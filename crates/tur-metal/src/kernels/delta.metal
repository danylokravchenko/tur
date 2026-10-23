// ---------------------------------------------------------------------------------------
// Gated DeltaNet
// ---------------------------------------------------------------------------------------

struct ConvParams {
    uint rows;
    uint channels;
    uint taps;
    uint x_stride;
};

// One sequence of a packed pass: its first packed row, token count, committed recurrent
// slot, offset of its written slots in the pass's slot list, whether it writes one slot
// per token, and its committed position. Sequences are consecutive and cover every row.
struct RecurrentSpan {
    uint row;
    uint tokens;
    uint read;
    uint writes;
    uint every;
    uint position;
};

// Recurrent state is updated in F32 and stored as T. Narrow storage rounds
// stochastically: the dropped mantissa bits are compared with a hash of the token
// position and element, so rounding is unbiased and errors average out over thousands of
// decode steps instead of drifting, while staying deterministic per request.
static inline uint round_bits(uint position, uint element) {
    uint h = position * 0x9E3779B1u ^ element * 0x85EBCA77u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    return h ^ (h >> 16);
}

template <typename T>
static inline T store_state(float v, uint position, uint element);

template <>
inline float store_state<float>(float v, uint, uint) {
    return v;
}

template <>
inline half store_state<half>(float v, uint position, uint element) {
    // Half keeps 10 of F32's 23 mantissa bits; below its normal range it rounds to nearest.
    if (fabs(v) < 6.103515625e-05f) {
        return half(v);
    }
    uint x = as_type<uint>(v) + (round_bits(position, element) & 0x1FFFu);
    return half(as_type<float>(x & 0xFFFFE000u));
}

template <>
inline bfloat store_state<bfloat>(float v, uint position, uint element) {
    uint x = as_type<uint>(v) + (round_bits(position, element) & 0xFFFFu);
    return bfloat(as_type<float>(x & 0xFFFF0000u));
}

static inline float conv_input(
    device const float *x, device const float *history, uint r, uint c, uint kept,
    constant ConvParams &p) {
    return r < kept ? history[r * p.channels + c] : x[ulong(r - kept) * p.x_stride + c];
}

// Depthwise causal convolution of every sequence of a packed pass over [history; x], then
// SiLU. Each sequence writes its final (taps-1)-row history into its first written slot,
// or the history after token t into its t-th slot when `every` is set. A sequence reads
// only its committed slot, which no sequence of the pass writes.
kernel void delta_conv_f32(
    device const float *x [[buffer(0)]],
    device float *pool [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant ConvParams &p [[buffer(4)]],
    device const RecurrentSpan *spans [[buffer(5)]],
    device const uint *row_span [[buffer(6)]],
    device const uint *slots [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.channels) {
        return;
    }
    uint row = gid / p.channels;
    uint c = gid % p.channels;
    RecurrentSpan s = spans[row_span[row]];
    uint t = row - s.row;
    uint kept = p.taps - 1;
    device const float *xs = x + ulong(s.row) * p.x_stride;
    device const float *history = pool + ulong(s.read) * kept * p.channels;
    device const float *wc = w + ulong(c) * p.taps;
    float acc = conv_input(xs, history, t, c, kept, p) * wc[0];
    for (uint j = 1; j < p.taps; ++j) {
        acc = acc + conv_input(xs, history, t + j, c, kept, p) * wc[j];
    }
    out[gid] = silu(acc);
    if (s.every != 0 || t + 1 == s.tokens) {
        ulong base = ulong(slots[s.writes + (s.every != 0 ? t : 0)]) * kept * p.channels;
        for (uint r = 0; r < kept; ++r) {
            pool[base + ulong(r) * p.channels + c] = conv_input(xs, history, t + 1 + r, c, kept, p);
        }
    }
}

struct ScanParams {
    uint sequences;
    uint heads;
    uint groups;
    uint key_width;
    uint key_dim;
    uint value_dim;
    uint mixed_stride;
    uint proj_stride;
    uint b_col;
    uint a_col;
};

// Gated delta rule over the sequences `order` lists, token by token, with the state held
// in registers. Threadgroups cover (sequence, head, column block); each simdgroup owns
// COLS value columns of one head and lane l holds key rows l + 32r.
// Per token: L2-normalise q/k, decay the state, read the prediction, write the
// beta-scaled correction and read the output with the query.
//
// The [dk, dv] state is read and written through a padded threadgroup tile so device
// accesses are contiguous rows of the threadgroup's columns; lane-owned rows alone would
// touch 16 bytes per cache line. State is stored as T (float, half or bfloat) and updated
// in F32.
template <uint R, uint COLS, typename T>
kernel void delta_scan(
    device const float *mixed [[buffer(0)]],
    device const float *proj [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *neg_a [[buffer(3)]],
    device T *pool [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant ScanParams &p [[buffer(6)]],
    device const RecurrentSpan *spans [[buffer(7)]],
    device const uint *slots [[buffer(8)]],
    device const uint *order [[buffer(9)]],
    threadgroup float *tile [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint dk = p.key_dim;
    uint dv = p.value_dim;
    uint column_groups = (dv + COLS - 1) / COLS;
    uint groups_per_tg = sgs;
    uint tgs_per_head = (column_groups + groups_per_tg - 1) / groups_per_tg;
    uint tgs_per_sequence = p.heads * tgs_per_head;
    uint index = tg / tgs_per_sequence;
    if (index >= p.sequences) {
        return;
    }
    RecurrentSpan s = spans[order[index]];
    uint local = tg % tgs_per_sequence;
    uint h = local / tgs_per_head;
    uint block = local % tgs_per_head;
    uint tile_cols = groups_per_tg * COLS;
    uint pitch = tile_cols + 1;
    uint tile_col0 = block * tile_cols;
    uint cg = block * groups_per_tg + sg;
    bool active = cg < column_groups;
    uint col0 = cg * COLS;
    uint kh = h / p.groups;
    ulong head_state = ulong(h) * dk * dv;
    ulong state_size = ulong(p.heads) * dk * dv;
    device const float *rows = mixed + ulong(s.row) * p.mixed_stride;
    device const float *gates = proj + ulong(s.row) * p.proj_stride;
    device const T *state_in = pool + ulong(s.read) * state_size;

    for (uint idx = tid; idx < dk * tile_cols; idx += threads) {
        uint i = idx / tile_cols;
        uint c = idx % tile_cols;
        uint j = tile_col0 + c;
        tile[i * pitch + c] = j < dv ? float(state_in[head_state + ulong(i) * dv + j]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float state[R][COLS];
    for (uint r = 0; r < R; ++r) {
        for (uint c = 0; c < COLS; ++c) {
            state[r][c] = tile[(lane + SIMD * r) * pitch + sg * COLS + c];
        }
    }

    if (active) {
        float scale = sqrt(float(dk));
        float bias = dt_bias[h];
        float a_neg = neg_a[h];
        uint q_col = kh * dk;
        uint k_col = p.key_width + kh * dk;
        uint v_col = 2 * p.key_width + h * dv + col0;

        // Software pipelining: inputs for token t + 1 load while token t updates the state.
        float q_next[R], k_next[R], v_next[COLS], a_next = 0.0f, b_next = 0.0f;
        for (uint r = 0; r < R; ++r) {
            q_next[r] = rows[q_col + lane + SIMD * r];
            k_next[r] = rows[k_col + lane + SIMD * r];
        }
        for (uint c = 0; c < COLS; ++c) {
            v_next[c] = col0 + c < dv ? rows[v_col + c] : 0.0f;
        }
        a_next = gates[p.a_col + h];
        b_next = gates[p.b_col + h];
        for (uint t = 0; t < s.tokens; ++t) {
            float q[R], k[R], v[COLS];
            float qs = 0.0f, ks = 0.0f;
            for (uint r = 0; r < R; ++r) {
                q[r] = q_next[r];
                k[r] = k_next[r];
                qs += q[r] * q[r];
                ks += k[r] * k[r];
            }
            for (uint c = 0; c < COLS; ++c) {
                v[c] = v_next[c];
            }
            float a = a_next;
            float b = b_next;
            if (t + 1 < s.tokens) {
                device const float *row = rows + ulong(t + 1) * p.mixed_stride;
                for (uint r = 0; r < R; ++r) {
                    q_next[r] = row[q_col + lane + SIMD * r];
                    k_next[r] = row[k_col + lane + SIMD * r];
                }
                for (uint c = 0; c < COLS; ++c) {
                    v_next[c] = col0 + c < dv ? row[v_col + c] : 0.0f;
                }
                device const float *pr = gates + ulong(t + 1) * p.proj_stride;
                a_next = pr[p.a_col + h];
                b_next = pr[p.b_col + h];
            }
            float q_inv = 1.0f / sqrt(simd_sum(qs) + 1e-6f);
            float k_inv = 1.0f / sqrt(simd_sum(ks) + 1e-6f);
            for (uint r = 0; r < R; ++r) {
                q[r] = q[r] * q_inv / scale;
                k[r] = k[r] * k_inv;
            }
            float decay = exp(softplus(a + bias) * a_neg);
            float beta = sigmoid(b);
            float prediction[COLS];
            for (uint c = 0; c < COLS; ++c) {
                float acc = 0.0f;
                for (uint r = 0; r < R; ++r) {
                    state[r][c] *= decay;
                    acc += k[r] * state[r][c];
                }
                prediction[c] = simd_sum(acc);
            }
            float output[COLS];
            for (uint c = 0; c < COLS; ++c) {
                float correction = (v[c] - prediction[c]) * beta;
                float acc = 0.0f;
                for (uint r = 0; r < R; ++r) {
                    state[r][c] += k[r] * correction;
                    acc += q[r] * state[r][c];
                }
                output[c] = simd_sum(acc);
            }
            device float *o = out + (ulong(s.row + t) * p.heads + h) * dv + col0;
            for (uint c = 0; c < COLS; ++c) {
                if (lane == c && col0 + c < dv) {
                    o[c] = output[c];
                }
            }
            // Per-token states serve speculative verification only; they skip staging.
            if (s.every != 0) {
                device T *st = pool + ulong(slots[s.writes + t]) * state_size + head_state;
                for (uint r = 0; r < R; ++r) {
                    for (uint c = 0; c < COLS; ++c) {
                        if (col0 + c < dv) {
                            ulong at = ulong(lane + SIMD * r) * dv + col0 + c;
                            st[at] = store_state<T>(
                                state[r][c], s.position + t, uint(head_state + at));
                        }
                    }
                }
            }
        }
    }
    if (s.every != 0) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < R; ++r) {
        for (uint c = 0; c < COLS; ++c) {
            tile[(lane + SIMD * r) * pitch + sg * COLS + c] = state[r][c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device T *final_state = pool + ulong(slots[s.writes]) * state_size + head_state;
    for (uint idx = tid; idx < dk * tile_cols; idx += threads) {
        uint i = idx / tile_cols;
        uint c = idx % tile_cols;
        uint j = tile_col0 + c;
        if (j < dv) {
            ulong at = ulong(i) * dv + j;
            final_state[at] = store_state<T>(
                tile[i * pitch + c], s.position + s.tokens - 1, uint(head_state + at));
        }
    }
}

#define SCAN(R, T, NAME) \
    template [[host_name("delta_scan_r" #R "_" #NAME)]] kernel void delta_scan<R, 4, T>( \
        device const float *, device const float *, device const float *, device const float *, \
        device T *, device float *, constant ScanParams &, device const RecurrentSpan *, \
        device const uint *, device const uint *, threadgroup float *, uint, uint, uint, uint, \
        uint, uint);
#define SCANS(T, NAME) SCAN(2, T, NAME) SCAN(4, T, NAME) SCAN(8, T, NAME)
SCANS(float, f32)
SCANS(half, f16)
SCANS(bfloat, bf16)

// One gated delta step for each single-token sequence `order` lists. A threadgroup covers
// (sequence, head, 64 value columns) with DK / 32 simdgroups: lane l owns value columns
// col0 + 2l and col0 + 2l + 1 and simdgroup g owns key rows 32g..32g+31, so every state
// load and store is a coalesced row segment. The two column reductions (prediction and
// output) sum the simdgroups' partials through threadgroup memory. State is stored as T
// (float, half or bfloat) and updated in F32.
template <uint DK, typename T>
kernel void delta_step(
    device const float *mixed [[buffer(0)]],
    device const float *proj [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *neg_a [[buffer(3)]],
    device T *pool [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant ScanParams &p [[buffer(6)]],
    device const RecurrentSpan *spans [[buffer(7)]],
    device const uint *slots [[buffer(8)]],
    device const uint *order [[buffer(9)]],
    threadgroup float2 *partial [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint GROUPS = DK / SIMD;
    uint dv = p.value_dim;
    uint blocks = (dv + 2 * SIMD - 1) / (2 * SIMD);
    uint per_sequence = p.heads * blocks;
    uint index = tg / per_sequence;
    if (index >= p.sequences) {
        return;
    }
    RecurrentSpan s = spans[order[index]];
    uint h = (tg % per_sequence) / blocks;
    uint j = (tg % blocks) * 2 * SIMD + 2 * lane;
    // Value dims are even (checked by the host), so a lane's pair is wholly in or out.
    bool active = j < dv;
    uint kh = h / p.groups;
    device const float *row = mixed + ulong(s.row) * p.mixed_stride;
    device const float *gates = proj + ulong(s.row) * p.proj_stride;
    device const float *q = row + kh * DK;
    device const float *k = row + p.key_width + kh * DK;
    float qs = 0.0f, ks = 0.0f;
    for (uint i = lane; i < DK; i += SIMD) {
        qs += q[i] * q[i];
        ks += k[i] * k[i];
    }
    float q_scale = 1.0f / (sqrt(simd_sum(qs) + 1e-6f) * sqrt(float(DK)));
    float k_scale = 1.0f / sqrt(simd_sum(ks) + 1e-6f);
    float decay = exp(softplus(gates[p.a_col + h] + dt_bias[h]) * neg_a[h]);
    float beta = sigmoid(gates[p.b_col + h]);
    device const float *vr = row + 2 * p.key_width + h * dv + j;
    float2 v = active ? float2(vr[0], vr[1]) : float2(0.0f);

    ulong state_size = ulong(p.heads) * DK * dv;
    ulong head = ulong(h) * DK * dv + j;
    device const T *state_in = pool + ulong(s.read) * state_size + head;
    uint i0 = sg * SIMD;
    float2 st[SIMD];
    float2 prediction = 0.0f;
    for (uint r = 0; r < SIMD; ++r) {
        st[r] = active ? float2(*((device const vec<T, 2> *)(state_in + ulong(i0 + r) * dv))) * decay
                       : float2(0.0f);
        prediction += k[i0 + r] * k_scale * st[r];
    }
    partial[sg * SIMD + lane] = prediction;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prediction = 0.0f;
    for (uint g = 0; g < GROUPS; ++g) {
        prediction += partial[g * SIMD + lane];
    }
    float2 correction = (v - prediction) * beta;
    float2 output = 0.0f;
    device T *state_out = pool + ulong(slots[s.writes]) * state_size + head;
    for (uint r = 0; r < SIMD; ++r) {
        st[r] += k[i0 + r] * k_scale * correction;
        output += q[i0 + r] * q_scale * st[r];
        if (active) {
            uint element = uint(head + ulong(i0 + r) * dv);
            *((device vec<T, 2> *)(state_out + ulong(i0 + r) * dv)) = vec<T, 2>(
                store_state<T>(st[r].x, s.position, element),
                store_state<T>(st[r].y, s.position, element + 1));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[sg * SIMD + lane] = output;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && active) {
        output = 0.0f;
        for (uint g = 0; g < GROUPS; ++g) {
            output += partial[g * SIMD + lane];
        }
        *((device float2 *)(out + (ulong(s.row) * p.heads + h) * dv + j)) = output;
    }
}

#define STEP(DK, T, NAME) \
    template [[host_name("delta_step_d" #DK "_" #NAME)]] kernel void delta_step<DK, T>( \
        device const float *, device const float *, device const float *, device const float *, \
        device T *, device float *, constant ScanParams &, device const RecurrentSpan *, \
        device const uint *, device const uint *, threadgroup float2 *, uint, uint, uint);
#define STEPS(T, NAME) STEP(64, T, NAME) STEP(128, T, NAME) STEP(256, T, NAME)
STEPS(float, f32)
STEPS(half, f16)
STEPS(bfloat, bf16)

