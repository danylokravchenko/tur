// Tur fused kernels. F32 activations; reductions and recurrent state accumulate in F32.
// Every kernel derives its work item from the dispatched threadgroup shape, so the host
// may choose any threadgroup width that is a multiple of the SIMD width.
#include <metal_stdlib>
using namespace metal;

#define SIMD 32

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-fabs(x))); }

// Sum over the whole threadgroup. Every thread must call it; `scratch` holds >= 32 floats.
static inline float threadgroup_sum(float v, threadgroup float *scratch, uint sg, uint lane, uint sgs) {
    v = simd_sum(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < sgs ? scratch[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// ---------------------------------------------------------------------------------------
// Normalization and elementwise primitives
// ---------------------------------------------------------------------------------------

struct RmsParams {
    uint rows;
    uint width;
    uint x_stride;
    uint r_stride;
    float eps;
};

// y = x * rsqrt(mean(x^2) + eps) * w, one threadgroup per row.
kernel void rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant RmsParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.x_stride;
    float acc = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        float v = xr[i];
        acc += v * v;
    }
    float total = threadgroup_sum(acc, scratch, sg, lane, sgs);
    float inv = 1.0f / sqrt(total / float(p.width) + p.eps);
    device float *yr = y + ulong(row) * p.width;
    for (uint i = tid; i < p.width; i += threads) {
        yr[i] = xr[i] * inv * w[i];
    }
}

struct ArgmaxParams {
    uint rows;
    uint cols;
    uint stride;
};

// Keep the larger value, or the lower index between equal values.
static inline void better(thread float &v, thread uint &i, float ov, uint oi) {
    if (oi != UINT_MAX && (i == UINT_MAX || ov > v || (ov == v && oi < i))) {
        v = ov;
        i = oi;
    }
}

// Greedy decoding without host logits: per row, the first maximum and whether every value
// is finite, as out[row] = (index, 1) or (index, 0). One threadgroup per row; indices are
// exact in F32 because the host bounds the vocabulary below 2^24.
kernel void argmax_f32(
    device const float *x [[buffer(0)]],
    device float2 *out [[buffer(1)]],
    constant ArgmaxParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float values[SIMD];
    threadgroup uint indices[SIMD];
    threadgroup uint flags[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.stride;
    float v = 0.0f;
    uint i = UINT_MAX;
    uint nonfinite = 0;
    for (uint c = tid; c < p.cols; c += threads) {
        float s = xr[c];
        // -inf marks a masked token; NaN and +inf are invalid.
        nonfinite |= (isnan(s) || s == INFINITY) ? 1u : 0u;
        better(v, i, s, c);
    }
    for (uint offset = SIMD / 2; offset > 0; offset /= 2) {
        better(v, i, simd_shuffle_xor(v, offset), simd_shuffle_xor(i, offset));
    }
    nonfinite = simd_max(nonfinite);
    if (lane == 0) {
        values[sg] = v;
        indices[sg] = i;
        flags[sg] = nonfinite;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        v = lane < sgs ? values[lane] : 0.0f;
        i = lane < sgs ? indices[lane] : UINT_MAX;
        nonfinite = lane < sgs ? flags[lane] : 0u;
        for (uint offset = SIMD / 2; offset > 0; offset /= 2) {
            better(v, i, simd_shuffle_xor(v, offset), simd_shuffle_xor(i, offset));
        }
        nonfinite = simd_max(nonfinite);
        if (lane == 0) {
            // A row without any unmasked token is invalid too.
            out[row] = float2(float(i), nonfinite == 0 && v > -INFINITY ? 1.0f : 0.0f);
        }
    }
}

// s = x + r (written out), y = rms(s) * w: residual addition fused with the next norm.
kernel void add_rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *r [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *s [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant RmsParams &p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.x_stride;
    device const float *rr = r + ulong(row) * p.r_stride;
    device float *sr = s + ulong(row) * p.width;
    float acc = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        float v = xr[i] + rr[i];
        sr[i] = v;
        acc += v * v;
    }
    float total = threadgroup_sum(acc, scratch, sg, lane, sgs);
    float inv = 1.0f / sqrt(total / float(p.width) + p.eps);
    device float *yr = y + ulong(row) * p.width;
    for (uint i = tid; i < p.width; i += threads) {
        yr[i] = (xr[i] + rr[i]) * inv * w[i];
    }
}

struct RopeParams {
    uint tokens;
    uint heads;
    uint dim;
    uint row_stride;
    uint head_stride;
    uint col_offset;
    uint half_rotary;
    float eps;
};

// Per-head RMSNorm followed by split-half rotary over the first 2 * half_rotary dimensions.
// Angles are position * inv_freq[f] in F32; the host computes inv_freq exactly as the
// reference does, because a one-ulp frequency difference grows linearly with position.
// Reads heads at an arbitrary row/head stride (e.g. the q half of an interleaved q|gate
// projection) and writes contiguous [tokens, heads, dim]. One simdgroup per (token, head).
kernel void qk_norm_rope_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const uint *positions [[buffer(2)]],
    device const float *inv_freq [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant RopeParams &p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.heads) {
        return;
    }
    uint t = item / p.heads;
    uint h = item % p.heads;
    device const float *xr = x + ulong(t) * p.row_stride + p.col_offset + ulong(h) * p.head_stride;
    float acc = 0.0f;
    for (uint i = lane; i < p.dim; i += SIMD) {
        float v = xr[i];
        acc += v * v;
    }
    float inv = 1.0f / sqrt(simd_sum(acc) / float(p.dim) + p.eps);
    float position = float(positions[t]);
    uint half_rotary = p.half_rotary;
    device float *yr = y + (ulong(t) * p.heads + h) * p.dim;
    for (uint i = lane; i < p.dim; i += SIMD) {
        float v = xr[i] * inv * w[i];
        if (i < 2 * half_rotary) {
            bool left = i < half_rotary;
            uint f = left ? i : i - half_rotary;
            uint partner = left ? i + half_rotary : i - half_rotary;
            float u = xr[partner] * inv * w[partner];
            float angle = position * inv_freq[f];
            float c = precise::cos(angle);
            float s = precise::sin(angle);
            v = left ? v * c - u * s : v * c + u * s;
        }
        yr[i] = v;
    }
}

struct GluParams {
    uint rows;
    uint width;
};

// out[r, i] = silu(gu[r, i]) * gu[r, width + i] over a fused gate|up projection.
kernel void swiglu_f32(
    device const float *gu [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant GluParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.width) {
        return;
    }
    uint r = gid / p.width;
    uint c = gid % p.width;
    device const float *row = gu + ulong(r) * 2 * p.width;
    out[gid] = silu(row[c]) * row[p.width + c];
}

struct GatedParams {
    uint rows;
    uint dim;
    uint heads;
    uint z_stride;
    uint z_col;
    float eps;
};

// out = rms(core) * w * silu(z) per [token, head] row; z is read in place from the projection.
kernel void gated_rms_f32(
    device const float *core [[buffer(0)]],
    device const float *z [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant GatedParams &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint row = tg * sgs + sg;
    if (row >= p.rows) {
        return;
    }
    uint t = row / p.heads;
    uint h = row % p.heads;
    device const float *cr = core + ulong(row) * p.dim;
    device const float *zr = z + ulong(t) * p.z_stride + p.z_col + ulong(h) * p.dim;
    float acc = 0.0f;
    for (uint i = lane; i < p.dim; i += SIMD) {
        float v = cr[i];
        acc += v * v;
    }
    float inv = 1.0f / sqrt(simd_sum(acc) / float(p.dim) + p.eps);
    device float *o = out + ulong(row) * p.dim;
    for (uint i = lane; i < p.dim; i += SIMD) {
        o[i] = cr[i] * inv * w[i] * silu(zr[i]);
    }
}

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

// ---------------------------------------------------------------------------------------
// Paged K/V: quantize-on-write of new rows and attention reading pages in place.
// Storage formats: F32 and F16 values, or Q8 two's-complement codes with one F16 scale
// per `group` head dimensions. Q8 groups are multiples of 32, so a group never splits a
// 32-dimension run of one register (writes) or a lane's 4-dimension vector (reads).
// ---------------------------------------------------------------------------------------

struct KvWriteParams {
    uint tokens;
    uint kv_heads;
    uint dim;
    uint page;
    uint group;
    uint k_stride;
    uint k_col;
    uint k_head_stride;
    uint v_stride;
    uint v_col;
    uint v_head_stride;
    float inv_127;
};

// Store new keys and values. One simdgroup per (K or V, token, head). Token t's row goes
// to logical position visible[t] - 1 of its page table. Q8 codes match the host
// reference: scale = half(max|x| * (1/127)), code = clamp(round(x / scale), +-127).
template <uint R, typename T, uint QUANT>
kernel void kv_write(
    device const float *k_src [[buffer(0)]],
    device const float *v_src [[buffer(1)]],
    device const uint *table [[buffer(2)]],
    device const uint *first_block [[buffer(3)]],
    device const uint *visible [[buffer(4)]],
    device T *k_data [[buffer(5)]],
    device half *k_scales [[buffer(6)]],
    device T *v_data [[buffer(7)]],
    device half *v_scales [[buffer(8)]],
    constant KvWriteParams &p [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint item = tg * sgs + sg;
    if (item >= 2 * p.tokens * p.kv_heads) {
        return;
    }
    uint h = item % p.kv_heads;
    uint rest = item / p.kv_heads;
    uint t = rest % p.tokens;
    bool value = rest >= p.tokens;
    uint j = visible[t] - 1;
    ulong slot = ulong(table[first_block[t] + j / p.page]) * p.page + j % p.page;
    device const float *src = value
        ? v_src + ulong(t) * p.v_stride + p.v_col + ulong(h) * p.v_head_stride
        : k_src + ulong(t) * p.k_stride + p.k_col + ulong(h) * p.k_head_stride;
    float x[R];
    for (uint r = 0; r < R; ++r) {
        x[r] = src[lane + SIMD * r];
    }
    device T *dst = (value ? v_data : k_data) + (slot * p.kv_heads + h) * p.dim;
    if (QUANT == 0) {
        for (uint r = 0; r < R; ++r) {
            dst[lane + SIMD * r] = T(x[r]);
        }
        return;
    }
    uint per = p.group / SIMD;
    device half *scales = (value ? v_scales : k_scales) + (slot * p.kv_heads + h) * (p.dim / p.group);
    for (uint g = 0; g * per < R; ++g) {
        float amax = 0.0f;
        for (uint r = g * per; r < (g + 1) * per; ++r) {
            amax = max(amax, simd_max(fabs(x[r])));
        }
        half scale = half(amax * p.inv_127);
        float divisor = max(float(scale), FLT_MIN);
        for (uint r = g * per; r < (g + 1) * per; ++r) {
            dst[lane + SIMD * r] = T(int(clamp(round(x[r] / divisor), -127.0f, 127.0f)));
        }
        if (lane == 0) {
            scales[g] = scale;
        }
    }
}

struct AttentionParams {
    uint tokens;
    uint heads;
    uint kv_heads;
    uint dim;
    uint page;
    uint splits;
    uint split_keys;
    uint gate_stride;
    uint gate_col;
    uint gate_head_stride;
    uint group;
    float scale;
};

// One simdgroup per (head, split, token). Lane l owns the W-wide vectors at dimensions
// W*l + 32*W*c, so each key and value row is read with vector loads; a Q8 lane needs one
// scale per vector. Online softmax keeps the running max/sum in F32. With one split the
// gated output is written directly; otherwise unnormalised partial sums and (max, sum)
// statistics go to scratch.
template <uint C, uint W, typename T, uint QUANT>
kernel void attention_kv(
    device const float *q [[buffer(0)]],
    device const T *k [[buffer(1)]],
    device const T *v [[buffer(2)]],
    device const uint *table [[buffer(3)]],
    device const uint *first_block [[buffer(4)]],
    device const uint *visible [[buffer(5)]],
    device const float *gate [[buffer(6)]],
    device float *out [[buffer(7)]],
    device float *partial [[buffer(8)]],
    device float *stats [[buffer(9)]],
    constant AttentionParams &p [[buffer(10)]],
    device const half *k_scales [[buffer(11)]],
    device const half *v_scales [[buffer(12)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    typedef vec<float, W> fv;
    typedef vec<T, W> tv;
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.heads * p.splits) {
        return;
    }
    uint h = item % p.heads;
    uint rest = item / p.heads;
    uint split = rest % p.splits;
    uint t = rest / p.splits;
    uint g = h / (p.heads / p.kv_heads);
    uint keys = visible[t];
    uint begin = split * p.split_keys;
    uint end = min(keys, begin + p.split_keys);
    uint base = first_block[t];
    uint groups = QUANT == 0 ? 0 : p.dim / p.group;

    fv qv[C], acc[C];
    uint dims[C], sidx[C];
    device const float *qr = q + (ulong(t) * p.heads + h) * p.dim;
    for (uint c = 0; c < C; ++c) {
        dims[c] = W * lane + SIMD * W * c;
        qv[c] = *(device const fv *)(qr + dims[c]);
        acc[c] = fv(0.0f);
        sidx[c] = QUANT == 0 ? 0 : dims[c] / p.group;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    for (uint j = begin; j < end; ++j) {
        ulong slot = ulong(table[base + j / p.page]) * p.page + j % p.page;
        ulong head = slot * p.kv_heads + g;
        device const T *kr = k + head * p.dim;
        device const T *vr = v + head * p.dim;
        float partial_dot = 0.0f;
        for (uint c = 0; c < C; ++c) {
            float d = dot(qv[c], fv(*(device const tv *)(kr + dims[c])));
            partial_dot += QUANT == 0 ? d : d * float(k_scales[head * groups + sidx[c]]);
        }
        float score = simd_sum(partial_dot) * p.scale;
        float m_new = max(m, score);
        float correction = exp(m - m_new);
        float weight = exp(score - m_new);
        l = l * correction + weight;
        for (uint c = 0; c < C; ++c) {
            float w = QUANT == 0 ? weight : weight * float(v_scales[head * groups + sidx[c]]);
            acc[c] = acc[c] * correction + w * fv(*(device const tv *)(vr + dims[c]));
        }
        m = m_new;
    }
    if (p.splits == 1) {
        device const float *gr = gate + ulong(t) * p.gate_stride + p.gate_col + ulong(h) * p.gate_head_stride;
        device float *o = out + (ulong(t) * p.heads + h) * p.dim;
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint c = 0; c < C; ++c) {
            for (uint i = 0; i < W; ++i) {
                o[dims[c] + i] = acc[c][i] * inv * sigmoid(gr[dims[c] + i]);
            }
        }
        return;
    }
    ulong slot = (ulong(t) * p.heads + h) * p.splits + split;
    device float *pp = partial + slot * p.dim;
    for (uint c = 0; c < C; ++c) {
        *(device fv *)(pp + dims[c]) = acc[c];
    }
    if (lane == 0) {
        stats[2 * slot] = m;
        stats[2 * slot + 1] = l;
    }
}

// Tiled prefill attention on the matrix units. A threadgroup of 4 simdgroups takes up to
// TQ consecutive query tokens of one sequence for one head; simdgroup s owns query rows
// 8s..8s+8 and keeps O as 8x8 F32 simdgroup matrices. Q is loaded one fragment at a time
// to bound register pressure at head dimension 256. Keys are
// swept in blocks of TK: each block's K, then V, is dequantized once into threadgroup
// memory, TD dimensions at a time, with vector loads, and shared by all TQ queries.
// S = QK^T, the online softmax and O += PV stay in registers; lanes holding the same row
// combine through two shuffles.
#define TQ 32
#define TK 32
#define TD 64

struct TiledParams {
    uint tiles;
    uint heads;
    uint kv_heads;
    uint page;
    uint group;
    uint gate_stride;
    uint gate_col;
    uint gate_head_stride;
    float scale;
};

// Row and first column of the two elements a lane holds in an 8x8 simdgroup matrix. Lanes
// sharing a row differ only in lane bits 0 and 3.
static inline ushort2 fragment_coord(ushort lane) {
    ushort quad = lane / 4;
    return ushort2((quad & 4) + (lane / 2) % 4, (quad & 2) * 2 + (lane % 2) * 2);
}

static inline float row_max(float x) {
    x = max(x, simd_shuffle_xor(x, 1));
    return max(x, simd_shuffle_xor(x, 8));
}

static inline float row_sum(float x) {
    x += simd_shuffle_xor(x, 1);
    return x + simd_shuffle_xor(x, 8);
}

// Four dequantized dimensions d..d+4 of one stored row; groups are multiples of 32.
template <typename T, uint QUANT>
static inline float4 kv_vector(device const T *data, device const half *scales, ulong head, uint dim, uint d, uint group) {
    float4 x = float4(*(device const vec<T, 4> *)(data + head * dim + d));
    return QUANT == 0 ? x : x * float(scales[head * (dim / group) + d / group]);
}

template <uint CHUNKS, typename T, uint QUANT>
kernel void attention_tiled(
    device const float *q [[buffer(0)]],
    device const T *k [[buffer(1)]],
    device const T *v [[buffer(2)]],
    device const uint *table [[buffer(3)]],
    device const uint *first_block [[buffer(4)]],
    device const uint *visible [[buffer(5)]],
    device const float *gate [[buffer(6)]],
    device float *out [[buffer(7)]],
    device const uint *tiles [[buffer(8)]],
    constant TiledParams &p [[buffer(9)]],
    device const half *k_scales [[buffer(10)]],
    device const half *v_scales [[buffer(11)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint D = TD * CHUNKS;
    uint h = tg % p.heads;
    uint tile = tg / p.heads;
    if (tile >= p.tiles) {
        return;
    }
    uint t0 = tiles[2 * tile];
    uint n = tiles[2 * tile + 1];
    uint g = h / (p.heads / p.kv_heads);
    threadgroup float *stage = shared;
    threadgroup uint *slots = (threadgroup uint *)(stage + TK * TD);

    ushort2 fc = fragment_coord(lane);
    uint row = 8 * sg + fc.x;
    // Rows past the tile repeat its last token and are never written.
    uint token = t0 + min(row, n - 1);
    uint limit = row < n ? visible[token] : 0;
    // Visible counts grow along a sequence, so the last token sees every key of the tile.
    uint keys = visible[t0 + n - 1];
    uint base = first_block[t0];

    simdgroup_float8x8 o[8 * CHUNKS];
    device const float *qr = q + (ulong(token) * p.heads + h) * D;
    // Every matrix-array index must become constant: dynamic indexing spills the
    // fragments to thread memory, including inside the dimension-chunk loops.
    #pragma clang loop unroll(full)
    for (uint x = 0; x < 8 * CHUNKS; ++x) {
        o[x] = simdgroup_float8x8(0.0f);
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    for (uint kb = 0; kb < keys; kb += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < TK) {
            uint j = kb + tid;
            slots[tid] = j < keys ? table[base + j / p.page] * p.page + j % p.page : 0;
        }
        simdgroup_float8x8 s[TK / 8];
        #pragma clang loop unroll(full)
        for (uint kt = 0; kt < TK / 8; ++kt) {
            s[kt] = simdgroup_float8x8(0.0f);
        }
        #pragma clang loop unroll(full)
        for (uint c = 0; c < CHUNKS; ++c) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < TK * TD / 4; i += threads) {
                uint key = i / (TD / 4);
                uint d = 4 * (i % (TD / 4));
                ulong head = ulong(slots[key]) * p.kv_heads + g;
                *(threadgroup float4 *)(stage + key * TD + d) = kb + key < keys
                    ? kv_vector<T, QUANT>(k, k_scales, head, D, c * TD + d, p.group)
                    : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint x = 0; x < TD / 8; ++x) {
                simdgroup_float8x8 a;
                // The clamped token also keeps partial query tiles inside the allocation.
                float2 pair = *(device const float2 *)(qr + c * TD + 8 * x + fc.y);
                a.thread_elements()[0] = pair[0] * p.scale;
                a.thread_elements()[1] = pair[1] * p.scale;
                #pragma clang loop unroll(full)
                for (uint kt = 0; kt < TK / 8; ++kt) {
                    simdgroup_float8x8 b;
                    simdgroup_load(b, stage + 8 * kt * TD + 8 * x, TD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[kt], a, b, s[kt]);
                }
            }
        }
        // Online softmax on this lane's two elements per key tile, all in row `row`.
        float block = -FLT_MAX;
        #pragma clang loop unroll(full)
        for (uint kt = 0; kt < TK / 8; ++kt) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                uint key = kb + 8 * kt + fc.y + e;
                float score = s[kt].thread_elements()[e];
                block = key < limit ? max(block, score) : block;
            }
        }
        float m_new = max(m, row_max(block));
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint kt = 0; kt < TK / 8; ++kt) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                uint key = kb + 8 * kt + fc.y + e;
                float weight = key < limit ? exp(s[kt].thread_elements()[e] - m_new) : 0.0f;
                s[kt].thread_elements()[e] = weight;
                total += weight;
            }
        }
        float correction = exp(m - m_new);
        l = l * correction + row_sum(total);
        m = m_new;
        #pragma clang loop unroll(full)
        for (uint x = 0; x < 8 * CHUNKS; ++x) {
            o[x].thread_elements()[0] *= correction;
            o[x].thread_elements()[1] *= correction;
        }
        #pragma clang loop unroll(full)
        for (uint c = 0; c < CHUNKS; ++c) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < TK * TD / 4; i += threads) {
                uint key = i / (TD / 4);
                uint d = 4 * (i % (TD / 4));
                ulong head = ulong(slots[key]) * p.kv_heads + g;
                *(threadgroup float4 *)(stage + key * TD + d) = kb + key < keys
                    ? kv_vector<T, QUANT>(v, v_scales, head, D, c * TD + d, p.group)
                    : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint kt = 0; kt < TK / 8; ++kt) {
                #pragma clang loop unroll(full)
                for (uint u = 0; u < TD / 8; ++u) {
                    simdgroup_float8x8 b;
                    simdgroup_load(b, stage + 8 * kt * TD + 8 * u, TD, ulong2(0, 0), false);
                    simdgroup_multiply_accumulate(o[8 * c + u], s[kt], b, o[8 * c + u]);
                }
            }
        }
    }
    if (row >= n) {
        return;
    }
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device const float *gr = gate + ulong(token) * p.gate_stride + p.gate_col + ulong(h) * p.gate_head_stride;
    device float *orow = out + (ulong(token) * p.heads + h) * D;
    #pragma clang loop unroll(full)
    for (uint x = 0; x < 8 * CHUNKS; ++x) {
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 2; ++e) {
            uint col = 8 * x + fc.y + e;
            orow[col] = o[x].thread_elements()[e] * inv * sigmoid(gr[col]);
        }
    }
}

// Combine split partials of one (token, head) and apply the sigmoid output gate.
template <uint R>
kernel void attention_reduce_f32(
    device const float *partial [[buffer(0)]],
    device const float *stats [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.heads) {
        return;
    }
    uint t = item / p.heads;
    uint h = item % p.heads;
    ulong first = ulong(item) * p.splits;
    float m = -FLT_MAX;
    for (uint s = 0; s < p.splits; ++s) {
        if (stats[2 * (first + s) + 1] > 0.0f) {
            m = max(m, stats[2 * (first + s)]);
        }
    }
    float l = 0.0f;
    float acc[R];
    for (uint r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (uint s = 0; s < p.splits; ++s) {
        float ls = stats[2 * (first + s) + 1];
        if (ls > 0.0f) {
            float factor = exp(stats[2 * (first + s)] - m);
            l += ls * factor;
            device const float *pp = partial + (first + s) * p.dim;
            for (uint r = 0; r < R; ++r) {
                acc[r] += pp[lane + SIMD * r] * factor;
            }
        }
    }
    device const float *gr = gate + ulong(t) * p.gate_stride + p.gate_col + ulong(h) * p.gate_head_stride;
    device float *o = out + (ulong(t) * p.heads + h) * p.dim;
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint r = 0; r < R; ++r) {
        uint i = lane + SIMD * r;
        o[i] = acc[r] * inv * sigmoid(gr[i]);
    }
}

#define TILED(R, NAME, T, QUANT) \
    template [[host_name("attention_tiled_r" #R "_" #NAME)]] kernel void attention_tiled<R / 2, T, QUANT>( \
        device const float *, device const T *, device const T *, device const uint *, \
        device const uint *, device const uint *, device const float *, device float *, \
        device const uint *, constant TiledParams &, device const half *, device const half *, \
        threadgroup float *, uint, uint, uint, uint, uint);
#define KV_FORMAT(R, C, W, NAME, T, QUANT) \
    TILED(R, NAME, T, QUANT) \
    template [[host_name("kv_write_r" #R "_" #NAME)]] kernel void kv_write<R, T, QUANT>( \
        device const float *, device const float *, device const uint *, device const uint *, \
        device const uint *, device T *, device half *, device T *, device half *, \
        constant KvWriteParams &, uint, uint, uint, uint); \
    template [[host_name("attention_r" #R "_" #NAME)]] kernel void attention_kv<C, W, T, QUANT>( \
        device const float *, device const T *, device const T *, device const uint *, \
        device const uint *, device const uint *, device const float *, device float *, \
        device float *, device float *, constant AttentionParams &, device const half *, \
        device const half *, uint, uint, uint, uint);
// R registers per lane = C vectors of W lanes' dimensions: 64 = 1 x 2, 128 = 1 x 4, 256 = 2 x 4.
#define ATTENTION(R, C, W) \
    KV_FORMAT(R, C, W, f32, float, 0) \
    KV_FORMAT(R, C, W, f16, half, 0) \
    KV_FORMAT(R, C, W, q8, char, 1) \
    template [[host_name("attention_reduce_r" #R "_f32")]] kernel void attention_reduce_f32<R>( \
        device const float *, device const float *, device const float *, device float *, \
        constant AttentionParams &, uint, uint, uint, uint);
ATTENTION(2, 1, 2)
ATTENTION(4, 1, 4)
ATTENTION(8, 2, 4)

// ---------------------------------------------------------------------------------------
// Packed small-batch matrix-vector products: y[b, n] = sum_k x[b, k] * W[n, k], b < B.
// Weights are GGML blocks, read once for all B input rows.
// ---------------------------------------------------------------------------------------

struct QmvParams {
    uint n;
    uint k;
    uint x_stride;
    uint y_stride;
};

struct block_q8_0 {
    half d;
    char qs[32];
};

struct block_q4_k {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

// Four lanes share one Q8_0 block (8 quants each); a simdgroup covers 8 blocks per step.
// Quants and activations load as vectors; each activation vector is reused by all NR
// rows, so activation traffic per weight byte falls as NR grows.
template <uint B, uint NR>
kernel void qmv_q8_0_f32(
    device const block_q8_0 *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant QmvParams &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint row0 = (tg * sgs + sg) * NR;
    if (row0 >= p.n) {
        return;
    }
    uint blocks = p.k / 32;
    uint ix = lane / 4;
    uint il = lane % 4;
    float acc[NR][B];
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < B; ++b) {
            acc[r][b] = 0.0f;
        }
    }
    for (uint ib = ix; ib < blocks; ib += 8) {
        float4 q_lo[NR], q_hi[NR];
        float scale[NR];
        for (uint r = 0; r < NR; ++r) {
            uint row = min(row0 + r, p.n - 1);
            device const block_q8_0 &blk = w[ulong(row) * blocks + ib];
            device const packed_char4 *qs = (device const packed_char4 *)(blk.qs + il * 8);
            scale[r] = float(blk.d);
            q_lo[r] = float4(char4(qs[0]));
            q_hi[r] = float4(char4(qs[1]));
        }
        for (uint b = 0; b < B; ++b) {
            device const float4 *xb =
                (device const float4 *)(x + ulong(b) * p.x_stride + ib * 32 + il * 8);
            float4 x_lo = xb[0];
            float4 x_hi = xb[1];
            for (uint r = 0; r < NR; ++r) {
                acc[r][b] += (dot(q_lo[r], x_lo) + dot(q_hi[r], x_hi)) * scale[r];
            }
        }
    }
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < B; ++b) {
            float total = simd_sum(acc[r][b]);
            if (lane == 0 && row0 + r < p.n) {
                y[ulong(b) * p.y_stride + row0 + r] = total;
            }
        }
    }
}

static inline uchar2 q4k_scale_min(uint j, device const uchar *q) {
    if (j < 4) {
        return uchar2(q[j] & 63, q[j + 4] & 63);
    }
    return uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// Eight lanes share one Q4_K super-block and a simdgroup keeps four blocks in flight.
// Lane slot s reads 16 packed bytes of 64-element group s/2: low nibbles are 16 values of
// sub-block 2*group, high nibbles 16 values of sub-block 2*group + 1.
template <uint B, uint NR>
kernel void qmv_q4k_f32(
    device const block_q4_k *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant QmvParams &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint row0 = (tg * sgs + sg) * NR;
    if (row0 >= p.n) {
        return;
    }
    uint blocks = p.k / 256;
    uint ix = lane / 8;
    uint slot = lane % 8;
    uint group = slot / 2;
    uint offset = 16 * (slot % 2);
    float acc[NR][B];
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < B; ++b) {
            acc[r][b] = 0.0f;
        }
    }
    for (uint ib = ix; ib < blocks; ib += 4) {
        uint4 packed[NR];
        float d_lo[NR], m_lo[NR], d_hi[NR], m_hi[NR];
        for (uint r = 0; r < NR; ++r) {
            uint row = min(row0 + r, p.n - 1);
            device const block_q4_k &blk = w[ulong(row) * blocks + ib];
            float d = float(blk.d);
            float dmin = float(blk.dmin);
            uchar2 first = q4k_scale_min(2 * group, blk.scales);
            uchar2 second = q4k_scale_min(2 * group + 1, blk.scales);
            d_lo[r] = d * float(first.x);
            m_lo[r] = dmin * float(first.y);
            d_hi[r] = d * float(second.x);
            m_hi[r] = dmin * float(second.y);
            packed[r] = *((device const uint4 *)(blk.qs + 32 * group + offset));
        }
        for (uint b = 0; b < B; ++b) {
            device const float4 *xl = (device const float4 *)(x + ulong(b) * p.x_stride + ib * 256 + 64 * group + offset);
            device const float4 *xh = xl + 8;
            float4 lo_x[4], hi_x[4];
            float sum_lo = 0.0f, sum_hi = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                lo_x[i] = xl[i];
                hi_x[i] = xh[i];
                sum_lo += lo_x[i].x + lo_x[i].y + lo_x[i].z + lo_x[i].w;
                sum_hi += hi_x[i].x + hi_x[i].y + hi_x[i].z + hi_x[i].w;
            }
            for (uint r = 0; r < NR; ++r) {
                float dot_lo = 0.0f, dot_hi = 0.0f;
                for (uint i = 0; i < 4; ++i) {
                    uint word = packed[r][i];
                    float4 lo = float4(word & 0xF, (word >> 8) & 0xF, (word >> 16) & 0xF, (word >> 24) & 0xF);
                    float4 hi = float4((word >> 4) & 0xF, (word >> 12) & 0xF, (word >> 20) & 0xF, word >> 28);
                    dot_lo += dot(lo, lo_x[i]);
                    dot_hi += dot(hi, hi_x[i]);
                }
                acc[r][b] += d_lo[r] * dot_lo - m_lo[r] * sum_lo + d_hi[r] * dot_hi - m_hi[r] * sum_hi;
            }
        }
    }
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < B; ++b) {
            float total = simd_sum(acc[r][b]);
            if (lane == 0 && row0 + r < p.n) {
                y[ulong(b) * p.y_stride + row0 + r] = total;
            }
        }
    }
}

// Small-batch Q8_0 matrix multiplication on the matrix units, computed transposed:
// y^T[n, b] = w[n, :] · x^T[:, b] for up to 8 * BT activation rows. A threadgroup owns
// 8 * NT output columns; its simdgroups split the Q8_0 blocks along k and reduce once at
// the end, so even narrow projections fill the GPU.
//
// Fragments load straight from device memory with no staging or barriers. Lane l of an
// 8x8 fragment holds (r, c) and (r, c + 1), r = (l/4 & 4) + (l/2) % 4, c = (l/4 & 2) * 2 +
// (l % 2) * 2. Within a block, step s of four maps fragment index c (a weight column or
// an activation row) to input k = 4c + s, so each lane's weights for the whole block are
// one 8-byte load and its activations one float4 per batch row, every value used.
// Accumulation is F32 throughout.
struct QmmParams {
    uint n;
    uint k;
    uint x_stride;
    uint y_stride;
    uint rows;
};

template <uint BT, uint NT>
kernel void qmm_q8_0_f32(
    device const block_q8_0 *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant QmmParams &p [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint qid = lane / 4;
    uint fr = (qid & 4) + (lane / 2) % 4;
    uint fc = (qid & 2) * 2 + (lane % 2) * 2;
    uint col0 = tg * 8 * NT;
    uint blocks = p.k / 32;
    simdgroup_float8x8 acc[NT][BT];
    for (uint t = 0; t < NT; ++t) {
        for (uint b = 0; b < BT; ++b) {
            acc[t][b] = simdgroup_float8x8(0.0f);
        }
    }
    // This lane's weight rows and activation rows.
    device const block_q8_0 *wr[NT];
    for (uint t = 0; t < NT; ++t) {
        wr[t] = w + ulong(min(col0 + t * 8 + fr, p.n - 1)) * blocks;
    }
    // This lane's activations k = 4 * fr + s of batch rows fc and fc + 1.
    uint xk = 4 * fr;
    for (uint ib = sg; ib < blocks; ib += sgs) {
        float4 xv[BT][2];
        for (uint b = 0; b < BT; ++b) {
            for (uint e = 0; e < 2; ++e) {
                uint row = b * 8 + fc + e;
                xv[b][e] = row < p.rows
                    ? *((device const float4 *)(x + ulong(row) * p.x_stride + ib * 32 + xk))
                    : float4(0.0f);
            }
        }
        float4 q_lo[NT], q_hi[NT];
        for (uint t = 0; t < NT; ++t) {
            device const block_q8_0 &blk = wr[t][ib];
            device const packed_char4 *qs = (device const packed_char4 *)(blk.qs + 4 * fc);
            float d = float(blk.d);
            q_lo[t] = float4(char4(qs[0])) * d;
            q_hi[t] = float4(char4(qs[1])) * d;
        }
        for (uint s = 0; s < 4; ++s) {
            simdgroup_float8x8 xm[BT];
            for (uint b = 0; b < BT; ++b) {
                xm[b].thread_elements()[0] = xv[b][0][s];
                xm[b].thread_elements()[1] = xv[b][1][s];
            }
            for (uint t = 0; t < NT; ++t) {
                simdgroup_float8x8 wm;
                wm.thread_elements()[0] = q_lo[t][s];
                wm.thread_elements()[1] = q_hi[t][s];
                for (uint b = 0; b < BT; ++b) {
                    simdgroup_multiply_accumulate(acc[t][b], wm, xm[b], acc[t][b]);
                }
            }
        }
    }
    // Reduce the k splits: partial[sg][t][b][r][c].
    for (uint t = 0; t < NT; ++t) {
        for (uint b = 0; b < BT; ++b) {
            threadgroup float *out = partial + ((sg * NT + t) * BT + b) * 64 + fr * 8 + fc;
            out[0] = acc[t][b].thread_elements()[0];
            out[1] = acc[t][b].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < NT * BT * 64; i += threads) {
        float total = 0.0f;
        for (uint g = 0; g < sgs; ++g) {
            total += partial[g * NT * BT * 64 + i];
        }
        uint t = i / (BT * 64);
        uint b = (i / 64) % BT;
        uint col = col0 + t * 8 + (i % 64) / 8;
        uint row = b * 8 + i % 8;
        if (row < p.rows && col < p.n) {
            y[ulong(row) * p.y_stride + col] = total;
        }
    }
}

#define QMM(BT, NT) \
    template [[host_name("qmm_q8_0_t" #BT "_f32")]] kernel void qmm_q8_0_f32<BT, NT>( \
        device const block_q8_0 *, device const float *, device float *, constant QmmParams &, \
        threadgroup float *, uint, uint, uint, uint, uint, uint);
// Two column tiles per threadgroup: four measured no faster on M1 (P6 kernel artifact).
QMM(1, 2)
QMM(2, 2)

// Rows per simdgroup shrink with the batch to bound accumulator registers.
#define QMV(B, NR8, NR4) \
    template [[host_name("qmv_q8_0_b" #B "_f32")]] kernel void qmv_q8_0_f32<B, NR8>( \
        device const block_q8_0 *, device const float *, device float *, constant QmvParams &, \
        uint, uint, uint, uint); \
    template [[host_name("qmv_q4k_b" #B "_f32")]] kernel void qmv_q4k_f32<B, NR4>( \
        device const block_q4_k *, device const float *, device float *, constant QmvParams &, \
        uint, uint, uint, uint);
QMV(1, 4, 2)
QMV(2, 4, 2)
QMV(3, 4, 2)
QMV(4, 4, 2)
QMV(5, 4, 2)
QMV(6, 4, 2)
QMV(7, 4, 2)
QMV(8, 4, 2)

// ---------------------------------------------------------------------------------------
// Penalties and sampling (host reference: `crate::sampling`)
// ---------------------------------------------------------------------------------------

struct MaskArgs {
    uint rows;
    uint cols;
    uint stride;
    uint words;
};

// Grammar masks, one thread per (mask, token): token t of logit row rows[r] becomes -inf
// unless bit t % 32 of word t / 32 of mask r is set.
kernel void mask_f32(
    device float *x [[buffer(0)]],
    device const uint *rows [[buffer(1)]],
    device const uint *bits [[buffer(2)]],
    constant MaskArgs &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.cols;
    uint c = gid % p.cols;
    if (r >= p.rows) {
        return;
    }
    if (((bits[r * p.words + c / 32] >> (c % 32)) & 1u) == 0) {
        x[ulong(rows[r]) * p.stride + c] = -INFINITY;
    }
}

struct PenaltyArgs {
    uint entries;
    uint stride;
};

// In place, one thread per distinct context token of a row: entries are (row, token,
// sampled count) and params (repetition, presence, frequency) per row.
kernel void penalize_f32(
    device float *x [[buffer(0)]],
    device const uint *entries [[buffer(1)]],
    device const float *params [[buffer(2)]],
    constant PenaltyArgs &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.entries) {
        return;
    }
    uint row = entries[3 * gid];
    float count = float(entries[3 * gid + 2]);
    device const float *rp = params + 3 * row;
    device float *v = x + ulong(row) * p.stride + entries[3 * gid + 1];
    float l = *v;
    l = l > 0.0f ? l / rp[0] : l * rp[0];
    *v = l - fma(rp[2], count, count > 0.0f ? rp[1] : 0.0f);
}

static inline float threadgroup_max(float v, threadgroup float *scratch, uint sg, uint lane, uint sgs) {
    v = simd_max(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = simd_max(lane < sgs ? scratch[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

static inline uint threadgroup_min_u(uint v, threadgroup uint *scratch, uint sg, uint lane, uint sgs) {
    v = simd_min(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m = simd_min(lane < sgs ? scratch[lane] : UINT_MAX);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

static inline uint threadgroup_max_u(uint v, threadgroup uint *scratch, uint sg, uint lane, uint sgs) {
    v = simd_max(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m = simd_max(lane < sgs ? scratch[lane] : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

// Exclusive prefix sum over threads in index order; `total` receives the full sum.
// Threadgroups are whole simdgroups.
static inline float threadgroup_scan(float v, threadgroup float *scratch, uint sg, uint lane, uint sgs, thread float &total) {
    float inner = simd_prefix_exclusive_sum(v);
    if (lane == SIMD - 1) {
        scratch[sg] = inner + v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = lane < sgs ? scratch[lane] : 0.0f;
    float before = simd_prefix_exclusive_sum(s);
    total = simd_sum(s);
    float base = simd_shuffle(before, ushort(sg));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return base + inner;
}

static inline ulong mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
    return z ^ (z >> 31);
}

static inline float uniform24(ulong seed, ulong draw, uint round) {
    ulong z = seed + draw * 0x9e3779b97f4a7c15ul + ulong(round) * 0xd1b54a32d192ed03ul;
    return float(mix64(z) >> 40) * (1.0f / 16777216.0f);
}

struct SampleArgs {
    uint rows;
    uint cols;
    uint stride;
    uint rounds;
};

// One threadgroup per row. Thread t owns tokens t, t + T, t + 2T, ... (coalesced passes);
// the CDF is inverted in thread-major order (thread 0's tokens, then thread 1's), a fixed
// permutation of the vocabulary, so draws follow the same distribution as the host
// reference. params: 7 words per row (temperature bits, top-k, top-p bits, seed lo/hi,
// draw lo/hi). out[row] = (token, status): status 1 sampled, 0 NaN/+inf or no positive
// weight, 2 not accepted within `rounds` rejection rounds.
kernel void sample_f32(
    device const float *x [[buffer(0)]],
    device const uint *params [[buffer(1)]],
    device float2 *out [[buffer(2)]],
    constant SampleArgs &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float fscratch[SIMD];
    threadgroup uint uscratch[SIMD];
    threadgroup uint chosen;
    threadgroup float chosen_w;
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.stride;
    device const uint *rp = params + 7 * row;
    float inv_t = 1.0f / as_type<float>(rp[0]);
    uint top_k = rp[1];
    float top_p = as_type<float>(rp[2]);
    ulong seed = ulong(rp[3]) | (ulong(rp[4]) << 32);
    ulong draw = ulong(rp[5]) | (ulong(rp[6]) << 32);
    bool truncating = (top_k > 0 && top_k < p.cols) || top_p < 1.0f;

    float m = -INFINITY;
    uint bad = 0;
    for (uint i = tid; i < p.cols; i += threads) {
        float v = xr[i];
        bad |= (isnan(v) || v == INFINITY) ? 1u : 0u;
        m = max(m, v);
    }
    m = threadgroup_max(m, fscratch, sg, lane, sgs);
    bad = threadgroup_max_u(bad, uscratch, sg, lane, sgs);
    if (bad != 0 || m == -INFINITY) {
        if (tid == 0) {
            out[row] = float2(0.0f, 0.0f);
        }
        return;
    }
    // Weights above `low` remain candidates; `s` is this thread's share of their mass.
    float low = 0.0f;
    float s = 0.0f;
    for (uint i = tid; i < p.cols; i += threads) {
        float w = exp((xr[i] - m) * inv_t);
        s += w > low ? w : 0.0f;
    }
    float mass;
    float prefix = threadgroup_scan(s, fscratch, sg, lane, sgs, mass);
    float total = mass;
    uint per_thread = (p.cols + threads - 1) / threads;
    for (uint round = 0; round < p.rounds; ++round) {
        float target = uniform24(seed, draw, round) * mass;
        uint owner = threadgroup_min_u(s > 0.0f && target < prefix + s ? tid : UINT_MAX, uscratch, sg, lane, sgs);
        if (owner == UINT_MAX) {
            owner = threadgroup_max_u(s > 0.0f ? tid : 0u, uscratch, sg, lane, sgs);
        }
        if (tid == owner) {
            uscratch[0] = as_type<uint>(target - prefix);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local = as_type<float>(uscratch[0]);
        // Simdgroup 0 scans the owner's tokens 32 at a time: the first past the target,
        // else (rounding) the last candidate.
        if (sg == 0) {
            float base = 0.0f;
            uint j = UINT_MAX;
            float wj = 0.0f;
            uint fallback = UINT_MAX;
            float fallback_w = 0.0f;
            for (uint k0 = 0; k0 < per_thread && j == UINT_MAX; k0 += SIMD) {
                uint i = owner + (k0 + lane) * threads;
                float w = i < p.cols ? exp((xr[i] - m) * inv_t) : 0.0f;
                w = w > low ? w : 0.0f;
                float r = base + simd_prefix_inclusive_sum(w);
                uint hit = simd_min(w > 0.0f && r > local ? lane : UINT_MAX);
                uint last = simd_max(w > 0.0f ? lane + 1 : 0u);
                if (hit != UINT_MAX) {
                    j = owner + (k0 + hit) * threads;
                    wj = simd_shuffle(w, ushort(hit));
                } else if (last > 0) {
                    fallback = owner + (k0 + last - 1) * threads;
                    fallback_w = simd_shuffle(w, ushort(last - 1));
                }
                base = simd_shuffle(r, ushort(SIMD - 1));
            }
            if (lane == 0) {
                chosen = j != UINT_MAX ? j : fallback;
                chosen_w = j != UINT_MAX ? wj : fallback_w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint j = chosen;
        float wj = chosen_w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!truncating) {
            if (tid == 0) {
                out[row] = float2(float(j), 1.0f);
            }
            return;
        }
        float heavier = 0.0f;
        s = 0.0f;
        for (uint i = tid; i < p.cols; i += threads) {
            float w = exp((xr[i] - m) * inv_t);
            if (w > wj) {
                s += w;
                heavier += 1.0f;
            }
        }
        float count = threadgroup_sum(heavier, fscratch, sg, lane, sgs);
        prefix = threadgroup_scan(s, fscratch, sg, lane, sgs, mass);
        if ((top_k == 0 || count < float(top_k)) && mass < top_p * total) {
            if (tid == 0) {
                out[row] = float2(float(j), 1.0f);
            }
            return;
        }
        low = wj;
    }
    if (tid == 0) {
        out[row] = float2(0.0f, 2.0f);
    }
}
