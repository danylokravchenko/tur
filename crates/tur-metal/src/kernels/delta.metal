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

static inline float4 conv_input4(
    device const float *x, device const float *history, uint r, uint c, uint kept,
    constant ConvParams &p) {
    return r < kept ? *(device const float4 *)(history + r * p.channels + c)
                    : *(device const float4 *)(x + ulong(r - kept) * p.x_stride + c);
}

// delta_conv_f32 over four consecutive channels per thread, with the same arithmetic per
// channel: one vector load per input row and one span lookup and row division per four
// outputs. The host routes here when channels, the row stride and the input offset keep
// every float4 aligned.
kernel void delta_conv4_f32(
    device const float *x [[buffer(0)]],
    device float *pool [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant ConvParams &p [[buffer(4)]],
    device const RecurrentSpan *spans [[buffer(5)]],
    device const uint *row_span [[buffer(6)]],
    device const uint *slots [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint quads = p.channels / 4;
    if (gid >= p.rows * quads) {
        return;
    }
    uint row = gid / quads;
    uint c = 4 * (gid % quads);
    RecurrentSpan s = spans[row_span[row]];
    uint t = row - s.row;
    uint kept = p.taps - 1;
    device const float *xs = x + ulong(s.row) * p.x_stride;
    device const float *history = pool + ulong(s.read) * kept * p.channels;
    device const float *wc = w + ulong(c) * p.taps;
    float4 acc = conv_input4(xs, history, t, c, kept, p)
        * float4(wc[0], wc[p.taps], wc[2 * p.taps], wc[3 * p.taps]);
    for (uint j = 1; j < p.taps; ++j) {
        float4 wj = float4(wc[j], wc[p.taps + j], wc[2 * p.taps + j], wc[3 * p.taps + j]);
        acc = acc + conv_input4(xs, history, t + j, c, kept, p) * wj;
    }
    *(device float4 *)(out + ulong(row) * p.channels + c) =
        float4(silu(acc.x), silu(acc.y), silu(acc.z), silu(acc.w));
    if (s.every != 0 || t + 1 == s.tokens) {
        ulong base = ulong(slots[s.writes + (s.every != 0 ? t : 0)]) * kept * p.channels;
        for (uint r = 0; r < kept; ++r) {
            *(device float4 *)(pool + base + ulong(r) * p.channels + c) =
                conv_input4(xs, history, t + 1 + r, c, kept, p);
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
    uint prep_stride;
};

// Per-token inputs of the scan, computed once per token instead of once per state column:
// row r of `prep` holds L2-normalised q (also scaled by 1/sqrt(key_dim)) and k per key
// head, then q·k per key head, then (decay, beta) per value head. One simdgroup per
// (row, key head); the gate columns of its value heads go to its first lanes.
kernel void delta_prepare_f32(
    device const float *mixed [[buffer(0)]],
    device const float *proj [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *neg_a [[buffer(3)]],
    device float *prep [[buffer(4)]],
    constant ScanParams &p [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint key_heads = p.heads / p.groups;
    uint item = tg * sgs + sg;
    if (item >= rows * key_heads) {
        return;
    }
    uint row = item / key_heads;
    uint kh = item % key_heads;
    uint dk = p.key_dim;
    device const float *q = mixed + ulong(row) * p.mixed_stride + kh * dk;
    device const float *k = q + p.key_width;
    float qs = 0.0f, ks = 0.0f;
    for (uint i = lane; i < dk; i += SIMD) {
        qs += q[i] * q[i];
        ks += k[i] * k[i];
    }
    float q_scale = 1.0f / (sqrt(simd_sum(qs) + 1e-6f) * sqrt(float(dk)));
    float k_scale = 1.0f / sqrt(simd_sum(ks) + 1e-6f);
    device float *out = prep + ulong(row) * p.prep_stride;
    float qk = 0.0f;
    for (uint i = lane; i < dk; i += SIMD) {
        float qn = q[i] * q_scale, kn = k[i] * k_scale;
        out[kh * dk + i] = qn;
        out[p.key_width + kh * dk + i] = kn;
        qk += qn * kn;
    }
    qk = simd_sum(qk);
    if (lane == 0) {
        out[2 * p.key_width + kh] = qk;
    }
    device const float *gates = proj + ulong(row) * p.proj_stride;
    for (uint g = lane; g < p.groups; g += SIMD) {
        uint h = kh * p.groups + g;
        device float *db = out + 2 * p.key_width + key_heads + 2 * h;
        db[0] = exp(softplus(gates[p.a_col + h] + dt_bias[h]) * neg_a[h]);
        db[1] = sigmoid(gates[p.b_col + h]);
    }
}

// Gated delta rule over the sequences `order` lists, token by token, with the state held
// in registers. Lane l of a simdgroup owns one value column and ROWS consecutive key
// rows: P = key_dim / ROWS lanes share a column, so a simdgroup covers 32 / P columns of
// one head. 32 rows measured fastest on M1 (64 doubles live registers and halves the
// simdgroups; 16 adds a shuffle step per reduction).
// Per token, with S the column's decayed-before state and q, k from `prep`,
//   a = k·S, b = q·S, pred = decay·a, corr = beta·(v - pred),
//   out = q·S' = decay·b + (q·k)·corr,  S' = decay·S + k·corr,
// so both dot products are lane-local over the lane's rows and combine across the P lanes
// of a column with log2(P) shuffles; nothing is recomputed per column and no threadgroup
// memory or barrier is needed. State loads and stores are row segments across a
// simdgroup's columns. Every loop over the state is fully unrolled: a dynamic index
// would spill the state registers to thread memory. State is stored as T (float,
// half or bfloat) and updated in F32.
template <uint ROWS, typename T>
kernel void delta_scan(
    device const float *mixed [[buffer(0)]],
    device const float *prep [[buffer(1)]],
    device T *pool [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant ScanParams &p [[buffer(4)]],
    device const RecurrentSpan *spans [[buffer(5)]],
    device const uint *slots [[buffer(6)]],
    device const uint *order [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint dk = p.key_dim;
    uint columns = SIMD * ROWS / dk;
    uint dv = p.value_dim;
    uint blocks = (dv + columns - 1) / columns;
    uint per_sequence = p.heads * blocks;
    uint item = tg * sgs + sg;
    uint index = item / per_sequence;
    if (index >= p.sequences) {
        return;
    }
    RecurrentSpan s = spans[order[index]];
    uint h = (item % per_sequence) / blocks;
    uint col = (item % blocks) * columns + lane % columns;
    bool active = col < dv;
    // Inactive lanes mirror the last column so every load stays in bounds.
    uint column = min(col, dv - 1);
    uint r0 = (lane / columns) * ROWS;
    uint kh = h / p.groups;
    uint key_heads = p.heads / p.groups;
    ulong state_size = ulong(p.heads) * dk * dv;
    ulong head_state = ulong(h) * dk * dv;

    float state[ROWS];
    device const T *state_in = pool + ulong(s.read) * state_size + head_state + column;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ROWS; ++i) {
        state[i] = float(state_in[ulong(r0 + i) * dv]);
    }
    uint q_col = kh * dk + r0;
    uint k_col = p.key_width + kh * dk + r0;
    uint qk_col = 2 * p.key_width + kh;
    uint gate_col = 2 * p.key_width + key_heads + 2 * h;
    uint v_col = 2 * p.key_width + h * dv + column;
    for (uint t = 0; t < s.tokens; ++t) {
        uint row = s.row + t;
        device const float *pr = prep + ulong(row) * p.prep_stride;
        device const float4 *qv = (device const float4 *)(pr + q_col);
        device const float4 *kv = (device const float4 *)(pr + k_col);
        float decay = pr[gate_col], beta = pr[gate_col + 1], qk = pr[qk_col];
        float v = mixed[ulong(row) * p.mixed_stride + v_col];
        float4 a4 = 0.0f, b4 = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ROWS / 4; ++i) {
            float4 st = float4(state[4 * i], state[4 * i + 1], state[4 * i + 2], state[4 * i + 3]);
            a4 = fma(kv[i], st, a4);
            b4 = fma(qv[i], st, b4);
        }
        float a = (a4.x + a4.y) + (a4.z + a4.w);
        float b = (b4.x + b4.y) + (b4.z + b4.w);
        for (uint offset = columns; offset < SIMD; offset *= 2) {
            a += simd_shuffle_xor(a, ushort(offset));
            b += simd_shuffle_xor(b, ushort(offset));
        }
        float correction = (v - decay * a) * beta;
        if (r0 == 0 && active) {
            out[(ulong(row) * p.heads + h) * dv + col] = decay * b + qk * correction;
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ROWS / 4; ++i) {
            float4 kc = kv[i] * correction;
            state[4 * i] = fma(state[4 * i], decay, kc.x);
            state[4 * i + 1] = fma(state[4 * i + 1], decay, kc.y);
            state[4 * i + 2] = fma(state[4 * i + 2], decay, kc.z);
            state[4 * i + 3] = fma(state[4 * i + 3], decay, kc.w);
        }
        // Per-token states serve speculative verification.
        if (s.every != 0 && active) {
            device T *st = pool + ulong(slots[s.writes + t]) * state_size + head_state + col;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < ROWS; ++i) {
                ulong at = ulong(r0 + i) * dv;
                st[at] = store_state<T>(state[i], s.position + t, uint(head_state + at + col));
            }
        }
    }
    if (s.every != 0 || !active) {
        return;
    }
    device T *final_state = pool + ulong(slots[s.writes]) * state_size + head_state + col;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ROWS; ++i) {
        ulong at = ulong(r0 + i) * dv;
        final_state[at] = store_state<T>(state[i], s.position + s.tokens - 1, uint(head_state + at + col));
    }
}

// Key rows per delta-scan lane (`DELTA_SCAN_ROWS` on the host); key_dim / SCAN_ROWS lanes share
// a value column.
#define SCAN_ROWS 32
#define SCAN(T, NAME) \
    template [[host_name("delta_scan_" #NAME)]] kernel void delta_scan<SCAN_ROWS, T>( \
        device const float *, device const float *, device T *, device float *, \
        constant ScanParams &, device const RecurrentSpan *, device const uint *, \
        device const uint *, uint, uint, uint, uint);
SCAN(float, f32)
SCAN(half, f16)
SCAN(bfloat, bf16)

// One gated delta step for each single-token sequence `order` lists. A threadgroup covers
// (sequence, head, 64 value columns) with DK / ROWS simdgroups: lane l owns value columns
// col0 + 2l and col0 + 2l + 1 and simdgroup g owns key rows ROWS·g..ROWS·g+ROWS-1, so every
// state load and store is a coalesced row segment. The two column reductions (prediction
// and output) sum the simdgroups' partials through threadgroup memory. Fewer rows per
// simdgroup keep fewer state registers per lane, so more simdgroups stay resident to hide
// memory latency. State is stored as T (float, half or bfloat) and updated in F32.
template <uint DK, uint ROWS, typename T>
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
    constexpr uint GROUPS = DK / ROWS;
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
    uint i0 = sg * ROWS;
    float2 st[ROWS];
    float2 prediction = 0.0f;
    for (uint r = 0; r < ROWS; ++r) {
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
    for (uint r = 0; r < ROWS; ++r) {
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

// Key rows per delta-step simdgroup: 16 is fastest on M1; 32 is the fallback for
// pipelines that cannot run DK / 16 simdgroups (DELTA_STEP_ROWS on the host).
#define STEP(DK, ROWS, T, NAME) \
    template [[host_name("delta_step_d" #DK "_r" #ROWS "_" #NAME)]] kernel void delta_step<DK, ROWS, T>( \
        device const float *, device const float *, device const float *, device const float *, \
        device T *, device float *, constant ScanParams &, device const RecurrentSpan *, \
        device const uint *, device const uint *, threadgroup float2 *, uint, uint, uint);
#define STEP_ROWS(DK, T, NAME) STEP(DK, 16, T, NAME) STEP(DK, 32, T, NAME)
#define STEPS(T, NAME) STEP_ROWS(64, T, NAME) STEP_ROWS(128, T, NAME) STEP_ROWS(256, T, NAME)
STEPS(float, f32)
STEPS(half, f16)
STEPS(bfloat, bf16)

