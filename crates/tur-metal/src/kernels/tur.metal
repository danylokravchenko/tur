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
// slot, offset of its written slots in the pass's slot list, and whether it writes one
// slot per token. Sequences are consecutive and cover every packed row.
struct RecurrentSpan {
    uint row;
    uint tokens;
    uint read;
    uint writes;
    uint every;
};

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

// Gated delta rule over every sequence of a packed pass, token by token, with the state
// held in registers. Threadgroups cover (sequence, head, column block); each simdgroup
// owns COLS value columns of one head and lane l holds key rows l + 32r.
// Per token: L2-normalise q/k, decay the state, read the prediction, write the
// beta-scaled correction and read the output with the query.
//
// The [dk, dv] state is read and written through a padded threadgroup tile so device
// accesses are contiguous rows of the threadgroup's columns; lane-owned rows alone would
// touch 16 bytes per cache line.
template <uint R, uint COLS>
kernel void delta_scan_f32(
    device const float *mixed [[buffer(0)]],
    device const float *proj [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *neg_a [[buffer(3)]],
    device float *pool [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant ScanParams &p [[buffer(6)]],
    device const RecurrentSpan *spans [[buffer(7)]],
    device const uint *slots [[buffer(8)]],
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
    uint sequence = tg / tgs_per_sequence;
    if (sequence >= p.sequences) {
        return;
    }
    RecurrentSpan s = spans[sequence];
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
    device const float *state_in = pool + ulong(s.read) * state_size;

    for (uint idx = tid; idx < dk * tile_cols; idx += threads) {
        uint i = idx / tile_cols;
        uint c = idx % tile_cols;
        uint j = tile_col0 + c;
        tile[i * pitch + c] = j < dv ? state_in[head_state + ulong(i) * dv + j] : 0.0f;
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
                device float *st = pool + ulong(slots[s.writes + t]) * state_size + head_state;
                for (uint r = 0; r < R; ++r) {
                    for (uint c = 0; c < COLS; ++c) {
                        if (col0 + c < dv) {
                            st[ulong(lane + SIMD * r) * dv + col0 + c] = state[r][c];
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
    device float *final_state = pool + ulong(slots[s.writes]) * state_size + head_state;
    for (uint idx = tid; idx < dk * tile_cols; idx += threads) {
        uint i = idx / tile_cols;
        uint c = idx % tile_cols;
        uint j = tile_col0 + c;
        if (j < dv) {
            final_state[ulong(i) * dv + j] = tile[i * pitch + c];
        }
    }
}

#define SCAN(R) \
    template [[host_name("delta_scan_r" #R "_f32")]] kernel void delta_scan_f32<R, 4>( \
        device const float *, device const float *, device const float *, device const float *, \
        device float *, device float *, constant ScanParams &, device const RecurrentSpan *, \
        device const uint *, threadgroup float *, uint, uint, uint, uint, uint, uint);
SCAN(2)
SCAN(4)
SCAN(8)

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
        float quant[NR][8];
        float scale[NR];
        for (uint r = 0; r < NR; ++r) {
            uint row = min(row0 + r, p.n - 1);
            device const block_q8_0 &blk = w[ulong(row) * blocks + ib];
            scale[r] = float(blk.d);
            for (uint i = 0; i < 8; ++i) {
                quant[r][i] = float(blk.qs[il * 8 + i]);
            }
        }
        for (uint b = 0; b < B; ++b) {
            device const float *xb = x + ulong(b) * p.x_stride + ib * 32 + il * 8;
            float xv[8];
            for (uint i = 0; i < 8; ++i) {
                xv[i] = xb[i];
            }
            for (uint r = 0; r < NR; ++r) {
                float dot = 0.0f;
                for (uint i = 0; i < 8; ++i) {
                    dot += quant[r][i] * xv[i];
                }
                acc[r][b] += dot * scale[r];
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
QMV(5, 2, 2)
QMV(6, 2, 2)
QMV(7, 2, 2)
QMV(8, 2, 2)
