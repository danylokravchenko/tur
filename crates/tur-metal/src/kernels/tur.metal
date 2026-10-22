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
    uint tokens;
    uint channels;
    uint taps;
    uint x_stride;
    uint every;
};

static inline float conv_input(
    device const float *x, device const float *history, uint r, uint c, constant ConvParams &p) {
    uint kept = p.taps - 1;
    return r < kept ? history[r * p.channels + c] : x[ulong(r - kept) * p.x_stride + c];
}

// Depthwise causal convolution of one sequence over [history; x], then SiLU. Writes the
// final (taps-1)-row history, or one history per token when `every` is set.
kernel void delta_conv_f32(
    device const float *x [[buffer(0)]],
    device const float *history [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    device float *history_out [[buffer(4)]],
    constant ConvParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.tokens * p.channels) {
        return;
    }
    uint t = gid / p.channels;
    uint c = gid % p.channels;
    device const float *wc = w + ulong(c) * p.taps;
    float acc = conv_input(x, history, t, c, p) * wc[0];
    for (uint j = 1; j < p.taps; ++j) {
        acc = acc + conv_input(x, history, t + j, c, p) * wc[j];
    }
    out[gid] = silu(acc);
    if (p.every != 0 || t + 1 == p.tokens) {
        uint kept = p.taps - 1;
        ulong base = ulong(p.every != 0 ? t : 0) * kept * p.channels;
        for (uint r = 0; r < kept; ++r) {
            history_out[base + ulong(r) * p.channels + c] = conv_input(x, history, t + 1 + r, c, p);
        }
    }
}

struct ScanParams {
    uint tokens;
    uint heads;
    uint groups;
    uint key_width;
    uint key_dim;
    uint value_dim;
    uint mixed_stride;
    uint proj_stride;
    uint b_col;
    uint a_col;
    uint every;
};

// Gated delta rule over one sequence, token by token, with the state held in registers.
// Each simdgroup owns COLS value columns of one head; lane l holds key rows l + 32r.
// Per token: L2-normalise q/k, decay the state, read the prediction, write the
// beta-scaled correction and read the output with the query.
//
// Threadgroups cover consecutive column groups of one head. The [dk, dv] state is read
// and written through a padded threadgroup tile so device accesses are contiguous rows of
// the threadgroup's columns; lane-owned rows alone would touch 16 bytes per cache line.
template <uint R, uint COLS>
kernel void delta_scan_f32(
    device const float *mixed [[buffer(0)]],
    device const float *proj [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *neg_a [[buffer(3)]],
    device const float *state_in [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *state_out [[buffer(6)]],
    constant ScanParams &p [[buffer(7)]],
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
    uint h = tg / tgs_per_head;
    if (h >= p.heads) {
        return;
    }
    uint tile_cols = groups_per_tg * COLS;
    uint pitch = tile_cols + 1;
    uint tile_col0 = (tg % tgs_per_head) * tile_cols;
    uint cg = (tg % tgs_per_head) * groups_per_tg + sg;
    bool active = cg < column_groups;
    uint col0 = cg * COLS;
    uint kh = h / p.groups;
    ulong head_state = ulong(h) * dk * dv;
    ulong state_size = ulong(p.heads) * dk * dv;

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
        if (p.tokens > 0) {
            for (uint r = 0; r < R; ++r) {
                q_next[r] = mixed[q_col + lane + SIMD * r];
                k_next[r] = mixed[k_col + lane + SIMD * r];
            }
            for (uint c = 0; c < COLS; ++c) {
                v_next[c] = col0 + c < dv ? mixed[v_col + c] : 0.0f;
            }
            a_next = proj[p.a_col + h];
            b_next = proj[p.b_col + h];
        }
        for (uint t = 0; t < p.tokens; ++t) {
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
            if (t + 1 < p.tokens) {
                device const float *row = mixed + ulong(t + 1) * p.mixed_stride;
                for (uint r = 0; r < R; ++r) {
                    q_next[r] = row[q_col + lane + SIMD * r];
                    k_next[r] = row[k_col + lane + SIMD * r];
                }
                for (uint c = 0; c < COLS; ++c) {
                    v_next[c] = col0 + c < dv ? row[v_col + c] : 0.0f;
                }
                device const float *pr = proj + ulong(t + 1) * p.proj_stride;
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
            device float *o = out + (ulong(t) * p.heads + h) * dv + col0;
            for (uint c = 0; c < COLS; ++c) {
                if (lane == c && col0 + c < dv) {
                    o[c] = output[c];
                }
            }
            // Per-token states serve speculative verification only; they skip staging.
            if (p.every != 0) {
                device float *s = state_out + ulong(t) * state_size + head_state;
                for (uint r = 0; r < R; ++r) {
                    for (uint c = 0; c < COLS; ++c) {
                        if (col0 + c < dv) {
                            s[ulong(lane + SIMD * r) * dv + col0 + c] = state[r][c];
                        }
                    }
                }
            }
        }
    }
    if (p.every != 0) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < R; ++r) {
        for (uint c = 0; c < COLS; ++c) {
            tile[(lane + SIMD * r) * pitch + sg * COLS + c] = state[r][c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < dk * tile_cols; idx += threads) {
        uint i = idx / tile_cols;
        uint c = idx % tile_cols;
        uint j = tile_col0 + c;
        if (j < dv) {
            state_out[head_state + ulong(i) * dv + j] = tile[i * pitch + c];
        }
    }
}

#define SCAN(R) \
    template [[host_name("delta_scan_r" #R "_f32")]] kernel void delta_scan_f32<R, 4>( \
        device const float *, device const float *, device const float *, device const float *, \
        device const float *, device float *, device float *, constant ScanParams &, \
        threadgroup float *, uint, uint, uint, uint, uint, uint);
SCAN(2)
SCAN(4)
SCAN(8)

// ---------------------------------------------------------------------------------------
// Attention: ragged queries over a paged K/V pool, direct grouped-query addressing.
// ---------------------------------------------------------------------------------------

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
    float scale;
};

// One simdgroup per (head, split, token); lane l owns dimensions l + 32r. Online softmax
// keeps the running max/sum in F32. With one split the gated output is written directly;
// otherwise unnormalised partial sums and (max, sum) statistics go to scratch.
template <uint R>
kernel void attention_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const uint *table [[buffer(3)]],
    device const uint *first_block [[buffer(4)]],
    device const uint *visible [[buffer(5)]],
    device const float *gate [[buffer(6)]],
    device float *out [[buffer(7)]],
    device float *partial [[buffer(8)]],
    device float *stats [[buffer(9)]],
    constant AttentionParams &p [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
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

    float qv[R], acc[R];
    device const float *qr = q + (ulong(t) * p.heads + h) * p.dim;
    for (uint r = 0; r < R; ++r) {
        qv[r] = qr[lane + SIMD * r];
        acc[r] = 0.0f;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    for (uint j = begin; j < end; ++j) {
        uint slot = table[base + j / p.page] * p.page + j % p.page;
        ulong row = (ulong(slot) * p.kv_heads + g) * p.dim;
        device const float *kr = k + row;
        float dot = 0.0f;
        for (uint r = 0; r < R; ++r) {
            dot += qv[r] * kr[lane + SIMD * r];
        }
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m, score);
        float correction = exp(m - m_new);
        float weight = exp(score - m_new);
        l = l * correction + weight;
        device const float *vr = v + row;
        for (uint r = 0; r < R; ++r) {
            acc[r] = acc[r] * correction + weight * vr[lane + SIMD * r];
        }
        m = m_new;
    }
    if (p.splits == 1) {
        device const float *gr = gate + ulong(t) * p.gate_stride + p.gate_col + ulong(h) * p.gate_head_stride;
        device float *o = out + (ulong(t) * p.heads + h) * p.dim;
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint r = 0; r < R; ++r) {
            uint i = lane + SIMD * r;
            o[i] = acc[r] * inv * sigmoid(gr[i]);
        }
        return;
    }
    ulong slot = (ulong(t) * p.heads + h) * p.splits + split;
    device float *pp = partial + slot * p.dim;
    for (uint r = 0; r < R; ++r) {
        pp[lane + SIMD * r] = acc[r];
    }
    if (lane == 0) {
        stats[2 * slot] = m;
        stats[2 * slot + 1] = l;
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

#define ATTENTION(R) \
    template [[host_name("attention_r" #R "_f32")]] kernel void attention_f32<R>( \
        device const float *, device const float *, device const float *, device const uint *, \
        device const uint *, device const uint *, device const float *, device float *, \
        device float *, device float *, constant AttentionParams &, uint, uint, uint, uint); \
    template [[host_name("attention_reduce_r" #R "_f32")]] kernel void attention_reduce_f32<R>( \
        device const float *, device const float *, device const float *, device float *, \
        constant AttentionParams &, uint, uint, uint, uint);
ATTENTION(2)
ATTENTION(4)
ATTENTION(8)

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
