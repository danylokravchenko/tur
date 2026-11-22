// ---------------------------------------------------------------------------------------
// Inkling 2: attention residuals over block states, loop-pass message gates, short causal
// convolutions with slot-addressed history (normalizing keys and values per head), and paged
// attention with in-place query clips, a learned relative position bias, per-token sinks and
// a sigmoid output gate.
// ---------------------------------------------------------------------------------------

#define RESIDUAL_ENTRIES 8u
#define RESIDUAL_QUERIES 4u
#define RESIDUAL_THREADS 256u

struct InklingResidualParams {
    uint tokens;
    uint width;
    uint blocks;
    uint queries;
    uint partial_stride;
    float eps;
    float scale;
};

// Attention of `queries` learned vectors over a token's block states and its partial state,
// each RMS-normalized without a gain: out[q][t] = sum_s softmax_s(q . n_s * scale) n_s.
// Block state s is its own contiguous [tokens, width] buffer b<s>, read in place. One
// threadgroup per token reads every state once for all norms and scores, then once more
// for the outputs, written [queries, tokens, width].
kernel void inkling_block_residual_f32(
    device const float *partial [[buffer(0)]],
    device const float *queries [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant InklingResidualParams &p [[buffer(3)]],
    device const float *b0 [[buffer(4)]],
    device const float *b1 [[buffer(5)]],
    device const float *b2 [[buffer(6)]],
    device const float *b3 [[buffer(7)]],
    device const float *b4 [[buffer(8)]],
    device const float *b5 [[buffer(9)]],
    device const float *b6 [[buffer(10)]],
    uint t [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SUMS = RESIDUAL_ENTRIES * (RESIDUAL_QUERIES + 1);
    threadgroup float shared[RESIDUAL_THREADS / SIMD][SUMS];
    threadgroup float weights[RESIDUAL_QUERIES][RESIDUAL_ENTRIES];
    uint entries = p.blocks + 1;
    ulong row = ulong(t) * p.width;
    device const float *states[RESIDUAL_ENTRIES] = {
        b0 + row, b1 + row, b2 + row, b3 + row, b4 + row, b5 + row, b6 + row, partial};
    states[p.blocks] = partial + ulong(t) * p.partial_stride;
    // Per entry: its sum of squares, then its dot product with each query.
    float local[SUMS];
    for (uint i = 0; i < SUMS; ++i) {
        local[i] = 0.0f;
    }
    for (uint d = tid; d < p.width; d += RESIDUAL_THREADS) {
        for (uint s = 0; s < entries; ++s) {
            float v = states[s][d];
            local[s] = fma(v, v, local[s]);
            for (uint q = 0; q < p.queries; ++q) {
                uint at = RESIDUAL_ENTRIES * (q + 1) + s;
                local[at] = fma(queries[q * p.width + d], v, local[at]);
            }
        }
    }
    for (uint i = 0; i < SUMS; ++i) {
        float v = simd_sum(local[i]);
        if (lane == 0) {
            shared[sg][i] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.queries) {
        float inv[RESIDUAL_ENTRIES], score[RESIDUAL_ENTRIES];
        float top = -INFINITY;
        for (uint s = 0; s < entries; ++s) {
            float squares = 0.0f, dot = 0.0f;
            for (uint g = 0; g < RESIDUAL_THREADS / SIMD; ++g) {
                squares += shared[g][s];
                dot += shared[g][RESIDUAL_ENTRIES * (tid + 1) + s];
            }
            inv[s] = rsqrt(squares / float(p.width) + p.eps);
            score[s] = dot * inv[s] * p.scale;
            top = max(top, score[s]);
        }
        float total = 0.0f;
        for (uint s = 0; s < entries; ++s) {
            score[s] = exp(score[s] - top);
            total += score[s];
        }
        // The weight of an entry already carries its normalization.
        for (uint s = 0; s < entries; ++s) {
            weights[tid][s] = score[s] / total * inv[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.width; d += RESIDUAL_THREADS) {
        float acc[RESIDUAL_QUERIES] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint s = 0; s < entries; ++s) {
            float v = states[s][d];
            for (uint q = 0; q < p.queries; ++q) {
                acc[q] = fma(weights[q][s], v, acc[q]);
            }
        }
        for (uint q = 0; q < p.queries; ++q) {
            out[(ulong(q) * p.tokens + t) * p.width + d] = acc[q];
        }
    }
}

struct InklingConvParams {
    uint rows;
    uint channels;
    uint taps;
    uint x_stride;
    // Floats per history row of a recurrent slot, and the first of them this convolution owns.
    uint pool_width;
    uint offset;
    uint dynamic;
    uint dyn_stride;
    uint dyn_col;
    uint head_dim;
    float divisor;
    // Nonzero: add row r of `residual` (stride res_stride) to the output.
    uint residual;
    uint res_stride;
    // The norm kernel: 0 clips each head's L2 norm to `threshold`, 1 divides it by its RMS.
    uint norm;
    float threshold;
    float eps;
};

static inline float short_conv_input(
    device const float *x, device const float *history, uint r, uint c, uint kept,
    constant InklingConvParams &p) {
    return r < kept ? history[r * p.pool_width + c] : x[ulong(r - kept) * p.x_stride + c];
}

// Channel c of packed row `row`: the depthwise causal convolution over [history; x] plus the
// input, divided by `divisor`. `taps` is [channels, taps] by lag (0 = the token itself).
// With `dynamic`, each token adds its own taps per head of `head_dim` channels from `dyn`
// (head h, lag l at column dyn_col + h * taps + l). Each sequence writes the (taps-1)-row
// history after each token its written slots list, as delta_conv_f32 does, into columns
// offset..offset + channels of the slot rows.
static inline float short_conv_channel(
    device const float *x, device float *pool, device const float *taps,
    constant InklingConvParams &p, device const RecurrentSpan *spans,
    device const uint *row_span, device const uint *slots, device const float *dyn,
    uint row, uint c) {
    RecurrentSpan s = spans[row_span[row]];
    uint t = row - s.row;
    uint kept = p.taps - 1;
    device const float *xs = x + ulong(s.row) * p.x_stride;
    device const float *history = pool + ulong(s.read) * kept * p.pool_width + p.offset;
    device const float *own = dyn + ulong(row) * p.dyn_stride + p.dyn_col + (c / p.head_dim) * p.taps;
    float acc = 0.0f;
    for (uint lag = 0; lag < p.taps; ++lag) {
        float w = taps[c * p.taps + lag] + (p.dynamic != 0 ? own[lag] : 0.0f);
        acc = fma(w, short_conv_input(xs, history, t + kept - lag, c, kept, p), acc);
    }
    uint written = written_after(s, slots, t);
    if (written < s.count) {
        ulong base = ulong(slots[s.writes + written]) * kept * p.pool_width + p.offset;
        for (uint r = 0; r < kept; ++r) {
            pool[base + ulong(r) * p.pool_width + c] = short_conv_input(xs, history, t + 1 + r, c, kept, p);
        }
    }
    return (acc + xs[ulong(t) * p.x_stride + c]) / p.divisor;
}

// Every channel of every row, one thread each, plus the residual row when given.
kernel void inkling_short_conv_f32(
    device const float *x [[buffer(0)]],
    device float *pool [[buffer(1)]],
    device const float *taps [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant InklingConvParams &p [[buffer(4)]],
    device const RecurrentSpan *spans [[buffer(5)]],
    device const uint *row_span [[buffer(6)]],
    device const uint *slots [[buffer(7)]],
    device const float *dyn [[buffer(8)]],
    device const float *residual [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.channels) {
        return;
    }
    uint row = gid / p.channels, c = gid % p.channels;
    float v = short_conv_channel(x, pool, taps, p, spans, row_span, slots, dyn, row, c);
    out[gid] = p.residual != 0 ? v + residual[ulong(row) * p.res_stride + c] : v;
}

// One threadgroup per (row, head of head_dim channels), a thread per channel: the
// convolution, then the head's L2 clip or RMS norm, so the attention reads normalized keys
// and values without another pass.
kernel void inkling_short_conv_norm_f32(
    device const float *x [[buffer(0)]],
    device float *pool [[buffer(1)]],
    device const float *taps [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant InklingConvParams &p [[buffer(4)]],
    device const RecurrentSpan *spans [[buffer(5)]],
    device const uint *row_span [[buffer(6)]],
    device const uint *slots [[buffer(7)]],
    device const float *dyn [[buffer(8)]],
    uint item [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[SIMD];
    uint heads = p.channels / p.head_dim;
    uint row = item / heads, c = (item % heads) * p.head_dim + d;
    float v = short_conv_channel(x, pool, taps, p, spans, row_span, slots, dyn, row, c);
    float sum = threadgroup_sum(v * v, scratch, sg, lane, sgs);
    float factor = p.norm == 0 ? min(p.threshold / (sqrt(sum) + 1e-6f), 1.0f)
                               : rsqrt(sum / float(p.head_dim) + p.eps);
    out[ulong(row) * p.channels + c] = v * factor;
}

struct InklingAttentionParams {
    uint tokens;
    uint heads;
    uint kv_heads;
    uint dim;
    uint page;
    uint group;
    float scale;
    uint window;
    uint rel_dim;
    uint extent;
    uint rel_stride;
    uint rel_col;
    uint sink_stride;
    uint sink_col;
    uint gate_stride;
    uint gate_col;
    uint splits;
    uint split_keys;
    // Queries are read from projection rows at q_col + h * dim, scaled by q_scale and
    // shrunk to L2 norm q_threshold; relative states likewise with rel_scale and
    // rel_threshold. With log_scaled both grow by 1 + log_alpha * ln(max((pos + 1) / floor, 1)).
    uint q_stride;
    uint q_col;
    float q_threshold;
    float rel_threshold;
    uint log_scaled;
    float log_alpha;
    float log_floor;
};

// Scale factor of a head whose scaled values have squared L2 norm `squares`: shrink to
// `threshold`, then grow by the token's log scaling.
static inline float clip_factor(float squares, float threshold, float tau) {
    return min(threshold / (sqrt(squares) + 1e-6f), 1.0f) * tau;
}

// Paged attention of one simdgroup per (token, head, split), as attention_kv reads pages.
// Splits partition each query's own window, so short batches still fill the GPU at long
// contexts; with one split the kernel writes the gated output, otherwise unnormalized
// partial sums and (max, sum) statistics that inkling_attention_reduce combines. Each
// score adds a bias by backward distance d = position - key when d < extent: the token's
// relative state (rel_dim values per head, one per lane) times column d of `proj`, summed in
// the same simdgroup reduction as q . k. A per-token sink logit joins each softmax as a key
// without a value, and the output is gated by sigmoid(gate).
template <uint R, typename T, uint QUANT>
kernel void inkling_attention(
    device const float *q [[buffer(0)]],
    device const T *k [[buffer(1)]],
    device const T *v [[buffer(2)]],
    device const uint *table [[buffer(3)]],
    device const uint *first_block [[buffer(4)]],
    device const uint *visible [[buffer(5)]],
    device const float *rel [[buffer(6)]],
    device const float *proj [[buffer(7)]],
    device const float *sinks [[buffer(8)]],
    device const float *gate [[buffer(9)]],
    device float *out [[buffer(10)]],
    constant InklingAttentionParams &p [[buffer(11)]],
    device const half *k_scales [[buffer(12)]],
    device const half *v_scales [[buffer(13)]],
    device float *partial [[buffer(14)]],
    device float *stats [[buffer(15)]],
    device const float *q_scale [[buffer(16)]],
    device const float *rel_scale [[buffer(17)]],
    device const uint *positions [[buffer(18)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    constexpr uint U = ATTENTION_KEYS;
    constexpr uint V = R * sizeof(T) <= 16 ? R : 16 / sizeof(T);
    constexpr uint C = R / V;
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.heads * p.splits) {
        return;
    }
    uint h = item % p.heads, rest = item / p.heads;
    uint split = rest % p.splits, t = rest / p.splits;
    uint g = h / (p.heads / p.kv_heads);
    uint keys = visible[t];
    uint first_key = p.window > 0 && keys > p.window ? keys - p.window : 0u;
    uint begin = first_key + split * p.split_keys;
    uint end = min(keys, begin + p.split_keys);
    uint base = first_block[t];
    uint groups = QUANT == 0 ? 0 : p.dim / p.group;
    uint dims[C], sidx[C];
    float qv[C][V], acc[C][V];
    device const float *qr = q + ulong(t) * p.q_stride + p.q_col + ulong(h) * p.dim;
    device const float *qs = q_scale + ulong(h) * p.dim;
    float squares = 0.0f;
    for (uint c = 0; c < C; ++c) {
        dims[c] = V * lane + SIMD * V * c;
        sidx[c] = QUANT == 0 ? 0 : dims[c] / p.group;
        kv_dims<V>(qr + dims[c], qv[c]);
        for (uint i = 0; i < V; ++i) {
            qv[c][i] *= qs[dims[c] + i];
            squares = fma(qv[c][i], qv[c][i], squares);
            acc[c][i] = 0.0f;
        }
    }
    float tau = p.log_scaled != 0
        ? 1.0f + p.log_alpha * log(max(float(positions[t] + 1) / p.log_floor, 1.0f))
        : 1.0f;
    float q_factor = clip_factor(simd_sum(squares), p.q_threshold, tau);
    for (uint c = 0; c < C; ++c) {
        for (uint i = 0; i < V; ++i) {
            qv[c][i] *= q_factor;
        }
    }
    // Lane r < rel_dim keeps component r of the relative state and reads row r of `proj`.
    bool rel_lane = lane < p.rel_dim;
    float rel_value = rel_lane
        ? rel[ulong(t) * p.rel_stride + p.rel_col + h * p.rel_dim + lane] * rel_scale[h * p.rel_dim + lane]
        : 0.0f;
    rel_value *= clip_factor(simd_sum(rel_value * rel_value), p.rel_threshold, tau);
    device const float *proj_row = proj + ulong(rel_lane ? lane : 0) * p.extent;
    // The sink joins the softmax as a key without a value: here with one split, in the
    // reduction otherwise.
    bool sink = p.splits == 1;
    float m = sink ? sinks[ulong(t) * p.sink_stride + p.sink_col + h] : -FLT_MAX;
    float l = sink ? 1.0f : 0.0f;
    for (uint j0 = begin; j0 < end; j0 += U) {
        uint rows[U];
        for (uint u = 0; u < U; ++u) {
            uint j = min(j0 + u, end - 1);
            rows[u] = (table[base + j / p.page] * p.page + j % p.page) * p.kv_heads + g;
        }
        float score[U];
        for (uint u = 0; u < U; ++u) {
            device const T *kr = k + ulong(rows[u]) * p.dim;
            float d = 0.0f;
            for (uint c = 0; c < C; ++c) {
                float kd[V];
                kv_dims<V>(kr + dims[c], kd);
                float ks = QUANT == 0 ? 1.0f : float(k_scales[ulong(rows[u]) * groups + sidx[c]]);
                float part = 0.0f;
                for (uint i = 0; i < V; ++i) {
                    part = fma(qv[c][i], kd[i], part);
                }
                d += part * ks;
            }
            uint distance = keys - 1 - min(j0 + u, end - 1);
            float bias = rel_lane && distance < p.extent ? rel_value * proj_row[distance] : 0.0f;
            score[u] = simd_sum(fma(d, p.scale, bias));
        }
        float m_new = m;
        for (uint u = 0; u < U; ++u) {
            m_new = j0 + u < end ? max(m_new, score[u]) : m_new;
        }
        float correction = exp(m - m_new);
        float weight[U];
        float total = 0.0f;
        for (uint u = 0; u < U; ++u) {
            weight[u] = j0 + u < end ? exp(score[u] - m_new) : 0.0f;
            total += weight[u];
        }
        l = l * correction + total;
        for (uint c = 0; c < C; ++c) {
            for (uint i = 0; i < V; ++i) {
                acc[c][i] *= correction;
            }
        }
        m = m_new;
        for (uint u = 0; u < U; ++u) {
            device const T *vr = v + ulong(rows[u]) * p.dim;
            for (uint c = 0; c < C; ++c) {
                float vd[V];
                kv_dims<V>(vr + dims[c], vd);
                float vs = QUANT == 0 ? 1.0f : float(v_scales[ulong(rows[u]) * groups + sidx[c]]);
                float w = weight[u] * vs;
                for (uint i = 0; i < V; ++i) {
                    acc[c][i] = fma(w, vd[i], acc[c][i]);
                }
            }
        }
    }
    if (p.splits > 1) {
        ulong slot = (ulong(t) * p.heads + h) * p.splits + split;
        device float *pp = partial + slot * p.dim;
        for (uint c = 0; c < C; ++c) {
            for (uint i = 0; i < V; ++i) {
                pp[dims[c] + i] = acc[c][i];
            }
        }
        if (lane == 0) {
            stats[2 * slot] = m;
            stats[2 * slot + 1] = l;
        }
        return;
    }
    device const float *gr = gate + ulong(t) * p.gate_stride + p.gate_col + ulong(h) * p.dim;
    device float *o = out + (ulong(t) * p.heads + h) * p.dim;
    for (uint c = 0; c < C; ++c) {
        for (uint i = 0; i < V; ++i) {
            uint d = dims[c] + i;
            o[d] = acc[c][i] / l * (1.0f / (1.0f + exp(-gr[d])));
        }
    }
}

// Combine the split partials of one (token, head) with its sink logit, then apply the
// sigmoid output gate. One simdgroup per (token, head); lane l owns dimensions l + 32 r.
template <uint R>
kernel void inkling_attention_reduce(
    device const float *partial [[buffer(0)]],
    device const float *stats [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device const float *gate [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant InklingAttentionParams &p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.heads) {
        return;
    }
    uint t = item / p.heads, h = item % p.heads;
    ulong first = ulong(item) * p.splits;
    float sink = sinks[ulong(t) * p.sink_stride + p.sink_col + h];
    float m = sink;
    for (uint s = 0; s < p.splits; ++s) {
        if (stats[2 * (first + s) + 1] > 0.0f) {
            m = max(m, stats[2 * (first + s)]);
        }
    }
    float l = exp(sink - m);
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
                acc[r] = fma(pp[lane + SIMD * r], factor, acc[r]);
            }
        }
    }
    device const float *gr = gate + ulong(t) * p.gate_stride + p.gate_col + ulong(h) * p.dim;
    device float *o = out + (ulong(t) * p.heads + h) * p.dim;
    for (uint r = 0; r < R; ++r) {
        uint d = lane + SIMD * r;
        o[d] = acc[r] / l * (1.0f / (1.0f + exp(-gr[d])));
    }
}

#define INKLING_ATTENTION(R, T, QUANT, NAME) \
    template [[host_name("inkling_attention_r" #R "_" #NAME)]] kernel void inkling_attention<R, T, QUANT>( \
        device const float *, device const T *, device const T *, device const uint *, device const uint *, \
        device const uint *, device const float *, device const float *, device const float *, \
        device const float *, device float *, constant InklingAttentionParams &, device const half *, \
        device const half *, device float *, device float *, device const float *, device const float *, \
        device const uint *, uint, uint, uint, uint);
INKLING_ATTENTION(2, float, 0, f32)
INKLING_ATTENTION(2, half, 0, f16)
INKLING_ATTENTION(2, char, 1, q8)
INKLING_ATTENTION(4, float, 0, f32)
INKLING_ATTENTION(4, half, 0, f16)
INKLING_ATTENTION(4, char, 1, q8)
INKLING_ATTENTION(8, float, 0, f32)
INKLING_ATTENTION(8, half, 0, f16)
INKLING_ATTENTION(8, char, 1, q8)
#define INKLING_REDUCE(R) \
    template [[host_name("inkling_attention_reduce_r" #R "_f32")]] kernel void inkling_attention_reduce<R>( \
        device const float *, device const float *, device const float *, device const float *, \
        device float *, constant InklingAttentionParams &, uint, uint, uint, uint);
INKLING_REDUCE(2)
INKLING_REDUCE(4)
INKLING_REDUCE(8)

struct InklingMessageParams {
    // Rows of `messages`: inputs × tokens, input-major.
    uint rows;
    uint tokens;
    uint width;
    // Nonzero: every input reads the same `tokens` rows of x.
    uint x_shared;
    uint x_stride;
    // Nonzero: each input has its own norm gain row; otherwise all share row 0.
    uint norm_rows;
    float eps;
};

// Row r = (input i, token t) of [x / rms(x) | m / rms(m)], each RMS without a gain: the
// pair a loop pass's message gate projects. One threadgroup per row.
kernel void inkling_message_pair_f32(
    device const float *x [[buffer(0)]],
    device const float *messages [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant InklingMessageParams &p [[buffer(3)]],
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
    device const float *xr = x + ulong(p.x_shared != 0 ? row % p.tokens : row) * p.x_stride;
    device const float *mr = messages + ulong(row) * p.width;
    float xs = 0.0f, ms = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        xs = fma(xr[i], xr[i], xs);
        ms = fma(mr[i], mr[i], ms);
    }
    float x_inv = rsqrt(threadgroup_sum(xs, scratch, sg, lane, sgs) / float(p.width) + p.eps);
    float m_inv = rsqrt(threadgroup_sum(ms, scratch, sg, lane, sgs) / float(p.width) + p.eps);
    device float *o = out + ulong(row) * 2 * p.width;
    for (uint i = tid; i < p.width; i += threads) {
        o[i] = xr[i] * x_inv;
        o[p.width + i] = mr[i] * m_inv;
    }
}

// Row r = (input i, token t): g = x - tanh(forget) * x + tanh(gain) * m, with `gates`
// holding forget then gain columns, followed by the input's RMS norm with gain
// norm[i] (or norm[0]). One threadgroup per row.
kernel void inkling_message_gate_f32(
    device const float *x [[buffer(0)]],
    device const float *messages [[buffer(1)]],
    device const float *gates [[buffer(2)]],
    device const float *norm [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant InklingMessageParams &p [[buffer(5)]],
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
    device const float *xr = x + ulong(p.x_shared != 0 ? row % p.tokens : row) * p.x_stride;
    device const float *mr = messages + ulong(row) * p.width;
    device const float *gr = gates + ulong(row) * 2 * p.width;
    device const float *w = norm + (p.norm_rows != 0 ? ulong(row / p.tokens) * p.width : 0);
    device float *o = out + ulong(row) * p.width;
    float sum = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        // Clamped as softcap clamps: tanh(±20) rounds to ±1, and larger arguments gave NaN.
        float forget = tanh(clamp(gr[i], -20.0f, 20.0f));
        float gain = tanh(clamp(gr[p.width + i], -20.0f, 20.0f));
        float v = xr[i] - forget * xr[i] + gain * mr[i];
        o[i] = v;
        sum = fma(v, v, sum);
    }
    float inv = rsqrt(threadgroup_sum(sum, scratch, sg, lane, sgs) / float(p.width) + p.eps);
    for (uint i = tid; i < p.width; i += threads) {
        o[i] = o[i] * inv * w[i];
    }
}

struct InklingGateGemvParams {
    uint groups;
    uint rows;
    uint outs;
    uint k;
    uint per;
};

// out[g][r][o] = sum_k x[g][r][k] * w[g][o][k] for small row counts: F32 activations and
// accumulation over BF16 weights (their raw bits, widened exactly), so stored BF16 gate
// weights multiply as their F32 values would. One simdgroup per `per` outputs of a row.
kernel void inkling_gate_gemv_f32(
    device const float *x [[buffer(0)]],
    device const ushort *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant InklingGateGemvParams &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint blocks = (p.outs + p.per - 1) / p.per;
    uint item = tg * sgs + sg;
    if (item >= p.groups * p.rows * blocks) {
        return;
    }
    uint b = item % blocks, row = item / blocks;
    uint g = row / p.rows;
    device const float *xr = x + ulong(row) * p.k;
    for (uint j = 0; j < p.per; ++j) {
        uint o = b * p.per + j;
        if (o >= p.outs) {
            break;
        }
        device const ushort *wr = w + (ulong(g) * p.outs + o) * p.k;
        float acc = 0.0f;
        for (uint i = lane; i < p.k; i += SIMD) {
            acc = fma(xr[i], as_type<float>(uint(wr[i]) << 16), acc);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            out[ulong(row) * p.outs + o] = acc;
        }
    }
}

// Routed experts of a few tokens after their selection (moe_route_sigmoid_f32), for decode
// and verification passes: each selected expert's gate|up GEMV with the softcapped SwiGLU,
// then the down GEMV of every selected expert with their weighted sum. The GEMVs walk rows
// as moe_gemv does and the sum runs in ascending expert ID as moe_scatter_f32 does, so each
// token's result is what the grouped kernels compute without their grouping tables.
#define SMALL_EXPERTS_K 8u
#define SMALL_EXPERTS_NR 4u

struct InklingSmallExpertsParams {
    // Byte offsets of the gate|up and down banks' scales.
    ulong up_scales;
    ulong down_scales;
    uint tokens;
    uint experts;
    uint top_k;
    uint hidden;
    uint inter;
    uint x_stride;
    float scale;
    float gate_cap;
    float up_cap;
};

// One simdgroup per (token, selected expert, four interleaved gate|up rows): the GEMV of
// moe_gemv, then two softcapped SwiGLU outputs with the expert's factors, as
// moe_softcap_glu_f32 computes them. Writes act[token][k][inter].
template <uint FORMAT>
kernel void inkling_experts_up_small(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *selected [[buffer(2)]],
    device const float *factors [[buffer(3)]],
    device float *act [[buffer(4)]],
    constant InklingSmallExpertsParams &p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    constexpr uint NR = SMALL_EXPERTS_NR;
    uint rows = 2 * p.inter, blocks = rows / NR;
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.top_k * blocks) {
        return;
    }
    uint block = item % blocks, assignment = item / blocks;
    uint token = assignment / p.top_k;
    uint expert = uint(selected[ulong(assignment) * 2]);
    uint col = block * NR;
    ulong base = ulong(expert) * rows;
    device const float *xr = x + ulong(token) * p.x_stride;
    float acc[NR];
    for (uint r = 0; r < NR; ++r) {
        acc[r] = 0.0f;
    }
    uint ix = lane / 4, start = (lane % 4) * 8;
    for (uint chunk = ix; chunk < p.hidden / 32; chunk += 8) {
        device const float *xb = xr + chunk * 32 + start;
        float4 xl = float4(xb[0], xb[1], xb[2], xb[3]);
        float4 xh = float4(xb[4], xb[5], xb[6], xb[7]);
        for (uint r = 0; r < NR; ++r) {
            float4 lo, hi;
            packed_weights8<FORMAT>(w, base + col + r, chunk, start, p.hidden, lo, hi, p.up_scales);
            acc[r] += dot(lo, xl) + dot(hi, xh);
        }
    }
    for (uint r = 0; r < NR; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (lane != 0) {
        return;
    }
    float up_factor = factors[2 * expert], down_factor = factors[2 * expert + 1];
    for (uint pair = 0; pair < NR / 2; ++pair) {
        float gate = acc[2 * pair] * up_factor, up = acc[2 * pair + 1] * up_factor;
        act[ulong(assignment) * p.inter + col / 2 + pair] =
            softcap(gate, p.gate_cap) * (1.0f / (1.0f + exp(-gate))) * softcap(up, p.up_cap) * down_factor;
    }
}

// One threadgroup per (token, four output rows) with a simdgroup per selected expert: its
// down GEMV over its activations, then, in threadgroup memory, their weighted sum in
// ascending expert ID. Writes y[token][hidden].
template <uint FORMAT>
kernel void inkling_experts_down_small(
    device const uchar *w [[buffer(0)]],
    device const float *act [[buffer(1)]],
    device const float *selected [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant InklingSmallExpertsParams &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint k [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NR = SMALL_EXPERTS_NR, K = SMALL_EXPERTS_K;
    threadgroup float partial[K][NR];
    uint blocks = (p.hidden + NR - 1) / NR;
    uint token = tg / blocks, col = (tg % blocks) * NR;
    device const float *sel = selected + ulong(token) * p.top_k * 2;
    ulong base = ulong(uint(sel[k * 2])) * p.hidden;
    device const float *xr = act + (ulong(token) * p.top_k + k) * p.inter;
    float acc[NR];
    for (uint r = 0; r < NR; ++r) {
        acc[r] = 0.0f;
    }
    uint ix = lane / 4, start = (lane % 4) * 8;
    for (uint chunk = ix; chunk < p.inter / 32; chunk += 8) {
        device const float *xb = xr + chunk * 32 + start;
        float4 xl = float4(xb[0], xb[1], xb[2], xb[3]);
        float4 xh = float4(xb[4], xb[5], xb[6], xb[7]);
        for (uint r = 0; r < NR; ++r) {
            float4 lo, hi;
            packed_weights8<FORMAT>(w, base + min(col + r, p.hidden - 1), chunk, start, p.inter, lo, hi,
                p.down_scales);
            acc[r] += dot(lo, xl) + dot(hi, xh);
        }
    }
    for (uint r = 0; r < NR; ++r) {
        acc[r] = simd_sum(acc[r]);
        if (lane == 0) {
            partial[k][r] = acc[r] * sel[k * 2 + 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= NR || col + tid >= p.hidden) {
        return;
    }
    // Ascending expert ID, as the grouped path's scatter sums.
    float sum = 0.0f;
    for (uint n = 0; n < p.top_k; ++n) {
        uint k_n = 0;
        for (uint j = 0; j < p.top_k; ++j) {
            uint rank = 0;
            for (uint i = 0; i < p.top_k; ++i) {
                rank += sel[i * 2] < sel[j * 2];
            }
            k_n = rank == n ? j : k_n;
        }
        sum += partial[k_n][tid];
    }
    y[ulong(token) * p.hidden + col + tid] = sum;
}

#define INKLING_SMALL_EXPERTS(FORMAT, NAME) \
    template [[host_name("inkling_experts_up_small_" #NAME)]] kernel void inkling_experts_up_small<FORMAT>( \
        device const uchar *, device const float *, device const float *, device const float *, device float *, \
        constant InklingSmallExpertsParams &, uint, uint, uint, uint); \
    template [[host_name("inkling_experts_down_small_" #NAME)]] kernel void inkling_experts_down_small<FORMAT>( \
        device const uchar *, device const float *, device const float *, device float *, \
        constant InklingSmallExpertsParams &, uint, uint, uint, uint);
INKLING_SMALL_EXPERTS(5, nvfp4)
INKLING_SMALL_EXPERTS(6, mxfp8)
