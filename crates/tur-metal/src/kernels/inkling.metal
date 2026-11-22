// ---------------------------------------------------------------------------------------
// Inkling 2: per-head norms, attention residuals over block states, short causal
// convolutions with slot-addressed history, and paged attention with a learned relative
// position bias, per-token sinks and a sigmoid output gate.
// ---------------------------------------------------------------------------------------

struct InklingNormParams {
    uint tokens;
    uint heads;
    uint dim;
    uint stride;
    uint col;
    // 0: scale by `scale`, then clip the head's L2 norm to `threshold`; 1: unweighted RMS.
    uint mode;
    uint scaled;
    float threshold;
    float eps;
    // Nonzero: also multiply by 1 + alpha * log(max((position + 1) / floor, 1)).
    uint log_scaled;
    float log_alpha;
    float log_floor;
};

// One simdgroup per (token, head): head h of row t starts at column col + h * dim.
// Writes contiguous [tokens, heads * dim] rows.
kernel void inkling_head_norm_f32(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device const uint *positions [[buffer(3)]],
    constant InklingNormParams &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint item = tg * sgs + sg;
    if (item >= p.tokens * p.heads) {
        return;
    }
    uint t = item / p.heads, h = item % p.heads;
    device const float *src = x + ulong(t) * p.stride + p.col + ulong(h) * p.dim;
    device float *dst = out + ulong(item) * p.dim;
    float sum = 0.0f;
    for (uint d = lane; d < p.dim; d += SIMD) {
        float v = src[d] * (p.scaled != 0 ? scale[h * p.dim + d] : 1.0f);
        sum = fma(v, v, sum);
    }
    sum = simd_sum(sum);
    float factor;
    if (p.mode == 0) {
        factor = min(p.threshold / (sqrt(sum) + 1e-6f), 1.0f);
        if (p.log_scaled != 0) {
            float ratio = max(float(positions[t] + 1) / p.log_floor, 1.0f);
            factor *= 1.0f + p.log_alpha * log(ratio);
        }
    } else {
        factor = rsqrt(sum / float(p.dim) + p.eps);
    }
    for (uint d = lane; d < p.dim; d += SIMD) {
        float v = src[d] * (p.scaled != 0 ? scale[h * p.dim + d] : 1.0f);
        dst[d] = v * factor;
    }
}

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
};

static inline float short_conv_input(
    device const float *x, device const float *history, uint r, uint c, uint kept,
    constant InklingConvParams &p) {
    return r < kept ? history[r * p.pool_width + c] : x[ulong(r - kept) * p.x_stride + c];
}

// Depthwise causal convolution plus the input, divided by `divisor`, for every sequence of
// a packed pass over [history; x]. `taps` is [channels, taps] by lag (0 = the token itself).
// With `dynamic`, each token adds its own taps per head of `head_dim` channels from `dyn`
// (head h, lag l at column dyn_col + h * taps + l). Each sequence writes the (taps-1)-row
// history after each token its written slots list, as delta_conv_f32 does, into columns
// offset..offset + channels of the slot rows.
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
    device const float *history = pool + ulong(s.read) * kept * p.pool_width + p.offset;
    device const float *own = dyn + ulong(row) * p.dyn_stride + p.dyn_col + (c / p.head_dim) * p.taps;
    float acc = 0.0f;
    for (uint lag = 0; lag < p.taps; ++lag) {
        float w = taps[c * p.taps + lag] + (p.dynamic != 0 ? own[lag] : 0.0f);
        acc = fma(w, short_conv_input(xs, history, t + kept - lag, c, kept, p), acc);
    }
    out[gid] = (acc + xs[ulong(t) * p.x_stride + c]) / p.divisor;
    uint written = written_after(s, slots, t);
    if (written < s.count) {
        ulong base = ulong(slots[s.writes + written]) * kept * p.pool_width + p.offset;
        for (uint r = 0; r < kept; ++r) {
            pool[base + ulong(r) * p.pool_width + c] = short_conv_input(xs, history, t + 1 + r, c, kept, p);
        }
    }
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
};

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
    device const float *qr = q + (ulong(t) * p.heads + h) * p.dim;
    for (uint c = 0; c < C; ++c) {
        dims[c] = V * lane + SIMD * V * c;
        sidx[c] = QUANT == 0 ? 0 : dims[c] / p.group;
        kv_dims<V>(qr + dims[c], qv[c]);
        for (uint i = 0; i < V; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    // Lane r < rel_dim keeps component r of the relative state and reads row r of `proj`.
    bool rel_lane = lane < p.rel_dim;
    float rel_value = rel_lane ? rel[ulong(t) * p.rel_stride + p.rel_col + h * p.rel_dim + lane] : 0.0f;
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
        device const half *, device float *, device float *, uint, uint, uint, uint);
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
        float v = xr[i] - tanh(gr[i]) * xr[i] + tanh(gr[p.width + i]) * mr[i];
        o[i] = v;
        sum = fma(v, v, sum);
    }
    float inv = rsqrt(threadgroup_sum(sum, scratch, sg, lane, sgs) / float(p.width) + p.eps);
    for (uint i = tid; i < p.width; i += threads) {
        o[i] = o[i] * inv * w[i];
    }
}
