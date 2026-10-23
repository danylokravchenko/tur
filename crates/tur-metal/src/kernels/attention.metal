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

