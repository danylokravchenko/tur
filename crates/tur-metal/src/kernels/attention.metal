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
    uint window;
    uint ungated;
    uint sinks;
};

// Keys per attention_kv step: their table lookups and K/V loads are all in flight at once,
// and the online softmax rescales once per step.
#define ATTENTION_KEYS 4

// R contiguous stored values of one row as F32 in one vector load (two for eight F32
// values). Rows are 16-byte aligned and callers start each lane's vector at a multiple of
// its size, so every load is naturally aligned.
template <uint R>
static inline void kv_dims(device const float *src, thread float (&x)[R]) {
    if (R == 2) {
        float2 v = *(device const float2 *)src;
        x[0] = v.x; x[1] = v.y;
        return;
    }
    for (uint i = 0; i < R; i += 4) {
        float4 v = *(device const float4 *)(src + i);
        x[i] = v.x; x[i + 1] = v.y; x[i + 2] = v.z; x[i + 3] = v.w;
    }
}

template <uint R>
static inline void kv_dims(device const half *src, thread float (&x)[R]) {
    uint words[4];
    if (R == 8) {
        uint4 v = *(device const uint4 *)src;
        words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
    } else if (R == 4) {
        uint2 v = *(device const uint2 *)src;
        words[0] = v.x; words[1] = v.y;
    } else {
        words[0] = *(device const uint *)src;
    }
    for (uint i = 0; i < R / 2; ++i) {
        half2 h = as_type<half2>(words[i]);
        x[2 * i] = float(h.x); x[2 * i + 1] = float(h.y);
    }
}

template <uint R>
static inline void kv_dims(device const char *src, thread float (&x)[R]) {
    if (R == 16) {
        uint4 words = *(device const uint4 *)src;
        for (uint i=0; i<4; ++i) {
            char4 v = as_type<char4>(words[i]);
            x[4*i]=float(v.x); x[4*i+1]=float(v.y);
            x[4*i+2]=float(v.z); x[4*i+3]=float(v.w);
        }
        return;
    }
    if (R == 2) {
        char2 c = as_type<char2>(*(device const ushort *)src);
        x[0] = float(c.x); x[1] = float(c.y);
        return;
    }
    uint words[2];
    if (R == 8) {
        uint2 v = *(device const uint2 *)src;
        words[0] = v.x; words[1] = v.y;
    } else {
        words[0] = *(device const uint *)src;
    }
    for (uint i = 0; i < R / 4; ++i) {
        float4 c = float4(as_type<char4>(words[i]));
        x[4 * i] = c.x; x[4 * i + 1] = c.y; x[4 * i + 2] = c.z; x[4 * i + 3] = c.w;
    }
}

// One simdgroup per (HG consecutive heads of one K/V head, split, token). Lane l reads
// each key and value row as C vectors of V = min(R, 16 / sizeof(T)) values, vector c at
// dimensions V*l + 32*V*c, so every load instruction covers one contiguous segment of the
// row: one 16-byte load per row for F16 or Q8 at head dimension 256, and for Q8 one scale
// per vector (a vector never straddles a 32-multiple group). Each loaded K and V vector
// serves all HG query heads. Keys are taken ATTENTION_KEYS at a time, so their loads are
// outstanding together. Online softmax keeps the running max/sum in F32. With one split
// the gated output is written directly; otherwise unnormalised partial sums and
// (max, sum) statistics go to scratch.
template <uint R, typename T, uint QUANT, uint HG>
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
    constexpr uint U = ATTENTION_KEYS;
    constexpr uint V = R * sizeof(T) <= 16 ? R : 16 / sizeof(T);
    constexpr uint C = R / V;
    uint head_groups = p.heads / HG;
    uint item = tg * sgs + sg;
    if (item >= p.tokens * head_groups * p.splits) {
        return;
    }
    uint h0 = (item % head_groups) * HG;
    uint rest = item / head_groups;
    uint split = rest % p.splits;
    uint t = rest / p.splits;
    uint g = h0 / (p.heads / p.kv_heads);
    uint keys = visible[t];
    // Splits partition each query's own window, so a sliding layer keeps every split busy
    // however long the context grows.
    uint first_key = p.window > 0 && keys > p.window ? keys - p.window : 0u;
    uint begin = first_key + split * p.split_keys;
    uint end = min(keys, begin + p.split_keys);
    uint base = first_block[t];
    uint groups = QUANT == 0 ? 0 : p.dim / p.group;
    uint dims[C], sidx[C];
    float qv[HG][C][V], acc[HG][C][V];
    for (uint c = 0; c < C; ++c) {
        dims[c] = V * lane + SIMD * V * c;
        sidx[c] = QUANT == 0 ? 0 : dims[c] / p.group;
    }
    for (uint x = 0; x < HG; ++x) {
        device const float *qr = q + (ulong(t) * p.heads + h0 + x) * p.dim;
        for (uint c = 0; c < C; ++c) {
            kv_dims<V>(qr + dims[c], qv[x][c]);
            for (uint i = 0; i < V; ++i) {
                acc[x][c][i] = 0.0f;
            }
        }
    }
    // A sink joins the softmax as one more key without a value: it enters the running
    // maximum and sum once, here for one split and in the reduction otherwise.
    bool sink = p.sinks != 0 && p.splits == 1;
    float m[HG], l[HG];
    for (uint x = 0; x < HG; ++x) {
        m[x] = sink ? gate[h0 + x] : -FLT_MAX;
        l[x] = sink ? 1.0f : 0.0f;
    }
    for (uint j0 = begin; j0 < end; j0 += U) {
        // Pool rows (slot * kv_heads + g; the host bounds them by 2^32) of this step's keys.
        // Keys past the range repeat the last one and get zero weight. A step inside one
        // page, the common case and uniform across the simdgroup, needs one table lookup.
        uint rows[U];
        if (j0 % p.page + U <= p.page && j0 + U <= end) {
            uint row = (table[base + j0 / p.page] * p.page + j0 % p.page) * p.kv_heads + g;
            for (uint u = 0; u < U; ++u) {
                rows[u] = row + u * p.kv_heads;
            }
        } else {
            for (uint u = 0; u < U; ++u) {
                uint j = min(j0 + u, end - 1);
                rows[u] = (table[base + j / p.page] * p.page + j % p.page) * p.kv_heads + g;
            }
        }
        float score[HG][U];
        for (uint u = 0; u < U; ++u) {
            device const T *kr = k + ulong(rows[u]) * p.dim;
            for (uint x = 0; x < HG; ++x) {
                score[x][u] = 0.0f;
            }
            for (uint c = 0; c < C; ++c) {
                float kd[V];
                kv_dims<V>(kr + dims[c], kd);
                float ks = QUANT == 0 ? 1.0f : float(k_scales[ulong(rows[u]) * groups + sidx[c]]);
                for (uint x = 0; x < HG; ++x) {
                    float d = 0.0f;
                    for (uint i = 0; i < V; ++i) {
                        d = fma(qv[x][c][i], kd[i], d);
                    }
                    score[x][u] += QUANT == 0 ? d : d * ks;
                }
            }
        }
        float weight[HG][U];
        for (uint x = 0; x < HG; ++x) {
            float m_new = m[x];
            for (uint u = 0; u < U; ++u) {
                score[x][u] = simd_sum(score[x][u]) * p.scale;
                m_new = j0 + u < end ? max(m_new, score[x][u]) : m_new;
            }
            float correction = exp(m[x] - m_new);
            float total = 0.0f;
            for (uint u = 0; u < U; ++u) {
                weight[x][u] = j0 + u < end ? exp(score[x][u] - m_new) : 0.0f;
                total += weight[x][u];
            }
            l[x] = l[x] * correction + total;
            for (uint c = 0; c < C; ++c) {
                for (uint i = 0; i < V; ++i) {
                    acc[x][c][i] *= correction;
                }
            }
            m[x] = m_new;
        }
        for (uint u = 0; u < U; ++u) {
            device const T *vr = v + ulong(rows[u]) * p.dim;
            for (uint c = 0; c < C; ++c) {
                float vd[V];
                kv_dims<V>(vr + dims[c], vd);
                float vs = QUANT == 0 ? 1.0f : float(v_scales[ulong(rows[u]) * groups + sidx[c]]);
                for (uint x = 0; x < HG; ++x) {
                    float w = QUANT == 0 ? weight[x][u] : weight[x][u] * vs;
                    for (uint i = 0; i < V; ++i) {
                        acc[x][c][i] = fma(w, vd[i], acc[x][c][i]);
                    }
                }
            }
        }
    }
    for (uint x = 0; x < HG; ++x) {
        uint h = h0 + x;
        if (p.splits == 1) {
            device const float *gr = gate + ulong(t) * p.gate_stride + p.gate_col + ulong(h) * p.gate_head_stride;
            device float *o = out + (ulong(t) * p.heads + h) * p.dim;
            float inv = l[x] > 0.0f ? 1.0f / l[x] : 0.0f;
            for (uint c = 0; c < C; ++c) {
                for (uint i = 0; i < V; ++i) {
                    o[dims[c] + i] = acc[x][c][i] * inv * (p.ungated ? 1.0f : sigmoid(gr[dims[c] + i]));
                }
            }
            continue;
        }
        ulong slot = (ulong(t) * p.heads + h) * p.splits + split;
        device float *pp = partial + slot * p.dim;
        for (uint c = 0; c < C; ++c) {
            for (uint i = 0; i < V; ++i) {
                pp[dims[c] + i] = acc[x][c][i];
            }
        }
        if (lane == 0) {
            stats[2 * slot] = m[x];
            stats[2 * slot + 1] = l[x];
        }
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
// Threads per tiled threadgroup (`TILE_THREADS`), the float4 loads each makes per staged
// chunk, and the key distance between them.
#define TT 128
#define LOADS (TK * TD / 4 / TT)
#define KEY_STRIDE (TT / (TD / 4))

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
    uint window;
    uint ungated;
    uint sinks;
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

// Pool slots of the keys this thread stages in the block at `kb`; keys past `keys` read slot 0.
static inline void kv_slots(thread uint *slot, device const uint *table, uint base, uint kb, uint keys, uint key0, uint page) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < LOADS; ++i) {
        uint j = kb + key0 + i * KEY_STRIDE;
        slot[i] = j < keys ? table[base + j / page] * page + j % page : 0;
    }
}

// This thread's share of one staged F32 or F16 chunk: dimensions d..d+4 of its keys as
// stored, zero past `keys`. Converting here would wait on the reads at once.
template <typename T>
static inline void kv_fetch(thread vec<T, 4> *raw, device const T *data, thread const uint *slot, uint kb, uint keys, uint key0, uint kv_heads, uint g, uint dim, uint d) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < LOADS; ++i) {
        ulong head = ulong(slot[i]) * kv_heads + g;
        raw[i] = kb + key0 + i * KEY_STRIDE < keys ? *(device const vec<T, 4> *)(data + head * dim + d) : vec<T, 4>(0);
    }
}

// Q·K over dimension chunk `c` of the staged keys, accumulated into the score tiles. F16
// and Q8 pools leave the score scale to the softmax, where it is applied once per block:
// per query fragment it costs them 3-8% of the kernel, while F32 pools run 9% faster with it here.
template <typename S, bool SCALE_QUERY>
static inline void tile_scores(thread simdgroup_float8x8 *s, threadgroup const S *stage, device const float *qr, uint c, ushort2 fc, float scale) {
    #pragma clang loop unroll(full)
    for (uint x = 0; x < TD / 8; ++x) {
        simdgroup_float8x8 a;
        // The clamped token also keeps partial query tiles inside the allocation.
        float2 pair = *(device const float2 *)(qr + c * TD + 8 * x + fc.y);
        a.thread_elements()[0] = SCALE_QUERY ? pair[0] * scale : pair[0];
        a.thread_elements()[1] = SCALE_QUERY ? pair[1] * scale : pair[1];
        #pragma clang loop unroll(full)
        for (uint kt = 0; kt < TK / 8; ++kt) {
            simdgroup_matrix<S, 8, 8> b;
            simdgroup_load(b, stage + 8 * kt * TD + 8 * x, TD, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s[kt], a, b, s[kt]);
        }
    }
}

// Online softmax on this lane's two elements per key tile of the block at `kb`, all in
// one row, whose visible keys are [lower, limit): scaled scores become weights, and the
// output and running sums are rescaled.
template <uint CHUNKS>
static inline void tile_softmax(thread simdgroup_float8x8 *s, thread simdgroup_float8x8 *o, thread float &m, thread float &l, uint kb, uint lower, uint limit, ushort2 fc, float scale) {
    float block = -FLT_MAX;
    #pragma clang loop unroll(full)
    for (uint kt = 0; kt < TK / 8; ++kt) {
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 2; ++e) {
            uint key = kb + 8 * kt + fc.y + e;
            float score = s[kt].thread_elements()[e] * scale;
            s[kt].thread_elements()[e] = score;
            block = key >= lower && key < limit ? max(block, score) : block;
        }
    }
    float m_new = max(m, row_max(block));
    float total = 0.0f;
    #pragma clang loop unroll(full)
    for (uint kt = 0; kt < TK / 8; ++kt) {
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 2; ++e) {
            uint key = kb + 8 * kt + fc.y + e;
            float weight = key >= lower && key < limit ? exp(s[kt].thread_elements()[e] - m_new) : 0.0f;
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
}

// P·V over dimension chunk `c` of the staged values, accumulated into the output tiles.
template <typename S>
static inline void tile_values(thread simdgroup_float8x8 *o, thread const simdgroup_float8x8 *s, threadgroup const S *stage, uint c) {
    #pragma clang loop unroll(full)
    for (uint kt = 0; kt < TK / 8; ++kt) {
        #pragma clang loop unroll(full)
        for (uint u = 0; u < TD / 8; ++u) {
            simdgroup_matrix<S, 8, 8> b;
            simdgroup_load(b, stage + 8 * kt * TD + 8 * u, TD, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(o[8 * c + u], s[kt], b, o[8 * c + u]);
        }
    }
}

// Staged element type: pools stage values as stored, so F16 multiplies in F16 against F32
// queries and weights with no rounding, and Q8 stages dequantized F32.
template <typename T> struct tile_stage { using type = T; };
template <> struct tile_stage<char> { using type = float; };

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

    ushort2 fc = fragment_coord(lane);
    uint row = 8 * sg + fc.x;
    // Rows past the tile repeat its last token and are never written.
    uint token = t0 + min(row, n - 1);
    uint limit = row < n ? visible[token] : 0;
    uint lower = p.window > 0 && limit > p.window ? limit - p.window : 0;
    // Visible counts grow along a sequence, so the last token sees every key of the tile
    // and the first token's window starts earliest; the sweep starts at its key block.
    uint keys = visible[t0 + n - 1];
    uint start = p.window > 0 && visible[t0] > p.window ? (visible[t0] - p.window) / TK * TK : 0;
    uint base = first_block[t0];

    simdgroup_float8x8 o[8 * CHUNKS];
    device const float *qr = q + (ulong(token) * p.heads + h) * D;
    // Every matrix-array index must become constant: dynamic indexing spills the
    // fragments to thread memory, including inside the dimension-chunk loops.
    #pragma clang loop unroll(full)
    for (uint x = 0; x < 8 * CHUNKS; ++x) {
        o[x] = simdgroup_float8x8(0.0f);
    }
    float m = p.sinks != 0 ? gate[h] : -FLT_MAX;
    float l = p.sinks != 0 ? 1.0f : 0.0f;
    uint key0 = tid / (TD / 4);
    uint d0 = 4 * (tid % (TD / 4));
    if (QUANT == 0) {
        // F32 and F16 chunks stream through two staging buffers. Each thread stages the
        // same keys of every chunk, and issues the next chunk's reads before this chunk's
        // matrix work so their latency overlaps it; one barrier per chunk orders the
        // buffers.
        using S = typename tile_stage<T>::type;
        constexpr bool SCALE_QUERY = is_same<T, float>::value;
        uint slot[LOADS];
        uint next_slot[LOADS];
        vec<T, 4> fetched[LOADS];
        kv_slots(slot, table, base, start, keys, key0, p.page);
        kv_fetch<T>(fetched, k, slot, start, keys, key0, p.kv_heads, g, D, d0);
        for (uint kb = start; kb < keys; kb += TK) {
            kv_slots(next_slot, table, base, kb + TK, keys, key0, p.page);
            simdgroup_float8x8 s[TK / 8];
            #pragma clang loop unroll(full)
            for (uint kt = 0; kt < TK / 8; ++kt) {
                s[kt] = simdgroup_float8x8(0.0f);
            }
            // 2 * CHUNKS steps per block keep every step's buffer constant.
            #pragma clang loop unroll(full)
            for (uint step = 0; step < 2 * CHUNKS; ++step) {
                threadgroup S *stage = (threadgroup S *)shared + (step % 2) * TK * TD;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < LOADS; ++i) {
                    *(threadgroup vec<S, 4> *)(stage + (key0 + i * KEY_STRIDE) * TD + d0) = vec<S, 4>(fetched[i]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (step + 1 < CHUNKS) {
                    kv_fetch<T>(fetched, k, slot, kb, keys, key0, p.kv_heads, g, D, (step + 1) * TD + d0);
                } else if (step + 1 < 2 * CHUNKS) {
                    kv_fetch<T>(fetched, v, slot, kb, keys, key0, p.kv_heads, g, D, (step + 1 - CHUNKS) * TD + d0);
                } else if (kb + TK < keys) {
                    kv_fetch<T>(fetched, k, next_slot, kb + TK, keys, key0, p.kv_heads, g, D, d0);
                }
                if (step < CHUNKS) {
                    tile_scores<S, SCALE_QUERY>(s, stage, qr, step, fc, p.scale);
                } else {
                    tile_values<S>(o, s, stage, step - CHUNKS);
                }
                if (step == CHUNKS - 1) {
                    tile_softmax<CHUNKS>(s, o, m, l, kb, lower, limit, fc, SCALE_QUERY ? 1.0f : p.scale);
                }
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < LOADS; ++i) {
                slot[i] = next_slot[i];
            }
        }
    } else {
        // Q8 chunks are read, dequantized and staged in one buffer, two barriers per
        // chunk: prefetching them, with their scales, was 9-42% slower at every shape.
        threadgroup uint *slots = (threadgroup uint *)(shared + TK * TD);
        for (uint kb = start; kb < keys; kb += TK) {
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
                    *(threadgroup float4 *)(shared + key * TD + d) = kb + key < keys
                        ? kv_vector<T, QUANT>(k, k_scales, head, D, c * TD + d, p.group)
                        : float4(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                tile_scores<float, false>(s, shared, qr, c, fc, p.scale);
            }
            tile_softmax<CHUNKS>(s, o, m, l, kb, lower, limit, fc, p.scale);
            #pragma clang loop unroll(full)
            for (uint c = 0; c < CHUNKS; ++c) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < TK * TD / 4; i += threads) {
                    uint key = i / (TD / 4);
                    uint d = 4 * (i % (TD / 4));
                    ulong head = ulong(slots[key]) * p.kv_heads + g;
                    *(threadgroup float4 *)(shared + key * TD + d) = kb + key < keys
                        ? kv_vector<T, QUANT>(v, v_scales, head, D, c * TD + d, p.group)
                        : float4(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                tile_values<float>(o, s, shared, c);
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
            orow[col] = o[x].thread_elements()[e] * inv * (p.ungated ? 1.0f : sigmoid(gr[col]));
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
    float m = p.sinks != 0 ? gate[h] : -FLT_MAX;
    for (uint s = 0; s < p.splits; ++s) {
        if (stats[2 * (first + s) + 1] > 0.0f) {
            m = max(m, stats[2 * (first + s)]);
        }
    }
    float l = p.sinks != 0 ? exp(gate[h] - m) : 0.0f;
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
        o[i] = acc[r] * inv * (p.ungated ? 1.0f : sigmoid(gr[i]));
    }
}

#define TILED(R, NAME, T, QUANT) \
    template [[host_name("attention_tiled_r" #R "_" #NAME)]] kernel void attention_tiled<R / 2, T, QUANT>( \
        device const float *, device const T *, device const T *, device const uint *, \
        device const uint *, device const uint *, device const float *, device float *, \
        device const uint *, constant TiledParams &, device const half *, device const half *, \
        threadgroup float *, uint, uint, uint, uint, uint);
#define ATTENTION_KV(R, NAME, T, QUANT, HG, PREFIX) \
    template [[host_name(PREFIX #R "_" #NAME)]] kernel void attention_kv<R, T, QUANT, HG>( \
        device const float *, device const T *, device const T *, device const uint *, \
        device const uint *, device const uint *, device const float *, device float *, \
        device float *, device float *, constant AttentionParams &, device const half *, \
        device const half *, uint, uint, uint, uint);
#define KV_FORMAT(R, NAME, T, QUANT) \
    TILED(R, NAME, T, QUANT) \
    template [[host_name("kv_write_r" #R "_" #NAME)]] kernel void kv_write<R, T, QUANT>( \
        device const float *, device const float *, device const uint *, device const uint *, \
        device const uint *, device T *, device half *, device T *, device half *, \
        constant KvWriteParams &, uint, uint, uint, uint); \
    ATTENTION_KV(R, NAME, T, QUANT, 1, "attention_r") \
    ATTENTION_KV(R, NAME, T, QUANT, 2, "attention_h2_r")
// R registers per lane: head dimension 64, 128 or 256 over 32 lanes.
#define ATTENTION(R) \
    KV_FORMAT(R, f32, float, 0) \
    KV_FORMAT(R, f16, half, 0) \
    KV_FORMAT(R, q8, char, 1) \
    template [[host_name("attention_reduce_r" #R "_f32")]] kernel void attention_reduce_f32<R>( \
        device const float *, device const float *, device const float *, device float *, \
        constant AttentionParams &, uint, uint, uint, uint);
ATTENTION(2)
ATTENTION(4)
ATTENTION(8)


// Global Gemma heads use 512 dimensions; the per-query kernel bounds register use.
KV_FORMAT(16, f32, float, 0)
KV_FORMAT(16, f16, half, 0)
KV_FORMAT(16, q8, char, 1)
template [[host_name("attention_reduce_r16_f32")]] kernel void attention_reduce_f32<16>(
    device const float *, device const float *, device const float *, device float *,
    constant AttentionParams &, uint, uint, uint, uint);
