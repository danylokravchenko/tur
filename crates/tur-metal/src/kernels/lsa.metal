// ---------------------------------------------------------------------------------------
// Latent sparse attention index: clustering closed blocks of keys into buckets, and
// choosing the keys a decoding query reads. Follows the flat design of Prefrontal-AI's
// reference (`lsa.core.reference`, `ward`, `index_space`): index vectors live in the
// position-free dims of every K/V head, carried through a per-layer map to 64 dims.
// Integers (pool and key positions) travel as F32, exact below 2^24.
// ---------------------------------------------------------------------------------------

#define LSA_BLOCK 512
#define LSA_POOL 4
#define LSA_POOLS 128
#define LSA_BUCKETS 64
#define LSA_BUCKET_POOLS 4
#define LSA_BUCKET_CAP 16
#define LSA_DIM 64
#define LSA_MAX_ROUNDS 256
#define LSA_MAX_P 1024
#define LSA_MAX_FREE 2048
#define LSA_SCORE_CHUNK 256
#define LSA_POOL_GROUPS 16
#define LSA_SG_POOLS 2

// A head's position-free dims run [first, first + split) then from `second` on.
struct LsaClusterParams {
    uint jobs;
    uint kv_heads;
    uint dim;
    uint first;
    uint split;
    uint second;
    uint page;
    uint group;
    uint free_dims;
};

struct LsaScoreParams {
    uint groups;
    uint heads;
    uint kv_heads;
    uint dim;
    uint first;
    uint split;
    uint second;
    uint free_dims;
    uint score_stride;
};

struct LsaSelectParams {
    uint groups;
    uint score_stride;
    uint top_k;
    uint top_p;
    uint window;
    uint sink;
    uint width;
    // Nonzero for groups of a tiled pass: sinks before the window, which runs on to the
    // group's last row.
    uint tiled;
};

// Head dim of position-free dim `e` of a head.
static inline uint lsa_free_dim(uint e, uint first, uint split, uint second) {
    return e < split ? first + e : second + (e - split);
}

template <typename T, uint QUANT>
static inline float lsa_key(device const T *data, device const half *scales, ulong row, uint e,
                            constant LsaClusterParams &p) {
    float x = float(data[row * p.dim + e]);
    if (QUANT != 0) {
        x *= float(scales[row * (p.dim / p.group) + e / p.group]);
    }
    return x;
}

// Grid (jobs, LSA_POOL_GROUPS): threadgroup g of a job sums the index keys of pools
// [8g, 8g + 8) of its block into `sums`, so one block spreads over 16 threadgroups. Each
// simdgroup projects 2 pools (8 keys) together, so every map row it loads serves 8 keys;
// lane l holds output dims l and l + 32. Index key = normalize(map^T x) over the
// concatenated position-free dims x of every K/V head; the reference's normalization of x
// first is a positive scale that the second one removes.
template <typename T, uint QUANT>
kernel void lsa_pools(
    device const T *k_data [[buffer(0)]],
    device const half *k_scales [[buffer(1)]],
    device const uint *table [[buffer(2)]],
    device const uint *jobs [[buffer(3)]],
    device const float *map [[buffer(4)]],
    device float *scratch [[buffer(5)]],
    constant LsaClusterParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint KEYS = LSA_SG_POOLS * LSA_POOL;
    constexpr uint PER_GROUP = LSA_POOLS / LSA_POOL_GROUPS;
    if (group.x >= p.jobs) {
        return;
    }
    uint first = jobs[group.x * 3], block = jobs[group.x * 3 + 1];
    device float *sums = scratch + ulong(group.x) * LSA_POOLS * LSA_DIM;
    uint free = p.free_dims / p.kv_heads;
    uint end = (group.y + 1) * PER_GROUP;
    for (uint pool = group.y * PER_GROUP + sg * LSA_SG_POOLS; pool < end;
         pool += sgs * LSA_SG_POOLS) {
        ulong rows[KEYS];
        float y0[KEYS], y1[KEYS];
        for (uint r = 0; r < KEYS; ++r) {
            uint j = block * LSA_BLOCK + pool * LSA_POOL + r;
            rows[r] = (ulong(table[first + j / p.page]) * p.page + j % p.page) * p.kv_heads;
            y0[r] = 0.0f;
            y1[r] = 0.0f;
        }
        for (uint d0 = 0; d0 < p.free_dims; d0 += SIMD) {
            uint d = d0 + lane;
            float x[KEYS];
            for (uint r = 0; r < KEYS; ++r) {
                x[r] = lsa_key<T, QUANT>(k_data, k_scales, rows[r] + d / free, lsa_free_dim(d % free, p.first, p.split, p.second), p);
            }
            device const float *m = map + ulong(d0) * LSA_DIM;
            for (uint i = 0; i < SIMD; ++i) {
                float m0 = m[i * LSA_DIM + lane], m1 = m[i * LSA_DIM + lane + SIMD];
                for (uint r = 0; r < KEYS; ++r) {
                    float xi = simd_broadcast(x[r], ushort(i));
                    y0[r] += xi * m0;
                    y1[r] += xi * m1;
                }
            }
        }
        for (uint q = 0; q < LSA_SG_POOLS; ++q) {
            float a0 = 0.0f, a1 = 0.0f;
            for (uint r = q * LSA_POOL; r < (q + 1) * LSA_POOL; ++r) {
                float inv = 1.0f / max(sqrt(simd_sum(y0[r] * y0[r] + y1[r] * y1[r])), 1e-12f);
                a0 += y0[r] * inv;
                a1 += y1[r] * inv;
            }
            sums[(pool + q) * LSA_DIM + lane] = a0;
            sums[(pool + q) * LSA_DIM + lane + SIMD] = a1;
        }
    }
}

// Gram matrix of a block's pool sums: entry (i, j) is sums_i . sums_j, read through its upper
// triangle so both sides of a pair see the same value.
static inline float lsa_gram(device const float *gram, uint i, uint j) {
    return gram[min(i, j) * LSA_POOLS + max(i, j)];
}

// Each alive bucket's cheapest partner under the cap, ties to the lower index: simdgroups
// own buckets and lanes their candidates.
static inline void lsa_partners(device const float *gram, threadgroup float *squares,
                                threadgroup float *costs, threadgroup uint *best,
                                threadgroup const uint *sizes, threadgroup const uint *alive,
                                uint sg, uint sgs, uint lane) {
    for (uint i = sg; i < LSA_POOLS; i += sgs) {
        if (alive[i] == 0) {
            continue;
        }
        float own = sqrt(max(squares[i], 0.0f));
        float cheapest = INFINITY;
        uint partner = i;
        for (uint j = lane; j < LSA_POOLS; j += SIMD) {
            if (j == i || alive[j] == 0 || sizes[i] + sizes[j] > LSA_BUCKET_CAP) {
                continue;
            }
            float merged = sqrt(max(squares[i] + squares[j] + 2.0f * lsa_gram(gram, i, j), 0.0f));
            float cost = own + sqrt(max(squares[j], 0.0f)) - merged;
            if (cost < cheapest) {
                cheapest = cost;
                partner = j;
            }
        }
        float least = simd_min(cheapest);
        uint pick = simd_min(cheapest == least ? partner : 0xFFFFFFFFu);
        if (lane == 0) {
            costs[i] = least;
            best[i] = least < INFINITY ? pick : i;
        }
    }
}

// Merge each kept bucket's partner into it: key sums, sizes, and the Gram matrix, whose
// rows then columns add the partner's. Partners are never kept, so neither pass reads an
// entry it writes.
static inline void lsa_merge(device float *sums, device float *gram, threadgroup const uint *keep,
                             threadgroup const uint *best, threadgroup uint *sizes,
                             threadgroup uint *alive, threadgroup uint *parent, uint tid,
                             uint threads, uint sg, uint sgs, uint lane) {
    for (uint i = sg; i < LSA_POOLS; i += sgs) {
        if (keep[i] == 0) {
            continue;
        }
        uint j = best[i];
        sums[i * LSA_DIM + lane] += sums[j * LSA_DIM + lane];
        sums[i * LSA_DIM + lane + SIMD] += sums[j * LSA_DIM + lane + SIMD];
        if (lane == 0) {
            sizes[i] += sizes[j];
            alive[j] = 0;
            parent[j] = i;
        }
    }
    for (uint e = tid; e < LSA_POOLS * LSA_POOLS; e += threads) {
        uint i = e / LSA_POOLS;
        if (keep[i] != 0) {
            gram[e] += gram[best[i] * LSA_POOLS + e % LSA_POOLS];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint e = tid; e < LSA_POOLS * LSA_POOLS; e += threads) {
        uint i = e % LSA_POOLS;
        if (keep[i] != 0) {
            gram[e] += gram[e - i + best[i]];
        }
    }
}

// Ward rounds over one block's pools, as `lsa.core.ward.ward_rounds`: each round every
// alive bucket finds its cheapest partner under the cap (ties to the lower index), and
// mutual pairs merge into the lower one, cheapest first, until 64 remain. Merging costs
// |a| + |b| - |a + b| over key sums, from the Gram matrix a merge updates linearly.
static inline void lsa_ward(device float *sums, device float *gram, threadgroup float *squares,
                            threadgroup uint *best, threadgroup float *costs,
                            threadgroup uint *sizes, threadgroup uint *alive,
                            threadgroup uint *parent, threadgroup uint *keep,
                            threadgroup uint *count, uint tid, uint threads, uint sg,
                            uint sgs, uint lane) {
    for (uint e = tid; e < LSA_POOLS * LSA_POOLS; e += threads) {
        uint i = e / LSA_POOLS, j = e % LSA_POOLS;
        if (j >= i) {
            float g = 0.0f;
            for (uint d = 0; d < LSA_DIM; ++d) {
                g += sums[i * LSA_DIM + d] * sums[j * LSA_DIM + d];
            }
            gram[e] = g;
            gram[j * LSA_POOLS + i] = g;
        }
    }
    for (uint i = tid; i < LSA_POOLS; i += threads) {
        sizes[i] = LSA_POOL;
        alive[i] = 1;
        parent[i] = i;
    }
    if (tid == 0) {
        count[0] = LSA_POOLS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint round = 0; round < LSA_MAX_ROUNDS; ++round) {
        if (count[0] <= LSA_BUCKETS) {
            break;
        }
        uint left = count[0] - LSA_BUCKETS;
        for (uint i = tid; i < LSA_POOLS; i += threads) {
            squares[i] = gram[i * LSA_POOLS + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lsa_partners(gram, squares, costs, best, sizes, alive, sg, sgs, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < LSA_POOLS; i += threads) {
            uint j = best[i];
            keep[i] = alive[i] != 0 && costs[i] < INFINITY && j > i && best[j] == i ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Mutual pairs merge cheapest first, no further than the block's count.
        uint ranked[LSA_POOLS / 32 + 1];
        uint n = 0;
        for (uint i = tid; i < LSA_POOLS; i += threads) {
            uint rank = 0;
            if (keep[i] != 0) {
                for (uint j = 0; j < LSA_POOLS; ++j) {
                    rank += keep[j] != 0 && (costs[j] < costs[i] || (costs[j] == costs[i] && j < i)) ? 1u : 0u;
                }
            }
            ranked[n++] = keep[i] != 0 && rank < left ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n = 0;
        for (uint i = tid; i < LSA_POOLS; i += threads) {
            keep[i] = ranked[n++];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lsa_merge(sums, gram, keep, best, sizes, alive, parent, tid, threads, sg, sgs, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        // A round without a merge ends the clustering, as the reference's does.
        if (tid == 0) {
            uint merges = 0, remaining = 0;
            for (uint i = 0; i < LSA_POOLS; ++i) {
                merges += keep[i];
                remaining += alive[i];
            }
            count[0] = merges == 0 ? 0u : remaining;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (count[0] == 0) {
            break;
        }
    }
}

// One threadgroup per job (`first`, `block`, `slot`), after `lsa_pools` filled its sums:
// cluster the pools into index block `slot`. Writes each bucket's normalized key sum and
// its pools in ascending order, -1 past its size; buckets are numbered by their first pool.
kernel void lsa_ward_f32(
    device const uint *jobs [[buffer(0)]],
    device float *scratch [[buffer(1)]],
    device float *grams [[buffer(2)]],
    device float *centroids [[buffer(3)]],
    device float *members [[buffer(4)]],
    constant LsaClusterParams &p [[buffer(5)]],
    uint job [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float squares[LSA_POOLS];
    threadgroup float costs[LSA_POOLS];
    threadgroup uint best[LSA_POOLS];
    threadgroup uint sizes[LSA_POOLS];
    threadgroup uint alive[LSA_POOLS];
    threadgroup uint parent[LSA_POOLS];
    threadgroup uint keep[LSA_POOLS];
    threadgroup uint count[1];
    if (job >= p.jobs) {
        return;
    }
    ulong slot = jobs[job * 3 + 2];
    device float *sums = scratch + ulong(job) * LSA_POOLS * LSA_DIM;
    device float *gram = grams + ulong(job) * LSA_POOLS * LSA_POOLS;
    lsa_ward(sums, gram, squares, best, costs, sizes, alive, parent, keep, count, tid, threads,
             sg, sgs, lane);
    // Every merged pool points at the pool it merged into; its bucket is the chain's end,
    // numbered by the alive pools before it.
    for (uint i = tid; i < LSA_POOLS; i += threads) {
        uint root = i;
        while (parent[root] != root) {
            root = parent[root];
        }
        best[i] = root;
    }
    device float *row = members + slot * LSA_BUCKETS * LSA_BUCKET_POOLS;
    for (uint i = tid; i < LSA_BUCKETS * LSA_BUCKET_POOLS; i += threads) {
        row[i] = -1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint i = tid; i < LSA_POOLS; i += threads) {
        uint root = best[i], label = 0, rank = 0;
        for (uint j = 0; j < root; ++j) {
            label += alive[j];
        }
        for (uint j = 0; j < i; ++j) {
            rank += best[j] == root ? 1u : 0u;
        }
        if (label < LSA_BUCKETS && rank < LSA_BUCKET_POOLS) {
            row[label * LSA_BUCKET_POOLS + rank] = float(i);
        }
    }
    for (uint i = sg; i < LSA_POOLS; i += sgs) {
        if (alive[i] == 0) {
            continue;
        }
        uint label = 0;
        for (uint j = 0; j < i; ++j) {
            label += alive[j];
        }
        float s0 = sums[i * LSA_DIM + lane], s1 = sums[i * LSA_DIM + lane + SIMD];
        float inv = 1.0f / max(sqrt(simd_sum(s0 * s0 + s1 * s1)), 1e-30f);
        if (label < LSA_BUCKETS) {
            device float *c = centroids + (slot * LSA_BUCKETS + label) * LSA_DIM;
            c[lane] = s0 * inv;
            c[lane + SIMD] = s1 * inv;
        }
    }
}

// Index query of q row `t` (query heads summed per K/V group over the position-free dims,
// carried through the map) into `y`, and the inverse of its norm.
static inline float lsa_query(device const float *q, device const float *map, uint t,
                              threadgroup float *x, threadgroup float *y,
                              constant LsaScoreParams &p, uint tid, uint threads) {
    uint free = p.free_dims / p.kv_heads, per = p.heads / p.kv_heads;
    device const float *qt = q + ulong(t) * p.heads * p.dim;
    for (uint d = tid; d < p.free_dims; d += threads) {
        uint h = d / free, e = lsa_free_dim(d % free, p.first, p.split, p.second);
        float s = 0.0f;
        for (uint g = 0; g < per; ++g) {
            s += qt[(h * per + g) * p.dim + e];
        }
        x[d] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < LSA_DIM; j += threads) {
        float s = 0.0f;
        for (uint d = 0; d < p.free_dims; ++d) {
            s += x[d] * map[ulong(d) * LSA_DIM + j];
        }
        y[j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = 0.0f;
    for (uint j = 0; j < LSA_DIM; ++j) {
        norm += y[j] * y[j];
    }
    return 1.0f / max(sqrt(norm), 1e-12f);
}

// Grid (chunks of 256 buckets, groups). Group `t` is q rows [rows[2t], rows[2t] + rows[2t + 1]);
// each threadgroup forms its group's index query and scores the chunk's buckets of the
// closed blocks before the group's first row: logical bucket b is bucket b % 64 of index
// block b / 64. A group of several rows scores with the sum of their normalized index
// queries, the mean of their scores; a single row scales once at the end, as the
// reference does.
kernel void lsa_score_f32(
    device const float *q [[buffer(0)]],
    device const float *map [[buffer(1)]],
    device const float *centroids [[buffer(2)]],
    device const uint *index_table [[buffer(3)]],
    device const uint *index_first [[buffer(4)]],
    device const uint *visible [[buffer(5)]],
    device float *scores [[buffer(6)]],
    constant LsaScoreParams &p [[buffer(7)]],
    device const uint *rows [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 width [[threads_per_threadgroup]]) {
    threadgroup float x[LSA_MAX_FREE];
    threadgroup float y[LSA_DIM];
    threadgroup float sum[LSA_DIM];
    uint tid = local.x, threads = width.x;
    uint t = group.y;
    if (t >= p.groups) {
        return;
    }
    uint pos = visible[t] - 1;
    uint buckets = (pos / LSA_BLOCK) * LSA_BUCKETS;
    uint start = group.x * LSA_SCORE_CHUNK;
    if (start >= buckets) {
        return;
    }
    uint r0 = rows[2 * t], n = rows[2 * t + 1];
    float inv = lsa_query(q, map, r0, x, y, p, tid, threads);
    threadgroup float *query = y;
    if (n > 1) {
        for (uint j = tid; j < LSA_DIM; j += threads) {
            sum[j] = y[j] * inv;
        }
        for (uint r = r0 + 1; r < r0 + n; ++r) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float scale = lsa_query(q, map, r, x, y, p, tid, threads);
            for (uint j = tid; j < LSA_DIM; j += threads) {
                sum[j] += y[j] * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        query = sum;
        inv = 1.0f;
    }
    for (uint b = start + tid; b < min(buckets, start + LSA_SCORE_CHUNK); b += threads) {
        uint phys = index_table[index_first[t] + b / LSA_BUCKETS];
        device const float4 *c = (device const float4 *)(centroids + (ulong(phys) * LSA_BUCKETS + b % LSA_BUCKETS) * LSA_DIM);
        float s = 0.0f;
        for (uint j = 0; j < LSA_DIM / 4; ++j) {
            float4 cj = c[j];
            s += query[4 * j] * cj.x + query[4 * j + 1] * cj.y + query[4 * j + 2] * cj.z + query[4 * j + 3] * cj.w;
        }
        scores[ulong(t) * p.score_stride + b] = s * inv;
    }
}

// Order-preserving map of finite floats to unsigned keys; -0 scores as +0, as torch ties them.
static inline uint lsa_order(float f) {
    uint u = as_type<uint>(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sums of `values[0..n)` in place, returning the total. Each thread owns
// one contiguous run, so any threadgroup width serves any n.
static inline uint lsa_scan(threadgroup uint *values, threadgroup uint *partial, uint n,
                            uint tid, uint threads, uint sg, uint sgs, uint lane) {
    uint per = (n + threads - 1) / threads;
    uint begin = min(n, tid * per), end = min(n, begin + per);
    uint local = 0;
    for (uint i = begin; i < end; ++i) {
        local += values[i];
    }
    uint inclusive = simd_prefix_inclusive_sum(local);
    if (lane == SIMD - 1) {
        partial[sg] = inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint s = lane < sgs ? partial[lane] : 0u;
        uint before = simd_prefix_exclusive_sum(s);
        uint all = simd_sum(s);
        partial[lane] = before;
        if (lane == 0) {
            partial[SIMD] = all;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint run = partial[sg] + inclusive - local;
    for (uint i = begin; i < end; ++i) {
        uint v = values[i];
        values[i] = run;
        run += v;
    }
    uint total = partial[SIMD];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// The P-th largest key among `n` scores (as `lsa_order`) and how many of the keys equal to
// it are needed, by four 8-bit radix passes.
static inline uint2 lsa_threshold(device const float *row, uint n, uint want,
                                  threadgroup atomic_uint *hist, threadgroup uint *shared,
                                  uint tid, uint threads) {
    uint prefix = 0, mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += threads) {
            atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += threads) {
            uint k = lsa_order(row[i]);
            if ((k & mask) == prefix) {
                atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint above = 0, bin = 255;
            for (;; --bin) {
                uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (above + c >= want || bin == 0) {
                    break;
                }
                above += c;
            }
            shared[0] = prefix | (bin << shift);
            shared[1] = want - above;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = shared[0];
        want = shared[1];
        mask |= 255u << shift;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return uint2(prefix, want);
}

// Bitonic sort of `items[0..n)` descending; entries past `n` are padded with 0.
static inline void lsa_sort(threadgroup ulong *items, uint n, uint tid, uint threads) {
    uint size = 1;
    while (size < n) {
        size <<= 1;
    }
    for (uint i = n + tid; i < size; i += threads) {
        items[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= size; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < size; i += threads) {
                uint l = i ^ j;
                if (l > i) {
                    ulong a = items[i], b = items[l];
                    bool descending = (i & k) == 0;
                    if (descending ? a < b : a > b) {
                        items[i] = b;
                        items[l] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Keys of logical bucket `bucket` that a query at `pos` reads beyond its sinks and window:
// writes up to `limit` positions from `out` and returns how many the bucket holds.
static inline uint lsa_bucket_keys(device const float *members, device const uint *index_table,
                                   uint first, uint bucket, uint sink, uint window_start,
                                   device float *out, uint limit) {
    uint block = bucket / LSA_BUCKETS;
    uint phys = index_table[first + block];
    device const float *row = members + (ulong(phys) * LSA_BUCKETS + bucket % LSA_BUCKETS) * LSA_BUCKET_POOLS;
    uint kept = 0;
    for (uint m = 0; m < LSA_BUCKET_POOLS; ++m) {
        float pool = row[m];
        if (pool < 0.0f) {
            break;
        }
        for (uint r = 0; r < LSA_POOL; ++r) {
            uint key = block * LSA_BLOCK + uint(pool) * LSA_POOL + r;
            if (key >= sink && key < window_start) {
                if (kept < limit) {
                    out[kept] = float(key);
                }
                ++kept;
            }
        }
    }
    return kept;
}

// One threadgroup per token, as `lsa.core.reference.read`: the top-p buckets by score
// (ties to the lower bucket), expanded best first until top-k keys outside the sinks and
// window are held, then the window, then the sinks; -1 past the row's length.
kernel void lsa_select_f32(
    device const float *scores [[buffer(0)]],
    device const float *members [[buffer(1)]],
    device const uint *index_table [[buffer(2)]],
    device const uint *index_first [[buffer(3)]],
    device const uint *visible [[buffer(4)]],
    device float *keys [[buffer(5)]],
    device float *lengths [[buffer(6)]],
    constant LsaSelectParams &p [[buffer(7)]],
    device const uint *rows [[buffer(8)]],
    uint t [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup ulong items[LSA_MAX_P];
    threadgroup uint counts[LSA_MAX_P];
    threadgroup uint partial[SIMD + 1];
    threadgroup uint shared[2];
    threadgroup atomic_uint cursor;
    if (t >= p.groups) {
        return;
    }
    uint pos = visible[t] - 1;
    uint buckets = (pos / LSA_BLOCK) * LSA_BUCKETS;
    uint window_start = pos + 1 > p.window ? pos + 1 - p.window : 0u;
    uint first = index_first[t];
    device const float *row = scores + ulong(t) * p.score_stride;
    device float *out = keys + ulong(t) * p.width;
    uint picked = min(p.top_p, buckets);
    if (buckets > p.top_p) {
        uint2 cut = lsa_threshold(row, buckets, picked, hist, shared, tid, threads);
        uint threshold = cut.x, ties = cut.y;
        if (tid == 0) {
            atomic_store_explicit(&cursor, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < buckets; i += threads) {
            uint k = lsa_order(row[i]);
            if (k > threshold) {
                uint at = atomic_fetch_add_explicit(&cursor, 1u, memory_order_relaxed);
                items[at] = (ulong(k) << 32) | ulong(0xFFFFFFFFu - i);
            }
        }
        // Ties at the threshold go to the lowest buckets, in order.
        if (sg == 0) {
            uint taken = 0, base = picked - ties;
            for (uint b0 = 0; b0 < buckets && taken < ties; b0 += SIMD) {
                uint i = b0 + lane;
                bool tie = i < buckets && lsa_order(row[i]) == threshold;
                uint offset = simd_prefix_exclusive_sum(tie ? 1u : 0u);
                if (tie && taken + offset < ties) {
                    items[base + taken + offset] = (ulong(threshold) << 32) | ulong(0xFFFFFFFFu - i);
                }
                taken += simd_sum(tie ? 1u : 0u);
            }
        }
    } else {
        for (uint i = tid; i < buckets; i += threads) {
            items[i] = (ulong(lsa_order(row[i])) << 32) | ulong(0xFFFFFFFFu - i);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_sort(items, picked, tid, threads);
    for (uint c = tid; c < picked; c += threads) {
        uint bucket = 0xFFFFFFFFu - uint(items[c] & 0xFFFFFFFFul);
        counts[c] = lsa_bucket_keys(members, index_table, first, bucket, p.sink, window_start, out, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = lsa_scan(counts, partial, picked, tid, threads, sg, sgs, lane);
    for (uint c = tid; c < picked; c += threads) {
        uint offset = counts[c];
        if (offset < p.top_k) {
            uint bucket = 0xFFFFFFFFu - uint(items[c] & 0xFFFFFFFFul);
            lsa_bucket_keys(members, index_table, first, bucket, p.sink, window_start,
                            out + offset, p.top_k - offset);
        }
    }
    uint expanded = min(total, p.top_k);
    uint tail = p.tiled != 0 ? rows[2 * t + 1] - 1 : 0u;
    uint recent = pos + 1 + tail - window_start;
    uint sinks = min(min(p.sink, window_start), pos + 1);
    uint recent_at = expanded + (p.tiled != 0 ? sinks : 0u);
    uint sinks_at = expanded + (p.tiled != 0 ? 0u : recent);
    for (uint i = tid; i < recent; i += threads) {
        out[recent_at + i] = float(window_start + i);
    }
    for (uint i = tid; i < sinks; i += threads) {
        out[sinks_at + i] = float(i);
    }
    uint length = expanded + recent + sinks;
    for (uint i = length + tid; i < p.width; i += threads) {
        out[i] = -1.0f;
    }
    if (tid == 0) {
        lengths[t] = float(length);
    }
}

#define LSA_POOLS_KERNEL(NAME, T, QUANT) \
    template [[host_name("lsa_pools_" #NAME)]] kernel void lsa_pools<T, QUANT>( \
        device const T *, device const half *, device const uint *, device const uint *, \
        device const float *, device float *, constant LsaClusterParams &, uint2, uint, uint, \
        uint);

LSA_POOLS_KERNEL(f32, float, 0)
LSA_POOLS_KERNEL(f16, half, 0)
LSA_POOLS_KERNEL(q8, char, 1)
