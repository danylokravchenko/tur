struct MoeRouteParams { uint tokens, experts, top_k, stride; };
struct ExpertParams { uint rows, experts, top_k, width, stride, out, routed; };
struct ScatterParams { uint tokens, top_k, width, stride, shared, shared_stride, gate_stride; };

// Assignments per histogram block; also the threads of the histogram, offset and scatter
// threadgroups, which cover up to 256 experts with one thread each.
#define MOE_BLOCK 256u
#define MOE_LANE_EXPERTS 8u
#define MOE_TILE 16u
// Groups above MOE_TILE rows split into MOE_WIDE-row tiles for moe_gemm_wide.
#define MOE_WIDE 32u

// One simdgroup per token; each lane keeps eight scores in registers. Top-k takes k
// simdgroup max reductions over exact F32 scores, ties to the lowest expert ID (both
// signs of zero compare equal), and normalizes in rank order as the host reference does.
kernel void moe_route_f32(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant MoeRouteParams &p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float scores[MOE_LANE_EXPERTS];
    bool taken[MOE_LANE_EXPERTS];
    bool valid = true;
    for (uint i = 0; i < MOE_LANE_EXPERTS; ++i) {
        uint e = lane + 32 * i;
        taken[i] = e >= p.experts;
        scores[i] = taken[i] ? -INFINITY : x[ulong(row) * p.stride + e];
        valid = valid && (taken[i] || isfinite(scores[i]));
    }
    valid = simd_all(valid);
    device float *out = y + ulong(row) * p.top_k * 2;
    if (!valid) {
        for (uint i = lane; i < p.top_k; i += 32) {
            // Keep every permutation row initialized even for a failed token. NaN
            // weights mark its output invalid without producing unchecked addresses.
            out[i * 2] = float(i);
            out[i * 2 + 1] = NAN;
        }
        return;
    }
    float top = 0.0f, sum = 0.0f;
    for (uint n = 0; n < p.top_k; ++n) {
        float best = -INFINITY;
        uint best_id = 0xFFFFFFFFu;
        for (uint i = 0; i < MOE_LANE_EXPERTS; ++i) {
            if (!taken[i] && (best_id == 0xFFFFFFFFu || scores[i] > best)) {
                best = scores[i];
                best_id = lane + 32 * i;
            }
        }
        float m = simd_max(best);
        uint id = simd_min(best_id != 0xFFFFFFFFu && best == m ? best_id : 0xFFFFFFFFu);
        for (uint i = 0; i < MOE_LANE_EXPERTS; ++i) taken[i] = taken[i] || lane + 32 * i == id;
        top = n == 0 ? m : top;
        float weight = exp(m - top);
        sum += weight;
        if (lane == 0) { out[n * 2] = float(id); out[n * 2 + 1] = weight; }
    }
    if (lane == 0) for (uint n = 0; n < p.top_k; ++n) out[n * 2 + 1] /= sum;
}

// Per block of assignments: each expert's count, and each assignment's rank among the
// earlier assignments of its expert in the block (kept in its permutation slot).
kernel void moe_histogram_f32(device const float *selected [[buffer(0)]], device float *hist [[buffer(1)]],
    device float *order [[buffer(2)]], constant MoeRouteParams &p [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ids[MOE_BLOCK];
    uint total = p.tokens * p.top_k, a = block * MOE_BLOCK + tid;
    ids[tid] = a < total ? uint(selected[ulong(a) * 2]) : 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a < total) {
        uint rank = 0;
        for (uint j = 0; j < tid; ++j) rank += ids[j] == ids[tid];
        order[ulong(a) * 3 + 2] = float(rank);
    }
    if (tid < p.experts) {
        uint count = 0;
        for (uint j = 0; j < MOE_BLOCK; ++j) count += ids[j] == tid;
        hist[ulong(block) * p.experts + tid] = float(count);
    }
}

// One threadgroup: block counts become per-expert exclusive prefixes, then expert offsets
// and tile starts are scanned across experts and every expert writes its own tiles
// (expert, first row, rows, wide): one tile for a group of at most MOE_TILE rows, MOE_WIDE-row
// wide tiles above it. Unused tile slots up to the bound get a negative expert.
kernel void moe_offsets_f32(device float *hist [[buffer(0)]], device float *groups [[buffer(1)]],
    device float *tiles [[buffer(2)]], constant MoeRouteParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint2 scan[MOE_BLOCK];
    uint total = p.tokens * p.top_k, blocks = (total + MOE_BLOCK - 1) / MOE_BLOCK;
    uint count = 0;
    if (tid < p.experts) {
        for (uint b = 0; b < blocks; ++b) {
            ulong at = ulong(b) * p.experts + tid;
            uint c = uint(hist[at]);
            hist[at] = float(count);
            count += c;
        }
    }
    bool wide = count > MOE_TILE;
    uint size = wide ? MOE_WIDE : MOE_TILE;
    uint2 own = uint2(count, (count + size - 1) / size);
    scan[tid] = own;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < MOE_BLOCK; step *= 2) {
        uint2 add = tid >= step ? scan[tid - step] : uint2(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint2 start = scan[tid] - own;
    if (tid < p.experts) {
        groups[tid] = float(start.x);
        for (uint row = 0; row < count; row += size) {
            ulong t = ulong(start.y + row / size) * 4;
            tiles[t] = float(tid); tiles[t + 1] = float(start.x + row);
            tiles[t + 2] = float(min(size, count - row)); tiles[t + 3] = float(wide);
        }
    }
    uint used = scan[MOE_BLOCK - 1].y;
    uint cap = min(total, (total + MOE_TILE - 1) / MOE_TILE + p.experts - 1);
    for (uint t = used + tid; t < cap; t += MOE_BLOCK) {
        tiles[ulong(t) * 4] = -1.0f; tiles[ulong(t) * 4 + 1] = 0.0f; tiles[ulong(t) * 4 + 2] = 0.0f;
        tiles[ulong(t) * 4 + 3] = 0.0f;
    }
}

// Rows of one expert keep ascending assignment order, so the permutation (and which rows
// share a tile) is deterministic. Slot 2 holds the block rank on entry, the row on exit.
kernel void moe_order_f32(device const float *selected [[buffer(0)]], device const float *hist [[buffer(1)]],
    device const float *groups [[buffer(2)]], device float *order [[buffer(3)]],
    constant MoeRouteParams &p [[buffer(4)]], uint a [[thread_position_in_grid]]) {
    if (a >= p.tokens * p.top_k) return;
    uint expert = uint(selected[ulong(a) * 2]);
    uint row = uint(groups[expert]) + uint(hist[ulong(a / MOE_BLOCK) * p.experts + expert])
        + uint(order[ulong(a) * 3 + 2]);
    order[ulong(row) * 3] = float(a); order[ulong(row) * 3 + 1] = float(expert);
    order[ulong(a) * 3 + 2] = float(row);
}

kernel void moe_gather_f32(device const float *x [[buffer(0)]], device const float *order [[buffer(1)]],
    device float *y [[buffer(2)]], constant ExpertParams &p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= ulong(p.rows) * p.width) return;
    uint row = tid / p.width, col = tid % p.width;
    y[tid] = x[ulong(uint(order[ulong(row) * 3]) / p.top_k) * p.stride + col];
}

// Input row of group row `r`: gathered rows are read directly; routed reads index the
// token through the permutation, so no gathered copy of the activations is needed.
inline device const float *expert_input(device const float *x, device const float *order,
    constant ExpertParams &p, uint r) {
    uint row = p.routed ? uint(order[ulong(r) * 3]) / p.top_k : r;
    return x + ulong(row) * p.stride;
}

// Q4_K GEMV over up to three rows, laid out like the dense kernel: eight lanes share a
// super-block, each reading 16 packed bytes (16 values of two sub-blocks), and the minima
// fold into one activation sum per sub-block.
template <uint NR> inline void expert_gemv_q4k(device const uchar *w, ulong base, uint col, uint out, uint width,
    thread device const float *const (&xs)[3], uint count, uint lane, thread float (&acc)[NR][3]) {
    uint blocks = width / 256, ix = lane / 8, slot = lane % 8, group = slot / 2, off = 16 * (slot % 2);
    for (uint ib = ix; ib < blocks; ib += 4) {
        uint4 packed[NR];
        float d_lo[NR], m_lo[NR], d_hi[NR], m_hi[NR];
        for (uint r = 0; r < NR; ++r) {
            device const block_q4_k &blk = ((device const block_q4_k *)w)[(base + min(col + r, out - 1)) * blocks + ib];
            float d = float(blk.d), dmin = float(blk.dmin);
            uchar2 first = q4k_scale_min(2 * group, blk.scales), second = q4k_scale_min(2 * group + 1, blk.scales);
            d_lo[r] = d * float(first.x); m_lo[r] = dmin * float(first.y);
            d_hi[r] = d * float(second.x); m_hi[r] = dmin * float(second.y);
            packed[r] = *((device const uint4 *)(blk.qs + 32 * group + off));
        }
        for (uint b = 0; b < 3; ++b) {
            if (b >= count) break;
            device const packed_float4 *xl = (device const packed_float4 *)(xs[b] + ib * 256 + 64 * group + off);
            float4 lo_x[4], hi_x[4];
            float sum_lo = 0.0f, sum_hi = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                lo_x[i] = float4(xl[i]); hi_x[i] = float4(xl[i + 8]);
                sum_lo += lo_x[i].x + lo_x[i].y + lo_x[i].z + lo_x[i].w;
                sum_hi += hi_x[i].x + hi_x[i].y + hi_x[i].z + hi_x[i].w;
            }
            for (uint r = 0; r < NR; ++r) {
                float dot_lo = 0.0f, dot_hi = 0.0f;
                for (uint i = 0; i < 4; ++i) {
                    uint word = packed[r][i];
                    dot_lo += dot(float4(word & 0xF, (word >> 8) & 0xF, (word >> 16) & 0xF, (word >> 24) & 0xF), lo_x[i]);
                    dot_hi += dot(float4((word >> 4) & 0xF, (word >> 12) & 0xF, (word >> 20) & 0xF, word >> 28), hi_x[i]);
                }
                acc[r][b] += d_lo[r] * dot_lo - m_lo[r] * sum_lo + d_hi[r] * dot_hi - m_hi[r] * sum_hi;
            }
        }
    }
}

// GEMV over small-group tiles of at most three rows; larger ones are left to moe_gemm. One simdgroup
// computes NR outputs (four, or two for Q4_K), as many as the dense GEMV, so decode keeps
// its parallelism.
template <uint FORMAT, uint NR> kernel void moe_gemv(device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]], device const float *tiles [[buffer(2)]],
    device float *y [[buffer(3)]], constant ExpertParams &p [[buffer(4)]],
    device const float *order [[buffer(5)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint columns = (p.out + NR * sgs - 1) / (NR * sgs), tile = tg / columns;
    uint col = (tg % columns) * NR * sgs + sg * NR;
    device const float *d = tiles + tile * 4;
    if (d[0] < 0.0f || d[3] != 0.0f || d[2] > 3.0f || col >= p.out) return;
    uint expert = uint(d[0]), offset = uint(d[1]), count = uint(d[2]);
    ulong base = ulong(expert) * p.out;
    device const float *xs[3];
    for (uint b = 0; b < 3; ++b) xs[b] = expert_input(x, order, p, offset + min(b, count - 1));
    float acc[NR][3];
    for (uint r = 0; r < NR; ++r) for (uint b = 0; b < 3; ++b) acc[r][b] = 0.0f;
    if (FORMAT == 2) {
        expert_gemv_q4k<NR>(w, base, col, p.out, p.width, xs, count, lane, acc);
    } else {
        uint ix = lane / 4, start = (lane % 4) * 8;
        for (uint chunk = ix; chunk < p.width / 32; chunk += 8) {
            float4 lo[NR], hi[NR];
            for (uint r = 0; r < NR; ++r) {
                packed_weights8<FORMAT>(w, base + min(col + r, p.out - 1), chunk, start, p.width, lo[r], hi[r]);
            }
            for (uint b = 0; b < 3; ++b) {
                if (b < count) {
                    device const float *xb = xs[b] + chunk * 32 + start;
                    float4 xl = float4(xb[0], xb[1], xb[2], xb[3]);
                    float4 xh = float4(xb[4], xb[5], xb[6], xb[7]);
                    for (uint r = 0; r < NR; ++r) acc[r][b] += dot(lo[r], xl) + dot(hi[r], xh);
                }
            }
        }
    }
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < 3; ++b) {
            float total = simd_sum(acc[r][b]);
            if (lane == 0 && b < count && col + r < p.out) y[ulong(offset + b) * p.out + col + r] = total;
        }
    }
}

// Matrix-unit GEMM over small-group tiles of four or more rows (F32 throughout).
// Each simdgroup owns a sixteen-row tile's 32 outputs over the whole reduction, reusing
// its vector-loaded weights across sixteen rows and its activations across 32 outputs,
// with no threadgroup memory or barriers.
template <uint FORMAT> kernel void moe_gemm(device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]], device const float *tiles [[buffer(2)]],
    device float *y [[buffer(3)]], constant ExpertParams &p [[buffer(4)]],
    device const float *order [[buffer(5)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint columns = (p.out + 32 * sgs - 1) / (32 * sgs), tile = tg / columns;
    uint col0 = (tg % columns) * 32 * sgs + sg * 32;
    device const float *d = tiles + tile * 4;
    if (d[0] < 0.0f || d[3] != 0.0f || d[2] <= 3.0f || col0 >= p.out) return;
    uint expert = uint(d[0]), offset = uint(d[1]), count = uint(d[2]);
    ulong base = ulong(expert) * p.out;
    uint qid = lane / 4;
    uint fr = (qid & 4) + (lane / 2) % 4, fc = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc[4][2];
    for (uint t = 0; t < 4; ++t) for (uint b = 0; b < 2; ++b) acc[t][b] = simdgroup_float8x8(0.0f);
    device const float *xs[2][2];
    bool live[2][2];
    for (uint b = 0; b < 2; ++b) {
        for (uint e = 0; e < 2; ++e) {
            uint row = b * 8 + fc + e;
            live[b][e] = row < count;
            xs[b][e] = expert_input(x, order, p, offset + min(row, count - 1)) + 4 * fr;
        }
    }
    ulong wr[4];
    for (uint t = 0; t < 4; ++t) wr[t] = base + min(col0 + t * 8 + fr, p.out - 1);
    for (uint chunk = 0; chunk < p.width / 32; ++chunk) {
        float4 xv[2][2];
        for (uint b = 0; b < 2; ++b) {
            for (uint e = 0; e < 2; ++e) {
                device const float *v = xs[b][e] + chunk * 32;
                xv[b][e] = live[b][e] ? float4(v[0], v[1], v[2], v[3]) : float4(0.0f);
            }
        }
        // Lane (fr, fc) holds weights 4·fc..4·fc+7 of its row; reduction index j of
        // step s is element 4j + s, matching the activations loaded at 4·fr + s.
        float4 lo[4], hi[4];
        for (uint t = 0; t < 4; ++t) packed_weights8<FORMAT>(w, wr[t], chunk, 4 * fc, p.width, lo[t], hi[t]);
        for (uint s = 0; s < 4; ++s) {
            simdgroup_float8x8 xm[2];
            for (uint b = 0; b < 2; ++b) { xm[b].thread_elements()[0] = xv[b][0][s]; xm[b].thread_elements()[1] = xv[b][1][s]; }
            for (uint t = 0; t < 4; ++t) {
                simdgroup_float8x8 wm;
                wm.thread_elements()[0] = lo[t][s]; wm.thread_elements()[1] = hi[t][s];
                for (uint b = 0; b < 2; ++b) simdgroup_multiply_accumulate(acc[t][b], wm, xm[b], acc[t][b]);
            }
        }
    }
    for (uint t = 0; t < 4; ++t) {
        for (uint b = 0; b < 2; ++b) {
            for (uint i = 0; i < 2; ++i) {
                uint row = b * 8 + fc + i, col = col0 + t * 8 + fr;
                if (row < count && col < p.out) y[ulong(offset + row) * p.out + col] = acc[t][b].thread_elements()[i];
            }
        }
    }
}

// Wide tiles of groups above MOE_TILE rows: ggml's mul_mm layout. A threadgroup of four
// simdgroups owns 64 outputs × 32 rows; each 32-wide reduction slice of weights is
// dequantized once into threadgroup memory as F16 and activations staged as F32, then each
// simdgroup multiplies a 32-output × 16-row block with F32 accumulation. Like Candle's
// tiled GEMM for dense projections above sixteen rows, weights round to F16.
template <uint FORMAT> kernel void moe_gemm_wide(device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]], device const float *tiles [[buffer(2)]],
    device float *y [[buffer(3)]], constant ExpertParams &p [[buffer(4)]],
    device const float *order [[buffer(5)]], threadgroup char *stage [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint columns = (p.out + 63) / 64, tile = tg / columns, col0 = (tg % columns) * 64;
    device const float *d = tiles + tile * 4;
    if (d[0] < 0.0f || d[3] == 0.0f) return;
    uint expert = uint(d[0]), offset = uint(d[1]), count = uint(d[2]);
    ulong base = ulong(expert) * p.out;
    threadgroup half *ws = (threadgroup half *)stage;                  // [64][32] outputs × k
    threadgroup float *xs = (threadgroup float *)(stage + 64 * 32 * 2); // [32][32] rows × k
    // Staging: sixteen weights of one output row, eight activations of one row per thread.
    uint w_row = tid / 2, w_start = (tid % 2) * 16;
    ulong w_global = base + min(col0 + w_row, p.out - 1);
    uint x_row = tid / 4, x_start = (tid % 4) * 8;
    device const float *x_src = expert_input(x, order, p, offset + min(x_row, count - 1)) + x_start;
    bool x_live = x_row < count;
    uint out0 = (sg % 2) * 32, row0 = (sg / 2) * 16;
    simdgroup_float8x8 acc[4][2];
    for (uint t = 0; t < 4; ++t) for (uint b = 0; b < 2; ++b) acc[t][b] = simdgroup_float8x8(0.0f);
    for (uint chunk = 0; chunk < p.width / 32; ++chunk) {
        for (uint h = 0; h < 2; ++h) {
            float4 lo, hi;
            packed_weights8<FORMAT>(w, w_global, chunk, w_start + 8 * h, p.width, lo, hi);
            threadgroup half *wd = ws + w_row * 32 + w_start + 8 * h;
            *((threadgroup half4 *)wd) = half4(lo); *((threadgroup half4 *)(wd + 4)) = half4(hi);
        }
        device const float *v = x_src + chunk * 32;
        threadgroup float *xd = xs + x_row * 32 + x_start;
        for (uint i = 0; i < 8; ++i) xd[i] = x_live ? v[i] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 xm[2];
            for (uint b = 0; b < 2; ++b) simdgroup_load(xm[b], xs + (row0 + b * 8) * 32 + k, 32);
            for (uint t = 0; t < 4; ++t) {
                // B[k][c] = W[out0 + 8t + c][k]: the staged weights loaded transposed.
                simdgroup_half8x8 wm;
                simdgroup_load(wm, ws + (out0 + t * 8) * 32 + k, 32, ulong2(0, 0), true);
                for (uint b = 0; b < 2; ++b) simdgroup_multiply_accumulate(acc[t][b], xm[b], wm, acc[t][b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Rows and outputs outside the tile are never written: stage, then copy guarded.
    threadgroup float *result = (threadgroup float *)stage;             // [32][64]
    for (uint t = 0; t < 4; ++t) {
        for (uint b = 0; b < 2; ++b) simdgroup_store(acc[t][b], result + (row0 + b * 8) * 64 + out0 + t * 8, 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * 64; i += 128) {
        uint row = i / 64, col = col0 + i % 64;
        if (row < count && col < p.out) y[ulong(offset + row) * p.out + col] = result[i];
    }
}

#define EXPERT_GEMM(FORMAT, NR, SUFFIX) \
template [[host_name("moe_gemv_" SUFFIX)]] kernel void moe_gemv<FORMAT, NR>(device const uchar *, device const float *, \
    device const float *, device float *, constant ExpertParams &, device const float *, uint, uint, uint, uint); \
template [[host_name("moe_gemm_" SUFFIX)]] kernel void moe_gemm<FORMAT>(device const uchar *, device const float *, \
    device const float *, device float *, constant ExpertParams &, device const float *, uint, uint, uint, uint); \
template [[host_name("moe_gemm_wide_" SUFFIX)]] kernel void moe_gemm_wide<FORMAT>(device const uchar *, \
    device const float *, device const float *, device float *, constant ExpertParams &, device const float *, \
    threadgroup char *, uint, uint, uint);
EXPERT_GEMM(0, 4, "f32")
EXPERT_GEMM(1, 4, "q8_0")
EXPERT_GEMM(2, 2, "q4k")

// One threadgroup per token: its experts are ranked by ID once, then every column sums
// them in ascending ID order without floating-point atomics. An optional shared expert
// is added with its sigmoid gate in the same pass.
kernel void moe_scatter_f32(device const float *x [[buffer(0)]], device const float *order [[buffer(1)]],
    device const float *selected [[buffer(2)]], device const float *shared [[buffer(3)]],
    device const float *gate [[buffer(4)]], device float *y [[buffer(5)]],
    constant ScatterParams &p [[buffer(6)]], uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    threadgroup uint rows[MOE_BLOCK];
    threadgroup float weights[MOE_BLOCK];
    if (tid < p.top_k) {
        ulong a = ulong(token) * p.top_k + tid;
        float expert = selected[a * 2];
        uint rank = 0;
        for (uint k = 0; k < p.top_k; ++k) rank += selected[(ulong(token) * p.top_k + k) * 2] < expert;
        rows[rank] = uint(order[a * 3 + 2]);
        weights[rank] = selected[a * 2 + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g = p.shared ? 1.0f / (1.0f + exp(-gate[ulong(token) * p.gate_stride])) : 0.0f;
    for (uint col = tid; col < p.width; col += threads) {
        float sum = 0.0f;
        for (uint n = 0; n < p.top_k; ++n) sum += x[ulong(rows[n]) * p.stride + col] * weights[n];
        if (p.shared) sum += g * shared[ulong(token) * p.shared_stride + col];
        y[ulong(token) * p.width + col] = sum;
    }
}
