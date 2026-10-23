struct MoeRouteParams { uint tokens, experts, top_k, stride; };
struct ExpertParams { uint rows, experts, top_k, width, stride, out; };

// One simdgroup per token; sorting ranks retain exact F32 scores and break ties by ID.
kernel void moe_route_f32(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant MoeRouteParams &p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    bool valid = true;
    float maximum = -INFINITY;
    for (uint e = lane; e < p.experts; e += 32) {
        float score = x[ulong(row) * p.stride + e];
        valid = valid && isfinite(score);
        maximum = max(maximum, score);
    }
    valid = simd_all(valid);
    maximum = simd_max(maximum);
    if (!valid) {
        for (uint i = lane; i < p.top_k; i += 32) {
            // Keep every permutation row initialized even for a failed token. NaN
            // weights mark its output invalid without producing unchecked addresses.
            y[(ulong(row) * p.top_k + i) * 2] = float(i);
            y[(ulong(row) * p.top_k + i) * 2 + 1] = NAN;
        }
        return;
    }
    float sum = 0.0f;
    for (uint e = lane; e < p.experts; e += 32) {
        float score = x[ulong(row) * p.stride + e];
        uint rank = 0;
        for (uint j = 0; j < p.experts; ++j) {
            float other = x[ulong(row) * p.stride + j];
            rank += other > score || (other == score && j < e);
        }
        if (rank < p.top_k) {
            float weight = exp(score - maximum);
            y[(ulong(row) * p.top_k + rank) * 2] = float(e);
            y[(ulong(row) * p.top_k + rank) * 2 + 1] = weight;
            sum += weight;
        }
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = lane; i < p.top_k; i += 32) {
        y[(ulong(row) * p.top_k + i) * 2 + 1] /= sum;
    }
}

kernel void moe_counts_f32(device const float *selected [[buffer(0)]], device float *counts [[buffer(1)]],
    constant MoeRouteParams &p [[buffer(2)]], uint expert [[thread_position_in_grid]]) {
    if (expert >= p.experts) return;
    uint count = 0;
    for (uint i = 0; i < p.tokens * p.top_k; ++i) count += selected[ulong(i) * 2] == float(expert);
    counts[expert] = float(count);
}

kernel void moe_offsets_f32(device const float *counts [[buffer(0)]], device float *groups [[buffer(1)]],
    device float *tiles [[buffer(2)]], device float *wide_tiles [[buffer(3)]],
    constant MoeRouteParams &p [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint offset = 0, tile = 0, wide_tile = 0;
    for (uint e = 0; e < p.experts; ++e) {
        uint count = uint(counts[e]);
        groups[e * 6] = float(offset); groups[e * 6 + 1] = float(count);
        groups[e * 6 + 2] = float(tile); groups[e * 6 + 3] = float((count + 7) / 8);
        groups[e * 6 + 4] = float(wide_tile); groups[e * 6 + 5] = float((count + 15) / 16);
        for (uint row = 0; row < count; row += 8) {
            tiles[(tile + row / 8) * 3] = float(e);
            tiles[(tile + row / 8) * 3 + 1] = float(offset + row);
            tiles[(tile + row / 8) * 3 + 2] = float(min(8u, count - row));
        }
        for (uint row = 0; row < count; row += 16) {
            wide_tiles[(wide_tile + row / 16) * 3] = float(e);
            wide_tiles[(wide_tile + row / 16) * 3 + 1] = float(offset + row);
            wide_tiles[(wide_tile + row / 16) * 3 + 2] = float(min(16u, count - row));
        }
        offset += count; tile += (count + 7) / 8; wide_tile += (count + 15) / 16;
    }
    uint rows = p.tokens * p.top_k;
    uint tile_cap = min(rows, (rows + 7) / 8 + p.experts - 1);
    uint wide_cap = min(rows, (rows + 15) / 16 + p.experts - 1);
    for (; tile < tile_cap; ++tile) { tiles[tile * 3] = -1.0f; tiles[tile * 3 + 1] = 0.0f; tiles[tile * 3 + 2] = 0.0f; }
    for (; wide_tile < wide_cap; ++wide_tile) { wide_tiles[wide_tile * 3] = -1.0f; wide_tiles[wide_tile * 3 + 1] = 0.0f; wide_tiles[wide_tile * 3 + 2] = 0.0f; }
}

kernel void moe_order_f32(device const float *selected [[buffer(0)]], device const float *groups [[buffer(1)]],
    device float *order [[buffer(2)]], constant MoeRouteParams &p [[buffer(3)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert >= p.experts) return;
    uint row = uint(groups[expert * 6]);
    for (uint i = 0; i < p.tokens * p.top_k; ++i) {
        if (selected[ulong(i) * 2] == float(expert)) {
            order[ulong(row) * 3] = float(i); order[ulong(row) * 3 + 1] = float(expert);
            order[ulong(i) * 3 + 2] = float(row);
            ++row;
        }
    }
}

kernel void moe_gather_f32(device const float *x [[buffer(0)]], device const float *order [[buffer(1)]],
    device float *y [[buffer(2)]], constant ExpertParams &p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= ulong(p.rows) * p.width) return;
    uint row = tid / p.width, col = tid % p.width;
    y[tid] = x[ulong(uint(order[ulong(row) * 3]) / p.top_k) * p.stride + col];
}

template <uint FORMAT> inline float expert_weight(device const uchar *w, ulong row, uint col, uint width) {
    if (FORMAT == 0) return ((device const float *)w)[row * width + col];
    if (FORMAT == 1) {
        device const block_q8_0 &b = ((device const block_q8_0 *)w)[row * (width / 32) + col / 32];
        return float(b.d) * float(b.qs[col % 32]);
    }
    device const block_q4_k &b = ((device const block_q4_k *)w)[row * (width / 256) + col / 256];
    uint inner = col % 256, sub = inner / 32;
    uchar2 scale = q4k_scale_min(sub, b.scales);
    uchar packed = b.qs[(inner / 64) * 32 + inner % 32];
    uint quant = (sub & 1) ? packed >> 4 : packed & 15;
    return float(b.d) * float(scale.x) * float(quant) - float(b.dmin) * float(scale.y);
}

// One matrix unit tile of one expert per simdgroup. Tile offsets come from the actual
// routing counts, so skew changes group sizes without padding tokens or host readback.
template <uint FORMAT> kernel void moe_gemm(device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]], device const float *tiles [[buffer(2)]],
    device float *y [[buffer(3)]], constant ExpertParams &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint columns = (p.out + 7) / 8, tile = tg / columns, col0 = (tg % columns) * 8;
    if (tiles[tile * 3] < 0.0f) return;
    uint expert = uint(tiles[tile * 3]), offset = uint(tiles[tile * 3 + 1]), count = uint(tiles[tile * 3 + 2]);
    uint row0 = 0;
    uint qid = lane / 4;
    uint fr = (qid & 4) + (lane / 2) % 4, fc = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc(0.0f);
    for (uint k = 0; k < p.width; k += 8) {
        simdgroup_float8x8 wm, xm;
        for (uint i = 0; i < 2; ++i) {
            wm.thread_elements()[i] = col0 + fr < p.out && k + fc + i < p.width
                ? expert_weight<FORMAT>(w, ulong(expert) * p.out + col0 + fr, k + fc + i, p.width) : 0.0f;
            xm.thread_elements()[i] = row0 + fc + i < count && k + fr < p.width
                ? x[ulong(offset + row0 + fc + i) * p.stride + k + fr] : 0.0f;
        }
        simdgroup_multiply_accumulate(acc, wm, xm, acc);
    }
    for (uint i = 0; i < 2; ++i) {
        if (row0 + fc + i < count && col0 + fr < p.out) {
            y[ulong(offset + row0 + fc + i) * p.out + col0 + fr] = acc.thread_elements()[i];
        }
    }
}

#define EXPERT_GEMM(NAME, FORMAT) \
template [[host_name(NAME)]] kernel void moe_gemm<FORMAT>(device const uchar *, device const float *, \
    device const float *, device float *, constant ExpertParams &, uint, uint);
EXPERT_GEMM("moe_gemm_f32", 0)
EXPERT_GEMM("moe_gemm_q4k", 2)

// Q8 groups use the measured GEMV/matrix split of the dense projection kernels.
// Each matrix tile reuses vector-loaded weights across sixteen token rows and splits
// the reduction across the threadgroup's simdgroups; groups of at most three rows use
// GEMV to avoid mostly empty matrix fragments.
kernel void moe_qmm_q8_0(device const block_q8_0 *w [[buffer(0)]],
    device const float *x [[buffer(1)]], device const float *tiles [[buffer(2)]],
    device float *y [[buffer(3)]], constant ExpertParams &p [[buffer(4)]],
    threadgroup float *partial [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint columns = (p.out + 15) / 16, tile = tg / columns, col0 = (tg % columns) * 16;
    if (tiles[tile * 3] < 0.0f) return;
    uint expert = uint(tiles[tile * 3]), offset = uint(tiles[tile * 3 + 1]), count = uint(tiles[tile * 3 + 2]);
    uint row0 = 0;
    uint blocks = p.width / 32;
    if (count <= 3) {
        for (uint part = sg; part < 4; part += sgs) {
            uint col = col0 + part * 4;
            float acc[4][3];
            for (uint r = 0; r < 4; ++r) for (uint b = 0; b < 3; ++b) acc[r][b] = 0.0f;
            uint ix = lane / 4, il = lane % 4;
            for (uint ib = ix; ib < blocks; ib += 8) {
                float4 lo[4], hi[4]; float scale[4];
                for (uint r = 0; r < 4; ++r) {
                    device const block_q8_0 &blk = w[(ulong(expert) * p.out + min(col + r, p.out - 1)) * blocks + ib];
                    device const packed_char4 *qs = (device const packed_char4 *)(blk.qs + il * 8);
                    scale[r] = float(blk.d); lo[r] = float4(char4(qs[0])); hi[r] = float4(char4(qs[1]));
                }
                for (uint b = 0; b < 3; ++b) {
                    if (b < count) {
                        device const float *xb = x + ulong(offset + b) * p.stride + ib * 32 + il * 8;
                        float4 xl = float4(xb[0], xb[1], xb[2], xb[3]);
                        float4 xh = float4(xb[4], xb[5], xb[6], xb[7]);
                        for (uint r = 0; r < 4; ++r) acc[r][b] += (dot(lo[r], xl) + dot(hi[r], xh)) * scale[r];
                    }
                }
            }
            for (uint r = 0; r < 4; ++r) {
                for (uint b = 0; b < 3; ++b) {
                    float total = simd_sum(acc[r][b]);
                    if (lane == 0 && b < count && col + r < p.out) y[ulong(offset + b) * p.out + col + r] = total;
                }
            }
        }
        return;
    }
    uint qid = lane / 4;
    uint fr = (qid & 4) + (lane / 2) % 4, fc = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc[2][2];
    for (uint t = 0; t < 2; ++t) for (uint b = 0; b < 2; ++b) acc[t][b] = simdgroup_float8x8(0.0f);
    device const block_q8_0 *wr[2];
    for (uint t = 0; t < 2; ++t) wr[t] = w + (ulong(expert) * p.out + min(col0 + t * 8 + fr, p.out - 1)) * blocks;
    for (uint ib = sg; ib < blocks; ib += sgs) {
        float4 xv[2][2];
        for (uint b = 0; b < 2; ++b) {
            for (uint e = 0; e < 2; ++e) {
                uint row = row0 + b * 8 + fc + e;
                device const float *v = x + ulong(offset + min(row, count - 1)) * p.stride + ib * 32 + 4 * fr;
                xv[b][e] = row < count ? float4(v[0], v[1], v[2], v[3]) : float4(0.0f);
            }
        }
        float4 lo[2], hi[2];
        for (uint t = 0; t < 2; ++t) {
            device const block_q8_0 &blk = wr[t][ib];
            device const packed_char4 *qs = (device const packed_char4 *)(blk.qs + 4 * fc);
            lo[t] = float4(char4(qs[0])) * float(blk.d); hi[t] = float4(char4(qs[1])) * float(blk.d);
        }
        for (uint s = 0; s < 4; ++s) {
            simdgroup_float8x8 xm[2];
            for (uint b = 0; b < 2; ++b) { xm[b].thread_elements()[0] = xv[b][0][s]; xm[b].thread_elements()[1] = xv[b][1][s]; }
            for (uint t = 0; t < 2; ++t) {
                simdgroup_float8x8 wm;
                wm.thread_elements()[0] = lo[t][s]; wm.thread_elements()[1] = hi[t][s];
                for (uint b = 0; b < 2; ++b) simdgroup_multiply_accumulate(acc[t][b], wm, xm[b], acc[t][b]);
            }
        }
    }
    for (uint t = 0; t < 2; ++t) {
        for (uint b = 0; b < 2; ++b) {
            threadgroup float *out = partial + ((sg * 2 + t) * 2 + b) * 64 + fr * 8 + fc;
            out[0] = acc[t][b].thread_elements()[0]; out[1] = acc[t][b].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 256; i += threads) {
        float total = 0.0f;
        for (uint g = 0; g < sgs; ++g) total += partial[g * 256 + i];
        uint t = i / 128, b = (i / 64) % 2;
        uint col = col0 + t * 8 + (i % 64) / 8, row = row0 + b * 8 + i % 8;
        if (row < count && col < p.out) y[ulong(offset + row) * p.out + col] = total;
    }
}

kernel void moe_scatter_f32(device const float *x [[buffer(0)]], device const float *order [[buffer(1)]],
    device const float *selected [[buffer(2)]], device float *y [[buffer(3)]],
    constant ExpertParams &p [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= ulong(p.rows / p.top_k) * p.width) return;
    uint token = tid / p.width, col = tid % p.width;
    float sum = 0.0f;
    // Walk selected experts in ascending ID order without an atomic floating-point sum.
    int previous = -1;
    for (uint n = 0; n < p.top_k; ++n) {
        int best = int(p.experts); uint index = 0;
        for (uint k = 0; k < p.top_k; ++k) {
            int expert = int(selected[(ulong(token) * p.top_k + k) * 2]);
            if (expert < 0) { y[tid] = NAN; return; }
            if (expert > previous && expert < best) { best = expert; index = k; }
        }
        uint assignment = token * p.top_k + index;
        uint row = uint(order[ulong(assignment) * 3 + 2]);
        sum += x[ulong(row) * p.stride + col] * selected[ulong(assignment) * 2 + 1];
        previous = best;
    }
    y[tid] = sum;
}
