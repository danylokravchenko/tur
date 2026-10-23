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
// Quants and activations load as vectors; each activation vector is reused by all NR
// rows, so activation traffic per weight byte falls as NR grows.
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
        float4 q_lo[NR], q_hi[NR];
        float scale[NR];
        for (uint r = 0; r < NR; ++r) {
            uint row = min(row0 + r, p.n - 1);
            device const block_q8_0 &blk = w[ulong(row) * blocks + ib];
            device const packed_char4 *qs = (device const packed_char4 *)(blk.qs + il * 8);
            scale[r] = float(blk.d);
            q_lo[r] = float4(char4(qs[0]));
            q_hi[r] = float4(char4(qs[1]));
        }
        for (uint b = 0; b < B; ++b) {
            device const float4 *xb =
                (device const float4 *)(x + ulong(b) * p.x_stride + ib * 32 + il * 8);
            float4 x_lo = xb[0];
            float4 x_hi = xb[1];
            for (uint r = 0; r < NR; ++r) {
                acc[r][b] += (dot(q_lo[r], x_lo) + dot(q_hi[r], x_hi)) * scale[r];
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

// Small-batch Q8_0 matrix multiplication on the matrix units, computed transposed:
// y^T[n, b] = w[n, :] · x^T[:, b] for up to 8 * BT activation rows. A threadgroup owns
// 8 * NT output columns; its simdgroups split the Q8_0 blocks along k and reduce once at
// the end, so even narrow projections fill the GPU.
//
// Fragments load straight from device memory with no staging or barriers. Lane l of an
// 8x8 fragment holds (r, c) and (r, c + 1), r = (l/4 & 4) + (l/2) % 4, c = (l/4 & 2) * 2 +
// (l % 2) * 2. Within a block, step s of four maps fragment index c (a weight column or
// an activation row) to input k = 4c + s, so each lane's weights for the whole block are
// one 8-byte load and its activations one float4 per batch row, every value used.
// Accumulation is F32 throughout.
struct QmmParams {
    uint n;
    uint k;
    uint x_stride;
    uint y_stride;
    uint rows;
};

template <uint BT, uint NT>
kernel void qmm_q8_0_f32(
    device const block_q8_0 *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant QmmParams &p [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint qid = lane / 4;
    uint fr = (qid & 4) + (lane / 2) % 4;
    uint fc = (qid & 2) * 2 + (lane % 2) * 2;
    uint col0 = tg * 8 * NT;
    uint blocks = p.k / 32;
    simdgroup_float8x8 acc[NT][BT];
    for (uint t = 0; t < NT; ++t) {
        for (uint b = 0; b < BT; ++b) {
            acc[t][b] = simdgroup_float8x8(0.0f);
        }
    }
    // This lane's weight rows and activation rows.
    device const block_q8_0 *wr[NT];
    for (uint t = 0; t < NT; ++t) {
        wr[t] = w + ulong(min(col0 + t * 8 + fr, p.n - 1)) * blocks;
    }
    // This lane's activations k = 4 * fr + s of batch rows fc and fc + 1.
    uint xk = 4 * fr;
    for (uint ib = sg; ib < blocks; ib += sgs) {
        float4 xv[BT][2];
        for (uint b = 0; b < BT; ++b) {
            for (uint e = 0; e < 2; ++e) {
                uint row = b * 8 + fc + e;
                xv[b][e] = row < p.rows
                    ? *((device const float4 *)(x + ulong(row) * p.x_stride + ib * 32 + xk))
                    : float4(0.0f);
            }
        }
        float4 q_lo[NT], q_hi[NT];
        for (uint t = 0; t < NT; ++t) {
            device const block_q8_0 &blk = wr[t][ib];
            device const packed_char4 *qs = (device const packed_char4 *)(blk.qs + 4 * fc);
            float d = float(blk.d);
            q_lo[t] = float4(char4(qs[0])) * d;
            q_hi[t] = float4(char4(qs[1])) * d;
        }
        for (uint s = 0; s < 4; ++s) {
            simdgroup_float8x8 xm[BT];
            for (uint b = 0; b < BT; ++b) {
                xm[b].thread_elements()[0] = xv[b][0][s];
                xm[b].thread_elements()[1] = xv[b][1][s];
            }
            for (uint t = 0; t < NT; ++t) {
                simdgroup_float8x8 wm;
                wm.thread_elements()[0] = q_lo[t][s];
                wm.thread_elements()[1] = q_hi[t][s];
                for (uint b = 0; b < BT; ++b) {
                    simdgroup_multiply_accumulate(acc[t][b], wm, xm[b], acc[t][b]);
                }
            }
        }
    }
    // Reduce the k splits: partial[sg][t][b][r][c].
    for (uint t = 0; t < NT; ++t) {
        for (uint b = 0; b < BT; ++b) {
            threadgroup float *out = partial + ((sg * NT + t) * BT + b) * 64 + fr * 8 + fc;
            out[0] = acc[t][b].thread_elements()[0];
            out[1] = acc[t][b].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < NT * BT * 64; i += threads) {
        float total = 0.0f;
        for (uint g = 0; g < sgs; ++g) {
            total += partial[g * NT * BT * 64 + i];
        }
        uint t = i / (BT * 64);
        uint b = (i / 64) % BT;
        uint col = col0 + t * 8 + (i % 64) / 8;
        uint row = b * 8 + i % 8;
        if (row < p.rows && col < p.n) {
            y[ulong(row) * p.y_stride + col] = total;
        }
    }
}

#define QMM(BT, NT) \
    template [[host_name("qmm_q8_0_t" #BT "_f32")]] kernel void qmm_q8_0_f32<BT, NT>( \
        device const block_q8_0 *, device const float *, device float *, constant QmmParams &, \
        threadgroup float *, uint, uint, uint, uint, uint, uint);
// Two column tiles per threadgroup: four measured no faster on M1 (P6 kernel artifact).
QMM(1, 2)
QMM(2, 2)

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
QMV(5, 4, 2)
QMV(6, 4, 2)
QMV(7, 4, 2)
QMV(8, 4, 2)

