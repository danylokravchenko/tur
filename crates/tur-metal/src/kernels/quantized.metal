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

// Eight consecutive dequantized weights of `row` from element 32 * chunk + start, where
// start is a multiple of four below 32. FORMAT: 0 F32, 1 Q8_0, 2 Q4_K, 3 MXFP4, whose scale
// bytes follow the blocks at byte offset `scales`.
constant float MXFP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
template <uint FORMAT> inline void packed_weights8(device const uchar *w, ulong row, uint chunk,
    uint start, uint width, thread float4 &lo, thread float4 &hi, ulong scales = 0) {
    if (FORMAT == 3) {
        ulong block = row * (width / 32) + chunk;
        // 2^(s - 127) as float bits, so no scale needs a power function.
        float d = as_type<float>(uint(w[scales + block]) << 23);
        // Eight values of a block are four bytes; start is even, so low nibbles lead.
        uint q = as_type<uint>(uchar4(*((device const packed_uchar4 *)(w + block * 16 + start / 2))));
        lo = float4(MXFP4_VALUES[q & 15], MXFP4_VALUES[(q >> 4) & 15], MXFP4_VALUES[(q >> 8) & 15], MXFP4_VALUES[(q >> 12) & 15]) * d;
        hi = float4(MXFP4_VALUES[(q >> 16) & 15], MXFP4_VALUES[(q >> 20) & 15], MXFP4_VALUES[(q >> 24) & 15], MXFP4_VALUES[q >> 28]) * d;
    } else if (FORMAT == 0) {
        device const packed_float4 *v = (device const packed_float4 *)((device const float *)w
            + row * width + chunk * 32 + start);
        lo = float4(v[0]); hi = float4(v[1]);
    } else if (FORMAT == 1) {
        device const block_q8_0 &b = ((device const block_q8_0 *)w)[row * (width / 32) + chunk];
        device const packed_char4 *qs = (device const packed_char4 *)(b.qs + start);
        float d = float(b.d);
        lo = float4(char4(qs[0])) * d; hi = float4(char4(qs[1])) * d;
    } else {
        device const block_q4_k &b = ((device const block_q4_k *)w)[row * (width / 256) + chunk / 8];
        uint sub = chunk % 8;
        uchar2 sm = q4k_scale_min(sub, b.scales);
        float d = float(b.d) * float(sm.x), m = float(b.dmin) * float(sm.y);
        device const packed_uchar4 *qs = (device const packed_uchar4 *)(b.qs + (sub / 2) * 32 + start);
        uchar4 q0 = uchar4(qs[0]), q1 = uchar4(qs[1]);
        if (sub & 1) { q0 >>= 4; q1 >>= 4; } else { q0 &= 15; q1 &= 15; }
        lo = float4(q0) * d - m; hi = float4(q1) * d - m;
    }
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

// Small-batch Q8_0 and Q4_K matrix multiplication on the matrix units, computed
// transposed: y^T[n, b] = w[n, :] · x^T[:, b] for up to 8 * BT activation rows. A
// threadgroup owns 8 * NT output columns; its simdgroups split the 32-element chunks along
// k and reduce once at the end, so even narrow projections fill the GPU. Dequantization
// and accumulation are F32, as in the GEMVs, so batch size changes no precision.
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

// This lane's activations k = 4 * fr + s of batch rows fc and fc + 1 in 32-element chunk
// `chunk`; rows past the batch read as zero.
template <uint BT>
static inline void qmm_activations(device const float *x, constant QmmParams &p, uint chunk,
    uint fr, uint fc, thread float4 (&xv)[BT][2]) {
    for (uint b = 0; b < BT; ++b) {
        for (uint e = 0; e < 2; ++e) {
            uint row = b * 8 + fc + e;
            xv[b][e] = row < p.rows
                ? *((device const float4 *)(x + ulong(row) * p.x_stride + chunk * 32 + 4 * fr))
                : float4(0.0f);
        }
    }
}

// One 32-element chunk: four k steps of NT × BT 8x8 multiplies. q_lo/q_hi hold this
// lane's weights 4·fc..4·fc+7 of its row in each column tile.
template <uint BT, uint NT>
static inline void qmm_chunk(thread simdgroup_float8x8 (&acc)[NT][BT], thread const float4 (&q_lo)[NT],
    thread const float4 (&q_hi)[NT], thread const float4 (&xv)[BT][2]) {
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

// Reduce the k splits through partial[sg][t][b][r][c] and store the threadgroup's tile.
template <uint BT, uint NT>
static inline void qmm_store(thread simdgroup_float8x8 (&acc)[NT][BT], threadgroup float *partial,
    device float *y, constant QmmParams &p, uint col0, uint fr, uint fc, uint sg, uint sgs,
    uint tid, uint threads) {
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

template <uint FORMAT, uint BT, uint NT>
kernel void qmm_f32(
    device const uchar *w [[buffer(0)]],
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
    // This lane's weight rows.
    ulong wr[NT];
    for (uint t = 0; t < NT; ++t) {
        wr[t] = min(col0 + t * 8 + fr, p.n - 1);
    }
    for (uint ib = sg; ib < blocks; ib += sgs) {
        float4 xv[BT][2];
        qmm_activations<BT>(x, p, ib, fr, fc, xv);
        float4 q_lo[NT], q_hi[NT];
        for (uint t = 0; t < NT; ++t) {
            packed_weights8<FORMAT>(w, wr[t], ib, 4 * fc, p.k, q_lo[t], q_hi[t]);
        }
        qmm_chunk<BT, NT>(acc, q_lo, q_hi, xv);
    }
    qmm_store<BT, NT>(acc, partial, y, p, col0, fr, fc, sg, sgs, tid, threads);
}

// Scale and minimum of Q4_K sub-block j from the 12 packed scale bytes held in `s`.
// Callers pass a j that is constant after unrolling, so every byte select is a fixed shift.
static inline uint q4k_byte(thread const uint (&s)[3], uint i) {
    return (s[i / 4] >> (8 * (i % 4))) & 0xFFu;
}

static inline float2 q4k_scale_min_reg(thread const uint (&s)[3], uint j) {
    if (j < 4) {
        return float2(q4k_byte(s, j) & 63u, q4k_byte(s, j + 4) & 63u);
    }
    return float2((q4k_byte(s, j + 4) & 0xFu) | ((q4k_byte(s, j - 4) >> 6) << 4),
        (q4k_byte(s, j + 4) >> 4) | ((q4k_byte(s, j) >> 6) << 4));
}

// Q4_K variant of qmm_f32. A simdgroup step covers one 256-element super-block: the
// header and scale bytes load once into registers, and each 8-byte quant load serves two
// sub-blocks (low nibbles, then high), so loads per weight fall about sevenfold against
// dequantizing every 32-element chunk independently.
template <uint BT, uint NT>
kernel void qmm_q4k_f32(
    device const block_q4_k *w [[buffer(0)]],
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
    uint blocks = p.k / 256;
    simdgroup_float8x8 acc[NT][BT];
    for (uint t = 0; t < NT; ++t) {
        for (uint b = 0; b < BT; ++b) {
            acc[t][b] = simdgroup_float8x8(0.0f);
        }
    }
    ulong wr[NT];
    for (uint t = 0; t < NT; ++t) {
        wr[t] = ulong(min(col0 + t * 8 + fr, p.n - 1)) * blocks;
    }
    for (uint ib = sg; ib < blocks; ib += sgs) {
        float d[NT], dmin[NT];
        uint scales[NT][3];
        device const uchar *qs[NT];
        for (uint t = 0; t < NT; ++t) {
            device const block_q4_k &blk = w[wr[t] + ib];
            d[t] = float(blk.d);
            dmin[t] = float(blk.dmin);
            device const uint *packed = (device const uint *)blk.scales;
            for (uint i = 0; i < 3; ++i) {
                scales[t][i] = packed[i];
            }
            qs[t] = blk.qs + 4 * fc;
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            uint2 q[NT];
            for (uint t = 0; t < NT; ++t) {
                q[t] = *((device const uint2 *)(qs[t] + 32 * j));
            }
            #pragma clang loop unroll(full)
            for (uint h = 0; h < 2; ++h) {
                float4 xv[BT][2];
                qmm_activations<BT>(x, p, ib * 8 + 2 * j + h, fr, fc, xv);
                float4 q_lo[NT], q_hi[NT];
                for (uint t = 0; t < NT; ++t) {
                    float2 sm = q4k_scale_min_reg(scales[t], 2 * j + h);
                    float scale = d[t] * sm.x, minimum = dmin[t] * sm.y;
                    uint2 nib = (q[t] >> (4 * h)) & 0x0F0F0F0Fu;
                    q_lo[t] = float4(as_type<uchar4>(nib.x)) * scale - minimum;
                    q_hi[t] = float4(as_type<uchar4>(nib.y)) * scale - minimum;
                }
                qmm_chunk<BT, NT>(acc, q_lo, q_hi, xv);
            }
        }
    }
    qmm_store<BT, NT>(acc, partial, y, p, col0, fr, fc, sg, sgs, tid, threads);
}

#define QMM(FORMAT, NAME, BT, NT) \
    template [[host_name("qmm_" NAME "_t" #BT "_f32")]] kernel void qmm_f32<FORMAT, BT, NT>( \
        device const uchar *, device const float *, device float *, constant QmmParams &, \
        threadgroup float *, uint, uint, uint, uint, uint, uint);
#define QMM_Q4K(BT, NT) \
    template [[host_name("qmm_q4k_t" #BT "_f32")]] kernel void qmm_q4k_f32<BT, NT>( \
        device const block_q4_k *, device const float *, device float *, constant QmmParams &, \
        threadgroup float *, uint, uint, uint, uint, uint, uint);
// Two column tiles per threadgroup: four measured no faster on M1 (P6 kernel artifact).
QMM(1, "q8_0", 1, 2)
QMM(1, "q8_0", 2, 2)
QMM_Q4K(1, 4)
QMM_Q4K(2, 4)

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

