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

// Pipelines over GGUF's other block formats specialize this to one `BlockFormat`, so the
// compiler removes every other decoder from them.
constant ushort BLOCK_FORMAT_VALUE [[function_constant(0)]];
constant uint BLOCK_FORMAT = is_function_constant_defined(BLOCK_FORMAT_VALUE) ? uint(BLOCK_FORMAT_VALUE) : 0;
enum : uint {
    B_F32, B_F16, B_BF16, B_Q4_0, B_Q4_1, B_Q5_0, B_Q5_1, B_Q8_0, B_Q2K, B_Q3K, B_Q4K, B_Q5K, B_Q6K,
    B_IQ1S, B_IQ1M, B_IQ2XXS, B_IQ2XS, B_IQ2S, B_IQ3XXS, B_IQ3S, B_IQ4NL, B_IQ4XS, B_MXFP4,
};

static inline uint block_u16(device const uchar *b, uint at) {
    return uint(b[at]) | (uint(b[at + 1]) << 8);
}
static inline uint block_u32(device const uchar *b, uint at) {
    return block_u16(b, at) | (block_u16(b, at + 2) << 16);
}
static inline float block_half(device const uchar *b, uint at) {
    return float(as_type<half>(ushort(block_u16(b, at))));
}
// Four consecutive bytes as one vector load. Blocks sit at any byte offset, and a whole-word
// load extracts all four quants at once where ggml's kernels and ours differ most in speed.
static inline uint4 block_bytes4(device const uchar *b) {
    return uint4(uchar4(*((device const packed_uchar4 *)b)));
}
// `value` with each lane negated where its bit of `signs`, from `first`, is set.
static inline float4 block_signed4(uint signs, uint first, float4 value) {
    uint4 bits = (uint4(signs) >> (uint4(0, 1, 2, 3) + first)) & 1u;
    return value * (1.0f - 2.0f * float4(bits));
}

// Signed scale `i` of Q3_K's 16, whose 12 bytes hold 4 low bits then 2 high bits each.
static inline float q3k_scale(device const uchar *s, uint i) {
    uint word = i / 4, byte = i % 4;
    uint nibble = word < 2 ? uint(s[word * 4 + byte]) & 15u : uint(s[(word - 2) * 4 + byte]) >> 4;
    return float(nibble | (((uint(s[8 + byte]) >> (2 * word)) & 3u) << 4)) - 32.0f;
}

// Eight weights from element 32 * chunk + start (start a multiple of eight) of `row`, in the
// format this pipeline was specialized to. Decoders follow ggml's dequantize_row_*.
static inline void block_weights8(device const uchar *w, ulong row, uint chunk, uint start, uint width,
    thread float4 &lo, thread float4 &hi) {
    uint f = BLOCK_FORMAT;
    bool wide = f == B_Q2K || f == B_Q3K || f == B_Q4K || f == B_Q5K || f == B_Q6K || f == B_IQ1S
        || f == B_IQ1M || f == B_IQ2XXS || f == B_IQ2XS || f == B_IQ2S || f == B_IQ3XXS || f == B_IQ3S
        || f == B_IQ4XS;
    uint size = 0;
    switch (f) {
        case B_F32: size = 128; break;
        case B_F16: case B_BF16: size = 64; break;
        case B_Q4_0: case B_IQ4NL: size = 18; break;
        case B_MXFP4: size = 17; break;
        case B_Q4_1: size = 20; break;
        case B_Q5_0: size = 22; break;
        case B_Q5_1: size = 24; break;
        case B_Q8_0: size = 34; break;
        case B_IQ1S: size = 50; break;
        case B_IQ1M: size = 56; break;
        case B_IQ2XXS: size = 66; break;
        case B_IQ2XS: size = 74; break;
        case B_IQ2S: size = 82; break;
        case B_Q2K: size = 84; break;
        case B_IQ3XXS: size = 98; break;
        case B_Q3K: case B_IQ3S: size = 110; break;
        case B_IQ4XS: size = 136; break;
        case B_Q4K: size = 144; break;
        case B_Q5K: size = 176; break;
        case B_Q6K: size = 210; break;
    }
    uint per_block = wide ? 8 : 1;
    device const uchar *b = w + (row * (width / (32 * per_block)) + chunk / per_block) * size;
    uint sub = chunk % per_block, l = start / 8;
    if (f == B_F32) {
        device const packed_float4 *v = (device const packed_float4 *)(b + 4 * start);
        lo = float4(v[0]); hi = float4(v[1]);
    } else if (f == B_F16) {
        for (uint j = 0; j < 4; ++j) {
            lo[j] = block_half(b, 2 * (start + j));
            hi[j] = block_half(b, 2 * (start + 4 + j));
        }
    } else if (f == B_BF16) {
        for (uint j = 0; j < 4; ++j) {
            lo[j] = as_type<float>(block_u16(b, 2 * (start + j)) << 16);
            hi[j] = as_type<float>(block_u16(b, 2 * (start + 4 + j)) << 16);
        }
    } else if (f == B_Q4_0 || f == B_Q4_1 || f == B_Q5_0 || f == B_Q5_1) {
        bool offset = f == B_Q4_1 || f == B_Q5_1, five = f == B_Q5_0 || f == B_Q5_1;
        float d = block_half(b, 0), m = offset ? block_half(b, 2) : 0.0f;
        uint at = offset ? 4 : 2;
        uint high = five ? block_u32(b, at) : 0;
        at += five ? 4 : 0;
        float center = offset ? 0.0f : (five ? 16.0f : 8.0f);
        uint nibble = start < 16 ? 0 : 4, i = start % 16;
        uint4 q0 = (block_bytes4(b + at + i) >> nibble) & 15u;
        uint4 q1 = (block_bytes4(b + at + i + 4) >> nibble) & 15u;
        q0 |= ((uint4(high) >> (uint4(0, 1, 2, 3) + start)) & 1u) << 4;
        q1 |= ((uint4(high) >> (uint4(4, 5, 6, 7) + start)) & 1u) << 4;
        lo = (float4(q0) - center) * d + m;
        hi = (float4(q1) - center) * d + m;
    } else if (f == B_Q8_0) {
        device const packed_char4 *qs = (device const packed_char4 *)(b + 2 + start);
        float d = block_half(b, 0);
        lo = float4(char4(qs[0])) * d; hi = float4(char4(qs[1])) * d;
    } else if (f == B_Q2K) {
        uint e0 = sub * 32 + start, group = e0 / 128, x0 = e0 % 128, scale = b[e0 / 16];
        float dl = block_half(b, 80) * float(scale & 15u), ml = block_half(b, 82) * float(scale >> 4);
        uint shift = 2 * (x0 / 32);
        device const uchar *qs = b + 16 + group * 32 + (x0 % 32) / 16 * 16 + x0 % 16;
        lo = dl * float4((block_bytes4(qs) >> shift) & 3u) - ml;
        hi = dl * float4((block_bytes4(qs + 4) >> shift) & 3u) - ml;
    } else if (f == B_Q3K) {
        uint e0 = sub * 32 + start, group = e0 / 128, x0 = e0 % 128, step = x0 / 32;
        uint at = (x0 % 32) / 16 * 16 + x0 % 16, bit = group * 4 + step, shift = 2 * step;
        float dl = block_half(b, 108) * q3k_scale(b + 96, e0 / 16);
        device const uchar *qs = b + 32 + group * 32 + at, *hm = b + at;
        // A clear high bit subtracts four, so the three bits read as an offset value.
        uint4 q0 = ((block_bytes4(qs) >> shift) & 3u) | (((block_bytes4(hm) >> bit) & 1u) << 2);
        uint4 q1 = ((block_bytes4(qs + 4) >> shift) & 3u) | (((block_bytes4(hm + 4) >> bit) & 1u) << 2);
        lo = dl * (float4(q0) - 4.0f);
        hi = dl * (float4(q1) - 4.0f);
    } else if (f == B_Q4K || f == B_Q5K) {
        uchar2 sm = q4k_scale_min(sub, b + 4);
        float dl = block_half(b, 0) * float(sm.x), ml = block_half(b, 2) * float(sm.y);
        uint e0 = sub * 32 + start, nibble = 4 * (sub % 2);
        device const uchar *qs = b + (f == B_Q5K ? 48 : 16) + e0 / 64 * 32 + e0 % 32, *high = b + 16 + e0 % 32;
        uint4 q0 = (block_bytes4(qs) >> nibble) & 15u, q1 = (block_bytes4(qs + 4) >> nibble) & 15u;
        if (f == B_Q5K) {
            q0 |= ((block_bytes4(high) >> sub) & 1u) << 4;
            q1 |= ((block_bytes4(high + 4) >> sub) & 1u) << 4;
        }
        lo = dl * float4(q0) - ml;
        hi = dl * float4(q1) - ml;
    } else if (f == B_Q6K) {
        uint e0 = sub * 32 + start, group = e0 / 128, x0 = e0 % 128, step = x0 / 32, l0 = x0 % 32;
        device const uchar *low = b + group * 64 + l0 + (step % 2) * 32, *high = b + 128 + group * 32 + l0;
        float dl = block_half(b, 208) * float(as_type<char>(b[192 + group * 8 + l0 / 16 + 2 * step]));
        uint nibble = step < 2 ? 0 : 4, top = 2 * step;
        uint4 q0 = ((block_bytes4(low) >> nibble) & 15u) | (((block_bytes4(high) >> top) & 3u) << 4);
        uint4 q1 = ((block_bytes4(low + 4) >> nibble) & 15u) | (((block_bytes4(high + 4) >> top) & 3u) << 4);
        lo = dl * (float4(q0) - 32.0f);
        hi = dl * (float4(q1) - 32.0f);
    } else if (f == B_IQ1S || f == B_IQ1M) {
        uint index;
        float dl, delta;
        if (f == B_IQ1S) {
            uint high = block_u16(b, 34 + 2 * sub);
            dl = block_half(b, 0) * float(2 * ((high >> 12) & 7u) + 1);
            delta = high & 0x8000u ? -0.125f : 0.125f;
            index = uint(b[2 + sub * 4 + l]) | (((high >> (3 * l)) & 7u) << 8);
        } else {
            uint s0 = block_u16(b, 48), s1 = block_u16(b, 50), s2 = block_u16(b, 52), s3 = block_u16(b, 54);
            uint bits = (s0 >> 12) | ((s1 >> 8) & 0xF0u) | ((s2 >> 4) & 0xF00u) | (s3 & 0xF000u);
            float d = float(as_type<half>(ushort(bits)));
            uint nibble = uint(b[32 + sub * 2 + l / 2]) >> (4 * (l % 2));
            index = uint(b[sub * 4 + l]) | ((nibble & 7u) << 8);
            delta = nibble & 8u ? -0.125f : 0.125f;
            uint scale = (block_u16(b, 48 + 2 * (sub / 2)) >> (6 * (sub % 2) + 3 * (l / 2))) & 7u;
            dl = d * float(2 * scale + 1);
        }
        lo = dl * (float4(as_type<char4>(IQ1S_GRID[2 * index])) + delta);
        hi = dl * (float4(as_type<char4>(IQ1S_GRID[2 * index + 1])) + delta);
    } else if (f == B_IQ2XXS || f == B_IQ2XS || f == B_IQ2S) {
        float d = block_half(b, 0), db;
        uint index, signs;
        constant uint *grid;
        if (f == B_IQ2XXS) {
            uint high = block_u32(b, 6 + sub * 8);
            db = d * (0.5f + float(high >> 28)) * 0.25f;
            index = b[2 + sub * 8 + l];
            signs = KSIGNS[(high >> (7 * l)) & 127u];
            grid = IQ2XXS_GRID;
        } else if (f == B_IQ2XS) {
            uint word = block_u16(b, 2 + 2 * (4 * sub + l));
            db = d * (0.5f + float((uint(b[66 + sub]) >> (4 * (l / 2))) & 15u)) * 0.25f;
            index = word & 511u;
            signs = KSIGNS[word >> 9];
            grid = IQ2XS_GRID;
        } else {
            db = d * (0.5f + float((uint(b[74 + sub]) >> (4 * (l / 2))) & 15u)) * 0.25f;
            index = uint(b[2 + sub * 4 + l]) | ((uint(b[66 + sub]) << (8 - 2 * l)) & 0x300u);
            signs = b[34 + sub * 4 + l];
            grid = IQ2S_GRID;
        }
        lo = block_signed4(signs, 0, db * float4(as_type<uchar4>(grid[2 * index])));
        hi = block_signed4(signs, 4, db * float4(as_type<uchar4>(grid[2 * index + 1])));
    } else if (f == B_IQ3XXS || f == B_IQ3S) {
        float d = block_half(b, 0), db;
        uint first = b[2 + sub * 8 + 2 * l], second = b[3 + sub * 8 + 2 * l], signs;
        constant uint *grid;
        if (f == B_IQ3XXS) {
            uint aux = block_u32(b, 66 + 4 * sub);
            db = d * (0.5f + float(aux >> 28)) * 0.5f;
            signs = KSIGNS[(aux >> (7 * l)) & 127u];
            grid = IQ3XXS_GRID;
        } else {
            uint high = b[66 + sub];
            first |= (high << (8 - 2 * l)) & 256u;
            second |= (high << (7 - 2 * l)) & 256u;
            db = d * float(1 + 2 * ((uint(b[106 + sub / 2]) >> (4 * (sub % 2))) & 15u));
            signs = b[74 + sub * 4 + l];
            grid = IQ3S_GRID;
        }
        lo = block_signed4(signs, 0, db * float4(as_type<uchar4>(grid[first])));
        hi = block_signed4(signs, 4, db * float4(as_type<uchar4>(grid[second])));
    } else {
        // IQ4_NL, IQ4_XS and MXFP4: nibbles select a table value.
        float d;
        uint at;
        if (f == B_IQ4NL) {
            d = block_half(b, 0);
            at = 2;
        } else if (f == B_MXFP4) {
            // 2^(e - 128) as float bits: E8M0 halved, since the table holds doubled values.
            d = b[0] == 0 ? 0.0f : as_type<float>((uint(b[0]) - 1) << 23);
            at = 1;
        } else {
            uint ls = ((uint(b[4 + sub / 2]) >> (4 * (sub % 2))) & 15u) | (((block_u16(b, 2) >> (2 * sub)) & 3u) << 4);
            d = block_half(b, 0) * (float(ls) - 32.0f);
            at = 8 + sub * 16;
        }
        uint nibble = start < 16 ? 0 : 4, i = start % 16;
        uint4 q0 = (block_bytes4(b + at + i) >> nibble) & 15u, q1 = (block_bytes4(b + at + i + 4) >> nibble) & 15u;
        constant char *table = f == B_MXFP4 ? KVALUES_FP4 : KVALUES_IQ4NL;
        lo = d * float4(table[q0.x], table[q0.y], table[q0.z], table[q0.w]);
        hi = d * float4(table[q1.x], table[q1.y], table[q1.z], table[q1.w]);
    }
}

// Eight consecutive dequantized weights of `row` from element 32 * chunk + start, where
// start is a multiple of four below 32. FORMAT: 0 F32, 1 Q8_0, 2 Q4_K, 3 MXFP4, whose scale
// bytes follow the blocks at byte offset `scales`, 4 the pipeline's specialized block format.
constant float MXFP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
template <uint FORMAT> inline void packed_weights8(device const uchar *w, ulong row, uint chunk,
    uint start, uint width, thread float4 &lo, thread float4 &hi, ulong scales = 0) {
    if (FORMAT == 4) {
        block_weights8(w, row, chunk, start, width, lo, hi);
    } else if (FORMAT == 3) {
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
    uint fr, uint fc, thread float4 (&xv)[BT][2], uint base = 0) {
    for (uint b = 0; b < BT; ++b) {
        for (uint e = 0; e < 2; ++e) {
            uint row = base + b * 8 + fc + e;
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
    uint tid, uint threads, uint base = 0) {
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
        uint row = base + b * 8 + i % 8;
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

// qmm_f32 over weights in the block format this pipeline is specialized to, up to 16 rows.
template <uint BT, uint NT>
kernel void qmm_blocks_f32(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant QmmParams &p [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint qid = lane / 4;
    uint fr = (qid & 4) + (lane / 2) % 4;
    uint fc = (qid & 2) * 2 + (lane % 2) * 2;
    uint col0 = tg.x * 8 * NT, base = tg.y * 8 * BT;
    uint blocks = p.k / 32;
    simdgroup_float8x8 acc[NT][BT];
    for (uint t = 0; t < NT; ++t) {
        for (uint b = 0; b < BT; ++b) {
            acc[t][b] = simdgroup_float8x8(0.0f);
        }
    }
    ulong wr[NT];
    for (uint t = 0; t < NT; ++t) {
        wr[t] = min(col0 + t * 8 + fr, p.n - 1);
    }
    for (uint ib = sg; ib < blocks; ib += sgs) {
        float4 xv[BT][2];
        qmm_activations<BT>(x, p, ib, fr, fc, xv, base);
        float4 q_lo[NT], q_hi[NT];
        for (uint t = 0; t < NT; ++t) {
            packed_weights8<4>(w, wr[t], ib, 4 * fc, p.k, q_lo[t], q_hi[t]);
        }
        qmm_chunk<BT, NT>(acc, q_lo, q_hi, xv);
    }
    qmm_store<BT, NT>(acc, partial, y, p, col0, fr, fc, sg, sgs, tid, threads.x, base);
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
#define QMM_BLOCKS(BT, NT) \
    template [[host_name("qmm_blocks_t" #BT "_f32")]] kernel void qmm_blocks_f32<BT, NT>( \
        device const uchar *, device const float *, device float *, constant QmmParams &, \
        threadgroup float *, uint3, uint, uint3, uint, uint, uint);
QMM_BLOCKS(1, 2)
QMM_BLOCKS(2, 2)

// One to three activation rows over weights in a specialized block format: four lanes
// share each decoded chunk and a simdgroup keeps eight chunks in flight, as qmv_q8_0_f32.
kernel void qmv_blocks_f32(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant QmmParams &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    constexpr uint NR = 4;
    uint row0 = (tg * sgs + sg) * NR;
    if (row0 >= p.n) {
        return;
    }
    uint start = (lane % 4) * 8;
    float acc[NR][3];
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < 3; ++b) {
            acc[r][b] = 0.0f;
        }
    }
    for (uint chunk = lane / 4; chunk < p.k / 32; chunk += 8) {
        float4 lo[NR], hi[NR];
        for (uint r = 0; r < NR; ++r) {
            packed_weights8<4>(w, min(row0 + r, p.n - 1), chunk, start, p.k, lo[r], hi[r]);
        }
        for (uint b = 0; b < p.rows; ++b) {
            device const float4 *xb =
                (device const float4 *)(x + ulong(b) * p.x_stride + chunk * 32 + start);
            float4 x_lo = xb[0], x_hi = xb[1];
            for (uint r = 0; r < NR; ++r) {
                acc[r][b] += dot(lo[r], x_lo) + dot(hi[r], x_hi);
            }
        }
    }
    for (uint r = 0; r < NR; ++r) {
        for (uint b = 0; b < p.rows; ++b) {
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
QMV(5, 4, 2)
QMV(6, 4, 2)
QMV(7, 4, 2)
QMV(8, 4, 2)

