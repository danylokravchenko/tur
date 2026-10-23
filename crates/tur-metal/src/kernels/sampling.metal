// ---------------------------------------------------------------------------------------
// Penalties and sampling (host reference: `crate::sampling`)
// ---------------------------------------------------------------------------------------

struct ArgmaxParams {
    uint rows;
    uint cols;
    uint stride;
};

// Keep the larger value, or the lower index between equal values.
static inline void better(thread float &v, thread uint &i, float ov, uint oi) {
    if (oi != UINT_MAX && (i == UINT_MAX || ov > v || (ov == v && oi < i))) {
        v = ov;
        i = oi;
    }
}

// Greedy decoding without host logits: per row, the first maximum and whether every value
// is finite, as out[row] = (index, 1) or (index, 0). One threadgroup per row; indices are
// exact in F32 because the host bounds the vocabulary below 2^24.
kernel void argmax_f32(
    device const float *x [[buffer(0)]],
    device float2 *out [[buffer(1)]],
    constant ArgmaxParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float values[SIMD];
    threadgroup uint indices[SIMD];
    threadgroup uint flags[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.stride;
    float v = 0.0f;
    uint i = UINT_MAX;
    uint nonfinite = 0;
    for (uint c = tid; c < p.cols; c += threads) {
        float s = xr[c];
        // -inf marks a masked token; NaN and +inf are invalid.
        nonfinite |= (isnan(s) || s == INFINITY) ? 1u : 0u;
        better(v, i, s, c);
    }
    for (uint offset = SIMD / 2; offset > 0; offset /= 2) {
        better(v, i, simd_shuffle_xor(v, offset), simd_shuffle_xor(i, offset));
    }
    nonfinite = simd_max(nonfinite);
    if (lane == 0) {
        values[sg] = v;
        indices[sg] = i;
        flags[sg] = nonfinite;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        v = lane < sgs ? values[lane] : 0.0f;
        i = lane < sgs ? indices[lane] : UINT_MAX;
        nonfinite = lane < sgs ? flags[lane] : 0u;
        for (uint offset = SIMD / 2; offset > 0; offset /= 2) {
            better(v, i, simd_shuffle_xor(v, offset), simd_shuffle_xor(i, offset));
        }
        nonfinite = simd_max(nonfinite);
        if (lane == 0) {
            // A row without any unmasked token is invalid too.
            out[row] = float2(float(i), nonfinite == 0 && v > -INFINITY ? 1.0f : 0.0f);
        }
    }
}

struct MaskArgs {
    uint rows;
    uint cols;
    uint stride;
    uint words;
};

// Grammar masks, one thread per (mask, token): token t of logit row rows[r] becomes -inf
// unless bit t % 32 of word t / 32 of mask r is set.
kernel void mask_f32(
    device float *x [[buffer(0)]],
    device const uint *rows [[buffer(1)]],
    device const uint *bits [[buffer(2)]],
    constant MaskArgs &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.cols;
    uint c = gid % p.cols;
    if (r >= p.rows) {
        return;
    }
    if (((bits[r * p.words + c / 32] >> (c % 32)) & 1u) == 0) {
        x[ulong(rows[r]) * p.stride + c] = -INFINITY;
    }
}

struct PenaltyArgs {
    uint entries;
    uint stride;
};

// In place, one thread per distinct context token of a row: entries are (row, token,
// sampled count) and params (repetition, presence, frequency) per row.
kernel void penalize_f32(
    device float *x [[buffer(0)]],
    device const uint *entries [[buffer(1)]],
    device const float *params [[buffer(2)]],
    constant PenaltyArgs &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.entries) {
        return;
    }
    uint row = entries[3 * gid];
    float count = float(entries[3 * gid + 2]);
    device const float *rp = params + 3 * row;
    device float *v = x + ulong(row) * p.stride + entries[3 * gid + 1];
    float l = *v;
    l = l > 0.0f ? l / rp[0] : l * rp[0];
    *v = l - fma(rp[2], count, count > 0.0f ? rp[1] : 0.0f);
}

static inline float threadgroup_max(float v, threadgroup float *scratch, uint sg, uint lane, uint sgs) {
    v = simd_max(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = simd_max(lane < sgs ? scratch[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

static inline uint threadgroup_min_u(uint v, threadgroup uint *scratch, uint sg, uint lane, uint sgs) {
    v = simd_min(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m = simd_min(lane < sgs ? scratch[lane] : UINT_MAX);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

static inline uint threadgroup_max_u(uint v, threadgroup uint *scratch, uint sg, uint lane, uint sgs) {
    v = simd_max(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m = simd_max(lane < sgs ? scratch[lane] : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

// Exclusive prefix sum over threads in index order; `total` receives the full sum.
// Threadgroups are whole simdgroups.
static inline float threadgroup_scan(float v, threadgroup float *scratch, uint sg, uint lane, uint sgs, thread float &total) {
    float inner = simd_prefix_exclusive_sum(v);
    if (lane == SIMD - 1) {
        scratch[sg] = inner + v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = lane < sgs ? scratch[lane] : 0.0f;
    float before = simd_prefix_exclusive_sum(s);
    total = simd_sum(s);
    float base = simd_shuffle(before, ushort(sg));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return base + inner;
}

static inline ulong mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
    return z ^ (z >> 31);
}

static inline float uniform24(ulong seed, ulong draw, uint round) {
    ulong z = seed + draw * 0x9e3779b97f4a7c15ul + ulong(round) * 0xd1b54a32d192ed03ul;
    return float(mix64(z) >> 40) * (1.0f / 16777216.0f);
}

struct SampleArgs {
    uint rows;
    uint cols;
    uint stride;
    uint rounds;
};

// One threadgroup per row. Thread t owns tokens t, t + T, t + 2T, ... (coalesced passes);
// the CDF is inverted in thread-major order (thread 0's tokens, then thread 1's), a fixed
// permutation of the vocabulary, so draws follow the same distribution as the host
// reference. params: 7 words per row (temperature bits, top-k, top-p bits, seed lo/hi,
// draw lo/hi). out[row] = (token, status): status 1 sampled, 0 NaN/+inf or no positive
// weight, 2 not accepted within `rounds` rejection rounds.
kernel void sample_f32(
    device const float *x [[buffer(0)]],
    device const uint *params [[buffer(1)]],
    device float2 *out [[buffer(2)]],
    constant SampleArgs &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float fscratch[SIMD];
    threadgroup uint uscratch[SIMD];
    threadgroup uint chosen;
    threadgroup float chosen_w;
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.stride;
    device const uint *rp = params + 7 * row;
    float inv_t = 1.0f / as_type<float>(rp[0]);
    uint top_k = rp[1];
    float top_p = as_type<float>(rp[2]);
    ulong seed = ulong(rp[3]) | (ulong(rp[4]) << 32);
    ulong draw = ulong(rp[5]) | (ulong(rp[6]) << 32);
    bool truncating = (top_k > 0 && top_k < p.cols) || top_p < 1.0f;

    float m = -INFINITY;
    uint bad = 0;
    for (uint i = tid; i < p.cols; i += threads) {
        float v = xr[i];
        bad |= (isnan(v) || v == INFINITY) ? 1u : 0u;
        m = max(m, v);
    }
    m = threadgroup_max(m, fscratch, sg, lane, sgs);
    bad = threadgroup_max_u(bad, uscratch, sg, lane, sgs);
    if (bad != 0 || m == -INFINITY) {
        if (tid == 0) {
            out[row] = float2(0.0f, 0.0f);
        }
        return;
    }
    // Weights above `low` remain candidates; `s` is this thread's share of their mass.
    float low = 0.0f;
    float s = 0.0f;
    for (uint i = tid; i < p.cols; i += threads) {
        float w = exp((xr[i] - m) * inv_t);
        s += w > low ? w : 0.0f;
    }
    float mass;
    float prefix = threadgroup_scan(s, fscratch, sg, lane, sgs, mass);
    float total = mass;
    uint per_thread = (p.cols + threads - 1) / threads;
    for (uint round = 0; round < p.rounds; ++round) {
        float target = uniform24(seed, draw, round) * mass;
        uint owner = threadgroup_min_u(s > 0.0f && target < prefix + s ? tid : UINT_MAX, uscratch, sg, lane, sgs);
        if (owner == UINT_MAX) {
            owner = threadgroup_max_u(s > 0.0f ? tid : 0u, uscratch, sg, lane, sgs);
        }
        if (tid == owner) {
            uscratch[0] = as_type<uint>(target - prefix);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local = as_type<float>(uscratch[0]);
        // Simdgroup 0 scans the owner's tokens 32 at a time: the first past the target,
        // else (rounding) the last candidate.
        if (sg == 0) {
            float base = 0.0f;
            uint j = UINT_MAX;
            float wj = 0.0f;
            uint fallback = UINT_MAX;
            float fallback_w = 0.0f;
            for (uint k0 = 0; k0 < per_thread && j == UINT_MAX; k0 += SIMD) {
                uint i = owner + (k0 + lane) * threads;
                float w = i < p.cols ? exp((xr[i] - m) * inv_t) : 0.0f;
                w = w > low ? w : 0.0f;
                float r = base + simd_prefix_inclusive_sum(w);
                uint hit = simd_min(w > 0.0f && r > local ? lane : UINT_MAX);
                uint last = simd_max(w > 0.0f ? lane + 1 : 0u);
                if (hit != UINT_MAX) {
                    j = owner + (k0 + hit) * threads;
                    wj = simd_shuffle(w, ushort(hit));
                } else if (last > 0) {
                    fallback = owner + (k0 + last - 1) * threads;
                    fallback_w = simd_shuffle(w, ushort(last - 1));
                }
                base = simd_shuffle(r, ushort(SIMD - 1));
            }
            if (lane == 0) {
                chosen = j != UINT_MAX ? j : fallback;
                chosen_w = j != UINT_MAX ? wj : fallback_w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint j = chosen;
        float wj = chosen_w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!truncating) {
            if (tid == 0) {
                out[row] = float2(float(j), 1.0f);
            }
            return;
        }
        float heavier = 0.0f;
        s = 0.0f;
        for (uint i = tid; i < p.cols; i += threads) {
            float w = exp((xr[i] - m) * inv_t);
            if (w > wj) {
                s += w;
                heavier += 1.0f;
            }
        }
        float count = threadgroup_sum(heavier, fscratch, sg, lane, sgs);
        prefix = threadgroup_scan(s, fscratch, sg, lane, sgs, mass);
        if ((top_k == 0 || count < float(top_k)) && mass < top_p * total) {
            if (tid == 0) {
                out[row] = float2(float(j), 1.0f);
            }
            return;
        }
        low = wj;
    }
    if (tid == 0) {
        out[row] = float2(0.0f, 2.0f);
    }
}
