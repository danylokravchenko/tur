// Fused RMSNorm + Linear projection kernel for Apple Metal.
//
// Computes: out[token, k] = dot(rms_norm(x[token, :]), proj_weight[k, :])
//   where rms_norm(x) = (x / rms(x)) * norm_weight
//
// Supports float32 and bfloat16. Accumulation is always in float32.
//
// Dispatch geometry
//   grid       = (N_tokens, 1, 1) threadgroups
//   threadgroup = (BLOCK_SIZE, 1, 1) threads   — must equal dispatch tg size
//   smem index 0: (H + BLOCK_SIZE) * sizeof(float) bytes

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared implementation (template over element type T)
// ---------------------------------------------------------------------------
template<typename T>
inline void rms_norm_linear_impl(
    device const T*    x,
    device const T*    norm_weight,
    device const T*    proj_weight,
    device       T*    out,
    constant uint&     H,
    constant uint&     K,
    constant float&    eps,
    threadgroup float* smem,          // (H + tg_sz) floats
    uint               token,         // threadgroup index = row to process
    uint               tid,           // lane within threadgroup
    uint               tg_sz          // == BLOCK_SIZE at dispatch time
) {
    device const T* x_row = x + (ulong)token * H;

    threadgroup float* norm_row = smem;         // H floats: normalised activation
    threadgroup float* reduce   = smem + H;     // tg_sz floats: reduction scratch

    // ── Phase 1: load row, accumulate partial sum-of-squares ─────────────────
    float local_ss = 0.0f;
    for (uint i = tid; i < H; i += tg_sz) {
        float v = float(x_row[i]);
        norm_row[i] = v;
        local_ss = fma(v, v, local_ss);
    }
    reduce[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: parallel tree reduction → sum-of-squares ────────────────────
    for (uint stride = tg_sz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(reduce[0] / float(H) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: normalise row in shared mem (mul rms_inv * norm_weight) ─────
    for (uint i = tid; i < H; i += tg_sz) {
        norm_row[i] = norm_row[i] * rms_inv * float(norm_weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: each thread computes K / tg_sz output elements ──────────────
    // Proj weight layout: [K, H] row-major (candle convention: [out_dim, in_dim])
    device T* out_row = out + (ulong)token * K;
    for (uint k = tid; k < K; k += tg_sz) {
        device const T* w_row = proj_weight + (ulong)k * H;
        float acc = 0.0f;
        for (uint i = 0; i < H; i++) {
            acc = fma(norm_row[i], float(w_row[i]), acc);
        }
        out_row[k] = T(acc);
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

kernel void rms_norm_linear_f32(
    device const float* x           [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* proj_weight [[buffer(2)]],
    device       float* out         [[buffer(3)]],
    constant uint&      H           [[buffer(4)]],
    constant uint&      K           [[buffer(5)]],
    constant float&     eps         [[buffer(6)]],
    threadgroup float*  smem        [[threadgroup(0)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]]
) {
    rms_norm_linear_impl<float>(
        x, norm_weight, proj_weight, out, H, K, eps, smem, token, tid, tg_sz
    );
}

kernel void rms_norm_linear_bf16(
    device const bfloat* x           [[buffer(0)]],
    device const bfloat* norm_weight [[buffer(1)]],
    device const bfloat* proj_weight [[buffer(2)]],
    device       bfloat* out         [[buffer(3)]],
    constant uint&       H           [[buffer(4)]],
    constant uint&       K           [[buffer(5)]],
    constant float&      eps         [[buffer(6)]],
    threadgroup float*   smem        [[threadgroup(0)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]]
) {
    rms_norm_linear_impl<bfloat>(
        x, norm_weight, proj_weight, out, H, K, eps, smem, token, tid, tg_sz
    );
}
