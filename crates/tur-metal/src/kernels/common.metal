// Tur fused kernels. F32 activations; reductions and recurrent state accumulate in F32.
// Every kernel derives its work item from the dispatched threadgroup shape, so the host
// may choose any threadgroup width that is a multiple of the SIMD width.
#include <metal_stdlib>
using namespace metal;

#define SIMD 32

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-fabs(x))); }

// Sum over the whole threadgroup. Every thread must call it; `scratch` holds >= 32 floats.
static inline float threadgroup_sum(float v, threadgroup float *scratch, uint sg, uint lane, uint sgs) {
    v = simd_sum(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < sgs ? scratch[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

