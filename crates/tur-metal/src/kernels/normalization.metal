// ---------------------------------------------------------------------------------------
// Normalization and elementwise primitives
// ---------------------------------------------------------------------------------------

struct RmsParams {
    uint rows;
    uint width;
    uint x_stride;
    uint r_stride;
    float eps;
};

// y = x * rsqrt(mean(x^2) + eps) * w, one threadgroup per row.
kernel void rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant RmsParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.x_stride;
    float acc = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        float v = xr[i];
        acc += v * v;
    }
    float total = threadgroup_sum(acc, scratch, sg, lane, sgs);
    float inv = 1.0f / sqrt(total / float(p.width) + p.eps);
    device float *yr = y + ulong(row) * p.width;
    for (uint i = tid; i < p.width; i += threads) {
        yr[i] = xr[i] * inv * w[i];
    }
}

// s = x + r (written out), y = rms(s) * w: residual addition fused with the next norm.
kernel void add_rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *r [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *s [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant RmsParams &p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.x_stride;
    device const float *rr = r + ulong(row) * p.r_stride;
    device float *sr = s + ulong(row) * p.width;
    float acc = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        float v = xr[i] + rr[i];
        sr[i] = v;
        acc += v * v;
    }
    float total = threadgroup_sum(acc, scratch, sg, lane, sgs);
    float inv = 1.0f / sqrt(total / float(p.width) + p.eps);
    device float *yr = y + ulong(row) * p.width;
    for (uint i = tid; i < p.width; i += threads) {
        yr[i] = (xr[i] + rr[i]) * inv * w[i];
    }
}

struct RopeParams {
    uint tokens;
    uint heads;
    uint dim;
    uint row_stride;
    uint head_stride;
    uint col_offset;
    uint half_rotary;
    float eps;
};

// Per-head RMSNorm followed by split-half rotary over the first 2 * half_rotary dimensions.
// Angles are position * inv_freq[f] in F32; the host computes inv_freq exactly as the
// reference does, because a one-ulp frequency difference grows linearly with position.
// Reads heads at an arbitrary row/head stride (e.g. the q half of an interleaved q|gate
// projection) and writes contiguous [tokens, heads, dim]. One simdgroup per (token, head).
kernel void rope_scaled_f32(
    device const float *x [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const uint *positions [[buffer(2)]],
    device const float *inv_freq [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant RopeParams &p [[buffer(5)]],
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
    device const float *xr = x + ulong(t) * p.row_stride + p.col_offset + ulong(h) * p.head_stride;
    float position = float(positions[t]);
    uint half_rotary = p.half_rotary;
    float amplitude = scale[0];
    device float *yr = y + (ulong(t) * p.heads + h) * p.dim;
    for (uint i = lane; i < 2 * half_rotary; i += SIMD) {
        bool left = i < half_rotary;
        uint f = left ? i : i - half_rotary;
        float angle = position * inv_freq[f];
        float c = precise::cos(angle) * amplitude;
        float s = precise::sin(angle) * amplitude;
        float v = xr[i];
        float u = xr[left ? i + half_rotary : i - half_rotary];
        yr[i] = left ? v * c - u * s : v * c + u * s;
    }
}

kernel void qk_norm_rope_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const uint *positions [[buffer(2)]],
    device const float *inv_freq [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant RopeParams &p [[buffer(5)]],
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
    device const float *xr = x + ulong(t) * p.row_stride + p.col_offset + ulong(h) * p.head_stride;
    float acc = 0.0f;
    for (uint i = lane; i < p.dim; i += SIMD) {
        float v = xr[i];
        acc += v * v;
    }
    float inv = 1.0f / sqrt(simd_sum(acc) / float(p.dim) + p.eps);
    float position = float(positions[t]);
    uint half_rotary = p.half_rotary;
    device float *yr = y + (ulong(t) * p.heads + h) * p.dim;
    for (uint i = lane; i < p.dim; i += SIMD) {
        float v = xr[i] * inv * w[i];
        if (i < 2 * half_rotary) {
            bool left = i < half_rotary;
            uint f = left ? i : i - half_rotary;
            uint partner = left ? i + half_rotary : i - half_rotary;
            float u = xr[partner] * inv * w[partner];
            float angle = position * inv_freq[f];
            float c = precise::cos(angle);
            float s = precise::sin(angle);
            v = left ? v * c - u * s : v * c + u * s;
        }
        yr[i] = v;
    }
}

struct GluParams {
    uint rows;
    uint width;
};

// out[r, i] = silu(gu[r, i]) * gu[r, width + i] over a fused gate|up projection.
kernel void swiglu_f32(
    device const float *gu [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant GluParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.width) {
        return;
    }
    uint r = gid / p.width;
    uint c = gid % p.width;
    device const float *row = gu + ulong(r) * 2 * p.width;
    out[gid] = silu(row[c]) * row[p.width + c];
}

struct GatedParams {
    uint rows;
    uint dim;
    uint heads;
    uint z_stride;
    uint z_col;
    float eps;
};

// out = rms(core) * w * silu(z) per [token, head] row; z is read in place from the projection.
kernel void gated_rms_f32(
    device const float *core [[buffer(0)]],
    device const float *z [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant GatedParams &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint row = tg * sgs + sg;
    if (row >= p.rows) {
        return;
    }
    uint t = row / p.heads;
    uint h = row % p.heads;
    device const float *cr = core + ulong(row) * p.dim;
    device const float *zr = z + ulong(t) * p.z_stride + p.z_col + ulong(h) * p.dim;
    float acc = 0.0f;
    for (uint i = lane; i < p.dim; i += SIMD) {
        float v = cr[i];
        acc += v * v;
    }
    float inv = 1.0f / sqrt(simd_sum(acc) / float(p.dim) + p.eps);
    device float *o = out + ulong(row) * p.dim;
    for (uint i = lane; i < p.dim; i += SIMD) {
        o[i] = cr[i] * inv * w[i] * silu(zr[i]);
    }
}


struct ResidualParams {
    uint rows;
    uint width;
    uint x_stride;
    uint r_stride;
    uint emit;
    float eps;
};

// s = (x + rms(r) * wr) * scale[0], and y = rms(s) * w when p.emit: a post-norm residual
// update fused with the layer scalar and the next pre-norm.
kernel void residual_rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *r [[buffer(1)]],
    device const float *wr [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    device const float *w [[buffer(4)]],
    device float *s [[buffer(5)]],
    device float *y [[buffer(6)]],
    constant ResidualParams &p [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[SIMD];
    if (row >= p.rows) {
        return;
    }
    device const float *xr = x + ulong(row) * p.x_stride;
    device const float *rr = r + ulong(row) * p.r_stride;
    float acc = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        float v = rr[i];
        acc += v * v;
    }
    float inv_r = 1.0f / sqrt(threadgroup_sum(acc, scratch, sg, lane, sgs) / float(p.width) + p.eps);
    float k = scale[0];
    device float *sr = s + ulong(row) * p.width;
    acc = 0.0f;
    for (uint i = tid; i < p.width; i += threads) {
        float v = (xr[i] + rr[i] * inv_r * wr[i]) * k;
        sr[i] = v;
        acc += v * v;
    }
    if (!p.emit) {
        return;
    }
    // Each thread reads back only the elements of s it wrote.
    float inv = 1.0f / sqrt(threadgroup_sum(acc, scratch, sg, lane, sgs) / float(p.width) + p.eps);
    device float *yr = y + ulong(row) * p.width;
    for (uint i = tid; i < p.width; i += threads) {
        yr[i] = sr[i] * inv * w[i];
    }
}

struct GeluMulParams {
    uint rows;
    uint width;
    uint a_stride;
    uint b_stride;
};

// out[r, i] = gelu_tanh(a[r, i]) * b[r, i]; a and b are read at their own row strides, e.g.
// the halves of a fused gate|up projection or one layer's slice of per-layer inputs.
kernel void gelu_mul_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant GeluMulParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.width) {
        return;
    }
    uint r = gid / p.width;
    uint c = gid % p.width;
    float x = a[ulong(r) * p.a_stride + c];
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[gid] = 0.5f * x * (1.0f + precise::tanh(inner)) * b[ulong(r) * p.b_stride + c];
}
