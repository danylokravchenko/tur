/// Qwen3.5 multimodal inference model.
///
/// Architecture: 24-layer hybrid model alternating between Gated Delta-Net linear
/// attention (18 layers) and standard full-attention (6 layers, every 4th starting
/// at index 3), plus a ViT-style vision encoder.  Vision tokens are merged into the
/// text embedding stream at positions marked by `image_token_id`.
use crate::backend::progress::ProgressReporter;
use crate::models::kv_cache::{KvCache, KvCacheImpl, PagedKvCache};
use crate::models::{ModelImpl, ModelInput};
use crate::models::layers;
use crate::weights::VarBuilderX;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Activation;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

#[cfg(feature = "flash-attn")]
use candle_flash_attn;


// ─── Configs ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_interleaved: Option<bool>,
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub hidden_act: Activation,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub image_token_id: u32,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub tie_word_embeddings: bool,
}

// ─── Partial Rotary Embedding (text full-attention only) ──────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Qwen35RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Number of dims that actually get rotated (partial_rotary_factor * head_dim).
    rotary_dim: usize,
}

impl Qwen35RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &TextConfig, dev: &Device) -> Result<Self> {
        let rotary_dim =
            (cfg.head_dim as f64 * cfg.rope_parameters.partial_rotary_factor) as usize;
        // rotary_dim must be even for the sin/cos split to work
        let rotary_dim = rotary_dim & !1;
        let half = rotary_dim / 2;
        let theta = cfg.rope_parameters.rope_theta;
        let max_seq = cfg.max_position_embeddings;

        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / (theta.powf(2.0 * i as f64 / rotary_dim as f64) as f32))
            .collect();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, half), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            rotary_dim,
        })
    }

    /// Apply partial RoPE to q and k (only the first `rotary_dim` dims are rotated).
    pub(crate) fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;

        let q_embed = self.apply_partial(q, &cos, &sin, head_dim)?;
        let k_embed = self.apply_partial(k, &cos, &sin, head_dim)?;
        Ok((q_embed, k_embed))
    }

    fn apply_partial(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        head_dim: usize,
    ) -> Result<Tensor> {
        let pass_dims = head_dim.saturating_sub(self.rotary_dim);
        let x_rot = x.narrow(3, 0, self.rotary_dim)?.contiguous()?;
        let x_embedded = candle_nn::rotary_emb::rope(&x_rot, cos, sin)?;
        if pass_dims == 0 {
            return Ok(x_embedded);
        }
        let x_pass = x.narrow(3, self.rotary_dim, pass_dims)?;
        Tensor::cat(&[x_embedded, x_pass.contiguous()?], 3)
    }
}

// ─── Causal conv1d helper ─────────────────────────────────────────────────────

/// Causal 1-D depthwise convolution.
///
/// Pads the left side with zeros so the output length equals the input length and
/// each position only sees past (and current) positions.
#[derive(Debug, Clone)]
struct CausalConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    channels: usize,
}

impl CausalConv1d {
    fn new(channels: usize, kernel_size: usize, vb: VarBuilderX) -> Result<Self> {
        // GGUF stores conv1d weight as [channels, kernel_size]; HF uses [channels, 1, kernel_size].
        let weight = if vb.is_qvar_builder() {
            vb.get((channels, kernel_size), "weight")?.unsqueeze(1)?
        } else {
            vb.get((channels, 1, kernel_size), "weight")?
        };
        let bias = vb.get(channels, "bias").ok();
        Ok(Self { weight, bias, kernel_size, channels })
    }

    /// Apply causal depthwise conv to `x` of shape [B, L, C].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        // [B, L, C] → [B, C, L]
        let x_bcl = x.permute((0, 2, 1))?.contiguous()?;

        // Left-pad with (kernel_size-1) zeros for causal behaviour
        let pad = self.kernel_size - 1;
        let zeros = Tensor::zeros((b, c, pad), x.dtype(), x.device())?;
        let padded = Tensor::cat(&[&zeros, &x_bcl], 2)?;

        // Depthwise grouped conv (groups = channels)
        let out = padded.conv1d(&self.weight, 0, 1, 1, self.channels)?;
        let out = if let Some(b) = &self.bias {
            out.broadcast_add(&b.unsqueeze(0)?.unsqueeze(2)?)?
        } else {
            out
        };
        // [B, C, L'] → [B, L', C]
        // L' should equal l
        let (_, _, l_out) = out.dims3()?;
        debug_assert_eq!(l_out, l);
        out.permute((0, 2, 1))
    }

    /// Update a ring-buffer `buf` of shape [B, C, kernel_size-1] with a single new
    /// K slice `k_t` of shape [B, 1, C] and return the conv output [B, 1, C].
    fn forward_step(&self, k_t: &Tensor, buf: &mut Option<Tensor>) -> Result<Tensor> {
        let (b, _one, c) = k_t.dims3()?;
        let k_t_bcd = k_t.permute((0, 2, 1))?; // [B, C, 1]

        let padded = if let Some(b_buf) = buf.as_ref() {
            Tensor::cat(&[b_buf, &k_t_bcd], 2)? // [B, C, kernel_size]
        } else {
            let zeros = Tensor::zeros((b, c, self.kernel_size - 1), k_t.dtype(), k_t.device())?;
            Tensor::cat(&[&zeros, &k_t_bcd], 2)? // [B, C, kernel_size]
        };

        // Update buffer: keep last (kernel_size-1) entries
        let pad = self.kernel_size - 1;
        *buf = Some(padded.narrow(2, 1, pad)?);

        // Apply weight: [channels, 1, kernel_size] × [B, C, kernel_size] grouped conv
        let out = padded.conv1d(&self.weight, 0, 1, 1, self.channels)?;
        let out = if let Some(b) = &self.bias {
            out.broadcast_add(&b.unsqueeze(0)?.unsqueeze(2)?)?
        } else {
            out
        };
        // [B, C, 1] → [B, 1, C]
        out.permute((0, 2, 1))
    }
}

// ─── Linear Attention (Gated Delta Net) ──────────────────────────────────────

/// Per-layer state for the Gated Delta-Net recurrence.
#[derive(Debug, Clone, Default)]
struct LinearAttnState {
    /// Delta-rule state: [B, num_heads, key_head_dim, value_head_dim]
    delta: Option<Tensor>,
    /// Causal-conv ring buffer: [B, conv_channels, kernel_size-1]
    conv_buf: Option<Tensor>,
}

impl LinearAttnState {
    fn reset(&mut self) {
        self.delta = None;
        self.conv_buf = None;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen35LinearAttention {
    in_proj_qkv: layers::LinearX,
    in_proj_z: layers::LinearX,
    in_proj_b: layers::LinearX,
    in_proj_a: layers::LinearX,
    conv1d: CausalConv1d,
    /// A_log parameter: [num_key_heads * key_head_dim]
    a_log: Tensor,
    /// dt_bias parameter: [num_key_heads * key_head_dim]
    dt_bias: Tensor,
    norm: layers::norm::NormX,
    out_proj: layers::LinearX,

    num_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    num_value_heads: usize,

    state: LinearAttnState,
}

impl Qwen35LinearAttention {
    pub(crate) fn new(cfg: &TextConfig, vb: VarBuilderX) -> Result<Self> {
        let nh = cfg.linear_num_key_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nvh = cfg.linear_num_value_heads;
        let hidden = cfg.hidden_size;
        let conv_k = cfg.linear_conv_kernel_dim;

        // GGUF: flat at block level; HF: sub-prefixed under "linear_attn."
        let is_gguf = vb.is_qvar_builder();
        let (qkv_name, z_name, b_name, a_name) = if is_gguf {
            ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha")
        } else {
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        };
        let (conv_name, a_log_name, dt_bias_name, norm_name, out_name) = if is_gguf {
            ("ssm_conv1d", "ssm_a", "ssm_dt.bias", "ssm_norm", "ssm_out")
        } else {
            ("conv1d", "A_log", "dt_bias", "norm", "out_proj")
        };

        let qkv_out = nh * kd + nh * kd + nvh * vd;
        let in_proj_qkv = layers::linear(hidden, qkv_out, vb.pp(qkv_name))?;
        let in_proj_z = layers::linear(hidden, nvh * vd, vb.pp(z_name))?;
        let in_proj_b = layers::linear(hidden, nh, vb.pp(b_name))?;
        // alpha (decay input) is per-head scalar, not per-element
        let in_proj_a = layers::linear(hidden, nh, vb.pp(a_name))?;

        // Conv1d applied to the full QKV tensor (all qkv_out channels)
        let conv1d = CausalConv1d::new(qkv_out, conv_k, vb.pp(conv_name))?;

        // A_log and dt_bias are per-head scalars [nh]
        let a_log = vb.get(nh, a_log_name)?;
        let dt_bias = vb.get(nh, dt_bias_name)?;

        // Norm is applied per-head over vd dims (state_size), not over nvh*vd total
        let norm = layers::norm::rms_norm(vd, cfg.rms_norm_eps, vb.pp(norm_name), vb.dtype(), false)?;
        let out_proj = layers::linear(nvh * vd, hidden, vb.pp(out_name))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            a_log,
            dt_bias,
            norm,
            out_proj,
            num_heads: nh,
            key_head_dim: kd,
            value_head_dim: vd,
            num_value_heads: nvh,
            state: LinearAttnState::default(),
        })
    }

    /// Numerically-stable softplus: log(1 + exp(x)).
    fn softplus(x: &Tensor) -> Result<Tensor> {
        // Clamp to avoid fp overflow before exp.
        let x_c = x.clamp(-20.0f64, 20.0f64)?;
        (x_c.exp()? + 1.0)?.log()
    }

    /// Compute per-head decay scalar: exp(-exp(A_log) * softplus(a + dt_bias))  [B, L, nh]
    fn compute_decay(&self, a: &Tensor) -> Result<Tensor> {
        let a_log_exp = self.a_log.exp()?;         // [nh]
        let dt = a.broadcast_add(&self.dt_bias)?;  // [B, L, nh]
        let sp = Self::softplus(&dt)?;             // [B, L, nh]
        let neg_decay = sp.broadcast_mul(&a_log_exp)?.neg()?;
        neg_decay.exp()  // [B, L, nh]
    }

    /// Apply the Gated Delta-Net recurrence across L tokens.
    ///
    /// * q, k, v: [B, L, nh, kd/vd]
    /// * beta:    [B, L, nh]  (per-head scalar)
    /// * decay:   [B, L, nh]  (per-head scalar, broadcast over kd/vd)
    fn gdn_recurrence(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        decay: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, nh, vd) = v.dims4()?;

        // Normalise k to unit-ball for stable linear attention.
        let k_norm_sq = k.sqr()?.sum_keepdim(3)?;
        let k_norm = (k_norm_sq + 1e-8)?.sqrt()?;
        let k = k.broadcast_div(&k_norm)?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(l);

        for t in 0..l {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?;       // [B, nh, kd]
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;       // [B, nh, kd]
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;       // [B, nh, vd]
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?; // [B, nh]
            let decay_t = decay.narrow(1, t, 1)?.squeeze(1)?; // [B, nh]

            // S shape: [B, nh, kd, vd]
            let s = if let Some(ref s) = self.state.delta {
                let s_k = s
                    .transpose(2, 3)?                 // [B, nh, vd, kd]
                    .matmul(&k_t.unsqueeze(3)?)?      // [B, nh, vd, 1]
                    .squeeze(3)?;                     // [B, nh, vd]

                let v_err = (v_t - s_k)?;            // [B, nh, vd]

                // outer product k ⊗ v_err: [B, nh, kd, vd]
                let upd = k_t.unsqueeze(3)?.matmul(&v_err.unsqueeze(2)?)?;

                // decay and beta: [B, nh] → [B, nh, 1, 1] to broadcast over [kd, vd]
                let beta_u = beta_t.unsqueeze(2)?.unsqueeze(3)?;
                let decay_u = decay_t.unsqueeze(2)?.unsqueeze(3)?;

                (decay_u.broadcast_mul(s)? + beta_u.broadcast_mul(&upd)?)?
            } else {
                let upd = k_t.unsqueeze(3)?.matmul(&v_t.unsqueeze(2)?)?;
                let beta_u = beta_t.unsqueeze(2)?.unsqueeze(3)?;
                beta_u.broadcast_mul(&upd)?
            };

            self.state.delta = Some(s.clone());

            // Output: s^T q  → [B, nh, vd]
            let o_t = s
                .transpose(2, 3)?              // [B, nh, vd, kd]
                .matmul(&q_t.unsqueeze(3)?)?   // [B, nh, vd, 1]
                .squeeze(3)?;                  // [B, nh, vd]

            outputs.push(o_t.reshape((b, 1, nh * vd))?);
        }

        Tensor::cat(&outputs, 1) // [B, L, nh*vd]
    }

    pub(crate) fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let nh = self.num_heads;
        let kd = self.key_head_dim;
        let vd = self.value_head_dim;
        let nvh = self.num_value_heads;

        // Projections
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;          // [B, L, nvh*vd]
        let b_raw = self.in_proj_b.forward(x)?;      // [B, L, nh]
        let a_raw = self.in_proj_a.forward(x)?;      // [B, L, nh]

        // Apply causal conv to the full QKV tensor before splitting.
        // forward_step expects [B, 1, C] and forward expects [B, L, C].
        let qkv_conv = if l == 1 {
            self.conv1d.forward_step(&qkv, &mut self.state.conv_buf)?  // [B, 1, qkv_out]
        } else {
            self.conv1d.forward(&qkv)?  // [B, L, qkv_out]
        };

        let q_raw = qkv_conv.narrow(2, 0, nh * kd)?;
        let k_raw = qkv_conv.narrow(2, nh * kd, nh * kd)?;
        let v_raw = qkv_conv.narrow(2, 2 * nh * kd, nvh * vd)?;

        let decay = self.compute_decay(&a_raw)?;     // [B, L, nh]
        let beta = candle_nn::ops::sigmoid(&b_raw)?; // [B, L, nh]

        // Reshape for per-head ops
        let q = q_raw.reshape((b, l, nh, kd))?;
        let k = k_raw.reshape((b, l, nh, kd))?;
        let v = v_raw.reshape((b, l, nvh, vd))?;

        // GDN recurrence → [B, L, nh*vd]
        let out = self.gdn_recurrence(&q, &k, &v, &beta, &decay)?;

        // Norm is per-head over vd, then flatten: [B, L, nh*vd] → reshape → norm → reshape
        let out = out.reshape((b, l, nh, vd))?;
        let out = self.norm.forward(&out.contiguous()?)?;  // norm over last dim (vd)
        let out = out.reshape((b, l, nh * vd))?;

        // Output gate and projection
        let gate = candle_nn::ops::sigmoid(&z)?;
        let out = (out * gate)?;
        self.out_proj.forward(&out)
    }

    pub(crate) fn clear_state(&mut self) {
        self.state.reset();
    }
}

// ─── Full Attention (with output gate) ───────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Qwen35FullAttention {
    // q_proj output is doubled: [num_heads * head_dim * 2] (query + sigmoid gate)
    q_proj: layers::LinearX,
    k_proj: layers::LinearX,
    v_proj: layers::LinearX,
    o_proj: layers::LinearX,
    q_norm: layers::norm::NormX,
    k_norm: layers::norm::NormX,

    rotary: Arc<Qwen35RotaryEmbedding>,
    kv_cache: KvCache,

    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Qwen35FullAttention {
    pub(crate) fn new(
        cfg: &TextConfig,
        rotary: Arc<Qwen35RotaryEmbedding>,
        vb: VarBuilderX,
    ) -> Result<Self> {
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let hidden = cfg.hidden_size;

        // GGUF: flat at block level; HF: sub-prefixed under "self_attn."
        let is_gguf = vb.is_qvar_builder();
        let (q_name, k_name, v_name, o_name, qn_name, kn_name) = if is_gguf {
            ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
        } else {
            ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        };

        // q_proj doubles head_dim to embed the output gate
        let q_proj = layers::linear(hidden, nh * hd * 2, vb.pp(q_name))?;
        let k_proj = layers::linear(hidden, nkv * hd, vb.pp(k_name))?;
        let v_proj = layers::linear(hidden, nkv * hd, vb.pp(v_name))?;
        let o_proj = layers::linear(nh * hd, hidden, vb.pp(o_name))?;

        let q_norm = layers::norm::rms_norm(hd, cfg.rms_norm_eps, vb.pp(qn_name), vb.dtype(), false)?;
        let k_norm = layers::norm::rms_norm(hd, cfg.rms_norm_eps, vb.pp(kn_name), vb.dtype(), false)?;

        let hidden_size = nh * hd;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            kv_cache: KvCache::new(2),
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: hd,
            hidden_size,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Project — q_proj output = [B, L, nh*hd*2]
        let q_full = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Split q into [query, gate]
        let qkd = self.num_heads * self.head_dim;
        let q = q_full.narrow(2, 0, qkd)?;
        let gate = q_full.narrow(2, qkd, qkd)?; // [B, L, nh*hd]

        // Reshape: (B, L, H, D) → (B, H, L, D)
        let q = q.reshape((b, l, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Per-head RMSNorm
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // Partial RoPE (first rotary_dim dims only)
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // Attention
        let on_cpu = x.device().is_cpu();

        #[cfg(not(feature = "flash-attn"))]
        let attn_out = if on_cpu {
            self.forward_cpu_flash(&q, &k, &v, offset, b, l)?
        } else {
            self.forward_standard(&q, &k, &v, attn_mask, b, l)?
        };

        #[cfg(feature = "flash-attn")]
        let attn_out = if !on_cpu {
            self.forward_flash(&q, &k, &v, b, l)?
        } else {
            self.forward_standard(&q, &k, &v, attn_mask, b, l)?
        };

        // Apply output gate: sigmoid(gate) applied element-wise
        let gate_sig = candle_nn::ops::sigmoid(&gate)?; // [B, L, nh*hd]
        let out = (attn_out * gate_sig)?;
        self.o_proj.forward(&out)
    }

    /// Batched forward with per-request paged caches.
    pub(crate) fn forward_paged(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        positions: &[usize],
        paged_caches: Vec<&mut PagedKvCache>,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q_full = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let qkd = self.num_heads * self.head_dim;
        let q = q_full.narrow(2, 0, qkd)?;
        let gate = q_full.narrow(2, qkd, qkd)?;

        let q = q.reshape((b, l, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self.rotary.apply_batch(&q, &k, positions)?;

        let mut output_parts: Vec<Tensor> = Vec::with_capacity(b);
        for (bi, cache) in paged_caches.into_iter().enumerate() {
            let q_r = q.narrow(0, bi, 1)?;
            let k_r = k.narrow(0, bi, 1)?;
            let v_r = v.narrow(0, bi, 1)?;
            let gate_r = gate.narrow(0, bi, 1)?;

            let (k_full, v_full) = cache.append(&k_r, &v_r)?;
            let offset = positions[bi];

            let attn_r = self.single_attn(&q_r, &k_full, &v_full, attn_mask, offset, l)?;
            let gate_sig = candle_nn::ops::sigmoid(&gate_r)?;
            let out_r = (attn_r * gate_sig)?;
            output_parts.push(self.o_proj.forward(&out_r)?);
        }
        Tensor::cat(&output_parts, 0)
    }

    fn single_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
        l: usize,
    ) -> Result<Tensor> {
        let on_cpu = q.device().is_cpu();

        #[cfg(not(feature = "flash-attn"))]
        if on_cpu {
            return self.forward_cpu_flash(q, k, v, offset, 1, l);
        }
        #[cfg(feature = "flash-attn")]
        if !on_cpu {
            return self.forward_flash(q, k, v, 1, l);
        }

        self.forward_standard(q, k, v, attn_mask, 1, l)
    }

    #[cfg(not(feature = "flash-attn"))]
    fn forward_cpu_flash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        offset: usize,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        use crate::models::attention::{AttnMask, flash_attn};
        let q_bshd = q.transpose(1, 2)?.contiguous()?;
        let k_bshd = k.transpose(1, 2)?.contiguous()?;
        let v_bshd = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let ctx = match q.dtype() {
            DType::F32 => flash_attn::<f32>(&q_bshd, &k_bshd, &v_bshd, scale,
                AttnMask::causal_with_offset(offset), None, None)?,
            DType::BF16 => {
                let q32 = q_bshd.to_dtype(DType::F32)?;
                let k32 = k_bshd.to_dtype(DType::F32)?;
                let v32 = v_bshd.to_dtype(DType::F32)?;
                flash_attn::<f32>(&q32, &k32, &v32, scale,
                    AttnMask::causal_with_offset(offset), None, None)?
                    .to_dtype(DType::BF16)?
            }
            dt => candle_core::bail!("unsupported dtype {:?} in cpu flash attn", dt),
        };
        // ctx is [B, H, S, D] → [B, S, H*D]
        ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))
    }

    #[cfg(feature = "flash-attn")]
    fn forward_flash(&self, q: &Tensor, k: &Tensor, v: &Tensor, b: usize, l: usize) -> Result<Tensor> {
        let q_bshd = q.transpose(1, 2)?.contiguous()?;
        let k_bshd = k.transpose(1, 2)?.contiguous()?;
        let v_bshd = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let causal = l > 1;
        let ctx = candle_flash_attn::flash_attn(&q_bshd, &k_bshd, &v_bshd, scale, causal)?;
        ctx.reshape((b, l, self.hidden_size))
    }

    fn forward_standard(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let k = repeat_kv(k.clone(), self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), self.num_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs.matmul(&v)?.transpose(1, 2)?.reshape((b, l, self.hidden_size))
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    pub(crate) fn get_kv_state(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.get_state()
    }

    pub(crate) fn set_kv_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        self.kv_cache.set_state(k, v)
    }
}

impl Qwen35RotaryEmbedding {
    /// Apply RoPE with per-batch element starting positions.
    pub(crate) fn apply_batch(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _nh, _seq, _hd) = q.dims4()?;
        let mut qs: Vec<Tensor> = Vec::with_capacity(batch);
        let mut ks: Vec<Tensor> = Vec::with_capacity(batch);
        for (bi, &off) in positions.iter().enumerate() {
            let q_b = q.narrow(0, bi, 1)?;
            let k_b = k.narrow(0, bi, 1)?;
            let (q_e, k_e) = self.apply(&q_b, &k_b, off)?;
            qs.push(q_e);
            ks.push(k_e);
        }
        Ok((Tensor::cat(&qs, 0)?, Tensor::cat(&ks, 0)?))
    }
}

// ─── MLP ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Qwen35MLP {
    gate_proj: layers::LinearX,
    up_proj: layers::LinearX,
    down_proj: layers::LinearX,
    act_fn: Activation,
}

impl Qwen35MLP {
    fn new(cfg: &TextConfig, vb: VarBuilderX) -> Result<Self> {
        let is_gguf = vb.is_qvar_builder();
        let (gate_name, up_name, down_name) = if is_gguf {
            ("ffn_gate", "ffn_up", "ffn_down")
        } else {
            ("gate_proj", "up_proj", "down_proj")
        };
        Ok(Self {
            gate_proj: layers::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp(gate_name))?,
            up_proj: layers::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp(up_name))?,
            down_proj: layers::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp(down_name))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen35MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ─── Decoder Layer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum LayerAttn {
    Linear(Qwen35LinearAttention),
    Full(Qwen35FullAttention),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attn: LayerAttn,
    mlp: Qwen35MLP,
    ln1: layers::norm::NormX,
    ln2: layers::norm::NormX,
}

impl DecoderLayer {
    fn new(
        cfg: &TextConfig,
        layer_type: &str,
        rotary: Arc<Qwen35RotaryEmbedding>,
        vb: VarBuilderX,
    ) -> Result<Self> {
        let is_gguf = vb.is_qvar_builder();
        let attn = if layer_type == "full_attention" {
            let attn_vb = if is_gguf { vb.clone() } else { vb.pp("self_attn") };
            LayerAttn::Full(Qwen35FullAttention::new(cfg, rotary, attn_vb)?)
        } else {
            let attn_vb = if is_gguf { vb.clone() } else { vb.pp("linear_attn") };
            LayerAttn::Linear(Qwen35LinearAttention::new(cfg, attn_vb)?)
        };

        let mlp_vb = if is_gguf { vb.clone() } else { vb.pp("mlp") };
        let mlp = Qwen35MLP::new(cfg, mlp_vb)?;
        let (ln1_name, ln2_name) = if is_gguf {
            ("attn_norm", "post_attention_norm")
        } else {
            ("input_layernorm", "post_attention_layernorm")
        };
        let ln1 = layers::norm::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(ln1_name), vb.dtype(), false)?;
        let ln2 = layers::norm::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(ln2_name), vb.dtype(), false)?;
        Ok(Self { attn, mlp, ln1, ln2 })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = match &mut self.attn {
            LayerAttn::Linear(la) => la.forward(&h)?,
            LayerAttn::Full(fa) => fa.forward(&h, mask, offset)?,
        };
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?.apply(&self.mlp)?;
        x + h2
    }

    fn forward_batch(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        positions: &[usize],
        paged_caches: Option<Vec<&mut PagedKvCache>>,
    ) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = match &mut self.attn {
            LayerAttn::Linear(la) => la.forward(&h)?,
            LayerAttn::Full(fa) => {
                if let Some(caches) = paged_caches {
                    fa.forward_paged(&h, mask, positions, caches)?
                } else {
                    // Fall back: use min position for single-offset call
                    let offset = *positions.iter().min().unwrap_or(&0);
                    fa.forward(&h, mask, offset)?
                }
            }
        };
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.attn {
            LayerAttn::Linear(la) => la.clear_state(),
            LayerAttn::Full(fa) => fa.clear_kv_cache(),
        }
    }

    fn get_kv_state(&self) -> Option<(Tensor, Tensor)> {
        match &self.attn {
            LayerAttn::Linear(_) => None, // linear attn state not exposed for prefix caching
            LayerAttn::Full(fa) => fa.get_kv_state(),
        }
    }

    fn set_kv_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        match &mut self.attn {
            LayerAttn::Linear(_) => Ok(()), // no-op for linear attention
            LayerAttn::Full(fa) => fa.set_kv_state(k, v),
        }
    }

}

// ─── Vision Encoder ───────────────────────────────────────────────────────────

/// Converts image patches to embeddings.
///
/// The HF implementation uses a Conv3d with temporal_patch_size=2 and patch_size=16.
/// We flatten the kernel into a linear projection:
/// input_dim = in_channels * temporal_patch_size * patch_size^2 = 3 * 2 * 256 = 1536
/// output_dim = vision hidden_size = 768.
#[derive(Debug, Clone)]
struct VisionPatchEmbed {
    proj: layers::LinearX,
}

impl VisionPatchEmbed {
    fn new(cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        let in_dim =
            cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let proj = layers::linear(in_dim, cfg.hidden_size, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    /// Embed a pre-flattened patch tensor of shape [num_patches, in_dim].
    fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        self.proj.forward(patches)
    }
}

/// Multi-head self-attention for ViT blocks.
#[derive(Debug, Clone)]
struct VisionAttention {
    qkv: layers::LinearX,
    proj: layers::LinearX,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        let hd = cfg.hidden_size / cfg.num_heads;
        let qkv = layers::linear(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("qkv"))?;
        let proj = layers::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("proj"))?;
        Ok(Self { qkv, proj, num_heads: cfg.num_heads, head_dim: hd })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        let qkv = self.qkv.forward(x)?
            .reshape((b, l, 3, nh, hd))?
            .permute((2, 0, 3, 1, 4))?; // [3, B, nh, L, hd]

        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?;
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [B, nh, L, hd]
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, nh * hd))?;
        self.proj.forward(&ctx)
    }
}

/// MLP inside a ViT block.
#[derive(Debug, Clone)]
struct VisionMLP {
    fc1: layers::LinearX,
    fc2: layers::LinearX,
    act: Activation,
}

impl VisionMLP {
    fn new(cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        Ok(Self {
            fc1: layers::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("linear_fc1"))?,
            fc2: layers::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("linear_fc2"))?,
            act: cfg.hidden_act,
        })
    }
}

impl Module for VisionMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

/// Single ViT transformer block.
#[derive(Debug, Clone)]
struct VisionBlock {
    norm1: layers::norm::NormX,
    norm2: layers::norm::NormX,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        Ok(Self {
            norm1: layers::norm::rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm1"), DType::F32, false)?,
            norm2: layers::norm::rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm2"), DType::F32, false)?,
            attn: VisionAttention::new(cfg, vb.pp("attn"))?,
            mlp: VisionMLP::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = (x + h)?;
        let h2 = self.norm2.forward(&x)?.apply(&self.mlp)?;
        x + h2
    }
}

/// Projects spatially-merged vision tokens to the text hidden size.
///
/// Groups 2×2 patches, applies RMSNorm, then a 2-layer MLP:
///   [4 * vision_hidden] → [4 * vision_hidden] → [text_hidden]
#[derive(Debug, Clone)]
struct VisionMerger {
    norm: layers::norm::NormX,
    fc1: layers::LinearX,
    fc2: layers::LinearX,
    spatial_merge_size: usize,
}

impl VisionMerger {
    fn new(v_cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        let merge_sq = v_cfg.spatial_merge_size * v_cfg.spatial_merge_size;
        let merged_dim = merge_sq * v_cfg.hidden_size;
        Ok(Self {
            norm: layers::norm::rms_norm(v_cfg.hidden_size, 1e-6, vb.pp("norm"), DType::F32, false)?,
            fc1: layers::linear(merged_dim, merged_dim, vb.pp("linear_fc1"))?,
            fc2: layers::linear(merged_dim, v_cfg.out_hidden_size, vb.pp("linear_fc2"))?,
            spatial_merge_size: v_cfg.spatial_merge_size,
        })
    }

    /// Merge spatial patches and project to text hidden size.
    ///
    /// `tokens`: [num_patches, vision_hidden] — the patch tokens for one image.
    /// Returns [num_visual_tokens, text_hidden].
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (np, vh) = tokens.dims2()?;
        let ms = self.spatial_merge_size;
        let ms_sq = ms * ms;

        // tokens per merged group
        let num_groups = np / ms_sq;
        if np % ms_sq != 0 {
            candle_core::bail!(
                "VisionMerger: num_patches {} is not divisible by spatial_merge_size^2 {}",
                np,
                ms_sq
            );
        }

        // Normalize per patch before merging
        let tokens = self.norm.forward(tokens)?;

        // Reshape: [num_patches, vh] → [num_groups, ms_sq * vh]
        let merged = tokens.reshape((num_groups, ms_sq * vh))?;

        // 2-layer MLP
        merged
            .apply(&self.fc1)?
            .apply(&candle_nn::Activation::Gelu)?
            .apply(&self.fc2)
    }
}

/// Full vision encoder: patch embedding + ViT blocks + spatial merger.
#[derive(Debug, Clone)]
pub struct VisionModel {
    patch_embed: VisionPatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: VisionMerger,
}

impl VisionModel {
    pub fn new(v_cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        let patch_embed = VisionPatchEmbed::new(v_cfg, vb.pp("patch_embed"))?;
        let blocks = (0..v_cfg.depth)
            .map(|i| VisionBlock::new(v_cfg, vb.pp("blocks").pp(&i.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let merger = VisionMerger::new(v_cfg, vb.pp("merger"))?;
        Ok(Self { patch_embed, blocks, merger })
    }

    /// Encode a flat patch tensor [num_patches, in_dim] into text-space embeddings
    /// [num_visual_tokens, text_hidden].
    pub fn encode_patches(&self, patches: &Tensor) -> Result<Tensor> {
        let mut h = self.patch_embed.forward(patches)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        self.merger.forward(&h)
    }
}

// ─── Text Model ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: layers::norm::NormX,
    device: Device,
    dtype: DType,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilderX) -> Result<Self> {
        Self::new_with_progress(cfg, vb, None)
    }

    pub fn new_with_progress(
        cfg: &TextConfig,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self> {
        let is_gguf = vb.is_qvar_builder();
        // GGUF: flat top-level names; HF safetensors: nested under "model.language_model"
        let embed_path = if is_gguf { "token_embd" } else { "model.language_model.embed_tokens" };
        let (embed_tokens, _) = layers::embedding(
            Some(cfg.vocab_size),
            cfg.hidden_size,
            vb.pp(embed_path),
            vb.dtype(),
        )?;

        let rotary = Arc::new(Qwen35RotaryEmbedding::new(vb.dtype(), cfg, &vb.device())?);

        if let Some(p) = progress {
            p.init_loading(cfg.num_hidden_layers);
        }

        let mut dec_layers: Vec<DecoderLayer> = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_path = if is_gguf { "blk" } else { "model.language_model.layers" };
        let vb_l = vb.pp(layers_path);
        for i in 0..cfg.num_hidden_layers {
            let lt = cfg.layer_types.get(i).map(String::as_str).unwrap_or("linear_attention");
            dec_layers.push(DecoderLayer::new(cfg, lt, rotary.clone(), vb_l.pp(&i.to_string()))?);
            if let Some(p) = progress {
                p.inc_loading();
            }
        }

        if let Some(p) = progress {
            p.finish_loading();
        }

        let norm_path = if is_gguf { "output_norm" } else { "model.language_model.norm" };
        let norm = layers::norm::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp(norm_path),
            vb.dtype(),
            false,
        )?;

        Ok(Self {
            embed_tokens,
            layers: dec_layers,
            norm,
            device: vb.device(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn get_kv_cache_state(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut out: Vec<(Tensor, Tensor)> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match layer.get_kv_state() {
                Some(kv) => out.push(kv),
                None => return Ok(Vec::new()), // linear-attn layer → no prefix caching
            }
        }
        Ok(out)
    }

    fn set_kv_cache_state(&mut self, states: Vec<(Tensor, Tensor)>) -> Result<()> {
        if states.len() != self.layers.len() {
            candle_core::bail!(
                "KV state length mismatch: expected {}, got {}",
                self.layers.len(),
                states.len()
            );
        }
        for (layer, (k, v)) in self.layers.iter_mut().zip(states) {
            layer.set_kv_state(k, v)?;
        }
        Ok(())
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    if j <= i + offset { 0.0 } else { minf }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        #[cfg(not(feature = "flash-attn"))]
        let needs_mask = !self.device.is_cpu() && l > 1;
        #[cfg(feature = "flash-attn")]
        let needs_mask = self.device.is_cpu() && l > 1;
        let causal = if needs_mask {
            Some(self.causal_mask(b, l, offset)?)
        } else {
            None
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }

    /// Forward with a pre-built embedding tensor (e.g. with image tokens merged in).
    pub fn forward_embeds(&mut self, embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = embeds.dims3()?;

        #[cfg(not(feature = "flash-attn"))]
        let needs_mask = !self.device.is_cpu() && l > 1;
        #[cfg(feature = "flash-attn")]
        let needs_mask = self.device.is_cpu() && l > 1;
        let causal = if needs_mask {
            Some(self.causal_mask(b, l, offset)?)
        } else {
            None
        };

        let mut h = embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }

    pub fn forward_batch(
        &mut self,
        input: &Tensor,
        positions: &[usize],
        mut paged_caches: Option<&mut [Vec<PagedKvCache>]>,
    ) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        if positions.len() != b {
            candle_core::bail!(
                "positions length {} must match batch size {}",
                positions.len(),
                b
            );
        }

        let mut h = self.embed_tokens.forward(input)?;

        #[cfg(not(feature = "flash-attn"))]
        let needs_mask = !self.device.is_cpu() && l > 1;
        #[cfg(feature = "flash-attn")]
        let needs_mask = self.device.is_cpu() && l > 1;
        let min_offset = *positions.iter().min().unwrap_or(&0);
        let causal = if needs_mask {
            Some(self.causal_mask(b, l, min_offset)?)
        } else {
            None
        };

        for (li, layer) in self.layers.iter_mut().enumerate() {
            let layer_caches = paged_caches.as_mut().map(|cs| {
                cs.iter_mut()
                    .map(|req| &mut req[li])
                    .collect::<Vec<_>>()
            });
            h = layer.forward_batch(&h, causal.as_ref(), positions, layer_caches)?;
        }
        self.norm.forward(&h)
    }
}

// ─── ModelForCausalLM ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    text: TextModel,
    lm_head: layers::LinearX,
    image_token_id: u32,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilderX) -> Result<Self> {
        Self::new_with_progress(cfg, vb, None)
    }

    pub fn new_with_progress(
        cfg: &Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self> {
        let text = TextModel::new_with_progress(&cfg.text_config, vb.clone(), progress)?;

        let lm_head = if cfg.tie_word_embeddings || cfg.text_config.tie_word_embeddings {
            layers::LinearX::Standard(candle_nn::Linear::new(
                text.embed_tokens.embeddings().clone(),
                None,
            ))
        } else {
            let lm_head_name = if vb.is_qvar_builder() { "output" } else { "lm_head" };
            layers::linear(cfg.text_config.hidden_size, cfg.text_config.vocab_size, vb.pp(lm_head_name))?
        };

        Ok(Self {
            text,
            lm_head,
            image_token_id: cfg.image_token_id,
        })
    }

    fn forward_text_only(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.text
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    fn merge_image_embeds(
        &self,
        token_ids: &Tensor,
        image_embeds: &[Tensor],
    ) -> Result<Tensor> {
        // Build the full embedding tensor replacing image_token_id positions with
        // the pre-encoded visual embeddings.
        let (b, l) = token_ids.dims2()?;
        let token_emb = self.text.embed_tokens.forward(token_ids)?; // [B, L, H]

        if image_embeds.is_empty() {
            return Ok(token_emb);
        }

        // For simplicity, replace image token positions with the image embeddings.
        // We only support a single batch element here (b==1).
        if b != 1 {
            candle_core::bail!("forward_modal with images only supports batch_size=1");
        }

        let ids_vec = token_ids.squeeze(0)?.to_vec1::<u32>()?;
        let img_id = self.image_token_id;
        let img_positions: Vec<usize> = ids_vec
            .iter()
            .enumerate()
            .filter(|&(_, id)| *id == img_id)
            .map(|(i, _)| i)
            .collect();

        if img_positions.is_empty() {
            return Ok(token_emb);
        }

        // Concatenate all image embeddings into one sequence of visual tokens.
        let vis_tokens = if image_embeds.len() == 1 {
            image_embeds[0].clone()
        } else {
            Tensor::cat(image_embeds, 0)?
        };

        // Replace image token positions with vision tokens (one-to-one mapping).
        // If there are more/fewer visual tokens than image_token_id placeholders, bail.
        let (num_vis, _) = vis_tokens.dims2()?;
        if num_vis != img_positions.len() {
            candle_core::bail!(
                "Number of visual tokens ({}) must equal number of image placeholders ({})",
                num_vis,
                img_positions.len()
            );
        }

        // Build output embedding by slicing and patching.
        let mut parts: Vec<Tensor> = Vec::new();
        let mut prev = 0usize;
        for (vi, &pos) in img_positions.iter().enumerate() {
            if pos > prev {
                parts.push(token_emb.narrow(1, prev, pos - prev)?);
            }
            let vis_tok = vis_tokens.narrow(0, vi, 1)?.unsqueeze(0)?; // [1, 1, vis_dim]
            parts.push(vis_tok);
            prev = pos + 1;
        }
        if prev < l {
            parts.push(token_emb.narrow(1, prev, l - prev)?);
        }

        Tensor::cat(&parts, 1)
    }
}

impl ModelImpl for ModelForCausalLM {
    fn name(&self) -> &'static str {
        "Qwen3.5"
    }

    fn num_layers(&self) -> usize {
        self.text.layers.len()
    }

    fn dtype(&self) -> DType {
        self.text.dtype
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_text_only(input, offset)
    }

    fn forward_modal(&mut self, input: ModelInput) -> Result<Tensor> {
        match input {
            ModelInput::Tokens { ids, offset } => self.forward_text_only(&ids, offset),
            ModelInput::Mixed { token_ids, audio_embeds: image_embeds, offset } => {
                let merged = self.merge_image_embeds(&token_ids, &image_embeds)?;
                let (_, l, _) = merged.dims3()?;
                self.text
                    .forward_embeds(&merged, offset)?
                    .narrow(1, l - 1, 1)?
                    .apply(&self.lm_head)
            }
            ModelInput::Embeddings { embeds, offset } => {
                let (_, l, _) = embeds.dims3()?;
                self.text
                    .forward_embeds(&embeds, offset)?
                    .narrow(1, l - 1, 1)?
                    .apply(&self.lm_head)
            }
        }
    }

    fn forward_batch(
        &mut self,
        input: &Tensor,
        positions: &[usize],
        paged_caches: Option<&mut [Vec<PagedKvCache>]>,
    ) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        if positions.len() != b {
            candle_core::bail!(
                "positions length {} must match batch size {}",
                positions.len(),
                b
            );
        }
        let hidden = self.text.forward_batch(input, positions, paged_caches)?;
        hidden.narrow(1, l - 1, 1)?.apply(&self.lm_head)
    }

    fn format_prompt(&self, prompt: &str, thinking: bool) -> String {
        let think_tag = if thinking { " /think" } else { " /no_think" };
        format!("<|im_start|>user\n{prompt}{think_tag}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn format_prompt_with_tools(
        &self,
        prompt: &str,
        tools: &[crate::backend::tools::ToolDefinition],
        thinking: bool,
    ) -> String {
        let think_tag = if thinking { " /think" } else { " /no_think" };
        let tools_json = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n\
             You may call one or more functions to assist with the user request.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\n\
             <tools>\n{tools_json}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\n\
             <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n\
             </tool_call><|im_end|>\n\
             <|im_start|>user\n{prompt}{think_tag}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    fn get_kv_cache_state(&self) -> Result<Vec<(Tensor, Tensor)>> {
        self.text.get_kv_cache_state()
    }

    fn set_kv_cache_state(&mut self, state: Vec<(Tensor, Tensor)>) -> Result<()> {
        self.text.set_kv_cache_state(state)
    }

    fn clear_kv_cache(&mut self) {
        self.text.clear_kv_cache();
    }
}
