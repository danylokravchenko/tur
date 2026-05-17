use crate::backend::progress::ProgressReporter;
use crate::models::kv_cache::{KvCache, KvCacheImpl, PagedKvCache};
use crate::models::layers;
use crate::weights::VarBuilderX;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Activation;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

#[cfg(feature = "flash-attn")]
use candle_flash_attn;

#[cfg(not(feature = "flash-attn"))]
use crate::models::attention::{AttnMask, flash_attn};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub(crate) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Apply RoPE with variable positions per batch element (q, k shape: B x H x L x D)
    /// positions[i] is the starting position for batch element i
    pub(crate) fn apply_batch(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, _num_heads, seq_len, _head_dim) = q.dims4()?;

        if positions.len() != batch_size {
            candle_core::bail!(
                "positions length {} must match batch size {}",
                positions.len(),
                batch_size
            );
        }

        // For each batch element, apply RoPE with its specific position offset
        let mut q_embeds = Vec::with_capacity(batch_size);
        let mut k_embeds = Vec::with_capacity(batch_size);

        for (b_idx, &offset) in positions.iter().enumerate() {
            let q_b = q.narrow(0, b_idx, 1)?; // [1, H, L, D]
            let k_b = k.narrow(0, b_idx, 1)?; // [1, H, L, D]

            let cos = self.cos.narrow(0, offset, seq_len)?;
            let sin = self.sin.narrow(0, offset, seq_len)?;

            let q_embed = candle_nn::rotary_emb::rope(&q_b.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k_b.contiguous()?, &cos, &sin)?;

            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }

        // Concatenate along batch dimension
        let q_result = Tensor::cat(&q_embeds, 0)?;
        let k_result = Tensor::cat(&k_embeds, 0)?;

        Ok((q_result, k_result))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3MLP {
    gate_proj: layers::LinearX,
    up_proj: layers::LinearX,
    down_proj: layers::LinearX,
    act_fn: Activation,
}

impl Qwen3MLP {
    pub(crate) fn new(cfg: &Config, vb: VarBuilderX) -> Result<Self> {
        // GGUF models use "ffn_gate", "ffn_up", "ffn_down" instead of "gate_proj", "up_proj", "down_proj"
        let (gate_name, up_name, down_name) = if vb.is_qvar_builder() {
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

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3Attention {
    // projections
    q_proj: layers::LinearX,
    k_proj: layers::LinearX,
    v_proj: layers::LinearX,
    o_proj: layers::LinearX,
    // norms
    q_norm: layers::norm::NormX,
    k_norm: layers::norm::NormX,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,
}

impl Qwen3Attention {
    pub(crate) fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilderX,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        // GGUF models use "attn_q", "attn_k", "attn_v", "attn_output" instead of "*_proj"
        let (q_name, k_name, v_name, o_name) = if vb.is_qvar_builder() {
            ("attn_q", "attn_k", "attn_v", "attn_output")
        } else {
            ("q_proj", "k_proj", "v_proj", "o_proj")
        };

        let q_proj = layers::linear(cfg.hidden_size, num_heads * head_dim, vb.pp(q_name))?;
        let k_proj = layers::linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp(k_name))?;
        let v_proj = layers::linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp(v_name))?;
        let o_proj = layers::linear(num_heads * head_dim, cfg.hidden_size, vb.pp(o_name))?;

        // GGUF models use "attn_q_norm" and "attn_k_norm"
        let (q_norm_name, k_norm_name) = if vb.is_qvar_builder() {
            ("attn_q_norm", "attn_k_norm")
        } else {
            ("q_norm", "k_norm")
        };

        let q_norm = layers::norm::rms_norm(
            head_dim,
            cfg.rms_norm_eps,
            vb.pp(q_norm_name),
            vb.dtype(),
            false,
        )?;
        let k_norm = layers::norm::rms_norm(
            head_dim,
            cfg.rms_norm_eps,
            vb.pp(k_norm_name),
            vb.dtype(),
            false,
        )?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        // dim=2 because we concatenate along the sequence dimension
        // For tensors of shape [batch, heads, seq, head_dim]
        let kv_cache = KvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Proj
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head RMSNorm — RmsNorm reduces over the last dim, so (B,H,L,D) works directly
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // 6. Attention dispatch: auto-select best available path
        //    - CPU (no flash-attn feature): fused CPU flash kernel
        //    - GPU (flash-attn feature):    CUDA flash attention
        //    - Fallback:                    standard matmul attention
        let on_cpu = x.device().is_cpu();

        #[cfg(not(feature = "flash-attn"))]
        if on_cpu {
            return self.forward_cpu_flash_attn(&q, &k, &v, offset, b, l);
        }
        #[cfg(feature = "flash-attn")]
        if !on_cpu {
            return self.forward_flash_attn(&q, &k, &v, offset, b, l);
        }

        self.forward_standard_attn(&q, &k, &v, attn_mask, b, l)
    }

    /// Batched forward pass with variable positions per request
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, hidden_size]
    /// * `attn_mask` - Optional attention mask
    /// * `positions` - Position offset for each batch element
    /// * `paged_caches` - Optional paged KV caches (one per request in batch)
    pub(crate) fn forward_batch(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        positions: &[usize],
        paged_caches: Option<Vec<&mut PagedKvCache>>,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        if positions.len() != b {
            candle_core::bail!(
                "positions length {} must match batch size {}",
                positions.len(),
                b
            );
        }

        // 1. Proj
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head RMSNorm — RmsNorm reduces over the last dim, so (B,H,L,D) works directly
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // 4. RoPE with variable positions
        let (q, k) = self.rotary_emb.apply_batch(&q, &k, positions)?;

        // 5 & 6. KV cache accumulation + attention dispatch.
        //
        // With paged caches each request has its own history of a different length, so we
        // cannot concatenate their K/V tensors along the batch dimension before running
        // attention.  Instead, compute attention independently per request and cat the
        // outputs (which all share the query sequence length `l`).
        if let Some(mut caches) = paged_caches {
            let mut output_parts = Vec::with_capacity(b);

            for (batch_idx, cache) in caches.iter_mut().enumerate() {
                let q_req = q.narrow(0, batch_idx, 1)?; // [1, H,    l,         D]
                let k_req = k.narrow(0, batch_idx, 1)?; // [1, kv_H, l,         D]
                let v_req = v.narrow(0, batch_idx, 1)?; // [1, kv_H, l,         D]

                let (k_full, v_full) = cache.append(&k_req, &v_req)?; // [1, kv_H, total, D]
                let offset = positions[batch_idx];

                let out =
                    self.single_request_attn(&q_req, &k_full, &v_full, attn_mask, offset, l)?;
                output_parts.push(out); // [1, l, hidden_size]
            }

            // All outputs share the same shape → safe to cat along the batch dim.
            return Tensor::cat(&output_parts, 0); // [B, l, hidden_size]
        }

        // Without paged caches: accumulate into the shared (single-request) KV cache.
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let on_cpu = x.device().is_cpu();

        #[cfg(not(feature = "flash-attn"))]
        if on_cpu {
            return self.forward_cpu_flash_attn(
                &q,
                &k,
                &v,
                *positions.iter().min().unwrap_or(&0),
                b,
                l,
            );
        }
        #[cfg(feature = "flash-attn")]
        if !on_cpu {
            return self.forward_flash_attn(
                &q,
                &k,
                &v,
                *positions.iter().min().unwrap_or(&0),
                b,
                l,
            );
        }

        self.forward_standard_attn(&q, &k, &v, attn_mask, b, l)
    }

    /// GPU flash attention path (requires flash-attn feature)
    #[cfg(feature = "flash-attn")]
    fn forward_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _offset: usize,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // Flash attention expects (B, S, H, D) format
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let causal = l > 1;
        let ctx = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?;

        // Output: (B, S, H, D) -> (B, L, hidden_size)
        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }

    /// CPU flash attention - optimized fused kernel for CPU
    ///
    /// The `flash_attn` dispatcher in candle-nn automatically selects:
    /// - B=1: single-batch optimized kernels (direct slice access)
    /// - B>1: packed varlen path (avoids batch-dim stride overhead)
    #[cfg(not(feature = "flash-attn"))]
    fn forward_cpu_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        offset: usize,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // CPU flash attention expects (B, S, H, D) format
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let ctx = match q.dtype() {
            DType::F32 => flash_attn::<f32>(
                &q,
                &k,
                &v,
                scale,
                AttnMask::causal_with_offset(offset),
                None,
                None,
            )?,
            DType::F64 => flash_attn::<f64>(
                &q,
                &k,
                &v,
                scale,
                AttnMask::causal_with_offset(offset),
                None,
                None,
            )?,
            DType::BF16 => {
                let q_f32 = q.to_dtype(DType::F32)?;
                let k_f32 = k.to_dtype(DType::F32)?;
                let v_f32 = v.to_dtype(DType::F32)?;
                let ctx_f32 = flash_attn::<f32>(
                    &q_f32,
                    &k_f32,
                    &v_f32,
                    scale,
                    AttnMask::causal_with_offset(offset),
                    None,
                    None,
                )?;
                ctx_f32.to_dtype(DType::BF16)?
            }
            dtype => candle_core::bail!("Unsupported dtype for CPU flash attention: {:?}", dtype),
        };

        // Output from CPU flash attention is (B, H, S, D), transpose to (B, S, H, D)
        let ctx = ctx.transpose(1, 2)?;

        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }

    /// Standard matmul-based attention (works on any device)
    fn forward_standard_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // GQA repeat_kv
        let k = repeat_kv(k.clone(), self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), self.num_kv_groups)?.contiguous()?;

        // Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// Compute attention for a single request (b=1).  Used by `forward_batch` when paged
    /// caches are active so each request is processed with its own variable-length KV.
    fn single_request_attn(
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
            return self.forward_cpu_flash_attn(q, k, v, offset, 1, l);
        }
        #[cfg(feature = "flash-attn")]
        if !on_cpu {
            return self.forward_flash_attn(q, k, v, offset, 1, l);
        }

        self.forward_standard_attn(q, k, v, attn_mask, 1, l)
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Get current KV cache state for this attention layer
    pub(crate) fn get_kv_state(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.get_state()
    }

    /// Restore KV cache state for this attention layer
    pub(crate) fn set_kv_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        self.kv_cache.set_state(k, v)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: layers::norm::NormX,
    ln2: layers::norm::NormX,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilderX) -> Result<Self> {
        // GGUF models don't have "self_attn" or "mlp" prefixes - components are directly under the block
        // SafeTensors: layers.0.self_attn.q_proj.weight
        // GGUF: blk.0.attn_q.weight
        let self_attn = if vb.is_qvar_builder() {
            Qwen3Attention::new(cfg, rotary, vb.clone())?
        } else {
            Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?
        };

        // SafeTensors: blk.0.mlp.gate_proj.weight
        // GGUF: blk.0.ffn_gate.weight
        let mlp = if vb.is_qvar_builder() {
            Qwen3MLP::new(cfg, vb.clone())?
        } else {
            Qwen3MLP::new(cfg, vb.pp("mlp"))?
        };

        // GGUF models use "attn_norm" and "ffn_norm" instead of "input_layernorm" and "post_attention_layernorm"
        let (ln1_name, ln2_name) = if vb.is_qvar_builder() {
            ("attn_norm", "ffn_norm")
        } else {
            ("input_layernorm", "post_attention_layernorm")
        };

        let ln1 = layers::norm::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp(ln1_name),
            vb.dtype(),
            false,
        )?;
        let ln2 = layers::norm::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp(ln2_name),
            vb.dtype(),
            false,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
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
        let h = self
            .self_attn
            .forward_batch(&h, mask, positions, paged_caches)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }

    fn get_kv_state(&self) -> Option<(Tensor, Tensor)> {
        self.self_attn.get_kv_state()
    }

    fn set_kv_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        self.self_attn.set_kv_state(k, v)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: layers::norm::NormX,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilderX) -> Result<Self> {
        Self::new_with_progress(cfg, vb, None)
    }

    pub fn new_with_progress(
        cfg: &Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self> {
        // GGUF models use "token_embd" instead of "model.embed_tokens"
        let embed_path = if vb.is_qvar_builder() {
            "token_embd"
        } else {
            "model.embed_tokens"
        };
        let (embed_tokens, _) = layers::embedding(
            Some(cfg.vocab_size),
            cfg.hidden_size,
            vb.pp(embed_path),
            vb.dtype(),
        )?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, &vb.device())?);

        // Initialize progress bar for layer loading
        if let Some(p) = progress {
            p.init_loading(cfg.num_hidden_layers);
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        // GGUF models use "blk" instead of "model.layers"
        let layers_path = if vb.is_qvar_builder() {
            "blk"
        } else {
            "model.layers"
        };
        let vb_l = vb.pp(layers_path);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                vb_l.pp(&i.to_string()),
            )?);

            // Report progress after each layer is loaded
            if let Some(p) = progress {
                p.inc_loading();
            }
        }

        // Finish loading progress
        if let Some(p) = progress {
            p.finish_loading();
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm: {
                // GGUF models use "output_norm" instead of "model.norm"
                let norm_path = if vb.is_qvar_builder() {
                    "output_norm"
                } else {
                    "model.norm"
                };
                layers::norm::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp(norm_path),
                    vb.dtype(),
                    false,
                )?
            },
            device: vb.device(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    /// Get KV cache state from all layers
    fn get_kv_cache_state(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            if let Some((k, v)) = layer.get_kv_state() {
                states.push((k, v));
            } else {
                // If any layer has no state, return empty vec
                return Ok(Vec::new());
            }
        }
        Ok(states)
    }

    /// Restore KV cache state to all layers
    fn set_kv_cache_state(&mut self, states: Vec<(Tensor, Tensor)>) -> Result<()> {
        if states.len() != self.layers.len() {
            candle_core::bail!(
                "KV cache state length mismatch: expected {} layers, got {}",
                self.layers.len(),
                states.len()
            );
        }
        for (layer, (k, v)) in self.layers.iter_mut().zip(states) {
            layer.set_kv_state(k, v)?;
        }
        Ok(())
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok { 0. } else { minf }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        // Build causal mask only for the standard attention fallback path.
        // Both CPU flash and GPU flash handle masking internally.
        #[cfg(not(feature = "flash-attn"))]
        let needs_mask = !self.device.is_cpu() && l > 1;
        #[cfg(feature = "flash-attn")]
        let needs_mask = self.device.is_cpu() && l > 1;
        let causal = if needs_mask {
            Some(self.causal_mask(b, l, offset, None)?)
        } else {
            None
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }

    /// Batched forward pass with variable positions per request
    ///
    /// # Arguments
    /// * `input` - Batched input tensor of shape [batch_size, seq_len]
    /// * `positions` - Position offset for each batch element
    ///
    /// # Returns
    /// Hidden states tensor of shape [batch_size, seq_len, hidden_size]
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

        // Build causal mask only for the standard attention fallback path.
        // Both CPU flash and GPU flash handle masking internally.
        #[cfg(not(feature = "flash-attn"))]
        let needs_mask = !self.device.is_cpu() && l > 1;
        #[cfg(feature = "flash-attn")]
        let needs_mask = self.device.is_cpu() && l > 1;

        // For batched execution with variable positions, use minimum offset for mask
        let min_offset = *positions.iter().min().unwrap_or(&0);
        let causal = if needs_mask {
            Some(self.causal_mask(b, l, min_offset, None)?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_caches = paged_caches.as_mut().map(|caches| {
                caches
                    .iter_mut()
                    .map(|req_caches| &mut req_caches[layer_idx])
                    .collect::<Vec<_>>()
            });
            h = layer.forward_batch(&h, causal.as_ref(), positions, layer_caches)?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: layers::LinearX,
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
        let base = Model::new_with_progress(cfg, vb.clone(), progress)?;
        let lm_head = if cfg.tie_word_embeddings {
            layers::LinearX::Standard(candle_nn::Linear::new(
                base.embed_tokens.embeddings().clone(),
                None,
            ))
        } else {
            // GGUF models use "output" instead of "lm_head"
            let lm_head_name = if vb.is_qvar_builder() {
                "output"
            } else {
                "lm_head"
            };
            layers::linear(cfg.hidden_size, cfg.vocab_size, vb.pp(lm_head_name))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

impl super::ModelImpl for ModelForCausalLM {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
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

        // Get hidden states for all positions
        let hidden_states = self.base.forward_batch(input, positions, paged_caches)?;

        // For each batch element, extract the last token's logits
        // hidden_states shape: [batch_size, seq_len, hidden_size]
        let last_hidden = hidden_states.narrow(1, l - 1, 1)?;

        // Apply language model head
        last_hidden.apply(&self.lm_head)
    }

    #[inline]
    fn format_prompt(prompt: &str, thinking: bool) -> String {
        let think_tag = if thinking { " /think" } else { " /no_think" };
        format!("<|im_start|>user\n{prompt}{think_tag}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn format_prompt_with_tools(
        prompt: &str,
        tools: &[crate::backend::tools::ToolDefinition],
        thinking: bool,
    ) -> String {
        let think_tag = if thinking { " /think" } else { " /no_think" };

        // Serialise each tool as a JSON object with the "function" wrapper
        // expected by the Qwen3 tool-calling template.
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
            "<|im_start|>system\n\
             You are a helpful assistant.\n\n\
             # Tools\n\n\
             You may call one or more functions to assist with the user request.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\n\
             <tools>\n\
             {tools_json}\n\
             </tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\n\
             <tool_call>\n\
             {{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n\
             </tool_call><|im_end|>\n\
             <|im_start|>user\n\
             {prompt}{think_tag}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    fn get_kv_cache_state(&self) -> Result<Vec<(Tensor, Tensor)>> {
        self.base.get_kv_cache_state()
    }

    fn set_kv_cache_state(&mut self, state: Vec<(Tensor, Tensor)>) -> Result<()> {
        self.base.set_kv_cache_state(state)
    }

    fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.base.layers.len()
    }

    fn name(&self) -> &'static str {
        "Qwen3"
    }

    fn dtype(&self) -> candle_core::DType {
        self.base.dtype
    }
}
