pub mod attention;
pub mod granite_41;
pub mod kv_cache;
pub mod layers;
pub mod qwen3;

use candle_core::{DType, Result, Tensor};
pub use granite_41::ModelForCausalLM as Granite41ModelForCausalLM;
pub use qwen3::{Config, Model, ModelForCausalLM as Qwen35ModelForCausalLM};

/// Input to [`ModelImpl::forward_modal`] for multimodal inference.
///
/// Text-only models use the [`Tokens`] variant (via the default impl that
/// delegates to [`ModelImpl::forward`]).  Multimodal models (e.g. Qwen3-Omni)
/// implement the [`Embeddings`] or [`Mixed`] variants to accept pre-encoded
/// audio/image embeddings alongside token IDs.
pub enum ModelInput {
    /// Token IDs only — standard autoregressive path.
    Tokens { ids: Tensor, offset: usize },
    /// Pre-encoded embeddings only (e.g. audio frame embeddings).
    Embeddings { embeds: Tensor, offset: usize },
    /// Token IDs interleaved with one or more audio embedding tensors.
    /// The model merges them (via its embedding table + concat) internally.
    Mixed {
        token_ids: Tensor,
        audio_embeds: Vec<Tensor>,
        offset: usize,
    },
}

pub trait ModelImpl {
    fn name(&self) -> &'static str;

    /// Returns the number of transformer layers in this model.
    fn num_layers(&self) -> usize;

    /// Returns the number of KV heads (after GQA, if any).
    fn num_kv_heads(&self) -> usize;

    /// Returns the per-head dimension (hidden_size / num_attention_heads).
    fn head_dim(&self) -> usize;

    /// Returns the dtype used for KV cache tensors.
    fn dtype(&self) -> DType;

    /// Forward pass for single request (existing)
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>;

    /// Multimodal forward pass.  Text-only models use the default implementation
    /// which delegates [`ModelInput::Tokens`] to [`forward`] and returns an error
    /// for the embedding variants.  Multimodal models override this to handle
    /// audio/image embeddings.
    fn forward_modal(&mut self, input: ModelInput) -> Result<Tensor> {
        match input {
            ModelInput::Tokens { ids, offset } => self.forward(&ids, offset),
            ModelInput::Embeddings { .. } | ModelInput::Mixed { .. } => {
                Err(candle_core::Error::Msg(format!(
                    "Model '{}' does not support modal (audio/image) inputs",
                    self.name()
                )))
            }
        }
    }

    // /// Forward pass for batched requests with variable positions
    // ///
    // /// # Arguments
    // /// * `input` - Batched input tensor of shape [batch_size, seq_len]
    // /// * `positions` - Position offset for each request in the batch
    // /// * `block_tables` - Optional block tables for PagedKvCache, one per request
    // ///
    // /// # Returns
    // /// Tensor of shape [batch_size, seq_len, vocab_size] with logits for each position
    // fn forward_batch(&mut self, input: &Tensor, positions: &[usize]) -> Result<Tensor>;

    /// Forward pass for batched requests with variable positions
    ///
    /// # Arguments
    /// * `input` - Batched input tensor of shape [batch_size, seq_len]
    /// * `positions` - Position offset for each request in the batch
    /// * `paged_caches` - Optional paged KV caches per request per layer
    ///   Format: Vec<Vec<PagedKvCache>> where outer vec is per-request, inner vec is per-layer
    ///
    /// # Returns
    /// Tensor of shape [batch_size, seq_len, vocab_size] with logits for each position
    fn forward_batch(
        &mut self,
        input: &Tensor,
        positions: &[usize],
        paged_caches: Option<&mut [Vec<kv_cache::PagedKvCache>]>,
    ) -> Result<Tensor>;

    /// Fallback prompt formatter used only when no chat template is loaded.
    ///
    /// In normal operation the pipeline renders prompts via the Jinja2 chat
    /// template from `tokenizer_config.json` / `chat_template.jinja` and never
    /// reaches this method.  Override only when the model ships without a
    /// tokenizer config and needs a hardcoded offline fallback.
    fn format_prompt(&self, prompt: &str, _thinking: bool) -> String {
        prompt.to_string()
    }

    /// Format a prompt with tool definitions injected as a system message.
    ///
    /// The default falls back to [`format_prompt`] (no tool information).
    /// Models that support function calling override this to emit the
    /// model-specific system-prompt+tools template.
    ///
    /// `prompt` should be the raw user message, not yet wrapped in a chat
    /// template — this method applies the full template including tools.
    fn format_prompt_with_tools(
        &self,
        prompt: &str,
        tools: &[crate::backend::tools::ToolDefinition],
        thinking: bool,
    ) -> String {
        let _ = tools;
        self.format_prompt(prompt, thinking)
    }

    /// Extract current KV cache state from all layers
    /// Returns Vec<(K, V)> where index corresponds to layer index
    fn get_kv_cache_state(&self) -> Result<Vec<(Tensor, Tensor)>>;

    /// Restore KV cache state to all layers
    /// The state vector must match the number of layers
    fn set_kv_cache_state(&mut self, state: Vec<(Tensor, Tensor)>) -> Result<()>;

    /// Clear KV cache in all layers
    fn clear_kv_cache(&mut self);

    /// EOS token IDs from the model config.
    /// Used to detect end-of-sequence without relying on hardcoded token strings.
    /// Returns an empty vec when the config does not specify EOS IDs.
    fn eos_token_ids(&self) -> Vec<u32> {
        vec![]
    }
}
