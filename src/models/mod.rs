pub mod attention;
pub mod kv_cache;
pub mod layers;
pub mod qwen3;

use candle_core::{Result, Tensor};
pub use qwen3::{Config, Model, ModelForCausalLM as Qwen35ModelForCausalLM};

pub trait ModelImpl {
    fn name(&self) -> &'static str;

    /// Forward pass for single request (existing)
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>;

    /// Forward pass for batched requests with variable positions
    ///
    /// # Arguments
    /// * `input` - Batched input tensor of shape [batch_size, seq_len]
    /// * `positions` - Position offset for each request in the batch
    ///
    /// # Returns
    /// Tensor of shape [batch_size, seq_len, vocab_size] with logits for each position
    fn forward_batch(&mut self, input: &Tensor, positions: &[usize]) -> Result<Tensor>;

    fn format_prompt(prompt: &str, thinking: bool) -> String;

    /// Extract current KV cache state from all layers
    /// Returns Vec<(K, V)> where index corresponds to layer index
    fn get_kv_cache_state(&self) -> Result<Vec<(Tensor, Tensor)>>;

    /// Restore KV cache state to all layers
    /// The state vector must match the number of layers
    fn set_kv_cache_state(&mut self, state: Vec<(Tensor, Tensor)>) -> Result<()>;

    /// Clear KV cache in all layers
    fn clear_kv_cache(&mut self);
}
