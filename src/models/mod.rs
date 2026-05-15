pub mod attention;
pub mod kv_cache;
pub mod layers;
pub mod qwen3;

use candle_core::{Result, Tensor};
pub use qwen3::{Config, Model, ModelForCausalLM as Qwen35ModelForCausalLM};

pub trait ModelImpl {
    fn name(&self) -> &'static str;
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>;
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
