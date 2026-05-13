pub mod attention;
pub mod layers;
pub mod qwen3;

use candle_core::{Result, Tensor};
pub use qwen3::{Config, Model, ModelForCausalLM as Qwen35ModelForCausalLM};

pub trait ModelImpl {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>;
}
