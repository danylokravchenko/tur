use candle_core::Device;
use tokenizers::Tokenizer;
use tracing::{info, trace};
use uuid::Uuid;

use crate::{
    ProgressReporter, Result, TurError,
    backend::{engine::InferenceEngine, tokenizer::TokenOutputStream},
    models::ModelImpl,
};

/// Request struct for text generation, suitable for scheduling and prefix-caching
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub id: Uuid,
    pub prompt: String,
    pub sample_len: usize,
}

impl GenerationRequest {
    /// Create a new request with auto-generated UUID
    pub fn new(prompt: String, sample_len: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            prompt,
            sample_len,
        }
    }
}

/// Statistics for generation runs
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub elapsed: std::time::Duration,
}

/// High-level text generation pipeline with output handling
pub struct TextGeneration<T: ModelImpl> {
    engine: InferenceEngine<T>,
    tokenizer: TokenOutputStream,
    progress: Option<ProgressReporter>,
    emit_output: bool,
}

/// Builder for TextGeneration
pub struct TextGenerationBuilder<T: ModelImpl> {
    model: T,
    tokenizer: Tokenizer,
    device: Device,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    progress: Option<ProgressReporter>,
    emit_output: bool,
}

impl<T: ModelImpl> TextGenerationBuilder<T> {
    pub fn new(model: T, tokenizer: Tokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
            seed: 299_792_458,
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            progress: None,
            emit_output: true,
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temp = Some(temp);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    pub fn repeat_last_n(mut self, n: usize) -> Self {
        self.repeat_last_n = n;
        self
    }

    pub fn progress(mut self, progress: ProgressReporter) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn emit_output(mut self, emit: bool) -> Self {
        self.emit_output = emit;
        self
    }

    pub fn build(self) -> TextGeneration<T> {
        let mut engine_builder = InferenceEngine::builder(self.model, self.device)
            .seed(self.seed)
            .repeat_penalty(self.repeat_penalty)
            .repeat_last_n(self.repeat_last_n);

        if let Some(temp) = self.temp {
            engine_builder = engine_builder.temperature(temp);
        }
        if let Some(top_p) = self.top_p {
            engine_builder = engine_builder.top_p(top_p);
        }

        let engine = engine_builder.build();
        TextGeneration {
            engine,
            tokenizer: TokenOutputStream::new(self.tokenizer),
            progress: self.progress,
            emit_output: self.emit_output,
        }
    }
}

impl<T: ModelImpl> TextGeneration<T> {
    /// Create a new builder for TextGeneration
    pub fn builder(model: T, tokenizer: Tokenizer, device: Device) -> TextGenerationBuilder<T> {
        TextGenerationBuilder::new(model, tokenizer, device)
    }

    /// Create from an existing inference engine
    pub fn from_engine(
        engine: InferenceEngine<T>,
        tokenizer: Tokenizer,
        progress: Option<ProgressReporter>,
    ) -> Self {
        Self {
            engine,
            tokenizer: TokenOutputStream::new(tokenizer),
            progress,
            emit_output: true,
        }
    }

    /// Access the underlying inference engine
    pub fn engine(&self) -> &InferenceEngine<T> {
        &self.engine
    }

    /// Access the underlying inference engine mutably
    pub fn engine_mut(&mut self) -> &mut InferenceEngine<T> {
        &mut self.engine
    }

    pub fn run(&mut self, request: &GenerationRequest) -> Result<GenerationStats> {
        let sample_len = request.sample_len;
        self.tokenizer.clear();
        trace!(
            request_id = %request.id,
            prompt_chars = request.prompt.chars().count(),
            sample_len,
            "starting text generation request",
        );
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(request.prompt.as_str(), true)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        trace!(
            request_id = %request.id,
            prompt_tokens = tokens.len(),
            "prompt encoded into tokens",
        );

        // Initialize generation progress bar
        if let Some(ref progress) = self.progress {
            progress.init_generation(sample_len);
        }

        let mut generated_tokens = 0usize;
        let (eos_token, eos_token2) = InferenceEngine::<T>::get_eos_tokens(&self.tokenizer)?;

        let start_gen = std::time::Instant::now();

        // Prefill phase
        let (first_token, _, cache_hit, cached_tokens) = self.engine.prefill(&tokens)?;
        trace!(
            request_id = %request.id,
            first_token,
            cache_hit,
            cached_tokens,
            "engine prefetched prompt and produced first token",
        );
        tokens.push(first_token);
        generated_tokens += 1;

        if let Some(ref progress) = self.progress {
            progress.inc_generation();
        }

        if first_token != eos_token
            && first_token != eos_token2
            && let Some(t) = self.tokenizer.next_token(first_token)?
        {
            trace!(
                request_id = %request.id,
                text_chars = t.chars().count(),
                "decoded first token into text chunk",
            );
            if let Some(ref progress) = self.progress {
                progress.print(&t);
            } else if self.emit_output {
                use std::io::Write;
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        // Decode phase
        for _ in 1..sample_len {
            if tokens[tokens.len() - 1] == eos_token || tokens[tokens.len() - 1] == eos_token2 {
                break;
            }

            let start_pos = tokens.len() - 1;
            let next_token = self.engine.decode_step(&tokens, start_pos)?;
            tokens.push(next_token);
            generated_tokens += 1;

            // Update progress bar
            if let Some(ref progress) = self.progress {
                progress.inc_generation();
            }

            if next_token == eos_token || next_token == eos_token2 {
                trace!(
                    request_id = %request.id,
                    next_token,
                    generated_tokens,
                    "decode loop reached EOS token",
                );
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                trace!(
                    request_id = %request.id,
                    next_token,
                    text_chars = t.chars().count(),
                    "decoded token into text chunk",
                );
                if let Some(ref progress) = self.progress {
                    progress.print(&t);
                } else if self.emit_output {
                    use std::io::Write;
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self
            .tokenizer
            .decode_rest()
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
        {
            trace!(
                request_id = %request.id,
                text_chars = rest.chars().count(),
                "flushed remaining decoded text",
            );
            if let Some(ref progress) = self.progress {
                progress.print(&rest);
            } else if self.emit_output {
                print!("{rest}");
            }
        }

        // Flush any remaining buffered text
        if let Some(ref progress) = self.progress {
            progress.flush_text();
        } else if self.emit_output {
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        trace!(
            request_id = %request.id,
            generated_tokens,
            elapsed_ms = dt.as_secs_f64() * 1000.0,
            "generation request completed",
        );

        // Finish generation progress bar
        if let Some(ref progress) = self.progress {
            progress.finish_generation(generated_tokens, dt.as_secs_f64());
        } else {
            info!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }
        Ok(GenerationStats {
            generated_tokens,
            elapsed: dt,
        })
    }

    pub fn format_prompt(&self, prompt: &str, thinking: bool) -> String {
        T::format_prompt(prompt, thinking)
    }

    pub fn with_output_enabled(mut self, emit_output: bool) -> Self {
        self.emit_output = emit_output;
        self
    }
}
