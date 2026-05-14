use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tracing::info;
use uuid::Uuid;

use crate::{
    ProgressReporter, Result, TurError,
    backend::tokenizer::{LogitsSampler, TokenOutputStream},
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

/// Detailed statistics separating prefill and decode phases
#[derive(Debug, Clone)]
pub struct DetailedGenerationStats {
    /// Time to encode prompt and perform first forward pass (prefill)
    pub prefill_ms: f64,
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Tokens generated per second during steady-state decode
    pub decode_tokens_per_sec: f64,
    /// Average latency per token during decode (ms)
    pub latency_per_token_ms: f64,
    /// Total tokens generated (excluding prompt)
    pub generated_tokens: usize,
    /// Total time for decode phase
    pub decode_ms: f64,
}

impl DetailedGenerationStats {
    pub fn report(&self, name: &str) {
        println!("\n=== Generation Stats: {} ===", name);
        println!("Prefill Phase:");
        println!("  Prompt tokens: {}", self.prompt_tokens);
        println!("  Prefill time: {:.2} ms", self.prefill_ms);
        println!(
            "  Prefill throughput: {:.2} tokens/sec",
            self.prompt_tokens as f64 / (self.prefill_ms / 1000.0)
        );
        println!("\nDecode Phase:");
        println!("  Generated tokens: {}", self.generated_tokens);
        println!("  Decode time: {:.2} ms", self.decode_ms);
        println!(
            "  Decode throughput: {:.2} tokens/sec",
            self.decode_tokens_per_sec
        );
        println!("  Latency per token: {:.2} ms", self.latency_per_token_ms);
        println!("========================================\n");
    }
}

/// Core inference engine without output handling
/// This is the reusable component for benchmarks and tests
pub struct InferenceEngine<T: ModelImpl> {
    model: T,
    device: Device,
    sampler: Box<dyn LogitsSampler>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

/// Builder for InferenceEngine
pub struct InferenceEngineBuilder<T: ModelImpl> {
    model: T,
    device: Device,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<T: ModelImpl> InferenceEngineBuilder<T> {
    pub fn new(model: T, device: Device) -> Self {
        Self {
            model,
            device,
            seed: 299_792_458, // Default seed
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
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

    pub fn build(self) -> InferenceEngine<T> {
        let sampler: Box<dyn LogitsSampler> =
            Box::new(LogitsProcessor::new(self.seed, self.temp, self.top_p));
        InferenceEngine {
            model: self.model,
            device: self.device,
            sampler,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
        }
    }
}

impl<T: ModelImpl> InferenceEngine<T> {
    /// Create a new builder for InferenceEngine
    pub fn builder(model: T, device: Device) -> InferenceEngineBuilder<T> {
        InferenceEngineBuilder::new(model, device)
    }

    /// Create with a custom sampler (for guidance, etc.)
    pub fn with_sampler(
        model: T,
        device: Device,
        sampler: Box<dyn LogitsSampler>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Self {
        Self {
            model,
            device,
            sampler,
            repeat_penalty,
            repeat_last_n,
        }
    }
    /// Get EOS tokens from tokenizer
    pub fn get_eos_tokens(tokenizer: &TokenOutputStream) -> Result<(u32, u32)> {
        let eos_token = tokenizer
            .get_token("<|endoftext|>")
            .ok_or_else(|| TurError::Tokenizer("cannot find <|endoftext|> token".to_string()))?;
        let eos_token2 = tokenizer
            .get_token("<|im_end|>")
            .ok_or_else(|| TurError::Tokenizer("cannot find <|im_end|> token".to_string()))?;
        Ok((eos_token, eos_token2))
    }

    /// Apply repeat penalty and optional guidance constraints to logits
    fn process_logits(&mut self, logits: Tensor, tokens: &[u32]) -> Result<Tensor> {
        // Apply repeat penalty
        let logits = if self.repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        // Apply guidance constraints if any
        if let Some(constrained) = self.sampler.apply_constraints(&logits, tokens)? {
            Ok(constrained)
        } else {
            Ok(logits)
        }
    }

    /// Perform prefill: encode full prompt and get first token
    /// Returns (first_token, prefill_duration)
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(u32, std::time::Duration)> {
        let start = std::time::Instant::now();

        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = self.process_logits(logits, tokens)?;
        let next_token = self.sampler.sample(&logits)?;

        Ok((next_token, start.elapsed()))
    }

    /// Perform single decode step with KV cache reuse
    /// Returns next token
    pub fn decode_step(&mut self, tokens: &[u32], start_pos: usize) -> Result<u32> {
        let context_size = 1;
        let pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[pos..];

        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = self.process_logits(logits, tokens)?;

        self.sampler.sample(&logits)
    }

    /// Run separated inference: prefill + decode with detailed stats
    pub fn run_separated(
        &mut self,
        tokens: &[u32],
        sample_len: usize,
        eos_tokens: (u32, u32),
    ) -> Result<DetailedGenerationStats> {
        let prompt_tokens = tokens.len();
        let mut token_vec = tokens.to_vec();

        // Prefill phase
        let (first_token, prefill_duration) = self.prefill(tokens)?;
        token_vec.push(first_token);
        let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;

        // Decode phase
        let mut generated_tokens = 1;
        let decode_start = std::time::Instant::now();

        for _ in 1..sample_len {
            let start_pos = token_vec.len() - 1;
            let next_token = self.decode_step(&token_vec, start_pos)?;
            token_vec.push(next_token);
            generated_tokens += 1;

            if next_token == eos_tokens.0 || next_token == eos_tokens.1 {
                break;
            }
        }

        let decode_elapsed = decode_start.elapsed();
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let decode_tokens_per_sec = generated_tokens as f64 / decode_elapsed.as_secs_f64();
        let latency_per_token_ms = decode_ms / generated_tokens as f64;

        Ok(DetailedGenerationStats {
            prefill_ms,
            prompt_tokens,
            decode_tokens_per_sec,
            latency_per_token_ms,
            generated_tokens,
            decode_ms,
        })
    }

    /// Access the underlying model
    pub fn model(&self) -> &T {
        &self.model
    }

    /// Access the underlying model mutably
    pub fn model_mut(&mut self) -> &mut T {
        &mut self.model
    }
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
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(request.prompt.as_str(), true)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();

        // Initialize generation progress bar
        if let Some(ref progress) = self.progress {
            progress.init_generation(sample_len);
        }

        let mut generated_tokens = 0usize;
        let (eos_token, eos_token2) = InferenceEngine::<T>::get_eos_tokens(&self.tokenizer)?;

        let start_gen = std::time::Instant::now();

        // Prefill phase
        let (first_token, _) = self.engine.prefill(&tokens)?;
        tokens.push(first_token);
        generated_tokens += 1;

        if let Some(ref progress) = self.progress {
            progress.inc_generation();
        }

        if first_token != eos_token
            && first_token != eos_token2
            && let Some(t) = self.tokenizer.next_token(first_token)?
        {
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
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
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
