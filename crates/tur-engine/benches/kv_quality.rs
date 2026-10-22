#![forbid(unsafe_code)]
//! K/V storage quality on a real checkpoint: teacher-forced next-token prediction over a
//! long held-out text, every K/V format against F32 K/V on identical weights. Reports,
//! by position bucket, the KL divergence from the F32-K/V distribution, top-1 agreement
//! and the mean negative log-likelihood. Prints one JSON document.
//!
//! `TUR_QWEN35_0_8B=<snapshot> TUR_EVAL_TEXT=<file> cargo bench --offline -p tur-engine
//! --bench kv_quality > out.json`; `TUR_EVAL_TOKENS` (default 8192) bounds the context and
//! `TUR_EVAL_KV` (default `f16,q8,q8_64,q8_256`) selects formats.

use candle_core::{D, DType, Device, Tensor};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout, invalid},
};
use tur_engine::hybrid::{
    Boundaries, HybridModel, ModelState, PackedForward, PoolConfig, SequenceForward, StatePool,
};
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

const CHUNK: usize = 256;
const BUCKETS: [usize; 5] = [0, 1024, 2048, 4096, 8192];

fn env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| TurError::Other(format!("set {name}")))
}

fn log_softmax(logits: &Tensor) -> Result<Tensor> {
    let shifted = logits.broadcast_sub(&logits.max_keepdim(D::Minus1)?)?;
    let total = shifted.exp()?.sum_keepdim(D::Minus1)?.log()?;
    Ok(shifted.broadcast_sub(&total)?)
}

/// One model with its own pool and committed state, advanced a chunk at a time.
struct Runner {
    model: Qwen35,
    pool: StatePool,
    state: ModelState,
}

impl Runner {
    fn new(dir: &Path, device: &Device, format: KvFormat, tokens: usize) -> Result<Self> {
        let model = Qwen35::load(
            dir,
            &LoadOptions {
                revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
                format: WeightFormat::Q8_0,
                device: device.clone(),
                execution: Execution::Fused,
                kv: KvLayout::new(16, format),
            },
        )?;
        let pool = StatePool::new(
            Arc::new(model.description()),
            device,
            PoolConfig {
                pages: model.description().pages_for(tokens) + 1,
                recurrent_slots: 2,
            },
        )?;
        let state = ModelState::new(&pool);
        Ok(Self { model, pool, state })
    }

    /// Log-probabilities `[chunk, vocab]` at every position of `chunk`, then commit it.
    fn advance(&mut self, chunk: &[u32]) -> Result<Tensor> {
        let mut reservation = self.state.reserve(chunk.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let start = self.state.position();
        let positions: Vec<usize> = (start..start + chunk.len()).collect();
        let rows: Vec<usize> = (0..chunk.len()).collect();
        let step = self
            .model
            .forward(&PackedForward {
                tokens: chunk,
                positions: &positions,
                sequences: &[SequenceForward {
                    tokens: 0..chunk.len(),
                    state: &binding,
                    logits: &rows,
                }],
                pool: &self.pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing step"))??;
        let log_probs = log_softmax(&step.logits)?;
        self.state = reservation.commit(chunk.len())?;
        drop(reservation);
        self.model.device().synchronize()?;
        self.pool.reclaim();
        Ok(log_probs)
    }
}

#[derive(Default, Clone)]
struct Bucket {
    positions: usize,
    kl: f64,
    kl_max: f64,
    agree: usize,
    nll: f64,
    reference_nll: f64,
}

impl Bucket {
    fn json(&self, from: usize, to: usize) -> Value {
        let n = self.positions.max(1) as f64;
        json!({
            "positions": [from, to], "count": self.positions,
            "mean_kl": self.kl / n, "max_kl": self.kl_max,
            "top1_agreement": self.agree as f64 / n,
            "mean_nll": self.nll / n, "reference_mean_nll": self.reference_nll / n,
        })
    }
}

fn main() -> Result<()> {
    let dir = env_path("TUR_QWEN35_0_8B")?;
    let text_path = env_path("TUR_EVAL_TEXT")?;
    let text = std::fs::read_to_string(&text_path)?;
    let limit: usize = std::env::var("TUR_EVAL_TOKENS")
        .unwrap_or_else(|_| "8192".into())
        .parse()
        .map_err(|_| invalid("TUR_EVAL_TOKENS"))?;
    let formats = std::env::var("TUR_EVAL_KV")
        .unwrap_or_else(|_| "f16,q8,q8_64,q8_256".into())
        .split(',')
        .map(|f| f.parse::<KvFormat>().map_err(TurError::Other))
        .collect::<Result<Vec<_>>>()?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let mut tokens = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| TurError::Tokenizer(e.to_string()))?
        .get_ids()
        .to_vec();
    tokens.truncate(limit + 1);
    let evaluated = tokens.len() - 1;
    let device = Device::new_metal(0)?;
    let mut reference = Runner::new(&dir, &device, KvFormat::F32, evaluated)?;
    let mut runners = formats
        .iter()
        .map(|&format| Runner::new(&dir, &device, format, evaluated))
        .collect::<Result<Vec<_>>>()?;
    let mut buckets = vec![vec![Bucket::default(); BUCKETS.len() - 1]; formats.len()];
    let inputs = &tokens[..evaluated];
    for (index, chunk) in inputs.chunks(CHUNK).enumerate() {
        let first = index * CHUNK;
        let targets =
            Tensor::new(&tokens[first + 1..first + 1 + chunk.len()], &device)?.unsqueeze(1)?;
        let exact = reference.advance(chunk)?;
        let exact_nll = exact
            .gather(&targets, 1)?
            .squeeze(1)?
            .neg()?
            .to_vec1::<f32>()?;
        let exact_top = exact.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let exact_probs = exact.exp()?;
        for (runner, buckets) in runners.iter_mut().zip(&mut buckets) {
            let log_probs = runner.advance(chunk)?;
            let kl = (&exact_probs * (&exact - &log_probs)?)?
                .sum(D::Minus1)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let nll = log_probs
                .gather(&targets, 1)?
                .squeeze(1)?
                .neg()?
                .to_vec1::<f32>()?;
            let top = log_probs.argmax(D::Minus1)?.to_vec1::<u32>()?;
            for i in 0..chunk.len() {
                let position = first + i;
                let Some(bucket) = BUCKETS
                    .windows(2)
                    .position(|w| (w[0]..w[1]).contains(&position))
                    .and_then(|b| buckets.get_mut(b))
                else {
                    continue;
                };
                bucket.positions += 1;
                bucket.kl += f64::from(kl[i]);
                bucket.kl_max = bucket.kl_max.max(f64::from(kl[i]));
                bucket.agree += usize::from(top[i] == exact_top[i]);
                bucket.nll += f64::from(nll[i]);
                bucket.reference_nll += f64::from(exact_nll[i]);
            }
        }
        eprintln!("evaluated {} of {evaluated} positions", first + chunk.len());
    }
    let results: Vec<Value> = formats
        .iter()
        .zip(&buckets)
        .map(|(format, buckets)| {
            let total = buckets.iter().fold(Bucket::default(), |a, b| Bucket {
                positions: a.positions + b.positions,
                kl: a.kl + b.kl,
                kl_max: a.kl_max.max(b.kl_max),
                agree: a.agree + b.agree,
                nll: a.nll + b.nll,
                reference_nll: a.reference_nll + b.reference_nll,
            });
            json!({
                "kv_format": format.to_string(),
                "all": total.json(0, evaluated),
                "buckets": buckets
                    .iter()
                    .zip(BUCKETS.windows(2))
                    .filter(|(b, _)| b.positions > 0)
                    .map(|(b, w)| b.json(w[0], w[1]))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let document = json!({
        "schema_version": 1,
        "checkpoint": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "weights": "q8_0", "reference_kv": "f32", "page_tokens": 16,
        "text": text_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "text_bytes": text.len(), "evaluated_positions": evaluated, "chunk": CHUNK,
        "metric": "teacher-forced next-token distributions; KL(F32-K/V || format) in nats",
        "results": results,
    });
    let output = serde_json::to_string_pretty(&document)?;
    std::io::Write::write_all(&mut std::io::stdout(), output.as_bytes())?;
    Ok(())
}
