//! State and weight storage quality on a real checkpoint: teacher-forced next-token
//! prediction over a long held-out text, each K/V format, recurrent state dtype or weight
//! variant against a reference with F32 K/V and F32 state. Reports, by position bucket,
//! the KL divergence from the reference distribution, top-1 agreement with the reference
//! and with the text, and the mean negative log-likelihood. Prints one JSON document.
//!
//! `TUR_QWEN35_0_8B=<snapshot> TUR_EVAL_TEXT=<file> cargo bench --offline -p tur
//! --bench perf -- kv_quality > out.json`; `TUR_EVAL_TOKENS` (default 8192) bounds the context,
//! `TUR_EVAL_KV` (default `f16,q8,q8_64,q8_256`, F32 state) and `TUR_EVAL_STATE` (default
//! none, F32 K/V; e.g. `f16,bf16`) select state variants, and `TUR_EVAL_CHUNK` (default
//! 256) sets tokens per pass. A chunk of 1 stores recurrent state after every token, as
//! decode does; longer chunks keep it in F32 registers within a chunk.
//!
//! Weights default to `Q8_0` for every model. `TUR_EVAL_REFERENCE=f32` sets the reference's
//! weights; `TUR_EVAL_WEIGHTS` (default none) adds variants with F32 K/V and state, each
//! `f32`, `q8_0`, `q4k` or a prepacked artifact directory. With `TUR_EVAL_MARKED=1`, text
//! between `⟦` and `⟧` (markers removed) is also scored separately, e.g. retrieval answers.
//! `TUR_EVAL_GENERATE=<n>` also keeps each model's state at 1K, 2K, 4K and 8K tokens and,
//! after the teacher-forced pass, forks those snapshots (a second turn reusing the first
//! turn's cached state) to feed the text's next token and decode `n` greedy tokens,
//! reporting how long each variant's continuation matches the reference's.

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
use tur_models::models::qwen3_5::Qwen35;
use tur_models::models::{Execution, LoadOptions, WeightFormat, Weights, parse_recurrent_state};
use tur_models::models::{Manifest, is_artifact};

const BUCKETS: [usize; 5] = [0, 1024, 2048, 4096, 8192];
/// Contexts whose states seed generated continuations.
const SNAPSHOTS: [usize; 4] = [1024, 2048, 4096, 8192];

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
    fn new(dir: &Path, device: &Device, variant: Variant, tokens: usize) -> Result<Self> {
        let Variant {
            kv: format,
            state: recurrent,
            weights,
        } = variant;
        let (dir, weights) = weights.load(dir)?;
        let model = Qwen35::load(
            &dir,
            &LoadOptions {
                revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
                weights,
                device: device.clone(),
                execution: Execution::Fused,
                kv: KvLayout::new(16, format),
                recurrent,
                mtp: None,
                audio: false,
            },
        )?;
        let pool = StatePool::new(
            Arc::new(model.description()),
            device,
            // Room for the snapshots and one continuation forked from each.
            PoolConfig {
                pages: model.description().pages_for(tokens) + 1 + SNAPSHOTS.len() * 16,
                window_pages: model.description().pages_for(tokens) + 1 + SNAPSHOTS.len() * 16,
                recurrent_slots: 4 + 2 * SNAPSHOTS.len(),
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
                embeddings: &[],
                tokens: chunk,
                positions: &positions,
                sequences: &[SequenceForward {
                    tokens: 0..chunk.len(),
                    state: &binding,
                    logits: &rows,
                    hidden: false,
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

    /// Greedy continuation of `steps` tokens from `state` after feeding `first`.
    fn generate(&self, mut state: ModelState, first: u32, steps: usize) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(steps);
        let mut next = first;
        while out.len() < steps {
            let mut reservation = state.reserve(1, Boundaries::Final)?;
            reservation.prepare()?;
            let binding = reservation.binding();
            let step = self
                .model
                .forward(&PackedForward {
                    embeddings: &[],
                    tokens: &[next],
                    positions: &[state.position()],
                    sequences: &[SequenceForward {
                        tokens: 0..1,
                        state: &binding,
                        logits: &[0],
                        hidden: false,
                    }],
                    pool: &self.pool,
                })?
                .pop()
                .ok_or_else(|| invalid("missing step"))??;
            next = step
                .logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?
                .first()
                .copied()
                .ok_or_else(|| invalid("empty logits"))?;
            out.push(next);
            state = reservation.commit(1)?;
            drop(reservation);
            self.model.device().synchronize()?;
            self.pool.reclaim();
        }
        Ok(out)
    }
}

/// K/V format, recurrent state dtype and weights of one evaluated model.
#[derive(Clone, Debug)]
struct Variant {
    kv: KvFormat,
    state: DType,
    weights: WeightVariant,
}

/// A conversion format of the checkpoint, or a prepacked artifact directory.
#[derive(Clone, Debug)]
enum WeightVariant {
    Format(WeightFormat),
    Artifact(PathBuf),
}

impl WeightVariant {
    fn parse(name: &str) -> Result<Self> {
        let path = PathBuf::from(name);
        if is_artifact(&path) {
            return Ok(Self::Artifact(path));
        }
        Ok(Self::Format(name.parse().map_err(TurError::Other)?))
    }

    fn name(&self) -> String {
        match self {
            Self::Format(format) => format!("{format:?}").to_lowercase(),
            Self::Artifact(path) => path.display().to_string(),
        }
    }

    fn load(&self, checkpoint: &Path) -> Result<(PathBuf, Weights)> {
        Ok(match self {
            Self::Format(format) => (checkpoint.to_path_buf(), (*format).into()),
            Self::Artifact(dir) => (
                dir.clone(),
                Weights::Prepacked(Arc::new(Manifest::read(dir)?)),
            ),
        })
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    std::env::var(name).map_or(Ok(default), |v| v.parse().map_err(|_| invalid(name)))
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
        .collect()
}

#[derive(Default, Clone)]
struct Bucket {
    positions: usize,
    kl: f64,
    kl_max: f64,
    agree: usize,
    correct: usize,
    reference_correct: usize,
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
            "top1_accuracy": self.correct as f64 / n,
            "reference_top1_accuracy": self.reference_correct as f64 / n,
            "mean_nll": self.nll / n, "reference_mean_nll": self.reference_nll / n,
        })
    }

    fn merge(&self, other: &Self) -> Self {
        Self {
            positions: self.positions + other.positions,
            kl: self.kl + other.kl,
            kl_max: self.kl_max.max(other.kl_max),
            agree: self.agree + other.agree,
            correct: self.correct + other.correct,
            reference_correct: self.reference_correct + other.reference_correct,
            nll: self.nll + other.nll,
            reference_nll: self.reference_nll + other.reference_nll,
        }
    }
}

/// Strip `⟦`/`⟧` markers and return the text with the byte ranges they enclosed.
fn marked_spans(text: &str) -> (String, Vec<std::ops::Range<usize>>) {
    let mut plain = String::with_capacity(text.len());
    let mut spans = Vec::new();
    let mut open = None;
    for c in text.chars() {
        match c {
            '⟦' => open = Some(plain.len()),
            '⟧' => {
                if let Some(start) = open.take() {
                    spans.push(start..plain.len());
                }
            }
            c => plain.push(c),
        }
    }
    (plain, spans)
}

/// The K/V formats, recurrent state dtypes and weights to compare, from the environment.
fn variants() -> Result<Vec<Variant>> {
    let base = WeightVariant::Format(WeightFormat::Q8_0);
    let mut variants = env_list("TUR_EVAL_KV", "f16,q8,q8_64,q8_256")
        .iter()
        .map(|f| {
            Ok(Variant {
                kv: f.parse::<KvFormat>().map_err(TurError::Other)?,
                state: DType::F32,
                weights: base.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for state in env_list("TUR_EVAL_STATE", "") {
        variants.push(Variant {
            kv: KvFormat::F32,
            state: parse_recurrent_state(&state).map_err(TurError::Other)?,
            weights: base.clone(),
        });
    }
    for weights in env_list("TUR_EVAL_WEIGHTS", "") {
        variants.push(Variant {
            kv: KvFormat::F32,
            state: DType::F32,
            weights: WeightVariant::parse(&weights)?,
        });
    }
    Ok(variants)
}

/// Per variant, how long its greedy continuation from each snapshot matches the
/// reference's.
fn continuations(
    reference: &Runner,
    runners: &[Runner],
    snapshots: Vec<(usize, ModelState, Vec<ModelState>)>,
    tokens: &[u32],
    generate: usize,
) -> Result<Vec<Vec<Value>>> {
    let mut continuations = vec![Vec::new(); runners.len()];
    for (context, exact_state, states) in snapshots {
        let next = tokens[context];
        let expected = reference.generate(exact_state, next, generate)?;
        for ((runner, state), out) in runners.iter().zip(states).zip(&mut continuations) {
            let actual = runner.generate(state, next, generate)?;
            let prefix = actual
                .iter()
                .zip(&expected)
                .take_while(|(a, b)| a == b)
                .count();
            out.push(json!({"context": context, "matched_prefix": prefix, "tokens": generate}));
        }
    }
    Ok(continuations)
}

/// One variant's results: overall, marked spans, continuations and position buckets.
fn variant_json(
    format: &Variant,
    buckets: &[Bucket],
    answers: &Bucket,
    (weight_bytes, kv_bytes): (usize, usize),
    continuations: &[Value],
    evaluated: usize,
) -> Value {
    let total = buckets.iter().fold(Bucket::default(), |a, b| a.merge(b));
    json!({
        "kv_format": format.kv.to_string(),
        "kv_token_bytes": kv_bytes,
        "recurrent_state": format!("{:?}", format.state).to_lowercase(),
        "weights": format.weights.name(),
        "weight_bytes": weight_bytes,
        "all": total.json(0, evaluated),
        "marked": (answers.positions > 0).then(|| answers.json(0, evaluated)),
        "continuations": continuations,
        "buckets": buckets
            .iter()
            .zip(BUCKETS.windows(2))
            .filter(|(b, _)| b.positions > 0)
            .map(|(b, w)| b.json(w[0], w[1]))
            .collect::<Vec<_>>(),
    })
}

pub(crate) fn run() -> Result<()> {
    let dir = env_path("TUR_QWEN35_0_8B")?;
    let text_path = env_path("TUR_EVAL_TEXT")?;
    let text = std::fs::read_to_string(&text_path)?;
    let limit = env_usize("TUR_EVAL_TOKENS", 8192)?;
    let chunk_tokens = Some(env_usize("TUR_EVAL_CHUNK", 256)?)
        .filter(|&c| c > 0)
        .ok_or_else(|| invalid("TUR_EVAL_CHUNK"))?;
    let formats = variants()?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let (text, spans) = if std::env::var("TUR_EVAL_MARKED").is_ok_and(|v| v == "1") {
        marked_spans(&text)
    } else {
        (text, Vec::new())
    };
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let mut tokens = encoding.get_ids().to_vec();
    // A target token is marked when it starts inside a marked span.
    let marked: Vec<bool> = encoding
        .get_offsets()
        .iter()
        .map(|&(start, _)| spans.iter().any(|s| s.contains(&start)))
        .collect();
    tokens.truncate(limit + 1);
    let evaluated = tokens.len() - 1;
    let device = Device::new_metal(0)?;
    let reference_weights = WeightVariant::parse(
        &std::env::var("TUR_EVAL_REFERENCE").unwrap_or_else(|_| "q8_0".into()),
    )?;
    let exact = Variant {
        kv: KvFormat::F32,
        state: DType::F32,
        weights: reference_weights.clone(),
    };
    let mut reference = Runner::new(&dir, &device, exact, evaluated)?;
    let mut runners = formats
        .iter()
        .map(|format| Runner::new(&dir, &device, format.clone(), evaluated))
        .collect::<Result<Vec<_>>>()?;
    let mut buckets = vec![vec![Bucket::default(); BUCKETS.len() - 1]; formats.len()];
    let mut answers = vec![Bucket::default(); formats.len()];
    let weight_bytes: Vec<usize> = runners.iter().map(|r| r.model.weight_bytes()).collect();
    let kv_bytes = runners
        .iter()
        .map(|r| r.model.description().kv_token_bytes())
        .collect::<Result<Vec<_>>>()?;
    let generate = env_usize("TUR_EVAL_GENERATE", 0)?;
    // (context, reference snapshot, each variant's snapshot)
    let mut snapshots: Vec<(usize, ModelState, Vec<ModelState>)> = Vec::new();
    let inputs = &tokens[..evaluated];
    for (index, chunk) in inputs.chunks(chunk_tokens).enumerate() {
        let first = index * chunk_tokens;
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
        for ((runner, buckets), answers) in runners.iter_mut().zip(&mut buckets).zip(&mut answers) {
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
                let target = tokens[position + 1];
                let Some(bucket) = BUCKETS
                    .windows(2)
                    .position(|w| (w[0]..w[1]).contains(&position))
                    .and_then(|b| buckets.get_mut(b))
                else {
                    continue;
                };
                let sample = Bucket {
                    positions: 1,
                    kl: f64::from(kl[i]),
                    kl_max: f64::from(kl[i]),
                    agree: usize::from(top[i] == exact_top[i]),
                    correct: usize::from(top[i] == target),
                    reference_correct: usize::from(exact_top[i] == target),
                    nll: f64::from(nll[i]),
                    reference_nll: f64::from(exact_nll[i]),
                };
                *bucket = bucket.merge(&sample);
                if marked.get(position + 1).copied().unwrap_or(false) {
                    *answers = answers.merge(&sample);
                }
            }
        }
        let end = first + chunk.len();
        if generate > 0 && SNAPSHOTS.contains(&end) && end < tokens.len() {
            snapshots.push((
                end,
                reference.state.clone(),
                runners.iter().map(|r| r.state.clone()).collect(),
            ));
        }
        if (first + chunk.len()).is_multiple_of(512) || first + chunk.len() == evaluated {
            eprintln!("evaluated {} of {evaluated} positions", first + chunk.len());
        }
    }
    let continuations = continuations(&reference, &runners, snapshots, &tokens, generate)?;
    let results: Vec<Value> = formats
        .iter()
        .zip(&buckets)
        .zip(answers.iter().zip(weight_bytes.iter().zip(&kv_bytes)))
        .zip(&continuations)
        .map(
            |(((format, buckets), (answers, (bytes, kv_bytes))), continuations)| {
                variant_json(
                    format,
                    buckets,
                    answers,
                    (*bytes, *kv_bytes),
                    continuations,
                    evaluated,
                )
            },
        )
        .collect();
    let document = json!({
        "schema_version": 2,
        "checkpoint": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "weights": "q8_0 unless a result names other weights",
        "reference_weights": reference_weights.name(), "reference_weight_bytes": reference.model.weight_bytes(),
        "reference_kv": "f32", "reference_state": "f32", "page_tokens": 16,
        "text": text_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "text_bytes": text.len(), "evaluated_positions": evaluated, "chunk": chunk_tokens,
        "metric": "teacher-forced next-token distributions; KL(F32 reference || variant) in nats",
        "results": results,
    });
    let output = serde_json::to_string_pretty(&document)?;
    std::io::Write::write_all(&mut std::io::stdout(), output.as_bytes())?;
    Ok(())
}
