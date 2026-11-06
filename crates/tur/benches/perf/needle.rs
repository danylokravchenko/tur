//! Long-context retrieval with dense and latent sparse decoding on a real checkpoint. Facts
//! are planted at evenly spaced depths of a long document of filler text; one prefill per
//! document serves every question, each asked from a snapshot of it and answered greedily.
//! The prompt prefills densely in both modes, so the answer tokens are where sparse
//! selection must find the planted keys. Prints one JSON document.
//!
//! `TUR_BENCH_MODEL=<snapshot> TUR_BENCH_TEXT=<filler.txt> TUR_BENCH_SPARSE_MAP=<map>
//! cargo bench --offline -p tur --bench perf -- needle > out.json`
//! `TUR_BENCH_CONTEXTS=16384,32768` sets the document lengths, `TUR_BENCH_NEEDLES=8` the
//! facts per document and `TUR_BENCH_SPARSE_CROSSOVER=<keys>` the sparse crossover.

use crate::model::{pool, prefilled};
use candle_core::Device;
use serde_json::{Value, json};
use std::path::PathBuf;
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout, invalid},
};
use tur_engine::hybrid::{
    Boundaries, HybridModel, ModelState, PackedForward, SequenceForward, StatePool,
};
use tur_models::models::{Execution, LoadOptions, SparseAttention, TextModel, WeightFormat};

/// Greedy answer tokens: Qwen spends one per digit, so a six-digit code and its leading
/// space take seven.
const ANSWER_TOKENS: usize = 10;
const NAMES: [&str; 12] = [
    "amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
    "kestrel", "lagoon",
];

fn var(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| TurError::Other(format!("set {name}")))
}

fn sizes(name: &str, default: &str) -> Result<Vec<usize>> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|v| v.parse().map_err(|_| invalid(name.to_owned())))
        .collect()
}

/// The code planted for fact `i`: six digits, none of them repeated in a row.
fn code(i: usize) -> String {
    format!("{}", 100_000 + (i * 7_919 + 3_571) * 37 % 900_000)
}

struct Document {
    tokens: Vec<u32>,
    /// Each fact's name, code and token position.
    facts: Vec<(&'static str, String, usize)>,
}

/// `context` tokens of `filler` with `needles` facts at evenly spaced depths.
fn document(
    tokenizer: &tokenizers::Tokenizer,
    filler: &[u32],
    context: usize,
    needles: usize,
) -> Result<Document> {
    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec())
    };
    let mut tokens = Vec::with_capacity(context);
    let mut facts = Vec::new();
    let mut taken = 0;
    for i in 0..needles {
        let depth = (2 * i + 1) * context / (2 * needles);
        let take = depth.saturating_sub(tokens.len());
        tokens.extend(filler.iter().cycle().skip(taken).take(take));
        taken += take;
        let name = NAMES[i % NAMES.len()];
        facts.push((name, code(i), tokens.len()));
        tokens.extend(encode(&format!(
            "\nThe access code for the {name} vault is {}.\n",
            code(i)
        ))?);
    }
    let rest = context.saturating_sub(tokens.len());
    tokens.extend(filler.iter().cycle().skip(taken).take(rest));
    tokens.truncate(context);
    Ok(Document { tokens, facts })
}

/// Commit `tokens` after `state` and return the last row's logits.
fn advance(
    model: &TextModel,
    pool: &StatePool,
    state: &ModelState,
    tokens: &[u32],
) -> Result<(Vec<f32>, ModelState)> {
    let mut reservation = state.reserve(tokens.len(), Boundaries::Final)?;
    reservation.prepare()?;
    let binding = reservation.binding();
    let positions: Vec<usize> = (state.position()..state.position() + tokens.len()).collect();
    let last = [tokens.len() - 1];
    let step = model
        .forward(&PackedForward {
            embeddings: &[],
            axial: &[],
            tokens,
            positions: &positions,
            sequences: &[SequenceForward {
                tokens: 0..tokens.len(),
                state: &binding,
                logits: &last,
                hidden: false,
            }],
            pool,
        })?
        .pop()
        .ok_or_else(|| invalid("missing step"))??;
    let logits = step
        .logits
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let next = reservation.commit(tokens.len())?;
    drop(reservation);
    model.device().synchronize()?;
    pool.reclaim();
    Ok((logits, next))
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Ask for each fact of `doc` from one prefill; returns each answer's text.
fn answers(
    model: &TextModel,
    tokenizer: &tokenizers::Tokenizer,
    doc: &Document,
) -> Result<Vec<String>> {
    let pool = pool(model, &[doc.tokens.len()], 64)?;
    let base = prefilled(model, &pool, &doc.tokens)?;
    let mut out = Vec::new();
    for (name, _, _) in &doc.facts {
        let question = format!(
            "\n\nQuestion: What is the access code for the {name} vault?\nAnswer: The access code for the {name} vault is"
        );
        let ids = tokenizer
            .encode(question.as_str(), false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        let (mut logits, mut state) = advance(model, &pool, &base, &ids)?;
        let mut generated = Vec::with_capacity(ANSWER_TOKENS);
        for _ in 0..ANSWER_TOKENS {
            let token = argmax(&logits);
            generated.push(token);
            (logits, state) = advance(model, &pool, &state, &[token])?;
        }
        drop(state);
        out.push(
            tokenizer
                .decode(&generated, true)
                .map_err(|e| TurError::Tokenizer(e.to_string()))?,
        );
    }
    Ok(out)
}

fn load(dir: &std::path::Path, sparse: Option<SparseAttention>) -> Result<TextModel> {
    let mut model = TextModel::load(
        dir,
        &LoadOptions {
            revision: dir.display().to_string(),
            weights: WeightFormat::Q8_0.into(),
            device: Device::new_metal(0)?,
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F16),
            recurrent: candle_core::DType::F32,
            mtp: None,
            audio: false,
            vision: false,
            video: false,
            expert_cache: None,
        },
    )?;
    if let Some(sparse) = sparse {
        model.sparse_attention(&sparse)?;
    }
    Ok(model)
}

pub(crate) fn run() -> Result<()> {
    let dir = PathBuf::from(var("TUR_BENCH_MODEL")?);
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let filler = tokenizer
        .encode(
            std::fs::read_to_string(var("TUR_BENCH_TEXT")?)?.as_str(),
            false,
        )
        .map_err(|e| TurError::Tokenizer(e.to_string()))?
        .get_ids()
        .to_vec();
    let contexts = sizes("TUR_BENCH_CONTEXTS", "16384,32768")?;
    let needles = sizes("TUR_BENCH_NEEDLES", "8")?
        .first()
        .copied()
        .unwrap_or(8);
    let mut sparse = SparseAttention::new(PathBuf::from(var("TUR_BENCH_SPARSE_MAP")?));
    if let Ok(keys) = std::env::var("TUR_BENCH_SPARSE_CROSSOVER") {
        sparse.crossover = keys
            .parse()
            .map_err(|_| invalid("TUR_BENCH_SPARSE_CROSSOVER"))?;
    }
    let documents = contexts
        .iter()
        .map(|&c| document(&tokenizer, &filler, c, needles))
        .collect::<Result<Vec<_>>>()?;
    let mut modes = Vec::new();
    for (label, settings) in [("dense", None), ("sparse", Some(sparse.clone()))] {
        let model = load(&dir, settings)?;
        let answered = documents
            .iter()
            .map(|doc| answers(&model, &tokenizer, doc))
            .collect::<Result<Vec<_>>>()?;
        modes.push((label, answered));
    }
    let mut cases: Vec<Value> = Vec::new();
    for (d, doc) in documents.iter().enumerate() {
        for (f, (name, code, position)) in doc.facts.iter().enumerate() {
            let [dense, sparse] = [&modes[0].1[d][f], &modes[1].1[d][f]];
            cases.push(json!({
                "context": doc.tokens.len(), "fact": name, "position": position,
                "depth": *position as f64 / doc.tokens.len() as f64, "code": code,
                "dense": dense, "sparse": sparse,
                "dense_correct": dense.contains(code.as_str()),
                "sparse_correct": sparse.contains(code.as_str()),
            }));
        }
    }
    let correct = |mode: &str, context: usize| {
        cases
            .iter()
            .filter(|c| c["context"] == context && c[format!("{mode}_correct")] == true)
            .count()
    };
    let summary: Vec<Value> = contexts
        .iter()
        .map(|&c| json!({"context": c, "facts": needles, "dense_correct": correct("dense", c), "sparse_correct": correct("sparse", c)}))
        .collect();
    let report = json!({
        "checkpoint": dir, "budget": format!("{sparse:?}"), "answer_tokens": ANSWER_TOKENS,
        "workload": "facts planted at evenly spaced depths of repeated filler text; one dense prefill per document; each question from a snapshot; greedy answers",
        "summary": summary, "cases": cases,
    });
    let text = serde_json::to_string_pretty(&report)?;
    std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?;
    Ok(())
}
