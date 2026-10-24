#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmarks emit JSON artifacts and timing summaries"
)]
#![forbid(unsafe_code)]
//! Host cost of prefix keys, lookups and eviction at serving prompt lengths, on a CPU pool
//! with one tiny layer of each kind (snapshot bytes do not affect these costs). Prints
//! one JSON document: `cargo bench --offline -p tur-engine --bench prefix`.

use candle_core::{DType, Device};
use serde_json::{Value, json};
use std::{hint::black_box, sync::Arc, time::Instant};
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, LayerStateKind, Modality, ModelDescription, ModelIdentity},
};
use tur_engine::backend::prefix::{PrefixCache, PrefixConfig};
use tur_metal::{
    pool::{PoolConfig, StatePool},
    state::{Boundaries, ModelState},
};

const PAGE: usize = 16;
const SPACING: usize = 256;
const LONGEST: usize = 32_768;

fn pool(entries: usize) -> Result<StatePool> {
    StatePool::new(
        Arc::new(ModelDescription {
            identity: ModelIdentity {
                revision: "prefix-bench".into(),
                state_format: 1,
            },
            layers: vec![
                LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 4,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 1,
                    conv_width: 2,
                    heads: 1,
                    key_dim: 1,
                    value_dim: 1,
                    dtype: DType::F32,
                },
            ],
            kv: KvLayout::new(PAGE, KvFormat::F32),
            modalities: vec![Modality::Text],
            vocab_size: 16,
            max_position: LONGEST,
        }),
        &Device::Cpu,
        PoolConfig {
            pages: entries * LONGEST / PAGE,
            recurrent_slots: entries + 1,
        },
    )
}

fn state(pool: &StatePool, tokens: usize) -> Result<ModelState> {
    let mut reservation = ModelState::new(pool).reserve(tokens, Boundaries::Final)?;
    reservation.prepare()?;
    reservation.commit(tokens)
}

/// Distinct tokens per `salt`.
fn tokens(length: usize, salt: u32) -> Vec<u32> {
    (0..length as u32)
        .map(|t| t.wrapping_mul(2_654_435_761).wrapping_add(salt) % 151_000)
        .collect()
}

/// Mean seconds of `f` over enough repetitions to take about 50 ms.
fn time(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    let mut runs = 0u32;
    while start.elapsed().as_secs_f64() < 0.05 || runs < 5 {
        f();
        runs += 1;
    }
    start.elapsed().as_secs_f64() / f64::from(runs)
}

fn main() -> Result<()> {
    let mut rows: Vec<Value> = Vec::new();
    for entries in [8, 64] {
        let pool = pool(entries)?;
        for length in [2048, 8192, LONGEST] {
            pool.reclaim();
            let mut cache = PrefixCache::new(PrefixConfig {
                spacing: SPACING,
                entries,
                state_share: None,
            });
            // Fill the cache with snapshots of other prompts at their deepest boundary,
            // and one of the probed prompt's own first half.
            for salt in 1..entries as u32 {
                let other = tokens(length / 2, salt);
                let keys = cache.keys(None, &other);
                cache.insert(&keys, &other, &state(&pool, other.len())?, None)?;
            }
            let probe = tokens(length, 0);
            let keys = cache.keys(None, &probe);
            cache.insert(
                &keys,
                &probe[..length / 2],
                &state(&pool, length / 2)?,
                None,
            )?;
            let miss = tokens(length, 9_999);
            let miss_keys = cache.keys(None, &miss);
            let hash = time(|| {
                black_box(cache.keys(None, black_box(&probe)));
            });
            let hit = time(|| {
                black_box(cache.lookup(&keys, &probe, |_| true).map(|hit| hit.length));
            });
            let missed = time(|| {
                black_box(cache.lookup(&miss_keys, &miss, |_| true).is_some());
            });
            eprintln!(
                "{entries} entries, {length} tokens: keys {:.1} µs, hit {:.1} µs, miss {:.1} µs",
                hash * 1e6,
                hit * 1e6,
                missed * 1e6
            );
            rows.push(json!({"entries": entries, "prompt_tokens": length,
                "keys_seconds": hash, "hit_seconds": hit, "miss_seconds": missed}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark": "prefix", "spacing": SPACING, "rows": rows})
        )?
    );
    Ok(())
}
