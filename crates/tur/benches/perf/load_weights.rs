//! Load time and resident bytes of one weight variant in a fresh process, until every
//! upload has completed. Peak process memory comes from the caller, e.g. `/usr/bin/time -l`.
//!
//! `cargo bench --offline -p tur --bench perf -- load_weights <dir> <variant>`,
//! where the variant is `f32`, `q8_0`, `q4k` (conversion of the `SafeTensors` checkpoint in
//! `<dir>`) or `prepacked` (the artifact in `<dir>`).

use candle_core::{DType, Device};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout},
};
use tur_models::models::Manifest;
use tur_models::models::TextModel;
use tur_models::models::{Execution, LoadOptions, Weights};

pub(crate) fn run(args: &[String]) -> Result<()> {
    let mut args = args.iter();
    let (Some(dir), Some(variant)) = (args.next(), args.next()) else {
        return Err(TurError::Other(
            "usage: perf -- load_weights <dir> <f32|q8_0|q4k|prepacked>".into(),
        ));
    };
    let dir = PathBuf::from(dir);
    let device = Device::new_metal(0)?;
    let Device::Metal(metal) = &device else {
        return Err(TurError::Other("expected Metal".into()));
    };
    let started = Instant::now();
    let weights = match variant.as_str() {
        "prepacked" => Weights::Prepacked(Arc::new(Manifest::read(&dir)?)),
        format => format
            .parse::<tur_models::models::WeightFormat>()
            .map_err(TurError::Other)?
            .into(),
    };
    let model = TextModel::load(
        &dir,
        &LoadOptions {
            revision: "load-measurement".into(),
            weights,
            device: device.clone(),
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F16),
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            expert_cache: None,
        },
    )?;
    device.synchronize()?;
    let seconds = started.elapsed().as_secs_f64();
    let document = serde_json::json!({
        "variant": variant,
        "load_seconds": seconds,
        "weight_bytes": model.weight_bytes(),
        "metal_allocated_bytes": metal.current_allocated_size(),
        "timing": "fresh process, page cache warm, through Device::synchronize after the last upload",
    });
    println!("{}", serde_json::to_string_pretty(&document)?);
    Ok(())
}
