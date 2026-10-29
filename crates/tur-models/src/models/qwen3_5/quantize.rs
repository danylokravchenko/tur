//! Offline weight preparation: calibrated packing, prepacked artifact writing, per-projection
//! sensitivity and precision allocation under a byte budget. Nothing here runs while serving.

use super::{Qwen35, config::Config, model::streaming::calibrate_layers};
use crate::models::artifact::{self, Conversion, Manifest, Writer};
use crate::models::{Embedding, Method, Packer, Precision, WeightPlan};
use candle_core::{D, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
use tur_core::{Result, execution::invalid};

/// The source config with its MTP head when the checkpoint has one, so an artifact can
/// serve speculative decoding too.
pub fn artifact_config(source: &Path) -> Result<Config> {
    let config = Config::from_json(&std::fs::read_to_string(source.join("config.json"))?)?;
    if config.supports_mtp() {
        config.with_mtp()
    } else {
        Ok(config)
    }
}

/// Pack every projection of the checkpoint at `conversion.source` by `plan` and write a
/// prepacked artifact, one tensor at a time. Small tensors are copied verbatim.
pub fn write_artifact(
    conversion: &Conversion<'_>,
    plan: &WeightPlan,
    packer: &Packer<'_>,
) -> Result<Manifest> {
    let config = artifact_config(conversion.source)?;
    let checkpoint = config.open_checkpoint(conversion.source)?;
    artifact::write(conversion, &config, &checkpoint, plan, packer)
}

/// Calibrate and pack one F32 layer at a time. Hidden states are spooled in temporary
/// files under the output directory and removed on success or failure. Peak memory
/// includes one F32 layer, its statistics and one calibration sample, never the full
/// F32 model or all layers' Hessians. The MTP head is packed without calibration.
/// `packer.calibration` must be `None`; statistics come from `samples`.
pub fn write_calibrated_artifact(
    conversion: &Conversion<'_>,
    plan: &WeightPlan,
    packer: &Packer<'_>,
    samples: &[Vec<u32>],
    device: &Device,
) -> Result<Manifest> {
    if packer.calibration.is_some() {
        return Err(invalid("streaming conversion collects its own calibration"));
    }
    let config = artifact_config(conversion.source)?;
    plan.validate(&config)?;
    let checkpoint = config.open_checkpoint(conversion.source)?;
    let mut writer = Writer::create(conversion.source, conversion.output)?;
    let projections = config.projections()?;
    let info = calibrate_layers(
        conversion.source,
        &config,
        samples,
        packer.method == Method::Gptq,
        device,
        conversion.output,
        |calibration| {
            let packer = Packer {
                calibration: Some(calibration),
                ..*packer
            };
            for projection in projections
                .iter()
                .filter(|p| calibration.sites.contains_key(&p.name))
            {
                packer.write(
                    &checkpoint,
                    projection,
                    plan.precision(&projection.name)?,
                    &mut writer,
                )?;
            }
            Ok(())
        },
    )?;
    for projection in projections.iter().filter(|p| p.name.starts_with("mtp.")) {
        packer.write(
            &checkpoint,
            projection,
            plan.precision(&projection.name)?,
            &mut writer,
        )?;
    }
    let packer = Packer {
        calibration: Some(&info),
        ..*packer
    };
    artifact::finish(conversion, &config, plan, &packer, &checkpoint, writer)
}

/// Teacher-forced next-token log-probabilities of an F32 model over fixed samples.
pub struct Reference {
    samples: Vec<(Vec<u32>, Tensor)>,
}

impl Reference {
    /// Log-probabilities `model` assigns at every position of each sample.
    pub fn new(model: &Qwen35, samples: Vec<Vec<u32>>) -> Result<Self> {
        let samples = samples
            .into_iter()
            .map(|tokens| {
                let probs = log_probs(model, &tokens)?;
                Ok((tokens, probs))
            })
            .collect::<Result<_>>()?;
        Ok(Self { samples })
    }

    /// Mean `KL(reference ‖ model)` in nats over every position of every sample.
    pub fn divergence(&self, model: &Qwen35) -> Result<f64> {
        let (mut total, mut count) = (0.0, 0usize);
        for (tokens, reference) in &self.samples {
            let candidate = log_probs(model, tokens)?;
            let kl = (reference.exp()? * (reference - candidate)?)?
                .sum(D::Minus1)?
                .to_vec1::<f32>()?;
            total += kl.iter().map(|&v| f64::from(v)).sum::<f64>();
            count += tokens.len();
        }
        Ok(total / count.max(1) as f64)
    }
}

fn log_probs(model: &Qwen35, tokens: &[u32]) -> Result<Tensor> {
    let rows: Vec<usize> = (0..tokens.len()).collect();
    let (_, logits) = model.trace(tokens, &rows)?;
    Ok(candle_nn::ops::log_softmax(&logits, D::Minus1)?)
}

/// Divergence of the model when only one projection is stored in one precision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sensitivity {
    pub name: String,
    pub precision: Precision,
    /// Mean teacher-forced `KL(F32 ‖ variant)` in nats on the calibration samples.
    pub kl: f64,
    pub bytes: usize,
}

/// Measure every projection of the F32 `model` at each of `precisions`, one projection
/// at a time with all others in F32, packed exactly as the artifact would pack them.
pub fn sensitivities(
    model: &mut Qwen35,
    reference: &Reference,
    precisions: &[Precision],
    packer: &Packer<'_>,
    mut progress: impl FnMut(&Sensitivity),
) -> Result<Vec<Sensitivity>> {
    let projections = model.config().projections()?;
    let mut measured = Vec::new();
    // Teacher-forced passes never run an MTP head.
    for projection in projections.iter().filter(|p| !p.name.starts_with("mtp.")) {
        let values = model
            .projection(&projection.name)?
            .dequantize()?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let factor = packer.factor(projection)?;
        for &precision in precisions {
            let weights = packer
                .pack_values(&values, projection, precision, factor.as_ref())?
                .upload(
                    precision,
                    (projection.rows, projection.cols),
                    model.device(),
                )?;
            let original = model.replace_projection(&projection.name, weights)?;
            let kl = reference.divergence(model);
            model.replace_projection(&projection.name, original)?;
            let sensitivity = Sensitivity {
                name: projection.name.clone(),
                precision,
                kl: kl?,
                bytes: precision.bytes(projection.rows, projection.cols)?,
            };
            progress(&sensitivity);
            measured.push(sensitivity);
        }
    }
    Ok(measured)
}

/// Choose each projection's precision under `budget` total bytes (projections plus the
/// embedding table), assuming divergences add. Every measured projection starts at its
/// narrowest measured precision; the upgrade with the largest divergence reduction per
/// added byte is applied while it fits. Unmeasured projections (an MTP head that the
/// passes never reach) keep `base`.
pub fn allocate(
    config: &Config,
    measured: &[Sensitivity],
    budget: usize,
    base: Precision,
    embedding: Embedding,
) -> Result<WeightPlan> {
    let mut options: BTreeMap<&str, Vec<&Sensitivity>> = BTreeMap::new();
    for s in measured {
        options.entry(&s.name).or_default().push(s);
    }
    for list in options.values_mut() {
        list.sort_by_key(|s| s.bytes);
    }
    let mut plan = WeightPlan {
        projections: config
            .projections()?
            .into_iter()
            .map(|p| (p.name, base))
            .collect(),
        embedding,
    };
    let mut current: BTreeMap<&str, &Sensitivity> = BTreeMap::new();
    for (name, list) in &options {
        let narrowest = list.first().ok_or_else(|| invalid("empty sensitivity"))?;
        current.insert(name, narrowest);
        plan.projections
            .insert((*name).to_owned(), narrowest.precision);
    }
    let mut used = plan.bytes(config)?;
    if used > budget {
        return Err(invalid(format!(
            "budget {budget} B is below the narrowest plan's {used} B"
        )));
    }
    loop {
        let best = options
            .iter()
            .flat_map(|(name, list)| {
                let now = current[name];
                list.iter()
                    .filter(move |s| s.bytes > now.bytes && s.kl < now.kl)
                    .map(move |s| (name, now, *s))
            })
            .filter(|(_, now, s)| used - now.bytes + s.bytes <= budget)
            .max_by(|a, b| {
                let gain = |(_, now, s): &(&&str, &Sensitivity, &Sensitivity)| {
                    (now.kl - s.kl) / (s.bytes - now.bytes) as f64
                };
                gain(a).total_cmp(&gain(b))
            });
        let Some((name, now, next)) = best else {
            break;
        };
        used = used - now.bytes + next.bytes;
        current.insert(name, next);
        plan.projections.insert((*name).to_owned(), next.precision);
    }
    plan.validate(config)?;
    Ok(plan)
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use crate::models::Projection;

    const SMALL: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/small-config.json");

    #[test]
    fn test_allocation_upgrades_the_best_divergence_per_byte_within_budget() -> Result<()> {
        let config = Config::from_json(SMALL)?;
        let projections = config.projections()?;
        let (a, b) = (&projections[0], &projections[1]);
        let entry = |p: &Projection, precision: Precision, kl: f64| -> Result<Sensitivity> {
            Ok(Sensitivity {
                name: p.name.clone(),
                precision,
                kl,
                bytes: precision.bytes(p.rows, p.cols)?,
            })
        };
        let measured = vec![
            entry(a, Precision::Q4k, 0.010)?,
            entry(a, Precision::Q8_0, 0.001)?,
            entry(b, Precision::Q4k, 0.002)?,
            entry(b, Precision::Q8_0, 0.0)?,
        ];
        let embedding = Embedding::Table(Precision::Bf16);
        let narrow = allocate(&config, &measured, usize::MAX, Precision::Q4k, embedding)?;
        assert_eq!(
            narrow
                .projections
                .values()
                .filter(|p| **p == Precision::Q8_0)
                .count(),
            2
        );
        let floor =
            WeightPlan::uniform(&config, crate::models::WeightFormat::Q4K)?.bytes(&config)?;
        let extra = |p: &Projection| -> Result<usize> {
            Ok(Precision::Q8_0.bytes(p.rows, p.cols)? - Precision::Q4k.bytes(p.rows, p.cols)?)
        };
        // Room for one upgrade: `a` gains more per byte unless `b` is much cheaper.
        let plan = allocate(
            &config,
            &measured,
            floor + extra(a)?.max(extra(b)?),
            Precision::Q4k,
            embedding,
        )?;
        let gain_a = 0.009 / extra(a)? as f64;
        let gain_b = 0.002 / extra(b)? as f64;
        let upgraded = if gain_a >= gain_b { a } else { b };
        assert_eq!(plan.precision(&upgraded.name)?, Precision::Q8_0);
        assert!(plan.bytes(&config)? <= floor + extra(a)?.max(extra(b)?));
        assert!(allocate(&config, &measured, floor - 1, Precision::Q4k, embedding).is_err());
        Ok(())
    }
}
