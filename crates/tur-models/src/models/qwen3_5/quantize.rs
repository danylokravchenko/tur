//! Offline weight preparation: calibrated packing, prepacked artifact writing, per-projection
//! sensitivity and precision allocation under a byte budget. Nothing here runs while serving.

use super::{
    Qwen35,
    artifact::{self, CalibrationInfo, Manifest, Quantizer, Writer},
    calibration::{Calibration, Site},
    config::{Config, Projection},
    plan::{Embedding, Precision, WeightPlan},
    weights::Checkpoint,
};
use candle_core::{D, DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
use tur_core::{Result, execution::invalid};
use tur_metal::{
    packed::PackedLinear,
    quantize::{Gptq, quantize},
};

/// Full-precision rows staged on the host at once while packing a projection.
pub(super) const STAGING_BYTES: usize = 64 << 20;

/// How weights are rounded into their block format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Round to nearest with GGML's own scale search.
    Rtn,
    /// K-quant scales searched under the calibration's per-column mean squared input.
    Imatrix,
    /// Importance-weighted scales plus GPTQ error compensation with the calibration
    /// Hessian. Expert banks, whose inputs differ per expert, fall back to `Imatrix`.
    Gptq,
}

impl std::str::FromStr for Method {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "rtn" => Ok(Self::Rtn),
            "imatrix" => Ok(Self::Imatrix),
            "gptq" => Ok(Self::Gptq),
            _ => Err(format!(
                "method must be rtn, imatrix or gptq, not '{value}'"
            )),
        }
    }
}

/// One projection's packed payload.
pub(super) enum Packed {
    Dense(Vec<f32>),
    Blocks(Vec<u8>),
}

/// Packs projections by one method, using calibration statistics when it has them.
#[derive(Clone, Copy)]
pub struct Packer<'a> {
    pub method: Method,
    /// GPTQ damping relative to the mean Hessian diagonal.
    pub damp: f64,
    pub calibration: Option<&'a Calibration>,
}

impl Packer<'_> {
    /// Round-to-nearest without statistics, as conversion at load time packs.
    pub const RTN: Packer<'static> = Packer {
        method: Method::Rtn,
        damp: 0.01,
        calibration: None,
    };

    fn site(&self, projection: &Projection) -> Option<&Site> {
        self.calibration.and_then(|c| c.sites.get(&projection.name))
    }

    /// The projection's GPTQ factor when this method compensates it; independent of the
    /// precision, so one factorization serves every candidate format.
    fn factor(&self, projection: &Projection) -> Result<Option<Gptq>> {
        if self.method != Method::Gptq || projection.experts > 1 {
            return Ok(None);
        }
        self.site(projection)
            .and_then(|s| s.hessian.as_deref())
            .map(|h| Gptq::new(h, projection.cols, self.damp))
            .transpose()
    }

    /// Pack `projection` streamed from `checkpoint` in row chunks.
    pub(super) fn pack(
        &self,
        checkpoint: &Checkpoint,
        projection: &Projection,
        precision: Precision,
    ) -> Result<Packed> {
        let per_expert = projection.rows / projection.experts;
        let chunk = if projection.experts > 1 {
            per_expert
        } else {
            (STAGING_BYTES / (projection.cols * 4)).max(1)
        };
        let gptq = self.factor(projection)?;
        let mut packed = Vec::new();
        let mut row = 0;
        for (part, _) in &projection.parts {
            for rows in checkpoint.take_rows(part, chunk)? {
                let values = rows?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expert = row / per_expert;
                row += values.len() / projection.cols;
                self.pack_rows(
                    &values,
                    projection,
                    precision,
                    expert,
                    gptq.as_ref(),
                    &mut packed,
                )?;
            }
        }
        Ok(finish(packed, precision))
    }

    /// Pack in-memory row-major `values` of `projection` with its precomputed `gptq` factor.
    fn pack_values(
        &self,
        values: &[f32],
        projection: &Projection,
        precision: Precision,
        gptq: Option<&Gptq>,
    ) -> Result<Packed> {
        let mut packed = Vec::new();
        for (expert, rows) in values
            .chunks(projection.rows / projection.experts * projection.cols)
            .enumerate()
        {
            self.pack_rows(rows, projection, precision, expert, gptq, &mut packed)?;
        }
        Ok(finish(packed, precision))
    }

    fn pack_rows(
        &self,
        values: &[f32],
        projection: &Projection,
        precision: Precision,
        expert: usize,
        gptq: Option<&Gptq>,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let Some(format) = precision.ggml() else {
            out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            return Ok(());
        };
        let packed = if let Some(gptq) = gptq {
            gptq.quantize(values, format)?
        } else {
            let importance = match self.method {
                Method::Rtn => None,
                Method::Imatrix | Method::Gptq => {
                    self.site(projection).and_then(|s| s.importance(expert))
                }
            };
            quantize(values, projection.cols, format, importance)?
        };
        out.extend_from_slice(&packed);
        Ok(())
    }
}

fn finish(bytes: Vec<u8>, precision: Precision) -> Packed {
    if precision.ggml().is_some() {
        Packed::Blocks(bytes)
    } else {
        Packed::Dense(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
    }
}

impl Packed {
    /// Upload as an `[rows, cols]` projection.
    pub(super) fn upload(
        self,
        precision: Precision,
        (rows, cols): (usize, usize),
        device: &Device,
    ) -> Result<PackedLinear> {
        match (self, precision.ggml()) {
            (Self::Blocks(bytes), Some(format)) => {
                PackedLinear::from_blocks(&bytes, format, (rows, cols), device)
            }
            (Self::Dense(values), None) => PackedLinear::dense(
                &Tensor::from_vec(values, (rows, cols), &Device::Cpu)?,
                device,
            ),
            _ => Err(invalid("packed payload does not match its precision")),
        }
    }
}

/// Source and destination of one artifact conversion.
pub struct Conversion<'a> {
    /// Source `SafeTensors` checkpoint directory.
    pub source: &'a Path,
    /// Exact identity of the source checkpoint (repository and commit).
    pub identity: &'a str,
    /// New artifact directory.
    pub output: &'a Path,
    /// Free-form description of the calibration data.
    pub calibration: &'a str,
}

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
    plan.validate(&config)?;
    let checkpoint = Checkpoint::open(conversion.source, &config)?;
    let mut writer = Writer::create(conversion.source, conversion.output)?;
    let projections = config.projections()?;
    let mut parts = std::collections::BTreeSet::new();
    for projection in &projections {
        parts.extend(projection.parts.iter().map(|(name, _)| name.clone()));
        let precision = plan.precision(&projection.name)?;
        match packer.pack(&checkpoint, projection, precision)? {
            Packed::Blocks(bytes) => writer.put(
                &projection.name,
                safetensors::Dtype::U8,
                vec![projection.rows, bytes.len() / projection.rows],
                &bytes,
            )?,
            Packed::Dense(values) => writer.put(
                &projection.name,
                safetensors::Dtype::F32,
                vec![projection.rows, projection.cols],
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
        }
    }
    let embedding = config.embedding_name();
    if let Embedding::Table(precision) = plan.embedding {
        let table = Projection {
            name: embedding.clone(),
            parts: vec![(embedding.clone(), config.vocab_size())],
            rows: config.vocab_size(),
            cols: config.hidden_size(),
            experts: 1,
        };
        let (dtype, shape, bytes) = match precision {
            Precision::Bf16 => {
                let table = checkpoint.take(&embedding)?.to_dtype(DType::BF16)?;
                (
                    safetensors::Dtype::BF16,
                    vec![config.vocab_size(), config.hidden_size()],
                    bf16_bytes(&table)?,
                )
            }
            precision => match Packer::RTN.pack(&checkpoint, &table, precision)? {
                Packed::Blocks(bytes) => (
                    safetensors::Dtype::U8,
                    vec![table.rows, bytes.len() / table.rows],
                    bytes,
                ),
                Packed::Dense(values) => (
                    safetensors::Dtype::F32,
                    vec![table.rows, table.cols],
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            },
        };
        writer.put(&embedding, dtype, shape, &bytes)?;
    }
    for name in config.weight_shapes()?.into_keys() {
        if parts.contains(&name) || name == embedding {
            continue;
        }
        let (entry, bytes) = checkpoint.take_raw(&name)?;
        writer.put(&name, entry.dtype, entry.shape, &bytes)?;
    }
    let manifest = Manifest {
        format_version: artifact::FORMAT_VERSION,
        source: conversion.identity.to_owned(),
        config_fingerprint: artifact::source_fingerprint(conversion.source)?,
        quantizer: Quantizer {
            name: "tur-quantize".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            method: packer.method,
            damp: packer.damp,
        },
        calibration: packer.calibration.map(|c| CalibrationInfo {
            samples: c.samples,
            tokens: c.tokens,
            fingerprint: format!("{:016x}", c.fingerprint),
            description: conversion.calibration.to_owned(),
        }),
        plan: plan.clone(),
    };
    writer.finish(&manifest, &config)?;
    Ok(manifest)
}

/// Little-endian BF16 bits of a BF16 tensor, via its raw host storage.
fn bf16_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let tensor = tensor.to_device(&Device::Cpu)?.contiguous()?;
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cpu(candle_core::CpuStorage::BF16(values)) => Ok(values
            [layout.start_offset()..layout.start_offset() + tensor.elem_count()]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect()),
        _ => Err(invalid("expected a host BF16 tensor")),
    }
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
            WeightPlan::uniform(&config, super::super::WeightFormat::Q4K)?.bytes(&config)?;
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
