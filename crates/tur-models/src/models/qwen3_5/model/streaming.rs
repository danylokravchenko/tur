//! Layer-wise reference calibration. Only one layer, its statistics and one sample
//! live in memory; hidden states between layers are spooled beside the output.

use super::{
    Arc, Boundaries, Config, DEFAULT_RECURRENT_STATE, Dims, Layer, Mixer, Mlp, ModelDescription,
    ModelState, Pass, PoolConfig, SequenceForward, Span, StatePool, Tensor, Tensors, WeightFormat,
    centred, math, plan,
};
use crate::quantization::{self, LayerCalibration};
use crate::quantization::{Calibration, Recorder};
use candle_core::Device;
use std::path::Path;
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, invalid},
};

/// Calibrate unquantized layers in order and consume their statistics before loading
/// the next layer. The head only needs normalized hidden states, not head weights.
pub(in crate::models::qwen3_5) fn calibrate_layers(
    source: &Path,
    config: &Config,
    samples: &[Vec<u32>],
    hessians: bool,
    device: &Device,
    scratch: &Path,
    mut consume: impl FnMut(&Calibration) -> Result<()>,
) -> Result<Calibration> {
    validate_samples(source, config, samples)?;
    let tensors = Tensors::open(source, config, &WeightFormat::F32.into(), device)?;
    let dims = Dims::new(config)?;
    let ids: Vec<u8> = samples
        .iter()
        .flatten()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    let info = Calibration {
        samples: samples.len(),
        tokens: samples.iter().map(Vec::len).sum(),
        fingerprint: plan::fnv1a(&ids),
        ..Calibration::default()
    };
    let description = config.description(
        "calibration".into(),
        KvLayout::new(16, KvFormat::F32),
        DEFAULT_RECURRENT_STATE,
    )?;
    let projections = config.projections()?;
    let embeddings = samples.iter().map(|sample| {
        tensors
            .checkpoint
            .gather_rows(&config.embedding_name(), sample)
    });
    let layers = (0..=config.layer_kinds().len()).map(|index| {
        let recorder = Recorder::new(hessians);
        let mut single = description.clone();
        let (body, names) = if index < config.layer_kinds().len() {
            single.layers = vec![description.layers[index].clone()];
            let prefix = format!("{}.layers.{index}.", config.text_prefix());
            let names = projections
                .iter()
                .filter(|p| p.name.starts_with(&prefix))
                .map(|p| p.name.clone())
                .collect();
            (
                Body::Layer(Box::new(Layer::load(&tensors, config, index)?)),
                names,
            )
        } else {
            let norm = centred(
                &tensors.take(&format!("{}.norm.weight", config.text_prefix()))?,
                device,
            )?;
            (Body::Head(norm), Vec::new())
        };
        Ok(Stage {
            body,
            dims,
            description: single,
            recorder,
            names,
            info: info.clone(),
        })
    });
    quantization::calibrate_layers(embeddings, layers, device, scratch, &mut consume)?;
    Ok(info)
}

fn validate_samples(source: &Path, config: &Config, samples: &[Vec<u32>]) -> Result<()> {
    let mut config = config.clone();
    let tokenizer_config = source.join("tokenizer_config.json");
    let tokenizer = source.join("tokenizer.json");
    if tokenizer_config.exists() && tokenizer.exists() {
        config.forbid_tokenizer_modalities(
            &std::fs::read_to_string(tokenizer_config)?,
            &std::fs::read_to_string(tokenizer)?,
        )?;
    }
    if samples.is_empty() || samples.iter().any(Vec::is_empty) {
        return Err(invalid("calibration requires nonempty samples"));
    }
    let max_position = config
        .description(
            "calibration".into(),
            KvLayout::new(16, KvFormat::F32),
            DEFAULT_RECURRENT_STATE,
        )?
        .max_position;
    if samples.iter().any(|s| s.len() > max_position) {
        return Err(invalid("calibration sample exceeds maximum positions"));
    }
    if samples
        .iter()
        .flatten()
        .any(|t| *t as usize >= config.vocab_size() || config.forbidden_tokens().contains(t))
    {
        return Err(invalid("calibration token is outside the text vocabulary"));
    }
    Ok(())
}

fn run_layer(
    layer: &Layer,
    dims: Dims,
    x: Tensor,
    description: &ModelDescription,
    recorder: &Recorder,
) -> Result<Tensor> {
    let len = x.dim(0)?;
    let pool = StatePool::new(
        Arc::new(description.clone()),
        x.device(),
        PoolConfig {
            pages: description.pages_for(len).max(1),
            recurrent_slots: 1,
        },
    )?;
    let mut reservation = ModelState::new(&pool).reserve(len, Boundaries::Final)?;
    reservation.prepare()?;
    let state = reservation.binding();
    let sequence = SequenceForward {
        tokens: 0..len,
        state: &state,
        logits: &[],
        hidden: false,
    };
    let span = Span {
        start: 0,
        len,
        past: 0,
        every: false,
        addresses: state.resolve(&pool)?,
        sequence: &sequence,
    };
    let pass = Pass {
        ids: Vec::new(),
        positions: (0..len).collect(),
        rows: Vec::new(),
        hidden: false,
        spans: &[span],
        pool: &pool,
    };
    let result = layer.reference(dims, x, &pass, 0, Some(recorder));
    // The pool must outlive all commands, including when reference execution fails.
    pool.device().synchronize()?;
    result
}

enum Body {
    Layer(Box<Layer>),
    Head(Tensor),
}

/// Qwen's adapter supplies reference math and projection identities; the shared driver
/// owns the lifetime of layers, activations and statistics.
struct Stage {
    body: Body,
    dims: Dims,
    description: ModelDescription,
    recorder: Recorder,
    names: Vec<String>,
    info: Calibration,
}

impl LayerCalibration for Stage {
    type Statistics = Calibration;

    fn forward(&self, input: Tensor) -> Result<Tensor> {
        match &self.body {
            Body::Layer(layer) => {
                run_layer(layer, self.dims, input, &self.description, &self.recorder)
            }
            Body::Head(norm) => {
                let h = math::rms(&input, norm, self.dims.eps)?;
                self.recorder.record_named("lm_head", &h)?;
                Ok(input)
            }
        }
    }

    fn finish(self) -> Result<Calibration> {
        let mut refs = Vec::new();
        if let Body::Layer(layer) = &self.body {
            projections!(@layer refs, layer);
        }
        let named: Vec<_> = self.names.into_iter().zip(refs).collect();
        self.recorder.finish(
            &named,
            self.info.samples,
            self.info.tokens,
            self.info.fingerprint,
        )
    }
}
