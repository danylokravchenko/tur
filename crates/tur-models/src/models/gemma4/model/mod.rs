//! Gemma 4 equations over the common packed weights and persistent state pools.
use super::config::Audio as AudioSettings;
use super::{Config, config::LayerKind};
use crate::audio::{Audio, AudioPrompt};
#[cfg(feature = "metal")]
use crate::models::paging::Paging;
use crate::{
    hybrid::{
        Addresses, Boundaries, DraftForward, DraftSequence, HybridModel, PackedForward,
        SequenceForward, SequenceStep, StatePool,
    },
    models::{
        Execution, LoadOptions, Packer, WeightPlan, Weights,
        artifact::{self, Conversion, Manifest},
        checkpoint::{Checkpoint, weight_files},
        math,
        weights::{Embeddings, Tensors, WeightLayout},
    },
};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeSet, path::Path, sync::Arc};
use tur_core::execution::EmbeddedInput;
use tur_core::{
    Result,
    execution::{ModelDescription, checked_elements, invalid},
};
use tur_metal::{packed::PackedLinear, pool::PoolConfig, state::ModelState};
mod assistant;
mod audio;
mod moe;

struct Layer {
    /// `q | k | v` on K/V owners (`q | k` when values reuse keys), queries alone otherwise.
    qkv: PackedLinear,
    o: PackedLinear,
    q_norm: Tensor,
    k_norm: Option<Tensor>,
    input_norm: Tensor,
    post_attention: Tensor,
    pre_ff: Tensor,
    post_ff: Tensor,
    scalar: Tensor,
    gate_up: PackedLinear,
    down: PackedLinear,
    ple: Option<LayerInput>,
    moe: Option<moe::Moe>,
}
struct LayerInput {
    gate: PackedLinear,
    projection: PackedLinear,
    norm: Tensor,
}
struct PerLayer {
    table: Vec<Tensor>,
    projection: PackedLinear,
    norm: Tensor,
}

/// Gemma 4 text tower; immutable weights, with all request state in the engine's pools.
pub struct Gemma4 {
    config: Config,
    description: ModelDescription,
    device: Device,
    embedding: Embeddings,
    head: PackedLinear,
    norm: Tensor,
    layers: Vec<Layer>,
    ple: Option<PerLayer>,
    /// The published drafter, when loaded for speculative decoding.
    assistant: Option<assistant::Assistant>,
    /// The audio encoder, when the checkpoint has one and the load requested it.
    audio: Option<audio::AudioTower>,
    #[cfg(feature = "metal")]
    fused: Option<Fused>,
}

/// Compiled kernels and per-model constants of fused execution.
#[cfg(feature = "metal")]
struct Fused {
    kernels: tur_metal::kernels::Kernels,
    /// Each layer's rotary inverse frequencies, zero for unrotated pairs.
    rope: Vec<Tensor>,
    /// Weight of the unweighted value norm, as wide as the widest head.
    ones: Tensor,
    /// One zero frequency: the value norm runs the query/key norm kernel without rotation.
    unrotated: Tensor,
    /// Unit scale of residual updates that the layer scalar does not follow.
    one: Tensor,
}

/// The drafter checkpoint of a Gemma draft head: Gemma 4 has no head inside the target.
fn assistant_dir(head: &crate::models::DraftHead) -> Result<&Path> {
    if head.vocab.is_some() {
        return Err(invalid(
            "Gemma 4 assistants choose candidate tokens by cluster; omit the draft vocabulary",
        ));
    }
    head.model
        .as_deref()
        .ok_or_else(|| invalid("Gemma 4 drafting needs an assistant checkpoint (--draft-model)"))
}

/// Storage of the audio encoder's projections when this load includes it: requested,
/// declared by the checkpoint and, for a prepacked artifact, packed into it.
fn audio_precision(config: &Config, options: &LoadOptions) -> Option<crate::models::Precision> {
    let text_f32 = match &options.weights {
        Weights::Prepacked(manifest) if !manifest.audio => return None,
        Weights::Prepacked(manifest) => plan_is_f32(&manifest.plan),
        Weights::Uniform(format) => *format == crate::models::WeightFormat::F32,
        Weights::Plan(_) => false,
    };
    (options.audio && config.audio.is_some()).then(|| audio::precision(text_f32))
}

fn format_plan(config: &Config, options: &LoadOptions) -> Result<WeightPlan> {
    #[cfg(feature = "metal")]
    if options.execution == Execution::Fused
        && options.device.is_metal()
        && [config.text.head_dim, config.text.global_head_dim]
            .iter()
            .any(|dim| !matches!(dim, 64 | 128 | 256 | 512))
    {
        return Err(invalid(
            "Gemma 4 Metal attention supports head dimensions 64, 128, 256 and 512",
        ));
    }
    let plan = options.weights.plan(config)?;
    plan.validate(config)?;
    Ok(plan)
}

fn checkpoint(dir: &Path, config: &Config) -> Result<Checkpoint> {
    let (checkpoint, _) = Checkpoint::scan(weight_files(dir)?)?;
    let shapes = config.weight_shapes()?;
    for (name, shape) in &shapes {
        let entry = checkpoint
            .entries()
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        if &entry.shape != shape {
            return Err(invalid(format!(
                "{name} has shape {:?}, expected {shape:?}",
                entry.shape
            )));
        }
        if !matches!(
            entry.candle_dtype(name)?,
            DType::F32 | DType::BF16 | DType::F16
        ) {
            return Err(invalid("unsupported Gemma 4 tensor dtype"));
        }
    }
    for (name, entry) in checkpoint.entries() {
        if !name.starts_with(&format!("{}.", config.prefix)) || shapes.contains_key(name) {
            continue;
        }
        // Published checkpoints retain unused K/V matrices in the sharing suffix.
        let mut known = false;
        for i in 0..config.layers() {
            if config.owners[i] == i {
                continue;
            }
            let prefix = format!("{}.layers.{i}.self_attn", config.prefix);
            let expected = if name == &format!("{prefix}.k_norm.weight") {
                Some(vec![config.dim(i)])
            } else if name == &format!("{prefix}.k_proj.weight")
                || name == &format!("{prefix}.v_proj.weight")
            {
                Some(vec![
                    config.kv_heads(i) * config.dim(i),
                    config.hidden_size(),
                ])
            } else {
                None
            };
            if let Some(expected) = expected {
                if entry.shape != expected {
                    return Err(invalid(format!("invalid unused shared K/V tensor {name}")));
                }
                entry.candle_dtype(name)?;
                known = true;
                break;
            }
        }
        if !known {
            return Err(invalid(format!("unexpected text tensor {name}")));
        }
    }
    Ok(checkpoint)
}

impl Config {
    fn layers(&self) -> usize {
        self.text.num_hidden_layers
    }
    /// The last layer of `kind` that owns its K/V: what sharing layers and the assistant read.
    pub(super) fn last_owner(&self, kind: LayerKind) -> Option<usize> {
        (0..self.layers())
            .rev()
            .find(|&i| self.owners[i] == i && self.text.layer_types[i] == kind)
    }
}

impl artifact::ArtifactLayout for Config {
    fn tensor_shapes(&self) -> Result<std::collections::BTreeMap<String, Vec<usize>>> {
        self.weight_shapes()
    }
    fn embedding_tensor(&self) -> String {
        format!("{}.embed_tokens.weight", self.prefix)
    }
    fn companions(
        &self,
        plan: &WeightPlan,
    ) -> Result<std::collections::BTreeMap<String, artifact::Companion>> {
        let Some(settings) = &self.audio else {
            return Ok(std::collections::BTreeMap::new());
        };
        let precision = audio::precision(plan_is_f32(plan));
        Ok(audio::stored_tensors(
            settings,
            self.hidden_size(),
            precision,
        ))
    }
}

/// Whether every projection of `plan` stays F32, so the audio encoder does too.
fn plan_is_f32(plan: &WeightPlan) -> bool {
    plan.projections
        .values()
        .all(|&precision| precision == crate::models::Precision::F32)
}

/// Pack the Gemma 4 checkpoint at `conversion.source` by `plan` and write a prepacked
/// artifact, one tensor at a time. Per-layer embedding tables and norms are copied as
/// stored, the audio encoder is packed beside the text tower, and unused sharing-layer K/V
/// and vision tensors are left out. Rounding is
/// round-to-nearest: calibrated packing is not implemented for Gemma 4.
pub fn write_artifact(
    conversion: &Conversion<'_>,
    plan: &WeightPlan,
    packer: &Packer<'_>,
) -> Result<Manifest> {
    if packer.calibration.is_some() || packer.method != crate::models::Method::Rtn {
        return Err(invalid(
            "Gemma 4 artifacts support round-to-nearest packing only (--method rtn)",
        ));
    }
    let config = Config::from_json(&std::fs::read_to_string(
        conversion.source.join("config.json"),
    )?)?;
    let checkpoint = checkpoint(conversion.source, &config)?;
    artifact::write(conversion, &config, &checkpoint, plan, packer)
}
impl Layer {
    fn load(t: &Tensors, c: &Config, i: usize) -> Result<Self> {
        let p = format!("{}.layers.{i}", c.prefix);
        let projection = |s: &str| t.projection(&format!("{p}.{s}"));
        let norm = |s: &str| -> Result<Tensor> {
            Ok(t.take(&format!("{p}.{s}.weight"))?
                .to_dtype(DType::F32)?
                .to_device(&t.device)?)
        };
        let own = c.owners[i] == i;
        Ok(Self {
            moe: if c.text.enable_moe_block {
                Some(moe::Moe::load(
                    t,
                    &p,
                    c.text
                        .num_experts
                        .ok_or_else(|| invalid("missing expert count"))?,
                    c.text
                        .top_k_experts
                        .ok_or_else(|| invalid("missing top-k"))?,
                )?)
            } else {
                None
            },
            qkv: projection(if own {
                "self_attn.qkv_proj"
            } else {
                "self_attn.q_proj"
            })?,
            o: projection("self_attn.o_proj")?,
            q_norm: norm("self_attn.q_norm")?,
            k_norm: own.then(|| norm("self_attn.k_norm")).transpose()?,
            input_norm: norm("input_layernorm")?,
            post_attention: norm("post_attention_layernorm")?,
            pre_ff: norm("pre_feedforward_layernorm")?,
            post_ff: norm("post_feedforward_layernorm")?,
            scalar: t
                .take(&format!("{p}.layer_scalar"))?
                .to_dtype(DType::F32)?
                .to_device(&t.device)?,
            gate_up: projection("mlp.gate_up_proj")?,
            down: projection("mlp.down_proj")?,
            ple: (c.text.hidden_size_per_layer_input > 0)
                .then(|| -> Result<LayerInput> {
                    Ok(LayerInput {
                        gate: projection("per_layer_input_gate")?,
                        projection: projection("per_layer_projection")?,
                        norm: norm("post_per_layer_input_norm")?,
                    })
                })
                .transpose()?,
        })
    }
}

impl Gemma4 {
    /// Load only the validated text tower; multimodal inputs and native drafting are rejected.
    pub fn load(dir: &Path, options: &LoadOptions) -> Result<Self> {
        let mut config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if dir.join("tokenizer_config.json").exists() && dir.join("tokenizer.json").exists() {
            crate::models::generation::forbid_tokenizer_modalities(
                &mut config.forbidden,
                &std::fs::read_to_string(dir.join("tokenizer_config.json"))?,
                &std::fs::read_to_string(dir.join("tokenizer.json"))?,
            )?;
        }
        let plan = format_plan(&config, options)?;
        let (checkpoint, prepacked) = match &options.weights {
            Weights::Prepacked(manifest) => {
                let (checkpoint, found) = artifact::open_for(dir, &config)?;
                if found != **manifest {
                    return Err(invalid(
                        "prepacked artifact changed after its load was planned",
                    ));
                }
                (checkpoint, true)
            }
            _ if artifact::is_artifact(dir) => {
                return Err(invalid(
                    "directory holds a prepacked artifact; load it as Weights::Prepacked",
                ));
            }
            _ => (checkpoint(dir, &config)?, false),
        };
        let t = Tensors {
            checkpoint,
            prepacked,
            plan,
            projections: config
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: options.device.clone(),
        };
        let head = t.projection("lm_head")?;
        let embedding = t.embedding(
            &format!("{}.embed_tokens.weight", config.prefix),
            config.vocab_size(),
            config.hidden_size(),
        )?;
        let norm = t
            .take(&format!("{}.norm.weight", config.prefix))?
            .to_dtype(DType::F32)?
            .to_device(&options.device)?;
        let layers = (0..config.layers())
            .map(|i| Layer::load(&t, &config, i))
            .collect::<Result<Vec<_>>>()?;
        let ple = if config.text.hidden_size_per_layer_input > 0 {
            let name = format!("{}.embed_tokens_per_layer.weight", config.prefix);
            Some(PerLayer {
                table: t.column_tables(
                    &name,
                    config.text.vocab_size_per_layer_input,
                    config.text.hidden_size_per_layer_input,
                    config.layers(),
                    ple_dtype(&config, &t.plan)?,
                )?,
                projection: t
                    .projection(&format!("{}.per_layer_model_projection", config.prefix))?,
                norm: t
                    .take(&format!(
                        "{}.per_layer_projection_norm.weight",
                        config.prefix
                    ))?
                    .to_dtype(DType::F32)?
                    .to_device(&options.device)?,
            })
        } else {
            None
        };
        let assistant = options
            .mtp
            .as_ref()
            .map(|head| assistant::Assistant::load(assistant_dir(head)?, &config, options))
            .transpose()?;
        let audio = match (audio_precision(&config, options), &config.audio) {
            (Some(precision), Some(settings)) => {
                // Clip framing tokens are ordinary prompt tokens once the encoder exists.
                config
                    .forbidden
                    .retain(|id| *id != settings.begin && *id != settings.end);
                // Its own Metal queue: a clip's GPU work interleaves with text passes instead
                // of holding every later one behind it in first-in, first-out order.
                let device = match &options.device {
                    Device::Metal(_) => Device::new_metal(0)?,
                    device => device.clone(),
                };
                Some(audio::AudioTower::load(
                    dir,
                    &t.checkpoint,
                    settings,
                    config.hidden_size(),
                    precision,
                    &device,
                )?)
            }
            _ => None,
        };
        let (description, _, _) = Self::memory_requirements(&config, options, 1, &[1])?;
        Self {
            config,
            description,
            device: options.device.clone(),
            embedding,
            head,
            norm,
            layers,
            ple,
            assistant,
            audio,
            #[cfg(feature = "metal")]
            fused: None,
        }
        .with_execution(options.execution)
    }

    /// Select common Metal kernels or the F32 Candle reference, retaining the same weights.
    pub fn with_execution(self, execution: Execution) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let fused = if execution == Execution::Fused && self.device.is_metal() {
                Some(self.fused()?)
            } else {
                None
            };
            Ok(Self { fused, ..self })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = execution;
            Ok(self)
        }
    }
    #[cfg(feature = "metal")]
    fn fused(&self) -> Result<Fused> {
        let c = &self.config;
        let rope = (0..c.layers())
            .map(|i| {
                let frequencies = inverse_frequencies(c.dim(i), c.rope(i));
                Ok(Tensor::new(frequencies.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let widest = c.text.head_dim.max(c.text.global_head_dim);
        Ok(Fused {
            kernels: tur_metal::kernels::Kernels::new(&self.device)?,
            rope,
            ones: Tensor::ones(widest, DType::F32, &self.device)?,
            unrotated: Tensor::zeros(1, DType::F32, &self.device)?,
            one: Tensor::ones(1, DType::F32, &self.device)?,
        })
    }
    #[cfg(feature = "metal")]
    fn fused_kernels(&self) -> Option<&tur_metal::kernels::Kernels> {
        self.fused.as_ref().map(|f| &f.kernels)
    }
    /// Device holding the immutable weights.
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Validated Gemma 4 settings.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// How a clip appears in prompts, when the audio encoder is loaded.
    pub fn audio_prompt(&self) -> Option<AudioPrompt> {
        let settings = self
            .config
            .audio
            .as_ref()
            .filter(|_| self.audio.is_some())?;
        Some(AudioPrompt {
            placeholder: settings.token,
            begin: settings.begin,
            end: settings.end,
            slot: self.config.text.pad_token_id.unwrap_or(0),
        })
    }
    /// Soft tokens a clip occupies in the prompt, or an error outside 40 ms to 30 s.
    pub fn audio_tokens(&self, clip: &Audio) -> Result<usize> {
        if self.audio.is_none() {
            return Err(invalid("this model has no audio encoder loaded"));
        }
        Some(audio::soft_tokens(clip.samples().len()))
            .filter(|&n| n > 0 && n <= AudioSettings::MAX_TOKENS)
            .ok_or_else(|| {
                invalid(format!(
                    "Gemma 4 audio clips must be between 40 ms and 30 s; this one is {:.2} s",
                    clip.duration().as_secs_f64()
                ))
            })
    }
    /// F32 `[audio_tokens, hidden]` soft tokens of a clip on the model device. The encoder
    /// has its own Metal queue, so this waits for its work and copies the rows across;
    /// call it off the thread that runs text passes when they should keep running.
    pub fn encode_audio(&self, clip: &Audio) -> Result<Tensor> {
        let rows = self
            .audio
            .as_ref()
            .ok_or_else(|| invalid("this model has no audio encoder loaded"))?
            .encode(clip)?;
        if rows.device().same_device(&self.device) {
            return Ok(rows);
        }
        Ok(rows.to_device(&Device::Cpu)?.to_device(&self.device)?)
    }
    /// Scaled token embeddings of the packed spans, with embedded inputs in place of
    /// their rows. Rows of sequences that failed validation are skipped with them.
    fn embed(&self, batch: &PackedForward<'_>, spans: &[Span<'_>], ids: &Tensor) -> Result<Tensor> {
        let x =
            (self.embedding.rows(ids, &self.head)? * (self.config.hidden_size() as f64).sqrt())?;
        if batch.embeddings.is_empty() {
            return Ok(x);
        }
        let mut pieces = Vec::new();
        let mut cursor = 0;
        for input in batch.embeddings {
            let Some(span) = spans.iter().find(|s| {
                s.sequence.tokens.start <= input.tokens.start
                    && input.tokens.end <= s.sequence.tokens.end
            }) else {
                continue;
            };
            let start = span.start + (input.tokens.start - span.sequence.tokens.start);
            if start < cursor
                || input.rows.dims2()? != (input.tokens.len(), self.config.hidden_size())
            {
                return Err(invalid("embedded input rows do not match the model width"));
            }
            if start > cursor {
                pieces.push(x.narrow(0, cursor, start - cursor)?);
            }
            pieces.push(input.rows.to_device(&self.device)?);
            cursor = start + input.tokens.len();
        }
        let rows = x.dim(0)?;
        if cursor < rows {
            pieces.push(x.narrow(0, cursor, rows - cursor)?);
        }
        Ok(Tensor::cat(&pieces, 0)?)
    }
    /// Whether common Metal kernels or Candle reference operations execute passes.
    pub fn execution(&self) -> Execution {
        #[cfg(feature = "metal")]
        if self.fused.is_some() {
            return Execution::Fused;
        }
        Execution::Reference
    }
    /// Resident bytes held by packed projection matrices.
    pub fn projection_bytes(&self) -> usize {
        let mut bytes = self.head.resident_bytes()
            + self
                .ple
                .as_ref()
                .map_or(0, |p| p.projection.resident_bytes());
        for layer in &self.layers {
            bytes += [&layer.qkv, &layer.o, &layer.gate_up, &layer.down]
                .iter()
                .map(|p| p.resident_bytes())
                .sum::<usize>();
            bytes += layer.ple.as_ref().map_or(0, |p| {
                p.gate.resident_bytes() + p.projection.resident_bytes()
            });
            bytes += layer.moe.as_ref().map_or(0, moe::Moe::projection_bytes);
        }
        bytes
    }
    /// Resident bytes loaded into the device.
    pub fn weight_bytes(&self) -> usize {
        let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let mut total = self.embedding.bytes() + self.head.resident_bytes() + bytes(&self.norm);
        if let Some(ple) = &self.ple {
            total += ple.table.iter().map(bytes).sum::<usize>()
                + bytes(&ple.norm)
                + ple.projection.resident_bytes();
        }
        for l in &self.layers {
            total += [&l.qkv, &l.o, &l.gate_up, &l.down]
                .iter()
                .map(|p| p.resident_bytes())
                .sum::<usize>();
            total += [
                &l.q_norm,
                &l.input_norm,
                &l.post_attention,
                &l.pre_ff,
                &l.post_ff,
                &l.scalar,
            ]
            .iter()
            .map(|t| bytes(t))
            .sum::<usize>()
                + l.k_norm.as_ref().map_or(0, bytes);
            if let Some(ple) = &l.ple {
                total +=
                    ple.gate.resident_bytes() + ple.projection.resident_bytes() + bytes(&ple.norm);
            }
            if let Some(moe) = &l.moe {
                total += moe.bytes();
            }
        }
        total
            + self
                .assistant
                .as_ref()
                .map_or(0, assistant::Assistant::bytes)
            + self.audio.as_ref().map_or(0, audio::AudioTower::bytes)
    }

    /// Plan packed weights, per-layer embedding storage and peak activations before loading.
    pub fn memory_requirements(
        config: &Config,
        options: &LoadOptions,
        tokens: usize,
        contexts: &[usize],
    ) -> Result<(ModelDescription, usize, usize)> {
        if options.expert_cache.is_some() {
            return Err(invalid("the expert cache reads MXFP4 banks (gpt-oss) only"));
        }
        let plan = format_plan(config, options)?;
        let audio = audio_precision(config, options);
        let description = config.description(
            format!("{}#{}", options.revision, options.weights.tag()),
            options.kv,
            audio.is_some(),
        )?;
        let projections = config.projection_layout()?;
        let mut accounted: BTreeSet<String> = projections
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(|(n, _)| n))
            .collect();
        accounted.insert(format!("{}.embed_tokens.weight", config.prefix));
        let mut weights = plan.bytes(config)?;
        if let (Some(precision), Some(settings)) = (audio, &config.audio) {
            weights = audio::planned_bytes(settings, config.hidden_size(), precision)?
                .checked_add(weights)
                .ok_or_else(|| invalid("Gemma 4 weight bytes overflow"))?;
        }
        if let Some(head) = &options.mtp {
            weights = assistant::Assistant::planned_bytes(assistant_dir(head)?, config, options)?
                .checked_add(weights)
                .ok_or_else(|| invalid("Gemma 4 weight bytes overflow"))?;
        }
        for (name, shape) in config.weight_shapes()? {
            if accounted.contains(&name) {
                continue;
            }
            let dtype = if name.ends_with("embed_tokens_per_layer.weight") {
                ple_dtype(config, &plan)?
            } else {
                DType::F32
            };
            weights = checked_elements(&shape)?
                .checked_mul(dtype.size_in_bytes())
                .and_then(|b| weights.checked_add(b))
                .ok_or_else(|| invalid("Gemma 4 weight bytes overflow"))?;
        }
        #[cfg(feature = "metal")]
        let attention = if options.execution == Execution::Fused && options.device.is_metal() {
            let kernels = tur_metal::kernels::Kernels::new(&options.device)?;
            Some(attention_workspace(
                config, &kernels, options.kv, tokens, contexts,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "metal"))]
        let attention = None;
        Ok((
            description,
            weights,
            workspace(config, tokens, contexts, attention)?,
        ))
    }
    fn linear(&self, w: &PackedLinear, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = self.fused_kernels() {
            return k.linear(w, x);
        }
        w.forward(x)
    }
    fn rms(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = self.fused_kernels() {
            let shape = x.shape().clone();
            let width = x.dim(candle_core::D::Minus1)?;
            return Ok(k
                .rms_norm(
                    &x.reshape((x.elem_count() / width, width))?,
                    w,
                    self.config.text.rms_norm_eps,
                )?
                .reshape(shape)?);
        }
        math::rms(x, w, self.config.text.rms_norm_eps)
    }
    fn inputs(&self, ids: &Tensor, embedded: &Tensor) -> Result<Option<Tensor>> {
        let Some(ple) = &self.ple else {
            return Ok(None);
        };
        let c = &self.config;
        let shape = (
            ids.elem_count(),
            c.layers(),
            c.text.hidden_size_per_layer_input,
        );
        let context = (self.linear(&ple.projection, embedded)? / (c.hidden_size() as f64).sqrt())?
            .reshape(shape)?;
        let context = self.rms(&context, &ple.norm)?;
        let gathered = ple
            .table
            .iter()
            .map(|table| table.index_select(ids, 0))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let identity = (Tensor::stack(&gathered, 1)?.to_dtype(DType::F32)?
            * (c.text.hidden_size_per_layer_input as f64).sqrt())?
        .reshape(shape)?;
        Ok(Some(
            ((context + identity)? * std::f64::consts::FRAC_1_SQRT_2)?,
        ))
    }
    fn run(
        &self,
        batch: &PackedForward<'_>,
        spans: &[Span<'_>],
        trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Vec<SequenceStep>> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for s in spans {
            tokens.extend_from_slice(&batch.tokens[s.sequence.tokens.clone()]);
            positions.extend_from_slice(&batch.positions[s.sequence.tokens.clone()]);
        }
        let ids = Tensor::new(tokens.as_slice(), &self.device)?;
        // Per-layer inputs read the embedded rows too, and the slot token's identity row.
        let x = self.embed(batch, spans, &ids)?;
        let inputs = self.inputs(&ids, &x)?;
        let pass = Pass {
            pool: batch.pool,
            spans,
            positions: &positions,
        };
        #[cfg(feature = "metal")]
        let hidden = match &self.fused {
            Some(fused) => self.run_fused(fused, &pass, x, inputs, trace)?,
            None => self.run_reference(&pass, x, inputs.as_ref(), trace)?,
        };
        #[cfg(not(feature = "metal"))]
        let hidden = self.run_reference(&pass, x, inputs.as_ref(), trace)?;
        spans
            .iter()
            .map(|s| {
                let rows = s
                    .sequence
                    .logits
                    .iter()
                    .map(|r| {
                        u32::try_from(s.start + r - s.sequence.tokens.start)
                            .map_err(|_| invalid("logit row overflow"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let logits = if rows.is_empty() {
                    Tensor::zeros((0, self.config.vocab_size()), DType::F32, &self.device)?
                } else {
                    let raw = self.linear(
                        &self.head,
                        &hidden.index_select(&Tensor::new(rows.as_slice(), &self.device)?, 0)?,
                    )?;
                    match self.config.text.final_logit_softcapping {
                        Some(cap) => ((raw / cap)?.tanh()? * cap)?,
                        None => raw,
                    }
                };
                Ok(SequenceStep {
                    logits,
                    hidden: s
                        .sequence
                        .hidden
                        .then(|| hidden.narrow(0, s.start, s.len))
                        .transpose()?,
                })
            })
            .collect()
    }
    /// `gelu(gate) * up` over the halves of a fused `gate | up` projection.
    fn gelu_product(&self, gate_up: &Tensor) -> Result<Tensor> {
        let width = gate_up.dim(1)? / 2;
        let (gate, up) = (
            gate_up.narrow(1, 0, width)?,
            gate_up.narrow(1, width, width)?,
        );
        #[cfg(feature = "metal")]
        if let Some(k) = self.fused_kernels() {
            return k.gelu_mul(&gate, &up);
        }
        Ok((gate.gelu()? * up)?)
    }
    /// Gated feed-forward update of one layer from residual `x` and its pre-norm `h`.
    fn mlp(&self, l: &Layer, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let activated = self.gelu_product(&self.linear(&l.gate_up, h)?)?;
        let dense = self.linear(&l.down, &activated)?;
        match &l.moe {
            Some(moe) => moe.forward(self, x, &dense),
            None => Ok(dense),
        }
    }
    /// Reference decoder over Candle operations; returns the final-normed hidden rows.
    fn run_reference(
        &self,
        pass: &Pass<'_, '_>,
        mut x: Tensor,
        inputs: Option<&Tensor>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        for (i, l) in self.layers.iter().enumerate() {
            let h = self.rms(&x, &l.input_norm)?;
            let attention = self.reference_attention(l, i, &h, pass)?;
            x = (x + self.rms(&attention, &l.post_attention)?)?;
            let h = self.rms(&x, &l.pre_ff)?;
            let mixed = self.mlp(l, &x, &h)?;
            x = (x + self.rms(&mixed, &l.post_ff)?)?;
            if let (Some(ple), Some(inputs)) = (&l.ple, inputs) {
                let gate = self.linear(&ple.gate, &x)?.gelu()?;
                let signal = (gate * inputs.narrow(1, i, 1)?.squeeze(1)?)?;
                x = (&x + self.rms(&self.linear(&ple.projection, &signal)?, &ple.norm)?)?;
            }
            x = x.broadcast_mul(&l.scalar)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        self.rms(&x, &self.norm)
    }
    fn reference_attention(
        &self,
        layer: &Layer,
        index: usize,
        hidden: &Tensor,
        pass: &Pass<'_, '_>,
    ) -> Result<Tensor> {
        let (pool, spans) = (pass.pool, pass.spans);
        let config = &self.config;
        let tokens = hidden.dim(0)?;
        let dim = config.dim(index);
        let heads = config.text.num_attention_heads;
        let (cos, sin) = rotary_tables(pass.positions, dim, config.rope(index), &self.device)?;
        let projected = self.linear(&layer.qkv, hidden)?;
        let query = projected
            .narrow(1, 0, heads * dim)?
            .reshape((tokens, heads, dim))?;
        let query = math::rotary(&self.rms(&query, &layer.q_norm)?, &cos, &sin)?;
        let store = pool.attention(config.state_layer(index))?;
        if let Some(norm) = &layer.k_norm {
            let width = config.kv_heads(index) * dim;
            let shape = (tokens, config.kv_heads(index), dim);
            let raw = projected.narrow(1, heads * dim, width)?.reshape(shape)?;
            let value = if config.unified_kv(index) {
                raw.clone()
            } else {
                projected
                    .narrow(1, heads * dim + width, width)?
                    .reshape(shape)?
            };
            let value = math::rms(
                &value,
                &Tensor::ones(dim, DType::F32, &self.device)?,
                config.text.rms_norm_eps,
            )?;
            let key = math::rotary(&self.rms(&raw, norm)?, &cos, &sin)?;
            scatter(store, spans, config.windowed(index), &key, &value)?;
        }
        let window = config.windowed(index).then_some(config.text.sliding_window);
        let outputs = spans
            .iter()
            .map(|s| {
                let pages = table(&s.addresses, config.windowed(index));
                let (keys, values) = store.gather(pages, s.past + s.len)?;
                math::attention(
                    &query.narrow(0, s.start, s.len)?,
                    &keys,
                    &values,
                    s.past,
                    1.0,
                    window,
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.linear(
            &layer.o,
            &Tensor::cat(&outputs, 0)?.reshape((tokens, heads * dim))?,
        )
    }
}
#[cfg(feature = "metal")]
impl Gemma4 {
    /// Fused decoder: every layer runs once over the packed batch on Tur's kernels, with
    /// residual updates fused into the following norm. Returns the final-normed rows.
    fn run_fused(
        &self,
        f: &Fused,
        pass: &Pass<'_, '_>,
        mut x: Tensor,
        inputs: Option<Tensor>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let k = &f.kernels;
        let eps = self.config.text.rms_norm_eps;
        let tokens = x.dim(0)?;
        let paging = self.paging(k, pass.pool, pass.spans, tokens)?;
        let positions = pass
            .positions
            .iter()
            .map(|&p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
            .collect::<Result<Vec<_>>>()?;
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        // Per-layer inputs as `[tokens, layers × width]`; each layer reads its columns.
        let inputs = inputs
            .map(|t| {
                t.reshape((
                    tokens,
                    self.config.layers() * self.config.text.hidden_size_per_layer_input,
                ))
            })
            .transpose()?;
        let first = self
            .layers
            .first()
            .ok_or_else(|| invalid("model has no layers"))?;
        let mut h = k.rms_norm(&x, &first.input_norm, eps)?;
        for (i, l) in self.layers.iter().enumerate() {
            let attention = self.fused_attention(f, l, i, &h, &positions, pass.pool, &paging)?;
            let (sum, normed) = k.residual_rms_norm(
                &x,
                &attention,
                &l.post_attention,
                &f.one,
                Some(&l.pre_ff),
                eps,
            )?;
            let normed = normed.ok_or_else(|| invalid("missing feed-forward input"))?;
            let mixed = self.mlp(l, &sum, &normed)?;
            // The last update fuses with the final norm.
            let next = self.layers.get(i + 1).map_or(&self.norm, |n| &n.input_norm);
            (x, h) = self.fused_update(f, l, i, &sum, &mixed, inputs.as_ref(), next)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        Ok(h)
    }
    /// Feed-forward residual, per-layer input injection and layer scalar, fused with the
    /// next norm: returns the next residual stream and its normed rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per operand of the fused residual update"
    )]
    fn fused_update(
        &self,
        fused: &Fused,
        layer: &Layer,
        index: usize,
        x: &Tensor,
        mixed: &Tensor,
        inputs: Option<&Tensor>,
        next: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let k = &fused.kernels;
        let eps = self.config.text.rms_norm_eps;
        let (x, h) = match (&layer.ple, inputs) {
            (Some(ple), Some(inputs)) => {
                let (x, _) =
                    k.residual_rms_norm(x, mixed, &layer.post_ff, &fused.one, None, eps)?;
                let width = self.config.text.hidden_size_per_layer_input;
                let column = index
                    .checked_mul(width)
                    .ok_or_else(|| invalid("per-layer input column overflow"))?;
                let signal =
                    k.gelu_mul(&k.linear(&ple.gate, &x)?, &inputs.narrow(1, column, width)?)?;
                let injected = k.linear(&ple.projection, &signal)?;
                k.residual_rms_norm(&x, &injected, &ple.norm, &layer.scalar, Some(next), eps)?
            }
            _ => k.residual_rms_norm(x, mixed, &layer.post_ff, &layer.scalar, Some(next), eps)?,
        };
        Ok((x, h.ok_or_else(|| invalid("missing next layer input"))?))
    }
    /// Projects `q | k | v` once and norms and rotates queries and keys in place from it;
    /// K/V owners store their normed values before attending.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input the attention dispatch binds"
    )]
    fn fused_attention(
        &self,
        fused: &Fused,
        layer: &Layer,
        index: usize,
        hidden: &Tensor,
        positions: &Tensor,
        pool: &StatePool,
        paging: &Paging,
    ) -> Result<Tensor> {
        let config = &self.config;
        let k = &fused.kernels;
        let paging = paging.layer(config.windowed(index))?;
        let eps = config.text.rms_norm_eps;
        let dim = config.dim(index);
        let heads = config.text.num_attention_heads;
        let rope = fused
            .rope
            .get(index)
            .ok_or_else(|| invalid("missing rotary frequencies"))?;
        let at = |count: usize, col_offset: usize| tur_metal::kernels::Heads {
            count,
            dim,
            head_stride: dim,
            col_offset,
        };
        let projected = k.linear(&layer.qkv, hidden)?;
        let query = k.qk_norm_rope(
            &projected,
            at(heads, 0),
            &layer.q_norm,
            positions,
            rope,
            eps,
        )?;
        let store = pool.attention(config.state_layer(index))?;
        if let Some(norm) = &layer.k_norm {
            let kv_heads = config.kv_heads(index);
            let key_col = heads * dim;
            // Unified K=V values are the raw key projection, read before key norm and RoPE.
            let value_col = if config.unified_kv(index) {
                key_col
            } else {
                key_col + kv_heads * dim
            };
            let key = k.qk_norm_rope(
                &projected,
                at(kv_heads, key_col),
                norm,
                positions,
                rope,
                eps,
            )?;
            let value = k.qk_norm_rope(
                &projected,
                at(kv_heads, value_col),
                &fused.ones.narrow(0, 0, dim)?,
                positions,
                &fused.unrotated,
                eps,
            )?;
            let width = kv_heads * dim;
            k.kv_write(
                &key.reshape((key.dim(0)?, width))?,
                at(kv_heads, 0),
                &value.reshape((value.dim(0)?, width))?,
                at(kv_heads, 0),
                store,
                paging,
            )?;
        }
        let window = config.windowed(index).then_some(config.text.sliding_window);
        k.linear(
            &layer.o,
            &k.attention_ungated(&query, store, paging, window, 1.0)?,
        )
    }
    /// Page addressing of this pass for full and windowed layers.
    fn paging(
        &self,
        k: &tur_metal::kernels::Kernels,
        pool: &StatePool,
        spans: &[Span<'_>],
        tokens: usize,
    ) -> Result<Paging> {
        Paging::new(
            k,
            &self.description,
            pool,
            spans.iter().map(|s| (&s.addresses, s.past, s.len)),
            tokens,
        )
    }
}

impl Gemma4 {
    fn check<'a>(
        &self,
        batch: &PackedForward<'_>,
        sequence: &'a SequenceForward<'a>,
        start: usize,
    ) -> Result<Span<'a>> {
        let tokens = batch
            .tokens
            .get(sequence.tokens.clone())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("empty or invalid sequence range"))?;
        let positions = batch
            .positions
            .get(sequence.tokens.clone())
            .ok_or_else(|| invalid("missing packed positions"))?;
        let past = sequence.state.past();
        let end = past
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("position overflow"))?;
        if end > self.config.max_position()
            || sequence.state.tokens() != tokens.len()
            || positions
                .iter()
                .enumerate()
                .any(|(i, &p)| past.checked_add(i) != Some(p))
            || tokens.iter().any(|id| {
                *id as usize >= self.config.vocab_size() || self.config.forbidden.contains(id)
            })
            || sequence.logits.iter().any(|r| !sequence.tokens.contains(r))
            || sequence.logits.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(invalid(
                "invalid Gemma 4 positions, tokens, state or logit rows",
            ));
        }
        let addresses = sequence.state.resolve(batch.pool)?;
        let pages = self.description.pages_for(end);
        if (self.description.has_full_attention() && addresses.pages.len() != pages)
            || (self.description.window().is_some() && addresses.windows.len() != pages)
        {
            return Err(invalid("sequence pages do not cover new tokens"));
        }
        Ok(Span {
            start,
            len: tokens.len(),
            past,
            addresses,
            sequence,
        })
    }
    /// Run a fresh sequence, with `embeddings` replacing rows of its token embeddings, and
    /// return per-layer residual streams and selected logits.
    pub fn trace(
        &self,
        tokens: &[u32],
        rows: &[usize],
        embeddings: &[EmbeddedInput],
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let pool = StatePool::new(
            Arc::new(self.description.clone()),
            &self.device,
            PoolConfig {
                pages: self.description.pages_for(tokens.len()).max(1),
                window_pages: self.description.pages_for(tokens.len()).max(1),
                recurrent_slots: 1,
            },
        )?;
        let mut reservation = ModelState::new(&pool).reserve(tokens.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let sequence = SequenceForward {
            tokens: 0..tokens.len(),
            state: &binding,
            logits: rows,
            hidden: false,
        };
        let sequences = [sequence];
        let batch = PackedForward {
            embeddings,
            tokens,
            positions: &positions,
            sequences: &sequences,
            pool: &pool,
        };
        let spans = [self.check(&batch, &sequences[0], 0)?];
        let mut trace = Vec::new();
        let step = self
            .run(&batch, &spans, Some(&mut trace))?
            .pop()
            .ok_or_else(|| invalid("missing Gemma 4 output"))?;
        Ok((trace, step.logits))
    }
}
impl Gemma4 {
    /// The last committed position of a draft sequence: the token after it and its
    /// hidden state, at the following position, over that sequence's pages.
    fn draft_row(
        &self,
        batch: &DraftForward<'_>,
        sequence: &DraftSequence<'_>,
    ) -> Result<(assistant::DraftRow, Addresses)> {
        let last = sequence
            .tokens
            .end
            .checked_sub(1)
            .filter(|&l| l >= sequence.tokens.start)
            .ok_or_else(|| invalid("empty draft sequence"))?;
        let token = *batch
            .tokens
            .get(last)
            .ok_or_else(|| invalid("draft token out of range"))?;
        let position = batch
            .positions
            .get(last)
            .and_then(|p| p.checked_add(1))
            .ok_or_else(|| invalid("draft position out of range"))?;
        let rows = sequence.hidden.dim(0)?;
        if token as usize >= self.config.vocab_size()
            || self.config.forbidden.contains(&token)
            || position >= self.config.max_position()
            || rows != sequence.tokens.len()
            || sequence.hidden.dim(1)? != self.config.hidden_size()
        {
            return Err(invalid(
                "invalid Gemma 4 draft token, position or hidden state",
            ));
        }
        let addresses = sequence.state.resolve(batch.pool)?;
        let pages = self.description.pages_for(position);
        if (self.description.has_full_attention() && addresses.pages.len() < pages)
            || (self.description.window().is_some() && addresses.windows.len() < pages)
        {
            return Err(invalid("draft pages do not cover the committed keys"));
        }
        Ok((
            assistant::DraftRow {
                token,
                hidden: sequence.hidden.narrow(0, rows - 1, 1)?,
                position,
                drafts: sequence.drafts,
            },
            addresses,
        ))
    }
}

/// Addressing shared by every layer of one packed pass.
struct Pass<'a, 'b> {
    pool: &'a StatePool,
    spans: &'a [Span<'b>],
    positions: &'a [usize],
}
struct Span<'a> {
    start: usize,
    len: usize,
    past: usize,
    addresses: Addresses,
    sequence: &'a SequenceForward<'a>,
}
/// The page table a layer's keys live in: window pages for windowed layers.
pub(super) fn table(addresses: &Addresses, windowed: bool) -> &[u32] {
    if windowed {
        &addresses.windows
    } else {
        &addresses.pages
    }
}
fn scatter(
    store: &tur_metal::pool::AttentionPool,
    spans: &[Span<'_>],
    windowed: bool,
    key: &Tensor,
    value: &Tensor,
) -> Result<()> {
    for s in spans {
        store.scatter(
            table(&s.addresses, windowed),
            s.past,
            &key.narrow(0, s.start, s.len)?,
            &value.narrow(0, s.start, s.len)?,
        )?;
    }
    Ok(())
}
/// Per-layer embedding tables stay F32 under an all-F32 plan, from a checkpoint or an
/// artifact alike, and BF16 under any packed plan.
fn ple_dtype(config: &Config, plan: &WeightPlan) -> Result<DType> {
    Ok(
        if *plan == WeightPlan::uniform(config, crate::models::WeightFormat::F32)? {
            DType::F32
        } else {
            DType::BF16
        },
    )
}
/// Inverse rotary frequencies over `dim / 2` pairs; proportional rotation leaves the pairs past
/// its rotated fraction at zero frequency.
fn inverse_frequencies(dim: usize, rope: &super::config::Rope) -> Vec<f32> {
    let rotated = (dim as f64 * rope.partial_rotary_factor) as usize / 2;
    (0..dim / 2)
        .map(|i| {
            if i < rotated {
                1.0 / (rope.rope_theta as f32).powf((2 * i) as f32 / dim as f32)
            } else {
                0.0
            }
        })
        .collect()
}
fn rotary_tables(
    positions: &[usize],
    dim: usize,
    rope: &super::config::Rope,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let frequencies = inverse_frequencies(dim, rope);
    let mut cos = Vec::with_capacity(positions.len() * dim / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for &position in positions {
        for &frequency in &frequencies {
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (positions.len(), dim / 2), device)?,
        Tensor::from_vec(sin, (positions.len(), dim / 2), device)?,
    ))
}
#[cfg(feature = "metal")]
fn attention_workspace(
    config: &Config,
    kernels: &tur_metal::kernels::Kernels,
    kv: tur_core::execution::KvLayout,
    tokens: usize,
    contexts: &[usize],
) -> usize {
    let visible = contexts.iter().copied().max().unwrap_or(0);
    (0..config.layers())
        .map(|i| {
            kernels.attention_scratch_bytes(
                tokens,
                config.text.num_attention_heads,
                config.kv_heads(i),
                config.dim(i),
                visible,
                kv.format,
            )
        })
        .max()
        .unwrap_or(0)
}
fn workspace(
    config: &Config,
    tokens: usize,
    contexts: &[usize],
    split: Option<usize>,
) -> Result<usize> {
    if tokens == 0 {
        return Ok(0);
    }

    // Bound live hidden/projection/PLE buffers, plus the reference score block or split-K
    // partials. Every temporary scales with admitted tokens and contexts.
    let width = config.text.intermediate_size
        * if config.text.use_double_wide_mlp {
            2
        } else {
            1
        };
    let ple = config
        .layers()
        .checked_mul(config.text.hidden_size_per_layer_input)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| invalid("per-layer input workspace overflow"))?;
    let activations = checked_elements(&[
        tokens,
        config.hidden_size() * 8
            + width * 4
            + ple
            + config.text.num_attention_heads * config.text.global_head_dim * 6,
        4,
    ])?;
    let attention = match split {
        Some(bytes) => bytes,
        None => checked_elements(&[
            tokens.min(256),
            contexts.iter().copied().max().unwrap_or(0),
            config.text.num_attention_heads,
            8,
        ])?,
    };
    let routed = if config.text.enable_moe_block {
        let width = checked_elements(&[config.hidden_size(), 2])?
            .checked_add(checked_elements(&[
                config.text.moe_intermediate_size.unwrap_or(0),
                4,
            ])?)
            .ok_or_else(|| invalid("expert workspace width overflow"))?;
        checked_elements(&[tokens, config.text.top_k_experts.unwrap_or(0), width, 4])?
    } else {
        0
    };
    activations
        .checked_add(attention)
        .and_then(|n| n.checked_add(routed))
        .ok_or_else(|| invalid("Gemma 4 workspace overflow"))
}
impl HybridModel for Gemma4 {
    fn description(&self) -> ModelDescription {
        self.description.clone()
    }
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
        let contexts = batch
            .sequences
            .iter()
            .map(|s| {
                s.state
                    .past()
                    .checked_add(s.tokens.len())
                    .ok_or_else(|| invalid("context overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "metal")]
        let attention = self.fused_kernels().map(|kernels| {
            attention_workspace(
                &self.config,
                kernels,
                self.description.kv,
                batch.tokens.len(),
                &contexts,
            )
        });
        #[cfg(not(feature = "metal"))]
        let attention = None;
        workspace(&self.config, batch.tokens.len(), &contexts, attention)
    }
    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        self.fused_kernels().cloned()
    }
    fn draft(&self, batch: &DraftForward<'_>) -> Result<Vec<Result<Option<Tensor>>>> {
        let assistant = self
            .assistant
            .as_ref()
            .ok_or_else(|| invalid("model has no draft head"))?;
        if batch.pool.description() != &self.description
            || !batch.pool.device().same_device(&self.device)
        {
            return Err(invalid("state pool belongs to another model or device"));
        }
        let mut out: Vec<Option<Result<Option<Tensor>>>> =
            batch.sequences.iter().map(|_| None).collect();
        let (mut rows, mut pages, mut live) = (Vec::new(), Vec::new(), Vec::new());
        for (i, sequence) in batch.sequences.iter().enumerate() {
            if sequence.drafts == 0 {
                out[i] = Some(Ok(None));
                continue;
            }
            match self.draft_row(batch, sequence) {
                Ok((row, table)) => {
                    rows.push(row);
                    pages.push(table);
                    live.push(i);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        if !rows.is_empty() {
            let pass = assistant::DraftPass {
                pool: batch.pool,
                rows: &rows,
                pages: &pages,
            };
            for (i, logits) in live
                .into_iter()
                .zip(self.assistant_logits(assistant, &pass)?)
            {
                out[i] = Some(Ok(Some(logits)));
            }
        }
        Ok(out
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("sequence was not drafted"))))
            .collect())
    }
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>> {
        if batch.pool.description() != &self.description
            || !batch.pool.device().same_device(&self.device)
        {
            return Err(invalid("state pool belongs to another model or device"));
        }
        let mut out: Vec<Option<Result<SequenceStep>>> =
            batch.sequences.iter().map(|_| None).collect();
        let mut spans = Vec::new();
        let mut live = Vec::new();
        let mut start = 0;
        for (i, sequence) in batch.sequences.iter().enumerate() {
            match self.check(batch, sequence, start) {
                Ok(s) => {
                    start += s.len;
                    spans.push(s);
                    live.push(i);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        if !spans.is_empty() {
            for (i, step) in live.into_iter().zip(self.run(batch, &spans, None)?) {
                out[i] = Some(Ok(step));
            }
        }
        Ok(out
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("sequence was not executed"))))
            .collect())
    }
}
