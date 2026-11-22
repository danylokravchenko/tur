//! Inkling 2 equations over the common packed weights and persistent state pools.
#[cfg(feature = "metal")]
mod fused;
mod reference;

use super::config::{Config, ExpertFormat, PREFIX, Stored};
use crate::{
    hybrid::{
        Addresses, Boundaries, HybridModel, PackedForward, SequenceForward, SequenceStep, StatePool,
    },
    models::{
        Execution, LoadOptions, Weights, artifact,
        checkpoint::{Checkpoint, weight_files},
        weights::{Embeddings, Tensors, WeightLayout},
    },
};
use candle_core::{DType, Device, Tensor};
use std::{path::Path, sync::Arc};
use tur_core::progress::Stage;
use tur_core::{
    Result,
    execution::{ModelDescription, checked_elements, invalid},
};
use tur_metal::{packed::PackedLinear, pool::PoolConfig, state::ModelState};

/// Attention weights of one layer. `query` stacks every row the query input feeds:
/// `q | output gate | relative states | sink logits`; `key` and `value` add the dynamic
/// convolution taps after their rows in full attention layers.
struct Attention {
    query: PackedLinear,
    key: PackedLinear,
    value: PackedLinear,
    output: PackedLinear,
    /// `[heads * head_dim]` and `[heads * d_rel]` learned scales before the L2 clips.
    q_scale: Tensor,
    rel_scale: Tensor,
    /// `[d_rel, extent]` bias of each relative state component by backward distance.
    rel_proj: Tensor,
    /// `[kv_heads * head_dim, taps]` convolution taps by lag (0 = the token itself).
    key_taps: Tensor,
    value_taps: Tensor,
}

/// Norms before the attention: one for sliding layers, one per input for full layers.
enum InputNorms {
    Shared(Tensor),
    /// `[3, hidden]` query, key and value gains.
    Separate(Tensor),
}

/// The query, key, value and MLP message gates of a looped layer, stacked so that several
/// run as one batched product. Gate `i` mixes a message of the previous loop pass into
/// its input: `(1 - tanh(forget)) * x + tanh(gain) * message`, both gates of a rank-`r`
/// projection of the normalized pair `[x, message]`.
struct MessageGates {
    /// `[4, 2 * hidden, rank]`, transposed for `pair @ down`.
    down: Tensor,
    /// `[4, rank, 2 * hidden]`: forget columns, then gain columns.
    up: Tensor,
}

struct Experts {
    /// `[experts, hidden]` F32 router and its selection-only bias.
    router: Tensor,
    bias: Tensor,
    gate_up: PackedLinear,
    down: PackedLinear,
    /// `[experts, 2]`: factors of the `gate_up` and `down` weights not multiplied into
    /// them (an NVFP4 checkpoint's second-level scale on packed banks), else ones.
    factors: Tensor,
}

struct Layer {
    /// `[1 or 3, hidden]` queries reading the attention input from the block states;
    /// layer 0 reads its input directly.
    attention_queries: Option<Tensor>,
    mlp_query: Tensor,
    norms: InputNorms,
    attention: Attention,
    post_norm: Tensor,
    /// `[hidden, taps]` taps of the convolutions after the attention and the experts.
    attention_taps: Tensor,
    mlp_taps: Tensor,
    experts: Experts,
    /// Query, key, value and MLP message gates of looped layers.
    gates: Option<MessageGates>,
}

/// Inkling 2 text tower: immutable weights, with all request state in the engine's pools.
pub struct Inkling2 {
    config: Config,
    description: ModelDescription,
    device: Device,
    embedding: Embeddings,
    embed_norm: Tensor,
    norm: Tensor,
    /// `[1, hidden]` query of the final attention residual.
    output_query: Tensor,
    /// `[looped layers, 4, hidden]` query, key, value and MLP message queries.
    message_queries: Tensor,
    head: PackedLinear,
    layers: Vec<Layer>,
    #[cfg(feature = "metal")]
    kernels: Option<tur_metal::kernels::Kernels>,
}

struct Span<'a> {
    start: usize,
    len: usize,
    past: usize,
    /// Tokens of this pass after which each written slot takes the convolution history.
    ends: Vec<usize>,
    addresses: Addresses,
    sequence: &'a SequenceForward<'a>,
}

/// Addressing shared by every layer of one packed pass.
struct Pass<'a, 'b> {
    pool: &'a StatePool,
    spans: &'a [Span<'b>],
    positions: &'a [usize],
}

/// The operations a pass runs, on the Metal kernels or the Candle reference. The driver,
/// [`Inkling2::decode`], owns the order of layers, passes, block states and messages.
trait Executor {
    /// `weight * x / sqrt(mean(x^2) + eps)` with the model's epsilon.
    fn norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor>;
    /// Attention of `queries` (`[q, hidden]`) over each token's `blocks` and `partial`
    /// state, each RMS-normalized without a gain. Returns `[q, tokens, hidden]`.
    fn residual(&self, blocks: &[Tensor], partial: &Tensor, queries: &Tensor) -> Result<Tensor>;
    /// Layer `index` as application `application` (its state lives at layers `2a` and
    /// `2a + 1`), with the previous loop pass's `[4, tokens, hidden]` messages after the
    /// first pass. A block boundary appends `x` to `blocks`.
    fn layer(
        &self,
        application: usize,
        index: usize,
        x: &Tensor,
        blocks: &mut Vec<Tensor>,
        messages: Option<&Tensor>,
    ) -> Result<Tensor>;
}

/// Paged K/V of layer application `application`.
fn attention_store(
    pool: &StatePool,
    application: usize,
) -> Result<&tur_metal::pool::AttentionPool> {
    pool.attention(2 * application)
}

/// The page table a layer's keys live in: window pages for windowed layers.
fn table(addresses: &Addresses, windowed: bool) -> &[u32] {
    if windowed {
        &addresses.windows
    } else {
        &addresses.pages
    }
}

fn checkpoint(dir: &Path, config: &Config) -> Result<Checkpoint> {
    let (checkpoint, _) = Checkpoint::scan(weight_files(dir)?)?;
    let shapes = config.weight_shapes()?;
    for (name, (shape, kind)) in &shapes {
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
        let stored = match entry.dtype {
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32 => {
                Stored::Float
            }
            safetensors::Dtype::U8 => Stored::Codes,
            safetensors::Dtype::F8_E4M3 => Stored::Fp8,
            dtype => return Err(invalid(format!("{name} has dtype {dtype:?}"))),
        };
        if stored != *kind {
            return Err(invalid(format!("{name} has dtype {:?}", entry.dtype)));
        }
    }
    // The vision tower is not read; any other tensor means a layout this code does not know.
    let vision = "model.visual.";
    if let Some(extra) = checkpoint
        .entries()
        .keys()
        .find(|n| !shapes.contains_key(*n) && !n.starts_with(vision))
    {
        return Err(invalid(format!("unexpected tensor {extra}")));
    }
    Ok(checkpoint)
}

/// Experts stay in their stored formats on the kernels, whatever precision the plan gives
/// projections, and are decoded to F32 only for reference execution. The formats are also
/// exact where wide expert tiles stage weights in F16 (E2M1 times an E4M3 scale, and E4M3
/// times a power of two at checkpoint scales), while a decoded NVFP4 weight, its expert's
/// factor multiplied in, would round there.
fn packed_experts(options: &LoadOptions) -> bool {
    options.execution == Execution::Fused && options.device.is_metal()
}

fn f32_tensor(tensors: &Tensors, name: &str) -> Result<Tensor> {
    Ok(tensors
        .take(name)?
        .to_dtype(DType::F32)?
        .to_device(&tensors.device)?)
}

/// A `[channels, 1, taps]` convolution weight, where the last tap weights the token itself,
/// as `[channels, taps]` by lag.
fn conv_taps(tensors: &Tensors, name: &str) -> Result<Tensor> {
    let weight = tensors.take(name)?.to_dtype(DType::F32)?.squeeze(1)?;
    let taps = weight.dim(1)?;
    let lags: Vec<u32> = (0..taps)
        .rev()
        .map(|t| u32::try_from(t).map_err(|_| invalid("too many convolution taps")))
        .collect::<Result<_>>()?;
    let order = Tensor::new(lags.as_slice(), &Device::Cpu)?;
    Ok(weight
        .index_select(&order, 1)?
        .contiguous()?
        .to_device(&tensors.device)?)
}

/// One expert bank, `[experts * rows, cols]`, with the per-expert factor of NVFP4 weights:
/// decoded with the factor multiplied in when dense, kept packed with it apart otherwise.
fn expert_bank(
    tensors: &Tensors,
    name: &str,
    format: ExpertFormat,
    packed: bool,
) -> Result<(PackedLinear, Vec<f32>)> {
    let (entry, values) = tensors.checkpoint.take_raw(name)?;
    let (_, scales) = tensors.checkpoint.take_raw(&format!("{name}_scale"))?;
    let [experts, rows, stored] = entry.shape.as_slice() else {
        return Err(invalid(format!("{name} is not a stack of matrices")));
    };
    let cols = match format {
        ExpertFormat::Nvfp4 => stored * 2,
        ExpertFormat::Mxfp8 => *stored,
    };
    let factors = match format {
        ExpertFormat::Nvfp4 => {
            let amax = tensors.take(&format!("{name}_amax"))?.to_vec1::<f32>()?;
            let denominator = tensors
                .take(&format!("{name}_outer_scale_denominator"))?
                .to_vec1::<f32>()?;
            let [denominator] = denominator.as_slice() else {
                return Err(invalid(format!("{name} has no single scale denominator")));
            };
            amax.iter().map(|a| a / denominator).collect()
        }
        ExpertFormat::Mxfp8 => vec![1.0; *experts],
    };
    let shape = (checked_elements(&[*experts, *rows])?, cols);
    let microscale = format.microscale();
    if packed {
        let bank = PackedLinear::microscaled(microscale, &values, &scales, shape, &tensors.device)?;
        return Ok((bank, factors));
    }
    // As Transformers dequantizes: each value times its block scale, then the factor.
    let mut decoded = microscale.decode(&values, &scales)?;
    let per_expert = rows * cols;
    for (chunk, factor) in decoded.chunks_exact_mut(per_expert).zip(&factors) {
        for value in chunk {
            *value *= factor;
        }
    }
    let bank = PackedLinear::dense(
        &Tensor::from_vec(decoded, shape, &Device::Cpu)?,
        &tensors.device,
    )?;
    Ok((bank, vec![1.0; *experts]))
}

impl Experts {
    fn load(tensors: &Tensors, config: &Config, index: usize, packed: bool) -> Result<Self> {
        let p = Config::layer_name(index);
        let format = config.expert_formats()?[index];
        let (gate_up, up_factors) = expert_bank(
            tensors,
            &Config::expert_name(index, "w13_weight"),
            format,
            packed,
        )?;
        let (down, down_factors) = expert_bank(
            tensors,
            &Config::expert_name(index, "w2_weight"),
            format,
            packed,
        )?;
        let factors: Vec<f32> = up_factors
            .iter()
            .zip(&down_factors)
            .flat_map(|(u, d)| [*u, *d])
            .collect();
        let experts = up_factors.len();
        Ok(Self {
            router: f32_tensor(tensors, &format!("{p}.mlp.gate.weight"))?,
            bias: f32_tensor(tensors, &format!("{p}.mlp.gate.bias"))?,
            gate_up,
            down,
            factors: Tensor::from_vec(factors, (experts, 2), &tensors.device)?,
        })
    }
}

impl MessageGates {
    /// The `attn`, `k`, `v` and `mlp` controllers under `prefix`.
    fn load(tensors: &Tensors, prefix: &str) -> Result<Self> {
        let (mut downs, mut ups) = (Vec::new(), Vec::new());
        for name in ["attn", "k", "v", "mlp"] {
            let take = |weight: &str| -> Result<Tensor> {
                Ok(tensors
                    .take(&format!("{prefix}.{name}.{weight}"))?
                    .to_dtype(DType::F32)?)
            };
            let up = Tensor::cat(&[take("forget_weight")?, take("gain_weight")?], 0)?;
            downs.push(take("down_weight")?.t()?);
            ups.push(up.t()?);
        }
        Ok(Self {
            down: Tensor::stack(&downs, 0)?.to_device(&tensors.device)?,
            up: Tensor::stack(&ups, 0)?.to_device(&tensors.device)?,
        })
    }

    /// Gate `i`'s `[2 * hidden, rank]` and `[rank, 2 * hidden]` projections.
    fn get(&self, i: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.down.get(i)?, self.up.get(i)?))
    }
}

impl Layer {
    fn load(tensors: &Tensors, config: &Config, index: usize, packed: bool) -> Result<Self> {
        let p = Config::layer_name(index);
        let a = format!("{p}.attn");
        let take = |name: &str| f32_tensor(tensors, name);
        let flat = |name: &str| -> Result<Tensor> { Ok(take(name)?.flatten_all()?) };
        let sliding = config.sliding(index);
        let (key_taps, value_taps) = if sliding {
            (
                conv_taps(tensors, &format!("{a}.k_sconv.weight"))?,
                conv_taps(tensors, &format!("{a}.v_sconv.weight"))?,
            )
        } else {
            (
                take(&format!("{a}.k_sconv.static_w"))?.t()?.contiguous()?,
                take(&format!("{a}.v_sconv.static_w"))?.t()?.contiguous()?,
            )
        };
        let norms = if sliding {
            InputNorms::Shared(take(&format!("{p}.attn_norm.weight"))?)
        } else {
            InputNorms::Separate(Tensor::stack(
                &[
                    take(&format!("{p}.q_input_norm.weight"))?,
                    take(&format!("{p}.k_input_norm.weight"))?,
                    take(&format!("{p}.v_input_norm.weight"))?,
                ],
                0,
            )?)
        };
        let attention_queries = (index > 0)
            .then(|| -> Result<Tensor> {
                let query = take(&format!("{PREFIX}.attn_res.attn_queries.{}", index - 1))?;
                let hidden = config.text.hidden_size;
                Ok(query.reshape((query.elem_count() / hidden, hidden))?)
            })
            .transpose()?;
        let gates = config
            .looped(index)
            .then(|| MessageGates::load(tensors, &format!("{p}.controllers")))
            .transpose()?;
        Ok(Self {
            attention_queries,
            mlp_query: take(&format!("{PREFIX}.attn_res.mlp_queries.{index}"))?.unsqueeze(0)?,
            norms,
            attention: Attention {
                query: tensors.projection(&format!("{a}.query"))?,
                key: tensors.projection(&format!("{a}.k"))?,
                value: tensors.projection(&format!("{a}.v"))?,
                output: tensors.projection(&format!("{a}.output"))?,
                q_scale: flat(&format!("{a}.q_l2_clip.scale"))?,
                rel_scale: flat(&format!("{a}.rel_l2_clip.scale"))?,
                rel_proj: take(&format!("{a}.rel_logits_proj.proj"))?,
                key_taps,
                value_taps,
            },
            post_norm: take(&format!("{p}.mlp_norm.weight"))?,
            attention_taps: conv_taps(tensors, &format!("{p}.attn_sconv.weight"))?,
            mlp_taps: conv_taps(tensors, &format!("{p}.mlp_sconv.weight"))?,
            experts: Experts::load(tensors, config, index, packed)?,
            gates,
        })
    }

    fn bytes(&self) -> usize {
        let a = &self.attention;
        let e = &self.experts;
        let mut tensors = vec![
            &self.mlp_query,
            &self.post_norm,
            &self.attention_taps,
            &self.mlp_taps,
            &a.q_scale,
            &a.rel_scale,
            &a.rel_proj,
            &a.key_taps,
            &a.value_taps,
            &e.router,
            &e.bias,
            &e.factors,
        ];
        tensors.extend(self.attention_queries.as_ref());
        match &self.norms {
            InputNorms::Shared(norms) | InputNorms::Separate(norms) => tensors.push(norms),
        }
        if let Some(gates) = &self.gates {
            tensors.extend([&gates.down, &gates.up]);
        }
        tensors.iter().map(|t| t.elem_count() * 4).sum()
    }

    fn projection_bytes(&self) -> usize {
        let a = &self.attention;
        [
            &a.query,
            &a.key,
            &a.value,
            &a.output,
            &self.experts.gate_up,
            &self.experts.down,
        ]
        .iter()
        .map(|p| p.resident_bytes())
        .sum()
    }
}

/// Bound live hidden and projection buffers, block states, routed rows, convolution
/// outputs and `split` bytes of attention split partials of one pass.
fn workspace(config: &Config, tokens: usize, split: usize) -> Result<usize> {
    let t = &config.text;
    let query = checked_elements(&[t.num_attention_heads, 2 * t.head_dim + t.d_rel + 1])?;
    let kv = checked_elements(&[t.num_key_value_heads, t.head_dim + t.sconv_kernel_size])?;
    // Hidden rows: the stream, its pre-loop copy, up to 8 block states twice (a pass's and
    // the previous pass's), the residual queries' outputs, message gates and norms.
    let activations = checked_elements(&[tokens, t.hidden_size * 32 + query * 2 + kv * 6, 4])?;
    let routed = checked_elements(&[
        tokens,
        t.num_experts_per_tok,
        t.intermediate_size * 5 + t.hidden_size * 2,
        4,
    ])?;
    let routing = checked_elements(&[tokens, t.num_experts_per_tok, 5, 4])?
        .checked_add(checked_elements(&[t.n_routed_experts, tokens.max(1), 8])?)
        .ok_or_else(|| invalid("Inkling 2 routing workspace overflow"))?;
    activations
        .checked_add(routed)
        .and_then(|n| n.checked_add(routing))
        .and_then(|n| n.checked_add(split))
        .ok_or_else(|| invalid("Inkling 2 workspace overflow"))
}

/// Split partials of the widest attention dispatch of a pass of `tokens` over sequences
/// reaching `context` keys: full layers read every key, sliding layers their window.
#[cfg(feature = "metal")]
fn split_workspace(
    config: &Config,
    kernels: &tur_metal::kernels::Kernels,
    tokens: usize,
    context: usize,
) -> usize {
    let t = &config.text;
    [context, context.min(t.sliding_window_size)]
        .into_iter()
        .map(|span| {
            kernels.relative_attention_scratch_bytes(
                tokens,
                t.num_attention_heads,
                t.head_dim,
                span,
            )
        })
        .max()
        .unwrap_or(0)
}

impl Inkling2 {
    /// Plan resident weights and peak activations before loading anything.
    pub fn memory_requirements(
        config: &Config,
        options: &LoadOptions,
        tokens: usize,
        contexts: &[usize],
    ) -> Result<(ModelDescription, usize, usize)> {
        let description = config.description(
            format!("{}#{}", options.revision, options.weights.tag()),
            options.kv,
        )?;
        let plan = options.weights.plan(config)?;
        plan.validate(config)?;
        let packed = packed_experts(options);
        let t = &config.text;
        let mut weights = plan
            .bytes(config)?
            .checked_add(checked_elements(&[config.resident_f32_elements()?, 4])?)
            .ok_or_else(|| invalid("Inkling 2 weight bytes overflow"))?;
        for format in config.expert_formats()? {
            let banks = [
                (2 * t.intermediate_size, t.hidden_size),
                (t.hidden_size, t.intermediate_size),
            ];
            for (rows, cols) in banks {
                let rows = checked_elements(&[t.n_routed_experts, rows])?;
                let bytes = if packed {
                    format.microscale().bytes(rows, cols)?
                } else {
                    checked_elements(&[rows, cols, 4])?
                };
                weights = weights
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("Inkling 2 weight bytes overflow"))?;
            }
        }
        #[cfg(feature = "metal")]
        let split = if options.device.is_metal() {
            let kernels = tur_metal::kernels::Kernels::new(&options.device)?;
            let context = contexts.iter().copied().max().unwrap_or(0);
            split_workspace(config, &kernels, tokens, context)
        } else {
            0
        };
        #[cfg(not(feature = "metal"))]
        let split = {
            let _ = contexts;
            0
        };
        Ok((description, weights, workspace(config, tokens, split)?))
    }

    /// Validate headers, then load the text tower onto `options.device`. Expert formats
    /// come from the directory's `hf_quant_config.json`.
    pub fn load(dir: &Path, options: &LoadOptions) -> Result<Self> {
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?
            .with_quantization(&std::fs::read_to_string(dir.join("hf_quant_config.json"))?)?;
        if matches!(options.weights, Weights::Prepacked(_) | Weights::Gguf(_))
            || artifact::is_artifact(dir)
        {
            return Err(invalid(
                "Inkling 2 loads its SafeTensors directly; it has no prepacked or GGUF form",
            ));
        }
        if options.mtp.is_some() || options.expert_cache.is_some() {
            return Err(invalid(
                "Inkling 2 loads without a draft head or an expert cache",
            ));
        }
        let plan = options.weights.plan(&config)?;
        plan.validate(&config)?;
        let packed = packed_experts(options);
        let device = &options.device;
        let tensors = Tensors {
            #[cfg(feature = "metal")]
            expert_cache: None,
            checkpoint: checkpoint(dir, &config)?,
            prepacked: false,
            plan,
            projections: config
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: device.clone(),
        };
        let mut loading = Stage::begin("Loading weights", config.layers() as u64, "layers");
        let layers = (0..config.layers())
            .map(|i| {
                let layer = Layer::load(&tensors, &config, i, packed);
                loading.advance(1);
                layer
            })
            .collect::<Result<Vec<_>>>()?;
        loading.finish();
        let take = |name: &str| f32_tensor(&tensors, &format!("{PREFIX}.{name}"));
        let hidden = config.text.hidden_size;
        // Each looped layer's query and MLP queries, then its key and value queries, as
        // one `[layers, 4, hidden]` stack of query, key, value and MLP queries.
        let pairs = |name: &str| -> Result<Tensor> { Ok(take(name)?.reshape(((), 2, hidden))?) };
        let (queries, kv_queries) = (
            pairs("middle_cycle_cross_pass_queries")?,
            pairs("middle_cycle_cross_pass_kv_queries")?,
        );
        let message_queries = Tensor::cat(
            &[
                queries.narrow(1, 0, 1)?,
                kv_queries,
                queries.narrow(1, 1, 1)?,
            ],
            1,
        )?
        .contiguous()?;
        Self {
            description: config.description(
                format!("{}#{}", options.revision, options.weights.tag()),
                options.kv,
            )?,
            device: device.clone(),
            embedding: tensors.embedding(
                &config.embedding_tensor_name(),
                config.padded_vocab(),
                hidden,
            )?,
            embed_norm: take("embed_norm.weight")?,
            norm: take("norm.weight")?,
            output_query: take("attn_res.output_query")?.unsqueeze(0)?,
            message_queries,
            head: tensors.projection("lm_head")?,
            layers,
            config,
            #[cfg(feature = "metal")]
            kernels: None,
        }
        .with_execution(options.execution)
    }

    /// Select Metal kernels or the F32 Candle reference. Packed experts run only on the
    /// kernels, so a reference pass over them fails rather than expanding them.
    pub fn with_execution(self, execution: Execution) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let kernels = if execution == Execution::Fused && self.device.is_metal() {
                Some(tur_metal::kernels::Kernels::new(&self.device)?)
            } else {
                None
            };
            Ok(Self { kernels, ..self })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = execution;
            Ok(self)
        }
    }

    /// Device holding the immutable weights.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Validated settings.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Resident bytes held by projections and expert banks.
    pub fn projection_bytes(&self) -> usize {
        self.head.resident_bytes()
            + self
                .layers
                .iter()
                .map(Layer::projection_bytes)
                .sum::<usize>()
    }

    /// Resident bytes loaded onto the device.
    pub fn weight_bytes(&self) -> usize {
        let globals = [
            &self.embed_norm,
            &self.norm,
            &self.output_query,
            &self.message_queries,
        ];
        self.projection_bytes()
            + self.embedding.bytes()
            + globals.iter().map(|t| t.elem_count() * 4).sum::<usize>()
            + self.layers.iter().map(Layer::bytes).sum::<usize>()
    }

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
            || tokens
                .iter()
                .any(|id| *id as usize >= self.config.vocab_size())
            || sequence.logits.iter().any(|r| !sequence.tokens.contains(r))
            || sequence.logits.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(invalid(
                "invalid Inkling 2 positions, tokens, state or logit rows",
            ));
        }
        let addresses = sequence.state.resolve(batch.pool)?;
        let ends = sequence.state.ends()?;
        let pages = self.description.pages_for(end);
        if addresses.pages.len() != pages
            || addresses.windows.len() != pages
            || addresses.writes.len() != ends.len()
        {
            return Err(invalid("sequence state does not cover its new tokens"));
        }
        Ok(Span {
            start,
            len: tokens.len(),
            past,
            ends,
            addresses,
            sequence,
        })
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
        let x = self
            .embedding
            .rows(&Tensor::new(tokens.as_slice(), &self.device)?, &self.head)?;
        let pass = Pass {
            pool: batch.pool,
            spans,
            positions: &positions,
        };
        #[cfg(feature = "metal")]
        let hidden = match &self.kernels {
            Some(k) => self.run_fused(k, &pass, &x, trace)?,
            None => self.run_reference(&pass, &x, trace)?,
        };
        #[cfg(not(feature = "metal"))]
        let hidden = self.run_reference(&pass, &x, trace)?;
        spans.iter().map(|s| self.step(s, &hidden)).collect()
    }

    /// One sequence's outputs from the final-normed rows of its pass: the head over its
    /// real vocabulary rows, softcapped.
    fn step(&self, s: &Span<'_>, hidden: &Tensor) -> Result<SequenceStep> {
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
            let selected = hidden.index_select(&Tensor::new(rows.as_slice(), &self.device)?, 0)?;
            let logits =
                self.linear(&self.head, &selected)?
                    .narrow(1, 0, self.config.vocab_size())?;
            let cap = self.config.text.final_logit_softcapping;
            ((logits / cap)?.tanh()? * cap)?
        };
        Ok(SequenceStep {
            logits,
            hidden: s
                .sequence
                .hidden
                .then(|| hidden.narrow(0, s.start, s.len))
                .transpose()?,
            replay: None,
        })
    }

    fn linear(&self, w: &PackedLinear, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = &self.kernels {
            return k.linear(w, x);
        }
        w.forward(x)
    }

    /// The text model over normalized embeddings: layers before the loop once, the looped
    /// layers once per pass, each later pass restarting from the pre-loop state and reading
    /// messages from the previous pass, then the remaining layers and the final residual.
    /// Returns the final-normed rows; `trace` receives every application's output.
    fn decode(
        &self,
        e: &impl Executor,
        x: &Tensor,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let l = self.config.text.loop_config;
        let mut x = e.norm(x, &self.embed_norm)?;
        let mut blocks = Vec::new();
        let mut application = 0;
        let mut apply =
            |index: usize, x: &Tensor, blocks: &mut Vec<Tensor>, messages: Option<&Tensor>| {
                let out = e.layer(application, index, x, blocks, messages)?;
                application += 1;
                if let Some(trace) = trace.as_mut() {
                    trace.push(out.clone());
                }
                Ok::<_, tur_core::TurError>(out)
            };
        for index in 0..l.core_start {
            x = apply(index, &x, &mut blocks, None)?;
        }
        let (pre, pre_blocks) = (x.clone(), blocks.clone());
        for index in l.core_start..l.core_end {
            x = apply(index, &x, &mut blocks, None)?;
        }
        for _ in 1..l.num_passes {
            // The first looped layer is a block boundary: its state is the pre-loop one.
            let previous = blocks
                .get(pre_blocks.len() + 1..)
                .ok_or_else(|| invalid("loop pass appended no block state"))?
                .to_vec();
            let last = x;
            (x, blocks) = (pre.clone(), pre_blocks.clone());
            for (offset, index) in (l.core_start..l.core_end).enumerate() {
                let queries = self.message_queries.get(offset)?;
                let messages = e.residual(&previous, &last, &queries)?;
                x = apply(index, &x, &mut blocks, Some(&messages))?;
            }
        }
        for index in l.core_end..self.config.layers() {
            x = apply(index, &x, &mut blocks, None)?;
        }
        let out = e.residual(&blocks, &x, &self.output_query)?.get(0)?;
        e.norm(&out, &self.norm)
    }

    /// Run a fresh sequence and return each layer application's output, in execution
    /// order, and the selected logits.
    pub fn trace(&self, tokens: &[u32], rows: &[usize]) -> Result<(Vec<Tensor>, Tensor)> {
        let pool = StatePool::new(
            Arc::new(self.description.clone()),
            &self.device,
            PoolConfig {
                pages: self.description.pages_for(tokens.len()).max(1),
                window_pages: self.description.pages_for(tokens.len()).max(1),
                recurrent_slots: 1,
                index_blocks: 0,
            },
        )?;
        let mut reservation = ModelState::new(&pool).reserve(tokens.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let sequences = [SequenceForward {
            tokens: 0..tokens.len(),
            state: &binding,
            logits: rows,
            hidden: false,
        }];
        let batch = PackedForward {
            embeddings: &[],
            axial: &[],
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
            .ok_or_else(|| invalid("missing Inkling 2 output"))?;
        Ok((trace, step.logits))
    }
}

impl HybridModel for Inkling2 {
    fn description(&self) -> ModelDescription {
        self.description.clone()
    }
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
        #[cfg(feature = "metal")]
        let split = match &self.kernels {
            Some(kernels) => {
                let mut context = 0;
                for s in batch.sequences {
                    let end = s.state.past().checked_add(s.tokens.len());
                    context = context.max(end.ok_or_else(|| invalid("context overflow"))?);
                }
                split_workspace(&self.config, kernels, batch.tokens.len(), context)
            }
            None => 0,
        };
        #[cfg(not(feature = "metal"))]
        let split = 0;
        workspace(&self.config, batch.tokens.len(), split)
    }
    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        self.kernels.clone()
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
