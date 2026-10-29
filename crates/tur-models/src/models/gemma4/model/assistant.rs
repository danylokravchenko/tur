//! Gemma 4 assistant: the published multi-token-prediction drafter of a Gemma 4 target.
//!
//! A small decoder whose layers all attend over the target's K/V: sliding layers read
//! the target's last sliding owner and full layers its last full owner, so the drafter
//! keeps no state. Its input joins the target's scaled embedding of the last seen token
//! with the target's final hidden state before it, at the last seen token's position.
//! Logits cover only the tokens of the top-scoring vocabulary clusters; every other
//! token is `-inf`.
use super::{Gemma4, StatePool, inverse_frequencies, rotary_tables};
use crate::models::{
    LoadOptions, WeightPlan, Weights,
    checkpoint::{Checkpoint, weight_files},
    gemma4::config::{LayerKind, TextConfig, validate_equations},
    math,
    weights::{Embedding, Projection, Tensors, WeightLayout},
};
use candle_core::{DType, Tensor};
use std::{collections::BTreeMap, path::Path};
use tur_core::{
    Result,
    execution::{checked_elements, invalid},
};
use tur_metal::packed::PackedLinear;

/// Validated assistant settings, checked against the target they draft for.
pub(super) struct AssistantConfig {
    text: TextConfig,
    /// Target residual width: the drafter reads and returns target hidden states.
    backbone: usize,
    centroids: usize,
    top_k: usize,
}

impl AssistantConfig {
    pub(super) fn from_json(json: &str, target: &super::Config) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        if raw["model_type"].as_str() != Some("gemma4_assistant")
            || raw["architectures"]
                .as_array()
                .is_none_or(|a| a.len() != 1 || a[0] != "Gemma4AssistantForCausalLM")
        {
            return Err(invalid("not a Gemma 4 assistant checkpoint"));
        }
        if raw["use_ordered_embeddings"].as_bool() != Some(true) {
            return Err(invalid(
                "Gemma 4 assistants without ordered embeddings are not implemented",
            ));
        }
        let size = |key: &str| {
            raw[key]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid(format!("missing assistant {key}")))
        };
        let text: TextConfig = serde_json::from_value(raw["text_config"].clone())?;
        validate_equations(&text)?;
        let config = Self {
            backbone: size("backbone_hidden_size")?,
            centroids: size("num_centroids")?,
            top_k: size("centroid_intermediate_top_k")?,
            text,
        };
        config.check(target)?;
        Ok(config)
    }

    /// Settings the drafter shares with its target: it reads the target's keys, values,
    /// embeddings and hidden states, so their geometry and rotary encodings must agree.
    fn check(&self, target: &super::Config) -> Result<()> {
        let (t, g) = (&self.text, &target.text);
        let rope = |kind: &str| {
            t.rope_parameters
                .get(kind)
                .map(|r| (r.rope_theta, r.partial_rotary_factor))
                == g.rope_parameters
                    .get(kind)
                    .map(|r| (r.rope_theta, r.partial_rotary_factor))
        };
        if self.backbone != g.hidden_size
            || t.vocab_size != g.vocab_size
            || t.head_dim != g.head_dim
            || t.global_head_dim != g.global_head_dim
            || t.num_key_value_heads != g.num_key_value_heads
            || t.attention_k_eq_v
            || t.sliding_window != g.sliding_window
            || t.num_kv_shared_layers != t.num_hidden_layers
            || t.hidden_size_per_layer_input != 0
            || t.enable_moe_block
            || t.use_double_wide_mlp
            || t.final_logit_softcapping.is_some()
            || !t.tie_word_embeddings
            || !rope("sliding_attention")
            || !rope("full_attention")
            || self.centroids == 0
            || !t.vocab_size.is_multiple_of(self.centroids)
            || self.top_k == 0
            || self.top_k > self.centroids
        {
            return Err(invalid("Gemma 4 assistant does not match its target"));
        }
        for kind in [LayerKind::SlidingAttention, LayerKind::FullAttention] {
            if t.layer_types.contains(&kind) && target.last_owner(kind).is_none() {
                return Err(invalid("target has no K/V owner for an assistant layer"));
            }
        }
        Ok(())
    }

    fn dim(&self, i: usize) -> usize {
        if self.text.layer_types[i] == LayerKind::FullAttention {
            self.text.global_head_dim
        } else {
            self.text.head_dim
        }
    }

    fn query(&self, i: usize) -> Result<usize> {
        checked_elements(&[self.text.num_attention_heads, self.dim(i)])
    }

    /// Every checkpoint tensor and its shape.
    fn weight_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let t = &self.text;
        let mut shapes = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| shapes.insert(name, shape);
        add(
            "model.embed_tokens.weight".into(),
            vec![t.vocab_size, t.hidden_size],
        );
        add("model.norm.weight".into(), vec![t.hidden_size]);
        add(
            "pre_projection.weight".into(),
            vec![t.hidden_size, 2 * self.backbone],
        );
        add(
            "post_projection.weight".into(),
            vec![self.backbone, t.hidden_size],
        );
        add(
            "masked_embedding.centroids.weight".into(),
            vec![self.centroids, t.hidden_size],
        );
        add("masked_embedding.token_ordering".into(), vec![t.vocab_size]);
        for i in 0..t.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let query = self.query(i)?;
            add(
                format!("{p}.self_attn.q_proj.weight"),
                vec![query, t.hidden_size],
            );
            add(
                format!("{p}.self_attn.o_proj.weight"),
                vec![t.hidden_size, query],
            );
            add(format!("{p}.self_attn.q_norm.weight"), vec![self.dim(i)]);
            add(
                format!("{p}.mlp.gate_proj.weight"),
                vec![t.intermediate_size, t.hidden_size],
            );
            add(
                format!("{p}.mlp.up_proj.weight"),
                vec![t.intermediate_size, t.hidden_size],
            );
            add(
                format!("{p}.mlp.down_proj.weight"),
                vec![t.hidden_size, t.intermediate_size],
            );
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                add(format!("{p}.{norm}.weight"), vec![t.hidden_size]);
            }
            add(format!("{p}.layer_scalar"), vec![1]);
        }
        Ok(shapes)
    }
}

impl WeightLayout for AssistantConfig {
    fn projection_layout(&self) -> Result<Vec<Projection>> {
        let t = &self.text;
        let single = |name: &str, rows: usize, cols: usize| {
            Projection::new(name.into(), vec![(format!("{name}.weight"), rows)], cols)
        };
        let mut projections = vec![
            single("pre_projection", t.hidden_size, 2 * self.backbone),
            single("masked_embedding.centroids", self.centroids, t.hidden_size),
            // The tied head; the drafter gathers its candidate rows.
            Projection::new(
                "lm_head".into(),
                vec![("model.embed_tokens.weight".into(), t.vocab_size)],
                t.hidden_size,
            ),
        ];
        for i in 0..t.num_hidden_layers {
            let p = format!("model.layers.{i}");
            projections.push(single(
                &format!("{p}.self_attn.q_proj"),
                self.query(i)?,
                t.hidden_size,
            ));
            projections.push(single(
                &format!("{p}.self_attn.o_proj"),
                t.hidden_size,
                self.query(i)?,
            ));
            projections.push(Projection::new(
                format!("{p}.mlp.gate_up_proj"),
                vec![
                    (format!("{p}.mlp.gate_proj.weight"), t.intermediate_size),
                    (format!("{p}.mlp.up_proj.weight"), t.intermediate_size),
                ],
                t.hidden_size,
            ));
            projections.push(single(
                &format!("{p}.mlp.down_proj"),
                t.hidden_size,
                t.intermediate_size,
            ));
        }
        Ok(projections)
    }
    fn embeddings_tied(&self) -> bool {
        true
    }
    fn vocabulary(&self) -> usize {
        self.text.vocab_size
    }
    fn residual_width(&self) -> usize {
        self.text.hidden_size
    }
}

struct AssistantLayer {
    q: PackedLinear,
    o: PackedLinear,
    q_norm: Tensor,
    input_norm: Tensor,
    post_attention: Tensor,
    pre_ff: Tensor,
    post_ff: Tensor,
    scalar: Tensor,
    gate_up: PackedLinear,
    down: PackedLinear,
    /// Target layer owning the keys and values this layer attends over, and its state layer.
    owner: usize,
    state_layer: usize,
    window: Option<usize>,
    rope: Tensor,
}

/// Loaded assistant weights; drafting state lives in the target's pools.
pub(super) struct Assistant {
    config: AssistantConfig,
    pre: PackedLinear,
    centroids: PackedLinear,
    head: PackedLinear,
    norm: Tensor,
    /// Token of each canonical cluster position: cluster `c` owns entries
    /// `c × vocab / centroids ..` of the ordering.
    ordering: Vec<u32>,
    layers: Vec<AssistantLayer>,
}

fn plan(config: &AssistantConfig, options: &LoadOptions) -> Result<WeightPlan> {
    let mut plan = match &options.weights {
        Weights::Prepacked(_) => {
            return Err(invalid(
                "Gemma 4 assistants load from SafeTensors checkpoints",
            ));
        }
        weights => weights.plan(config)?,
    };
    // The head is gathered row by row, never multiplied whole.
    plan.embedding = Embedding::Head;
    plan.validate(config)?;
    Ok(plan)
}

impl Assistant {
    /// Load the assistant checkpoint in `dir` for `target`, packing its projections like
    /// the target's.
    pub(super) fn load(dir: &Path, target: &super::Config, options: &LoadOptions) -> Result<Self> {
        let config =
            AssistantConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?, target)?;
        let plan = plan(&config, options)?;
        let (checkpoint, _) = Checkpoint::scan(weight_files(dir)?)?;
        let shapes = config.weight_shapes()?;
        for (name, entry) in checkpoint.entries() {
            if shapes.get(name) != Some(&entry.shape) {
                return Err(invalid(format!("unexpected assistant tensor {name}")));
            }
        }
        let t = Tensors {
            checkpoint,
            prepacked: false,
            plan,
            projections: config
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: options.device.clone(),
        };
        let small = |name: &str| -> Result<Tensor> {
            Ok(t.take(name)?.to_dtype(DType::F32)?.to_device(&t.device)?)
        };
        let ordering = ordering(&t.checkpoint, &config)?;
        let mut layers = Vec::with_capacity(config.text.num_hidden_layers);
        for i in 0..config.text.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let kind = config.text.layer_types[i];
            let owner = target
                .last_owner(kind)
                .ok_or_else(|| invalid("target has no K/V owner for an assistant layer"))?;
            let rope = target.rope(owner);
            layers.push(AssistantLayer {
                q: t.projection(&format!("{p}.self_attn.q_proj"))?,
                o: t.projection(&format!("{p}.self_attn.o_proj"))?,
                q_norm: small(&format!("{p}.self_attn.q_norm.weight"))?,
                input_norm: small(&format!("{p}.input_layernorm.weight"))?,
                post_attention: small(&format!("{p}.post_attention_layernorm.weight"))?,
                pre_ff: small(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                post_ff: small(&format!("{p}.post_feedforward_layernorm.weight"))?,
                scalar: small(&format!("{p}.layer_scalar"))?,
                gate_up: t.projection(&format!("{p}.mlp.gate_up_proj"))?,
                down: t.projection(&format!("{p}.mlp.down_proj"))?,
                owner,
                state_layer: target.state_layer(owner),
                // Transformers masks the drafter bidirectionally: its sliding queries see
                // the last `window + 1` target keys, one more than a target query does.
                window: (kind == LayerKind::SlidingAttention)
                    .then_some(config.text.sliding_window + 1),
                rope: Tensor::new(
                    inverse_frequencies(config.dim(i), rope).as_slice(),
                    &t.device,
                )?,
            });
        }
        Ok(Self {
            pre: t.projection("pre_projection")?,
            centroids: t.projection("masked_embedding.centroids")?,
            head: t.projection("lm_head")?,
            norm: small("model.norm.weight")?,
            ordering,
            layers,
            config,
        })
    }

    /// Resident bytes of the assistant's weights.
    pub(super) fn bytes(&self) -> usize {
        let dense = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let mut total = self.pre.resident_bytes()
            + self.centroids.resident_bytes()
            + self.head.resident_bytes()
            + dense(&self.norm)
            + self.ordering.len() * 4;
        for l in &self.layers {
            total += [&l.q, &l.o, &l.gate_up, &l.down]
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
            .map(|t| dense(t))
            .sum::<usize>();
        }
        total
    }

    /// Planned resident bytes of the assistant in `dir`, before loading it.
    pub(super) fn planned_bytes(
        dir: &Path,
        target: &super::Config,
        options: &LoadOptions,
    ) -> Result<usize> {
        let config =
            AssistantConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?, target)?;
        let plan = plan(&config, options)?;
        let accounted: std::collections::BTreeSet<String> = config
            .projection_layout()?
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(|(name, _)| name))
            .collect();
        let mut bytes = plan.bytes(&config)?;
        for (name, shape) in config.weight_shapes()? {
            // The post-projection feeds only chained drafts, which Tur does not run.
            if !accounted.contains(&name) && name != "post_projection.weight" {
                bytes = checked_elements(&shape)?
                    .checked_mul(4)
                    .and_then(|b| bytes.checked_add(b))
                    .ok_or_else(|| invalid("assistant weight bytes overflow"))?;
            }
        }
        Ok(bytes)
    }
}

/// The cluster ordering as token IDs, which must be a permutation of the vocabulary.
fn ordering(checkpoint: &Checkpoint, config: &AssistantConfig) -> Result<Vec<u32>> {
    let (entry, bytes) = checkpoint.take_raw("masked_embedding.token_ordering")?;
    let vocab = config.text.vocab_size;
    if entry.dtype != safetensors::Dtype::I64 || bytes.len() != vocab * 8 {
        return Err(invalid(
            "assistant token ordering must be I64 over the vocabulary",
        ));
    }
    let mut seen = vec![false; vocab];
    let mut ordering = Vec::with_capacity(vocab);
    for raw in bytes.as_chunks::<8>().0 {
        let token = usize::try_from(i64::from_le_bytes(*raw))
            .ok()
            .filter(|&t| t < vocab)
            .ok_or_else(|| invalid("assistant token ordering leaves the vocabulary"))?;
        if std::mem::replace(&mut seen[token], true) {
            return Err(invalid("assistant token ordering repeats a token"));
        }
        ordering.push(u32::try_from(token).map_err(|_| invalid("token exceeds u32"))?);
    }
    Ok(ordering)
}

/// One draft: the token at the last seen position and the target's final hidden state
/// before it, at that token's position.
pub(super) struct DraftRow {
    pub(super) token: u32,
    pub(super) hidden: Tensor,
    pub(super) position: usize,
}

/// Target addressing of a draft pass: each row's page table; row `r` sees the first
/// `rows[r].position` keys.
pub(super) struct DraftPass<'a> {
    pub(super) pool: &'a StatePool,
    pub(super) rows: &'a [DraftRow],
    pub(super) pages: &'a [crate::hybrid::Addresses],
}

impl Gemma4 {
    /// Draft logits `[1, vocab]` for each row of `pass`.
    pub(super) fn assistant_logits(
        &self,
        assistant: &Assistant,
        pass: &DraftPass<'_>,
    ) -> Result<Vec<Tensor>> {
        let ids: Vec<u32> = pass.rows.iter().map(|r| r.token).collect();
        let ids = Tensor::new(ids.as_slice(), &self.device)?;
        let embedded =
            (self.embedding.rows(&ids, &self.head)? * (self.config.hidden_size() as f64).sqrt())?;
        let hidden = Tensor::cat(
            &pass
                .rows
                .iter()
                .map(|r| r.hidden.clone())
                .collect::<Vec<_>>(),
            0,
        )?;
        let mut x = self.linear(&assistant.pre, &Tensor::cat(&[embedded, hidden], 1)?)?;
        for layer in &assistant.layers {
            let h = self.rms(&x, &layer.input_norm)?;
            let attention = self.assistant_attention(assistant, layer, &h, pass)?;
            x = (x + self.rms(&attention, &layer.post_attention)?)?;
            let h = self.rms(&x, &layer.pre_ff)?;
            let mlp = self.linear(
                &layer.down,
                &self.gelu_product(&self.linear(&layer.gate_up, &h)?)?,
            )?;
            x = (x + self.rms(&mlp, &layer.post_ff)?)?.broadcast_mul(&layer.scalar)?;
        }
        let normed = self.rms(&x, &assistant.norm)?;
        let scores = self
            .linear(&assistant.centroids, &normed)?
            .to_vec2::<f32>()?;
        scores
            .iter()
            .enumerate()
            .map(|(r, row)| self.cluster_logits(assistant, row, &normed.narrow(0, r, 1)?))
            .collect()
    }

    /// Logits of the tokens in the `top_k` best-scoring clusters, `-inf` elsewhere.
    fn cluster_logits(
        &self,
        assistant: &Assistant,
        scores: &[f32],
        normed: &Tensor,
    ) -> Result<Tensor> {
        let c = &assistant.config;
        let per = c.text.vocab_size / c.centroids;
        let mut clusters: Vec<usize> = (0..c.centroids).collect();
        // Highest scores first; ties keep the lower cluster.
        clusters.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let candidates: Vec<u32> = clusters[..c.top_k]
            .iter()
            .flat_map(|&k| assistant.ordering[k * per..(k + 1) * per].iter().copied())
            .collect();
        let ids = Tensor::new(candidates.as_slice(), &self.device)?;
        let logits = assistant
            .head
            .rows(&ids)?
            .matmul(&normed.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut full = vec![f32::NEG_INFINITY; c.text.vocab_size];
        for (&token, logit) in candidates.iter().zip(logits) {
            full[token as usize] = logit;
        }
        Ok(Tensor::from_vec(
            full,
            (1, c.text.vocab_size),
            &self.device,
        )?)
    }

    fn assistant_attention(
        &self,
        assistant: &Assistant,
        layer: &AssistantLayer,
        h: &Tensor,
        pass: &DraftPass<'_>,
    ) -> Result<Tensor> {
        let rows = h.dim(0)?;
        let heads = assistant.config.text.num_attention_heads;
        let dim = layer.q_norm.dim(0)?;
        let store = pass.pool.attention(layer.state_layer)?;
        let query = self.linear(&layer.q, h)?;
        let positions: Vec<usize> = pass.rows.iter().map(|r| r.position).collect();
        #[cfg(feature = "metal")]
        if let Some(k) = self.fused_kernels() {
            let ids = positions
                .iter()
                .map(|&p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
                .collect::<Result<Vec<_>>>()?;
            let at = tur_metal::kernels::Heads {
                count: heads,
                dim,
                head_stride: dim,
                col_offset: 0,
            };
            let query = k.qk_norm_rope(
                &query,
                at,
                &layer.q_norm,
                &Tensor::new(ids.as_slice(), &self.device)?,
                &layer.rope,
                self.config.text.rms_norm_eps,
            )?;
            let paging = self.draft_paging(k, pass)?;
            return self.linear(
                &layer.o,
                &k.attention_ungated(
                    &query,
                    store,
                    paging.layer(layer.window.is_some())?,
                    layer.window,
                    1.0,
                )?,
            );
        }
        let (cos, sin) =
            rotary_tables(&positions, dim, self.config.rope(layer.owner), &self.device)?;
        let query = math::rotary(
            &self.rms(&query.reshape((rows, heads, dim))?, &layer.q_norm)?,
            &cos,
            &sin,
        )?;
        let outputs = pass
            .rows
            .iter()
            .zip(pass.pages)
            .enumerate()
            .map(|(r, (row, addresses))| {
                let pages = super::table(addresses, layer.window.is_some());
                let (keys, values) = store.gather(pages, row.position)?;
                // The query sees the keys before its own position, as if it followed them.
                math::attention(
                    &query.narrow(0, r, 1)?,
                    &keys,
                    &values,
                    row.position - 1,
                    1.0,
                    layer.window,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.linear(
            &layer.o,
            &Tensor::cat(&outputs, 0)?.reshape((rows, heads * dim))?,
        )
    }

    /// Page addressing of a draft pass: one query per row, seeing its position's keys.
    #[cfg(feature = "metal")]
    fn draft_paging(
        &self,
        k: &tur_metal::kernels::Kernels,
        pass: &DraftPass<'_>,
    ) -> Result<super::Paging> {
        let (mut full, mut windows) = (Vec::new(), Vec::new());
        let mut first = Vec::with_capacity(pass.rows.len());
        let mut visible = Vec::with_capacity(pass.rows.len());
        for (row, addresses) in pass.rows.iter().zip(pass.pages) {
            let base = full.len().max(windows.len());
            first.push(u32::try_from(base).map_err(|_| invalid("page table overflow"))?);
            full.extend_from_slice(&addresses.pages);
            windows.extend_from_slice(&addresses.windows);
            let end = base + addresses.pages.len().max(addresses.windows.len());
            full.resize(end, 0);
            windows.resize(end, 0);
            visible.push(u32::try_from(row.position).map_err(|_| invalid("context overflow"))?);
        }
        super::Paging::new(
            k,
            &self.description,
            pass.pool,
            (&full, &windows),
            &first,
            &visible,
        )
    }
}
