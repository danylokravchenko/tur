//! Permanent expert removal: a smaller checkpoint in the published layout, with exactly the
//! same number of experts left in every layer. Kept weights are copied byte for byte, and
//! removing a router row is what masking that expert's logit before top-k does.
use super::Config;
use crate::models::checkpoint::{Checkpoint, weight_files};
use serde::Serialize;
use std::{
    io::{BufWriter, Write},
    os::unix::fs::FileExt,
    path::Path,
};
use tur_core::{Result, execution::invalid};

/// Serving files carried over from the source checkpoint.
const COPIED: [&str; 5] = [
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
];
const CHUNK: usize = 8 << 20;

/// What a pruned checkpoint keeps and where it came from, stored beside its weights.
#[derive(Clone, Debug, Serialize)]
pub struct Pruning {
    /// Exact identity of the source checkpoint.
    pub source: String,
    /// Free-form description of the calibration data.
    pub calibration: String,
    /// Original expert IDs of each layer, ascending. Index in this list is the new ID.
    pub kept: Vec<Vec<usize>>,
}

/// One layer's experts measured on calibration tokens, for every expert and not only the
/// ones each token routes to, because pruning sends tokens to their next choices.
#[derive(Clone, Debug, Default)]
pub struct LayerProbe {
    tokens: usize,
    /// `[tokens, experts]` router scores.
    logits: Vec<f32>,
    /// `[tokens, experts, experts]` inner products of the experts' outputs on each token.
    /// How far a layer's output moves under any routing weights follows from these alone.
    overlaps: Vec<f32>,
}

impl LayerProbe {
    /// Add one pass: `values` holds `[rows, width]` outputs whose rows `rows` names as
    /// `(token, expert)`, and `scores` the `[tokens, experts]` router scores.
    pub fn add(
        &mut self,
        experts: usize,
        width: usize,
        rows: &[(usize, usize)],
        values: &[f32],
        scores: &[f32],
    ) -> Result<()> {
        let tokens = scores.len() / experts.max(1);
        let square = experts * experts;
        if rows.len() != tokens * experts || values.len() != rows.len() * width {
            return Err(invalid("expert probe rows do not cover every expert"));
        }
        let mut at = vec![0usize; rows.len()];
        for (row, &(token, expert)) in rows.iter().enumerate() {
            *at.get_mut(token * experts + expert)
                .ok_or_else(|| invalid("expert probe row outside its pass"))? = row;
        }
        let start = self.overlaps.len();
        self.overlaps.resize(start + tokens * square, 0.0);
        let (at, share) = (&at, (tokens / 8).max(1) * square);
        std::thread::scope(|scope| {
            for (chunk, block) in self.overlaps[start..].chunks_mut(share).enumerate() {
                scope.spawn(move || {
                    for (i, cell) in block.iter_mut().enumerate() {
                        let index = chunk * share + i;
                        let (token, pair) = (index / square, index % square);
                        let row = |e: usize| {
                            let begin = at[token * experts + e] * width;
                            &values[begin..begin + width]
                        };
                        *cell = row(pair / experts)
                            .iter()
                            .zip(row(pair % experts))
                            .map(|(a, b)| a * b)
                            .sum();
                    }
                });
            }
        });
        self.logits.extend_from_slice(scores);
        self.tokens += tokens;
        Ok(())
    }
}

/// What a pruning search measured on calibration text.
#[derive(Clone, Debug)]
pub struct ExpertProbe {
    pub experts: usize,
    pub top_k: usize,
    pub layers: Vec<LayerProbe>,
}

/// Selected experts and routing weights of one token among `kept`, as the router gives them.
fn routing(scores: &[f32], kept: &[bool], top_k: usize) -> Result<Vec<(usize, f32)>> {
    let ids: Vec<usize> = (0..scores.len()).filter(|&e| kept[e]).collect();
    let visible: Vec<f32> = ids.iter().map(|&e| scores[e]).collect();
    let (chosen, weights) = crate::models::math::route(&visible, top_k)?;
    Ok(chosen.into_iter().map(|i| ids[i]).zip(weights).collect())
}

/// Squared distance between a token's outputs under two routings, from its overlaps.
fn moved(overlaps: &[f32], experts: usize, full: &[(usize, f32)], now: &[(usize, f32)]) -> f64 {
    let mut delta: Vec<(usize, f32)> = full.to_vec();
    for &(expert, weight) in now {
        match delta.iter_mut().find(|(e, _)| *e == expert) {
            Some(entry) => entry.1 -= weight,
            None => delta.push((expert, -weight)),
        }
    }
    let mut total = 0.0;
    for &(a, wa) in &delta {
        for &(b, wb) in &delta {
            total = (f64::from(wa) * f64::from(wb))
                .mul_add(f64::from(overlaps[a * experts + b]), total);
        }
    }
    total
}

/// One layer's pruning: survivors with the cheapest to lose first, and the removed experts
/// with the most valuable (removed last) first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    pub kept: Vec<usize>,
    pub dropped: Vec<usize>,
}

/// Remove experts from one layer one at a time, always the one whose loss moves the layer's
/// output least over the calibration tokens, including where tokens reroute to next.
fn plan_layer(probe: &LayerProbe, experts: usize, top_k: usize, keep: usize) -> Result<LayerPlan> {
    let square = experts * experts;
    let score = |t: usize| &probe.logits[t * experts..(t + 1) * experts];
    let overlap = |t: usize| &probe.overlaps[t * square..(t + 1) * square];
    let mut kept = vec![true; experts];
    let full = (0..probe.tokens)
        .map(|t| routing(score(t), &kept, top_k))
        .collect::<Result<Vec<_>>>()?;
    let mut now = full.clone();
    let mut dropped = Vec::new();
    // Removing an expert changes only the tokens that route to it.
    let removal_costs = |kept: &[bool], now: &[Vec<(usize, f32)>]| -> Result<Vec<(f64, usize)>> {
        let mut costs = Vec::new();
        for expert in (0..experts).filter(|&e| kept[e]) {
            let mut without = kept.to_vec();
            without[expert] = false;
            let mut cost = 0.0;
            for t in 0..probe.tokens {
                if now[t].iter().any(|(e, _)| *e == expert) {
                    let next = routing(score(t), &without, top_k)?;
                    cost += moved(overlap(t), experts, &full[t], &next)
                        - moved(overlap(t), experts, &full[t], &now[t]);
                }
            }
            costs.push((cost, expert));
        }
        costs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        Ok(costs)
    };
    for _ in keep..experts {
        let (_, expert) = *removal_costs(&kept, &now)?
            .first()
            .ok_or_else(|| invalid("no expert left to remove"))?;
        kept[expert] = false;
        dropped.insert(0, expert);
        for (t, routes) in now.iter_mut().enumerate() {
            if routes.iter().any(|(e, _)| *e == expert) {
                *routes = routing(score(t), &kept, top_k)?;
            }
        }
    }
    Ok(LayerPlan {
        kept: removal_costs(&kept, &now)?
            .into_iter()
            .map(|cost| cost.1)
            .collect(),
        dropped,
    })
}

/// A pruning of every layer down to exactly `keep` experts, from local output movement alone.
pub fn plan_pruning(probe: &ExpertProbe, keep: usize) -> Result<Vec<LayerPlan>> {
    if keep < probe.top_k || keep > probe.experts {
        return Err(invalid(format!(
            "cannot keep {keep} of {} experts with {} per token",
            probe.experts, probe.top_k
        )));
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = probe
            .layers
            .iter()
            .map(|layer| scope.spawn(move || plan_layer(layer, probe.experts, probe.top_k, keep)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| invalid("pruning search thread failed"))?
            })
            .collect()
    })
}

/// Improve `plans` on what actually matters: try swapping a layer's cheapest survivors with
/// its most valuable removed experts, and keep a swap only when `measure`, end-to-end KL over
/// each layer's ascending survivors, falls. Returns the first and final divergence.
pub fn refine(
    plans: &mut [LayerPlan],
    evaluations: usize,
    mut measure: impl FnMut(&[Vec<usize>]) -> Result<f64>,
) -> Result<(f64, f64)> {
    let kept_sets = |plans: &[LayerPlan]| -> Vec<Vec<usize>> {
        plans
            .iter()
            .map(|plan| {
                let mut kept = plan.kept.clone();
                kept.sort_unstable();
                kept
            })
            .collect()
    };
    let first = measure(&kept_sets(plans))?;
    let mut best = first;
    let layers = plans.len().max(1);
    for evaluation in 0..evaluations {
        let (layer, rank) = (evaluation % layers, evaluation / layers);
        let plan = &mut plans[layer];
        if rank >= plan.kept.len() || rank >= plan.dropped.len() {
            break;
        }
        std::mem::swap(&mut plan.kept[rank], &mut plan.dropped[rank]);
        let divergence = measure(&kept_sets(plans))?;
        if divergence < best {
            best = divergence;
        } else {
            let plan = &mut plans[layer];
            std::mem::swap(&mut plan.kept[rank], &mut plan.dropped[rank]);
        }
    }
    Ok((first, best))
}

/// Mean `KL(reference ‖ candidate)` in nats over matching log-probability rows.
pub fn divergence(reference: &[Vec<Vec<f32>>], candidate: &[Vec<Vec<f32>>]) -> f64 {
    let (mut total, mut rows) = (0.0, 0usize);
    for (r, c) in reference.iter().flatten().zip(candidate.iter().flatten()) {
        total += r
            .iter()
            .zip(c)
            .map(|(p, q)| f64::from(p.exp()) * f64::from(p - q))
            .sum::<f64>();
        rows += 1;
    }
    total / rows.max(1) as f64
}

/// One output tensor: its layout, and the source byte ranges it copies, in order.
struct Planned {
    name: String,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    ranges: Vec<(usize, u64, usize)>,
}

/// The layer whose experts `name` indexes by its first dimension, when it does.
fn expert_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let by_expert = rest.starts_with("mlp.experts.") || rest.starts_with("mlp.router.");
    by_expert.then(|| layer.parse().ok()).flatten()
}

fn plan(checkpoint: &Checkpoint, kept: &[Vec<usize>]) -> Result<Vec<Planned>> {
    let mut planned = Vec::new();
    for (name, entry) in checkpoint.entries() {
        let (file, offset) = checkpoint.located(name)?;
        let (shape, ranges) = match expert_layer(name) {
            Some(layer) => {
                let kept = kept
                    .get(layer)
                    .ok_or_else(|| invalid(format!("no kept experts for layer {layer}")))?;
                let (&experts, rest) = entry
                    .shape
                    .split_first()
                    .ok_or_else(|| invalid(format!("{name} has no expert dimension")))?;
                let each = entry.bytes.checked_div(experts).unwrap_or(0);
                let ranges = kept
                    .iter()
                    .map(|&expert| {
                        let skip = (expert as u64)
                            .checked_mul(each as u64)
                            .and_then(|skip| offset.checked_add(skip))
                            .ok_or_else(|| invalid("expert offset overflow"))?;
                        Ok((file, skip, each))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut shape = vec![kept.len()];
                shape.extend_from_slice(rest);
                (shape, ranges)
            }
            None => (entry.shape.clone(), vec![(file, offset, entry.bytes)]),
        };
        planned.push(Planned {
            name: name.clone(),
            dtype: entry.dtype,
            shape,
            ranges,
        });
    }
    Ok(planned)
}

/// Write the checkpoint at `source` with only `pruning.kept` experts into `output`. Each
/// payload byte is written once, into a partial file renamed into place when complete.
pub fn prune(source: &Path, output: &Path, pruning: &Pruning) -> Result<()> {
    let weights = output.join("model.safetensors");
    if weights.exists() {
        return Err(invalid(format!("{} already exists", weights.display())));
    }
    let raw = std::fs::read_to_string(source.join("config.json"))?;
    let config = Config::from_json(&raw)?;
    let keep = pruning.kept.first().map_or(0, Vec::len);
    if pruning.kept.len() != config.layers()
        || keep < config.text.num_experts_per_tok
        || pruning.kept.iter().any(|kept| {
            kept.len() != keep
                || kept.windows(2).any(|pair| pair[0] >= pair[1])
                || kept
                    .last()
                    .is_none_or(|&e| e >= config.text.num_local_experts)
        })
    {
        return Err(invalid(
            "every layer must keep the same ascending experts, at least one token's worth",
        ));
    }
    let (checkpoint, _) = Checkpoint::scan(weight_files(source)?)?;
    let planned = plan(&checkpoint, &pruning.kept)?;
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        serde_json::json!({ "format": "pt", "tur_pruned": serde_json::to_string(pruning)? }),
    );
    let mut offset = 0usize;
    for tensor in &planned {
        let size: usize = tensor.ranges.iter().map(|r| r.2).sum();
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, offset + size],
            }),
        );
        offset += size;
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    std::fs::create_dir_all(output)?;
    let partial = output.join("model.safetensors.partial");
    {
        let files = checkpoint.open_files()?;
        let mut out = BufWriter::with_capacity(CHUNK, std::fs::File::create(&partial)?);
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        let mut buffer = vec![0u8; CHUNK];
        for tensor in &planned {
            for &(file, start, size) in &tensor.ranges {
                let handle = files
                    .get(file)
                    .ok_or_else(|| invalid("pruned tensor has no file"))?;
                let mut done = 0;
                while done < size {
                    let step = CHUNK.min(size - done);
                    handle.read_exact_at(&mut buffer[..step], start + done as u64)?;
                    out.write_all(&buffer[..step])?;
                    done += step;
                }
            }
        }
        out.flush()?;
    }
    for name in COPIED {
        if source.join(name).is_file() {
            std::fs::copy(source.join(name), output.join(name))?;
        }
    }
    let mut settings: serde_json::Value = serde_json::from_str(&raw)?;
    settings["num_local_experts"] = keep.into();
    std::fs::write(
        output.join("config.json"),
        serde_json::to_string_pretty(&settings)?,
    )?;
    std::fs::write(
        output.join("tur-pruned.json"),
        serde_json::to_string_pretty(pruning)?,
    )?;
    std::fs::rename(&partial, &weights)?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    /// Experts 0 and 1 output the same thing and 2 is orthogonal: one twin can go at no cost,
    /// though each carries more routing mass than expert 2, which must stay.
    fn twins() -> ExpertProbe {
        let experts = 3;
        let overlaps = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut layer = LayerProbe::default();
        for scores in [
            [2.0, 1.9, 0.0],
            [1.9, 2.0, -1.0],
            [2.0, 2.0, 0.5],
            [0.0, 0.1, 2.0],
        ] {
            layer.logits.extend_from_slice(&scores);
            layer.overlaps.extend_from_slice(&overlaps);
            layer.tokens += 1;
        }
        ExpertProbe {
            experts,
            top_k: 1,
            layers: vec![layer],
        }
    }

    #[test]
    fn test_search_removes_redundant_experts_rather_than_the_lightest() -> Result<()> {
        let plan = plan_pruning(&twins(), 2)?;
        assert!(plan[0].kept.contains(&2), "{plan:?}");
        assert_eq!(plan[0].kept.len() + plan[0].dropped.len(), 3);
        assert!(plan_pruning(&twins(), 0).is_err() && plan_pruning(&twins(), 4).is_err());
        Ok(())
    }

    #[test]
    fn test_overlaps_reproduce_the_distance_between_real_outputs() -> Result<()> {
        // Two experts, width 2, one token: outputs (1, 0) and (1, 1).
        let mut layer = LayerProbe::default();
        layer.add(2, 2, &[(0, 1), (0, 0)], &[1.0, 1.0, 1.0, 0.0], &[0.0, 0.0])?;
        assert_eq!(layer.overlaps, [1.0, 1.0, 1.0, 2.0]);
        // Half of each against all of the first: distance (0, 0.5), squared 0.25.
        let distance = moved(&layer.overlaps, 2, &[(0, 0.5), (1, 0.5)], &[(0, 1.0)]);
        assert!((distance - 0.25).abs() < 1e-9);
        assert!(
            layer
                .add(2, 2, &[(0, 0)], &[1.0, 1.0], &[0.0, 0.0])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_refine_keeps_a_swap_only_when_divergence_falls() -> Result<()> {
        let mut plans = vec![LayerPlan {
            kept: vec![0, 1],
            dropped: vec![2, 3],
        }];
        // Only expert 3 helps: swapping 0 for 2 is undone, then 1 for 3 stays.
        let (first, best) = refine(&mut plans, 2, |kept| {
            Ok(if kept[0].contains(&3) { 0.5 } else { 1.0 })
        })?;
        assert_eq!((first, best), (1.0, 0.5));
        assert_eq!(plans[0].kept, [0, 3]);
        assert_eq!(plans[0].dropped, [2, 1]);
        Ok(())
    }

    #[test]
    fn test_only_expert_and_router_tensors_are_indexed_by_expert() {
        assert_eq!(
            expert_layer("model.layers.3.mlp.experts.down_proj_blocks"),
            Some(3)
        );
        assert_eq!(expert_layer("model.layers.12.mlp.router.bias"), Some(12));
        assert_eq!(expert_layer("model.layers.3.self_attn.q_proj.weight"), None);
        assert_eq!(expert_layer("lm_head.weight"), None);
    }
}
