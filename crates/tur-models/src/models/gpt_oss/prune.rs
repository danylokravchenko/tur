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

/// The `keep` experts of each layer with the most routing mass, ties to the lower ID.
pub fn keep_by_mass(mass: &[Vec<f64>], keep: usize) -> Result<Vec<Vec<usize>>> {
    mass.iter()
        .map(|layer| {
            if keep == 0 || keep > layer.len() {
                return Err(invalid(format!(
                    "cannot keep {keep} of {} experts",
                    layer.len()
                )));
            }
            let mut order: Vec<usize> = (0..layer.len()).collect();
            order.sort_by(|&a, &b| layer[b].total_cmp(&layer[a]).then(a.cmp(&b)));
            order.truncate(keep);
            order.sort_unstable();
            Ok(order)
        })
        .collect()
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

    #[test]
    fn test_keep_by_mass_takes_the_heaviest_in_id_order_and_breaks_ties_low() -> Result<()> {
        let mass = vec![vec![1.0, 5.0, 3.0, 5.0], vec![0.0, 0.0, 0.0, 9.0]];
        assert_eq!(keep_by_mass(&mass, 2)?, [vec![1, 3], vec![0, 3]]);
        assert!(keep_by_mass(&mass, 0).is_err() && keep_by_mass(&mass, 5).is_err());
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
