//! `SafeTensors` checkpoint discovery, header validation and payload loading without mmap.

use super::config::Config;
use candle_core::{DType, Device, Tensor};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};
use tur_core::{Result, execution::invalid};

/// Headers larger than this are rejected before allocation.
const MAX_HEADER_BYTES: u64 = 256 << 20;

struct Entry {
    file: usize,
    shape: Vec<usize>,
    dtype: DType,
    offset: u64,
    bytes: usize,
}

/// A validated set of text-tower tensors spread over one or more `SafeTensors` files.
pub(super) struct Checkpoint {
    files: Vec<PathBuf>,
    entries: BTreeMap<String, Entry>,
}

fn header(path: &Path) -> Result<(safetensors::tensor::Metadata, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length > MAX_HEADER_BYTES {
        return Err(invalid("safetensors header is too large"));
    }
    let mut bytes = vec![0u8; usize::try_from(length).map_err(|_| invalid("header size"))?];
    file.read_exact(&mut bytes)?;
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
    let mut tensors: Vec<(String, safetensors::tensor::TensorInfo)> = map
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, info)| Ok((name, serde_json::from_value(info)?)))
        .collect::<Result<_>>()?;
    tensors.sort_unstable_by_key(|(_, info)| info.data_offsets);
    let metadata = safetensors::tensor::Metadata::new(None, tensors)
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    let start = length
        .checked_add(8)
        .ok_or_else(|| invalid("safetensors header overflow"))?;
    if start.checked_add(metadata.data_len() as u64) != Some(file.metadata()?.len()) {
        return Err(invalid("safetensors file length disagrees with its header"));
    }
    Ok((metadata, start))
}

fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index)?)?;
        let map = raw["weight_map"]
            .as_object()
            .ok_or_else(|| invalid("index has no weight_map"))?;
        let mut files: Vec<PathBuf> = map
            .values()
            .map(|f| {
                let name = f
                    .as_str()
                    .ok_or_else(|| invalid("index weight_map has a non-string file"))?;
                let mut parts = Path::new(name).components();
                if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
                    return Err(invalid(
                        "checkpoint index filename must stay inside its directory",
                    ));
                }
                Ok(dir.join(name))
            })
            .collect::<Result<_>>()?;
        files.sort();
        files.dedup();
        return Ok(files);
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(invalid("no safetensors checkpoint found"))
}

impl Checkpoint {
    /// Match every header against the config before any payload is read. Tensors outside
    /// the text tower (vision, MTP) are ignored; unexpected text tensors are rejected.
    pub(super) fn open(dir: &Path, config: &Config) -> Result<Self> {
        let expected = config.weight_shapes()?;
        let prefix = format!("{}.", config.text_prefix());
        let files = weight_files(dir)?;
        let mut entries = BTreeMap::new();
        for (index, path) in files.iter().enumerate() {
            let (metadata, start) = header(path)?;
            for (name, info) in metadata.tensors() {
                let text = name.starts_with(&prefix) || name == "lm_head.weight";
                let Some(shape) = expected.get(&name) else {
                    if text && !(name == "lm_head.weight" && config.tied_embeddings()) {
                        return Err(invalid(format!("unexpected text tensor {name}")));
                    }
                    continue;
                };
                let dtype = match info.dtype {
                    safetensors::Dtype::BF16 => DType::BF16,
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::F32 => DType::F32,
                    dtype => {
                        return Err(invalid(format!("unsupported dtype {dtype:?} for {name}")));
                    }
                };
                let actual = &info.shape;
                if actual != shape {
                    return Err(invalid(format!(
                        "{name} has shape {actual:?}, expected {shape:?}"
                    )));
                }
                if entries
                    .insert(
                        name.clone(),
                        Entry {
                            file: index,
                            shape: actual.clone(),
                            dtype,
                            offset: start
                                .checked_add(info.data_offsets.0 as u64)
                                .ok_or_else(|| invalid("weight offset overflow"))?,
                            bytes: info.data_offsets.1 - info.data_offsets.0,
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(format!("duplicate tensor {name}")));
                }
            }
        }
        if let Some(missing) = expected.keys().find(|name| !entries.contains_key(*name)) {
            return Err(invalid(format!("missing tensor {missing}")));
        }
        Ok(Self { files, entries })
    }

    /// Read just this tensor, after every header has passed validation. Packing a layer
    /// can now release its host payload before the next layer is read: `MoE` checkpoints
    /// never require all unquantized experts or a complete shard in host memory.
    pub(super) fn take(&mut self, name: &str) -> Result<Tensor> {
        let entry = self
            .entries
            .remove(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let mut file = std::fs::File::open(&self.files[entry.file])?;
        file.seek(SeekFrom::Start(entry.offset))?;
        let mut bytes = vec![0u8; entry.bytes];
        file.read_exact(&mut bytes)?;
        Ok(Tensor::from_raw_buffer(
            &bytes,
            entry.dtype,
            &entry.shape,
            &Device::Cpu,
        )?)
    }

    /// Read a rank-three tensor one leading index at a time, as `[dims[1], dims[2]]`
    /// matrices: an expert bank streams one expert's payload instead of the whole bank.
    pub(super) fn take_split(&mut self, name: &str) -> Result<Split> {
        let entry = self
            .entries
            .remove(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let &[count, rows, cols] = entry.shape.as_slice() else {
            return Err(invalid(format!("{name} is not a stack of matrices")));
        };
        let mut file = std::fs::File::open(&self.files[entry.file])?;
        file.seek(SeekFrom::Start(entry.offset))?;
        Ok(Split {
            file,
            dtype: entry.dtype,
            shape: [rows, cols],
            bytes: entry.bytes.checked_div(count).unwrap_or(0),
            remaining: count,
        })
    }
}

/// Matrices of one stacked checkpoint tensor, read in order on demand.
pub(super) struct Split {
    file: std::fs::File,
    dtype: DType,
    shape: [usize; 2],
    bytes: usize,
    remaining: usize,
}

impl Iterator for Split {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Result<Tensor>> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let mut bytes = vec![0u8; self.bytes];
        Some(
            self.file
                .read_exact(&mut bytes)
                .map_err(Into::into)
                .and_then(|()| {
                    Ok(Tensor::from_raw_buffer(
                        &bytes,
                        self.dtype,
                        &self.shape,
                        &Device::Cpu,
                    )?)
                }),
        )
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn inspect(json: &str, payload: &[u8]) -> Result<bool> {
        let path = std::env::temp_dir().join(format!(
            "tur-weight-header-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(payload);
        std::fs::write(&path, bytes)?;
        let valid = header(&path).is_ok();
        std::fs::remove_file(path)?;
        Ok(valid)
    }

    #[test]
    fn test_streaming_headers_validate_layout_and_payload_length_before_reading_weights()
    -> Result<()> {
        let valid = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(inspect(valid, &[0; 4])?);
        assert!(!inspect(valid, &[])?);
        assert!(!inspect(valid, &[0; 8])?);
        assert!(!inspect(
            r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
            &[0; 5]
        )?);
        assert!(!inspect(
            r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 4]
        )?);
        Ok(())
    }
}
