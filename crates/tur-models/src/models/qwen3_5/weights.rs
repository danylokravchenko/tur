//! `SafeTensors` checkpoint discovery, header validation and payload loading without mmap.

use super::config::Config;
use candle_core::{DType, Device, Tensor};
use std::{
    collections::{BTreeMap, HashMap},
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};
use tur_core::{Result, execution::invalid};

/// Headers larger than this are rejected before allocation.
const MAX_HEADER_BYTES: u64 = 256 << 20;

/// One tensor's location and declared layout.
#[derive(Clone)]
pub(super) struct Entry {
    file: usize,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: safetensors::Dtype,
    offset: u64,
    pub(super) bytes: usize,
}

impl Entry {
    /// Candle element type of a readable tensor.
    pub(super) fn candle_dtype(&self, name: &str) -> Result<DType> {
        match self.dtype {
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::U8 => Ok(DType::U8),
            dtype => Err(invalid(format!("unsupported dtype {dtype:?} for {name}"))),
        }
    }
}

/// Validated tensors spread over one or more `SafeTensors` files, read on demand.
pub(super) struct Checkpoint {
    files: Vec<PathBuf>,
    entries: BTreeMap<String, Entry>,
}

/// Parse and check one file's header: tensor table, `__metadata__` strings and the offset
/// of its payload, which must end exactly at the end of the file.
fn header(path: &Path) -> Result<(safetensors::tensor::Metadata, HashMap<String, String>, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length > MAX_HEADER_BYTES {
        return Err(invalid("safetensors header is too large"));
    }
    let mut bytes = vec![0u8; usize::try_from(length).map_err(|_| invalid("header size"))?];
    file.read_exact(&mut bytes)?;
    let mut map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
    let strings = map
        .remove("__metadata__")
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    let mut tensors: Vec<(String, safetensors::tensor::TensorInfo)> = map
        .into_iter()
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
    Ok((metadata, strings, start))
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
    /// Every tensor of `files` with each file's metadata strings; no payload is read.
    pub(super) fn scan(files: Vec<PathBuf>) -> Result<(Self, Vec<HashMap<String, String>>)> {
        let mut entries = BTreeMap::new();
        let mut metadata = Vec::with_capacity(files.len());
        for (index, path) in files.iter().enumerate() {
            let (header, strings, start) = header(path)?;
            metadata.push(strings);
            for (name, info) in header.tensors() {
                let entry = Entry {
                    file: index,
                    shape: info.shape.clone(),
                    dtype: info.dtype,
                    offset: start
                        .checked_add(info.data_offsets.0 as u64)
                        .ok_or_else(|| invalid("weight offset overflow"))?,
                    bytes: info.data_offsets.1 - info.data_offsets.0,
                };
                if entries.insert(name.clone(), entry).is_some() {
                    return Err(invalid(format!("duplicate tensor {name}")));
                }
            }
        }
        Ok((Self { files, entries }, metadata))
    }

    /// Match every header of a source checkpoint against the config before any payload
    /// is read. Tensors outside the text tower (vision, an unloaded MTP head) are dropped;
    /// unexpected text tensors are rejected.
    pub(super) fn open(dir: &Path, config: &Config) -> Result<Self> {
        let expected = config.weight_shapes()?;
        let prefix = format!("{}.", config.text_prefix());
        let (mut checkpoint, _) = Self::scan(weight_files(dir)?)?;
        let mut entries = BTreeMap::new();
        for (name, entry) in std::mem::take(&mut checkpoint.entries) {
            let text = name.starts_with(&prefix) || name == "lm_head.weight";
            let Some(shape) = expected.get(&name) else {
                if text && !(name == "lm_head.weight" && config.tied_embeddings()) {
                    return Err(invalid(format!("unexpected text tensor {name}")));
                }
                continue;
            };
            if entry.dtype == safetensors::Dtype::U8 {
                return Err(invalid(format!("unsupported dtype U8 for {name}")));
            }
            entry.candle_dtype(&name)?;
            if &entry.shape != shape {
                return Err(invalid(format!(
                    "{name} has shape {:?}, expected {shape:?}",
                    entry.shape
                )));
            }
            entries.insert(name, entry);
        }
        if let Some(missing) = expected.keys().find(|name| !entries.contains_key(*name)) {
            return Err(invalid(format!("missing tensor {missing}")));
        }
        checkpoint.entries = entries;
        Ok(checkpoint)
    }

    pub(super) fn entries(&self) -> &BTreeMap<String, Entry> {
        &self.entries
    }

    fn locate(&self, name: &str) -> Result<(Entry, std::fs::File)> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?
            .clone();
        let mut file = std::fs::File::open(&self.files[entry.file])?;
        file.seek(SeekFrom::Start(entry.offset))?;
        Ok((entry, file))
    }

    /// Read just this tensor's payload, after every header has passed validation.
    pub(super) fn take_raw(&self, name: &str) -> Result<(Entry, Vec<u8>)> {
        let (entry, mut file) = self.locate(name)?;
        let mut bytes = vec![0u8; entry.bytes];
        file.read_exact(&mut bytes)?;
        Ok((entry, bytes))
    }

    /// Read this tensor onto the host in its stored element type.
    pub(super) fn take(&self, name: &str) -> Result<Tensor> {
        let (entry, bytes) = self.take_raw(name)?;
        Ok(Tensor::from_raw_buffer(
            &bytes,
            entry.candle_dtype(name)?,
            &entry.shape,
            &Device::Cpu,
        )?)
    }

    /// Read a matrix, or a stack of matrices as one `[Σ rows, cols]` matrix, `chunk`
    /// rows at a time: packing can release each chunk before the next is read, so no
    /// projection needs its full-precision payload in host memory at once.
    pub(super) fn take_rows(&self, name: &str, chunk: usize) -> Result<Rows> {
        let (entry, file) = self.locate(name)?;
        let dtype = entry.candle_dtype(name)?;
        let (rows, cols) = match entry.shape.as_slice() {
            [rows, cols] => (*rows, *cols),
            [count, rows, cols] => (count * rows, *cols),
            _ => return Err(invalid(format!("{name} is not a matrix"))),
        };
        if chunk == 0 || rows * cols * dtype.size_in_bytes() != entry.bytes {
            return Err(invalid(format!("{name} cannot be read in row chunks")));
        }
        Ok(Rows {
            file,
            dtype,
            cols,
            chunk,
            remaining: rows,
        })
    }
}

/// Row chunks of one checkpoint matrix, read in order on demand.
pub(super) struct Rows {
    file: std::fs::File,
    dtype: DType,
    cols: usize,
    chunk: usize,
    remaining: usize,
}

impl Iterator for Rows {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Result<Tensor>> {
        if self.remaining == 0 {
            return None;
        }
        let rows = self.chunk.min(self.remaining);
        self.remaining -= rows;
        let mut bytes = vec![0u8; rows * self.cols * self.dtype.size_in_bytes()];
        Some(
            self.file
                .read_exact(&mut bytes)
                .map_err(Into::into)
                .and_then(|()| {
                    Ok(Tensor::from_raw_buffer(
                        &bytes,
                        self.dtype,
                        &[rows, self.cols],
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
