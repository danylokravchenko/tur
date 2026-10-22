//! SafeTensors checkpoint discovery, header validation and payload loading without mmap.

use super::config::Config;
use candle_core::{DType, Device, Tensor, safetensors::Load};
use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::{Path, PathBuf},
};
use tur_core::{Result, execution::invalid};

/// Headers larger than this are rejected before allocation.
const MAX_HEADER_BYTES: u64 = 256 << 20;

struct Entry {
    file: usize,
    shape: Vec<usize>,
}

/// A validated set of text-tower tensors spread over one or more SafeTensors files.
pub(super) struct Checkpoint {
    files: Vec<PathBuf>,
    entries: BTreeMap<String, Entry>,
}

fn header(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut file = std::fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length > MAX_HEADER_BYTES {
        return Err(invalid("safetensors header is too large"));
    }
    let mut bytes = vec![0u8; usize::try_from(length).map_err(|_| invalid("header size"))?];
    file.read_exact(&mut bytes)?;
    match serde_json::from_slice(&bytes)? {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(invalid("safetensors header is not an object")),
    }
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
            .map(|f| f.as_str().map(|f| dir.join(f)))
            .collect::<Option<_>>()
            .ok_or_else(|| invalid("index weight_map has a non-string file"))?;
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
            for (name, info) in header(path)? {
                if name == "__metadata__" {
                    continue;
                }
                let text = name.starts_with(&prefix) || name == "lm_head.weight";
                let Some(shape) = expected.get(&name) else {
                    if text && !(name == "lm_head.weight" && config.tied_embeddings()) {
                        return Err(invalid(format!("unexpected text tensor {name}")));
                    }
                    continue;
                };
                let dtype = info["dtype"].as_str().unwrap_or_default();
                if !matches!(dtype, "BF16" | "F16" | "F32") {
                    return Err(invalid(format!("unsupported dtype {dtype} for {name}")));
                }
                let actual: Vec<usize> = serde_json::from_value(info["shape"].clone())?;
                if &actual != shape {
                    return Err(invalid(format!(
                        "{name} has shape {actual:?}, expected {shape:?}"
                    )));
                }
                if entries
                    .insert(
                        name.clone(),
                        Entry {
                            file: index,
                            shape: actual,
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

    /// Read each file once and return text tensors on the CPU in their stored dtype.
    pub(super) fn load(self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::with_capacity(self.entries.len());
        for (index, path) in self.files.iter().enumerate() {
            let names: Vec<&String> = self
                .entries
                .iter()
                .filter(|(_, e)| e.file == index)
                .map(|(n, _)| n)
                .collect();
            if names.is_empty() {
                continue;
            }
            let bytes = std::fs::read(path)?;
            let file = safetensors::SafeTensors::deserialize(&bytes)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            for name in names {
                let view = file
                    .tensor(name)
                    .map_err(|e| invalid(format!("{name}: {e}")))?;
                let tensor = view.load(&Device::Cpu)?;
                let entry = &self.entries[name];
                if tensor.dims() != entry.shape.as_slice()
                    || !matches!(tensor.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    return Err(invalid(format!("{name} payload disagrees with header")));
                }
                tensors.insert(name.clone(), tensor);
            }
        }
        Ok(tensors)
    }
}
