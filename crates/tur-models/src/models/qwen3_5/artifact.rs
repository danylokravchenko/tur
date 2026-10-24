//! Prepacked weight artifacts: one `SafeTensors` file whose projections hold the exact
//! GGML blocks the kernels read, beside copies of the source's config, tokenizer and
//! template. Loading uploads each tensor as stored; nothing is converted or expanded.

use super::{
    config::Config,
    plan::{Embedding, Precision, WeightPlan, fnv1a},
    quantize::Method,
    weights::Checkpoint,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use tur_core::{Result, execution::invalid};

/// File name of the weights inside an artifact directory.
pub const ARTIFACT_FILE: &str = "tur-weights.safetensors";
/// Layout version of [`ARTIFACT_FILE`]; readers reject any other.
pub const FORMAT_VERSION: u32 = 1;
/// Serving files copied from the source checkpoint when present.
const COPIED: [&str; 6] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
];

/// Identity and contents of a prepacked artifact, stored in its header.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    /// Exact source checkpoint identity (repository and commit).
    pub source: String,
    /// FNV-1a of the source `config.json` bytes, which the artifact carries verbatim.
    pub config_fingerprint: String,
    pub quantizer: Quantizer,
    pub calibration: Option<CalibrationInfo>,
    pub plan: WeightPlan,
}

/// The conversion that produced an artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quantizer {
    pub name: String,
    pub version: String,
    pub method: Method,
    /// GPTQ Hessian damping relative to its mean diagonal.
    pub damp: f64,
}

/// The calibration data statistics were collected from; never the held-out evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationInfo {
    pub samples: usize,
    pub tokens: usize,
    /// FNV-1a of the calibration token IDs.
    pub fingerprint: String,
    pub description: String,
}

impl Manifest {
    /// Stable identity of the artifact's numerical contents, for state namespaces.
    pub fn identity(&self) -> String {
        format!(
            "tur{}-{:016x}",
            self.format_version,
            fnv1a(serde_json::to_string(self).unwrap_or_default().as_bytes())
        )
    }

    /// Read and validate the manifest of the artifact in `dir` against its `config.json`,
    /// without reading any weight payload.
    pub fn read(dir: &Path) -> Result<Self> {
        let (_, manifest) = open(dir)?;
        Ok(manifest)
    }
}

/// Whether `dir` holds a prepacked artifact.
pub fn is_artifact(dir: &Path) -> bool {
    dir.join(ARTIFACT_FILE).is_file()
}

fn config_fingerprint(dir: &Path) -> Result<String> {
    Ok(format!(
        "{:016x}",
        fnv1a(&std::fs::read(dir.join("config.json"))?)
    ))
}

fn open(dir: &Path) -> Result<(Checkpoint, Manifest)> {
    let (checkpoint, metadata) = Checkpoint::scan(vec![dir.join(ARTIFACT_FILE)])?;
    let json = metadata
        .first()
        .and_then(|m| m.get("tur"))
        .ok_or_else(|| invalid("weights file has no Tur manifest; not a prepacked artifact"))?;
    let manifest: Manifest = serde_json::from_str(json)?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(invalid(format!(
            "artifact format {} is not supported (expected {FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    if manifest.config_fingerprint != config_fingerprint(dir)? {
        return Err(invalid(
            "artifact config.json differs from the one it was packed with",
        ));
    }
    Ok((checkpoint, manifest))
}

/// Stored layout of every tensor an artifact holds for `config` under `plan`.
fn expected(config: &Config, plan: &WeightPlan) -> Result<BTreeMap<String, Stored>> {
    let projections = config.projections()?;
    let mut stored = BTreeMap::new();
    let mut parts = std::collections::BTreeSet::new();
    for p in &projections {
        parts.extend(p.parts.iter().map(|(name, _)| name.clone()));
        stored.insert(
            p.name.clone(),
            Stored::matrix(plan.precision(&p.name)?, p.rows, p.cols)?,
        );
    }
    let embedding = config.embedding_name();
    if let Embedding::Table(precision) = plan.embedding {
        stored.insert(
            embedding.clone(),
            Stored::matrix(precision, config.vocab_size(), config.hidden_size())?,
        );
    }
    for (name, shape) in config.weight_shapes()? {
        if !parts.contains(&name) && name != embedding {
            stored.insert(name, Stored::Float(shape));
        }
    }
    Ok(stored)
}

/// How one tensor is stored in an artifact.
#[derive(Debug, PartialEq, Eq)]
enum Stored {
    /// A small tensor in its checkpoint dtype and shape.
    Float(Vec<usize>),
    /// A matrix in exactly this dtype and stored shape.
    Exact(safetensors::Dtype, Vec<usize>),
}

impl Stored {
    fn matrix(precision: Precision, rows: usize, cols: usize) -> Result<Self> {
        Ok(match precision {
            Precision::F32 => Self::Exact(safetensors::Dtype::F32, vec![rows, cols]),
            Precision::Bf16 => Self::Exact(safetensors::Dtype::BF16, vec![rows, cols]),
            packed => Self::Exact(safetensors::Dtype::U8, vec![rows, packed.bytes(1, cols)?]),
        })
    }
}

/// Open the artifact in `dir` for `config`, checking every header entry against the plan
/// before any payload is read. Tensors of an unloaded MTP head are ignored.
pub(super) fn open_for(dir: &Path, config: &Config) -> Result<(Checkpoint, Manifest)> {
    let (checkpoint, manifest) = open(dir)?;
    manifest.plan.validate(config)?;
    let expected = expected(config, &manifest.plan)?;
    for (name, entry) in checkpoint.entries() {
        let Some(stored) = expected.get(name) else {
            if name.starts_with("mtp.") && !config.mtp() {
                continue;
            }
            return Err(invalid(format!("unexpected artifact tensor {name}")));
        };
        let matches = match stored {
            Stored::Float(shape) => {
                &entry.shape == shape
                    && matches!(
                        entry.dtype,
                        safetensors::Dtype::BF16
                            | safetensors::Dtype::F16
                            | safetensors::Dtype::F32
                    )
            }
            Stored::Exact(dtype, shape) => entry.dtype == *dtype && &entry.shape == shape,
        };
        if !matches {
            return Err(invalid(format!(
                "artifact tensor {name} is {:?} {:?}, expected {stored:?}",
                entry.dtype, entry.shape
            )));
        }
    }
    if let Some(missing) = expected
        .keys()
        .find(|name| !checkpoint.entries().contains_key(*name))
    {
        return Err(invalid(format!("artifact is missing {missing}")));
    }
    Ok((checkpoint, manifest))
}

/// Streams tensors to a temporary payload file, then writes the header and payload as
/// one `SafeTensors` file: host memory holds one tensor at a time.
pub(super) struct Writer {
    dir: PathBuf,
    payload: PathBuf,
    data: BufWriter<std::fs::File>,
    entries: Vec<(String, safetensors::Dtype, Vec<usize>, usize, usize)>,
    offset: usize,
}

impl Writer {
    /// Start an artifact in `dir`, creating it, and copy the source's serving files.
    pub(super) fn create(source: &Path, dir: &Path) -> Result<Self> {
        if is_artifact(dir) {
            return Err(invalid(format!(
                "{} already holds an artifact",
                dir.display()
            )));
        }
        std::fs::create_dir_all(dir)?;
        for name in COPIED {
            let from = source.join(name);
            if from.is_file() {
                std::fs::copy(&from, dir.join(name))?;
            }
        }
        let payload = dir.join(format!("{ARTIFACT_FILE}.payload"));
        Ok(Self {
            dir: dir.to_path_buf(),
            data: BufWriter::new(std::fs::File::create(&payload)?),
            payload,
            entries: Vec::new(),
            offset: 0,
        })
    }

    pub(super) fn put(
        &mut self,
        name: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<()> {
        self.data.write_all(bytes)?;
        let end = self.offset + bytes.len();
        self.entries
            .push((name.to_owned(), dtype, shape, self.offset, end));
        self.offset = end;
        Ok(())
    }

    /// Write the header carrying `manifest`, then the payload, and validate the result.
    pub(super) fn finish(mut self, manifest: &Manifest, config: &Config) -> Result<()> {
        self.data.flush()?;
        drop(self.data);
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".into(),
            serde_json::json!({ "tur": serde_json::to_string(manifest)? }),
        );
        for (name, dtype, shape, start, end) in &self.entries {
            header.insert(
                name.clone(),
                serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, end] }),
            );
        }
        let mut header = serde_json::to_vec(&header)?;
        header.resize(header.len().next_multiple_of(8), b' ');
        let partial = self.dir.join(format!("{ARTIFACT_FILE}.partial"));
        {
            let mut out = BufWriter::new(std::fs::File::create(&partial)?);
            out.write_all(&(header.len() as u64).to_le_bytes())?;
            out.write_all(&header)?;
            let mut payload = std::fs::File::open(&self.payload)?;
            let mut buffer = vec![0u8; 8 << 20];
            loop {
                let read = payload.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                out.write_all(&buffer[..read])?;
            }
            out.flush()?;
        }
        std::fs::remove_file(&self.payload)?;
        std::fs::rename(&partial, self.dir.join(ARTIFACT_FILE))?;
        open_for(&self.dir, config).map(|_| ())
    }
}

/// The fingerprint of `dir`'s `config.json`, recorded when an artifact is packed from it.
pub(super) fn source_fingerprint(dir: &Path) -> Result<String> {
    config_fingerprint(dir)
}
