//! Prepacked weight artifacts: one `SafeTensors` file whose projections hold the exact
//! GGML blocks the kernels read, beside copies of the source's config, tokenizer and
//! template. Loading uploads each tensor as stored; nothing is converted or expanded.
//! Every model family writes and opens artifacts through [`ArtifactLayout`].

use crate::models::checkpoint::Checkpoint;
use crate::models::weights::{WeightLayout, fnv1a};
use crate::models::{Embedding, Method, Packer, Precision, Projection, STAGING_BYTES, WeightPlan};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufWriter, Seek, Write},
    path::{Path, PathBuf},
};
use tur_core::{Result, execution::invalid};

/// File name of the weights inside an artifact directory.
pub const ARTIFACT_FILE: &str = "tur-weights.safetensors";
/// Layout version of [`ARTIFACT_FILE`]; readers reject any other.
pub(crate) const FORMAT_VERSION: u32 = 1;
/// Serving files copied from the source checkpoint when present.
const COPIED: [&str; 8] = [
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
];

/// What an artifact stores beyond the loaded projections: every checkpoint tensor's
/// shape, the embedding table's name, and tensors of optional parts left unloaded.
pub(crate) trait ArtifactLayout: WeightLayout {
    fn tensor_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>>;
    fn embedding_tensor(&self) -> String;
    /// Whether an artifact tensor belongs to an optional part this layout does not load.
    fn unloaded(&self, _name: &str) -> bool {
        false
    }
    /// Tensors of an optional part stored beside the rest, each with the precision of a
    /// packed matrix or `None` to copy it as it is. An artifact holds all of them or none.
    fn companions(&self, _plan: &WeightPlan) -> Result<BTreeMap<String, Companion>> {
        Ok(BTreeMap::new())
    }
}

/// One tensor of an artifact's optional part.
#[derive(Clone, Debug)]
pub(crate) struct Companion {
    pub(crate) shape: Vec<usize>,
    /// Stored packed in this precision; `None` copies the checkpoint's tensor.
    pub(crate) packed: Option<Precision>,
}

impl Companion {
    fn stored(&self) -> Result<Stored> {
        match (self.packed, self.shape.as_slice()) {
            (Some(precision), &[rows, cols]) => Stored::matrix(precision, rows, cols),
            _ => Ok(Stored::Float(self.shape.clone())),
        }
    }
}

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
    /// Whether the artifact holds its model's optional audio encoder.
    #[serde(default)]
    pub audio: bool,
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
fn expected(config: &impl ArtifactLayout, plan: &WeightPlan) -> Result<BTreeMap<String, Stored>> {
    let projections = config.projection_layout()?;
    let mut stored = BTreeMap::new();
    let mut parts = std::collections::BTreeSet::new();
    for p in &projections {
        parts.extend(p.parts.iter().map(|(name, _)| name.clone()));
        stored.insert(
            p.name.clone(),
            Stored::matrix(plan.precision(&p.name)?, p.rows, p.cols)?,
        );
    }
    let embedding = config.embedding_tensor();
    if let Embedding::Table(precision) = plan.embedding {
        stored.insert(
            embedding.clone(),
            Stored::matrix(precision, config.vocabulary(), config.residual_width())?,
        );
    }
    for (name, shape) in config.tensor_shapes()? {
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
/// before any payload is read. Tensors of unloaded optional parts are ignored.
pub(crate) fn open_for(dir: &Path, config: &impl ArtifactLayout) -> Result<(Checkpoint, Manifest)> {
    let (checkpoint, manifest) = open(dir)?;
    manifest.plan.validate(config)?;
    let mut expected = expected(config, &manifest.plan)?;
    if manifest.audio {
        for (name, companion) in config.companions(&manifest.plan)? {
            expected.insert(name, companion.stored()?);
        }
    }
    for (name, entry) in checkpoint.entries() {
        let Some(stored) = expected.get(name) else {
            if config.unloaded(name) {
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

/// Bytes reserved for the header at the start of an artifact, far above the few hundred
/// kilobytes of any supported model's tensor table and manifest. Spaces pad what is unused.
const HEADER_RESERVE: usize = 4 << 20;

/// Streams tensors straight into their final place in a partial `SafeTensors` file, then
/// fills in the reserved header and renames it. Every payload byte reaches the disk once,
/// and host memory holds one tensor at a time.
pub(crate) struct Writer {
    dir: PathBuf,
    partial: PathBuf,
    data: BufWriter<std::fs::File>,
    entries: Vec<(String, safetensors::Dtype, Vec<usize>, usize, usize)>,
    offset: usize,
}

impl Writer {
    /// Start an artifact in `dir`, creating it, and copy the source's serving files.
    pub(crate) fn create(source: &Path, dir: &Path) -> Result<Self> {
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
        let partial = dir.join(format!("{ARTIFACT_FILE}.partial"));
        let mut data = BufWriter::new(std::fs::File::create(&partial)?);
        data.seek(std::io::SeekFrom::Start(HEADER_RESERVE as u64))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            partial,
            data,
            entries: Vec::new(),
            offset: 0,
        })
    }

    pub(crate) fn put(
        &mut self,
        name: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<()> {
        self.put_stream(name, dtype, shape, |out| Ok(out.write_all(bytes)?))
    }

    /// Write a tensor in bounded chunks without retaining its complete payload.
    pub(crate) fn put_stream(
        &mut self,
        name: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        write: impl FnOnce(&mut dyn Write) -> Result<()>,
    ) -> Result<()> {
        let size = tur_core::execution::checked_elements(&shape)?
            .checked_mul(dtype.bitsize() / 8)
            .ok_or_else(|| invalid("artifact tensor size overflow"))?;
        let end = self
            .offset
            .checked_add(size)
            .ok_or_else(|| invalid("artifact payload size overflow"))?;
        write(&mut self.data)?;
        if self.data.stream_position()? != (HEADER_RESERVE + end) as u64 {
            return Err(invalid(format!(
                "artifact tensor {name} has an incorrect payload size"
            )));
        }
        self.entries
            .push((name.to_owned(), dtype, shape, self.offset, end));
        self.offset = end;
        Ok(())
    }

    /// Write the header carrying `manifest`, publish the file, and validate the result.
    pub(crate) fn finish(
        mut self,
        manifest: &Manifest,
        config: &impl ArtifactLayout,
    ) -> Result<()> {
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
        let room = HEADER_RESERVE - 8;
        if header.len() > room {
            return Err(invalid(format!(
                "artifact header of {} bytes exceeds its {room} reserved bytes",
                header.len()
            )));
        }
        header.resize(room, b' ');
        self.data.seek(std::io::SeekFrom::Start(0))?;
        self.data.write_all(&(room as u64).to_le_bytes())?;
        self.data.write_all(&header)?;
        self.data.flush()?;
        drop(self.data);
        std::fs::rename(&self.partial, self.dir.join(ARTIFACT_FILE))?;
        open_for(&self.dir, config).map(|_| ())
    }
}

/// Source and destination of one artifact conversion.
pub struct Conversion<'a> {
    /// Source `SafeTensors` checkpoint directory.
    pub source: &'a Path,
    /// Exact identity of the source checkpoint (repository and commit).
    pub identity: &'a str,
    /// New artifact directory.
    pub output: &'a Path,
    /// Free-form description of the calibration data.
    pub calibration: &'a str,
}

impl Packer<'_> {
    /// Pack `projection` from `checkpoint` in `precision` and stream it into `writer`.
    pub(crate) fn write(
        &self,
        checkpoint: &Checkpoint,
        projection: &Projection,
        precision: Precision,
        writer: &mut Writer,
    ) -> Result<()> {
        let (dtype, shape) = match precision.ggml() {
            Some(_) => (
                safetensors::Dtype::U8,
                vec![projection.rows, precision.bytes(1, projection.cols)?],
            ),
            None => (
                safetensors::Dtype::F32,
                vec![projection.rows, projection.cols],
            ),
        };
        writer.put_stream(&projection.name, dtype, shape, |out| {
            self.pack_chunks(checkpoint, projection, precision, |bytes| {
                Ok(out.write_all(bytes)?)
            })
        })
    }
}

/// Pack every projection of `checkpoint` by `plan` without calibration statistics and
/// write a complete artifact, one tensor at a time.
pub(crate) fn write(
    conversion: &Conversion<'_>,
    config: &impl ArtifactLayout,
    checkpoint: &Checkpoint,
    plan: &WeightPlan,
    packer: &Packer<'_>,
) -> Result<Manifest> {
    plan.validate(config)?;
    let mut writer = Writer::create(conversion.source, conversion.output)?;
    for projection in config.projection_layout()? {
        packer.write(
            checkpoint,
            &projection,
            plan.precision(&projection.name)?,
            &mut writer,
        )?;
    }
    finish(conversion, config, plan, packer, checkpoint, writer)
}

/// Write the embedding table and every remaining checkpoint tensor verbatim, then the
/// manifest; `writer` already holds the packed projections.
pub(crate) fn finish(
    conversion: &Conversion<'_>,
    config: &impl ArtifactLayout,
    plan: &WeightPlan,
    packer: &Packer<'_>,
    checkpoint: &Checkpoint,
    mut writer: Writer,
) -> Result<Manifest> {
    let parts: std::collections::BTreeSet<_> = config
        .projection_layout()?
        .into_iter()
        .flat_map(|p| p.parts.into_iter().map(|(name, _)| name))
        .collect();
    let embedding = config.embedding_tensor();
    if let Embedding::Table(precision) = plan.embedding {
        let table = Projection {
            name: embedding.clone(),
            parts: vec![(embedding.clone(), config.vocabulary())],
            rows: config.vocabulary(),
            cols: config.residual_width(),
            experts: 1,
        };
        if precision == Precision::Bf16 {
            put_bf16(
                &mut writer,
                checkpoint,
                &embedding,
                vec![table.rows, table.cols],
            )?;
        } else {
            Packer::RTN.write(checkpoint, &table, precision, &mut writer)?;
        }
    }
    for (name, shape) in config.tensor_shapes()? {
        if parts.contains(&name) || name == embedding {
            continue;
        }
        let bytes = checked_bytes(&shape)?;
        let entry = checkpoint
            .entries()
            .get(&name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        // Multi-gigabyte BF16 tables (Gemma's per-layer embeddings) stream by rows.
        if entry.dtype == safetensors::Dtype::BF16 && bytes > STAGING_BYTES && shape.len() == 2 {
            put_bf16(&mut writer, checkpoint, &name, shape)?;
        } else {
            let (entry, bytes) = checkpoint.take_raw(&name)?;
            writer.put(&name, entry.dtype, entry.shape, &bytes)?;
        }
    }
    let companions = config.companions(plan)?;
    for (name, companion) in &companions {
        if let (Some(precision), &[rows, cols]) = (companion.packed, companion.shape.as_slice()) {
            let projection = Projection::new(name.clone(), vec![(name.clone(), rows)], cols);
            Packer::RTN.write(checkpoint, &projection, precision, &mut writer)?;
        } else {
            let (entry, bytes) = checkpoint.take_raw(name)?;
            writer.put(name, entry.dtype, entry.shape, &bytes)?;
        }
    }
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        source: conversion.identity.to_owned(),
        config_fingerprint: config_fingerprint(conversion.source)?,
        quantizer: Quantizer {
            name: "tur-quantize".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            method: packer.method,
            damp: packer.damp,
        },
        calibration: packer.calibration.map(|c| CalibrationInfo {
            samples: c.samples,
            tokens: c.tokens,
            fingerprint: format!("{:016x}", c.fingerprint),
            description: conversion.calibration.to_owned(),
        }),
        plan: plan.clone(),
        audio: !companions.is_empty(),
    };
    writer.finish(&manifest, config)?;
    Ok(manifest)
}

fn checked_bytes(shape: &[usize]) -> Result<usize> {
    tur_core::execution::checked_elements(shape)?
        .checked_mul(2)
        .ok_or_else(|| invalid("artifact tensor size overflow"))
}

/// Stream the 2-D tensor `name` into `writer` as BF16, a staging chunk of rows at a time.
fn put_bf16(
    writer: &mut Writer,
    checkpoint: &Checkpoint,
    name: &str,
    shape: Vec<usize>,
) -> Result<()> {
    let cols = shape.get(1).copied().unwrap_or(1).max(1);
    writer.put_stream(name, safetensors::Dtype::BF16, shape, |out| {
        for rows in checkpoint.take_rows(name, (STAGING_BYTES / (cols * 4)).max(1))? {
            out.write_all(&bf16_bytes(&rows?.to_dtype(DType::BF16)?)?)?;
        }
        Ok(())
    })
}

/// Little-endian BF16 bits of a BF16 tensor, via its raw host storage.
fn bf16_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let tensor = tensor.to_device(&Device::Cpu)?.contiguous()?;
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cpu(candle_core::CpuStorage::BF16(values)) => Ok(values
            [layout.start_offset()..layout.start_offset() + tensor.elem_count()]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect()),
        _ => Err(invalid("expected a host BF16 tensor")),
    }
}
