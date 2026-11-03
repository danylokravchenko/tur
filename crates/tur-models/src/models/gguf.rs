//! GGUF weight files as llama.cpp's converter writes them: tensors in ggml's block formats
//! under ggml's names, loaded as stored into a model configured by its own checkpoint's
//! `config.json`. The tokenizer comes from that checkpoint too, and must match the file's
//! token list; a chat template in the file replaces the checkpoint's.

use crate::models::{
    checkpoint::{Adjust, Checkpoint, Entry},
    weights::{Embedding, Precision, Projection, WeightLayout, WeightPlan, fnv1a},
};
use std::{
    collections::BTreeMap,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
use tur_core::{Result, execution::invalid};
use tur_metal::blocks::BlockFormat;

/// Tensor tables and metadata larger than this are not GGUF files any model here needs.
const MAX_ENTRIES: u64 = 1 << 24;
const MAX_STRING_BYTES: u64 = 1 << 30;
const DEFAULT_ALIGNMENT: u64 = 32;
/// The only array this reader keeps: the token strings, indexed by ID.
const TOKENS: &str = "tokenizer.ggml.tokens";

/// What a model reads from one GGUF tensor.
pub enum GgufTensor {
    /// The checkpoint tensor `name`, after undoing what llama.cpp's converter did to it.
    Stored { name: String, adjust: Adjust },
    /// Nothing: a tensor derivable from the config.
    Ignored,
}

/// One tensor's location in a set of split files.
#[derive(Clone, Debug)]
struct Located {
    name: String,
    file: usize,
    /// Dimensions outermost first, as checkpoints order them; ggml stores the reverse.
    shape: Vec<usize>,
    ggml_type: u32,
    offset: u64,
}

/// A validated GGUF file, or the shards of a split one; no weight payload has been read.
#[derive(Debug)]
pub struct GgufFile {
    files: Vec<PathBuf>,
    tensors: Vec<Located>,
    architecture: String,
    /// Identity of the numerical contents, for state namespaces.
    fingerprint: u64,
    /// `tokenizer.chat_template`, which a fine-tune may have changed.
    chat_template: Option<String>,
    /// `tokenizer.ggml.tokens`: every token string by ID.
    tokens: Option<Vec<String>>,
    /// `tokenizer.ggml.eos_token_id`.
    eos_token: Option<u32>,
}

struct Reader<R> {
    inner: R,
    length: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.inner.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn string(&mut self) -> Result<String> {
        let length = self.u64()?;
        let left = self.length.saturating_sub(self.inner.stream_position()?);
        if length > MAX_STRING_BYTES || length > left {
            return Err(invalid("GGUF string runs past the file"));
        }
        let mut bytes = vec![0; length as usize];
        self.inner.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid("GGUF string is not UTF-8"))
    }
    /// An array of strings, after its value type.
    fn strings(&mut self) -> Result<Vec<String>> {
        let (element, count) = (self.u32()?, self.u64()?);
        if element != 8 || count > MAX_ENTRIES {
            return Err(invalid(format!("GGUF {TOKENS} is not a list of strings")));
        }
        (0..count).map(|_| self.string()).collect()
    }
    fn skip(&mut self, bytes: u64) -> Result<()> {
        let at = self.inner.stream_position()?;
        match at.checked_add(bytes) {
            Some(end) if end <= self.length => {
                self.inner.seek(SeekFrom::Start(end))?;
                Ok(())
            }
            _ => Err(invalid("GGUF value runs past the file")),
        }
    }
    /// A metadata value's size, skipped; `Some` holds a scalar or string this reader keeps.
    fn value(&mut self, kind: u32) -> Result<Option<Value>> {
        let width = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            5 | 6 => 4,
            11 | 12 => 8,
            4 => return Ok(Some(Value::Number(u64::from(self.u32()?)))),
            10 => return Ok(Some(Value::Number(self.u64()?))),
            8 => return Ok(Some(Value::Text(self.string()?))),
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if element == 8 {
                    for _ in 0..count {
                        let length = self.u64()?;
                        self.skip(length)?;
                    }
                } else if element == 9 {
                    return Err(invalid("GGUF nested arrays are not supported"));
                } else {
                    let width = match element {
                        0 | 1 | 7 => 1,
                        2 | 3 => 2,
                        4..=6 => 4,
                        10..=12 => 8,
                        other => return Err(invalid(format!("GGUF value type {other}"))),
                    };
                    self.skip(
                        count
                            .checked_mul(width)
                            .ok_or_else(|| invalid("GGUF array size overflows"))?,
                    )?;
                }
                return Ok(None);
            }
            other => return Err(invalid(format!("GGUF value type {other}"))),
        };
        self.skip(width)?;
        Ok(None)
    }
}

enum Value {
    Number(u64),
    Text(String),
    Strings(Vec<String>),
}

/// Header of one file: its tensors, with absolute payload offsets, and the metadata kept.
fn read_header(path: &Path, file: usize) -> Result<(Vec<Located>, BTreeMap<String, Value>)> {
    let handle = std::fs::File::open(path)?;
    let length = handle.metadata()?.len();
    let mut reader = Reader {
        inner: BufReader::new(handle),
        length,
    };
    if &reader.take::<4>()? != b"GGUF" {
        return Err(invalid(format!("{} is not a GGUF file", path.display())));
    }
    let version = reader.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(invalid(format!("GGUF version {version} is not supported")));
    }
    let (tensor_count, metadata_count) = (reader.u64()?, reader.u64()?);
    if tensor_count > MAX_ENTRIES || metadata_count > MAX_ENTRIES {
        return Err(invalid("GGUF header declares too many entries"));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        if key == TOKENS && kind == 9 {
            metadata.insert(key, Value::Strings(reader.strings()?));
        } else if let Some(value) = reader.value(kind)? {
            metadata.insert(key, value);
        }
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dims = reader.u32()?;
        if dims == 0 || dims > 4 {
            return Err(invalid(format!("GGUF tensor {name} has {dims} dimensions")));
        }
        let mut shape = (0..dims)
            .map(|_| usize::try_from(reader.u64()?).map_err(|_| invalid("GGUF dimension")))
            .collect::<Result<Vec<_>>>()?;
        shape.reverse();
        let ggml_type = reader.u32()?;
        let offset = reader.u64()?;
        tensors.push(Located {
            name,
            file,
            shape,
            ggml_type,
            offset,
        });
    }
    let alignment = match metadata.get("general.alignment") {
        Some(Value::Number(alignment)) if *alignment > 0 => *alignment,
        _ => DEFAULT_ALIGNMENT,
    };
    let start = reader
        .inner
        .stream_position()?
        .checked_next_multiple_of(alignment)
        .ok_or_else(|| invalid("GGUF data offset overflows"))?;
    for tensor in &mut tensors {
        tensor.offset = start
            .checked_add(tensor.offset)
            .filter(|&offset| offset <= length)
            .ok_or_else(|| invalid(format!("GGUF tensor {} is outside its file", tensor.name)))?;
        if let Some(format) = BlockFormat::from_ggml(tensor.ggml_type) {
            let (rows, cols) = rows_cols(&tensor.shape)?;
            let end = format
                .bytes(rows, cols)
                .ok()
                .and_then(|bytes| tensor.offset.checked_add(bytes as u64));
            if end.is_none_or(|end| end > length) {
                return Err(invalid(format!(
                    "GGUF tensor {} runs past its file",
                    tensor.name
                )));
            }
        }
    }
    Ok((tensors, metadata))
}

/// Rows and row width of a tensor; every dimension but the innermost counts rows.
fn rows_cols(shape: &[usize]) -> Result<(usize, usize)> {
    let (cols, outer) = shape
        .split_last()
        .ok_or_else(|| invalid("GGUF tensor has no dimensions"))?;
    let rows = outer
        .iter()
        .try_fold(1usize, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(|| invalid("GGUF tensor shape overflows"))?;
    Ok((rows, *cols))
}

/// `-00001-of-00003.gguf` as `(stem, shard count)`.
pub(crate) fn split_name(name: &str) -> Option<(&str, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    let (stem, count) = stem.rsplit_once("-of-")?;
    let (stem, index) = stem.rsplit_once('-')?;
    (index == "00001" && count.len() == 5)
        .then(|| count.parse().ok().map(|count| (stem, count)))
        .flatten()
}

/// File name of shard `index` (from one) of `count`.
pub(crate) fn shard_name(stem: &str, index: usize, count: usize) -> String {
    format!("{stem}-{index:05}-of-{count:05}.gguf")
}

impl GgufFile {
    /// Read and check the headers of `path`, and of its other shards when it is the first
    /// of a split file. Tensors must lie inside their files.
    pub fn open(path: &Path) -> Result<Self> {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("GGUF path has no file name"))?;
        let files = match split_name(name) {
            Some((stem, count)) => (1..=count)
                .map(|index| path.with_file_name(shard_name(stem, index, count)))
                .collect(),
            None => vec![path.to_path_buf()],
        };
        let mut tensors = Vec::new();
        // llama.cpp's splitter writes the model's metadata into the first shard only.
        let mut first = None;
        for (file, shard) in files.iter().enumerate() {
            let (found, metadata) = read_header(shard, file)?;
            tensors.extend(found);
            first.get_or_insert(metadata);
        }
        let mut metadata = first.unwrap_or_default();
        let Some(Value::Text(architecture)) = metadata.remove("general.architecture") else {
            return Err(invalid("GGUF file names no general.architecture"));
        };
        let chat_template = match metadata.remove("tokenizer.chat_template") {
            Some(Value::Text(template)) => Some(template),
            _ => None,
        };
        let tokens = match metadata.remove(TOKENS) {
            Some(Value::Strings(tokens)) => Some(tokens),
            _ => None,
        };
        let eos_token = match metadata.get("tokenizer.ggml.eos_token_id") {
            Some(&Value::Number(id)) => {
                Some(u32::try_from(id).map_err(|_| invalid("GGUF eos_token_id is out of range"))?)
            }
            _ => None,
        };
        let table = tensors
            .iter()
            .map(|t| format!("{} {:?} {} {}\n", t.name, t.shape, t.ggml_type, t.offset))
            .collect::<Vec<_>>()
            .concat();
        Ok(Self {
            files,
            fingerprint: fnv1a(table.as_bytes()),
            tensors,
            architecture,
            chat_template,
            tokens,
            eos_token,
        })
    }

    /// The file's chat template, when it carries one.
    pub(crate) fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Check that every token `tokenizer` defines is spelled as the file spells it, so the
    /// checkpoint the tokenizer came from was not edited apart from the file. IDs the
    /// tokenizer leaves unused may hold anything: llama.cpp's `[PAD<id>]` filler, or special
    /// tokens of a newer tokenizer revision. Returns the file's end-of-sequence token, if any.
    pub(crate) fn check_tokenizer(&self, tokenizer: &tokenizers::Tokenizer) -> Result<Option<u32>> {
        for (id, token) in self.tokens.iter().flatten().enumerate() {
            let id = u32::try_from(id).map_err(|_| invalid("GGUF token list is too long"))?;
            if let Some(expected) = tokenizer.id_to_token(id)
                && expected != *token
            {
                return Err(invalid(format!(
                    "GGUF token {id} is {token:?}, but the checkpoint's tokenizer has \
                     {expected:?}; the file needs the tokenizer it was converted with"
                )));
            }
        }
        if let Some(eos) = self.eos_token
            && self
                .tokens
                .as_ref()
                .is_some_and(|tokens| eos as usize >= tokens.len())
        {
            return Err(invalid(format!("GGUF eos_token_id {eos} is not a token")));
        }
        Ok(self.eos_token)
    }

    /// Stable identity of the weights, for state namespaces.
    pub(crate) fn identity(&self) -> String {
        format!("gguf-{:016x}", self.fingerprint)
    }

    /// Bytes of every tensor by format, for reporting what a file holds.
    pub fn formats(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for tensor in &self.tensors {
            if let Some(format) = BlockFormat::from_ggml(tensor.ggml_type) {
                *counts.entry(format.name()).or_default() += 1;
            }
        }
        counts
    }

    /// Checkpoint view of the file under `config`'s tensor names. Every tensor the config
    /// expects must exist in the expected shape and a readable format, before any payload.
    pub(crate) fn checkpoint(&self, config: &impl WeightLayout) -> Result<Checkpoint> {
        if config.gguf_architectures().is_empty() {
            return Err(invalid(
                "this model family does not load GGUF files; use its SafeTensors checkpoint",
            ));
        }
        if !config
            .gguf_architectures()
            .contains(&self.architecture.as_str())
        {
            return Err(invalid(format!(
                "GGUF architecture {} does not match this checkpoint, which reads {:?}",
                self.architecture,
                config.gguf_architectures()
            )));
        }
        let expected = config.stored_shapes()?;
        let mut entries = BTreeMap::new();
        for tensor in &self.tensors {
            let (name, adjust) = match config.gguf_tensor(&tensor.name) {
                Some(GgufTensor::Stored { name, adjust }) => (name, adjust),
                Some(GgufTensor::Ignored) => continue,
                None => {
                    return Err(invalid(format!(
                        "GGUF tensor {} has no place in this model",
                        tensor.name
                    )));
                }
            };
            // A tied checkpoint's head is its embedding, even if the file repeats it.
            if name == "lm_head.weight" && config.embeddings_tied() {
                continue;
            }
            if !expected.contains_key(&name) && config.gguf_redundant(&name) {
                continue;
            }
            let shape = expected_shape(&expected, &name, &tensor.shape)?;
            let format = BlockFormat::from_ggml(tensor.ggml_type).ok_or_else(|| {
                invalid(format!(
                    "GGUF tensor {} uses ggml type {}, which Tur cannot read",
                    tensor.name, tensor.ggml_type
                ))
            })?;
            let (rows, cols) = rows_cols(&tensor.shape)?;
            let entry = Entry::stored(
                tensor.file,
                shape,
                format,
                tensor.offset,
                format.bytes(rows, cols)?,
                adjust,
            );
            if entries.insert(name.clone(), entry).is_some() {
                return Err(invalid(format!("GGUF file repeats {name}")));
            }
        }
        for name in expected.keys() {
            let found = name.strip_suffix(".gate_up_proj").map_or_else(
                || entries.contains_key(name),
                |prefix| {
                    entries.contains_key(&format!("{prefix}.gate_proj"))
                        && entries.contains_key(&format!("{prefix}.up_proj"))
                },
            );
            if !found {
                return Err(invalid(format!("GGUF file has no tensor for {name}")));
            }
        }
        Ok(Checkpoint::from_entries(self.files.clone(), entries))
    }

    /// Weight plan of the file under `config`: every projection as stored, with its exact
    /// bytes. A head only the block kernels read cannot serve embeddings, which then get
    /// a BF16 table.
    pub(crate) fn plan(&self, config: &impl WeightLayout) -> Result<WeightPlan> {
        let checkpoint = self.checkpoint(config)?;
        let mut stored = BTreeMap::new();
        let mut projections = BTreeMap::new();
        for projection in config.projection_layout()? {
            let parts = stored_parts(&projection);
            let bytes = if let ([gate, up], true) = (parts.as_slice(), projection.experts > 1) {
                bank_format(
                    checkpoint.entry(gate)?.stored_format()?,
                    checkpoint.entry(up)?.stored_format()?,
                )
                .bytes(projection.rows, projection.cols)?
            } else {
                let mut bytes = 0usize;
                for part in &parts {
                    bytes = checkpoint
                        .entry(part)?
                        .bytes
                        .checked_add(bytes)
                        .ok_or_else(|| invalid("weight total overflow"))?;
                }
                bytes
            };
            projections.insert(projection.name.clone(), Precision::Stored);
            stored.insert(projection.name, bytes);
        }
        let name = config.embedding_tensor_name();
        let table = checkpoint.entry(&name)?;
        let format = table.stored_format()?;
        let embedding = match (
            format.candle().is_some() || format == BlockFormat::F32,
            config.embeddings_tied(),
        ) {
            (true, true) => Embedding::Head,
            (true, false) if format == BlockFormat::F32 => Embedding::Table(Precision::F32),
            (true, false) => {
                stored.insert(name, table.bytes);
                Embedding::Table(Precision::Stored)
            }
            (false, _) => Embedding::Table(Precision::Bf16),
        };
        Ok(WeightPlan {
            projections,
            embedding,
            stored,
        })
    }
}

/// The one format of an expert bank stored as `gate` and `up`: the expert kernels read a
/// single format, so tensors that differ are packed again at least as wide as the wider.
pub(crate) fn bank_format(gate: BlockFormat, up: BlockFormat) -> BlockFormat {
    use BlockFormat::{Q2K, Q3K, Q4K, Q5K, Q6K, Q8_0};
    if gate == up {
        return gate;
    }
    let bits = |format: BlockFormat| format.block_bytes() as f64 / format.block_values() as f64;
    let wider = bits(gate).max(bits(up));
    [Q2K, Q3K, Q4K, Q5K, Q6K, Q8_0]
        .into_iter()
        .find(|&format| bits(format) >= wider)
        .unwrap_or(Q8_0)
}

/// Parts a projection stores in a GGUF file, in stacking order. llama.cpp splits a stacked
/// expert bank's gate and up rows into one tensor each.
pub(crate) fn stored_parts(projection: &Projection) -> Vec<String> {
    projection
        .name
        .strip_suffix(".gate_up_proj")
        .filter(|_| projection.experts > 1)
        .map_or_else(
            || {
                projection
                    .parts
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect()
            },
            |prefix| vec![format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
        )
}

/// The checkpoint's shape for `name`, accepting the single-element dimensions the converter
/// removes, and half-height parts of a stacked gate-and-up bank.
fn expected_shape(
    expected: &BTreeMap<String, Vec<usize>>,
    name: &str,
    found: &[usize],
) -> Result<Vec<usize>> {
    let unit =
        |shape: &[usize]| -> Vec<usize> { shape.iter().copied().filter(|&d| d != 1).collect() };
    let mismatch =
        |shape: &[usize]| invalid(format!("GGUF has {name} as {found:?}, expected {shape:?}"));
    if let Some(shape) = expected.get(name) {
        return if shape.as_slice() == found || unit(shape) == unit(found) {
            Ok(shape.clone())
        } else {
            Err(mismatch(shape))
        };
    }
    let bank = name
        .strip_suffix(".gate_proj")
        .or_else(|| name.strip_suffix(".up_proj"))
        .and_then(|prefix| expected.get(&format!("{prefix}.gate_up_proj")));
    let shape = bank.ok_or_else(|| invalid(format!("this model has no tensor {name}")))?;
    match shape.as_slice() {
        &[experts, rows, cols] if found == [experts, rows / 2, cols] => Ok(found.to_vec()),
        _ => Err(mismatch(shape)),
    }
}
