//! Per-projection storage precision under a resident byte budget.

mod loading;
mod packing;
use super::{WeightFormat, gguf::GgufTensor};
use candle_core::quantized::GgmlDType;
pub(crate) use loading::{Embeddings, Tensors};
pub(crate) use packing::STAGING_BYTES;
pub use packing::{Method, Packer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tur_core::{
    Result,
    execution::{checked_elements, invalid},
};
use tur_metal::quantize::packed_bytes;

/// One loaded projection: checkpoint matrices stacked along output rows into the matrix
/// the model multiplies by. Names are stable keys for weight plans, calibration
/// statistics and prepacked artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// Checkpoint module path of the loaded matrix, e.g. `….layers.0.self_attn.kv_proj`.
    pub name: String,
    /// Checkpoint tensors in stacking order, with the output rows each contributes.
    pub parts: Vec<(String, usize)>,
    pub rows: usize,
    pub cols: usize,
    /// Experts of a stacked expert bank, each owning `rows / experts` rows; 1 otherwise.
    pub experts: usize,
}

impl Projection {
    pub(crate) fn new(name: String, parts: Vec<(String, usize)>, cols: usize) -> Self {
        let rows = parts.iter().map(|(_, rows)| rows).sum();
        Self {
            name,
            parts,
            rows,
            cols,
            experts: 1,
        }
    }

    pub(crate) fn bank(name: String, experts: usize, rows: usize, cols: usize) -> Self {
        Self {
            parts: vec![(name.clone(), experts * rows)],
            name,
            rows: experts * rows,
            cols,
            experts,
        }
    }
}

/// Architecture-owned matrix layout used by the common weight planner.
pub trait WeightLayout {
    /// Loaded projection names, their checkpoint parts and checked dimensions.
    fn projection_layout(&self) -> Result<Vec<Projection>>;
    /// Whether input embeddings and the LM head share checkpoint weights.
    fn embeddings_tied(&self) -> bool;
    /// Padded vocabulary width of the projection and embedding.
    fn vocabulary(&self) -> usize;
    /// Width of the residual stream and input embedding rows.
    fn residual_width(&self) -> usize;
    /// Whether an optional native draft head is included in this layout.
    fn loads_mtp(&self) -> bool {
        false
    }
    /// `general.architecture` values of the GGUF files this model reads; none by default.
    fn gguf_architectures(&self) -> &'static [&'static str] {
        &[]
    }
    /// What this model reads from the GGUF tensor `name`, or `None` for one it cannot use.
    fn gguf_tensor(&self, _name: &str) -> Option<GgufTensor> {
        None
    }
    /// Whether a GGUF file may hold the checkpoint tensor `name` that this model never reads.
    fn gguf_redundant(&self, _name: &str) -> bool {
        false
    }
    /// Every checkpoint tensor's name and shape, for matching a GGUF file against.
    fn stored_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        Err(invalid("this model has no GGUF layout"))
    }
    /// Checkpoint name of the input embedding.
    fn embedding_tensor_name(&self) -> String {
        String::new()
    }
}

/// Storage of one weight matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    /// Embedding tables only: projections multiply in F32 or packed blocks.
    Bf16,
    Q8_0,
    /// Embedding tables only: Candle's 2–16-row kernels for Q6K/Q5K are 2.5–3.5× slower
    /// than Tur's `Q8_0`/Q4K ones on the M1, so projections exclude them.
    Q6k,
    Q5k,
    Q4k,
    /// As a GGUF file stores it, possibly mixing formats; the plan records its bytes.
    Stored,
}

impl Precision {
    /// Projection precisions with Tur kernels at every batch size, widest first.
    pub const PROJECTIONS: [Self; 3] = [Self::F32, Self::Q8_0, Self::Q4k];

    /// GGML block format, or `None` for unpacked storage.
    pub fn ggml(self) -> Option<GgmlDType> {
        match self {
            Self::F32 | Self::Bf16 | Self::Stored => None,
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::Q6k => Some(GgmlDType::Q6K),
            Self::Q5k => Some(GgmlDType::Q5K),
            Self::Q4k => Some(GgmlDType::Q4K),
        }
    }

    /// Stored bytes of a `[rows, cols]` matrix.
    pub fn bytes(self, rows: usize, cols: usize) -> Result<usize> {
        match self {
            Self::F32 => checked_elements(&[rows, cols, 4]),
            Self::Bf16 => checked_elements(&[rows, cols, 2]),
            _ => packed_bytes(
                self.ggml().ok_or_else(|| invalid("unpacked precision"))?,
                rows,
                cols,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q6k => "q6k",
            Self::Q5k => "q5k",
            Self::Q4k => "q4k",
            Self::Stored => "stored",
        }
    }
}

impl std::str::FromStr for Precision {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        [
            Self::F32,
            Self::Bf16,
            Self::Q8_0,
            Self::Q6k,
            Self::Q5k,
            Self::Q4k,
        ]
        .into_iter()
        .find(|p| p.name() == value.to_ascii_lowercase())
        .ok_or_else(|| format!("unknown precision '{value}' (f32, bf16, q8_0, q6k, q5k, q4k)"))
    }
}

impl From<WeightFormat> for Precision {
    fn from(format: WeightFormat) -> Self {
        match format {
            WeightFormat::F32 => Self::F32,
            WeightFormat::Q8_0 => Self::Q8_0,
            WeightFormat::Q4K => Self::Q4k,
        }
    }
}

/// Where input embeddings are gathered from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Embedding {
    /// Rows of the tied LM head, in its precision: no second table.
    Head,
    /// A separate table in this precision.
    Table(Precision),
}

/// Storage precision of every projection and of the embedding. Norms, gates, convolution,
/// decay parameters and expert routers always stay F32.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightPlan {
    pub projections: BTreeMap<String, Precision>,
    pub embedding: Embedding,
    /// Exact bytes of what is [`Precision::Stored`], by projection or embedding tensor name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stored: BTreeMap<String, usize>,
}

impl WeightPlan {
    /// Every projection in `format`. F32 shares a tied head with the embedding; packed
    /// formats keep a BF16 table, from which only gathered rows become F32.
    pub fn uniform(config: &impl WeightLayout, format: WeightFormat) -> Result<Self> {
        let precision = Precision::from(format);
        let embedding = match (format, config.embeddings_tied()) {
            (WeightFormat::F32, true) => Embedding::Head,
            (WeightFormat::F32, false) => Embedding::Table(Precision::F32),
            _ => Embedding::Table(Precision::Bf16),
        };
        Ok(Self {
            projections: config
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name, precision))
                .collect(),
            embedding,
            stored: BTreeMap::new(),
        })
    }

    /// Bytes of the first `rows` of projection `name`, of `total` rows and `cols` columns.
    pub(crate) fn leading_bytes(
        &self,
        name: &str,
        (rows, total, cols): (usize, usize, usize),
    ) -> Result<usize> {
        match self.precision(name)? {
            Precision::Stored => self
                .stored
                .get(name)
                .and_then(|bytes| bytes.checked_mul(rows))
                .map(|bytes| bytes.div_ceil(total.max(1)))
                .ok_or_else(|| invalid(format!("{name} has no stored size"))),
            precision => precision.bytes(rows, cols),
        }
    }

    /// Whether this plan loads a GGUF file's tensors as stored.
    pub(crate) fn is_stored(&self) -> bool {
        !self.stored.is_empty()
    }

    /// Precision of projection `name`.
    pub fn precision(&self, name: &str) -> Result<Precision> {
        self.projections
            .get(name)
            .copied()
            .ok_or_else(|| invalid(format!("weight plan has no precision for {name}")))
    }

    /// Check that the plan covers `config`'s projections in executable precisions (plus,
    /// at most, those of an MTP head the config does not load) and that a shared
    /// embedding has a tied head to share.
    pub fn validate(&self, config: &impl WeightLayout) -> Result<()> {
        let projections = config.projection_layout()?;
        let names: std::collections::BTreeSet<&str> =
            projections.iter().map(|p| p.name.as_str()).collect();
        if let Some(extra) = self
            .projections
            .keys()
            .find(|n| !names.contains(n.as_str()) && (config.loads_mtp() || !n.starts_with("mtp.")))
        {
            return Err(invalid(format!(
                "weight plan names {extra}, which the model does not have"
            )));
        }
        for p in &projections {
            let precision = self.precision(&p.name)?;
            if precision == Precision::Stored {
                if !self.stored.contains_key(&p.name) {
                    return Err(invalid(format!("{} has no stored size", p.name)));
                }
                continue;
            }
            if !Precision::PROJECTIONS.contains(&precision) {
                return Err(invalid(format!(
                    "{} cannot use {}: projections are f32, q8_0 or q4k",
                    p.name,
                    precision.name()
                )));
            }
            precision.bytes(p.rows, p.cols)?;
        }
        match self.embedding {
            Embedding::Head if !config.embeddings_tied() => Err(invalid(
                "an untied checkpoint has no head to gather embeddings from",
            )),
            Embedding::Table(Precision::Stored) | Embedding::Head => Ok(()),
            Embedding::Table(precision) => precision
                .bytes(config.vocabulary(), config.residual_width())
                .map(|_| ()),
        }
    }

    /// Resident bytes of the projections and embedding table.
    pub fn bytes(&self, config: &impl WeightLayout) -> Result<usize> {
        let mut total = match self.embedding {
            Embedding::Head => 0,
            Embedding::Table(Precision::Stored) => self
                .stored
                .get(&config.embedding_tensor_name())
                .copied()
                .ok_or_else(|| invalid("stored embedding has no size"))?,
            Embedding::Table(precision) => {
                precision.bytes(config.vocabulary(), config.residual_width())?
            }
        };
        for p in config.projection_layout()? {
            let bytes = match self.precision(&p.name)? {
                Precision::Stored => self
                    .stored
                    .get(&p.name)
                    .copied()
                    .ok_or_else(|| invalid(format!("{} has no stored size", p.name)))?,
                precision => precision.bytes(p.rows, p.cols)?,
            };
            total = total
                .checked_add(bytes)
                .ok_or_else(|| invalid("weight total overflow"))?;
        }
        Ok(total)
    }

    /// Short identity of the plan for state namespaces: the uniform precision's name, or
    /// a digest of the whole plan when mixed.
    pub fn tag(&self) -> String {
        let mut precisions = self.projections.values();
        match precisions.next() {
            Some(first) if precisions.all(|p| p == first) => {
                let embedding = match self.embedding {
                    Embedding::Head => "head".to_owned(),
                    Embedding::Table(p) => p.name().to_owned(),
                };
                format!("{}+{embedding}", first.name())
            }
            _ => format!(
                "mixed-{:016x}",
                fnv1a(serde_json::to_string(self).unwrap_or_default().as_bytes())
            ),
        }
    }
}

/// 64-bit FNV-1a: a stable content fingerprint for identities, not a security hash.
/// Storage of a `[_, cols]` projection in `precision`: the same format, or F32 when its
/// rows do not tile into the format's blocks (small test widths).
pub(crate) fn tiled(precision: Precision, cols: usize) -> Precision {
    match precision.ggml() {
        Some(format) if !cols.is_multiple_of(format.block_size()) => Precision::F32,
        _ => precision,
    }
}

/// Upload a `[rows, cols]` matrix as a projection stored in [`tiled`] `precision`.
pub(crate) fn upload_projection(
    weight: &candle_core::Tensor,
    precision: Precision,
    device: &candle_core::Device,
) -> Result<tur_metal::packed::PackedLinear> {
    tiled(precision, weight.dim(1)?).ggml().map_or_else(
        || tur_metal::packed::PackedLinear::dense(weight, device),
        |format| tur_metal::packed::PackedLinear::quantized(weight, format, device),
    )
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::Config;

    const SMALL: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/small-config.json");

    #[test]
    fn test_uniform_plans_validate_and_mixed_plans_reject_unexecutable_precisions() -> Result<()> {
        let config = Config::from_json(SMALL)?;
        for format in [WeightFormat::F32, WeightFormat::Q8_0, WeightFormat::Q4K] {
            let plan = WeightPlan::uniform(&config, format)?;
            plan.validate(&config)?;
            assert!(!plan.tag().starts_with("mixed"));
        }
        let mut plan = WeightPlan::uniform(&config, WeightFormat::Q4K)?;
        let q4k = plan.bytes(&config)?;
        let first = config.projection_layout()?.remove(0);
        plan.projections.insert(first.name.clone(), Precision::Q8_0);
        plan.validate(&config)?;
        assert!(plan.tag().starts_with("mixed-"));
        assert_eq!(
            plan.bytes(&config)? - q4k,
            Precision::Q8_0.bytes(first.rows, first.cols)?
                - Precision::Q4k.bytes(first.rows, first.cols)?
        );
        plan.projections.insert(first.name.clone(), Precision::Q6k);
        assert!(plan.validate(&config).is_err());
        plan.projections.remove(&first.name);
        assert!(plan.validate(&config).is_err());
        plan.projections.insert(first.name.clone(), Precision::Q4k);
        plan.projections
            .insert("model.unknown".into(), Precision::Q4k);
        assert!(plan.validate(&config).is_err());
        let mut plan = WeightPlan::uniform(&config, WeightFormat::Q8_0)?;
        plan.embedding = Embedding::Head;
        assert_eq!(plan.validate(&config).is_ok(), config.tied_embeddings());
        Ok(())
    }

    #[test]
    fn test_plans_round_trip_through_json() -> Result<()> {
        let config = Config::from_json(SMALL)?;
        let mut plan = WeightPlan::uniform(&config, WeightFormat::Q4K)?;
        plan.embedding = Embedding::Table(Precision::Q6k);
        let json = serde_json::to_string(&plan)?;
        assert!(json.contains("\"q4k\"") && json.contains("{\"table\":\"q6k\"}"));
        assert_eq!(serde_json::from_str::<WeightPlan>(&json)?, plan);
        assert_eq!("Q8_0".parse::<Precision>(), Ok(Precision::Q8_0));
        Ok(())
    }
}
