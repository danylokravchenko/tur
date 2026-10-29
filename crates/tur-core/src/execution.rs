//! Host-side contracts checked before device submission.

use candle_core::DType;
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};
use uuid::Uuid;

use crate::{Result, TurError};

/// Exact checkpoint/configuration/adapter/position-policy identity and state format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelIdentity {
    pub revision: String,
    pub state_format: u32,
}

/// Input kinds a model accepts. Packed execution currently carries text tokens only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

/// State required by one model layer. Attention storage follows the model's [`KvLayout`];
/// recurrent layers declare the storage dtype of their matrix state, while the short
/// convolution history is always F32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerStateKind {
    Attention {
        kv_heads: usize,
        head_dim: usize,
        /// Most recent keys any later query or draft of a sequence reads. A windowed layer
        /// stores only the pages covering them, in the pool's window space.
        window: Option<usize>,
    },
    Recurrent {
        conv_channels: usize,
        conv_width: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        /// Storage of the `[heads, key_dim, value_dim]` matrix state: F32, F16 or BF16.
        /// Updates always accumulate in F32; a narrower dtype rounds the state each time
        /// it is stored.
        dtype: DType,
    },
}

/// Version of the paged K/V memory layout. Bump when page, scale or code layouts change, so
/// checkpoints and persisted prefixes from another layout are rejected rather than misread.
/// 2: windowed attention layers keep only their recent pages, in a separate space.
pub const KV_LAYOUT_VERSION: u32 = 2;

/// Storage format of attention keys and values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvFormat {
    F32,
    F16,
    /// Symmetric 8-bit codes with one F16 scale per `group` consecutive head dimensions.
    /// Codes are two's-complement bytes in U8 storage; the scale is `max |x| / 127` of
    /// the group.
    Q8 {
        group: usize,
    },
}

impl KvFormat {
    /// Default Q8 group: one scale per 32 dimensions.
    pub const Q8: Self = Self::Q8 { group: 32 };

    /// Dtype of the stored codes or values.
    pub fn data_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Q8 { .. } => DType::U8,
        }
    }

    /// Head dimensions sharing one scale, or `None` for unscaled formats.
    pub fn group(self) -> Option<usize> {
        match self {
            Self::Q8 { group } => Some(group),
            Self::F32 | Self::F16 => None,
        }
    }

    /// Reject groups that do not tile `head_dim` in whole 32-lane SIMD rows.
    pub fn validate(self, head_dim: usize) -> Result<()> {
        if let Some(group) = self.group()
            && (group == 0 || !group.is_multiple_of(32) || !head_dim.is_multiple_of(group))
        {
            return Err(invalid(format!(
                "K/V group {group} must be a multiple of 32 dividing head dim {head_dim}"
            )));
        }
        Ok(())
    }

    /// Bytes of one token's keys (or values): codes plus F16 scales.
    pub fn token_bytes(self, kv_heads: usize, head_dim: usize) -> Result<usize> {
        let data = checked_elements(&[kv_heads, head_dim, self.data_dtype().size_in_bytes()])?;
        let scales = match self.group() {
            Some(group) if group > 0 => checked_elements(&[kv_heads, head_dim / group, 2])?,
            Some(_) => return Err(invalid("zero K/V group")),
            None => 0,
        };
        data.checked_add(scales)
            .ok_or_else(|| invalid("K/V token size overflow"))
    }
}

impl std::fmt::Display for KvFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::F16 => f.write_str("f16"),
            Self::Q8 { group } => write!(f, "q8_{group}"),
        }
    }
}

impl std::str::FromStr for KvFormat {
    type Err = String;

    /// `f32`, `f16`, `q8` (group 32) or `q8_<group>`.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let value = value.to_ascii_lowercase();
        match value.as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "q8" => Ok(Self::Q8),
            _ => value
                .strip_prefix("q8_")
                .and_then(|group| group.parse().ok())
                .map(|group| Self::Q8 { group })
                .ok_or_else(|| "K/V format must be f32, f16, q8 or q8_<group>".into()),
        }
    }
}

/// Paged K/V layout shared by every attention layer: page capacity, storage format and
/// layout version. Page size and quantization group are validated independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvLayout {
    pub page_tokens: usize,
    pub format: KvFormat,
    pub version: u32,
}

impl KvLayout {
    /// A layout at the current [`KV_LAYOUT_VERSION`].
    pub fn new(page_tokens: usize, format: KvFormat) -> Self {
        Self {
            page_tokens,
            format,
            version: KV_LAYOUT_VERSION,
        }
    }
}

/// Model-declared layout; this describes state, not model-specific configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDescription {
    pub identity: ModelIdentity,
    pub layers: Vec<LayerStateKind>,
    /// Paged storage of every attention layer.
    pub kv: KvLayout,
    pub modalities: Vec<Modality>,
    pub vocab_size: usize,
    pub max_position: usize,
}

/// Error for invalid execution descriptors and state transitions.
pub fn invalid(message: impl Into<String>) -> TurError {
    TurError::InvalidExecution(message.into())
}

/// Compute an allocation size without overflow or zero dimensions.
pub fn checked_elements(dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |size, &dim| {
        if dim == 0 {
            return Err(invalid("zero state dimension"));
        }
        size.checked_mul(dim)
            .ok_or_else(|| invalid("state allocation overflow"))
    })
}

fn state_dtype_bytes(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(dtype.size_in_bytes()),
        _ => Err(invalid("state dtype must be F32, F16 or BF16")),
    }
}

impl LayerStateKind {
    /// Whether this is a windowed attention layer.
    pub fn windowed(&self) -> bool {
        matches!(
            self,
            Self::Attention {
                window: Some(_),
                ..
            }
        )
    }

    /// K plus V bytes for one token in `format`, including scales; zero for recurrent layers.
    pub fn attention_token_bytes(&self, format: KvFormat) -> Result<usize> {
        match *self {
            Self::Attention {
                kv_heads, head_dim, ..
            } => format
                .token_bytes(kv_heads, head_dim)?
                .checked_mul(2)
                .ok_or_else(|| invalid("K/V token size overflow")),
            Self::Recurrent { .. } => Ok(0),
        }
    }

    /// Convolution history plus recurrent matrix bytes at one boundary; zero for attention.
    pub fn recurrent_boundary_bytes(&self) -> Result<usize> {
        match *self {
            Self::Attention { .. } => Ok(0),
            Self::Recurrent {
                conv_channels,
                conv_width,
                heads,
                key_dim,
                value_dim,
                dtype,
            } => {
                let bytes = state_dtype_bytes(dtype)?;
                checked_elements(&[conv_channels, conv_width, DType::F32.size_in_bytes()])?
                    .checked_add(checked_elements(&[heads, key_dim, value_dim, bytes])?)
                    .ok_or_else(|| invalid("recurrent payload overflow"))
            }
        }
    }
}

impl ModelDescription {
    /// K/V payload of one token summed over full (unwindowed) attention layers:
    /// `2 × kv_heads × head_dim × element bytes` plus scales. Recurrent layers add nothing.
    pub fn kv_token_bytes(&self) -> Result<usize> {
        self.token_bytes(false)
    }

    /// K/V payload of one token summed over windowed attention layers.
    pub fn window_token_bytes(&self) -> Result<usize> {
        self.token_bytes(true)
    }

    fn token_bytes(&self, windowed: bool) -> Result<usize> {
        self.layers
            .iter()
            .filter(|kind| kind.windowed() == windowed)
            .try_fold(0usize, |total, kind| {
                total
                    .checked_add(kind.attention_token_bytes(self.kv.format)?)
                    .ok_or_else(|| invalid("K/V token size overflow"))
            })
    }

    /// Bytes of one page across every full attention layer.
    pub fn page_bytes(&self) -> Result<usize> {
        self.kv_token_bytes()?
            .checked_mul(self.kv.page_tokens)
            .ok_or_else(|| invalid("page size overflow"))
    }

    /// Bytes of one window page across every windowed attention layer.
    pub fn window_page_bytes(&self) -> Result<usize> {
        self.window_token_bytes()?
            .checked_mul(self.kv.page_tokens)
            .ok_or_else(|| invalid("page size overflow"))
    }

    /// The window shared by every windowed attention layer, if any layer is windowed.
    pub fn window(&self) -> Option<usize> {
        self.layers.iter().find_map(|kind| match kind {
            LayerStateKind::Attention { window, .. } => *window,
            LayerStateKind::Recurrent { .. } => None,
        })
    }

    /// First logical page a windowed layer still needs at `position`: later queries and
    /// drafts read only keys from `position - window` on.
    pub fn first_window_page(&self, position: usize) -> usize {
        self.window().map_or(0, |window| {
            position.saturating_sub(window) / self.kv.page_tokens.max(1)
        })
    }

    /// Window pages a sequence at `position` holds.
    pub fn window_pages_for(&self, position: usize) -> usize {
        if self.window().is_some() {
            self.pages_for(position) - self.first_window_page(position)
        } else {
            0
        }
    }

    /// Convolution histories plus recurrent matrices of every recurrent layer at one boundary.
    pub fn recurrent_slot_bytes(&self) -> Result<usize> {
        self.layers.iter().try_fold(0usize, |total, kind| {
            total
                .checked_add(kind.recurrent_boundary_bytes()?)
                .ok_or_else(|| invalid("recurrent slot size overflow"))
        })
    }

    /// Whether any layer keeps full (unwindowed) paged attention state.
    pub fn has_full_attention(&self) -> bool {
        self.layers
            .iter()
            .any(|kind| matches!(kind, LayerStateKind::Attention { window: None, .. }))
    }

    /// Whether any layer keeps paged attention state.
    pub fn has_attention(&self) -> bool {
        self.layers
            .iter()
            .any(|kind| matches!(kind, LayerStateKind::Attention { .. }))
    }

    /// Whether any layer keeps recurrent state.
    pub fn has_recurrent(&self) -> bool {
        self.layers
            .iter()
            .any(|kind| matches!(kind, LayerStateKind::Recurrent { .. }))
    }

    /// Pages holding `positions` tokens; zero without attention layers.
    pub fn pages_for(&self, positions: usize) -> usize {
        if self.has_attention() {
            positions.div_ceil(self.kv.page_tokens.max(1))
        } else {
            0
        }
    }

    /// Resident state of one sequence at a token boundary: whole pages, the window pages
    /// still read, plus one recurrent slot, excluding allocator and grammar overhead.
    pub fn state_payload_bytes(&self, position: usize) -> Result<usize> {
        if position > self.max_position {
            return Err(invalid("state position exceeds model limit"));
        }
        self.full_pages_for(position)
            .checked_mul(self.page_bytes()?)
            .zip(
                self.window_pages_for(position)
                    .checked_mul(self.window_page_bytes()?),
            )
            .and_then(|(p, w)| p.checked_add(w))
            .and_then(|b| b.checked_add(self.recurrent_slot_bytes().ok()?))
            .ok_or_else(|| invalid("model payload overflow"))
    }

    /// Full-attention pages holding `positions` tokens; zero without such layers.
    pub fn full_pages_for(&self, positions: usize) -> usize {
        if self.has_full_attention() {
            self.pages_for(positions)
        } else {
            0
        }
    }

    /// Pool memory one pass reserves for `tokens` new tokens after `start`: the pages and
    /// window pages it opens plus `boundaries` recurrent slots (1 ordinarily, `tokens` for
    /// speculation). A copy-on-write tail page, when needed, adds one page.
    pub fn state_update_bytes(
        &self,
        start: usize,
        tokens: usize,
        boundaries: usize,
    ) -> Result<usize> {
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| invalid("position overflow"))?;
        let opened = self.pages_for(end) - self.pages_for(start);
        let pages = if self.has_full_attention() { opened } else { 0 };
        let windows = if self.window().is_some() { opened } else { 0 };
        let slots = if self.has_recurrent() { boundaries } else { 0 };
        pages
            .checked_mul(self.page_bytes()?)
            .zip(windows.checked_mul(self.window_page_bytes()?))
            .zip(slots.checked_mul(self.recurrent_slot_bytes()?))
            .and_then(|((p, w), s)| p.checked_add(w)?.checked_add(s))
            .ok_or_else(|| invalid("model update overflow"))
    }

    /// Validate every dimension, the K/V layout and byte sizes before allocating state.
    pub fn validate(&self) -> Result<()> {
        if self.identity.revision.is_empty()
            || self.identity.state_format == 0
            || self.layers.is_empty()
            || self.vocab_size == 0
            || self.max_position == 0
        {
            return Err(invalid("incomplete model description"));
        }
        if !self.modalities.contains(&Modality::Text) {
            return Err(invalid("packed execution requires text input support"));
        }
        if self.kv.version != KV_LAYOUT_VERSION {
            return Err(invalid(format!(
                "K/V layout version {} is not the supported {KV_LAYOUT_VERSION}",
                self.kv.version
            )));
        }
        checked_elements(&[self.kv.page_tokens])?;
        u32::try_from(self.kv.page_tokens).map_err(|_| invalid("page size exceeds u32"))?;
        self.vocab_size
            .checked_mul(4)
            .ok_or_else(|| invalid("logits byte size overflow"))?;
        let window = self.window();
        for kind in &self.layers {
            if let LayerStateKind::Attention {
                kv_heads,
                head_dim,
                window: layer_window,
            } = *kind
            {
                checked_elements(&[kv_heads, head_dim])?;
                self.kv.format.validate(head_dim)?;
                if layer_window.is_some_and(|w| w == 0 || Some(w) != window) {
                    return Err(invalid("windowed layers must share one non-zero window"));
                }
            }
            kind.attention_token_bytes(self.kv.format)?;
            kind.recurrent_boundary_bytes()?;
        }
        self.state_payload_bytes(self.max_position)?;
        Ok(())
    }
}

/// Request identity plus monotonic revision. Revisions are never reused on restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateHandle {
    pub request: Uuid,
    pub revision: u64,
}

/// Current state metadata supplied by the sole execution owner.
#[derive(Clone, Debug)]
pub struct StateMetadata {
    pub handle: StateHandle,
    pub position: usize,
    /// The request's grammar reached a terminal state; further outputs are not sampled.
    pub finished: bool,
}

/// One contiguous sequence in a packed input. Output positions index packed tokens.
///
/// A speculative sequence is `[next token, draft_1, ..., draft_m]` with an output at every
/// position. The engine samples the target at each position and accepts drafts while they
/// match, so acceptance can never disagree with the sampled tokens.
#[derive(Clone, Debug)]
pub struct SequenceDescriptor {
    pub state: StateHandle,
    pub tokens: Range<usize>,
    /// Strictly increasing packed positions whose logits are sampled.
    pub outputs: Vec<usize>,
    pub speculative: bool,
}

/// Untrusted execution input; validate against current request metadata before use.
#[derive(Clone, Debug)]
pub struct PackedExecution {
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub sequences: Vec<SequenceDescriptor>,
}

/// Immutable execution input with checked ranges, positions and unique state writes.
#[derive(Clone, Debug)]
pub struct ValidatedExecution {
    packed: PackedExecution,
    requirements: ExecutionRequirements,
}

/// Checked payload retained by one submission until completion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionRequirements {
    pub packed_input_bytes: usize,
    /// F32 logits for requested output positions only.
    pub logits_bytes: usize,
    /// Pool pages opened and recurrent slots written by this pass, including any
    /// copy-on-write tail page.
    pub state_update_bytes: usize,
    /// Model-declared kernel scratch for this plan.
    pub workspace_bytes: usize,
}

impl ExecutionRequirements {
    /// Sum of all retained payloads; use this for admission against a device budget.
    pub fn total_bytes(&self) -> Result<usize> {
        self.packed_input_bytes
            .checked_add(self.logits_bytes)
            .and_then(|t| t.checked_add(self.state_update_bytes))
            .and_then(|t| t.checked_add(self.workspace_bytes))
            .ok_or_else(|| invalid("execution requirement overflow"))
    }
}

impl ValidatedExecution {
    /// Return the validated packed input without allowing mutation.
    pub fn packed(&self) -> &PackedExecution {
        &self.packed
    }

    /// Payload bounds checked before execution, excluding model workspace.
    pub fn requirements(&self) -> &ExecutionRequirements {
        &self.requirements
    }

    /// Take ownership of the validated input for execution.
    pub fn into_parts(self) -> (PackedExecution, ExecutionRequirements) {
        (self.packed, self.requirements)
    }
}

impl PackedExecution {
    /// Reject stale handles, aliased writes, holes, invalid tokens and out-of-range outputs.
    pub fn validate(
        self,
        model: &ModelDescription,
        states: &[StateMetadata],
    ) -> Result<ValidatedExecution> {
        model.validate()?;
        if self.tokens.is_empty()
            || self.sequences.is_empty()
            || self.positions.len() != self.tokens.len()
        {
            return Err(invalid("empty or mismatched packed input"));
        }
        if self
            .tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |t| t >= model.vocab_size))
        {
            return Err(invalid("token outside vocabulary"));
        }
        let states: HashMap<_, _> = states.iter().map(|s| (s.handle, s)).collect();
        let mut requirements = ExecutionRequirements {
            packed_input_bytes: checked_elements(&[
                self.tokens.len(),
                size_of::<u32>() + size_of::<usize>(),
            ])?,
            ..Default::default()
        };
        let mut seen = HashSet::with_capacity(self.sequences.len());
        let mut outputs = 0usize;
        let mut cursor = 0;
        for sequence in &self.sequences {
            if sequence.tokens.start != cursor
                || sequence.tokens.end <= cursor
                || sequence.tokens.end > self.tokens.len()
            {
                return Err(invalid("token ranges must partition the packed input"));
            }
            if !seen.insert(sequence.state.request) {
                return Err(invalid("overlapping request state writes"));
            }
            let state = states
                .get(&sequence.state)
                .ok_or_else(|| invalid("unknown or stale state handle"))?;
            let length = sequence.tokens.len();
            let end = state
                .position
                .checked_add(length)
                .ok_or_else(|| invalid("position overflow"))?;
            if end > model.max_position {
                return Err(invalid("sequence exceeds positional limit"));
            }
            let positions = self
                .positions
                .get(sequence.tokens.clone())
                .ok_or_else(|| invalid("position range out of bounds"))?;
            for (offset, &position) in positions.iter().enumerate() {
                if state.position.checked_add(offset) != Some(position) {
                    return Err(invalid("noncontiguous sequence positions"));
                }
            }
            if !sequence.outputs.windows(2).all(|w| w[0] < w[1])
                || sequence
                    .outputs
                    .iter()
                    .any(|p| !sequence.tokens.contains(p))
            {
                return Err(invalid(
                    "outputs must be strictly increasing within the range",
                ));
            }
            if sequence.speculative && sequence.outputs.len() != length {
                return Err(invalid(
                    "speculative sequences need an output at every token",
                ));
            }
            let boundaries = if sequence.speculative { length } else { 1 };
            requirements.state_update_bytes = requirements
                .state_update_bytes
                .checked_add(model.state_update_bytes(state.position, length, boundaries)?)
                .ok_or_else(|| invalid("batch state update overflow"))?;
            outputs = outputs
                .checked_add(sequence.outputs.len())
                .ok_or_else(|| invalid("output count overflow"))?;
            cursor = sequence.tokens.end;
        }
        if cursor != self.tokens.len() {
            return Err(invalid("unassigned packed tokens"));
        }
        requirements.logits_bytes = outputs
            .checked_mul(model.vocab_size)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("logits byte size overflow"))?;
        Ok(ValidatedExecution {
            packed: self,
            requirements,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn attention(kv_heads: usize, head_dim: usize) -> LayerStateKind {
        LayerStateKind::Attention {
            kv_heads,
            head_dim,
            window: None,
        }
    }

    fn recurrent() -> LayerStateKind {
        LayerStateKind::Recurrent {
            conv_channels: 1,
            conv_width: 2,
            heads: 1,
            key_dim: 1,
            value_dim: 1,
            dtype: DType::F32,
        }
    }

    fn description(layers: Vec<LayerStateKind>, format: KvFormat) -> ModelDescription {
        ModelDescription {
            identity: ModelIdentity {
                revision: "fixture".into(),
                state_format: 1,
            },
            layers,
            kv: KvLayout::new(2, format),
            modalities: vec![Modality::Text],
            vocab_size: 4,
            max_position: 8,
        }
    }

    #[test]
    fn test_checked_state_dimensions() {
        assert!(checked_elements(&[usize::MAX, 2]).is_err());
        assert!(checked_elements(&[1, 0]).is_err());
        assert_eq!(checked_elements(&[2, 3]).ok(), Some(6));
    }

    #[test]
    fn test_execution_payload_requirements_and_aggregate_overflow() -> Result<()> {
        let mut model = description(vec![attention(1, 1)], KvFormat::F32);
        let state = StateHandle {
            request: Uuid::new_v4(),
            revision: 0,
        };
        let plan = PackedExecution {
            tokens: vec![1, 2, 3],
            positions: vec![1, 2, 3],
            sequences: vec![SequenceDescriptor {
                state,
                tokens: 0..3,
                outputs: vec![1],
                speculative: false,
            }],
        };
        let validated = plan.validate(
            &model,
            &[StateMetadata {
                handle: state,
                position: 1,
                finished: false,
            }],
        )?;
        let requirements = validated.requirements();
        assert_eq!(
            requirements.packed_input_bytes,
            3 * (4 + size_of::<usize>())
        );
        assert_eq!(requirements.logits_bytes, 16);
        // Positions 1..4 already own page 0 and open page 1: one 2-token page of K and V.
        assert_eq!(requirements.state_update_bytes, 16);
        assert_eq!(model.state_payload_bytes(0)?, 0);
        assert_eq!(model.state_payload_bytes(3)?, 32);
        model.max_position = 1;
        model.layers = vec![attention(usize::MAX / 16, 1); 3];
        assert!(model.validate().is_err());
        Ok(())
    }

    #[test]
    fn test_kv_formats_count_codes_and_scales_per_attention_layer() -> Result<()> {
        // Qwen3.5-0.8B: six attention layers with 2 K/V heads of 256 dims among 24.
        let mut layers = vec![recurrent(); 18];
        layers.extend(vec![attention(2, 256); 6]);
        let bytes = |format| description(layers.clone(), format).kv_token_bytes();
        assert_eq!(bytes(KvFormat::F32)?, 6 * 2 * 2 * 256 * 4);
        assert_eq!(bytes(KvFormat::F16)?, 6 * 2 * 2 * 256 * 2);
        // 512 one-byte codes plus 2 heads × 8 groups of F16 scales, for K and V.
        assert_eq!(bytes(KvFormat::Q8)?, 6 * 2 * (512 + 32));
        assert_eq!(bytes(KvFormat::Q8 { group: 128 })?, 6 * 2 * (512 + 8));
        let model = description(layers, KvFormat::Q8);
        assert_eq!(model.page_bytes()?, 2 * 6 * 2 * 544);
        assert_eq!(model.recurrent_slot_bytes()?, 18 * 12);
        assert_eq!(model.pages_for(5), 3);
        Ok(())
    }

    #[test]
    fn test_recurrent_dtype_sizes_the_matrix_and_keeps_the_history_f32() -> Result<()> {
        let layer = |dtype| LayerStateKind::Recurrent {
            conv_channels: 6,
            conv_width: 3,
            heads: 2,
            key_dim: 4,
            value_dim: 8,
            dtype,
        };
        // History: 3 × 6 F32 values; matrix: 2 × 4 × 8 values of the declared dtype.
        assert_eq!(layer(DType::F32).recurrent_boundary_bytes()?, 72 + 256);
        for dtype in [DType::F16, DType::BF16] {
            assert_eq!(layer(dtype).recurrent_boundary_bytes()?, 72 + 128);
            assert!(
                description(vec![layer(dtype)], KvFormat::F32)
                    .validate()
                    .is_ok()
            );
        }
        Ok(())
    }

    #[test]
    fn test_kv_layout_groups_pages_versions_and_modalities_are_validated() {
        assert!(
            description(vec![attention(1, 64)], KvFormat::Q8)
                .validate()
                .is_ok()
        );
        for group in [0, 16, 48, 128] {
            assert!(
                description(vec![attention(1, 64)], KvFormat::Q8 { group })
                    .validate()
                    .is_err(),
                "group {group}"
            );
        }
        let mut model = description(vec![attention(1, 64)], KvFormat::F16);
        model.kv.page_tokens = 0;
        assert!(model.validate().is_err());
        let mut model = description(vec![attention(1, 64)], KvFormat::F16);
        model.kv.version = KV_LAYOUT_VERSION + 1;
        assert!(model.validate().is_err());
        let mut model = description(vec![attention(1, 1)], KvFormat::F32);
        model.modalities = vec![Modality::Image];
        assert!(model.validate().is_err());
        let mut model = description(vec![recurrent()], KvFormat::F32);
        model.layers = vec![LayerStateKind::Recurrent {
            conv_channels: 1,
            conv_width: 2,
            heads: 1,
            key_dim: 1,
            value_dim: 1,
            dtype: DType::U8,
        }];
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_kv_format_names_round_trip() -> Result<()> {
        for (format, name) in [
            (KvFormat::F32, "f32"),
            (KvFormat::F16, "f16"),
            (KvFormat::Q8, "q8_32"),
            (KvFormat::Q8 { group: 64 }, "q8_64"),
        ] {
            assert_eq!(format.to_string(), name);
            assert_eq!(name.parse::<KvFormat>().map_err(invalid)?, format);
        }
        assert_eq!("q8".parse::<KvFormat>(), Ok(KvFormat::Q8 { group: 32 }));
        assert!("q4".parse::<KvFormat>().is_err());
        assert!("q8_x".parse::<KvFormat>().is_err());
        Ok(())
    }

    #[test]
    fn test_speculative_recurrent_boundaries_are_charged_per_token() -> Result<()> {
        let model = description(vec![recurrent()], KvFormat::F32);
        assert_eq!(model.state_update_bytes(0, 4, 1)?, 12);
        assert_eq!(model.state_update_bytes(0, 4, 4)?, 48);
        assert_eq!(model.pages_for(4), 0);
        Ok(())
    }
}
