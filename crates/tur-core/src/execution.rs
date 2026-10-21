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

/// State required by one model layer, including its storage dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerStateKind {
    Attention {
        kv_heads: usize,
        head_dim: usize,
        page_tokens: usize,
        dtype: DType,
    },
    Recurrent {
        conv_channels: usize,
        conv_width: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        dtype: DType,
    },
}

/// Model-declared layout; this describes state, not model-specific configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDescription {
    pub identity: ModelIdentity,
    pub layers: Vec<LayerStateKind>,
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
    /// K plus V bytes for one token; zero for recurrent layers.
    pub fn attention_token_bytes(&self) -> Result<usize> {
        match *self {
            Self::Attention {
                kv_heads,
                head_dim,
                dtype,
                ..
            } => checked_elements(&[kv_heads, head_dim, 2, state_dtype_bytes(dtype)?]),
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
                checked_elements(&[conv_channels, conv_width, bytes])?
                    .checked_add(checked_elements(&[heads, key_dim, value_dim, bytes])?)
                    .ok_or_else(|| invalid("recurrent payload overflow"))
            }
        }
    }
}

impl ModelDescription {
    /// State tensor payload at a token boundary, excluding allocator/grammar overhead.
    pub fn state_payload_bytes(&self, position: usize) -> Result<usize> {
        if position > self.max_position {
            return Err(invalid("state position exceeds model limit"));
        }
        self.layers.iter().try_fold(0usize, |total, kind| {
            let recurrent = kind.recurrent_boundary_bytes()?;
            let bytes = kind
                .attention_token_bytes()?
                .checked_mul(position)
                .and_then(|b| b.checked_add(recurrent))
                .ok_or_else(|| invalid("layer payload overflow"))?;
            total
                .checked_add(bytes)
                .ok_or_else(|| invalid("model payload overflow"))
        })
    }

    /// Model outputs retained until completion for `tokens` new tokens and `boundaries`
    /// recurrent snapshots (1 for ordinary execution, `tokens` for speculation).
    pub fn state_update_bytes(&self, tokens: usize, boundaries: usize) -> Result<usize> {
        self.layers.iter().try_fold(0usize, |total, kind| {
            let attention = kind
                .attention_token_bytes()?
                .checked_mul(tokens)
                .ok_or_else(|| invalid("attention update overflow"))?;
            let recurrent = kind
                .recurrent_boundary_bytes()?
                .checked_mul(boundaries)
                .ok_or_else(|| invalid("recurrent update overflow"))?;
            total
                .checked_add(attention)
                .and_then(|t| t.checked_add(recurrent))
                .ok_or_else(|| invalid("model update overflow"))
        })
    }

    /// Validate every dimension, including byte sizes, before allocating state.
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
        self.vocab_size
            .checked_mul(4)
            .ok_or_else(|| invalid("logits byte size overflow"))?;
        for kind in &self.layers {
            if let LayerStateKind::Attention { page_tokens, .. } = *kind {
                checked_elements(&[page_tokens])?;
            }
            kind.attention_token_bytes()?;
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
    /// New K/V for every token plus recurrent state at each retained boundary.
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
                std::mem::size_of::<u32>() + std::mem::size_of::<usize>(),
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
                .checked_add(model.state_update_bytes(length, boundaries)?)
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
mod tests {
    use super::*;

    fn attention(kv_heads: usize, dtype: DType) -> LayerStateKind {
        LayerStateKind::Attention {
            kv_heads,
            head_dim: 1,
            page_tokens: 2,
            dtype,
        }
    }

    fn description(layers: Vec<LayerStateKind>) -> ModelDescription {
        ModelDescription {
            identity: ModelIdentity {
                revision: "fixture".into(),
                state_format: 1,
            },
            layers,
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
        let mut model = description(vec![attention(1, DType::F32)]);
        let state = StateHandle {
            request: Uuid::new_v4(),
            revision: 0,
        };
        let plan = PackedExecution {
            tokens: vec![1, 2],
            positions: vec![0, 1],
            sequences: vec![SequenceDescriptor {
                state,
                tokens: 0..2,
                outputs: vec![1],
                speculative: false,
            }],
        };
        let validated = plan.validate(
            &model,
            &[StateMetadata {
                handle: state,
                position: 0,
                finished: false,
            }],
        )?;
        let requirements = validated.requirements();
        assert_eq!(
            requirements.packed_input_bytes,
            2 * (4 + std::mem::size_of::<usize>())
        );
        assert_eq!(requirements.logits_bytes, 16);
        assert_eq!(requirements.state_update_bytes, 16);
        assert_eq!(model.state_payload_bytes(0)?, 0);
        model.max_position = 1;
        model.layers = vec![attention(usize::MAX / 16, DType::F32); 3];
        assert!(model.validate().is_err());
        Ok(())
    }

    #[test]
    fn test_state_dtype_and_modalities_are_validated() {
        assert_eq!(
            description(vec![attention(1, DType::BF16)])
                .state_payload_bytes(2)
                .ok(),
            Some(8)
        );
        assert!(
            description(vec![attention(1, DType::U8)])
                .validate()
                .is_err()
        );
        let mut model = description(vec![attention(1, DType::F32)]);
        model.modalities = vec![Modality::Image];
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_speculative_recurrent_boundaries_are_charged_per_token() -> Result<()> {
        let model = description(vec![LayerStateKind::Recurrent {
            conv_channels: 1,
            conv_width: 2,
            heads: 1,
            key_dim: 1,
            value_dim: 1,
            dtype: DType::F32,
        }]);
        assert_eq!(model.state_update_bytes(4, 1)?, 12);
        assert_eq!(model.state_update_bytes(4, 4)?, 48);
        Ok(())
    }
}
