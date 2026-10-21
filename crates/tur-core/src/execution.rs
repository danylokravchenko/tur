//! Host-side contracts checked before device submission.

use std::{collections::HashSet, ops::Range};
use uuid::Uuid;

use crate::{Result, TurError};

/// Exact checkpoint/configuration/adapter/position-policy identity and state format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelIdentity {
    pub revision: String,
    pub state_format: u32,
}

/// State required by one model layer. P2 state storage uses F32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerStateKind {
    Attention {
        kv_heads: usize,
        head_dim: usize,
        page_tokens: usize,
    },
    Recurrent {
        conv_channels: usize,
        conv_width: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    },
}

/// Model-declared layout; this describes state, not model-specific configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDescription {
    pub identity: ModelIdentity,
    pub layers: Vec<LayerStateKind>,
    pub vocab_size: usize,
    pub max_position: usize,
}

/// Reuse the existing error hierarchy for invalid execution/state transitions.
pub fn invalid(message: impl Into<String>) -> TurError {
    TurError::InvalidPhase(message.into())
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

impl ModelDescription {
    /// State tensor payload at a token boundary, excluding allocator/grammar overhead.
    pub fn state_payload_bytes(&self, position: usize) -> Result<usize> {
        if position > self.max_position {
            return Err(invalid("state position exceeds model limit"));
        }
        self.layers.iter().try_fold(0usize, |total, kind| {
            let bytes = match *kind {
                LayerStateKind::Attention {
                    kv_heads, head_dim, ..
                } => checked_elements(&[kv_heads, head_dim, 8])?
                    .checked_mul(position)
                    .ok_or_else(|| invalid("attention payload overflow"))?,
                LayerStateKind::Recurrent {
                    conv_channels,
                    conv_width,
                    heads,
                    key_dim,
                    value_dim,
                } => checked_elements(&[conv_channels, conv_width, 4])?
                    .checked_add(checked_elements(&[heads, key_dim, value_dim, 4])?)
                    .ok_or_else(|| invalid("recurrent payload overflow"))?,
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| invalid("model payload overflow"))
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
        self.vocab_size
            .checked_mul(4)
            .ok_or_else(|| invalid("logits byte size overflow"))?;
        for kind in &self.layers {
            match *kind {
                LayerStateKind::Attention {
                    kv_heads,
                    head_dim,
                    page_tokens,
                } => {
                    checked_elements(&[page_tokens, kv_heads, head_dim, 4])?;
                    checked_elements(&[self.max_position, kv_heads, head_dim, 8])?;
                }
                LayerStateKind::Recurrent {
                    conv_channels,
                    conv_width,
                    heads,
                    key_dim,
                    value_dim,
                } => {
                    checked_elements(&[conv_width, conv_channels, 4])?;
                    checked_elements(&[heads, key_dim, value_dim, 4])?;
                }
            }
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
}

/// One contiguous sequence in a packed input. Output positions index packed tokens.
#[derive(Clone, Debug)]
pub struct SequenceDescriptor {
    pub state: StateHandle,
    pub tokens: Range<usize>,
    pub outputs: Vec<usize>,
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

/// Checked tensor payload requirements; excludes backend-specific temporary allocations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionRequirements {
    pub packed_input_bytes: usize,
    /// All token-step logits are retained until completion, even when not selected as outputs.
    pub retained_logits_bytes: usize,
    /// Conservative bound for one state snapshot, detached input and update per token.
    pub reference_state_bytes: usize,
}

impl ValidatedExecution {
    /// Return the validated packed input without allowing mutation.
    pub fn packed(&self) -> &PackedExecution {
        &self.packed
    }

    /// Payload bounds checked before execution; these are not an admission budget.
    pub fn requirements(&self) -> &ExecutionRequirements {
        &self.requirements
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
        let mut requirements = ExecutionRequirements {
            packed_input_bytes: checked_elements(&[
                self.tokens.len(),
                std::mem::size_of::<u32>() + std::mem::size_of::<usize>(),
            ])?,
            retained_logits_bytes: checked_elements(&[self.tokens.len(), model.vocab_size, 4])?,
            reference_state_bytes: 0,
        };
        if self
            .tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |t| t >= model.vocab_size))
        {
            return Err(invalid("token outside vocabulary"));
        }
        let mut seen = HashSet::new();
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
                .iter()
                .find(|s| s.handle == sequence.state)
                .ok_or_else(|| invalid("unknown or stale state handle"))?;
            let end = state
                .position
                .checked_add(sequence.tokens.len())
                .ok_or_else(|| invalid("position overflow"))?;
            if end > model.max_position {
                return Err(invalid("sequence exceeds positional limit"));
            }
            let payload = model
                .state_payload_bytes(end)?
                .checked_mul(sequence.tokens.len())
                .and_then(|bytes| bytes.checked_mul(3))
                .ok_or_else(|| invalid("reference state payload overflow"))?;
            requirements.reference_state_bytes = requirements
                .reference_state_bytes
                .checked_add(payload)
                .ok_or_else(|| invalid("batch state payload overflow"))?;
            let positions = self
                .positions
                .get(sequence.tokens.clone())
                .ok_or_else(|| invalid("position range out of bounds"))?;
            for (offset, &position) in positions.iter().enumerate() {
                if state.position.checked_add(offset) != Some(position) {
                    return Err(invalid("noncontiguous sequence positions"));
                }
            }
            let mut outputs = HashSet::new();
            for &position in &sequence.outputs {
                if !sequence.tokens.contains(&position) || !outputs.insert(position) {
                    return Err(invalid("invalid or duplicate output position"));
                }
            }
            checked_elements(&[sequence.outputs.len().max(1), model.vocab_size, 4])?;
            cursor = sequence.tokens.end;
        }
        if cursor != self.tokens.len() {
            return Err(invalid("unassigned packed tokens"));
        }
        Ok(ValidatedExecution {
            packed: self,
            requirements,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_state_dimensions() {
        assert!(checked_elements(&[usize::MAX, 2]).is_err());
        assert!(checked_elements(&[1, 0]).is_err());
        assert_eq!(checked_elements(&[2, 3]).ok(), Some(6));
    }

    #[test]
    fn test_execution_payload_requirements_and_aggregate_overflow() -> Result<()> {
        let mut model = ModelDescription {
            identity: ModelIdentity {
                revision: "fixture".into(),
                state_format: 1,
            },
            layers: vec![LayerStateKind::Attention {
                kv_heads: 1,
                head_dim: 1,
                page_tokens: 2,
            }],
            vocab_size: 4,
            max_position: 8,
        };
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
            }],
        };
        let validated = plan.validate(
            &model,
            &[StateMetadata {
                handle: state,
                position: 0,
            }],
        )?;
        assert_eq!(
            validated.requirements().packed_input_bytes,
            2 * (4 + std::mem::size_of::<usize>())
        );
        assert_eq!(validated.requirements().retained_logits_bytes, 32);
        assert_eq!(validated.requirements().reference_state_bytes, 96);
        assert_eq!(model.state_payload_bytes(0)?, 0);
        model.max_position = 1;
        let kind = LayerStateKind::Attention {
            kv_heads: usize::MAX / 16,
            head_dim: 1,
            page_tokens: 1,
        };
        model.layers = vec![kind; 3];
        assert!(model.validate().is_err());
        Ok(())
    }
}
