#![forbid(unsafe_code)]
//! Immutable page and recurrent snapshots with staged, boundary-selected updates.

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{LayerStateKind, ModelDescription, invalid},
};

/// Committed layer state handed to a model. Tensors share storage with live state and
/// checkpoints, so models must treat them as read-only (no `slice_set`/`zero_set`).
pub enum LayerRead {
    /// Pages in order; each tensor is `[page tokens, kv_heads, head_dim]`.
    Attention { pages: Vec<(Tensor, Tensor)> },
    /// `[conv_width, conv_channels]` history and `[heads, key_dim, value_dim]` matrix.
    Recurrent {
        convolution: Tensor,
        recurrent: Tensor,
    },
}

/// Recurrent states a model must return for one sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundaries {
    /// Only the state after the last token.
    Final,
    /// The state after every token, so any accepted prefix can be published.
    EveryToken,
}

/// Complete update for one layer over a packed sequence of `tokens` new tokens.
pub enum LayerUpdate {
    /// `[tokens, kv_heads, head_dim]` keys and values.
    Attention { key: Tensor, value: Tensor },
    /// `[boundaries, conv_width, conv_channels]` and `[boundaries, heads, key_dim, value_dim]`,
    /// where boundaries is 1 for [`Boundaries::Final`] and `tokens` for [`Boundaries::EveryToken`].
    Recurrent {
        convolution: Tensor,
        recurrent: Tensor,
    },
}

/// Validated, detached model outputs awaiting acceptance.
pub struct StagedUpdate {
    tokens: usize,
    boundaries: Boundaries,
    layers: Vec<LayerUpdate>,
}

impl StagedUpdate {
    /// Number of new tokens covered by this update.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

#[derive(Clone)]
struct Page {
    key: Tensor,
    value: Tensor,
    tokens: usize,
}

#[derive(Clone)]
enum LayerState {
    Attention(Vec<Arc<Page>>),
    Recurrent {
        convolution: Tensor,
        recurrent: Tensor,
    },
}

/// An immutable, whole-model state at exactly one token boundary.
#[derive(Clone)]
pub struct ModelState {
    description: Arc<ModelDescription>,
    device: Device,
    position: usize,
    layers: Vec<LayerState>,
}

fn check(tensor: &Tensor, shape: &[usize], dtype: DType, device: &Device) -> Result<()> {
    if tensor.dims() != shape || tensor.dtype() != dtype || !tensor.device().same_device(device) {
        return Err(invalid("state tensor has incorrect shape, dtype or device"));
    }
    Ok(())
}

/// Append `[n, kv_heads, head_dim]` K/V, filling a partial tail page before opening new pages.
fn append_pages(
    pages: &mut Vec<Arc<Page>>,
    key: &Tensor,
    value: &Tensor,
    page_tokens: usize,
) -> Result<()> {
    let total = key.dim(0)?;
    let mut offset = 0;
    if let Some(tail) = pages.last().filter(|p| p.tokens < page_tokens) {
        let take = (page_tokens - tail.tokens).min(total);
        let page = Page {
            key: Tensor::cat(&[&tail.key, &key.narrow(0, 0, take)?], 0)?,
            value: Tensor::cat(&[&tail.value, &value.narrow(0, 0, take)?], 0)?,
            tokens: tail.tokens + take,
        };
        pages.pop();
        pages.push(Arc::new(page));
        offset = take;
    }
    while offset < total {
        let take = page_tokens.min(total - offset);
        pages.push(Arc::new(Page {
            key: key.narrow(0, offset, take)?,
            value: value.narrow(0, offset, take)?,
            tokens: take,
        }));
        offset += take;
    }
    Ok(())
}

impl ModelState {
    /// Allocate zero recurrent state and empty attention page tables after layout validation.
    pub fn allocate(description: Arc<ModelDescription>, device: &Device) -> Result<Self> {
        description.validate()?;
        let mut layers = Vec::with_capacity(description.layers.len());
        for kind in &description.layers {
            layers.push(match *kind {
                LayerStateKind::Attention { .. } => LayerState::Attention(Vec::new()),
                LayerStateKind::Recurrent {
                    conv_channels,
                    conv_width,
                    heads,
                    key_dim,
                    value_dim,
                    dtype,
                } => LayerState::Recurrent {
                    convolution: Tensor::zeros((conv_width, conv_channels), dtype, device)?,
                    recurrent: Tensor::zeros((heads, key_dim, value_dim), dtype, device)?,
                },
            });
        }
        Ok(Self {
            description,
            device: device.clone(),
            position: 0,
            layers,
        })
    }

    /// Device owning the immutable state buffers.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Exact token boundary of every layer in this snapshot.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Complete identity and layout used to validate restore.
    pub fn description(&self) -> &ModelDescription {
        &self.description
    }

    /// Zero-copy model inputs sharing committed storage.
    pub fn bind_layers(&self) -> Vec<LayerRead> {
        self.layers
            .iter()
            .map(|layer| match layer {
                LayerState::Attention(pages) => LayerRead::Attention {
                    pages: pages
                        .iter()
                        .map(|p| (p.key.clone(), p.value.clone()))
                        .collect(),
                },
                LayerState::Recurrent {
                    convolution,
                    recurrent,
                } => LayerRead::Recurrent {
                    convolution: convolution.clone(),
                    recurrent: recurrent.clone(),
                },
            })
            .collect()
    }

    /// Detached copies for inspection; mutating them cannot change any snapshot.
    pub fn read_layers(&self) -> Result<Vec<LayerRead>> {
        self.bind_layers()
            .into_iter()
            .map(|layer| {
                Ok(match layer {
                    LayerRead::Attention { pages } => LayerRead::Attention {
                        pages: pages
                            .iter()
                            .map(|(k, v)| Ok((k.copy()?, v.copy()?)))
                            .collect::<Result<_>>()?,
                    },
                    LayerRead::Recurrent {
                        convolution,
                        recurrent,
                    } => LayerRead::Recurrent {
                        convolution: convolution.copy()?,
                        recurrent: recurrent.copy()?,
                    },
                })
            })
            .collect()
    }

    /// Validate every layer's update for `tokens` new tokens, then detach it from model storage.
    pub fn stage(
        &self,
        updates: Vec<LayerUpdate>,
        tokens: usize,
        boundaries: Boundaries,
    ) -> Result<StagedUpdate> {
        let end = self
            .position
            .checked_add(tokens)
            .ok_or_else(|| invalid("position overflow"))?;
        if tokens == 0 || end > self.description.max_position || updates.len() != self.layers.len()
        {
            return Err(invalid("invalid layer update count or position"));
        }
        let snapshots = match boundaries {
            Boundaries::Final => 1,
            Boundaries::EveryToken => tokens,
        };
        for (kind, update) in self.description.layers.iter().zip(&updates) {
            match (kind, update) {
                (
                    LayerStateKind::Attention {
                        kv_heads,
                        head_dim,
                        dtype,
                        ..
                    },
                    LayerUpdate::Attention { key, value },
                ) => {
                    check(key, &[tokens, *kv_heads, *head_dim], *dtype, &self.device)?;
                    check(value, &[tokens, *kv_heads, *head_dim], *dtype, &self.device)?;
                }
                (
                    LayerStateKind::Recurrent {
                        conv_channels,
                        conv_width,
                        heads,
                        key_dim,
                        value_dim,
                        dtype,
                    },
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    },
                ) => {
                    check(
                        convolution,
                        &[snapshots, *conv_width, *conv_channels],
                        *dtype,
                        &self.device,
                    )?;
                    check(
                        recurrent,
                        &[snapshots, *heads, *key_dim, *value_dim],
                        *dtype,
                        &self.device,
                    )?;
                }
                _ => return Err(invalid("layer update kind does not match model")),
            }
        }
        let layers = updates
            .iter()
            .map(|update| {
                Ok(match update {
                    LayerUpdate::Attention { key, value } => LayerUpdate::Attention {
                        key: key.copy()?,
                        value: value.copy()?,
                    },
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    } => LayerUpdate::Recurrent {
                        convolution: convolution.copy()?,
                        recurrent: recurrent.copy()?,
                    },
                })
            })
            .collect::<Result<_>>()?;
        Ok(StagedUpdate {
            tokens,
            boundaries,
            layers,
        })
    }

    /// Construct the snapshot after the first `accepted` staged tokens; this one is unchanged.
    /// Rejected suffixes are never truncated out of recurrent state: the exact saved
    /// boundary is selected instead, which requires [`Boundaries::EveryToken`].
    pub fn apply(&self, staged: &StagedUpdate, accepted: usize) -> Result<Self> {
        if accepted == 0 {
            return Ok(self.clone());
        }
        let partial = accepted < staged.tokens;
        if accepted > staged.tokens || (partial && staged.boundaries == Boundaries::Final) {
            return Err(invalid("accepted prefix has no saved boundary"));
        }
        let position = self
            .position
            .checked_add(accepted)
            .ok_or_else(|| invalid("position overflow"))?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for ((kind, old), update) in self
            .description
            .layers
            .iter()
            .zip(&self.layers)
            .zip(&staged.layers)
        {
            layers.push(match (kind, old, update) {
                (
                    LayerStateKind::Attention { page_tokens, .. },
                    LayerState::Attention(pages),
                    LayerUpdate::Attention { key, value },
                ) => {
                    let mut pages = pages.clone();
                    // Copy a partial accept so pages do not pin rejected draft storage.
                    let (key, value) = if partial {
                        (
                            key.narrow(0, 0, accepted)?.copy()?,
                            value.narrow(0, 0, accepted)?.copy()?,
                        )
                    } else {
                        (key.clone(), value.clone())
                    };
                    append_pages(&mut pages, &key, &value, *page_tokens)?;
                    LayerState::Attention(pages)
                }
                (
                    LayerStateKind::Recurrent { .. },
                    LayerState::Recurrent { .. },
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    },
                ) => match staged.boundaries {
                    Boundaries::Final => LayerState::Recurrent {
                        convolution: convolution.get(0)?,
                        recurrent: recurrent.get(0)?,
                    },
                    Boundaries::EveryToken => LayerState::Recurrent {
                        convolution: convolution.get(accepted - 1)?.copy()?,
                        recurrent: recurrent.get(accepted - 1)?.copy()?,
                    },
                },
                _ => return Err(invalid("inconsistent internal layer state")),
            });
        }
        Ok(Self {
            description: Arc::clone(&self.description),
            device: self.device.clone(),
            position,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tur_core::execution::{Modality, ModelIdentity};

    fn description(page_tokens: usize) -> Arc<ModelDescription> {
        Arc::new(ModelDescription {
            identity: ModelIdentity {
                revision: "fixture".into(),
                state_format: 1,
            },
            layers: vec![
                LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 1,
                    page_tokens,
                    dtype: DType::F32,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 1,
                    conv_width: 1,
                    heads: 1,
                    key_dim: 1,
                    value_dim: 1,
                    dtype: DType::F32,
                },
            ],
            modalities: vec![Modality::Text],
            vocab_size: 8,
            max_position: 8,
        })
    }

    fn flatten(state: &ModelState) -> Result<Vec<Vec<f32>>> {
        let mut observed = Vec::new();
        for layer in state.read_layers()? {
            match layer {
                LayerRead::Attention { pages } => {
                    for (key, value) in pages {
                        observed.push(key.flatten_all()?.to_vec1::<f32>()?);
                        observed.push(value.flatten_all()?.to_vec1::<f32>()?);
                    }
                }
                LayerRead::Recurrent {
                    convolution,
                    recurrent,
                } => {
                    observed.push(convolution.flatten_all()?.to_vec1::<f32>()?);
                    observed.push(recurrent.flatten_all()?.to_vec1::<f32>()?);
                }
            }
        }
        Ok(observed)
    }

    fn updates(values: &[f32], boundaries: &[f32]) -> Result<Vec<LayerUpdate>> {
        let n = values.len();
        let b = boundaries.len();
        Ok(vec![
            LayerUpdate::Attention {
                key: Tensor::from_slice(values, (n, 1, 1), &Device::Cpu)?,
                value: Tensor::from_slice(values, (n, 1, 1), &Device::Cpu)?,
            },
            LayerUpdate::Recurrent {
                convolution: Tensor::from_slice(boundaries, (b, 1, 1), &Device::Cpu)?,
                recurrent: Tensor::from_slice(boundaries, (b, 1, 1, 1), &Device::Cpu)?,
            },
        ])
    }

    #[test]
    fn test_allocation_rejects_invalid_layout() {
        assert!(ModelState::allocate(description(0), &Device::Cpu).is_err());
    }

    #[test]
    fn test_published_state_does_not_alias_model_output_tensors() -> Result<()> {
        let initial = ModelState::allocate(description(2), &Device::Cpu)?;
        let outputs = updates(&[3.0], &[5.0])?;
        let staged = initial.stage(
            outputs
                .iter()
                .map(|u| match u {
                    LayerUpdate::Attention { key, value } => LayerUpdate::Attention {
                        key: key.clone(),
                        value: value.clone(),
                    },
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    } => LayerUpdate::Recurrent {
                        convolution: convolution.clone(),
                        recurrent: recurrent.clone(),
                    },
                })
                .collect(),
            1,
            Boundaries::Final,
        )?;
        for update in &outputs {
            match update {
                LayerUpdate::Attention { key, value } => {
                    key.zero_set()?;
                    value.zero_set()?;
                }
                LayerUpdate::Recurrent {
                    convolution,
                    recurrent,
                } => {
                    convolution.zero_set()?;
                    recurrent.zero_set()?;
                }
            }
        }
        let next = initial.apply(&staged, 1)?;
        assert_eq!(
            flatten(&next)?,
            vec![vec![3.0], vec![3.0], vec![5.0], vec![5.0]]
        );
        assert_eq!(initial.position(), 0);
        assert_eq!(next.position(), 1);
        Ok(())
    }

    #[test]
    fn test_packed_append_fills_tail_then_opens_pages() -> Result<()> {
        let initial = ModelState::allocate(description(2), &Device::Cpu)?;
        let one = initial.apply(
            &initial.stage(updates(&[1.0], &[1.0])?, 1, Boundaries::Final)?,
            1,
        )?;
        let staged = one.stage(
            updates(&[2.0, 3.0, 4.0, 5.0], &[9.0])?,
            4,
            Boundaries::Final,
        )?;
        let next = one.apply(&staged, 4)?;
        assert_eq!(next.position(), 5);
        let observed = flatten(&next)?;
        assert_eq!(observed[0], vec![1.0, 2.0]);
        assert_eq!(observed[2], vec![3.0, 4.0]);
        assert_eq!(observed[4], vec![5.0]);
        assert_eq!(observed[6], vec![9.0]);
        Ok(())
    }

    #[test]
    fn test_partial_accept_selects_exact_recurrent_boundary() -> Result<()> {
        let initial = ModelState::allocate(description(2), &Device::Cpu)?;
        let every = initial.stage(
            updates(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])?,
            3,
            Boundaries::EveryToken,
        )?;
        let two = initial.apply(&every, 2)?;
        assert_eq!(
            flatten(&two)?,
            vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![20.0], vec![20.0]]
        );
        assert_eq!(initial.apply(&every, 0)?.position(), 0);
        assert!(initial.apply(&every, 4).is_err());
        let last = initial.stage(updates(&[1.0, 2.0], &[7.0])?, 2, Boundaries::Final)?;
        assert!(initial.apply(&last, 1).is_err());
        Ok(())
    }

    #[test]
    fn test_stage_rejects_wrong_boundary_count_and_dtype() -> Result<()> {
        let initial = ModelState::allocate(description(2), &Device::Cpu)?;
        assert!(
            initial
                .stage(updates(&[1.0, 2.0], &[1.0])?, 2, Boundaries::EveryToken)
                .is_err()
        );
        let wrong_dtype = vec![
            LayerUpdate::Attention {
                key: Tensor::zeros((1, 1, 1), DType::F16, &Device::Cpu)?,
                value: Tensor::zeros((1, 1, 1), DType::F16, &Device::Cpu)?,
            },
            LayerUpdate::Recurrent {
                convolution: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
                recurrent: Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?,
            },
        ];
        assert!(initial.stage(wrong_dtype, 1, Boundaries::Final).is_err());
        Ok(())
    }
}
