#![forbid(unsafe_code)]
//! Immutable page and recurrent snapshots. Tensor boundaries copy to prevent alias writes.

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{LayerStateKind, ModelDescription, invalid},
};

/// Detached model inputs. Mutating these tensors cannot mutate a checkpoint.
pub enum LayerRead {
    Attention {
        pages: Vec<(Tensor, Tensor)>,
    },
    Recurrent {
        convolution: Tensor,
        recurrent: Tensor,
    },
}

/// Complete update for one layer at one token boundary.
pub enum LayerUpdate {
    Attention {
        key: Tensor,
        value: Tensor,
    },
    Recurrent {
        convolution: Tensor,
        recurrent: Tensor,
    },
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

fn check(tensor: &Tensor, shape: &[usize], device: &Device) -> Result<()> {
    if tensor.dims() != shape
        || tensor.dtype() != DType::F32
        || !tensor.device().same_device(device)
    {
        return Err(invalid("state tensor has incorrect shape, dtype or device"));
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
                } => LayerState::Recurrent {
                    convolution: Tensor::zeros((conv_width, conv_channels), DType::F32, device)?,
                    recurrent: Tensor::zeros((heads, key_dim, value_dim), DType::F32, device)?,
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

    /// Detach all inputs so third-party Candle in-place operations cannot corrupt shared state.
    pub fn read_layers(&self) -> Result<Vec<LayerRead>> {
        self.layers
            .iter()
            .map(|layer| {
                Ok(match layer {
                    LayerState::Attention(pages) => LayerRead::Attention {
                        pages: pages
                            .iter()
                            .map(|p| Ok((p.key.copy()?, p.value.copy()?)))
                            .collect::<Result<_>>()?,
                    },
                    LayerState::Recurrent {
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

    /// Validate all updates, then construct a new snapshot without changing the old one.
    /// Full pages are shared; appending copies only a partial attention tail.
    pub fn advance(&self, updates: &[LayerUpdate]) -> Result<Self> {
        let position = self
            .position
            .checked_add(1)
            .ok_or_else(|| invalid("position overflow"))?;
        if position > self.description.max_position || updates.len() != self.layers.len() {
            return Err(invalid("invalid layer update count or position"));
        }
        // Validate every layer before issuing even a copy for this transition.
        for (kind, update) in self.description.layers.iter().zip(updates) {
            match (kind, update) {
                (
                    LayerStateKind::Attention {
                        kv_heads, head_dim, ..
                    },
                    LayerUpdate::Attention { key, value },
                ) => {
                    check(key, &[*kv_heads, *head_dim], &self.device)?;
                    check(value, &[*kv_heads, *head_dim], &self.device)?;
                }
                (
                    LayerStateKind::Recurrent {
                        conv_channels,
                        conv_width,
                        heads,
                        key_dim,
                        value_dim,
                    },
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    },
                ) => {
                    check(convolution, &[*conv_width, *conv_channels], &self.device)?;
                    check(recurrent, &[*heads, *key_dim, *value_dim], &self.device)?;
                }
                _ => return Err(invalid("layer update kind does not match model")),
            }
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for ((kind, old), update) in self
            .description
            .layers
            .iter()
            .zip(&self.layers)
            .zip(updates)
        {
            layers.push(match (kind, old, update) {
                (
                    LayerStateKind::Attention { page_tokens, .. },
                    LayerState::Attention(pages),
                    LayerUpdate::Attention { key, value },
                ) => {
                    let mut pages = pages.clone();
                    let key = key.copy()?.unsqueeze(0)?;
                    let value = value.copy()?.unsqueeze(0)?;
                    let page = if let Some(tail) = pages.last().filter(|p| p.tokens < *page_tokens)
                    {
                        let tokens = tail
                            .tokens
                            .checked_add(1)
                            .ok_or_else(|| invalid("page length overflow"))?;
                        let page = Page {
                            key: Tensor::cat(&[&tail.key, &key], 0)?,
                            value: Tensor::cat(&[&tail.value, &value], 0)?,
                            tokens,
                        };
                        pages.pop();
                        page
                    } else {
                        Page {
                            key,
                            value,
                            tokens: 1,
                        }
                    };
                    pages.push(Arc::new(page));
                    LayerState::Attention(pages)
                }
                (
                    LayerStateKind::Recurrent { .. },
                    LayerState::Recurrent { .. },
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    },
                ) => LayerState::Recurrent {
                    convolution: convolution.copy()?,
                    recurrent: recurrent.copy()?,
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
    use tur_core::execution::ModelIdentity;

    #[test]
    fn test_allocation_rejects_invalid_layout() {
        let description = ModelDescription {
            identity: ModelIdentity {
                revision: "fixture".into(),
                state_format: 1,
            },
            layers: vec![LayerStateKind::Attention {
                kv_heads: 1,
                head_dim: 1,
                page_tokens: 0,
            }],
            vocab_size: 8,
            max_position: 8,
        };
        assert!(ModelState::allocate(Arc::new(description), &Device::Cpu).is_err());
    }

    #[test]
    fn test_published_state_does_not_alias_model_output_tensors() -> Result<()> {
        let description = Arc::new(ModelDescription {
            identity: ModelIdentity {
                revision: "fixture".into(),
                state_format: 1,
            },
            layers: vec![
                LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 1,
                    page_tokens: 2,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 1,
                    conv_width: 1,
                    heads: 1,
                    key_dim: 1,
                    value_dim: 1,
                },
            ],
            vocab_size: 8,
            max_position: 8,
        });
        let initial = ModelState::allocate(description, &Device::Cpu)?;
        let key = Tensor::new(&[[3.0f32]], &Device::Cpu)?;
        let value = Tensor::new(&[[4.0f32]], &Device::Cpu)?;
        let convolution = Tensor::new(&[[5.0f32]], &Device::Cpu)?;
        let recurrent = Tensor::new(&[[[6.0f32]]], &Device::Cpu)?;
        let next = initial.advance(&[
            LayerUpdate::Attention {
                key: key.clone(),
                value: value.clone(),
            },
            LayerUpdate::Recurrent {
                convolution: convolution.clone(),
                recurrent: recurrent.clone(),
            },
        ])?;
        key.zero_set()?;
        value.zero_set()?;
        convolution.zero_set()?;
        recurrent.zero_set()?;
        let mut observed = Vec::new();
        for layer in next.read_layers()? {
            match layer {
                LayerRead::Attention { pages } => {
                    for (key, value) in pages {
                        observed.extend(key.flatten_all()?.to_vec1::<f32>()?);
                        observed.extend(value.flatten_all()?.to_vec1::<f32>()?);
                    }
                }
                LayerRead::Recurrent {
                    convolution,
                    recurrent,
                } => {
                    observed.extend(convolution.flatten_all()?.to_vec1::<f32>()?);
                    observed.extend(recurrent.flatten_all()?.to_vec1::<f32>()?);
                }
            }
        }
        assert_eq!(observed, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(initial.position(), 0);
        assert_eq!(next.position(), 1);
        Ok(())
    }
}
