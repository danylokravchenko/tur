//! Parts of encoder towers that see their input whole: Whisper's audio encoder and the
//! image towers.
use super::{Precision, checkpoint::Checkpoint, weights::upload_projection};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::{layer_norm, softmax_last_dim};
use std::collections::BTreeMap;
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// A `LayerNorm` with its checkpoint's bias and the reference's epsilon.
pub(crate) struct Norm {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) eps: f32,
}

impl Norm {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(layer_norm(x, &self.weight, &self.bias, self.eps)?)
    }
}

/// A projection and its optional bias.
pub(crate) struct Linear {
    pub(crate) weight: PackedLinear,
    pub(crate) bias: Option<Tensor>,
}

impl Linear {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weight.forward(x)?;
        Ok(match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        })
    }
}

/// Reads a tower's tensors onto its device and counts the bytes they hold.
pub(crate) struct Loader<'a> {
    pub(crate) checkpoint: &'a Checkpoint,
    pub(crate) precision: Precision,
    pub(crate) device: &'a Device,
    pub(crate) eps: f32,
    pub(crate) bytes: usize,
    /// Input width of each packed matrix, which an artifact's bytes alone do not say.
    pub(crate) packed_cols: BTreeMap<String, usize>,
}

impl Loader<'_> {
    pub(crate) fn dense(&mut self, name: &str) -> Result<Tensor> {
        let tensor = self
            .checkpoint
            .take(name)?
            .to_dtype(DType::F32)?
            .to_device(self.device)?;
        self.bytes += tensor.elem_count() * 4;
        Ok(tensor)
    }
    pub(crate) fn projection(&mut self, name: &str) -> Result<PackedLinear> {
        let entry = self.checkpoint.entry(name)?;
        let linear = if entry.dtype == safetensors::Dtype::U8 {
            // A prepacked artifact's blocks, uploaded as stored.
            let (entry, bytes) = self.checkpoint.take_raw(name)?;
            let (&[rows, _], Some(format), Some(cols)) = (
                entry.shape.as_slice(),
                self.precision.ggml(),
                self.packed_cols.get(name),
            ) else {
                return Err(invalid(format!("{name} is not a packed matrix")));
            };
            PackedLinear::from_blocks(&bytes, format, (rows, *cols), self.device)?
        } else {
            upload_projection(&self.checkpoint.take(name)?, self.precision, self.device)?
        };
        self.bytes += linear.resident_bytes();
        Ok(linear)
    }
    pub(crate) fn linear(&mut self, name: &str, bias: bool) -> Result<Linear> {
        Ok(Linear {
            weight: self.projection(&format!("{name}.weight"))?,
            bias: bias
                .then(|| self.dense(&format!("{name}.bias")))
                .transpose()?,
        })
    }
    pub(crate) fn norm(&mut self, name: &str) -> Result<Norm> {
        Ok(Norm {
            weight: self.dense(&format!("{name}.weight"))?,
            bias: self.dense(&format!("{name}.bias"))?,
            eps: self.eps,
        })
    }
}

/// `[heads, tokens, head_dim]` view of `[tokens, heads · head_dim]` rows.
pub(crate) fn heads(x: &Tensor, heads: usize) -> Result<Tensor> {
    let (tokens, width) = x.dims2()?;
    Ok(x.reshape((tokens, heads, width / heads))?
        .transpose(0, 1)?
        .contiguous()?)
}

/// The usual attention scale of a head width.
#[allow(clippy::cast_precision_loss, reason = "head widths are small")]
pub(crate) fn head_scale(dim: usize) -> f64 {
    (dim as f64).powf(-0.5)
}

/// Scaled dot-product attention of `[heads, q, dim]` queries over `[heads, k, dim]`
/// keys and values, optionally causal with `past` cached keys before the queries.
pub(crate) fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: Option<usize>,
    scale: f64,
) -> Result<Tensor> {
    let rows = q.dim(1)?;
    let mut scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
    if let Some(past) = causal {
        let keys = k.dim(1)?;
        let mask: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..keys).map(move |c| {
                    if c <= past + r {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        scores = scores.broadcast_add(&Tensor::from_vec(mask, (rows, keys), q.device())?)?;
    }
    Ok(softmax_last_dim(&scores)?.matmul(v)?)
}

/// `[tokens, heads · dim]` rows of `[heads, tokens, dim]` outputs.
pub(crate) fn merge(x: &Tensor) -> Result<Tensor> {
    let (heads, tokens, dim) = x.dims3()?;
    Ok(x.transpose(0, 1)?
        .contiguous()?
        .reshape((tokens, heads * dim))?)
}

/// Attention of `[tokens, heads · dim]` rows over keys and values that every query sees,
/// `group` heads at a time: bounds the `[group, tokens, tokens]` score buffer.
pub(crate) fn attend_whole(
    (q, k, v): (&Tensor, &Tensor, &Tensor),
    count: usize,
    group: usize,
    scale: f64,
) -> Result<Tensor> {
    let (q, k, v) = (heads(q, count)?, heads(k, count)?, heads(v, count)?);
    let groups = (0..count)
        .step_by(group)
        .map(|start| {
            let n = group.min(count - start);
            attend(
                &q.narrow(0, start, n)?,
                &k.narrow(0, start, n)?,
                &v.narrow(0, start, n)?,
                None,
                scale,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    merge(&Tensor::cat(&groups, 0)?)
}

/// Metal attention over rows that every query sees whole: the paged kernels read the
/// projections in place, as one page of a pool that needs no copy. 10× faster than
/// Candle's matmuls and softmax on Whisper's encoder shape.
#[cfg(feature = "metal")]
pub(crate) struct WholeAttention {
    pub(crate) kernels: tur_metal::kernels::Kernels,
    pub(crate) head_dim: usize,
}

#[cfg(feature = "metal")]
impl WholeAttention {
    /// Kernels for `device` when it is Metal and has kernels for the head width.
    pub(crate) fn new(head_dim: usize, device: &Device) -> Result<Option<Self>> {
        if !device.is_metal() || !matches!(head_dim, 64 | 128 | 256) {
            return Ok(None);
        }
        Ok(Some(Self {
            kernels: tur_metal::kernels::Kernels::new(device)?,
            head_dim,
        }))
    }

    /// `[keys, heads · dim]` rows as a one-page pool, read in place.
    pub(crate) fn pool(&self, k: &Tensor, v: &Tensor) -> Result<tur_metal::pool::AttentionPool> {
        use tur_core::execution::KvFormat;
        use tur_metal::pool::{AttentionPool, KvPool};
        let (keys, width) = k.dims2()?;
        let (heads, dim) = (width / self.head_dim, self.head_dim);
        Ok(AttentionPool {
            key: KvPool {
                data: k.reshape((keys, heads, dim))?,
                scales: None,
            },
            value: KvPool {
                data: v.reshape((keys, heads, dim))?,
                scales: None,
            },
            format: if k.dtype() == DType::F16 {
                KvFormat::F16
            } else {
                KvFormat::F32
            },
            page_tokens: keys,
            kv_heads: heads,
            head_dim: dim,
        })
    }

    /// `queries` tokens that each see every key of a `keys`-token pool.
    pub(crate) fn whole(&self, queries: usize, keys: usize) -> Result<tur_metal::kernels::PagedKv> {
        let visible = u32::try_from(keys).map_err(|_| invalid("too many encoder keys"))?;
        self.kernels
            .paged_kv(keys, keys, &[0], &vec![0; queries], &vec![visible; queries])
    }

    /// `[queries, heads · dim]` attention of query rows over `pool`, scaled by `scale`.
    pub(crate) fn attend(
        &self,
        q: &Tensor,
        pool: &tur_metal::pool::AttentionPool,
        kv: &tur_metal::kernels::PagedKv,
        scale: f64,
    ) -> Result<Tensor> {
        let (queries, width) = q.dims2()?;
        let dim = self.head_dim;
        self.kernels.attention_ungated(
            &q.reshape((queries, width / dim, dim))?,
            pool,
            kv,
            None,
            scale,
        )
    }
}
