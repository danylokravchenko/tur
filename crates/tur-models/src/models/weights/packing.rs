//! Streaming projection packing shared by checkpoint loaders and offline quantization.
use super::{Precision, Projection};
use crate::{
    models::checkpoint::Checkpoint,
    quantization::{Calibration, Site},
};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use tur_core::{Result, execution::invalid};
use tur_metal::{
    packed::PackedLinear,
    quantize::{Gptq, quantize},
};

/// Full-precision rows staged on the host at once while packing a projection.
pub(crate) const STAGING_BYTES: usize = 64 << 20;

/// How weights are rounded into their block format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Round to nearest with GGML's own scale search.
    Rtn,
    /// K-quant scales searched under the calibration's per-column mean squared input.
    Imatrix,
    /// Importance-weighted scales plus GPTQ error compensation with the calibration
    /// Hessian. Expert banks, whose inputs differ per expert, fall back to `Imatrix`.
    Gptq,
}

impl std::str::FromStr for Method {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "rtn" => Ok(Self::Rtn),
            "imatrix" => Ok(Self::Imatrix),
            "gptq" => Ok(Self::Gptq),
            _ => Err(format!(
                "method must be rtn, imatrix or gptq, not '{value}'"
            )),
        }
    }
}

/// One projection's packed payload.
pub(crate) enum Packed {
    Dense(Vec<f32>),
    Blocks(Vec<u8>),
}

/// Packs projections by one method, using calibration statistics when it has them.
#[derive(Clone, Copy)]
pub struct Packer<'a> {
    pub method: Method,
    /// GPTQ damping relative to the mean Hessian diagonal.
    pub damp: f64,
    pub calibration: Option<&'a Calibration>,
}

impl Packer<'_> {
    /// Round-to-nearest without statistics, as conversion at load time packs.
    pub const RTN: Packer<'static> = Packer {
        method: Method::Rtn,
        damp: 0.01,
        calibration: None,
    };

    pub(crate) fn site(&self, projection: &Projection) -> Option<&Site> {
        self.calibration.and_then(|c| c.sites.get(&projection.name))
    }

    /// The projection's GPTQ factor when this method compensates it; independent of the
    /// precision, so one factorization serves every candidate format.
    pub(crate) fn factor(&self, projection: &Projection) -> Result<Option<Gptq>> {
        if self.method != Method::Gptq || projection.experts > 1 {
            return Ok(None);
        }
        self.site(projection)
            .and_then(|s| s.hessian.as_deref())
            .map(|h| Gptq::new(h, projection.cols, self.damp))
            .transpose()
    }

    /// Pack `projection` streamed from `checkpoint` in row chunks.
    pub(crate) fn pack(
        &self,
        checkpoint: &Checkpoint,
        projection: &Projection,
        precision: Precision,
    ) -> Result<Packed> {
        let mut packed = Vec::new();
        self.pack_chunks(checkpoint, projection, precision, |bytes| {
            packed.extend_from_slice(bytes);
            Ok(())
        })?;
        Ok(finish(packed, precision))
    }

    /// Share rounding and chunk order between in-memory loading and streamed artifacts.
    pub(crate) fn pack_chunks(
        &self,
        checkpoint: &Checkpoint,
        projection: &Projection,
        precision: Precision,
        mut consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let per_expert = projection.rows / projection.experts;
        let chunk = if projection.experts > 1 {
            per_expert
        } else {
            (STAGING_BYTES / (projection.cols * 4)).max(1)
        };
        let gptq = self.factor(projection)?;
        let mut row = 0;
        for (part, _) in &projection.parts {
            for rows in checkpoint.take_rows(part, chunk)? {
                let values = rows?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expert = row / per_expert;
                row += values.len() / projection.cols;
                let mut packed = Vec::new();
                self.pack_rows(
                    &values,
                    projection,
                    precision,
                    expert,
                    gptq.as_ref(),
                    &mut packed,
                )?;
                consume(&packed)?;
            }
        }
        Ok(())
    }

    /// Pack in-memory row-major `values` of `projection` with its precomputed `gptq` factor.
    pub(crate) fn pack_values(
        &self,
        values: &[f32],
        projection: &Projection,
        precision: Precision,
        gptq: Option<&Gptq>,
    ) -> Result<Packed> {
        let mut packed = Vec::new();
        for (expert, rows) in values
            .chunks(projection.rows / projection.experts * projection.cols)
            .enumerate()
        {
            self.pack_rows(rows, projection, precision, expert, gptq, &mut packed)?;
        }
        Ok(finish(packed, precision))
    }

    fn pack_rows(
        &self,
        values: &[f32],
        projection: &Projection,
        precision: Precision,
        expert: usize,
        gptq: Option<&Gptq>,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let Some(format) = precision.ggml() else {
            out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            return Ok(());
        };
        let packed = if let Some(gptq) = gptq {
            gptq.quantize(values, format)?
        } else {
            let importance = match self.method {
                Method::Rtn => None,
                Method::Imatrix | Method::Gptq => {
                    self.site(projection).and_then(|s| s.importance(expert))
                }
            };
            quantize(values, projection.cols, format, importance)?
        };
        out.extend_from_slice(&packed);
        Ok(())
    }
}

fn finish(bytes: Vec<u8>, precision: Precision) -> Packed {
    if precision.ggml().is_some() {
        Packed::Blocks(bytes)
    } else {
        Packed::Dense(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
    }
}

impl Packed {
    /// Upload as an `[rows, cols]` projection.
    pub(crate) fn upload(
        self,
        precision: Precision,
        (rows, cols): (usize, usize),
        device: &Device,
    ) -> Result<PackedLinear> {
        match (self, precision.ggml()) {
            (Self::Blocks(bytes), Some(format)) => {
                PackedLinear::from_blocks(&bytes, format, (rows, cols), device)
            }
            (Self::Dense(values), None) => PackedLinear::dense(
                &Tensor::from_vec(values, (rows, cols), &Device::Cpu)?,
                device,
            ),
            _ => Err(invalid("packed payload does not match its precision")),
        }
    }
}
