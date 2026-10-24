//! Calibration statistics of projection inputs, recorded during F32 reference passes over
//! calibration text. They steer offline quantization only; serving never collects them.

use candle_core::{DType, Tensor};
use std::{cell::RefCell, collections::BTreeMap, collections::HashMap};
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// Input statistics of one projection.
#[derive(Clone, Debug)]
pub struct Site {
    /// Input rows observed, per expert (one entry for an ordinary projection).
    pub tokens: Vec<usize>,
    /// Mean squared input of each column, per expert: the importance of each weight column.
    pub squares: Vec<Vec<f32>>,
    /// Mean outer product `E[x xᵀ]`, row-major, when second moments were collected.
    pub hessian: Option<Vec<f32>>,
}

impl Site {
    /// Column importance of `expert`, or `None` if calibration never routed to it.
    pub fn importance(&self, expert: usize) -> Option<&[f32]> {
        self.tokens
            .get(expert)
            .filter(|&&t| t > 0)
            .and_then(|_| self.squares.get(expert))
            .map(Vec::as_slice)
    }
}

/// Statistics of every projection the calibration passes reached, by projection name.
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    pub sites: BTreeMap<String, Site>,
    pub samples: usize,
    pub tokens: usize,
    /// FNV-1a of the calibration token IDs, for the artifact manifest.
    pub fingerprint: u64,
}

struct Accumulator {
    tokens: Vec<usize>,
    squares: Vec<Option<Tensor>>,
    hessian: Option<Tensor>,
}

/// Sums inputs on the device during reference passes, keyed by weight identity; names
/// are attached afterwards from the model's projection order.
pub(super) struct Recorder {
    hessians: bool,
    sums: RefCell<HashMap<usize, Accumulator>>,
}

impl Recorder {
    pub(super) fn new(hessians: bool) -> Self {
        Self {
            hessians,
            sums: RefCell::new(HashMap::new()),
        }
    }

    /// Record the `[rows, in]` input of `layer`.
    pub(super) fn record(&self, layer: &PackedLinear, x: &Tensor) -> Result<()> {
        self.record_expert(layer, 0, 1, x)
    }

    /// Record the input routed to `expert` of `experts` sharing the stacked `layer`.
    /// Second moments are kept for ordinary projections only.
    pub(super) fn record_expert(
        &self,
        layer: &PackedLinear,
        expert: usize,
        experts: usize,
        x: &Tensor,
    ) -> Result<()> {
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let rows = x.dim(0)?;
        if x.dim(1)? != layer.in_features() || expert >= experts {
            return Err(invalid("calibration input does not match its projection"));
        }
        let mut sums = self.sums.borrow_mut();
        let entry = sums
            .entry(std::ptr::from_ref(layer).addr())
            .or_insert_with(|| Accumulator {
                tokens: vec![0; experts],
                squares: vec![None; experts],
                hessian: None,
            });
        if entry.tokens.len() != experts {
            return Err(invalid("calibration expert count changed"));
        }
        entry.tokens[expert] += rows;
        let squares = x.sqr()?.sum(0)?;
        entry.squares[expert] = Some(match entry.squares[expert].take() {
            Some(sum) => (sum + squares)?,
            None => squares,
        });
        if self.hessians && experts == 1 {
            let outer = x.t()?.matmul(&x)?;
            entry.hessian = Some(match entry.hessian.take() {
                Some(sum) => (sum + outer)?,
                None => outer,
            });
        }
        Ok(())
    }

    /// Normalise the sums into means and name them by `projections`.
    pub(super) fn finish(
        self,
        projections: &[(String, &PackedLinear)],
        samples: usize,
        tokens: usize,
        fingerprint: u64,
    ) -> Result<Calibration> {
        let mut sums = self.sums.into_inner();
        let mut sites = BTreeMap::new();
        for (name, layer) in projections {
            let Some(sum) = sums.remove(&std::ptr::from_ref(*layer).addr()) else {
                continue;
            };
            let cols = layer.in_features();
            let squares = sum
                .squares
                .iter()
                .zip(&sum.tokens)
                .map(|(squares, &n)| match squares {
                    Some(s) => Ok((s / n.max(1) as f64)?.to_vec1::<f32>()?),
                    None => Ok(vec![0.0; cols]),
                })
                .collect::<Result<Vec<_>>>()?;
            let hessian = sum
                .hessian
                .map(|h| -> Result<Vec<f32>> {
                    let n = sum.tokens[0].max(1) as f64;
                    Ok((h / n)?.flatten_all()?.to_vec1::<f32>()?)
                })
                .transpose()?;
            sites.insert(
                name.clone(),
                Site {
                    tokens: sum.tokens,
                    squares,
                    hessian,
                },
            );
        }
        if !sums.is_empty() {
            return Err(invalid(
                "calibration recorded a projection the model does not list",
            ));
        }
        Ok(Calibration {
            sites,
            samples,
            tokens,
            fingerprint,
        })
    }
}
