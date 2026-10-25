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

#[derive(Hash, PartialEq, Eq)]
enum Identity {
    Projection(usize),
    Named(String),
}

struct Accumulator {
    cols: usize,
    tokens: Vec<usize>,
    squares: Vec<Option<Tensor>>,
    hessian: Option<Tensor>,
}

/// Sums inputs on the device during reference passes, keyed by weight identity; names
/// are attached afterwards from the model's projection order.
pub(crate) struct Recorder {
    hessians: bool,
    sums: RefCell<HashMap<Identity, Accumulator>>,
}

impl Recorder {
    pub(crate) fn new(hessians: bool) -> Self {
        Self {
            hessians,
            sums: RefCell::new(HashMap::new()),
        }
    }

    /// Record the `[rows, in]` input of `layer`.
    pub(crate) fn record(&self, layer: &PackedLinear, x: &Tensor) -> Result<()> {
        self.record_expert(layer, 0, 1, x)
    }

    /// Record the input routed to `expert` of `experts` sharing the stacked `layer`.
    /// Second moments are kept for ordinary projections only.
    pub(crate) fn record_expert(
        &self,
        layer: &PackedLinear,
        expert: usize,
        experts: usize,
        x: &Tensor,
    ) -> Result<()> {
        if x.dim(1)? != layer.in_features() || expert >= experts {
            return Err(invalid("calibration input does not match its projection"));
        }
        self.accumulate(
            Identity::Projection(std::ptr::from_ref(layer).addr()),
            expert,
            experts,
            x,
        )
    }

    /// Record a named projection input without loading weights that are never multiplied.
    pub(crate) fn record_named(&self, name: &str, x: &Tensor) -> Result<()> {
        self.accumulate(Identity::Named(name.to_owned()), 0, 1, x)
    }

    fn accumulate(&self, key: Identity, expert: usize, experts: usize, x: &Tensor) -> Result<()> {
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let rows = x.dim(0)?;
        let cols = x.dim(1)?;
        let mut sums = self.sums.borrow_mut();
        let entry = sums.entry(key).or_insert_with(|| Accumulator {
            cols,
            tokens: vec![0; experts],
            squares: vec![None; experts],
            hessian: None,
        });
        if entry.tokens.len() != experts || entry.cols != cols {
            return Err(invalid("calibration input dimensions changed"));
        }
        entry.tokens[expert] = entry.tokens[expert]
            .checked_add(rows)
            .ok_or_else(|| invalid("calibration token count overflow"))?;
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
    pub(crate) fn finish(
        self,
        projections: &[(String, &PackedLinear)],
        samples: usize,
        tokens: usize,
        fingerprint: u64,
    ) -> Result<Calibration> {
        let mut sums = self.sums.into_inner();
        let mut sites = BTreeMap::new();
        let mut named: Vec<_> = projections
            .iter()
            .map(|(name, layer)| {
                (
                    name.clone(),
                    Identity::Projection(std::ptr::from_ref(*layer).addr()),
                )
            })
            .collect();
        named.extend(sums.keys().filter_map(|key| match key {
            Identity::Named(name) => Some((name.clone(), Identity::Named(name.clone()))),
            Identity::Projection(_) => None,
        }));
        for (name, key) in named {
            let Some(sum) = sums.remove(&key) else {
                continue;
            };
            let cols = sum.cols;
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
                name,
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
