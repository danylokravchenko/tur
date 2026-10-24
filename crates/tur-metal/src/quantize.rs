#![forbid(unsafe_code)]
//! Offline host quantization into GGML block formats: round-to-nearest, importance-weighted
//! K-quant scale search, and GPTQ error compensation onto the same block grids. Every
//! method emits ordinary GGML blocks, so the existing packed kernels execute the result.

use candle_core::{
    Device, Tensor,
    quantized::{GgmlDType, QTensor},
};
use tur_core::{Result, execution::invalid};

/// Block formats a projection may be packed in.
pub const FORMATS: [GgmlDType; 4] = [
    GgmlDType::Q8_0,
    GgmlDType::Q6K,
    GgmlDType::Q5K,
    GgmlDType::Q4K,
];

/// Packed bytes of `rows` rows of `cols` values in `format`.
pub fn packed_bytes(format: GgmlDType, rows: usize, cols: usize) -> Result<usize> {
    if !FORMATS.contains(&format) {
        return Err(invalid(format!("{format:?} is not a projection format")));
    }
    if !cols.is_multiple_of(format.block_size()) {
        return Err(invalid(format!(
            "input width {cols} is not a multiple of the {format:?} block size"
        )));
    }
    (cols / format.block_size())
        .checked_mul(format.type_size())
        .and_then(|row| row.checked_mul(rows))
        .ok_or_else(|| invalid("packed size overflow"))
}

/// Pack row-major `weights` (`cols` per row). With `importance` (one non-negative weight
/// per input column, e.g. the mean squared activation), K-quant scales minimise the
/// importance-weighted error instead of the plain one; `Q8_0` has a single scale per
/// 32 values and ignores it, as GGML does.
pub fn quantize(
    weights: &[f32],
    cols: usize,
    format: GgmlDType,
    importance: Option<&[f32]>,
) -> Result<Vec<u8>> {
    let rows = weights.len().checked_div(cols).unwrap_or(0);
    let bytes = packed_bytes(format, rows, cols)?;
    if rows == 0 || rows * cols != weights.len() {
        return Err(invalid("weights do not form whole rows"));
    }
    let tensor = Tensor::from_slice(weights, (rows, cols), &Device::Cpu)?;
    let packed = match importance {
        Some(importance) if format != GgmlDType::Q8_0 => {
            if importance.len() != cols || !importance.iter().all(|w| *w >= 0.0) {
                return Err(invalid(
                    "importance needs one non-negative weight per column",
                ));
            }
            // GGML's weighted search divides by each sub-block's total weight, so columns
            // calibration never activated get a small floor instead of zero.
            let mean = importance.iter().sum::<f32>() / cols as f32;
            if mean > 0.0 {
                let floor = mean * 1e-6;
                let importance: Vec<f32> = importance.iter().map(|w| w.max(floor)).collect();
                let packed = QTensor::quantize_imatrix(&tensor, &importance, format)?;
                return repair(weights, cols, format, packed.data()?.into_owned(), bytes);
            }
            QTensor::quantize(&tensor, format)?
        }
        _ => QTensor::quantize(&tensor, format)?,
    };
    let data = packed.data()?.into_owned();
    if data.len() != bytes {
        return Err(invalid("quantizer produced an unexpected size"));
    }
    Ok(data)
}

/// Re-pack with plain rounding every row whose weighted encoding decodes to non-finite
/// values. GGML's weighted search squares the weights themselves, so rows of near-zero
/// weights (|w| ≈ 1e-37 in real checkpoints) underflow to all-zero weights and NaN scales.
fn repair(
    weights: &[f32],
    cols: usize,
    format: GgmlDType,
    mut packed: Vec<u8>,
    bytes: usize,
) -> Result<Vec<u8>> {
    if packed.len() != bytes {
        return Err(invalid("quantizer produced an unexpected size"));
    }
    let rows = weights.len() / cols;
    let row_bytes = bytes / rows;
    let storage = candle_core::quantized::QStorage::from_data(
        std::borrow::Cow::Borrowed(&packed),
        &Device::Cpu,
        format,
    )?;
    let decoded = QTensor::new(storage, (rows, cols))?
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (row, values) in decoded.chunks_exact(cols).enumerate() {
        if values.iter().all(|v| v.is_finite()) {
            continue;
        }
        let plain = quantize(&weights[row * cols..(row + 1) * cols], cols, format, None)?;
        packed[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(&plain);
    }
    Ok(packed)
}

/// `GPTQ` state for one input space: the upper Cholesky factor of the damped inverse
/// Hessian `(XᵀX)⁻¹`, which orders error compensation from the first column to the last.
pub struct Gptq {
    cols: usize,
    upper: Vec<f32>,
    /// Columns no calibration input ever activated; their weights are irrelevant.
    dead: Vec<bool>,
    importance: Vec<f32>,
}

impl Gptq {
    /// Factor the `cols × cols` row-major second-moment matrix `hessian` (`Σ x xᵀ / n`)
    /// in f64, adding `damp` times its mean diagonal. Damping grows tenfold until the
    /// matrix is positive definite, at most four times.
    pub fn new(hessian: &[f32], cols: usize, damp: f64) -> Result<Self> {
        if cols == 0 || hessian.len() != cols * cols || damp.is_nan() || damp <= 0.0 {
            return Err(invalid("GPTQ needs a square Hessian and positive damping"));
        }
        let diagonal: Vec<f64> = (0..cols)
            .map(|i| f64::from(hessian[i * cols + i]))
            .collect();
        let dead: Vec<bool> = diagonal.iter().map(|&d| d.is_nan() || d <= 0.0).collect();
        let live = dead.iter().filter(|d| !**d).count().max(1);
        let mean = diagonal
            .iter()
            .zip(&dead)
            .filter(|(_, d)| !**d)
            .map(|(v, _)| v)
            .sum::<f64>()
            / live as f64;
        let mean = if mean > 0.0 { mean } else { 1.0 };
        let mut lambda = damp * mean;
        for _ in 0..4 {
            let mut h: Vec<f64> = hessian.iter().map(|&v| f64::from(v)).collect();
            for i in 0..cols {
                if dead[i] {
                    h[i * cols..(i + 1) * cols].fill(0.0);
                    for row in 0..cols {
                        h[row * cols + i] = 0.0;
                    }
                    h[i * cols + i] = 1.0;
                } else {
                    h[i * cols + i] += lambda;
                }
            }
            if let Some(upper) = inverse_upper_factor(h, cols) {
                return Ok(Self {
                    cols,
                    upper,
                    dead,
                    importance: diagonal.iter().map(|&d| d.max(0.0) as f32).collect(),
                });
            }
            lambda *= 10.0;
        }
        Err(invalid(
            "calibration Hessian is not positive definite after damping",
        ))
    }

    /// Quantize row-major `weights` (`cols` per row) in `format` (`Q8_0` or Q4K). Each
    /// block's scales are chosen from the already compensated weights when compensation
    /// reaches it, by the same importance-weighted search as [`quantize`]; each value is
    /// then rounded onto that grid and its error spread over the remaining columns.
    pub fn quantize(&self, weights: &[f32], format: GgmlDType) -> Result<Vec<u8>> {
        let cols = self.cols;
        let rows = weights.len().checked_div(cols).unwrap_or(0);
        if rows == 0 || rows * cols != weights.len() {
            return Err(invalid("weights do not form whole rows"));
        }
        let grid = &Grid::new(format)?;
        let row_bytes = packed_bytes(format, 1, cols)?;
        let mut out = vec![0u8; rows * row_bytes];
        let tile = 32;
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(rows.div_ceil(tile))
            .max(1);
        let per_thread = rows.div_ceil(tile).div_ceil(threads) * tile;
        std::thread::scope(|scope| {
            let handles: Vec<_> = weights
                .chunks(per_thread * cols)
                .zip(out.chunks_mut(per_thread * row_bytes))
                .map(|(weights, out)| {
                    scope.spawn(move || -> Result<()> {
                        for (weights, out) in weights
                            .chunks(tile * cols)
                            .zip(out.chunks_mut(tile * row_bytes))
                        {
                            self.tile(weights, out, grid)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .try_for_each(|h| h.join().map_err(|_| invalid("GPTQ worker failed"))?)
        })?;
        Ok(out)
    }

    /// Compensate one tile of rows; rows are independent given the shared factor.
    fn tile(&self, weights: &[f32], out: &mut [u8], grid: &Grid) -> Result<()> {
        let cols = self.cols;
        let rows = weights.len() / cols;
        let mut w = weights.to_vec();
        for row in w.chunks_exact_mut(cols) {
            for (x, dead) in row.iter_mut().zip(&self.dead) {
                if *dead {
                    *x = 0.0;
                }
            }
        }
        let blocks = cols / grid.block;
        let row_bytes = blocks * grid.bytes;
        let mut slab = vec![0f32; rows * grid.block];
        let mut errors = vec![0f32; rows];
        for b in 0..blocks {
            let start = b * grid.block;
            for (r, row) in w.chunks_exact(cols).enumerate() {
                slab[r * grid.block..(r + 1) * grid.block]
                    .copy_from_slice(&row[start..start + grid.block]);
            }
            let encoded = quantize(
                &slab,
                grid.block,
                grid.format,
                Some(&self.importance[start..start + grid.block]),
            )?;
            for r in 0..rows {
                out[r * row_bytes + b * grid.bytes..r * row_bytes + (b + 1) * grid.bytes]
                    .copy_from_slice(&encoded[r * grid.bytes..(r + 1) * grid.bytes]);
            }
            for j in start..start + grid.block {
                let u = &self.upper[j * cols..(j + 1) * cols];
                let pivot = u[j];
                for (r, row) in w.chunks_exact_mut(cols).enumerate() {
                    let block = &mut out[r * row_bytes + b * grid.bytes..][..grid.bytes];
                    let value = row[j];
                    let dequantized = grid.set(block, j - start, value);
                    errors[r] = (value - dequantized) / pivot;
                }
                for (row, &error) in w.chunks_exact_mut(cols).zip(&errors) {
                    for (x, &h) in row[j + 1..].iter_mut().zip(&u[j + 1..]) {
                        *x = (-error).mul_add(h, *x);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Element access to one GGML block whose scales are already encoded.
struct Grid {
    format: GgmlDType,
    block: usize,
    bytes: usize,
}

impl Grid {
    fn new(format: GgmlDType) -> Result<Self> {
        match format {
            GgmlDType::Q8_0 | GgmlDType::Q4K => Ok(Self {
                format,
                block: format.block_size(),
                bytes: format.type_size(),
            }),
            _ => Err(invalid(format!(
                "GPTQ supports Q8_0 and Q4K, not {format:?}"
            ))),
        }
    }

    /// Round `value` onto element `index`'s grid, store it and return its dequantized
    /// value, exactly as GGML's dequantization reads it back.
    fn set(&self, block: &mut [u8], index: usize, value: f32) -> f32 {
        if self.format == GgmlDType::Q8_0 {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let q = if d == 0.0 {
                0.0
            } else {
                (value / d).round().clamp(-128.0, 127.0)
            };
            block[2 + index] = (q as i8).to_le_bytes()[0];
            d * q
        } else {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scale, min) = scale_min_k4(index / 32, &block[4..16]);
            let (d, m) = (d * f32::from(scale), dmin * f32::from(min));
            let q = if d == 0.0 {
                0
            } else {
                ((value + m) / d).round().clamp(0.0, 15.0) as u8
            };
            let (chunk, within) = (index / 64, index % 64);
            let byte = &mut block[16 + chunk * 32 + within % 32];
            *byte = if within < 32 {
                (*byte & 0xF0) | q
            } else {
                (*byte & 0x0F) | (q << 4)
            };
            d.mul_add(f32::from(q), -m)
        }
    }
}

/// Six-bit scale and min of Q4K sub-block `j` from its packed 12 scale bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x3FF);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Upper factor `U` with `UᵀU = H⁻¹` for a symmetric positive definite `h`, or `None`.
fn inverse_upper_factor(mut h: Vec<f64>, n: usize) -> Option<Vec<f32>> {
    cholesky(&mut h, n)?;
    // Row j of `inverse` is column j of L⁻¹: the forward substitution L x = e_j, whose
    // entries before j are zero. Columns are independent.
    let mut inverse = vec![0f64; n * n];
    parallel_rows(&mut inverse, n, |j, x| {
        x[j] = 1.0 / h[j * n + j];
        for i in j + 1..n {
            let row = &h[i * n + j..i * n + i];
            let sum = dot(row, &x[j..i]);
            x[i] = -sum / h[i * n + i];
        }
    });
    // H⁻¹ = L⁻ᵀ L⁻¹: entry (i, j) is the dot product of columns i and j of L⁻¹.
    let mut product = vec![0f64; n * n];
    parallel_rows(&mut product, n, |i, row| {
        let a = &inverse[i * n..(i + 1) * n];
        for (j, value) in row[..=i].iter_mut().enumerate() {
            let b = &inverse[j * n..(j + 1) * n];
            *value = dot(&a[i..], &b[i..]);
        }
    });
    cholesky(&mut product, n)?;
    // The lower factor M of H⁻¹ = M Mᵀ gives U = Mᵀ.
    let mut upper = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            upper[j * n + i] = product[i * n + j] as f32;
        }
    }
    Some(upper)
}

/// Run `f(row index, row)` over every `n`-wide row of `values`, rows dealt round-robin to
/// the available cores so that triangular workloads stay balanced.
fn parallel_rows(values: &mut [f64], n: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let mut shares: Vec<Vec<(usize, &mut [f64])>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, row) in values.chunks_exact_mut(n).enumerate() {
        shares[i % threads].push((i, row));
    }
    let f = &f;
    std::thread::scope(|scope| {
        for share in shares {
            scope.spawn(move || {
                for (i, row) in share {
                    f(i, row);
                }
            });
        }
    });
}

/// Dot product over eight independent accumulators, so the reduction vectorizes (an
/// ordered float sum cannot be reassociated by the compiler).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0f64; 8];
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, b_tail) = b[..a.len()].as_chunks::<8>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..8 {
            lanes[lane] = x[lane].mul_add(y[lane], lanes[lane]);
        }
    }
    let tail: f64 = a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f64>() + tail
}

/// In-place lower Cholesky factor of row-major `a` (lower triangle read and written).
fn cholesky(a: &mut [f64], n: usize) -> Option<()> {
    for j in 0..n {
        let rest = &mut a[j * n..];
        let (row_j, below) = rest.split_at_mut(n);
        let diagonal = row_j[j] - dot(&row_j[..j], &row_j[..j]);
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return None;
        }
        let pivot = diagonal.sqrt();
        row_j[j] = pivot;
        let row_j = &row_j[..j];
        for row_i in below.chunks_exact_mut(n) {
            row_i[j] = (row_i[j] - dot(&row_i[..j], row_j)) / pivot;
        }
    }
    Some(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0)
            })
            .collect()
    }

    fn dequantize(bytes: &[u8], format: GgmlDType, rows: usize, cols: usize) -> Result<Vec<f32>> {
        let storage = candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Borrowed(bytes),
            &Device::Cpu,
            format,
        )?;
        Ok(QTensor::new(storage, (rows, cols))?
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?)
    }

    /// Correlated inputs: a few shared factors plus noise, with uneven column scales.
    fn calibration(tokens: usize, cols: usize) -> Vec<f32> {
        let factors = values(tokens * 4, 7);
        let noise = values(tokens * cols, 11);
        let mixing = values(4 * cols, 13);
        (0..tokens * cols)
            .map(|i| {
                let (t, c) = (i / cols, i % cols);
                let shared: f32 = (0..4)
                    .map(|f| factors[t * 4 + f] * mixing[f * cols + c])
                    .sum();
                0.3f32.mul_add(noise[i], shared) * if c % 7 == 0 { 4.0 } else { 1.0 }
            })
            .collect()
    }

    fn hessian(x: &[f32], cols: usize) -> Vec<f32> {
        let tokens = x.len() / cols;
        let mut h = vec![0f64; cols * cols];
        for row in x.chunks_exact(cols) {
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] = f64::from(row[i]).mul_add(f64::from(row[j]), h[i * cols + j]);
                }
            }
        }
        h.iter().map(|v| (v / tokens as f64) as f32).collect()
    }

    /// Mean squared output error of `w_hat` against `w` over inputs `x`.
    fn output_error(w: &[f32], w_hat: &[f32], x: &[f32], cols: usize) -> f64 {
        let mut total = 0.0;
        for (row, row_hat) in w.chunks_exact(cols).zip(w_hat.chunks_exact(cols)) {
            for input in x.chunks_exact(cols) {
                let e: f32 = row
                    .iter()
                    .zip(row_hat)
                    .zip(input)
                    .map(|((a, b), x)| (a - b) * x)
                    .sum();
                total += f64::from(e * e);
            }
        }
        total / (w.len() / cols * x.len() / cols) as f64
    }

    #[test]
    fn test_grid_rounds_dequantized_values_back_to_their_own_quants() -> Result<()> {
        let cols = 256;
        let w = values(2 * cols, 17);
        for format in [GgmlDType::Q8_0, GgmlDType::Q4K] {
            let grid = Grid::new(format)?;
            let packed = quantize(&w, cols, format, None)?;
            let decoded = dequantize(&packed, format, 2, cols)?;
            let mut rewritten = packed.clone();
            for (b, block) in rewritten.chunks_exact_mut(grid.bytes).enumerate() {
                for i in 0..grid.block {
                    let value = decoded[b * grid.block + i];
                    assert_eq!(grid.set(block, i, value), value, "{format:?} {b} {i}");
                }
            }
            assert_eq!(rewritten, packed, "{format:?}");
        }
        Ok(())
    }

    #[test]
    fn test_packed_sizes_reject_partial_blocks_and_non_projection_formats() {
        assert_eq!(packed_bytes(GgmlDType::Q4K, 3, 512).ok(), Some(3 * 2 * 144));
        assert_eq!(packed_bytes(GgmlDType::Q8_0, 2, 64).ok(), Some(2 * 2 * 34));
        assert!(packed_bytes(GgmlDType::Q4K, 1, 300).is_err());
        assert!(packed_bytes(GgmlDType::F16, 1, 256).is_err());
    }

    #[test]
    fn test_gptq_blocks_decode_to_their_grid_and_reduce_output_error() -> Result<()> {
        let (rows, cols) = (48, 512);
        let w: Vec<f32> = values(rows * cols, 3).iter().map(|v| v * 0.05).collect();
        let x = calibration(1024, cols);
        let h = hessian(&x, cols);
        let gptq = Gptq::new(&h, cols, 0.01)?;
        for format in [GgmlDType::Q8_0, GgmlDType::Q4K] {
            let rtn = dequantize(&quantize(&w, cols, format, None)?, format, rows, cols)?;
            let importance: Vec<f32> = (0..cols).map(|i| h[i * cols + i]).collect();
            let weighted = dequantize(
                &quantize(&w, cols, format, Some(&importance))?,
                format,
                rows,
                cols,
            )?;
            let packed = gptq.quantize(&w, format)?;
            assert_eq!(packed.len(), packed_bytes(format, rows, cols)?);
            let compensated = dequantize(&packed, format, rows, cols)?;
            let (rtn, weighted, compensated) = (
                output_error(&w, &rtn, &x, cols),
                output_error(&w, &weighted, &x, cols),
                output_error(&w, &compensated, &x, cols),
            );
            assert!(
                compensated < 0.8 * rtn && compensated < weighted,
                "{format:?}: rtn {rtn:e}, weighted {weighted:e}, gptq {compensated:e}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_gptq_ignores_dead_columns_and_rejects_bad_inputs() -> Result<()> {
        let cols = 256;
        let mut x = calibration(512, cols);
        for row in x.chunks_exact_mut(cols) {
            row[5] = 0.0;
        }
        let gptq = Gptq::new(&hessian(&x, cols), cols, 0.01)?;
        let w = values(4 * cols, 5);
        let decoded = dequantize(&gptq.quantize(&w, GgmlDType::Q4K)?, GgmlDType::Q4K, 4, cols)?;
        assert!(decoded.iter().all(|v| v.is_finite()));
        assert!(gptq.quantize(&w[..cols - 1], GgmlDType::Q4K).is_err());
        assert!(gptq.quantize(&w, GgmlDType::Q6K).is_err());
        assert!(Gptq::new(&[1.0; 3], 2, 0.01).is_err());
        assert!(Gptq::new(&[1.0, 0.0, 0.0, 1.0], 2, 0.0).is_err());
        Ok(())
    }

    #[test]
    fn test_importance_weighting_changes_k_quant_scales_only() -> Result<()> {
        let (rows, cols) = (8, 256);
        let w = values(rows * cols, 9);
        let importance: Vec<f32> = (0..cols)
            .map(|i| if i < 8 { 100.0 } else { 0.01 })
            .collect();
        let plain = quantize(&w, cols, GgmlDType::Q4K, None)?;
        let weighted = quantize(&w, cols, GgmlDType::Q4K, Some(&importance))?;
        assert_eq!(plain.len(), weighted.len());
        assert_ne!(plain, weighted);
        // Q8_0 has no importance-weighted search.
        assert_eq!(
            quantize(&w, cols, GgmlDType::Q8_0, None)?,
            quantize(&w, cols, GgmlDType::Q8_0, Some(&importance))?
        );
        assert!(quantize(&w, cols, GgmlDType::Q4K, Some(&importance[1..])).is_err());
        // Rows of near-zero weights, as real checkpoints hold, fall back to plain rounding.
        let mut tiny = w.clone();
        tiny[cols..2 * cols].iter_mut().for_each(|v| *v *= 1e-37);
        let raw = QTensor::quantize_imatrix(
            &Tensor::from_slice(&tiny, (rows, cols), &Device::Cpu)?,
            &vec![1.0; cols],
            GgmlDType::Q4K,
        )?;
        assert!(
            !raw.dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        let packed = quantize(&tiny, cols, GgmlDType::Q4K, Some(&vec![1.0; cols]))?;
        assert!(
            dequantize(&packed, GgmlDType::Q4K, rows, cols)?
                .iter()
                .all(|v| v.is_finite())
        );
        assert_eq!(
            packed[144..288],
            quantize(&tiny[cols..2 * cols], cols, GgmlDType::Q4K, None)?[..]
        );
        // Sub-blocks no calibration input reached still get finite scales.
        let silent: Vec<f32> = (0..cols).map(|i| if i < 64 { 0.0 } else { 1.0 }).collect();
        for importance in [silent, vec![0.0; cols]] {
            let packed = quantize(&w, cols, GgmlDType::Q4K, Some(&importance))?;
            let decoded = dequantize(&packed, GgmlDType::Q4K, rows, cols)?;
            assert!(decoded.iter().all(|v| v.is_finite()));
        }
        Ok(())
    }
}
