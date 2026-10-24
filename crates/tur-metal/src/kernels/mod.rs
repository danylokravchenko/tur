#![forbid(unsafe_code)]
//! Fused Metal kernels for hybrid decoders.
//!
//! Activations are F32. Inputs are validated (dtype, device, strides, extents and index
//! ranges) before encoding; nothing falls back to the CPU. Kernels only enqueue work: the
//! caller fences with [`crate::completion::Completion`] before reading results.

mod moe;
mod runtime;
pub use moe::{ExpertRoutes, SharedExpert};

use crate::{
    packed::PackedLinear,
    pool::{AttentionPool, KvPool},
    sampling::{MAX_ROUNDS, PenaltyEntry, PenaltyParams, SampleParams},
};
use candle_core::{DType, Device, Tensor, quantized::GgmlDType};
pub use runtime::{Capabilities, LAYOUT_VERSION, Output, Tuning};
use runtime::{Pipeline, Runtime, View, to_u32};
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{KvFormat, invalid},
};

/// Key rows owned by one delta-step simdgroup, in order of preference: 16 is fastest on M1
/// and 32 needs half the threads, for pipelines that cannot run `dk / 16` simdgroups.
const DELTA_STEP_ROWS: [usize; 2] = [16, 32];
/// Key rows held by one delta-scan lane (`SCAN_ROWS` in the MSL).
const DELTA_SCAN_ROWS: usize = 32;
/// Output rows per GEMV simdgroup (`NR` in the MSL `QMV` instantiations).
fn gemv_rows(format: GgmlDType, batch: usize) -> usize {
    match format {
        GgmlDType::Q8_0 if batch <= 8 => 4,
        _ => 2,
    }
}

#[repr(C)]
struct RmsParams {
    rows: u32,
    width: u32,
    x_stride: u32,
    r_stride: u32,
    eps: f32,
}

#[repr(C)]
struct RopeParams {
    tokens: u32,
    heads: u32,
    dim: u32,
    row_stride: u32,
    head_stride: u32,
    col_offset: u32,
    half_rotary: u32,
    eps: f32,
}

#[repr(C)]
struct GluParams {
    rows: u32,
    width: u32,
}

#[repr(C)]
struct GatedParams {
    rows: u32,
    dim: u32,
    heads: u32,
    z_stride: u32,
    z_col: u32,
    eps: f32,
}

#[repr(C)]
struct ConvParams {
    rows: u32,
    channels: u32,
    taps: u32,
    x_stride: u32,
}

#[repr(C)]
struct ScanParams {
    sequences: u32,
    heads: u32,
    groups: u32,
    key_width: u32,
    key_dim: u32,
    value_dim: u32,
    mixed_stride: u32,
    proj_stride: u32,
    b_col: u32,
    a_col: u32,
    prep_stride: u32,
}

#[repr(C)]
struct AttentionParams {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    page: u32,
    splits: u32,
    split_keys: u32,
    gate_stride: u32,
    gate_col: u32,
    gate_head_stride: u32,
    group: u32,
    scale: f32,
}

#[repr(C)]
struct TiledParams {
    tiles: u32,
    heads: u32,
    kv_heads: u32,
    page: u32,
    group: u32,
    gate_stride: u32,
    gate_col: u32,
    gate_head_stride: u32,
    scale: f32,
}

/// Query tokens per tile (8 rows per simdgroup), key block and staged dimensions of the
/// tiled kernel (`TQ`, `TK`, `TD` in the MSL).
const TILE_QUERIES: usize = 32;
/// Threads of one tiled-attention threadgroup: a simdgroup per eight query rows.
const TILE_THREADS: usize = TILE_QUERIES / 8 * 32;
const TILE_KEYS: usize = 32;
const TILE_DIMS: usize = 64;

#[repr(C)]
struct KvWriteParams {
    tokens: u32,
    kv_heads: u32,
    dim: u32,
    page: u32,
    group: u32,
    k_stride: u32,
    k_col: u32,
    k_head_stride: u32,
    v_stride: u32,
    v_col: u32,
    v_head_stride: u32,
    inv_127: f32,
}

#[repr(C)]
struct ArgmaxParams {
    rows: u32,
    cols: u32,
    stride: u32,
}

#[repr(C)]
struct MaskArgs {
    rows: u32,
    cols: u32,
    stride: u32,
    words: u32,
}

#[repr(C)]
struct PenaltyArgs {
    entries: u32,
    stride: u32,
}

#[repr(C)]
struct SampleArgs {
    rows: u32,
    cols: u32,
    stride: u32,
    rounds: u32,
}

#[repr(C)]
struct QmmParams {
    n: u32,
    k: u32,
    x_stride: u32,
    y_stride: u32,
    rows: u32,
}

/// Output columns per threadgroup of the small-batch matrix kernels (`8 * NT` in the MSL).
fn qmm_columns(format: GgmlDType) -> usize {
    match format {
        GgmlDType::Q4K => 32,
        _ => 16,
    }
}

#[repr(C)]
struct QmvParams {
    n: u32,
    k: u32,
    x_stride: u32,
    y_stride: u32,
}

/// Heads read in place from projection rows: head `h` of token `t` starts at column
/// `col_offset + h * head_stride` of row `t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heads {
    pub count: usize,
    pub dim: usize,
    pub head_stride: usize,
    pub col_offset: usize,
}

/// Gated `DeltaNet` geometry. The projection row holds `[q | k | v | z | b | a]`, with the
/// `b` and `a` gate columns at `b_col` and `a_col`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub b_col: usize,
    pub a_col: usize,
}

impl DeltaShape {
    pub fn key_width(&self) -> usize {
        self.key_heads * self.key_dim
    }

    pub fn value_width(&self) -> usize {
        self.value_heads * self.value_dim
    }

    /// Convolved channels: `q | k | v`.
    pub fn channels(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }

    /// Floats per row of the recurrent prefill's prepared inputs: q and k scales and q·k
    /// per key head, then (decay, beta) per value head, rounded to whole float4s.
    pub fn prepared_width(&self) -> usize {
        (3 * self.key_heads + 2 * self.value_heads).next_multiple_of(4)
    }
}

/// Validated paged K/V addressing for a packed batch. Key `j` of query token `t` lives in
/// pool slot `table[first_block[t] + j / page] * page + j % page`; token `t` attends to
/// keys `0..visible[t]` and, when written, is stored at key `visible[t] - 1`. One
/// addressing serves every attention layer of a pass: page ids span all layers.
pub struct PagedKv {
    table: Tensor,
    first_block: Tensor,
    visible: Tensor,
    /// `(first token, count)` query tiles for the tiled kernel, or `None` when some
    /// sequence's tokens are not consecutive positions sharing one page table.
    tiles: Option<(Tensor, usize)>,
    page: usize,
    slots: usize,
    tokens: usize,
    max_visible: usize,
}

impl PagedKv {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn max_visible(&self) -> usize {
        self.max_visible
    }

    /// Query tiles of the tiled kernel, if the layout allows it.
    pub fn tiles(&self) -> Option<usize> {
        self.tiles.as_ref().map(|(_, count)| *count)
    }
}

/// Split runs of tokens sharing a page table, at consecutive positions, into tiles of at
/// most `TILE_QUERIES`; `None` if any run breaks that shape.
fn query_tiles(first_block: &[u32], visible: &[u32]) -> Option<Vec<u32>> {
    let mut tiles = Vec::new();
    let mut start = 0;
    while start < visible.len() {
        let mut end = start + 1;
        while end < visible.len() && first_block[end] == first_block[start] {
            if visible[end - 1].checked_add(1) != Some(visible[end]) {
                return None;
            }
            end += 1;
        }
        for tile in (start..end).step_by(TILE_QUERIES) {
            tiles.extend([
                u32::try_from(tile).ok()?,
                u32::try_from((end - tile).min(TILE_QUERIES)).ok()?,
            ]);
        }
        start = end;
    }
    Some(tiles)
}

/// One sequence's recurrent addressing in a packed pass: its token count, the slot holding
/// its committed state, and the slots receiving its state after the last token (one) or
/// after every token (one per token).
#[derive(Clone, Copy, Debug)]
pub struct RecurrentSpan<'a> {
    pub tokens: usize,
    /// Committed tokens before this pass. Seeds the stochastic rounding of 16-bit state,
    /// so stored values depend only on the request, not on batch composition or slots.
    pub position: usize,
    pub read: u32,
    pub writes: &'a [u32],
}

/// Validated recurrent addressing of a packed pass: consecutive sequences covering every
/// packed row. Uploaded once per pass and shared by every recurrent layer, whose pools
/// have equal slot counts, so each layer's convolution and scan is one dispatch.
pub struct RecurrentSpans {
    /// `[row, tokens, read, first write, every, position]` per sequence (`RecurrentSpan`
    /// in the MSL).
    spans: Tensor,
    /// Sequence of each packed row.
    rows: Tensor,
    slots: Tensor,
    /// Sequence orders with their counts: single-token sequences (the decode step), the
    /// rest (the scan), and all of them (the scan alone). Each may be empty.
    single: (Tensor, usize),
    multi: (Tensor, usize),
    all: (Tensor, usize),
    sequences: usize,
    tokens: usize,
    /// Highest slot read or written, checked against each pool.
    max_slot: u32,
}

impl RecurrentSpans {
    pub fn sequences(&self) -> usize {
        self.sequences
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    fn check(&self, capacity: usize, rows: usize) -> Result<()> {
        if self.max_slot as usize >= capacity || rows != self.tokens {
            return Err(invalid(
                "recurrent spans exceed the slot pool or do not cover the packed rows",
            ));
        }
        Ok(())
    }
}

/// Split-K choice for one attention dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionSplit {
    pub splits: usize,
    pub split_keys: usize,
}

/// Compiled kernels for one Metal device plus their launch tuning.
#[derive(Clone)]
pub struct Kernels {
    runtime: Arc<Runtime>,
    tuning: Tuning,
}

impl Kernels {
    /// Query capabilities and compile every pipeline for `device`, which must be Metal.
    pub fn new(device: &Device) -> Result<Self> {
        let runtime = Runtime::new(device)?;
        Ok(Self {
            tuning: runtime.tuning,
            runtime: Arc::new(runtime),
        })
    }

    pub fn device(&self) -> &Device {
        &self.runtime.device
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.runtime.capabilities
    }

    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// Replace launch tuning, e.g. with benchmarked per-device values.
    pub fn with_tuning(mut self, tuning: Tuning) -> Result<Self> {
        if tuning.row_threads == 0
            || tuning.simdgroups == 0
            || tuning.scan_simdgroups == 0
            || tuning.elementwise_threads == 0
            || tuning.attention_parallelism == 0
            || [tuning.attention_head_group, tuning.attention_q8_head_group]
                .iter()
                .any(|hg| ![1, 2].contains(hg))
            || tuning.min_split_keys == 0
            || [tuning.gemv_q8_0_rows, tuning.gemv_q4k_rows]
                .iter()
                .any(|&(min, max)| min == 0 || max > 8)
            || [tuning.qmm_q8_0_rows, tuning.qmm_q4k_rows]
                .iter()
                .any(|&(min, max)| min == 0 || max > 16)
        {
            return Err(invalid("invalid kernel tuning"));
        }
        self.tuning = tuning;
        Ok(self)
    }

    /// Allocate an output that several dispatches fill before [`Kernels::finish`].
    pub fn output(&self, rows: usize, cols: usize) -> Result<Output> {
        self.runtime.output(rows, cols)
    }

    /// Turn a filled output into an F32 tensor of `shape`.
    pub fn finish(&self, output: Output, shape: impl Into<candle_core::Shape>) -> Result<Tensor> {
        output.into_tensor(&self.runtime.metal, shape)
    }

    fn view<'a>(&self, tensor: &'a Tensor) -> Result<View<'a>> {
        self.runtime.view(tensor, DType::F32)
    }

    fn pipeline(&self, name: &str) -> Result<&Pipeline> {
        self.runtime.pipeline(name)
    }

    /// One simdgroup per item: (threadgroups, threads).
    fn simdgroup_launch(&self, pipeline: &Pipeline, items: usize) -> (usize, usize) {
        let simdgroups = pipeline.simdgroups(self.tuning.simdgroups).max(1);
        (items.div_ceil(simdgroups), simdgroups * 32)
    }

    /// First maximum of each row of `x` (`[rows, cols]`) and whether the row is valid:
    /// returns F32 `[rows, 2]` of (index, 1.0 or 0.0). −∞ marks a masked token; NaN, +∞ or
    /// a row of only −∞ is invalid. Ties keep the lowest index.
    pub fn argmax(&self, x: &Tensor) -> Result<Tensor> {
        let xv = self.view(x)?;
        // Indices travel as F32, exact below 2^24.
        if xv.rows == 0 || xv.cols == 0 || xv.cols > 1 << 24 {
            return Err(invalid(
                "argmax needs 1..=2^24 columns and at least one row",
            ));
        }
        let params = ArgmaxParams {
            rows: to_u32(xv.rows, "rows")?,
            cols: to_u32(xv.cols, "columns")?,
            stride: to_u32(xv.row_stride, "row stride")?,
        };
        let pipeline = self.pipeline("argmax_f32")?;
        let out = self.output(xv.rows, 2)?;
        // One threadgroup per row; the widest group shortens each row's pass.
        let threads = pipeline.threads(self.capabilities().max_threads_per_threadgroup);
        self.runtime
            .dispatch(pipeline, "tur_argmax", xv.rows, threads, |e| {
                xv.bind(e, 0)?;
                out.bind(e, 1);
                e.set_bytes(2, &params);
                Ok(())
            })?;
        self.finish(out, (xv.rows, 2))
    }

    /// Apply grammar masks in place: logit row `rows[r]` keeps only the tokens allowed by
    /// mask `r`, `words` LSB-first `u32` bitset words starting at `bits[r * words]`; every
    /// other token becomes −∞.
    pub fn mask(&self, logits: &Tensor, rows: &[u32], bits: &[u32]) -> Result<()> {
        let xv = self.view(logits)?;
        if rows.is_empty() {
            return Ok(());
        }
        let words = xv.cols.div_ceil(32);
        let mut sorted = rows.to_vec();
        sorted.sort_unstable();
        if bits.len() != rows.len() * words
            || sorted.windows(2).any(|pair| pair[0] == pair[1])
            || sorted.last().is_some_and(|&row| row as usize >= xv.rows)
        {
            return Err(invalid(
                "grammar masks need one full bitset per distinct row inside the batch",
            ));
        }
        let count = rows.len() * xv.cols;
        let args = MaskArgs {
            rows: to_u32(rows.len(), "masked rows")?,
            cols: to_u32(xv.cols, "columns")?,
            stride: to_u32(xv.row_stride, "row stride")?,
            words: to_u32(words, "mask words")?,
        };
        to_u32(count, "masked elements")?;
        let rows = self.runtime.indices(rows)?;
        let bits = self.runtime.indices(bits)?;
        let rv = self.runtime.view(&rows, DType::U32)?;
        let bv = self.runtime.view(&bits, DType::U32)?;
        let pipeline = self.pipeline("mask_f32")?;
        let threads = pipeline.threads(self.tuning.elementwise_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_mask",
            count.div_ceil(threads),
            threads,
            |e| {
                xv.bind_mut(e, 0)?;
                rv.bind(e, 1)?;
                bv.bind(e, 2)?;
                e.set_bytes(3, &args);
                Ok(())
            },
        )
    }

    /// Penalize rows of `logits` (`[rows, vocab]`) in place: `params[r]` applies to the
    /// `(row, entry)` pairs of row `r`, which must name distinct tokens. Same arithmetic as
    /// [`crate::sampling::penalize`].
    pub fn penalize(
        &self,
        logits: &Tensor,
        params: &[PenaltyParams],
        entries: &[(u32, PenaltyEntry)],
    ) -> Result<()> {
        let xv = self.view(logits)?;
        if params.len() != xv.rows {
            return Err(invalid("one penalty parameter set per logit row"));
        }
        if entries.is_empty() {
            return Ok(());
        }
        params.iter().try_for_each(PenaltyParams::validate)?;
        let mut keys: Vec<(u32, u32)> = entries.iter().map(|(r, e)| (*r, e.token)).collect();
        keys.sort_unstable();
        if keys.windows(2).any(|pair| pair[0] == pair[1])
            || keys
                .iter()
                .any(|&(row, token)| row as usize >= xv.rows || token as usize >= xv.cols)
        {
            return Err(invalid(
                "penalty entries must be distinct tokens inside their rows",
            ));
        }
        let words: Vec<u32> = entries
            .iter()
            .flat_map(|(row, e)| [*row, e.token, e.count])
            .collect();
        let factors: Vec<f32> = params
            .iter()
            .flat_map(|p| [p.repetition, p.presence, p.frequency])
            .collect();
        let words = self.runtime.indices(&words)?;
        let factors = Tensor::from_slice(&factors, factors.len(), self.device())?;
        let wv = self.runtime.view(&words, DType::U32)?;
        let fv = self.view(&factors)?;
        let args = PenaltyArgs {
            entries: to_u32(entries.len(), "penalty entries")?,
            stride: to_u32(xv.row_stride, "row stride")?,
        };
        let pipeline = self.pipeline("penalize_f32")?;
        let threads = pipeline.threads(self.tuning.elementwise_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_penalize",
            entries.len().div_ceil(threads),
            threads,
            |e| {
                xv.bind_mut(e, 0)?;
                wv.bind(e, 1)?;
                fv.bind(e, 2)?;
                e.set_bytes(3, &args);
                Ok(())
            },
        )
    }

    /// Sample one token per row of `logits` (`[rows, vocab]`, penalties applied) with
    /// [`crate::sampling`]'s algorithm. Returns F32 `[rows, 2]` of (token, status word; see
    /// [`crate::sampling::SampleStatus::from_word`]).
    pub fn sample(&self, logits: &Tensor, params: &[SampleParams]) -> Result<Tensor> {
        let xv = self.view(logits)?;
        // Token ids travel as F32, exact below 2^24.
        if xv.rows == 0 || xv.cols == 0 || xv.cols > 1 << 24 || params.len() != xv.rows {
            return Err(invalid(
                "sampling needs 1..=2^24 columns and one parameter set per row",
            ));
        }
        params.iter().try_for_each(SampleParams::validate)?;
        let words: Vec<u32> = params.iter().flat_map(SampleParams::words).collect();
        let words = self.runtime.indices(&words)?;
        let pv = self.runtime.view(&words, DType::U32)?;
        let args = SampleArgs {
            rows: to_u32(xv.rows, "rows")?,
            cols: to_u32(xv.cols, "columns")?,
            stride: to_u32(xv.row_stride, "row stride")?,
            rounds: MAX_ROUNDS,
        };
        let pipeline = self.pipeline("sample_f32")?;
        let out = self.output(xv.rows, 2)?;
        // One threadgroup per row: the widest group shortens each row's passes.
        let threads = pipeline.threads(self.capabilities().max_threads_per_threadgroup);
        self.runtime
            .dispatch(pipeline, "tur_sample", xv.rows, threads, |e| {
                xv.bind(e, 0)?;
                pv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &args);
                Ok(())
            })?;
        self.finish(out, (xv.rows, 2))
    }

    /// `x * rsqrt(mean(x²) + eps) * weight` per row; returns `[rows, width]`.
    pub fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let xv = self.view(x)?;
        let wv = self.view(weight)?;
        wv.expect(1, xv.cols, "norm weight")?;
        let params = RmsParams {
            rows: to_u32(xv.rows, "rows")?,
            width: to_u32(xv.cols, "width")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
            r_stride: 0,
            eps: eps as f32,
        };
        let pipeline = self.pipeline("rms_norm_f32")?;
        let out = self.output(xv.rows, xv.cols)?;
        let threads = pipeline.threads(self.tuning.row_threads);
        self.runtime
            .dispatch(pipeline, "tur_rms_norm", xv.rows, threads, |e| {
                xv.bind(e, 0)?;
                wv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &params);
                Ok(())
            })?;
        let shape = (xv.rows, xv.cols);
        self.finish(out, shape)
    }

    /// Residual addition fused with the following norm: returns `(x + r, rms(x + r) * w)`.
    pub fn add_rms_norm(
        &self,
        x: &Tensor,
        r: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let xv = self.view(x)?;
        let rv = self.view(r)?;
        let wv = self.view(weight)?;
        rv.expect(xv.rows, xv.cols, "residual")?;
        wv.expect(1, xv.cols, "norm weight")?;
        let params = RmsParams {
            rows: to_u32(xv.rows, "rows")?,
            width: to_u32(xv.cols, "width")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
            r_stride: to_u32(rv.row_stride, "residual stride")?,
            eps: eps as f32,
        };
        let pipeline = self.pipeline("add_rms_norm_f32")?;
        let sum = self.output(xv.rows, xv.cols)?;
        let normed = self.output(xv.rows, xv.cols)?;
        let threads = pipeline.threads(self.tuning.row_threads);
        self.runtime
            .dispatch(pipeline, "tur_add_rms_norm", xv.rows, threads, |e| {
                xv.bind(e, 0)?;
                rv.bind(e, 1)?;
                wv.bind(e, 2)?;
                sum.bind(e, 3);
                normed.bind(e, 4);
                e.set_bytes(5, &params);
                Ok(())
            })?;
        let shape = (xv.rows, xv.cols);
        Ok((self.finish(sum, shape)?, self.finish(normed, shape)?))
    }

    /// Rotary inverse frequencies `1 / theta^(2i / rotary_dim)` in F32, computed on the
    /// host exactly as the reference does; upload once per model.
    pub fn rotary_frequencies(&self, rotary_dim: usize, theta: f64) -> Result<Tensor> {
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            return Err(invalid("rotary dimension must be even and non-zero"));
        }
        let values: Vec<f32> = (0..rotary_dim / 2)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        Ok(Tensor::from_vec(values, rotary_dim / 2, self.device())?)
    }

    /// Per-head `RMSNorm` then split-half rotary over the first `2 * inv_freq.len()`
    /// dimensions at angle `position * inv_freq[i]`. `x` is `[tokens, width]`; returns
    /// contiguous `[tokens, heads, dim]`.
    pub fn qk_norm_rope(
        &self,
        x: &Tensor,
        heads: Heads,
        weight: &Tensor,
        positions: &Tensor,
        inv_freq: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let wv = self.view(weight)?;
        let pv = self.runtime.view(positions, DType::U32)?;
        let fv = self.view(inv_freq)?;
        wv.expect(1, heads.dim, "head norm weight")?;
        pv.expect(1, xv.rows, "positions")?;
        let half_rotary = fv.cols;
        let last = heads
            .count
            .checked_sub(1)
            .and_then(|h| h.checked_mul(heads.head_stride))
            .and_then(|o| o.checked_add(heads.col_offset))
            .and_then(|o| o.checked_add(heads.dim))
            .ok_or_else(|| invalid("invalid head geometry"))?;
        if fv.rows != 1 || last > xv.cols || heads.dim == 0 || 2 * half_rotary > heads.dim {
            return Err(invalid("heads or rotary dimensions exceed the projection"));
        }
        let params = RopeParams {
            tokens: to_u32(xv.rows, "tokens")?,
            heads: to_u32(heads.count, "heads")?,
            dim: to_u32(heads.dim, "head dim")?,
            row_stride: to_u32(xv.row_stride, "row stride")?,
            head_stride: to_u32(heads.head_stride, "head stride")?,
            col_offset: to_u32(heads.col_offset, "column offset")?,
            half_rotary: to_u32(half_rotary, "rotary dim")?,
            eps: eps as f32,
        };
        let items = xv
            .rows
            .checked_mul(heads.count)
            .ok_or_else(|| invalid("head count overflow"))?;
        let pipeline = self.pipeline("qk_norm_rope_f32")?;
        let out = self.output(items, heads.dim)?;
        let (groups, threads) = self.simdgroup_launch(pipeline, items);
        self.runtime
            .dispatch(pipeline, "tur_qk_norm_rope", groups, threads, |e| {
                xv.bind(e, 0)?;
                wv.bind(e, 1)?;
                pv.bind(e, 2)?;
                fv.bind(e, 3)?;
                out.bind(e, 4);
                e.set_bytes(5, &params);
                Ok(())
            })?;
        let shape = (xv.rows, heads.count, heads.dim);
        self.finish(out, shape)
    }

    /// `silu(gate) * up` over a fused contiguous `[rows, 2 * width]` gate|up projection.
    pub fn swiglu(&self, gate_up: &Tensor) -> Result<Tensor> {
        let gv = self.view(gate_up)?;
        gv.contiguous("gate|up projection")?;
        if !gv.cols.is_multiple_of(2) {
            return Err(invalid("gate|up width must be even"));
        }
        let width = gv.cols / 2;
        let params = GluParams {
            rows: to_u32(gv.rows, "rows")?,
            width: to_u32(width, "width")?,
        };
        let count = gv.rows * width;
        to_u32(count, "elements")?;
        let pipeline = self.pipeline("swiglu_f32")?;
        let out = self.output(gv.rows, width)?;
        let threads = pipeline.threads(self.tuning.elementwise_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_swiglu",
            count.div_ceil(threads),
            threads,
            |e| {
                gv.bind(e, 0)?;
                out.bind(e, 1);
                e.set_bytes(2, &params);
                Ok(())
            },
        )?;
        self.finish(out, (gv.rows, width))
    }

    /// `rms(core) * weight * silu(z)` for contiguous `[tokens * heads, dim]` core rows; `z`
    /// head `h` of token `t` is read in place at column `z_col + h * dim` of `z` row `t`.
    /// Returns `[tokens, heads * dim]`.
    pub fn gated_rms(
        &self,
        core: &Tensor,
        z: &Tensor,
        z_col: usize,
        heads: usize,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        let cv = self.view(core)?;
        let zv = self.view(z)?;
        let wv = self.view(weight)?;
        cv.contiguous("delta core")?;
        wv.expect(1, cv.cols, "gated norm weight")?;
        if heads == 0
            || zv.rows.checked_mul(heads) != Some(cv.rows)
            || z_col + heads * cv.cols > zv.cols
        {
            return Err(invalid("gated norm rows or z columns do not match"));
        }
        let params = GatedParams {
            rows: to_u32(cv.rows, "rows")?,
            dim: to_u32(cv.cols, "dim")?,
            heads: to_u32(heads, "heads")?,
            z_stride: to_u32(zv.row_stride, "z stride")?,
            z_col: to_u32(z_col, "z column")?,
            eps: eps as f32,
        };
        let pipeline = self.pipeline("gated_rms_f32")?;
        let out = self.output(cv.rows, cv.cols)?;
        let (groups, threads) = self.simdgroup_launch(pipeline, cv.rows);
        self.runtime
            .dispatch(pipeline, "tur_gated_rms", groups, threads, |e| {
                cv.bind(e, 0)?;
                zv.bind(e, 1)?;
                wv.bind(e, 2)?;
                out.bind(e, 3);
                e.set_bytes(4, &params);
                Ok(())
            })?;
        self.finish(out, (zv.rows, heads * cv.cols))
    }

    /// Validate and upload the recurrent addressing of a packed pass. Every sequence has
    /// at least one token and writes one slot or one per token; no slot is written twice
    /// or both read and written in the pass. Forks may share a committed slot.
    pub fn recurrent_spans(&self, spans: &[RecurrentSpan<'_>]) -> Result<RecurrentSpans> {
        if spans.is_empty() {
            return Err(invalid("a pass has at least one recurrent sequence"));
        }
        let mut table = Vec::with_capacity(spans.len() * 6);
        let mut rows = Vec::new();
        let mut slots = Vec::new();
        for (index, span) in spans.iter().enumerate() {
            if span.tokens == 0 || (span.writes.len() != 1 && span.writes.len() != span.tokens) {
                return Err(invalid(
                    "a recurrent sequence writes one slot or one per token",
                ));
            }
            table.extend([
                to_u32(rows.len(), "packed row")?,
                to_u32(span.tokens, "tokens")?,
                span.read,
                to_u32(slots.len(), "slot offset")?,
                u32::from(span.writes.len() > 1),
                to_u32(span.position, "position")?,
            ]);
            rows.resize(rows.len() + span.tokens, to_u32(index, "sequence")?);
            slots.extend_from_slice(span.writes);
        }
        let mut written = slots.clone();
        written.sort_unstable();
        if written.windows(2).any(|pair| pair[0] == pair[1])
            || spans
                .iter()
                .any(|span| written.binary_search(&span.read).is_ok())
        {
            return Err(invalid(
                "recurrent slots must be written once and distinct from every committed slot",
            ));
        }
        let max_slot = spans
            .iter()
            .map(|span| span.read)
            .chain(written.last().copied())
            .max()
            .unwrap_or(0);
        let all: Vec<u32> = (0..to_u32(spans.len(), "sequences")?).collect();
        let (single, multi): (Vec<u32>, Vec<u32>) = all
            .iter()
            .partition(|&&i| spans.get(i as usize).is_some_and(|span| span.tokens == 1));
        let counts = [single.len(), multi.len(), all.len()];
        // Uploads are never empty; an unused order list is a placeholder of one entry.
        let placeholder = |ids: &[u32]| {
            if ids.is_empty() {
                vec![0]
            } else {
                ids.to_vec()
            }
        };
        let [table_view, rows_view, slots_view, single, multi, all] =
            self.runtime.indices_packed([
                &table,
                &rows,
                &slots,
                &placeholder(&single),
                &placeholder(&multi),
                &placeholder(&all),
            ])?;
        Ok(RecurrentSpans {
            spans: table_view,
            tokens: rows.len(),
            rows: rows_view,
            slots: slots_view,
            single: (single, counts[0]),
            multi: (multi, counts[1]),
            all: (all, counts[2]),
            sequences: spans.len(),
            max_slot,
        })
    }

    /// Causal depthwise convolution of every sequence of a packed pass: `x` rows (first
    /// `channels` columns) follow each sequence's history in its committed slot of `pool`
    /// (`[slots, taps - 1, channels]`), with `weight` `[channels, taps]`, then `SiLU`.
    /// Returns `[rows, channels]` and writes each sequence's history after its last token,
    /// or after every token, into its written slots. One dispatch.
    pub fn delta_conv(
        &self,
        x: &Tensor,
        spans: &RecurrentSpans,
        pool: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let hv = self.view(pool)?;
        let wv = self.view(weight)?;
        let (_, kept, channels) = pool.dims3()?;
        let taps = wv.cols;
        if taps < 2 || kept != taps - 1 {
            return Err(invalid("convolution needs at least two taps"));
        }
        hv.contiguous("convolution history pool")?;
        spans.check(hv.rows, xv.rows)?;
        wv.expect(channels, taps, "convolution weight")?;
        wv.contiguous("convolution weight")?;
        if xv.cols < channels {
            return Err(invalid("convolution input is narrower than its channels"));
        }
        let params = ConvParams {
            rows: to_u32(xv.rows, "rows")?,
            channels: to_u32(channels, "channels")?,
            taps: to_u32(taps, "taps")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
        };
        let count = xv.rows * channels;
        to_u32(count, "elements")?;
        let table = self.runtime.view(&spans.spans, DType::U32)?;
        let rows = self.runtime.view(&spans.rows, DType::U32)?;
        let slots = self.runtime.view(&spans.slots, DType::U32)?;
        let out = self.output(xv.rows, channels)?;
        // Four channels per thread when every row of the input and history stays 16-byte
        // aligned.
        let vector = channels.is_multiple_of(4)
            && xv.row_stride.is_multiple_of(4)
            && xv.offset().is_multiple_of(16)
            && hv.offset().is_multiple_of(16);
        let (pipeline, items) = if vector {
            (self.pipeline("delta_conv4_f32")?, count / 4)
        } else {
            (self.pipeline("delta_conv_f32")?, count)
        };
        let threads = pipeline.threads(self.tuning.elementwise_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_delta_conv",
            items.div_ceil(threads),
            threads,
            |e| {
                xv.bind(e, 0)?;
                hv.bind_mut(e, 1)?;
                wv.bind(e, 2)?;
                out.bind(e, 3);
                e.set_bytes(4, &params);
                table.bind(e, 5)?;
                rows.bind(e, 6)?;
                slots.bind(e, 7)?;
                Ok(())
            },
        )?;
        self.finish(out, (xv.rows, channels))
    }

    /// Gated delta rule over every sequence of a packed pass: `mixed` holds the convolved
    /// `q | k | v` rows, `proj` the same rows' `b`/`a` gates. Each sequence starts from
    /// its committed slot of `pool` (`[slots, heads, key_dim, value_dim]`, F32, F16 or
    /// BF16; updates accumulate in F32) and writes its state after the last token, or
    /// after every token, into its written slots, rounded to the pool's dtype. Returns
    /// `[rows * heads, value_dim]`. Single-token sequences take one coalesced step dispatch
    /// (unless tuning disables it) and the rest one scan dispatch.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per kernel binding; a struct would only rename them"
    )]
    pub fn delta_scan(
        &self,
        mixed: &Tensor,
        proj: &Tensor,
        spans: &RecurrentSpans,
        shape: &DeltaShape,
        dt_bias: &Tensor,
        neg_a: &Tensor,
        pool: &Tensor,
    ) -> Result<Tensor> {
        let mv = self.view(mixed)?;
        let pv = self.view(proj)?;
        let bv = self.view(dt_bias)?;
        let av = self.view(neg_a)?;
        let storage = match pool.dtype() {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            other => return Err(invalid(format!("no delta kernels for {other:?} state"))),
        };
        let sv = self.runtime.view(pool, pool.dtype())?;
        let heads = shape.value_heads;
        let (dk, dv) = (shape.key_dim, shape.value_dim);
        if shape.key_heads == 0 || !heads.is_multiple_of(shape.key_heads) {
            return Err(invalid("value heads must be a multiple of key heads"));
        }
        if !matches!(dk, 64 | 128 | 256) {
            return Err(invalid(format!("no delta scan kernel for key dim {dk}")));
        }
        // The scan reads q and k rows of `mixed` as float4s, aligned when every row is.
        let aligned = mv.row_stride.is_multiple_of(4) && mv.offset().is_multiple_of(16);
        let name = if aligned {
            format!("delta_scan_{storage}")
        } else {
            format!("delta_scan_unaligned_{storage}")
        };
        let prep_stride = shape.prepared_width();
        bv.expect(1, heads, "dt bias")?;
        av.expect(1, heads, "decay rate")?;
        if pool.dims().get(1..) != Some(&[heads, dk, dv][..]) {
            return Err(invalid("recurrent pool does not match the delta geometry"));
        }
        sv.contiguous("recurrent pool")?;
        spans.check(sv.rows, mv.rows)?;
        if mv.cols < shape.channels()
            || shape.b_col + heads > pv.cols
            || shape.a_col + heads > pv.cols
            || pv.rows != mv.rows
        {
            return Err(invalid("delta scan inputs do not match the delta geometry"));
        }
        // Without the step kernel every sequence goes through the scan.
        let (stepped, scanned) = if self.tuning.delta_step {
            (&spans.single, &spans.multi)
        } else {
            (&spans.single, &spans.all)
        };
        let stepped_count = if self.tuning.delta_step { stepped.1 } else { 0 };
        let params = |sequences: usize| -> Result<ScanParams> {
            Ok(ScanParams {
                sequences: to_u32(sequences, "sequences")?,
                heads: to_u32(heads, "heads")?,
                groups: to_u32(heads / shape.key_heads, "head groups")?,
                key_width: to_u32(shape.key_width(), "key width")?,
                key_dim: to_u32(dk, "key dim")?,
                value_dim: to_u32(dv, "value dim")?,
                mixed_stride: to_u32(mv.row_stride, "mixed stride")?,
                proj_stride: to_u32(pv.row_stride, "projection stride")?,
                b_col: to_u32(shape.b_col, "b column")?,
                a_col: to_u32(shape.a_col, "a column")?,
                prep_stride: to_u32(prep_stride, "prepared stride")?,
            })
        };
        let table = self.runtime.view(&spans.spans, DType::U32)?;
        let slots = self.runtime.view(&spans.slots, DType::U32)?;
        let out = self.output(mv.rows * heads, dv)?;
        let bind = |e: &candle_metal_kernels::metal::ComputeCommandEncoder,
                    params: &ScanParams,
                    order: &View<'_>|
         -> Result<()> {
            mv.bind(e, 0)?;
            pv.bind(e, 1)?;
            bv.bind(e, 2)?;
            av.bind(e, 3)?;
            sv.bind_mut(e, 4)?;
            out.bind(e, 5);
            e.set_bytes(6, params);
            table.bind(e, 7)?;
            slots.bind(e, 8)?;
            order.bind(e, 9)?;
            Ok(())
        };
        if stepped_count > 0 {
            // One simdgroup per `rows` key rows; each stages 32 float2 partials.
            let (pipeline, threads) = DELTA_STEP_ROWS
                .iter()
                .map(|rows| -> Result<_> {
                    let pipeline = self.pipeline(&format!("delta_step_d{dk}_r{rows}_{storage}"))?;
                    Ok((pipeline, dk / rows * 32))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .find(|(pipeline, threads)| pipeline.threads(*threads) == *threads)
                .ok_or_else(|| invalid("no delta step pipeline fits a threadgroup"))?;
            if !dv.is_multiple_of(2) {
                return Err(invalid("the delta step kernel needs an even value dim"));
            }
            let groups = stepped_count
                .checked_mul(heads * dv.div_ceil(64))
                .ok_or_else(|| invalid("delta step grid overflow"))?;
            to_u32(groups, "delta step threadgroups")?;
            let order = self.runtime.view(&stepped.0, DType::U32)?;
            let params = params(stepped_count)?;
            self.runtime
                .dispatch(pipeline, "tur_delta_step", groups, threads, |e| {
                    bind(e, &params, &order)?;
                    e.set_threadgroup_memory_length(0, threads * 8);
                    Ok(())
                })?;
        }
        if scanned.1 > 0 {
            let params = params(scanned.1)?;
            let rows = to_u32(mv.rows, "rows")?;
            let prep = self.output(mv.rows, prep_stride)?;
            let pipeline = self.pipeline("delta_prepare_f32")?;
            let (groups, threads) = self.simdgroup_launch(pipeline, mv.rows * shape.key_heads);
            self.runtime
                .dispatch(pipeline, "tur_delta_prepare", groups, threads, |e| {
                    mv.bind(e, 0)?;
                    pv.bind(e, 1)?;
                    bv.bind(e, 2)?;
                    av.bind(e, 3)?;
                    prep.bind(e, 4);
                    e.set_bytes(5, &params);
                    e.set_bytes(6, &rows);
                    Ok(())
                })?;
            let pipeline = self.pipeline(&name)?;
            // dk / DELTA_SCAN_ROWS lanes share a value column.
            let columns = 32 * DELTA_SCAN_ROWS / dk;
            let items = scanned
                .1
                .checked_mul(heads * dv.div_ceil(columns))
                .ok_or_else(|| invalid("delta scan grid overflow"))?;
            let simdgroups = pipeline.simdgroups(self.tuning.scan_simdgroups).max(1);
            let groups = items.div_ceil(simdgroups);
            to_u32(groups, "delta scan threadgroups")?;
            let order = self.runtime.view(&scanned.0, DType::U32)?;
            self.runtime
                .dispatch(pipeline, "tur_delta_scan", groups, simdgroups * 32, |e| {
                    mv.bind(e, 0)?;
                    e.set_input_buffer(1, Some(prep.buffer()), 0);
                    sv.bind_mut(e, 2)?;
                    out.bind(e, 3);
                    e.set_bytes(4, &params);
                    table.bind(e, 5)?;
                    slots.bind(e, 6)?;
                    order.bind(e, 7)?;
                    Ok(())
                })?;
        }
        self.finish(out, (mv.rows * heads, dv))
    }

    /// Validate and upload paged addressing for `slots` pool rows of `page` tokens.
    /// Every table entry must address a whole page inside the pool, and every token's
    /// visible keys must be covered by table entries.
    pub fn paged_kv(
        &self,
        page: usize,
        slots: usize,
        table: &[u32],
        first_block: &[u32],
        visible: &[u32],
    ) -> Result<PagedKv> {
        if page == 0 || first_block.len() != visible.len() || visible.is_empty() {
            return Err(invalid("invalid paged K/V layout"));
        }
        let pages = slots / page;
        if table.iter().any(|&entry| entry as usize >= pages) {
            return Err(invalid("page table entry outside the K/V pool"));
        }
        let mut max_visible = 0;
        for (&first, &keys) in first_block.iter().zip(visible) {
            let blocks = (keys as usize).div_ceil(page);
            if keys == 0 || first as usize + blocks > table.len() {
                return Err(invalid("token keys exceed its page table"));
            }
            max_visible = max_visible.max(keys as usize);
        }
        let (table_view, first_view, visible_view, tiles) =
            if let Some(tiles) = query_tiles(first_block, visible) {
                let [table, first, keys, uploaded] =
                    self.runtime
                        .indices_packed([table, first_block, visible, &tiles])?;
                (table, first, keys, Some((uploaded, tiles.len() / 2)))
            } else {
                let [table, first, keys] =
                    self.runtime.indices_packed([table, first_block, visible])?;
                (table, first, keys, None)
            };
        Ok(PagedKv {
            table: table_view,
            first_block: first_view,
            visible: visible_view,
            tiles,
            page,
            slots,
            tokens: visible.len(),
            max_visible,
        })
    }

    /// Split keys across threadgroups only when tokens × heads alone cannot fill the GPU.
    pub fn attention_split(
        &self,
        tokens: usize,
        heads: usize,
        max_visible: usize,
    ) -> AttentionSplit {
        let items = tokens.saturating_mul(heads).max(1);
        let min_keys = self.tuning.min_split_keys;
        let wanted = self.tuning.attention_parallelism.div_ceil(items);
        let splits = wanted.min(max_visible.div_ceil(min_keys)).max(1);
        let split_keys = max_visible.div_ceil(splits).max(1);
        AttentionSplit {
            splits: max_visible.div_ceil(split_keys).max(1),
            split_keys,
        }
    }

    /// Threadgroup memory of the tiled kernel's staging buffer and physical slots.
    fn tiled_memory_bytes() -> usize {
        (TILE_KEYS * TILE_DIMS + TILE_KEYS) * 4
    }

    /// Whether a pass with `kv` addressing and `heads` query heads runs the tiled kernel.
    pub fn attention_tiled(&self, kv: &PagedKv, heads: usize, dim: usize) -> bool {
        kv.tiles().is_some_and(|tiles| {
            tiles.saturating_mul(heads) >= self.tuning.attention_tiled_groups
                // Sparse query tiles waste matrix work and lose split-K parallelism.
                && (self.tuning.attention_tiled_groups == 0
                    || kv.tokens >= tiles.saturating_mul(TILE_QUERIES / 2))
                && matches!(dim, 64 | 128 | 256)
                && self.capabilities().max_threads_per_threadgroup >= TILE_THREADS
                && Self::tiled_memory_bytes() <= self.capabilities().threadgroup_memory_bytes
        })
    }

    /// Query heads that one per-query attention simdgroup serves over a `format` pool: the
    /// tuned count when it divides the query heads per K/V head, else 1.
    pub fn attention_head_group(&self, heads: usize, kv_heads: usize, format: KvFormat) -> usize {
        let group = heads.checked_div(kv_heads).unwrap_or(1);
        let tuned = match format {
            KvFormat::Q8 { .. } => self.tuning.attention_q8_head_group,
            KvFormat::F32 | KvFormat::F16 => self.tuning.attention_head_group,
        };
        if group.is_multiple_of(tuned) {
            tuned
        } else {
            1
        }
    }

    /// Scratch bytes of one attention dispatch over a `format` pool (split partials and
    /// statistics).
    pub fn attention_scratch_bytes(
        &self,
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        max_visible: usize,
        format: KvFormat,
    ) -> usize {
        let hg = self.attention_head_group(heads, kv_heads, format);
        let split = self.attention_split(tokens, heads / hg, max_visible);
        if split.splits == 1 {
            return 0;
        }
        tokens
            .saturating_mul(heads)
            .saturating_mul(split.splits)
            .saturating_mul(dim + 2)
            .saturating_mul(4)
    }

    /// Validate a K/V pool in `format` against `slots` rows of `kv_heads × dim`.
    fn kv_views<'a>(
        &self,
        pool: &'a KvPool,
        format: KvFormat,
        slots: usize,
        kv_heads: usize,
        dim: usize,
    ) -> Result<(View<'a>, Option<View<'a>>)> {
        if pool.data.dims() != [slots, kv_heads, dim] {
            return Err(invalid("K/V pool does not match its page addressing"));
        }
        let data = self.runtime.view(&pool.data, format.data_dtype())?;
        data.contiguous("K/V pool")?;
        let scales = match (format.group(), &pool.scales) {
            (None, None) => None,
            (Some(group), Some(scales)) => {
                if scales.dims() != [slots, kv_heads, dim / group] {
                    return Err(invalid("K/V scales do not match the pool"));
                }
                let view = self.runtime.view(scales, DType::F16)?;
                view.contiguous("K/V scales")?;
                Some(view)
            }
            _ => return Err(invalid("K/V scales do not match the format")),
        };
        Ok((data, scales))
    }

    fn kv_kernel(prefix: &str, dim: usize, format: KvFormat) -> Result<String> {
        let registers = match dim {
            64 => 2,
            128 => 4,
            256 => 8,
            _ => return Err(invalid(format!("no {prefix} kernel for head dim {dim}"))),
        };
        let suffix = match format {
            KvFormat::F32 => "f32",
            KvFormat::F16 => "f16",
            KvFormat::Q8 { .. } => "q8",
        };
        Ok(format!("{prefix}_r{registers}_{suffix}"))
    }

    /// Store new keys and values into `pool`, quantizing to its format. Token `t` of the
    /// pass writes logical position `visible[t] - 1` of its page table in `kv`. Key head
    /// `h` of token `t` is read at column `key_heads.col_offset + h * head_stride` of `key`
    /// row `t`; values likewise from `value`.
    pub fn kv_write(
        &self,
        key: &Tensor,
        key_heads: Heads,
        value: &Tensor,
        value_heads: Heads,
        pool: &AttentionPool,
        kv: &PagedKv,
    ) -> Result<()> {
        let (kv_heads, dim) = (pool.kv_heads, pool.head_dim);
        if kv_heads == 0 {
            return Err(invalid("attention pool has no K/V heads"));
        }
        let kv_view = self.view(key)?;
        let vv = self.view(value)?;
        for (view, heads) in [(&kv_view, key_heads), (&vv, value_heads)] {
            let last = heads.col_offset + (kv_heads - 1) * heads.head_stride + dim;
            if heads.count != kv_heads
                || heads.dim != dim
                || view.rows != kv.tokens
                || last > view.cols
            {
                return Err(invalid("new K/V rows do not match the pool"));
            }
        }
        if kv.page != pool.page_tokens {
            return Err(invalid("page addressing does not match the pool"));
        }
        let (k_data, k_scales) = self.kv_views(&pool.key, pool.format, kv.slots, kv_heads, dim)?;
        let (v_data, v_scales) =
            self.kv_views(&pool.value, pool.format, kv.slots, kv_heads, dim)?;
        let table = self.runtime.view(&kv.table, DType::U32)?;
        let first = self.runtime.view(&kv.first_block, DType::U32)?;
        let visible = self.runtime.view(&kv.visible, DType::U32)?;
        let params = KvWriteParams {
            tokens: to_u32(kv.tokens, "tokens")?,
            kv_heads: to_u32(kv_heads, "kv heads")?,
            dim: to_u32(dim, "head dim")?,
            page: to_u32(kv.page, "page")?,
            group: to_u32(pool.format.group().unwrap_or(32), "group")?,
            k_stride: to_u32(kv_view.row_stride, "key stride")?,
            k_col: to_u32(key_heads.col_offset, "key column")?,
            k_head_stride: to_u32(key_heads.head_stride, "key head stride")?,
            v_stride: to_u32(vv.row_stride, "value stride")?,
            v_col: to_u32(value_heads.col_offset, "value column")?,
            v_head_stride: to_u32(value_heads.head_stride, "value head stride")?,
            inv_127: (1.0f64 / 127.0) as f32,
        };
        let pipeline = self.pipeline(&Self::kv_kernel("kv_write", dim, pool.format)?)?;
        let (groups, threads) = self.simdgroup_launch(pipeline, 2 * kv.tokens * kv_heads);
        self.runtime
            .dispatch(pipeline, "tur_kv_write", groups, threads, |e| {
                kv_view.bind(e, 0)?;
                vv.bind(e, 1)?;
                table.bind(e, 2)?;
                first.bind(e, 3)?;
                visible.bind(e, 4)?;
                k_data.bind_mut(e, 5)?;
                // Unscaled formats bind the data again; every declared buffer must be bound.
                k_scales.as_ref().unwrap_or(&k_data).bind_mut(e, 6)?;
                v_data.bind_mut(e, 7)?;
                v_scales.as_ref().unwrap_or(&v_data).bind_mut(e, 8)?;
                e.set_bytes(9, &params);
                Ok(())
            })
    }

    /// Causal grouped-query attention with online softmax over a paged pool in any K/V
    /// format, gated by `sigmoid(gate)`. `q` is `[tokens, heads, dim]`. Gate head `h` of
    /// token `t` is at column `gate.col_offset + h * gate.head_stride` of `gate_source`
    /// row `t`. Returns `[tokens, heads * dim]`.
    pub fn attention(
        &self,
        q: &Tensor,
        pool: &AttentionPool,
        kv: &PagedKv,
        gate_source: &Tensor,
        gate: Heads,
    ) -> Result<Tensor> {
        let (tokens, heads, dim) = q.dims3()?;
        let kv_heads = pool.kv_heads;
        if kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || pool.head_dim != dim
            || kv.page != pool.page_tokens
            || tokens != kv.tokens
            || gate.count != heads
            || gate.dim != dim
        {
            return Err(invalid("attention shapes do not match"));
        }
        let hg = self.attention_head_group(heads, kv_heads, pool.format);
        let name = if hg == 1 {
            Self::kv_kernel("attention", dim, pool.format)?
        } else {
            Self::kv_kernel(&format!("attention_h{hg}"), dim, pool.format)?
        };
        let reduce = Self::kv_kernel("attention_reduce", dim, KvFormat::F32)?;
        let (k_data, k_scales) = self.kv_views(&pool.key, pool.format, kv.slots, kv_heads, dim)?;
        let (v_data, v_scales) =
            self.kv_views(&pool.value, pool.format, kv.slots, kv_heads, dim)?;
        let qv = self.view(q)?;
        let gv = self.view(gate_source)?;
        qv.contiguous("queries")?;
        if gv.rows != tokens || gate.col_offset + (heads - 1) * gate.head_stride + dim > gv.cols {
            return Err(invalid("attention gate exceeds its projection"));
        }
        if self.attention_tiled(kv, heads, dim) {
            return self.attention_tiles(
                &qv,
                pool,
                kv,
                &gv,
                gate,
                (k_data, k_scales),
                (v_data, v_scales),
            );
        }
        // The per-query kernel addresses pool rows (slot × K/V head) in 32 bits.
        to_u32(
            kv.slots
                .checked_mul(kv_heads)
                .ok_or_else(|| invalid("K/V pool rows overflow"))?,
            "K/V pool rows",
        )?;
        let table = self.runtime.view(&kv.table, DType::U32)?;
        let first = self.runtime.view(&kv.first_block, DType::U32)?;
        let visible = self.runtime.view(&kv.visible, DType::U32)?;
        // Splits fill the GPU with simdgroups, each serving `hg` heads.
        let split = self.attention_split(tokens, heads / hg, kv.max_visible);
        let params = AttentionParams {
            tokens: to_u32(tokens, "tokens")?,
            heads: to_u32(heads, "heads")?,
            kv_heads: to_u32(kv_heads, "kv heads")?,
            dim: to_u32(dim, "head dim")?,
            page: to_u32(kv.page, "page")?,
            splits: to_u32(split.splits, "splits")?,
            split_keys: to_u32(split.split_keys, "split keys")?,
            gate_stride: to_u32(gv.row_stride, "gate stride")?,
            gate_col: to_u32(gate.col_offset, "gate column")?,
            gate_head_stride: to_u32(gate.head_stride, "gate head stride")?,
            group: to_u32(pool.format.group().unwrap_or(32), "group")?,
            scale: (dim as f64).sqrt().recip() as f32,
        };
        let items = tokens * heads;
        let out = self.output(items, dim)?;
        let scratch = if split.splits > 1 {
            Some((
                self.output(items * split.splits, dim)?,
                self.output(items * split.splits, 2)?,
            ))
        } else {
            None
        };
        let pipeline = self.pipeline(&name)?;
        let (groups, threads) = self.simdgroup_launch(pipeline, items / hg * split.splits);
        self.runtime
            .dispatch(pipeline, "tur_attention", groups, threads, |e| {
                qv.bind(e, 0)?;
                k_data.bind(e, 1)?;
                v_data.bind(e, 2)?;
                table.bind(e, 3)?;
                first.bind(e, 4)?;
                visible.bind(e, 5)?;
                gv.bind(e, 6)?;
                out.bind(e, 7);
                // Unused with one split, but every declared buffer must be bound.
                let (partial, stats) = scratch.as_ref().map_or((&out, &out), |(p, s)| (p, s));
                partial.bind(e, 8);
                stats.bind(e, 9);
                e.set_bytes(10, &params);
                k_scales.as_ref().unwrap_or(&k_data).bind(e, 11)?;
                v_scales.as_ref().unwrap_or(&v_data).bind(e, 12)?;
                Ok(())
            })?;
        if let Some((partial, stats)) = &scratch {
            let pipeline = self.pipeline(&reduce)?;
            let (groups, threads) = self.simdgroup_launch(pipeline, items);
            self.runtime
                .dispatch(pipeline, "tur_attention_reduce", groups, threads, |e| {
                    e.set_input_buffer(0, Some(partial.buffer()), 0);
                    e.set_input_buffer(1, Some(stats.buffer()), 0);
                    gv.bind(e, 2)?;
                    out.bind(e, 3);
                    e.set_bytes(4, &params);
                    Ok(())
                })?;
        }
        self.finish(out, (tokens, heads * dim))
    }

    /// The tiled matrix-unit kernel over `kv`'s query tiles; same contract as `attention`.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per kernel binding; a struct would only rename them"
    )]
    fn attention_tiles(
        &self,
        qv: &View<'_>,
        pool: &AttentionPool,
        kv: &PagedKv,
        gv: &View<'_>,
        gate: Heads,
        (k_data, k_scales): (View<'_>, Option<View<'_>>),
        (v_data, v_scales): (View<'_>, Option<View<'_>>),
    ) -> Result<Tensor> {
        let (tiles, count) = kv
            .tiles
            .as_ref()
            .ok_or_else(|| invalid("page addressing has no query tiles"))?;
        let (heads, dim) = (gate.count, pool.head_dim);
        let params = TiledParams {
            tiles: to_u32(*count, "tiles")?,
            heads: to_u32(heads, "heads")?,
            kv_heads: to_u32(pool.kv_heads, "kv heads")?,
            page: to_u32(kv.page, "page")?,
            group: to_u32(pool.format.group().unwrap_or(32), "group")?,
            gate_stride: to_u32(gv.row_stride, "gate stride")?,
            gate_col: to_u32(gate.col_offset, "gate column")?,
            gate_head_stride: to_u32(gate.head_stride, "gate head stride")?,
            scale: (dim as f64).sqrt().recip() as f32,
        };
        let pipeline = self.pipeline(&Self::kv_kernel("attention_tiled", dim, pool.format)?)?;
        if pipeline.threads(TILE_THREADS) != TILE_THREADS {
            return Err(invalid(
                "tiled attention pipeline cannot run a full query tile",
            ));
        }
        let tiles = self.runtime.view(tiles, DType::U32)?;
        let table = self.runtime.view(&kv.table, DType::U32)?;
        let first = self.runtime.view(&kv.first_block, DType::U32)?;
        let visible = self.runtime.view(&kv.visible, DType::U32)?;
        let out = self.output(kv.tokens * heads, dim)?;
        let groups = count
            .checked_mul(heads)
            .ok_or_else(|| invalid("tile count overflow"))?;
        self.runtime
            .dispatch(pipeline, "tur_attention_tiled", groups, TILE_THREADS, |e| {
                qv.bind(e, 0)?;
                k_data.bind(e, 1)?;
                v_data.bind(e, 2)?;
                table.bind(e, 3)?;
                first.bind(e, 4)?;
                visible.bind(e, 5)?;
                gv.bind(e, 6)?;
                out.bind(e, 7);
                tiles.bind(e, 8)?;
                e.set_bytes(9, &params);
                k_scales.as_ref().unwrap_or(&k_data).bind(e, 10)?;
                v_scales.as_ref().unwrap_or(&v_data).bind(e, 11)?;
                e.set_threadgroup_memory_length(0, Self::tiled_memory_bytes());
                Ok(())
            })?;
        self.finish(out, (kv.tokens, heads * dim))
    }

    /// Project `x` (`[rows, in]`) through `layer`. Packed `Q8_0/Q4K` weights with a row count
    /// in the tuned matrix range use the matrix-unit kernel, and in the GEMV range the
    /// small-batch GEMV; both read each weight once for all rows. Everything else uses
    /// Candle's dense, one-row GEMV or tiled packed kernels.
    pub fn linear(&self, layer: &PackedLinear, x: &Tensor) -> Result<Tensor> {
        let rows = x.dim(0)?;
        let range = |(min, max): (usize, usize)| (min..=max).contains(&rows);
        match layer.packed_buffer() {
            Some((buffer, format @ GgmlDType::Q8_0)) if range(self.tuning.qmm_q8_0_rows) => {
                self.qmm(layer, buffer, format, x)
            }
            Some((buffer, format @ GgmlDType::Q4K)) if range(self.tuning.qmm_q4k_rows) => {
                self.qmm(layer, buffer, format, x)
            }
            Some((buffer, format @ GgmlDType::Q8_0)) if range(self.tuning.gemv_q8_0_rows) => {
                self.gemv(layer, buffer, format, x)
            }
            Some((buffer, format @ GgmlDType::Q4K)) if range(self.tuning.gemv_q4k_rows) => {
                self.gemv(layer, buffer, format, x)
            }
            _ => layer.forward(x),
        }
    }

    /// Small-batch `Q8_0` or Q4K projection on the matrix units: up to 16 rows, each
    /// weight read once, F32 dequantization and accumulation.
    fn qmm(
        &self,
        layer: &PackedLinear,
        weights: &candle_metal_kernels::metal::Buffer,
        format: GgmlDType,
        x: &Tensor,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let (n, k) = (layer.out_features(), layer.in_features());
        let name = match format {
            GgmlDType::Q8_0 => "q8_0",
            GgmlDType::Q4K => "q4k",
            _ => return Err(invalid("small-batch projection supports Q8_0 and Q4K")),
        };
        if xv.cols != k || xv.rows == 0 || xv.rows > 16 {
            return Err(invalid("small-batch projection input does not match"));
        }
        let bytes = n
            .checked_mul(k / format.block_size())
            .and_then(|b| b.checked_mul(format.type_size()))
            .ok_or_else(|| invalid("packed weight size overflows"))?;
        if !k.is_multiple_of(format.block_size()) || weights.length() < bytes {
            return Err(invalid("packed weight buffer is smaller than its shape"));
        }
        if !xv.row_stride.is_multiple_of(4) || !xv.offset().is_multiple_of(16) {
            return layer.forward(x);
        }
        let tiles = xv.rows.div_ceil(8);
        let params = QmmParams {
            n: to_u32(n, "rows")?,
            k: to_u32(k, "columns")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
            y_stride: to_u32(n, "output stride")?,
            rows: to_u32(xv.rows, "batch rows")?,
        };
        let pipeline = self.pipeline(&format!("qmm_{name}_t{tiles}_f32"))?;
        // Simdgroups of a threadgroup split k over the same output columns.
        let threads = pipeline.simdgroups(self.tuning.simdgroups).max(1) * 32;
        let columns = qmm_columns(format);
        let groups = n.div_ceil(columns);
        let shared = threads / 32 * columns * tiles * 8 * 4;
        if shared > self.capabilities().threadgroup_memory_bytes {
            return Err(invalid(
                "small-batch projection partials exceed threadgroup memory",
            ));
        }
        let out = self.output(xv.rows, n)?;
        self.runtime
            .dispatch(pipeline, "tur_qmm", groups, threads, |e| {
                e.set_input_buffer(0, Some(weights), 0);
                xv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &params);
                e.set_threadgroup_memory_length(0, shared);
                Ok(())
            })?;
        self.finish(out, (xv.rows, n))
    }

    fn gemv(
        &self,
        layer: &PackedLinear,
        weights: &candle_metal_kernels::metal::Buffer,
        format: GgmlDType,
        x: &Tensor,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let (n, k) = (layer.out_features(), layer.in_features());
        if xv.cols != k || xv.rows == 0 || xv.rows > 8 {
            return Err(invalid("GEMV input does not match the projection"));
        }
        let blocks = k / format.block_size();
        let bytes = n
            .checked_mul(blocks)
            .and_then(|b| b.checked_mul(format.type_size()))
            .ok_or_else(|| invalid("packed weight size overflows"))?;
        if !k.is_multiple_of(format.block_size()) || weights.length() < bytes {
            return Err(invalid("packed weight buffer is smaller than its shape"));
        }
        // Both kernels read activations as float4.
        if !xv.row_stride.is_multiple_of(4) || !xv.offset().is_multiple_of(16) {
            return layer.forward(x);
        }
        let prefix = if format == GgmlDType::Q8_0 {
            "q8_0"
        } else {
            "q4k"
        };
        let name = format!("qmv_{prefix}_b{}_f32", xv.rows);
        let params = QmvParams {
            n: to_u32(n, "rows")?,
            k: to_u32(k, "columns")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
            y_stride: to_u32(n, "output stride")?,
        };
        let pipeline = self.pipeline(&name)?;
        let out = self.output(xv.rows, n)?;
        let (groups, threads) =
            self.simdgroup_launch(pipeline, n.div_ceil(gemv_rows(format, xv.rows)));
        self.runtime
            .dispatch(pipeline, "tur_gemv", groups, threads, |e| {
                e.set_input_buffer(0, Some(weights), 0);
                xv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &params);
                Ok(())
            })?;
        self.finish(out, (xv.rows, n))
    }
}

#[cfg(test)]
mod tests {
    use super::query_tiles;

    #[test]
    fn test_query_tiles_preserve_sequence_boundaries_and_tails() {
        let first = [vec![0; 33], vec![4; 7]].concat();
        let visible = [(1..=33).collect::<Vec<_>>(), (10..17).collect()].concat();
        assert_eq!(
            query_tiles(&first, &visible),
            Some(vec![0, 32, 32, 1, 33, 7])
        );
    }

    #[test]
    fn test_query_tiles_reject_gaps_at_tile_boundaries_and_overflow() {
        let mut visible: Vec<u32> = (1..=33).collect();
        visible[32] = 34;
        assert_eq!(query_tiles(&[0; 33], &visible), None);
        assert_eq!(query_tiles(&[0; 2], &[u32::MAX, 0]), None);
    }
}
