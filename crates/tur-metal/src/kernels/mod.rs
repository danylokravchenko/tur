#![forbid(unsafe_code)]
//! Fused Metal kernels for hybrid decoders.
//!
//! Activations are F32. Inputs are validated (dtype, device, strides, extents and index
//! ranges) before encoding; nothing falls back to the CPU. Kernels only enqueue work: the
//! caller fences with [`crate::completion::Completion`] before reading results.

mod runtime;

use crate::{
    packed::PackedLinear,
    pool::{AttentionPool, KvPool},
};
use candle_core::{DType, Device, Tensor, quantized::GgmlDType};
pub use runtime::{Capabilities, LAYOUT_VERSION, Output, Tuning};
use runtime::{Pipeline, Runtime, View, to_u32};
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{KvFormat, invalid},
};

/// Columns of value state owned by one delta-scan simdgroup (`COLS` in the MSL).
const SCAN_COLUMNS: usize = 4;
/// Output rows per GEMV simdgroup (`NR` in the MSL `QMV` instantiations).
fn gemv_rows(format: GgmlDType, batch: usize) -> usize {
    match format {
        GgmlDType::Q8_0 if batch <= 4 => 4,
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
    tokens: u32,
    channels: u32,
    taps: u32,
    x_stride: u32,
    every: u32,
}

#[repr(C)]
struct ScanParams {
    tokens: u32,
    heads: u32,
    groups: u32,
    key_width: u32,
    key_dim: u32,
    value_dim: u32,
    mixed_stride: u32,
    proj_stride: u32,
    b_col: u32,
    a_col: u32,
    every: u32,
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

/// Gated DeltaNet geometry. The projection row holds `[q | k | v | z | b | a]`, with the
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

/// Recurrent pool slots written by one sequence: one per retained boundary. Uploaded once
/// per pass and shared by every recurrent layer, whose pools have equal slot counts.
pub struct Slots {
    indices: Tensor,
    values: Vec<u32>,
}

impl Slots {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Check against a pool of `capacity` slots and the slot being read.
    fn check(&self, capacity: usize, read: u32, tokens: usize) -> Result<()> {
        if (self.len() != 1 && self.len() != tokens)
            || self
                .values
                .iter()
                .any(|&s| s as usize >= capacity || s == read)
            || read as usize >= capacity
        {
            return Err(invalid(
                "recurrent slots must be one or one per token, inside the pool and distinct from \
                 the committed slot",
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
            || tuning.min_split_keys == 0
            || [tuning.gemv_q8_0_rows, tuning.gemv_q4k_rows]
                .iter()
                .any(|&(min, max)| min == 0 || max > 8)
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

    /// Per-head RMSNorm then split-half rotary over the first `2 * inv_freq.len()`
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

    /// Upload the recurrent slots one sequence writes.
    pub fn slots(&self, values: &[u32]) -> Result<Slots> {
        if values.is_empty() {
            return Err(invalid("a pass writes at least one recurrent slot"));
        }
        Ok(Slots {
            indices: self.runtime.indices(values)?,
            values: values.to_vec(),
        })
    }

    /// Causal depthwise convolution of one sequence: rows `row..row + tokens` of `x`
    /// (first `channels` columns) after the history in slot `read` of `pool`
    /// (`[slots, taps - 1, channels]`), with `weight` `[channels, taps]`, then SiLU into
    /// `out` rows `out_row..`. Writes the history after the last token into slot
    /// `writes[0]`, or after token `t` into `writes[t]` when one slot per token is given.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_conv(
        &self,
        x: &Tensor,
        row: usize,
        tokens: usize,
        pool: &Tensor,
        read: u32,
        writes: &Slots,
        weight: &Tensor,
        out: &Output,
        out_row: usize,
    ) -> Result<()> {
        let xv = self.view(x)?;
        let hv = self.view(pool)?;
        let wv = self.view(weight)?;
        let (_, kept, channels) = pool.dims3()?;
        let taps = wv.cols;
        if taps < 2 || tokens == 0 || kept != taps - 1 {
            return Err(invalid("convolution needs at least two taps and one token"));
        }
        hv.contiguous("convolution history pool")?;
        writes.check(hv.rows, read, tokens)?;
        wv.expect(channels, taps, "convolution weight")?;
        wv.contiguous("convolution weight")?;
        if xv.cols < channels || out.cols() != channels || row + tokens > xv.rows {
            return Err(invalid(
                "convolution input or output does not match its channels",
            ));
        }
        let params = ConvParams {
            tokens: to_u32(tokens, "tokens")?,
            channels: to_u32(channels, "channels")?,
            taps: to_u32(taps, "taps")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
            every: u32::from(writes.len() > 1),
        };
        let count = tokens * channels;
        to_u32(count, "elements")?;
        let slots = self.runtime.view(&writes.indices, DType::U32)?;
        let pipeline = self.pipeline("delta_conv_f32")?;
        let threads = pipeline.threads(self.tuning.elementwise_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_delta_conv",
            count.div_ceil(threads),
            threads,
            |e| {
                xv.bind_row(e, 0, row)?;
                hv.bind_row(e, 1, read as usize)?;
                wv.bind(e, 2)?;
                out.bind_rows(e, 3, out_row, tokens)?;
                hv.bind_mut(e, 4)?;
                e.set_bytes(5, &params);
                slots.bind(e, 6)?;
                Ok(())
            },
        )
    }

    /// Gated delta rule over one sequence: `mixed` rows `row..row + tokens` hold the
    /// convolved `q | k | v`, `proj` the same rows' `b`/`a` gates. Starts from slot `read`
    /// of `pool` (`[slots, heads, key_dim, value_dim]`), writes `[tokens, heads, value_dim]`
    /// outputs into `out` rows `out_row..` and the state after the last token into slot
    /// `writes[0]`, or after token `t` into `writes[t]`.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_scan(
        &self,
        mixed: &Tensor,
        proj: &Tensor,
        row: usize,
        tokens: usize,
        shape: &DeltaShape,
        dt_bias: &Tensor,
        neg_a: &Tensor,
        pool: &Tensor,
        read: u32,
        writes: &Slots,
        out: &Output,
        out_row: usize,
    ) -> Result<()> {
        let mv = self.view(mixed)?;
        let pv = self.view(proj)?;
        let bv = self.view(dt_bias)?;
        let av = self.view(neg_a)?;
        let sv = self.view(pool)?;
        let heads = shape.value_heads;
        let (dk, dv) = (shape.key_dim, shape.value_dim);
        if shape.key_heads == 0 || !heads.is_multiple_of(shape.key_heads) || tokens == 0 {
            return Err(invalid("value heads must be a multiple of key heads"));
        }
        let name = match dk {
            64 => "delta_scan_r2_f32",
            128 => "delta_scan_r4_f32",
            256 => "delta_scan_r8_f32",
            _ => return Err(invalid(format!("no delta scan kernel for key dim {dk}"))),
        };
        bv.expect(1, heads, "dt bias")?;
        av.expect(1, heads, "decay rate")?;
        if pool.dims().get(1..) != Some(&[heads, dk, dv][..]) {
            return Err(invalid("recurrent pool does not match the delta geometry"));
        }
        sv.contiguous("recurrent pool")?;
        writes.check(sv.rows, read, tokens)?;
        if mv.cols < shape.channels()
            || shape.b_col + heads > pv.cols
            || shape.a_col + heads > pv.cols
            || row + tokens > mv.rows
            || row + tokens > pv.rows
            || out.cols() != heads * dv
        {
            return Err(invalid("delta scan inputs do not match the delta geometry"));
        }
        let params = ScanParams {
            tokens: to_u32(tokens, "tokens")?,
            heads: to_u32(heads, "heads")?,
            groups: to_u32(heads / shape.key_heads, "head groups")?,
            key_width: to_u32(shape.key_width(), "key width")?,
            key_dim: to_u32(dk, "key dim")?,
            value_dim: to_u32(dv, "value dim")?,
            mixed_stride: to_u32(mv.row_stride, "mixed stride")?,
            proj_stride: to_u32(pv.row_stride, "projection stride")?,
            b_col: to_u32(shape.b_col, "b column")?,
            a_col: to_u32(shape.a_col, "a column")?,
            every: u32::from(writes.len() > 1),
        };
        let slots = self.runtime.view(&writes.indices, DType::U32)?;
        let pipeline = self.pipeline(name)?;
        // Halve the threadgroup until its padded state tile fits threadgroup memory.
        let tile_bytes =
            |simdgroups: usize| (dk * (simdgroups * SCAN_COLUMNS + 1) * 4).next_multiple_of(16);
        let mut simdgroups = pipeline.simdgroups(self.tuning.scan_simdgroups).max(1);
        while simdgroups > 1
            && tile_bytes(simdgroups) > self.capabilities().threadgroup_memory_bytes
        {
            simdgroups /= 2;
        }
        if tile_bytes(simdgroups) > self.capabilities().threadgroup_memory_bytes {
            return Err(invalid("delta state tile exceeds threadgroup memory"));
        }
        let per_head = dv.div_ceil(SCAN_COLUMNS).div_ceil(simdgroups);
        self.runtime.dispatch(
            pipeline,
            "tur_delta_scan",
            heads * per_head,
            simdgroups * 32,
            |e| {
                mv.bind_row(e, 0, row)?;
                pv.bind_row(e, 1, row)?;
                bv.bind(e, 2)?;
                av.bind(e, 3)?;
                sv.bind_row(e, 4, read as usize)?;
                out.bind_rows(e, 5, out_row, tokens)?;
                sv.bind_mut(e, 6)?;
                e.set_bytes(7, &params);
                slots.bind(e, 8)?;
                e.set_threadgroup_memory_length(0, tile_bytes(simdgroups));
                Ok(())
            },
        )
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
        let tiles = query_tiles(first_block, visible)
            .map(|tiles| -> Result<(Tensor, usize)> {
                Ok((self.runtime.indices(&tiles)?, tiles.len() / 2))
            })
            .transpose()?;
        Ok(PagedKv {
            table: self.runtime.indices(table)?,
            first_block: self.runtime.indices(first_block)?,
            visible: self.runtime.indices(visible)?,
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
                && self.capabilities().max_threads_per_threadgroup >= 128
                && Self::tiled_memory_bytes() <= self.capabilities().threadgroup_memory_bytes
        })
    }

    /// Scratch bytes of one attention dispatch (split partials and statistics).
    pub fn attention_scratch_bytes(
        &self,
        tokens: usize,
        heads: usize,
        dim: usize,
        max_visible: usize,
    ) -> usize {
        let split = self.attention_split(tokens, heads, max_visible);
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
        let name = Self::kv_kernel("attention", dim, pool.format)?;
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
        let table = self.runtime.view(&kv.table, DType::U32)?;
        let first = self.runtime.view(&kv.first_block, DType::U32)?;
        let visible = self.runtime.view(&kv.visible, DType::U32)?;
        let split = self.attention_split(tokens, heads, kv.max_visible);
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
        let (groups, threads) = self.simdgroup_launch(pipeline, items * split.splits);
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
    #[allow(clippy::too_many_arguments)]
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
        if pipeline.threads(128) != 128 {
            return Err(invalid("tiled attention needs 128-thread threadgroups"));
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
            .dispatch(pipeline, "tur_attention_tiled", groups, 128, |e| {
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

    /// Project `x` (`[rows, in]`) through `layer`. Packed Q8_0/Q4K weights with a row count
    /// in the tuned GEMV range use the small-batch GEMV, which reads each weight once for
    /// all rows; everything else uses Candle's dense, one-row GEMV or tiled packed kernels.
    pub fn linear(&self, layer: &PackedLinear, x: &Tensor) -> Result<Tensor> {
        let rows = x.dim(0)?;
        let range = |(min, max): (usize, usize)| (min..=max).contains(&rows);
        match layer.packed_buffer() {
            Some((buffer, format @ GgmlDType::Q8_0)) if range(self.tuning.gemv_q8_0_rows) => {
                self.gemv(layer, buffer, format, x)
            }
            Some((buffer, format @ GgmlDType::Q4K)) if range(self.tuning.gemv_q4k_rows) => {
                self.gemv(layer, buffer, format, x)
            }
            _ => layer.forward(x),
        }
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
        // Q4K reads activations as float4.
        if format == GgmlDType::Q4K
            && (!xv.row_stride.is_multiple_of(4) || !xv.offset().is_multiple_of(16))
        {
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
