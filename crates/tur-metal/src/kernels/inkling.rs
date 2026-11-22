//! Inkling 2's kernels: attention residuals over block states, loop-pass message gates,
//! short causal convolutions whose history lives in recurrent slots (normalizing keys and
//! values per head), and paged attention that clips its queries and relative states as it
//! reads them, with a learned relative position bias, per-token sinks and a sigmoid gate.
use super::{
    Kernels, PagedKv, RecurrentSpans,
    runtime::{View, to_u32},
};
use crate::pool::AttentionPool;
use candle_core::{DType, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use tur_core::{
    Result,
    execution::{KvFormat, invalid},
};

/// Most states one attention residual reads, block states plus the partial one.
pub const RESIDUAL_ENTRIES: usize = 8;
/// Most learned queries of one attention residual.
pub const RESIDUAL_QUERIES: usize = 4;
const RESIDUAL_THREADS: usize = 256;

#[repr(C)]
struct ResidualParams {
    tokens: u32,
    width: u32,
    blocks: u32,
    queries: u32,
    partial_stride: u32,
    eps: f32,
    scale: f32,
}

#[repr(C)]
struct MessageParams {
    rows: u32,
    tokens: u32,
    width: u32,
    x_shared: u32,
    x_stride: u32,
    norm_rows: u32,
    eps: f32,
}

#[repr(C)]
struct GateGemvParams {
    groups: u32,
    rows: u32,
    outs: u32,
    k: u32,
    per: u32,
}

/// Most rows per group [`Kernels::gate_projection`] multiplies with its GEMV; each row reads
/// the weights again, from cache at these sizes.
pub const GATE_GEMV_ROWS: usize = 16;

#[repr(C)]
struct ShortConvParams {
    rows: u32,
    channels: u32,
    taps: u32,
    x_stride: u32,
    pool_width: u32,
    offset: u32,
    dynamic: u32,
    dyn_stride: u32,
    dyn_col: u32,
    head_dim: u32,
    divisor: f32,
    residual: u32,
    res_stride: u32,
    norm: u32,
    threshold: f32,
    eps: f32,
}

#[repr(C)]
struct RelativeParams {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    page: u32,
    group: u32,
    scale: f32,
    window: u32,
    rel_dim: u32,
    extent: u32,
    rel_stride: u32,
    rel_col: u32,
    sink_stride: u32,
    sink_col: u32,
    gate_stride: u32,
    gate_col: u32,
    splits: u32,
    split_keys: u32,
    q_stride: u32,
    q_col: u32,
    q_threshold: f32,
    rel_threshold: f32,
    log_scaled: u32,
    log_alpha: f32,
    log_floor: f32,
}

/// How a normalizing [`Kernels::short_conv`] normalizes each head.
#[derive(Clone, Copy, Debug)]
pub enum HeadNorm {
    /// Shrink the head to L2 norm `threshold` when longer: `x * min(threshold / (|x| + 1e-6), 1)`.
    Clip { threshold: f32 },
    /// `x / sqrt(mean(x^2) + eps)`, with no gain.
    Rms { eps: f32 },
}

/// Position-dependent growth of queries and relative states in long contexts.
#[derive(Clone, Copy, Debug)]
pub struct LogScale<'a> {
    /// `[tokens]` U32 positions.
    pub positions: &'a Tensor,
    pub alpha: f32,
    pub floor: f32,
}

/// One short convolution of [`Kernels::short_conv`].
#[derive(Clone, Copy, Debug)]
pub struct ShortConv<'a> {
    /// `[channels, taps]` F32 weights by lag: column 0 weights the token itself.
    pub taps: &'a Tensor,
    /// The `[slots, taps - 1, width]` history pool and the first column of it this
    /// convolution owns.
    pub pool: &'a Tensor,
    pub offset: usize,
    /// Per-token taps added to `taps`: `(rows, first column, channels per head)`, with
    /// head `h`'s lag `l` at column `first + h * taps + l`.
    pub dynamic: Option<(&'a Tensor, usize, usize)>,
    /// What the convolution plus its input is divided by.
    pub divisor: f32,
    /// `[rows, channels]` rows added to the output, such as the residual stream.
    pub residual: Option<&'a Tensor>,
    /// Normalize each head of this many channels afterwards. Excludes `residual`.
    pub norm: Option<(usize, HeadNorm)>,
}

/// Most channels per head of a normalizing convolution: one threadgroup holds a head.
const CONV_NORM_DIM: usize = 1024;

/// Heads read in place from projection rows, multiplied by a scale per dimension and shrunk
/// to an L2 norm: `x * scale * min(threshold / (|x * scale| + 1e-6), 1)`.
#[derive(Clone, Copy, Debug)]
pub struct Clipped<'a> {
    /// `[tokens, cols]` rows; head 0 starts at column `col`, and heads follow contiguously.
    pub rows: &'a Tensor,
    pub col: usize,
    /// `[heads * dim]` scale.
    pub scale: &'a Tensor,
    pub threshold: f32,
}

/// What [`Kernels::relative_attention`] adds to plain paged attention. The relative states,
/// `sinks` and `gate` are read in place from projection rows, at the column where their
/// head 0 starts.
#[derive(Clone, Copy, Debug)]
pub struct Relative<'a> {
    /// `rel_dim` relative state components per head.
    pub rel: Clipped<'a>,
    /// Growth of the queries and relative states in long contexts.
    pub log_scale: Option<LogScale<'a>>,
    /// `[rel_dim, extent]`: bias of each relative state component by backward distance.
    pub proj: &'a Tensor,
    /// One sink logit per head of each token, from column `sinks.1`.
    pub sinks: (&'a Tensor, usize),
    /// The pre-sigmoid output gate of each head dimension, from column `gate.1`.
    pub gate: (&'a Tensor, usize),
    pub window: Option<usize>,
    pub scale: f32,
}

impl Kernels {
    /// Scratch bytes of one [`Self::relative_attention`] dispatch: split partials and
    /// statistics when `tokens × heads` queries over at most `span` keys need splits.
    pub fn relative_attention_scratch_bytes(
        &self,
        tokens: usize,
        heads: usize,
        dim: usize,
        span: usize,
    ) -> usize {
        let split = self.attention_split(tokens, heads, span);
        if split.splits == 1 {
            return 0;
        }
        tokens
            .saturating_mul(heads)
            .saturating_mul(split.splits)
            .saturating_mul(dim + 2)
            .saturating_mul(4)
    }

    /// Parameters of the message kernels over `[inputs * tokens, width]` messages, input
    /// major, and `x` with as many rows or `tokens` rows every input shares.
    fn message_params(
        x: &View<'_>,
        m: &View<'_>,
        tokens: usize,
        eps: f32,
    ) -> Result<MessageParams> {
        let shared = x.rows != m.rows;
        if x.cols != m.cols
            || tokens == 0
            || (shared && x.rows != tokens)
            || !(eps.is_finite() && eps >= 0.0)
        {
            return Err(invalid("message rows do not match their inputs"));
        }
        m.contiguous("messages")?;
        Ok(MessageParams {
            rows: to_u32(m.rows, "message rows")?,
            tokens: to_u32(tokens, "tokens")?,
            width: to_u32(m.cols, "width")?,
            x_shared: u32::from(shared),
            x_stride: to_u32(x.row_stride, "row stride")?,
            norm_rows: 0,
            eps,
        })
    }

    /// `[x / rms(x) | m / rms(m)]` for each message (`[inputs, tokens, width]`) and its
    /// input's row of `x` (`[inputs * tokens, width]`, or `[tokens, width]` shared by every
    /// input), both RMS norms without a gain. Returns `[inputs, tokens, 2 * width]`.
    pub fn message_pair(&self, x: &Tensor, messages: &Tensor, eps: f32) -> Result<Tensor> {
        let (inputs, tokens, width) = messages.dims3()?;
        let messages = messages.reshape((inputs * tokens, width))?;
        let (xv, mv) = (self.view(x)?, self.view(&messages)?);
        let p = Self::message_params(&xv, &mv, tokens, eps)?;
        let pipeline = self.pipeline("inkling_message_pair_f32")?;
        let out = self.output(mv.rows, 2 * mv.cols)?;
        let threads = pipeline.threads(self.tuning.row_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_inkling_message_pair",
            mv.rows,
            threads,
            |e| {
                xv.bind(e, 0)?;
                mv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &p);
                Ok(())
            },
        )?;
        self.finish(out, (inputs, tokens, 2 * width))
    }

    /// `x @ w^T` per group for `x` `[groups, rows, k]` F32 and `w` `[groups, outs, k]` BF16:
    /// F32 accumulation over the exactly widened weights, so BF16-stored loop-gate weights
    /// multiply as their F32 values do at half the bytes. Up to [`GATE_GEMV_ROWS`] rows per
    /// group take the GEMV; more convert the weights once and use the matrix multiply.
    /// Returns `[groups, rows, outs]`.
    pub fn gate_projection(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (groups, rows, k) = x.dims3()?;
        let (w_groups, outs, w_k) = w.dims3()?;
        if (w_groups, w_k) != (groups, k) || w.dtype() != DType::BF16 || outs == 0 || k == 0 {
            return Err(invalid("gate projection shapes do not match"));
        }
        if rows > GATE_GEMV_ROWS {
            return Ok(x.matmul(&w.to_dtype(DType::F32)?.transpose(1, 2)?)?);
        }
        let x = x.reshape((groups * rows, k))?;
        let w = w.reshape((groups * outs, k))?;
        let (xv, wv) = (self.view(&x)?, self.runtime.view(&w, DType::BF16)?);
        xv.contiguous("gate projection input")?;
        wv.contiguous("gate projection weights")?;
        // Long rows give each simdgroup one output; short ones four, so a launch stays
        // a few thousand simdgroups either way.
        let per = if k >= 256 { 1 } else { 4 };
        let p = GateGemvParams {
            groups: to_u32(groups, "groups")?,
            rows: to_u32(rows, "rows")?,
            outs: to_u32(outs, "outputs")?,
            k: to_u32(k, "width")?,
            per,
        };
        let items = groups * rows * outs.div_ceil(per as usize);
        let pipeline = self.pipeline("inkling_gate_gemv_f32")?;
        let (launch, threads) = self.simdgroup_launch(pipeline, items);
        let out = self.output(groups * rows, outs)?;
        self.runtime
            .dispatch(pipeline, "tur_inkling_gate_gemv", launch, threads, |e| {
                xv.bind(e, 0)?;
                wv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &p);
                Ok(())
            })?;
        self.finish(out, (groups, rows, outs))
    }

    /// Mix each message into its input as [`Self::message_pair`] pairs them:
    /// `x - tanh(forget) * x + tanh(gain) * m`, with `gates` (`[inputs, tokens, 2 * width]`)
    /// holding the forget then the gain columns, then RMS-normalize it with its input's row
    /// of `norm` (`[inputs, width]`, or one `[width]` gain for all). Returns
    /// `[inputs * tokens, width]`.
    pub fn message_gate(
        &self,
        x: &Tensor,
        messages: &Tensor,
        gates: &Tensor,
        norm: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let (inputs, tokens, width) = messages.dims3()?;
        let messages = messages.reshape((inputs * tokens, width))?;
        let gates = gates.reshape((inputs * tokens, 2 * width))?;
        let (xv, mv) = (self.view(x)?, self.view(&messages)?);
        let (gv, nv) = (self.view(&gates)?, self.view(norm)?);
        let mut p = Self::message_params(&xv, &mv, tokens, eps)?;
        gv.expect(mv.rows, 2 * mv.cols, "message gates")?;
        gv.contiguous("message gates")?;
        nv.contiguous("message norm")?;
        if nv.cols != mv.cols || (nv.rows != 1 && nv.rows != inputs) {
            return Err(invalid("message norm needs one gain row or one per input"));
        }
        p.norm_rows = u32::from(nv.rows != 1);
        let pipeline = self.pipeline("inkling_message_gate_f32")?;
        let out = self.output(mv.rows, mv.cols)?;
        let threads = pipeline.threads(self.tuning.row_threads);
        self.runtime.dispatch(
            pipeline,
            "tur_inkling_message_gate",
            mv.rows,
            threads,
            |e| {
                xv.bind(e, 0)?;
                mv.bind(e, 1)?;
                gv.bind(e, 2)?;
                nv.bind(e, 3)?;
                out.bind(e, 4);
                e.set_bytes(5, &p);
                Ok(())
            },
        )?;
        self.finish(out, (mv.rows, mv.cols))
    }

    /// Attention of each of `queries` (`[queries, width]`) over every token's block states
    /// (each a contiguous `[tokens, width]` tensor, read in place) and its `partial` state
    /// (`[tokens, width]`), each RMS-normalized without a gain: per query, the softmax over
    /// states of `q · n_s * scale` weights the normalized states `n_s`. Returns
    /// `[queries, tokens, width]`.
    pub fn block_residual(
        &self,
        blocks: &[Tensor],
        partial: &Tensor,
        queries: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let pv = self.view(partial)?;
        let qv = self.view(queries)?;
        let (tokens, width) = (pv.rows, pv.cols);
        let count = blocks.len();
        let bvs = blocks
            .iter()
            .map(|b| {
                let view = self.view(b)?;
                view.expect(tokens, width, "block state")?;
                view.contiguous("block state")?;
                Ok(view)
            })
            .collect::<Result<Vec<_>>>()?;
        qv.contiguous("residual queries")?;
        if qv.cols != width
            || qv.rows == 0
            || qv.rows > RESIDUAL_QUERIES
            || count + 1 > RESIDUAL_ENTRIES
            || !eps.is_finite()
        {
            return Err(invalid(
                "attention residuals take 1-4 queries over at most 8 states of the residual width",
            ));
        }
        let p = ResidualParams {
            tokens: to_u32(tokens, "tokens")?,
            width: to_u32(width, "width")?,
            blocks: to_u32(count, "block states")?,
            queries: to_u32(qv.rows, "queries")?,
            partial_stride: to_u32(pv.row_stride, "partial stride")?,
            eps,
            scale: (width as f64).sqrt().recip() as f32,
        };
        let out = self.output(qv.rows * tokens, width)?;
        self.runtime.dispatch(
            self.pipeline("inkling_block_residual_f32")?,
            "tur_inkling_block_residual",
            tokens,
            RESIDUAL_THREADS,
            |e| {
                pv.bind(e, 0)?;
                qv.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &p);
                // Slots past the block states are never read; any bound buffer will do.
                for slot in 0..RESIDUAL_ENTRIES - 1 {
                    bvs.get(slot).unwrap_or(&pv).bind(e, 4 + slot)?;
                }
                Ok(())
            },
        )?;
        self.finish(out, (qv.rows, tokens, width))
    }

    /// `(conv(x) + x) / divisor` over every sequence of a packed pass, the depthwise causal
    /// convolution reading each sequence's history from its committed slot and writing it
    /// after each token its written slots name, in columns `offset..offset + channels` of
    /// the pool rows. Returns `[rows, channels]`.
    pub fn short_conv(
        &self,
        x: &Tensor,
        spans: &RecurrentSpans,
        conv: ShortConv<'_>,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let hv = self.view(conv.pool)?;
        let tv = self.view(conv.taps)?;
        let (slots, kept, width) = conv.pool.dims3()?;
        let (channels, taps) = (tv.rows, tv.cols);
        hv.contiguous("convolution history pool")?;
        tv.contiguous("convolution taps")?;
        spans.check(slots, xv.rows)?;
        if taps < 2
            || kept != taps - 1
            || xv.cols < channels
            || conv.offset + channels > width
            || !(conv.divisor.is_finite() && conv.divisor > 0.0)
        {
            return Err(invalid(
                "short convolution does not fit its input or history",
            ));
        }
        let dv = conv.dynamic.map(|(d, _, _)| self.view(d)).transpose()?;
        let (dyn_stride, dyn_col, head_dim) = match (&dv, conv.dynamic) {
            (Some(view), Some((_, col, head_dim))) => {
                let heads = channels / head_dim.max(1);
                if head_dim == 0
                    || !channels.is_multiple_of(head_dim)
                    || view.rows != xv.rows
                    || col + heads * taps > view.cols
                {
                    return Err(invalid("dynamic taps do not cover the convolution heads"));
                }
                (view.row_stride, col, head_dim)
            }
            _ => (0, 0, conv.norm.map_or(1, |(dim, _)| dim)),
        };
        let rv = conv.residual.map(|r| self.view(r)).transpose()?;
        if let Some(rv) = &rv {
            rv.expect(xv.rows, channels, "convolution residual")?;
        }
        let (norm, threshold, eps) = match conv.norm {
            None => (0, 0.0, 0.0),
            Some((dim, norm)) => {
                let (mode, threshold, eps) = match norm {
                    HeadNorm::Clip { threshold } if threshold.is_finite() && threshold > 0.0 => {
                        (0, threshold, 0.0)
                    }
                    HeadNorm::Rms { eps } if eps.is_finite() && eps >= 0.0 => (1, 0.0, eps),
                    _ => return Err(invalid("invalid convolution norm threshold or epsilon")),
                };
                if dim != head_dim
                    || dim == 0
                    || dim > CONV_NORM_DIM
                    || !channels.is_multiple_of(dim)
                    || rv.is_some()
                {
                    return Err(invalid("normalized convolution heads do not fit"));
                }
                (mode, threshold, eps)
            }
        };
        let p = ShortConvParams {
            rows: to_u32(xv.rows, "rows")?,
            channels: to_u32(channels, "channels")?,
            taps: to_u32(taps, "taps")?,
            x_stride: to_u32(xv.row_stride, "row stride")?,
            pool_width: to_u32(width, "pool width")?,
            offset: to_u32(conv.offset, "pool offset")?,
            dynamic: u32::from(dv.is_some()),
            dyn_stride: to_u32(dyn_stride, "dynamic stride")?,
            dyn_col: to_u32(dyn_col, "dynamic column")?,
            head_dim: to_u32(head_dim, "head dim")?,
            divisor: conv.divisor,
            residual: u32::from(rv.is_some()),
            res_stride: to_u32(rv.as_ref().map_or(0, |r| r.row_stride), "residual stride")?,
            norm,
            threshold,
            eps,
        };
        let count = xv.rows * channels;
        to_u32(count, "elements")?;
        let table = self.runtime.view(&spans.spans, DType::U32)?;
        let rows = self.runtime.view(&spans.rows, DType::U32)?;
        let slot_list = self.runtime.view(&spans.slots, DType::U32)?;
        let out = self.output(xv.rows, channels)?;
        let bind = |e: &ComputeCommandEncoder| -> Result<()> {
            xv.bind(e, 0)?;
            hv.bind_mut(e, 1)?;
            tv.bind(e, 2)?;
            out.bind(e, 3);
            e.set_bytes(4, &p);
            table.bind(e, 5)?;
            rows.bind(e, 6)?;
            slot_list.bind(e, 7)?;
            dv.as_ref().unwrap_or(&xv).bind(e, 8)?;
            Ok(())
        };
        if conv.norm.is_some() {
            // A threadgroup per head, a thread per channel; whole simdgroups for the reduction.
            let pipeline = self.pipeline("inkling_short_conv_norm_f32")?;
            if !head_dim.is_multiple_of(32) || pipeline.threads(head_dim) != head_dim {
                return Err(invalid(
                    "normalized convolution heads do not fit a threadgroup",
                ));
            }
            self.runtime.dispatch(
                pipeline,
                "tur_inkling_short_conv_norm",
                xv.rows * channels / head_dim,
                head_dim,
                bind,
            )?;
        } else {
            let pipeline = self.pipeline("inkling_short_conv_f32")?;
            let threads = pipeline.threads(self.tuning.elementwise_threads);
            self.runtime.dispatch(
                pipeline,
                "tur_inkling_short_conv",
                count.div_ceil(threads),
                threads,
                |e| {
                    bind(e)?;
                    // Unread without a residual, but every declared buffer must be bound.
                    rv.as_ref().unwrap_or(&xv).bind(e, 9)
                },
            )?;
        }
        self.finish(out, (xv.rows, channels))
    }

    /// Validated parameters of [`Self::relative_attention`], without its split.
    #[allow(
        clippy::too_many_lines,
        reason = "one check or field per kernel input; splitting would only scatter them"
    )]
    fn relative_params(
        &self,
        (q, heads): (&Clipped<'_>, usize),
        pool: &AttentionPool,
        kv: &PagedKv,
        relative: &Relative<'_>,
    ) -> Result<RelativeParams> {
        let (tokens, kv_heads, dim) = (kv.tokens, pool.kv_heads, pool.head_dim);
        let [rel_dim, extent] = relative.proj.dims() else {
            return Err(invalid("relative bias projection is not a matrix"));
        };
        let (rel_dim, extent) = (*rel_dim, *extent);
        let (qv, rv) = (self.view(q.rows)?, self.view(relative.rel.rows)?);
        let (sv, gv) = (self.view(relative.sinks.0)?, self.view(relative.gate.0)?);
        self.view(relative.proj)?
            .contiguous("relative bias projection")?;
        let scales = [
            (q.scale, heads * dim),
            (relative.rel.scale, heads * rel_dim),
        ];
        for (scale, len) in scales {
            let view = self.view(scale)?;
            view.contiguous("clip scale")?;
            if view.rows * view.cols != len {
                return Err(invalid("clip scale does not cover its heads"));
            }
        }
        // Each lane loads its query dimensions as aligned vectors of up to four floats.
        let aligned = qv.row_stride.is_multiple_of(4) && q.col.is_multiple_of(4);
        if kv_heads == 0
            || heads == 0
            || !heads.is_multiple_of(kv_heads)
            || kv.page != pool.page_tokens
            || !aligned
            || qv.rows != tokens
            || q.col + heads * dim > qv.cols
            || rel_dim == 0
            || rel_dim > 32
            || extent == 0
            || rv.rows != tokens
            || relative.rel.col + heads * rel_dim > rv.cols
            || sv.rows != tokens
            || relative.sinks.1 + heads > sv.cols
            || gv.rows != tokens
            || relative.gate.1 + heads * dim > gv.cols
            || relative.window == Some(0)
            || !(relative.scale.is_finite() && relative.scale > 0.0)
            || ![q.threshold, relative.rel.threshold]
                .iter()
                .all(|t| t.is_finite() && *t > 0.0)
        {
            return Err(invalid("relative attention shapes do not match"));
        }
        if let Some(l) = relative.log_scale
            && (l.positions.dims() != [tokens] || !(l.floor > 0.0 && l.alpha.is_finite()))
        {
            return Err(invalid("log scaling needs one position per query"));
        }
        let log = relative.log_scale;
        Ok(RelativeParams {
            tokens: to_u32(tokens, "tokens")?,
            heads: to_u32(heads, "heads")?,
            kv_heads: to_u32(kv_heads, "kv heads")?,
            dim: to_u32(dim, "head dim")?,
            page: to_u32(kv.page, "page")?,
            group: to_u32(pool.format.group().unwrap_or(32), "group")?,
            scale: relative.scale,
            window: to_u32(relative.window.unwrap_or(0), "window")?,
            rel_dim: to_u32(rel_dim, "relative dim")?,
            extent: to_u32(extent, "relative extent")?,
            rel_stride: to_u32(rv.row_stride, "relative stride")?,
            rel_col: to_u32(relative.rel.col, "relative column")?,
            sink_stride: to_u32(sv.row_stride, "sink stride")?,
            sink_col: to_u32(relative.sinks.1, "sink column")?,
            gate_stride: to_u32(gv.row_stride, "gate stride")?,
            gate_col: to_u32(relative.gate.1, "gate column")?,
            splits: 1,
            split_keys: 0,
            q_stride: to_u32(qv.row_stride, "query stride")?,
            q_col: to_u32(q.col, "query column")?,
            q_threshold: q.threshold,
            rel_threshold: relative.rel.threshold,
            log_scaled: u32::from(log.is_some()),
            log_alpha: log.map_or(0.0, |l| l.alpha),
            log_floor: log.map_or(1.0, |l| l.floor),
        })
    }

    /// Paged attention of `heads` queries per token over `pool`, the queries read from
    /// projection rows and clipped as `q` says, with each score `q · k * scale` plus the
    /// learned bias at its backward distance, a sink logit per token and head in each
    /// softmax, and the output gated by `sigmoid(gate)`. Windowed layers read the last
    /// `window` keys. Returns `[tokens, heads * dim]`.
    pub fn relative_attention(
        &self,
        (q, heads): (Clipped<'_>, usize),
        pool: &AttentionPool,
        kv: &PagedKv,
        relative: Relative<'_>,
    ) -> Result<Tensor> {
        let mut p = self.relative_params((&q, heads), pool, kv, &relative)?;
        let (tokens, kv_heads, dim) = (kv.tokens, pool.kv_heads, pool.head_dim);
        let qv = self.view(q.rows)?;
        let rv = self.view(relative.rel.rows)?;
        let pv = self.view(relative.proj)?;
        let sv = self.view(relative.sinks.0)?;
        let gv = self.view(relative.gate.0)?;
        let (qs, rs) = (self.view(q.scale)?, self.view(relative.rel.scale)?);
        let positions = relative
            .log_scale
            .map(|l| self.runtime.view(l.positions, DType::U32))
            .transpose()?;
        let (k_data, k_scales) = self.kv_views(&pool.key, pool.format, kv.slots, kv_heads, dim)?;
        let (v_data, v_scales) =
            self.kv_views(&pool.value, pool.format, kv.slots, kv_heads, dim)?;
        let table = self.runtime.view(&kv.table, DType::U32)?;
        let first = self.runtime.view(&kv.first_block, DType::U32)?;
        let visible = self.runtime.view(&kv.visible, DType::U32)?;
        // A query reads at most `window` keys, from its window's first key.
        let span = relative
            .window
            .map_or(kv.max_visible, |w| kv.max_visible.min(w));
        let split = self.attention_split(tokens, heads, span);
        (p.splits, p.split_keys) = (
            to_u32(split.splits, "splits")?,
            to_u32(split.split_keys, "split keys")?,
        );
        let items = tokens * heads;
        let pipeline = self.pipeline(&Self::kv_kernel("inkling_attention", dim, pool.format)?)?;
        let (groups, threads) = self.simdgroup_launch(pipeline, items * split.splits);
        let out = self.output(tokens, heads * dim)?;
        let scratch = if split.splits > 1 {
            Some((
                self.output(items * split.splits, dim)?,
                self.output(items * split.splits, 2)?,
            ))
        } else {
            None
        };
        self.runtime
            .dispatch(pipeline, "tur_inkling_attention", groups, threads, |e| {
                qv.bind(e, 0)?;
                k_data.bind(e, 1)?;
                v_data.bind(e, 2)?;
                table.bind(e, 3)?;
                first.bind(e, 4)?;
                visible.bind(e, 5)?;
                rv.bind(e, 6)?;
                pv.bind(e, 7)?;
                sv.bind(e, 8)?;
                gv.bind(e, 9)?;
                out.bind(e, 10);
                e.set_bytes(11, &p);
                // Unread for unquantized K/V; any bound buffer will do.
                k_scales.as_ref().unwrap_or(&k_data).bind(e, 12)?;
                v_scales.as_ref().unwrap_or(&v_data).bind(e, 13)?;
                // Unused with one split, but every declared buffer must be bound.
                let (partial, stats) = scratch.as_ref().map_or((&out, &out), |(p, s)| (p, s));
                partial.bind(e, 14);
                stats.bind(e, 15);
                qs.bind(e, 16)?;
                rs.bind(e, 17)?;
                // Unread without log scaling.
                positions.as_ref().unwrap_or(&qv).bind(e, 18)?;
                Ok(())
            })?;
        if let Some((partial, stats)) = &scratch {
            let reduce = Self::kv_kernel("inkling_attention_reduce", dim, KvFormat::F32)?;
            let pipeline = self.pipeline(&reduce)?;
            let (groups, threads) = self.simdgroup_launch(pipeline, items);
            self.runtime.dispatch(
                pipeline,
                "tur_inkling_attention_reduce",
                groups,
                threads,
                |e| {
                    partial.bind(e, 0);
                    stats.bind(e, 1);
                    sv.bind(e, 2)?;
                    gv.bind(e, 3)?;
                    out.bind(e, 4);
                    e.set_bytes(5, &p);
                    Ok(())
                },
            )?;
        }
        self.finish(out, (tokens, heads * dim))
    }
}
