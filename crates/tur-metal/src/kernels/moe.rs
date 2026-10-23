//! Device-owned routing and grouped expert projections. No routing readback or padding.
use super::{Kernels, PackedLinear, runtime::to_u32};
use candle_core::{Tensor, quantized::GgmlDType};
use tur_core::{Result, execution::invalid};

/// Validated device routing, shared by gather, both expert projections and scatter.
/// Construction is private: every expert and permutation index comes from checked kernels.
pub struct ExpertRoutes {
    selected: Tensor,
    tiles: Tensor,
    wide_tiles: Tensor,
    order: Tensor,
    tokens: usize,
    experts: usize,
    top_k: usize,
}

#[repr(C)]
struct RouteParams {
    tokens: u32,
    experts: u32,
    top_k: u32,
    stride: u32,
}
#[repr(C)]
struct ExpertParams {
    rows: u32,
    experts: u32,
    top_k: u32,
    width: u32,
    stride: u32,
    out: u32,
}

impl ExpertRoutes {
    fn rows(&self) -> usize {
        self.tokens * self.top_k
    }
    fn params(&self, width: usize, stride: usize, out: usize) -> Result<ExpertParams> {
        Ok(ExpertParams {
            rows: to_u32(self.rows(), "assignments")?,
            experts: to_u32(self.experts, "experts")?,
            top_k: to_u32(self.top_k, "top-k")?,
            width: to_u32(width, "width")?,
            stride: to_u32(stride, "row stride")?,
            out: to_u32(out, "output width")?,
        })
    }
}

impl Kernels {
    /// Select up to 256 experts with F32 logits and normalized top-k probabilities.
    /// Ties keep the lower expert ID. Nonfinite rows propagate NaNs through scatter.
    /// All metadata stays on this device and has a bounded size proportional to tokens × k.
    pub fn route(&self, logits: &Tensor, top_k: usize) -> Result<ExpertRoutes> {
        let x = self.view(logits)?;
        let (tokens, experts) = (x.rows, x.cols);
        let assignments = tokens
            .checked_mul(top_k)
            .ok_or_else(|| invalid("MoE assignments overflow"))?;
        if tokens == 0
            || experts == 0
            || experts > 256
            || top_k == 0
            || top_k > experts
            || assignments > 1 << 24
        {
            return Err(invalid(
                "MoE routing needs nonempty rows, 1..=256 experts, valid top-k and ≤2^24 assignments",
            ));
        }
        let p = RouteParams {
            tokens: to_u32(tokens, "tokens")?,
            experts: to_u32(experts, "experts")?,
            top_k: to_u32(top_k, "top-k")?,
            stride: to_u32(x.row_stride, "stride")?,
        };
        let selected = self.output(tokens, top_k * 2)?;
        self.runtime.dispatch(
            self.pipeline("moe_route_f32")?,
            "tur_moe_route",
            tokens,
            32,
            |e| {
                x.bind(e, 0)?;
                selected.bind(e, 1);
                e.set_bytes(2, &p);
                Ok(())
            },
        )?;
        let selected = self.finish(selected, (tokens, top_k * 2))?;
        let sv = self.view(&selected)?;
        let counts = self.output(experts, 1)?;
        self.runtime.dispatch(
            self.pipeline("moe_counts_f32")?,
            "tur_moe_counts",
            experts.div_ceil(32),
            32,
            |e| {
                sv.bind(e, 0)?;
                counts.bind(e, 1);
                e.set_bytes(2, &p);
                Ok(())
            },
        )?;
        let counts = self.finish(counts, (experts, 1))?;
        let cv = self.view(&counts)?;
        let groups = self.output(experts, 6)?;
        let tile_count = assignments.min(assignments.div_ceil(8) + experts - 1);
        let wide_count = assignments.min(assignments.div_ceil(16) + experts - 1);
        let tiles = self.output(tile_count, 3)?;
        let wide_tiles = self.output(wide_count, 3)?;
        self.runtime.dispatch(
            self.pipeline("moe_offsets_f32")?,
            "tur_moe_offsets",
            1,
            32,
            |e| {
                cv.bind(e, 0)?;
                groups.bind(e, 1);
                tiles.bind(e, 2);
                wide_tiles.bind(e, 3);
                e.set_bytes(4, &p);
                Ok(())
            },
        )?;
        let groups = self.finish(groups, (experts, 6))?;
        let gv = self.view(&groups)?;
        let order = self.output(assignments, 3)?;
        self.runtime.dispatch(
            self.pipeline("moe_order_f32")?,
            "tur_moe_order",
            experts.div_ceil(32),
            32,
            |e| {
                sv.bind(e, 0)?;
                gv.bind(e, 1)?;
                order.bind(e, 2);
                e.set_bytes(3, &p);
                Ok(())
            },
        )?;
        drop(gv);
        drop(sv);
        Ok(ExpertRoutes {
            selected,
            tiles: self.finish(tiles, (tile_count, 3))?,
            wide_tiles: self.finish(wide_tiles, (wide_count, 3))?,
            order: self.finish(order, (assignments, 3))?,
            tokens,
            experts,
            top_k,
        })
    }

    /// Gather each token once per selected expert into contiguous groups without padding.
    pub fn expert_gather(&self, x: &Tensor, routes: &ExpertRoutes) -> Result<Tensor> {
        let xv = self.view(x)?;
        if xv.rows != routes.tokens {
            return Err(invalid("expert gather token count differs"));
        }
        let order = self.view(&routes.order)?;
        let p = routes.params(xv.cols, xv.row_stride, 0)?;
        let out = self.output(routes.rows(), xv.cols)?;
        let elements = routes
            .rows()
            .checked_mul(xv.cols)
            .ok_or_else(|| invalid("expert gather overflow"))?;
        to_u32(elements, "gather elements")?;
        self.runtime.dispatch(
            self.pipeline("moe_gather_f32")?,
            "tur_moe_gather",
            elements.div_ceil(128),
            128,
            |e| {
                xv.bind(e, 0)?;
                order.bind(e, 1)?;
                out.bind(e, 2);
                e.set_bytes(3, &p);
                Ok(())
            },
        )?;
        self.finish(out, (routes.rows(), xv.cols))
    }

    /// Matrix-unit GEMM over expert groups, reading dense, `Q8_0` or Q4K weights in place.
    /// `layer` flattens `[experts, out, in]` to `[experts*out, in]`; inputs are gathered
    /// by [`Self::expert_gather`]. No expert is expanded or independently submitted.
    pub fn expert_linear(
        &self,
        layer: &PackedLinear,
        x: &Tensor,
        routes: &ExpertRoutes,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let width = layer.in_features();
        let out_width = layer.out_features() / routes.experts;
        if xv.rows != routes.rows()
            || xv.cols != width
            || out_width == 0
            || !layer.out_features().is_multiple_of(routes.experts)
        {
            return Err(invalid("grouped expert projection shape differs"));
        }
        let p = routes.params(width, xv.row_stride, out_width)?;
        let (name, packed) = match layer.packed_buffer() {
            Some((buffer, GgmlDType::Q8_0)) => ("moe_qmm_q8_0", Some(buffer)),
            Some((buffer, GgmlDType::Q4K)) => ("moe_gemm_q4k", Some(buffer)),
            None if layer.format().is_none() => ("moe_gemm_f32", None),
            _ => return Err(invalid("unsupported expert weight format")),
        };
        let dense = layer.dense_tensor().map(|t| self.view(t)).transpose()?;
        if let Some(dense) = &dense {
            dense.contiguous("expert weights")?;
        }
        if let Some(buffer) = packed
            && buffer.length() < layer.resident_bytes()
        {
            return Err(invalid("expert weights exceed packed buffer"));
        }
        let q8 = layer.format() == Some(GgmlDType::Q8_0);
        let tile_size = if q8 { 16 } else { 8 };
        let tiles = self.view(if q8 {
            &routes.wide_tiles
        } else {
            &routes.tiles
        })?;
        let launch = tiles
            .rows
            .checked_mul(out_width.div_ceil(tile_size))
            .ok_or_else(|| invalid("expert GEMM grid overflow"))?;
        to_u32(launch, "expert tiles")?;
        let out = self.output(routes.rows(), out_width)?;
        let pipeline = self.pipeline(name)?;
        let threads = if q8 {
            pipeline.simdgroups(self.tuning.simdgroups).max(1) * 32
        } else {
            32
        };
        let partials = threads / 32 * 4 * 64 * 4;
        if q8 && partials > self.capabilities().threadgroup_memory_bytes {
            return Err(invalid("grouped expert partials exceed threadgroup memory"));
        }
        self.runtime
            .dispatch(pipeline, "tur_moe_gemm", launch, threads, |e| {
                if let Some(buffer) = packed {
                    e.set_input_buffer(0, Some(buffer), 0);
                } else {
                    dense
                        .as_ref()
                        .ok_or_else(|| invalid("missing expert weights"))?
                        .bind(e, 0)?;
                }
                xv.bind(e, 1)?;
                tiles.bind(e, 2)?;
                out.bind(e, 3);
                e.set_bytes(4, &p);
                if q8 {
                    e.set_threadgroup_memory_length(0, partials);
                }
                Ok(())
            })?;
        self.finish(out, (routes.rows(), out_width))
    }

    /// Weighted scatter to token order, accumulated in expert ID order for stable rounding.
    pub fn expert_scatter(&self, x: &Tensor, routes: &ExpertRoutes) -> Result<Tensor> {
        let xv = self.view(x)?;
        if xv.rows != routes.rows() {
            return Err(invalid("expert scatter row count differs"));
        }
        let order = self.view(&routes.order)?;
        let selected = self.view(&routes.selected)?;
        let p = routes.params(xv.cols, xv.row_stride, 0)?;
        let out = self.output(routes.tokens, xv.cols)?;
        let elements = routes
            .tokens
            .checked_mul(xv.cols)
            .ok_or_else(|| invalid("expert scatter overflow"))?;
        to_u32(elements, "scatter elements")?;
        self.runtime.dispatch(
            self.pipeline("moe_scatter_f32")?,
            "tur_moe_scatter",
            elements.div_ceil(128),
            128,
            |e| {
                xv.bind(e, 0)?;
                order.bind(e, 1)?;
                selected.bind(e, 2)?;
                out.bind(e, 3);
                e.set_bytes(4, &p);
                Ok(())
            },
        )?;
        self.finish(out, (routes.tokens, xv.cols))
    }
}
