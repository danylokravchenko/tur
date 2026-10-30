//! Device-owned routing and grouped expert projections. No routing readback or padding.
use super::{Kernels, PackedLinear, runtime::to_u32};
use candle_core::{Tensor, quantized::GgmlDType};
use tur_core::{Result, execution::invalid};

/// Assignments per histogram block and threads of the block-wide routing kernels.
const BLOCK: usize = 256;
/// Rows of one expert tile; groups above it use wider tiles (32 rows, fewer tiles).
const TILE: usize = 16;
/// Staged F16 weights `[64, 32]` and F32 activations `[32, 32]`, reused for the
/// `[32, 64]` F32 result.
const WIDE_STAGE_BYTES: usize = 64 * 32 * 2 + 32 * 32 * 4;
/// Simdgroups of one matrix-unit threadgroup, each owning 32 outputs of a tile.
const GEMM_SIMDGROUPS: usize = 2;

/// Validated device routing, shared by gather, both expert projections and scatter.
/// Construction is private: every expert and permutation index comes from checked kernels.
pub struct ExpertRoutes {
    selected: Tensor,
    tiles: Tensor,
    order: Tensor,
    tokens: usize,
    experts: usize,
    top_k: usize,
    /// Banks the weights hold when `tiles` address slots of an arena rather than experts.
    slots: Option<usize>,
}

#[repr(C)]
struct RouteParams {
    tokens: u32,
    experts: u32,
    top_k: u32,
    stride: u32,
    scaled: u32,
}
#[repr(C)]
struct ExpertParams {
    rows: u32,
    experts: u32,
    top_k: u32,
    width: u32,
    stride: u32,
    out: u32,
    routed: u32,
    scales: u64,
}
#[repr(C)]
struct GluParams {
    rows: u32,
    width: u32,
    stride: u32,
    alpha: f32,
    limit: f32,
}
#[repr(C)]
struct ScatterParams {
    tokens: u32,
    top_k: u32,
    width: u32,
    stride: u32,
    shared: u32,
    shared_stride: u32,
    gate_stride: u32,
}

/// A shared expert's output and its gate logit, added by [`Kernels::expert_combine`].
pub struct SharedExpert<'a> {
    /// `[tokens, width]` shared expert output.
    pub output: &'a Tensor,
    /// `[tokens, 1]` pre-sigmoid gate.
    pub gate: &'a Tensor,
}

impl ExpertRoutes {
    /// Tile rows `[expert, first row, rows, wide]`; waits for all GPU work before this.
    pub(crate) fn tile_values(&self) -> Result<Vec<f32>> {
        Ok(self.tiles.flatten_all()?.to_vec1::<f32>()?)
    }
    /// `(expert, weight)` of every assignment, in token order; waits for all GPU work.
    pub fn selections(&self) -> Result<Vec<(usize, f32)>> {
        let values = self.selected.flatten_all()?.to_vec1::<f32>()?;
        Ok(values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0] as usize, pair[1]))
            .collect())
    }
    fn rows(&self) -> usize {
        self.tokens * self.top_k
    }
    fn params(
        &self,
        width: usize,
        stride: usize,
        out: usize,
        routed: bool,
    ) -> Result<ExpertParams> {
        Ok(ExpertParams {
            rows: to_u32(self.rows(), "assignments")?,
            experts: to_u32(self.experts, "experts")?,
            top_k: to_u32(self.top_k, "top-k")?,
            width: to_u32(width, "width")?,
            stride: to_u32(stride, "row stride")?,
            out: to_u32(out, "output width")?,
            routed: u32::from(routed),
            scales: 0,
        })
    }
}

impl Kernels {
    /// Select up to 256 experts with F32 logits and normalized top-k probabilities.
    /// Ties keep the lower expert ID. Nonfinite rows propagate NaNs through scatter.
    /// All metadata stays on this device and has a bounded size proportional to tokens × k;
    /// the permutation keeps each expert's rows in assignment order.
    pub fn route(&self, logits: &Tensor, top_k: usize) -> Result<ExpertRoutes> {
        self.routing(logits, top_k, None)
    }

    /// Normalized top-k routing with a learned scale applied to each selected expert's
    /// probability after normalization. Selection remains based on the original logits.
    pub fn route_scaled(
        &self,
        logits: &Tensor,
        top_k: usize,
        scales: &Tensor,
    ) -> Result<ExpertRoutes> {
        self.routing(logits, top_k, Some(scales))
    }

    fn routing(
        &self,
        logits: &Tensor,
        top_k: usize,
        scales: Option<&Tensor>,
    ) -> Result<ExpertRoutes> {
        let x = self.view(logits)?;
        let (tokens, experts) = (x.rows, x.cols);
        let assignments = tokens
            .checked_mul(top_k)
            .ok_or_else(|| invalid("MoE assignments overflow"))?;
        if tokens == 0
            || experts == 0
            || experts > BLOCK
            || top_k == 0
            || top_k > experts
            || assignments > 1 << 24
        {
            return Err(invalid(
                "MoE routing needs nonempty rows, 1..=256 experts, valid top-k and ≤2^24 assignments",
            ));
        }
        let sv = scales.map(|s| self.view(s)).transpose()?;
        if let Some(sv) = &sv {
            sv.expect(1, experts, "expert scales")?;
            sv.contiguous("expert scales")?;
        }
        let p = RouteParams {
            tokens: to_u32(tokens, "tokens")?,
            experts: to_u32(experts, "experts")?,
            top_k: to_u32(top_k, "top-k")?,
            stride: to_u32(x.row_stride, "stride")?,
            scaled: u32::from(scales.is_some()),
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
                sv.as_ref().unwrap_or(&x).bind(e, 3)?;
                Ok(())
            },
        )?;
        let selected = self.finish(selected, (tokens, top_k * 2))?;
        let sv = self.view(&selected)?;
        let blocks = assignments.div_ceil(BLOCK);
        let hist = self.output(blocks, experts)?;
        let order = self.output(assignments, 3)?;
        self.runtime.dispatch(
            self.pipeline("moe_histogram_f32")?,
            "tur_moe_histogram",
            blocks,
            BLOCK,
            |e| {
                sv.bind(e, 0)?;
                hist.bind(e, 1);
                order.bind(e, 2);
                e.set_bytes(3, &p);
                Ok(())
            },
        )?;
        let groups = self.output(experts, 1)?;
        // Wide tiles hold more rows, so the small-tile bound covers both kinds.
        let tile_count = assignments.min(assignments.div_ceil(TILE) + experts - 1);
        let tiles = self.output(tile_count, 4)?;
        self.runtime.dispatch(
            self.pipeline("moe_offsets_f32")?,
            "tur_moe_offsets",
            1,
            BLOCK,
            |e| {
                hist.bind(e, 0);
                groups.bind(e, 1);
                tiles.bind(e, 2);
                e.set_bytes(3, &p);
                Ok(())
            },
        )?;
        let hist = self.finish(hist, (blocks, experts))?;
        let groups = self.finish(groups, (experts, 1))?;
        let (hv, gv) = (self.view(&hist)?, self.view(&groups)?);
        self.runtime.dispatch(
            self.pipeline("moe_order_f32")?,
            "tur_moe_order",
            blocks,
            BLOCK,
            |e| {
                sv.bind(e, 0)?;
                hv.bind(e, 1)?;
                gv.bind(e, 2)?;
                order.bind(e, 3);
                e.set_bytes(4, &p);
                Ok(())
            },
        )?;
        drop((hv, gv, sv));
        Ok(ExpertRoutes {
            selected,
            tiles: self.finish(tiles, (tile_count, 4))?,
            order: self.finish(order, (assignments, 3))?,
            tokens,
            experts,
            top_k,
            slots: None,
        })
    }

    /// `routes` reading `tiles` whose expert field now holds a slot of an arena of `slots`
    /// banks. Only the weight kernels read that field: the permutation and the scatter keep
    /// original expert IDs, so biases and accumulation order do not change.
    pub(crate) fn with_slots(
        &self,
        routes: ExpertRoutes,
        tiles: Vec<f32>,
        slots: usize,
    ) -> Result<ExpertRoutes> {
        if tiles.len() != routes.tiles.elem_count() {
            return Err(invalid("expert slot tiles changed shape"));
        }
        Ok(ExpertRoutes {
            tiles: crate::experts::tile_tensor(tiles, self.device())?,
            slots: Some(slots),
            ..routes
        })
    }

    /// Gather each token once per selected expert into contiguous groups without padding.
    /// [`Self::expert_linear_routed`] reads the same rows without this copy.
    pub fn expert_gather(&self, x: &Tensor, routes: &ExpertRoutes) -> Result<Tensor> {
        let xv = self.view(x)?;
        if xv.rows != routes.tokens {
            return Err(invalid("expert gather token count differs"));
        }
        let order = self.view(&routes.order)?;
        let p = routes.params(xv.cols, xv.row_stride, 0, false)?;
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
    /// `layer` flattens `[experts, out, in]` to `[experts*out, in]`; `x` holds one row per
    /// assignment in group order (an [`Self::expert_gather`] or a previous projection).
    /// No expert is expanded or independently submitted.
    pub fn expert_linear(
        &self,
        layer: &PackedLinear,
        x: &Tensor,
        routes: &ExpertRoutes,
    ) -> Result<Tensor> {
        self.grouped(layer, x, routes, false)
    }

    /// [`Self::expert_linear`] over token-major `x` (one row per token): each group row
    /// reads its token through the routing permutation instead of a gathered copy.
    pub fn expert_linear_routed(
        &self,
        layer: &PackedLinear,
        x: &Tensor,
        routes: &ExpertRoutes,
    ) -> Result<Tensor> {
        self.grouped(layer, x, routes, true)
    }

    fn grouped(
        &self,
        layer: &PackedLinear,
        x: &Tensor,
        routes: &ExpertRoutes,
        routed: bool,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        let width = layer.in_features();
        let banks = routes.slots.unwrap_or(routes.experts);
        let out_width = layer.out_features() / banks;
        let rows = if routed { routes.tokens } else { routes.rows() };
        if xv.rows != rows
            || xv.cols != width
            || !width.is_multiple_of(32)
            || out_width == 0
            || !layer.out_features().is_multiple_of(banks)
        {
            return Err(invalid(
                "grouped expert projection shape differs or width is not a multiple of 32",
            ));
        }
        let mut p = routes.params(width, xv.row_stride, out_width, routed)?;
        // GEMV outputs per simdgroup match the dense GEMV kernels.
        let (suffix, gemv_rows, packed) = match (layer.packed_buffer(), layer.mxfp4_buffer()) {
            (_, Some((buffer, scales))) => {
                p.scales = scales as u64;
                ("mxfp4", 4, Some(buffer))
            }
            (Some((buffer, GgmlDType::Q8_0)), _) => ("q8_0", 4, Some(buffer)),
            (Some((buffer, GgmlDType::Q4K)), _) => ("q4k", 4, Some(buffer)),
            (None, None) if layer.format().is_none() => ("f32", 4, None),
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
        let tiles = self.view(&routes.tiles)?;
        let order = self.view(&routes.order)?;
        let out = self.output(routes.rows(), out_width)?;
        if WIDE_STAGE_BYTES > self.capabilities().threadgroup_memory_bytes {
            return Err(invalid("wide expert tiles exceed threadgroup memory"));
        }
        // Every tile goes to exactly one kernel: within groups of at most sixteen rows, GEMV
        // for at most three rows and F32 matrix units otherwise; larger groups' wide tiles
        // use F16-staged weights. Each launches over all tiles and skips the others.
        let gemv = self.pipeline(&format!("moe_gemv_{suffix}"))?;
        let gemv_simdgroups = gemv.simdgroups(self.tuning.simdgroups).max(1);
        // A token selects each expert at most once, so no group has more rows than the
        // pass has tokens: the GEMM (4–16 rows) and wide (17+) tiers are skipped when no
        // tile can reach them, instead of launching grids that all exit.
        let tiers = [
            (gemv, gemv_simdgroups, gemv_rows * gemv_simdgroups, 0, 1),
            (
                self.pipeline(&format!("moe_gemm_{suffix}"))?,
                GEMM_SIMDGROUPS,
                32 * GEMM_SIMDGROUPS,
                0,
                4,
            ),
            (
                self.pipeline(&format!("moe_gemm_wide_{suffix}"))?,
                4,
                64,
                WIDE_STAGE_BYTES,
                TILE + 1,
            ),
        ];
        for (pipeline, simdgroups, outputs, stage, smallest) in tiers {
            if routes.tokens < smallest {
                continue;
            }
            let launch = tiles
                .rows
                .checked_mul(out_width.div_ceil(outputs))
                .ok_or_else(|| invalid("expert GEMM grid overflow"))?;
            to_u32(launch, "expert tiles")?;
            self.runtime
                .dispatch(pipeline, "tur_moe_gemm", launch, simdgroups * 32, |e| {
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
                    order.bind(e, 5)?;
                    if stage > 0 {
                        e.set_threadgroup_memory_length(0, stage);
                    }
                    Ok(())
                })?;
        }
        self.finish(out, (routes.rows(), out_width))
    }

    /// `x + bias[expert]` over one row per assignment in group order; `bias` is
    /// `[experts, width]` and stays indexed by original expert IDs.
    pub fn expert_bias(&self, x: &Tensor, bias: &Tensor, routes: &ExpertRoutes) -> Result<Tensor> {
        let xv = self.view(x)?;
        let bv = self.view(bias)?;
        xv.expect(routes.rows(), xv.cols, "expert output")?;
        bv.expect(routes.experts, xv.cols, "expert bias")?;
        bv.contiguous("expert bias")?;
        let order = self.view(&routes.order)?;
        let p = routes.params(xv.cols, xv.row_stride, xv.cols, false)?;
        let out = self.output(routes.rows(), xv.cols)?;
        self.runtime.dispatch(
            self.pipeline("moe_bias_f32")?,
            "tur_moe_bias",
            routes.rows(),
            BLOCK,
            |e| {
                xv.bind(e, 0)?;
                order.bind(e, 1)?;
                bv.bind(e, 2)?;
                out.bind(e, 3);
                e.set_bytes(4, &p);
                Ok(())
            },
        )?;
        self.finish(out, (routes.rows(), xv.cols))
    }

    /// `(clamp(up) + 1) * gate * sigmoid(gate * alpha)` over interleaved gate|up columns of
    /// one row per assignment, after adding each expert's `[experts, 2 * width]` bias.
    /// Gates clamp above at `limit`, and ups to `±limit`. Returns `[rows, width]`.
    pub fn expert_glu(
        &self,
        gate_up: &Tensor,
        bias: &Tensor,
        routes: &ExpertRoutes,
        alpha: f32,
        limit: f32,
    ) -> Result<Tensor> {
        let xv = self.view(gate_up)?;
        let bv = self.view(bias)?;
        xv.expect(routes.rows(), xv.cols, "expert gate|up")?;
        bv.expect(routes.experts, xv.cols, "expert gate|up bias")?;
        bv.contiguous("expert gate|up bias")?;
        if !xv.cols.is_multiple_of(2) {
            return Err(invalid("expert gate|up width is odd"));
        }
        let width = xv.cols / 2;
        let order = self.view(&routes.order)?;
        let p = GluParams {
            rows: to_u32(routes.rows(), "assignments")?,
            width: to_u32(width, "width")?,
            stride: to_u32(xv.row_stride, "row stride")?,
            alpha,
            limit,
        };
        let out = self.output(routes.rows(), width)?;
        self.runtime.dispatch(
            self.pipeline("moe_glu_f32")?,
            "tur_moe_glu",
            routes.rows(),
            BLOCK,
            |e| {
                xv.bind(e, 0)?;
                order.bind(e, 1)?;
                bv.bind(e, 2)?;
                out.bind(e, 3);
                e.set_bytes(4, &p);
                Ok(())
            },
        )?;
        self.finish(out, (routes.rows(), width))
    }

    /// Weighted scatter to token order, accumulated in expert ID order for stable rounding.
    pub fn expert_scatter(&self, x: &Tensor, routes: &ExpertRoutes) -> Result<Tensor> {
        self.combine(x, routes, None)
    }

    /// [`Self::expert_scatter`] plus a shared expert's output times the sigmoid of its
    /// gate, in one pass.
    pub fn expert_combine(
        &self,
        x: &Tensor,
        routes: &ExpertRoutes,
        shared: SharedExpert<'_>,
    ) -> Result<Tensor> {
        self.combine(x, routes, Some(shared))
    }

    fn combine(
        &self,
        x: &Tensor,
        routes: &ExpertRoutes,
        shared: Option<SharedExpert<'_>>,
    ) -> Result<Tensor> {
        let xv = self.view(x)?;
        if xv.rows != routes.rows() {
            return Err(invalid("expert scatter row count differs"));
        }
        let views = match shared {
            Some(s) => Some((self.view(s.output)?, self.view(s.gate)?)),
            None => None,
        };
        if let Some((output, gate)) = &views {
            output.expect(routes.tokens, xv.cols, "shared expert output")?;
            gate.expect(routes.tokens, 1, "shared expert gate")?;
        }
        let order = self.view(&routes.order)?;
        let selected = self.view(&routes.selected)?;
        let p = ScatterParams {
            tokens: to_u32(routes.tokens, "tokens")?,
            top_k: to_u32(routes.top_k, "top-k")?,
            width: to_u32(xv.cols, "width")?,
            stride: to_u32(xv.row_stride, "row stride")?,
            shared: u32::from(views.is_some()),
            shared_stride: to_u32(
                views.as_ref().map_or(0, |(o, _)| o.row_stride),
                "shared stride",
            )?,
            gate_stride: to_u32(
                views.as_ref().map_or(0, |(_, g)| g.row_stride),
                "gate stride",
            )?,
        };
        routes
            .tokens
            .checked_mul(xv.cols)
            .ok_or_else(|| invalid("expert scatter overflow"))?;
        let out = self.output(routes.tokens, xv.cols)?;
        self.runtime.dispatch(
            self.pipeline("moe_scatter_f32")?,
            "tur_moe_scatter",
            routes.tokens,
            BLOCK,
            |e| {
                xv.bind(e, 0)?;
                order.bind(e, 1)?;
                selected.bind(e, 2)?;
                // Unread when there is no shared expert; any bound buffer will do.
                let (output, gate) = views.as_ref().map_or((&xv, &xv), |(o, g)| (o, g));
                output.bind(e, 3)?;
                gate.bind(e, 4)?;
                out.bind(e, 5);
                e.set_bytes(6, &p);
                Ok(())
            },
        )?;
        self.finish(out, (routes.tokens, xv.cols))
    }
}
