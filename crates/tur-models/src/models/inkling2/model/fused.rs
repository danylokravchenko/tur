//! The decoder on the Metal kernels.
use super::{Executor, Inkling2, InputNorms, Layer, Pass, reference::RESIDUAL_EPS};
use crate::models::paging::Paging;
use candle_core::Tensor;
use tur_core::{Result, execution::invalid};
use tur_metal::kernels::{
    Clipped, HeadNorm, Heads, Kernels, LogScale, RecurrentSpan, RecurrentSpans, Relative,
    SMALL_EXPERT_TOKENS, ShortConv, SmallExperts,
};

/// Addressing and per-pass inputs every layer shares.
pub(super) struct Fused<'a, 'b> {
    model: &'a Inkling2,
    k: &'a Kernels,
    pass: &'a Pass<'a, 'b>,
    paging: Paging,
    recurrent: RecurrentSpans,
    /// `[tokens]` U32 positions, for log scaling.
    positions: Tensor,
}

fn heads(count: usize, dim: usize, col_offset: usize) -> Heads {
    Heads {
        count,
        dim,
        head_stride: dim,
        col_offset,
    }
}

impl Fused<'_, '_> {
    fn config(&self) -> &super::Config {
        &self.model.config
    }

    fn eps(&self) -> f32 {
        self.config().text.rms_norm_eps as f32
    }

    /// The normalized query, key and value inputs. A looped layer mixes the previous pass's
    /// messages into all three with one pair kernel, one batched low-rank product and one
    /// kernel that applies the gates and the norms.
    fn inputs(
        &self,
        layer: &Layer,
        x: &Tensor,
        blocks: &[Tensor],
        messages: Option<&Tensor>,
    ) -> Result<[Tensor; 3]> {
        let (tokens, hidden) = x.dims2()?;
        // `[tokens, hidden]` rows every input shares, or `[3 * tokens, hidden]`.
        let read = match &layer.attention_queries {
            None => x.clone(),
            Some(queries) => {
                let read = self.residual(blocks, x, queries)?;
                read.reshape((read.dim(0)? * tokens, hidden))?
            }
        };
        let input = |t: &Tensor, i: usize| -> Result<Tensor> {
            if t.dim(0)? == tokens {
                Ok(t.clone())
            } else {
                Ok(t.narrow(0, i * tokens, tokens)?)
            }
        };
        if let (Some(messages), Some(gates)) = (messages, &layer.gates) {
            let norms = match &layer.norms {
                InputNorms::Shared(norm) => norm.unsqueeze(0)?,
                InputNorms::Separate(norms) => norms.clone(),
            };
            let messages = messages.narrow(0, 0, 3)?;
            let pair = self.k.message_pair(&read, &messages, self.eps())?;
            let low = self
                .k
                .gate_projection(&pair, &gates.down.narrow(0, 0, 3)?)?;
            let mix = self.k.gate_projection(&low, &gates.up.narrow(0, 0, 3)?)?;
            let out = self
                .k
                .message_gate(&read, &messages, &mix, &norms, self.eps())?;
            return Ok([input(&out, 0)?, input(&out, 1)?, input(&out, 2)?]);
        }
        Ok(match &layer.norms {
            InputNorms::Shared(norm) => {
                let normed = self.norm(&read, norm)?;
                [normed.clone(), normed.clone(), normed]
            }
            InputNorms::Separate(norms) => [
                self.norm(&input(&read, 0)?, &norms.get(0)?)?,
                self.norm(&input(&read, 1)?, &norms.get(1)?)?,
                self.norm(&input(&read, 2)?, &norms.get(2)?)?,
            ],
        })
    }

    /// The experts' normalized input: the MLP message mixed in first in looped layers.
    fn mlp_input(&self, layer: &Layer, h: &Tensor, messages: Option<&Tensor>) -> Result<Tensor> {
        let (Some(messages), Some(gates)) = (messages, &layer.gates) else {
            return self.norm(h, &layer.post_norm);
        };
        let message = messages.narrow(0, 3, 1)?;
        let pair = self.k.message_pair(h, &message, self.eps())?;
        let low = self
            .k
            .gate_projection(&pair, &gates.down.narrow(0, 3, 1)?)?;
        let mix = self.k.gate_projection(&low, &gates.up.narrow(0, 3, 1)?)?;
        self.k
            .message_gate(h, &message, &mix, &layer.post_norm, self.eps())
    }

    /// One convolved side of the attention from its `projected` rows: the short
    /// convolution, reading dynamic taps from the rows' last columns in full attention
    /// layers, with each head normalized as `norm` says in the same pass.
    fn side(
        &self,
        projected: &Tensor,
        taps: &Tensor,
        (application, index, column): (usize, usize, usize),
        norm: HeadNorm,
    ) -> Result<Tensor> {
        let t = &self.config().text;
        let width = t.num_key_value_heads * t.head_dim;
        let sliding = self.config().sliding(index);
        let taps_count = t.sconv_kernel_size;
        let divisor = if sliding {
            taps_count + 1
        } else {
            2 * taps_count + 1
        };
        self.k.short_conv(
            projected,
            &self.recurrent,
            ShortConv {
                taps,
                pool: &self.pass.pool.recurrent(2 * application + 1)?.convolution,
                offset: column,
                dynamic: (!sliding).then_some((projected, width, t.head_dim)),
                divisor: (divisor as f32).sqrt(),
                residual: None,
                norm: Some((t.head_dim, norm)),
            },
        )
    }

    fn attention(&self, application: usize, index: usize, inputs: &[Tensor; 3]) -> Result<Tensor> {
        let config = self.config();
        let t = &config.text;
        let a = &self.model.layers[index].attention;
        let (count, dim, rel, kv) = (
            t.num_attention_heads,
            t.head_dim,
            t.d_rel,
            t.num_key_value_heads,
        );
        let sliding = config.sliding(index);
        let columns = config.conv_columns();
        let projected = self.k.linear(&a.query, &inputs[0])?;
        let keys = self.k.linear(&a.key, &inputs[1])?;
        let values = self.k.linear(&a.value, &inputs[2])?;
        let log_scale = (!sliding).then_some(LogScale {
            positions: &self.positions,
            alpha: t.log_scaling_alpha as f32,
            floor: t.log_scaling_n_floor as f32,
        });
        // The attention clips the queries and relative states as it reads them.
        let clip = |col, scale, threshold: f64| Clipped {
            rows: &projected,
            col,
            scale,
            threshold: threshold as f32,
        };
        let key = self.side(
            &keys,
            &a.key_taps,
            (application, index, columns.key),
            HeadNorm::Clip {
                threshold: t.qk_l2_clip_threshold as f32,
            },
        )?;
        let value = self.side(
            &values,
            &a.value_taps,
            (application, index, columns.value),
            HeadNorm::Rms {
                eps: t.rms_norm_eps as f32,
            },
        )?;
        let store = super::attention_store(self.pass.pool, application)?;
        let paged = self.paging.layer(sliding)?;
        self.k.kv_write(
            &key,
            heads(kv, dim, 0),
            &value,
            heads(kv, dim, 0),
            store,
            paged,
        )?;
        let attended = self.k.relative_attention(
            (clip(0, &a.q_scale, t.qk_l2_clip_threshold), count),
            store,
            paged,
            Relative {
                rel: clip(2 * count * dim, &a.rel_scale, t.rel_l2_clip_threshold),
                log_scale,
                proj: &a.rel_proj,
                sinks: (&projected, 2 * count * dim + count * rel),
                gate: (&projected, count * dim),
                window: sliding.then_some(t.sliding_window_size),
                scale: (dim as f64).sqrt().recip() as f32,
            },
        )?;
        self.k.linear(&a.output, &attended)
    }

    /// The routed experts of `h`: three fused dispatches for decode-sized passes, the
    /// grouped kernels (which read each selected expert once for all its rows) beyond.
    fn experts(&self, layer: &Layer, h: &Tensor) -> Result<Tensor> {
        let t = &self.config().text;
        let e = &layer.experts;
        if h.dim(0)? <= SMALL_EXPERT_TOKENS {
            return self.k.small_experts(
                h,
                SmallExperts {
                    router: &e.router,
                    bias: &e.bias,
                    gate_up: &e.gate_up,
                    down: &e.down,
                    factors: &e.factors,
                    top_k: t.num_experts_per_tok,
                    scale: t.route_scale as f32,
                    caps: (t.mlp_gate_softcap as f32, t.mlp_up_softcap as f32),
                },
            );
        }
        let logits = h.matmul(&e.router.t()?)?;
        let routes = self.k.route_sigmoid(
            &logits,
            t.num_experts_per_tok,
            &e.bias,
            t.route_scale as f32,
        )?;
        let gate_up = self.k.expert_linear_routed(&e.gate_up, h, &routes)?;
        let act = self.k.expert_softcap_glu(
            &gate_up,
            &e.factors,
            &routes,
            (t.mlp_gate_softcap as f32, t.mlp_up_softcap as f32),
        )?;
        let out = self.k.expert_linear(&e.down, &act, &routes)?;
        self.k.expert_scatter(&out, &routes)
    }

    /// The short convolution after a sublayer, plus `residual` when given.
    fn conv(
        &self,
        application: usize,
        x: &Tensor,
        (taps, column): (&Tensor, usize),
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let count = self.config().text.sconv_kernel_size;
        self.k.short_conv(
            x,
            &self.recurrent,
            ShortConv {
                taps,
                pool: &self.pass.pool.recurrent(2 * application + 1)?.convolution,
                offset: column,
                dynamic: None,
                divisor: ((count + 1) as f32).sqrt(),
                residual,
                norm: None,
            },
        )
    }
}

impl Executor for Fused<'_, '_> {
    fn norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        self.k.rms_norm(x, weight, self.config().text.rms_norm_eps)
    }

    fn residual(&self, blocks: &[Tensor], partial: &Tensor, queries: &Tensor) -> Result<Tensor> {
        self.k
            .block_residual(blocks, partial, queries, RESIDUAL_EPS as f32)
    }

    fn layer(
        &self,
        application: usize,
        index: usize,
        x: &Tensor,
        blocks: &mut Vec<Tensor>,
        messages: Option<&Tensor>,
    ) -> Result<Tensor> {
        let config = self.config();
        let layer = &self.model.layers[index];
        let columns = config.conv_columns();
        let inputs = self.inputs(layer, x, blocks, messages)?;
        let attended = self.attention(application, index, &inputs)?;
        // A block boundary restarts the residual stream; elsewhere the convolution adds it.
        let boundary = config.boundary(index);
        let residual = self.conv(
            application,
            &attended,
            (&layer.attention_taps, columns.attention),
            (!boundary).then_some(x),
        )?;
        if boundary {
            blocks.push(x.clone());
        }
        let h = self.residual(blocks, &residual, &layer.mlp_query)?.get(0)?;
        let routed = self.experts(layer, &self.mlp_input(layer, &h, messages)?)?;
        self.conv(
            application,
            &routed,
            (&layer.mlp_taps, columns.mlp),
            Some(&residual),
        )
    }
}

impl Inkling2 {
    pub(super) fn run_fused(
        &self,
        k: &Kernels,
        pass: &Pass<'_, '_>,
        x: &Tensor,
        capture: super::Capture<'_>,
    ) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let paging = Paging::new(
            k,
            &self.description,
            pass.pool,
            pass.spans.iter().map(|s| (&s.addresses, s.past, s.len)),
            tokens,
        )?;
        let recurrent = k.recurrent_spans(
            &pass
                .spans
                .iter()
                .map(|s| RecurrentSpan {
                    tokens: s.len,
                    position: s.past,
                    read: s.addresses.read,
                    writes: &s.addresses.writes,
                    ends: &s.ends,
                })
                .collect::<Vec<_>>(),
        )?;
        let positions = pass
            .positions
            .iter()
            .map(|&p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
            .collect::<Result<Vec<_>>>()?;
        let fused = Fused {
            model: self,
            k,
            pass,
            paging,
            recurrent,
            positions: Tensor::new(positions.as_slice(), &self.device)?,
        };
        self.decode(&fused, x, capture)
    }
}
