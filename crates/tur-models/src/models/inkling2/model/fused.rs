//! The decoder on the Metal kernels.
use super::{Executor, Inkling2, InputNorms, Layer, Pass, reference::RESIDUAL_EPS};
use crate::models::paging::Paging;
use candle_core::Tensor;
use tur_core::{Result, execution::invalid};
use tur_metal::kernels::{
    HeadNorm, Heads, Kernels, LogScale, RecurrentSpan, RecurrentSpans, Relative, ShortConv,
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
            let mix = self
                .k
                .message_pair(&read, &messages, self.eps())?
                .matmul(&gates.down.narrow(0, 0, 3)?)?
                .matmul(&gates.up.narrow(0, 0, 3)?)?;
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
        let mix = self
            .k
            .message_pair(h, &message, self.eps())?
            .matmul(&gates.down.narrow(0, 3, 1)?)?
            .matmul(&gates.up.narrow(0, 3, 1)?)?;
        self.k
            .message_gate(h, &message, &mix, &layer.post_norm, self.eps())
    }

    /// One convolved side of the attention: its projection, then the short convolution
    /// reading dynamic taps from the projection's last columns in full attention layers.
    fn side(
        &self,
        input: &Tensor,
        weights: &tur_metal::packed::PackedLinear,
        taps: &Tensor,
        (application, index, column): (usize, usize, usize),
    ) -> Result<Tensor> {
        let t = &self.config().text;
        let width = t.num_key_value_heads * t.head_dim;
        let sliding = self.config().sliding(index);
        let projected = self.k.linear(weights, input)?;
        let taps_count = t.sconv_kernel_size;
        let divisor = if sliding {
            taps_count + 1
        } else {
            2 * taps_count + 1
        };
        self.k.short_conv(
            &projected,
            &self.recurrent,
            ShortConv {
                taps,
                pool: &self.pass.pool.recurrent(2 * application + 1)?.convolution,
                offset: column,
                dynamic: (!sliding).then_some((&projected, width, t.head_dim)),
                divisor: (divisor as f32).sqrt(),
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
        let log_scale = (!sliding).then_some(LogScale {
            positions: &self.positions,
            alpha: t.log_scaling_alpha as f32,
            floor: t.log_scaling_n_floor as f32,
        });
        let clip = |scale, threshold: f64| HeadNorm::Clip {
            scale: Some(scale),
            threshold: threshold as f32,
            log_scale,
        };
        let q = self
            .k
            .head_norm(
                &projected,
                heads(count, dim, 0),
                clip(&a.q_scale, t.qk_l2_clip_threshold),
            )?
            .reshape((inputs[0].dim(0)?, count, dim))?;
        let states = self.k.head_norm(
            &projected,
            heads(count, rel, 2 * count * dim),
            clip(&a.rel_scale, t.rel_l2_clip_threshold),
        )?;
        let key = self.side(
            &inputs[1],
            &a.key,
            &a.key_taps,
            (application, index, columns.key),
        )?;
        let key = self.k.head_norm(
            &key,
            heads(kv, dim, 0),
            HeadNorm::Clip {
                scale: None,
                threshold: t.qk_l2_clip_threshold as f32,
                log_scale: None,
            },
        )?;
        let value = self.side(
            &inputs[2],
            &a.value,
            &a.value_taps,
            (application, index, columns.value),
        )?;
        let value = self.k.head_norm(
            &value,
            heads(kv, dim, 0),
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
            &q,
            store,
            paged,
            Relative {
                rel: &states,
                proj: &a.rel_proj,
                sinks: (&projected, 2 * count * dim + count * rel),
                gate: (&projected, count * dim),
                window: sliding.then_some(t.sliding_window_size),
                scale: (dim as f64).sqrt().recip() as f32,
            },
        )?;
        self.k.linear(&a.output, &attended)
    }

    fn experts(&self, layer: &Layer, h: &Tensor) -> Result<Tensor> {
        let t = &self.config().text;
        let e = &layer.experts;
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

    fn conv(&self, application: usize, x: &Tensor, taps: &Tensor, column: usize) -> Result<Tensor> {
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
        let attended = self.conv(
            application,
            &attended,
            &layer.attention_taps,
            columns.attention,
        )?;
        let residual = if config.boundary(index) {
            blocks.push(x.clone());
            attended
        } else {
            (x + attended)?
        };
        let h = self.residual(blocks, &residual, &layer.mlp_query)?.get(0)?;
        let routed = self.experts(layer, &self.mlp_input(layer, &h, messages)?)?;
        let routed = self.conv(application, &routed, &layer.mlp_taps, columns.mlp)?;
        Ok((residual + routed)?)
    }
}

impl Inkling2 {
    pub(super) fn run_fused(
        &self,
        k: &Kernels,
        pass: &Pass<'_, '_>,
        x: &Tensor,
        trace: Option<&mut Vec<Tensor>>,
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
        self.decode(&fused, x, trace)
    }
}
