# Model and checkpoints

Tur implements Qwen3.5-family hybrid decoders and Gemma 4 text decoders in `tur-models`.
`ModelConfig` selects the model-specific configuration from checkpoint fields;
`TextModel` dispatches execution through the existing `HybridModel` contract.
Both families use the same engine, state pools, memory planner, packed projections,
streaming weight loader, norms, rotary operations and attention kernels.

## Qwen architecture

The decoder stacks two kinds of layers in the order the checkpoint's `layer_types`
lists:

- **Gated DeltaNet (linear attention).** A depthwise causal convolution over the last few
  inputs, then a gated delta-rule recurrence over a per-head key × value matrix. The state
  is the convolution history plus the recurrent matrix, and its size is independent of
  context length.
- **Gated full attention.** Grouped-query attention with partial rotary embeddings
  and a sigmoid output gate. The state is the keys and values of every past token.

Each layer then runs a SwiGLU MLP, or, for MoE checkpoints, routed experts plus a
sigmoid-gated shared expert. Qwen3.5 checkpoints also ship a one-layer
multi-token-prediction (MTP) head, which Tur can load as a drafter (see
[Speculative decoding](speculation.md)).

## Gemma 4 architecture

Gemma 4 interleaves sliding-window and full attention. It uses learned Q/K RMS norms,
unscaled Q·K scores, an unweighted value RMS norm, GELU-gated feed-forward layers,
pre- and post-sublayer norms, a learned layer scalar, scaled embeddings and final
logit softcapping. Global heads use proportional RoPE: the rotated coordinates span
halves of the full head, with zero frequencies for the unrotated pairs.

E2B and E4B add per-layer embeddings (PLE): token-specific layer inputs combine with a
normalized projection of the initial embedding. A gated projection injects each layer's
PLE signal after its feed-forward update. E2B also doubles the MLP width in its K/V-sharing
suffix. Sharing layers reuse the last non-sharing layer of the same attention kind;
only physical K/V owners allocate pool state. Sliding owners keep only the window pages
their later queries can read (see [State memory](memory.md)); the drafter's one extra
key is part of their window.

The 31B and 12B Unified text decoders share the dense equations without PLE. The
26B-A4B branch adds normalized top-k experts, learned per-expert scales and separately
normalized dense/routed branches. Global K=V reuses the raw key projection for values,
before key normalization and RoPE. Vision and audio encoders are not executed.

E2B is the real-checkpoint development model:
`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
The other sizes have configuration and small architecture-fixture coverage; their full
checkpoints have not been executed.

## Two execution paths

- **Fused** is the Metal path used for serving. Every layer runs once over the packed
  batch on Tur's kernels. Activations are F32, weights stay packed, and state is read and
  written in the persistent pools.
- **Reference** is a slow, readable F32 implementation on Candle operations. It is the
  numerical oracle: the fused path is tested against it on identical weights, and the
  reference is tested against Transformers. `--cpu-reference` runs it on the CPU.

## Validation

`ModelConfig::from_json` accepts a checkpoint only through its explicit architecture
identifiers:

| `model_type` | `architectures` |
| --- | --- |
| `qwen3_5` | `Qwen3_5ForConditionalGeneration` (settings in `text_config`) |
| `qwen3_5_text` | `Qwen3_5ForCausalLM` |
| `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` (settings in `text_config`) |
| `qwen3_5_moe_text` | `Qwen3_5MoeForCausalLM` |
| `gemma4` | `Gemma4ForConditionalGeneration` (settings in `text_config`) |
| `gemma4_text` | `Gemma4ForCausalLM` |
| `gemma4_unified` | `Gemma4UnifiedForConditionalGeneration` (settings in `text_config`) |
| `gemma4_unified_text` | `Gemma4UnifiedForCausalLM` |

The Qwen parser rejects settings outside its implemented equations:

- quantization configs, RoPE scaling and sliding windows;
- MoE settings on dense architectures, unnormalized routing and MLP-only layers;
- activations other than SiLU, attention bias or dropout, and non-swish output gates;
- non-F32 recurrent state dtypes;
- inconsistent dimensions or head counts, invalid partial-rotary settings, and M-RoPE
  layouts other than interleaved text-only sections.

Gemma validates its own head widths, sharing topology, PLE and expert dimensions,
GELU activation, causal text attention, default local RoPE and proportional global RoPE.
Attention biases, dropout, arbitrary RoPE scaling, bidirectional text attention and
external quantization configurations are rejected. Gemma 4 drafts with its published
assistant (`gemma4_assistant`), a separate checkpoint validated against its target:
same vocabulary, head widths, K/V heads, window and rotary settings, and every layer
sharing target K/V (see [Speculative decoding](speculation.md)). Gemma artifacts are
written with round-to-nearest packing (see [Prepacked weights](../guide/quantization.md)).

Every weight shape is derived and checked before any payload is read. SafeTensors files
are discovered from the index, their headers validated, and payloads read without mmap.
For a remote model, `config.json` is fetched and validated before any weight file is
downloaded, and the repository must report exactly the requested commit.

## Tokens and templates

- **EOS** comes from `generation_config.json`, the config, and the tokenizer's
  `eos_token`; any of them ends generation.
- **Chat template.** The checkpoint's Jinja template (`chat_template.jinja`, or the one in
  `tokenizer_config.json`) renders every request with MiniJinja. Thinking controls are
  template variables. Nothing is appended to prompts, and a missing or invalid template
  fails loading. Tokenizer special-token variables, including Gemma's `bos_token`,
  are passed to the template; Python mapping `get` and string methods retain their semantics.
- **Tools.** Function definitions are passed to the template. Generated Qwen XML tool
  calls are parsed with argument types taken from each tool's JSON schema. Gemma
  `call:name{...}` blocks use their typed argument grammar. Tool parsing decodes protocol
  tokens separately, so special delimiters survive without appearing in user text.
- **Text only.** Image, video, audio and vision-boundary token IDs from the config are
  forbidden in prompts, and multimodal content is rejected.

## Weights

Projections are stored as F32, Q8_0 or Q4K blocks (`PackedLinear`). Embedding tables can
also be BF16, Q6K or Q5K, or gathered from the packed LM head on tied checkpoints. See
[Prepacked weights](../guide/quantization.md) for the formats and the offline
quantizer. Gemma supports load-time F32/Q8_0/Q4K conversion where matrix columns fit
the block format; the 26B expert width supports Q8_0 but not Q4K. PLE tables stay BF16
in packed modes and F32 in uniform F32 mode.

## Decisions

- **The architecture comes from checkpoint fields, never from the model's name.**
  Mapping every "qwen3" name to one decoder would silently run the wrong mathematics.
- **Correctness comes from pinned external references.** The 0.8B checkpoint matches
  Transformers 5.20.2 (CPU, F32) per layer and in final logits, at relative error
  ≤ 3.5e-6 against a 1e-4 gate. Whole-prompt, chunked and incremental execution agree,
  and so do sequences entering and leaving a batch. Qwen3.8-27B is validated against its
  pinned config, tensor headers and template, but has not been executed. The MoE path
  passes an independent small Transformers fixture.
- **Ordinary Qwen3 and Granite were deleted**, not kept as regression models, together
  with their engine, cache and the `/think` prompt heuristic.
- **Image, video and audio are rejected** rather than silently dropped, until their
  encoders and position handling exist.
- **Recurrent accumulation stays F32.** The checkpoint's state dtype is F32, and
  quantization error in recurrent state persists through every later update.
- **Share execution infrastructure across model families.** Model-specific equations
  remain in each adapter; weight plans, packing, tokenizer metadata and state ownership
  have one common implementation.
- **Gemma conformance and quantization quality are separate gates.** Small pinned
  Transformers 5.20.2 F32 fixtures cover dense PLE/sharing, MoE and Unified equations
  per layer and in logits on CPU and Metal at 1e-4 relative error. The real E2B Q8_0
  checkpoint passes the existing 95% next-token agreement floor and greedy-prefix /
  termination checks. This is a short conformance case, not a held-out quality study.
- **Run Gemma on Qwen's fused primitives.** One `q | k | v` and one `gate | up` matrix per
  layer, in-place query/key/value norms and rotary from precomputed frequencies, a strided
  GELU product and residual updates fused with the layer scalar and the next norm. On the
  M1 (E2B, Q8_0, F16 K/V, two alternating rounds) 512-token prefill rose from 323 to
  417 tokens/s (1.29×), decode at 128 tokens from 19.6 to 20.6 (one sequence) and 48.6
  to 51.8 tokens/s (four), and engine latency fell from 50.7 to 48.0 ms per token.
- **Keep PLE buffers bounded.** Store each layer's columns separately and fence streamed
  uploads before staging the next chunk; E2B's BF16 PLE table alone is 4.375 GiB.
