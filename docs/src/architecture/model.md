# Model and checkpoints

Tur implements one model family: the Qwen3.5 hybrid decoder, used by Qwen3.5, Qwen3.8
(dense) and Qwen3.6 (mixture-of-experts). The code is in `tur-models`.

## Architecture

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

## Two execution paths

- **Fused** is the Metal path used for serving. Every layer runs once over the packed
  batch on Tur's kernels. Activations are F32, weights stay packed, and state is read and
  written in the persistent pools.
- **Reference** is a slow, readable F32 implementation on Candle operations. It is the
  numerical oracle: the fused path is tested against it on identical weights, and the
  reference is tested against Transformers. `--cpu-reference` runs it on the CPU.

## Validation

`Config::from_json` accepts a checkpoint only through its explicit architecture
identifiers:

| `model_type` | `architectures` |
| --- | --- |
| `qwen3_5` | `Qwen3_5ForConditionalGeneration` (settings in `text_config`) |
| `qwen3_5_text` | `Qwen3_5ForCausalLM` |
| `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` (settings in `text_config`) |
| `qwen3_5_moe_text` | `Qwen3_5MoeForCausalLM` |

It then rejects anything Tur does not implement:

- quantization configs, RoPE scaling and sliding windows;
- MoE settings on dense architectures, unnormalized routing and MLP-only layers;
- activations other than SiLU, attention bias or dropout, and non-swish output gates;
- non-F32 recurrent state dtypes;
- inconsistent dimensions or head counts, invalid partial-rotary settings, and M-RoPE
  layouts other than interleaved text-only sections.

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
  fails loading.
- **Tools.** Function definitions are passed to the template. Generated Qwen XML tool
  calls are parsed with argument types taken from each tool's JSON schema.
- **Text only.** Image, video, audio and vision-boundary token IDs from the config are
  forbidden in prompts, and multimodal content is rejected.

## Weights

Projections are stored as F32, Q8_0 or Q4K blocks (`PackedLinear`). Embedding tables can
also be BF16, Q6K or Q5K, or gathered from the packed LM head on tied checkpoints. See
[Prepacked weights](../guide/quantization.md) for the formats and the offline
quantizer.

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
