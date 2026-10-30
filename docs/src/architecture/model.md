# Model and checkpoints

`tur-models` implements Qwen3.5-family hybrid decoders, Gemma 4 decoders with Gemma's audio encoder, and Whisper for transcription. `ModelConfig` picks the model from checkpoint fields, and `TextModel` runs it through the shared `HybridModel` contract. Both families share the engine, state pools, memory planner, packed projections, weight loader, norms, rotary embeddings and attention kernels, so an improvement to one of those helps both.

## Qwen: a hybrid of recurrence and attention

Most of a Qwen3.5 model's layers are Gated DeltaNet layers, a form of linear attention. Instead of remembering every past token, a DeltaNet layer keeps a fixed-size matrix per head that it updates with each new token. Its state is the same size at token 10 and token 100,000, which is why these models handle long contexts cheaply. Every few layers, a full attention layer keeps the keys and values of every past token for precise recall. The checkpoint's `layer_types` lists the order.

- A Gated DeltaNet layer runs a short causal convolution over its last few inputs, then a gated delta-rule update of its key-by-value matrix. Its state is the convolution history plus that matrix.
- A gated full attention layer is grouped-query attention with partial rotary embeddings and a sigmoid output gate. Its state is the keys and values of every past token.

Each layer then runs a SwiGLU MLP or, in MoE checkpoints, routed experts plus a gated shared expert. Qwen3.5 checkpoints, dense and MoE alike, also ship a one-layer multi-token-prediction (MTP) head, which Tur can use to draft tokens (see [Speculative decoding](speculation.md)).

This mix is why Tur pages attention state but keeps recurrent state in slots. It's also why prefix reuse has to snapshot exact token boundaries: unlike a key-value cache, a recurrent state can't be cut back to an earlier token.

## Gemma 4

Gemma 4 alternates sliding-window attention, where each token sees only the last few hundred tokens, with full attention. It uses learned Q/K norms, unscaled attention scores, a value norm, GELU-gated feed-forward layers, norms before and after each sublayer, a learned per-layer scale, scaled embeddings and a soft cap on the final logits. Its global heads use proportional RoPE, which rotates only part of each head.

E2B and E4B add per-layer embeddings (PLE): each layer gets an extra token-specific input, injected through a gated projection after its feed-forward step. The last layers of E2B reuse the keys and values of an earlier layer of the same kind instead of computing their own, so only the layers that produce K/V get memory in the pool. Sliding layers keep only the pages their window can still reach (see [State memory](memory.md)).

The 31B and 12B Unified models use the same equations without PLE. The 26B-A4B model adds normalized top-k experts with learned per-expert scales. Its global layers reuse the raw key projection as values. E2B and E4B also carry an audio encoder, which [Audio](audio.md) describes; the vision encoder is not run.

E2B (`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`) is the checkpoint used in development. The other sizes are covered by configuration checks and small architecture fixtures, but their full checkpoints haven't been run.

## Two execution paths

The fused path is what serves requests. Each layer runs once over the whole packed batch on Tur's Metal kernels, activations are F32, weights stay packed, and state is read and written in the persistent pools.

The reference path is a slow, readable F32 implementation built from Candle operations. It is the numerical oracle: the fused path is tested against it on identical weights, and it is tested against Transformers. `--cpu-reference` runs it on the CPU.

## Validation

`ModelConfig::from_json` accepts a checkpoint only through its explicit architecture identifiers:

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
| `gpt_oss` | `GptOssForCausalLM` |

The Qwen parser rejects any setting Tur doesn't implement, rather than running the wrong maths:

- quantization configs, RoPE scaling and sliding windows;
- MoE settings on dense architectures, unnormalized routing and MLP-only layers;
- activations other than SiLU, attention bias or dropout, and output gates other than swish;
- recurrent state dtypes other than F32;
- inconsistent dimensions or head counts, invalid partial-rotary settings, and M-RoPE layouts other than interleaved text-only sections.

Gemma checks its own head widths, K/V sharing layout, PLE and expert dimensions, activation, attention mask and RoPE settings, and rejects attention biases, dropout, arbitrary RoPE scaling, bidirectional text attention and external quantization configs. Gemma's assistant drafter (`gemma4_assistant`) is a separate checkpoint, checked against its target model: same vocabulary, head widths, K/V heads, window and rotary settings (see [Speculative decoding](speculation.md)).

Tur derives and checks every weight shape before reading any data. It finds SafeTensors files through the index, validates their headers, and reads payloads without memory-mapping. For a remote model, it downloads and validates `config.json` before any weights, and the repository must report exactly the requested commit.

## Tokens and templates

- Any end-of-sequence token named in `generation_config.json`, the config, or the tokenizer's `eos_token` ends generation.
- The checkpoint's own Jinja chat template (`chat_template.jinja`, or the one in `tokenizer_config.json`) renders every request through MiniJinja. Thinking controls are template variables, and Tur never appends anything to the prompt. A missing or invalid template fails at load time. Tokenizer special tokens, such as Gemma's `bos_token`, are passed to the template, and Python-style `get` and string methods behave as in Transformers.
- Function definitions go to the template. Tur parses Qwen's XML tool calls using argument types from each tool's JSON schema, and Gemma's `call:name{...}` blocks with their own typed grammar. Protocol tokens are decoded separately, so special delimiters never leak into user-visible text.
- Image, video and vision-boundary token IDs from the config are rejected in prompts, and so are audio placeholders unless the model's audio encoder is loaded. With the encoder, each audio part must render exactly one placeholder.

## Weights

Projections are stored as F32, Q8_0 or Q4K blocks (`PackedLinear`). Embedding tables can also be BF16, Q6K or Q5K, or read from the packed output layer on checkpoints that share it. [Prepacked weights](../guide/quantization.md) covers the formats and the offline quantizer. Gemma converts to F32, Q8_0 or Q4K at load time where a matrix's width fits the block format; the 26B expert width supports Q8_0 but not Q4K. Gemma's PLE tables stay BF16 in packed modes.

## Decisions

- The architecture comes from checkpoint fields, never from the model's name. Mapping every "qwen3" name to one decoder would silently run the wrong maths.
- Correctness comes from pinned external references. The 0.8B checkpoint matches Transformers 5.20.2 (CPU, F32) per layer and in final logits within 3.5e-6 relative error, against a 1e-4 gate. Whole-prompt, chunked and incremental execution agree, and so do requests joining and leaving a batch. Qwen3.8-27B is validated against its pinned config, tensor headers and template, but hasn't been run. The MoE path passes an independent small Transformers fixture.
- Image and video inputs are rejected, not silently dropped, until their encoders and position handling exist. Audio reaches Gemma 4 E2B and E4B through their encoder (see [Audio](audio.md)).
- Recurrent accumulation stays F32. The checkpoint's state dtype is F32, and rounding error in recurrent state carries into every later token.
- Model families share execution infrastructure. Each adapter keeps its own equations, while weight plans, packing, tokenizer metadata and state ownership have one implementation.
- Gemma conformance and quantization quality are separate gates. Small pinned Transformers F32 fixtures cover the dense PLE and sharing, MoE and Unified equations per layer and in logits on CPU and Metal at 1e-4 relative error. The real E2B Q8_0 checkpoint passes the 95% next-token agreement floor plus greedy-prefix and termination checks. That is a short conformance check, not a held-out quality study.
- Gemma runs on Qwen's fused building blocks: one `q | k | v` and one `gate | up` matrix per layer, in-place norms and rotary embeddings, and residual updates fused with the layer scale and the next norm. On E2B that made 512-token prefill 1.29× faster and cut engine latency per token by 5%.
- Gemma's PLE table (4.4 GiB on E2B) is stored per layer and uploaded in bounded chunks, so loading never needs one multi-gigabyte buffer.
