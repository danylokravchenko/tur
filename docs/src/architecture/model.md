# Model and checkpoints

`tur-models` implements Qwen3.5-family hybrid decoders, Gemma 4 decoders with Gemma's audio encoder, gpt-oss and Inkling 2's looped decoder, vision encoders for Qwen and Gemma, and Whisper for transcription. `ModelConfig` picks the model from checkpoint fields, and `TextModel` runs it through the shared `HybridModel` contract. Both families share the engine, state pools, memory planner, packed projections, weight loader, norms, rotary embeddings and attention kernels, so an improvement to one of those helps both.

## Qwen: a hybrid of recurrence and attention

Most of a Qwen3.5 model's layers are Gated DeltaNet layers, a form of linear attention. Instead of remembering every past token, a DeltaNet layer keeps a fixed-size matrix per head that it updates with each new token. Its state is the same size at token 10 and token 100,000, which is why these models handle long contexts cheaply. Every few layers, a full attention layer keeps the keys and values of every past token for precise recall. The checkpoint's `layer_types` lists the order.

- A Gated DeltaNet layer runs a short causal convolution over its last few inputs, then a gated delta-rule update of its key-by-value matrix. Its state is the convolution history plus that matrix.
- A gated full attention layer is grouped-query attention with partial rotary embeddings and a sigmoid output gate. Its state is the keys and values of every past token.

Each layer then runs a SwiGLU MLP or, in MoE checkpoints, routed experts plus a gated shared expert. Qwen3.5 checkpoints, dense and MoE alike, also ship a one-layer multi-token-prediction (MTP) head, which Tur can use to draft tokens (see [Speculative decoding](speculation.md)).

This mix is why Tur pages attention state but keeps recurrent state in slots. It's also why prefix reuse has to snapshot exact token boundaries: unlike a key-value cache, a recurrent state can't be cut back to an earlier token.

## Gemma 4

Gemma 4 alternates sliding-window attention, where each token sees only the last few hundred tokens, with full attention. It uses learned Q/K norms, unscaled attention scores, a value norm, GELU-gated feed-forward layers, norms before and after each sublayer, a learned per-layer scale, scaled embeddings and a soft cap on the final logits. Its global heads use proportional RoPE, which rotates only part of each head.

E2B and E4B add per-layer embeddings (PLE): each layer gets an extra token-specific input, injected through a gated projection after its feed-forward step. The last layers of E2B and E4B reuse the keys and values of an earlier layer of the same kind instead of computing their own, so only the layers that produce K/V get memory in the pool. Sliding layers keep only the pages their window can still reach (see [State memory](memory.md)). Because the sharing layers store nothing, a prompt row whose output nobody reads can stop at the last layer that produces K/V. In a prefill chunk that is every row but the last, so on E2B most prompt tokens skip 20 of its 35 layers, the ones with double-width feed-forward.

The 31B and 12B Unified models use the same equations without PLE. The 26B-A4B model adds normalized top-k experts with learned per-expert scales. Its global layers reuse the raw key projection as values. E2B and E4B also carry an audio encoder, which [Audio](audio.md) describes; the vision encoder is not run.

E2B (`google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`) is the checkpoint used in development. The other sizes are covered by configuration checks and small architecture fixtures, but their full checkpoints haven't been run.

## Inkling 2

Inkling 2 Mini is a looped mixture-of-experts decoder. It stores 24 layers, but the middle 12 run three times with shared weights, so each token passes through 48 layer applications. Every pass after the first restarts from the state before the loop and reads messages from the pass before it: per looped layer, four learned queries attend over that pass's block states and output, and small tanh gates mix the results into the layer's query, key, value and expert inputs. Each application attends over its own earlier tokens, so each keeps its own keys, values and convolution history.

Residual sums stop at block boundaries, every six layers. A boundary layer's input becomes a block state and its residual stream restarts. Each sublayer reads its input by attention over the block states and the current stream: a learned query scores every RMS-normalized state and the softmax mixes them. Full attention layers read separate query, key and value inputs this way.

Attention has no rotary embedding. A 16-dimensional relative state per token and head, times a learned table, adds a bias to each score by backward distance, up to 1,024 tokens. Keys and values first pass through a four-tap causal convolution (in full layers each token adds its own taps), queries and keys are L2-clipped, values are RMS-normalized, each token adds a sink logit per head, and a sigmoid gate scales the output. Full layers also scale queries and relative states up logarithmically past position 8,192, and sliding layers see 1,024 tokens. A second convolution follows the attention and another the experts. Each token routes to 8 of 128 experts by sigmoid score plus a bias that only steers selection, and the experts' SwiGLU soft-caps its gate and up halves. The final logits are soft-capped too, over the unembedding's 200,058 real rows.

The experts of layers 0 to 19 are NVFP4: two E2M1 values per byte, an E4M3 scale per 16 values and a factor per expert. Layers 20 to 23 are MXFP8, and `hf_quant_config.json` names each format. The experts stay packed on the GPU, about 15.7 GB of the 16.7 GB a Q8_0 load holds. In the state pools each application owns two layers: paged K/V, and a recurrent slot whose convolution history holds the key, value, attention and expert convolutions side by side. Tur reads the text model and the block drafter in `mtp.safetensors` (see [Speculative decoding](speculation.md)); the vision tower is not run.

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
| `inkling2_loop` | `Inkling2LoopForConditionalGeneration` (settings in `text_config`) |

The Qwen parser rejects any setting Tur doesn't implement, rather than running the wrong maths:

- quantization configs, RoPE scaling and sliding windows;
- MoE settings on dense architectures, unnormalized routing and MLP-only layers;
- activations other than SiLU, attention bias or dropout, and output gates other than swish;
- recurrent state dtypes other than F32;
- inconsistent dimensions or head counts, invalid partial-rotary settings, and M-RoPE layouts other than interleaved text-only sections.

Gemma checks its own head widths, K/V sharing layout, PLE and expert dimensions, activation, attention mask and RoPE settings, and rejects attention biases, dropout, arbitrary RoPE scaling, bidirectional text attention and external quantization configs. Gemma's assistant drafter (`gemma4_assistant`) is a separate checkpoint, checked against its target model: same vocabulary, head widths, K/V heads, window and rotary settings (see [Speculative decoding](speculation.md)).

Inkling 2 checks the loop and block layout against Transformers' own rules, takes sliding layers from `local_layer_ids` and requires them to agree with any `layer_types`, and rejects biases, varying loop counts, shared experts and head layouts that differ between sliding and full layers.

Tur derives and checks every weight shape before reading any data, for GGUF files too, whose tensors each model maps onto its checkpoint names, undoing what llama.cpp's converter changed (RMS norms store `weight + 1` and DeltaNet decay stores `-exp(A_log)`). It finds SafeTensors files through the index, validates their headers, and reads payloads without memory-mapping. For a remote model, it downloads and validates `config.json` before any weights, and the repository must report exactly the requested commit.

## Tokens and templates

- Any end-of-sequence token named in `generation_config.json`, the config, or the tokenizer's `eos_token` ends generation.
- The checkpoint's own Jinja chat template (`chat_template.jinja`, or the one in `tokenizer_config.json`) renders every request through MiniJinja. Thinking controls are template variables, and Tur never appends anything to the prompt. A missing or invalid template fails at load time. Tokenizer special tokens, such as Gemma's `bos_token`, are passed to the template, and Python-style `get` and string methods, and `tojson` with `sort_keys` and `separators`, behave as in Transformers. Hugging Face's `{% generation %}` blocks, which only mark assistant text for training, render their contents.
- Function definitions go to the template. Tur parses Qwen's XML tool calls using argument types from each tool's JSON schema, including `type` lists and `anyOf` or `oneOf` alternatives, and Gemma's `call:name{...}` blocks with their own typed grammar.
- Calls are split from the answer by token. Every token from a call's opening marker to its closing one goes to the call, and only those spans are parsed, with special tokens kept. Qwen's `<tool_call>` isn't a special token, so decoding the text would keep its markup, and Gemma's markers are, so decoding would drop only the markers and leave the call's body. A span that doesn't parse as an offered call rejoins the answer as text. Reasoning is split the same way: Qwen's from the prompt's open `<think>` to `</think>`, and Gemma's thought channel from its markers, so neither reaches the answer.
- Inkling 2 and gpt-oss write their reasoning, answer and tool calls as separate messages, which Tur splits by their special tokens. Inkling 2's token that opens a message's body names its kind: `<|content_thinking|>` is reasoning, `<|content_text|>` the answer, and `<|content_invoke_tool_json|>` a call to the tool its header names. gpt-oss names the channel in a header before `<|message|>`. Output that never opens a message reaches the answer unchanged. Tur rejects structured output and tool grammars for Inkling 2.
- Image, video and vision-boundary token IDs from the config are rejected in prompts, and so are audio placeholders, unless the matching encoder is loaded. With an encoder, each audio, image or video part must render exactly one placeholder.

## Weights

Projections are stored as F32, Q8_0 or Q4K blocks (`PackedLinear`), or in whatever mix of formats a [GGUF file](../guide/gguf.md) holds. Embedding tables can also be BF16, Q6K or Q5K, or read from the packed output layer on checkpoints that share it. [Prepacked weights](../guide/quantization.md) covers the formats and the offline quantizer. Gemma converts to F32, Q8_0 or Q4K at load time where a matrix's width fits the block format; the 26B expert width supports Q8_0 but not Q4K. Gemma's PLE tables stay BF16 in packed modes. Expert banks in MXFP4, NVFP4 or MXFP8 are uploaded as stored, values then scale bytes, and only the expert kernels read them.

## Decisions

- The architecture comes from checkpoint fields, never from the model's name. Mapping every "qwen3" name to one decoder would silently run the wrong maths.
- Correctness comes from pinned external references. The 0.8B checkpoint matches Transformers 5.20.2 (CPU, F32) per layer and in final logits within 3.5e-6 relative error, against a 1e-4 gate. Whole-prompt, chunked and incremental execution agree, and so do requests joining and leaving a batch. Qwen3.8-27B is validated against its pinned config, tensor headers and template, but hasn't been run. The MoE path passes an independent small Transformers fixture.
- Images and video reach Qwen3.5 and Gemma 4 through their vision encoders (see [Images and video](vision.md)), and audio reaches Gemma 4 E2B and E4B through theirs (see [Audio](audio.md)).
- Recurrent accumulation stays F32. The checkpoint's state dtype is F32, and rounding error in recurrent state carries into every later token.
- Model families share execution infrastructure. Each adapter keeps its own equations, while weight plans, packing, tokenizer metadata and state ownership have one implementation.
- Gemma conformance and quantization quality are separate gates. Small pinned Transformers F32 fixtures cover the dense PLE and sharing, MoE and Unified equations per layer and in logits on CPU and Metal at 1e-4 relative error. The real E2B Q8_0 checkpoint passes the 95% next-token agreement floor plus greedy-prefix and termination checks. That is a short conformance check, not a held-out quality study.
- Gemma runs on Qwen's fused building blocks: one `q | k | v` and one `gate | up` matrix per layer, in-place norms and rotary embeddings, and residual updates fused with the layer scale and the next norm. On E2B that made 512-token prefill 1.29× faster and cut engine latency per token by 5%.
- Prompt rows that need no output skip Gemma's K/V-sharing layers, as in vLLM's KV-sharing fast prefill. On E2B with Q8_0 weights, 512-token prefill went from 429 to 1192 tokens/s and 2048-token prefill from 351 to 1012, while decode, where every row needs logits, stayed within noise. The sharing layers run on the logit rows and, when a draft head takes hidden states, on each sequence's last row. The other hidden rows are left at zero; the assistant reads only the last committed row, which is always kept.
- Gemma's PLE table (4.4 GiB on E2B) is stored per layer and uploaded in bounded chunks, so loading never needs one multi-gigabyte buffer.
- Tool calls leave the answer by token, not by decoded text. Qwen's tokenizer keeps `<tool_call>`, which isn't a special token, in decoded text, so under `tool_choice=auto` a stream carried the whole call block in `delta.content` before the parsed call, and a streaming agent showed the markup. Matching the marker tokens works the same for special and ordinary markers, and needs no buffering of text that might start a marker.
- Inkling 2 is checked against a tiny fixture in its published layout (original names, NVFP4 and MXFP8 experts, padded unembedding), whose goldens come from Transformers 5.24.0 loading and dequantizing it itself. The reference matches every layer application and the logits within 1e-5 relative error on CPU and Metal, and the fused path within 3e-5, against a 1e-4 gate; decoding token by token through the pools matches Transformers' cache. On the real checkpoint (`thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`), with F32 projections and the published experts, every layer application ends within 2.2e-6 relative error of Transformers' own layers and the logits within 2.8e-6.
- Inkling 2's experts stay packed on the GPU whatever precision the plan gives projections. Wide expert tiles stage weights in F16, where E2M1 times an E4M3 scale and checkpoint-scale MXFP8 values are exact; the per-expert NVFP4 factor is applied in F32 in the expert activation. Decoding the experts to F32 multiplies that factor in and rounds in the same tiles: on the fixture, a 24-token pass then ended 2.4e-3 from Transformers' logits, while packed experts keep every layer application within 3e-5.
- Inkling 2's message gates run as fused kernels. Built from Candle operations, each gate took about ten dispatches, and a looped application has four. Pairing the inputs in one kernel, projecting the three attention gates in one batched product and gating and normalizing in one kernel brought decode on the M3 Pro from 52.2 to 47.4 ms per token with one sequence (Q8_0 projections, 128 tokens of context) and from 37.3 to 41.8 tokens/s with four.
- Inkling 2's message gates stay in their stored BF16, half the bytes of F32, and a GEMV widens them exactly with F32 accumulation, so they multiply as before. Up to 16 rows per gate take the GEMV; prefill converts the weights per call for the matrix multiply. Alternating runs on the M3 Pro: decode with four sequences fell from 95–96 to 87–88 ms per step, where Candle's batched F32 product handled the many small groups poorly, and with one sequence by about 0.4 ms.
- Inkling 2's drafter has no Transformers implementation, so its reference is an F32 transcription of vLLM's (`scripts/inkling2_reference.py --tiny-mtp`, matching SGLang's) over the tiny fixture's application inputs: the drafted logits match within 8e-6. On the real checkpoint, drafts it gave confidence of at least 0.9 were accepted 90% of the time on prose and 93% on code.
- Candle 0.11.1's multi-row Metal GGML GEMM reads its input from the start of the buffer, ignoring the tensor's start offset, so a row range of a larger activation multiplied the wrong rows. `PackedLinear::forward` copies such a view first. The one-row path and Tur's own kernels honor the offset.
- An Inkling 2 application's convolution histories share one recurrent slot, so the engine snapshots, commits and reuses them like DeltaNet state. The pools pair each history with a recurrent matrix; these slots carry a one-element placeholder for it.
