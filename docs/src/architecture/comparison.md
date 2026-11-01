# Tur and other engines

Pick Tur to serve several requests at once from a supported model on an Apple Silicon Mac, or to run a mixture-of-experts (MoE) model that doesn't fit in that Mac's memory. Its design treats memory traffic, recurrent state and steady replies as the core problems, where most other engines start from broad model and hardware support. Tur covers few models and has a narrow API, so the other engines on this page win for many setups, and the table below points to them.

| You want | Use |
| --- | --- |
| Fast batched decode and low memory use on a Mac serving Qwen3.5-family models to several clients or agents | Tur |
| An MoE model larger than GPU memory on a Mac, with bounded memory, no SSD writes and more than one request at a time | Tur |
| Prompt caches that survive a restart, a menu bar app, several models, or images and embeddings on a Mac | oMLX |
| The widest choice of models and hardware, GGUF files, or a library that runs everywhere | llama.cpp |
| A model running in one command, with switching handled for you | Ollama |
| Production serving on NVIDIA, AMD or TPU clusters | SGLang |
| The largest MoE models on consumer GPUs or a large Mac, one request at a time | Strata |

Other projects' features come from their documentation as of November 2026 and change quickly, so check upstream before deciding.

## At a glance

| | Tur | llama.cpp | oMLX | Ollama | SGLang | Strata |
| --- | --- | --- | --- | --- | --- | --- |
| Hardware | Apple Silicon | CPUs and GPUs of most vendors | Apple Silicon | Mac, Linux, Windows | Data-center GPUs, TPUs, NPUs; Macs through MLX | Consumer NVIDIA, AMD and Intel GPUs; Apple Silicon |
| Models | Qwen3.5 family, Gemma 4, gpt-oss, Whisper | Very broad, as GGUF | Broad, through MLX | Broad, from its library | Broad | Selected large MoE models |
| Execution | Own Metal kernels over Candle | Own Metal backend | MLX, optional native kernels | llama.cpp, or MLX | MLX on a Mac | Own Metal engine on a Mac |
| Requests at once on a Mac | Continuous batching | Parallel slots | Continuous batching, 8 by default | 1 by default; [no concurrency on the MLX engine yet](https://github.com/ollama/ollama/issues/17280) | [Guide](https://docs.sglang.io/docs/hardware-platforms/apple_metal) shows single-request runs only | [One](https://github.com/Niko1221/Strata/blob/5cfe96ee6a0827acf83f4c6f5ac74dd21c16fe9a/docs/MAC.md) |
| Prompt cache | In memory, shared pages | [In slots and host memory](https://github.com/ggml-org/llama.cpp/pull/16391) | Memory and SSD, survives restart | Slots, or [MLX snapshots](https://ollama.com/blog/mlx) | RadixAttention; not documented for MLX | Not compared |
| Weights larger than memory | Expert cache | Memory-mapped files | No expert offload documented | Not documented | Not documented for Macs | Expert streaming |
| Images and embeddings | No | Images through its multimodal library | Yes | Yes | Yes | Images in Qwen3.8 |
| Structured output over HTTP | No, only `tool_choice=auto` | Yes | Yes | Yes | Yes | Not compared |

## How Tur's design differs

Most of these engines start from generality: a tensor library or a broad server that many models and devices can plug into. Tur starts from one machine and one model shape, then builds each layer around them. That narrowness costs it model coverage, and every new family needs its own implementation and validation. It pays for the choices below. The descriptions of other engines come from their documentation, not from auditing their code.

| Concern | Tur's approach | What you get | Elsewhere |
| --- | --- | --- | --- |
| Compute | Kernels fitted to these models and to each batch shape, with Candle where it wins | Fused steps, and a kernel chosen by measurement at each shape | llama.cpp and MLX run models through general tensor libraries; oMLX adds optional native kernels for some families |
| Recurrent state | A first-class kind of state with its own slots and exact boundaries | Prefix reuse and rollback that work on hybrid models by construction | Prompt caches usually grow from K/V, which can be cut at any length |
| Memory | One plan for weights, state, scratch and drafts, reserved before loading | A configuration that doesn't fit fails at startup | oMLX evicts models against a process limit; Ollama's RAM [scales with parallel requests times context](https://docs.ollama.com/faq) |
| Scheduling | Decodes first, with prompts sized from measured pass times | New prompts can at most double the wait for the next token | oMLX uses first-come, first-served; Strata and Ollama's MLX engine queue requests |
| Failure | One owner, per-request errors, generation-tagged handles | One failed request doesn't take down the others | Not compared |

### State built for hybrid models

Hybrid models keep attention state and recurrent state side by side, and the two behave differently. You can trim cached keys and values to any length, but the recurrent state after token `n` exists only if a pass ended exactly at `n`. Tur treats this as a basic constraint: the scheduler stops prompt chunks at snapshot positions, so the boundaries a later request needs always exist. Each pass also writes new recurrent slots next to the committed ones, so rejecting a speculative draft means picking a slot, with nothing to recompute.

Snapshots share pages and slots instead of copying them, so taking one costs almost nothing, and a page is copied only when a request appends to a page someone else still holds. oMLX's block cache also uses copy-on-write. llama.cpp's [host-memory cache](https://github.com/ggml-org/llama.cpp/pull/16391) instead keeps prompt states in RAM and swaps them into the context. Tur's choice ties reuse to the GPU pools it reads every token, while theirs can hold more prompts than the context, and oMLX's survive a restart.

### Memory as a promise

Tur works out its whole memory use before reading a weight, and `--context-tokens` is a promise that every concurrent request gets that much room. Running out is then a planned event. Snapshots go first, then the newest request gives up its state and later recomputes it, with its random stream intact. A server you leave running is less likely to push the Mac into swap partway through a job, though the plan can't see what other applications do after startup. The cost is that Tur holds its full plan even while idle.

### Scheduling for steady replies

Every pass mixes replying requests and new prompts. Tur's scheduler times each pass on the GPU and gives prompts the largest share that keeps the pass within its target, so a long prompt arriving doesn't stall every stream. On a mixed workload this made the slowest 5% of gaps between tokens 3.7× shorter than running prompts unchecked, in exchange for 7% less throughput. Tur still processes prompts in arrival order, with no priorities or per-client fairness yet.

### Speed-ups that don't change the answer

Tur treats optimizations as exact by default. Prefix restores compare every token, and speculative decoding publishes only tokens drawn from the model's own distribution. The same seed gives the same tokens as normal decoding, and speculation turns itself off when measured costs say it won't pay. Expert-cache logits are bit-identical to keeping those experts resident. The one exception is batch shape, which can change a result's last bits because kernels add numbers in different orders. `PipelineConfig::reproducible` runs requests one at a time when that matters.

### A small, strict runtime

Tur is one Rust binary, with no Python environment to manage, unlike oMLX and SGLang. The code has no `unsafe` and no panics. One thread owns each model's requests and memory, so no locks are held during execution, and every shape, offset and page index is checked before GPU work is submitted. Defaults follow the same discipline: a custom kernel stays only if it beats the existing one by more than run-to-run noise, and the losers are deleted and listed in [Rejected experiments](../design/rejected.md).

## Why Tur is faster on a Mac

A Mac's GPU spends most of a decode step waiting for memory: weights, attention keys and values, and recurrent state. Tur reduces the bytes each token reads, through four choices.

Each pass reads the weights once for the whole batch. Tur packs every running request's next token into one pass with no padding, and it uses the GPU's matrix units for the small batches that generic kernels handle poorly. Q4K at 2 to 8 rows runs 2.5× faster on them than on Candle's kernel, so one extra sequence adds little to a step.

Recurrent state is read in the order the GPU prefers. This cut an eight-sequence pass by a third and raised serving throughput at 8 and 16 requests about 1.7×. Hybrid models carry this state for every sequence, and it's now the main per-sequence cost of decoding.

Attention reads state where it lives. Keys and values stay in fixed pages that kernels read in place, and norms, gates and residual additions are fused into neighbouring kernels, which saves a trip through memory for each step.

Weights hold fewer bytes. Tur's Q4K uses Q4_K for every projection, while GGUF's Q4_K_M mixes in Q6_K blocks. With one sequence, reading weights is the limit, so Q4K decodes about 1.6× faster than Q8_0 inside Tur. The cost is quality, which [Choosing a quantization](../guide/choosing-quantization.md) measures.

Two more features skip work and don't appear in the kernel numbers below. [Prefix reuse](prefix-reuse.md) shares state without copying it, and [speculative decoding](speculation.md) reaches up to 2.9× on repetitive text, with output identical to normal decoding.

These causes come from measuring Tur against its own variants. We haven't profiled the other engines, so they explain why Tur got faster rather than proving why another engine is slower.

## How much faster

These results are for llama.cpp 0.8.0 on an M3 Pro with 36 GB, Qwen3.5-0.8B and Q8_0 weights, unless a row says otherwise. Both engines use their usual format in the 4-bit row, so it compares engines and weights together.

| Test | Tur | llama.cpp | Difference |
| --- | ---: | ---: | ---: |
| Decode, 1 sequence | 123 tok/s | 113 tok/s | +9% |
| Decode, 1 sequence, 4-bit | 182 tok/s | 141 tok/s | +29% |
| Decode, 8 sequences | 729 tok/s | 456 tok/s | +60% |
| Prefill, 2K tokens | 3,799 tok/s | 3,383 tok/s | +12% |
| Qwen3.6-35B, 4-bit, decode, 1 sequence | 47.1 tok/s | 35.6 tok/s | +32% |
| Qwen3.6-35B, 4-bit, burst of 8, end to end | 101.5 tok/s | 75.0 tok/s | +35% |
| Qwen3.8-27B, 4-bit, decode, 8 sequences | 52.5 tok/s | 41.1 tok/s | +28% |

Against oMLX on an M1 with 16 GB and 4-bit Qwen3.5-0.8B, using its [published results](https://omlx.ai/benchmarks/performance/yw52053o), not a rerun on our Mac:

| Sequences | Tur | oMLX | Peak memory, Tur / oMLX |
| --- | ---: | ---: | ---: |
| 1 | 90 tok/s | 73.7 tok/s | 0.71 GB / 2.17 GB |
| 2 | 114 tok/s | 123 tok/s | |
| 4 | 202 tok/s | 198 tok/s | |

[Measured performance](../guide/performance.md) has the methods, versions and remaining rows.

Tur doesn't win everywhere:

- Prefill is mostly a compute problem, so kernels matter less there. llama.cpp is 6 to 11% ahead at 8K tokens on the M1, and level on Qwen3.8-27B.
- oMLX matches Tur from two sequences on, in the published results above.
- MLX uses the M5's Neural Accelerators, and [Ollama reports](https://ollama.com/blog/mlx) faster first tokens from them. Tur doesn't use them and hasn't been measured on an M5, so its lead may not hold on that chip.

We haven't benchmarked Ollama or SGLang. Ollama on GGUF runs llama.cpp, so the llama.cpp numbers are a guide to that path, before Ollama's own overhead. SGLang is built for data-center GPUs, and Tur can't run there at all.

## Offloading experts to the SSD

"SSD" means different things across these engines, and only some of them help a model fit:

- Tur and Strata read MoE experts from the SSD into a fixed arena. Each token uses few experts, so a model larger than memory can run.
- llama.cpp memory-maps its model file and leaves paging to the OS. That lets a larger file load, but a mapped file sets no limit on residency and gets no read-ahead from the model's routers. We haven't measured it on a model that doesn't fit.
- oMLX writes K/V blocks to the SSD so prompts survive eviction and restarts. This saves prefill, and the weights still have to fit. It also adds steady SSD writes, which is why [Tur doesn't do it](../research/serving-technique-gaps.md#not-recommended).

Strata is the closest comparison, and [its Metal expert stream](https://github.com/Niko1221/Strata/blob/46c152a1310f77cd2f0ce2c7cd213962df14fe83/metal/src/engine/stream.h) was the reference for Tur's [expert cache](expert-cache.md). Both keep dense weights resident, map expert IDs to arena slots on the device, read misses on demand and read ahead using the next layer's likely experts. They differ in what they optimize:

| | Tur | Strata on a Mac |
| --- | --- | --- |
| Built for | Serving smaller MoE models on small Macs | The largest MoE models on consumer hardware |
| Models | gpt-oss-20b, Qwen3.6-35B-A3B | Qwen3.8-Flash-Next, DeepSeek V4.1-Flash; Qwen3.6 doesn't run on its Mac engine |
| Smallest measured Mac | M1 with 16 GB | M4 Mac mini with 32 GB, M1 Max with 34 GB |
| Requests at once | Several, sharing the arena | One, others queue |
| Arena size | You set it, and the load refuses a size that would cause swapping | Chosen automatically: Metal's share or available memory minus a reserve, whichever is smaller |
| Arena on large Macs | Not wired, not measured | Wired from 64 GB up, so macOS doesn't compress it |
| SSD writes | None during a run, and page cache bypassed | Optional one-time copy that stores each expert's matrices together, which it measured at 17% faster decode and 30% faster prompts |
| Long prompts | Prefill chunks grow to 1,024 tokens, so each pass reads fewer experts | No equivalent documented |

Their speeds can't be compared directly, because no model and Mac appear on both sides. Strata reports 13 to 14 tok/s on Flash-Next Q2_0 with one request on the 32 GB Mac mini, using an arena of 11.5 to 14.7 GB. Tur reports 3.1 to 3.3 tok/s on gpt-oss-20b on a 16 GB M1, using 1 to 2 GiB. Tur has no support for Flash-Next, whose 512 experts per layer exceed its router's limit of 256. Both results show that speed depends on how many reads the arena avoids.

Choose Tur's cache when your Mac can't hold the model, your model is supported and more than one client needs answers. Its total throughput rises slightly with more requests (2.1 to 2.7 tok/s from one to four on gpt-oss-20b), while each request slows down. Choose Strata for Flash-Next or DeepSeek-class models, for a desktop app, or for a discrete GPU.

Offloading is the fallback in either engine. If the whole model fits, keep it resident: on the 36 GB M3 Pro, Qwen3.6-35B decodes 47.1 tok/s resident, while 3 tok/s on gpt-oss-20b suits batch work and short answers better than conversation. [Expert cache](expert-cache.md) covers sizing and SSD wear.

## What Tur leaves out

Tur rejects images and video, serves one chat model per process, and has no embeddings or Anthropic-style API. Grammar-constrained output works in the engine, but the HTTP server doesn't expose it, so `response_format` and forced tool choice aren't available to clients yet. Qwen3.6-35B and Qwen3.8-27B run on real checkpoints, though their quality against Transformers hasn't been measured. If you need any of these today, choose oMLX or llama.cpp on a Mac. [Scope and principles](../design/scope.md) has the evidence for each model, and [Open work](../design/open.md) tracks what's next.
