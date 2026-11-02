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

## The other engines

### llama.cpp

llama.cpp runs on the CPUs and GPUs of most vendors and loads a very broad range of models as GGUF. On a Mac it uses its own Metal backend and serves concurrent requests through parallel slots. It keeps prompt caches [in slots and host memory](https://github.com/ggml-org/llama.cpp/pull/16391), handles larger-than-memory weights by memory-mapping them, takes images through its multimodal library, and offers structured output over HTTP.

### oMLX

oMLX runs on Apple Silicon and covers a broad range of models through MLX, with optional native kernels. It uses continuous batching, with 8 requests at once by default, and its prompt cache lives in memory and on the SSD, so it survives a restart. It supports images, embeddings and structured output, and documents no expert offload.

### Ollama

Ollama runs on Mac, Linux and Windows and offers a broad model library, running each model on llama.cpp or MLX. It serves one request at a time by default, and there is [no concurrency on the MLX engine yet](https://github.com/ollama/ollama/issues/17280). Prompts are cached in slots or as [MLX snapshots](https://ollama.com/blog/mlx). It supports images, embeddings and structured output, and doesn't document running weights larger than memory.

### SGLang

SGLang is built for data-center GPUs, TPUs and NPUs, covers a broad range of models, and reaches Macs through MLX. Its [Mac guide](https://docs.sglang.io/docs/hardware-platforms/apple_metal) shows single-request runs only. Its RadixAttention prompt cache and any handling of larger-than-memory weights aren't documented for Macs. It supports images, embeddings and structured output.

### Strata

Strata runs selected large MoE models on consumer NVIDIA, AMD and Intel GPUs and on Apple Silicon, with its own Metal engine on a Mac. That engine serves [one request at a time](https://github.com/Niko1221/Strata/blob/5cfe96ee6a0827acf83f4c6f5ac74dd21c16fe9a/docs/MAC.md). It streams experts to run models larger than memory and takes images in Qwen3.8. We haven't compared its prompt cache or structured output.

## How Tur's design differs

Most of these engines start from generality: a tensor library or a broad server that many models and devices can plug into. Tur starts from one machine and one model shape, then builds each layer around them. That narrowness costs it model coverage, and every new family needs its own implementation and validation. It pays for the choices below. The descriptions of other engines come from their documentation, not from auditing their code.

### State built for hybrid models

Hybrid models keep attention state and recurrent state side by side, and the two behave differently. You can trim cached keys and values to any length, but the recurrent state after token `n` exists only if a pass ended exactly at `n`. Prompt caches usually grow from K/V, which can be cut anywhere. Tur treats recurrent boundaries as a basic constraint instead: the scheduler stops prompt chunks at snapshot positions, so the boundaries a later request needs always exist. Each pass also writes new recurrent slots next to the committed ones, so rejecting a speculative draft means picking a slot, with nothing to recompute.

Snapshots share pages and slots instead of copying them, so taking one costs almost nothing, and a page is copied only when a request appends to a page someone else still holds. oMLX's block cache also uses copy-on-write. llama.cpp's [host-memory cache](https://github.com/ggml-org/llama.cpp/pull/16391) instead keeps prompt states in RAM and swaps them into the context. Tur's choice ties reuse to the GPU pools it reads every token, while theirs can hold more prompts than the context, and oMLX's survive a restart.

### Memory as a promise

Tur works out its whole memory use before reading a weight, and `--context-tokens` is a promise that every concurrent request gets that much room. Running out is then a planned event. Snapshots go first, then the newest request gives up its state and later recomputes it, with its random stream intact. A server you leave running is less likely to push the Mac into swap partway through a job, though the plan can't see what other applications do after startup. The cost is that Tur holds its full plan even while idle. Elsewhere, oMLX evicts models against a process limit, and Ollama's RAM [scales with parallel requests times context](https://docs.ollama.com/faq).

### Scheduling for steady replies

Every pass mixes replying requests and new prompts. Tur's scheduler times each pass on the GPU and gives prompts the largest share that keeps the pass within its target, so a long prompt arriving doesn't stall every stream. On a mixed workload this made the slowest 5% of gaps between tokens 3.7× shorter than running prompts unchecked, in exchange for 7% less throughput. Tur still processes prompts in arrival order, with no priorities or per-client fairness yet. oMLX also uses first-come, first-served, while Strata and Ollama's MLX engine queue whole requests.

### Speed-ups that don't change the answer

Tur treats optimizations as exact by default. Prefix restores compare every token, and speculative decoding publishes only tokens drawn from the model's own distribution. The same seed gives the same tokens as normal decoding, and speculation turns itself off when measured costs say it won't pay. Expert-cache logits are bit-identical to keeping those experts resident. The one exception is batch shape, which can change a result's last bits because kernels add numbers in different orders. `PipelineConfig::reproducible` runs requests one at a time when that matters.

### A small, strict runtime

Tur is one Rust binary, with no Python environment to manage, unlike oMLX and SGLang. The code has no `unsafe` and no panics. One thread owns each model's requests and memory, so no locks are held during execution, and every shape, offset and page index is checked before GPU work is submitted. Errors are per request and handles are tagged by generation, so one failed request doesn't take down the others. Defaults follow the same discipline: a custom kernel stays only if it beats the existing one by more than run-to-run noise, and the losers are deleted and listed in [Rejected experiments](../design/rejected.md). llama.cpp and MLX run models through general tensor libraries, and oMLX adds optional native kernels for some families.

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

Against oMLX on an M1 with 16 GB and 4-bit Qwen3.5-0.8B, using its [published results](https://omlx.ai/benchmarks/performance/yw52053o) rather than a rerun on our Mac, Tur leads with one sequence, at 90 tok/s to 73.7 and with a peak of 0.71 GB to 2.17 GB. oMLX leads at two sequences, 123 tok/s to 114, and the two are level at four, with Tur at 202 and oMLX at 198.

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

Strata is the closest comparison, and [its Metal expert stream](https://github.com/Niko1221/Strata/blob/46c152a1310f77cd2f0ce2c7cd213962df14fe83/metal/src/engine/stream.h) was the reference for Tur's [expert cache](expert-cache.md). Both keep dense weights resident, map expert IDs to arena slots on the device, read misses on demand and read ahead using the next layer's likely experts.

They differ in what they optimize. Tur serves smaller MoE models on small Macs: gpt-oss-20b and Qwen3.6-35B-A3B, measured down to an M1 with 16 GB. Strata runs the largest MoE models on consumer hardware: Qwen3.8-Flash-Next and DeepSeek V4.1-Flash, measured down to an M4 Mac mini with 32 GB and an M1 Max with 34 GB. Qwen3.6 doesn't run on its Mac engine. Tur serves several requests that share the arena, while Strata serves one and queues the others.

You set Tur's arena size, and the load refuses a size that would cause swapping. Strata chooses automatically: Metal's share or available memory minus a reserve, whichever is smaller. From 64 GB up it wires the arena so macOS doesn't compress it, which Tur doesn't do and hasn't measured.

Tur writes nothing to the SSD during a run and bypasses the page cache. Strata offers an optional one-time copy that stores each expert's matrices together, which it measured at 17% faster decode and 30% faster prompts. Tur grows prefill chunks to 1,024 tokens on long prompts, so each pass reads fewer experts, and Strata documents no equivalent.

Their speeds can't be compared directly, because no model and Mac appear on both sides. Strata reports 13 to 14 tok/s on Flash-Next Q2_0 with one request on the 32 GB Mac mini, using an arena of 11.5 to 14.7 GB. Tur reports 3.1 to 3.3 tok/s on gpt-oss-20b on a 16 GB M1, using 1 to 2 GiB. Tur has no support for Flash-Next, whose 512 experts per layer exceed its router's limit of 256. Both results show that speed depends on how many reads the arena avoids.

Choose Tur's cache when your Mac can't hold the model, your model is supported and more than one client needs answers. Its total throughput rises slightly with more requests (2.1 to 2.7 tok/s from one to four on gpt-oss-20b), while each request slows down. Choose Strata for Flash-Next or DeepSeek-class models, for a desktop app, or for a discrete GPU.

Offloading is the fallback in either engine. If the whole model fits, keep it resident: on the 36 GB M3 Pro, Qwen3.6-35B decodes 47.1 tok/s resident, while 3 tok/s on gpt-oss-20b suits batch work and short answers better than conversation. [Expert cache](expert-cache.md) covers sizing and SSD wear.

## What Tur leaves out

Tur rejects images and video, serves one chat model per process, and has no embeddings or Anthropic-style API. Qwen3.6-35B and Qwen3.8-27B run on real checkpoints, though their quality against Transformers hasn't been measured. If you need any of these today, choose oMLX or llama.cpp on a Mac. [Scope and principles](../design/scope.md) has the evidence for each model, and [Open work](../design/open.md) tracks what's next.
