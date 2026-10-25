<div align="center">
  <img src="assets/mascot.png" alt="Tur mascot, a red Cossack with a sweeping moustache, flexing an arm" width="320">
  <h1>Tur</h1>
  <p><strong>Local inference with a little muscle.</strong></p>
  <p>A Rust inference engine for hybrid language models on Apple Silicon. Fused Metal kernels. Continuous batching. Your Mac.</p>
  <p> <a href="#why-tur-is-fast">Why it's fast</a> · <a href="#get-started">Get started</a> · <a href="#what-tur-can-do">Features</a> · <a href="#measured-on-a-mac">Performance</a> · <a href="docs/src/SUMMARY.md">Read the book</a> </p>
  <p><strong>Rust · Metal · Apple Silicon · Apache-2.0</strong></p>
</div>

---

Welcome to Tur. Run a prompt from your terminal, serve a model through an OpenAI-style HTTP API, or build on the Rust library. One `tur` binary handles generation, serving and quantization.

**Tur specializes the inference path for Qwen's hybrid architecture on Apple Silicon.** From Metal kernels to request scheduling, the goal is to move less data and do more useful work per GPU pass.

## Why Tur is fast

Generating a token repeatedly reads model weights and updates attention and recurrent state. Tur targets those costs directly:

- **Kernels tuned for the actual workload.** Fused norms, gates, activations and residual additions cut GPU dispatches and intermediate memory traffic. DeltaNet kernels arrange recurrent-state reads for efficient memory access, and quantized projection kernels use Apple's matrix units for small batches. Kernels are chosen by batch shape and measurements; Candle handles shapes where it wins.
- **Several requests share one model pass.** Continuous batching packs active decodes and prompt chunks together without padding. Batched projections reuse weights across requests, getting more work out of each weight read.
- **State stays where the GPU uses it.** Attention kernels read paged K/V directly, without gathering or concatenating the cache each step. Recurrent state lives in persistent pools, and prefix snapshots share state instead of copying it.
- **Sampling stays on the GPU.** The default path applies penalties, grammar masks and sampling alongside model execution. Only the selected token and a status word return to the CPU, avoiding a full vocabulary-sized logit transfer each step.
- **Quantized weights reduce bandwidth demand.** Single-sequence decode is limited by weight bandwidth on the measured M1. Q4K reduces the bytes read and reaches **92 tokens/s**, compared with **58 tokens/s** for Q8_0 on Qwen3.5-0.8B.

These choices have measured effects: reorganizing recurrent-state access alone cut an eight-sequence model pass from **43.4 ms to 28.6 ms** on the M1. The [kernel decisions](docs/src/architecture/kernels.md#decisions) and [memory design](docs/src/architecture/memory.md) explain the details.

For shared or predictable text, prefix reuse skips already-computed prompt tokens, and optional speculative decoding verifies several draft tokens in one pass. Their gains depend on the workload; see the [measurements below](#measured-on-a-mac).

## Get started

You'll need an **Apple Silicon Mac** and **Rust installed through [rustup](https://rustup.rs/)**. The checkout pins Rust 1.99.0. The first build takes a few minutes; the first run downloads the model into your Hugging Face cache.

### 1. Install

```sh
git clone https://github.com/danylokravchenko/tur.git
cd tur
./install.sh
```

The binary goes into `~/.cargo/bin`; make sure it's on your `PATH`. To install into `~/.local/bin`, use `./install.sh --root "$HOME/.local"`.

### 2. Say hello

Pin the model revision so every run loads the same weights:

```sh
export TUR_MODEL=Qwen/Qwen3.5-0.8B
export TUR_REVISION=2fc06364715b967f1860aea9cf38778875588b17

tur run --prompt 'Write one sentence about the Moon.' --temperature 0 -n 64 --thinking false
```

### 3. Make it an API

```sh
tur serve --served-model-name qwen
```

From another terminal, send your first request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"thinking":false}'
```

For an OpenAI client, set the base URL to `http://127.0.0.1:8080/v1` and the model to `qwen`. See the [HTTP API guide](docs/src/guide/http-api.md) for streaming, tool calls and structured output.

**Make it yours.** Copy [`tur.example.toml`](tur.example.toml) to `tur.toml` to keep model and tuning settings in a file. Flags override file values. The [configuration guide](docs/src/guide/configuration.md) covers every key and its precedence; the [full walkthrough](docs/src/guide/quickstart.md) covers local downloads, offline use and quantization.

## What Tur can do

| Feature | What it gives you |
| --- | --- |
| **Packed continuous batching** | One pass per step for scheduled requests, without padding. |
| **Paged state** | Persistent K/V and recurrent state pools, with F32, F16 or Q8 K/V. |
| **Prefix reuse** | Zero-copy snapshots at exact token boundaries to reuse shared prompts. |
| **Device sampling** | Temperature, top-k, top-p, penalties and grammar masks on the GPU. |
| **Exact speculative decoding** | Prompt-lookup and MTP drafts verified to match ordinary decoding. |
| **Calibrated quantization** | GPTQ, importance-weighted and mixed-precision prepacked weights. |

### Models and validation

| Model | Status |
| --- | --- |
| **Qwen3.5-0.8B** | Validated against Transformers per layer and in logits, on CPU and Metal. |
| **Qwen3.8-27B** | Same code path; config and headers validated. Not yet executed; needs more than 16 GB. |
| **Qwen3.6-35B-A3B (MoE)** | Experimental. |
| **Gemma 4 E2B** | Text inference validated with a pinned Transformers reference; Q8_0 checkpoint generation tested. |
| **Gemma 4 E4B / 12B / 26B-A4B / 31B** | Architecture fixtures and configuration coverage; full checkpoints not executed. |

Tur currently accepts **text input only**. Inference runs on Metal; CPU execution serves as a numerical reference for tests.

### Smaller weights, ready to serve

With the [Hugging Face CLI](docs/src/guide/quickstart.md#3-download) installed and representative calibration text in `calib.txt`, prepack calibrated 4-bit weights:

```sh
hf download Qwen/Qwen3.5-0.8B --revision "$TUR_REVISION" --local-dir ~/models/qwen3.5-0.8b
tur quantize --model ~/models/qwen3.5-0.8b --revision "$TUR_REVISION" \
  --output ~/models/qwen3.5-0.8b-q4k --calibration calib.txt
tur serve --model ~/models/qwen3.5-0.8b-q4k --revision q4k-v1 --served-model-name qwen
```

Start with [choosing a quantization](docs/src/guide/choosing-quantization.md) for format and quality tradeoffs, then use the [tuning recipes](docs/src/guide/tuning.md) for your workload.

## Measured on a Mac

All measurements so far come from an **Apple M1 with 16 GB**. These are measured workloads, with full settings and reproduction details in the [performance guide](docs/src/guide/performance.md).

**Faster decoding than llama.cpp in our M1 benchmarks:** with Qwen3.5-0.8B and Q8_0 weights, Tur delivers **up to 1.55× the batched decode throughput** at 8–16 sequences on the same Mac. At 8K-token prefill, llama.cpp is 6–11% faster. See the [full comparison and methodology](docs/src/guide/performance.md#compared-with-llamacpp) for versions, formats and workload details.

| Workload | Result |
| --- | --- |
| Single-sequence decode, Q8_0 weights | **58 tokens/s** |
| Single-sequence decode, Q4K weights | **92 tokens/s** |
| Prefill, Q8_0 weights, 2K-token prompt | **1,311 tokens/s** |
| Prompt-lookup speculation, repetitive output | **Up to 2.9×** ordinary decoding |

For serving, Qwen3.5-0.8B with Q8_0 weights, 128-token prompts and 64 sampled output tokens gives the following aggregate output throughput. Time to first token and inter-token latency are medians:

| Concurrent requests | Output tokens/s | Time to first token | Inter-token latency |
| --- | --- | --- | --- |
| 1 | 49.6 | 0.14 s | 17.8 ms |
| 4 | 121 | 0.45 s | 22.7 ms |
| 8 | 186 | 0.87 s | 26.9 ms |
| 16 | 204 | 1.63 s | 46.0 ms |

Prefix reuse also helps shared prompts: eight users arriving with the same cold 1.5K-token prompt get a first token in **1.7 s** at the median, compared with **9.6 s** under the earlier eviction policy. See the [measurements](docs/src/guide/performance.md#prefix-reuse) for cache budgets and workload details.

## Find your way around

The [Tur book](docs/src/SUMMARY.md) covers usage, architecture and the measurements behind engineering decisions. Run `just docs` to read it locally.

| I want to… | Start here |
| --- | --- |
| Run my first model | [Quickstart](docs/src/guide/quickstart.md) |
| Configure the CLI or server | [Configuration](docs/src/guide/configuration.md) · [Command line](docs/src/guide/cli.md) |
| Connect an app | [HTTP API](docs/src/guide/http-api.md) · [Rust library](docs/src/guide/library.md) |
| Choose weights and tune performance | [Quantization](docs/src/guide/choosing-quantization.md) · [Tuning](docs/src/guide/tuning.md) · [Benchmarks](docs/src/guide/performance.md) |
| Understand the engine | [Architecture](docs/src/architecture/overview.md) · [Scope](docs/src/design/scope.md) |
| Explore what's next | [Open work](docs/src/design/open.md) · [Rejected experiments](docs/src/design/rejected.md) |

## Build with us

Contributions are welcome. Start with the [development workflow](docs/src/development/workflow.md) and [AGENTS.md](AGENTS.md) for the project's conventions. Install [`just`](https://github.com/casey/just) to run the workspace tasks:

```sh
just ci           # format, lint, check with and without Metal, and run offline tests
just bench <case> # kernels | model | serving | prefix | kv_quality | tensorops | load_weights
```

Tests need direct Metal access on an Apple Silicon Mac. Real-checkpoint tests run separately; see the [testing guide](docs/src/development/testing.md).

| Crate | Responsibility |
| --- | --- |
| `tur-core` | Shared errors and checked packed-execution descriptors |
| `tur-metal` | State pools, completion fences, Metal kernels, packed weights, sampler |
| `tur-models` | Checkpoint loading and validation, Qwen and Gemma models, templates, tools |
| `tur-engine` | Engine, scheduler, prefix cache, speculation, pipeline, memory plan |
| `tur` | Facade and HTTP server; integration tests and benchmarks |
| `tur-cli` | The `tur` binary |

No unsafe code, no panics, and performance claims backed by measurements on the target Mac.

---

<div align="center">
  <p>Built for Apple Silicon. Written in Rust.</p>
  <p>Licensed under <a href="LICENSE">Apache-2.0</a>.</p>
</div>
