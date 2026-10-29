<div align="center">
  <img src="assets/mascot.png" alt="Tur mascot, a red Cossack with a sweeping moustache, flexing an arm" width="320">
  <h1>Tur</h1>
  <p><strong>Local inference with a little muscle.</strong></p>
  <p>A Rust inference engine for hybrid language models on Apple Silicon.</p>
  <p> <a href="#why-tur-is-fast">Why it's fast</a> · <a href="#get-started">Get started</a> · <a href="#what-tur-can-do">Features</a> · <a href="#measured-on-a-mac">Performance</a> · <a href="docs/src/SUMMARY.md">Read the book</a> </p>
  <p><strong>Rust · Metal · Apple Silicon · Apache-2.0</strong></p>
</div>

---

Tur runs Qwen3.5-family and Gemma 4 text models on your Mac's GPU. You can generate from the terminal, serve an OpenAI-compatible HTTP API, or embed the engine in a Rust program. One `tur` binary handles generation, serving and quantization.

General-purpose engines treat every model the same way. Tur is built around one family of models on one kind of hardware, so each part of the path, from Metal kernels to request scheduling, can be shaped to what those models need.

## Why Tur is fast

On a Mac, producing a token is mostly a memory problem. The GPU has to read the model's weights and each request's attention and recurrent state, and it spends more time waiting for those bytes than computing with them. Tur's design follows from that:

- Several requests share each pass over the weights. Continuous batching packs every active request's next token, plus chunks of new prompts, into one model pass with no padding. The weights are read once and used for all of them, so eight users cost far less than eight times one user.
- Kernels do more per memory read. Norms, gates, activations and residual additions are fused into neighbouring kernels, and the recurrent-state kernels read memory in the order the GPU prefers. Where Candle's own kernel is faster, Tur uses Candle's.
- State stays where the GPU reads it. Attention keys and values live in fixed pages that kernels read in place, so nothing is copied or reassembled per token. Recurrent state lives in reusable slots.
- Sampling happens on the GPU. Temperature, top-k, top-p, penalties and grammar masks run right after the model pass, and only the chosen token travels back to the CPU.
- Smaller weights mean fewer bytes to read. With 4-bit weights, a single conversation decodes about 1.6× faster than with 8-bit ones.

Two more features skip work entirely. Prefix reuse keeps the state of prompts it has already processed, so a shared system prompt or an agent's growing conversation is computed once. Speculative decoding guesses several tokens ahead and checks them all in one pass, which pays off on repetitive output such as code edits. Both give exactly the output you would get without them.

The [architecture chapters](docs/src/architecture/overview.md) explain each mechanism and the measurements that decided it.

## Get started

You need an Apple Silicon Mac and Rust installed through [rustup](https://rustup.rs/). The first build takes a few minutes, and the first run downloads the model into your Hugging Face cache.

### 1. Install

```sh
git clone https://github.com/danylokravchenko/tur.git
cd tur
./install.sh
```

The binary goes into `~/.cargo/bin`, so make sure that is on your `PATH`. Use `./install.sh --root "$HOME/.local"` to install into `~/.local/bin` instead.

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

From another terminal:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"thinking":false}'
```

Any OpenAI client works with the base URL `http://127.0.0.1:8080/v1` and the model `qwen`. The [HTTP API guide](docs/src/guide/http-api.md) covers streaming, tool calls and structured output.

To keep your settings, copy [`tur.example.toml`](tur.example.toml) to `tur.toml`; flags still override it. The [configuration guide](docs/src/guide/configuration.md) lists every key, and the [quickstart](docs/src/guide/quickstart.md) walks through local downloads, offline use and quantization.

## What Tur can do

| Feature | What it gives you |
| --- | --- |
| Continuous batching | Many requests share each model pass, with no padding |
| Paged state | Attention and recurrent state in fixed pools, with F32, F16 or Q8 keys and values |
| Prefix reuse | Shared prompts and multi-turn conversations are computed once |
| GPU sampling | Temperature, top-k, top-p, penalties and grammar masks without a round trip to the CPU |
| Exact speculative decoding | Drafted tokens verified in one pass, with output identical to normal decoding |
| Calibrated quantization | GPTQ, importance-weighted and mixed-precision 4-bit weights that load instantly |

### Models

| Model | Status |
| --- | --- |
| Qwen3.5-0.8B | Validated layer by layer against Transformers, on CPU and Metal |
| Qwen3.8-27B | Same code path and validated config; not yet run, since it needs more than 16 GB |
| Qwen3.6-35B-A3B (MoE) | Experimental |
| Gemma 4 E2B | Validated against Transformers; real checkpoint generation tested |
| Gemma 4 E4B, 12B, 26B-A4B, 31B | Architecture covered by small fixtures; full checkpoints not yet run |

Tur accepts text input only.

### Smaller weights

With the [Hugging Face CLI](docs/src/guide/quickstart.md#3-download) installed and some text that resembles your traffic in `calib.txt`, you can prepack calibrated 4-bit weights:

```sh
hf download Qwen/Qwen3.5-0.8B --revision "$TUR_REVISION" --local-dir ~/models/qwen3.5-0.8b
tur quantize --model ~/models/qwen3.5-0.8b --revision "$TUR_REVISION" \
  --output ~/models/qwen3.5-0.8b-q4k --calibration calib.txt
tur serve --model ~/models/qwen3.5-0.8b-q4k --revision q4k-v1 --served-model-name qwen
```

[Choosing a quantization](docs/src/guide/choosing-quantization.md) explains the trade-offs, and the [tuning recipes](docs/src/guide/tuning.md) match settings to workloads.

## Measured on a Mac

These numbers come from an Apple M1 with 16 GB running Qwen3.5-0.8B. The [performance guide](docs/src/guide/performance.md) has the full tables and how to reproduce them.

| Workload | Result |
| --- | --- |
| One conversation, 8-bit weights | 58 tokens/s |
| One conversation, 4-bit weights | 92 tokens/s |
| Prefill of a 2K-token prompt | 1,311 tokens/s |
| Prompt-lookup speculation on repetitive output | Up to 2.9× normal decoding |

Under load, through the full server pipeline with sampling, 8-bit weights, 128-token prompts and 64-token replies:

| Concurrent requests | Output tokens/s | Time to first token | Time between tokens |
| --- | --- | --- | --- |
| 1 | 50 | 0.14 s | 18 ms |
| 4 | 121 | 0.45 s | 23 ms |
| 8 | 186 | 0.87 s | 27 ms |
| 16 | 204 | 1.63 s | 46 ms |

On the same Mac, Tur decodes up to 1.55× faster than llama.cpp with batches of 8 to 16 sequences. llama.cpp is 6 to 11% faster at prefilling 8K-token prompts. The [comparison](docs/src/guide/performance.md#compared-with-llamacpp) lists versions and methods.

Prefix reuse matters most when many users share a prompt. Eight users who send the same new 1.5K-token prompt at once get their first token in 1.7 s at the median; without prefix reuse they wait 9.5 s.

## Find your way around

The [Tur book](docs/src/SUMMARY.md) covers usage, architecture and the measurements behind each decision. Run `just docs` to read it locally.

| I want to | Start here |
| --- | --- |
| Run my first model | [Quickstart](docs/src/guide/quickstart.md) |
| Configure the CLI or server | [Configuration](docs/src/guide/configuration.md), [Command line](docs/src/guide/cli.md) |
| Connect an app | [HTTP API](docs/src/guide/http-api.md), [Rust library](docs/src/guide/library.md) |
| Choose weights and tune performance | [Quantization](docs/src/guide/choosing-quantization.md), [Tuning](docs/src/guide/tuning.md), [Benchmarks](docs/src/guide/performance.md) |
| Understand the engine | [Architecture](docs/src/architecture/overview.md), [Scope](docs/src/design/scope.md) |
| See what's next | [Open work](docs/src/design/open.md), [Rejected experiments](docs/src/design/rejected.md) |

## Build with us

Contributions are welcome. Read the [development workflow](docs/src/development/workflow.md) and [AGENTS.md](AGENTS.md) for the project's conventions, and install [`just`](https://github.com/casey/just) to run the workspace tasks:

```sh
just ci           # format, lint, check with and without Metal, and run offline tests
just bench <case> # kernels | model | serving | prefix | kv_quality | tensorops | load_weights
```

Tests need direct Metal access on an Apple Silicon Mac. Real-checkpoint tests run separately, as the [testing guide](docs/src/development/testing.md) explains.

| Crate | Responsibility |
| --- | --- |
| `tur-core` | Shared errors and checked packed-execution descriptors |
| `tur-metal` | State pools, completion fences, Metal kernels, packed weights, sampler |
| `tur-models` | Checkpoint loading and validation, Qwen and Gemma models, templates, tools |
| `tur-engine` | Engine, scheduler, prefix cache, speculation, pipeline, memory plan |
| `tur` | Facade and HTTP server; integration tests and benchmarks |
| `tur-cli` | The `tur` binary |

The code has no `unsafe` and no panics, and every performance claim comes from a measurement on a Mac.

---

<div align="center">
  <p>Built for Apple Silicon. Written in Rust.</p>
  <p>Licensed under <a href="LICENSE">Apache-2.0</a>.</p>
</div>
