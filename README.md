# Tur

Tur is a Rust inference engine for hybrid language models on Apple Silicon. It runs
Qwen3.5-family checkpoints, which interleave Gated DeltaNet recurrent layers with gated
attention, on fused Metal kernels:

- **Packed continuous batching:** one pass per step for every scheduled request, with no
  padding.
- **Paged state:** persistent K/V and recurrent state pools; K/V in F32, F16 or Q8.
- **Prefix reuse:** zero-copy snapshots at exact token boundaries.
- **Device sampling:** temperature, top-k, top-p, penalties and grammar masks on the GPU.
- **Exact speculative decoding:** drafts from prompt lookup and the checkpoint's own MTP
  head, verified so output matches ordinary decoding.
- **Calibrated quantization:** GPTQ, importance-weighted and mixed-precision prepacked
  weights.

It ships as one `tur` binary (`run`, `serve`, `quantize`), an OpenAI-style HTTP API,
and a Rust library.

| Model | Status |
| --- | --- |
| Qwen3.5-0.8B | Validated against Transformers per layer and in logits, on CPU and Metal |
| Qwen3.8-27B | Same code path, config and headers validated; not yet executed (needs more than 16 GB) |
| Qwen3.6-35B-A3B (MoE) | Experimental |

Text input only. All measurements so far come from one M1 with 16 GB.

## Performance

Qwen3.5-0.8B with Q8_0 weights on an M1 (16 GB), serving 128-token prompts with 64 sampled
output tokens:

| Concurrent requests | Output tok/s | TTFT p50 | Inter-token p50 |
| --- | --- | --- | --- |
| 1 | 50 | 0.14 s | 17.8 ms |
| 4 | 121 | 0.45 s | 22.7 ms |
| 8 | 186 | 0.87 s | 26.9 ms |
| 16 | 204 | 1.63 s | 46.0 ms |

- Single-sequence decode: 58 tok/s with Q8_0, 92 tok/s with Q4K.
- Prefill: about 1,300 tok/s at 2K tokens.
- Prefix reuse: a cold shared 1.5K-token prompt reached by eight users gets its first
  token in 1.7 s instead of 9.6 s under the earlier eviction policy.
- Speculation: prompt lookup reaches up to 2.9× on repetitive output.

All tables and settings: [Measured performance](docs/src/guide/performance.md).

## Quickstart

```sh
./install.sh                          # installs to ~/.cargo/bin by default

export TUR_MODEL=Qwen/Qwen3.5-0.8B
export TUR_REVISION=2fc06364715b967f1860aea9cf38778875588b17

tur run --prompt 'Write one sentence about the Moon.' --temperature 0 -n 64 --thinking false
tur serve --served-model-name qwen        # http://127.0.0.1:8080/v1/chat/completions
```

To keep model and tuning settings in a file, copy [`tur.example.toml`](tur.example.toml)
to `tur.toml`, then use `tur run` or `tur serve`. Flags override file values; see
[Configuration](docs/src/guide/configuration.md) for precedence and all available keys.

Use `./install.sh --root "$HOME/.local"` to install into `~/.local/bin` instead.
Ensure the destination's `bin` directory is on your `PATH`.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
```

Prepack calibrated 4-bit weights, then serve them:

```sh
hf download Qwen/Qwen3.5-0.8B --revision "$TUR_REVISION" --local-dir ~/models/qwen3.5-0.8b
tur quantize --model ~/models/qwen3.5-0.8b --revision "$TUR_REVISION" \
  --output ~/models/qwen3.5-0.8b-q4k --calibration calib.txt
tur serve --model ~/models/qwen3.5-0.8b-q4k --revision q4k-v1
```

## Documentation

The book in [`docs/`](docs/src/SUMMARY.md) is the full documentation. Serve it with
`just docs`.

- **Guide:**
  - [Quickstart](docs/src/guide/quickstart.md)
  - [Measured performance](docs/src/guide/performance.md)
  - [Configuration](docs/src/guide/configuration.md)
  - [Command line](docs/src/guide/cli.md)
  - [HTTP API](docs/src/guide/http-api.md)
  - [Choosing a quantization](docs/src/guide/choosing-quantization.md)
  - [Tuning recipes](docs/src/guide/tuning.md)
  - [Rust library](docs/src/guide/library.md)
- **Architecture:** [Overview](docs/src/architecture/overview.md), then one chapter per
  component: model, engine, memory, scheduling, prefix reuse, sampling, speculation,
  kernels and serving. Each chapter ends with the decisions and measurements behind it.
- **Design:**
  - [Scope](docs/src/design/scope.md)
  - [Rejected experiments](docs/src/design/rejected.md)
  - [Open work](docs/src/design/open.md)

## Development

An Apple Silicon Mac with Rust installed through rustup. `rust-toolchain.toml` pins
Rust 1.99.0 with rustfmt and Clippy. [`just`](https://github.com/casey/just) runs every
task:

```sh
just ci           # fmt-check, strict clippy, checks with and without Metal, ~25 s test suite
just bench <case> # kernels | model | serving | prefix | kv_quality | tensorops | load_weights
```

| Crate | Owns |
| --- | --- |
| `tur-core` | Shared errors and checked packed-execution descriptors |
| `tur-metal` | State pools, completion fences, Metal kernels, packed weights, sampler |
| `tur-models` | Checkpoint loading and validation, the Qwen3.5 model, templates, tools |
| `tur-engine` | Engine, scheduler, prefix cache, speculation, pipeline, memory plan |
| `tur` | Facade and HTTP server; all integration tests and benchmarks |
| `tur-cli` | The `tur` binary |

No unsafe code, no panics, and every default backed by a measurement; see
[Workflow and rules](docs/src/development/workflow.md) and [AGENTS.md](AGENTS.md).

Licensed under Apache-2.0.
