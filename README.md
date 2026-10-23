# Tur

Tur is a Rust inference engine being redesigned for Apple Silicon. It runs the
Qwen3.5/3.8 dense hybrid text model through the packed `HybridEngine`: fused Metal
kernels (P4) over persistent, paged and optionally quantized K/V and recurrent state
pools (P5). The CLI and HTTP worker use this path. Ordinary Qwen3, Granite 4.1 and
their legacy engine, cache, tests and benchmark have been removed.

An experimental Qwen3.5 MoE text path also implements Qwen3.6-35B-A3B's expert layout.
Small-model conformance and pinned checkpoint headers pass; full 35B execution,
expert-kernel performance and quality gates remain open. See the
[P8 report](docs/apple-silicon-p8-speculation-moe.md).

The real Qwen3.5-0.8B checkpoint was validated against Transformers on CPU and Metal.
Qwen3.8-27B shares the implementation and has pinned config, tensor-header and
template validation; it has **not been executed** on the development Mac. Measurements
come from one M1; production serving hardening, quality budgets and release gates are
later stages. See the [redesign plan](docs/apple-silicon-redesign.md), the
[P3](docs/apple-silicon-p3-qwen.md), [P4](docs/apple-silicon-p4-metal.md) and
[P5](docs/apple-silicon-p5-memory.md) reports.

## Build and test

Rust 1.99 or later is required. Metal is enabled by default on macOS.

```sh
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo nextest run --workspace
cargo build --bin tur --bin server
```

The offline suite uses small tensor and tokenizer fixtures. Real checkpoint tests
are explicitly ignored and require the environment variables documented in the
[P3 report](docs/apple-silicon-p3-qwen.md). `--cpu-reference` on either binary selects
the CPU numerical reference for development checks.

Every workspace member inherits the shared Rust and Clippy lints, including forbidden
unsafe Rust and denied unwrap/expect, explicit panic and lock-across-await paths.
Fallible tests allow assertions while propagating setup and task errors with `?`.
`--no-default-features` builds the CPU reference; obsolete CUDA features are removed.

## CLI

Use a fully qualified Hugging Face repository and an immutable commit. Checkpoint
architecture fields determine support; model names do not select an architecture.
The loader rejects ordinary Qwen3, Granite and unsupported architectures or tensor
layouts. Qwen3.5 MoE is experimental; its 35B Q4K text weights alone exceed the
development Mac's 16 GB memory. Native MTP and Gemma 4 execution are disabled.

```sh
cargo run --bin tur -- \
  --model-id Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --weight-format q8_0 --prompt 'Write one sentence about the Moon.' \
  --temperature 0 --max-tokens 64
```

For a local SafeTensors checkpoint, replace `--model-id` with
`--weight-path /path/to/checkpoint` and provide its identity with `--revision`.
The directory must contain `config.json`, `tokenizer.json`, `tokenizer_config.json`,
and `model.safetensors` or `model.safetensors.index.json` plus its shards.
`chat_template.jinja` takes precedence over the tokenizer config's template.
`generation_config.json` is optional. Invalid metadata or templates fail loading.
Remote sources need Hugging Face access; use a local directory for offline loading.

`--weight-format f32|q8_0|q4k` explicitly converts the SafeTensors projection matrices
into the selected resident storage. Q8_0 is the adapter default. These are not GGUF
imports, and `q4k` does not promise a Q4_K_M quality recipe. Packed storage and
quantization limitations are documented in the P3 report.

Sampling controls (`--temperature`, `--top-k`, `--top-p`) inherit checkpoint defaults
when omitted. `--temperature 0` selects greedy generation. Thinking controls are
passed to the checkpoint template: `--thinking true|false`,
`--reasoning-effort low|medium|xhigh`, and `--preserve-thinking true|false`.
Omitting them preserves the template's defaults. No `/think` suffix is injected.
`--tools /path/to/tools.json` accepts an array of `{name, description, parameters}`
function definitions. `--prefill-chunk-size` defaults to 256.

`--speculative-draft 4` enables adaptive prompt-lookup speculation on either binary.
It is disabled by default. This throughput mode can accelerate repetitive output but
delivers tokens in bursts with longer pauses, uses extra state memory, and automatically
falls back to ordinary decoding with F16/BF16 recurrent state. Measurements and exact
verification semantics are in the [P8 report](docs/apple-silicon-p8-speculation-moe.md).

K/V state is paged: `--kv-format f32|f16|q8|q8_<group>` (default `f16`) selects storage
and `--kv-page-tokens` (default 16) the page size. `q8` halves K/V memory relative to
`f16` with a small, measured quality cost. `--state-memory-mb` sets the memory for K/V
pages and recurrent state; by default each concurrent request gets room for an 8K-token
context within the device's working set, and a budget beyond the working set is
rejected. The budget includes the permanent zero recurrent slot, two usable recurrent
slots per concurrent request, retained snapshots, speculative checkpoints when enabled,
and whole K/V pages; page rounding stays within the requested bytes.

## HTTP server

The binary loads one resident hybrid model before opening the listener. Model load
or template failures are startup errors. Choose an API ID with `--served-model-name`.

```sh
cargo run --bin server -- \
  --weight-path /path/to/checkpoint --revision my-pinned-checkpoint \
  --weight-format q8_0 --served-model-name qwen-hybrid \
  --max-batch-size 4 --prefill-chunk-size 256 --port 8080
```

Remote loading uses the same `--model-id` and `--revision` flags as the CLI.

```sh
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-hybrid","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0,"thinking":false}'
```

Set `"stream":true` for SSE text chunks, a final usage/finish chunk and `[DONE]`.
Omitting `model` selects the configured model. Request-level `temperature`, `top_k`,
`top_p`, `seed`, `thinking` (alias `enable_thinking`), `reasoning_effort` and
`preserve_thinking` are honored. Conversation turns preserve `reasoning_content`
and assistant function calls; tool response turns are rendered by the checkpoint.
Function tools use the usual `{type:"function",function:{name,description,parameters}}`
envelope. Only automatic tool selection is supported. Complete Qwen hybrid XML calls
are parsed with schema-aware argument types; the server returns function calls but
does not execute tools.

Only text input is supported. Image, video and audio content parts and placeholder
tokens are rejected. Positive generation limits must fit the model context. Usage
counts prompt tokens and sampled completion tokens, including EOS. Finish reasons
are `stop`, `length`, or `tool_calls`.

The worker bounds intake and live request counts, packs one prefill chunk or decode
token per sequence, isolates per-request failures, and releases state on completion
or client cancellation. The server takes the same K/V and state-memory flags as the
CLI. Requests that can never fit the state pool are rejected. Memory pressure can evict
prefix snapshots and preempt running state for recomputation. Packed scheduling, prefix
reuse, device sampling and bounded streaming are implemented; see the
[P6](docs/apple-silicon-p6-scheduling.md) and [P7](docs/apple-silicon-p7-serving.md) reports.
Production hardening and release gates remain open; measurements are development evidence.

## Workspace

`tur-core` owns checked descriptors and shared errors; `tur-metal` owns state pools,
completion and Metal kernels; `tur-models` owns the hybrid decoder and checkpoint metadata;
`tur-engine` owns execution and the text pipeline. `tur` provides facade exports and
the HTTP library; `tur-cli` and `tur-server` provide the binaries. Breaking Rust and
CLI changes are deliberate; legacy compatibility aliases and flags are removed.

[P0 baseline artifacts](docs/apple-silicon-baseline.md) remain as historical
measurement evidence. Their ordinary-model runners were retired with the legacy
implementation. The synthetic Metal probe remains available.

Licensed under Apache-2.0.
