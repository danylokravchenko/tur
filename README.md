# Tur

Tur is a Rust inference engine being redesigned for Apple Silicon. P3 is complete:
Qwen3.5/3.8 dense hybrid text execution, checkpoint loading, the CLI and the HTTP
worker use the packed `HybridEngine` path. Ordinary Qwen3, Granite 4.1 and their
legacy engine, cache, tests and benchmark have been removed.

The real Qwen3.5-0.8B checkpoint was validated against Transformers on CPU and Metal.
Qwen3.8-27B shares the implementation and has pinned config, tensor-header and
template validation; it has **not been executed** on the development Mac.
This remains reference execution. Production performance, quality budgets and
serving hardening are later stages. See the [redesign plan](docs/apple-silicon-redesign.md)
and [P3 evidence and limitations](docs/apple-silicon-p3-qwen.md).

## Build and test

Rust 1.99 or later is required. Metal is enabled by default on macOS.

```sh
cargo check --workspace --all-targets
cargo nextest run --workspace
cargo build --bin tur --bin server
```

The offline suite uses small tensor and tokenizer fixtures. Real checkpoint tests
are explicitly ignored and require the environment variables documented in the
[P3 report](docs/apple-silicon-p3-qwen.md). `--cpu-reference` on either binary selects
the CPU numerical reference for development checks.

## CLI

Use a fully qualified Hugging Face repository and an immutable commit. Checkpoint
architecture fields determine support; model names do not select an architecture.
The loader rejects ordinary Qwen3, Granite, MoE and unsupported tensor layouts.

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
or client cancellation. Memory-based admission, prefix reuse, production scheduling,
bounded streaming output and comprehensive serving controls remain P6/P7/P9 work.
There are no performance or production-readiness claims.

## Workspace

`tur-core` owns checked descriptors and shared errors; `tur-metal` owns hybrid state
and completion; `tur-models` owns the hybrid decoder and checkpoint metadata;
`tur-engine` owns execution and the text pipeline. `tur` provides facade exports and
the HTTP library; `tur-cli` and `tur-server` provide the binaries. Breaking Rust and
CLI changes are deliberate; legacy compatibility aliases and flags are removed.

[P0 baseline artifacts](docs/apple-silicon-baseline.md) remain as historical
measurement evidence. Their ordinary-model runners were retired with the legacy
implementation. The synthetic Metal probe remains available.

Licensed under Apache-2.0.
