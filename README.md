# Tur

Tur is a Rust inference engine for Apple Silicon. It runs Qwen3.5/3.8 dense hybrid
text models (Gated DeltaNet plus gated attention) on fused Metal kernels, over paged
and optionally quantized K/V and recurrent state. It also does packed continuous
batching, prefix reuse, device sampling and exact speculative decoding.

The Qwen3.5-0.8B checkpoint is validated against Transformers on CPU and Metal.
Qwen3.8-27B shares the code path but has not been executed. The Qwen3.6 MoE path is
experimental. Every measurement so far comes from one M1. See
[design decisions](docs/decisions.md) for what was chosen, what was rejected and what
is still open.

## Build and test

Rust 1.99 or later. Metal is enabled by default; `--no-default-features` builds the
CPU reference only.

```sh
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo nextest run --workspace
cargo build --release --bin tur
```

Integration tests for every crate are linked into one binary,
`crates/tur/tests/integration`, with fixtures in `crates/tur/tests/fixtures`. Test builds
optimize dependencies, `tur-metal` and `tur-models`, because their host reference math
dominates test time otherwise. Benchmarks and measurement tools are one binary as well:
`cargo bench -p tur --bench perf -- <kernels|model|serving|prefix|kv_quality|tensorops|load_weights>`.

The offline suite uses small tensor and tokenizer fixtures. Real-checkpoint tests are
`#[ignore]`d and read `TUR_QWEN35_0_8B` (a local snapshot of the pinned 0.8B), plus
`TUR_QWEN35_GOLDEN` and `TUR_QWEN35_MTP_GOLDEN`, which are written by
`scripts/qwen3_5_reference.py` and `scripts/qwen3_5_mtp_reference.py`.

## Usage

One binary, three commands:

```sh
tur run      --model <dir|org/name> --revision <rev> [--prompt ...]   # one completion to stdout
tur serve    --model <dir|org/name> --revision <rev> [--listen ...]   # OpenAI-style HTTP API
tur quantize --model <dir> --revision <rev> --output <dir>            # prepack weights offline
```

`--model` is a local checkpoint directory or a fully qualified Hugging Face repository.
A remote `--revision` must be a 40-character commit. For a local directory it is the
identity you assign. A directory needs `config.json`, `tokenizer.json`,
`tokenizer_config.json` and SafeTensors weights. `chat_template.jinja` and
`generation_config.json` are optional. `TUR_MODEL` and `TUR_REVISION` set both flags
from the environment.

```sh
cargo run --release --bin tur -- run \
  --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --prompt 'Write one sentence about the Moon.' --temperature 0 --max-tokens 64
```

### Shared by `run` and `serve`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--weight-format f32\|q8_0\|q4k` | `q8_0` | Convert SafeTensors projections at load time. Omit it for a prepacked artifact |
| `--kv-format f32\|f16\|q8\|q8_<group>` | `f16` | K/V storage; `q8` halves memory at a small measured quality cost |
| `--kv-page-tokens` | 16 | Tokens per K/V page |
| `--recurrent-state f32\|f16` | `f32` | F16 halves recurrent memory; its error grows with decode length |
| `--state-memory-mb` | derived | K/V and recurrent memory; default fits an 8K context per concurrent request |
| `--prefill-chunk-size` | 256 | Largest prefill chunk of one request per pass |
| `--speculative-draft N` | 0 | Prompt-lookup drafts per decode. Throughput mode: tokens arrive in bursts |
| `--mtp` | off | Also draft with the checkpoint's MTP head; disables prefix reuse |
| `--cpu-reference` | off | Run the CPU numerical reference |

`run` also takes the sampling flags `--temperature`, `--top-k` and `--top-p`, the
penalty flags, and `--seed`. Sampling flags you omit inherit the checkpoint's defaults;
`--temperature 0` is greedy. It also takes `--thinking`, `--reasoning-effort` and
`--preserve-thinking`, which are passed to the checkpoint template, and `--tools` with
a JSON array of `{name, description, parameters}`.

### Serving

```sh
cargo run --release --bin tur -- serve \
  --model /path/to/checkpoint --revision my-checkpoint --served-model-name qwen-hybrid \
  --max-batch-size 8 --listen 127.0.0.1:8080

curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen-hybrid","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The model loads and validates before the listener binds. `serve` also takes
`--max-batch-tokens`, `--max-waiting`, `--request-timeout-secs` and
`--prefix-snapshots` (0 disables prefix reuse).

The endpoints are `/v1/models` and `/v1/chat/completions`. `"stream": true` returns SSE
text chunks, a final usage chunk and `[DONE]`. Requests can set `temperature`, `top_k`,
`top_p`, the penalties, `seed`, `thinking` (alias `enable_thinking`),
`reasoning_effort` and `preserve_thinking`. Function tools use the usual
`{type:"function",function:{...}}` envelope with automatic selection. Calls are parsed
and returned but never executed. Only text input is accepted. Finish reasons are `stop`,
`length` and `tool_calls`. A streaming client that stops reading is cancelled with a
`slow_client` error.

### Prepacked weights

`tur quantize` writes a versioned artifact that `run` and `serve` load directly. By
default it uses Q4K with GPTQ, which needs `--calibration` text files. Use
`--method rtn` for uncalibrated rounding. `--budget-bits 5.5` upgrades the most
sensitive projections to `q8_0`. `--report` writes calibration and plan details as
JSON.

## Workspace

| Crate | Owns |
| --- | --- |
| `tur-core` | Shared errors and checked packed-execution descriptors |
| `tur-metal` | State pools, completion fences, Metal kernels, packed weights |
| `tur-models` | Checkpoint loading and validation, the Qwen3.5 model, templates |
| `tur-engine` | Engine, scheduler, prefix cache, speculation, sampling, pipeline |
| `tur` | Facade re-exports and the HTTP server library |
| `tur-cli` | The `tur` binary |

Licensed under Apache-2.0.
