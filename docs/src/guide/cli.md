# Command line

`tur` has three commands:

```text
tur run       Generate one completion and stream it to stdout
tur serve     Serve one resident model over an OpenAI-style HTTP API
tur quantize  Prepack weights offline, optionally calibrated, for direct loading
```

`tur <command> --help` prints every flag. This page lists their defaults and groups them
by purpose.

## Configuration

`run` and `serve` read `./tur.toml` when it exists. Start with the repository's
`tur.example.toml`; see [Configuration](configuration.md) for every key, file-only
scheduler and streaming settings, and examples.

Explicit CLI arguments override `TUR_MODEL` / `TUR_REVISION`, which override file values.
Omitted settings keep their existing defaults. `--config <path>` selects another file;
`--no-config` disables discovery. Both flags work before or after the command and are
mutually exclusive. Prompts, tools files, CPU reference mode and quantization jobs use
CLI arguments. `[generation]` defaults apply to `run`; HTTP generation settings stay in
each request.

## Model and state (`run`, `serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model <dir or org/name>` | required via flag, `TUR_MODEL` or TOML | An existing directory loads locally; anything else names a Hugging Face repository |
| `--revision <rev>` | required via flag, `TUR_REVISION` or TOML | A 40-character commit for remote models; your own identity for local ones |
| `--weight-format f32\|q8_0\|q4k` | `q8_0` | Convert SafeTensors projections into this resident format at load time. Omit it for a [prepacked artifact](quantization.md), which loads as stored |
| `--kv-format f32\|f16\|q8\|q8_<group>` | `f16` | Storage of attention keys and values; `q8` uses one scale per 32 values |
| `--kv-page-tokens <n>` | 16 | Tokens per K/V page |
| `--recurrent-state f32\|f16` | `f32` | Storage of DeltaNet matrix state |
| `--cpu-reference` | off | Run the F32 CPU reference instead of Metal (development only) |

Gemma 4 uses the same `run` and `serve` commands. For the pinned E2B development model:

```sh
tur run --model google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --weight-format q8_0 --temperature 0 --thinking false --prompt 'What is the capital of France?'
```

Input remains text-only. Gemma native `--mtp` and offline `tur quantize` are not
implemented. Use load-time conversion; Q4K additionally requires every projection's
input width to fit its 256-element blocks.

## Passes and memory (`run`, `serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prefill-chunk-size <n>` | 256 | Largest prefill chunk one request contributes to one pass |
| `--state-memory-mb <n>` | derived | Memory for K/V pages and recurrent state. By default each concurrent request gets room for an 8K-token context within the device's working set |
| `--speculative-draft <n>` | 0 | Most prompt-lookup drafts per decode; 0 disables speculation |
| `--mtp [true\|false]` | off | Also draft with the checkpoint's multi-token-prediction head; implies at least one draft and disables prefix reuse |

Speculation needs F32 recurrent state; with `--recurrent-state f16` it is switched off
with a warning. See [Speculative decoding](../architecture/speculation.md) for when it
pays.

## Generation (`run`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prompt <text>` | `Who are you?` | One user turn, rendered with the checkpoint's chat template |
| `-n`, `--max-tokens <n>` | 128 | Generated-token limit, including EOS |
| `--temperature`, `--top-k`, `--top-p` | checkpoint defaults | Sampling; `--temperature 0` is greedy |
| `--repetition-penalty <x>` | none | Above 1 discourages every prompt or generated token |
| `--presence-penalty <x>` | none | Subtracted once from each generated token's logit |
| `--frequency-penalty <x>` | none | Subtracted per occurrence of each generated token |
| `--seed <n>` | 299792458 | Seed of the request's random stream |
| `--thinking true\|false` | template default | Passed to the chat template |
| `--reasoning-effort low\|medium\|xhigh` | template default | Passed to the chat template |
| `--preserve-thinking true\|false` | template default | Passed to the chat template |
| `--tools <file>` | none | JSON array of `{name, description, parameters}` function definitions |

Sampling controls you omit inherit the checkpoint's `generation_config.json`. Thinking
controls you omit leave the template's own defaults in place; nothing is appended to the
prompt.

## Serving (`serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen <addr>` | `127.0.0.1:8080` | Bound after the model loads |
| `--served-model-name <id>` | `qwen-hybrid` | The model ID clients send and `/v1/models` lists |
| `--max-batch-size <n>` | 8 | Requests holding model state at once |
| `--max-batch-tokens <n>` | 2048 or more | Tokens in one packed pass; at least one prefill chunk beside every decode |
| `--max-waiting <n>` | 256 | Queued requests before new ones are rejected with HTTP 429 |
| `--request-timeout-secs <n>` | none | Fail requests not finished this long after arrival (HTTP 504) |
| `--prefix-snapshots <n>` | one per running request | Prefix snapshots kept beside running requests; 0 disables prefix reuse |

## Quantization (`quantize`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model <dir>` | required | Source SafeTensors checkpoint directory |
| `--revision <rev>` | required | Identity of the source, recorded in the artifact |
| `--output <dir>` | required | New artifact directory |
| `--method rtn\|imatrix\|gptq` | `gptq` | Rounding method; `imatrix` and `gptq` need calibration text |
| `--weight-format q8_0\|q4k\|f32` | `q4k` | Every projection's format, or the narrowest one under `--budget-bits` |
| `--budget-bits <bits>` | none | Average bits per projection weight; upgrades the most sensitive projections to `q8_0` |
| `--embedding head\|f32\|bf16\|q8_0\|q6k\|q5k\|q4k` | `head` if tied, else `bf16` | Embedding storage; `head` gathers rows from the packed LM head |
| `--calibration <files>` | none | Comma-separated calibration text files |
| `--calibration-tokens <n>` | 16384 | Calibration tokens, split into samples |
| `--sample-tokens <n>` | 512 | Tokens per calibration sample |
| `--sensitivity-tokens <n>` | 1024 | Tokens over which sensitivities are measured |
| `--damp <x>` | 0.01 | GPTQ damping relative to the mean Hessian diagonal |
| `--report <file>` | stdout | Calibration, sensitivity and plan details as JSON |

See [Prepacked weights](quantization.md) for the workflow.

## Logging

Logs go to stderr through `tracing`. `RUST_LOG` sets the filter (default `info`), for
example `RUST_LOG=tur_engine=debug`.


## Decisions

- **Preserve common flags as overrides.** Model, memory, generation and serving flags
  remain available alongside TOML; `--mtp false` can undo a file's boolean default.
  Less common scheduler and stream bounds are file-only to keep the CLI manageable.
- **Keep invocation inputs explicit.** Prompts, tool files, CPU reference mode and
  quantization jobs remain command-line inputs; logging remains in `RUST_LOG`.
