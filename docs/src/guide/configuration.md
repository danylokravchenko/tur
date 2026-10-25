# Configuration

Keep persistent model and deployment settings in `tur.toml`. Use command-line arguments
for occasional overrides and the input to a particular invocation. `run` and `serve`
share the model and engine settings; each also has its own section.

## Start with a file

Copy the repository's commented [`tur.example.toml`](https://github.com/danylokravchenko/tur/blob/main/tur.example.toml)
to `tur.toml` in your working directory. A minimal file is:

```toml
[model]
source = "Qwen/Qwen3.5-0.8B"
revision = "2fc06364715b967f1860aea9cf38778875588b17"

[server]
served_model_name = "qwen"
max_batch_size = 4

[generation]
max_tokens = 256
thinking = false
```

Then:

```sh
tur run --prompt 'Write one sentence about the Moon.'
tur serve

tur serve --config ./deploy/mac.toml       # select a different file
tur serve --config ./deploy/mac.toml --listen 127.0.0.1:9000
tur run --no-config --model ./models/qwen --revision local-v1 --prompt 'Hello'
```

`--config` and `--no-config` can appear before or after the command. They are mutually
exclusive. The default discovery checks only `./tur.toml`; it does not search parent
or home directories. A missing default file is fine. An explicitly selected file must
exist. Unreadable files, malformed TOML, unknown sections or keys, and invalid types are
errors. Files are read once at startup; restart to apply changes.

## Precedence and paths

For each setting, the first supplied value wins:

1. An explicit CLI argument.
2. `TUR_MODEL` or `TUR_REVISION`, for the model source or revision.
3. The selected TOML file.
4. Tur's built-in or checkpoint-derived default.

Every key and section is optional. A model source and revision must be provided through
one of the first three sources. Zero and false are explicit values: for example,
`--prefix-snapshots 0` disables file-configured prefix reuse, and `--mtp false` disables
file-configured MTP. Plain `--mtp` enables it.

A file's local model paths are relative to the file's directory. Use `./` or `../` to
identify a relative local path explicitly; an existing directory is also recognized.
Other values such as `Qwen/Qwen3.5-0.8B` name Hugging Face repositories. CLI and environment
paths are relative to the working directory. TOML strings do not expand `~` or environment
variables. Use an absolute path or a relative path instead.

The development `just run` and `just serve` recipes supply the pinned checkpoint as
explicit arguments; those arguments override the file's model selection. Use the `tur`
binary when the file should select the model.

Derived defaults use the resolved settings. For example, omitting `max_batch_tokens`
sizes the token cap from the effective batch size and prefill chunk, and omitting
`prefix_snapshots` retains one snapshot per effective running request, within its memory
share. Omitting sampling and thinking controls preserves checkpoint defaults.

## What belongs where

| Location | Settings |
| --- | --- |
| `[model]` | Checkpoint source and revision, weight format, K/V format and page size, recurrent state format |
| `[engine]` | Prefill chunk size, state memory, speculation, advanced scheduler controls and device sampling |
| `[server]` | Listener and model ID, concurrency and queue bounds, request deadline, prefix retention and streaming bounds |
| `[generation]` | Reusable defaults for `tur run`: output limit, seed, sampling and thinking |
| CLI only | Command, config selection, prompt, tools JSON file, CPU reference mode, quantization job inputs and settings |
| HTTP request | Sampling, output limit, seed, thinking, tools, grammar, cache salt and other request-specific controls |
| Environment | `RUST_LOG`, Hugging Face cache and authentication (`HF_HOME`, `HF_TOKEN`), optional `TUR_MODEL` and `TUR_REVISION` overrides |

`[generation]` applies only to `run`. The server continues to use each HTTP request's
settings with the checkpoint's defaults for omitted sampling and thinking controls.
`quantize` uses explicit job arguments and does not discover `tur.toml`; passing
`--config` to it is an error. This keeps a resident model's defaults from selecting a
quantization source or output accidentally.

## Keys shared with CLI arguments

The file uses snake_case keys corresponding to the CLI's kebab-case flags. See
[Command line](cli.md) for the values and defaults of these options.

| Section | Keys | CLI mapping |
| --- | --- | --- |
| `model` | `source` | `--model` / `TUR_MODEL` |
| `model` | `revision` | `--revision` / `TUR_REVISION` |
| `model` | `weight_format`, `kv_format`, `kv_page_tokens`, `recurrent_state` | Same name with hyphens |
| `engine` | `prefill_chunk_size`, `state_memory_mb`, `speculative_draft`, `mtp` | Same name with hyphens |
| `server` | `listen`, `served_model_name`, `max_batch_size`, `max_batch_tokens`, `max_waiting`, `request_timeout_secs`, `prefix_snapshots` | Same name with hyphens |
| `generation` | `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `thinking`, `reasoning_effort`, `preserve_thinking` | Same name with hyphens |

Formats, listener addresses and reasoning effort are strings; counts, seeds and durations
are integers; sampling controls are numbers; thinking and MTP are booleans. Durations
carry their unit in the key. `state_memory_mb` uses MiB (1,048,576 bytes).

An explicit `prefix_snapshots` count removes the default 1/8 state-memory share cap,
matching `--prefix-snapshots`. The state pool's overall memory limit still applies.

## Advanced file settings

These controls are available in TOML and the Rust library. They have no additional CLI
flags. The defaults remain the measured settings described in the architecture chapters.

| Key | Default | Meaning |
| --- | --- | --- |
| `engine.device_sampling` | `true` | Sample unconstrained requests on the GPU; false uses the host sampler |
| `engine.max_waiting_tokens` | 1,048,576 | Total prompt tokens allowed in the waiting queue; excess submissions get HTTP 429 |
| `engine.min_prefill_tokens` | `min(prefill_chunk_size, 32)` | Minimum prefill grant while work exists; positive, at most the chunk size and below the pass token cap |
| `engine.target_pass_ms` | 60 | Positive target pass duration while decodes run; the adaptive scheduler also allows twice the estimated decode cost |
| `engine.max_queue_wait_ms` | 2000 | Queue age after which the oldest prefill ignores the cost target; 0 applies the guard immediately |
| `engine.prefix_spacing` | 256 | Tokens between snapshots; positive multiple of the K/V page size while reuse is enabled |
| `server.stream_channel_capacity` | 64 | Positive number of queued text messages per streaming request |
| `server.stream_max_buffered_bytes` | 16384 | Positive pending-text limit per streaming request; exceeding it cancels a slow client |

The waiting request and token bounds may be zero to reject waiting work. Scheduler,
page-size, prefix-spacing and stream constraints are checked before loading weights.
Invalid combinations are errors, including a prefill minimum greater than its chunk or
pass token cap.

Internal algorithm thresholds, MTP vocabulary and calibration policy, device memory
accounting and kernel selection remain library implementation choices. The file exposes
operational limits without adding flags for every engine field.

## Decisions

- **One optional startup file for `run` and `serve`.** Shared settings reuse the CLI's
  existing argument types and resolve into `ModelFactory`, `PipelineConfig` and
  `WorkerConfig`; no engine configuration hierarchy is added.
- **CLI > environment > file > defaults.** Explicit values, including zero and false,
  override file defaults; derived limits are calculated after merging. Tests cover both.
- **Strict parsing and explicit discovery.** Unknown keys fail instead of silently losing
  a deployment setting. Only the current directory is searched, and `--no-config`
  bypasses discovery.
- **Deployment defaults and invocation inputs have separate homes.** Model, engine and
  server limits persist; prompts, tool files, HTTP request controls and quantization jobs
  remain explicit inputs. Advanced scheduler and stream bounds use TOML instead of more
  CLI flags.
