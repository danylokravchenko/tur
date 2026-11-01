# Configuration

Settings you keep, such as the model and server limits, belong in `tur.toml`. Command-line flags are for one-off overrides and for the inputs of a single run. `run` and `serve` share the model and engine settings, and each has a section of its own.

## Start with a file

Copy the repository's commented [`tur.example.toml`](https://github.com/danylokravchenko/tur/blob/main/tur.example.toml) to `tur.toml` in your working directory. A minimal file looks like this:

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

tur serve --config ./deploy/mac.toml       # use a different file
tur serve --config ./deploy/mac.toml --listen 127.0.0.1:9000
tur run --no-config --model ./models/qwen --revision local-v1 --prompt 'Hello'
```

`--config` and `--no-config` can go before or after the command, but not together. Tur looks only for `./tur.toml`, never in parent or home directories, and runs fine without it. A file you name with `--config` must exist. Unreadable files, malformed TOML, unknown sections or keys and wrong types are errors, so a typo can't silently drop a setting. Tur reads the file once at startup, so restart to apply changes.

## Precedence and paths

For each setting, the first value found wins:

1. An explicit CLI argument.
2. `TUR_MODEL` or `TUR_REVISION`, for the model source and revision.
3. The TOML file.
4. Tur's default, or the checkpoint's.

Every key is optional, but the model source and revision must come from one of the first three. Zero and false count as values: `--prefix-snapshots 0` turns off prefix reuse that the file enabled, and `--mtp false` turns off MTP that the file enabled. Plain `--mtp` turns it on.

Local model paths in the file are relative to the file's directory. Start a relative path with `./` or `../` to mark it as local; an existing directory is also recognized. Anything else, such as `Qwen/Qwen3.5-0.8B`, names a Hugging Face repository. Paths given on the command line or in the environment are relative to the working directory. TOML strings don't expand `~` or environment variables, so use a plain absolute or relative path.

The development recipes `just run` and `just serve` pass the pinned checkpoint as arguments, which override the file's model. Use the `tur` binary directly when you want the file to choose the model.

Defaults that depend on other settings are worked out after merging. If you leave out `max_batch_tokens`, Tur sizes it from the batch size and prefill chunk you ended up with. If you leave out `prefix_snapshots`, it keeps one snapshot per running request. Sampling and thinking settings you leave out keep the checkpoint's defaults.

## What belongs where

| Location | Settings |
| --- | --- |
| `[model]` | Checkpoint source and revision, weight format, K/V format and page size, recurrent state format |
| `[engine]` | Prefill chunk size, context limit, state memory, speculation, advanced scheduler controls, device sampling |
| `[server]` | Listener and model ID, concurrency and queue limits, request deadline, prefix retention, streaming limits |
| `[generation]` | Defaults for `tur run`: output limit, seed, sampling and thinking |
| `[whisper]` | Speech-to-text checkpoint for `serve` and `transcribe`: source, revision, weight format |
| CLI only | Command, config selection, prompt, tools file, CPU reference mode, quantization jobs |
| HTTP request | Sampling, output limit, seed, thinking, tools, grammar, cache salt and other per-request controls |
| Environment | `RUST_LOG`, Hugging Face cache and token (`HF_HOME`, `HF_TOKEN`), optional `TUR_MODEL` and `TUR_REVISION` |

`[generation]` applies only to `run`. The server takes generation settings from each HTTP request and falls back to the checkpoint's defaults. `quantize` takes only explicit arguments and rejects `--config`, so a server's file can never pick a quantization source or output by accident.

## Keys shared with CLI flags

File keys are the CLI flags in snake_case. See [Command line](cli.md) for values and defaults.

| Section | Keys | CLI flag |
| --- | --- | --- |
| `model` | `source` | `--model` / `TUR_MODEL` |
| `model` | `revision` | `--revision` / `TUR_REVISION` |
| `model` | `weight_format`, `kv_format`, `kv_page_tokens`, `recurrent_state`, `audio_encoder`, `expert_cache_mb` | Same name with hyphens |
| `engine` | `prefill_chunk_size`, `context_tokens`, `state_memory_mb`, `speculative_draft`, `mtp`, `draft_model`, `draft_revision` | Same name with hyphens |
| `server` | `listen`, `served_model_name`, `max_batch_size`, `max_batch_tokens`, `max_waiting`, `request_timeout_secs`, `prefix_snapshots`, `transcription_model_name`, `max_waiting_transcriptions`, `shutdown_timeout_secs` | Same name with hyphens |
| `whisper` | `source`, `revision`, `weight_format` | `--whisper`, `--whisper-revision`, `--whisper-weight-format` |
| `generation` | `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `thinking`, `reasoning_effort`, `preserve_thinking` | Same name with hyphens |

Formats, addresses and reasoning effort are strings. Counts, seeds and durations are integers, and durations name their unit in the key. Sampling controls are numbers, and thinking and MTP are booleans. `state_memory_mb` is in MiB.

Setting `prefix_snapshots` explicitly lifts the default cap of 1/8 of state memory for snapshots, as `--prefix-snapshots` does. The state pool's total size still applies.

## Advanced settings

These controls exist in the file and the Rust library but have no CLI flag. Their defaults are the measured settings described in the architecture chapters, and most deployments never need to change them.

| Key | Default | Meaning |
| --- | --- | --- |
| `engine.device_sampling` | `true` | Sample on the GPU; `false` samples on the CPU |
| `engine.max_waiting_tokens` | 1,048,576 | Total prompt tokens allowed in the queue; more gets HTTP 429 |
| `engine.min_prefill_tokens` | `min(prefill_chunk_size, 32)` | Smallest prefill share of a pass while prompts wait; positive, at most the chunk size and below the pass token cap |
| `engine.target_pass_ms` | 60 | Target pass length while requests are decoding; the scheduler also allows twice the decode-only cost |
| `engine.max_queue_wait_ms` | 2000 | Queue age after which the oldest prompt ignores the pass target; 0 applies this immediately |
| `engine.prefix_spacing` | 256, or 1024 with an expert cache | Tokens between prefix snapshots; a positive multiple of the K/V page size |
| `engine.prefix_restorable_first` | `false` | Let requests with a cached prefix start ahead of older uncached ones; see [Prefix reuse](../architecture/prefix-reuse.md) |
| `server.stream_channel_capacity` | 64 | Queued messages per streaming request |
| `server.stream_max_buffered_bytes` | 16384 | Unsent text allowed per streaming request before a slow client is cancelled |

The waiting-request and waiting-token limits may be zero, which rejects any request that would have to wait. Tur checks scheduler, page-size, spacing and streaming settings before loading weights, and an impossible combination, such as a prefill minimum larger than its chunk, is an error.

Internal thresholds, kernel selection and memory accounting are not configurable. The file exposes operational limits, not every engine field.

## Decisions

- One optional startup file serves `run` and `serve`. It reuses the CLI's argument types and resolves into `ModelFactory`, `PipelineConfig` and `WorkerConfig`, so there is no second configuration hierarchy.
- Precedence is CLI, then environment, then file, then defaults. Explicit zero and false override the file, and derived limits are computed after merging. Tests cover both.
- Parsing is strict and discovery is explicit. Unknown keys fail instead of silently losing a deployment setting, and only the current directory is searched.
- Deployment defaults and per-run inputs live apart. Model, engine and server limits persist in the file; prompts, tool files, HTTP request controls and quantization jobs stay explicit. Rarely changed scheduler and stream limits live only in the file to keep the CLI small.
