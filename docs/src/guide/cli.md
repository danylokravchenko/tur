# Command line

`tur` has four commands:

```text
tur run         Generate one completion and stream it to stdout
tur serve       Serve one model over an OpenAI-compatible HTTP API
tur transcribe  Transcribe a WAV file with a Whisper checkpoint
tur quantize    Prepack weights offline, optionally calibrated, for instant loading
tur prune       Experiment: keep the experts a gpt-oss checkpoint routes to most
```

`tur <command> --help` prints every flag. This page groups the flags by purpose and gives their defaults.

## Configuration

`run`, `serve` and `transcribe` read `./tur.toml` if it exists; see [Configuration](configuration.md) for the file format and the settings that only the file exposes. Flags override `TUR_MODEL` and `TUR_REVISION`, which override the file. `--config <path>` picks another file and `--no-config` skips it.

## Model and state (`run`, `serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model <dir or org/name>` | required (flag, `TUR_MODEL` or file) | An existing directory loads locally; anything else names a Hugging Face repository |
| `--revision <rev>` | required (flag, `TUR_REVISION` or file) | A 40-character commit for remote models, or a name you choose for local ones |
| `--weight-format f32\|q8_0\|q4k` | `q8_0` | Convert SafeTensors weights to this format while loading. Leave it out for a [prepacked artifact](quantization.md) |
| `--kv-format f32\|f16\|q8\|q8_<group>` | `f16` | Storage for attention keys and values; `q8` uses one scale per 32 values |
| `--kv-page-tokens <n>` | 16 | Tokens per K/V page |
| `--recurrent-state f32\|f16` | `f32` | Storage for DeltaNet recurrent state |
| `--audio-encoder [true\|false]` | `true` | Load the checkpoint's audio encoder when it has one (Gemma 4 E2B and E4B); `false` saves its memory |
| `--expert-cache-mb <MiB>` | every expert resident | Keep gpt-oss experts on disk and read them into this much memory; see [Expert cache](../architecture/expert-cache.md) |
| `--cpu-reference` | off | Run the slow F32 CPU reference instead of Metal, for development |

Gemma 4 uses the same commands. For the E2B model used in development:

```sh
tur run --model google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --weight-format q8_0 --temperature 0 --thinking false --prompt 'What is the capital of France?'
```

Gemma's `q4k` needs every projection's input width to be a multiple of 256. Gemma drafts speculative tokens with Google's published assistant model rather than `--mtp`:

```sh
tur run --model google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --draft-model google/gemma-4-E2B-it-assistant \
  --draft-revision 2d874ef7d29f9a30599a1e4b3c1cbc9595f005df --prompt 'Explain river deltas.'
```

## Passes and memory (`run`, `serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prefill-chunk-size <n>` | 256, or 1024 with `--expert-cache-mb` | Most prompt tokens one request adds to one pass |
| `--state-memory-mb <n>` | derived | Memory for K/V pages and recurrent state. By default, room for an 8K-token context per concurrent request |
| `--speculative-draft <n>` | 0 | Most tokens drafted from the request's own text per step; 0 turns speculation off |
| `--mtp [true\|false]` | off | Also draft with the checkpoint's multi-token-prediction head; turns off prefix reuse |
| `--draft-model <dir\|repo>` | none | Draft with a separate assistant checkpoint (Gemma 4); turns off prefix reuse |
| `--draft-revision <commit>` | none | Commit of a remote `--draft-model`, or a name for a local one |

Speculation needs F32 recurrent state. With `--recurrent-state f16`, Tur logs a warning and decodes normally. [Speculative decoding](../architecture/speculation.md) explains when it pays.

## Generation (`run`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prompt <text>` | `Who are you?` | One user turn, rendered with the checkpoint's chat template |
| `-n`, `--max-tokens <n>` | 128 | Most tokens to generate, counting EOS |
| `--temperature`, `--top-k`, `--top-p` | checkpoint defaults | Sampling; `--temperature 0` is greedy |
| `--repetition-penalty <x>` | none | Above 1 discourages any token already in the prompt or output |
| `--presence-penalty <x>` | none | Subtracted once from each generated token's logit |
| `--frequency-penalty <x>` | none | Subtracted once per occurrence of each generated token |
| `--seed <n>` | 299792458 | Seed of the request's random stream |
| `--thinking true\|false` | template default | Passed to the chat template |
| `--reasoning-effort low\|medium\|xhigh` | template default | Passed to the chat template |
| `--preserve-thinking true\|false` | template default | Passed to the chat template |
| `--tools <file>` | none | JSON array of `{name, description, parameters}` function definitions |
| `--audio <file.wav>` | none | A clip placed before the prompt, for models with an audio encoder |

Sampling settings you leave out come from the checkpoint's `generation_config.json`. Thinking settings you leave out keep the template's defaults, and Tur adds nothing to the prompt.

## Serving (`serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen <addr>` | `127.0.0.1:8080` | Opened after the model loads |
| `--served-model-name <id>` | `qwen-hybrid` | The model ID clients send and `/v1/models` lists |
| `--max-batch-size <n>` | 8 | Requests running at once |
| `--max-batch-tokens <n>` | 2048 or more | Tokens in one pass; always room for one prompt chunk beside every decode |
| `--max-waiting <n>` | 256 | Queued requests before new ones get HTTP 429 |
| `--request-timeout-secs <n>` | none | Fail requests not finished this long after arrival (HTTP 504) |
| `--prefix-snapshots <n>` | one per running request | Prefix snapshots kept for reuse; 0 turns prefix reuse off |
| `--transcription-model-name <id>` | `whisper-1` | The model ID of `--whisper` at `/v1/audio/*` and in `/v1/models` |
| `--max-waiting-transcriptions <n>` | 16 | Queued clips before new ones get HTTP 429 |

## Speech to text (`serve`, `transcribe`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--whisper <dir or org/name>` | none | Whisper checkpoint; `serve` loads it beside the chat model |
| `--whisper-revision <rev>` | required with `--whisper` | A 40-character commit for remote models, or a name for local ones |
| `--whisper-weight-format f32\|q8_0\|q4k` | `q8_0` | Storage for Whisper's projections |
| `--language <code>` | detected | `transcribe` only: spoken language, such as `en` |
| `--translate` | off | `transcribe` only: write English text instead of a transcript |
| `--format text\|srt\|vtt\|words` | `text` | `transcribe` only: plain text, timed subtitles, or one `start`, `end`, `word` line per word |
| `--prompt <text>` | none | `transcribe` only: text the decoder reads before every window |
| `--condition-on-previous-text` | off | `transcribe` only: also read the text decoded so far |
| `--temperature <t>` | retries | `transcribe` only: sample at this temperature instead of retrying doubtful windows |

```sh
tur transcribe --whisper openai/whisper-large-v3-turbo \
  --whisper-revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 speech.wav

tur run --model google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --audio speech.wav --prompt 'Summarize this recording.' --thinking false
```

## Quantization (`quantize`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model <dir>` | required | Source SafeTensors checkpoint directory |
| `--revision <rev>` | required | Name of the source, recorded in the artifact |
| `--output <dir>` | required | New artifact directory |
| `--method rtn\|imatrix\|gptq` | `gptq` | Rounding method; `imatrix` and `gptq` need calibration text |
| `--weight-format q8_0\|q4k\|f32` | `q4k` | Format of every projection, or the base format under `--budget-bits` |
| `--budget-bits <bits>` | none | Average bits per weight; upgrades the most sensitive projections to `q8_0` |
| `--embedding head\|f32\|bf16\|q8_0\|q6k\|q5k\|q4k` | `head` if tied, else `bf16` | Embedding storage; `head` reads embeddings from the packed output layer |
| `--calibration <files>` | none | Comma-separated calibration text files |
| `--calibration-tokens <n>` | 16384 | Calibration tokens, split into samples |
| `--sample-tokens <n>` | 512 | Tokens per calibration sample |
| `--sensitivity-tokens <n>` | 1024 | Tokens used to measure each projection's sensitivity |
| `--damp <x>` | 0.01 | GPTQ damping, relative to the mean Hessian diagonal |
| `--report <file>` | stdout | Calibration, sensitivity and plan details as JSON |

[Prepacked weights](quantization.md) walks through the workflow.

## Logging

Logs go to stderr through `tracing`. `RUST_LOG` sets the filter, for example `RUST_LOG=tur_engine=debug`; the default is `info`.

## Decisions

- Common flags stay available as overrides beside the file, and `--mtp false` can undo a file's `true`. Rarely changed scheduler and stream limits live only in the file to keep the CLI manageable.
- Per-run inputs stay explicit: prompts, tool files, CPU reference mode and quantization jobs are always flags, and logging stays in `RUST_LOG`.
