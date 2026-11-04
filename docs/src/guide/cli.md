# Command line

Run `tur` alone for a welcome screen listing the commands:

```text
tur run         Chat with a model, or answer one prompt: the answer to stdout, reasoning to stderr
tur serve       Serve one model over an OpenAI-compatible HTTP API, with a live dashboard
tur list        List the model names and which are downloaded
tur pull        Download a model ahead of its first run
tur status      Print a running server's state once
tur top         Watch a running server live
tur stop        Stop a server on this machine
tur transcribe  Transcribe a WAV file with a Whisper checkpoint
tur quantize    Prepack weights offline, optionally calibrated, for instant loading
tur prune       Experiment: write a gpt-oss checkpoint with fewer experts
```

`tur <command> --help` prints every flag. This page groups the flags by purpose and gives their defaults.

## What you see

Every command shares one way of reporting. Long work draws a progress bar with an estimate: downloads, weight loading, calibration, packing, sensitivity measurement, expert probing and transcription. Each stage leaves a line with what it did and how long it took, so a finished job reads as a short record. A summary follows.

All of this goes to stderr. Stdout holds only the command's result, so `tur run … > answer.txt` and `tur quantize … --json | jq` stay clean. When stderr isn't a terminal, bars become a plain line every 10 seconds and logs keep their timestamps, threads and targets for whatever collects them.

| Flag or variable | Effect |
| --- | --- |
| `-q`, `--quiet` | Only warnings, errors and the result |
| `NO_COLOR` | No color; the banner appears without its animation |
| `RUST_LOG` | Log filter. In a terminal the default is `warn`, or `info` for `serve`; otherwise `info` |

## Watching a server

In a terminal, `tur serve` opens a dashboard once its listener is up, with traffic, memory, efficiency, a throughput chart and the server's logs. `q` closes it and the server keeps running, printing the logs it held and any that follow. Ctrl-C drains the server, and a second Ctrl-C stops it at once. `--dashboard=false`, or `dashboard = false` under `[server]`, keeps plain logs instead.

From another terminal or machine, `tur top` shows the same dashboard without logs, and `tur status` prints one reading:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url <http://host:port>` | `TUR_URL`, else `[server].listen`, else `127.0.0.1:8080` | Server to read |
| `--force` | off | `stop` only: fail running requests and stop at once |
| `--interval <seconds>` | 1 | `top` only: time between readings |
| `--json` | off | `status` only: print `/diagnostics` as it is |

`tur status` exits nonzero when nothing answers, or when the server is draining or a worker has stopped, so a script can gate on it. Both read [`/diagnostics`](http-api.md), so they agree with what any other client sees.

`tur stop` works like Ctrl-C from anywhere on the machine: running requests finish, up to `--shutdown-timeout-secs`, and it returns once the process has exited and released its device memory. It takes the same `--url` and refuses a server on another machine.

Throughput counts tokens through the model, prompts and speculative drafts included, because a request's generated tokens only count once it finishes. Device busy is the share of wall time a pass was on the device. First-token and request latencies are upper bounds from the histogram's buckets, over the server's whole life.

## Configuration

`run`, `serve`, `transcribe`, `status`, `top` and `stop` read `./tur.toml` if it exists; see [Configuration](configuration.md) for the file format and the settings that only the file exposes. Flags override the file. A model on the command line replaces the file's source and revision together, because the file's revision names a commit of its own model. `--config <path>` picks another file and `--no-config` skips it.

## Choosing a model (`run`, `serve`, `pull`)

Every one of these commands takes the model as its first argument:

```sh
tur run qwen                # a name from `tur list`
tur run qwen3.5:0.8b        # the same model by its full name
tur run openai/gpt-oss-20b  # a Hugging Face repository
tur run ./my-checkpoint     # a directory
```

| Name | Full name | Repository | What it brings |
| --- | --- | --- | --- |
| `qwen` | `qwen3.5:0.8b` | `Qwen/Qwen3.5-0.8B` | Its commit |
| `gemma` | `gemma4:e2b` | `google/gemma-4-E2B-it` | Its commit |
| `gpt-oss` | `gpt-oss:20b` | `openai/gpt-oss-20b` | Its commit, `--expert-cache-mb 2048`, and for `serve` `--max-batch-size 1` |

Flags and the file override what a name brings. To run `gpt-oss` with every expert resident on a Mac with room for it, name the repository, which brings nothing. Without a model, `run` and `serve` use `[model].source` from the file.

`tur pull <model>` downloads without loading and prints the snapshot's directory. `tur list` marks a model as downloaded when the cache holds every weight file of its pinned commit.

## Model and state (`run`, `serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--revision <rev>` | The name's commit, a repository's latest, or a directory's name | A 40-character commit for remote models, or a name you choose for local ones |
| `--weight-format f32\|q8_0\|q4k` | `q8_0` | Convert SafeTensors weights to this format while loading. Leave it out for a [prepacked artifact](quantization.md) |
| `--gguf <path or org/name/file.gguf>` | none (flag or `TUR_GGUF`) | Load weights from a [GGUF file](gguf.md) as stored; the model argument then supplies the config and a tokenizer that must match the file; a template in the file replaces the checkpoint's |
| `--gguf-revision <commit>` | the repository's latest (flag or `TUR_GGUF_REVISION`) | 40-character commit of a remote `--gguf` file |
| `--kv-format f32\|f16\|q8\|q8_<group>` | `f16` | Storage for attention keys and values; `q8` uses one scale per 32 values |
| `--kv-page-tokens <n>` | 16 | Tokens per K/V page |
| `--recurrent-state f32\|f16` | `f32` | Storage for DeltaNet recurrent state |
| `--audio-encoder [true\|false]` | `true` | Load the checkpoint's audio encoder when it has one (Gemma 4 E2B and E4B); `false` saves its memory |
| `--vision-encoder [true\|false]` | `true` | Load the checkpoint's vision encoder when it has one (Qwen3.5 and Gemma 4); `false` saves its memory |
| `--video-input [true\|false]` | `true` | Accept video through the vision encoder; `false` refuses it before any decoder reads the file, for untrusted clients, and images still work |
| `--expert-cache-mb <MiB>` | every expert resident | Keep the experts of gpt-oss, a Qwen artifact or a Qwen GGUF file on disk and read them into this much memory; see [Expert cache](../architecture/expert-cache.md) |
| `--cpu-reference` | off | Run the slow F32 CPU reference instead of Metal, for development |

Gemma 4 uses the same commands. For the E2B model used in development:

```sh
tur run gemma 'What is the capital of France?' --temperature 0 --thinking false
```

Gemma's `q4k` needs every projection's input width to be a multiple of 256. Gemma drafts speculative tokens with Google's published assistant model rather than `--mtp`:

```sh
tur run gemma 'Explain river deltas.' --draft-model google/gemma-4-E2B-it-assistant \
  --draft-revision 2d874ef7d29f9a30599a1e4b3c1cbc9595f005df
```

## Passes and memory (`run`, `serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prefill-chunk-size <n>` | 256, or 1024 with `--expert-cache-mb` | Most prompt tokens one request adds to one pass |
| `--context-tokens <n>` | what the pool and model allow | Most tokens one request may hold, prompt and output together. Sizes the state pool for that many per concurrent request; a start that can't fit it fails and names what would |
| `--state-memory-mb <n>` | derived | Memory for K/V pages and recurrent state. By default, room for an 8K-token context per concurrent request |
| `--speculative-draft <n>` | 0 | Most tokens drafted from the request's own text per step; 0 turns speculation off |
| `--mtp [true\|false]` | off | Also draft with the checkpoint's multi-token-prediction head |
| `--draft-model <dir\|repo>` | none | Draft with a separate assistant checkpoint (Gemma 4) |
| `--draft-revision <commit>` | none | Commit of a remote `--draft-model`, or a name for a local one |

Speculation needs F32 recurrent state. With `--recurrent-state f16`, Tur logs a warning and decodes normally. [Speculative decoding](../architecture/speculation.md) explains when it pays.

## Generation (`run`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `<prompt>...`, `--prompt <text>` | piped input, else a chat | One user turn, rendered with the checkpoint's chat template. Use `--prompt` when the file selects the model |
| `-n`, `--max-tokens <n>` | 1024, or what the context leaves | Most tokens to generate, counting EOS |
| `--temperature`, `--top-k`, `--top-p` | checkpoint defaults | Sampling; `--temperature 0` is greedy |
| `--repetition-penalty <x>` | none | Above 1 discourages any token already in the prompt or output |
| `--presence-penalty <x>` | none | Subtracted once from each generated token's logit |
| `--frequency-penalty <x>` | none | Subtracted once per occurrence of each generated token |
| `--seed <n>` | 299792458 | Seed of the request's random stream |
| `--thinking true\|false` | template default | Passed to the chat template |
| `--reasoning-effort low\|medium\|xhigh` | template default | Passed to the chat template |
| `--preserve-thinking true\|false` | template default | Passed to the chat template |
| `--tools <file>` | none | JSON array of `{name, description, parameters}` function definitions |
| `--audio <file.wav>` | none | A clip placed before the prompt, for models with an audio encoder. Needs a prompt |
| `--image <file>` | none | An image placed before the prompt, for models with a vision encoder: PNG, JPEG, WebP or GIF. Repeat for several. Needs a prompt |
| `--video <file.mp4>` | none | A video placed before the prompt, for models with a vision encoder: H.264 or HEVC in MP4. Repeat for several. Needs a prompt |

With no prompt in a terminal, `run` opens a chat. Each turn sends the whole conversation, so prefix reuse answers follow-ups without recomputing it. `/clear` forgets the conversation, and `/bye` or Ctrl-D leaves. A turn that fails, such as one past the context limit, prints its error and leaves the conversation as it was.

Sampling settings you leave out come from the checkpoint's `generation_config.json`. Thinking settings you leave out keep the template's defaults, and Tur adds nothing to the prompt.

## Serving (`serve`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen <addr>` | `127.0.0.1:8080` | Opened after the model loads |
| `--served-model-name <id>` | the model argument's last part | The model ID clients send and `/v1/models` lists |
| `--max-batch-size <n>` | 8 | Requests running at once |
| `--max-batch-tokens <n>` | 2048 or more | Tokens in one pass; always room for one prompt chunk beside every decode |
| `--max-waiting <n>` | 256 | Queued requests before new ones get HTTP 429 |
| `--request-timeout-secs <n>` | none | Fail requests not finished this long after arrival (HTTP 504) |
| `--prefix-snapshots <n>` | one per running request | Prefix snapshots kept for reuse; 0 turns prefix reuse off |
| `--dashboard [true\|false]` | `true` in a terminal | Show the live dashboard once serving |
| `--shutdown-timeout-secs <n>` | 30 | How long accepted requests may take to finish after SIGINT or SIGTERM. A second signal ends the wait |
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

tur run gemma 'Summarize this recording.' --audio speech.wav --thinking false

tur run qwen 'What does this sign say?' --image sign.jpg

tur run qwen 'What happens in this clip?' --video clip.mp4
```

## Quantization (`quantize`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model <dir or org/name>` | required | Source SafeTensors checkpoint: a directory, or a Hugging Face repository at `--revision` |
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
| `--report <file>` | none | Write calibration, sensitivity and plan details as JSON |
| `--json` | off | Print those details on stdout |

[Prepacked weights](quantization.md) walks through the workflow. `tur prune` takes `--json` the same way, and its `--model` also accepts a repository.

## Logging

Logs go to stderr through `tracing`. `RUST_LOG` sets the filter, for example `RUST_LOG=tur_engine=debug`.

## Decisions

- Common flags stay available as overrides beside the file, and `--mtp false` can undo a file's `true`. Rarely changed scheduler and stream limits live only in the file to keep the CLI manageable.
- Progress is `tracing` events from a `Stage` in `tur-core`, not callbacks: no signature changes, a library user pays for a filtered event, and any front end can subscribe to the `tur_progress` target.
- Monitoring reads the public `/diagnostics` over HTTP, in `serve` too, so there is one source of truth and it works remotely. Bars use `indicatif` and the dashboard `ratatui`; the client stays on `std` because it only makes plain GETs.
- `tur stop` signals the process named by `/diagnostics` rather than calling a shutdown route. The API port has no authentication and allows any origin, so a route would let any web page stop the server. The signal limits stopping to the machine and the user running the server, and `stop` checks that the process is a `tur` binary, because a forwarded port can report a PID that means something else locally.
- The banner animates for bare `tur`, `serve`, `quantize` and `prune`. `run`, `list`, `pull`, `transcribe`, `status` and `top` start without it, because people run them often or in scripts.
- The model and prompt are positional because they are what every invocation changes. `TUR_MODEL` and `TUR_REVISION` are gone: they existed to shorten a command that names now shorten, and a positional argument read from the environment would make `tur run hello` ambiguous between a model and a prompt. `tur.toml` still selects a model for a directory.
- `tur run` answers up to 1024 tokens, the server's default, fitted to the context. A limit of 128 ends models that think by default before their answer begins.
- Per-run inputs stay explicit: prompts, tool files, CPU reference mode and quantization jobs are always flags, and logging stays in `RUST_LOG`.
