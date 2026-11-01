<div align="center">
  <img src="assets/mascot.png" alt="Tur mascot, a red Cossack with a sweeping moustache, flexing an arm" width="320">
  <h1>Tur</h1>
  <p><strong>Local inference with a little muscle.</strong></p>
  <p>A Rust inference engine for hybrid language models on Apple Silicon.</p>
  <p>Built to get the most out of your Mac.</p>
  <p> <a href="#why-tur">Why Tur</a> · <a href="#how-it-works">How it works</a> · <a href="#performance">Performance</a> · <a href="#models">Models</a> · <a href="#get-started">Get started</a> · <a href="docs/src/SUMMARY.md">Read the book</a> </p>
  <p><strong>Rust · Metal · Apple Silicon · Apache-2.0</strong></p>
</div>

---

Tur runs language models on your Mac's GPU with Metal kernels fitted to those models. One `tur` binary handles generation, an OpenAI-compatible HTTP server, Whisper transcription and quantization, and you can also embed the engine in a Rust program.

## Why Tur

Choose Tur if you want one of these on an Apple Silicon Mac:

- Several clients or agents sharing one model. Requests share each model pass, so eight sequences decode 60% faster than on llama.cpp on the same Mac.
- A mixture-of-experts model larger than your Mac's memory. Tur keeps most experts on the SSD and reads them into a fixed cache, with several requests at once and no SSD writes during a run.
- A server you can leave running. Tur plans all its memory before loading a weight, so a configuration that doesn't fit fails at startup, and one failed request doesn't affect the others.

Tur supports few models and doesn't accept images or video. It has no embeddings, it serves one chat model per process, and its HTTP API doesn't offer structured output yet. For those, or for other hardware, look at oMLX, llama.cpp, Ollama, SGLang or Strata. [Tur and other engines](docs/src/architecture/comparison.md) compares them.

## How it works

Producing a token on a Mac is mostly a memory problem: the GPU spends more time waiting for weights and request state than computing with them. Tur is built around one family of models on this one kind of hardware, so it can shape each layer to read fewer bytes.

- Continuous batching packs every request's next token and chunks of new prompts into one pass with no padding, so the weights are read once for all of them.
- Metal kernels fuse neighbouring steps and read recurrent state in the order the GPU prefers. Tur keeps a custom kernel only where it measurably beats Candle's.
- Attention and recurrent state live in fixed pools that kernels read in place, so nothing is copied or reassembled per token.
- Sampling runs on the GPU right after the model pass, and only the chosen token returns to the CPU.
- Calibrated 4-bit weights load in a fraction of a second and hold fewer bytes to read, which makes a single conversation about 1.6× faster than 8-bit.

Prefix reuse computes a shared system prompt or an agent's growing conversation once. Speculative decoding drafts several tokens and verifies them in one pass, which pays off on repetitive output such as code edits. Neither changes the generated text.

The [architecture chapters](docs/src/architecture/overview.md) explain each mechanism and the measurements behind it.

## Performance

Tur against llama.cpp on an M3 Pro with 36 GB, in tokens per second. 4-bit rows use each engine's usual format, so they compare engines and weights together.

| Test | Tur | llama.cpp | Difference |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B, 8-bit, decode, 1 sequence | 123 | 113 | +9% |
| Qwen3.5-0.8B, 8-bit, decode, 8 sequences | 729 | 456 | +60% |
| Qwen3.5-0.8B, 4-bit, decode, 1 sequence | 182 | 141 | +29% |
| Qwen3.5-0.8B, 8-bit, prefill, 2K tokens | 3,799 | 3,383 | +12% |
| Qwen3.6-35B, 4-bit, decode, 1 sequence | 47.1 | 35.6 | +32% |

Tur doesn't win everywhere: on a 16 GB M1, llama.cpp prefills 8K-token prompts 6 to 11% faster. Prefill is mostly compute, so Tur's savings on memory reads help less there.

Other results for Qwen3.5-0.8B on that 16 GB M1:

- Through the full server with sampling, eight concurrent requests produce 186 tokens/s in total, with 27 ms between tokens.
- Eight users behind a cached 1.5K-token system prompt get their first token in 0.56 s at the median, against 9.5 s without prefix reuse.
- Prompt-lookup speculation reaches up to 2.9× normal decoding on repetitive text.

[Measured performance](docs/src/guide/performance.md) has the full tables, versions and methods.

## Models

Tur supports the Qwen3.5 family, Gemma 4 and gpt-oss for text, plus Whisper for transcription. Gemma 4 E2B and E4B also accept WAV audio. [Scope and principles](docs/src/design/scope.md) lists how far each model has been validated.

## Get started

You need an Apple Silicon Mac. Install the prebuilt binary, which goes into `~/.local/bin`:

```sh
curl -fsSL https://github.com/danylokravchenko/tur/releases/latest/download/install.sh | sh
```

To build from source instead, or to see the installer's options, read [Requirements and checkpoints](docs/src/guide/getting-started.md).

Generate some text. The first run downloads the model into your Hugging Face cache, and pinning the revision means every run loads the same weights:

```sh
export TUR_MODEL=Qwen/Qwen3.5-0.8B
export TUR_REVISION=2fc06364715b967f1860aea9cf38778875588b17

tur run --prompt 'Write one sentence about the Moon.' --temperature 0 -n 64 --thinking false
```

Serve it as an API:

```sh
tur serve --served-model-name qwen
```

Any OpenAI client then works with the base URL `http://127.0.0.1:8080/v1` and the model `qwen`:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"thinking":false}'
```

## Documentation

The [Tur book](https://danylokravchenko.github.io/tur/docs/) covers usage, architecture and the measurements behind each decision, and the [benchmarks page](https://danylokravchenko.github.io/tur/benchmarks/) lets you explore them. Run `just docs` to read the book locally.

| I want to | Start here |
| --- | --- |
| Run my first model, offline use included | [Quickstart](docs/src/guide/quickstart.md) |
| Configure the CLI or server | [Configuration](docs/src/guide/configuration.md), [Command line](docs/src/guide/cli.md) |
| Connect an app | [HTTP API](docs/src/guide/http-api.md), [Rust library](docs/src/guide/library.md) |
| Make weights smaller | [Choosing a quantization](docs/src/guide/choosing-quantization.md), [Prepacked weights](docs/src/guide/quantization.md) |
| Match settings to my workload | [Tuning](docs/src/guide/tuning.md), [Best practice](docs/src/guide/best-practice.md) |
| Understand the engine | [Architecture](docs/src/architecture/overview.md), [Scope](docs/src/design/scope.md) |
| See what's next | [Open work](docs/src/design/open.md), [Rejected experiments](docs/src/design/rejected.md) |

## Contributing

Contributions are welcome. Read the [development workflow](docs/src/development/workflow.md) and [AGENTS.md](AGENTS.md), install [`just`](https://github.com/casey/just), and run `just ci` before you open a pull request. Tests need direct Metal access on an Apple Silicon Mac.

---

<div align="center">
  <p>Built for Apple Silicon. Written in Rust.</p>
  <p>Licensed under <a href="LICENSE">Apache-2.0</a>.</p>
</div>
