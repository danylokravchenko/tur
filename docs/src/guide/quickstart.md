# Quickstart

This walkthrough installs Tur, chats with a model and serves it. Then it quantizes the model and serves the smaller version. It uses Qwen3.5-0.8B, which runs on any Apple Silicon Mac.

## 1. Install

```sh
curl -fsSL https://github.com/danylokravchenko/tur/releases/latest/download/install.sh | sh
tur --help
```

The script installs the prebuilt binary into `~/.local/bin`, so you don't need Rust. If that directory isn't on your `PATH`, the script adds it to `~/.zshrc` or `~/.bash_profile`; open a new terminal if `tur` isn't found. Running the script again replaces an existing installation. [Requirements and checkpoints](getting-started.md#download) lists its settings and explains how to build from source.

## 2. Chat

```sh
tur run qwen
```

The first run downloads the model into the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`), and later runs load it from there. Then you get a prompt. `/clear` forgets the conversation, and `/bye` or Ctrl-D leaves.

`tur list` shows the other models and which ones you already have, and `tur pull gemma` downloads one without running it. Each name is pinned to an exact commit, so it always loads the same weights. Gated models need `HF_TOKEN`.

## 3. Ask for one answer

Give a prompt, or pipe one in, and Tur prints the answer and exits:

```sh
# Thinking off for a direct answer.
tur run qwen 'Write one sentence about the Moon.' --thinking false

# Sampled with the checkpoint's own defaults, thinking on (the template's default).
tur run qwen 'Explain why the sky is blue.'

# Piped input is the prompt.
cat notes.txt | tur run qwen

# Tools: the model may answer with a function call.
cat > tools.json <<'EOF'
[{"name": "get_weather", "description": "Current weather for a city",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
EOF
tur run qwen 'What is the weather in Kyiv?' --tools tools.json --thinking false
```

The answer goes to stdout. Reasoning, loading progress and a timing summary go to stderr:

```text
  ✓ Loading weights  24/24 layers in 1.6 s
  26 tokens at 35.1/s · first token 117 ms · 741 ms total · stop
```

Set `RUST_LOG=info` to also see the memory plan. See [Command line](cli.md) for how every command reports its work.

## 4. Serve

```sh
tur serve qwen
```

When it prints `Listening on http://127.0.0.1:8080`, the model is loaded and ready. Clients call it by the name you gave it, `qwen` here.

```sh
# One response.
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"thinking":false}'

# Streaming.
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","stream":true,"messages":[{"role":"user","content":"Count to five"}]}'
```

Any OpenAI client works. With Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Name three moons of Jupiter."}],
    max_tokens=64,
    extra_body={"thinking": False},
)
print(reply.choices[0].message.content)
```

## 5. Run a model that has no name

`tur list` holds only models Tur has generated from. Anything else that Tur supports loads from a Hugging Face repository or a directory:

```sh
tur run Qwen/Qwen3.5-4B
tur run ./models/my-checkpoint
```

A repository loads at its latest commit, and Tur prints that commit with the `--revision` flag that keeps it. Add the flag when a later run must load the same weights. For offline use, download into a directory with the Hugging Face CLI (`pip install -U huggingface_hub`) and run that directory:

```sh
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir ~/models/qwen3.5-0.8b
tur run ~/models/qwen3.5-0.8b
```

Older versions of `huggingface_hub` call the command `huggingface-cli download`.

## 6. Quantize

`tur quantize` writes a prepacked copy of the model. It loads almost instantly, and with calibration its 4-bit weights lose about half as much quality as plain rounding on most text. Calibration needs text that resembles your traffic, roughly 16K tokens (about 60 KB of English), so mix the kinds of text you serve:

```sh
cat my-docs/*.md my-code/*.rs > calib.txt
tur quantize --model ~/models/qwen3.5-0.8b --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output ~/models/qwen3.5-0.8b-q4k-gptq --calibration calib.txt \
  --report ~/models/qwen3.5-0.8b-q4k-gptq.json
```

Without calibration text, `--method rtn` uses plain rounding. You still get the fast load.

## 7. Serve the quantized model

```sh
tur serve ~/models/qwen3.5-0.8b-q4k-gptq --served-model-name qwen
```

Leave out `--weight-format`, because the artifact loads exactly as stored. The memory plan should now show about 425 MB of weights, a third of the default 8-bit load.

## Next

- [Choosing a quantization](choosing-quantization.md) helps you pick a format for your Mac and model.
- [Tuning](tuning.md) has recipes for latency, throughput, memory and agents.
- [Best practice](best-practice.md) has a starting configuration for each model.
- [HTTP API](http-api.md) documents every request field and error.
