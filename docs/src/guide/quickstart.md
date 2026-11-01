# Quickstart

This walkthrough installs Tur, downloads a model, chats with it and serves it. Then it quantizes the model and serves the smaller version. It uses Qwen3.5-0.8B, which runs on any Apple Silicon Mac.

## 1. Install

```sh
curl -fsSL https://github.com/danylokravchenko/tur/releases/latest/download/install.sh | sh
tur --help
```

The script installs the prebuilt binary into `~/.local/bin`, so you don't need Rust. If that directory isn't on your `PATH`, the script adds it to `~/.zshrc` or `~/.bash_profile`; open a new terminal if `tur` isn't found. Running the script again replaces an existing installation. [Requirements and checkpoints](getting-started.md#download) lists its settings and explains how to build from source.

## 2. Pick and pin a model

Tur asks for an exact revision of every model, so the same command always loads the same weights. To look up the current commit of a Hugging Face repository:

```sh
curl -s https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
```

This book uses `2fc06364715b967f1860aea9cf38778875588b17`. Keep it in your shell:

```sh
export TUR_MODEL=Qwen/Qwen3.5-0.8B
export TUR_REVISION=2fc06364715b967f1860aea9cf38778875588b17
```

`tur` reads both variables, so later commands can leave out `--model` and `--revision`.

## 3. Download

The simplest option is to let Tur download. Give it the repository name, and the files go to the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`). Gated models need `HF_TOKEN`.

```sh
tur run --prompt 'Hello'      # downloads on first use
```

For offline use and for quantization, download into a directory instead. This needs the Hugging Face CLI (`pip install -U huggingface_hub`):

```sh
hf download Qwen/Qwen3.5-0.8B --revision "$TUR_REVISION" --local-dir ~/models/qwen3.5-0.8b
export TUR_MODEL=~/models/qwen3.5-0.8b    # a directory loads locally
export TUR_REVISION=qwen3.5-0.8b@2fc0636  # for a local directory, any stable name you choose
```

Older versions of `huggingface_hub` call the command `huggingface-cli download`.

## 4. Generate

```sh
# Greedy and short, with thinking off for a direct answer.
tur run --prompt 'Write one sentence about the Moon.' --temperature 0 -n 64 --thinking false

# Sampled with the checkpoint's own defaults, thinking on (the template's default).
tur run --prompt 'Explain why the sky is blue.' -n 512

# Tools: the model may answer with a function call.
cat > tools.json <<'EOF'
[{"name": "get_weather", "description": "Current weather for a city",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
EOF
tur run --prompt 'What is the weather in Kyiv?' --tools tools.json --thinking false
```

Text goes to stdout. Loading progress and a timing summary go to stderr:

```text
  ✓ Loading weights  24/24 layers in 1.6 s
  26 tokens at 35.1/s · first token 117 ms · 741 ms total · stop
```

Set `RUST_LOG=info` to also see the memory plan. See [Command line](cli.md) for how every command reports its work.

## 5. Serve

```sh
tur serve --served-model-name qwen --max-batch-size 8
```

When it prints `Listening on http://127.0.0.1:8080`, the model is loaded and ready.

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

## 6. Quantize

`tur quantize` writes a prepacked copy of the model. It loads almost instantly, and with calibration its 4-bit weights lose about half as much quality as plain rounding on most text. Calibration needs text that resembles your traffic, roughly 16K tokens (about 60 KB of English), so mix the kinds of text you serve:

```sh
cat my-docs/*.md my-code/*.rs > calib.txt
tur quantize --model ~/models/qwen3.5-0.8b --revision "$TUR_REVISION" \
  --output ~/models/qwen3.5-0.8b-q4k-gptq --calibration calib.txt \
  --report ~/models/qwen3.5-0.8b-q4k-gptq.json
```

Without calibration text, `--method rtn` uses plain rounding. You still get the fast load.

## 7. Serve the quantized model

```sh
tur serve --model ~/models/qwen3.5-0.8b-q4k-gptq --revision q4k-gptq-v1 --served-model-name qwen
```

Leave out `--weight-format`, because the artifact loads exactly as stored. The memory plan should now show about 425 MB of weights, a third of the default 8-bit load.

## Next

- [Choosing a quantization](choosing-quantization.md) helps you pick a format for your Mac and model.
- [Tuning](tuning.md) has recipes for latency, throughput, memory and agents.
- [Best practice](best-practice.md) has a starting configuration for each model.
- [HTTP API](http-api.md) documents every request field and error.
