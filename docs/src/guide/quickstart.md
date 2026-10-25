# Quickstart: from zero to a quantized server

This walkthrough installs Tur, downloads a model, chats with it, serves it, and then
quantizes it and serves the smaller artifact. It uses Qwen3.5-0.8B, which runs on any
Apple Silicon Mac.

## 1. Install

```sh
git clone https://github.com/danylokravchenko/tur && cd tur
./install.sh                          # puts `tur` in ~/.cargo/bin by default
tur --help
```

Install Rust through rustup first; the checkout selects Rust 1.99.0 automatically.
Building takes a few minutes the first time. The script also replaces an existing
installation. Use `./install.sh --root "$HOME/.local"` for `~/.local/bin`, and ensure
the destination's `bin` directory is on your `PATH`. `cargo build --release --bin tur`
works too; the binary is then `target/release/tur`.

## 2. Pick and pin a model

Tur needs an exact revision for every model, so the same command always loads the same
weights. Look up the current commit of a Hugging Face repository:

```sh
curl -s https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
```

This book uses `2fc06364715b967f1860aea9cf38778875588b17`. Keep it in your shell:

```sh
export TUR_MODEL=Qwen/Qwen3.5-0.8B
export TUR_REVISION=2fc06364715b967f1860aea9cf38778875588b17
```

`tur` reads both variables, so later commands can omit `--model` and `--revision`.

## 3. Download

You have two options.

**Let Tur download.** Give the repository name; files go to the Hugging Face cache
(`~/.cache/huggingface`, or `HF_HOME`). Gated models need `HF_TOKEN`.

```sh
tur run --prompt 'Hello'      # downloads on first use
```

**Download to a directory.** This is better for offline use, quantization and serving.
It needs the Hugging Face CLI (`pip install -U huggingface_hub`):

```sh
hf download Qwen/Qwen3.5-0.8B --revision "$TUR_REVISION" --local-dir ~/models/qwen3.5-0.8b
export TUR_MODEL=~/models/qwen3.5-0.8b    # a directory: loads locally
export TUR_REVISION=qwen3.5-0.8b@2fc0636  # for local dirs, any stable identity
```

Older `huggingface_hub` versions spell the command `huggingface-cli download`.

## 4. Generate

```sh
# Greedy and short; thinking off for a direct answer.
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

The memory plan and timing are logged to stderr:

```text
INFO tur::run: Memory plan resident_bytes=3017015772 weights=1309441280 state_pool=181469184 scratch_peak=251789312
INFO tur::run: Generation complete prompt_tokens=19 generated_tokens=26 elapsed=741ms finish_reason="stop"
```

## 5. Serve

```sh
tur serve --served-model-name qwen --max-batch-size 8
```

Once it prints `Listening on http://127.0.0.1:8080`, it is ready.

```sh
# One-shot.
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

A prepacked artifact loads faster and can be smaller and more accurate than load-time
conversion. GPTQ needs calibration text that resembles your traffic, roughly 16K tokens
(about 60 KB of English). Mix the kinds of text you serve:

```sh
cat my-docs/*.md my-code/*.rs > calib.txt
tur quantize --model ~/models/qwen3.5-0.8b --revision "$TUR_REVISION" \
  --output ~/models/qwen3.5-0.8b-q4k-gptq --calibration calib.txt \
  --report ~/models/qwen3.5-0.8b-q4k-gptq.json
```

No calibration text at hand? `--method rtn` packs without it. The result is the same as
load-time conversion, but loads in a fraction of the time.

## 7. Serve the artifact

```sh
tur serve --model ~/models/qwen3.5-0.8b-q4k-gptq --revision q4k-gptq-v1 --served-model-name qwen
```

Do not pass `--weight-format`; the artifact loads as stored. The memory plan should show
resident weights of about 425 MB, against 1,309 MB for load-time Q8_0.

## Next

- [Choosing a quantization](choosing-quantization.md): which format for which Mac and
  model.
- [Tuning](tuning.md): recipes for latency, throughput, memory and agents.
- [HTTP API](http-api.md): every request field and error.
