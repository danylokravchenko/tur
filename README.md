# Tur - a strong inference engine

> ![tur-mascot](./assets/mascot.png)

A high-performance Rust inference engine for transformer models, built on [Candle](https://github.com/huggingface/candle). Features optimized attention kernels, prefix caching, continuous batching with paged KV memory, and flexible quantization support.

## Features

- **Fast Inference**: Optimized CPU flash attention and GPU flash attention kernels; BF16 on Metal/CUDA, F32 on CPU
- **Prefix Caching**: Automatic KV cache reuse for common prompt prefixes, with per-request paged cache support in batch mode
- **Continuous Batching**: Process multiple concurrent requests together via batched prefill and decode; up to `max_batch_size` requests in flight at once
- **Chunked Prefill**: Split long prompts into fixed-size chunks processed across scheduler iterations, bounding peak attention-matrix memory and letting decode requests interleave with in-progress prefills
- **Paged KV Cache**: Block-based memory allocator (`BlockAllocator`) isolates KV state per request, enabling true multi-request concurrency without interference
- **Scheduling Policies**: FCFS (default), Priority, and Shortest-Job-First to control request ordering and reduce average latency
- **Quantization**: `Q4_K_M`, `Q5_K_M`, `Q8_0`, and other GGUF quantization formats
- **Guided Generation**: Grammar-constrained decoding via [llguidance](https://github.com/guidance-ai/llguidance); JSON schema, Lark, and regex grammars with ~50 µs per-token overhead
- **Thinking Mode**: Enable chain-of-thought reasoning
- **Detailed Statistics**: Per-request prefill/decode timing, cache hit rate, and tokens-per-second reporting
- **Tooling Support**: Define your tools which a model can call

## Supported Models

| Family | HuggingFace ID examples | Quantization | Thinking | Tools |
| ------ | ----------------------- | ------------ | -------- | ----- |
| **Qwen3** | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | SafeTensors (BF16) or GGUF (`Q4_K_M`, `Q8_0`, …) | ✅ `--thinking` flag injects `/think` tag | ✅ `<tool_call>` blocks |
| **Granite 4.1** | `ibm-granite/granite-4.1-3b` | SafeTensors (BF16) or GGUF (`Q4_K_M`, `Q8_0`, …) | ⚠️ inherent model behaviour; `--thinking` flag ignored | ✅ `<tool_call>` blocks via chat template |

Auto-detection reads `model_type` from `config.json` (`"qwen3"`, `"granite"`), so passing the HuggingFace repo ID is enough — no extra flags required.

## Quick Start

### Basic Usage

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200
```

### With Thinking Mode

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200 --thinking
```

### With Quantization

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200 \
  --quantization Q4_K_M \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20
```

## Prefix Caching

Prefix caching skips redundant computation for requests that share a common prompt prefix (e.g., a system prompt) by restoring previously computed KV states instead of re-running the forward pass over the shared tokens.

### Enabling Prefix Cache

```bash
cargo run -- --model-id 'Qwen3-0.6B' --prefix-cache \
  --cache-max-entries 100 \
  --cache-max-tokens 2048
```

### Performance Impact

For prompts with ~50% prefix overlap:

- **Prefill time**: ~50% reduction
- **Time to first token**: ~50% faster
- **Memory overhead**: ~1–2% for cache metadata

In batch mode the prefix cache integrates with paged KV caches so each request independently restores its cached prefix without interfering with other in-flight requests.

## Continuous Batching

Continuous batching processes multiple concurrent requests in a single forward pass rather than one at a time, significantly increasing throughput on multi-request workloads.

### Enabling Batching

```bash
cargo run -- --model-id 'Qwen3-0.6B' --enable-batching \
  --max-batch-size 16 \
  --max-prefill-batch 8 \
  --max-decode-batch 16 \
  --scheduling-policy fcfs
```

### Scheduling Policies

| Policy | Flag | Behavior |
| ------ | ---- | -------- |
| FCFS | `fcfs` | Admit requests in arrival order (default) |
| Priority | `priority` | Use caller-supplied priority field |
| SJF | `shortest_job_first` | Shorter prompts run first; reduces average latency |

### Paged KV Cache

Each request in a batch receives its own isolated KV cache backed by a shared `BlockAllocator`. Blocks are allocated on demand and freed when the request completes, so long-running requests don't starve shorter ones of memory.

### Batch Performance

Batch processing amortizes the fixed cost of a forward pass across multiple requests:

- **Throughput**: Near-linear scaling with batch size up to hardware memory limits
- **Prefill + prefix cache**: Cached prefix tokens are skipped per-request even inside a batch — requests sharing a system prompt pay the prefill cost only for their unique suffix
- **Decode**: All in-flight requests advance by one token per forward pass

## Chunked Prefill

Without chunked prefill, a request with a long prompt monopolises the GPU for the entire duration of its prefill forward pass. Every decode-phase request in the same batch stalls and waits, increasing their time-to-next-token proportionally to the long prompt's length.

Chunked prefill breaks the prompt into slices of at most `chunk_size` tokens. Each scheduler iteration processes one chunk, then immediately runs the pending decode batch. Decode latency is bounded by `chunk_size` rather than the full prompt length, and peak attention-matrix memory per iteration drops from O(prompt²) to O(chunk²).

### Enabling Chunked Prefill

Chunked prefill is a batching-mode feature — paged KV caches are required to hold intermediate KV state between chunks.

```bash
--prefill-chunk-size=512 # process at most 512 prompt tokens per scheduler iteration
```

Setting `prefill-chunk-size` larger than any prompt in the workload degrades gracefully to single-shot prefill with no behavioural change.

### Choosing a Chunk Size

| Chunk size | Effect |
| ---------- | ------ |
| Small (≤ 64) | Very low per-iteration prefill cost; more scheduler iterations needed per prompt; best for latency-sensitive workloads with many concurrent decode requests |
| Medium (128–512) | Balanced trade-off; recommended starting point |
| Large (≥ 1024) | Similar to single-shot; useful mainly to cap memory spikes on extremely long prompts |

### How it interacts with other features

- **Prefix cache**: The first chunk (`kv_start_pos = 0`) checks the prefix cache normally. Subsequent chunks skip the cache lookup and write new KV entries at their respective offsets.
- **Scheduling policy**: All three policies (FCFS, Priority, SJF) work unchanged; chunked prefill only affects how many tokens of each selected request are processed per iteration.
- **Token budget**: `max_prefill_tokens` is enforced against the chunk size, not the full prompt length, so oversized prompts no longer bypass the token budget guard.

## Guided Generation

Guided generation constrains the output to tokens that are valid under a grammar, guaranteeing syntactically correct structured output without relying on prompt engineering.

### When to use it

| Use case | Grammar type |
| -------- | ------------ |
| JSON output parsed by the caller | JSON schema |
| Enum / classification | Regex or Lark |
| Tool calls / agent protocols | JSON schema |
| Domain-specific syntax (SQL, config) | Lark |

Free-text generation, open-ended responses, and any case where the model reliably follows format instructions without enforcement are better served without guidance.

### Usage

Build a `ParserFactory` once (expensive; tied to tokenizer vocabulary) and share it across requests. Requests without a grammar run unconstrained regardless of whether a factory is configured.

## Tool Calling

Tool calling lets the model invoke external functions by emitting structured `<tool_call>` blocks in its output. The pipeline injects the tool schema into the prompt automatically and parses any calls from the generated text, returning them in a response.

### How it works

1. You define tools with a name, description, and JSON Schema parameters.
2. Attach them to a `GenerationRequest` via `.with_tools(…)`.
3. The pipeline formats the raw user message using the model's chat template with tools injected in `<tools>` XML tags (Qwen3 format).
4. After generation, the pipeline scans the output for `<tool_call>…</tool_call>` blocks and deserialises each one into a `ToolCall`.

### API example

```json
{
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "City and country, e.g. Paris, France"
    },
    "unit": {
      "type": "string",
      "enum": ["celsius", "fahrenheit"]
    }
  },
  "required": ["location"]
}
```

### Prompt formatting

When tools are present, the pipeline formats calls internally — you do **not** need to pre-format the prompt yourself. Pass the raw user message as the `prompt` field.

### Combining with thinking mode

Set `--enable-thinking` on the call to enable chain-of-thought reasoning alongside tool calling (supported on Qwen3 via the `/think` tag)

### Combining with guided generation

Tool calling and grammar-constrained generation can be used together. Attach both `.with_tools(…)` and `.with_grammar(…)` to the same request to restrict which tokens the model can emit while still parsing tool calls from the output.

## OpenAI-Compatible Server

Tur ships a web server (`src/bin/server.rs`) that exposes an OpenAI-compatible REST API.  Any client that speaks the OpenAI Chat Completions protocol — `curl`, the `openai` Python SDK, LangChain, LlamaIndex, etc. — works without modification.

### Starting the server

```bash
# Minimal — load a model from HuggingFace and listen on 127.0.0.1:8080
cargo run --bin server -- --model-id Qwen/Qwen3-0.6B

# Quantized model, custom host/port, and larger batch size
cargo run --bin server -- \
  --model-id Qwen/Qwen3-0.6B \
  --quantization Q4_K_M \
  --host 0.0.0.0 \
  --port 8080 \
  --max-batch-size 16
```

### Chat completions — non-streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is the capital of France?"}
    ]
  }'
```

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1748000000,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The capital of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 9, "total_tokens": 9}
}
```

### Chat completions — streaming (SSE)

Add `"stream": true` to receive tokens as Server-Sent Events as they are generated:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  --no-buffer \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true
  }'
```

Each SSE event carries a `chat.completion.chunk` object.  The stream ends with `data: [DONE]`.

### Tool calling

Define tools in your request; the server injects the schema into the prompt and parses `<tool_call>` blocks from the model output:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

When the model decides to call a tool the response looks like:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Extended thinking (Qwen3)

Pass `"thinking": true` to enable chain-of-thought reasoning before the final answer:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Solve: if 2x + 3 = 11, what is x?"}],
    "thinking": true
  }'
```

### Parallel request processing and continuous batching

The server uses **continuous batching**: instead of processing requests one at a time, every incoming request is submitted to a shared `ContinuousBatchScheduler`. Each scheduler iteration:

1. **Admits** any newly queued requests (memory-permitting).
2. **Prefills** a batch of admitted requests in a single GPU forward pass.
3. **Decodes** all in-progress requests together, advancing every sequence by one token per iteration.
4. **Streams** each new token to its originating HTTP response immediately.
5. **Completes** requests that hit EOS or `max_tokens`, freeing their paged KV cache blocks.

This means `N` concurrent clients each get tokens interleaved in real time rather than waiting for the previous client to finish.  Throughput scales near-linearly with batch size up to the VRAM limit.

**Firing parallel requests with curl:**

```bash
# Launch four simultaneous requests in the background
for i in 1 2 3 4; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"Qwen/Qwen3-0.6B\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Tell me a fact number $i\"}]
    }" &
done
wait
```

**Streaming four responses in parallel:**

```bash
for i in 1 2 3 4; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    --no-buffer \
    -d "{
      \"model\": \"Qwen/Qwen3-0.6B\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Fact $i\"}],
      \"stream\": true
    }" > /tmp/stream_$i.txt &
done
wait
grep -h "content" /tmp/stream_*.txt | head -20
```

**Tuning `--max-batch-size`:**

| Workload | Recommendation |
| --- | --- |
| Single user / interactive | `4` — low latency, minimal overhead |
| Multi-user API server | `8`–`16` — default; good throughput/latency balance |
| Batch inference / offline | `32`+ — maximise GPU utilisation; needs more VRAM |

The scheduler's memory-based admission control prevents OOM: requests that would exceed available KV cache memory are queued until earlier requests complete and free their blocks.

## Advanced Configuration

### Sampling Parameters

```bash
--temperature 0.7      # Randomness (0.0 = greedy, 1.0 = creative)
--top-p 0.9            # Nucleus sampling threshold
--top-k 50             # Top-k sampling limit
--repeat-penalty 1.1   # Penalize token repetition
--seed 42              # Random seed for reproducibility
```

### Model Sources

```bash
--model-id Qwen3-0.6B               # Download from HuggingFace
--weight-path /local/path/to/model  # Load from local directory
```

## Performance

Tur achieves competitive performance through:

1. **Optimized Attention**: Custom CPU flash attention and GPU flash attention (Metal/CUDA)
2. **Prefix Caching**: KV state reuse for shared prefixes, integrated with paged memory
3. **Continuous Batching**: Amortized forward-pass cost across concurrent requests
4. **Paged Memory**: Block allocator eliminates memory fragmentation in multi-request workloads
5. **Quantization**: 4-bit and 8-bit GGUF models reduce memory footprint and improve cache utilization

### Benchmarks

```bash
cargo bench
```

Available benchmark groups:

| Group | What it measures |
| ----- | ---------------- |
| `prefill` | Prompt encoding + first forward pass latency |
| `decode` | Steady-state token generation throughput |
| `full_pipeline` | Cold-start model load + prefill + decode |
| `prefix_cache` | Prefill with vs. without prefix cache on repeated prefixes |
| `prefix_cache_lengths` | Cache hit speedup across varying prefix lengths |
| `batch_prefill` | Batched prefill throughput (batch sizes 1/2/4), no-cache vs. with paged prefix cache |
| `chunked_prefill` | Single-shot vs. chunked prefill (chunk sizes 16/32/64) for each prompt length; shows total overhead and per-chunk latency |

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built on [Candle](https://github.com/huggingface/candle) by Hugging Face
- Inspired by [vLLM](https://github.com/vllm-project/vllm)'s prefix caching and paged attention design
