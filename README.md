# Tur - a strong inference engine

> ![tur-mascot](./assets/mascot.png)

A high-performance Rust inference engine for transformer models, built on [Candle](https://github.com/huggingface/candle). Features optimized attention kernels, prefix caching, continuous batching with paged KV memory, and flexible quantization support.

## Features

- **Fast Inference**: Optimized CPU flash attention and GPU flash attention kernels; BF16 on Metal/CUDA, F32 on CPU
- **Prefix Caching**: Automatic KV cache reuse for common prompt prefixes, with per-request paged cache support in batch mode
- **Continuous Batching**: Process multiple concurrent requests together via batched prefill and decode; up to `max_batch_size` requests in flight at once
- **Paged KV Cache**: Block-based memory allocator (`BlockAllocator`) isolates KV state per request, enabling true multi-request concurrency without interference
- **Scheduling Policies**: FCFS (default), Priority, and Shortest-Job-First to control request ordering and reduce average latency
- **Quantization**: `Q4_K_M`, `Q5_K_M`, `Q8_0`, and other GGUF quantization formats
- **Thinking Mode**: Enable chain-of-thought reasoning
- **Detailed Statistics**: Per-request prefill/decode timing, cache hit rate, and tokens-per-second reporting

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

### Documentation

```bash
cargo doc --open
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built on [Candle](https://github.com/huggingface/candle) by Hugging Face
- Inspired by [vLLM](https://github.com/vllm-project/vllm)'s prefix caching and paged attention design
