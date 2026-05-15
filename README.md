# Tur - a strong inference engine

> ![tur-mascot](./assets/mascot.png)

A high-performance Rust inference engine for transformer models, built on [Candle](https://github.com/huggingface/candle). Features optimized attention kernels, prefix caching, and flexible quantization support.

## Features

- 🚀 **Fast Inference**: Optimized CPU and GPU attention kernels
- 💾 **Prefix Caching**: Automatic KV cache reuse for common prompt prefixes
- 🔢 **Quantization**: Support for `Q4_K_M` and other GGUF quantization formats
- 🧠 **Thinking Mode**: Enable chain-of-thought reasoning
- 📊 **Detailed Statistics**: Track prefill/decode performance and cache efficiency

## Quick Start

### Basic Usage

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200
```

### With Thinking Mode

Enable chain-of-thought reasoning:

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200 --thinking
```

### With Quantization

Use quantized models for reduced memory usage:

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200 \
  --quantization Q4_K_M \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20
```

## Prefix Caching

Prefix caching dramatically improves performance for requests with common prompt prefixes by reusing previously computed KV cache states.

### Benefits

- **Faster Response Times**: Skip redundant computation for cached prefixes
- **Higher Throughput**: Process more requests with the same hardware
- **Reduced Latency**: Especially beneficial for multi-turn conversations and batch processing

### Performance Impact

For prompts with 50% prefix overlap:

- **Prefill Time**: ~50% reduction
- **Time to First Token**: ~50% faster
- **Memory**: Minimal overhead (~1-2% for cache metadata)

## Advanced Configuration

### Sampling Parameters

```bash
--temperature 0.7    # Randomness (0.0 = deterministic, 1.0 = creative)
--top-p 0.9         # Nucleus sampling threshold
--top-k 50          # Top-k sampling limit
--repeat-penalty 1.1 # Penalize token repetition
--seed 42           # Random seed for reproducibility
```

## Performance

Tur achieves competitive performance through:

1. **Optimized Attention**: Custom CPU flash attention kernels and GPU flash attention
2. **Prefix Caching**: Automatic KV cache reuse for common prefixes
3. **Efficient Memory**: Minimal allocations and Arc-based tensor sharing
4. **Quantization**: Support for 4-bit and 8-bit quantized models

### Benchmarks

Run benchmarks to measure performance on your hardware:

```bash
cargo bench
```

### Documentation

Generate and view documentation:

```bash
cargo doc --open
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built on [Candle](https://github.com/huggingface/candle) by Hugging Face
- Inspired by [vLLM](https://github.com/vllm-project/vllm)'s prefix caching design
