# Tur - a strong inference engine

> ![tur-mascot](./assets/mascot.png)

For normal:

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200
```

For thinking add `--thinking` flag.

For quant:

```bash
cargo run -- --model-id 'Qwen3-0.6B' --sample-len 200 --quantization Q4_K_M --temperature 0.6 --top-p 0.95 --top-k 20 
```
