# Expert cache

Routed experts are most of a large MoE model, yet each token uses only a few of them. The expert cache keeps a bounded subset in memory and reads the rest from the SSD when a token routes to them, so a model that doesn't fit in the GPU's working set can still run. gpt-oss-20b holds 9.5 GiB of experts in 12.8 GiB of weights, and it generates on a 16 GB Mac with a 1 GiB cache:

```sh
tur run --model openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --expert-cache-mb 2048 --prompt 'Hello'
```

The cache reads gpt-oss's MXFP4 banks only. Other models reject the flag.

## How it works

The weights outside the routed experts stay resident. Experts live in one fixed arena of slots, and the kernels read them in place. After a layer's router runs, the host reads that layer's routes, loads any missing experts into slots, and rewrites only the expert number in the routes' tiles to a slot number. Biases and accumulation order keep using original expert IDs, so logits are bit-identical to keeping every expert resident.

Reading the routes waits for all earlier GPU work, so no kernel can be using a slot while it is overwritten, and copies complete before the layer's kernels are encoded. A failed read fails that pass and leaves the slot without an owner, and a later pass reads it again.

## Protecting the SSD

SSD wear comes from writes, and the cache makes none. Checkpoint files open read-only, evicted slots are clean and simply overwritten, and reads bypass the page cache so that the arena really bounds what experts occupy. The way a cache can still cost writes is indirect: taking memory other applications hold pushes them into swap. So a load refuses a cache larger than what macOS can spare without swapping, minus a reserve of 3.5 GiB for state, scratch and everything else.

## Choosing a size

The cache must hold every expert of one layer, which is 423 MiB on gpt-oss-20b, because one pass can route to all of them. Beyond that, more memory means fewer reads. Generating 160 tokens from a 91-token prompt on an M1 with 16 GB:

| Cache | Hit rate | Tokens/s | Read per token |
| --- | ---: | ---: | ---: |
| 512 MiB | 16% | 1.6 | 1.0 GiB |
| 1 GiB | 34% | 1.9 | 0.8 GiB |
| 2 GiB | 52% | 2.1 | 0.6 GiB |
| 3 GiB | 67% | 2.7 | 0.4 GiB |
| 4 GiB | 79% | 3.7 | 0.3 GiB |

Waiting for reads takes two thirds to four fifths of the time at every size, so a faster SSD or better hit rate is what speeds this up. These are single-request numbers: concurrent requests and prefill touch more experts.

## Pruning experts

`tur prune` writes a smaller checkpoint that keeps the same number of experts in every layer, copying their bytes unchanged and dropping the others' router rows. It keeps the experts that calibration text routes to most. That ranking is not good enough yet. Keeping 16 of 32 experts saves 4.7 GiB, but on held-out text the top token matches the full model only 78% of the time on code and documentation and 55% on prose, with KL divergences of 0.21 and 1.41 nats, and greedy generation falls into repetition. Treat it as an experiment.

## Decisions

Measured on gpt-oss-20b on an M1 with 16 GB:

- Correctness comes from Transformers. A tiny MXFP4 fixture and the real checkpoint match its layers and logits within 1e-4 relative error, the real one through a 1 GiB cache with F32 projections.
- Up to 4 GiB of cache, system-wide SSD writes during a run were usually under 100 MiB when nothing swapped (386 MiB at most), which is background activity. A 6 GiB cache swapped out 2.4 GB in one run, which is why the load checks spare memory. 2 GiB with about 6 GiB to spare swapped nothing in repeated runs; 3 and 4 GiB swapped up to 830 MiB under the same conditions.
- Below one token's experts (96 on gpt-oss-20b), eviction takes the expert needed furthest ahead, since passes visit layers in order. At or above it, eviction is LRU, which keeps what routing repeats.
- The arena sits outside Candle's buffer pool, which rounds sizes up to a power of two.
- Pruning by routing mass isn't recommended, for the reasons above. Published results for this kind of pruning (RCO) search on KL divergence directly, which Tur doesn't do.
