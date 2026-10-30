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

Reading ahead overlaps the SSD with the GPU. When a layer routes, the next layer's router also scores the same hidden state, and the experts it ranks highest are read in the background while the GPU works. Those reads only fill staging buffers and never touch a slot, so they can't race a kernel. Four in five turn out to be used.

The embedding table stays on the SSD as well. It is over a gigabyte, and a token only needs its own row.

## Protecting the SSD

SSD wear comes from writes, and the cache makes none. Checkpoint files open read-only, evicted slots are clean and simply overwritten, and reads bypass the page cache so that the arena really bounds what experts occupy. The way a cache can still cost writes is indirect: taking memory other applications hold pushes them into swap. So a load refuses a cache larger than what macOS can spare without swapping, minus a reserve of 2.5 GiB for state, scratch and everything else.

## Choosing a size

The cache must hold every expert of one layer, which is 423 MiB on gpt-oss-20b, because one pass can route to all of them. Beyond that, more memory means fewer reads. Generating 160 tokens from a 91-token prompt on an M1 with 16 GB:

| Cache | Hit rate | Tokens/s |
| --- | ---: | ---: |
| 1 GiB | 34% | 2.9 |
| 2 GiB | 52% | 3.3 |

These are single-request numbers: concurrent requests touch more experts. Larger caches raise the hit rate further (67% at 3 GiB and 79% at 4 GiB), though on a busy 16 GB Mac the spare-memory check may refuse them.

Long prompts behave differently. Each pass reads at most the whole expert set however many tokens it holds, so fewer, larger passes read far less. With a cache, prefill chunks and prefix snapshots default to every 1,024 tokens, and a 2,037-token prompt takes 18 s instead of 32 s at the usual 256. A repeat of that prompt with a different ending takes 10 s, because prefix reuse skips its first 1,024 tokens.

## Pruning experts

`tur prune` writes a smaller checkpoint that keeps the same number of experts in every layer, copying their bytes unchanged and dropping the others' router rows. It keeps the experts that calibration text routes to most. That ranking is not good enough yet. Keeping 16 of 32 experts saves 4.7 GiB, but on held-out text the top token matches the full model only 78% of the time on code and documentation and 55% on prose, with KL divergences of 0.21 and 1.41 nats, and greedy generation falls into repetition. Treat it as an experiment.

## Decisions

Measured on gpt-oss-20b on an M1 with 16 GB:

- Correctness comes from Transformers. A tiny MXFP4 fixture and the real checkpoint match its layers and logits within 1e-4 relative error, the real one through a 1 GiB cache with F32 projections.
- Up to 4 GiB of cache, system-wide SSD writes during a run were usually under 100 MiB when nothing swapped (386 MiB at most), which is background activity. A 6 GiB cache swapped out 2.4 GB in one run, which is why the load checks spare memory. Under the 2.5 GiB reserve, 1 and 2 GiB swapped nothing; before it, 3 and 4 GiB swapped up to 830 MiB.
- A miss cost more in our own pipeline than expected: for 160 tokens, 33 s of reads at about 4 GB/s, 24 s of staging uploads and 7 s of copies. Uploading inside the read threads plus reading ahead cut 83 s to 55 s at 1 GiB, with the same hit rate and logits.
- Eviction stays as it is. Replaying a real trace reproduced the live hit rates, and LFU and LRU-2 gained at most two points. Only foreknowledge does better (55% at 81 slots, 89% at 324), which is why reading ahead pays off.
- The MXFP4 kernels stay on a lookup table. It is as fast as Q8_0 at the bank's shape, and an arithmetic decode is 1.3 to 2.1× slower. A GPU round trip costs about 0.2 ms, so the wait at each of 24 layers isn't worth removing.
- Below one token's experts (96 on gpt-oss-20b), eviction takes the expert needed furthest ahead, since passes visit layers in order. At or above it, eviction is LRU, which keeps what routing repeats.
- The arena sits outside Candle's buffer pool, which rounds sizes up to a power of two.
- Pruning by routing mass isn't recommended, for the reasons above. Published results for this kind of pruning (RCO) search on KL divergence directly, which Tur doesn't do.
