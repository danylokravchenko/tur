# Expert cache

Routed experts are most of a large MoE model, yet each token uses only a few of them. The expert cache keeps a bounded subset in memory and reads the rest from the SSD when a token routes to them, so a model that doesn't fit in the GPU's working set can still run. gpt-oss-20b holds 9.5 GiB of experts in 12.8 GiB of weights, and it generates on a 16 GB Mac with a 1 GiB cache:

```sh
tur run --model openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --expert-cache-mb 2048 --prompt 'Hello'
```

The cache also serves Qwen3.5-family MoE checkpoints, from a prepacked artifact: the banks have to be on disk exactly as the kernels read them, so write one with `tur quantize` first. Its banks can be Q8_0 or Q4K, with one format per projection across layers. A native MTP head reads its own expert bank through the same cache. Dense models and Gemma 4 reject the flag, and so do Qwen checkpoints converted at load time.

## How it works

The weights outside the routed experts stay resident. Experts live in one fixed arena of slots, and the kernels read them in place. After a layer's router runs, the host reads that layer's routes, loads any missing experts into slots, and rewrites only the expert number in the routes' tiles to a slot number. Biases and accumulation order keep using original expert IDs, so logits are bit-identical to keeping every expert resident.

Reading the routes waits for all earlier GPU work, so no kernel can be using a slot while it is overwritten, and copies complete before the layer's kernels are encoded. A failed read fails that pass and leaves the slot without an owner, and a later pass reads it again.

Reading ahead overlaps the SSD with the GPU. When a layer routes, the next layer's router also scores the same hidden state, and the experts it ranks highest are read in the background while the GPU works and while the routed layer's own reads and copies finish. Those reads only fill staging buffers and never touch a slot, so they can't race a kernel. Four in five turn out to be used.

The embedding table stays on the SSD as well. It is over a gigabyte, and a token only needs its own row.

## Protecting the SSD

SSD wear comes from writes, and the cache makes none. Checkpoint files open read-only, evicted slots are clean and simply overwritten, and reads bypass the page cache so that the arena really bounds what experts occupy. The way a cache can still cost writes is indirect: taking memory other applications hold pushes them into swap. So a load refuses a cache larger than what macOS can spare without swapping, minus a reserve of 2.5 GiB for state, scratch and everything else.

## Choosing a size

The cache must hold every expert of one layer, which is 423 MiB on gpt-oss-20b, because one pass can route to all of them. Beyond that, more memory means fewer reads. Generating 160 tokens from a 91-token prompt on an M1 with 16 GB:

| Cache | Hit rate | Tokens/s |
| --- | ---: | ---: |
| 1 GiB | 34% | 3.1 |
| 2 GiB | 52% | 3.3 |

The 2 GiB row predates reading ahead during a layer's own reads and copies. These are single-request numbers: concurrent requests touch more experts. Larger caches raise the hit rate further (67% at 3 GiB and 79% at 4 GiB), though on a busy 16 GB Mac the spare-memory check may refuse them.

Long prompts behave differently. Each pass reads at most the whole expert set however many tokens it holds, so fewer, larger passes read far less. With a cache, prefill chunks and prefix snapshots default to every 1,024 tokens, and a 2,037-token prompt takes 18 s instead of 32 s at the usual 256. A repeat of that prompt with a different ending takes 10 s, because prefix reuse skips its first 1,024 tokens.

## Serving several requests

A cache shares its slots between requests, so more of them mean more misses per token. Total throughput still rises slightly while each request slows down. Through `tur serve` with a 1 GiB cache on gpt-oss-20b:

| Concurrent requests | Total tokens/s | Each request, tokens/s | Median gap between tokens |
| ---: | ---: | ---: | ---: |
| 1 | 2.1 | 2.6 | 0.32 s |
| 2 | 2.5 | 1.4 | 0.64 s |
| 4 | 2.7 | 0.7 | 1.1 s |

A long prompt can arrive while another request is streaming. In that case the stream kept going at 1.6 tokens/s, with a 1.1 s 95th-percentile gap, while the 2,000-token prompt reached its first token in 21 s.

A cancelled request needs no special handling, because reads and copies finish inside the pass that started them and never outlive it. Disconnecting a client during decode, during prefill, or three at once left the server running, and the next request's answer was identical to one from an untouched server.

## Pruning experts

Pruning is a separate step from the cache. The cache never changes what the model computes; `tur prune` writes a smaller checkpoint that permanently lacks some experts, so tokens that would use them go to their next choice. It keeps the same number of experts in every layer and copies their bytes unchanged. Nothing prunes unless you run this command, and the source checkpoint stays untouched.

To choose, it runs every expert on calibration text and removes, one at a time, whichever moves the layer's output least, including where tokens reroute. `--search-evaluations` then tries swaps and keeps those that lower end-to-end KL. `--plan-only` writes the choice without a checkpoint, and the `experts` benchmark can judge it by hiding those experts from the routers.

None of this is good enough yet, so treat pruning as an experiment. Keeping 16 of 32 experts saves 4.7 GiB, but on held-out code and documentation the top token matches the full model about 80% of the time, and on prose under 60%. The limit is generalization rather than search: experts removed because calibration text rarely needs them are what other text needs.

## Decisions

Measured on gpt-oss-20b on an M1 with 16 GB:

- Correctness comes from Transformers. A tiny MXFP4 fixture and the real checkpoint match its layers and logits within 1e-4 relative error, the real one through a 1 GiB cache with F32 projections.
- Up to 4 GiB of cache, system-wide SSD writes during a run were usually under 100 MiB when nothing swapped (386 MiB at most), which is background activity. A 6 GiB cache swapped out 2.4 GB in one run, which is why the load checks spare memory. Under the 2.5 GiB reserve, 1 and 2 GiB swapped nothing; before it, 3 and 4 GiB swapped up to 830 MiB.
- A miss cost more in our own pipeline than expected: for 160 tokens, 33 s of reads at about 4 GB/s, 24 s of staging uploads and 7 s of copies. Uploading inside the read threads plus reading ahead cut 83 s to 55 s at 1 GiB, with the same hit rate and logits.
- Decode at small caches is limited by the bytes it reads: 1 GiB reads about 1 GB per token, and four threads read random expert-sized blocks at about 5.7 GB/s before uploads. Starting the read-ahead before a layer's own reads and copies, rather than after, cut 160 tokens from 56 s to 52 s with the same cache statistics.
- Eviction stays as it is. Replaying a real trace reproduced the live hit rates, and LFU and LRU-2 gained at most two points. Only foreknowledge does better (55% at 81 slots, 89% at 324), which is why reading ahead pays off.
- The MXFP4 kernels stay on a lookup table. It is as fast as Q8_0 at the bank's shape, and an arithmetic decode is 1.3 to 2.1× slower. A GPU round trip costs about 0.2 ms, so the wait at each of 24 layers isn't worth removing.
- A Qwen artifact through the cache gives bit-identical logits to keeping its banks resident, in Q8_0 and Q4K under eviction. This is checked on a small fixture. The real Qwen3.6-35B-A3B Q4K artifact has also run through a 4 GiB cache, serving 16 requests at once where keeping every expert resident runs out of GPU memory on a 36 GB Mac (see [Measured performance](../guide/performance.md#qwen36-35b-a3b)); its logits haven't been compared with the resident run.
- Serving four concurrent requests on a 5.9 GiB plan swapped out 251 MiB over a run of several minutes, where single requests swapped nothing. The spare-memory check sees the machine at load time only, so keep `--max-batch-size` small on a tight machine.
- Below one token's experts (96 on gpt-oss-20b), eviction takes the expert needed furthest ahead, since passes visit layers in order. At or above it, eviction is LRU, which keeps what routing repeats.
- The arena sits outside Candle's buffer pool, which rounds sizes up to a power of two.
- Pruning isn't recommended. Keeping 16 of 32 experts, held-out KL on code and documentation, then prose: routing mass 0.21 and 1.41 nats; output-movement search on 512 tokens 0.26 and 1.45; on 4,096 mixed tokens 0.18 and 1.11; plus 48 KL-tested swaps 0.19 and 1.14, even though those lowered calibration KL by 28%. These rest on 256 held-out tokens per text, and no task benchmark has run. Keeping 24 of 32 on 512 tokens gave 0.05 and 0.71.
