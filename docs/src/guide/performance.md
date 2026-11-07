# Measured performance

This page collects Tur's current numbers. Each architecture chapter explains the mechanism behind them, and [`just bench`](../development/benchmarks.md) reproduces them; re-measure on your own Mac before relying on a figure.

Each section names its model and checkpoint. Qwen3.5-0.8B, Gemma 4 E2B, gpt-oss-20b and Whisper ran on an Apple M1 with 16 GB. Qwen3.6-35B-A3B and Qwen3.8-27B ran on an Apple M3 Pro with 36 GB, which the first of them describes.

Every timed region ends when the GPU finishes, and serving rows are the best of three runs.

## Qwen3.5-0.8B

These sections use an Apple M1 with 16 GB, on AC power with low-power mode off, and Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`, except the last subsection, which ran on an M3 Pro. K/V is F16 in 16-token pages, recurrent state is F32 and prefill chunks are 256 tokens. The comparison subsections set Tur against llama.cpp and MLX on this model.

### Single model pass

Raw model speed, without scheduling or HTTP (`model` benchmark, synthetic tokens).

Decode throughput in tokens/s across all sequences, at a 128-token context:

| Weights | 1 seq | 2 seq | 4 seq | 8 seq | 16 seq |
| --- | --- | --- | --- | --- | --- |
| Q8_0, converted at load | 58 | 111 | 186 | 317 | 367 |
| Q8_0 artifact | 56 | 110 | 184 | 317 | 366 |
| Q4K, converted at load | 92 | 114 | 204 | 347 | 365 |
| Q4K GPTQ artifact | 96 | 114 | 206 | 349 | 365 |
| Mixed 5.5-bit GPTQ artifact | 79 | 111 | 197 | 339 | 356 |

With one sequence, decode is limited by how fast the GPU reads weights, so Q4K is about 1.6× faster than Q8_0. From two sequences on, the weights are shared across the batch, each sequence's recurrent state becomes the main cost, and the formats converge.

Decode at longer contexts, in tokens/s:

| Weights / K/V | 1 seq at 2K | 4 seq at 2K | 1 seq at 8K | 4 seq at 8K |
| --- | --- | --- | --- | --- |
| Q8_0 / F16 | 57 | 168 | 50 | 138 |
| Q8_0 / Q8 | 56 | 171 | 52 | 144 |
| Q4K / F16 | 87 | 184 | 75 | 149 |
| Q4K / Q8 | 88 | 187 | 78 | 156 |

Prefill, in prompt tokens/s for one sequence:

| Weights | 2K tokens | 8K tokens |
| --- | --- | --- |
| Q8_0 | 1,335 | 1,129 |
| Q4K | 1,211 | 1,075 |

A 2K-token prompt takes about 1.5 s, and an 8K-token prompt about 7.3 s.

### Serving

Bursts through the full scheduled pipeline: N requests arrive at once, each with a 128-token prompt and 64 output tokens, using Q8_0 weights and sampling at temperature 0.7, top-k 20 and top-p 0.95. Output tokens/s is the total across the burst, time to first token (TTFT) counts from submission, and inter-token latency (ITL) is the gap between tokens of one request.

| Concurrent | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50.2 | 0.15 s | 0.15 s | 17.4 ms | 17.9 ms | 18.5 ms |
| 2 | 88.9 | 0.24 s | 0.24 s | 18.3 ms | 19.1 ms | 24.2 ms |
| 4 | 122 | 0.45 s | 0.48 s | 22.0 ms | 23.0 ms | 44.8 ms |
| 8 | 187 | 0.89 s | 0.94 s | 25.9 ms | 27.0 ms | 42.5 ms |
| 16 | 207 | 1.68 s | 1.74 s | 44.8 ms | 46.2 ms | 182 ms |

Greedy decoding is at most 7% faster, because sampling runs on the GPU.

With mixed prompt lengths, sixteen requests at once where half have 128-token prompts and half 2K-token prompts, the burst produced 55 output tokens/s with a median TTFT of 8.0 s and a median ITL of 219 ms. The eight long prompts add up to 16K prefill tokens, about 13 s of GPU work, which the scheduler interleaves with the short requests' replies.

### Requests arriving over time

48 requests with 512-token prompts and 64 output tokens arrive at random (a Poisson process), sampled as above.

| Arrival rate | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p99 | GPU idle |
| --- | --- | --- | --- | --- | --- | --- |
| 2/s | 93.1 | 3.33 s | 6.60 s | 55 ms | 258 ms | 4.8% |
| 4/s | 105.5 | 6.64 s | 14.5 s | 53 ms | 249 ms | 5.2% |

Both rates ask for more prefill than one M1 can do, so the queue grows and TTFT climbs over the run. The server accepts new requests while the GPU is busy, which keeps it idle only about 5% of the time.

### Grammar-constrained output

A JSON-like grammar masks every token; greedy decoding.

| Concurrent | Masks on CPU: tok/s, ITL p50 | Masks on GPU (default): tok/s, ITL p50 |
| --- | --- | --- |
| 1 | 42.3, 20.1 ms | 47.6, 17.4 ms |
| 4 | 106.0, 30.4 ms | 129.5, 22.0 ms |
| 8 | 158.1, 37.7 ms | 197.8, 25.9 ms |
| 16 | 176.5, 64.2 ms | 221.0, 45.4 ms |

With masks applied on the GPU, constrained output runs as fast as unconstrained output.

### Prefix reuse

The `reuse` workloads, three runs each:

| Workload | Result |
| --- | --- |
| 8 users with the same new 1.5K-token prompt at once | TTFT p50 1.69 s, 91.6 output tok/s |
| 8 users behind an already cached 1.5K-token prompt | TTFT p50 0.56 s |
| The same 8 users with prefix reuse off | TTFT p50 9.46 s |
| 4 agents × 5 steps over a 1K-token manual | 83% of prompt tokens cached, 9.9 s total |
| 16 distinct 512-token prompts (nothing to reuse) | TTFT p50 3.83 s, 57.1 tok/s |

Multi-turn chat: four conversations of six turns each, with a 256-token first message, 64-token follow-ups and up to 48 output tokens per turn (`TUR_BENCH_WORKLOADS=prefix`):

| Snapshots kept | Prompt tokens cached | Last-turn TTFT p50 | Six-turn total |
| --- | --- | --- | --- |
| 8 | 44.5% | 1.46 s | 13.2 s |
| 16 | 68.3% | 0.79 s | 10.9 s |

### Speculative decoding

Tur verifies drafts into one recurrent slot per sequence and replays the accepted prefix after a rejection, reducing recurrent-state writes. These runs use Q8_0 weights, 80 output tokens per request and the best of three repeats after warmup, in total output tokens/s. Each prompt runs in a separate benchmark invocation with a fresh pipeline for each configuration, so adaptive draft estimates carry over only from that prompt's warmup and earlier repeats.

Greedy:

| Prompt | Seqs | Normal | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 54.8 | 58.1 | 71.9 |
|  | 4 | 161.8 | 163.6 | 189.6 |
|  | 8 | 259.3 | 257.4 | 249.0 |
| Code | 1 | 54.1 | 54.5 | 71.2 |
|  | 4 | 160.6 | 150.0 | 188.5 |
|  | 8 | 255.1 | 241.9 | 237.1 |
| Repetitive | 1 | 52.8 | 161.8 | 154.9 |
|  | 4 | 148.3 | 294.2 | 287.5 |
|  | 8 | 227.6 | 318.7 | 306.6 |

Sampled at temperature 0.7, top-k 20, top-p 0.95:

| Prompt | Seqs | Normal | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 54.5 | 55.2 | 62.8 |
|  | 4 | 160.6 | 153.6 | 152.3 |
|  | 8 | 258.3 | 256.6 | 248.7 |
| Code | 1 | 54.3 | 60.7 | 70.6 |
|  | 4 | 159.6 | 157.4 | 188.9 |
|  | 8 | 254.7 | 250.7 | 232.1 |
| Repetitive | 1 | 52.3 | 158.1 | 156.6 |
|  | 4 | 137.2 | 161.4 | 181.7 |
|  | 8 | 211.8 | 235.4 | 224.2 |

Prompt lookup reaches 3.1× normal decoding on repetitive text with one request. The MTP head adds up to 1.3× on prose and code with one to four requests, but gives no gain on sampled prose at four. While more than four requests decode, the head stops proposing drafts; maintaining its attention state still costs time.

The planner retains measured costs and calibrated head acceptance across requests. Running prose and code before repetition on one pipeline reduced greedy repetitive Lookup + MTP from 153 to 81 tokens/s at one request and from 278 to 204 at four: single-token head drafts displaced longer lookup chains. At eight requests, lookup largely stopped drafting and Lookup + MTP fell from 301 to 213 tokens/s. The isolated tables above avoid this history effect, which remains relevant to long-lived servers.

The widest spread between the three repeats was 5% for greedy and 17% for sampled output, including changes in adaptive draft selection. In a separate paired comparison with 64 output tokens, replay improved repetitive throughput by 6 to 8% at one, four and eight requests and reduced planned resident memory by 74 to 297 MiB after charging replay records; code throughput ranged from 3% slower to 2% faster (see [replay measurements](../research/hybrid-state-from-vllm.md#what-tur-built-replay)).

Every run, greedy and sampled, reproduced normal decoding's token IDs.

### Long contexts with sparse attention

[Sparse attention](sparse-attention.md) (`--sparse-attention auto`) with the map `tur sparse-map` fitted on four books, Q8_0 weights, F16 K/V. Past 16K keys each token reads about 4,600 chosen keys, in prefill and in decode; below 16K both modes read every key.

Decode of one sequence at a long context (`model` benchmark, `TUR_BENCH_SPARSE_MAP`), in tokens/s, with the time per step. The llama.cpp column is the same Q8_0 checkpoint as a GGUF (`llama-bench -p 2048 -n 32 -d 32768,65536,131072 -r 2`, llama.cpp 0.7.0 with its defaults, flash attention chosen automatically and F16 K/V), as in [Compared with llama.cpp](#compared-with-llamacpp):

| Context | Dense | Sparse attention | llama.cpp | Step, dense | Step, sparse | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 32K | 38.9 | 46.0 | 36.8 | 25.7 ms | 21.8 ms | 1.18× |
| 64K | 30.7 | 48.0 | 29.2 | 32.6 ms | 20.9 ms | 1.56× |
| 128K | 21.7 | 47.5 | 20.2 | 46.2 ms | 21.1 ms | 2.19× |

Prefill of one 2048-token chunk after a long context (`TUR_BENCH_ONLY=chunk`), in prompt tokens/s, with the time per chunk; llama.cpp's is `pp2048` at the same depth:

| Context before the chunk | Dense | Sparse attention | llama.cpp | Chunk, dense | Chunk, sparse | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 32K | 481 | 825 | 471 | 4.26 s | 2.48 s | 1.7× |
| 64K | 292 | 733 | 279 | 7.01 s | 2.79 s | 2.5× |
| 128K | 165 | 679 | 159 | 12.37 s | 3.02 s | 4.1× |

Dense, Tur decodes 5 to 7% faster than llama.cpp at every depth and prefills 2 to 5% faster after 32K to 128K of context, where attention over the context dominates; its prefill attention kernel keeps two threadgroups on each GPU core (see [Metal kernels](../architecture/kernels.md#decisions)). With sparse attention Tur decodes 1.25 to 2.35 times as fast as llama.cpp and prefills 1.75 to 4.3 times as fast, but its outputs differ slightly from dense attention while llama.cpp's don't. The serving rows above ran before the current prefill attention kernel, which prefills 39 to 61% faster after 32K to 64K of context, so their dense first-token times are now shorter.

With 128 tokens of context the same build decodes 57.5 tokens/s and prefills a 2K prompt at 1,311 tokens/s, so sparse attention keeps decode at 83% of its short-context speed out to 128K, where dense falls to 38%.

Bursts of long prompts through the full server (`TUR_BENCH_WORKLOADS=long`), cold prefix cache, greedy decoding, 256 output tokens per request, one run each. Output tokens/s counts the whole burst, prefill included:

| Prompt, requests | TTFT p50, dense | TTFT p50, sparse | ITL p50, dense | ITL p50, sparse | Output tok/s, dense | Output tok/s, sparse |
| --- | --- | --- | --- | --- | --- | --- |
| 32K, 1 | 61.3 s | 39.4 s | 25.3 ms | 21.2 ms | 3.8 | 5.7 |
| 32K, 4 | 229 s | 152 s | 44.0 ms | 29.1 ms | 3.4 | 5.0 |
| 64K, 1 | 194 s | 79.7 s | 32.4 ms | 21.8 ms | 1.3 | 3.0 |
| 64K, 4 | 737 s | 308 s | 78.2 ms | 31.9 ms | 1.4 | 2.9 |

The first token arrives 1.6 to 2.4 times sooner and the gap between tokens shrinks up to 2.5 times, most with four 64K requests, whose K/V reads dominate dense decode. Prompts still prefill below 16K keys densely, so the gain grows with prompt length. On the needle check (eight codes hidden in a held-out book, asked one at a time) sparse attention answered 23 of 24 correctly at 16K, 32K and 64K, against 22 for dense attention.

### Loading

A fresh process with a warm file cache, timed until the last weight reaches the GPU. Peak memory comes from `/usr/bin/time -l`.

| Weights | Load time | Weight memory | Peak RSS |
| --- | --- | --- | --- |
| Q4K, converted at load | 8.9 s | 933 MB | 1.46 GB |
| Q8_0, converted at load | 1.58 s | 1,309 MB | 2.11 GB |
| Q4K GPTQ artifact | 0.16 s | 425 MB | 0.71 GB |
| Mixed 5.5-bit GPTQ artifact | 0.20 s | 521 MB | 0.80 GB |
| Q8_0 artifact | 0.27 s | 801 MB | 1.20 GB |

### Quality of weight formats

Teacher-forced over about 8K held-out tokens per domain, against F32 weights. KL is the mean divergence in nats per token (lower is better), and top-1 is how often the format's most likely token matches F32's.

| Weights | Prose KL / top-1 | Code KL / top-1 | Retrieval KL / top-1 | Tools KL / top-1 |
| --- | --- | --- | --- | --- |
| Q8_0 | 0.0007 / 98.4% | 0.0004 / 99.2% | 0.0003 / 98.3% | 0.0014 / 99.4% |
| Mixed 5.5-bit GPTQ | 0.0346 / 88.6% | 0.0182 / 94.2% | 0.0144 / 90.9% | 0.0663 / 96.0% |
| Q4K GPTQ | 0.0704 / 84.4% | 0.0343 / 92.1% | 0.0296 / 89.6% | 0.1003 / 94.5% |
| Q4K importance-weighted | 0.0941 / 83.3% | 0.0510 / 90.8% | 0.0368 / 88.9% | 0.0986 / 94.7% |
| Q4K plain rounding | 0.1413 / 79.7% | 0.0832 / 88.4% | 0.0577 / 87.0% | 0.1142 / 93.9% |

For K/V, F16 is indistinguishable from F32, and Q8 stays within 1e-4 nats of F32 with 99.3 to 99.8% top-1 agreement.

### Compared with llama.cpp

llama.cpp 0.7.0 (build 11669, Homebrew) on the same M1, with unsloth's Qwen3.5-0.8B GGUF files (revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`) and its defaults: Metal and F16 K/V. Its figures are the best of three runs. Q8_0 is the like-for-like format. GGUF's Q4_K_M mixes Q4_K and Q6_K blocks (497 MiB), while Tur's Q4K uses Q4_K for every projection.

Single sequence, in tokens/s (llama.cpp from `llama-bench`):

| Test | Tur Q8_0 | llama.cpp Q8_0 | Tur Q4K | llama.cpp Q4_K_M |
| --- | --- | --- | --- | --- |
| Decode at 128 tokens | 58 | 51.4 | 92 | 65.6 |
| Decode at 2K | 57 | 49.4 | 87 | 63.5 |
| Decode at 8K | 50 | 46.2 | 75 | 58.9 |
| Prefill 2K | 1,335 | 1,323 | 1,211 | 1,188 |
| Prefill 8K | 1,129 | 1,142 | 1,075 | 1,037 |

Batched decode with Q8_0, total tokens/s at about 128 tokens of context (Tur from the `model` benchmark, llama.cpp from `llama-batched-bench`):

| Sequences | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Tur | 58 | 111 | 186 | 317 | 367 |
| llama.cpp | 51.6 | 85.6 | 138 | 201 | 230 |

Bursts of N requests with 128-token prompts and 64 output tokens, in output tokens/s. Tur's figures go through the full pipeline with sampling. llama.cpp's are `llama-batched-bench` wall time for the same prefill and decode, with no scheduler and no sampling.

| Concurrent | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Tur Q8_0, sampled | 50.2 | 88.9 | 122 | 187 | 207 |
| llama.cpp Q8_0 | 47.6 | 75.5 | 114 | 155 | 170 |
| llama.cpp Q4_K_M | 59.4 | 78.7 | 114 | 151 | 160 |

Tur decodes 1.1 to 1.4× faster with one sequence and up to 1.6× faster with 8 to 16, and finishes bursts of 8 to 16 requests 1.2× faster end to end. Prefill is level at 2K and 8K tokens, within 1 to 4% either way, while after 32K to 128K of context Tur's dense prefill is 2 to 5% faster (see [Long contexts with sparse attention](#long-contexts-with-sparse-attention)).

#### On the same GGUF files

Both engines loading the same file, on the same M1 with F16 K/V: `Q3_K_M` (Q3_K, Q4_K, Q5_K, Q6_K and Q8_0) and `UD-Q2_K_XL` (which adds Q2_K and IQ2, IQ3 and IQ4 blocks). Both files hold identical weights in each engine, so these rows compare engines alone. llama.cpp 0.7.0 is the mean of three repeats, with standard deviations of at most 3.4 tokens/s; Tur's decode is the median of timed steps and its prefill the median of 2 to 4 passes.

| GGUF test | Tur Q3_K_M | llama.cpp Q3_K_M | Tur UD-Q2_K_XL | llama.cpp UD-Q2_K_XL |
| --- | --- | --- | --- | --- |
| Decode, 1 sequence, 128 tokens | 70 | 65.7 | 70 | 69.0 |
| Decode, 1 sequence, 2K | 68 | 62.5 | 68 | 66.1 |
| Decode, 4 sequences | 168 | 135 | 175 | 134 |
| Decode, 8 sequences | 293 | 200 | 306 | 204 |
| Prefill 512 | 1,246 | 1,237 | 1,307 | 1,263 |
| Prefill 2K | 1,199 | 1,208 | 1,251 | 1,233 |

With several sequences Tur decodes 1.2 to 1.3× faster at 4 and about 1.5× at 8 on both files. With one, it is 7 to 9% ahead on `Q3_K_M` and 1 to 3% ahead on `UD-Q2_K_XL`. Prefill is within 1% on `Q3_K_M` and 1 to 3% ahead on `UD-Q2_K_XL`; llama.cpp's only lead is `Q3_K_M` at 2K tokens. Tur's figures are the mean of two runs. Both files load in about a second, against 9 s for converting SafeTensors to Q4K.

```sh
llama-bench -m <gguf> -p 2048,8192 -n 0 -r 3
llama-bench -m <gguf> -p 0 -n 64 -d 128,2048,8192 -r 3
llama-batched-bench -m <gguf> -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16
```

### Compared with MLX and oMLX

These are published community results for Qwen3.5-0.8B on an M1 (8-core GPU) with 16 GB, not reruns on this Mac: mlx_lm from [whatcani.run](https://www.whatcani.run/?device=Apple+M1:8:Apple+M1:8:16) with 4,096 prompt and 1,024 output tokens, and oMLX 0.7.3 from [omlx.ai](https://omlx.ai/benchmarks/performance/yw52053o) at temperature 0.7. Tur has no 4K-context row, so its 2K and 8K figures bracket theirs. MLX's OptiQ-4bit (620 MB) is a mixed 4-bit format; Tur's Q4K GPTQ artifact is 425 MB.

Single sequence, in tokens/s:

| Engine, weights | Prefill | Decode | Peak memory |
| --- | --- | --- | --- |
| Tur Q8_0 | 1,335 at 2K, 1,129 at 8K | 57 at 2K, 50 at 8K | 1.20 GB RSS (artifact) |
| mlx_lm 8-bit | 814 at 4K | 46.1 at 4K | 2.46 GB |
| Tur Q4K GPTQ | 1,211 at 2K, 1,075 at 8K | 87 at 2K, 75 at 8K | 0.71 GB RSS |
| mlx_lm OptiQ-4bit | 946 at 4K | 56.0 at 4K | 3.50 GB |
| oMLX OptiQ-4bit | 1,025 at 1K, 990 at 4K | 73.7 at 1K, 68.6 at 4K | 2.17 GB footprint |

Batched 4-bit decode, total tokens/s (Tur at 128 tokens of context; oMLX doesn't state its context):

| Sequences | 1 | 2 | 4 |
| --- | --- | --- | --- |
| Tur Q4K GPTQ | 96 | 114 | 206 |
| oMLX OptiQ-4bit | 73.7 | 123 | 198 |

With one sequence, Tur decodes 1.1 to 1.3× faster than oMLX and 1.1 to 1.6× faster than mlx_lm, with half or less of their reported peak memory. Its 4-bit prefill is level with oMLX's and ahead of mlx_lm's. With two to four sequences the lead disappears: oMLX scales better from one sequence to two, because Tur's batched decode is limited by recurrent-state traffic rather than weights.

### On an M3 Pro

The same checkpoint on the Apple M3 Pro with 36 GB that the larger Qwen sections use, against llama.cpp 0.8.0 (build 11960, Homebrew) with the same unsloth GGUF files as above and its defaults: Metal and F16 K/V. Tur's weights are converted at load from the checkpoint. llama.cpp's figures are the mean of three `llama-bench` repeats, with a standard deviation of at most 1.4 tokens/s on decode and 27 on prefill, except the batched row, which is one `llama-batched-bench` run. Tur's decode is the median of 12 timed steps and its prefill the median of 2 to 4 passes. GGUF's Q4_K_M mixes Q4_K and Q6_K blocks while Tur's Q4K uses Q4_K for every projection, so the 4-bit columns compare engines with their usual weights.

Single sequence, in tokens/s:

| Test | Tur Q8_0 | llama.cpp Q8_0 | Tur Q4K | llama.cpp Q4_K_M |
| --- | --- | --- | --- | --- |
| Decode at 128 tokens | 123.4 | 113.0 | 181.6 | 140.9 |
| Decode at 2K | 119.6 | 109.8 | 172.6 | 136.3 |
| Prefill 512 | 3,853 | 3,446 | 3,635 | 3,277 |
| Prefill 2K | 3,799 | 3,383 | 3,611 | 3,208 |

Batched decode with Q8_0, total tokens/s at about 128 tokens of context (Tur from the `model` benchmark, llama.cpp from `llama-batched-bench -c 8192 -ub 512` with 128-token prompts and 64 generated tokens):

| Sequences | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| Tur | 123 | 240 | 420 | 729 |
| llama.cpp | 113 | 172 | 320 | 456 |

Tur prefills 11 to 13% faster and decodes one Q8_0 sequence 9% faster. With more sequences the lead grows to 31 to 60%, and Tur's 4-bit decode is 27 to 29% ahead of Q4_K_M. Tur's 4-bit prefill is 5 to 6% slower than its Q8_0 prefill.

#### Long contexts, tiled attention and MTP

The tiled dense attention kernels, sparse attention and the MTP head, on the same M3 Pro. Q8_0 is converted at load from the checkpoint; Q4K is a plain-rounding artifact (`tur quantize --method rtn --weight-format q4k`, 414 MiB). K/V is F16. Each sparse run uses an index map fitted on the weights it serves, but from 131K tokens of this repository's documentation and Rust source rather than books, and no retrieval check was run on these maps, so these rows show speed only. Decode is the median of 12 timed steps and prefill the median of two passes.

The tiled prefill kernel against the per-query one (`kernels` benchmark, F16 K/V, Qwen3.5 attention shapes):

| Prefill shape | Tiled | Per-query | Speedup |
| --- | --- | --- | --- |
| 2048 tokens | 4.8 ms | 14.9 ms | 3.1× |
| 8192 tokens | 73 ms | 623 ms | 8.5× |
| 2048-token chunk after 32K | 165 ms | 2,775 ms | 16.9× |

Below 128 tokens the per-query kernel is faster (0.4 to 0.6× at 16 to 64 tokens), and the two meet at 128. For one decode token, selecting keys and attending to them takes 0.25 ms at 32K against 0.81 ms for dense attention, 0.26 ms against 1.65 ms at 64K, and 0.29 ms against 4.1 ms at 128K.

Prefill of a 2048-token chunk after a long context, in prompt tokens/s:

| Context before the chunk | Q8_0 dense | Q8_0 sparse | Q4K dense | Q4K sparse |
| --- | --- | --- | --- | --- |
| 4K | 3,154 | 3,148 | 3,043 | 2,984 |
| 32K | 1,379 | 2,885 | 1,324 | 2,769 |
| 64K | 800 | 2,632 | 805 | 2,569 |

Decode in tokens/s, with the time per step:

| Sequences, context | Q8_0 dense | Q8_0 sparse | Q4K dense | Q4K sparse |
| --- | --- | --- | --- | --- |
| 1 at 4K | 117 (8.6 ms) | 111 (9.0 ms) | 168 (6.0 ms) | 164 (6.1 ms) |
| 1 at 32K | 80 (12.5 ms) | 100 (10.0 ms) | 98 (10.2 ms) | 135 (7.4 ms) |
| 4 at 32K | 119 (33.6 ms) | 294 (13.6 ms) | 132 (30.2 ms) | 347 (11.5 ms) |
| 1 at 64K | 56 (17.9 ms) | 97 (10.3 ms) | 69 (14.5 ms) | 134 (7.5 ms) |
| 4 at 64K | 63 (63.1 ms) | 297 (13.5 ms) | 65 (61.8 ms) | 335 (11.9 ms) |

Below the 16K crossover sparse attention reads every key and the two columns agree within run-to-run noise. Past it, prefill stays within 17% of its 4K speed out to 64K, and four sequences at 64K take 1.3× the step time of one, where dense takes 3.5×.

Short prompts through the full server (`serving` benchmark, 64 output tokens, greedy), in output tokens/s:

| Concurrent requests | 1 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| Q8_0 | 109 | 302 | 436 | 546 |
| Q4K | 150 | 317 | 442 | 490 |

Four drafts from the MTP head against normal decoding (`TUR_BENCH_WORKLOADS=speculation`, 128 output tokens, greedy, best of three repeats), in output tokens/s:

| Prompt, requests | Q8_0 normal | Q8_0 MTP | Q4K normal | Q4K MTP |
| --- | --- | --- | --- | --- |
| Repetitive, 1 | 115 | 430 | 156 | 512 |
| Repetitive, 4 | 364 | 753 | 424 | 758 |
| Novel prose, 1 | 117 | 161 | 161 | 202 |
| Novel prose, 4 | 381 | 476 | 454 | 510 |
| Code, 1 | 117 | 167 | 159 | 188 |
| Code, 4 | 382 | 473 | 447 | 510 |

The head gains 3.3 to 3.8× on repetitive text with one request and 1.2 to 1.4× on prose and code. With four requests the gain falls to 1.8 to 2.1× on repetitive text and 1.1 to 1.25× on the rest. The same prompts with sparse attention enabled never reach the 16K crossover, so they measure its fixed cost. Q4K matched or beat the runs above (169 against 156 tokens/s for one normal request, 203 against 202 with MTP on novel prose). Q8_0 was about 7% slower with one request (106 to 109 against 115 to 117 tokens/s) and 5 to 8% slower with four, which these runs don't explain.

One long prompt through the server (`TUR_BENCH_WORKLOADS=long TUR_BENCH_MTP=0,1`, 128 output tokens, cold prefix cache, one run each). The prompt is a synthetic list of repeated words, so the MTP rows show cost at depth, not the head's usual acceptance:

| Prompt, weights | Mode | TTFT p50 | ITL p50 (MTP: gap between bursts) | Tokens per pass |
| --- | --- | --- | --- | --- |
| 32K, Q8_0 | dense | 16.5 s | 12.8 ms | 1.00 |
|  | dense + MTP | 18.0 s | 20.0 ms | 1.94 |
|  | sparse | 12.5 s | 10.1 ms | 1.00 |
|  | sparse + MTP | 14.1 s | 12.9 ms | 1.83 |
| 64K, Q8_0 | dense | 48.4 s | 18.1 ms | 1.00 |
|  | dense + MTP | 53.9 s | 19.6 ms | 1.28 |
|  | sparse | 25.2 s | 10.3 ms | 1.00 |
|  | sparse + MTP | 30.8 s | 12.6 ms | 1.12 |
| 32K, Q4K | dense | 16.7 s | 9.7 ms | 1.00 |
|  | dense + MTP | 18.3 s | 16.9 ms | 1.54 |
|  | sparse | 13.1 s | 7.5 ms | 1.00 |
|  | sparse + MTP | 14.5 s | 8.5 ms | 1.08 |
| 64K, Q4K | dense | 48.8 s | 14.6 ms | 1.00 |
|  | dense + MTP | 54.5 s | 16.5 ms | 1.39 |
|  | sparse | 26.1 s | 7.7 ms | 1.00 |
|  | sparse + MTP | 31.8 s | 9.5 ms | 1.08 |

Sparse attention halves the 64K first-token time and cuts the gap between tokens by 40 to 50%. MTP does not pay at these depths on this prompt: the first token arrives 10 to 20% later and the gap between bursts is longer than plain decoding's gap between tokens in every row. Accepted drafts vary, from 1.08 tokens per pass on sparse Q4K to 1.94 on dense Q8_0 at 32K, and these runs don't isolate why.

## Gemma 4 E2B

`google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, Q8_0 weights and F16 K/V, measured with `just bench-gemma`. Ranges span the measurement rounds. E2B does about three times the work per token of the 0.8B Qwen, and its 262K-token output layer keeps single-sequence decode near 20 tokens/s.

| Case | Result |
| --- | --- |
| Prefill, 512 tokens | 1192 tokens/s |
| Prefill, 2048 tokens | 1012 tokens/s |
| Decode, 1 sequence at 128 tokens | 19.8 to 21.3 tokens/s |
| Decode, 4 sequences at 128 tokens | 51.8 tokens/s |
| Decode, 1 sequence at 8K tokens | 18.9 to 20.0 tokens/s |
| Decode, 4 sequences at 8K tokens | 44.6 tokens/s |
| State memory, 1 sequence at 8K tokens | 56.8 MB |
| State memory, 4 sequences at 8K tokens | 227 MB |

Speculative decoding, 96 greedy tokens, total output tokens/s, with the assistant drafting one token per pass:

| Prompt | Seqs | Normal | Lookup (4 drafts) | Lookup + assistant |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 20.0 to 20.3 | 19.7 to 20.0 | 28.5 to 28.9 |
| | 4 | 48.6 to 48.8 | 45.6 to 46.7 | 58.9 to 61.1 |
| Code | 1 | 19.9 to 20.1 | 20.1 to 20.5 | 23.3 to 26.1 |
| | 4 | 47.7 to 48.1 | 46.5 to 47.3 | 64.1 to 65.4 |
| Repetitive | 1 | 19.4 to 19.6 | 64.3 to 65.0 | 58.7 to 61.2 |
| | 4 | 43.7 to 45.2 | 99.4 to 110.3 | 111.4 to 113.2 |

With one sequence, the assistant now chains several drafts per pass, which measured another 1.14× on prose and 1.5× on repetitive text; see [Speculative decoding](../architecture/speculation.md#decisions). Every run reproduced normal decoding's tokens.

| Weights | Load time | Weight memory | Peak RSS |
| --- | --- | --- | --- |
| Q8_0, converted at load | 11.1 to 12.3 s | 7.93 GB | 2.2 to 2.7 GB |
| Q8_0 artifact (`tur quantize --method rtn`) | 4.8 to 5.5 s | 7.12 GB | 3.2 GB |

The artifact reads embeddings from its packed output layer, which saves the 805 MB embedding table. Of either total, 4.4 GiB is Gemma's per-layer embedding table.

### Sparse attention on E2B

`--sparse-attention auto` with a map fitted on four books, `model` benchmark, one sequence. The 16 GB M1 ran out of memory for a dense 2048-token chunk after 64K, so that row is missing.

| Case | Dense | Sparse attention |
| --- | --- | --- |
| Decode at 32K | 17.5 tokens/s (57.3 ms) | 18.4 tokens/s (54.3 ms) |
| Decode at 64K | 15.2 tokens/s (66.0 ms) | 19.5 tokens/s (51.3 ms) |
| Prefill, 2048-token chunk after 32K | 188 tokens/s (10.9 s) | 806 tokens/s (2.54 s) |

E2B spends 47 ms of each decode step on weights, so the context's share is small and so is the decode gain; prefill gains 4.3×. Both modes answered all 16 needle questions at 16K and 32K with identical text.

## gpt-oss-20b

This model holds 12.8 GiB of weights, more than a 16 GB M1's GPU working set, so it runs through the [expert cache](../architecture/expert-cache.md) with most experts on the SSD. One request, Q8_0 attention and output projections, MXFP4 experts as published:

| Cache | Decode, tokens/s | Hit rate |
| --- | ---: | ---: |
| 1 GiB | 3.1 | 34% |
| 2 GiB | 3.3 | 52% |

A 2,037-token prompt takes 18 s with a 1 GiB cache, about 115 tokens/s. Repeating it with a different ending takes 10 s. These numbers come from `tur run` on a Mac with an editor and browser open, and concurrent requests are not yet measured.

## Whisper large-v3-turbo

`openai/whisper-large-v3-turbo` at revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, on clips spoken by macOS `say`, measured with `just bench whisper`. Language is detected from the first window unless a row says otherwise, and every time excludes loading the checkpoint.

The encoder always reads a full 30-second window, so a short clip costs almost as much as a long window: most of a transcription's time is one encoder pass per window. A window with no audible frame skips that pass, so silence costs almost nothing.

| Case | Q8_0 | F32 |
| --- | --- | --- |
| Log-mel features, one window | 0.12 s | 0.12 s |
| Encoder, one window | 2.01 s | 2.17 s |
| Decoder step at 30 tokens | 4.6 ms, about 215 tokens/s | 14.7 ms |
| Decoder step at 270 tokens | 5.5 to 6.1 ms | 14.2 to 14.8 ms |
| Transcribing a 2.66 s clip | 2.31 s | 2.59 s |
| Transcribing an 80 s clip (3 windows, 287 tokens) | 8.5 s, 9.4× real time | 11.9 s, 6.7× real time |
| Transcribing a 45 s silent clip | 0.12 s | not measured |
| Loading the checkpoint | about 2.9 s | not measured |
| Weight memory | about 0.85 GB | 3.2 GB |

One encoder layer spends about 13 ms on attention and about 40 ms on projections, which run on Candle's GEMM at 1.2 to 1.6 TFLOP/s. A decoder step is mostly the output head and cross-attention, both limited by how fast the GPU can read memory, so Tur reads fewer bytes there: a Q4K head and F16 cross-attention keys and values. Word timestamps added about 1.7 s to the 80 s clip through the CLI.

## Images and video

Qwen3.5-0.8B with Q8_0 weights, encoding on the tower's own queue, while a 400-token greedy request decodes (`just bench vision` with `TUR_QWEN35_0_8B`):

| Case | Result |
| --- | --- |
| Encoding 256×256, 64 soft tokens | 0.05 s |
| Encoding 512×512, 256 soft tokens | 0.24 s |
| Encoding 1024×1024, 1024 soft tokens | 1.56 s |
| Encoding 3000×3000, scaled to 1024 soft tokens | 1.59 s |
| Encoding 1024×1024 beside a decoding request | 2.23 s |
| Encoding a 5 s 320×320 clip, 500 soft tokens | 0.47 s |
| Encoding a 5 s 640×640 clip, 2000 soft tokens | 2.35 s |
| Encoding a 30 s 640×640 clip, 1920 soft tokens | 1.95 s |
| Decoding 1080p H.264, one thread | 98 frames/s |
| Decoding 1080p HEVC, one thread | 88 frames/s (78 at 10-bit) |
| Repeating an encoded input, rows reused | under 10 µs |
| Text pass, alone: median, slowest | 17.6 ms, 86 ms |
| Text pass beside that image: median, slowest | 18.4 ms, 80 ms |

Gemma 4 E2B encodes a 640×480 image, 266 soft tokens from about 2400 patches, in 1.64 s.

## Gemma 4 audio

E2B with Q8_0 weights encoding clips on their own queue, while a 400-token greedy request decodes (`just bench whisper` with `TUR_GEMMA4_E2B`):

| Case | Result |
| --- | --- |
| Encoding a 5 s clip, alone | 0.21 s |
| Encoding a 30 s clip, alone | 1.45 s |
| Encoding a 30 s clip beside a decoding request | 2.70 s |
| Text pass, alone: median, slowest | 47 ms, 109 ms |
| Text pass beside that clip: median, slowest | 48 ms, 165 ms |

## Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is a mixture-of-experts model with 35 billion parameters, of which 3 billion are active for each token. Every layer routes a token to 8 of its 256 experts, so decode reads only a small part of the expert weights per step, while the full bank has to be resident or read from the SSD. The rows below use a Q4K artifact made with `tur quantize --method rtn` from the pinned `Qwen/Qwen3.6-35B-A3B` snapshot. Its weights take 20.3 GB.

These runs use an Apple M3 Pro with an 18-core GPU and 36 GB of unified memory, on macOS 27.0.1, on AC power with low-power mode off. The GPU's recommended working set is 30.2 GB, so the model fits but leaves little room for long batches. Serving rows are the best of three runs.

Single sequence, from the `model` benchmark (median of 12 decode steps or 2 to 4 prefill passes, F16 K/V):

| Case | Result |
| --- | --- |
| Prefill, 512 tokens | 640 tokens/s |
| Prefill, 2,048 tokens | 718 tokens/s |
| Decode, 1 sequence at 128 tokens | 47.2 tokens/s (21.2 ms per step) |
| Decode, 1 sequence at 2K tokens | 47.6 tokens/s |
| Decode, 4 sequences at 128 tokens | 94.3 tokens/s |
| Decode, 4 sequences at 2K tokens | 89.1 tokens/s |
| Decode, 8 sequences at 128 tokens | 142 tokens/s |
| Load time | 5.5 s |

Bursts through the full pipeline: N requests arrive at once, each with a 128-token prompt and 64 output tokens, sampled at temperature 0.7, top-k 20 and top-p 0.95:

| Concurrent | Output tok/s | TTFT p50 | ITL p50 | ITL p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- |
| 1 | 37.8 | 0.34 s | 21.3 ms | 22.4 ms | 22.6 ms |
| 2 | 49.0 | 0.54 s | 32.7 ms | 34.1 ms | 34.3 ms |
| 4 | 70.0 | 0.95 s | 43.5 ms | 45.7 ms | 46.2 ms |
| 8 | 94.8 | 1.76 s | 55.2 ms | 66.3 ms | 66.9 ms |

Sixteen concurrent requests cannot run with every expert resident: each attempt failed with a Metal out-of-memory error. With `TUR_BENCH_EXPERT_CACHE_MB=4096`, the experts stay on the SSD behind a 4 GiB cache and the same burst completes at 25.7 output tokens/s, with a range of 25.3 to 25.7 across three runs. TTFT p50 is 6.2 s, and ITL p50 is 364 to 398 ms with a p95 of 1.25 s, because each sequence waits for its experts to be read on every step. The cache is the only way this burst runs on this Mac, and its figures are for a different configuration from the resident rows above.

Against llama.cpp and a reported MLX-family result on the same chip class:

| Engine, weights | Prefill | Decode, 1 sequence |
| --- | --- | --- |
| Tur, Q4K artifact | 640 tokens/s at 512, 718 at 2K | 47.2 tokens/s at 128 tokens, 47.6 at 2K |
| llama.cpp, unsloth UD-Q4_K_M | 665 tokens/s at 512, 650 at 2K | 35.6 tokens/s at 128 tokens, 34.8 at 2K |
| oMLX v0.7.0.dev2, 4-bit | 632 tokens/s at 1K | 39.2 tokens/s at 1K |

llama.cpp 0.8.0 (build 11960, Homebrew) ran on this Mac with `llama-bench -r 3` and its defaults: Metal and F16 K/V, flash attention chosen automatically. The GGUF is unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M` (revision `a483e9e6cbd595906af30beda3187c2663a1118c`, 22.1 GB). It mixes quantization types per tensor and is not the same format as Tur's RTN Q4K artifact, so the comparison is between engines with their usual 4-bit weights, not between kernels on identical bits. Quality was not measured for either. The oMLX row is a reported result from an M3 Pro with an 18-core GPU, not a rerun here, and it uses a 1K prompt.

Batched decode with 128-token prompts and 64 generated tokens, as output tokens per second over the whole burst. llama.cpp comes from `llama-batched-bench -c 8192 -ub 512` and is the best of three runs for 1 to 8 sequences and one run for 16; Tur is the serving benchmark above:

| Concurrent | Tur | llama.cpp |
| --- | --- | --- |
| 1 | 37.8 | 31.3 |
| 2 | 49.0 | 45.8 |
| 4 | 70.0 | 62.1 |
| 8 | 94.8 | 75.0 |
| 16 | 25.7 with a 4 GiB expert cache | 81.4 |

Tur is ahead of llama.cpp from 1 to 8 sequences, by 21% at one and 26% at eight. At 16 sequences llama.cpp keeps every weight resident and completes the burst, while Tur ran out of GPU memory and only completes it through the expert cache, so that row compares different configurations. Tur samples at temperature 0.7, top-k 20 and top-p 0.95 on the device, and `llama-batched-bench` has its own sampling, so the harnesses are not identical.

Speculative decoding with the model's own MTP head (`--mtp`, with prompt lookup and 4 drafts) speeds up decoding on this checkpoint. Each row is a burst of N requests with 128 output tokens, sampled at temperature 0.7, top-k 20 and top-p 0.95, as the best of three runs. Normal is decoding without speculation. These prompts are a repeated pattern, a new prose question and a code request, so they are not the burst prompts in the serving table above:

| Requests | Prompt | Normal tok/s | Lookup + MTP tok/s | Speedup |
| --- | --- | --- | --- | --- |
| 1 | Repetitive | 42.9 | 90.8 | 2.11× |
| 1 | New prose | 44.8 | 48.0 | 1.07× |
| 1 | Code | 45.2 | 53.4 | 1.18× |
| 2 | Repetitive | 59.5 | 120.8 | 2.03× |
| 2 | New prose | 58.9 | 74.5 | 1.27× |
| 2 | Code | 58.6 | 77.7 | 1.33× |
| 4 | Repetitive | 98.8 | 167.3 | 1.69× |
| 4 | New prose | 89.1 | 105.0 | 1.18× |
| 4 | Code | 89.4 | 117.4 | 1.31× |

On repetitive text prompt lookup supplies every draft and all of them are accepted. On new prose and code, MTP supplies nearly all the drafts, and 78 to 80% of its drafts were accepted on prose (105 of 133 at one request) and 83 to 87% on code. A pass with drafts costs more than a normal pass, about 34 ms against 21 ms for one request, so the gain on new text is smaller than the acceptance suggests. Time to first token is unchanged. The head adds 0.5 to 0.6 GiB to the resident plan at one to four requests. Output equality with normal decoding was not compared in this round.

Each concurrency ran in its own process. One process running one request and then four ran out of GPU memory at four, with or without the head, so the plan for the larger burst does not fit beside the smaller one's. Whether the first pipeline's memory is released late is not established.

### Long contexts with sparse attention

Only 10 of this model's 40 layers use full attention; the other 30 keep a fixed-size recurrent state, so long contexts cost less here than in a dense transformer. Sparse attention reads the 10 full-attention layers through an index map fitted on the Q4K weights (`tur sparse-map`, 5 minutes). The map comes from 131K tokens of this repository's documentation and Rust source, and no retrieval check was run on it, so these rows show speed only. Single sequence, from the `model` benchmark with F16 K/V. The chunk columns time one 2,048-token prefill chunk after a context of that length, and the decode columns time one step at that context:

| Context | Prefill chunk, dense | Prefill chunk, sparse | Decode, dense | Decode, sparse |
| --- | --- | --- | --- | --- |
| 128 tokens | | | 47.4 tokens/s (21.1 ms) | 48.7 tokens/s (20.6 ms) |
| 32K | 343 tokens/s (6.0 s) | 622 tokens/s (3.3 s) | 36.3 tokens/s (27.5 ms) | 40.6 tokens/s (24.6 ms) |
| 64K | 227 tokens/s (9.0 s) | 598 tokens/s (3.4 s) | 29.2 tokens/s (34.2 ms) | 40.1 tokens/s (25.0 ms) |

Dense attention slows with every doubling of the context, while the sparse run stays close to its 32K speed: a 2,048-token chunk after 64K takes 3.4 s instead of 9.0 s, and decode loses 0.4 ms per step between 32K and 64K instead of 6.7 ms. Decode of one sequence at 64K is within 18% of its 128-token speed.

One 32K-token prompt through the server (`TUR_BENCH_WORKLOADS=long`, 128 output tokens, one request, cold prefix cache, one run each). With MTP, ITL is zero inside a burst, so the gap between bursts is the figure to read:

| Mode | TTFT p50 | ITL p50 | Gap between bursts | Tokens per pass |
| --- | --- | --- | --- | --- |
| Dense | 74.9 s | 27.5 ms | 27.5 ms | 1.00 |
| Sparse | 60.2 s | 24.4 ms | 24.4 ms | 1.00 |
| Dense + MTP | 78.7 s | 29.0 ms | 29.0 ms | 1.32 |
| Sparse + MTP | 63.9 s | 27.8 ms | 38.8 ms | 1.75 |

Sparse attention cuts the first-token time by 20%, less than the chunk benchmark suggests, because the first 16K tokens of the prompt run dense, below the crossover. MTP does not pay at this depth on this synthetic prompt: the first token arrives 5 to 6% later, and the gap between bursts is 38.8 ms against 24.4 ms without it, for 1.75 tokens per pass. The 64K prompt was not run through the server.

The `model` benchmark ran each context and mode in its own process, because one process running the whole sweep (dense and sparse, up to 64K, with batches of 8) ran out of GPU memory.

## Qwen3.8-27B

Qwen3.8-27B is a dense model with 64 layers: every token goes through all 27 billion parameters, where the MoE above uses 3 billion. Decode reads the projection weights once per step whatever the batch, so extra sequences add little to a step, while prefill does the arithmetic of all 27 billion parameters for every prompt token. The rows below use a Q4K artifact made with `tur quantize --method rtn` (no calibration) from the pinned `Qwen/Qwen3.8-27B` snapshot. Quantizing took 3 min 50 s. The artifact file is 17.2 GB, of which the projections are 14.4 GB, and it loads in 4.2 s. Quality against Transformers has not been measured.

The machine is the M3 Pro described above, with F16 K/V. Single-sequence figures come from the `model` benchmark, each case group in its own process, because one process running the whole sweep ran out of GPU memory partway through.

| Case | Result |
| --- | --- |
| Prefill, 512 / 2K / 4K / 8K tokens | 96.0 / 93.7 / 93.5 / 92.3 tokens/s |
| Decode, 1 sequence at 128 tokens | 8.6 tokens/s (about 116 ms per step) |
| Decode, 2 / 4 / 8 sequences at 128 tokens | 15.2 / 29.0 / 52.5 tokens/s |
| Decode, 1 sequence at 2K / 4K / 8K tokens | 8.69 / 8.50 / 8.45 tokens/s |
| Decode, 4 sequences at 2K / 4K / 8K tokens | 28.1 / 27.2 / 23.3 tokens/s |

Repeats of the same 128-token step varied by up to 7% between processes (114 to 121 ms). A step takes about 116 ms with one sequence and 152 ms with eight, so batching is nearly free: eight sequences deliver six times the tokens per second of one. Context costs little for one sequence (3% slower at 8K than at 2K), while four sequences lose 17% over the same range. At 8K the four sequences hold 2.6 GiB of K/V and recurrent state.

Bursts through the full pipeline, with 128-token prompts and 64 output tokens, sampled at temperature 0.7, top-k 20 and top-p 0.95. Rows are the best of three runs, and the range is the output rate across the three:

| Concurrent | Output tok/s | Range | TTFT p50 | ITL p50 | ITL p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.8 | 6.4 to 6.8 | 2.0 s | 115 ms | 119 ms | 121 ms |
| 2 | 10.3 | 9.8 to 10.3 | 4.1 s | 132 ms | 136 ms | 137 ms |
| 4 | 15.3 | 14.6 to 15.3 | 7.7 s | 141 ms | 144 ms | 146 ms |
| 8 | 20.2 | | 15.4 s | 156 ms | 159 ms | 174 ms |

The burst rates are lower than the `model` benchmark's because they include prefill, which takes the 2 s to first token at one request (about 180 tokens at 90 tokens/s), and sampling. Eight concurrent requests complete too, at 20.2 tokens/s from a single run. Earlier builds ran out of GPU memory at this burst because the state pool occupied more device memory than the plan counted.

Against llama.cpp 0.8.0 (build 11960, Homebrew) on the same Mac, with unsloth's `Qwen3.8-27B-UD-Q4_K_M` GGUF (revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`, 15.3 GiB) and its defaults: Metal and F16 K/V. The GGUF mixes quantization types per tensor and is not the same format as Tur's RTN Q4K artifact, so this compares engines with their usual 4-bit weights, not kernels on identical bits. Quality was not measured for either. This is a short run: prefill at 512 to 4K tokens is the mean of two `llama-bench` repeats and everything else is one run, so differences of a few percent are within noise. Decode is 32 tokens after the stated context.

| Test | Tur | llama.cpp |
| --- | --- | --- |
| Prefill, 512 tokens | 96.0 tokens/s | 97.1 |
| Prefill, 2K | 93.7 | 96.3 |
| Prefill, 4K | 93.5 | 94.4 |
| Prefill, 8K | 92.3 | 93.4 |
| Decode, 1 sequence at 128 tokens | 8.6 | 7.95 |
| Decode, 1 sequence at 2K | 8.69 | 7.78 |
| Decode, 1 sequence at 8K | 8.45 | 7.10 |

Batched decode at about 128 tokens of context, in tokens/s over all sequences (Tur from the `model` benchmark, llama.cpp from `llama-batched-bench -c 8192 -ub 512` with 128-token prompts, as its generation rate):

| Sequences | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Tur | 8.6 | 15.2 | 29.0 | 52.5 |
| llama.cpp | 7.3 | 11.8 | 23.2 | 41.1 |

Prefill is level, with llama.cpp 1 to 3% ahead. Both engines are limited by compute there. Decode is where they differ: Tur is 8% faster with one sequence at 128 tokens, 19% at 8K, and 25 to 29% faster with 2 to 8 sequences. Bursts through the serving pipeline were not compared, because the two harnesses use different output lengths and sampling. MLX was not run on this model.
