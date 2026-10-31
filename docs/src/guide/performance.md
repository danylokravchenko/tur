# Measured performance

This page collects Tur's current numbers. Each architecture chapter explains the mechanism behind them, and [`just bench`](../development/benchmarks.md) reproduces them; re-measure on your own Mac before relying on a figure.

Unless a section says otherwise, every table uses:

- an Apple M1 with 16 GB, on AC power with low-power mode off;
- Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`;
- F16 K/V in 16-token pages, F32 recurrent state and 256-token prefill chunks.

Every timed region ends when the GPU finishes, and serving rows are the best of three runs.

## Single model pass

Raw model speed, without scheduling or HTTP (`model` benchmark, synthetic tokens).

Decode throughput in tokens/s across all sequences, at a 128-token context:

| Weights | 1 seq | 2 seq | 4 seq | 8 seq | 16 seq |
| --- | --- | --- | --- | --- | --- |
| Q8_0, converted at load | 58 | 108 | 181 | 311 | 357 |
| Q8_0 artifact | 57 | 111 | 181 | 310 | 356 |
| Q4K, converted at load | 92 | 112 | 200 | 334 | 353 |
| Q4K GPTQ artifact | 90 | 114 | 202 | 339 | 350 |
| Mixed 5.5-bit GPTQ artifact | 79 | 111 | 197 | 339 | 356 |

With one sequence, decode is limited by how fast the GPU reads weights, so Q4K is about 1.6× faster than Q8_0. From two sequences on, the weights are shared across the batch, each sequence's recurrent state becomes the main cost, and the formats converge.

Decode at longer contexts, in tokens/s:

| Weights / K/V | 1 seq at 2K | 4 seq at 2K | 1 seq at 8K | 4 seq at 8K |
| --- | --- | --- | --- | --- |
| Q8_0 / F16 | 56 | 165 | 50 | 134 |
| Q8_0 / Q8 | 55 | 170 | 52 | 140 |
| Q4K / F16 | 89 | 184 | 75 | 149 |
| Q4K / Q8 | 85 | 181 | 79 | 153 |

Prefill, in prompt tokens/s for one sequence:

| Weights | 2K tokens | 8K tokens |
| --- | --- | --- |
| Q8_0 | 1,311 | 1,031 |
| Q4K | 1,210 | 982 |

A 2K-token prompt takes about 1.6 s, and an 8K-token prompt about 8 s.

## Serving

Bursts through the full scheduled pipeline: N requests arrive at once, each with a 128-token prompt and 64 output tokens, using Q8_0 weights and sampling at temperature 0.7, top-k 20 and top-p 0.95. Output tokens/s is the total across the burst, time to first token (TTFT) counts from submission, and inter-token latency (ITL) is the gap between tokens of one request.

| Concurrent | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 49.6 | 0.14 s | 0.14 s | 17.8 ms | 18.8 ms | 19.0 ms |
| 2 | 85.8 | 0.25 s | 0.25 s | 18.8 ms | 19.8 ms | 24.8 ms |
| 4 | 121 | 0.45 s | 0.47 s | 22.7 ms | 23.8 ms | 44.7 ms |
| 8 | 186 | 0.87 s | 0.91 s | 26.9 ms | 27.9 ms | 46.6 ms |
| 16 | 204 | 1.63 s | 1.70 s | 46.0 ms | 48.0 ms | 178 ms |

Greedy decoding is only 2 to 8% faster, because sampling runs on the GPU.

With mixed prompt lengths, sixteen requests at once where half have 128-token prompts and half 2K-token prompts, the burst produced 55 output tokens/s with a median TTFT of 8.2 s and a median ITL of 217 ms. The eight long prompts add up to 16K prefill tokens, about 13 s of GPU work, which the scheduler interleaves with the short requests' replies.

## Requests arriving over time

48 requests with 512-token prompts and 64 output tokens arrive at random (a Poisson process), sampled as above.

| Arrival rate | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p99 | GPU idle |
| --- | --- | --- | --- | --- | --- | --- |
| 2/s | 93.1 | 3.33 s | 6.60 s | 55 ms | 258 ms | 4.8% |
| 4/s | 105.5 | 6.64 s | 14.5 s | 53 ms | 249 ms | 5.2% |

Both rates ask for more prefill than one M1 can do, so the queue grows and TTFT climbs over the run. The server accepts new requests while the GPU is busy, which keeps it idle only about 5% of the time.

## Grammar-constrained output

A JSON-like grammar masks every token; greedy decoding.

| Concurrent | Masks on CPU: tok/s, ITL p50 | Masks on GPU (default): tok/s, ITL p50 |
| --- | --- | --- |
| 1 | 42.3, 20.1 ms | 47.6, 17.4 ms |
| 4 | 106.0, 30.4 ms | 129.5, 22.0 ms |
| 8 | 158.1, 37.7 ms | 197.8, 25.9 ms |
| 16 | 176.5, 64.2 ms | 221.0, 45.4 ms |

With masks applied on the GPU, constrained output runs as fast as unconstrained output.

## Prefix reuse

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

## Speculative decoding

80 output tokens per request, three repeats each, total output tokens/s.

Greedy:

| Prompt | Seqs | Normal | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 54.1 | 53.5 | 72.3 |
| | 4 | 157.1 | 156.5 | 194.1 |
| | 8 | 249.7 | 250.6 | 244.0 |
| Code | 1 | 53.7 | 54.0 | 72.3 |
| | 4 | 155.9 | 157.6 | 193.5 |
| | 8 | 247.7 | 248.0 | 240.8 |
| Repetitive | 1 | 52.1 | 151.5 | 144.7 |
| | 4 | 145.0 | 272.9 | 262.2 |
| | 8 | 221.9 | 301.2 | 289.3 |

Sampled at temperature 0.7, top-k 20, top-p 0.95:

| Prompt | Seqs | Normal | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 53.0 | 53.1 | 65.2 |
| | 4 | 155.4 | 156.1 | 177.3 |
| | 8 | 249.0 | 248.7 | 241.2 |
| Code | 1 | 53.1 | 53.0 | 73.0 |
| | 4 | 155.6 | 155.9 | 192.7 |
| | 8 | 247.3 | 247.6 | 238.7 |
| Repetitive | 1 | 51.5 | 153.3 | 146.2 |
| | 4 | 134.5 | 157.6 | 179.5 |
| | 8 | 214.3 | 226.4 | 229.0 |

Prompt lookup gives up to 2.9× on repetitive text. On new text, the MTP head adds 1.1 to 1.4× with one to four sequences. With eight sequences the head costs a few percent, so Tur switches it off there by itself. In every configuration the output was identical to normal decoding.

## Loading

A fresh process with a warm file cache, timed until the last weight reaches the GPU. Peak memory comes from `/usr/bin/time -l`.

| Weights | Load time | Weight memory | Peak RSS |
| --- | --- | --- | --- |
| Q4K, converted at load | 8.9 s | 933 MB | 1.46 GB |
| Q8_0, converted at load | 1.58 s | 1,309 MB | 2.11 GB |
| Q4K GPTQ artifact | 0.16 s | 425 MB | 0.71 GB |
| Mixed 5.5-bit GPTQ artifact | 0.20 s | 521 MB | 0.80 GB |
| Q8_0 artifact | 0.27 s | 801 MB | 1.20 GB |

## Quality of weight formats

Teacher-forced over about 8K held-out tokens per domain, against F32 weights. KL is the mean divergence in nats per token (lower is better), and top-1 is how often the format's most likely token matches F32's.

| Weights | Prose KL / top-1 | Code KL / top-1 | Retrieval KL / top-1 | Tools KL / top-1 |
| --- | --- | --- | --- | --- |
| Q8_0 | 0.0007 / 98.4% | 0.0004 / 99.2% | 0.0003 / 98.3% | 0.0014 / 99.4% |
| Mixed 5.5-bit GPTQ | 0.0346 / 88.6% | 0.0182 / 94.2% | 0.0144 / 90.9% | 0.0663 / 96.0% |
| Q4K GPTQ | 0.0704 / 84.4% | 0.0343 / 92.1% | 0.0296 / 89.6% | 0.1003 / 94.5% |
| Q4K importance-weighted | 0.0941 / 83.3% | 0.0510 / 90.8% | 0.0368 / 88.9% | 0.0986 / 94.7% |
| Q4K plain rounding | 0.1413 / 79.7% | 0.0832 / 88.4% | 0.0577 / 87.0% | 0.1142 / 93.9% |

For K/V, F16 is indistinguishable from F32, and Q8 stays within 1e-4 nats of F32 with 99.3 to 99.8% top-1 agreement.

## Gemma 4 E2B

`google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, Q8_0 weights and F16 K/V, measured with `just bench-gemma`. Ranges span the measurement rounds. E2B does about three times the work per token of the 0.8B Qwen, and its 262K-token output layer keeps single-sequence decode near 20 tokens/s.

| Case | Result |
| --- | --- |
| Prefill, 512 tokens | 417 tokens/s |
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

## Gemma 4 audio

E2B with Q8_0 weights encoding clips on their own queue, while a 400-token greedy request decodes (`just bench whisper` with `TUR_GEMMA4_E2B`):

| Case | Result |
| --- | --- |
| Encoding a 5 s clip, alone | 0.21 s |
| Encoding a 30 s clip, alone | 1.45 s |
| Encoding a 30 s clip beside a decoding request | 2.70 s |
| Text pass, alone: median, slowest | 47 ms, 109 ms |
| Text pass beside that clip: median, slowest | 48 ms, 165 ms |

## Compared with llama.cpp

llama.cpp 0.7.0 (build 11669, Homebrew) on the same Mac, with unsloth's Qwen3.5-0.8B GGUF files (revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`) and its defaults: Metal and F16 K/V. Its figures are the best of three runs. Q8_0 is the like-for-like format. GGUF's Q4_K_M mixes Q4_K and Q6_K blocks (497 MiB), while Tur's Q4K uses Q4_K for every projection.

Single sequence, in tokens/s (llama.cpp from `llama-bench`):

| Test | Tur Q8_0 | llama.cpp Q8_0 | Tur Q4K | llama.cpp Q4_K_M |
| --- | --- | --- | --- | --- |
| Decode at 128 tokens | 58 | 51.4 | 92 | 65.6 |
| Decode at 2K | 56 | 49.4 | 89 | 63.5 |
| Decode at 8K | 50 | 46.2 | 75 | 58.9 |
| Prefill 2K | 1,311 | 1,323 | 1,210 | 1,188 |
| Prefill 8K | 1,031 | 1,142 | 982 | 1,037 |

Batched decode with Q8_0, total tokens/s at about 128 tokens of context (Tur from the `model` benchmark, llama.cpp from `llama-batched-bench`):

| Sequences | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Tur | 58 | 108 | 181 | 311 | 357 |
| llama.cpp | 51.6 | 85.6 | 138 | 201 | 230 |

Bursts of N requests with 128-token prompts and 64 output tokens, in output tokens/s. Tur's figures go through the full pipeline with sampling. llama.cpp's are `llama-batched-bench` wall time for the same prefill and decode, with no scheduler and no sampling.

| Concurrent | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Tur Q8_0, sampled | 49.6 | 85.8 | 121 | 186 | 204 |
| llama.cpp Q8_0 | 47.6 | 75.5 | 114 | 155 | 170 |
| llama.cpp Q4_K_M | 59.4 | 78.7 | 114 | 151 | 160 |

Tur decodes 1.1 to 1.4× faster with one sequence and up to 1.55× faster with 8 to 16, and finishes bursts of 8 to 16 requests 1.2× faster end to end. Prefill is level at 2K tokens; at 8K, llama.cpp is 6 to 11% faster.

```sh
llama-bench -m <gguf> -p 2048,8192 -n 0 -r 3
llama-bench -m <gguf> -p 0 -n 64 -d 128,2048,8192 -r 3
llama-batched-bench -m <gguf> -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16
```

## Compared with MLX and oMLX

These are published community results for Qwen3.5-0.8B on an M1 (8-core GPU) with 16 GB, not reruns on this Mac: mlx_lm from [whatcani.run](https://www.whatcani.run/?device=Apple+M1:8:Apple+M1:8:16) with 4,096 prompt and 1,024 output tokens, and oMLX 0.7.3 from [omlx.ai](https://omlx.ai/benchmarks/performance/yw52053o) at temperature 0.7. Tur has no 4K-context row, so its 2K and 8K figures bracket theirs. MLX's OptiQ-4bit (620 MB) is a mixed 4-bit format; Tur's Q4K GPTQ artifact is 425 MB.

Single sequence, in tokens/s:

| Engine, weights | Prefill | Decode | Peak memory |
| --- | --- | --- | --- |
| Tur Q8_0 | 1,311 at 2K, 1,031 at 8K | 56 at 2K, 50 at 8K | 1.20 GB RSS (artifact) |
| mlx_lm 8-bit | 814 at 4K | 46.1 at 4K | 2.46 GB |
| Tur Q4K GPTQ | 1,210 at 2K, 982 at 8K | 89 at 2K, 75 at 8K | 0.71 GB RSS |
| mlx_lm OptiQ-4bit | 946 at 4K | 56.0 at 4K | 3.50 GB |
| oMLX OptiQ-4bit | 1,025 at 1K, 990 at 4K | 73.7 at 1K, 68.6 at 4K | 2.17 GB footprint |

Batched 4-bit decode, total tokens/s (Tur at 128 tokens of context; oMLX doesn't state its context):

| Sequences | 1 | 2 | 4 |
| --- | --- | --- | --- |
| Tur Q4K GPTQ | 90 | 114 | 202 |
| oMLX OptiQ-4bit | 73.7 | 123 | 198 |

With one sequence, Tur decodes 1.1 to 1.3× faster than oMLX and 1.1 to 1.6× faster than mlx_lm, with half or less of their reported peak memory. Its 4-bit prefill is level with oMLX's and ahead of mlx_lm's. With two to four sequences the lead disappears: oMLX scales better from one sequence to two, because Tur's batched decode is limited by recurrent-state traffic rather than weights.
