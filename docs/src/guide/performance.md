# Measured performance

These numbers combine two measurement rounds on the same Mac. The first round ran during
development (2026-10-23 and 2026-10-24); the second reran every benchmark on 2026-10-25
at commit `0b339ec`. Each figure is the better of the two rounds: higher throughput,
lower latency, load time or memory. The rounds agree within run-to-run noise except
where later scheduling work made serving faster. Re-measure with
[`just bench`](../development/benchmarks.md) before relying on a number.

**Setup for every table:**

- Hardware: Apple M1 with 16 GB, macOS 26.6.2, not in low-power mode.
- Model: Qwen3.5-0.8B (`2fc06364715b967f1860aea9cf38778875588b17`), except
  [Gemma 4 E2B](#gemma-4-e2b).
- Defaults unless noted: F16 K/V with 16-token pages, F32 recurrent state, 256-token
  prefill chunks.
- Timing: every timed region ends at device completion.
- Serving rows are the best of three runs.

## Single model pass

Raw model speed, without scheduling or HTTP (`model` benchmark, synthetic tokens).

**Decode throughput** (tokens/s across all sequences, 128-token context):

| Weights | 1 seq | 2 seq | 4 seq | 8 seq | 16 seq |
| --- | --- | --- | --- | --- | --- |
| Q8_0 (load-time) | 58 | 108 | 181 | 311 | 357 |
| Q8_0 artifact | 57 | 111 | 181 | 310 | 356 |
| Q4K (load-time) | 92 | 112 | 200 | 334 | 353 |
| Q4K GPTQ artifact | 90 | 114 | 202 | 339 | 350 |
| Mixed 5.5-bit GPTQ artifact | 79 | 111 | 197 | 339 | 356 |

At one sequence, decode is bound by weight bandwidth, so Q4K is about 1.6× faster than
Q8_0. From two sequences onward, batched kernels and recurrent-state traffic dominate,
and the formats converge.

**Decode at longer context** (tokens/s):

| Weights / K/V | 1 seq @ 2K | 4 seq @ 2K | 1 seq @ 8K | 4 seq @ 8K |
| --- | --- | --- | --- | --- |
| Q8_0 / F16 | 56 | 165 | 50 | 134 |
| Q8_0 / Q8 | 55 | 170 | 52 | 140 |
| Q4K / F16 | 89 | 184 | 75 | 149 |
| Q4K / Q8 | 85 | 181 | 79 | 153 |

**Prefill** (prompt tokens/s, one sequence):

| Weights | 2K tokens | 8K tokens |
| --- | --- | --- |
| Q8_0 | 1,311 | 1,031 |
| Q4K | 1,210 | 982 |
| Mixed 5.5-bit | 1,245 | — |

A 2K-token prompt therefore takes about 1.6 s; an 8K-token prompt about 7.9 s.

## Serving

Closed-loop bursts through the scheduled pipeline: N requests submitted at once, each
with a 128-token prompt and 64 output tokens; Q8_0 weights. "Output tok/s" is aggregate
over the burst. TTFT counts from submission.

**Sampled** (T=0.7, top-k 20, top-p 0.95; device sampling):

| Concurrent | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 49.6 | 0.14 s | 0.14 s | 17.8 ms | 18.8 ms | 19.0 ms |
| 2 | 85.8 | 0.25 s | 0.25 s | 18.8 ms | 19.8 ms | 24.8 ms |
| 4 | 121 | 0.45 s | 0.47 s | 22.7 ms | 23.8 ms | 44.7 ms |
| 8 | 186 | 0.87 s | 0.91 s | 26.9 ms | 27.9 ms | 46.6 ms |
| 16 | 204 | 1.63 s | 1.70 s | 46.0 ms | 48.0 ms | 178 ms |

**Greedy** (T=0):

| Concurrent | Output tok/s | TTFT p50 | ITL p50 | ITL p99 |
| --- | --- | --- | --- | --- |
| 1 | 51.4 | 0.13 s | 17.2 ms | 18.8 ms |
| 2 | 88.7 | 0.25 s | 18.4 ms | 20.4 ms |
| 4 | 131 | 0.45 s | 22.1 ms | 30.7 ms |
| 8 | 191 | 0.87 s | 26.4 ms | 57.5 ms |
| 16 | 208 | 1.61 s | 45.1 ms | 178 ms |

Sampled serving runs within 2–8% of greedy, because sampling runs on the GPU.

**Mixed prompts.** Sixteen requests at once, half with 128-token and half with 2K-token
prompts:

| Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 |
| --- | --- | --- | --- | --- |
| 55.3 | 8.23 s | 14.1 s | 217 ms | 288 ms |

The eight 2K-token prompts are 16K prefill tokens, about 13 s of prefill on this GPU,
which the scheduler interleaves with the short requests' decodes. Sampled and greedy
runs agree within 1%.

## Open-loop arrivals

48 requests with 512-token prompts and 64 output tokens arrive as a Poisson process,
sampled as above.

| Rate | Intake | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p99 | Device idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2/s | serial | 91.7 | 3.62 s | 7.49 s | 55 ms | 284 ms | 9.4% |
| 2/s | overlapped | 93.1 | 3.33 s | 6.60 s | 55 ms | 258 ms | 4.8% |
| 4/s | serial | 101.6 | 8.43 s | 15.3 s | 49 ms | 379 ms | 9.6% |
| 4/s | overlapped | 105.5 | 6.64 s | 14.5 s | 53 ms | 249 ms | 5.2% |

Both rates exceed what one M1 sustains with 512-token prompts: about 1,000 prompt
tokens/s of prefill alone is needed at 2/s. Queues therefore grow, and TTFT climbs over
the run. Overlapped intake is the default in `tur serve`.

## Grammar-constrained output

JSON-like grammar, every token masked; greedy.

| Concurrent | Host masks: tok/s, ITL p50 | Device masks (default): tok/s, ITL p50 |
| --- | --- | --- |
| 1 | 42.3, 20.1 ms | 47.6, 17.4 ms |
| 4 | 106.0, 30.4 ms | 129.5, 22.0 ms |
| 8 | 158.1, 37.7 ms | 197.8, 25.9 ms |
| 16 | 176.5, 64.2 ms | 221.0, 45.4 ms |

With device masks, constrained generation runs at the speed of unconstrained generation.

## Prefix reuse

The `reuse` workloads, three runs each. "Before" is the earlier reuse policy (LRU
eviction, no waiting for a producer), measured during development; it is no longer in
the tree.

| Workload | Metric | Before | Now |
| --- | --- | --- | --- |
| 8 users, cold 1.5K-token system prompt at once | TTFT p50 | 9.58 s | 1.69 s |
| | Output tok/s | 23.6 | 91.6 |
| 8 users, cold 300-token system prompt | TTFT p50 | 3.32 s | 2.08 s |
| 4 agents × 5 steps on a 1K-token manual | Prompt tokens cached | 72% | 83% |
| | Wall time | 12.3 s | 9.9 s |
| | First-step TTFT p50 | 3.21 s | 1.07 s |
| Hot prefix changing under cold churn | Hot tokens restored | 0 | 6,144 |
| | Wall time | 35.3 s | 32.5 s |
| 16 distinct 512-token prompts (no reuse) | TTFT p50, tok/s | 3.86 s, 56.9 | 3.83 s, 57.1 |
| Warm vs cold contention (16 requests, 4 slots) | TTFT p50, tok/s | 4.95 s, 44.7 | 4.88 s, 45.3 |

Eight users behind an already *warm* 1.5K-token system prompt get their first token in
0.56 s p50. Without prefix snapshots, the same burst takes 9.46 s.

Template-divergence snapshots were measured separately on 2026-10-25: four conversations
of six turns, a 256-token initial user message, 64-token follow-ups and up to 48 output
tokens per turn, greedy. The M1 ran macOS 26.6.2 on AC power with low-power mode off,
Qwen3.5-0.8B at revision `2fc06364`, Q8_0 weights, F16 K/V and F32 recurrent state.
Baseline `ae3f9de` and the implementation used equal snapshot budgets and 256-token
spacing. Values are three-run medians with min–max ranges:

| Snapshots | Metric | Ordinary boundaries | Also predict template divergence |
| --- | --- | --- | --- |
| 8 | Prompt tokens cached | 40.9% | 44.5% |
| | Last-turn TTFT p50 | 1.498 s (1.495–1.506) | 1.462 s (1.462–1.475) |
| | Six-turn wall time | 13.17 s (13.17–13.41) | 13.17 s (13.00–13.18) |
| 16 | Prompt tokens cached | 52.1% | 68.3% |
| | Last-turn TTFT p50 | 1.134 s (1.120–1.149) | 0.792 s (0.775–0.837) |
| | Six-turn wall time | 12.18 s (12.05–12.26) | 10.86 s (10.78–10.89) |

The extra boundary helps only while retained: at eight snapshots the total-time gain
was within noise. With 16, the conversations finished 11% sooner and last-turn TTFT
fell 30%. Single-turn, tool-loop and churn controls stayed within their run-to-run
spread; first turns take no extra pass. Reproduce with `TUR_BENCH_WORKLOADS=prefix
TUR_BENCH_SPACING=256 TUR_BENCH_PREFIX_ENTRIES=8 just bench serving`, then repeat with 16
(set the pinned local checkpoint in `TUR_BENCH_MODEL`).

## Speculative decoding

80 output tokens per request; three repeats each; output tok/s (aggregate).

**Greedy:**

| Prompt | Seqs | Ordinary | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 54.1 | 53.5 | 72.3 |
| | 4 | 157.1 | 156.5 | 194.1 |
| | 8 | 249.7 | 250.6 | 244.0 |
| Code | 1 | 53.7 | 54.0 | 72.3 |
| | 4 | 155.9 | 157.6 | 193.5 |
| | 8 | 247.7 | 248.0 | 240.8 |
| Repetitive | 1 | 52.1 | **151.5** | 144.7 |
| | 4 | 145.0 | **272.9** | 262.2 |
| | 8 | 221.9 | **301.2** | 289.3 |

**Sampled** (T=0.7, top-k 20, top-p 0.95):

| Prompt | Seqs | Ordinary | Lookup (4 drafts) | Lookup + MTP |
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

How to read these:

- **Repetitive text:** prompt lookup gives up to 2.9×.
- **Novel text at one to four sequences:** the MTP head gives 1.14–1.37×, about 1.35× at
  one sequence (1.23× on sampled prose) and 1.14–1.24× at four.
- **Eight sequences:** MTP costs 2–4%, so it gates itself off.
- **Exactness:** outputs are identical to ordinary decoding in every configuration.

## Loading

Fresh process, page cache warm, through the last upload; peak RSS from
`/usr/bin/time -l`.

| Weights | Load time | Resident weights | Peak RSS |
| --- | --- | --- | --- |
| Q4K, converted at load | 8.9 s | 933 MB | 1.46 GB |
| Q8_0, converted at load | 1.58 s | 1,309 MB | 2.11 GB |
| Q4K GPTQ artifact | 0.16 s | 425 MB | 0.71 GB |
| Mixed 5.5-bit GPTQ artifact | 0.20 s | 521 MB | 0.80 GB |
| Q8_0 artifact | 0.27 s | 801 MB | 1.20 GB |

## Gemma 4 E2B

Measured 2026-10-29 on the same M1 with `google/gemma-4-E2B-it`
(`3e22461f65e89153144f8adb70e3b8c2cc9845a7`), Q8_0 weights and F16 K/V, through
`just bench-gemma` and its variables. Each row alternates the earlier and current code
in one session; ranges are the rounds. E2B does about three times the 0.8B Qwen's work
per token, and its 262K-token LM head makes decode bandwidth-bound near 20 tokens/s.

**Single model pass** (fused `q|k|v` and `gate|up`, in-place norms and rotary,
residuals fused with the next norm, sliding splits over the window):

| Case | Before | After |
| --- | --- | --- |
| Prefill, 512 tokens | 323 tok/s | 417 tok/s |
| Decode, 1 seq @ 128 | 19.4–19.8 | 19.8–21.3 |
| Decode, 4 seq @ 128 | 48.6 | 51.8 |
| Decode, 1 seq @ 8K | 16.4–18.1 | 18.9–20.0 |
| Decode, 4 seq @ 8K | 42.6 | 44.6 |
| Engine latency per token | 50.7 ms | 48.0 ms |

**Window K/V** (sliding owners store only their window, 8K context):

| | Full pages | Window pages |
| --- | --- | --- |
| K/V bytes per token past the window | 18,432 | 6,144 |
| State, 1 sequence | 151 MB | 56.8 MB |
| State, 4 sequences | 604 MB | 227 MB |
| Decode, 1 / 4 seq | 18.2–18.8 / 37.4–43.9 | 18.2–19.2 / 39.4–44.5 |
| Prefill, 2,048 tokens | 301–334 | 345–354 |

**Sliding-layer attention** (8 heads, one K/V head of 256, 512-key window): the tiled
kernel takes 13.0 ms against the per-query kernel's 20.1 ms at 2,048 queries over F16,
and loses below 256 queries, so ungated passes tile from 64 threadgroups. End-to-end
prefill moves within noise: projections dominate E2B prefill.

**Speculative decoding** (96 greedy tokens, two repeats, output tok/s):

| Prompt | Seqs | Ordinary | Lookup (4 drafts) | Lookup + assistant |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 20.0–20.3 | 19.7–20.0 | 28.5–28.9 |
| | 4 | 48.6–48.8 | 45.6–46.7 | 58.9–61.1 |
| Code | 1 | 19.9–20.1 | 20.1–20.5 | 23.3–26.1 |
| | 4 | 47.7–48.1 | 46.5–47.3 | 64.1–65.4 |
| Repetitive | 1 | 19.4–19.6 | 64.3–65.0 | 58.7–61.2 |
| | 4 | 43.7–45.2 | 99.4–110.3 | 111.4–113.2 |

The assistant (`--draft-model google/gemma-4-E2B-it-assistant`) drafted one token per
pass, 63–66% accepted, at about 2.9 ms each. All 36 runs reproduced ordinary decoding's
token IDs.

**Chained assistant drafts** (lookup plus the assistant, measured on 2026-10-29 against
one draft per pass, alternating builds over two rounds, output tok/s):

| Prompt | Seqs | One draft | Chained |
| --- | --- | --- | --- |
| Novel prose | 1 | 26.3–29.9 | 31.6–33.8 |
| | 4 | 59.6–62.0 | 54.8–58.3 |
| Code | 1 | 28.7–29.9 | 27.1–29.8 |
| | 4 | 64.9–66.3 | 56.2–58.5 |
| Repetitive | 1 | 33.2–33.6 | 50.0–50.5 |
| | 4 | 62.8–70.9 | 84.1–87.6 |

Chains therefore run only when one sequence decodes; above that the assistant drafts one
token per pass. Every run reproduced the single-draft token IDs.

**Loading:**

| Weights | Load time | Resident weights | Peak RSS |
| --- | --- | --- | --- |
| Q8_0, converted at load | 11.1–12.3 s | 7.93 GB | 2.2–2.7 GB |
| Q8_0 artifact (`tur quantize --method rtn`) | 4.8–5.5 s | 7.12 GB | 3.2 GB |

The artifact gathers embeddings from its packed LM head (`--embedding head`), which saves
the 805 MB BF16 embedding table. 4.4 GiB of either is the BF16 per-layer embedding.

## Quality of weight formats

Teacher-forced over 8,192 held-out tokens per domain (7,206 for retrieval), against F32
weights. KL is mean nats per token, lower is better; top-1 is agreement with F32's
argmax. The rerun reproduced these values.

| Weights | Prose KL / top-1 | Code KL / top-1 | Retrieval KL / top-1 | Tools KL / top-1 |
| --- | --- | --- | --- | --- |
| Q8_0 | 0.0007 / 98.4% | 0.0004 / 99.2% | 0.0003 / 98.3% | 0.0014 / 99.4% |
| Mixed 5.5-bit GPTQ | 0.0346 / 88.6% | 0.0182 / 94.2% | 0.0144 / 90.9% | 0.0663 / 96.0% |
| Q4K GPTQ | 0.0704 / 84.4% | 0.0343 / 92.1% | 0.0296 / 89.6% | 0.1003 / 94.5% |
| Q4K importance-weighted | 0.0941 / 83.3% | 0.0510 / 90.8% | 0.0368 / 88.9% | 0.0986 / 94.7% |
| Q4K RTN | 0.1413 / 79.7% | 0.0832 / 88.4% | 0.0577 / 87.0% | 0.1142 / 93.9% |

K/V formats, from the same evaluation method:

- Q8 K/V stays within 1e-4 nats of F32 K/V, with 99.3–99.8% top-1 agreement.
- F16 K/V is indistinguishable from F32 (KL below 1e-7, 100% top-1).

## Compared with llama.cpp

llama.cpp 0.7.0 (build 11669, Homebrew) on the same Mac, with unsloth's Qwen3.5-0.8B GGUF
files (revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`) and its defaults: Metal,
F16 K/V. Its figures are the best of three runs. Q8_0 is the like-for-like format; GGUF
Q4_K_M mixes Q4_K and Q6_K blocks (497 MiB), against Tur's all-Q4K projections.

**Single sequence** (tokens/s; llama.cpp from `llama-bench`):

| Test | Tur Q8_0 | llama.cpp Q8_0 | Tur Q4K | llama.cpp Q4_K_M |
| --- | --- | --- | --- | --- |
| Decode at 128 tokens | 58 | 51.4 | 92 | 65.6 |
| Decode at 2K | 56 | 49.4 | 89 | 63.5 |
| Decode at 8K | 50 | 46.2 | 75 | 58.9 |
| Prefill 2K | 1,311 | 1,323 | 1,210 | 1,188 |
| Prefill 8K | 1,031 | 1,142 | 982 | 1,037 |

**Batched decode** (Q8_0, aggregate tokens/s at about 128 tokens of context; Tur from the
`model` benchmark, llama.cpp from `llama-batched-bench`):

| Sequences | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Tur | 58 | 108 | 181 | 311 | 357 |
| llama.cpp | 51.6 | 85.6 | 138 | 201 | 230 |

**Bursts** of N requests with 128-token prompts and 64 output tokens (output tokens/s over
the burst). Tur's figures run through the scheduled pipeline with sampling; llama.cpp's
are `llama-batched-bench` wall time for the same prefill and decode, with no scheduler and
no sampling.

| Concurrent | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Tur Q8_0, sampled | 49.6 | 85.8 | 121 | 186 | 204 |
| llama.cpp Q8_0 | 47.6 | 75.5 | 114 | 155 | 170 |
| llama.cpp Q4_K_M | 59.4 | 78.7 | 114 | 151 | 160 |

Tur decodes 1.1–1.4× faster at one sequence and up to 1.55× faster at 8–16, and leads
end-to-end bursts by 1.2× at 8–16 requests. Prefill is level at 2K tokens; at 8K,
llama.cpp is 6–11% faster.

```sh
llama-bench -m <gguf> -p 2048,8192 -n 0 -r 3
llama-bench -m <gguf> -p 0 -n 64 -d 128,2048,8192 -r 3
llama-batched-bench -m <gguf> -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16
```

## Compared with MLX and oMLX

Published community results for Qwen3.5-0.8B on an M1 (8-core GPU) with 16 GB: mlx_lm
from [whatcani.run](https://www.whatcani.run/?device=Apple+M1:8:Apple+M1:8:16) (4,096
prompt and 1,024 output tokens) and oMLX 0.7.3 from
[omlx.ai](https://omlx.ai/benchmarks/performance/yw52053o) (macOS 27.0.1, 2026-10-25,
T=0.7). Both are single submissions on another machine, not reruns on this Mac. Tur has
no 4K-context row, so its 2K and 8K figures bracket the 4K results. MLX's OptiQ-4bit
(620 MB) is a mixed 4-bit format, against Tur's all-Q4K GPTQ artifact (425 MB).

**Single sequence** (tokens/s):

| Engine, weights | Prefill | Decode | Peak memory |
| --- | --- | --- | --- |
| Tur Q8_0 | 1,311 at 2K, 1,031 at 8K | 56 at 2K, 50 at 8K | 1.20 GB RSS (artifact) |
| mlx_lm 8-bit | 814 at 4K | 46.1 at 4K | 2.46 GB |
| Tur Q4K GPTQ | 1,210 at 2K, 982 at 8K | 89 at 2K, 75 at 8K | 0.71 GB RSS |
| mlx_lm OptiQ-4bit | 946 at 4K | 56.0 at 4K | 3.50 GB |
| oMLX OptiQ-4bit | 1,025 at 1K, 990 at 4K | 73.7 at 1K, 68.6 at 4K | 2.17 GB footprint |

**Batched decode** (4-bit, aggregate tokens/s; Tur at 128 tokens of context, oMLX's
context unstated):

| Sequences | 1 | 2 | 4 |
| --- | --- | --- | --- |
| Tur Q4K GPTQ | 90 | 114 | 202 |
| oMLX OptiQ-4bit | 73.7 | 123 | 198 |

At one sequence Tur decodes 1.1–1.3× faster than oMLX and 1.1–1.6× faster than mlx_lm.
Its 4-bit prefill is level with oMLX's and ahead of mlx_lm's; at 8-bit it prefills
1.3–1.6× faster than mlx_lm. With two to four sequences the lead
disappears: oMLX scales 1.67× from one sequence to two against Tur's 1.27×, where
recurrent-state traffic rather than weight bandwidth bounds Tur's decode.
