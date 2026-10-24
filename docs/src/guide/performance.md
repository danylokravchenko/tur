# Measured performance

These are the final measurements recorded during development, taken from the benchmark
artifacts under `docs/baselines/` at commit `99b073e`. They are historical: the code has
been refactored since, without intended performance changes. Re-measure with
[`just bench`](../development/benchmarks.md) before relying on a number.

**Setup for every table:**

- Hardware: Apple M1 with 16 GB.
- Model: Qwen3.5-0.8B (`2fc06364715b967f1860aea9cf38778875588b17`).
- Defaults unless noted: F16 K/V with 16-token pages, F32 recurrent state, 256-token
  prefill chunks.
- Timing: every timed region ends at device completion.
- Spreads: ranges in brackets are min–max over repeated runs.

## Single model pass

Raw model speed, without scheduling or HTTP (`model` benchmark, synthetic tokens).
Sources: `2026-10-24-m1-p11/speed-{1,2}.json` (two-run mean),
`2026-10-24-m1-p10/model-after-*.json` and `2026-10-24-m1-p10/prefill-after-*.json`.

**Decode throughput** (tokens/s across all sequences, 128-token context):

| Weights | 1 seq | 2 seq | 4 seq | 8 seq | 16 seq |
| --- | --- | --- | --- | --- | --- |
| Q8_0 (load-time) | 57 | 108 | 180 | 307 | 357 |
| Q8_0 artifact | 57 | 109 | 181 | 310 | 354 |
| Q4K (load-time) | 91 | 112 | 200 | 334 | 353 |
| Q4K GPTQ artifact | 85 | 111 | 197 | 339 | 350 |
| Mixed 5.5-bit GPTQ artifact | 79 | 111 | 197 | 339 | 356 |

At one sequence, decode is bound by weight bandwidth, so Q4K is about 1.6× faster than
Q8_0. From two sequences onward, batched kernels and recurrent-state traffic dominate,
and the formats converge.

**Decode at longer context** (tokens/s):

| Weights / K/V | 1 seq @ 2K | 4 seq @ 2K | 1 seq @ 8K | 4 seq @ 8K |
| --- | --- | --- | --- | --- |
| Q8_0 / F16 | 55 | 165 | 50 | 133 |
| Q8_0 / Q8 | 55 | 170 | 51–52 | 135–140 |
| Q4K / F16 | 89 | 184 | 75 | 149 |
| Q4K / Q8 | — | — | 79 | 153 |

**Prefill** (prompt tokens/s, one sequence):

| Weights | 2K tokens | 8K tokens |
| --- | --- | --- |
| Q8_0 | 1,263–1,311 | 989 |
| Q4K | 1,188–1,204 | — |
| Mixed 5.5-bit | 1,233 | — |

A 2K-token prompt therefore takes about 1.6 s; an 8K-token prompt about 8.3 s.

## Serving

Closed-loop bursts through the scheduled pipeline: N requests submitted at once, each
with a 128-token prompt and 64 output tokens; Q8_0 weights. "Output tok/s" is aggregate
over the burst. TTFT counts from submission.

**Sampled** (T=0.7, top-k 20, top-p 0.95; device sampling). Source:
`2026-10-24-m1-p15/final-serving.json`, which agrees with the three runs of
`2026-10-23-m1-p7-gaps/serving-sampled-*.json`.

| Concurrent | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 48.9 | 0.15 s | 0.15 s | 17.8 ms | 18.8 ms | 19.9 ms |
| 2 | 78.4 [76.7–80.8] | 0.30 s | 0.30 s | 20.4 ms | 22.6 ms | 27.3 ms |
| 4 | 118 | 0.47 s | 0.50 s | 23.0 ms | 23.8 ms | 44.7 ms |
| 8 | 184 | 0.88 s | 0.93 s | 27.2 ms | 27.9 ms | 48.3 ms |
| 16 | 200 | 1.69 s | 1.78 s | 46.3 ms | 48.0 ms | 178 ms |

The concurrency-2 row comes from the P7 runs only.

**Greedy** (T=0). Source: three runs of `2026-10-23-m1-p7-gaps/serving-greedy-*.json`.

| Concurrent | Output tok/s | TTFT p50 | ITL p50 | ITL p99 |
| --- | --- | --- | --- | --- |
| 1 | 46.2 [44.2–47.2] | 0.17 s | 18.8 ms | 21.5 ms |
| 2 | 78.7 [77.5–80.4] | 0.30 s | 20.1 ms | 33.6 ms |
| 4 | 114 [109–116] | 0.53 s | 24.6 ms | 34.6 ms |
| 8 | 168 [160–174] | 1.02 s | 29.5 ms | 66.9 ms |
| 16 | 186 [180–190] | 1.88 s | 50.0 ms | 215 ms |

Sampled serving matches greedy within noise, because sampling runs on the GPU. The
sampled P15 run is newer and slightly faster, after the P13–P15 changes.

**Mixed prompts.** Sixteen requests at once, half with 128-token and half with 2K-token
prompts:

| Run | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 |
| --- | --- | --- | --- | --- | --- |
| P15 sampled | 55.1 | 8.25 s | 14.1 s | 218 ms | 288 ms |
| P7 greedy (3 runs) | 47.2 [45.9–47.9] | 9.57 s | 16.7 s | 258 ms | 345 ms |

The eight 2K-token prompts are 16K prefill tokens, about 13 s of prefill on this GPU,
which the scheduler interleaves with the short requests' decodes.

## Open-loop arrivals

48 requests with 512-token prompts and 64 output tokens arrive as a Poisson process,
sampled as above. Source: `2026-10-23-m1-p7-gaps/arrivals.json`.

| Rate | Intake | Output tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p99 | Device idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2/s | serial | 83.0 | 4.62 s | 10.7 s | 61 ms | 340 ms | 9.4% |
| 2/s | overlapped | 85.9 | 3.96 s | 9.32 s | 60 ms | 290 ms | 4.8% |
| 4/s | serial | 89.2 | 10.5 s | 19.1 s | 53 ms | 379 ms | 9.6% |
| 4/s | overlapped | 93.2 | 9.13 s | 17.7 s | 53 ms | 346 ms | 5.2% |

Both rates exceed what one M1 sustains with 512-token prompts: about 1,000 prompt
tokens/s of prefill alone is needed at 2/s. Queues therefore grow, and TTFT climbs over
the run. Overlapped intake is the default in `tur serve`.

## Grammar-constrained output

JSON-like grammar, every token masked; greedy. Source:
`2026-10-23-m1-p7-gaps/grammar-device{0,1}.json`.

| Concurrent | Host masks: tok/s, ITL p50 | Device masks (default): tok/s, ITL p50 |
| --- | --- | --- |
| 1 | 38.0, 23.3 ms | 46.1, 18.4 ms |
| 4 | 90.4, 33.0 ms | 115.7, 24.1 ms |
| 8 | 127.6, 44.9 ms | 172.0, 29.0 ms |
| 16 | 139.9, 81.6 ms | 192.2, 50.2 ms |

With device masks, constrained generation runs at the speed of unconstrained generation.

## Prefix reuse

The `reuse` workloads, three runs each. Sources: `2026-10-24-m1-p15/final-reuse-{1,2,3}.json`
against `baseline-reuse*.json`. The baseline is reuse with LRU eviction and no producer
waiting.

| Workload | Metric | Baseline | Final |
| --- | --- | --- | --- |
| 8 users, cold 1.5K-token system prompt at once | TTFT p50 | 9.58 s | 1.72 s |
| | Output tok/s | 23.6 | 89.7 |
| 8 users, cold 300-token system prompt | TTFT p50 | 3.32 s | 2.10 s |
| 4 agents × 5 steps on a 1K-token manual | Prompt tokens cached | 72% | 83% |
| | Wall time | 12.3 s | 9.9 s |
| | First-step TTFT p50 | 3.21 s | 1.07 s |
| Hot prefix changing under cold churn | Hot tokens restored | 0 | 6,144 |
| | Wall time | 35.3 s | 32.7 s |
| 16 distinct 512-token prompts (no reuse) | TTFT p50, tok/s | 3.86 s, 56.9 | 3.87 s, 56.8 |
| Warm vs cold contention (16 requests, 4 slots) | TTFT p50, tok/s | 4.95 s, 44.7 | 4.90 s, 44.8 |

Eight users behind an already *warm* 1.5K-token system prompt get their first token in
0.56 s p50 (`2026-10-24-m1-p15/final-prefix.json`). Earlier P7 measurements put the
no-reuse cost of the same burst at 11.0 s.

## Speculative decoding

80 output tokens per request; three repeats each; output tok/s (aggregate). Sources:
`2026-10-24-m1-p14/final-{greedy,sampled}-{ordinary-lookup,mtp}.json`.

**Greedy:**

| Prompt | Seqs | Ordinary | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 52.1 | 53.5 | 70.9–72.0 |
| | 4 | 155.6 | 155.1 | 175.4–187.2 |
| | 8 | 248.2 | 248.0 | 235.9–237.4 |
| Code | 1 | 53.6 | 53.0 | 71.0–72.3 |
| | 4 | 155.6 | 155.2 | 192.0–192.2 |
| | 8 | 245.9 | 245.7 | 238.1–239.7 |
| Repetitive | 1 | 51.5 | **149.4** | 144.1–144.7 |
| | 4 | 143.5 | **269.9** | 260.2–262.1 |
| | 8 | 217.5 | **294.9** | 284.4–285.0 |

**Sampled** (T=0.7, top-k 20, top-p 0.95):

| Prompt | Seqs | Ordinary | Lookup (4 drafts) | Lookup + MTP |
| --- | --- | --- | --- | --- |
| Novel prose | 1 | 52.8 | 52.9 | 64.7–65.2 |
| | 4 | 155.1 | 153.6 | 173.1–177.3 |
| | 8 | 246.2 | 246.5 | 237.5–239.8 |
| Code | 1 | 52.7 | 52.8 | 72.4–72.7 |
| | 4 | 154.1 | 152.3 | 188.9–192.7 |
| | 8 | 244.4 | 244.2 | 235.7–237.4 |
| Repetitive | 1 | 51.5 | 147.8 | 142.9–143.2 |
| | 4 | 133.9 | 145.8 | 176.3–178.5 |
| | 8 | 212.1 | 217.6 | 219.0–219.6 |

How to read these:

- **Repetitive text:** prompt lookup gives up to 2.9×.
- **Novel text at one to four sequences:** the MTP head gives 1.12–1.38×, about 1.35× at
  one sequence and 1.12–1.25× at four.
- **Eight sequences:** MTP costs 3–5%, so it gates itself off.
- **Exactness:** outputs are identical to ordinary decoding in every configuration.

## Loading

Fresh process, page cache warm, through the last upload. Source: `2026-10-24-m1-p11/load.jsonl`.

| Weights | Load time | Resident weights | Peak RSS |
| --- | --- | --- | --- |
| Q4K, converted at load | 8.9–9.1 s | 933 MB | 1.46 GB |
| Q8_0, converted at load | 1.6 s | 1,309 MB | 2.14 GB |
| Q4K GPTQ artifact | 0.24 s | 425 MB | 0.71 GB |
| Mixed 5.5-bit GPTQ artifact | 0.20 s | 521 MB | 0.80 GB |
| Q8_0 artifact | 0.40 s | 801 MB | 1.20 GB |

## Quality of weight formats

Teacher-forced over 8,192 held-out tokens per domain (7,206 for retrieval), against F32
weights. KL is mean nats per token, lower is better; top-1 is agreement with F32's
argmax. Source: `2026-10-24-m1-p11/quality-*.json`.

| Weights | Prose KL / top-1 | Code KL / top-1 | Retrieval KL / top-1 | Tools KL / top-1 |
| --- | --- | --- | --- | --- |
| Q8_0 | 0.0008 / 98.4% | 0.0004 / 99.2% | 0.0003 / 98.3% | 0.0014 / 99.4% |
| Mixed 5.5-bit GPTQ | 0.0346 / 88.6% | 0.0182 / 94.2% | 0.0144 / 90.9% | 0.0663 / 96.0% |
| Q4K GPTQ | 0.0704 / 84.4% | 0.0343 / 92.1% | 0.0296 / 89.6% | 0.1003 / 94.5% |
| Q4K importance-weighted | 0.0941 / 83.3% | 0.0510 / 90.8% | 0.0368 / 88.9% | 0.0986 / 94.7% |
| Q4K RTN | 0.1413 / 79.7% | 0.0832 / 88.4% | 0.0577 / 87.0% | 0.1142 / 93.9% |

K/V formats, from the same evaluation method:

- Q8 K/V stays within 1e-4 nats of F32 K/V, with 99.2% top-1 agreement.
- F16 K/V is indistinguishable from F32 (KL 7.5e-8).
