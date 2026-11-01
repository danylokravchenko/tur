# Tuning

The defaults suit one model serving interactive users. This page explains what each setting trades, gives recipes for common situations, and shows how to measure the result. [Best practice](best-practice.md) has a starting configuration for each model, and the [Architecture](../architecture/overview.md) chapters explain the mechanisms behind them.

## Keep settings in a file

Put the settings you reuse in [`tur.toml`](configuration.md) instead of repeating flags:

```toml
[engine]
prefill_chunk_size = 512

[server]
max_batch_size = 2
request_timeout_secs = 120
```

`tur serve --max-batch-size 4` then overrides just that one setting for one start.

## Read the memory plan first

Every start logs a plan like this:

```text
Memory plan resident_bytes=3017015772 weights=1309441280 state_pool=181469184 scratch_peak=251789312
```

The total is the weights, plus the state pool (K/V pages, recurrent state, prefix snapshots and speculative drafts), plus kernel scratch space and in-flight outputs, plus a reserve of a tenth of the GPU's working set. Tur reserves all of it before reading any weight and refuses to start if it doesn't fit, so macOS never ends up swapping model memory. If a start fails for lack of memory, shrink one of these:

| Part | Setting | Effect |
| --- | --- | --- |
| Weights | `--weight-format q4k`, or a [prepacked](quantization.md) artifact | Q4K takes about 70% of Q8_0's memory at load time, and about a third as an artifact on models with shared embeddings |
| K/V pages | `--kv-format q8` | Half of F16's K/V memory, almost no quality change, slower prefill |
| Recurrent state | `--recurrent-state f16` | Half the recurrent memory and slightly faster batched decode; error grows on long outputs, and speculation turns off |
| State pool | `--state-memory-mb` | By default, room for an 8K-token context per concurrent request |
| Snapshots | `--prefix-snapshots N` | By default, one per running request, using at most 1/8 of the pool |
| Concurrency | `--max-batch-size` | Each running request holds recurrent state and its own K/V pages |

The same numbers stay available while the server runs, at [`/diagnostics`](http-api.md#get-diagnostics) and `/metrics`.

## Set the context you need

`--context-tokens` says how long one conversation may get, and Tur plans around it:

```sh
tur serve --context-tokens 32768 --max-batch-size 4 --kv-format q8
```

The state pool is sized so that all four requests can reach 32K tokens at once, and kernel scratch only for that length. If the plan doesn't fit the GPU's working set, the start fails with the largest context that would fit. A request beyond the limit gets HTTP 400 with `context_length_exceeded`, and one that leaves out `max_tokens` is cut off where the context ends.

Without the flag, Tur sizes the pool for 8K tokens per request and lets one request use as much of the pool as the model allows. It shrinks the pool to fit the device, and logs a warning when that leaves room for fewer full contexts than concurrent requests. With both `--context-tokens` and `--state-memory-mb`, the pool you set must hold one full context; requests beyond what it holds wait or recompute rather than fail.

At startup Tur also compares the plan with what macOS can spare without compressing or swapping, and warns if the plan is larger. While serving, it reads spare memory every 10 seconds and warns when spare memory falls below 1 GiB or macOS starts writing to swap.

## What each setting trades

| Setting | Raise it to | Lower it to | Default |
| --- | --- | --- | --- |
| `--max-batch-size` | Serve more users at once, with more total tokens/s | Cut per-request latency and memory | 8 |
| `--prefill-chunk-size` | Finish long prompts sooner | Interleave prompts more finely with streaming replies | 256 |
| `--max-batch-tokens` | Allow bigger prefill passes | Keep passes short | 2048 or more |
| `--max-waiting` | Absorb bursts | Turn overload into fast 429s | 256 |
| `--state-memory-mb` | Longer contexts, more snapshots | Free memory for weights or other apps | derived |
| `--prefix-snapshots` | Keep more shared prompts warm | Free pool memory | one per running request |
| `--speculative-draft` | More tokens per pass on repetitive output | Steadier token timing | 0 |

You don't need to balance prompts against replies by hand. The scheduler resizes the prompt share of every pass to keep each pass near 60 ms, or twice the cost of the replies alone if that's longer. These settings only set the limits it works within (see [Scheduling](../architecture/scheduling.md)).

## Recipes

### A local assistant: one user, lowest latency

```sh
tur serve --max-batch-size 2 --prefill-chunk-size 512
```

A small batch keeps every pass short, and bigger prompt chunks finish a long first prompt in fewer passes. Add `--mtp` to generate new text 1.1 to 1.4× faster with one to four users. Output is identical, but tokens arrive in small bursts, and prefix reuse turns off.

### A chat service: many users at once

```sh
tur serve --max-batch-size 16 --max-waiting 512 --request-timeout-secs 120 --kv-format q8
```

A bigger batch raises total throughput, though each user's tokens arrive a little slower as it grows. Q8 K/V halves the memory for keys and values, which leaves room for the extra conversations. The timeout fails stuck requests with HTTP 504, and the queue limit turns overload into quick 429s instead of endless waits. Leave prefix reuse on, since every user shares the system prompt.

### Agents and shared system prompts

```sh
tur serve --max-batch-size 8 --prefix-snapshots 16
```

Agent loops resend a growing conversation each step, and prefix reuse skips the part already computed; in one agent benchmark, 83% of prompt tokens came from the cache. When several users hit the same new prompt at once, one computes it and the others wait for it rather than each computing their own copy. Leave `--mtp` off, because it disables prefix reuse.

If most requests share a cached prompt and their first token matters more than total throughput, let them start ahead of older uncached requests:

```toml
[engine]
prefix_restorable_first = true
```

In the benchmark, cached requests stuck behind uncached ones got their first token 5× sooner, while uncached requests waited 15% longer and throughput fell 10%. An uncached request is never passed over for more than `max_queue_wait_ms` (2 s by default).

Use the `cache_salt` request field to keep tenants' caches apart, and check `usage.prompt_tokens_details.cached_tokens` in responses to see what was reused.

### Code editing, extraction and structured output

```sh
tur serve --speculative-draft 4
```

Prompt lookup drafts tokens by copying from the request's own text, which works well when the output quotes code, repeats keys or echoes the input. On repetitive output a single request ran about 3× faster. Drafting watches its own hit rate and cost, and stops when it no longer pays, for example when many requests are running.

### Long contexts on a 16 GB Mac

```sh
tur serve --max-batch-size 2 --kv-format q8 --state-memory-mb 3072
```

K/V memory grows with context while recurrent state doesn't, so long contexts are mostly a K/V cost and Q8 K/V roughly halves it. An explicit pool size gives the few running requests more room than the default 8K tokens each. A request that could never fit fails immediately with an out-of-memory error.

### The smallest footprint

```sh
tur quantize --model ~/models/m --revision m1 --output ~/models/m-q4k --calibration calib.txt
tur serve --model ~/models/m-q4k --revision m1-q4k --kv-format q8 --recurrent-state f16 --max-batch-size 2
```

This shrinks every part of the plan. F16 recurrent state costs some accuracy on long outputs and disables speculation, so [check quality on your own text](choosing-quantization.md#check-quality-on-your-own-text) first.

### Reproducible output

Batching can change logits in their last bits, because kernels add numbers in a different order at different batch shapes. The same seed with different neighbours can therefore pick a different token now and then. For bit-for-bit reproducible output, use the library's `PipelineConfig::reproducible`, which runs one request at a time in fixed chunks, without prefix reuse or speculation. Making concurrent batches bit-exact was tried and cost up to half the speed (see [Rejected experiments](../design/rejected.md)).

## Measure the result

For a single request, the HTTP `usage` block and the library's `GenerationStats` report prompt, cached and generated tokens, and the CLI logs elapsed time and finish reason.

Under load, the serving benchmark reports throughput, time to first token and time between tokens for your settings:

```sh
TUR_QWEN35_0_8B=$HOME/models/qwen3.5-0.8b TUR_BENCH_WORKLOADS=decode,mixed \
TUR_BENCH_CONCURRENCY=1,4,8,16 just bench serving > serving.json
```

Before adopting a different K/V, state or weight format, run `kv_quality` (see [Benchmarks](../development/benchmarks.md)).

Change one setting at a time, repeat each run at least three times, and compare on the same Mac in the same power mode.
