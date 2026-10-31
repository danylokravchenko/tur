# Benchmarks

Every benchmark and measurement tool is a case of one bench binary:

```sh
just bench <case> [args] > out.json
# same as: cargo bench --offline -p tur --bench perf -- <case> [args]
```

`experts <dir> <cache MiB> <text> [reference dir]` measures the expert cache: its reads, hit rate and wait, system-wide SSD writes and swap-outs, and KL against a reference checkpoint.

Each case prints one JSON document to stdout and progress to stderr. Timed regions always end when the GPU finishes; timing only the moment work is queued would measure nothing. `model` and `serving` take a local Qwen or Gemma checkpoint in `TUR_BENCH_MODEL`, and the Qwen quality studies use `TUR_QWEN35_0_8B`.

| Case | Measures | Main knobs |
| --- | --- | --- |
| `kernels` | Each custom kernel against Candle's path at Qwen3.5 shapes | `TUR_BENCH_ONLY=delta\|linear\|paged\|tiled\|sample\|moe`, `TUR_BENCH_LINEAR`, `TUR_BENCH_MOE`, `TUR_BENCH_SHAPE` |
| `model` | Model-level prefill and batched decode at several contexts, per weight and K/V format; engine overhead | `TUR_BENCH_FORMATS` (formats or artifact dirs), `TUR_BENCH_KV`, `TUR_BENCH_PAGES`, `TUR_BENCH_PREFILL`, `TUR_BENCH_CONTEXTS`, `TUR_BENCH_BATCHES`, `TUR_BENCH_CONTEXT_BATCHES`, `TUR_BENCH_REFERENCE=1`, `TUR_BENCH_ONLY=prefill,decode,engine` |
| `serving` | Throughput, time to first token and inter-token latency through the pipeline | `TUR_BENCH_WORKLOADS=decode,mixed,grammar,arrivals,prefix,reuse,speculation`, `TUR_BENCH_CONCURRENCY`, `TUR_BENCH_OUTPUT`, `TUR_BENCH_SAMPLING=0.7,20,0.95` |
| `prefix` | CPU cost of prefix keys, lookups and eviction (no checkpoint) | none |
| `kv_quality` | Teacher-forced KL, top-1 and NLL of K/V, state and weight variants against an F32 reference | `TUR_EVAL_TEXT`, `TUR_EVAL_KV`, `TUR_EVAL_STATE`, `TUR_EVAL_WEIGHTS`, `TUR_EVAL_TOKENS`, `TUR_EVAL_GENERATE` |
| `tensorops` | Metal 4 TensorOps availability and F32 GEMM against Candle | none |
| `whisper` | Whisper's encoder operations at their shapes; with checkpoints, whole-window encoding and transcription, and text passes during a concurrent Gemma clip | `TUR_AUDIO_SPEECH`, `TUR_WHISPER`, `TUR_GEMMA4_E2B` (all optional) |
| `load_weights <dir> <variant>` | Load time and memory of `f32`, `q8_0`, `q4k` or `prepacked` | wrap in `/usr/bin/time -l` for peak memory |

The module doc of each case in `crates/tur/benches/perf/` lists every variable. The serving workloads are:

- `decode`: short prompts at each concurrency.
- `mixed`: 128- and 2K-token prompts together.
- `grammar`: every token masked.
- `arrivals`: requests arriving at random, with serial or overlapped intake.
- `prefix`: a shared system prompt and four six-turn conversations at several snapshot spacings. `TUR_BENCH_PREFIX_ENTRIES` sets the snapshot capacity (16 by default; use 8 to match default serving).
- `reuse`: shared uncached prompts, contention, agent loops, churn and a no-reuse control. `TUR_BENCH_RESTORABLE_FIRST=1` lets cached prefixes jump the queue.
- `speculation`: prompt lookup and MTP over repetitive, novel and code prompts. `TUR_BENCH_DRAFT_MODEL=<dir>` drafts with a Gemma 4 assistant instead.

Gemma 4 E2B does about three times the work per token of the 0.8B Qwen, and its default sweep prefills 32K tokens in one case. `just bench-gemma` runs a short profile on the local checkpoint in `TUR_GEMMA4_E2B`: Q8_0 weights, F16 K/V, a 512-token prefill, decode at batches 1 and 4 at 128 tokens and one sequence at 4K (past the 512-token window), and 16 engine tokens. Any `TUR_BENCH_*` variable overrides its default:

```sh
TUR_GEMMA4_E2B=<snapshot dir> just bench-gemma > gemma4.json
TUR_BENCH_ONLY=decode TUR_BENCH_CONTEXTS=8192 just bench-gemma > gemma4-8k.json
```

Gemma's large per-layer embedding tables make some wide sweeps exceed a 16 GB Mac, so start with the short profile and size measurements to the machine.

## Methodology

- Compare configurations on one Mac with the same checkpoint, weights, prompts, output lengths, sampling and cache state. Record the chip, macOS version, power mode and git revision.
- Run serving results at least three times and report the spread. A difference within run-to-run noise is no difference.
- Check quality before speed. For any format change, run `kv_quality` on held-out prose, code, tool calls and JSON, and retrieval text first. The budget so far is a mean KL of at most 1e-3 and top-1 agreement of at least 98% per domain.
- Record both outcomes: the winner's numbers in the relevant chapter, and the loser in [Rejected experiments](../design/rejected.md).
