# Benchmarks

Every benchmark and measurement tool is a case of one bench binary:

```sh
just bench <case> [args] > out.json
# = cargo bench --offline -p tur --bench perf -- <case> [args]
```

Each case prints one JSON document to stdout and progress to stderr. Timed regions always
end at device completion; timing only the enqueueing of GPU work would measure nothing.
Most cases need a real checkpoint in `TUR_QWEN35_0_8B`.

| Case | Measures | Main knobs |
| --- | --- | --- |
| `kernels` | Each custom kernel against Candle's path at Qwen3.5 shapes | `TUR_BENCH_ONLY=delta\|linear\|paged\|tiled\|sample\|moe`, `TUR_BENCH_LINEAR`, `TUR_BENCH_MOE`, `TUR_BENCH_SHAPE` |
| `model` | Model-level prefill and batched decode at several contexts, per weight and K/V format; engine overhead | `TUR_BENCH_FORMATS` (formats or artifact dirs), `TUR_BENCH_KV`, `TUR_BENCH_PAGES`, `TUR_BENCH_PREFILL`, `TUR_BENCH_CONTEXTS`, `TUR_BENCH_REFERENCE=1`, `TUR_BENCH_ONLY=prefill,decode,engine` |
| `serving` | Throughput, TTFT and inter-token latency through the pipeline | `TUR_BENCH_WORKLOADS=decode,mixed,grammar,arrivals,prefix,reuse,speculation`, `TUR_BENCH_CONCURRENCY`, `TUR_BENCH_OUTPUT`, `TUR_BENCH_SAMPLING=0.7,20,0.95` |
| `prefix` | Host cost of prefix keys, lookups and eviction (CPU, no checkpoint) | none |
| `kv_quality` | Teacher-forced KL, top-1 and NLL of K/V, state and weight variants against an F32 reference | `TUR_EVAL_TEXT`, `TUR_EVAL_KV`, `TUR_EVAL_STATE`, `TUR_EVAL_WEIGHTS`, `TUR_EVAL_TOKENS`, `TUR_EVAL_GENERATE` |
| `tensorops` | Metal 4 TensorOps capability and F32 GEMM against Candle | none |
| `load_weights <dir> <variant>` | Load time and resident bytes of `f32`, `q8_0`, `q4k` or `prepacked` | wrap in `/usr/bin/time -l` for peak memory |

Each case's module doc in `crates/tur/benches/perf/` lists every variable. The serving
workloads:

- `decode`: short prompts at each concurrency.
- `mixed`: 128- and 2K-token prompts together.
- `grammar`: every token masked.
- `arrivals`: open-loop Poisson arrivals, serial versus overlapped intake.
- `prefix`: a shared system prompt at several snapshot spacings.
- `reuse`: cold shared prompts, contention, agent loops, churn and no reuse.
- `speculation`: prompt lookup and MTP over repetitive, novel and code prompts.

## Methodology

- **Same machine, same everything.** Compare configurations on one Mac with the same
  checkpoint, weights, prompts, output lengths, sampling and cache state. Record the
  chip, macOS, power mode and git revision.
- **Repeat.** Use at least three runs for serving results and report the spread. A
  difference within run-to-run noise is no difference.
- **Quality before speed.** For any format change, run `kv_quality` on held-out prose,
  code, tool/JSON and retrieval text first. The budget used so far is mean KL ≤ 1e-3 and
  top-1 agreement ≥ 98% per domain.
- **Report both outcomes.** Record the winner's numbers in the relevant chapter, and the
  loser in [Rejected experiments](../design/rejected.md).
