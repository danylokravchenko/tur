# P13: reusable execution plans, overlap and batch invariance

2026-10-24, Apple M1 (8 GPU cores, 16 GB), macOS 26.6.2, rustc 1.99.0, pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`. Base commit `5e7efcf` (P12), `pre-release`
profile. Only the M1 and the 0.8B checkpoint are measured. P9 has not run, so this package
measures against the base commit only.

## Outcome

**No speed gain is available on this M1, and batch invariance is rejected on cost.**
Host encoding is 13–25% of each pass, but it runs while the GPU executes earlier command
buffers, and the device idles only 28–38 µs between passes. Neither buffer reuse nor
deeper submission can shorten a pass by more than about 0.3%. A batch-invariant execution
mode was built and validated: bit-identical across batch size, order and chunking, and
more accurate on prefill. It costs 0.52–0.60× prefill throughput and 0.59–0.87×
single-sequence decode, so it is not kept. The implementation is preserved as
[a patch](baselines/2026-10-24-m1-p13/rejected-batch-invariance.patch) against `5e7efcf`.

| Item | Result | Status |
| --- | --- | --- |
| 1. Host allocations and buffer reuse | Buffer creation is ~19% of encoding, but encoding is off the critical path | **Not pursued**; instrumentation kept |
| 2. Reusable planning, command reuse | Committing work to the GPU earlier or later changes nothing beyond noise | **Not pursued** |
| 3. Next-pass overlap, submission depth 2 | Device idle 28–38 µs per pass (0.2–0.3%); bound ≤ 1.003× | **Rejected by bound** |
| 4. Batch-invariant execution | 0 of 15 comparisons differ (default: 11 of 15); prefill 0.52–0.60×, decode ×1 0.59–0.87× | **Built, rejected on cost** |

Kept: the model benchmark now reports encode time per pass, engine submit time,
host time after device completion and device idle time, and can select cases
(`TUR_BENCH_ONLY`) and the engine case's length (`TUR_BENCH_ENGINE_TOKENS`).

## Where pass time goes

### Encoding

Host time from a synchronized device until the forward call returns, against the whole
pass to completion, Q8 K/V ([encode](baselines/2026-10-24-m1-p13/encode.json)):

| Decode | Q8_0 pass / encode | Q4K pass / encode |
| --- | ---: | ---: |
| ×1, 128 context | 17.66 / 2.55 ms | 10.59 / 1.40 ms |
| ×4, 128 context | 22.69 / 3.73 ms | 20.15 / 3.33 ms |
| ×16, 128 context | 45.93 / 6.63 ms | 45.62 / 6.37 ms |
| ×1, 8K context | 19.67 / 2.78 ms | 13.30 / 2.90 ms |
| ×4, 8K context | 29.03 / 4.65 ms | 26.84 / 4.81 ms |

A 10 s `sample` profile of the engine decode loop (Q8_0, Q8 K/V, one sequence) puts
`HybridEngine::submit` at 13.6% of main-thread samples; the rest is the fence wait. Within
submit, by self time ([summary](baselines/2026-10-24-m1-p13/encode-profile-summary.txt)):

| Share of submit | Where |
| ---: | --- |
| 24% | Metal driver dispatch and argument binding |
| 19% | `MTLBuffer` creation: kernel outputs, Candle's quantized GEMV outputs, per-pass index uploads |
| 15% | malloc/free (Candle's hazard sets, the driver's own C++ allocations) |
| 13% | Candle's per-binding hazard hash maps |
| 5% | Objective-C runtime |
| 4% | Debug labels, command encoder lock and fences |

Buffers are re-created every pass because Candle's fence (`Device::synchronize`) releases
every pooled buffer no live tensor holds, and every host upload creates a fresh buffer.
Resident device memory stays flat after warmup (the existing allocation-bound test
asserts this), so this is churn, not growth. The profiler ran the `tur-engine` benchmark
with the system allocator; the `tur` binaries use jemalloc, which affects only the Rust
part of the malloc share.

### Commit granularity

Candle commits a command buffer every 50 dispatches; the GPU cannot start a pass before
its first commit. `CANDLE_METAL_COMPUTE_PER_BUFFER` sets that interval. Decode step times,
Q8 K/V ([8](baselines/2026-10-24-m1-p13/commit-every-8.json),
[16](baselines/2026-10-24-m1-p13/commit-every-16.json),
[25](baselines/2026-10-24-m1-p13/commit-every-25.json),
[50](baselines/2026-10-24-m1-p13/commit-every-50.json),
[100](baselines/2026-10-24-m1-p13/commit-every-100.json)):

| Case | 8 | 16 | 25 | 50 (two runs) | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 ×1, 128 | 17.75 | 18.04 | 17.49 | 17.04 / 17.71 | 17.45 |
| Q8_0 ×16, 128 | 45.31 | 45.14 | 45.24 | 45.43 / 45.22 | 45.49 |
| Q8_0 ×4, 8K | 29.05 | 28.92 | 28.82 | 29.16 / 29.35 | 29.86 |
| Q8_0 engine | 18.23 | 17.86 | 17.65 | 17.93 / 17.92 | 18.56 |
| Q4K ×1, 128 | 11.33 | 10.87 | 10.67 | 10.51 / 10.48 | 10.54 |
| Q4K ×4, 8K | 27.08 | 26.44 | 26.59 | 27.37 / 27.29 | 27.54 |
| Q4K engine | 11.61 | 11.29 | 11.03 | 11.07 / 11.16 | 11.46 |

The two default runs differ by up to 4%, and no other setting beats the default beyond that
spread; committing every 8 dispatches is 5–8% slower for single-sequence Q4K. The GPU
therefore does not wait for encoding: each dispatch takes longer on the GPU (about 35–60 µs
in decode) than to encode (about 8–10 µs). Removing encoding cost would free CPU time,
not shorten passes. The `commit-every-50.json` file holds the second default run.

### Between passes

Engine decode loop, one sequence, 200 tokens; idle is from device completion to the next
submission ([engine split](baselines/2026-10-24-m1-p13/engine-split.json)):

| Weights | Token | Submit | After completion | Device idle |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 17.73 ms | 2.98 ms | 36 µs | 38 µs |
| Q4K | 11.10 ms | 2.39 ms | 27 µs | 28 µs |

P7 measured 0.06–0.33 ms of idle per pass through the serving pipeline, at most 0.6% of a
pass at concurrency 16. With the plan's bound `1 / ((1 - f) + f / s)`, removing all of it
gives at most 1.003× here and 1.006× in serving. A second in-flight pass would also need
device-resident token handoff and per-submission completion events, which Candle does not
expose; it was not built.

## Batch invariance (rejected)

Default kernels choose by pass shape: the projection kernel by row count (Candle's
one-row GEMV, the custom GEMV at 2–3 rows, the matrix-unit kernel at 4–16, Candle's
F16-staged tiled GEMM above 16), attention split-K by the pass's tokens and longest
context, tiled attention by tile count, and the recurrent step kernel for single-token
sequences. A sequence's logits therefore depend on its batch, order and chunking.

The patch adds `Execution::Invariant`, which fixes every choice to the sequence's own
shape: matrix-unit projections for every row count (16-row blocks beyond), 512-key
attention splits, per-query attention only, and the recurrent scan for every token. It
rejects dense and expert models, whose kernels it cannot keep invariant. It also adds
`ModelFactory::execution` and a `--batch-invariant` CLI and server flag.

**Correctness** (small fixture, Q8 K/V, Q8_0 and Q4K weights). Decodes beside contexts
needing one to three attention splits, an empty sequence's prefill chunk, four batch
orders, and one prompt in four chunkings including single-token chunks:

- Invariant: 0 of 15 comparisons differ in any bit of logits or committed state.
- Default kernels: 11 of 15 differ, for both formats.
- Prefill against the reference fed one token at a time (F32 one-row GEMV): invariant
  8.1e-7 (Q8_0) and 1.1e-5 (Q4K), default 2.4e-3 and 1.2e-2.

The patch also contains an end-to-end pipeline test (sampled requests alone, batched with
prefix snapshots, under preemption, and speculatively verified); it was written but not
run before removal.

**Cost.** Interleaved runs on the 0.8B checkpoint, best of each side, invariant over
default; F16 and Q8 K/V agree and are merged
([before 1](baselines/2026-10-24-m1-p13/rejected-invariant-before-1.json),
[after 1](baselines/2026-10-24-m1-p13/rejected-invariant-after-1.json),
[before 2](baselines/2026-10-24-m1-p13/rejected-invariant-before-2.json)):

| Workload | Q8_0 | Q4K |
| --- | ---: | ---: |
| Prefill 128 / 2,048 tokens | 0.52–0.57× | 0.57–0.60× |
| Decode ×1, 128–8K context | 0.84–0.87× | 0.59–0.70× |
| Decode ×2–16 | 0.84–0.95× | 0.87–1.01× |
| Engine loop, one sequence | 0.83–0.85× | 0.63× |

Prefill re-reads every weight per 16-row block instead of once in a tiled GEMM, and loses
tiled attention. A single row runs in an 8-row matrix-unit tile instead of a one-row GEMV,
which costs most against Candle's Q4K GEMV.

This is not the existing reproducibility mode (`PipelineConfig::reproducible`). That mode
serializes requests to remove batching altogether; invariance would keep batching but
change kernels. Nothing in Tur needs batch-independent tokens yet, so neither cost is
justified by default. To revisit: apply the patch on `5e7efcf` and run its tests. A
cheaper variant would need a single kernel family that is fast at one row and at prefill
widths; none exists in Tur today.

## Validation

- The invariant-mode tests above passed with the patch applied.
- After removal, the tree differs from `5e7efcf` only in `crates/tur-engine/benches/qwen3_5.rs`.

## Remaining limits

- One model, one device; one profile run and one or two benchmark runs per configuration,
  without confidence intervals.
- Hosts with slower CPUs relative to their GPU (or larger, faster GPUs) may expose
  encoding on the critical path; re-run the commit-granularity sweep there before
  reusing these conclusions.
- Buffer reuse would still cut CPU use per pass by an estimated 15–20% of encoding, which
  matters only if the CPU becomes contended (for example by tokenization under heavy
  intake).

## Reproduction

```bash
export TUR_QWEN35_0_8B=<snapshot>
TUR_BENCH_ONLY=decode,engine TUR_BENCH_FORMATS=q8_0,q4k TUR_BENCH_KV=q8 TUR_BENCH_CONTEXTS=8192 \
  cargo bench --offline -p tur-engine --bench qwen3_5 --profile pre-release            # encode
CANDLE_METAL_COMPUTE_PER_BUFFER=<8|16|25|50|100> <same command>                          # commit sweep
TUR_BENCH_ONLY=engine TUR_BENCH_FORMATS=q8_0,q4k TUR_BENCH_KV=q8 TUR_BENCH_ENGINE_TOKENS=200 \
  cargo bench --offline -p tur-engine --bench qwen3_5 --profile pre-release            # engine split
# profile: run the engine case with TUR_BENCH_ENGINE_TOKENS=3000, then `sample <pid> 10`
git apply docs/baselines/2026-10-24-m1-p13/rejected-batch-invariance.patch             # on 5e7efcf
cargo nextest run --offline --cargo-profile pre-release -p tur-engine --test qwen3_5_fused invariant
TUR_BENCH_EXECUTION=<fused|invariant> TUR_BENCH_FORMATS=q8_0,q4k TUR_BENCH_KV=f16,q8 \
  TUR_BENCH_PREFILL=128,2048 TUR_BENCH_CONTEXTS=2048,8192 \
  cargo bench --offline -p tur-engine --bench qwen3_5 --profile pre-release
```
