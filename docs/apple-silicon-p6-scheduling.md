# P6 packed execution and scheduling

Status (2026-10-22): every serving iteration is one validated packed pass. Every layer
now runs once over the whole batch, including the `DeltaNet` convolution and scan, which
dispatched once per sequence before. A new scheduler selects each pass:

- **Decode first:** running decodes go first, rotated fairly when they exceed the token
  budget.
- **Bounded prefill:** prefill chunks fill a budget sized from measured pass costs.
- **Pressure:** under memory pressure the newest running request is preempted and later
  recomputes its tokens, with its RNG and grammar state intact.

Admission adds a bounded waiting queue (requests and tokens), deadlines and a device
memory budget shared by every model on the device.

Measured on the P0 M1 with Qwen3.5-0.8B (Q8_0 weights, F16 K/V):

- **One pass per iteration:** a 16-request decode burst finishes 64 tokens per request
  in 64–65 passes.
- **Batching:** a batch-8 decode pass takes 28.6 ms, against 147 ms for eight batch-1
  passes (43 ms before the [batched decode](#batched-decode) kernels).
- **Mixed prompts:** with 128- and 2K-token prompts arriving together, decode
  inter-token latency p95 falls 3.7× (1.75 s → 0.47 s) and median time to first token
  falls 37%. Output throughput drops 7%, and the slowest first token is 14% later.
- **Decode-only bursts:** unchanged within noise.

[Batched decode](#batched-decode) followed and removed most of the per-sequence cost.
From batch 1 to 8 a serving pass now grows by about 1.4 ms per added sequence, against
6.4 ms before. At 8 concurrent requests the batched pass is now within ~1.4 ms of the
model pass, and greedy decode output rises 1.72× (96.6 → 166 tokens/s). The rest is
recurrent-state bandwidth; see [Limits](#limits).

## Packed execution

| Requirement | Implementation |
| --- | --- |
| Combine projections across variable-length requests | Already true since P4: embeddings, projections, norms, rotary, MLP and attention run once over the packed rows; attention reads per-token page tables |
| Recurrent layers | `Kernels::recurrent_spans` validates and uploads one span table per pass (first row, tokens, committed slot, written slots, per-token flag). `delta_conv` and `delta_scan` each dispatch once per layer over every sequence (threadgroups cover sequence × head × column block) and return the packed output. The per-sequence entry points were deleted |
| Track actual token boundaries | Spans must be consecutive and cover every packed row. No slot may be written twice, and no slot may be both read and written in a pass; forks may share a committed slot. Slot ids are checked against each pool at dispatch |
| Gather only required logits | Unchanged: the LM head runs on the requested rows only. Requests now ask for logits only when a chunk reaches their last known token |
| No padding | Packed rows are exactly the scheduled tokens; there is no padding anywhere on the main path |

Decode and prefill share one pass and one kernel sequence. No monolithic kernel was
needed.

## Scheduler

`tur_engine::backend::scheduler` is a pure policy over request descriptors. The pipeline
supplies a `fits` callback that accumulates pool demand, so the policy is unit-tested
without a model.

Each iteration:

1. **Expire deadlines.** Requests past their deadline fail with a deadline error and
   release their state.
2. **Decodes.** Running decodes, one token each, ordered by the ticket of their last
   chunk. This is strict rotation: a decode skipped for lack of budget runs first next
   time. While prefill that can run exists, `min_prefill_tokens` of the pass budget is
   held back for it.
3. **Prefill.** Chunks in arrival order, admitting waiting requests while fewer than
   `max_running` hold state. The budget is the largest power-of-two step whose
   estimated pass meets a target:
   - **Target:** `max(target_pass, 2 × decode-only estimate)`. Prefill may slow a pass
     to twice its decode cost, never below the configured target.
   - **Without decodes:** prefill takes the whole pass.
   - **Damping:** the budget moves at most a factor of two per pass and never drops
     below the minimum, so one noisy timing cannot starve prefill.
   - **Starvation guard:** a request older than `max_queue_wait` gets a full chunk.
4. **Memory.** A chunk that admits a request must leave one page of growth per running
   sequence free.
   - **Decode:** if a decode does not fit, the newest running request is preempted.
   - **Nothing fits:** the newest running request is preempted, or the request fails
     with `MemoryAllocation` if it is alone. Admission already rejects requests that
     cannot fit the empty pool.
   - **Progress:** the oldest request always makes progress.

The cost model keeps moving averages of measured pass durations in power-of-two buckets
of packed tokens × context. An unmeasured shape borrows the nearest measured bucket,
scaled linearly by tokens.

**Recompute preemption.** `HybridEngine::release_state` drops every layer's pages and
slot but keeps RNG draws, sampled history and the grammar parser. The pipeline holds
each request's prompt plus generated tokens. After preemption it re-prefills them all;
the final chunk's logits produce the *next* token, so no token is sampled twice and the
output stream is unchanged. Recurrent snapshots and speculative boundaries are charged
through the same per-pass `Demand` (pages plus written slots).

**Defaults** (`SchedulerConfig::new(max_running, max_chunk)`):

| Setting | Default |
| --- | --- |
| Waiting queue | 256 requests and 1M prompt tokens |
| Pass size | 2048 tokens, or one chunk beside every decode if larger |
| Minimum prefill per pass | 32 tokens |
| Target pass | 60 ms |
| Queue-age cap | 2 s |

The server exposes `--max-batch-tokens`, `--max-waiting`, `--target-pass-ms` and
`--request-timeout-secs`.

The first default throttled prefill against the 60 ms target even with no decodes
running, and the M1 cannot run a 16-way decode pass in 60 ms. That lengthened decode
bursts' time to first token by up to 1.7× (`decode_heavy` c=16: 2.2 s → 3.6 s p95). It
was replaced by the decode-relative target above; only the final measurements are kept.

## Shared device budget

`DeviceBudget` is a lock-free ledger of the device working set, shared across pipelines
by cloning. `InferencePipeline::with_budget` plans memory beside what other models have
leased (`PlanInputs::reserved`) and leases its planned resident bytes. A second model
that does not fit fails at load with `MemoryAllocation`; dropping a pipeline returns its
lease. `InferencePipeline::new` uses a budget of its own, and the server passes one
budget per device to its worker. The server still serves one resident model.
Multi-model arbitration is tested at the planning level, not with two loaded
checkpoints.

## Validation

| Exit criterion | Evidence |
| --- | --- |
| Mixed prompt/context lengths, batch reshuffling | `scheduling.rs`: twelve requests (prompts 1–37 tokens, limits 1–23, half sampled at T=0.9/top-p 0.95) on random weights, arriving all at once, staggered and in reverse, in 9-token passes. Every output equals the request run alone |
| Cancellation | The same workload under a 4-page pool, cancelled while waiting, mid-prefill and mid-decode. Survivors match their solo outputs and every page and slot is released |
| OOM injection | That pool cannot hold the concurrent requests: preemptions occur and every recomputed output is identical. `pipeline_integration.rs`: two requests that each outgrow a shared 2-page pool now both finish (previously the newer failed). Kernel/engine pressure tests from P5 still pass |
| Bounded fairness | 8 decoders in 6-token passes never wait more than 2 passes between tokens. A 40-token prompt arriving among them gets its first token within 20 passes. Scheduler unit tests: 30 decodes over 6 passes each run exactly 4 times; budget halving and the noise floor; the starvation guard |
| Queue limits and deadlines | Request/token queue limits reject submissions; an expired request fails and releases its state |
| Kernels | Ragged conv/scan against F64 references over mixed decode, prefill, speculative and forked sequences (key dims 64/128/256); span-validation failures before encoding |
| Model | Fused synthetic tests and the real-checkpoint comparisons (`--ignored`) pass unchanged |
| Shared budget | Two plans cannot overcommit one budget; concurrent leases never exceed the limit |

## Measurements

Apple M1 (8 GPU cores, 16 GB), Qwen3.5-0.8B, Q8_0 weights, F16 K/V, 16-token pages. All
timings end at device completion.

**Ragged DeltaNet**, conv plus scan for one layer, batched decode
([kernels](baselines/2026-10-22-m1-p6-kernels/delta.json)):

| Sequences | 0.8B ragged | 0.8B per-sequence | Speedup | 27B shape speedup |
| --- | --- | --- | --- | --- |
| 1 | 0.078 ms | 0.077 ms | 0.99× | 0.87× |
| 4 | 0.302 ms | 0.353 ms | 1.17× | 1.05× |
| 8 | 0.633 ms | 0.769 ms | 1.21× | 1.07× |
| 16 | 1.255 ms | 1.540 ms | 1.23× | 1.10× |

The win is dispatch overhead only. The scan streams each sequence's F32 state, 1 MB per
0.8B layer, in and out every token, so its cost stays linear in batch. Single-sequence
27B-shape scans are 13% slower through the span indirection, about 0.03 ms per layer.

**Model decode pass**, 128-token context
([model](baselines/2026-10-22-m1-p6-model/qwen3_5.json)):

| Sequences | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| Pass | 18.6 ms | 20.1 ms | 26.8 ms | 43.4 ms |

At 2K-token contexts, batch 1 and batch 4 passes take 19.2 ms and 28.7 ms. These are
within 2% of P5 at batch 8; the 18 recurrent layers' share of the batch-8 pass is about
11 ms.

**Serving**: closed-loop bursts of greedy chat requests, 64 output tokens, 256-token
prefill chunks
([before](baselines/2026-10-22-m1-p6-serving/before.json),
[after](baselines/2026-10-22-m1-p6-serving/after.json)). "Before" is the P5/tiled-attention
pipeline at `0619547`, measured with the same harness.

| Workload | Output tok/s | TTFT p50 / p95 | ITL p50 / p95 / p99 | Passes |
| --- | --- | --- | --- | --- |
| 128-token prompts, c=1, before → after | 43.6 → 42.6 | 0.17 / 0.17 → 0.18 / 0.18 s | 20 / 39 / 41 → 21 / 40 / 41 ms | 64 → 64 |
| c=4 | 87.0 → 89.0 | 0.54 → 0.55 s | 36 / 75 / 119 → 36 / 73 / 109 ms | 64 → 64 |
| c=8 | 95.1 → 96.6 | 1.06 → 1.06 s | 67 / 136 / 195 → 66 / 130 / 195 ms | 64 → 64 |
| c=16 | 106.4 → 108.7 | 2.15 / 2.15 → 1.97 / 2.22 s | 118 / 238 / 359 → 114 / 233 / 329 ms | 64 → 65 |
| 8 × 128 + 8 × 2K prompts, c=16 | 43.7 → 40.5 | 15.8 / 15.8 → 10.0 / 18.0 s | 120 / 1746 / 1895 → 285 / 468 / 887 ms | 72 → 121 |

The pass counts show packed execution: 16 concurrent requests complete 64 tokens each
in 64 passes. The old pipeline also packed passes; P6's batching change is the
recurrent dispatch above.

The mixed row shows the policy trade:

- **Before:** every pass carried every prefill chunk, so running decodes stalled behind
  2K-token prefills for up to 1.9 s.
- **After:** decodes keep running and prefill is spread over more passes. Short prompts
  get their first token sooner. The slowest 2K prompts finish prefill later, and
  smaller passes cost 7% of throughput.

`--target-pass-ms` and `--max-batch-tokens` move this trade.

## Batched decode

After the scheduler work, a batch-8 decode pass still cost 3.4 ms per added sequence on
the device and about 2.2 ms per sequence on the host. Profiling one batch-8 serving pass
showed where:

- **Projections:** above 4 rows the GEMV fell to ~25 GB/s, and Candle's tiled GEMM ran at
  about half the weight-bandwidth floor.
- **DeltaNet decode:** the scan streamed state through threadgroup tiles at ~26 GB/s.
- **Host:** each sequence's 1 MB of logits came back through its own blit-and-wait
  round trip (1.3 ms), then was sampled with two more full copies (0.8 ms).

Four changes, each tested against references:

| Change | Implementation | Effect (M1, 0.8B shapes) |
| --- | --- | --- |
| Small-batch Q8_0 projections on the matrix units | `qmm_q8_0_t{1,2}`: transposed 8×8 F32 MMA tiles for 1–16 activation rows. Each lane loads its fragment straight from device memory. A per-block K permutation makes a lane's weights one 8-byte load and its activations two `float4` loads. Threadgroups split K across simdgroups and reduce once; no staging and no barriers in the loop. Dispatch: Candle at 1 row, the GEMV at 2–3, this kernel at 4–16 | At 8 rows: LM head 5.7 ms (GEMV 10.8, Candle 8.9); `gate_up` 0.142 ms (Candle 0.287); `down` 0.066 ms (Candle 0.197). Flat in rows up to 8 (47–61 GB/s); at 16 rows, F32 MMA-bound at parity with Candle's F16-tile GEMM but at F32 precision |
| Coalesced recurrent decode step | `delta_step_d{64,128,256}`: one token per sequence. Lanes own value-column pairs and simdgroups own key-row blocks, so state rows are 256-byte coalesced loads and stores; two reductions cross simdgroups through threadgroup memory. Single-token sequences of a pass use it, the rest the scan, as two dispatches into one output | Conv + recurrence at batch 8: 0.50 ms against 0.68 (ragged scan) and 0.81 (per sequence). State traffic reaches 36–49 GB/s, close to the M1's mixed read/write limit |
| One logits copy per pass | Every sequence's rows are concatenated on the device before the fence; `finish` copies once | Host copy at batch 8: 3.4 ms per pass against 10.7 ms |
| Greedy reduction on the device | For greedy, grammar-free requests the engine enqueues `argmax_f32` (first maximum and a finiteness flag per row), copying 8 bytes per row. Other requests keep full rows. The host greedy path is a chunked, vectorizable scan. `HybridModel::kernels` lets the engine reuse the model's compiled kernels | Host work at batch 8: about 1.4 ms per pass against 17 ms |

`Q4K` has no matrix-unit kernel yet: above its 2–4-row GEMV range it still uses Candle's
F16-tile GEMM.

**Model decode pass**, 128-token context, Q8_0, F16 K/V
([model](baselines/2026-10-22-m1-p6-batch-decode/qwen3_5.json)):

| Sequences | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Before | 18.6 ms | 20.1 ms | 26.8 ms | 43.4 ms | — |
| After | 18.4 ms | 19.5 ms | 23.8 ms | 28.6 ms | 48.6 ms |
| Tokens/s | 54 | 102 | 168 | 280 | 329 |

**Serving**, the same greedy bursts as above
([serving](baselines/2026-10-22-m1-p6-batch-decode/serving.json),
[kernels](baselines/2026-10-22-m1-p6-batch-decode/linear.json),
[recurrent](baselines/2026-10-22-m1-p6-batch-decode/delta.json)):

| Concurrency | Pass p50, P6 → now | ITL p95 | Output tok/s |
| --- | --- | --- | --- |
| 1 | 20.7 → 19.9 ms | 40.1 → 40.0 ms | 42.6 → 44.5 |
| 4 | 36.1 → 26.2 ms | 73.1 → 53.1 ms | 89.0 → 113.5 |
| 8 | 65.7 → 30.0 ms | 130.1 → 60.8 ms | 96.6 → 166.0 |
| 16 | 114.0 → 51.4 ms | 232.7 → 103.6 ms | 108.7 → 180.9 |
| Mixed 128/2K, c=16 | 121.5 → 71.3 ms | 468 → 469 ms | 40.5 → 45.4 |

Correctness:

- **Matrix kernel:** within 1e-5 of the F64 dequantized reference at every row count
  1–16, contiguous and strided.
- **Step kernel:** within 1e-5 of the scalar recurrence, mixed with scan sequences, and
  with the step disabled.
- **Argmax:** keeps the first maximum and flags non-finite rows.
- **Fused model:** decode batches of 3, 6 and 12 stay within 1e-4 of one-row reference
  logits.
- **Real checkpoint:** the reference comparisons and greedy generation (now through the
  device argmax) pass.

## Limits

- **Recurrent state bandwidth bounds decode scaling.** Every sequence reads and writes
  its F32 recurrent state each token: about 36 MB per token on 0.8B and much more on
  27B. That is about 7.5 ms of a batch-8 pass and 15 ms at batch 16, linear in batch at
  ~39 GB/s. Storing it as F16/BF16 would halve the traffic but needs the separate
  quality evaluation the plan requires for recurrent state.
- **9–16 rows are MMA-bound** at F32 (16 rows cost twice 8). F16 MMA tiles would
  double throughput at Candle's precision; not adopted without a quality budget.
- **Sampled requests** (temperature, top-k/top-p, grammars) still copy full rows and
  sample on the CPU, about 0.8 ms per sequence plus one shared copy. GPU sampling is P7.
- **Typed errors.** Queue-full and deadline failures use `TurError::Other`, so HTTP
  still reports them as 500. Typed variants (for 429/503/504 mapping) need an approved
  change to the shared error type.
- **Device overhead per model.** Each plan reserves a tenth of the working set for
  runtime/OS overhead. Two models on one device reserve it twice, which is conservative.
- **Not yet done.** Preemption always recomputes; checkpoint retention is a P7 decision
  with prefix reuse. Cost buckets are per process and are not persisted. Only the M1 is
  measured, and the 27B is not executed. Multi-model serving is planned at the budget
  level but the server loads one model.

## Commands

```sh
cargo test --offline --release --workspace
TUR_QWEN35_0_8B=<snapshot> cargo test --offline --release -p tur-engine --test qwen3_5_fused -- --ignored --test-threads 1
TUR_BENCH_ONLY=delta cargo bench --offline -p tur-metal --bench kernels
TUR_BENCH_ONLY=linear cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_FORMATS=q8_0 TUR_BENCH_KV=f16 TUR_BENCH_PAGES=16 TUR_BENCH_PREFILL=128 TUR_BENCH_CONTEXTS=2048 cargo bench --offline -p tur-engine --bench qwen3_5
TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench serving
```
