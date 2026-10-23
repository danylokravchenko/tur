# P6 packed execution and scheduling

Status (implemented 2026-10-22; reviewed 2026-10-23): every serving iteration is one validated packed pass. Every layer
now runs once over the whole batch, including the `DeltaNet` convolution and scan, which
dispatched once per sequence before. A new scheduler selects each pass:

- **Decode first:** running decodes go first, rotated fairly when they exceed the token
  budget.
- **Bounded prefill:** prefill chunks fill a budget sized from measured pass costs.
- **Pressure:** under memory pressure the newest running request is preempted and later
  recomputes its tokens, with its RNG and grammar state intact.

The batched-decode follow-up splits recurrence into step and scan dispatches when both
shapes occur in a pass; it does not loop over requests on the host.

Admission adds a bounded waiting queue (requests and tokens), deadlines and a device
memory budget shared by every model on the device.

Measured on the P0 M1 with Qwen3.5-0.8B (Q8_0 weights, F16 K/V):

- **One pass per iteration:** a 16-request decode burst finishes 64 tokens per request
  in 64–65 passes.
- **Batching:** a batch-8 decode pass takes 28.6 ms, against 147 ms for eight batch-1
  passes (43 ms before the [batched decode](#batched-decode) kernels).
- **Mixed prompts:** with 128- and 2K-token prompts arriving together, decoded-text
  delivery interval p95 falls 3.7× (1.75 s → 0.47 s) and median time to first text
  falls 37%. Output throughput drops 7%, and the slowest first text is 14% later.
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
| Recurrent layers | `Kernels::recurrent_spans` validates and uploads one span table per pass (first row, tokens, committed slot, written slots, per-token flag). `delta_conv` dispatches once per layer. Recurrence uses one coalesced step dispatch for single-token sequences and one scan dispatch for the rest (threadgroups cover sequence × head × column block), returning one packed output. The per-sequence entry points were deleted |
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

1. **Expire deadlines.** Timeouts start at submission, before worker intake waiting or
   tokenization. Expiration runs before execution and before publication: work finishing
   after the deadline fails and its state is released. A running GPU pass is fenced,
   not interrupted.
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
   - **Starvation guard:** the oldest prefill older than `max_queue_wait` ignores the
     cost target, within the pass token cap. If younger requests hold the memory or
     running slots it needs, the newest younger request is preempted. Older work is
     not evicted to admit newer work; the age cap is not a global completion deadline.
4. **Memory.** A chunk that admits a request must leave one page of growth per running
   sequence free.
   - **Decode:** if a decode does not fit, the newest running request is preempted.
   - **Nothing fits:** the newest running request is preempted, or the request fails
     with `MemoryAllocation` if it is alone. Admission already rejects requests that
     cannot fit the empty pool.
   - **Progress:** aged prefill reclaims resources from younger requests. Decode
     rotation bounds gaps when work fits; requests behind older work can still wait
     for that work to finish or reach its deadline.

The cost model keeps moving averages of measured **seconds per token** in power-of-two
buckets of packed tokens × context. Estimates multiply that rate by the requested token
count, including within the measured bucket. An unmeasured shape borrows the nearest
measured bucket. This normalization fixes the old 7-token/70 ms sample estimating the
same shape at 122.5 ms.

**Recompute preemption.** `HybridEngine::release_state` drops every layer's pages and
slot but keeps RNG draws, sampled history and the grammar parser. The pipeline holds
each request's prompt, generated tokens and committed pass boundaries. After preemption
it replays each original chunk without splitting or combining it, preserving F16
rounding positions. The scheduler reserves enough tokens for each indivisible replay
chunk even when the adaptive cost budget is smaller. The final new token's logits produce
the *next* output; already emitted tokens are not sampled again. RNG continuity is
preserved. Floating-point differences between batch-dependent projection kernels can
still affect logits, so arbitrary batch shapes do not have a bitwise continuation
guarantee. Recurrent snapshots and speculative boundaries are charged
through the same per-pass `Demand` (pages plus written slots).

**Defaults** (`SchedulerConfig::new(max_running, max_chunk)`):

| Setting | Default |
| --- | --- |
| Waiting queue | 256 requests and 1M prompt tokens |
| Pass size | 2048 tokens, or one chunk beside every decode if larger |
| Minimum prefill per pass | 32 tokens |
| Target pass | 60 ms |
| Queue-age cap | 2 s |

Worker intake holds at most 256 messages and rejects a full channel immediately. Each
intake drain handles at most 32 messages or 2 ms of admission work (plus the currently
executing admission), then returns to execution, cancellation and expiration. The first
blocking admission when idle is separate from that drain.
Pipeline waiting limits apply after tokenization. Queue-full errors map to HTTP 429,
deadlines to 504, and memory/worker unavailability to 503. Once SSE headers are sent,
these failures are emitted as structured error events followed by `[DONE]`.

The server exposes `--max-batch-tokens`, `--max-waiting`, `--target-pass-ms` and
`--request-timeout-secs`.

The first default throttled prefill against the 60 ms target even with no decodes
running, and the M1 cannot run a 16-way decode pass in 60 ms. That lengthened decode
bursts' time to first text by up to 1.7× (`decode_heavy` c=16: 2.2 s → 3.6 s p95). It
was replaced by the decode-relative target above; only the final measurements are kept.

## Shared device budget

`DeviceBudget` is a lock-free ledger shared across pipelines by cloning.
`ModelFactory::create_model_with` validates serving metadata, then invokes a reservation
callback before reading weight payloads or uploading tensors. `Qwen35::memory_requirements`
derives the state description, exact packed weight sizes (including tied heads) and
workspace from config; the loaded model uses the same workspace calculation.
`InferencePipeline::with_budget` plans beside other leases (`PlanInputs::reserved`) and
atomically leases its resident plan before loading. A model that does not fit is rejected
before weight loading. Loading failures and pipeline drops return their leases.

Workspace covers the model's supported context limit, not just the default 8K pool-sizing
context, and in-flight memory includes both full logits and their gathered copy. A plan
is a resident/device-workspace reservation, not a bound on checkpoint-file reads and
CPU conversion buffers during startup. `InferencePipeline::new` creates its own budget;
callers loading multiple models on one device must use `with_budget` with one shared
budget. The server still serves one resident model. Separate tests load two tiny models
on CPU and Metal, reject a third, execute pressure/replay work, and verify lease release.

## Validation

| Exit criterion | Evidence |
| --- | --- |
| Mixed prompt/context lengths, batch reshuffling | `scheduling.rs`: twelve requests (prompts 1–37 tokens, limits 1–23, half sampled at T=0.9/top-p 0.95) on random weights, arriving all at once, staggered and in reverse, in 9-token passes. Every output equals the request run alone |
| Cancellation | The same workload under a 4-page pool, cancelled while waiting, mid-prefill and mid-decode. Survivors match their solo outputs and every page and slot is released |
| OOM injection | That pool cannot hold the concurrent requests: preemptions occur and every recomputed output is identical. `pipeline_integration.rs`: two requests that each outgrow a shared 2-page pool now both finish (previously the newer failed). Kernel/engine pressure tests from P5 still pass |
| Fairness with explicit resource assumptions | 8 decoders in 6-token passes never wait more than 2 passes between tokens. A 40-token prompt arriving among them gets its first token within 20 passes. Scheduler unit tests: 30 decodes over 6 passes each run exactly 4 times; budget halving and the noise floor; the starvation guard, including reclaiming younger state and running slots |
| Queue limits and deadlines | Refilling intake cannot monopolize execution; queue-full admission rejects immediately. Submission-time deadlines include intake waiting, late results are not published, and HTTP/SSE preserve error codes |
| Kernels | Ragged conv/scan against F64 references over mixed decode, prefill, speculative and forked sequences (key dims 64/128/256); span-validation failures before encoding |
| Model | Fused synthetic tests are rerun after review; real-checkpoint differential comparisons (`--ignored`) retain their earlier evidence, and the corrected serving benchmark loads the real checkpoint |
| Shared budget | Two loaded models on CPU and Metal share a budget; a third is rejected before loading. Corrupt-payload load failures return their lease. Preflight sizes match F32/Q8_0/Q4K, tied and untied layouts; concurrent leases never exceed the limit |

## Measurements

Apple M1 (8 GPU cores, 16 GB), Qwen3.5-0.8B, Q8_0 weights, F16 K/V, 16-token pages. All
timings end at device completion.

**Historical latency correction (2026-10-23):** the October 22 serving harness timed
`new_tokens` text fragments, which can combine several sampled tokens. Its JSON keys
`ttft_seconds` and `itl_seconds` therefore describe first text and intervals between text
fragments. Those artifacts remain unchanged; the historical tables below use corrected
labels. Throughput and completed-pass timings are unaffected by this labeling error.
The corrected harness consumes `BatchStepOutput::sampled_tokens` (including EOS), checks
that every generated token has a timestamp, and uses `GenerationStats::first_token` for
submission-to-first-token latency. Fresh token-based results are recorded below.

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

**Serving (historical text-fragment latencies)**: closed-loop bursts of greedy chat requests, 64 output tokens, 256-token
prefill chunks
([before](baselines/2026-10-22-m1-p6-serving/before.json),
[after](baselines/2026-10-22-m1-p6-serving/after.json)). "Before" is the P5/tiled-attention
pipeline at `0619547`, measured with the same harness.

| Workload | Output tok/s | First text p50 / p95 | Text interval p50 / p95 / p99 | Passes |
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
  get their first text sooner. The slowest 2K prompts finish prefill later, and
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
F16-tile GEMM. (Historical: P8 added one, and the [kernel performance pass](apple-silicon-kernel-performance.md) made Q4K use
`qmm_q4k` for 2–16 rows.)

**Model decode pass**, 128-token context, Q8_0, F16 K/V
([model](baselines/2026-10-22-m1-p6-batch-decode/qwen3_5.json)):

| Sequences | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Before | 18.6 ms | 20.1 ms | 26.8 ms | 43.4 ms | — |
| After | 18.4 ms | 19.5 ms | 23.8 ms | 28.6 ms | 48.6 ms |
| Tokens/s | 54 | 102 | 168 | 280 | 329 |

**Serving (historical text-fragment latencies)**, the same greedy bursts as above
([serving](baselines/2026-10-22-m1-p6-batch-decode/serving.json),
[kernels](baselines/2026-10-22-m1-p6-batch-decode/linear.json),
[recurrent](baselines/2026-10-22-m1-p6-batch-decode/delta.json)):

| Concurrency | Pass p50, P6 → now | Text interval p95 | Output tok/s |
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

## Recurrent state storage

Recurrent matrices can now be stored as F16 or BF16 (`--recurrent-state f16`;
`LoadOptions::recurrent`, `ModelFactory::recurrent`):

- **Updates:** the scan and step kernels still accumulate in F32 and round only when
  storing.
- **History:** the convolution history stays F32 (about 7% of a slot).
- **Identity:** the dtype is part of the state layout, so checkpoints from another layout
  are rejected.
- **Rounding:** 16-bit stores round stochastically. Random bits come from a hash of the
  token position and element, so rounding is unbiased yet deterministic per request:
  independent of slot IDs and batch order. Replay retains the original pass boundaries
  so it performs stores at the same positions; merging decode steps into a prefill
  chunk would change the numerical result. Different batch-dependent projection and
  reduction paths can still change logits. The CPU reference rounds to nearest.

**Quality** was measured on the P5 held-out text (8,192 tokens, SHA-256 `d4b954c2…`),
against F32 state with F32 K/V
([artifacts](baselines/2026-10-22-m1-p6-state-quality/)). A 256-token chunk keeps the
state in F32 registers within each chunk. Chunk 1 stores it after every token, as decode
does, so rounding error compounds:

| Pass | State | Mean KL | KL 0–1K → 4–8K | Top-1 | Mean NLL (F32: 3.6998) |
| --- | --- | --- | --- | --- | --- |
| 256-token chunks, nearest | f16 | 1.1e-7 | 2.0e-8 → 1.6e-7 | 99.98% | 3.6997 |
| 256-token chunks, nearest | bf16 | 7.2e-6 | 1.2e-6 → 1.1e-5 | 99.88% | 3.6999 |
| Decode, nearest | f16 | 1.2e-4 | 4.5e-6 → 2.1e-4 | 99.40% | 3.7014 |
| Decode, nearest | bf16 | 6.2e-3 | 3.6e-4 → 1.0e-2 | 95.42% | 3.7014 |
| Decode, stochastic | f16 | 5.2e-5 | 9.8e-6 → 7.7e-5 | 99.43% | 3.6999 |
| Decode, stochastic | bf16 | 2.5e-3 | 4.9e-4 → 3.5e-3 | 96.98% | 3.6957 |

Stochastic rounding halves F16's decode error and removes its NLL bias, but the error
still grows about 8× from the first 1K tokens to 4–8K. BF16's 7-bit mantissa is not
usable for decode.

**Speed:** F16 state saves 6–7% of a batch-8/16 decode pass (31.3 → 29.5 ms and
51.5 → 47.8 ms) and adds 4–5% of serving throughput at 8–16 requests. Batch 1–2 is
unchanged or up to 3% slower from the rounding hash. The saving is smaller than halving the
state traffic would suggest (about 4 ms at batch 8): the step kernel does not reach
twice its element rate with 16-bit storage. That was not investigated further.

**Decision:** F32 stays the default (`DEFAULT_RECURRENT_STATE`). F16 is an opt-in
capacity option, like Q8 K/V: on average it is closer to F32 than Q8 K/V is to F16
(5.2e-5 vs 9.7e-5 nats), and it nearly halves recurrent memory per request: about 9 MB
less on 0.8B and 73 MB less on 27B (P5: 20.2 MB and 157 MB with F32). Its context-growing error is not bounded beyond 8K tokens. The
CLI and server accept `f32` and `f16` only; BF16 remains in the kernels and benches.

## Small-batch projections beyond 8 rows

At 16 rows the matrix-unit kernel costs about twice 8 rows (LM head 9.4 against
5.4 ms). Diagnostics on the LM head:

| Variant | 16-row LM head | Finding |
| --- | --- | --- |
| Current (two row tiles, two column tiles) | 9.4 ms | — |
| Second tile's multiplies and loads removed (invalid output) | 5.7 ms | The second tile's work costs 3.7 ms for 2G multiply-adds |
| Second tile's activation loads removed (invalid output) | 7.8 ms | Loads and registers account for a share |
| Contiguous K permutation (`k = 4c + s`): half the activation loads, none wasted | 9.6 ms | Kept (neutral; every loaded value used), not a gain |
| One column tile per threadgroup (fewer registers) | 11.8 ms | Worse: amortizing activations over columns matters more |
| Half-precision weight fragments of exact codes, block scale on F32 partials | 33.3 ms | Mixed half × F32 matrix multiplies are much slower on the M1 |

None of the precision-preserving variants improves 9–16 rows; the kernel sits near the
M1's F32 matrix-unit throughput there. It stays the dispatch for 4–16 rows: faster than
Candle's F16-tile GEMM on narrow projections (`down` 0.15 against 0.19 ms), at parity
on `gate_up` and DeltaNet `in_proj`, 7% slower on the LM head, and in F32 rather than
F16 precision.

## Limits

- **Recurrent state bandwidth bounds decode scaling.** With F32 state every sequence
  reads and writes about 36 MB per token on 0.8B (much more on 27B). That is about
  7.5 ms of a batch-8 pass and 15 ms at batch 16, linear in batch at ~39 GB/s (47–53 GB/s
  after the [kernel performance pass](apple-silicon-kernel-performance.md), which gives each step simdgroup 16 key rows). F16
  state trims 6–7% at a measured quality cost (above). An in-register multi-token
  decode path only applies to speculation (P8).
- **9–16 rows are bound by F32 matrix-unit throughput** (above). Only F16 activations
  would go faster, at Candle's precision; not adopted without a quality budget.
- **Sampled requests** (temperature, top-k/top-p, grammars) still copy full rows and
  sample on the CPU, about 0.8 ms per sequence plus one shared copy. GPU sampling is P7.
- **Device overhead per model.** Each plan reserves a tenth of the working set for
  runtime/OS overhead. Two models on one device reserve it twice, which is conservative.
- **Not yet done.** Preemption always recomputes; checkpoint retention is a P7 decision
  with prefix reuse. Cost buckets are per process and are not persisted. Only the M1 is
  measured, and the 27B is not executed. Budget arbitration is exercised with two loaded tiny models on CPU and Metal;
  the server loads one model. Full checkpoint multi-model serving is not measured.

## Corrected sampled-token serving measurement (2026-10-23)

Same M1 and pinned Qwen3.5-0.8B checkpoint, Q8_0 weights, F16 K/V, F32 recurrent state,
256-token prefill chunks, 64 requested output tokens, and concurrency 1/2/4/8/16.
Short/mixed prompt targets are raw tokenizer lengths; rendered token counts are in the
artifact. Each case is one warmed burst; these are observations without repeat-run
confidence intervals. Loading is excluded.

[Raw results](baselines/2026-10-23-m1-p6-review/serving.json),
[source hashes and settings](baselines/2026-10-23-m1-p6-review/manifest.json).
Latencies below count sampled tokens, including EOS. TTFT starts before rendering and
tokenization; ITL uses token publication timestamps. The old text-fragment percentiles
are not a token-latency baseline and should not be used to claim a token-latency speedup.

| Workload | Output tok/s | TTFT p50 / p95 (s) | ITL p50 / p95 / p99 (ms) | Passes |
| --- | --- | --- | --- | --- |
| Short, c=1 | 45.9 | 0.15 / 0.15 | 19.3 / 20.4 / 21.7 | 64 |
| Short, c=2 | 79.9 | 0.29 / 0.29 | 20.2 / 21.1 / 22.3 | 64 |
| Short, c=4 | 115.4 | 0.52 / 0.55 | 24.3 / 34.4 / 42.3 | 64 |
| Short, c=8 | 168.8 | 1.01 / 1.07 | 29.6 / 30.6 / 64.0 | 64 |
| Short, c=16 | 184.7 | 1.93 / 2.02 | 50.1 / 51.8 / 202.7 | 65 |
| Mixed 128/2K, c=16 | 46.0 | 9.52 / 16.64 | 261.8 / 341.1 / 350.3 | 121 |

## Commands

```sh
cargo test --offline --release --workspace
TUR_QWEN35_0_8B=<snapshot> cargo test --offline --release -p tur-engine --test qwen3_5_fused -- --ignored --test-threads 1
TUR_BENCH_ONLY=delta cargo bench --offline -p tur-metal --bench kernels
TUR_BENCH_ONLY=linear cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_FORMATS=q8_0 TUR_BENCH_KV=f16 TUR_BENCH_PAGES=16 TUR_BENCH_PREFILL=128 TUR_BENCH_CONTEXTS=2048 cargo bench --offline -p tur-engine --bench qwen3_5
TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench serving
TUR_QWEN35_0_8B=<snapshot> TUR_EVAL_TEXT=<text> TUR_EVAL_KV= TUR_EVAL_STATE=f16,bf16 TUR_EVAL_CHUNK=1 cargo bench --offline -p tur-engine --bench kv_quality
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_STATE=f32,f16 cargo bench --offline -p tur-engine --bench qwen3_5
```

## Review fixes and API additions (2026-10-23)

- `GenerationRequest::submitted_at`: optional original arrival, assigned by the worker
  before enqueueing; direct struct literals add `submitted_at: None`.
- `BatchStepOutput::sampled_tokens`: `(request, token ID, Instant)` for every published
  token, including EOS, independent of decoded text fragments.
- `Candidate::minimum`: indivisible original chunk size during replay; one otherwise.
- `ModelFactory::create_model_with` and `Qwen35::memory_requirements`: reserve resident
  memory using metadata before loading weights, with no duplicated memory-plan type.
- `TurError::{QueueFull, DeadlineExceeded, WorkerUnavailable}`: typed admission/lifecycle
  errors mapped by the HTTP adapter. Full intake returns a ready error future instead
  of allowing unbounded sender waiting.

Regression suites cover the non-power-of-two cost sample, aged prefill blocked by younger
state/slots, preserved replay chunks under token/cost pressure, F16 continuations under
pool pressure, token timestamps despite buffered punctuation, deadlines, error mapping,
and loaded-model budget arbitration. No dependencies were added.

Review validation: **76 targeted tests passed**, including the Metal pressure/replay,
workspace and loaded-model budget tests. The three ignored real-checkpoint differential
tests were not rerun; the corrected serving benchmark uses the cached pinned 0.8B
checkpoint. `cargo clippy --offline --workspace --all-targets -- -D warnings`, workspace
checks with and without default features, and formatting checks pass. The existing
upstream `block` future-compatibility warning remains.

```sh
cargo nextest run --offline --release -p tur-engine -p tur --lib --test scheduling --test pipeline_integration --test hybrid_serving --test server_integration --test qwen3_5_fused --test scheduling_metal --test-threads 1
```
