# P7 prefix reuse, asynchronous overlap and GPU sampling

Status (implemented 2026-10-23). Measured on the P0 M1 with Qwen3.5-0.8B (Q8_0 weights, F16
K/V, F32 recurrent state):

- **GPU sampling.** Penalties, temperature, top-k and top-p run on the device, with a
  per-request counter-based RNG. Only the token returns to the host. With sampling at
  T=0.7, top-k 20 and top-p 0.95, inter-token latency (ITL) p50 at 8 and 16 concurrent
  requests falls 1.4× against the host sampler (39.9 → 28.5 ms and 72.6 → 50.5 ms).
  Sampled serving now runs within about 1 ms of greedy serving.
- **Prefix reuse.** Snapshots are zero-copy and sit at aligned token boundaries. Hash
  hits are verified token by token, namespaces are kept apart, and snapshots are evicted
  before any running request is preempted. Eight users behind a shared 1.5K-token system
  prompt get their first token 18× sooner (time to first token, TTFT, p50 11.0 → 0.61 s),
  and the prefill work drops 17× (13,059 → 771 packed tokens).
- **Overlap.** Each pass is submitted, then finished. The completion fence runs on its own
  thread, and the host copy of the outputs is folded into that fence. Intake,
  tokenization and disconnect handling run while the device executes. With no work queued
  before submission, the device's idle time per pass is at most 0.06–0.31 ms, against
  0.40–0.68 ms when the host copy followed the fence. This idle time is a lower bound.
- **Bounded streaming.** Each streaming client gets a bounded channel plus a coalescing
  buffer. A client that stalls past the bound is cancelled with an explicit error, and its
  state is released.

All workspace tests pass (192), as do the three ignored real-checkpoint tests, strict Clippy, and checks with and
without default features. No dependencies were added; the unused `candle-transformers`
was removed. Artifacts: [baselines/2026-10-23-m1-p7](baselines/2026-10-23-m1-p7/)
([manifest](baselines/2026-10-23-m1-p7/manifest.json)).

## GPU sampling

**Semantics** (`tur_metal::sampling`). Penalties apply first:

- **Repetition** divides positive logits and multiplies negative ones, for every prompt or
  sampled token.
- **Presence and frequency** subtract once, or per occurrence, for every sampled token.

Grammar masks apply next (host only). Weights are then
`w = exp((logit − max) / temperature)`. Token `j` survives truncation when fewer than
`top_k` tokens are strictly heavier and their mass is below `top_p` of the total. These
are Candle's top-k-then-top-p semantics, except that tied weights at the boundary are all
kept.

Penalties see the history committed before the pass plus that pass's speculative drafts,
never outputs sampled within the same pass. The host and the device therefore agree on
them even though the device builds them at submission. Prompt tokens enter the history
through `HybridEngine::add_context`. Generation configs may set `repetition_penalty`; the
server and CLI accept all three penalties.

**Algorithm.** The kernel never sorts. Each round inverts the CDF of the weights above a
pivot, using a 24-bit uniform `u(seed, draw, round)`:

- **Accept:** if the drawn token is in the truncated set, it is the sample.
- **Reject:** otherwise every token at most as heavy is outside the set too, so the pivot
  rises to the drawn weight and the next round redraws from the strictly heavier rest.

This is exact rejection sampling from the renormalized truncated distribution. The
heaviest token is always accepted, so the rounds terminate. A row the kernel does not
settle in 64 rounds is reported as unconverged, and the host resamples that penalized row
explicitly; no measured row needed this (`SamplingStats::fallback_rows` counts them).

**Kernels.**

- **`penalize_f32`:** one thread per distinct penalized token, in place.
- **`sample_f32`:** one 1,024-thread threadgroup per row. Passes are strided and
  coalesced. The CDF is inverted in thread-major order, a fixed permutation of the
  vocabulary, and simdgroup 0 scans the owning thread's tokens 32 at a time.
- **Batch layout:** every device row of a pass is concatenated once. Greedy rows come
  first and use `argmax_f32`, which now also uses the full threadgroup width. One
  penalty dispatch and one dispatch each for argmax and sampling cover the whole batch.

The host reference uses the same semantics and RNG. It inverts in token order, selects
top-k candidates with a size-k heap, and uses a branch-free, vectorizable exponential.

**Routing** (`SamplingStats`):

- **Device:** requests without a grammar, when the model runs on Metal kernels.
- **Host:** grammar-constrained requests, and every CPU-reference request.
- **Configuration:** `PipelineConfig::device_sampling = false` forces the host path.

**Kernel cost**, one 248,320-entry Qwen3.5 row per sequence, T=0.7, top-k 20, top-p 0.95
([kernels](baselines/2026-10-23-m1-p7/sampling-kernels.json)):

| Rows | Device sample | Device argmax (was) | Penalties, 1K tokens/row | Host copy + sample |
| --- | --- | --- | --- | --- |
| 1 | 0.155 ms | 0.053 ms (0.202) | 0.016 ms | 0.33 + 0.91 ms |
| 4 | 0.334 ms | 0.060 ms (0.209) | 0.039 ms | 0.66 + 3.58 ms |
| 8 | 0.564 ms | 0.136 ms (0.293) | 0.058 ms | 1.18 + 7.21 ms |
| 16 | 0.828 ms | 0.274 ms (0.369) | 0.078 ms | 2.06 + 14.41 ms |

**Rejected kernel layout.** The first kernel gave each thread a contiguous segment, so
each simdgroup touched 32 cache lines per load. It cost 0.82/1.94/2.27/3.68 ms at
1/4/8/16 rows ([artifact](baselines/2026-10-23-m1-p7/sampling-kernels-segmented.json)).

**Host reference cost.** The host sampler started at about 2.2 ms per row. Branch-free
folds and the polynomial exponential brought it to 0.9 ms, close to the Candle sampler
that P6 used (about 0.8 ms per sequence).

**Serving**, the same bursts as P6 with sampled requests: device sampling against the host
sampler in one build ([device](baselines/2026-10-23-m1-p7/serving-sampled-device.json),
[host](baselines/2026-10-23-m1-p7/serving-sampled-host.json)):

| Workload | ITL p50 / p95, device | ITL p50 / p95, host | Output tok/s | Post-completion host time per pass |
| --- | --- | --- | --- | --- |
| c=1 | 18.6 / 19.9 ms | 21.8 / 23.5 ms | 46.2 vs 41.2 | 0.04 vs 2.61 ms |
| c=4 | 24.9 / 27.6 ms | 33.6 / 35.1 ms | 111.3 vs 88.6 | 0.07 vs 9.24 ms |
| c=8 | 28.5 / 30.3 ms | 39.9 / 47.0 ms | 169.0 vs 131.8 | 0.12 vs 11.05 ms |
| c=16 | 50.5 / 53.3 ms | 72.6 / 81.4 ms | 181.8 vs 145.5 | 0.31 vs 21.55 ms |
| Mixed 128/2K, c=16 | 253.6 / 340.5 ms | 279.1 / 371.4 ms | 48.0 vs 41.9 | 0.17 vs 17.55 ms |

Sampled ITL at c=8 and c=16 (28.5 and 50.5 ms) matches greedy (29.2 and 49.9 ms) within
1 ms. Throughput comparisons between modes are approximate, because sampled requests stop
at different EOS positions; ITL is the reliable metric.

**Validation.**

- **Statistics:** chi-square tests at the 99.9% level pass for the host reference (12
  tokens, five truncation settings, 40K draws) and for the device kernel (10 tokens, four
  settings, 20K draws).
- **Full vocabulary:** on a 248,320-token row, bucketed chi-square tests (dominant tokens
  plus the rest of the kept set) pass on both paths. Every draw falls inside the exact
  F64 truncated set.
- **Penalties:** device penalties match the reference bit for bit on strided rows.
  Duplicate entries and rows outside the batch are rejected before encoding.
- **Engine:** a presence penalty steers greedy decoding away from sampled tokens, and
  penalty history rolls back with speculative rejection.
- **Batch order:** on Metal, six penalized, sampled requests give identical text alone,
  batched, and batched in reversed arrival order. Every row was decided on the device.

**Reproducibility.** Determinism is validated separately from distributional correctness:

- **Batch order:** draws depend only on the request seed and its draw counter, never on
  batch order or slot.
- **Run to run:** a batched schedule repeated in a fresh pipeline gives identical tokens
  (Metal test).
- **Batch composition:** batch-dependent projection kernels can still change logits in
  their last bits. `PipelineConfig::reproducible(chunk)` runs one request at a time in
  fixed chunks without prefix reuse. Its tokens are then independent of other traffic
  and arrival order (Metal test).
- **Across paths:** the host and device paths sample the same distribution with the same
  stream, but the same uniform can select different tokens on them. CPU-reference output
  is reproducible on the CPU only.

## Prefix reuse

`tur_engine::backend::prefix`. A snapshot is a committed `ModelState`: the page table plus
the recurrent slot written at exactly that boundary. A recurrent state exists only where a
pass ended. The pipeline therefore never lets a prefill chunk cross a multiple of the
spacing, so every aligned position of every request is a valid snapshot. Decode commits
every position anyway.

- **Insertion:** at every aligned commit, a reference to the request's own leases, with
  no copy. The spacing must be a multiple of the page size, so restoring a snapshot never
  copies a tail page.
- **Lookup:** keys chain a per-process keyed hash over aligned blocks, seeded by the
  request's `cache_salt` (the HTTP field of the same name). The deepest hit that leaves
  at least one prompt token to compute logits is used, and only after comparing every
  token. Colliding entries are rejected and counted. The pool pins model identity,
  quantization, K/V layout and recurrent dtype, so entries cannot cross them.
- **Restore:** at admission, for the oldest waiting requests that fit the free running
  slots, through `HybridEngine::adopt`. A waiting request that missed is not looked up
  again until a new snapshot is inserted.
- **Memory:** each retained snapshot adds one recurrent slot to the plan
  (`PlanInputs::retained`; 20.2 MB on 0.8B). Pages are shared with live requests.
- **Pressure:** eviction removes never-hit entries first, then the least recently used.
  It runs before the scheduler preempts or fails a request, and while snapshots keep the
  oldest admissible request's next chunk from fitting. That second case stops once
  evicting no longer frees pages.
- **Preemption:** a request that restored a prefix replays its own pass boundaries only
  above the restore point. Below it, it prefills in aligned chunks or restores again.
- **Defaults:** one snapshot per running sequence, 256-token spacing.
  `PrefixConfig::DISABLED` turns reuse off; the server flags are `--prefix-snapshots`
  and `--prefix-spacing`.

`GenerationStats::cached_tokens` and the HTTP `usage.prompt_tokens_details.cached_tokens`
report the restored tokens.

**Measurements** (T=0.7, top-k 20, top-p 0.95; [artifact](baselines/2026-10-23-m1-p7/prefix.json)).

*Shared system prompt.* One request warms a 1,536-token system prompt, then eight users
with distinct 48-token turns arrive together:

| Spacing | TTFT p50 / max | Packed tokens | Wall | Output tok/s |
| --- | --- | --- | --- | --- |
| Off | 11.00 / 11.08 s | 13,059 | 12.2 s | 20.9 |
| 256 | 0.61 / 0.69 s | 771 | 1.75 s | 145.9 |

Every spacing from 64 to 512 restores all 1,536 tokens for each of the eight users, since
1,536 is a multiple of each.

*Multi-turn conversations.* Four conversations of six turns run together. Each turn
re-renders the history with the template (thinking off) and adds a 64-token user turn;
each table cell gives cached tokens out of the turn's prompt tokens, with TTFT p50:

| Spacing | Turn 2 | Turn 4 | Turn 6 | Cached / prompt tokens, all turns | Snapshots inserted / evicted |
| --- | --- | --- | --- | --- | --- |
| Off | 0/1,560, 1.30 s | 0/2,540, 2.10 s | 0/3,516, 2.90 s | 0 / 13,765 | — |
| 64 | 1,024, 0.52 s | 1,792, 0.70 s | 1,792, 1.53 s | 7,936 | 126 / 110 |
| 128 | 1,024, 0.51 s | 1,536, 0.89 s | 2,048, 1.28 s | 8,192 | 56 / 40 |
| 256 | 1,024, 0.49 s | 1,024, 1.29 s | 2,048, 1.27 s | 7,168 | 26 / 10 |
| 512 | 0, 1.30 s | 0, 2.10 s | 2,048, 1.27 s | 4,096 | 11 / 3 |

**Spacing decision: 256 tokens.** Spacing trades restorable depth against snapshot churn,
since each snapshot costs a slot:

- **64:** churns through its 16 entries and loses the deep snapshots.
- **128:** restores 14% more tokens than 256, but with twice the insertions. It also
  splits the default 256-token prefill chunks, and it gains nothing on the final turns.
- **512:** misses every prompt shorter than 1K tokens.

256 matches the default prefill chunk, so alignment never splits a chunk. Recomputation
is cheap enough on this model that SSD spill was not pursued. Persisted caches are not
implemented; they would need revision and format validation and atomic writes.

**Validation** (`prefix_reuse.rs`, unit tests).

- **Equivalence:** a repeated prompt and prompts sharing 36 of 40 tokens restore 32
  tokens and generate exactly the text of cold runs with different chunking.
- **Namespaces:** salts isolate entries.
- **Pressure:** with eight concurrent requests over a 48-token shared prefix in a 7-page
  pool, snapshots are evicted, requests are still preempted and recomputed, and every
  output equals its cold run.
- **No leaks:** after clearing the cache, no page or slot remains leased.
- **Unit tests:** the deepest verified hit, forged-hash collisions, the eviction order,
  key extension and disabled alignment.

## Asynchronous overlap

`InferencePipeline::begin_step` selects and submits a pass, and `finish_step` publishes
it; `step_streaming` runs both. Between the two, callers may submit and cancel requests.
Results for requests cancelled during the pass are dropped, and their pages and slots are
released at completion.

The engine hands each submission to a `Notifier` thread. Its synchronize commits the
encoded tail at once; Candle otherwise commits only every 50 encoders or at a fence. The
outputs of the pass (full host rows and device token pairs) are copied as part of that
fence. That copy is queued last, so waiting for it waits for the pass, and no second
device round trip remains on the critical path.

Only the notifier thread waits while a pass runs. Two concurrent synchronizations of one
device could otherwise return before the other's work completed. A failed fence now
marks the batch failed, so a retried fence (which Candle reports as successful) cannot
publish its outputs.

**Depth.** At most one pass is in flight. The next pass needs this pass's sampled tokens,
and Candle exposes only a device-wide fence. Depth 2 would need device-resident token
feedback and per-submission completion events; it was not attempted.

**Worker.** The serving worker submits, then does intake (template rendering and
tokenization), disconnect cancellation, and pass-dependent delivery while the device
runs. Pass timing (`PassTiming`, on each `BatchStepOutput`) splits:

- `schedule`: expiry, prefix restore, scheduling and packing;
- `encode`: validation, reservation and encoding;
- `overlap`: host work beside the running device;
- `wait`: until the completion event;
- `publish`: after completion;
- `device`: from encoding to completion;
- `idle_before`: from the previous completion to this encoding. The device had nothing
  queued then; it is a lower bound on idle time, since work starts as buffers commit.

The cost model now records `device` time, so host work that overlaps a pass no longer
inflates estimates.

**Greedy serving**, the P6 bursts ([after](baselines/2026-10-23-m1-p7/serving-greedy.json),
[readback after fence](baselines/2026-10-23-m1-p7/serving-greedy-readback-after-fence.json),
[P6](baselines/2026-10-23-m1-p6-review/serving.json)):

| Concurrency | Output tok/s, P6 → P7 | ITL p50 / p95, P6 → P7 | Schedule | Encode | Publish | Idle before, readback after fence → P7 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 45.9 → 47.3 | 19.3 / 20.4 → 18.4 / 19.6 ms | 0.01 ms | 2.6 ms | 0.05 ms | 0.40 → 0.06 ms |
| 2 | 79.9 → 81.8 | 20.2 / 21.1 → 19.6 / 20.3 ms | 0.01 ms | 2.8 ms | 0.06 ms | 0.42 → 0.07 ms |
| 4 | 115.4 → 121.9 | 24.3 / 34.4 → 23.6 / 25.2 ms | 0.01 ms | 2.6 ms | 0.07 ms | 0.46 → 0.08 ms |
| 8 | 168.8 → 171.7 | 29.6 / 30.6 → 29.2 / 30.6 ms | 0.02 ms | 4.5 ms | 0.15 ms | 0.57 → 0.18 ms |
| 16 | 184.7 → 188.5 | 50.1 / 51.8 → 49.9 / 51.4 ms | 0.03 ms | 4.9 ms | 0.27 ms | 0.68 → 0.31 ms |
| Mixed, c=16 | 46.0 → 47.4 | 261.8 / 341.1 → 254.2 / 343.3 ms | 0.02 ms | 4.3 ms | 0.14 ms | 0.61 → 0.17 ms |

**What changed.** CPU scheduling time is 10–30 µs per pass. Encoding (2.5–5 ms) overlaps
the device as buffers commit. The visible idle gap is now 0.3–0.6% of a pass. The
throughput change against P6 (2–6%) is within run-to-run noise for single runs, but it
is consistent with the idle-gap cut and the faster argmax.

**Where overlap does not help.** These closed-loop bursts queue all their requests up
front, so the window has no host work to hide. The window pays off with streaming
arrivals, whose rendering and tokenization would otherwise sit on the critical path.

## Bounded streaming and disconnects

`WorkerMsg::token_tx` is a bounded `tokio::mpsc::Sender`, and non-streaming requests have
none. The worker never waits on a client:

- **Coalescing:** each pass sends at most one message per request, holding text the
  channel cannot take in a per-request buffer.
- **Stalled clients:** a buffer beyond `StreamConfig::max_buffered_bytes` (default
  16 KiB) cancels the request with `TurError::SlowClient`, which maps to HTTP 408 or an
  SSE `slow_client` error event. A stalled client therefore holds at most
  `channel_capacity + 1` messages (default 64) of at most that size.
- **Completion:** the SSE adapter emits any text the channel could not take from the
  final text, so completion does not wait either.
- **Disconnects:** these are detected every pass, while the device runs.
- **Server flags:** `--stream-channel-capacity` and `--stream-buffer-kb`.

**Tests.**

- A client that never reads is cancelled with `SlowClient`, holding exactly its channel
  capacity, and its running slot serves the next request.
- Cancellations between `begin_step` and `finish_step` publish nothing, release every
  page and slot, and leave survivors' outputs unchanged.
- The existing disconnect test still reclaims a dropped stream before admitting the next
  request.

## API changes

These are breaking changes, authorized in [Breaking changes and legacy removal](apple-silicon-redesign.md#breaking-changes-and-legacy-removal).

- `SamplingOptions` gains `repetition_penalty`, `presence_penalty` and
  `frequency_penalty`. `GenerationDefaults` gains `repetition_penalty`. The HTTP request
  accepts all three penalties and `cache_salt`.
- `HybridEngine`: `add_context`, `model_state`, `adopt`, `sampling_stats`,
  `set_device_sampling`, `completed_at`, `in_flight`.
- `InferencePipeline`: `begin_step`, `finish_step`, `prefix_stats`,
  `clear_prefix_cache`, `sampling_stats`. `PipelineConfig` gains `prefix` and
  `device_sampling`, and `PipelineConfig::reproducible` is added.
- `GenerationRequest::cache_salt`, `GenerationStats::cached_tokens`,
  `BatchStepOutput::timing` (`PassTiming`).
- `PoolConfig::for_budget` takes `retained` snapshot slots, and `PlanInputs::retained` is
  added.
- `Completion::notify`, `Completion::take_copy` and `Notifier` are added. `tur_metal::sampling` and
  `Kernels::{penalize, sample}` are new.
- `TurError::SlowClient` is a new error variant. `WorkerMsg::token_tx` is now an
  `Option` of a bounded sender, and `WorkerConfig::stream` (`StreamConfig`) is added.

## Gap closure (2026-10-23)

These follow-ups closed the gaps that the first P7 pass left open. Artifacts:
[baselines/2026-10-23-m1-p7-gaps](baselines/2026-10-23-m1-p7-gaps/). Across the
workspace, 201 tests and the three real-checkpoint tests pass; strict Clippy passes, as
do checks with and without default features.

**Grammar masks on the device.**

- **API:** `GenerationRequest::grammar` accepts any llguidance grammar. The parser factory
  for the tokenizer is built on the first constrained request.
- **Mask timing:** for a single-output, non-speculative grammar sequence, the engine
  computes the mask at submission, while the device runs the model. It uploads an
  LSB-first bitset, and `mask_f32` sets disallowed logits to −∞ after the penalties.
  Argmax and sampling then run on the device, and the grammar commits at publication.
  A grammar that stops at submission publishes nothing for that output. Speculative or
  multi-output grammar sequences still sample on the host, because their later masks
  depend on tokens sampled within the pass.
- **Argmax semantics:** −∞ now means a masked token, and a fully masked row is invalid.

The table compares grammar-constrained requests (a JSON-like regex the grammar
terminates; T=0.7, top-k 20, top-p 0.95) with device and host masking
([device](baselines/2026-10-23-m1-p7-gaps/grammar-device1.json),
[host](baselines/2026-10-23-m1-p7-gaps/grammar-device0.json)):

| Concurrency | ITL p50, device vs host | Pass p50 | Encode (mask computation included) | Post-completion host time |
| --- | --- | --- | --- | --- |
| 1 | 18.4 vs 23.3 ms | 18.4 vs 23.6 ms | 3.3 vs 1.6 ms | 0.13 vs 4.5 ms |
| 4 | 24.1 vs 33.0 ms | 24.1 vs 33.0 ms | 4.9 vs 1.0 ms | 0.30 vs 9.3 ms |
| 8 | 29.0 vs 44.9 ms | 29.1 vs 44.5 ms | 6.0 vs 0.8 ms | 0.52 vs 15.9 ms |
| 16 | 50.2 vs 81.6 ms | 50.2 vs 81.2 ms | 10.4 vs 1.0 ms | 1.31 vs 31.1 ms |

With masks on the device, constrained passes cost the same as unconstrained ones at
every concurrency. Mask computation takes about 0.34 ms per request and runs during
`encode`, while the device executes the committed command buffers.

**Prompt-end and reply snapshots.**

- **Keys:** keys now chain over every token, so snapshots can sit at any committed
  boundary, not only aligned ones.
- **Prompt end:** the commit that completes a prompt leaves a snapshot holding an
  allocated copy of that prompt's last logits. A request whose whole prompt matches
  restores it, then `HybridEngine::sample_logits` samples its first output with the
  request's penalties, grammar and RNG stream. That takes no model pass, so the first
  token arrives at admission.
- **Reply end:** a finished request leaves a snapshot after its last consumed token.
- **Snapshot cost:** an unaligned snapshot makes its next appender copy one tail page.
  The memory plan charges one F32 logit row per retained snapshot.
- **Replay boundaries:** recomputation accepts only restore lengths that are its own
  pass boundaries, or that lie at or below its first restore. Below that point, chunks
  also stop at the first restore, so original pass boundaries are preserved.

Tests:

- An exact repeat restores all 40 tokens, needs one pass fewer than its outputs, and
  generates the cold run's text.
- A next turn whose history renders identically restores the prompt plus the reply
  minus its unconsumed last token, and matches the cold run.
- On Metal, cached logits sample exactly the token the original pass sampled.

A real Qwen3.5 template re-renders the previous assistant turn without the generation
prompt's empty think block, so reply snapshots do not match its next turn. On the
multi-turn workload, reuse is unchanged at 7,168 of 13,765 prompt tokens (last-turn TTFT
p50 2.91 → 1.28 s with reuse; [prefix](baselines/2026-10-23-m1-p7-gaps/prefix.json)).

**Retained snapshots scale with state size.** `PrefixConfig::state_share` defaults to
1/8: retained snapshots may take at most that share of the state pool's bytes
(`MemoryPlan::retained` reports the result). A 27B deployment on a small budget
therefore keeps proportionally fewer 157 MB states. An explicit `--prefix-snapshots`
count is honored as given.

**Streaming arrivals.** Open-loop Poisson arrivals of 48 requests with 512-token prompts
and 64 sampled outputs each. Each request is rendered and tokenized either before the
next pass (serial) or while the device runs the current one (overlapped). TTFT starts at
the scheduled arrival ([arrivals](baselines/2026-10-23-m1-p7-gaps/arrivals.json)):

| Rate | Mode | Output tok/s | TTFT p50 / p95 | ITL p50 / p95 / p99 | Device idle (lower bound) |
| --- | --- | --- | --- | --- | --- |
| 2/s | serial | 83.0 | 4.62 / 10.66 s | 61.1 / 289.9 / 340.2 ms | 9.4% |
| 2/s | overlapped | 85.9 | 3.96 / 9.32 s | 60.3 / 268.3 / 290.1 ms | 4.8% |
| 4/s | serial | 89.2 | 10.48 / 19.13 s | 52.7 / 296.2 / 378.5 ms | 9.6% |
| 4/s | overlapped | 93.2 | 9.13 / 17.70 s | 52.5 / 290.5 / 345.6 ms | 5.2% |

Rendering and tokenization took 1.6–1.8 s of host time per run, about 36 ms per request.
Overlapping that work halves device idle time, cuts TTFT p50 by 13–14% and ITL p99 by
9–15%, and adds 3.5–4.5% throughput. Both rates exceed what the M1 sustains with this
model, so queues grow and TTFT is in seconds.

**Run-to-run spread.** The closed-loop bursts were run three times each, with prompts
unique per burst so that no burst restores another's snapshots. Each cell gives the mean,
then the minimum–maximum
([greedy](baselines/2026-10-23-m1-p7-gaps/serving-greedy-1.json) and repeats 2–3,
[sampled](baselines/2026-10-23-m1-p7-gaps/serving-sampled-1.json) and repeats 2–3):

| Workload | Greedy tok/s | Greedy ITL p50 | Sampled tok/s | Sampled ITL p50 | P6 greedy tok/s |
| --- | --- | --- | --- | --- | --- |
| c=1 | 46.2 (44.2–47.2) | 18.8 (18.2–19.8) ms | 45.7 (44.6–46.7) | 19.1 (18.6–20.0) ms | 45.9 |
| c=2 | 78.7 (77.5–80.4) | 20.1 (19.4–21.2) ms | 78.4 (76.7–80.8) | 20.4 (19.7–21.1) ms | 79.9 |
| c=4 | 114.0 (109.3–116.4) | 24.6 (23.6–26.1) ms | 107.2 (105.5–108.7) | 24.6 (24.1–25.3) ms | 115.4 |
| c=8 | 168.3 (160.1–173.8) | 29.5 (28.1–31.4) ms | 166.1 (158.7–169.9) | 29.6 (28.6–31.3) ms | 168.8 |
| c=16 | 185.9 (179.6–190.3) | 50.0 (48.9–51.3) ms | 176.1 (167.0–185.1) | 51.8 (50.2–52.9) ms | 184.7 |
| Mixed, c=16 | 47.2 (45.9–47.9) | 258.4 (254.3–265.7) ms | 46.5 (45.9–47.8) | 262.3 (255.4–265.9) ms | 46.0 |

Every P6 greedy result lies inside the P7 range, so the 2–6% gain in single runs above is
within noise. Device idle time is 0.06–0.33 ms per pass (p50). Sampled ITL stays within
2 ms of greedy.

**Copy semantics.** On Metal, Candle's `Tensor::copy` shares the source buffer instead of
copying it. In the first P7 pass this let the in-place penalties of a single device row
write into the model's own logits. Retained logits exposed it: the cached first token
mismatched for one of eight seeds. Every copy that must be independent now uses
`force_contiguous`; this covers the device rows, retained logits, the copy-on-write
temporary, and `ModelState::read_layers`, whose "detached" recurrent tensors had aliased
pool slots. Kernel tests pin the behavior.

## Limits

- **One device.** Only the M1 was measured, three runs per serving case and one per other
  case, without confidence intervals. The 27B is not executed.
- **Snapshot memory.** Each retained snapshot costs a full recurrent slot (20.2 MB on
  0.8B, about 157 MB on 27B), bounded by default to 1/8 of the state pool. Eviction does
  not rank snapshot kinds: reply snapshots can displace prompt-end snapshots.
- **Multi-turn reuse on real templates.** Reply snapshots miss when a template re-renders
  history differently from the generated text, as Qwen's does with its empty think
  block. Reuse then falls back to aligned snapshots and loses up to 255 tokens per turn.
  Fixing it would need a pass boundary where the next turn's rendering diverges.
- **Not implemented:** persisted caches and SSD spill.
- **Sampling.** The CPU and GPU paths do not give token-identical samples. Uniforms have
  24 bits, so tokens below about 2⁻²⁴ of the remaining mass are effectively never drawn.
  Speculative or multi-output grammar sequences sample on the host.
- **Overlap.** In-flight depth is one. Overlap hides only host work that does not depend
  on the current pass.

## Commands

```sh
cargo test --offline --release --workspace
TUR_QWEN35_0_8B=<snapshot> cargo test --offline --release -p tur-engine --test qwen3_5_fused -- --ignored --test-threads 1
TUR_BENCH_ONLY=sample cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench serving
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_SAMPLING=0.7,20,0.95 [TUR_BENCH_DEVICE_SAMPLING=0] cargo bench --offline -p tur-engine --bench serving
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_WORKLOADS=prefix TUR_BENCH_SAMPLING=0.7,20,0.95 TUR_BENCH_SPACING=0,64,128,256,512 cargo bench --offline -p tur-engine --bench serving
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_WORKLOADS=arrivals TUR_BENCH_RATE=2,4 TUR_BENCH_SAMPLING=0.7,20,0.95 cargo bench --offline -p tur-engine --bench serving
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_WORKLOADS=grammar TUR_BENCH_CONCURRENCY=1,4,8,16 TUR_BENCH_SAMPLING=0.7,20,0.95 [TUR_BENCH_DEVICE_SAMPLING=0] cargo bench --offline -p tur-engine --bench serving
```
