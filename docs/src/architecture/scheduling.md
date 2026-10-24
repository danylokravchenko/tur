# Scheduling

`InferencePipeline` (`tur-engine`) turns requests into packed passes. Its `Scheduler` is
a pure policy over request descriptors. The pipeline supplies a callback that checks
whether each candidate chunk fits the state pool, so the policy is unit-tested without a
model.

## One iteration

`begin_step` runs these steps, then submits the pass without waiting for it:

1. **Expire deadlines.** A request's deadline counts from submission, including time
   waiting for worker intake. Requests past their deadline fail before execution, and
   again before publication: work that finishes late fails, and its state is released.
   A pass already on the GPU is fenced, not interrupted.
2. **Restore prefixes.** Waiting requests whose prompts start with a retained snapshot
   adopt it and skip that part of prefill (see [Prefix reuse](prefix-reuse.md)).
3. **Decodes first.** Every running decode gets one token, in strict rotation by when it
   last ran, so a decode skipped for lack of budget runs first next time. While prefill
   work exists, `min_prefill_tokens` of the pass is held back for it.
4. **Prefill.** Chunks of waiting and partially prefilled requests follow in arrival
   order, admitting new requests while fewer than `max_running` hold state. The prefill
   budget is the largest power-of-two token count whose estimated pass time meets the
   target `max(target_pass, 2 × decode-only estimate)`, so prefill can at most double a
   pass's decode cost.
   - **Without decodes,** prefill takes the whole pass.
   - **Damping.** The budget moves at most a factor of two per pass and never drops
     below the minimum, so one noisy timing cannot starve prefill.
   - **Queue-age guard.** The oldest prefill waiting longer than `max_queue_wait`
     ignores the cost target, within the pass's token cap. If younger requests hold the
     memory or running slots it needs, the newest of them is preempted.
   - **Chunk boundaries.** Chunks never cross a prefix-snapshot boundary, so snapshots
     land on aligned positions.
5. **Memory.** A chunk that admits a request must leave one page of growth for every
   running sequence. A decode that does not fit preempts the newest running request.

Pass costs are measured per shape and context bucket, from completed device time, and
feed the next iteration's estimates.

## Preemption

Under memory pressure, the newest running request releases its state and goes back to
waiting. When it runs again it recomputes its prompt and generated tokens with its RNG
stream and grammar state intact, and with its original chunk boundaries, so recurrent
rounding matches the first run. Older work is never evicted to admit newer work.
Retained prefix snapshots are evicted before any running request is preempted.

## Limits

| Setting | Default | Flag |
| --- | --- | --- |
| `max_running` | 8 (serve), 1 (run) | `--max-batch-size` |
| `max_waiting` | 256 | `--max-waiting` |
| `max_waiting_tokens` | 1,048,576 prompt tokens | library only |
| `max_batch_tokens` | `max(2048, chunk + max_running)` | `--max-batch-tokens` |
| `max_chunk` | 256 | `--prefill-chunk-size` |
| `min_prefill_tokens` | `min(max_chunk, 32)` | library only |
| `target_pass` | 60 ms | library only |
| `max_queue_wait` | 2 s | library only |

Submissions beyond `max_waiting` or `max_waiting_tokens` are rejected immediately
(`QueueFull`, HTTP 429). A positive output limit that cannot fit the model's context,
and a request that cannot fit the state pool even alone, are rejected at submission.

## Decisions

- **Decode priority with bounded, adaptive prefill.** On a mixed 128/2K-token workload,
  this cut p95 decoded-text intervals 3.7× against the earlier unscheduled loop. Throughput
  fell 7% and first-text latency p95 rose 14%; decode-only serving is unchanged.
- **Recompute, not swap.** Preemption recomputes rather than copying state out of the
  pool, and preserves rounding boundaries and RNG, so a preempted request produces what
  it would have produced uninterrupted (barring batch-shape effects, see below).
- **Batch shape can change the last bits.** Kernels reduce in different orders at
  different shapes. Stress tests compare mixed lengths, reshuffled arrivals, cancellation
  and pool exhaustion against solo runs on random weights. `PipelineConfig::reproducible`
  serializes requests when exact reproducibility matters. Concurrent batch invariance was
  built and rejected on cost.
- **One shared device budget.** Several models on one device lease from one budget
  instead of each assuming the whole device. Arbitration is tested with two loaded tiny
  models; the server loads one.
- **Admitting restorable prefixes first was rejected.** It lowered warm TTFT 5.1× but
  raised cold TTFT 17% and cut throughput 10% (see
  [Rejected experiments](../design/rejected.md)).
