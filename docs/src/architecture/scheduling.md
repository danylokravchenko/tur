# Scheduling

Every pass, the scheduler decides which requests get to run and how much of each. It has to balance two kinds of work. Requests that are already replying want one token each, quickly and steadily. New prompts want large chunks processed so their first token arrives soon. Give prompts too much of a pass and every streaming reply stutters; give them too little and new users wait.

`InferencePipeline` (in `tur-engine`) turns requests into packed passes. Its `Scheduler` is a pure policy over request descriptions. The pipeline passes it a callback that checks whether a chunk fits the state pool, so the policy is unit-tested without a model.

## One iteration

`begin_step` runs these steps and then submits the pass without waiting for it:

1. Expire deadlines. A deadline counts from submission, including time spent waiting to be accepted. Requests past their deadline fail before execution and again before their results are published, and their memory is released. A pass already on the GPU is allowed to finish.
2. Restore prefixes. Waiting requests whose prompts start with a cached snapshot take it over and skip that part of their prompt (see [Prefix reuse](prefix-reuse.md)), oldest first. With `engine.prefix_restorable_first`, such requests take free places ahead of older uncached ones, until the oldest has waited `max_queue_wait`.
3. Decodes go first. Every replying request gets one token, in strict rotation by when it last ran, so a decode skipped for lack of room runs first next time. While prompts are waiting, `min_prefill_tokens` of the pass is held back for them.
4. Prompts fill the rest, in arrival order, admitting new requests while fewer than `max_running` hold state. The prompt share is sized from measured costs (see below).
5. Memory is checked last. A chunk that admits a new request must leave one page of growth for every running request. If a decode doesn't fit, the newest running request is preempted.

### Sizing the prompt share

The scheduler measures how long each pass took on the GPU, by shape and context length, and uses that to predict the next one. It gives prompts the largest power-of-two token count whose predicted pass time stays within `max(target_pass, 2 × the decode-only time)`. In other words, prompt work can at most double the time replying users wait for their next token, and short passes are allowed to grow to the target.

- When nobody is decoding, prompts take the whole pass.
- The share changes by at most a factor of two per pass and never drops below the minimum, so one noisy timing can't starve prompts.
- If the oldest waiting prompt has waited longer than `max_queue_wait`, it ignores the time target, within the pass's token cap. If younger requests hold the memory or places it needs, the newest of them is preempted.
- Chunks stop at prefix snapshot positions, so recurrent state exists at each exact boundary.

## Preemption

When memory runs out, the newest running request gives up its state and goes back to waiting. When it runs again, it recomputes its prompt and the tokens it already generated, with its random stream and grammar state intact and its original chunk boundaries, so its output matches what it would have produced uninterrupted. Older work is never evicted to make room for newer work, and cached prefix snapshots are evicted before any running request is preempted.

## Limits

| Setting | Default | CLI / TOML |
| --- | --- | --- |
| `max_running` | 8 (serve), 1 (run) | `--max-batch-size` |
| `max_waiting` | 256 | `--max-waiting` |
| `max_waiting_tokens` | 1,048,576 prompt tokens | TOML: `engine.max_waiting_tokens` |
| `max_batch_tokens` | `max(2048, chunk + max_running)` | `--max-batch-tokens` |
| `max_chunk` | 256 | `--prefill-chunk-size` |
| `min_prefill_tokens` | `min(max_chunk, 32)` | TOML: `engine.min_prefill_tokens` |
| `target_pass` | 60 ms | TOML: `engine.target_pass_ms` |
| `max_queue_wait` | 2 s | TOML: `engine.max_queue_wait_ms` |

Flag-backed settings also have [TOML equivalents](../guide/configuration.md).

Submissions beyond `max_waiting` or `max_waiting_tokens` are rejected immediately (`QueueFull`, HTTP 429). A request whose output limit can't fit the model's context, or that can't fit the state pool even on its own, is rejected at submission.

## Decisions

- Decodes get priority, and prompts get a bounded, adaptive share. On a mixed workload of 128- and 2K-token prompts, this made the slowest 5% of gaps between streamed tokens 3.7× shorter than running prompts unchecked. Throughput fell 7% and the slowest first-token times rose 14%; decode-only serving is unchanged.
- Preemption recomputes rather than copying state out. It preserves chunk boundaries and the random stream, so a preempted request produces what it would have produced uninterrupted, apart from batch-shape effects.
- Batch shape can change the last bits of a result, because kernels add numbers in different orders at different shapes. Stress tests compare mixed lengths, reshuffled arrivals, cancellation and memory exhaustion against solo runs on random weights. `PipelineConfig::reproducible` runs requests one at a time when exact reproducibility matters; bit-exact batching was built and rejected because of its cost.
- Several models on one GPU share one budget instead of each assuming the whole device. This is tested with two small models loaded at once; the server loads one.
- Letting cached prefixes jump the queue is opt-in. Under contention it got cached requests their first token 5× sooner, but uncached requests waited 15% longer and throughput fell 10%, so arrival order stays the default (see [Prefix reuse](prefix-reuse.md)).
- Operational limits can live in TOML. Scheduler timing and waiting-token limits are file settings, and the measured defaults are unchanged.
