# P15: prefix-aware scheduling and retention

2026-10-24, Apple M1 (8 GPU cores, 16 GB), macOS 26.6.2, pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`, Q8_0 weights, F16 K/V, F32 recurrent state,
`pre-release` profile, sampled at T=0.7, top-k 20, top-p 0.95. Base commit `2012dcc` (P14).
Only the M1 and the 0.8B checkpoint are measured. P9 has not run, so this package
measures against the base commit only. P7's prefix reuse is the baseline; its historical
18× TTFT gain on a warmed prompt is not counted again.

## Outcome

**Requests that reach a cold shared prefix together now wait for the oldest to compute
it, and eviction now keeps the snapshots that save the most recomputation.** A burst of
eight users behind a cold 1.5K-token system prompt gets its first token 5.6× sooner
(TTFT p50 9.57 → 1.72 s), with 5.7× fewer prefill tokens and 3.8× the output
throughput. An agent loop caches 83% of its prompt tokens instead of 72% and finishes
1.25× sooner. Under churn, a recurring prefix is now restored in 3 of 5 waves, where P7's
LRU order never restored it. Traffic without reuse, the P7 warmed-prefix workloads and
ordinary decode and mixed traffic are unchanged within noise.

| Item | Result | Status |
| --- | --- | --- |
| 1. Locality, lookup cost, saved prefill, retained bytes | Six reuse workloads added; lookup cost now follows the entry count: a 32K-token hit 152 → 6.6 µs, a miss 294 → < 0.1 µs | Measured; **length index kept** |
| 2a. Wait for an in-flight producer | Cold shared burst TTFT 5.6× lower; agent first step 3.0× lower | **Kept** (default) |
| 2b. Admit restorable prefixes first | Warm TTFT 5.1× lower, but cold TTFT +17% and output throughput −10% | **Rejected** ([patch](baselines/2026-10-24-m1-p15/rejected-hits-first.patch)) |
| 3a. Eviction by saved compute per retained byte | Churn: hot prefix restored in 3 of 5 waves per phase (P7 order: 0) | **Kept** (default) |
| 3b. Checkpoint spacing | 128 caches 12% more multi-turn tokens than 256 at equal last-turn TTFT | 256 kept |
| 3c. Radix index | Not built: the indexed hash lookup already costs ≤ 7 µs at 32K tokens | **Rejected by bound** |
| 4. Isolation, verification, boundaries, CoW; cold, low-reuse, churn | Preserved and tested; no low-reuse regression | Validated |

## Workloads and baseline

`TUR_BENCH_WORKLOADS=reuse` runs six workloads on the default pipeline configuration
(`PipelineConfig::new`, 256-token spacing, retention capped at 1/8 of state memory: 8
entries at 8 running, 4 at 4 running). Each snapshot keeps one 20.2 MB recurrent slot
plus the pages only it holds.

| Workload | Shape |
| --- | --- |
| `cold_shared` | 8 users, cold 1,536-token system prompt, 48-token turns, 32 outputs |
| `contention` | 4 running slots, 16 requests: 8 behind a warmed 1K-token prompt, 8 cold 1K-token prompts |
| `agent` | 4 agents, shared 1K-token tool manual, 5 steps each adding a 48-token reply and a 128-token tool result |
| `churn` | 4 slots; per wave one request on a hot 1K-token prefix and three cold 1K-token prompts; the hot prefix changes after 5 waves |
| `no_reuse` | 16 distinct 512-token prompts at 8 running |
| `short_shared` | 8 users share a cold 300-token system prompt (one aligned block), then 256-token turns |

Base commit, three runs ([1](baselines/2026-10-24-m1-p15/baseline-reuse.json),
[2](baselines/2026-10-24-m1-p15/baseline-reuse-2.json),
[3](baselines/2026-10-24-m1-p15/baseline-reuse-3.json)). The gaps:

- **No reuse inside a burst.** All eight `cold_shared` users are admitted in the same
  pass, so each computes the 1.5K-token prompt: 0 tokens cached, 13,053 packed tokens.
- **Recurring prefixes lost under churn.** Each wave inserts about 24 snapshots into 4
  entries. P7 evicts never-hit entries first, then the least recently used, so the hot
  request's snapshots (inserted first) go first, and the hot prefix never hits.
- **Lookup cost grows with prompt length.** A lookup probed every prefix length (about
  9 ns per token); a waiting request repeats it whenever a snapshot is inserted
  ([lookup](baselines/2026-10-24-m1-p15/baseline-lookup.json)).

## Waiting for an in-flight producer

At admission, waiting requests are taken in arrival order. A first-time request younger
than the queue-age limit (`max_queue_wait`, 2 s) waits, unscheduled and without a slot,
when an older request that is running or admitted by the same pass shares a prefix with
it and has not yet reached the deepest snapshot it will insert within the shared tokens.
That snapshot is an aligned boundary, or the producer's prompt end when the waiter
extends the whole prompt. A request waits only if the cache holds nothing as deep, and
only after the shared tokens are compared. When the producer passes that boundary, the
waiter restores the snapshot like any other hit. A cancelled or finished producer
releases its waiters at the next pass, and a waiter that reaches the queue-age limit
computes on its own. Shared lengths come from a binary search over the chained keys,
which also keeps namespaces apart.

The first version let waiters wait on any older request, including ones still queued.
On `contention` that serialized the warmed requests behind each other: warm TTFT rose
4.37 → 5.56 s, because the cache already held their prefix. Restricting producers to
running or admitted requests, and skipping prefixes the cache already holds, made it
neutral.

## Eviction by saved compute per retained byte

An entry's value is the tokens it covers beyond the deepest retained entry it extends,
times how often its prefix was recently inserted or restored, over its slot plus the
pages only it holds (`ModelState::exclusive`). Inserting a prefix that was computed
before counts as reuse even after its entry was evicted. Counts are halved every 32
events per entry (at least 256), so prefixes that stop recurring lose their protection.
Ties go to the least recently used. A new entry is always kept.

Counting only an entry's tokens beyond its ancestor matters on agent and chat
traffic. A first version used the full length, which favored deep prompt-end and
reply-end snapshots over the aligned snapshot the next step restores. Qwen3.5's
template re-renders the previous reply differently, so those deep snapshots never match
the next turn (P7). That version cached fewer agent tokens than P7's order.

Attribution on `agent`, single runs ([await only](baselines/2026-10-24-m1-p15/ab2-lru.json),
[eviction only](baselines/2026-10-24-m1-p15/ab2-no-await.json),
[both](baselines/2026-10-24-m1-p15/ab3-all.json)): waiting with P7's order caches 24,448
tokens, both together 24,825, and the new eviction alone 20,715, below the base's 21,570. Without
waiting, four agents computing the manual together count as four reuses of every
aligned block, which inflates its shallow entries. The two changes are kept together;
eviction alone is not a recommended configuration.

## Results

Three runs of the final code ([1](baselines/2026-10-24-m1-p15/final-reuse-1.json),
[2](baselines/2026-10-24-m1-p15/final-reuse-2.json),
[3](baselines/2026-10-24-m1-p15/final-reuse-3.json)) against the base runs. Each cell is
the three-run mean; run-to-run spread is under 3% for every row below.

| Workload | Metric | Base | P15 | Change |
| --- | --- | ---: | ---: | ---: |
| `cold_shared` | TTFT p50 / max | 9.58 / 9.64 s | 1.72 / 1.79 s | 5.6× lower |
| | Packed tokens | 13,053 | 2,301 | 5.7× fewer |
| | Output tok/s | 23.6 | 89.7 | 3.8× |
| | ITL p95 | 48 ms | 48 ms | — |
| `short_shared` | TTFT p50 | 3.32 s | 2.10 s | 1.58× lower |
| | Packed tokens | 4,833 | 3,041 | 1.6× fewer |
| | Output tok/s | 58.7 | 81.5 | 1.39× |
| `agent` | Cached / prompt tokens | 21,570 / 29,895 | 24,825 / 29,895 | 72% → 83% |
| | Wall, 5 steps | 12.34 s | 9.90 s | 1.25× |
| | TTFT p50, step 1 / step 5 | 3.21 / 1.33 s | 1.07 / 1.01 s | |
| `churn` | Hot tokens cached, phase 1 / phase 2 | 0 / 0 | 3,072 / 3,072 | |
| | Wall, both phases | 35.33 s | 32.65 s | −7.6% |
| `contention` | TTFT mean, warm / cold | 4.41 / 5.53 s | 4.36 / 5.48 s | neutral |
| `no_reuse` | Output tok/s, TTFT p50 | 56.9, 3.86 s | 56.8, 3.87 s | neutral |

In `churn`, the hot request of each phase's first two waves still misses: a prefix needs
a second insertion before it outranks single-use entries. The base inserts 243 snapshots
and evicts 239; P15 inserts 219 and evicts 215.

P7 workloads ([base](baselines/2026-10-24-m1-p15/baseline-prefix.json),
[P15](baselines/2026-10-24-m1-p15/final-prefix.json)) and default traffic
([base](baselines/2026-10-24-m1-p15/baseline-serving.json),
[P15](baselines/2026-10-24-m1-p15/final-serving.json)), single runs:

| Workload | Base | P15 |
| --- | ---: | ---: |
| Warmed shared prompt, spacing 256: TTFT p50, out tok/s | 555 ms, 156.0 | 563 ms, 156.3 |
| Multi-turn, spacing 256: cached tokens, last-turn TTFT p50 | 7,168, 1.11 s | 7,168, 1.13 s |
| Multi-turn, spacing 128: cached tokens, last-turn TTFT p50 | 7,168, 1.49 s | 8,064, 1.13 s |
| Decode ×1 / ×4 / ×8 / ×16, out tok/s | 49.0 / 115.9 / 183.8 / 200.6 | 48.9 / 117.7 / 183.5 / 199.6 |
| Mixed ×16: out tok/s, TTFT p50 | 54.9, 8.30 s | 55.1, 8.25 s |

**Spacing stays at 256.** With the new eviction, 128-token spacing caches 12% more
multi-turn tokens, but last-turn TTFT is the same as at 256. It also inserts twice the
snapshots and splits the default 256-token prefill chunks. The extra depth is below
one chunk of prefill on this model.

## Lookup cost and the radix option

[Base](baselines/2026-10-24-m1-p15/baseline-lookup.json) and
[final](baselines/2026-10-24-m1-p15/final-lookup.json) host timings, CPU pool, 8 entries
(64 entries are the same within noise):

| Prompt tokens | Keys (once per request) | Hit, base → P15 | Miss, base → P15 |
| ---: | ---: | ---: | ---: |
| 2,048 | 37–78 µs | 11.7 → 0.5 µs | 18.2 → < 0.1 µs |
| 8,192 | 147 µs | 37.3 → 1.5 µs | 71.5 → < 0.1 µs |
| 32,768 | 0.58–1.5 ms | 152 → 6.6 µs | 294 → < 0.1 µs |

A `BTreeMap` of entry lengths makes a lookup probe only lengths at which entries exist.
A hit's remaining cost is comparing its tokens. A radix tree would walk the matched
tokens too, so it cannot beat this. Hashing keys at submission (about 18 ns per token)
is now the largest cost. It runs during intake, which overlaps device work since P7.
Hashing by aligned blocks would cut it, but only off the critical path, so it was not done.

## Rejected and deferred

- **Restorable prefixes first** ([patch](baselines/2026-10-24-m1-p15/rejected-hits-first.patch)).
  Restorable requests pass older misses until the oldest waiting request reaches the
  queue-age limit ([run](baselines/2026-10-24-m1-p15/ab3-hits-first.json)). On
  `contention`, warm TTFT falls 4.37 → 0.86 s, but cold TTFT rises 5.49 → 6.40 s and
  output throughput falls 44.9 → 40.3 tok/s, since short warm prefills no longer overlap
  cold ones. That starves unrelated requests, which the exit criteria exclude. A
  latency-SLO scheduler could revisit it.
- **Not carrying a decode-free prefill budget into mixed passes**
  ([patch](baselines/2026-10-24-m1-p15/rejected-budget-carry.patch)). Tried against an
  ITL spike that turned out to come from waiters waiting on other waiters, which the
  producer rule fixed. On its own it slowed `contention` and `churn` by 4–5%
  ([with](baselines/2026-10-24-m1-p15/ab2-all.json),
  [without](baselines/2026-10-24-m1-p15/ab2-carry.json)).
- **Admission control for new entries** (TinyLFU-style). Not built: it would refuse an
  agent's deeper next-step snapshot in favor of the older one it extends.
- **MTP and prefix reuse** remain mutually exclusive (P8/P14 boundary-state issue,
  unchanged).

## Changes

- `prefix`: a length index for lookups; `PrefixCache::depth` (a lookup without its
  side effects) and `reachable`; `PrefixKeys::get` and `shared`; eviction by saved
  compute per retained byte with aging reuse counts; `clear` also drops the counts.
  Entries keep their chain keys (8 bytes per token of host memory).
- `tur_metal::state::ModelState::exclusive`: pages and slot only this state holds.
- `pipeline`: waiting requests wait for in-flight producers (`awaits`), and waiting ones
  are left out of scheduling for that pass.
- Benchmarks: `TUR_BENCH_WORKLOADS=reuse` in `serving` (per-request TTFT and cached
  tokens are now in every run's JSON), and a new `prefix` host benchmark.

## Validation

- Offline workspace nextest: 263 passed, 8 checkpoint-dependent skipped
  ([tail](baselines/2026-10-24-m1-p15/nextest-tail.log)). Strict Clippy on all targets
  ([tail](baselines/2026-10-24-m1-p15/clippy-tail.log)), formatting and the
  no-default-features check pass.
- New integration tests (random weights, where a wrong state changes text): a cold
  shared burst runs one producer, three waiters restore at least 48 tokens, and every
  output equals its cold run; cancelling the producer mid-burst lets the waiters
  complete with cold-run outputs and leaks no state; requests past the queue-age limit
  and requests in other namespaces never wait.
- New unit tests: lookups after eviction and clearing, `depth` counting nothing, shared
  keys and reachable depths, a recomputed prefix outranking newer single-use entries, an
  extension evicted before the entry it extends, reuse counts aging away, and `exclusive`
  accounting across shared pages and clones.
- The namespace test now holds 16 entries. With 8, the new eviction drops the tenant's
  8-token prompt-end extension, so an exact repeat restores 32 of 40 tokens instead of
  40; isolation is unchanged.

## Remaining limits

- One model, one device, three runs of the reuse workloads and one of the others,
  without confidence intervals. The workloads are synthetic shapes, not a traffic
  trace.
- Waiting is bounded only by the queue-age limit. A request with a deadline shorter than
  that may wait until its deadline expires. A producer slowed by many concurrent decodes
  delays its waiters by the same amount.
- Requests that share a prefix but not their first aligned block, or a short prompt
  that only a longer prompt contains, are not grouped as producer and waiter.
- Exact-repeat prompt-end snapshots rank low under the new eviction. They save one pass
  beyond their few tokens, and the value estimate does not count that pass.

## Reproduction

```bash
export TUR_QWEN35_0_8B=<snapshot>
TUR_BENCH_WORKLOADS=reuse TUR_BENCH_SAMPLING=0.7,20,0.95 \
  cargo bench --offline -p tur-engine --bench serving --profile pre-release
TUR_BENCH_WORKLOADS=prefix TUR_BENCH_SPACING=128,256 TUR_BENCH_SAMPLING=0.7,20,0.95 \
  cargo bench --offline -p tur-engine --bench serving --profile pre-release
TUR_BENCH_WORKLOADS=decode,mixed TUR_BENCH_CONCURRENCY=1,4,8,16 TUR_BENCH_SAMPLING=0.7,20,0.95 \
  cargo bench --offline -p tur-engine --bench serving --profile pre-release
cargo bench --offline -p tur-engine --bench prefix --profile pre-release
cargo nextest run --offline --workspace --test-threads 1
```

Base runs used `2012dcc` with this package's `serving.rs`. The `ab-*`, `ab2-*` and `ab3-*`
artifacts came from intermediate builds with temporary switches (`TUR_P15_AWAIT`,
`TUR_P15_EVICTION`, `TUR_P15_HITS_FIRST`, `TUR_P15_BUDGET`), since removed. `ab-*`
predate the producer and marginal-value fixes, and `ab2-hits-first` predates a fix that
had disabled waiting in that mode.
