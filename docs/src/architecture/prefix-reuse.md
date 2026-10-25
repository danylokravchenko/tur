# Prefix reuse

A hybrid model's recurrent state cannot be truncated like a key-value cache: the state
after token `n` exists only if a pass ended exactly at `n`. Prefix reuse therefore
stores exact boundaries. Each `PrefixCache` entry is a committed `ModelState` (page table
plus recurrent slot) after some request's first `n` tokens.

## Where snapshots are taken

- **Every multiple of the spacing** (256 tokens). The pipeline never lets a prefill chunk
  cross one, so these boundaries always exist.
- **The end of each prompt**, together with the logits of its last token. An identical
  prompt then samples its first output without running a model pass.
- **The last consumed token of each finished request.** A conversation's next turn can
  resume right after the reply.

Snapshots share leases with the request that produced them, so taking one copies
nothing. It keeps one recurrent slot and its pages alive. A snapshot that ends inside a
page makes its first appender copy that one tail page.

## Lookup

Keys chain a keyed hash over tokens, seeded by a namespace: the tenant, or the request's
`cache_salt`. Equal prefixes in different namespaces never meet, and a hash hit is used
only after comparing every token. The pool pins the model's identity and state layout,
so entries cannot cross models, quantizations or positional settings.

A lookup probes only the lengths at which entries exist, so its cost follows the number
of entries rather than the prompt's length.

## Waiting for a producer

When several requests reach the same cold prefix together, only the oldest computes it.
The others wait until its snapshot exists and then restore it. The set of waiters is
rebuilt every pass, so a cancelled or finished producer stops holding them back at once.
Waiting is bounded by the scheduler's queue-age guard.

## Retention and eviction

By default the cache keeps one snapshot per running request (`--prefix-snapshots`),
capped at 1/8 of the state pool's bytes. Under memory pressure, snapshots are evicted
before any running request is preempted.

Eviction removes the entry that saves the fewest tokens per retained byte. An entry's
saving is the tokens it covers beyond the deepest retained entry it extends, times how
often its prefix was recently computed or restored. Its cost is its recurrent slot plus
the pages only it holds. Counts are halved periodically, so prefixes that stop recurring
lose their protection. A new entry is always kept. Pages and slots return to the pool
only when no request or entry holds them.

## Decisions

Measured on Qwen3.5-0.8B on an M1:

- **Zero-copy, verified snapshots.** Warmed prefixes generate exactly the text of cold
  runs. Eight users behind a 1.5K-token system prompt reached their first token 18×
  sooner than without reuse (11.0 → 0.61 s p50), with 17× less prefill work.
- **256-token spacing**, chosen from a 64–512 sweep and kept after re-measurement: 128
  cached 12% more multi-turn tokens at equal last-turn TTFT.
- **Prompt-end snapshots keep their logits.** An exact repeat needs no prefill pass.
- **Wait for an in-flight producer.** A burst of eight users behind a cold 1.5K-token
  prompt got its first token 5.6× sooner (9.58 → 1.72 s), with 5.7× fewer prefill tokens.
  An agent's first step was 3.0× faster.
- **Evict by saved compute per retained byte.** Under churn, a recurring prefix survived
  in 3 of 5 waves, where LRU order kept it in none. An agent loop cached 83% of its prompt
  tokens instead of 72% and finished 1.25× sooner. Low-reuse, decode-only and mixed
  traffic were unchanged.
- **Restorable prefixes first, opt-in.** With `engine.prefix_restorable_first`, waiting
  requests with a restorable snapshot take free slots ahead of older cold ones until the
  oldest waits the queue-age limit. Under warm-cold contention (16 requests, 4 slots)
  warm TTFT fell 5.4× (4.43 → 0.82 s) while cold TTFT rose 15% and throughput fell 10%
  (44.6 → 39.9 tok/s); the other reuse workloads were unchanged. It stays off by
  default because it slows unrelated requests.
- **No radix tree.** The indexed hash lookup costs 6.6 µs for a 32K-token hit, down from
  152 µs, and under 0.1 µs for a miss.
- **Prefix reuse and the MTP drafter are mutually exclusive.** A snapshot at boundary B
  cannot hold the head's entry for B − 1, which depends on the token sampled after B, so
  enabling `--mtp` disables reuse.
- **Known gap.** When a template re-renders history differently from the generated text
  (Qwen's empty think block), reply snapshots miss. Reuse then falls back to aligned
  snapshots and loses up to 255 tokens per turn.
