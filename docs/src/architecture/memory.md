# State memory

All per-request state lives in one persistent `StatePool` per model (`tur-metal`),
allocated once at load and never grown.

## Pools

- **Attention pages.** Page `p` owns token slots `p × page_tokens ..` in every attention
  layer's key and value tensors, so one page table addresses the whole model. Keys and
  values are stored as F32, F16 or Q8. Q8 is quantized on write, with one scale per
  `group` values (32 by default), and dequantized on read inside the attention kernels.
- **Recurrent slots.** Slot `s` spans every recurrent layer's convolution history and
  matrix state. Slot 0 is a permanent zero state shared by every fresh sequence and never
  written.

Kernels read pages and slots in place through page tables and slot indices. Nothing is
gathered or concatenated per pass.

Gemma's per-layer embedding table is split by decoder layer into column tables. E2B
uses 35 BF16 tables of 128 MiB each, rather than one multi-gigabyte Metal buffer.
The shared loader bounds host chunks to 64 MiB and synchronizes each completed upload
before proceeding, allowing Candle to reclaim device staging buffers. These are loading
fences; request execution keeps its existing completion ownership.

Only Gemma layers that produce their own K/V allocate attention pool layers. Reusing
layers address those same physical owners. Sliding-window masking limits attention,
while page leases retain the full logical prefix for restore and speculation.

## Snapshots and leases

A `ModelState` is a page table plus one recurrent slot at an exact token boundary.
Cloning it shares every lease, which is how prefix snapshots cost nothing to take.

- **Leases are reference counted.** When the last holder drops one, the id is retired.
  Retired ids return to the free list only through `StatePool::reclaim`, which the engine
  calls when no unfenced work can reference them. Reclaiming bumps the id's generation,
  so an id captured before reuse no longer resolves.
- **Tokens visible to a snapshot are never rewritten.** A sequence appends into its
  partially filled tail page in place only if no other snapshot holds that page.
  Otherwise the page's valid prefix is first copied to a fresh page (copy-on-write).
- **Recurrent state is never updated in place.** Each pass writes new slots, and
  `Reservation::commit` publishes the accepted boundary without copying.

## The memory plan

At load, before reading any weight, the pipeline computes a `MemoryPlan` and leases it
from a `DeviceBudget`:

```text
resident = packed weights + attention KV + recurrent state
         + retained checkpoints + activation/scratch peak
         + speculative state + in-flight buffers + runtime overhead
```

- K/V pages, recurrent slots, retained snapshots and speculative boundaries all come
  from the state pool, so its size bounds their sum.
- Scratch is the kernel workspace of the largest planned pass.
- In-flight covers packed inputs, logits and draft indices.
- The overhead is a tenth of the device's recommended working set.

By default the state pool gives every concurrent request room for an 8K-token context.
`--state-memory-mb` sets it explicitly. Every request holds two recurrent slots, and K/V
pages are whole. A plan beyond the device's working set, or beyond what other models
already leased from the same budget, fails before loading. The loaded weights must match
the plan exactly.

At run time the scheduler checks every chunk against the pool. A request that can never
fit, even alone, fails with an out-of-memory error instead of waiting forever.

## Decisions

Measured on Qwen3.5-0.8B on an M1:

- **Paged K/V, read in place.** No whole-cache reconstruction on decode: a decode at
  1,500 tokens allocates 32 KB more than one at 40, where a rebuild would cost 1.5 MB.
  Pools are allocated once, and churn tests show no growth.
- **F16 K/V by default.** It matches F32 quality (KL 7.5e-8) and was the fastest format
  in every model measurement.
- **Q8 is the capacity option.** Within 1e-4 nats of F32 over 8K held-out tokens (99.2%
  top-1 agreement), at half of F16's memory, but slower to prefill. On a 16 GB Mac
  running the 27B, memory rather than speed binds, so Q8 is the likely default there.
  That still needs measuring on hardware that can run it.
- **Q8 groups of 32.** Groups of 64 and 256 save 1.5% and 2.9% of Q8 memory, are no
  faster, and raise KL 1.3× and 2.8×.
- **16-token pages.** Sizes 16–128 measured within 3%. The smallest wastes the least tail
  memory and copies the least on copy-on-write.
- **No high-precision recent window.** Q8 error does not grow with context, so such a
  window would have nothing to correct.
- **F32 recurrent state by default.** F16 with stochastic rounding
  (`--recurrent-state f16`) gives 6–7% faster batched decode and half the recurrent
  memory, but its KL from F32 grows with context (5.2e-5 nats on average), and
  speculation requires F32.
- **One budget for everything.** Weights, pages, slots, snapshots, scratch, in-flight
  outputs and drafts are charged to one plan, leased before loading, with OS headroom.
  Admission never relies on a sequence count alone.
- **Low-bit K/V was rejected.** Every 4-bit format missed the quality budget by 10–30×
  (see [Rejected experiments](../design/rejected.md)).

- **Account for Gemma PLE and shared K/V explicitly.** Resident PLE tables and activation
  peaks enter the existing memory plan. Only K/V owners allocate state. E2B checkpoint
  execution validates bounded staging; synthetic F32 fixtures validate shared-cache
  chunking and rollback.
