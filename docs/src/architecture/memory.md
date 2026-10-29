# State memory

Every request needs memory for its attention keys and values and for its recurrent state. Tur keeps all of it in one `StatePool` per model (in `tur-metal`), allocated once at startup and never grown. Fixed pools make memory use predictable, avoid allocation on the hot path, and let kernels read state exactly where it lives.

## Pools

Attention pages hold keys and values. Page `p` owns the same run of token positions in every attention layer, so a single page table describes a request's whole attention state. Keys and values are stored as F32, F16 or Q8. Q8 is quantized on write, with one scale per group of 32 values by default, and dequantized inside the attention kernels.

Window pages serve attention layers that only look back a fixed distance, like Gemma's sliding layers. They have their own space, indexed by the same logical page numbers, and a request keeps only the window pages its future queries and drafts can still reach. Page 0 is a never-written placeholder for released pages.

Recurrent slots hold the DeltaNet state. Slot `s` spans every recurrent layer's convolution history and matrix. Slot 0 is a permanent zero state shared by every fresh request.

Kernels read pages and slots in place through page tables and slot indices, so nothing is gathered or concatenated before a pass.

### Gemma specifics

Only Gemma layers that produce their own K/V get attention pool layers, and layers that reuse K/V read from those. Sliding layers keep one key more than Gemma's window (513 on E2B), because the assistant drafter reads one key further back than the model itself. The budget gives each request and each retained snapshot enough window pages for its window plus one pass, and the rest goes to full pages.

Gemma's per-layer embedding table is split into one table per decoder layer, 35 tables of 128 MiB on E2B, rather than one multi-gigabyte Metal buffer. The loader uploads them in chunks of at most 64 MiB and waits for each to finish, so Candle can reuse its staging buffers.

## Snapshots and leases

A `ModelState` is a page table plus one recurrent slot at an exact token boundary. Cloning it shares every page and slot instead of copying them, which is why prefix snapshots cost nothing to take.

- Pages and slots are reference counted. When the last holder drops one, it is retired, and it returns to the free list only through `StatePool::reclaim`, which the engine calls once no unfinished GPU work can use it. Reclaiming bumps the id's generation, so an old reference can't reach the reused memory.
- Tokens a snapshot can see are never overwritten. A request writes into its partly filled last page directly only if nobody else holds that page; otherwise it first copies the page's valid part to a fresh page (copy-on-write).
- Recurrent state is never updated in place. Each pass writes new slots, and `Reservation::commit` makes the accepted one current without copying.

## The memory plan

At startup, before reading any weight, the pipeline works out a `MemoryPlan` and reserves it from a `DeviceBudget`:

```text
resident = packed weights + attention K/V + recurrent state
         + retained snapshots + activation and scratch peak
         + speculative state + in-flight buffers + runtime overhead
```

K/V pages, recurrent slots, snapshots and speculative state all come from the state pool, so its size caps their total. Scratch is the kernel workspace of the largest planned pass, in-flight covers packed inputs, logits and draft indices, and the overhead is a tenth of the GPU's recommended working set.

By default the state pool gives every concurrent request room for an 8K-token context; `--state-memory-mb` sets it explicitly. A plan larger than the GPU's working set, or than what other models on the same budget have left, fails before loading, and the loaded weights must match the plan exactly. That way Tur fails at startup with a clear error instead of pushing macOS into swap halfway through serving.

While running, the scheduler checks every chunk against the pool. A request that can't fit even on its own fails with an out-of-memory error instead of waiting forever.

## Decisions

Measured on Qwen3.5-0.8B on an M1 unless noted:

- Paged K/V is read in place. A decode never rebuilds the cache: a decode at 1,500 tokens allocates 32 KB more than one at 40, where a rebuild would cost 1.5 MB. Pools are allocated once, and churn tests show no growth.
- F16 K/V is the default. It matches F32 quality and was the fastest format in every model measurement.
- Q8 is the capacity option. It stays within 1e-4 nats of F32 over 8K held-out tokens at half of F16's memory, but prefills more slowly. On a 16 GB Mac running the 27B, memory rather than speed is the limit, so Q8 is the likely default there; that still needs measuring on a Mac that can run it.
- Q8 uses groups of 32. Groups of 64 and 256 save only 1.5% and 3% of memory, aren't faster, and raise the error 1.3× and 2.8×.
- Pages hold 16 tokens. Sizes from 16 to 128 measured within 3% of each other, and the smallest wastes the least memory at the end of a sequence and copies the least on copy-on-write.
- There is no high-precision window for recent tokens. Q8 error doesn't grow with context, so such a window would have nothing to correct.
- Recurrent state is F32 by default. F16 with stochastic rounding (`--recurrent-state f16`) gives 6 to 7% faster batched decode and halves recurrent memory, but its error grows with context, and speculation needs F32.
- Everything is charged to one budget. Weights, pages, slots, snapshots, scratch, outputs and drafts form one plan, reserved before loading with room left for the OS. Admission never relies on a request count alone.
- 4-bit K/V was rejected. Every 4-bit format missed the quality budget by 10 to 30× (see [Rejected experiments](../design/rejected.md)).
- Gemma's sliding layers store only their window. They are 12 of E2B's 15 K/V-producing layers, and keeping only the window plus the current page cuts K/V memory per token beyond the window to a third. At 8K tokens, one request needs 57 MB instead of 151 MB, with decode and prefill speed unchanged within noise. A ring buffer was not built: every pass would copy the whole window, and attention would need a second key source inside each chunk.
- Gemma's PLE tables and shared K/V enter the same memory plan. Only K/V-producing layers allocate state; the E2B checkpoint validates bounded staging, and synthetic F32 fixtures validate shared-cache chunking and rollback.
