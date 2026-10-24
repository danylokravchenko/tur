# Tur design decisions

What Tur is built to do, what was chosen and what was rejected. All measurements are
from one M1 (16 GB) running `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`
unless stated otherwise. The phase reports, benchmark artifacts and rejected patches
that back these decisions are in git history at commit `99b073e`, under `docs/`
(for example `git show 99b073e:docs/baselines/2026-10-24-m1-p13/rejected-batch-invariance.patch`).

## Scope

- **Platform.** Apple Silicon macOS with Metal. CPU execution is a numerical
  reference for tests (`--cpu-reference`), not a production backend. Intel Macs, CUDA and
  Linux are out of scope.
- **Models.** Qwen3.5/3.8 dense hybrid text models (Gated DeltaNet plus gated full
  attention). The architecture is selected from checkpoint fields, never from the model
  name. Ordinary Qwen3 and Granite were deleted, not kept as regression models.
  Correctness comes from pinned external references (Transformers, MLX-LM).
- **Validation level.** The 0.8B checkpoint matches Transformers per layer and in logits
  (relative error ≤ 3.5e-6 against a 1e-4 gate) on CPU and Metal. Qwen3.8-27B shares
  the code path and matches its pinned config, tensor headers and template, but it has
  not been executed: its 4-bit weights exceed this Mac.
- **Experimental.** The Qwen3.6-35B-A3B MoE text path passes an independent small
  Transformers fixture and pinned headers only. Gemma 4 and the flagship's expert MTP
  head are not implemented. Image, video and audio inputs are rejected.
- **Breaking changes.** Breaking Rust, HTTP and CLI changes are allowed and preferred
  over compatibility shims (user decision, 2026-10-21). Replaced code is deleted in the
  same change.
- **Safety.** No unsafe Rust (this rules out NEON intrinsics; hot host loops are written
  so the compiler can auto-vectorize them). No unwrap, expect or panic. New dependencies
  need approval.

## Workspace

```text
tur-core    shared errors, checked packed-execution descriptors
tur-metal   state pools, completion fences, Metal runtime and MSL kernels, packed weights
tur-models  checkpoint validation and loading, Qwen3.5 model, chat templates, tools
tur-engine  HybridEngine, scheduler, prefix cache, speculation, sampling, pipeline
tur         facade re-exports, HTTP server library, integration tests
tur-cli     the `tur` binary: run, serve, quantize
```

Dependencies point down the list. Model-specific configuration stays with its model;
there is no universal `Config` and one error hierarchy (`tur-core`).

## Execution contracts

- A model declares its state layers: paged attention K/V, or recurrent convolution and
  matrix state, each with its own dtype. A request owns its state; weights are shared and
  immutable.
- `HybridModel::forward` takes one packed batch: tokens, positions, per-sequence ranges,
  committed state, requested logits and recurrent boundaries. There is no per-request loop.
- The engine validates descriptors with checked arithmetic, enqueues one pass and fences
  before it reads outputs. Each request publishes independently, so one request's model,
  sampling or grammar failure never fails the others. A failed fence quarantines the engine.
- State handles are generation-tagged, so stale and foreign handles fail to resolve.
  Resources are reclaimed only after every command that references them completes.
- Recurrent state is double-buffered: new slots are written beside committed ones. This
  gives exact rollback without truncation or copies, at two recurrent slots per request
  (20.2 MB on 0.8B, 157 MB on 27B).

## Kernels and dispatch

Fused Metal kernels are the default on Metal. A custom kernel replaces Candle only for
the shapes where it measurably wins.

| Operation | Condition | Kernel |
| --- | --- | --- |
| Q8_0 projection | 1 / 2–3 / 4–16 / >16 rows | Candle `mul_mv` / `qmv_q8_0` / `qmm_q8_0` / Candle F16 tiles |
| Q4K projection | 1 / 2–16 / >16 rows | Candle `mul_mv` / `qmm_q4k` / Candle F16 tiles |
| Recurrent decode | single-token sequences | `delta_step`, 16 key rows per simdgroup |
| Recurrent prefill and verification | multi-token sequences | `delta_prepare` (scales only), then the transposed `delta_scan` |
| Convolution | aligned float4 rows / otherwise | `delta_conv4` / `delta_conv` |
| Prefill attention | ≥ 16 threadgroups and ≥ 16 queries per 32-query tile | `attention_tiled_r*` (matrix units, pages read in place) |
| Decode attention, Q8 pools | below the tiled crossover, even query heads per K/V head | `attention_h2_r*_q8` (two heads per simdgroup) |
| Decode attention, F32/F16 pools | below the tiled crossover | `attention_r*`, split-K over ≥ 64-key ranges |
| MoE experts | ≤ 3 / 4–16 / > 16 rows per expert | GEMV / matrix units / wide F16-staged tiles |

- Prefill projections stay on Candle's tiled GEMM, about 2 TFLOPS on 2K tokens, near the
  M1's F32 peak. No custom prefill GEMM was written.
- Softmax, reductions and recurrent accumulation stay in F32. F16/BF16 activations are
  deferred until a quality budget exists.
- Launch shapes are M1 tuning, not constants for every device.

## Memory

- **K/V is paged and read in place.** Persistent pools, 16-token pages, copy-on-write
  tails, and no per-pass concatenation.
- **K/V format: F16 by default.** It matches F32 quality (KL 7.5e-8) and was the fastest
  format. Q8 with group size 32 is the capacity option: within 1e-4 nats of F32 at half
  of F16's memory, but slower to prefill. Groups of 64 and 256 save under 3% of memory
  and raise KL 1.3–2.8×. Page sizes from 16 to 128 tokens measured within 3%, so the
  smallest is used. There is no high-precision recent window, because Q8 error does not
  grow with context.
- **Recurrent state: F32 by default.** F16 with stochastic rounding is opt-in
  (`--recurrent-state f16`): 6–7% faster batched decode and half the recurrent memory,
  but its error grows with context, and speculation requires F32.
- **One budget.** Weights, K/V pages, recurrent slots, snapshots, scratch, in-flight
  outputs and draft state are charged to one plan, which is leased from a shared device
  budget before any weight is read. A budget beyond the device working set is refused.
  The default state pool gives each concurrent request room for an 8K-token context.

## Scheduling and serving

- **Decode first.** Each pass runs every decode first, in strict rotation. Prefill gets
  a budget that targets `max(60 ms, 2 × decode-only estimate)`. That budget is damped to
  at most a factor of two per pass, never drops below a minimum, and is overridden for
  requests that have waited more than 2 s.
- **Preemption.** Under memory pressure the newest request is preempted. It recomputes
  later with its RNG, grammar state and original rounding boundaries preserved. Older
  work is never evicted to admit newer work. Requests that can never fit are rejected.
- **Limits.** The server bounds intake, waiting requests and waiting tokens, enforces
  deadlines before publication, and rejects positive output limits that exceed the
  context.
- **Sampling on the device.** Penalties, temperature, top-k, top-p and grammar masks run
  on the GPU with a per-request counter-based RNG that does not depend on batch order.
  Only the sampled token returns to the host.
- **Overlap.** The completion fence and the output copy run on their own thread, while
  intake, tokenization and disconnect handling run during device execution. In-flight
  depth is one.
- **Streaming.** Each client gets a bounded channel (64 messages) plus a coalescing
  buffer. A client whose buffer passes 16 KiB is cancelled with `slow_client`, and its
  state is released.
- **Reproducibility.** `PipelineConfig::reproducible` serializes requests. Concurrent
  batch invariance is not offered (see Rejected).

## Prefix reuse

- Snapshots are zero-copy. They sit at aligned 256-token boundaries (chosen by a 64–512
  sweep), at prompt ends with retained logits, and at reply ends. Hash hits are verified
  token by token, and namespaces include the model revision and formats.
- Retention defaults to one snapshot per running request, capped at 1/8 of the state pool.
- Requests that reach a cold shared prefix together wait for the oldest one to compute
  it. A burst of eight users behind a cold 1.5K-token prompt got its first token 5.6×
  sooner.
- Eviction keeps the snapshots that save the most recomputation per retained byte.
- Lookups probe only the lengths that hold entries: a 32K-token hit takes 6.6 µs, so a
  radix index is not needed.

## Speculation

- **Opt-in throughput mode** (`--speculative-draft N`). Drafts come from prompt lookup:
  suffix n-grams of 1–4 tokens matched in the request's own history. Draft length adapts
  to measured acceptance and verification cost. Tokens arrive in bursts.
- **Exact target matching.** Every position is sampled from the target and drafts are
  accepted while they match, so output equals ordinary decoding token for token. There
  is no stochastic `min(1, p/q)` correction and there are no draft trees.
  Grammar-constrained requests never draft.
- **Native MTP** (`--mtp`). The checkpoint's own head drafts from the 32,768 lowest token
  IDs through a copy of the LM head's leading rows, which cuts a head pass from 6.5 to
  2.4–3.1 ms at unchanged acceptance. Each draft carries a calibrated confidence. With
  prompt lookup this is 1.20–1.38× ordinary decoding on novel text at one to four
  sequences, greedy and sampled. The head is gated off above four decodes, where its
  upkeep costs 3–4%.
- **Constraints.** MTP and prefix reuse are mutually exclusive: a snapshot cannot hold
  the head's entry for the token before its boundary. Enabling MTP disables prefix reuse.
  Speculation falls back to ordinary decoding unless recurrent state is F32.

## Weights

- `--weight-format f32|q8_0|q4k` converts SafeTensors projections at load time; `q8_0`
  is the default. These are not GGUF imports.
- `tur quantize` writes versioned prepacked artifacts. Each records its source,
  quantizer and calibration identity, is validated before any payload is read, and loads
  byte-identical to load-time conversion. Q4K loads in 0.2 s instead of 9 s, with half
  the peak RSS.
- **GPTQ is the default method.** At equal bytes it cuts Q4K's KL from F32 by 49–59% on
  prose, code and retrieval text, and by 12% on tool/JSON text.
- **Mixed precision is opt-in** (`--budget-bits`). Greedy Q4K→Q8_0 upgrades by measured
  sensitivity: at 5.5 bits KL falls another 34–58% for +23% bytes, at 0.86× decode.
- Tied checkpoints gather the embedding from the packed LM head, which cuts Q4K resident
  weights from 933 to 425 MB at unchanged quality. This is the artifact default.
- Q5K/Q6K are used only for gathered embedding tables: without Tur small-batch kernels
  they are 2.5–3.5× slower in projections.

## Rejected

| Experiment | Result | Recover from `99b073e` |
| --- | --- | --- |
| Chunked (WY/UT) DeltaNet prefill | Correct, but 0.41–0.66× the token scan at every shape | `docs/baselines/2026-10-24-m1-p10/rejected-chunked-delta.patch` |
| Low-bit K/V: groupwise Q4, KIVI, TurboQuant ± QJL | Every 4-bit format missed the quality budget (mean KL ≤ 1e-3, top-1 ≥ 98%) by 10–30×. The only passing format saved 1.17× over Q8 and decoded 2–7% slower | `docs/baselines/2026-10-24-m1-p12/rejected-low-bit-formats.patch` |
| Batch-invariant execution | Bit-identical across batch size, order and chunking, but 0.52–0.60× prefill and 0.59–0.87× single-sequence decode | `docs/baselines/2026-10-24-m1-p13/rejected-batch-invariance.patch` |
| Admitting restorable prefixes first | Warm TTFT 5.1× lower, but cold TTFT +17% and throughput −10%: starves unrelated requests | `docs/baselines/2026-10-24-m1-p15/rejected-hits-first.patch` |
| Carrying a decode-free prefill budget into mixed passes | 4–5% slower on contention and churn | `docs/baselines/2026-10-24-m1-p15/rejected-budget-carry.patch` |
| Pass overlap, submission depth 2, command reuse | Host encoding already overlaps the GPU and the device idles only 28–38 µs per pass, so the gain is bounded at about 1.003×. Not built | — |
| Metal TensorOps for projections | Works without unsafe bindings, but F32 GEMM ran at only 0.86–1.08× Candle. Not adopted on M1 | — |
| Kernel variants: hoisted Q4K unpack, ggml-layout Q4K GEMV, 8-row `delta_step`, 16/64-row scans, 8-key attention steps, 64-query attention tiles | Each lost at representative shapes | — |
| Kernel-tier speculation cost model, learned drafters | Bounded at about 10% by draft cost; no compatible drafter exists for the 0.8B | — |

Rejected on M1 does not mean rejected everywhere. Chunked prefill, TensorOps and
pass overlap should be re-measured on GPUs that are faster relative to the host CPU.

## Open

- Run the flagship 27B and the 35B MoE on hardware that fits them, then measure quality,
  K/V format choice (Q8 is the likely default under 16 GB) and the MoE wide tier, which
  trails Candle by 1.2–1.5×. The MoE work also needs trained routing traces.
- Comparative baseline against pinned MLX-LM and llama.cpp at matched quantization, and
  held-out quality budgets with confidence intervals.
- Chained MTP drafts, the next speculation candidate (an estimated 10% at one sequence).
- Layer-at-a-time calibration, so quantization scales to the flagship.
- Production hardening: health and readiness, metrics, graceful shutdown, stop sequences,
  long thermal and leak runs, and a documented macOS capability floor.
- Multi-turn reuse loses up to 255 tokens per turn when a template re-renders history
  differently from the generated text.
