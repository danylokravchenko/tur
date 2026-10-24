# P10: parallel prefill and native Metal math

2026-10-24, Apple M1 (8 GPU cores, 16 GB), macOS 26.6.2, pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`. Base commit `165e9b8` (the kernel performance
pass). Every speedup below compares the base commit and this change built in separate
target directories and run interleaved on the same Mac; tables give the best median of
the interleaved runs. Only the M1 is measured.

P9 has not run, so there is no frozen comparative baseline or held-out quality budget
yet. This package therefore measures against the base commit only and changes no precision
policy: activations, accumulation and recurrent updates stay in F32.

## Outcome

| Item | Result | Status |
| --- | --- | --- |
| 1–2. Chunked (WY/UT) DeltaNet prefill | Correct against the scalar recurrence at every tested boundary, but 0.41–0.66× the token scan on M1 at every shape and batch | **Rejected on M1**; removed from the tree, preserved as a [patch](baselines/2026-10-24-m1-p10/rejected-chunked-delta.patch) |
| Prefill inputs (found while profiling item 1) | The prepare pass writes scales instead of normalised q/k copies: prepared scratch −98%; 1.07–1.46× in the kernel benchmark from 2K tokens, but model prefill unchanged within noise | **Kept** for memory |
| 3. GQA K/V reuse in decode attention | Two query heads per simdgroup: 1.06–1.25× attention over Q8 pools, 1.02–1.09× Q8 K/V model decode at 8K context; F32/F16 tie or lose; four heads lose everywhere | **Kept for Q8** |
| 3. Tiled/split-K/projection crossovers | Not retuned: the per-format head-group choice is the only dispatch change | Unchanged |
| 4. Grouped-MoE wide tier | Needs trained routing traces from a checkpoint this Mac cannot run | **Deferred** |
| 5. Metal TensorOps | Available on M1 through runtime compilation without unsafe bindings; F32 GEMM 0.86–1.08× Candle's | **Evaluated; not adopted on M1** |
| 6. F16/BF16 activations | Need P9's quality budgets first | **Deferred** |

## Model level

Fused passes on the 0.8B checkpoint, page 16, F32 recurrent state, two interleaved runs per
side, best median of each ([before](baselines/2026-10-24-m1-p10/model-before-1.json),
[after](baselines/2026-10-24-m1-p10/model-after-1.json)). Decode steps are per-pass times.

| Workload | Q8_0, F16 K/V | Q8_0, Q8 K/V | Q4K, F16 K/V | Q4K, Q8 K/V |
| --- | ---: | ---: | ---: | ---: |
| Prefill 2,048 tokens | 1.548 → 1.586 s (0.98×) | 1.572 → 1.599 s (0.98×) | 1.694 → 1.701 s (1.00×) | 1.721 → 1.713 s (1.00×) |
| Decode ×1, context 128 | 17.4 → 17.6 ms | 17.6 → 17.0 ms | 10.7 → 10.5 ms | 10.8 → 10.7 ms |
| Decode ×16, context 128 | 43.6 → 43.9 ms | 43.6 → 43.9 ms | 45.3 → 44.3 ms | 45.4 → 44.2 ms |
| Decode ×1 at 8K | 19.6 → 19.9 ms (0.98×) | 20.0 → 19.7 ms (1.02×) | 13.0 → 13.3 ms (0.98×) | 13.3 → 12.5 ms (1.07×) |
| Decode ×4 at 8K | 28.8 → 30.0 ms (0.96×) | 29.8 → 28.1 ms (1.06×) | 26.8 → 26.9 ms (1.00×) | 28.1 → 25.7 ms (1.09×) |
| One sequence through `HybridEngine` | 57.7 → 58.2 tok/s | 57.9 → 56.7 tok/s | 92.3 → 92.1 tok/s | 91.1 → 92.8 tok/s |

Only the Q8 K/V columns at 8K context move beyond the spread of the other cells, by the
amount the attention share predicts. Every other cell, including all F16 K/V cells, whose
kernels are unchanged, varies by up to ±4% between runs. Prefill was repeated alone at
2K and 8K tokens with three interleaved runs per side (Q8_0 weights, F16 K/V): 1.557 →
1.562 s and 8.09 → 8.12 s best samples, no change
([before](baselines/2026-10-24-m1-p10/prefill-before-1.json),
[after](baselines/2026-10-24-m1-p10/prefill-after-1.json)).

## Recurrent prefill inputs

`delta_prepare_f32` used to write, for every packed row, L2-normalised q and k plus q·k
and the gates: 4,144 floats per row on the 0.8B checkpoint, 34 MB per layer at 2K tokens,
which the scan then read back. It now writes per key head the two normalising scales and
q·k, then decay and beta per value head (`DeltaShape::prepared_width`, 80 floats per row on
0.8B). The scan reads raw q and k from the convolution output and applies the scales to its
two dot products and to the correction (three multiplies per token instead of a vector
multiply per loaded element), so it reads the same bytes and the prepare pass writes 2%
of them. Rows of `mixed` that are not 16-byte aligned select `delta_scan_unaligned_*`,
which loads `packed_float4`; the model's rows always take the aligned kernel.

Single sequence, one committed state, F32 state, kernel time to completion. Three
interleaved runs per side ([before](baselines/2026-10-24-m1-p10/delta-before-1.json),
[after](baselines/2026-10-24-m1-p10/delta-after-1.json), runs 2–3 alongside).

| Tokens | 0.8B before → after | 27B shapes before → after |
| ---: | ---: | ---: |
| 16 | 0.098 → 0.106 ms (0.93×) | 0.265 → 0.277 ms (0.95×) |
| 128 | 0.562 → 0.507 ms (1.11×) | 1.425 → 1.449 ms (0.98×) |
| 512 | 2.32 → 2.21 ms (1.05×) | 5.97 → 5.99 ms (1.00×) |
| 2,048 | 10.28 → 9.02 ms (1.14×) | 28.3 → 26.3 ms (1.07×) |
| 8,192 | 54.1 → 37.1 ms (1.46×) | 130.2 → 118.7 ms (1.10×) |

The kernel speedup does not reach the model: whole-pass prefill is unchanged within noise
(above), where 1.14× on 12% of a 2K pass would predict about 1.5%. The microbenchmark
allocates fresh prepared and output buffers on each timed call, which the model's pooled
passes avoid; that is a likely but unverified explanation. The change is kept for memory:
the prepared region of the peak workspace plan falls from 64 MiB to 1 MiB at 2K tokens on
0.8B (power-of-two rounded), and in proportion at longer passes.

Short sequences lose 7–8 µs per layer in the kernel benchmark: the scan's per-token loads
now come from `mixed` rows instead of the freshly written, cache-resident prepared rows. This includes
multi-token speculative verification windows, which use the same scan; at 18 recurrent
layers that is about 0.14 ms per 16-token verification pass. The peak workspace plan
charges the smaller prepared region.

The summation order changed: the scales multiply the reduced dot products rather than
each element. Differential tests against the F64 recurrence keep their tolerances and the
real-checkpoint tests pass unchanged. Applying the scales to every loaded element instead
kept the old arithmetic but was 10–20% slower at 16–128 tokens
([rejected, single run](baselines/2026-10-24-m1-p10/rejected-scan-scaled-vectors.json)).

## Chunked DeltaNet prefill (rejected on M1)

Implemented in two kernels and verified, then removed because it never won:

- `delta_chunk_prepare` (one threadgroup per chunk and value head, fully parallel):
  cumulative log decay by simdgroup prefix sum, `KKᵀ` and `QKᵀ` on the matrix units,
  the unit-triangular inverse `T = (I + A)⁻¹` by forward substitution with each thread
  owning one column (A in the lower triangle, T transposed in the upper, no barriers),
  then `U = T diag(β) V` and `W = T diag(β e^γ) K` written to scratch. Every exponent is
  of a non-positive cumulative-decay difference, so extreme decays underflow instead of
  overflowing.
- `delta_chunk_scan` (sequential over a sequence's chunks): each simdgroup keeps a
  `dk × 8` state slice in matrix registers and computes `V' = U − W S`,
  `O = (e^γ Q) S + P V'` and `S' = e^{γ_L} S + (e^{γ_L−γ} K)ᵀ V'`, with W, Q, Kᵀ and P staged
  in threadgroup memory per chunk. The final state is stored once with the scan's
  stochastic rounding (same position and element seeds).

Chunks of 32 and 64 were swept; 128 needs a 66 KB triangular solve, above the 32 KB of
threadgroup memory. Speculative sequences that store every token's state stayed on the
scan. Differential tests covered partial, exact and spilling chunks, a 600-token prefill,
grouped heads with partial value blocks, key dims 64/128/256, F32/F16/BF16 state, near-total
forgetting and near-perfect memory, mixed decode/speculative/prefill passes, a prompt split
at a snapshot boundary and then verified, a bitwise-identical replay, and untouched
committed slots. A sign flip in one product failed them.

Final design against the final (faster) token scan
([artifact](baselines/2026-10-24-m1-p10/rejected-chunked-delta.json)):

| Shape | 16 tokens | 128 | 2,048 | 8,192 | 8 × 256 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B, chunk 32 | 0.41× | 0.58× | 0.55× | 0.47× | 0.53× |
| 0.8B, chunk 64 | 0.16× | 0.26× | 0.30× | 0.28× | 0.28× |
| 27B shapes, chunk 32 | 0.53× | 0.58× | 0.51× | 0.60× | 0.50× |

Why it loses here:

- **Throughput, not latency.** Splitting the same 2,048 tokens into 8 or 16 prompts gives
  the sequential kernel 8–16× more independent simdgroups and leaves the per-token cost
  unchanged, so occupancy is not the limit.
- **Low operand reuse.** Ablations at 2K tokens (0.8B, chunk 32, prepare pass plus the
  sequential kernel, about 15.3 ms): removing the matrix products leaves 3.3 ms, removing
  staging 8.5 ms ([artifacts](baselines/2026-10-24-m1-p10/)). The products run at about
  0.5 TFLOPS, against 1.7 TFLOPS for Candle's large F32 GEMM on the same GPU: each
  simdgroup owns only eight state columns, so every 8×8 product needs a freshly loaded
  operand. Owning more columns multiplies the state registers per simdgroup.
- **More work.** Chunking adds the `KKᵀ`, `QKᵀ`, `T V` and `T K` products (about a third more
  arithmetic than the recurrence) to products that, at this efficiency, already cost about
  what the scalar scan's FMAs do.

Variants that did not help: no staging (direct masked device loads), 25 ms at 2K; a
17 KB staging tile (one threadgroup resident per core) against an 8.7 KB tile, faster only
at one or two simdgroups per threadgroup; split accumulator chains; staging P and a
pre-transposed K, 15.3 → 14.8 ms; 1–16 simdgroups per threadgroup (4 best). Chunk 64 was
always slower than 32: its 16.6 KB triangular solve limits residency.

Upper bound for the model: the scan is 12% of 2K prefill, so even removing it would give
at most 1.14×. A device whose matrix units sustain several times the M1's F32 rate, or a
TensorOps formulation of the recurrence, is where this should be retried; the patch
applies to `165e9b8`.

## Grouped-query decode attention

`attention_kv` takes a head-group parameter: one simdgroup serves `HG` consecutive query
heads of one K/V head, so each loaded and dequantized key and value vector serves `HG`
dot products and accumulations. With `HG = 1` it is the previous kernel, arithmetic
unchanged. Split-K sizing counts simdgroups, so a grouped dispatch gets more key splits.
`Tuning::attention_head_group` (F32/F16 pools) and `attention_q8_head_group` (Q8 pools)
select 1 or 2; a group that does not divide the query heads per K/V head falls back to 1.

Qwen3.5-0.8B attention shape (8 query heads, 2 K/V heads, dimension 256), page 16
([artifact](baselines/2026-10-24-m1-p10/paged-head-groups.json)):

| Decode | F32 | F16 | Q8 g32 | Q8 g64 | Q8 g256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K context | 0.72× | 1.10× | 1.15× | 1.11× | 1.06× |
| 8K context | 0.99× | 0.97× | 1.20× | 1.23× | 1.23× |
| 32K context | 1.01× | 0.91× | 1.20× | 1.22× | 1.22× |
| 2K × 8 sequences | 0.99× | 0.91× | 1.25× | 1.25× | 1.25× |

Q8 was bound by per-key dequantization work, which pairing halves; F32 and F16 are closer
to bandwidth and lose occupancy to the second head's registers. Four heads per simdgroup
were 0.76–0.99× in every cell (same artifact; that kernel was removed). Q8 therefore uses
pairs by default and F32/F16 keep one head. Prefill above the tiled crossover is unchanged.

## Metal TensorOps

`examples/tensorops_gemm.rs` compiles MetalPerformancePrimitives `matmul2d` kernels at
runtime (language version 4.0) over `tensor_inline` views of existing buffers, so the host
creates no `MTLTensor`: objc2-metal's buffer-backed tensor constructor is unsafe, and is not
used. On the M1 the F32 GEMM matches Candle's results exactly and roughly its speed:

| m × n × k | Candle F32 | TensorOps, 128 × 64 tile, 8 simdgroups | 64 × 64 tile, 4 simdgroups |
| --- | ---: | ---: | ---: |
| 2048 × 8192 × 1024 | 1.70 TFLOPS | 1.76 TFLOPS (1.03×) | 1.39 TFLOPS (0.82×) |
| 128 × 8192 × 1024 | 1.44 TFLOPS | 1.23 TFLOPS (0.86×) | 1.22 TFLOPS (0.85×) |
| 2048 × 1024 × 3584 | 1.55 TFLOPS | 1.67 TFLOPS (1.08×) | 1.56 TFLOPS (1.01×) |

Smaller tiles (32 × 32, 64 × 32) reach 0.36–0.59×, and every configuration's output equals
Candle's exactly ([artifact](baselines/2026-10-24-m1-p10/tensorops-gemm.jsonl); a single
run, so the 128 × 8192 Candle figure differs from an earlier probe's 1.21 TFLOPS).
The M1 has no GPU neural accelerators, so parity is the expected ceiling. The probe is the
starting point for M5 measurements of projections, attention and chunked recurrence;
quantized inputs and cooperative tensors were not evaluated.

## Deferred

- **Grouped-MoE wide tier (item 4).** The P8 gap is measured with synthetic routing; the
  item requires trained routing traces, which need the 35B checkpoint to run, and it does
  not fit in 16 GB. The small Transformers fixture's routing is not representative.
- **F16/BF16 activations and tiles (item 6).** These are quality experiments; P9's held-out
  budgets do not exist yet. Recurrent accumulation stays F32.
- **Crossover retuning (item 3).** The tiled and split-K crossovers were last swept in the
  kernel performance pass on the same device; nothing in this change moves them except the
  grouped kernel's split count, which is derived, not tuned.

## Default dispatch after P10

| Operation | Condition | Kernel |
| --- | --- | --- |
| Recurrent prefill and verification | multi-token sequences; aligned / unaligned `mixed` rows | `delta_prepare` (scales only), then `delta_scan_*` / `delta_scan_unaligned_*` |
| Recurrent decode | single-token sequences | `delta_step` (unchanged) |
| Decode attention, Q8 pools | below the tiled crossover, even query heads per K/V head | `attention_h2_r*_q8` (two heads per simdgroup) |
| Decode attention, F32/F16 pools | below the tiled crossover | `attention_r*` (unchanged) |
| Everything else | | as in the [kernel performance pass](apple-silicon-kernel-performance.md#default-dispatch-after-this-pass) |

Twelve pipelines were added: nine paired-head attention variants and three unaligned
scans. A cold shader compile (new source, empty system cache) of the whole library rises
from 0.71 to 0.84–0.87 s; warm loads are unchanged.

## Validation

- [Workspace tests](baselines/2026-10-24-m1-p10/nextest.log): 235 passed, 8 checkpoint
  tests skipped. The recurrent differential test now runs aligned and unaligned `mixed`
  rows; the paged GQA test runs one and two heads per simdgroup for every K/V format, with
  and without splits, plus a 6:2 head layout that cannot pair and the 999-key decode.
- [Real-checkpoint tests](baselines/2026-10-24-m1-p10/nextest-checkpoint.log): all 8 pass
  on the pinned 0.8B checkpoint (per-layer and logit agreement with Transformers, fused
  versus reference logits and greedy generation, K/V-format quality, quantized coherence,
  MTP head). The golden files' hashes are in the manifest.
- [Shader validation](baselines/2026-10-24-m1-p10/shader-validation.log): the same 13 tests
  fail on the base commit and on this change. Four are the allocation and workspace tests
  the kernel performance pass reported; nine abort because the validation layer caps
  Candle's 1,024-thread 2-D threadgroups at 896, which that pass did not see. Tur's own
  dispatches clamp to each pipeline's limit.
- `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check` and
  `cargo check --workspace --no-default-features` pass. The upstream `block` crate's
  future-incompatibility warning remains.

## Reproduction

```sh
TUR_BENCH_ONLY=delta cargo bench --offline -p tur-metal --bench kernels --profile pre-release > delta.json
TUR_BENCH_ONLY=paged cargo bench --offline -p tur-metal --bench kernels --profile pre-release > paged.json
cargo run --offline --release -p tur-metal --example tensorops_gemm > tensorops-gemm.jsonl
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_FORMATS=q8_0,q4k TUR_BENCH_KV=f16,q8 TUR_BENCH_PAGES=16 \
  TUR_BENCH_PREFILL=2048 TUR_BENCH_CONTEXTS=8192 cargo bench --offline -p tur-engine --bench qwen3_5 --profile pre-release > model.json
```

A base-commit worktree needs its own `CARGO_TARGET_DIR`: with a shared one, Cargo treats
the other tree's identically named artifacts as fresh and runs them. The
[manifest](baselines/2026-10-24-m1-p10/manifest.json) records sources, toolchain,
checkpoint and golden-file hashes.
