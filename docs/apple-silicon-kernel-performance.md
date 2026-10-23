# Kernel performance pass

2026-10-23, after P8. A review of the Metal kernels found four kernels well below the M1's
measured limits and two with clear causes in the code. This pass fixes them, plus the
recurrent convolution, the MoE expert projections and per-pass index uploads. Every
change was measured against the unchanged base commit on the same Mac, kept only where
it won, and checked by the existing differential and real-checkpoint tests. No public
API, dependency or precision policy changed: activations, accumulation and recurrent
updates stay in F32.

## End-to-end results

Apple M1 (8 GPU cores, 16 GB), pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`, F32 recurrent state, 16-token pages, fused
model passes ending at device completion. Decode steps are per-pass times; context 128
unless noted. [Before](baselines/2026-10-23-m1-kernel-perf/model-before.json),
[after](baselines/2026-10-23-m1-kernel-perf/model-after.json).

| Workload | Q8_0, F16 K/V | Q8_0, Q8 K/V | Q4K, F16 K/V | Q4K, Q8 K/V |
| --- | --- | --- | --- | --- |
| Prefill 2,048 tokens | 1.83 → 1.62 s (1.13×)¹ | 1.83 → 1.60 s (1.14×) | 1.94 → 1.70 s (1.14×) | 1.96 → 1.72 s (1.14×) |
| Decode ×1 | 18.2 → 18.2 ms | 18.4 → 17.9 ms (1.03×) | 11.4 → 10.8 ms (1.05×) | 11.7 → 11.0 ms (1.06×) |
| Decode ×2 | 19.6 → 19.1 ms (1.02×) | 19.5 → 18.7 ms (1.04×) | 20.3 → 17.6 ms (1.16×) | 20.0 → 18.0 ms (1.11×) |
| Decode ×4 | 23.5 → 22.7 ms (1.04×) | 23.9 → 22.3 ms (1.07×) | 26.5 → 20.0 ms (1.32×) | 25.7 → 20.8 ms (1.24×) |
| Decode ×8 | 27.9 → 26.5 ms (1.05×) | 28.2 → 26.2 ms (1.08×) | 30.2 → 23.5 ms (1.29×) | 29.7 → 24.0 ms (1.24×) |
| Decode ×16 | 48.4 → 44.7 ms (1.08×) | 49.3 → 45.3 ms (1.09×) | 56.2 → 44.9 ms (1.25×) | 56.4 → 45.0 ms (1.25×) |
| Decode ×1 at 8K | 20.7 → 20.1 ms (1.03×) | 21.4 → 20.3 ms (1.05×) | 14.3 → 13.0 ms (1.10×) | 15.1 → 13.7 ms (1.10×) |
| Decode ×4 at 8K | 31.6 → 30.5 ms (1.04×) | 34.8 → 30.4 ms (1.14×) | 33.8 → 27.0 ms (1.25×) | 36.6 → 28.5 ms (1.29×) |
| One sequence through `HybridEngine` | 54.8 → 57.3 tok/s | 54.2 → 56.8 tok/s | 80.6 → 91.0 tok/s | 85.2 → 91.1 tok/s |

¹ From an isolated [rerun](baselines/2026-10-23-m1-kernel-perf/model-after-q8_0-f16-rerun.json):
the full matrix measured this first configuration at 1.71 s, with its two samples 3%
apart. Prefill medians use two samples; the other cells agree within a few percent with
an earlier full run of the same matrix.

Single-sequence Q8_0 decode barely moves: its projections already stream weights at
57–66 GB/s, close to the bandwidth the M1 sustains. The gains come from Q4K batches,
recurrent state traffic, Q8 attention and prefill. At 2K prefill the token-serial
DeltaNet scan falls from about 21% to 12% of the pass (18 × 10.4 ms of ~1.6 s), the
convolution to 3% and tiled attention is about 10%. Candle's tiled projection GEMMs are
now roughly two thirds of prefill.

## Changes

Kernel timings below are medians of fifteen completed-work samples after three warmups,
from the [kernel artifacts](baselines/2026-10-23-m1-kernel-perf/).

### Q4K small-batch projections

`qmm_q4k_t{1,2}_f32` replaces the shared per-format path for Q4K on the matrix units. A
simdgroup step covers one 256-value super-block: its header and twelve scale bytes load
once into registers, the scale and minimum of each sub-block decode with constant shifts,
and each 8-byte quant load serves two sub-blocks (low nibbles, then high). The previous
path decoded the scales and re-read every quant byte for each 32-value chunk, so the
kernel was bound by dequantization, not memory. Threadgroups now cover 32 output columns
(16 for Q8_0). The arithmetic is unchanged: F32 dequantization and accumulation.

| Shape (n × k) | 1–8 rows, before | 1–8 rows, after | 16 rows, before → after | Candle tiles, 2–16 rows |
| --- | ---: | ---: | ---: | ---: |
| 0.8B `in_proj` 8224 × 1024 | 212–216 µs | 140–145 µs | 408 → 319 µs | 357–362 µs |
| 0.8B `gate_up` 7168 × 1024 | 185–190 µs | 126–134 µs | 350 → 273 µs | 311–318 µs |
| 0.8B `down` 1024 × 3584 | 100–110 µs | 62–64 µs | 190 → 162 µs | 205–213 µs |
| 0.8B LM head 248320 × 1024 | 6.11–6.28 ms | 3.74–4.07 ms | 12.24 → 9.54 ms | 9.6–10.1 ms |
| 27B `gate_up` 34816 × 5120 | 4.15–4.22 ms | 2.50–2.79 ms | 7.99 → 5.98 ms | 6.6–6.7 ms |
| 27B `down` 5120 × 17408 | 2.12–2.18 ms | 1.29–1.65 ms | 4.03 → 3.01 ms | 3.51–3.54 ms |

The kernel now beats the Q4K GEMV from two rows (`in_proj` 140 against 168 µs, LM head
3.83 against 4.87 ms) and Candle's F16 tiles at every count up to sixteen, which it
previously lost by 10–25%. Q4K therefore uses it for 2–16 rows (`qmm_q4k_rows: (2, 16)`);
the Q4K GEMV stays reachable through tuning overrides only. Candle's `mul_mv` keeps one
row and its F16 tiles keep more than sixteen.
[Before](baselines/2026-10-23-m1-kernel-perf/linear-before.json),
[after](baselines/2026-10-23-m1-kernel-perf/linear-after.json). Q8_0 timings are
unchanged within noise.

### Recurrent decode step

`delta_step` simdgroups now own 16 key rows instead of 32, which halves each lane's
state registers (32 instead of 64) and doubles the simdgroups resident to hide memory
latency. The layout and reductions are otherwise unchanged. A 32-row variant is
compiled as well; the host selects it only where a pipeline cannot run the larger
threadgroup (up to 512 threads at key dimension 256), as under Metal shader validation.

| Sequences | 0.8B, before → after | 27B shapes, before → after |
| ---: | ---: | ---: |
| 1 | 57 → 37 µs | 164 → 92 µs |
| 4 | 214 → 142 µs | 675 → 505 µs |
| 8 | 476 → 356 µs | 1,348 → 960 µs |
| 16 | 955 → 675 µs | 2,621 → 1,900 µs |

State traffic rises from 35–43 GB/s to 47–53 GB/s at 4–16 sequences. The 57–68 GB/s at
one or two sequences is partly served from the system cache, since the benchmark
re-reads the same state.

### DeltaNet prefill scan

The token-serial scan is redesigned; chunked prefill (P10) remains open.

- **`delta_prepare_f32`**, one simdgroup per (token, key head), writes per-token inputs to
  a scratch row: L2-normalised q (also scaled by 1/√dk) and k, q·k per key head, and
  decay and beta per value head. Previously every state-column simdgroup of a head
  recomputed the norms and the exp/softplus/sigmoid chain inside the sequential loop.
- **`delta_scan_{f32,f16,bf16}`** is transposed: a lane owns one value column and 32 key
  rows, so dk / 32 lanes share a column. With `a = k·S` and `b = q·S` over the
  decayed-before state, `out = decay·b + (q·k)·corr` equals `q·S'`, so one pass over the
  lane's state gives both dot products. They combine across a column's lanes with at
  most three shuffles, where the old layout needed ten simdgroup reductions per token.
  No threadgroup memory or barrier remains, and per-token speculative state stores are
  row segments across a simdgroup's columns instead of 16-byte pieces of cache lines.

| Shape | Tokens | Before | After |
| --- | ---: | ---: | ---: |
| 0.8B | 128 | 1.34 ms | 0.56 ms (2.4×) |
| 0.8B | 2,048 | 21.0 ms | 10.4 ms (2.0×) |
| 27B | 128 | 3.89 ms | 1.43 ms (2.7×) |
| 27B | 2,048 | 65.1 ms | 28.1 ms (2.3×) |

The scratch row is charged in the fused workspace plan.
[Before](baselines/2026-10-23-m1-kernel-perf/delta-before.json),
[after](baselines/2026-10-23-m1-kernel-perf/delta-after.json).

### DeltaNet convolution

`delta_conv4_f32` handles four channels per thread with vector loads and stores, a
quarter of the span lookups and row divisions, and the same per-channel arithmetic. The
host routes to it when the channel count, row stride and offsets keep every float4
aligned, and otherwise keeps the scalar kernel. 0.8B: 3.87 → 2.56 ms at 2,048 tokens,
205 → 149 µs at 128 and 30 → 18 µs at eight decoding sequences; 27B: 6.80 → 5.05 ms at
2,048 tokens.

### Decode attention

The per-query kernel (decode, and prefill below the tiled crossover) now has each lane
load 16-byte vectors, a lane's successive vectors 32 vectors apart, so every load
instruction covers one contiguous row segment. At head dimension 256 that is one load per
row for F16 or Q8 (with one scale per vector) instead of two to four. Keys are taken
four per step: one page-table lookup with 32-bit row arithmetic serves a step inside a
page, all four rows' loads are outstanding together, and the online softmax rescales once
per step. The host now rejects pools whose slot × K/V-head row count exceeds 2^32.

| Paged decode, page 16 | F32 | F16 | Q8 group 32 |
| --- | ---: | ---: | ---: |
| 2K context | 165 → 164 µs | 104 → 79 µs | 153 → 109 µs |
| 8K context | 711 → 688 µs | 426 → 419 µs | 583 → 426 µs |
| 32K context | 2,447 → 2,619 µs | 1,381 → 1,298 µs | 1,966 → 1,317 µs |
| 2K context × 8 sequences | 1,226 → 1,279 µs | 676 → 655 µs | 963 → 670 µs |

Q8 is 1.37–1.50× faster at every context and now costs the same as F16. F32 ranges from
0.90× to 1.03× (worst at 32K and in one 2K case). Q8 still does not reach its halved
byte count, so it remains bound by per-key work rather than bandwidth.
[Before](baselines/2026-10-23-m1-kernel-perf/paged-before.json),
[after](baselines/2026-10-23-m1-kernel-perf/paged-after.json).

The faster per-query kernel moves the tiled crossover only for F16 at 64 queries
(per-query 0.063 against tiled 0.087 ms). Raising the threshold would also reroute
"chunk 64 after 2048", where tiled is 3× faster, so the
[tiled dispatch policy](apple-silicon-tiled-attention.md) is unchanged.
[Tiled sweep](baselines/2026-10-23-m1-kernel-perf/tiled-before.json).

### MoE expert projections

Flagship gate/up shape (256 experts, top-8, 2048 → 1024), synthetic routing as in the
[P8 report](apple-silicon-p8-speculation-moe.md#expert-projection-performance-gate-open);
grouped timings include routing.

- **Q4K GEMV tier (groups of one to three rows)** uses ggml's `mul_mv_q4_K` lane layout:
  quants are masked in place from 16-bit words, with each nibble's 16/256/4096 multiple
  folded into per-sub-block sums; scales and minima unpack from three 16-bit words; and
  four outputs per simdgroup share each lane's activations.
- **GEMM tier (four to sixteen rows)** loads Q4K one super-block at a time as `qmm_q4k`
  does, and multiplies one activation fragment instead of two when a group has at most
  eight rows.
- **Unreachable tiers are skipped.** A token selects each expert at most once, so with at
  most three tokens no group reaches the GEMM tier, and with at most sixteen none reaches
  the wide tier.

| Tokens | Routing | Q4K grouped, before → after | Q4K individual | Q8_0 grouped, before → after | Q8_0 individual |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Dispersed | 0.309 → 0.202 ms | 0.205 ms | 0.345 → 0.337 ms | 0.345 ms |
| 8 | Dispersed | 2.27 → 1.40 ms | 1.58 ms | 2.45 → 2.47 ms | 2.80 ms |
| 8 | Concentrated | 0.941 → 0.401 ms | 0.355 ms | 0.682 → 0.538 ms | 0.427 ms |
| 32 | Dispersed | 7.42 → 4.90 ms | 5.04 ms | 6.76 → 6.54 ms | 7.01 ms |
| 128 | Dispersed | 19.2 → 9.74 ms | 9.44 ms | 14.7 → 12.2 ms | 12.2 ms |
| 128 | Concentrated | 4.71 → 4.35 ms | 2.85 ms | 3.83 → 3.76 ms | 2.54 ms |
| 2,048 | Dispersed | 83.4 → 82.6 ms | 57.3 ms | 72.6 → 73.5 ms | 50.3 ms |

Q4K single-token decode now matches Candle's dense GEMV instead of trailing it by 1.5×.
Groups above sixteen rows (the wide tier) are unchanged and remain 1.2–1.5× behind
Candle's per-expert tiled GEMM, so the P8 performance gate stays open. The "before"
artifact already includes the tier skip; its kernels are the base commit's.
[Before](baselines/2026-10-23-m1-kernel-perf/moe-before.json),
[after](baselines/2026-10-23-m1-kernel-perf/moe-after.json).

### Per-pass index uploads

Each `Tensor::from_slice` upload creates a new `MTLBuffer` and commits the residency set
(Candle does not pool these), and a pass's addressing is uploaded before its first command
buffer commits, while the GPU waits. Page addressing (table, first block, visible keys,
query tiles) and recurrent addressing (spans, rows, slots and three order lists) now go up
as one buffer each, viewed in place: ten uploads become two. This was not timed separately.

## Default dispatch after this pass

| Operation | Rows or condition | Kernel |
| --- | --- | --- |
| Q8_0 projection | 1 / 2–3 / 4–16 / >16 | Candle `mul_mv` / `qmv_q8_0` / `qmm_q8_0` / Candle F16 tiles (unchanged) |
| Q4K projection | 1 / 2–16 / >16 | Candle `mul_mv` / `qmm_q4k` / Candle F16 tiles |
| Recurrent decode | single-token sequences | `delta_step`, 16 key rows per simdgroup (32 if the pipeline cannot fit it) |
| Recurrent prefill and verification | multi-token sequences | `delta_prepare` then the transposed `delta_scan` |
| Convolution | aligned float4 rows / otherwise | `delta_conv4` / `delta_conv` |
| Attention | below / above the tiled crossover | per-query `attention_r*` / `attention_tiled_r*` (policy unchanged) |
| MoE experts | ≤3 / 4–16 / >16 rows per expert | GEMV (Q4K in ggml layout) / matrix units / wide F16-staged tiles |

## Rejected variants

All measured on the same M1 and not kept; artifacts are in the
[baseline directory](baselines/2026-10-23-m1-kernel-perf/).

| Variant | Result |
| --- | --- |
| Q4K GEMV unpacking each row's nibbles once per block, before the batch loop | 1.6–1.8× slower at every row count (`in_proj`, one row: 105 → 184 µs): 64 unpacked floats per lane cut occupancy ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-qmv-q4k-hoisted-unpack.json)) |
| Dense Q4K GEMV in ggml's layout | One row beats Candle on 0.8B shapes (`in_proj` 70 against 82 µs, LM head 2.37 against 2.50 ms) but loses 25% on 27B `down` (1.10 against 0.87 ms) and is 2.5–3× slower at 2–6 rows. Routing it by shape would need a new public tuning field ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-qmv-q4k-ggml-layout.json)) |
| `qmm_q4k` with 16 columns per threadgroup | 5–9% slower than 32 on most shapes; 32 loses 5–8% only on 27B `down`, outweighed by 27B `gate_up` (in the hoisted-unpack artifact above) |
| `delta_step` with 8 key rows per simdgroup | Slower (27B, one sequence: 99 → 153 µs) and needs 1,024 threads at key dimension 256 ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-delta-step-8-rows.json)) |
| Transposed scan with 64 rows per lane | 30.5 ms at 2K on 0.8B, 1.45× slower than the original: register pressure ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-delta-scan-64-rows.json)) |
| Transposed scan with 16 rows per lane | 10–22% slower than 32: one more shuffle step per reduction ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-delta-scan-16-rows.json)) |
| Decode attention with 8 keys per step | Up to 23% slower at short F32 contexts, 2–8% faster at long ones ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-attention-8-keys-per-step.json)) |
| Tiled attention with 64-query tiles (8 simdgroups) | 0.86–0.89× on 512–8,192-token prefill and 0.65× on four 32-token chunks; faster only around 64 queries ([artifact](baselines/2026-10-23-m1-kernel-perf/rejected-tiled-64-query-tiles.json)) |

## Validation

- [Workspace tests](baselines/2026-10-23-m1-kernel-perf/nextest.log): 235 passed, 8
  checkpoint tests skipped. They include the differential kernel tests (Q8_0/Q4K
  projections at 1–16 rows with partial tiles and strided inputs, the delta step and scan
  over ragged, forked and speculative sequences at key dimensions 64/128/256 in F32/F16/BF16
  state, paged attention for every format, split and GQA ratio, and grouped experts under
  every routing skew) and the fused-model and MoE fixture regressions.
- [Real-checkpoint tests](baselines/2026-10-23-m1-kernel-perf/nextest-checkpoint.log): all
  8 pass on the pinned 0.8B checkpoint, including per-layer and logit agreement with
  Transformers, fused-versus-reference logits and greedy generation, K/V-format quality,
  coherent quantized generation and the MTP head.
- [Shader validation](baselines/2026-10-23-m1-kernel-perf/shader-validation.log)
  (`MTL_SHADER_VALIDATION=1`): the same four allocation and workspace tests fail as on the
  base commit, whose measurements the validation layer's own allocations change; no other
  test fails. Validation lowers pipeline thread limits, which is what exposed the need for
  the 32-row `delta_step` fallback.
- `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check` and
  `cargo check --workspace --no-default-features` pass. The upstream `block` crate's
  future-incompatibility warning remains.

Numerical order changes are covered by those tolerances: the scan's output identity
(algebraically exact), four-key softmax steps, and ggml's summation order in the Q4K
expert GEMV. Q4K dequantization in the matrix-unit kernels is arithmetically unchanged.

## Remaining limits

- **Prefill GEMMs** are now about two thirds of 2K prefill, in Candle's F16-tile kernel at
  roughly 1.6 TFLOPS. Only a custom tiled GEMM or lower-precision activations could move
  them; neither is attempted here.
- **Chunked DeltaNet prefill (P10)** competes with a 10.4 ms scan at 2K tokens, not 21 ms.
  At 12% of prefill, removing the scan entirely would give at most about 1.14×.
- **Tiled prefill attention** runs at about 25% of the M1's F32 matrix-unit peak: staging
  is single-buffered with sixteen barriers per 32-key block at head dimension 256.
- **MoE wide tier** (more than sixteen rows per expert) trails Candle by 1.2–1.5×.
- **Single-sequence decode** spends roughly a quarter of a Q8_0 pass outside projection
  weight streaming, consistent with about 10 µs per dependent dispatch. Fusing dispatches
  could recover a few percent; nothing is claimed without a profiler-backed breakdown.
- Only the M1 is measured; launch shapes remain per-device tuning starting points.

## Reproduction

```sh
TUR_BENCH_ONLY=linear cargo bench --offline -p tur-metal --bench kernels --profile pre-release > linear.json
TUR_BENCH_ONLY=delta cargo bench --offline -p tur-metal --bench kernels --profile pre-release > delta.json
TUR_BENCH_ONLY=paged cargo bench --offline -p tur-metal --bench kernels --profile pre-release > paged.json
TUR_BENCH_ONLY=moe TUR_BENCH_MOE=q8_0,q4k cargo bench --offline -p tur-metal --bench kernels --profile pre-release > moe.json
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_FORMATS=q8_0,q4k TUR_BENCH_KV=f16,q8 TUR_BENCH_PAGES=16 \
  TUR_BENCH_PREFILL=2048 TUR_BENCH_CONTEXTS=8192 cargo bench --offline -p tur-engine --bench qwen3_5 --profile pre-release > model.json
cargo nextest run --offline --cargo-profile pre-release --workspace --test-threads 1
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 cargo nextest run --offline --cargo-profile pre-release -p tur-metal -p tur-engine --test-threads 1 --no-fail-fast
```

The [manifest](baselines/2026-10-23-m1-kernel-perf/manifest.json) records the base
commit, source hashes, toolchain and checkpoint revision.
