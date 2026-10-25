# Rejected experiments

These experiments were built or bounded on the M1 and rejected as defaults. The
partial-model pilot retains its research tooling. Older deleted implementations
remain recoverable from commit `99b073e`:

```sh
git ls-tree -r --name-only 99b073e docs | grep rejected-   # find the patch's path
git show 99b073e:<path> > change.patch
```

| Experiment | Result | Patch |
| --- | --- | --- |
| Permanent low-influence recurrent-layer removal from Qwen3.5-0.8B, without recovery | Keeping 22/20/18 layers removes 5.7/11.5/17.2% of text parameters. Every candidate fails a code-generation smoke check; full-model Q4K uses less resident weight memory and has lower KL on all four small held-out samples. [Protocol and measurements](../research/partial-model-pilot.md). Research tooling retained; no runtime default changed | — |
| Activation-energy MLP width pruning of Qwen3.5-0.8B, without recovery | Retaining 75/50/25% MLP width keeps every layer and removes 8.8/17.6/26.3% of text parameters. All candidates fail the code-generation smoke check; full Q4K is smaller in resident weight bytes and has lower KL in every sample. [Protocol and measurements](../research/partial-model-pilot.md). Research tooling retained | — |
| Chunked (WY/UT) DeltaNet prefill on the matrix units | Correct at every boundary, but 0.41–0.66× the token scan at every shape and batch | `rejected-chunked-delta.patch` |
| Low-bit K/V: groupwise Q4, KIVI-style asymmetric Q4, TurboQuant with and without QJL, 5/6-bit variants | Every 4-bit format missed the quality budget (mean KL ≤ 1e-3, top-1 ≥ 98% per domain) by 10–30×. Values dominate the error, and each extra bit cuts KL about 4×. The only passing format (6-bit TurboQuant keys, Q8 values) saved 1.17× over Q8 and decoded 2–7% slower | `rejected-low-bit-formats.patch` |
| Batch-invariant execution | Bit-identical across batch size, order and chunking (default kernels differ in 11 of 15 comparisons), but 0.52–0.60× prefill and 0.59–0.87× single-sequence decode | `rejected-batch-invariance.patch` |
| Admitting restorable prefixes first by default | Warm TTFT 5.4× lower, but cold TTFT +15% and throughput −10% under contention: it slows unrelated requests. Kept as the opt-in `engine.prefix_restorable_first` | — |
| Template-divergence snapshots on first turns | Cold single-request TTFT +15% (136 → 156 ms) from the extra prefill pass. Predict boundaries only once a prompt contains assistant history | — |
| Carrying a decode-free prefill budget into mixed passes | 4–5% slower under contention and churn | `rejected-budget-carry.patch` |
| Prefetched staging over Q8 pools in tiled attention | 9–42% slower at every shape; with the scale reads replaced by constants still 2% slower at 8K, so better scale handling cannot win | — |
| Prefetching tiled-attention chunks two steps ahead | Slower than one step at 8K (F16: 393 against 379 ms; F32: 402 against 376), from the extra registers | — |
| Next-pass overlap, submission depth 2, command reuse | Not built: encoding already overlaps the GPU and the device idles 28–38 µs per pass, so the gain is bounded at about 1.003× | — |
| Metal TensorOps for projections | Available on the M1 without unsafe bindings, but F32 GEMM ran at only 0.86–1.08× Candle | — |
| Q4K GEMV unpacking each block before the batch loop | 1.6–1.8× slower: unpacked values cut occupancy | — |
| Dense Q4K GEMV in ggml's layout | 1.18–1.21× Candle's one-row GEMV on 0.8B `in_proj` and `gate_up` in the kernel benchmark, where their 4–5 MB of weights stay cached between repetitions; ties on the 143 MB LM head and loses on 27B shapes. Routed to one-row projections with at most 1024 inputs, single-sequence Q4K decode did not change beyond noise (−3 to −4%, Q8_0 control −3 to +2%): decode streams its weights from memory, where both kernels are bandwidth-bound. 2.5–3× slower at 2–6 rows | — |
| `delta_step` with 8 key rows per simdgroup | Slower (27B, one sequence: 99 → 153 µs) | — |
| Transposed DeltaNet scan with 16 or 64 rows per lane | 10–22% and 1.45× slower than 32 | — |
| Decode attention with 8 keys per step | Up to 23% slower at short contexts, 2–8% faster at long ones | — |
| Tiled attention with 64-query tiles | 0.86–0.89× on long prefill, 0.65× on short chunks | — |
| Four query heads per simdgroup in decode attention | Lost at every shape | — |
| Recent high-precision K/V window | Not built: Q8 error is flat across positions | — |
| Radix tree for prefix lookup | Not built: indexed hash lookup costs ≤ 7 µs at 32K tokens | — |
| TinyLFU-style admission for snapshots | Not built: it would refuse an agent's deeper next-step snapshot in favor of the one it extends | — |
| Kernel-tier speculation cost model | Not built: bounded at about 10% by the cost of a draft at eight sequences | — |
| Learned drafters (EAGLE-3, DFlash, DSpark), draft models | No compatible drafter or smaller Qwen3.5 exists for these checkpoints | — |

## Retry on different hardware

The M1 decided these results; a GPU that is faster relative to its CPU, or one with
faster matrix units, may decide otherwise. Re-measure before reusing a patch:

- chunked DeltaNet prefill, and TensorOps, on matrix-unit-rich GPUs (M5's neural
  accelerators);
- pass overlap, where host encoding could reach the critical path;
- low-bit K/V on the flagship, where K/V is a larger share of memory.
