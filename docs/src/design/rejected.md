# Rejected experiments

Every row here was built or bounded, measured on the M1, and removed from the tree. Keeping the list stops the same idea from being tried twice without new evidence. Patches remain recoverable from commit `99b073e`:

```sh
git ls-tree -r --name-only 99b073e docs | grep rejected-   # find the patch's path
git show 99b073e:<path> > change.patch
```

| Experiment | Result | Patch |
| --- | --- | --- |
| Chunked (WY/UT) DeltaNet prefill on the matrix units | Correct at every boundary, but 0.41 to 0.66× the speed of the token scan at every shape and batch | `rejected-chunked-delta.patch` |
| 4-bit K/V: groupwise Q4, KIVI-style asymmetric Q4, TurboQuant with and without QJL, 5- and 6-bit variants | Every 4-bit format missed the quality budget (mean KL at most 1e-3, top-1 at least 98% per domain) by 10 to 30×. Values cause most of the error, and each extra bit cuts KL about 4×. The only passing format (6-bit TurboQuant keys, Q8 values) saved just 1.17× over Q8 and decoded 2 to 7% slower | `rejected-low-bit-formats.patch` |
| Batch-invariant execution | Bit-identical across batch size, order and chunking, but 0.52 to 0.60× prefill speed and 0.59 to 0.87× single-sequence decode | `rejected-batch-invariance.patch` |
| Letting cached prefixes jump the queue by default | Cached requests got their first token 5× sooner, but uncached ones waited 15% longer and throughput fell 10%. Kept as the opt-in `engine.prefix_restorable_first` | none |
| Template-rewrite snapshots on first turns | The extra prefill pass made a single uncached request 15% slower to its first token. These boundaries are taken only once a prompt has assistant history | none |
| Carrying a decode-free prefill budget into mixed passes | 4 to 5% slower under contention and churn | `rejected-budget-carry.patch` |
| Prefetched staging over Q8 pools in tiled attention | 9 to 42% slower at every shape, and still 2% slower at 8K with the scale reads replaced by constants, so better scale handling can't win | none |
| Prefetching tiled-attention chunks two steps ahead | Slower than one step at 8K, because of the extra registers | none |
| Overlapping the next pass, submission depth 2, command reuse | Not built: encoding already overlaps the GPU and the GPU idles only tens of microseconds per pass, so the gain is at most 0.3% | none |
| Metal TensorOps for projections | Available on the M1 without unsafe bindings, but F32 GEMM ran at only 0.86 to 1.08× Candle | none |
| Q4K GEMV unpacking each block before the batch loop | 1.6 to 1.8× slower, because the unpacked values cut occupancy | none |
| Dense Q4K GEMV in ggml's layout | About 1.2× Candle's one-row GEMV on small 0.8B matrices that stay in cache, but tied on the output layer and lost on 27B shapes. In real single-sequence decode, which streams weights from memory, it made no difference beyond noise, and it was 2.5 to 3× slower at 2 to 6 rows | none |
| `delta_step` with 8 key rows per simdgroup | Slower (27B, one sequence: 99 µs became 153 µs) | none |
| Transposed DeltaNet scan with 16 or 64 rows per lane | 10 to 22% and 1.45× slower than 32 | none |
| Decode attention with 8 keys per step | Up to 23% slower at short contexts, only 2 to 8% faster at long ones | none |
| Tiled attention with 64-query tiles | 0.86 to 0.89× on long prefill, 0.65× on short chunks | none |
| Four query heads per simdgroup in decode attention | Lost at every shape | none |
| A high-precision K/V window for recent tokens | Not built: Q8 error is flat across positions | none |
| Radix tree for prefix lookup | Not built: the indexed hash lookup costs at most 7 µs at 32K tokens | none |
| TinyLFU-style admission for snapshots | Not built: it would refuse an agent's deeper next-step snapshot in favor of the one it extends | none |
| Kernel-level speculation cost model | Not built: the cost of a draft at eight sequences caps the gain at about 10% | none |
| Learned drafters (EAGLE-3, DFlash, DSpark) and separate draft models for Qwen | No compatible drafter or smaller Qwen3.5 exists for these checkpoints | none |
| Gemma 4 assistant chains with more than one request decoding | With four requests on E2B, against one draft per pass: 6% slower on prose and 13% on code, though 28% faster on repetitive text. Limited by `CHAIN_MAX_DECODES` | none |

## Worth retrying on other hardware

The M1 decided these results. A GPU that is faster relative to its CPU, or one with faster matrix units, may decide otherwise, so re-measure before reusing a patch:

- chunked DeltaNet prefill and TensorOps, on GPUs with strong matrix units such as the M5's neural accelerators;
- pass overlap, where CPU-side encoding could become the bottleneck;
- 4-bit K/V on the 27B, where K/V is a larger share of memory.
