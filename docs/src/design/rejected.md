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
| Running the shortest remaining prompt first by default | At light load, short prompts got their median first token about 4× sooner, but the slowest 5% of token gaps grew by 15 to 50%. Under overload the short-prompt median was no better, and about 18% worse at two arrivals per second. Kept as the opt-in `engine.shortest_prompt_first` | none |
| Carrying a decode-free prefill budget into mixed passes | 4 to 5% slower under contention and churn | `rejected-budget-carry.patch` |
| Other layouts for dense F16 prefill attention, 2048 queries after 32K keys on the 0.8B, against the per-row kernel's 800 ms | Holding the tile's query fragments in registers: 970 ms as F32 and 857 ms as F16. F16 softmax weights for P.V: 853 ms. Loading key and value fragments straight from the pool with no staging or barriers: 1,298 ms, and 1,100 ms with F16 queries and weights. A 2x2 block of score fragments per simdgroup with the softmax in threadgroup memory: 840 ms. Against the 32-row layout `attention_split_f16` settled on (492 ms): 8-key blocks, 593 ms; 64-row tiles of 8 simdgroups, 664 to 677 ms with 32-key blocks and 652 ms with 16-key blocks; declaring the threadgroup size so the compiler could keep more registers changed nothing | none |
| Prefetched staging over Q8 pools in tiled attention | 9 to 42% slower at every shape, and still 2% slower at 8K with the scale reads replaced by constants, so better scale handling can't win | none |
| Prefetching tiled-attention chunks two steps ahead | Slower than one step at 8K, because of the extra registers | none |
| Overlapping the next pass, submission depth 2, command reuse | Not built: encoding already overlaps the GPU and the GPU idles only tens of microseconds per pass, so the gain is at most 0.3% | none |
| Matrix-unit GEMM for Whisper's projections | At Whisper's shapes (1500 rows, 1280 and 5120 wide), Q8_0 tiles reached 0.4 to 1.0 TFLOP/s and F32 tiles 0.2 to 0.8, against Candle's 1.2 to 1.6, across five tile shapes and two to eight simdgroups | none |
| Fused `layer_norm` kernel for Whisper | Slowed the encoder by about 9% at 1500 rows (2.02 s to 2.20 s per window), and showed no gain beyond noise on the decoder's single row | none |
| Q8 cross-attention keys and values in Whisper's decoder | Beside a Q4K head, an 80 s clip took 8.99 and 8.50 s against 8.98 and 8.57 s without them, within noise, for more code than F16 | none |
| Metal TensorOps for projections | Available on the M1 without unsafe bindings, but F32 GEMM ran at only 0.86 to 1.08× Candle | none |
| Q4K GEMV unpacking each block before the batch loop | 1.6 to 1.8× slower, because the unpacked values cut occupancy | none |
| Dense Q4K GEMV in ggml's layout | About 1.2× Candle's one-row GEMV on small 0.8B matrices that stay in cache, but tied on the output layer and lost on 27B shapes. In real single-sequence decode, which streams weights from memory, it made no difference beyond noise, and it was 2.5 to 3× slower at 2 to 6 rows | none |
| `delta_step` with 8 key rows per simdgroup | Slower (27B, one sequence: 99 µs became 153 µs) | none |
| Transposed DeltaNet scan with 16 or 64 rows per lane | 10 to 22% and 1.45× slower than 32 | none |
| Decode attention with 8 keys per step | Up to 23% slower at short contexts, only 2 to 8% faster at long ones | none |
| Tiled attention with 64-query tiles | 0.86 to 0.89× on long prefill, 0.65× on short chunks | none |
| Tiled attention with 64-key blocks | About 15% slower at 2K tokens over F16 pools | none |
| Fewer tiled barriers through 128- or 256-dimension F16 chunks | 128 dimensions showed no gain beyond noise, and 256 was 25 to 45% slower despite two barriers per block instead of eight | none |
| Keeping tiled query fragments in registers across key blocks | 9 to 18% slower than reloading them on prefill from empty, and level on a chunk after 8K | none |
| Staging the query tile as F16 in threadgroup memory | About 15% slower than scaling once per block, and it would round queries | none |
| `fast::exp` in the tiled softmax | No gain beyond noise | none |
| Four query heads per simdgroup in decode attention | Lost at every shape | none |
| A high-precision K/V window for recent tokens | Not built: Q8 error is flat across positions | none |
| Radix tree for prefix lookup | Not built: the indexed hash lookup costs at most 7 µs at 32K tokens | none |
| TinyLFU-style admission for snapshots | Not built: it would refuse an agent's deeper next-step snapshot in favor of the one it extends | none |
| Kernel-level speculation cost model | Not built: the cost of a draft at eight sequences caps the gain at about 10% | none |
| Learned drafters (EAGLE-3, DFlash, DSpark) and separate draft models for Qwen | No compatible drafter or smaller Qwen3.5 exists for these checkpoints | none |
| Per-tenant n-gram follower table beside prompt lookup, as in llama.cpp's `ngram-mod` | Built and run on Qwen3.5-0.8B over 16 mixed requests on an M1: about 1.21× against lookup's 1.19× with one request and 0.967× against 0.970× with four (one round). Too little to justify a tenant-keyed cache to bound and account for | none |
| Bigram follower tables | Replay only: 1.22 to 1.23× against lookup's 1.23 to 1.24× with one request, and below lookup with four | none |
| Token Recycling's chained top candidates | Replay only, without its draft tree: about 0.91× with one request and 0.37× with four, because reading logits back through the logprobs path cost 3.5 to 9.6 ms per row. With free readback it only ties lookup at about 1.24×, and its top-8 candidates held the next token only 42% of the time | none |
| Gemma 4 assistant chains with more than one request decoding | With four requests on E2B, against one draft per pass: 6% slower on prose and 13% on code, though 28% faster on repetitive text. Limited by `CHAIN_MAX_DECODES` | none |
| Plain LRU for an expert cache smaller than one token's experts | On gpt-oss-20b, each token touches 96 experts in layer order, so 40 or 81 slots hit nothing at 1.37 tokens/s. Evicting the expert needed furthest ahead hits 34% at 1.87 tokens/s with 81 slots, but only 65% against LRU's 79% with 324, so LRU stays above the crossover | none |
| Block kernels at every batch size for GGUF formats Candle also reads | At one row 0.41 to 0.85× of Candle's speed and at 128 rows 0.43 to 0.64×, against 1.3 to 1.9× at 2 to 8 rows. Routing each format to the faster kernel replaced it for one-format tensors | none |
| Codebooks staged into threadgroup memory for one-row IQ decode, as ggml does | One-row IQ2 and IQ3 fell from 0.8× of Candle's Q3_K to 0.63×, because each small threadgroup copies the whole table | none |
| Arithmetic MXFP4 decode instead of a lookup table | 1.3 to 2.1× slower at gpt-oss-20b's bank shape, while the table matches Q8_0 | none |
| Reading experts ahead by two or three layers | On gpt-oss-20b with a 1 GiB cache, 160 tokens took 52.5 s and 54.3 s against 51.5 s at one layer, because they read 9% and 13% more bytes for almost no further used experts | none |
| Serving expert reads through the page cache | On gpt-oss-20b with a 1 GiB cache, 160 tokens took 53 to 55 s against 56 s with `F_NOCACHE`, and 51.5 to 52.9 s against 52.2 to 52.4 s with read-ahead also off | none |
| Skipping experts with little routing weight | On gpt-oss-20b at floors of 0.1 and 0.2, KL from the exact model was 0.14 to 0.30 nats and top-token agreement 81 to 91%. Decode gained nothing at 0.1 and 8% at 0.2, though prefetch didn't yet honor the floor | none |
| Evicting snapshots and lowering `max_running` when spare memory runs low | Not built: neither returns memory to macOS. The state pool is zero-filled at startup, so an idle 0.8B server already has a 2,007 MB footprint against 2,005 MB of device allocations, and evicted pages only rejoin the pool's free list. A burst of four requests added 13 MB of allocations after its passes, and gpt-oss-20b plans 1.28 GB of scratch at one request or four. The running limit decides the pool's size at startup, and throttling afterwards would only cost throughput and prefix hits | none |
| Whole-prompt chunks or 256-token chunks behind an expert cache instead of layer groups | On gpt-oss-20b with a 1 GiB cache and a decode after each pass, a 2,048-token prompt in one pass took 16.4 s, but stalled the decode for 16.3 s and left no snapshots inside the prompt. 256-token chunks took 42.6 s with a 6.0 s worst gap, against 22.8 s and 3.2 s for four groups per 1,024-token chunk | none |
| Reading expert misses directly into a shared arena, without staging uploads or blits | On gpt-oss-20b with a 1 GiB cache, 160 tokens took 45.8 s (one run) against 48.7 to 49.5 s, with the same hits and misses and no GPU slowdown. Reads stay limited by the SSD at about 4.5 GB/s, and most experts arrive through read-ahead, which still needs a copy. Candle exposes shared memory only as a raw pointer, so this would have added Tur's first `unsafe` for 6% | none |

## Worth retrying on other hardware

The M1 decided these results. A GPU that is faster relative to its CPU, or one with faster matrix units, may decide otherwise, so re-measure before reusing a patch:

- chunked DeltaNet prefill and TensorOps, on GPUs with strong matrix units such as the M5's neural accelerators;
- pass overlap, where CPU-side encoding could become the bottleneck;
- 4-bit K/V on the 27B, where K/V is a larger share of memory.
