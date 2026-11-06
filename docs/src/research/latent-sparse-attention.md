# Latent sparse attention

Created: 2026-11-05. Status: steps 1 and 2 of the [plan](#plan) are measured on Qwen3.5-0.8B (decode cost on the M1, quality in Transformers with a training-free map), and both cleared their gates. Steps 3 to 5 are built: the index kernels match the reference implementation exactly on its golden fixture, the engine keeps index blocks in the state pool, `tur sparse-map` fits maps natively, and `--sparse-attention auto` decodes the 0.8B at 128K tokens 2.2× faster. The retrieval budget (step 6) is still open. The plan starts with the 0.8B, then moves to Qwen3.8-27B with the paper's released index maps.

Decoding one token reads every key and value in every full attention layer, so at long contexts the K/V reads outgrow the weight reads and decode slows down with each token of context. Latent Sparse Attention ([Prefrontal-AI, arXiv 2610.42147](https://arxiv.org/abs/2610.42147), [code](https://github.com/Prefrontal-AI/latent-sparse-attention), Apache-2.0) caps those reads. It groups keys into small clusters once, and each query scores the clusters before reading only the keys of the best ones. Attention over the keys it picks stays exact, with the model's own kernel. This page records what the method does, where it would pay in Tur, and how to build it.

## How it works

The index lives in the position-free part of the attention score. Qwen rotates the first 64 of each head's 256 dimensions, and the other 192 contribute to the score independently of position, so clustering on them groups keys by content. A key's index input concatenates those 192 dimensions from every K/V head (768 on Qwen3.8-27B, 384 on the 0.8B). A query's index input sums the query heads of each K/V group and concatenates the groups the same way, so one key list serves every head of the layer. Both are L2-normalized, mapped to 64 dimensions by the layer's index projection, and normalized again.

Keys are indexed in blocks of 512, once each block is complete:

- Groups of 4 consecutive index keys are summed into one pool, which gives 128 pools per block.
- Ward's agglomerative clustering merges the pools into 64 buckets of at most 16 keys. Merging two buckets costs `|a| + |b| - |a + b|` over their key sums. Each round, every bucket finds its cheapest allowed partner (ties go to the lower index), and every pair that picked each other merges, cheapest first in the round that reaches 64. Nothing is random, so the result is deterministic.
- The index stores each bucket's normalized key sum as its centroid, plus which pools it holds. A block's buckets never change after that.

A decoding query scores every centroid of the complete blocks before its own, keeps the best `p = 1024` buckets (ties to the lower bucket), and reads their keys best bucket first until it has `k = 4096` long-range keys. Keys in the sinks or the window don't count toward `k`, because the query reads those anyway. Attention then runs over the long-range keys, the 512 most recent keys and 4 sink keys at the start: about 4,612 keys per query at any context length. The paper's decode kernel picks buckets with a threshold estimated from a sample of scores rather than a full sort.

The projection comes from one of three sources:

| Variant | Projection | Cost to obtain |
| --- | --- | --- |
| D-LSA, distilled | Trained against the frozen model's attention, with a softmax temperature | About 352M tokens of long documents. Released for Qwen3.8-27B and GLM-5.3-Flash |
| D-LSA, training-free | The leading principal directions of the layer's keys | One pass over calibration text |
| N-LSA | None; the model is trained with sparse attention from scratch | Pretraining, so out of scope for Tur |

The paper evaluates D-LSA on Qwen3.8-27B, the same hybrid checkpoint Tur pins, at contexts up to 10M tokens with YaRN. Up to 1M tokens it stays within 5 RULER points of full attention, and at 10M it scores 63.9 against full attention's 59.9. MMLU-Pro, GPQA-Diamond, AIME25 and MATH500 stay within their confidence intervals. The training-free map takes each layer's top 64 principal directions of the index input over eight long documents. On Qwen3.8-27B it scores close to the distilled map in the served tests, though offline its picks hold less of the dense attention weight at 1M tokens (0.23 against 0.30). On GLM-5.3-Flash it trails the distilled map. The speedups are reported only at very long contexts on H200 and B300: on Qwen at 10M tokens, time per output token is 2.9× better end to end and attention 42.5× faster per layer. The paper gives no decode timings between 32K and 256K and no break-even point against dense attention. All of these numbers are the authors' own and haven't been reproduced.

## Where it pays in Tur

Qwen3.5, 3.6 and 3.8 place one full attention layer after every three Gated DeltaNet layers. DeltaNet state doesn't grow with context, so only the attention layers would change. Their K/V read per token of context, with F16 K/V and the pinned configs:

| Model | Attention layers | K/V heads | K/V per context token |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 6 | 2 | 12 KiB |
| Qwen3.6-35B-A3B | 10 | 2 | 20 KiB |
| Qwen3.8-27B | 16 | 4 | 64 KiB |

On the 0.8B, the time a decode step gains over a 128-token context is almost all attention, since nothing else in the model grows with context. Measured with `just bench model` on the M1, Q8_0 weights, F16 K/V, one sequence:

| Context | Step | Tokens/s | Context-dependent time | Share of the step |
| --- | --- | --- | --- | --- |
| 128 | 16.8 ms | 59.4 | | |
| 8K | 20.7 ms | 48.3 | 3.9 ms | 19% |
| 32K | 26.3 ms | 38.1 | 9.5 ms | 36% |
| 64K | 32.1 ms | 31.2 | 15.3 ms | 48% |
| 128K | 45.2 ms | 22.1 | 28.4 ms | 63% |

Between 64K and 128K the extra K/V reads run at about 61 GB/s, near the M1's peak, so dense attention has little left to gain from kernel work. With reads capped at about 4,612 keys, a step should cost about what it does at a 5K context plus the selection, around 18 to 19 ms at any context: roughly 1.4× at 32K, 1.7× at 64K and 2.4× at 128K, if selection costs what it should. At 8K, LSA can't help, because 4,612 keys is more than half the context. The default pool also plans 8K tokens per request, so LSA only matters once `context_tokens` is raised.

A bandwidth estimate for Qwen3.8-27B with Q4K weights (about 15 GB read per step), one sequence, F16 K/V, not measured:

| Context | Dense bytes per step | With LSA | Ceiling on the speedup |
| --- | --- | --- | --- |
| 32K | 17.1 GB | 15.3 GB | 1.1× |
| 128K | 23.6 GB | 15.3 GB | 1.5× |
| 256K | 32 GB | 15.3 GB | 2.1× |

Batches gain more, because the weights are read once for the batch and the K/V once per sequence: four sequences at 64K would read 32 GB dense and 16 GB with LSA. The estimate assumes attention reads memory as fast as the projections do. If it reads slower, which the 0.8B numbers suggest on the M1, the real gain is larger.

`TUR_BENCH_ONLY=lsa just bench kernels` times one attention layer of the 0.8B (8 query heads, 2 K/V heads, F16 pool) on the M1 with the paper's budget. Clustering costs 0.37 ms per block, which decode pays once every 512 tokens; prefill, clustering many blocks at once, pays 93 µs each:

| Context, sequences | Dense attention | Selection | Sparse attention | Sparse total | Speedup |
| --- | --- | --- | --- | --- | --- |
| 32K, 1 | 1.29 ms | 0.22 ms | 0.24 ms | 0.46 ms | 2.8× |
| 64K, 1 | 2.45 ms | 0.23 ms | 0.24 ms | 0.47 ms | 5.2× |
| 128K, 1 | 4.78 ms | 0.37 ms | 0.25 ms | 0.62 ms | 7.7× |
| 32K, 4 | 5.35 ms | 0.33 ms | 0.74 ms | 1.07 ms | 5.0× |
| 64K, 4 | 9.73 ms | 0.57 ms | 0.77 ms | 1.34 ms | 7.3× |

Six attention layers at 4.78 ms account for the 28.4 ms that context adds to a 128K decode step, so with sparse attention the step should fall from 45 ms to about 21 ms. Clustering adds about 2.2 ms to the one step in 512 that completes a block on all six layers.

Through the whole model (`TUR_BENCH_SPARSE_MAP=<map> just bench model`), with the map `tur sparse-map` fitted on four books, Q8_0 weights, F16 K/V, one sequence on the M1:

| Context | Dense step | Sparse step | Tokens/s, dense | Tokens/s, sparse | Speedup |
| --- | --- | --- | --- | --- | --- |
| 32K | 25.7 ms | 21.8 ms | 38.9 | 46.0 | 1.18× |
| 64K | 32.6 ms | 20.9 ms | 30.7 | 48.0 | 1.56× |
| 128K | 46.2 ms | 21.1 ms | 21.7 | 47.5 | 2.19× |

The sparse step stays near 21 ms from 64K on, as the kernel measurements predicted. These steps don't close a block, so the clustering stall once every 512 tokens isn't in them.

Through the serving pipeline (`TUR_BENCH_WORKLOADS=long just bench serving`), with prompts prefilled cold, greedy decoding and 256 output tokens per request, and `auto` reading prefill sparsely too:

| Prompt, requests | Inter-token p50, dense | Inter-token p50, sparse | First token, dense | First token, sparse | Output tok/s, dense | Output tok/s, sparse |
| --- | --- | --- | --- | --- | --- | --- |
| 32K, 1 | 25.3 ms | 21.2 ms | 61.3 s | 39.4 s | 3.8 | 5.7 |
| 32K, 4 | 44.0 ms | 29.1 ms | 229 s | 152 s | 3.4 | 5.0 |
| 64K, 1 | 32.4 ms | 21.8 ms | 194 s | 79.7 s | 1.3 | 3.0 |
| 64K, 4 | 78.2 ms | 31.9 ms | 737 s | 308 s | 1.4 | 2.9 |

The decode gain grows with concurrency, because every request adds its own K/V reads while the weights are read once per pass: four requests at 64K decode faster than one did dense. The first-token gain grows with the prompt, because dense prefill attention grows with the square of the prompt while sparse prefill reads a fixed budget past 16K. With sparse decode alone, the first token came up to 3% later than dense, from clustering every block during prefill.

### Sparse prefill

Dense prefill attention grows with the square of the prompt, so past 32K it dominates time to first token. Sparse prefill gives the same budget to each tile of 32 consecutive prompt queries past the crossover. The tile scores buckets with the sum of its rows' normalized index queries, which ranks buckets by the mean of the rows' scores, and the tiled attention kernel reads the tile's list: the selected keys before the first row's window, the sinks, then every key from that window's start to the last row. Row `r` of the tile reads the list up to its own position, so causality holds and the matrix units still share each staged key across the tile's 32 queries. Lists per query would give up that sharing, since each query would stage its own keys.

`auto` includes it. It changes every prompt token's hidden state rather than only the generated ones, so errors could compound through the layers; the needle check below and the E2B check found no loss, and `SparseAttention::prefill` turns it off for experiments. One 2048-token chunk after a long context, on the 0.8B with Q8_0 weights and F16 K/V on the M1 (`TUR_BENCH_ONLY=chunk just bench model`):

| Context before the chunk | Dense | Sparse prefill | Speedup |
| --- | --- | --- | --- |
| 32K | 5.91 s (346 tokens/s) | 2.48 s (825 tokens/s) | 2.4× |
| 64K | 11.30 s (181 tokens/s) | 2.79 s (733 tokens/s) | 4.0× |

With sparse prefill as well, the needle check below answered 23 of its 24 questions correctly against dense attention's 22, and 22 answers matched dense attention's text exactly. One of the two that differ is a 16K question dense attention got wrong; the other has the right code and continues differently.

## Which models

The method needs no change to model weights, only an index projection per attention layer. Qwen3.8-27B can use the released distilled maps. Qwen3.5-0.8B and Qwen3.6-35B-A3B need training-free maps from a calibration pass, which nobody has evaluated on models that size: a small model's keys may cluster less cleanly, which is why the quality check below comes before any kernel work. Gemma 4's full attention layers rotate only a quarter of each 512-dimension head, with proportional rotation: dims `i` and `i + 256` rotate together for `i` below 64. Their position-free dims are two runs of 192, `[64, 256)` and `[320, 512)`, which `IndexMap::split_rotary` describes, so Gemma 4 takes training-free maps too. Its sliding window layers read at most the window and stay dense. On E2B and E4B, the full layers past the K/V-sharing boundary read an earlier full layer's keys: only the owner clusters them, and the sharer selects through the owner's map and index. gpt-oss rotates every dimension, so its keys have no position-free part to index; it would need unrotated keys stored beside the rotated ones.

On Gemma 4 E2B (Q8_0 weights, F16 K/V, M1), with a map `tur sparse-map` fitted on the same four books (its three K/V-owning full layers, in 7 minutes):

| Measure | Dense | LSA, sparse prefill | Speedup |
| --- | --- | --- | --- |
| Decode step at 32K | 57.3 ms | 54.3 ms | 1.06× |
| Decode step at 64K | 66.0 ms | 51.3 ms | 1.29× |
| 2048-token prefill chunk after 32K | 10.92 s | 2.54 s | 4.3× |

E2B decodes a short context in 47 ms, almost all of it weights, so context adds only 10 ms at 32K and 19 ms at 64K; sparse reads cut it to 4 to 8 ms, within run-to-run noise of each other. The prefill gain is larger than the 0.8B's at the same context: E2B's full layers have 512-dimension heads, twice Qwen's, so each key costs dense prefill twice the work. A 2048-token chunk after 64K ran out of the M1's 16 GB with dense attention, so the 64K rows cover decode only. The needle check with sparse prefill answered all 16 questions at 16K and 32K correctly in both modes, with identical text.

## Design

### Index state

An index block covers 512 tokens, which is 32 of Tur's 16-token [pages](../architecture/memory.md). Per attention layer, `IndexPool` holds each block's 64 centroids as F32 (16 KiB) and each bucket's pools, up to 4, as F32 integers (1 KiB), as other kernels pass indices. That comes to about 0.8% of the 27B's F16 K/V and 1.7% of the 0.8B's, and the memory plan will count it with the pages. F16 centroids would halve it if the selections still match.

Index blocks follow the same rules as pages. A block is written once, when its 512th token is written, and never changes after that. Two requests with the same first `512n` tokens have the same blocks, so snapshots and restored prefixes share them by reference count, with no copy-on-write. Keys after the last complete block always fall inside the 512-key window, so they never need an index. Keys that are both in a selected bucket and in the window are read only once.

With Q8 pools, clustering reads the dequantized keys, as attention does.

### Kernels

The kernels live in `tur-metal`'s `lsa.metal` and `lsa.rs`:

- `lsa_pools_{f32,f16,q8}` and `lsa_ward_f32` cluster complete blocks, from a list of jobs (page-table offset, block, index slot). `lsa_pools` spreads each block over 16 threadgroups. It reads the unrotated dimensions straight from the K pool (rotation doesn't touch them), dequantizing Q8, then projects and normalizes each key and sums the pools; each simdgroup projects 8 keys together, so a map row it loads serves all eight. `lsa_ward` then gives each block one threadgroup. It forms the 128×128 Gram matrix of the pool sums once, and every merge adds the partner's row and column to it, since sums merge linearly. Each round then reads its costs from the matrix instead of recomputing 16K dot products. The sums and the matrix don't fit the M1's 32 KiB of threadgroup memory, so they live in device scratch per job. Decode runs the pair once every 512 tokens; prefill runs it for every block a chunk completes. Splitting the projection out and adding the Gram matrix cut one block from 2.06 to 0.37 ms on the M1, and prefill's bulk clustering from 270 to 93 µs per block.
- `lsa_score_f32`: a grid of 256-bucket chunks by tokens. Each threadgroup forms its token's index query and scores its chunk of centroids, so long contexts spread over the whole GPU.
- `lsa_select_f32`: one threadgroup per token. It finds the exact `p`-th best score with four 8-bit radix passes, collects the buckets above it plus the lowest-numbered ties, sorts them, and expands them into key positions with a prefix sum. The paper's kernel estimates the threshold from a sample instead; the exact version is what makes Tur's lists identical to the reference's. At 128K that is 16,384 centroids, 4 MiB of reads per layer.
- `lsa_score_f32` and `lsa_select_f32` also take query groups: each row of the index table is one token, or one query tile of a tiled pass, scored by the sum of its rows' normalized index queries. A tile's list puts the sinks before the window and runs the window on to the tile's last row. A single token keeps the reference's arithmetic and order, so decode lists stay identical to the reference's.
- `attention_tiled` reads an optional key list per tile, through the same page-table lookup, clamped to the tile's last key. Each row's visible length is the list length less the keys past its own position, so the online softmax and its masking are unchanged.
- `attention_kv` reads an optional key list (`Kernels::attention_sparse`). Each listed position goes through the page table on its own, splits partition the list instead of a range, and the online softmax and gate stay as they are. A position at or past the token's visible keys is clamped, so a malformed list can't read another sequence's pages. Prefill keeps the tiled kernel. Entry `i` becomes page `table[J[i] / 16]` and slot `J[i] % 16`, and the existing splits, online softmax and gate stay as they are. A pool holds 4 consecutive keys, so a bucket expands into runs of 4 contiguous rows inside a page.

### Engine

- `[model] sparse_attention` (`--sparse-attention`) is `off` or `auto`, and `off` is the default, because outputs change compared with dense attention and every golden test assumes dense attention. Under `auto`, a request reads sparsely once its context passes the crossover; below it, and for models without a map, attention stays dense. A model can default to `auto` only after it passes the retrieval budget.
- The crossover defaults to 16K keys. Measured on the 0.8B with sparse reads forced from the first key, sparse decode is no slower than dense from 8K (20.2 against 20.9 ms) but gains only 4% at 16K and 12% at 24K, where the budget is still a large share of the context; the needle check above matched dense from 16K. A per-device value in the tuning table would replace it once more Macs are measured. The budget (`k`, `p`, window, sinks) is an advanced setting with the paper's values as defaults.
- Verification runs one selection per verified position, as plain decoding would at that position, so speculation stays exact with respect to the sparse model. The MTP head's own attention layer stays dense.
- Prefill below the crossover stays dense and tiled, and only builds index blocks; past it, each query tile selects its own keys. A pass sorts its rows into dense rows, decoding rows that select per query, and prefill rows that select per tile, and dispatches each kind on its own. With serving bursts of long prompts on the 0.8B, `auto` brought the first token 1.6 to 2.4 times sooner and cut the median gap between tokens up to 2.5 times (see [Measured performance](../guide/performance.md#long-contexts-with-sparse-attention)).

## Validation

Model correctness follows the usual rule: a pinned external reference first. The reference repository ships a plain PyTorch oracle for the flat design (`src/lsa/core/reference.py`, `ward.py` and `index_space.py`), which its own CUDA kernels are tested against. `scripts/lsa_reference.py golden` runs it on synthetic keys, queries and a map that the test rebuilds bit for bit from an integer hash, so the fixture holds only the reference's outputs. Tur's kernels reproduce its buckets and key lists exactly, for three blocks, eight query positions and two budgets, and its centroids within 1e-5. A second test checks F32, F16 and Q8 pools, four K/V heads, 128-dimension heads and two sequences in one pass against an F64 scalar reference of the same algorithm, also exactly. Making either the merge cost or the window boundary wrong fails both tests. Sparse attention matches an F64 scalar reference over exactly the selected keys within 1e-5 for every format, with and without split-K, and reading each position off by one fails it. The same scalar check covers Gemma 4's split rotation (512-dimension heads, two free runs) exactly, and the tiled path: each tile's list matches the reference built from the summed group query, and each row's tiled attention matches the scalar reference over its prefix of the list, in F16 and Q8. Through whole models, a budget that holds every key reproduces dense logits within 1e-4 on Qwen and 2e-5 on a Gemma 4 fixture with a K/V-sharing full layer, for prefill chunks, decoding, verification, mixed passes and passes that keep only their last row; on Gemma a one-bucket budget moves them 40 times past that bound, so the sparse path is the one under test.

For quality, the dense model is the reference. The first check needs no Tur kernels: `scripts/lsa_reference.py` runs Qwen3.5-0.8B in Transformers, prefills a document densely, and runs its last 64 tokens once densely and once with each attention layer reading only the keys the reference selects. It reports how much of each layer's dense attention weight the selected keys hold, against the same number of most recent keys and against the best single list for all heads, and how far next-token logits move.

The training-free map was fitted on four books at 32K tokens with the Python reference; `tur sparse-map` fits the same subspaces (mean squared cosine 0.97 to 0.99 per layer, against 0.17 for a random one), and its map gives the same results at 32K within 0.01 of weight held and 0.001 of KL and tested on three other books and Tur's Rust sources, with the paper's budget of 4,612 keys:

| Context | Weight held, LSA | Most recent keys | Best list | Mean KL | Top-1 agreement | Change in NLL |
| --- | --- | --- | --- | --- | --- | --- |
| 16K | 0.81 to 0.97 | 0.50 to 0.92 | 0.93 to 0.99 | 0.002 to 0.005 | 89 to 98% | 0.000 to +0.016 |
| 32K | 0.67 to 0.90 | 0.34 to 0.78 | 0.88 to 0.97 | 0.010 to 0.020 | 84 to 97% | -0.045 to +0.030 |
| 64K | 0.59 to 0.91 | 0.35 to 0.87 | 0.84 to 0.97 | 0.002 to 0.019 | 92 to 98% | 0.000 to +0.017 |

Next-token quality barely moves at any of the three lengths, while the weight held falls as the fixed budget becomes a smaller share of the context. LSA holds clearly more than a window of recent keys in every layer but one. The first attention layer (layer 3) is the exception at 64K, where LSA held 0.61 to 0.70 and recent keys 0.65 to 0.74 on three of the four texts, so that layer may need a wider window or a larger budget. The gap to the best list, about 0.15 to 0.25 at 64K, is what a distilled map would close. Top-1 agreement over 64 tokens moves 1.6 points per token, so read it coarsely.

These numbers measure how much attention weight survives, not whether a single fact can still be found. `just bench needle` checks that through Tur itself: it plants eight six-digit access codes at evenly spaced depths of a held-out book, prefills the document once, asks for each code from a snapshot of it and answers greedily, decoding densely and then sparsely. The prompt prefills densely in both modes, so the answer tokens are where sparse selection must find the planted keys. On the 0.8B with the map from `tur sparse-map`:

| Context | Dense correct | Sparse correct | Answers identical |
| --- | --- | --- | --- |
| 16K | 6 of 8 | 6 of 8 | 8 of 8 |
| 32K | 8 of 8 | 8 of 8 | 7 of 8 |
| 64K | 8 of 8 | 8 of 8 | 8 of 8 |

Sparse decoding retrieved every code dense decoding did. The two misses at 16K are the model's own: both modes give a neighbouring fact's code. The one answer that differs at 32K has the right code and continues differently after it. This is one model, one filler text and 24 questions; a RULER-style set with confidence intervals, declared in advance, still gates making `auto` a default for any model.

## Plan

1. Measure attention's share of decode on the 0.8B (M1) and the 27B and 35B-A3B (M3 Pro) at 32K, 64K and 128K. If it stays under about a quarter at 64K, stop and add a row to [Rejected experiments](../design/rejected.md).
2. Write `scripts/lsa_reference.py`: training-free maps for the 0.8B, attention recall and logit KL against dense, and golden fixtures for the kernels.
3. Build `lsa_cluster` and `lsa_select` with CPU references and kernel tests.
4. Add the position list to `attention_kv` and test it against the scalar reference.
5. Wire in index blocks: the pool and memory plan, snapshot sharing, the config flag, the crossover and verification.
6. Benchmark decode against dense at 32K to 128K and check the quality budget. Document the result in [Metal kernels](../architecture/kernels.md) and [State memory](../architecture/memory.md), or reject it.

## Open questions

- Whether a single crossover holds across models, or whether each Qwen needs its own.
- How often real keys produce near-ties. The GPU sums dot products in a different order from the reference, so a near-tie can flip a merge or a pick; on the fixtures none did. A flip changes which keys are read, not the attention over them.
