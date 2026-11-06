# Long contexts with sparse attention

Decoding one token normally reads every key and value the request has written, in every full attention layer. At 128K tokens of context those reads take longer than reading the model's weights, so each new token gets slower as the context grows. Latent sparse attention (LSA) caps those reads: past 16K keys, each decoding token reads about 4,600 keys chosen for it, however long the context is. On Qwen3.5-0.8B on an M1, a decode step at 128K drops from 46 ms to 21 ms, the same time it takes at 32K.

LSA is an experiment and is off by default. It changes outputs compared with dense attention, and so far it has been checked on two small models, Qwen3.5-0.8B and Gemma 4 E2B. It works on Qwen3.5-family models, dense and MoE, and on Gemma 4's full attention layers. gpt-oss reads every key, because it rotates every dimension of its keys and leaves nothing position-free to cluster.

## How it works

LSA groups the keys into small clusters once, ahead of time. Each query then scores the clusters and reads only the keys in the best ones.

1. Each time a request completes a block of 512 keys in a full attention layer, Tur clusters that block. It sums consecutive keys in groups of four, then merges the groups into 64 buckets of up to 16 keys each, always merging the two groups whose sum loses the least length. Each bucket keeps its normalized key sum, its centroid. A block's index takes 17 KiB per layer, about 2% of the K/V it describes on the 0.8B.
2. A decoding query scores every centroid of the closed blocks before it. Clustering and scoring use only the dimensions rotary position encoding leaves alone (192 of each head's 256 on Qwen, 384 of 512 on Gemma 4's full layers), so they compare keys by content, not position. A per-layer map, the index map, projects those dimensions to 64 before comparing.
3. The query takes its 1,024 best buckets and reads up to 4,096 of their keys. It also reads the 512 most recent keys and the first 4. Attention over those keys runs in the same kernel as dense attention and is exact over the keys it reads.

Clustering costs 0.37 ms per block on the M1. Decoding pays it once every 512 tokens. Prefill clusters every block it completes, at about 93 µs each.

The index map is the one thing LSA needs beyond the model's weights. Tur fits one without training: `tur sparse-map` runs calibration text through the model and keeps, per full attention layer, the 64 directions along which its keys vary most. The map depends only on the model, so you fit it once and reuse it.

## When it helps

LSA helps long decodes and does nothing for short ones. Below the crossover of 16K keys every query reads densely, so a request pays only for clustering. Past it, decode time stays nearly flat as the context grows. With several long requests decoding at once, the gain is larger, because each request adds its own K/V reads while the batch reads the weights once.

Measured on Qwen3.5-0.8B, Q8_0 weights, F16 K/V, on an M1:

| Workload | Dense | LSA |
| --- | --- | --- |
| One request decoding at 128K, per token | 46.2 ms | 21.1 ms |
| One request decoding at 64K, per token | 32.6 ms | 20.9 ms |
| Four 64K requests decoding together, median inter-token latency | 78.2 ms | 33.3 ms |
| Four 64K requests, time to first token | 787 s | 815 s |

With `auto`, prefill still reads every key, so the time to first token stays about the same (up to 3% later). If your requests carry long prompts and short answers, most of their time goes to prefill and `auto` saves little.

### Sparse prefill

`--sparse-attention prefill` reads prompts sparsely too. Prefill runs in tiles of 32 consecutive prompt tokens, and past the crossover each tile scores the buckets with its tokens' averaged index query and gets one list: the budget's 4,096 selected keys, the sinks, and every key from 512 before the tile's first token up to its last. Each token reads that list up to its own position. Sharing one list across the tile keeps the matrix units busy, so a chunk's prefill attention grows little with context past the crossover. On the 0.8B, one 2048-token prefill chunk takes:

| Context before the chunk | Dense | Sparse prefill |
| --- | --- | --- |
| 32K | 5.9 s | 2.5 s |
| 64K | 11.3 s | 2.8 s |

Sparse prefill changes the hidden state of every prompt token, not only the generated ones, so its errors can build up through the layers. It is a separate setting for that reason.

### Gemma 4

Gemma 4 alternates sliding window layers, which read at most 512 or 1,024 recent keys and stay dense, with full attention layers, which LSA indexes. On E2B and E4B the last full layers reuse an earlier layer's keys; they select through that layer's index, so only the owner clusters. Most of a small Gemma's decode step is weights, so the decode gain is smaller than Qwen's, while prefill gains more. On E2B on an M1:

| Measure | Dense | LSA with sparse prefill |
| --- | --- | --- |
| Decode step at 64K | 66.0 ms | 51.3 ms |
| 2048-token prefill chunk after 32K | 10.9 s | 2.5 s |

Speculative decoding still works. Verification selects keys for each drafted position as plain decoding would, so accepted tokens match what the sparse model would have produced one at a time. The MTP head's own attention stays dense.

## What it costs in quality

Sparse reads can miss a key that dense attention would have weighted. Tur's retrieval check hides eight six-digit codes at evenly spaced depths of a long book and asks for each one. On the 0.8B, LSA answered correctly every question that dense attention answered, at 16K, 32K and 64K, and 23 of its 24 answers matched dense attention's text exactly. With sparse prefill as well, it answered 23 of 24 correctly against dense attention's 22, and 22 answers matched exactly. On Gemma 4 E2B with sparse prefill, both modes answered all 16 questions at 16K and 32K with identical text. That is one model, one book and one kind of question, so check your own workload before relying on it.

## Setting it up

### 1. Pick calibration text

Use a few long documents that resemble what the model will serve: books for prose, code for a coding assistant, transcripts for chat. Four to eight documents of at least 32K tokens each work well; 32K tokens is about 120 KB of English text. Keep the text you will evaluate on out of the calibration set, so the check measures the map rather than its training data.

### 2. Fit the index map

```sh
tur sparse-map --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output qwen-0.8b-map.safetensors --calibration book1.txt,book2.txt,book3.txt,book4.txt
```

Tur prefills the first 32K tokens of each file as one sequence, samples 8,192 keys per document from every full attention layer, and fits the map. On the 0.8B, four documents took five minutes on an M1, and seven on Gemma 4 E2B. The map file is small (590 KB for the 0.8B's six full attention layers).

Fit the map on the weights you will serve. `--weight-format` (default `q8_0`) picks the format for a checkpoint, and a prepacked artifact loads as stored. For an MoE model larger than GPU memory, `--expert-cache-mb` keeps the experts on disk, as it does for serving. [Command line](cli.md#index-maps-sparse-map) lists every flag.

A map belongs to one model. Loading it into a model with other full attention layers or head sizes fails with an error naming the layer.

### 3. Serve with it

```sh
tur serve Qwen/Qwen3.5-0.8B --sparse-attention auto --sparse-map qwen-0.8b-map.safetensors
```

Or in `tur.toml`:

```toml
[model]
source = "Qwen/Qwen3.5-0.8B"
revision = "2fc06364715b967f1860aea9cf38778875588b17"
sparse_attention = "auto"
sparse_map = "qwen-0.8b-map.safetensors"
```

A relative `sparse_map` resolves against the directory of the config file. `tur run` takes the same flags.

With `auto`, every request decodes densely until its context passes 16K keys and sparsely after that. `sparse_attention = "prefill"` adds sparse prefill past the same crossover. The memory plan reserves index blocks next to the K/V pages, about 2% more state memory on the 0.8B.

### 4. Check it on your workload

Compare a few long requests with and without `--sparse-attention auto`, using greedy sampling so the outputs are comparable. To repeat Tur's retrieval check on your model and your own held-out text:

```sh
TUR_BENCH_MODEL=<checkpoint dir> TUR_BENCH_TEXT=held-out.txt \
  TUR_BENCH_SPARSE_MAP=qwen-0.8b-map.safetensors TUR_BENCH_CONTEXTS=16384,32768,65536 \
  just bench needle > needle.json
```

The `summary` field counts the questions each mode answered correctly at each context. `TUR_BENCH_SPARSE_PREFILL=1` checks sparse prefill as well. `TUR_BENCH_SPARSE_MAP=<map> just bench model` and `just bench serving` measure speed the same way; [Benchmarks](../development/benchmarks.md) describes both.

## Limits

- Qwen3.5-family and Gemma 4 models on Metal only, with fused execution (the default).
- The budget (4,096 keys from the best 1,024 buckets, a 512-key window and 4 sinks) and the 16K crossover are fixed defaults. The Rust library's `SparseAttention` exposes them for experiments.

[Research: latent sparse attention](../research/latent-sparse-attention.md) covers the method's origin, the kernels, every measurement and the open questions.
