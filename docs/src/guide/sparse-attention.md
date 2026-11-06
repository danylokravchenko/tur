# Long contexts with sparse attention

Without sparse attention, every prompt token and every generated token reads every key and value the request has written, in every full attention layer. At 64K tokens of context those reads dominate: prefilling a 2048-token chunk takes twice as long as at 32K, and each new token gets slower as the context grows. Latent sparse attention (LSA) caps those reads. Past 16K keys, each token reads about 4,600 keys chosen for it, however long the context is. On Qwen3.5-0.8B on an M1, a 64K prompt's first token arrives in 80 s instead of 194 s, and a decode step at 128K drops from 46 ms to 21 ms, the same time it takes at 32K.

LSA is an experiment and is off by default. It changes outputs compared with dense attention, and so far it has been checked on two small models, Qwen3.5-0.8B and Gemma 4 E2B. It works on Qwen3.5-family models, dense and MoE, and on Gemma 4's full attention layers. gpt-oss reads every key, because it rotates every dimension of its keys and leaves nothing position-free to cluster.

## When to use it

Turn it on when your requests regularly pass 16K tokens: long documents, whole codebases, agent loops whose history keeps growing, several long conversations at once.

- Below 16K keys every token reads densely, so short chats gain nothing and lose almost nothing: the only cost is clustering each completed block of 512 keys.
- Past 16K both prefill and decode stay nearly flat as the context grows, so the gain grows with the prompt. A 32K prompt arrives 1.6 times sooner on the 0.8B and a 64K one 2.4 times sooner.
- Several long requests at once gain the most, because each one adds its own K/V reads while the batch shares one read of the weights. Four 64K requests decode 2.5 times faster.
- Models whose decode step is mostly weights gain less in decode. Gemma 4 E2B spends 47 ms per token on weights, so its decode at 64K improves 1.3 times, while its prefill improves 4.3 times.
- Leave it off when you need outputs identical to dense attention, for example to compare against another engine or a reference.

[Best practice](best-practice.md#long-contexts) has a configuration to start from.

## How it works

LSA groups the keys into small clusters once, ahead of time. Each query then scores the clusters and reads only the keys in the best ones.

1. Each time a request completes a block of 512 keys in a full attention layer, Tur clusters that block. It sums consecutive keys in groups of four, then merges the groups into 64 buckets of up to 16 keys each, always merging the two groups whose sum loses the least length. Each bucket keeps its normalized key sum, its centroid. A block's index takes 17 KiB per layer, about 2% of the K/V it describes on the 0.8B.
2. A query scores every centroid of the closed blocks before it. Clustering and scoring use only the dimensions rotary position encoding leaves alone (192 of each head's 256 on Qwen, 384 of 512 on Gemma 4's full layers), so they compare keys by content, not position. A per-layer map, the index map, projects those dimensions to 64 before comparing.
3. The query takes its 1,024 best buckets and reads up to 4,096 of their keys. It also reads the 512 most recent keys and the first 4. Attention over those keys runs in the same kernels as dense attention and is exact over the keys it reads.

Decoding tokens select one by one. Prefill runs in tiles of 32 consecutive prompt tokens, and each tile scores the buckets with its tokens' averaged index query and gets one list: the selected keys, the first 4, and every key from 512 before the tile's first token up to its last. Each token reads that list up to its own position, so a token never sees keys after it, and the GPU's matrix units share every key they load across the tile's 32 tokens.

Clustering costs 0.37 ms per block on the M1. Decoding pays it once every 512 tokens. Prefill clusters every block it completes, at about 93 µs each.

On Gemma 4, the sliding window layers read at most 512 or 1,024 recent keys and stay dense; LSA indexes the full attention layers. On E2B and E4B the last full layers reuse an earlier layer's keys, so they select through that layer's index and only the owner clusters.

Speculative decoding still works. Verification selects keys for each drafted position as plain decoding would, so accepted tokens match what the sparse model would have produced one at a time. The MTP head's own attention stays dense.

The index map is the one thing LSA needs beyond the model's weights. Tur fits one without training: `tur sparse-map` runs calibration text through the model and keeps, per full attention layer, the 64 directions along which its keys vary most. The map depends only on the model, so you fit it once and reuse it.

## What it gains

Bursts of long prompts through the full server on Qwen3.5-0.8B, Q8_0 weights, F16 K/V, on an M1, 256 output tokens per request:

| Prompt, requests | First token, dense | First token, LSA | Gap between tokens, dense | Gap between tokens, LSA |
| --- | --- | --- | --- | --- |
| 32K, 1 | 61 s | 39 s | 25.3 ms | 21.2 ms |
| 32K, 4 | 229 s | 152 s | 44.0 ms | 29.1 ms |
| 64K, 1 | 194 s | 80 s | 32.4 ms | 21.8 ms |
| 64K, 4 | 737 s | 308 s | 78.2 ms | 31.9 ms |

Against llama.cpp on the same Q8_0 weights, one sequence decodes 47.5 tokens/s at 128K against llama.cpp's 20.2, and a 2048-token chunk after 128K prefills at 679 tokens/s against 159. [Measured performance](performance.md#long-contexts-with-sparse-attention) has every depth, dense Tur alongside, and Gemma 4 E2B.

## What it costs in quality

Sparse reads can miss a key that dense attention would have weighted. Tur's retrieval check hides eight six-digit codes at evenly spaced depths of a long book and asks for each one. On the 0.8B, LSA answered 23 of 24 questions correctly at 16K, 32K and 64K, against 22 for dense attention, and 22 answers matched dense attention's text exactly; of the two that differ, one is a question dense attention got wrong and the other has the right code and continues differently. On Gemma 4 E2B both modes answered all 16 questions at 16K and 32K with identical text. That is two small models, one book and one kind of question, so check your own workload before relying on it.

## Setting it up

### 1. Pick calibration text

Use a few long documents that resemble what the model will serve: books for prose, code for a coding assistant, transcripts for chat. Four to eight documents of at least 32K tokens each work well; 32K tokens is about 120 KB of English text. Keep the text you will evaluate on out of the calibration set, so the check measures the map rather than its training data.

### 2. Fit the index map

```sh
tur sparse-map --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output qwen-0.8b-map.safetensors --calibration book1.txt,book2.txt,book3.txt,book4.txt
```

Tur prefills the first 32K tokens of each file as one sequence, samples 8,192 keys per document from every full attention layer, and fits the map. On an M1, four documents took five minutes on the 0.8B and seven on Gemma 4 E2B. The map file is small (590 KB for the 0.8B's six full attention layers).

Fit the map on the weights you will serve. `--weight-format` (default `q8_0`) picks the format for a checkpoint, and a prepacked artifact loads as stored. For an MoE model larger than GPU memory, `--expert-cache-mb` keeps the experts on disk, as it does for serving. [Command line](cli.md#index-maps-sparse-map) lists every flag.

A map belongs to one model. Loading it into a model with other full attention layers or head sizes fails with an error naming the layer.

### 3. Serve with it

```sh
tur serve Qwen/Qwen3.5-0.8B --sparse-attention auto --sparse-map qwen-0.8b-map.safetensors \
  --context-tokens 65536
```

Or in `tur.toml`:

```toml
[model]
source = "Qwen/Qwen3.5-0.8B"
revision = "2fc06364715b967f1860aea9cf38778875588b17"
sparse_attention = "auto"
sparse_map = "qwen-0.8b-map.safetensors"

[engine]
context_tokens = 65536
```

A relative `sparse_map` resolves against the directory of the config file. `tur run` takes the same flags.

With `auto`, a request reads every key until its context passes 16K keys and sparsely after that, in prefill and in decode. Set `context_tokens` to the longest request you accept, so the state pool is sized for it; by default it holds about 8K tokens per concurrent request. The memory plan reserves index blocks next to the K/V pages, about 2% more state memory on the 0.8B.

### 4. Check it on your workload

Compare a few long requests with and without `--sparse-attention auto`, using greedy sampling so the outputs are comparable. To repeat Tur's retrieval check on your model and your own held-out text:

```sh
TUR_BENCH_MODEL=<checkpoint dir> TUR_BENCH_TEXT=held-out.txt \
  TUR_BENCH_SPARSE_MAP=qwen-0.8b-map.safetensors TUR_BENCH_CONTEXTS=16384,32768,65536 \
  just bench needle > needle.json
```

The `summary` field counts the questions each mode answered correctly at each context. `TUR_BENCH_SPARSE_PREFILL=0` keeps the sparse mode's prefill dense, to tell decode's effect from prefill's. `TUR_BENCH_SPARSE_MAP=<map> just bench model` and `just bench serving` measure speed the same way; [Benchmarks](../development/benchmarks.md) describes both.

## Limits

- Qwen3.5-family and Gemma 4 models on Metal only, with fused execution (the default).
- The budget (4,096 keys from the best 1,024 buckets, a 512-key window and 4 sinks) and the 16K crossover are fixed defaults. The Rust library's `SparseAttention` exposes them, and whether prefill reads sparsely, for experiments.

[Research: latent sparse attention](../research/latent-sparse-attention.md) covers the method's origin, the kernels, every measurement and the open questions.
