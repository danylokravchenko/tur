# Best practice

This page gives a starting configuration for each model Tur supports: which weights to use, which speculation to turn on, and whether to use the expert cache. Save a block as `tur.toml` next to your models and run `tur serve`. [Tuning](tuning.md) explains what each setting trades, and [Measured performance](performance.md) has the full tables.

| Model | Type | Best weights | Speculation | Expert cache | Evidence |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | Dense | Q8_0 artifact | Lookup and MTP for one or two users | Not supported | Measured, M1 16 GB |
| Qwen3.5-4B, 9B | Dense | Q8_0 artifact; Q4K on 16 GB for the 9B | As the 0.8B | Not supported | Not run |
| Qwen3.8-27B | Dense | Q4K artifact | MTP not measured | Not supported | Speed measured, M3 Pro 36 GB; quality not |
| Qwen3.6-35B-A3B | MoE | Q4K artifact | Lookup and MTP, up to four requests | Only above 8 requests, or if it doesn't fit | Speed measured, M3 Pro 36 GB; quality not |
| Gemma 4 E2B | Dense | Q8_0 artifact | Lookup and assistant | Not supported | Measured, M1 16 GB |
| Gemma 4 E4B, 12B, 31B | Dense | Q8_0 artifact if it fits | Assistant not run | Not supported | Not run |
| Gemma 4 26B-A4B | MoE | Q8_0 (Q4K can't pack its experts) | Assistant not run | Not supported | Not run |
| gpt-oss-20b | MoE | Q8_0 at load, MXFP4 experts as published | No MTP head | 2 GiB on 16 GB | Measured, M1 16 GB |
| Whisper large-v3-turbo | Speech | Q8_0 | None | None | Measured, M1 16 GB |

"Not run" rows are starting points worked out from the configuration and the rules below. Read the memory plan that Tur logs at startup, and [check quality on your own text](choosing-quantization.md#check-quality-on-your-own-text) before you rely on them.

## Dense and MoE models

A dense model runs every parameter for every token. On the large ones, prefill is limited by arithmetic: Qwen3.8-27B prefills about 92 tokens/s, so a 2K-token prompt takes 22 s, while batching is nearly free because one pass reads the weights once for all sequences. Keep prefix reuse on so a prompt is computed only once, and raise the batch size before you worry about decode speed.

An MoE model uses a few experts per token, so it runs faster than its size suggests, but the whole bank has to live somewhere. Keep every expert resident when the model fits. The expert cache exists to run what doesn't fit, and reading experts from the SSD costs far more than batching saves: Qwen3.6-35B-A3B makes 101.5 tokens/s across 8 resident requests and 25.7 across 16 through a 4 GiB cache.

These rules apply to every model:

- Prepack weights with `tur quantize` if you restart often. Artifacts load in a fraction of the time, and on models with shared embeddings they drop the separate embedding table.
- Leave out `weight_format` when loading an artifact, because setting it is an error.
- MTP and the Gemma assistant turn off prefix reuse. Use them for single requests with fresh prompts, and leave them off for chat history and agent loops, which resend the same prompt.
- Prompt lookup doesn't affect prefix reuse and costs little when it has nothing to draft: nothing measurable on the 0.8B and about 5% on Gemma E2B with four requests. That makes `speculative_draft = 4` a cheap addition, though tokens may arrive in small bursts.
- Speculation needs `recurrent_state = "f32"`, the default. Outputs match normal decoding with every speculative configuration.
- Set `context_tokens` if you need more than 8K tokens per request, so the start fails early with the largest context that fits.

## Qwen3.5-0.8B

Use a Q8_0 artifact. It agrees with F32 on the top token 98 to 99% of the time and holds 801 MB. Q4K decodes one sequence 1.6× faster (90 tokens/s against 57), but from two sequences on the formats converge and its top-1 agreement falls to 84 to 95%. If one conversation's speed matters most, a mixed 5.5-bit GPTQ artifact is the better compromise: 79 tokens/s, 521 MB, and about half the KL divergence of Q4K GPTQ.

```sh
tur quantize --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output ./models/qwen3.5-0.8b-q8 --weight-format q8_0 --method rtn

# Optional, for one conversation:
tur quantize --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output ./models/qwen3.5-0.8b-mixed --budget-bits 5.5 --calibration prose.txt,code.txt
```

With one or two users, turn on MTP. New prose and code generate 1.1 to 1.4× faster, and repetitive text about 3× faster through lookup:

```toml
[model]
source = "./models/qwen3.5-0.8b-q8"
revision = "q8-v1"

[engine]
prefill_chunk_size = 512
speculative_draft = 4
mtp = true

[server]
served_model_name = "qwen-0.8b"
max_batch_size = 2
```

For a chat service or agents, leave MTP off and keep more snapshots. With 16 snapshots, four six-turn conversations served 68% of prompt tokens from the cache, against 44.5% with 8. Eight or more requests is where MTP stops helping anyway.

```toml
[model]
source = "./models/qwen3.5-0.8b-q8"
revision = "q8-v1"
kv_format = "q8"

[engine]
speculative_draft = 4

[server]
served_model_name = "qwen-0.8b"
max_batch_size = 16
prefix_snapshots = 32
request_timeout_secs = 120
```

## Qwen3.5-4B and 9B

These sizes haven't been run through Tur. They share the 0.8B's code path, so start from its configuration and its artifact commands. By arithmetic, the 4B's weights take about 4.3 GB at Q8_0 and 2.3 GB at Q4K, and the 9B's take about 9.6 GB and 5.1 GB. On a 16 GB Mac, use Q8_0 for the 4B and Q4K for the 9B; GPTQ is the default method for a Q4K artifact, and it cut KL divergence by about half on the 0.8B.

## Qwen3.8-27B

Use a Q4K artifact, which is 17.2 GB and needs a Mac with at least 32 GB. Q8_0 takes about 28.7 GB and won't fit on 36 GB. Plain rounding is the measured path: it takes under 4 minutes and loads in 4.2 s. GPTQ should give better quality, as it does on the 0.8B, but neither its quality nor its peak memory has been measured on this model.

```sh
tur quantize --model Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --output ./models/qwen3.8-27b-q4k --method rtn
```

The checkpoint ships an MTP head, but it hasn't been measured here. Leave it off to begin with, because prefix reuse is worth more at 92 tokens/s of prefill: repeating a 2K-token prompt costs 22 s without reuse. Lookup helped repetitive output on every model where it was measured, though not yet on this one. Four requests at once kept a median of 141 ms between tokens, against 115 ms for one, and eight also complete at 156 ms if you need the throughput.

```toml
[model]
source = "./models/qwen3.8-27b-q4k"
revision = "q4k-rtn-v1"

[engine]
speculative_draft = 4

[server]
served_model_name = "qwen-27b"
max_batch_size = 4
request_timeout_secs = 600
```

To try MTP with one user and fresh prompts, add `mtp = true` and compare against this configuration on your own prompts. At 8K tokens, four requests hold 2.6 GiB of K/V and recurrent state. If longer contexts don't fit, `kv_format = "q8"` halves the K/V part, though its effect on this model's prefill hasn't been measured.

## Qwen3.6-35B-A3B

Use a Q4K artifact, whose weights take 20.3 GB. The source checkpoint is about 67 GiB, so check your disk before downloading it. As with the 27B, plain rounding is the measured path. Calibrated methods hold a large MoE layer in F32 while they work, and their peak memory on this model hasn't been measured.

```sh
tur quantize --model Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output ./models/qwen3.6-35b-q4k --method rtn
```

With one to four requests on a 36 GB Mac, keep experts resident and turn on MTP. It gave 1.09 to 1.34× on new prose and code, and up to 1.99× on repetitive text, for 0.8 to 1.7 GB more memory. The head switches itself off above four decoding requests, so there's no reason to allow more.

```toml
[model]
source = "./models/qwen3.6-35b-q4k"
revision = "q4k-rtn-v1"

[engine]
speculative_draft = 4
mtp = true

[server]
served_model_name = "qwen-35b"
max_batch_size = 4
```

For chat history and agents, set `mtp = false` and raise `max_batch_size` to 8, which is the most that runs resident on 36 GB and gives 101.5 tokens/s. Sixteen resident requests run out of GPU memory.

Use the cache only for 16 requests at once, or on a Mac where the model doesn't fit. A 4 GiB cache ran 16 requests at 25.7 tokens/s with a median of 315 ms between tokens. The cache needs an artifact, and Tur raises the prefill chunk and snapshot spacing to 1,024 tokens for you. Leave MTP off, because the head hasn't been measured through the cache.

```toml
[model]
source = "./models/qwen3.6-35b-q4k"
revision = "q4k-rtn-v1"
expert_cache_mb = 4096

[server]
served_model_name = "qwen-35b"
max_batch_size = 16
```

Smaller Macs haven't run this model. By arithmetic from its tensor header, the weights outside the routed experts take about 2 GB at Q4K, and the cache must hold one layer's 256 experts, about 430 MiB. Start at 2 GiB with `max_batch_size = 1` or 2, and raise the cache while the spare-memory check accepts it.

## Gemma 4 E2B

Use a Q8_0 artifact, which holds 7.12 GB and loads in about 5 s, against 7.93 GB and about 12 s when converting at load time. Gemma's artifacts use plain rounding, because calibration is available only for Qwen. E2B's widths allow Q4K, but no quality or speed figures exist for it. Most of the weight memory is the 4.4 GiB per-layer embedding table, which stays BF16 in packed modes, so Q4K wouldn't save as much as it does on Qwen.

```sh
tur quantize --model google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --output ./models/gemma-e2b-q8 --weight-format q8_0 --method rtn
```

Gemma drafts with Google's assistant model, and `mtp = true` is an error. With one request, the assistant and lookup gave 1.42× on prose and 1.17 to 1.30× on code when drafting one token per pass. Tur now chains drafts for a single request, which added another 1.14× on prose. With four requests they gave 1.23× and 1.35×.

```toml
[model]
source = "./models/gemma-e2b-q8"
revision = "q8-v1"
audio_encoder = false  # Remove this line to accept audio in chat requests.

[engine]
speculative_draft = 4
draft_model = "google/gemma-4-E2B-it-assistant"
draft_revision = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"

[server]
served_model_name = "gemma-e2b"
max_batch_size = 4
```

For chat history and agents, remove `draft_model` and `draft_revision` so that prefix reuse stays on. Keep `speculative_draft = 4`, which alone gave 3.2× on repetitive prompts.

## Gemma 4 E4B, 12B and 31B

These checkpoints pass configuration checks and small architecture fixtures, but the full weights haven't been run, so check their output before you trust it. They are dense and take the E2B's artifact command with their own repository and revision. Their widths allow Q4K, whose quality on Gemma is unmeasured, so prefer Q8_0 where it fits.

| Model | Q8_0, by arithmetic | Q4K, by arithmetic |
| --- | --- | --- |
| E4B | Above E2B's 7.12 GB, with its own embedding table | Not estimated |
| 12B | About 12.8 GB | About 6.8 GB |
| 31B | About 33 GB | About 17.4 GB |

On these estimates the 31B needs Q4K on 36 GB, and the 12B needs it on 16 GB. Write the artifact from a pinned commit, then start with this:

```toml
[model]
source = "./models/gemma-12b-q8"
revision = "q8-v1"

[engine]
speculative_draft = 4

[server]
served_model_name = "gemma-12b"
max_batch_size = 4
```

Google publishes an assistant for each size, but Tur has only run the E2B's. If you try one, set `draft_model` and `draft_revision` as for the E2B and compare on your own prompts.

## Gemma 4 26B-A4B

This is Gemma's MoE model, and it hasn't been run on full weights either. Use Q8_0: its 704-wide experts don't fit Q4K's 256-column blocks. The expert cache rejects Gemma, so every expert has to stay resident. At about 27.6 GB of weights by arithmetic, it won't fit a 36 GB Mac's 30.2 GB working set once the reserve and state are counted, so plan on 48 GB or more.

```toml
[model]
source = "./models/gemma-26b-q8"
revision = "q8-v1"

[engine]
speculative_draft = 4

[server]
served_model_name = "gemma-26b"
max_batch_size = 2
```

## gpt-oss-20b

Leave out `weight_format`, so attention and output projections convert to Q8_0, and let the experts run as the MXFP4 blocks they ship in. There are no artifacts to make for this model. Its 2,880-wide projections don't fit Q4K blocks, and `tur prune` costs too much quality to recommend. It has no MTP head, so `mtp = true` is an error.

Its 12.8 GiB of weights are more than a 16 GB Mac's GPU working set, so the cache is required there. A 2 GiB cache gave a 52% hit rate and 3.3 tokens/s, and swapped nothing. Hit rates reach 67% at 3 GiB and 79% at 4 GiB, but the spare-memory check may refuse those sizes on a busy Mac, and larger caches have swapped. The cache must hold at least one layer's experts, 423 MiB.

```toml
[model]
source = "openai/gpt-oss-20b"
revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
expert_cache_mb = 2048

[server]
served_model_name = "gpt-oss-20b"
max_batch_size = 1
```

Each extra request slows the others, since they share the cache's slots. Total throughput rises only from 2.1 to 2.7 tokens/s between one and four requests, each request's speed falls from 2.6 to 0.7 tokens/s, and four requests swapped 251 MiB on a tight machine. Keep the prompt prefix stable: repeating a 2,037-token prompt with a different ending takes 10 s instead of 18 s.

On a Mac with room for 12.8 GiB, remove `expert_cache_mb` so that no token waits for the SSD. That configuration hasn't been measured.

## Whisper large-v3-turbo

Whisper loads beside the chat model and takes the `[whisper]` section. Keep the default Q8_0: it uses about 0.85 GB against 3.2 GB for F32, transcribes an 80 s clip faster (9.4× real time against 6.7×), and matched Transformers' transcript on a pinned clip. Q4K is accepted but unmeasured.

```toml
[whisper]
source = "openai/whisper-large-v3-turbo"
revision = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
weight_format = "q8_0"
```
