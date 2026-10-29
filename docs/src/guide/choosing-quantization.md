# Choosing a quantization

Three questions decide the format: does the model fit, how much quality can you give up, and how fast must it load? Speed barely differs between formats once several requests are running, because batched decoding is limited by recurrent state rather than weights. With one conversation, smaller weights decode faster.

## The formats

| Format | Bits per weight | How you get it | Quality |
| --- | --- | --- | --- |
| `f32` | 32 | `--weight-format f32` | Reference, for testing |
| `q8_0` | 8.5 | `--weight-format q8_0` (the default) | Close to F32 |
| `q4k`, plain rounding | 4.5 | `--weight-format q4k`, or `quantize --method rtn` | Noticeable loss |
| `q4k`, GPTQ | 4.5 | `quantize` (the default method) | About half the loss of plain rounding |
| Mixed Q4K and Q8_0 | your budget, e.g. 5.5 | `quantize --budget-bits 5.5` | A third to half less again |

Qwen3.5's small models share one matrix between the input embedding and the output layer. Artifacts take advantage of that and keep no separate embedding table, while load-time conversion keeps one.

## Measured on Qwen3.5-0.8B

Memory for the weights, as `tur run` reports it:

| Weights | Weight memory | Total with one 8K-token request |
| --- | --- | --- |
| F32, converted at load | 3,010 MB | 4.72 GB |
| Q8_0, converted at load | 1,309 MB | 3.02 GB |
| Q4K, converted at load | 933 MB | 2.64 GB |
| Q8_0 artifact | 801 MB | |
| Q4K artifact | 425 MB | |

The total also includes the state pool, kernel scratch space, and a reserve of a tenth of the GPU's working set.

Quality, as KL divergence from F32 on held-out text (lower is better):

- GPTQ cuts Q4K's divergence by about half on prose, code and retrieval text, and by 12% on tool calls and JSON.
- Mixed 5.5-bit cuts it by another third to half, for 23% more memory.
- Q8_0 agrees with F32 on the top token about 98 to 99% of the time.

A Q4K artifact loads in 0.2 s, against 9 s for converting at load time.

## Estimating other models

Weights take roughly `parameters × bits / 8` bytes:

| Model | Q8_0 | Q4K | Fits a 16 GB Mac? |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 1.3 GB | 0.9 GB (0.4 GB artifact) | Yes |
| Qwen3.5-4B | about 4.3 GB | about 2.3 GB | Yes |
| Qwen3.5-9B | about 9.6 GB | about 5.1 GB | Q4K only |
| Qwen3.8-27B | about 28.7 GB | about 15.2 GB | No; Q4K needs a 32 GB Mac |

Only the 0.8B row is measured; the others are arithmetic. macOS lets the GPU use about two thirds to three quarters of RAM, and Tur refuses to start if its plan doesn't fit, so check the plan it logs.

Each running request also needs state. Recurrent state is fixed per request (20 MB on the 0.8B in F32, about 157 MB on the 27B), and K/V pages grow with the context.

## Recommendations

1. Start with `q8_0`, the default. It is close to F32, needs no calibration, and leaves room on small models.
2. If you need memory, prepack Q4K with GPTQ on text that resembles your traffic. On models with shared embeddings that's about a third of Q8_0's weight memory, and it keeps most of Q8_0's quality.
3. If quality matters at a Q4K budget, add `--budget-bits 5.5`. It costs about 23% more memory and 14% of single-conversation speed.
4. If you restart often or run several instances, prepack even at Q8_0 (`tur quantize --weight-format q8_0 --method rtn`). It loads in a fraction of the time and, with shared embeddings, saves the separate embedding table.
5. Q4K is the floor. Tur has no fast kernels for narrower formats, and 4-bit K/V failed its quality budget.

## Check quality on your own text

The `kv_quality` benchmark compares weight variants against a reference on any text:

```sh
TUR_QWEN35_0_8B=$HOME/models/qwen3.5-0.8b \
TUR_EVAL_TEXT=held-out.txt \
TUR_EVAL_KV= \
TUR_EVAL_WEIGHTS=q4k,$HOME/models/qwen3.5-0.8b-q4k-gptq \
just bench kv_quality > quality.json
```

The reference uses Q8_0 weights unless you set `TUR_EVAL_REFERENCE=f32`. An empty `TUR_EVAL_KV` skips the K/V format variants. The report gives KL divergence, top-1 agreement and negative log-likelihood for each context length. `TUR_QWEN35_0_8B` can point at any Qwen3.5-family checkpoint directory. Use text that is not in your calibration set.
