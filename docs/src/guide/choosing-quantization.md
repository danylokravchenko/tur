# Choosing a quantization

Three things decide the format: whether the model fits, how much quality you can give
up, and how fast it needs to load. Speed at equal formats barely changes, because all
formats run on packed kernels.

## The formats

| Format | Bits per weight | How you get it | Quality vs F32 |
| --- | --- | --- | --- |
| `f32` | 32 | `--weight-format f32` | Reference; for testing |
| `q8_0` | 8.5 | `--weight-format q8_0` (the default) | Near-lossless |
| `q4k` RTN | 4.5 | `--weight-format q4k`, or `quantize --method rtn` | Noticeable loss |
| `q4k` GPTQ | 4.5 | `quantize` (default method) | Loss roughly halved vs RTN |
| mixed Q4K/Q8_0 | your budget, e.g. 5.5 | `quantize --budget-bits 5.5` | Loss halved again vs GPTQ |

Embeddings are separate. On tied checkpoints (Qwen3.5's small models), artifacts reuse
the packed LM head for embedding lookups and store no embedding table at all. Load-time
conversion keeps a separate table.

## Measured on Qwen3.5-0.8B

Resident weights, from the memory plan `tur run` logs:

| Weights | Resident weights | Planned total (1 request, 8K context) |
| --- | --- | --- |
| F32 (load-time) | 3,010 MB | 4.72 GB |
| Q8_0 (load-time) | 1,309 MB | 3.02 GB |
| Q4K (load-time) | 933 MB | 2.64 GB |
| Q8_0 artifact, head embedding | 801 MB | |
| Q4K artifact, head embedding | 425 MB | |

The planned total also includes the state pool, scratch, and an overhead reserve of a
tenth of the device's working set.

Quality, as KL divergence from F32 on held-out text (lower is better):

- **GPTQ vs Q4K RTN, at equal bytes:** 50% lower on prose, 59% lower on code, 49% lower
  on retrieval, 12% lower on tool/JSON.
- **Mixed 5.5-bit vs Q4K GPTQ:** another 34–58% lower, for 23% more bytes.
- **Q8_0** meets the predeclared 0.95 top-1 agreement floor against the reference.
- **Q4K RTN** reached 0.821 top-1 agreement, with coherent output.

Load time for Q4K: 0.2 s from an artifact, against 9 s for load-time conversion, with
half the peak memory.

## Estimating other models

Weights take roughly `parameters × bits / 8` bytes, plus embeddings:

| Model | Q8_0 | Q4K | Fits a 16 GB Mac? |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | ~1.3 GB | ~0.9 GB (0.4 GB artifact) | Yes |
| Qwen3.5-4B (estimate) | ~4.3 GB | ~2.3 GB | Yes |
| Qwen3.5-9B (estimate) | ~9.6 GB | ~5.1 GB | Q4K only |
| Qwen3.8-27B (estimate) | ~28.7 GB | ~15.2 GB | No; needs a 32 GB+ Mac for Q4K |

The estimates are arithmetic, not measurements. Tur refuses a plan larger than the
device's recommended working set (about two thirds to three quarters of RAM on most
Macs), so check the plan it logs.

Per running request you also need state:

- recurrent slots: 20.2 MB on 0.8B, about 157 MB on 27B, in F32;
- K/V pages, which grow with context.

## Recommendations

1. **Start with `q8_0`** (the default). It is near-lossless, needs no calibration, and
   leaves room on small models.
2. **Need memory? Prepack Q4K with GPTQ** on calibration text that resembles your
   traffic. That is about a third of Q8_0's weight bytes on tied models, and keeps most
   of Q8_0's quality.
3. **Quality-sensitive at Q4K budgets?** Add `--budget-bits 5.5`. It costs about 23% more
   bytes and about 14% single-sequence decode speed.
4. **Restart often, or run many instances?** Prepack even at Q8_0
   (`tur quantize --weight-format q8_0 --method rtn`). It loads in a fraction of the time
   and, on tied models, drops the separate embedding table (1,309 → 801 MB).
5. **Q4K is the floor.** It is the narrowest projection format Tur packs. Q5K and Q6K
   are offered only for embedding tables, because Tur has no fast small-batch kernels
   for them, and K/V below 8 bits failed its quality budget.

## Check quality on your own text

The `kv_quality` benchmark compares weight variants against a reference on any text:

```sh
TUR_QWEN35_0_8B=$HOME/models/qwen3.5-0.8b \
TUR_EVAL_TEXT=held-out.txt \
TUR_EVAL_KV= \
TUR_EVAL_WEIGHTS=q4k,$HOME/models/qwen3.5-0.8b-q4k-gptq \
just bench kv_quality > quality.json
```

The reference uses Q8_0 weights unless `TUR_EVAL_REFERENCE=f32` is set. An empty
`TUR_EVAL_KV` skips the K/V format variants. The report gives KL divergence, top-1
agreement and negative log-likelihood per context bucket. `TUR_QWEN35_0_8B` can point
at any Qwen3.5-family checkpoint directory. Use text that is *not* in your calibration
set.
