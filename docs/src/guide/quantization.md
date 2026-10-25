# Prepacked weights

Tur keeps projection weights packed in GGML block formats, and its kernels read those
blocks directly. A SafeTensors checkpoint can be packed in two ways:

- **At load time:** `--weight-format f32|q8_0|q4k` on `run` or `serve` (default `q8_0`).
  This is plain round-to-nearest packing, repeated on every start.
- **Offline:** `tur quantize` writes a prepacked artifact. It can use calibrated
  rounding and per-projection precision, and it loads as stored.

These are not GGUF imports, and `q4k` is Q4_K packing rather than a promise of
llama.cpp's Q4_K_M recipe.

## Writing an artifact

```sh
tur quantize \
  --model /models/qwen3.5-0.8b --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output /models/qwen3.5-0.8b-q4k-gptq \
  --calibration prose.txt,code.txt,tools.txt \
  --report /models/qwen3.5-0.8b-q4k-gptq.json
```

Then load it without `--weight-format`:

```sh
tur serve --model /models/qwen3.5-0.8b-q4k-gptq --revision q4k-gptq-v1
```

Passing `--weight-format` with an artifact is an error. The artifact is one
`tur-weights.safetensors` file whose header carries a manifest, next to verbatim copies
of the source's config, tokenizer, template and generation defaults. The manifest
records:

- the format version (readers reject any other);
- the source checkpoint identity, and a fingerprint of its `config.json`;
- the quantizer, its version, method and damping;
- the calibration sample count, token count and fingerprint;
- the per-projection weight plan.

All of it is validated before any payload is read. The file is written to a `.partial`
path and renamed into place, so a failed run leaves no half-written artifact.

## Choosing a method

| Method | Needs calibration | What it does |
| --- | --- | --- |
| `rtn` | no | Round-to-nearest; identical to load-time conversion |
| `imatrix` | yes | Importance-weighted K-quant scale search from input statistics |
| `gptq` (default) | yes | GPTQ error compensation onto the same block grids |

Every method emits ordinary GGML blocks, so the same kernels execute the result, and
decode speed is unchanged at equal formats.

Calibration text must never overlap the text you evaluate quality on. Without
`--budget-bits`, GPTQ and imatrix load one F32 layer, run the calibration samples through
it, release it, and pack its projections before loading the next layer. Later layers
receive the unquantized reference activations, preserving resident calibration's
statistics. Embeddings are gathered directly from the checkpoint; calibrating the LM
head does not load its weights. Artifact payloads are written in chunks, including
embedding tables.

Peak memory depends on the largest F32 layer, its input statistics and one sample's
execution workspace. GPTQ also needs that layer's Hessians and one projection's
factorization workspace. These still need to fit; a large MoE layer can remain limiting.
Use `imatrix` to avoid Hessians, or `rtn` to avoid calibration entirely. Reducing
`--sample-tokens` reduces activation workspace.

Hidden states are spooled to two temporary files under the output directory, requiring
up to `2 × calibration_tokens × hidden_size × 4` additional disk bytes. They are removed
on success and ordinary errors. A forcibly terminated process can leave a
`.tur-calibration-*` directory that can be deleted before retrying. Artifact publication
also needs space for both the temporary payload and the completed weights file.

## Mixed precision

`--budget-bits 5.5` measures each projection's sensitivity: the mean KL divergence from
F32 on the calibration samples when only that projection is quantized. It then upgrades
projections from `--weight-format` to `q8_0`, applying the largest divergence reduction
per added byte first, until the budget is spent. The allocation assumes divergences add,
which the measurements show is only approximately true. Q5K and Q6K are
available only for embedding tables, because Tur has no small-batch kernels for them.
This whole-model sensitivity measurement still keeps the complete F32 model and all
calibration statistics in memory; `--budget-bits` does not use layer-wise calibration.

## Decisions

Measured on Qwen3.5-0.8B on an M1:

- **Load-time `q8_0` stays the default** for SafeTensors checkpoints. It needs no
  calibration data and is the most accurate packed format.
- **GPTQ is the default artifact method.** At equal bytes it cuts Q4K's KL divergence
  from F32 by 49–59% on held-out prose, code and retrieval text, and by 12% on tool/JSON
  text, where it ties with `imatrix`.
- **Uniform calibrated conversion streams layers.** Shared calibration infrastructure
  owns disk-backed activations and layer/statistics lifetimes; model adapters supply
  loading and reference execution. Dense, tied and MoE fixture payloads match resident
  calibration on CPU and Metal for GPTQ and imatrix. This removes the whole-model F32
  requirement without calibrating against already quantized layers. Large-checkpoint
  peak memory has not yet been measured.
- **Mixed precision is opt-in.** At 5.5 bits per weight, KL falls another 34–58% for 23%
  more resident bytes, at 0.86× single-sequence decode speed.
- **Tied checkpoints gather embeddings from the packed LM head** (`--embedding head`, the
  artifact default). This cuts Q4K resident weights from 933 to 425 MB at unchanged
  quality, and Q8_0 from 1,309 to 801 MB.
- **Prepacked artifacts load directly.** They load byte-identical to load-time
  conversion: Q4K in 0.2 s instead of 9 s, with half the peak memory.
- **Projections are F32, Q8_0 or Q4K only.** Candle's Q5K/Q6K kernels at 2–16 rows are
  2.5–3.5× slower than Tur's Q8_0/Q4K kernels.
- **AWQ-style rescaling and DWQ are not implemented.**
