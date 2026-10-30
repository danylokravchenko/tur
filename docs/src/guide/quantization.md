# Prepacked weights

Tur stores projection weights in packed block formats (8-bit Q8_0 and 4-bit Q4K, from GGML), and its kernels read those blocks directly. Fewer bytes per weight means less memory and, when one conversation is decoding, more tokens per second, because that work is limited by how fast the GPU can read weights.

You can pack a SafeTensors checkpoint in two ways:

- At load time, with `--weight-format f32|q8_0|q4k` on `run` or `serve` (the default is `q8_0`). This uses plain rounding and repeats the work on every start.
- Offline, with `tur quantize`, which writes a prepacked artifact. It can calibrate the rounding on your text, give sensitive projections more bits, and it loads as stored in a fraction of a second.

These are not GGUF files, and `q4k` means Q4_K blocks for every projection, not llama.cpp's Q4_K_M mix.

Calibration (`imatrix`, `gptq` and `--budget-bits`) is available for the Qwen family. Gemma 4 artifacts use plain rounding: `tur quantize --method rtn --weight-format q8_0`. An E2B Q8_0 artifact loads in about 5 s, against about 12 s for converting at load time.

## Writing an artifact

```sh
tur quantize \
  --model /models/qwen3.5-0.8b --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output /models/qwen3.5-0.8b-q4k-gptq \
  --calibration prose.txt,code.txt,tools.txt \
  --report /models/qwen3.5-0.8b-q4k-gptq.json
```

Then load it without `--weight-format`, which is an error with an artifact:

```sh
tur serve --model /models/qwen3.5-0.8b-q4k-gptq --revision q4k-gptq-v1
```

The artifact is one `tur-weights.safetensors` file next to copies of the source's config, tokenizer, template and generation defaults. Its header holds a manifest with the format version, the source checkpoint's identity and a fingerprint of its `config.json`, the quantizer settings, a fingerprint of the calibration data, and the format chosen for each projection. Tur checks all of it before reading any weights, so an artifact can't be loaded against the wrong config. The file is written under a temporary name and renamed when complete, so a failed run never leaves a half-written artifact.

## Choosing a method

| Method | Needs calibration | What it does |
| --- | --- | --- |
| `rtn` | no | Rounds each weight to the nearest value; the same result as load-time conversion |
| `imatrix` | yes | Chooses block scales that protect the weights your text uses most |
| `gptq` (default) | yes | Rounds column by column and corrects the remaining columns for the error so far |

Every method produces ordinary blocks, so the same kernels run the result and speed is the same at equal formats. Only quality differs: on held-out text GPTQ roughly halves the error of plain 4-bit rounding.

Keep calibration text separate from the text you judge quality on, or the measurement will flatter the result.

### Memory and disk

GPTQ and imatrix calibrate one layer at a time. Tur loads a layer in F32, runs the calibration samples through it, packs its projections, frees it, and moves on. Each layer sees the activations the unquantized model would produce, so errors don't compound from one layer to the next, and the whole F32 model never has to fit in memory.

Peak memory is set by the largest F32 layer, its input statistics and one sample's workspace. GPTQ also holds that layer's Hessians. A large MoE layer can still be the limit; `imatrix` avoids the Hessians, `rtn` avoids calibration entirely, and a smaller `--sample-tokens` shrinks the workspace.

Between layers, hidden states go to two temporary files in the output directory, up to `2 × calibration_tokens × hidden_size × 4` bytes. They are removed when the run ends normally. A killed process can leave a `.tur-calibration-*` directory, which you can delete before retrying. Writing the artifact also needs room for a temporary copy of the weights file.

## Mixed precision

`--budget-bits 5.5` gives some projections 8 bits and the rest the base format, within an average of 5.5 bits per weight. Tur first measures how much each projection hurts on its own: the KL divergence from F32 on the calibration text when only that projection is quantized. It then upgrades projections to `q8_0` in order of quality gained per extra byte until the budget is spent. This assumes the errors add up, which is only roughly true.

Mixed precision measures the whole model at once, so it needs the full F32 model and its calibration statistics in memory.

## Decisions

Measured on Qwen3.5-0.8B on an M1:

- Load-time `q8_0` stays the default for SafeTensors checkpoints. It needs no calibration data and is the most accurate packed format.
- GPTQ is the default artifact method. At equal size it cuts Q4K's KL divergence from F32 by about half on held-out prose, code and retrieval text, and by 12% on tool calls and JSON, where it ties with `imatrix`.
- Calibration streams one layer at a time. Dense, tied and MoE fixtures produce the same weights as calibrating the whole model at once, on CPU and Metal. Peak memory on large checkpoints hasn't been measured yet.
- Mixed precision is opt-in. At 5.5 bits per weight it cuts KL by another third to half, but costs 23% more memory and 14% of single-sequence decode speed.
- Tied checkpoints read embeddings from the packed output layer (`--embedding head`, the artifact default). That drops the separate embedding table at unchanged quality: a Q4K artifact of the 0.8B holds 425 MB of weights instead of 933 MB.
- Prepacked artifacts load directly, byte-identical to load-time conversion. The 0.8B Q4K artifact loads in 0.2 s instead of 9 s, with half the peak memory.
- Projections are F32, Q8_0 or Q4K only. Candle's Q5K and Q6K kernels at 2 to 16 rows are 2.5 to 3.5× slower than Tur's Q8_0 and Q4K kernels, so those formats are offered only for embedding tables.
- AWQ-style rescaling and DWQ are not implemented.
- Artifacts are written in one pass, into a reserved header and payload in the final file, so a conversion costs its size in SSD writes rather than twice that.
- gpt-oss ships its experts in MXFP4 and runs them as stored, so it has no artifacts. `tur prune` is its only size reduction, and it is an experiment: see [Expert cache](../architecture/expert-cache.md).
