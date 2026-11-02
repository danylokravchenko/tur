# GGUF files

Tur loads GGUF files directly, so you can run the quant you already like from Hugging Face, whatever its size or mix of formats. The file's tensors go to the GPU as stored, with nothing converted, and memory use stays close to the file's size. You choose the file, so you also choose the quality: Tur doesn't check it.

```sh
tur run \
  --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --gguf unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf \
  --gguf-revision 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 \
  --prompt 'What is the capital of France?'
```

`--gguf` takes a local path or `organization/repository/file.gguf`. A remote file needs `--gguf-revision`, a 40-character commit, for the same reason `--revision` does: branches can move. Give the first shard of a split file (`-00001-of-00003.gguf`) and Tur finds the others.

`--model` still names the original checkpoint, because the config, tokenizer and chat template come from there. Tur downloads only those files, not its weights. A local directory holding just `config.json`, `tokenizer.json`, `tokenizer_config.json` and the template works too. `--weight-format` is an error with `--gguf`.

Before reading any weights, Tur checks the file's architecture and every tensor's name, shape and format against the config, so a file for the wrong model fails at once.

## Formats

| Family | Formats |
| --- | --- |
| Floating point | F32, F16, BF16 |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K |
| Legacy | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 |
| Codebook | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| Microscaling | MXFP4 |

A name like `Q3_K_M` or `UD-Q2_K_XL` describes a mix. Real files combine several of these, often within one matrix: a DeltaNet layer's input projection in a `Q3_K_M` file holds Q4_K, Q3_K and Q8_0 rows. Tur keeps each run of rows in its own format and sends it to the fastest kernel for that format and batch size, then joins the outputs.

NVFP4, TQ1_0, TQ2_0, Q1_0, Q2_0 and F8_E4M3 fail with an error naming the tensor.

## Models

| Model | GGUF `general.architecture` | Evidence |
| --- | --- | --- |
| Qwen3.5-0.8B | `qwen35` | Eleven real quants, including pure IQ and legacy files, gave llama.cpp's greedy text in the text compared, apart from one near-tie |
| Qwen3.8-27B | `qwen35` | Header of a real file plans correctly; not run |
| Qwen3.6-35B-A3B | `qwen35moe` | Headers of three real files plan correctly, including MXFP4 experts, and sampled expert blocks decode to plausible values; not run |
| Gemma 4 E2B | `gemma4` | A real Q3_K_S file gives the same probabilities as its weights decoded into Tur's Transformers-validated SafeTensors path |

Other Gemma 4 sizes should share E2B's tensor names, but we haven't checked their files. The 26B expert model's expert tensors have no mapping and fail at load.

## Limitations

- gpt-oss: GGUF files don't load, and fail at load with an error. gpt-oss checkpoints ship their experts in MXFP4 inside SafeTensors and run them as stored, so use those; [Expert cache](../architecture/expert-cache.md) covers running them from disk.
- `--expert-cache-mb` needs a prepacked artifact, because GGUF stores each bank's gate and up rows in separate tensors, so experts aren't laid out as the cache reads them. Every expert stays resident.
- Audio: a GGUF holds only the text model, so Gemma's audio encoder isn't loaded and audio requests are rejected.
- MTP drafting works only when the file carries the head, as Qwen3.8's do; the 0.8B files don't. The head's norm convention is inferred from llama.cpp's converter and hasn't run on real weights.
- An expert bank whose gate and up tensors differ in format is packed again, because expert kernels read one format per bank. This rounds its weights again and costs memory. Bytes are planned exactly.
- Gemma's per-layer embedding table is expanded to BF16, as it is for SafeTensors, so E2B's weights take about 5.8 GB from a 2.4 GB file.
- Calibration (`tur quantize`) reads SafeTensors. GGUF files are used as they are.
- Tur's Gemma output differs from llama.cpp's on the same file. Tur's matches its validated path on the decoded weights, so this comes from the two engines' Gemma code, and we haven't traced it further.

## Performance

With several sequences Tur decodes 1.2 to 1.5× faster than llama.cpp on the same file. With one, it is slightly ahead on files without codebook blocks and about 10% behind on the one measured file that has them. Prefill is within 4% either way. See [Compared with llama.cpp](performance.md#compared-with-llamacpp). If you can convert the checkpoint yourself, [`tur quantize`](quantization.md) gives faster single-sequence decode, because Q4K and Q8_0 have kernels specialized further.

## Decisions

- Tensors load as stored. Decoding to one format would cost memory and fidelity the user chose, and re-encoding would change the quant they picked.
- Mixed matrices run per part. Up to eight rows, each part takes its own fastest kernel and the outputs concatenate, which brought one-row decode from 0.43 to 0.54× of Candle's one-format speed to 0.84 to 0.98×. Above eight rows every part writes its columns of one output through the wide block kernel, with no copy, which raised `UD-Q2_K_XL` prefill from 1,137 to 1,291 tokens/s at 512 tokens. This costs `Q3_K` parts some kernel speed, at 0.77× of Candle's, but no real file measured got slower.
- The model's config, tokenizer and template come from its own checkpoint rather than the file's metadata, so shape validation and templates share one path with SafeTensors.
- Decoders are checked against ggml's own `dequantize_row_*`, on host and device, in every packed format (`scripts/gguf_blocks_reference.py`).
