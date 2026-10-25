# Dense Qwen3.5-0.8B pruning pilot

This is the first local experiment for the [partial-model initiative](partial-model-initiative.md),
run on 2026-10-25. It tests two independent reductions of the existing dense checkpoint:
remove whole recurrent layers, or shrink every layer's MLP while keeping all layers.
No recovery training or distillation has been applied. Flash-Next's MoE architecture
is not implemented by this pilot.

## Reference and protocol

- Apple M1, MacBookPro17,1, 16 GB memory, macOS 26.6.2.
- `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
- The text tower has **752,393,024 parameters**: 254,279,680 in the tied embedding,
  264,241,152 in dense MLPs and 233,872,192 in mixers and norms. Vision and MTP are
  excluded from all candidates and are not loaded by the text-only reference run.
- Calibration uses the first 256 untemplated tokens of [calibration.txt](partial-model-pilot/calibration.txt).
  The four other authored text files are disjoint diagnostic samples: code, prose,
  JSON/tool text and arithmetic. Their fingerprints and measurements are in
  [results.json](partial-model-pilot/results.json).
- Quality comparisons use Tur's fused Metal path, Q8_0 weights unless explicitly
  Q4K, F32 activations/K/V/recurrent state, 16-token pages, 64-token chunks and at most
  256 tokens per sample. Each model receives identical token IDs. The benchmark
  rejects separate checkpoint directories whose tokenizer bytes differ.
- An additional standalone CLI generation uses the original chat template, thinking
  off, greedy sampling, seed 299792458 and a 128-token output cap. It asks for a Python
  `sum_even` function. Outputs are saved as data and were not executed.

The four diagnostic samples have 215/155/210/161 scored next-token positions, 741 in
total. They are too small to estimate coding pass rates, general reasoning accuracy,
JSON validity or retrieval performance. No quality-loss allowance or latency ceiling
has been agreed. There is no final benchmark split or equal-total-memory comparison.

## Permanent layer removal

`just bench layer_influence` ranks layers by mean per-token
`1 - cosine(layer input, layer output)`, measured with Q8_0 weights. The first layer
has no traced input and is protected. The last layer and every full-attention layer
are also protected. Lower-ranked remaining layers are removed jointly, without
rescaling. The [calibration report](partial-model-pilot/layer-influence.json) records
all scores and the original config.

| Variant | Removed original layers | Text parameters | Resident weight MB | Mean KL | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| Full Q8_0 / all-layer export | none | 752,393,024 | 1,309.4 | 0 | 100% |
| Full Q4K | none | 752,393,024 | 933.5 | 0.0831 | 86.9% |
| 22 layers, Q8_0 | 20, 21 | 709,282,304 | 1,263.5 | 0.2588 | 78.3% |
| 20 layers, Q8_0 | 12, 20, 21, 22 | 666,171,584 | 1,217.5 | 1.1167 | 48.4% |
| 18 layers, Q8_0 | 9, 12, 13, 20, 21, 22 | 623,060,864 | 1,171.6 | 1.6448 | 35.9% |

MB are decimal. KL is `KL(full Q8_0 || candidate)` in nats; agreement is with the
full Q8_0 top token. Combined values weight each domain by its scored positions.
Per-domain KL, maximum KL, agreement, target-token accuracy and NLL are retained in
the results JSON. The all-layer export has zero measured KL and 100% top-1 agreement
in all four samples, establishing a control for the export conversion.

The original Q8_0 and Q4K models both produce the requested `sum_even` function. The
22-layer model emits an invented module import and no function. The 20-layer and
18-layer models become repetitive and reach the output cap. Full Q4K has lower KL
in every domain and uses fewer resident weight bytes than any pruned Q8_0 candidate.
This unrecovered layer-selection method is rejected on this checkpoint.

## Shrinking layers instead of removing them

Width pruning keeps the complete 24-layer sequence. For each layer, remove matching
rows of `gate_proj` and `up_proj`, and the corresponding columns of `down_proj`.
Preserve the original order of retained neurons. This keeps the residual width,
attention/recurrent dimensions, tokenizer, vocabulary and output head unchanged.

`just bench neuron_influence` reuses Tur's existing F32 calibration recorder. For
neuron `j`, the score is `E[a_j²] × ||W_down[:, j]||²`, where `a` is the post-SwiGLU
activation. This estimates the squared output energy of each individual neuron;
it ignores correlations between neurons and does not establish joint importance.
Each layer retains its highest-scoring neurons independently. The exporter copies
their original tensor bytes and performs no training or compensating rescaling.

| MLP width in every layer | Text parameters | Parameter reduction | Resident weight MB | Mean KL | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3,584 (100%) | 752,393,024 | 0% | 1,309.4 | 0 | 100% |
| 2,688 (75%) | 686,332,736 | 8.8% | 1,239.3 | 0.5136 | 66.4% |
| 1,792 (50%) | 620,272,448 | 17.6% | 1,169.1 | 1.1625 | 52.0% |
| 896 (25%) | 554,212,160 | 26.3% | 1,098.9 | 2.0449 | 40.5% |

All variants keep 24 layers. The all-width control has zero measured KL and 100%
top-1 agreement in all four samples. At about 620M parameters, halving MLP width preserves
more top-token agreement than the similarly sized 18-layer candidate, but both fail
the code-generation check. All three reduced-width outputs become repetitive and
reach the output cap; none produces a usable function. Full Q4K remains smaller in
resident weight bytes and has lower KL in every diagnostic domain. This initial
unrecovered neuron-selection method is also rejected as a quality configuration.

`results.json` contains the width measurements and generation outputs under `width`.
The complete neuron calibration report is retained locally in
`target/partial-neuron-influence.json`; the results record its SHA256, calibration
fingerprints and reproduction method. Each local candidate manifest includes every
retained neuron ID and the complete calibration scores. No candidate is promoted.

Reducing this model's MLP width to 25% cannot reduce the entire model to 25%:
the embedding alone already contains 254M parameters. For a larger MoE model,
the same coupled pruning can shrink every expert's intermediate width while
retaining its layers and expert IDs. The unchanged components still set a floor;
the actual inventory is needed before claiming that a 125B model becomes 30B.
Reducing residual width and attention/recurrent heads would require additional
coordinated transformations and validation. [Minitron](https://arxiv.org/abs/2407.14679)
provides evidence for combining structured width pruning with recovery/distillation;
it does not establish unrecovered quality on Qwen's hybrid architecture.

## Export and reproduce

`scripts/partial_model.py` needs Python 3.11+ and no third-party packages. It reads
and validates SafeTensors headers, streams tensor bytes, converts the nested model
to an ordinary `Qwen3_5ForCausalLM` text checkpoint, removes vision/MTP weights and
sets MTP depth to zero. The destination must not exist. A complete directory is
published by rename; ordinary failures clean up its staging directory.

Each output contains `partial-model.json`: source revision, config and shard SHA256
hashes, copied tokenizer/template hashes, tensor precision, parameter counts, weight
hash, calibration provenance, original layer IDs and retained neuron IDs where
applicable. Copied tokenizer/template/generation files retain their source bytes.
The source checkpoint is never overwritten.

```sh
export TUR_QWEN35_0_8B="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
pilot_revision=2fc06364715b967f1860aea9cf38778875588b17
pilot_data="$PWD/docs/src/research/partial-model-pilot"

# F32 activation statistics for shrinking each feed-forward block.
just bench neuron_influence "$TUR_QWEN35_0_8B" "$pilot_revision" \
  "$pilot_data/calibration.txt" 256 > /tmp/tur-neuron-influence.json

python3 -B -I scripts/partial_model.py \
  --model "$TUR_QWEN35_0_8B" --revision "$pilot_revision" \
  --neuron-scores /tmp/tur-neuron-influence.json --width 1792 \
  --output "$PWD/target/qwen35-width-1792"

TUR_EVAL_TEXT="$pilot_data/code.txt" TUR_EVAL_KV='' \
TUR_EVAL_WEIGHTS="$PWD/target/qwen35-width-1792,q4k" \
TUR_EVAL_TOKENS=256 TUR_EVAL_CHUNK=64 just bench kv_quality > /tmp/tur-width-code.json

cargo run --release --bin tur -- run \
  --model "$PWD/target/qwen35-width-1792" --revision "$pilot_revision-width-1792" \
  --temperature 0 --thinking false --seed 299792458 -n 128 \
  --prompt 'Write a Python function sum_even(numbers) that returns the sum of the even integers in numbers. Return only the code.'

python3 -B -I scripts/test_partial_model.py
```

Repeat quality comparisons with `prose.txt`, `json.txt` and `reasoning.txt`, and widths
2688/896/3584. Widths used here are multiples of 32 for Q8_0 packing. The all-width
3584 export is an identity control. Layer removal instead uses `layer_influence`,
`--scores <report> --remove 2|4|6`; an empty `--drop-layers ''` exports all text layers.
Use a fresh destination for every export. Depth and width pruning cannot be combined
in one export in this pilot, keeping their initial effects separately measurable.

## Decisions and limits

- Keep pruning offline and reuse ordinary checkpoint loading, calibration and the
  existing quality benchmark. There are no serving flags, alternate execution paths,
  new dependencies or changes to shared model descriptors.
- Reject the measured unrecovered layer-removal candidates. Preserve their outputs
  and calibration to distinguish a failed selection strategy from an exporter error.
- Reject the measured unrecovered width candidates as quality configurations too.
  Shrinking dimensions is mechanically valid, but this activation-energy ranking
  alone does not preserve generation. Investigate better calibration, nonuniform
  width allocation and recovery/distillation before attempting a fourfold reduction.
- Resident weight bytes are measured by Tur; total physical footprint, startup peak,
  swap, TTFT and throughput are not measured here. In particular, load-time conversion
  keeps a separate F32 embedding table; the existing prepacked tied-head option can
  reduce it. These numbers are not a best-packing or device-fit claim.
- These diagnostics do not establish retained coding/reasoning capability or support
  for Flash-Next. Larger held-out task evaluation and recovery remain open work.
