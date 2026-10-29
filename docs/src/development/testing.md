# Testing

```sh
just test                       # whole offline suite, about 25 s after the first build
just test -E 'test(prefix)'     # tests whose name contains "prefix"
just test -E 'binary(integration) and test(/^scheduling::/)'
```

Tests use [cargo-nextest](https://nexte.st). They need direct Metal access, and fail
rather than skip without it.

## Layout

- **Unit tests** sit next to their code in `#[cfg(test)]` modules.
- **Integration tests** for every crate are linked into one binary,
  `crates/tur/tests/integration/`. `main.rs` declares one module per area:

  | Module | Covers |
  | --- | --- |
  | `kernels` | Each Metal kernel against an independent scalar F64 reference |
  | `reference` | Template fidelity; real-checkpoint agreement with Transformers |
  | `contracts` | `HybridEngine` state, fork/restore, speculation, cancellation, failures |
  | `engine`, `fused` | Qwen3.5 through the engine: reference versus fused paths, formats, boundaries |
  | `moe` | Qwen MoE conformance against a Transformers fixture |
  | `gemma4` | Dense PLE/sharing, MoE and Unified conformance; cache lifecycle; canonical template bytes; prepacked artifacts; assistant drafts |
  | `quantization` | Calibration, GPTQ, budget allocation, prepacked artifacts |
  | `pipeline`, `scheduling`, `scheduling_metal` | Admission, stress, preemption, deadlines, budgets |
  | `prefix_reuse`, `speculation`, `prepacked` | Pipeline-level reuse, drafting and artifact loading |
  | `serving`, `server` | The worker and the HTTP API |

  `common.rs` builds tiny random checkpoints for pipeline and serving tests, and
  `model_common.rs` holds model and engine fixtures. Add a module rather than a new test
  target.
- **Fixtures** live in `crates/tur/tests/fixtures/`: pinned configs, templates and tensor
  headers of real checkpoints (`checkpoints/`), the MoE golden outputs, and rendered
  template cases.

## What the tests establish

- **Kernels** against scalar references, on non-aligned shapes, page boundaries, every
  K/V format and long contexts.
- **Fused versus reference** execution on identical weights: per-layer and logit
  agreement, chunked and incremental prefill, page sizes and formats.
- **Stress and isolation.** Mixed lengths, reshuffled arrivals, cancellation mid-pass,
  memory pressure and preemption, each compared with solo runs on random weights, where
  leaked or misplaced state changes the text.
- **Exactness.** Prefix restores and speculative acceptance reproduce fresh computation.
  Device sampling passes chi-square tests.
- **Bounds.** No growth across churn, bounded streaming memory, queue limits and
  deadlines.

## Real-checkpoint tests

Tests that need a real checkpoint are `#[ignore]`d:

```sh
export TUR_QWEN35_0_8B=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export TUR_QWEN35_GOLDEN=/tmp/qwen35-golden.safetensors
export TUR_QWEN35_MTP_GOLDEN=/tmp/qwen35-mtp-golden.safetensors
python -I scripts/qwen3_5_reference.py --model "$TUR_QWEN35_0_8B" --golden "$TUR_QWEN35_GOLDEN"
python -I scripts/qwen3_5_mtp_reference.py --model "$TUR_QWEN35_0_8B" --golden "$TUR_QWEN35_MTP_GOLDEN"
just test-checkpoint -E 'not test(gemma4)'
```

The Gemma 4 E2B checkpoint is a second real-model test mule. Its test is ignored in the
offline suite and needs the pinned local checkpoint and generated CPU reference:

```sh
export TUR_GEMMA4_E2B=/models/gemma-4-E2B-it
export TUR_GEMMA4_GOLDEN=/tmp/gemma4-golden.safetensors
python -I scripts/gemma4_reference.py --model "$TUR_GEMMA4_E2B" --golden "$TUR_GEMMA4_GOLDEN"
just test-checkpoint -E 'test(gemma4_e2b_checkpoint)'
```

Download `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The generator streams one F32 text layer at a
time and gathers PLE rows on demand, keeping the CPU reference within a smaller memory
budget. It records per-layer states, prompt logits and a greedy continuation. Q8_0
uses the same 95% next-token agreement floor as Qwen, plus greedy-prefix and EOS checks;
the offline F32 fixtures use the separate 1e-4 equation-conformance gate.

The assistant drafter has its own ignored test. It drafts at 51 positions of a
939-token prompt, past the sliding window, and requires 90% agreement with
Transformers F32 drafts from the Q8_0 target:

```sh
export TUR_GEMMA4_ASSISTANT=/models/gemma-4-E2B-it-assistant   # 2d874ef7d29f9a30599a1e4b3c1cbc9595f005df
export TUR_GEMMA4_ASSISTANT_GOLDEN=/tmp/gemma4-assistant.safetensors
python -I scripts/gemma4_assistant_reference.py --model "$TUR_GEMMA4_E2B" \
  --assistant "$TUR_GEMMA4_ASSISTANT" --golden "$TUR_GEMMA4_ASSISTANT_GOLDEN"
just test-checkpoint -E 'test(e2b_assistant)'
```

Regenerate offline Gemma fixtures with:

```sh
python -I scripts/gemma4_reference.py --tiny crates/tur/tests/fixtures/gemma4
python -I scripts/gemma4_assistant_reference.py --tiny crates/tur/tests/fixtures/gemma4
python -I scripts/gemma4_reference.py --templates crates/tur/tests/fixtures/gemma4/templates.json
```

## Reference scripts

`scripts/` regenerates golden data with pinned Transformers 5.20.2 and torch on the CPU
in F32:

| Script | Produces |
| --- | --- |
| `qwen3_5_reference.py` | Per-layer hidden states, logits and greedy continuations of the 0.8B; rendered template cases |
| `qwen3_5_mtp_reference.py` | The MTP head assembled from Transformers modules, with next-next-token accuracy |
| `qwen3_5_moe_reference.py` | The offline MoE oracle on deterministic weights |
| `gemma4_reference.py` | Dense, MoE and Unified F32 fixtures; Gemma template bytes; real E2B logits and greedy continuation |
| `gemma4_assistant_reference.py` | The assistant drafter over the dense fixture, and teacher-forced E2B assistant drafts past the sliding window |
| `qwen3_6_reference_headers.py` | Qwen3.6 tensor headers via byte-range requests; no weights are downloaded |

Run them with `python -I` and treat downloaded files as data.
