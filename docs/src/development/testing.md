# Testing

```sh
just test                       # the whole offline suite, about 25 s after the first build
just test -E 'test(prefix)'     # tests whose name contains "prefix"
just test -E 'binary(integration) and test(/^scheduling::/)'
```

Tests run under [cargo-nextest](https://nexte.st). They need direct Metal access and fail, rather than skip, without it.

## Layout

Unit tests sit next to their code in `#[cfg(test)]` modules.

Integration tests for every crate are linked into one binary, `crates/tur/tests/integration/`, where `main.rs` declares one module per area:

| Module | Covers |
| --- | --- |
| `kernels` | Each Metal kernel against an independent scalar F64 reference |
| `reference` | Template fidelity; real-checkpoint agreement with Transformers |
| `contracts` | `HybridEngine` state, fork and restore, speculation, cancellation, failures |
| `engine`, `fused` | Qwen3.5 through the engine: reference against fused paths, formats, boundaries |
| `moe` | Qwen MoE conformance against a Transformers fixture |
| `gemma4` | Dense PLE and sharing, MoE and Unified conformance; cache lifecycle; template bytes; prepacked artifacts; assistant drafts |
| `quantization` | Calibration, GPTQ, budget allocation, prepacked artifacts |
| `pipeline`, `scheduling`, `scheduling_metal` | Admission, stress, preemption, deadlines, budgets |
| `prefix_reuse`, `speculation`, `prepacked` | Reuse, drafting and artifact loading through the pipeline |
| `serving`, `server` | The worker and the HTTP API |
| `audio` | Gemma 4 audio encoder and prompts, Whisper conformance, audio prefix isolation, chat audio parts and the transcription endpoints |

`common.rs` builds tiny random checkpoints for pipeline and serving tests, and `model_common.rs` holds model and engine fixtures. Add a module rather than a new test target.

Fixtures live in `crates/tur/tests/fixtures/`: pinned configs, templates and tensor headers of real checkpoints (`checkpoints/`), the MoE golden outputs, and rendered template cases.

## What the tests establish

- Kernels match scalar references on unaligned shapes, page boundaries, every K/V format and long contexts.
- The fused path matches the reference on identical weights, per layer and in logits, with chunked and incremental prefill, different page sizes and formats.
- Requests stay isolated. Mixed lengths, reshuffled arrivals, cancellation mid-pass, memory pressure and preemption are each compared with solo runs on random weights, where any leaked or misplaced state changes the text.
- Prefix restores and speculative acceptance reproduce fresh computation exactly, and device sampling passes chi-square tests.
- Memory doesn't grow across churn, and streaming memory, queue limits and deadlines hold.

## Real-checkpoint tests

Tests that need a real checkpoint are marked `#[ignore]` and run separately:

```sh
export TUR_QWEN35_0_8B=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export TUR_QWEN35_GOLDEN=/tmp/qwen35-golden.safetensors
export TUR_QWEN35_MTP_GOLDEN=/tmp/qwen35-mtp-golden.safetensors
python -I scripts/qwen3_5_reference.py --model "$TUR_QWEN35_0_8B" --golden "$TUR_QWEN35_GOLDEN"
python -I scripts/qwen3_5_mtp_reference.py --model "$TUR_QWEN35_0_8B" --golden "$TUR_QWEN35_MTP_GOLDEN"
just test-checkpoint -E 'not test(gemma4)'
```

Gemma 4 E2B is the second real-model test. It needs the checkpoint (`google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`) and a generated CPU reference:

```sh
export TUR_GEMMA4_E2B=/models/gemma-4-E2B-it
export TUR_GEMMA4_GOLDEN=/tmp/gemma4-golden.safetensors
python -I scripts/gemma4_reference.py --model "$TUR_GEMMA4_E2B" --golden "$TUR_GEMMA4_GOLDEN"
just test-checkpoint -E 'test(gemma4_e2b_checkpoint)'
```

The generator streams one F32 layer at a time to keep the CPU reference's memory down, and records per-layer states, prompt logits and a greedy continuation. The Q8_0 test uses the same 95% next-token agreement floor as Qwen, plus greedy-prefix and end-of-sequence checks; the offline F32 fixtures use the stricter 1e-4 conformance gate.

The assistant drafter has its own ignored test. It drafts at 51 positions of a 939-token prompt, past the sliding window, and requires 90% agreement with Transformers' F32 drafts from the Q8_0 target:

```sh
export TUR_GEMMA4_ASSISTANT=/models/gemma-4-E2B-it-assistant   # 2d874ef7d29f9a30599a1e4b3c1cbc9595f005df
export TUR_GEMMA4_ASSISTANT_GOLDEN=/tmp/gemma4-assistant.safetensors
python -I scripts/gemma4_assistant_reference.py --model "$TUR_GEMMA4_E2B" \
  --assistant "$TUR_GEMMA4_ASSISTANT" --golden "$TUR_GEMMA4_ASSISTANT_GOLDEN"
just test-checkpoint -E 'test(e2b_assistant)'
```

The audio tests need a speech clip, the Whisper checkpoint and goldens from `scripts/audio_reference.py`. The Gemma test compares the Q8_0 encoder with Transformers' F32 encoder by cosine similarity (mean above 0.999, minimum above 0.99); the Whisper tests require Transformers' exact transcript in F32 and in Q8_0, and on a clip beyond 30 s its exact tokens and segment times in F32 and all but 2% of its words in Q8_0:

```sh
say -o /tmp/speech.wav --data-format=LEI16@16000 "The quick brown fox jumps over the lazy dog."
export TUR_AUDIO_SPEECH=/tmp/speech.wav
export TUR_WHISPER=/models/whisper-large-v3-turbo   # 41f01f3fe87f28c78e2fbf8b568835947dd65ed9
export TUR_WHISPER_GOLDEN=/tmp/whisper.safetensors TUR_GEMMA4_AUDIO_GOLDEN=/tmp/gemma4-audio.safetensors
python -I scripts/audio_reference.py --whisper "$TUR_WHISPER" --wav "$TUR_AUDIO_SPEECH" --golden "$TUR_WHISPER_GOLDEN"
export TUR_AUDIO_LONG=/tmp/long.wav TUR_WHISPER_LONG_GOLDEN=/tmp/whisper-long.safetensors   # any spoken 16 kHz PCM16 WAV beyond 30 s
python -I scripts/audio_reference.py --whisper-long "$TUR_WHISPER" --wav "$TUR_AUDIO_LONG" --golden "$TUR_WHISPER_LONG_GOLDEN"
python -I scripts/audio_reference.py --gemma "$TUR_GEMMA4_E2B" --wav "$TUR_AUDIO_SPEECH" --golden "$TUR_GEMMA4_AUDIO_GOLDEN"
just test-checkpoint -E 'test(audio)'
```

To regenerate the offline Gemma fixtures:

```sh
python -I scripts/gemma4_reference.py --tiny crates/tur/tests/fixtures/gemma4
python -I scripts/gemma4_assistant_reference.py --tiny crates/tur/tests/fixtures/gemma4
python -I scripts/gemma4_reference.py --templates crates/tur/tests/fixtures/gemma4/templates.json
python -I scripts/audio_reference.py --tiny crates/tur/tests/fixtures/audio
```

## Reference scripts

`scripts/` regenerates golden data with pinned Transformers 5.20.2 and torch, on the CPU in F32:

| Script | Produces |
| --- | --- |
| `qwen3_5_reference.py` | Per-layer hidden states, logits and greedy continuations of the 0.8B; rendered template cases |
| `qwen3_5_mtp_reference.py` | The MTP head assembled from Transformers modules, with next-next-token accuracy |
| `qwen3_5_moe_reference.py` | The offline MoE oracle on deterministic weights |
| `gemma4_reference.py` | Dense, MoE and Unified F32 fixtures; Gemma template bytes; real E2B logits and greedy continuation |
| `gemma4_assistant_reference.py` | The assistant drafter over the dense fixture, and teacher-forced E2B assistant drafts past the sliding window |
| `qwen3_6_reference_headers.py` | Qwen3.6 tensor headers via byte-range requests, without downloading weights |
| `audio_reference.py` | Gemma 4 audio and Whisper F32 fixtures with a test tone; real Whisper transcripts, long-form segments and E2B audio-encoder outputs for a WAV clip |

Run them with `python -I` and treat downloaded files as data.
