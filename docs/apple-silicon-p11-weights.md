# P11: calibrated weights and bounded loading

2026-10-24, Apple M1 (8 GPU cores, 16 GB), macOS 26.6.2, pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`. Base commit `9a78ff6` (P10). Only the M1 and
the 0.8B checkpoint are measured; the flagship cannot run on this Mac.

P9 has not run, so there is no frozen comparative baseline or predeclared quality budget.
Quality is measured against the F32 checkpoint on held-out text disjoint from the
calibration data, one run per configuration, without confidence intervals.

## Outcome

| Item | Result | Status |
| --- | --- | --- |
| 1. Calibrated conversion | At equal bytes, GPTQ cuts Q4K's KL by 50% on prose, 59% on code and 49% on retrieval text, and 12% on tool/JSON text (tied with imatrix there). AWQ-style rescaling and DWQ not implemented (reasons below) | **GPTQ kept** |
| 2. Sensitivity and byte-budget allocation | Per-projection sensitivity plus greedy Q4K→`Q8_0` upgrades at 5.5 bits per weight: a further 34–58% lower KL for +23% resident bytes, 0.86× single-sequence decode | **Kept, opt-in** |
| 3. Formats with efficient Metal execution | Q5K/Q6K lack Tur small-batch kernels; Candle's path is 2.5–3.5× slower than Tur's `Q8_0`/Q4K at 2–16 rows. Projections are F32, `Q8_0` or Q4K; Q5K/Q6K only for gathered embedding tables | **Measured** |
| 4. Prepacked artifacts, bounded loading | Versioned artifacts with pinned source, quantizer and calibration identities, validated before any payload is read, loaded byte-identical to load-time conversion. Q4K loads 37–53× faster (0.17–0.24 s vs 8.9–9.2 s) with half the peak RSS | **Kept** |
| Tied embedding from the packed head (found while sizing item 4) | Q4K resident weights 933 → 425 MB (2.2×) at unchanged quality; `Q8_0` 1,309 → 801 MB | **Kept** (artifact default; opt-in for load-time conversion) |
| 5. Load time, peak memory, quality-vs-bytes | All measured below; decode and prefill speed unchanged at equal formats | **Measured** |

Exit gate: on this checkpoint, the selected Q4K GPTQ artifact improves quality at equal
bytes in every evaluated domain, and packed-kernel conformance passes (artifacts are
byte-identical to conversion; real-checkpoint and workspace suites pass). Same-bit
calibration is not a speedup claim; speed is unchanged.

## Held-out quality

Teacher-forced next-token distributions, 8,192 positions per text (7,206 for retrieval),
with F32 weights, F32 K/V and F32 recurrent state as the reference. Every artifact gathers
its input embedding from the packed tied head. Calibration used 16,384 tokens from other
sources ([data](#data)). Results:
[prose](baselines/2026-10-24-m1-p11/quality-prose.json),
[code](baselines/2026-10-24-m1-p11/quality-code.json),
[tools](baselines/2026-10-24-m1-p11/quality-tools.json),
[retrieval](baselines/2026-10-24-m1-p11/quality-retrieval.json), and the mixed artifact's
`quality-mixed-*.json`.

Mean KL from the F32 distribution (nats), with top-1 agreement in brackets:

| Weights | Resident bytes | Prose | Code | Tools/JSON | Retrieval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4K, round to nearest | 424.9 MB | 0.141 (79.7%) | 0.083 (88.4%) | 0.114 (93.9%) | 0.058 (87.0%) |
| Q4K, imatrix | 424.9 MB | 0.094 (83.3%) | 0.051 (90.8%) | 0.099 (94.7%) | 0.037 (88.9%) |
| **Q4K, GPTQ** | 424.9 MB | **0.070** (84.4%) | **0.034** (92.1%) | 0.100 (94.5%) | **0.030** (89.6%) |
| Mixed GPTQ, 5.5 bits/weight | 521.1 MB | 0.035 (88.6%) | 0.018 (94.2%) | 0.066 (96.0%) | 0.014 (90.9%) |
| `Q8_0`, round to nearest | 800.9 MB | 0.0008 (98.4%) | 0.0004 (99.2%) | 0.0014 (99.4%) | 0.0003 (98.3%) |

Mean negative log-likelihood of the text (lower is better):

| Weights | Prose | Code | Tools/JSON | Retrieval |
| --- | ---: | ---: | ---: | ---: |
| F32 reference | 3.700 | 1.582 | 0.728 | 1.268 |
| Q4K, round to nearest | 3.852 | 1.647 | 0.766 | 1.323 |
| Q4K, GPTQ | 3.771 | 1.615 | 0.764 | 1.303 |
| Mixed GPTQ | 3.726 | 1.599 | 0.748 | 1.290 |
| `Q8_0` | 3.700 | 1.582 | 0.727 | 1.268 |

The tool/JSON text gains least from calibration. Half of the chat-template renderings
were calibration data, so the remaining gap is probably not a shortage of matching
format; the text is dominated by the MCP schema, whose tokens are already predicted at
94% agreement. Retrieval answers do not discriminate: every variant reproduces 299–300
of 300 answer digits (F32: 300), so the test shows that Q4K preserves this 7K-token
recall and nothing finer.

Load-time Q4K conversion with a BF16 embedding table measured 933.5 MB at prose KL 0.139,
and `Q8_0` 1,309 MB at 0.0007: sharing the tied head costs no measurable quality.

## Calibration methods

- **Round to nearest** is GGML's own scale search, as load-time conversion always does.
- **imatrix** collects each projection input's mean squared activation per column during
  F32 reference passes and gives it to Candle's importance-weighted K-quant search
  (llama.cpp's imatrix weighting). `Q8_0` has one scale per 32 values and ignores it.
- **GPTQ** also collects `E[x xᵀ]` per projection, factors the damped inverse (1% of the
  mean diagonal) in f64, and quantizes columns in order onto the exact GGML `Q8_0`/Q4K
  grid: each block's scales come from the importance-weighted search over the already
  compensated weights, then each value is rounded onto that grid and its error spread
  over the remaining columns. Rows are independent and processed in parallel tiles.
  Expert banks, whose inputs differ per expert, fall back to imatrix per expert.
- **AWQ-style scaling** was not implemented: folding per-channel scales needs the
  preceding operation to absorb their inverse, which the shared norms, gated attention
  output and per-head gated `DeltaNet` norm do not uniformly allow, and imatrix already
  covers its salience criterion. **DWQ** is unnecessary while GPTQ beats the simpler
  options.

Three failures appeared on the real checkpoint and are fixed with regression tests:

- Some rows of layer 0's recurrent input projection hold weights near 1e-37. The weighted
  search squares the weights, which underflow to zero, giving NaN block scales and NaN
  logits for both imatrix and GPTQ. Rows whose weighted encoding does not decode to finite
  values are re-packed by plain rounding.
- Columns calibration never activates would give zero-weight sub-blocks; importance has a
  floor of 1e-6 of its mean.
- Divergence measurement converted to F64 on the device, which Candle's Metal backend
  lacks; per-position KL is now summed on the host in f64, with a Metal test.

## Mixed precision

Sensitivity is the teacher-forced KL on 1,024 calibration tokens when one projection is
GPTQ-packed in Q4K or `Q8_0` and every other projection stays F32: 206 measurements over
103 projections (the MTP head is never reached by teacher forcing and keeps the base
format). Allocation starts from Q4K and repeatedly applies the upgrade with the largest KL
reduction per added byte that fits the budget, assuming divergences add
([plan and sensitivities](baselines/2026-10-24-m1-p11/quantize-mixed55-gptq.json)).

At 5.5 bits per projection weight, 48 of 109 projections became `Q8_0`: 13 of 24
`gate_up`, 13 of 18 recurrent output projections, every attention `kv` and `o`, 7 `down`
and 3 recurrent input projections, concentrated in the first layers. The LM head is the
most sensitive single projection (KL 0.0094 alone) but its 127 MB upgrade did not fit the
106 MB of headroom; a larger budget or an untied head would choose it first. The sum of
single-projection KLs (0.036) is about half the measured whole-model prose KL of uniform
GPTQ Q4K (0.070), so errors do not add exactly; the allocation is a ranking heuristic, and
the result above is measured, not predicted.

Quality against resident bytes on held-out prose: 424.9 MB → KL 0.070, 521.1 MB → 0.035,
800.9 MB → 0.0008 (and round-to-nearest 424.9 MB → 0.141).

## Speed

Fused passes, F16 K/V, page 16, best median of two interleaved runs
([run 1](baselines/2026-10-24-m1-p11/speed-1.json),
[run 2](baselines/2026-10-24-m1-p11/speed-2.json)). Decode cells are per-step times.

| Workload | Q4K conversion | Q4K GPTQ artifact | Mixed artifact | `Q8_0` conversion | `Q8_0` artifact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill 2,048 tokens | 1.708 s | 1.706 s | 1.658 s | 1.561 s | 1.561 s |
| Decode ×1, context 128 | 10.95 ms | 10.87 ms | 12.41 ms | 17.36 ms | 17.34 ms |
| Decode ×4, context 2,048 | 21.3 ms | 21.6 ms | 22.4 ms | 24.2 ms | 24.2 ms |
| Decode ×16, context 128 | 45.0 ms | 45.5 ms | 44.2 ms | 44.4 ms | 45.0 ms |
| One sequence through `HybridEngine` | 90.9 tok/s | 92.4 tok/s | 79.8 tok/s | 57.3 tok/s | 57.3 tok/s |

Equal formats run at equal speed: artifacts change no kernel. The mixed plan's
single-sequence decode costs what its extra bytes predict (+23% bytes, 1.13× step time),
and batched decode and prefill are unchanged within noise.

## Kernel formats

Packed projection time at Qwen3.5-0.8B and Qwen3.8-27B shapes, best of Tur's and
Candle's kernels per case ([artifact](baselines/2026-10-24-m1-p11/linear-formats.json)).

| Shape, rows | `Q8_0` | Q4K | Q5K | Q6K |
| --- | ---: | ---: | ---: | ---: |
| 0.8B lm_head, 1 | 4.71 ms | 2.37 ms | 3.71 ms | 4.33 ms |
| 0.8B lm_head, 4 | 5.01 ms | 3.88 ms | 10.59 ms | 10.24 ms |
| 27B gate_up, 1 | 3.17 ms | 1.73 ms | 2.10 ms | 2.52 ms |
| 27B gate_up, 8 | 3.81 ms | 2.74 ms | 7.38 ms | 7.10 ms |
| 27B down, 128 | 11.8 ms | 13.7 ms | 14.9 ms | 14.2 ms |

Q5K/Q6K are competitive at one row and in prefill but have no small-batch kernel, so they
would slow batched decode and speculative verification.

## Artifacts and loading

`tur-quantize` writes a directory holding `tur-weights.safetensors` and copies of the
source's config, tokenizer, chat template, generation config and licence. The weights
file stores every projection under its loaded name (`….self_attn.kv_proj`,
`….linear_attn.in_proj`, `….mlp.gate_up_proj`, `lm_head`), already stacked and packed as
the kernels read it, plus the small tensors verbatim. Its header carries a manifest:
format version, source identity, an FNV-1a fingerprint of `config.json`, quantizer method
and damping, calibration samples, tokens and token fingerprint, and the weight plan.

Loading checks the version, the config fingerprint and every tensor's dtype and shape
against the plan before reading payloads, then uploads each tensor as stored. The state
identity becomes `<revision>#tur1-<manifest digest>`, so prefix snapshots never mix
artifacts. `tur` and `server` detect an artifact when `--weight-format` is omitted, and
giving a format for an artifact is an error.

Behind this, one declarative projection table (`Config::projections`) replaces the
loader's name-suffix heuristics and drives loading, memory planning, artifacts,
calibration and sensitivity. Load-time conversion reads each projection in row chunks of
at most 64 MiB of F32, and BF16 embedding tables are filled on the device chunk by chunk
instead of passing through two or three full host copies.

Serving load in a fresh process, page cache warm, until the last upload completes; peak
RSS from `/usr/bin/time -l`; two interleaved runs
([data](baselines/2026-10-24-m1-p11/load.jsonl)):

| Weights | Load | Resident weights | Metal allocated | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Q4K, converted at load | 8.90–9.15 s | 933 MB | 964 MB | 1.47 GB |
| Q4K GPTQ artifact | 0.17–0.24 s | 425 MB | 427 MB | 0.71 GB |
| Mixed GPTQ artifact | 0.19–0.20 s | 521 MB | 523 MB | 0.80 GB |
| `Q8_0`, converted at load | 1.60–1.61 s | 1,309 MB | 1,340 MB | 2.11–2.14 GB |
| `Q8_0` artifact | 0.28–0.40 s | 801 MB | 803 MB | 1.20 GB |

Most of the Q4K conversion time is host K-quant search; artifacts skip it entirely.

Conversion-tool cost (fresh process; reports:
[rtn](baselines/2026-10-24-m1-p11/quantize-q4k-rtn.json),
[imatrix](baselines/2026-10-24-m1-p11/quantize-q4k-imatrix.json),
[gptq](baselines/2026-10-24-m1-p11/quantize-q4k-gptq.json),
[mixed](baselines/2026-10-24-m1-p11/quantize-mixed55-gptq.json),
[q8](baselines/2026-10-24-m1-p11/quantize-q8-rtn.json)):

| Artifact | Wall time | Calibration pass | Peak RSS |
| --- | ---: | ---: | ---: |
| `Q8_0`, round to nearest | 2.8 s | — | 0.45 GB |
| Q4K, round to nearest | 10.2 s | — | 0.33 GB |
| Q4K, imatrix | 124 s | 62 s | 4.1 GB |
| Q4K, GPTQ | 297 s | 73 s | 5.1 GB |
| Mixed GPTQ, 5.5 bits/weight | 949 s | 72 s | 5.1 GB |

Calibration loads the whole model in F32 and keeps every projection's second moments, so
it does not scale to the flagship on a 16–36 GB Mac (about 100 GB of F32 weights and
80 GB of Hessians at 27B): that needs layer-at-a-time calibration. Round-to-nearest
artifacts stream and work at that size.

## Data

Built by [make_data.py](baselines/2026-10-24-m1-p11/make_data.py) from pinned local
sources. The held-out prose is `git show 7ca1ecb:docs/<P5 files>`, concatenated as in P5.

| File | Sources | SHA-256 prefix |
| --- | --- | --- |
| calibration | Qwen3.5-0.8B model card, `candle-core 0.11.0` `tensor.rs`, `url 2.5.8` URL-test JSON, even chat-template fixture cases (≈18K characters each) | `048cd208b4fc6036` |
| held-out prose | P5 text | `d4b954c2b213bef2` |
| held-out code | `tokio 1.52.3` multi-thread worker and mpsc channel | `5fb0ba59bac76daa` |
| held-out tools | odd chat-template fixture cases, `rmcp 3.5.1` client JSON-RPC schema | `fcd94f70420b17fb` |
| held-out retrieval | 120 vault codes, `regex 1.13.1` docs as distance, 60 questions with marked answers | `e6bc1d5cbdea129e` |

## Reproduction

```bash
cargo build --release --offline -p tur-cli
target/release/tur-quantize --weight-path <snapshot> --revision Qwen/Qwen3.5-0.8B@2fc06364… \
  --output <dir> --method gptq --format q4k --embedding head --calibration <calibration.txt> \
  [--budget-bits 5.5]
TUR_QWEN35_0_8B=<snapshot> TUR_EVAL_TEXT=<text> TUR_EVAL_KV= TUR_EVAL_REFERENCE=f32 \
  TUR_EVAL_WEIGHTS=<artifact dirs> [TUR_EVAL_MARKED=1] \
  cargo bench --offline -p tur-engine --bench kv_quality
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_FORMATS=q4k,<artifact dirs> TUR_BENCH_KV=f16 \
  TUR_BENCH_PREFILL=2048 TUR_BENCH_CONTEXTS=2048 cargo bench --offline -p tur-engine --bench qwen3_5
cargo run --release --offline -p tur-models --example load_weights -- <dir> <q4k|q8_0|prepacked>
TUR_BENCH_ONLY=linear TUR_BENCH_LINEAR=q8_0,q4k,q5k,q6k cargo bench --offline -p tur-metal --bench kernels
```

## Validation

- [Workspace tests, release](baselines/2026-10-24-m1-p11/nextest.log): 253 passed,
  8 checkpoint tests skipped. [Real-checkpoint tests](baselines/2026-10-24-m1-p11/nextest-checkpoint.log):
  all 8 pass, including Transformers parity of F32 layers, greedy generation and the MTP head.
  Workspace clippy is clean.
- New tests: GPTQ/imatrix encoding, dead-row repair and importance floor (`tur-metal`
  `quantize`); packed row gathers and block round trips (`packed`); plans, projection
  coverage and allocation (`tur-models`); artifact equality with load-time conversion on
  CPU and Metal, validation failures, calibration statistics, calibrated divergence and
  Metal divergence (`tur-engine/tests/prepacked.rs`); factory and pipeline serving with
  MTP (`tur/tests/prepacked.rs`).
- A debug-profile workspace run once failed a timing assertion in `tur::scheduling` under
  parallel load; it passes in release, the profile of the recorded logs.

## Remaining limits

- One model, one device, one run per configuration; no P9 budget or confidence intervals.
- The 0.8B checkpoint is dense: expert-bank calibration is tested only on the MoE fixture.
- Calibration does not fit the flagship on Macs this size; layer-at-a-time calibration is
  the prerequisite.
- Head-shared embeddings are the artifact default but not yet the default for load-time
  conversion.
- One mixed budget was evaluated; the allocation assumes additive divergences, which the
  measurements show is only approximate.
