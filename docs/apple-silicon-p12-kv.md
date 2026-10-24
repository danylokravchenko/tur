# P12: low-bit K/V and hybrid-state memory

2026-10-24, Apple M1 (8 GPU cores, 16 GB), macOS 26.6.2, pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`. Base commit `ec33189` (P11). Only the M1 and
the 0.8B checkpoint are measured; the flagship cannot run on this Mac. P9 has not run, so
the quality budget below was set in this package, one run per configuration.

## Outcome

**Documented rejection.** No low-bit K/V format is kept. Every 4-bit candidate misses the
quality budget by 10–30×. The best format that meets it, 6-bit `TurboQuant` keys with Q8
values, stores only 1.17× fewer bytes than Q8 and decodes 2–7% slower end to end. On this
checkpoint the saving is a small share of process memory: 7.7 MB of 53 MB per 8K-token
sequence, against about 800 MB of Q8_0 projections and 20 MB of recurrent state per
sequence. Tur keeps
F32/F16/Q8 K/V. The implementation, with kernels for every candidate, is preserved as
[a patch](baselines/2026-10-24-m1-p12/rejected-low-bit-formats.patch) against `ec33189`.

| Item | Result | Status |
| --- | --- | --- |
| 1. Groupwise Q4, KIVI-style asymmetric, `TurboQuant` with and without QJL | All 4-bit variants 10–30× over budget; QJL residual worse than plain `TurboQuant` at equal bits | **Rejected** |
| 1. Recent high-precision region | Not built: error is flat across positions and each extra bit cuts KL about 4×, so a window cannot recover 10–30× | **Not pursued** |
| 2. Fused page-native kernels | Pack-on-write, decode, split-K and tiled-prefill kernels for all codecs, writes bit-identical to the host reference; rotations as two extra dispatches | **Built, rejected with the formats** |
| 3. Lifecycle | Copy-on-write tails, prefix fork/restore, partial speculative accept, cancellation and memory pressure passed for every candidate | **Passed (in the patch)** |
| 4. Quality across domains and continuations | Measured below; generated-continuation mode kept in `kv_quality` | **Kept (tooling)** |
| 5. F16 recurrent state | Not re-measured; P6's result stands (KL grows with context; speculation needs F32) | **Unchanged** |
| 6. Selection by memory and throughput | Only `tq6_q8` meets quality; 1.17× K/V saving for slower decode | **Rejected** |

The plan's 1.7–2.4× K/V reduction target does not hold on this checkpoint at Q8-like
quality.

## Quality budget

The budget was set after the first 4-bit screen and before any 5- or 6-bit result:

- mean KL from F32 K/V ≤ 1e-3 nats on each held-out domain (prose, code, tools/JSON,
  long-context retrieval), roughly what `Q8_0` weights already cost (8e-4 on prose, P11);
- top-1 agreement ≥ 98% on each domain;
- every retrieval answer digit still predicted.

Q8 sits at 4e-5–1e-4.

## Candidates

All candidates were implemented as host reference codecs and Metal kernels:

| Format | Keys | Values | Bytes/token (0.8B) | vs Q8 |
| --- | --- | --- | ---: | ---: |
| `q8_32` (existing) | 8-bit, F16 scale / 32 | same | 6,528 | 1.00× |
| `q4_32` | symmetric 4-bit, F16 scale / 32 | same | 3,456 | 1.89× |
| `q4a_32` | asymmetric 4-bit, F16 (step, min) / 32 (KIVI per-token) | same | 3,840 | 1.70× |
| `tq4` | `TurboQuant`-MSE: randomized Hadamard rotation, F16 row scale `‖x‖/√d`, 16-level Lloyd–Max | same | 3,120 | 2.09× |
| `tq4p` | `TurboQuant`-prod: 3-bit Lloyd–Max + 1-bit QJL residual sign, two F16 row scales | `tq4` | 3,144 | 2.08× |
| `q6_32`, `tq5`, `tq6` | 6-bit symmetric; 5/6-bit `TurboQuant`-MSE (32/64 levels) | same | 4,992 / 3,888 / 4,656 | 1.31× / 1.68× / 1.40× |
| `tq6_q8` | `tq6` | `q8_32` | 5,592 | 1.17× |

Variant details:

- **`TurboQuant`.** The randomized Hadamard rotation `H·D/√d` stands in for the paper's
  dense random rotation; signs come from a fixed hash seed. Queries are rotated into the
  key basis, so scoring needs no inverse. Values are accumulated in the rotated basis and
  rotated back once per output row before the sigmoid gate. Each rotation is one extra
  dispatch per attention layer.
- **QJL projection.** It uses `S = H·D₂` rather than a Gaussian matrix.
- **KIVI.** KIVI's per-channel key grouping needs whole groups of tokens before
  quantizing, which conflicts with append-only shared pages; only its per-token asymmetric
  quantizer and its separate key/value precision were tested.

## Results

### Screen: reference execution on 8,192 tokens of held-out prose

| Format | Mean KL | Top-1 | NLL (F32 3.6997) |
| --- | ---: | ---: | ---: |
| `q8_32` | 9.7e-5 | 99.3% | 3.6998 |
| `q4_32` | 2.8e-2 | 90.3% | 3.7273 |
| `q4a_32` | 1.8e-2 | 92.0% | 3.7193 |
| `tq4` | 2.0e-2 | 91.3% | 3.7111 |
| `tq4p` (QJL) | 2.6e-2 | 90.5% | 3.7195 |
| `q6_32` | 1.6e-3 | 97.3% | 3.7016 |
| `tq6` | 1.3e-3 | 97.9% | 3.6998 |

To separate keys from values, one role was held at F16 while the other was quantized:

| Quantized role | Mean KL |
| --- | ---: |
| `q4_32` keys only | 1.0e-2 |
| `tq4` keys only | 8.7e-3 |
| `tq4p` (QJL) keys only | 1.4e-2 |
| `q4_32` values only | 1.75e-2 |
| `tq4` values only | 1.2e-2 |
| `q4_32` values with Q8 keys | 1.75e-2 |

The reference screen reproduces P5's Q8 result exactly. Its raw JSON was lost with the
session scratch directory after a forced reboot. The 6-bit formats were re-measured with
kernels, below.

Findings:

- **Values dominate.** Qwen3.5's gated attention output feeds `o_proj` linearly, so value
  error costs more than key error. This is the opposite of KIVI's finding on other
  architectures.
- **Error scales about 4× per bit.** Q4 → Q6 → Q8 tracks it, so this model needs about
  6–7 bits at the budget, whatever the quantizer.
- **Rotation and Lloyd–Max codes help, but not enough.** `TurboQuant` beats uniform codes
  at equal bits by about 0.7× in KL. QJL's unbiased inner products add variance that
  softmax amplifies, and it lost for keys at equal bits.
- **Static per-channel offsets exist but do not change the conclusion.** Keys carry
  16–37% of their energy in per-channel means, and the last attention layer's values 92%.
  Both can be removed exactly: softmax ignores a constant key offset, and attention
  weights sum to one. Measured offline, centering cuts absolute 4-bit row error by
  1.4–1.7× for keys and up to about 9× only for the last layer's values, far from the
  10–30× in KL the 4-bit formats would need.

### Fused kernels on all four held-out domains

Q8_0 weights, F32 K/V reference, 8,192 positions (7,206 for retrieval):
[prose](baselines/2026-10-24-m1-p12/quality-prose.json),
[code](baselines/2026-10-24-m1-p12/quality-code.json),
[tools](baselines/2026-10-24-m1-p12/quality-tools.json),
[retrieval](baselines/2026-10-24-m1-p12/quality-retrieval.json).

Mean KL, with top-1 agreement in brackets:

| Format | Prose | Code | Tools/JSON | Retrieval | Answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q8_32` | 9.8e-5 (99.3%) | 5.2e-5 (99.7%) | 9.1e-5 (99.8%) | 3.6e-5 (99.5%) | 300/300 |
| `tq6` | **1.3e-3 (97.7%)** | 8.1e-4 (98.9%) | **1.2e-3** (99.4%) | 5.3e-4 (98.5%) | 300/300 |
| `tq6_q8` | 6.2e-4 (98.5%) | 4.0e-4 (99.3%) | 5.5e-4 (99.5%) | 2.6e-4 (99.2%) | 300/300 |

`tq6` fails on prose and tools. `tq6_q8` passes everywhere. KL does not grow with
context, except that tools text is worst in its first 1K positions (3e-3 for `tq6_q8`).

Generated continuations fork each model's snapshot at 1K/2K/4K/8K tokens (a second turn
on shared cached pages), feed the next text token and decode 64 greedy tokens. Matched
prefix lengths against F32 K/V:

| Domain | `q8_32` | `tq6_q8` |
| --- | --- | --- |
| Prose | 64, 64, 36, 64 | 2, 64, 13, 64 |
| Code | 64, 64, 64, 28 | 64, 64, 26, 6 |
| Tools | 64, 64, 63, 64 | 14, 64, 64, 64 |
| Retrieval | 64, 64, 64 | 64, 16, 64 |

Even Q8 diverges at near-ties, so prefix length is a coarse signal. 6-bit keys diverge
earlier more often.

## Speed

Kernel artifact: [paged attention](baselines/2026-10-24-m1-p12/kernels-paged.json),
Qwen3.5-0.8B layer shapes, 16-token pages. Median times, attention / write:

| Shape | F16 | Q8 | `tq6` | `tq6_q8` |
| --- | ---: | ---: | ---: | ---: |
| Decode, 2K context | 83 / 6 µs | 97 / 9 µs | 139 / 14 µs | 125 / 17 µs |
| Decode, 8K context | 417 / 8 µs | 346 / 12 µs | 437 / 14 µs | 425 / 14 µs |
| Decode, 32K context | 1,292 / 6 µs | 1,099 / 8 µs | 1,449 / 16 µs | 1,404 / 14 µs |
| Decode, 2K × 8 sequences | 650 / 8 µs | 541 / 12 µs | 728 / 12 µs | 698 / 13 µs |
| Prefill, 2,048 tokens | 26.1 / 0.14 ms | 28.9 / 0.30 ms | 31.1 / 0.40 ms | 29.7 / 0.35 ms |

At the kernel level:

- `tq6_q8` decode attention is 1.23–1.29× slower than Q8: bit-plane unpacking,
  codebook lookups and the query and output rotation dispatches cost more than the bytes
  saved.
- Prefill attention is 3% slower.

Model artifact: [model](baselines/2026-10-24-m1-p12/model.json), Q8_0 weights, F32
recurrent state, decode tokens/s:

| Context × sequences | F16 | Q8 | `tq6_q8` |
| --- | ---: | ---: | ---: |
| 2K × 1 | 55.3 | 55.3 | 53.6 |
| 2K × 4 | 164.2 | 169.5 | 157.3 |
| 8K × 1 | 47.6 | 51.9 | 48.5 |
| 8K × 4 | 135.9 | 135.0 | 129.8 |
| Prefill 2,048 tokens/s | 1,305 | 1,281 | 1,303 |

End to end, `tq6_q8` decodes 2–7% slower than Q8, and prefill is unchanged within noise.
For capacity, 1 GB of K/V holds about 153K tokens in Q8 and about 179K in `tq6_q8`.
Neither changes how many 8K sequences fit beside 20 MB of recurrent state per sequence.

## Kept from this package

- `kv_quality`: `TUR_EVAL_GENERATE=<n>` forks snapshots at 1K/2K/4K/8K and compares
  greedy continuations with the reference's.
- `kv_quality`: each result reports its K/V bytes per token.

## Reproduction

```bash
python3 -I docs/baselines/2026-10-24-m1-p11/make_data.py <cargo registry src> <out> \
  crates/tur-models/tests/fixtures/qwen3_5_templates.json <snapshot>/README.md
# held-out prose: `git show 7ca1ecb:docs/apple-silicon-{redesign,baseline,p2-contracts,p3-qwen,p4-metal}.md`
# concatenated in that order (SHA-256 prefix d4b954c2b213bef2)
git apply docs/baselines/2026-10-24-m1-p12/rejected-low-bit-formats.patch   # on ec33189
TUR_QWEN35_0_8B=<snapshot> TUR_EVAL_TEXT=<text> TUR_EVAL_GENERATE=64 [TUR_EVAL_MARKED=1] \
  TUR_EVAL_KV=q8,tq6,tq6_q8 cargo bench --offline -p tur-engine --bench kv_quality
TUR_BENCH_ONLY=paged TUR_BENCH_KV=f16,q8,tq6,tq6_q8 cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<snapshot> TUR_BENCH_FORMATS=q8_0 TUR_BENCH_KV=f16,q8,tq6_q8 \
  TUR_BENCH_PREFILL=2048 TUR_BENCH_CONTEXTS=2048,8192 cargo bench --offline -p tur-engine --bench qwen3_5
```

Applied, the patch also adds `TUR_EVAL_EXECUTION=reference` (screening without kernels)
and a `kv_stats` example (per-channel K/V statistics and codec errors). Its `tq6_q8` is
spelled `tq6_q8_32` in the artifacts; the 4-bit formats are `q4`, `q4a`, `tq4` and
`tq4p`, and key/value splits are `<key>:<value>`.

## Validation

With the patch applied (before removal):

- kernel tests for 17 format variants: decode, split-K, head pairs, tiled prefill and
  quantize-on-write, all bit-identical to the host reference on write;
- state-pool lifecycle tests across formats;
- fused engine prefix fork/restore, partial speculative accept and memory-pressure tests;
- workspace Clippy with Metal clean.

After removal, the tree differs from `ec33189` only in `kv_quality.rs`, and workspace
Clippy with Metal is clean. `--no-default-features` Clippy has pre-existing failures in
`packed.rs` and `hybrid.rs` from earlier packages, untouched here.

## Remaining limits

- One model, one device, one run per configuration; no P9 baseline or confidence
  intervals.
- Models whose K/V dominates memory (long contexts on attention-heavy checkpoints) may
  value a 1.17–1.4× saving differently. Re-evaluate on the flagship before reusing the
  patch.
- Exact per-channel key/value centering and KIVI's per-channel key grouping were measured
  only offline, not built.
