# P14: confidence-aware speculative decoding

2026-10-24, Apple M1 (8 GPU cores, 16 GB), macOS 26.6.2, rustc 1.99.0, pinned Qwen3.5-0.8B
`2fc06364715b967f1860aea9cf38778875588b17`, Q8_0 weights, F16 K/V, F32 recurrent state,
`pre-release` profile. Base commit `cdb73ee` (P13). Only the M1 and the 0.8B checkpoint
are measured. P9 has not run, so this package measures against the base commit only.

## Outcome

**The native MTP head now pays on novel text, greedy and sampled: 1.20–1.38× over
ordinary decoding at one to four sequences. Against the best P8 policy per workload the
gain is 1.09–1.36×.** On this 0.8B model, a head pass cost about 6.5 ms beside an
18 ms decode, and the head's projection onto the full 248K vocabulary was most of it.
The head now drafts only from the 32K lowest token IDs (byte-level BPE numbers tokens in
merge order, so these are the most frequent). That cuts a head pass to 2.4–3.1 ms at
unchanged acceptance. Each draft also carries the head's own probability, calibrated
online against observed acceptance. With both changes, sampled requests use the head too;
P8 had to disable it for them. The target-matching verifier is unchanged, so outputs are
exact: all 1,278 same-seed request comparisons against ordinary decoding produced identical
token IDs.

The result is within the plan's exploratory target of 1.20–1.50× for novel text at low
concurrency. It comes from a cheaper version of the checkpoint's existing drafter, not
from a new learned one: no compatible learned drafter exists for this checkpoint
(item 3).

| Item | Result | Status |
| --- | --- | --- |
| 1. Best P8 policy, MTP upkeep and memory | Re-measured below; the head pass is the dominant draft cost | Baseline recorded |
| 2. Verification cost and confidence | Per-draft calibrated confidence: ≥ 0.8 → 99% accepted, < 0.4 → 20% (greedy). More stable throughput; enables sampled MTP | **Kept** |
| 2. Kernel-tier cost model | Not built: even with a cheap head, a draft costs ~2.4 ms against 3.25 ms per decode token at 8 sequences | **Rejected by bound** (gate stays at 4) |
| 3. Draft model or learned drafter | No smaller Qwen3.5 exists; no EAGLE-3/DFlash/DSpark drafter targets the 0.8B | **Deferred** (needs training) |
| 3a. Reduced draft vocabulary | Head pass 6.5 → 2.4–3.1 ms; acceptance unchanged at 32K | **Kept** (default 32,768) |
| 4. Stochastic proposals, trees | Not introduced; deterministic target matching is kept | Not applicable |
| 5. Workload matrix | Prose, code, repetitive; greedy and sampled; 1/4/8 sequences | Measured below |

## Where the draft cost went

The P8 draft-cost fits on this commit give a base of 17.6–18.5 ms per decode-only pass at
one sequence and 0.7–1.7 ms per verified draft. The head pass, which runs between
passes and blocks the next one, cost about 6.5 ms. A draft accepted 75% of the time
therefore barely repaid the head. With a 32K draft vocabulary, the measured head pass is
2.4 ms at one sequence (novel greedy), 2.9–3.1 ms at four, and about 1.0 ms when it only
writes keys and values.

Coverage check before building it: in P8's recorded greedy outputs, 98.2% of novel-prose
tokens and 100% of the repetitive workload's tokens had IDs below 32,768. A draft
outside the subset is simply never proposed; verification is unaffected.

Vocabulary sweep, greedy, lookup + MTP, three-run means in tok/s, acceptance in parentheses
([16K](baselines/2026-10-24-m1-p14/vocab16k-greedy.json),
[32K](baselines/2026-10-24-m1-p14/vocab32k-greedy.json),
[64K](baselines/2026-10-24-m1-p14/vocab64k-greedy.json); full vocabulary from the
[baseline](baselines/2026-10-24-m1-p14/baseline-greedy.json)):

| Workload | Full (248K) | 64K | 32K | 16K |
| --- | ---: | ---: | ---: | ---: |
| Novel ×1 | 58.4 (75%) | 68.1 (71%) | 65.2 (64%) | 68.2 (62%) |
| Novel ×4 | 155.5 (75%) | 187.5 (75%) | 192.9 (75%) | 172.4 (61%) |
| Code ×1 | 53.3 (65%) | 70.5 (76%) | 72.1 (72%) | 72.9 (72%) |
| Code ×4 | 177.0 (84%) | 191.6 (83%) | 193.1 (79%) | 194.2 (79%) |

16K loses prose drafts at four sequences, and 64K costs about 1 ms more per head pass.
32K is the default (`speculation::MTP_VOCAB`, CLI and server `--mtp-vocab`; 0 uses the
whole vocabulary). The novel ×1 cells without confidence vary widely between runs
(32K: 54–72 tok/s), which the next change addresses. Extra resident memory: 34 MiB of Q8_0
head rows, charged to the weight plan before loading.

## Calibrated confidence

The head returns `p(draft)`, its softmax probability over the draft vocabulary, computed on
the device beside its argmax. The pipeline keeps a reliability table (ten bins, starting at
the bin's own confidence, decayed over about 200 drafts), separately for greedy and sampled
requests. The planner weighs each draft by its bin's acceptance rather than by the
request's history.

Reliability, cumulative over the final lookup + MTP runs at one and four sequences
(accepted / verified):

| Confidence | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | 2/18 (11%) | 10/35 (29%) | 156/226 (69%) | 147/242 (61%) | 111/143 (78%) | 181/209 (87%) | 129/129 (100%) | 552/560 (99%) |
| Sampled | 69/180 (38%) | 59/181 (33%) | 84/188 (45%) | 106/191 (55%) | 101/133 (76%) | 92/140 (66%) | 182/198 (92%) | 990/1002 (99%) |

At the top the raw probability is underconfident, because a greedy draft is accepted
whenever it is the target's argmax. Sampled acceptance is flatter: a sampled target
accepts low-confidence drafts more and mid-confidence ones less, so the two tables are
kept separate.

Confidence alone (greedy, 32K) left mean throughput unchanged but removed the low runs:
novel ×1 went from 65.2 (54.4–72.1) to 71.0 (69.8–72.4) tok/s
([artifact](baselines/2026-10-24-m1-p14/vocab32k-confidence-greedy.json)).

### Sampled requests and the head's gate

With a cheap head and per-draft confidence, sampled requests (temperature 0.7, top-k 20,
top-p 0.95) gain 1.09–1.38× from the head
([artifact](baselines/2026-10-24-m1-p14/sampled-head-vocab32k-confidence.json)). P8
disabled the head for them. Target matching keeps their distribution exact; acceptance is
the target's probability of the draft.

How the head decides whether to run at all mattered. Its per-request acceptance estimate
gates the head (below 0.3 it stops) and feeds its profitability check. For sampled
requests, rejected low-confidence drafts pulled that estimate down and stopped the head
computing its likely drafts too. After a repetitive workload in the same pipeline,
sampled ×1 code fell to 54.8 tok/s. Three variants were measured on the full matrix:

| Head estimate from | Greedy novel ×4 | Sampled novel ×1 | Sampled code ×1 |
| --- | ---: | ---: | ---: |
| Verification outcomes (P8) | 186.5 | 58.8 (53–62) | 54.8 (51–63) |
| Calibrated confidence of computed drafts | 175.4 (160–184) | 64.7 | 72.4 |
| Verification for greedy, confidence for sampled (**kept**) | **187.2** | **65.2** | **72.7** |

Artifacts for the superseded variants are in
[`superseded/`](baselines/2026-10-24-m1-p14/superseded/).

## Final measurements

Each case warms separately, then runs three bursts with an 80-token output limit; seeds
equal request indices; prefix reuse disabled; one pipeline per configuration serves
repeat, novel, then code. Ordinary and lookup-only from
[greedy](baselines/2026-10-24-m1-p14/final-greedy-ordinary-lookup.json) and
[sampled](baselines/2026-10-24-m1-p14/final-sampled-ordinary-lookup.json) runs (their MTP rows
are the superseded confidence-only variant; those code paths are otherwise unchanged); lookup + MTP
from [greedy](baselines/2026-10-24-m1-p14/final-greedy-mtp.json) and
[sampled](baselines/2026-10-24-m1-p14/final-sampled-mtp.json) runs of the final code; the P8
policy from the [greedy](baselines/2026-10-24-m1-p14/baseline-greedy.json) and
[sampled](baselines/2026-10-24-m1-p14/baseline-sampled.json) baselines on `cdb73ee`.
Output tokens per second, three-run means:

| Mode | Workload | Seqs | Ordinary | Best P8 policy | P14 lookup + MTP | vs ordinary | vs best P8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | Novel | 1 | 52.1 | 58.4 (MTP) | **72.0** | 1.38× | 1.23× |
| Greedy | Novel | 4 | 155.6 | 156.7 (lookup) | **187.2** | 1.20× | 1.19× |
| Greedy | Code | 1 | 53.6 | 53.5 (ordinary) | **72.3** | 1.35× | 1.35× |
| Greedy | Code | 4 | 155.6 | 177.0 (MTP) | **192.2** | 1.24× | 1.09× |
| Sampled | Novel | 1 | 52.8 | 53.2 (lookup) | **65.2** | 1.23× | 1.23× |
| Sampled | Novel | 4 | 155.1 | 155.1 (ordinary) | **177.3** | 1.14× | 1.14× |
| Sampled | Code | 1 | 52.7 | 53.4 (lookup) | **72.7** | 1.38× | 1.36× |
| Sampled | Code | 4 | 154.1 | 154.9 (lookup) | **192.7** | 1.25× | 1.24× |
| Greedy | Novel / code | 8 | 248.2 / 245.9 | ordinary | 237.4 / 239.7 | 0.96× / 0.97× | — |
| Greedy | Repeat | 1 / 4 / 8 | 51.5 / 143.5 / 217.5 | lookup 149.4 / 269.9 / 295.5 | 144.7 / 262.1 / 284.4 | — | 0.96–0.97× |
| Sampled | Repeat | 1 / 4 / 8 | 51.5 / 133.9 / 212.1 | lookup 147.8 / 145.8 / 218.8 | 142.9 / 178.5 / 219.6 | — | 0.97× / 1.22× / 1.00× |

- **Where it loses.** At eight sequences the head computes no drafts (the four-sequence
  gate) but still writes its keys and values: 3–4% below ordinary decoding. On purely
  repetitive greedy text, lookup alone is 3–4% faster for the same reason. MTP is an
  explicit option (`--mtp`), so both remain operator choices; neither regime gains from
  enabling it.
- **Latency.** ITL p95 rises from 18–19 to 22.5–23 ms at one sequence and from about 24 to
  31–32 ms at four. Like P8's modes, this is an opt-in throughput mode, not the default.
- **Memory.** Reserved resident plan, lookup → lookup + MTP: +74 MB at one sequence
  (3,088,767,516 → 3,162,944,028 bytes), +124 MB at four and +192 MB at eight. This
  includes the head layer, its K/V state layer and the 34 MiB draft head.
- **Exactness.** Every run matched the baseline's ordinary decoding token for token: 117
  greedy and 117 sampled comparisons in the final lookup + MTP runs, 1,278 over all runs.

## Per-position acceptance (prompt lookup)

The planner models lookup acceptance at depth `i` as `rate^i`. Measured conditional
acceptance per depth (final runs, cumulative):

| Workload | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
| --- | ---: | ---: | ---: | ---: |
| Greedy ×4 | 260/279 (93%) | 256/268 (96%) | 240/248 (97%) | 236/240 (98%) |
| Sampled ×4 | 48/58 (83%) | 48/48 (100%) | 45/45 (100%) | 42/42 (100%) |

Conditional acceptance rises with depth: once a lookup continuation starts matching, it
keeps matching. A single rate is therefore conservative for deep drafts. The pipeline now
counts this (`SpeculationStats::depth_drafted/depth_accepted`), but a per-depth model was
not built. Where lookup applies, drafts already fill the four-token limit, and the
remaining loss is in regimes where the cost fit, not the estimate, declines drafts.

## Not built

- **Kernel-tier cost model, more sequences.** With the four-sequence gate lifted
  ([ungated](baselines/2026-10-24-m1-p14/ungated-c8-confidence1.json)), eight sequences
  stayed at 234–241 tok/s. The fitted cost there is 26 ms per pass plus 2.3–2.5 ms per
  draft, against 3.25 ms per decode token. Even fully accepted drafts gain under 0.9 ms
  each, about 10% at best before the head pass. A row-tier cost model cannot recover more
  than that, so it was not built. Dropping the head's K/V upkeep while gated (3–4%) would
  need stored hidden states to catch up; not built.
- **Chained MTP drafts.** A second draft through the head, using its own output state,
  would need speculative head K/V writes past the committed position. With the cheaper
  head, the estimated gain is about 10% at one sequence; it is the next candidate.
- **Learned drafters.** No smaller Qwen3.5 checkpoint exists to act as a draft model.
  Published hybrid-Qwen drafters target much larger models (a DFlash speculator for
  Qwen3.5-397B-A17B, a DSpark speculator for a Qwen3.6 MoE). SpecForge's Qwen3.5
  EAGLE-3/DFlash training support is still a proposal. Training a matching drafter is
  a separately sized prerequisite. The DSpark Qwen3.6 drafter becomes relevant when the
  flagship runs on a larger Mac. Sources:
  [Red Hat speculators](https://github.com/vllm-project/speculators),
  [Qwen3.5-397B DFlash speculator](https://huggingface.co/inference-optimization/Qwen3.5-397B-A17B-speculator.dflash-extra),
  [SpecForge Qwen3.5 RFC](https://github.com/sgl-project/SpecForge/issues/486),
  [vLLM speculative decoding](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus).
- **Stochastic proposals and trees.** Deterministic target matching stays the only
  algorithm; no `min(1, p/q)` correction was introduced. Trees need linear verification
  to win first at the concurrency where branch state pays, which this device does not
  show.
- **Combinations.** Prefix reuse stays incompatible with MTP (P8's boundary-state issue,
  unchanged); quantized K/V and narrow recurrent state were not re-measured with MTP.
  Grammar-constrained requests still never draft.

## Changes

- `PackedLinear::leading_rows` copies a projection's first rows (packed bytes are a
  prefix). `LoadOptions::draft_vocab` / `ModelFactory::draft_vocab` give the MTP head its
  own LM head over those rows, included in `memory_requirements` and `weight_bytes`.
- `HybridEngine::draft` returns `Draft { token, confidence }`. `speculation::Calibration`
  learns acceptance per confidence bin; `SpeculationConfig::mtp_vocab` (default 32,768 via
  `with_mtp`); `--mtp-vocab` on the CLI and server.
- The head runs for sampled requests. Greedy requests keep their verification-based head
  estimate; sampled ones follow the calibrated confidence of their computed drafts.
- `SpeculationStats` counts lookup acceptance per depth and head acceptance per
  confidence bin. The serving benchmark adds a code workload (`TUR_BENCH_SPECULATION`),
  runs lookup and lookup + MTP together (`TUR_BENCH_MTP=0,1`) and takes
  `TUR_BENCH_MTP_VOCAB`.

## Validation

- Offline workspace nextest: 255 passed, 8 checkpoint-dependent skipped. Strict Clippy on
  all targets and the no-default-features check pass.
- New tests: leading rows equal the full projection's first outputs (F32, Q8_0, Q4K);
  MTP with a reduced vocabulary preserves greedy and sampled outputs on the reference and
  fused Q8_0 Metal paths, releases all state, and drafts for sampled requests; the planned
  weights include exactly the copied rows; an empty draft vocabulary is rejected;
  calibration and per-depth counters.
- Same-seed token identity on the real checkpoint, as above.

## Remaining limits

- One model, one device, one run of three repeats per configuration, without confidence
  intervals. The two lowest workloads per cell still vary by up to ±3%.
- The 32K bound relies on merge-order token IDs; another tokenizer needs its own coverage
  check. A larger model, whose LM head is a smaller share of its weights, gains less from
  the reduced vocabulary but more from MTP generally; not measured.
- The workloads are one prose prompt, one code prompt and one repetitive prompt, not a
  quality or traffic distribution.

## Reproduction

```bash
export TUR_QWEN35_0_8B=<snapshot>
# Baseline on cdb73ee and the final matrix (add TUR_BENCH_SAMPLING=0.7,20,0.95 for sampled):
TUR_BENCH_WORKLOADS=speculation TUR_BENCH_SPECULATION=repeat,novel,code TUR_BENCH_MTP=0,1 \
  TUR_BENCH_CONCURRENCY=1,4,8 TUR_BENCH_OUTPUT=80 TUR_BENCH_REPEATS=3 \
  cargo bench --offline -p tur-engine --bench serving --profile pre-release
# Vocabulary sweep:
TUR_BENCH_MTP_VOCAB=<16384|32768|65536|0> TUR_BENCH_WORKLOADS=speculation \
  TUR_BENCH_SPECULATION=novel,code TUR_BENCH_DRAFT=4 TUR_BENCH_MTP=1 TUR_BENCH_CONCURRENCY=1,4 \
  TUR_BENCH_OUTPUT=80 TUR_BENCH_REPEATS=3 cargo bench --offline -p tur-engine --bench serving --profile pre-release
cargo nextest run --offline --workspace --test-threads 1
```

Note: the vocabulary-sweep and confidence artifacts predate the final code (their
`TUR_BENCH_MTP_CONFIDENCE` switch was folded in), and `ungated-*` came from a temporary
`MTP_MAX_DECODES = 16` build.
