# P8: speculative verification and expert expansion

2026-10-23. **P8 is in progress.** Prompt-lookup speculation and the Qwen3.5 MoE text
path are implemented. Native MTP and Gemma 4 execution are disabled. Full Qwen3.6-35B-A3B
checkpoint execution, trained-model routing traces and quality gates still need a
larger-memory Mac. This report distinguishes those gates from the small-model evidence.

## Exact verification and proposals

The existing `HybridEngine` target-matching verifier is now connected to the scheduled
pipeline. A decoding sequence supplies its pending input token plus a bounded set of
proposals, with logits requested at **every** position. Attention pages and convolution
and recurrent state are reserved transactionally. Each recurrent layer retains a slot
at each candidate boundary. Publication selects the accepted boundary, advances RNG and
penalty history only for published samples, and releases rejected state after completion.
Cancellation cannot publish the pending transaction.

Proposals are deterministic prompt lookup: match the longest suffix n-gram in the
request's own prompt/output history, verify hash hits against tokens, then propose its
earlier continuation. A continuation that reaches the end of the history repeats with the
period between the two occurrences, so a short cycle (`4 4 4`, `a b a b`) fills the draft
limit instead of proposing a single period. Defaults when enabled: n-grams 1–4, up to four
drafts per request and sixteen across a pass. Limits remain configurable through
`SpeculationConfig`.

The sampling rule is **target matching**, not a probabilistic draft-model interface:

1. Draw from the target's full distribution after temperature, top-k/top-p and penalties.
2. If it equals the proposed next token, accept that token and inspect the next row.
3. On the first mismatch, publish that target draw and discard the remaining proposals.
4. When every proposal matches, publish the last row's target draw as a bonus token.

For a deterministic proposal `d`, acceptance probability is `p(d)`. A mismatching draw
already has the residual distribution `p(x | x != d)`. Combining the two branches gives
the target distribution exactly; no additional acceptance RNG draw or proposal logits
are involved. This can accept less than a stochastic draft model with the usual
`min(1, p/q)` correction. No implementation or performance claim for that algorithm is
made here.

Device sampling constructs each candidate row with its proper RNG offset and the
penalty counts of earlier proposals. Only rows up to the accepted boundary advance
bookkeeping. Grammar-constrained requests use ordinary decoding so their masks remain
on the device. EOS proposals are excluded; an EOS correction terminates publication.
Output, context, packed-pass and prefix-boundary limits cap drafting before submission.

### Adaptation and bounded memory

Each request tracks conditional draft acceptance. A decaying linear fit of completed
decode-only pass time against total drafts is kept separately per decode count. The
planner chooses draft depths in order of expected acceptance while predicted tokens per
second improve. A fit that was never identified explores drafted and ordinary passes.
Afterwards, the last identified per-draft cost stands in whenever recent passes stop
varying their drafts, while the base keeps following recent passes. Previously the fit
forgot its slope roughly 80 undrafted passes later and forced a full-draft exploration
pass, matching the p95 spikes measured on low-reuse prompts. Unsuccessful requests
recover toward the shared prior so they can be retried.

Decodes also draft beside prefill chunks once costs are identified: the scheduler's
estimate for the prefill tokens joins the pass base, so drafts are weighed against the
whole pass they lengthen. Mixed passes never explore unidentified costs and do not feed
the draft fit. The scheduler's prefill cost model receives drafted passes as their
undrafted tokens and time (fitted per-draft cost removed), or not at all before that
cost is known.

Memory planning charges **all simultaneously possible drafts**, capped by running
sequences and the pass limit. It reserves their recurrent checkpoints and logit rows
before loading weights. Scheduling additionally checks actual page and slot demand
beside the pass's ordinary demand; proposals shrink or yield when it does not fit. No
growth headroom is held back for drafts: rejected drafts return their leases when the
pass completes, and accepted ones are progress a decode would lease anyway. (Holding
admission's page-per-sequence headroom disabled drafting entirely under page pressure.)

The host n-gram index retains at most two 8K-token windows, pruning periodically. Its
hash-table capacity is charged to the shared memory budget; preemption drops the index
and rebuilds it on resume. Context size cannot create an unbounded proposal index.
The request's complete token history remains available for normal recomputation.

**Default: disabled.** Both adapters expose `--speculative-draft N` (zero by default).
Narrow recurrent state automatically disables drafting, with a warning: its ordinary per-pass rounding
is not reproduced by the F32 intermediates of a multi-token verification. The validated
speculation path requires F32 recurrent state.
This is an explicit throughput mode: accepted tokens arrive in bursts, with longer
pauses between passes. `PipelineConfig::reproducible` also disables speculation.
The target-matching sampling guarantee does not promise bit-identical logits between
different batch/chunk shapes: existing F16-tile and recurrent rounding differences still
apply. The reproducibility mode fixes those execution choices separately. On the MoE
fixture, verification logits match one-at-a-time decoding within 7e-7 relative for F32,
`Q8_0`, and Q4K windows inside its 2–4-row GEMV range. Wider Q4K windows run Candle's
F16-staged tiles, as batched Q4K decoding of as many rows already does, and drift by
**1.1e-2**, partly through flipped close routing decisions. A test now bounds this
instead of avoiding the regime; removing it needs an F32 matrix-unit dense Q4K kernel.

### Serving measurements

Artifacts: [P8 results](baselines/2026-10-23-m1-p8/). Apple M1, 8 GPU cores, 16 GB unified
memory; pinned Qwen3.5-0.8B `2fc06364715b967f1860aea9cf38778875588b17`, Q8_0 weights,
F16 K/V, F32 recurrent state, chunk 256. Prefix reuse is disabled. Each case warms
separately, then runs three bursts with an 80-token output limit. Seeds equal request
indices. Timings include completed GPU work and publication; loading is excluded.

The `repeat` prompt asks the model to continue a color pattern. `novel` asks about
database crash recovery. These are bounded comparison workloads, not a general quality
evaluation. Raw artifacts retain every token ID, acceptance counts, pass times, memory
reservations and the learned cost fit. Inter-token latency is measured between sampled
token publications, including zero-length gaps within a speculative burst.

Final three-run means ([greedy](baselines/2026-10-23-m1-p8/speculation-greedy.json),
[sampled](baselines/2026-10-23-m1-p8/speculation-sampled.json),
[summary](baselines/2026-10-23-m1-p8/summary.json)); sampled uses temperature 0.7,
top-k 20 and top-p 0.95:

| Mode/workload | Concurrency | Output tok/s, ordinary → speculative | ITL p95, ordinary → speculative |
| --- | ---: | ---: | ---: |
| Greedy repeat | 1 | 48.1 → 144.6 | 20.0 → 24.2 ms |
| Greedy repeat | 4 | 130.8 → 258.1 | 25.6 → 47.4 ms |
| Greedy repeat | 8 | 195.5 → 277.4 | 32.6 → 52.9 ms |
| Sampled repeat | 1 | 46.8 → 142.4 | 21.1 → 24.3 ms |
| Sampled repeat | 4 | 123.4 → 136.7 | 26.6 → 33.0 ms |
| Sampled repeat | 8 | 189.3 → 199.3 | 32.3 → 43.8 ms |
| Greedy novel | 8 | 225.2 → 220.8 | 30.7 → 35.3 ms |
| Sampled novel | 8 | 222.8 → 218.3 | 30.8 → 33.4 ms |

All **156 same-seed request comparisons** produced identical token IDs. Greedy repeat
accepted all 2,178 proposed tokens across the three concurrency levels. Sampled repeat
accepted 891/931 proposals, but cost adaptation often chose ordinary passes at higher
concurrency. Sampled novel accepted none of 48 proposals at concurrency eight; further
drafting was suppressed except for bounded exploration. Low-reuse prose remained within
about 3.3% of ordinary throughput across the measured cases.

**The p95 latency gate is not met** for throughput mode:
its bursts can increase p95 inter-token latency substantially. This is why enabling it
requires an explicit option and why it is not the default.

These measurements predate the later planner, proposal, headroom and mixed-pass changes
and have not been rerun. The serving bench now also reports burst gaps (between the first
publications of a request's successive passes) and tokens per burst: ITL p50 is zero
inside speculative bursts and says little about them.

At concurrency eight, the reserved resident plan grows from 3,998,108,236 to
4,386,671,436 bytes (+371 MiB). Separate processes measured with `/usr/bin/time -l`
and a 40-token output limit observed maximum RSS of 2.83 → 3.76 GiB and peak macOS
memory footprint of 5.24 → 5.77 GiB. These process peaks include loading, driver state
and allocator caching; they are not a measurement of speculative tensors alone. Both
runs fit this machine, but the memory experiment does not establish a bound for other
contexts or sustained churn. See `memory-draft{0,4}.{json,log}` in the artifact directory.

## Qwen3.6-35B-A3B text path

The pinned checkpoint declares `qwen3_5_moe`, not a new dense architecture. The existing
Qwen3.5 attention and DeltaNet implementations are reused with a distinct expert MLP:

- F32 router projection, stable normalized top-k selection, deterministic low-ID ties.
- 256 routed experts, eight selected per token, expert intermediate width 512.
- One shared expert, width 512, multiplied by a sigmoid gate.
- Packed expert banks: `[experts, 2*intermediate, hidden]` gate/up and
  `[experts, hidden, intermediate]` down projections. F32, Q8_0 and Q4K storage is
  supported; router and shared gate stay F32.
- Routing, per-expert counts, prefix offsets, permutation construction, grouped GEMM and
  weighted scatter stay on the device. Expert groups have their actual row counts; no
  uniform routing or per-expert host submission is assumed.
- Top-k is k simdgroup max reductions over register-held scores (previously every score
  was ranked against every other, O(experts²) device reads per token). The stable
  permutation comes from per-block histograms of 256 assignments, one scan and a
  parallel scatter; previously single threads scanned every assignment per expert.
- Every format (F32, `Q8_0`, Q4K) uses device-built sixteen-row tile descriptors with two
  kernels. A GEMV handles tiles of at most three rows, with the dense GEMV's outputs per
  simdgroup (two for Q4K's vectorized super-block layout). A matrix-unit kernel handles
  larger tiles: each simdgroup owns 32 outputs over the whole reduction, with no
  threadgroup memory. The gate/up projection reads each token through the permutation,
  so no gathered `[tokens·k, hidden]` copy is made.
- Scatter ranks each token's experts once per threadgroup and accumulates in expert-ID
  order without floating-point atomics; the sigmoid-gated shared expert is added in the
  same pass. Invalid router scores propagate an invalid result without becoming an
  unchecked expert index.

Config identifiers, head layout, quantization widths, routing limits, tensor names,
shapes and dtypes are validated. Unsupported routing normalization and sparse-layer
layouts are rejected. The factory, pipeline and adapters share this implementation;
there is no separate dense-model alias pretending to implement MoE.

The checkpoint loader validates every shard header with SafeTensors metadata validation
before reading any payload. It then seeks directly to each needed tensor. Packing a
layer releases its host payload before the next layer is read, avoiding retention of
the complete BF16 checkpoint. Expert banks stream one expert at a time: each expert's
payload is read, converted to F32 and quantized before the next, and the packed bytes
are uploaded once. For a Qwen3.6 gate/up bank, full-precision transients fall from about
6 GiB (BF16 read, its copy and a full F32 conversion) to about 16 MiB per expert; the
packed bank itself (about 0.3 GiB in Q4K) is held on the host until its upload. Other tensors
still transiently hold their payload twice; the largest is the embedding (about 1 GiB of
BF16). Index filenames must remain in the checkpoint directory. Loading memory is not yet
measured on the full 35B checkpoint.

GGML quantization reads a tensor's storage buffer rather than its layout, so quantizing an
F32 row-range view packed the wrong rows. Weights loaded from BF16 checkpoints were
unaffected (conversion copies), but packing now always quantizes an owned host copy.

The CPU reference path routes on the host, then projects each selected expert's tokens
at once through only that expert's packed rows. It previously multiplied the whole bank
per token and kept one expert's outputs, about 32× the necessary work for 256 experts.

### Pins and independent conformance

[Reference pins and hashes](references/p8/pins.json):

| Reference | Immutable revision |
| --- | --- |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| [Transformers implementation](https://github.com/huggingface/transformers/tree/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/qwen3_5_moe) | `ea52711b4850c332be3f778396ce882ab84a9011` |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B/tree/24548b62aa021d562695c04aaf7758a1ea47990b) | `24548b62aa021d562695c04aaf7758a1ea47990b` |

The pinned Qwen config and **all 693 text tensor names/shapes** are checked against
actual shard headers. `scripts/p8_reference_headers.py` fetches headers only and records
their SHA-256 hashes; it downloads no model payloads. Nineteen MTP tensors are present,
but the proposal algorithm and its tensor execution are unimplemented, so MTP remains
disabled despite `mtp_num_hidden_layers: 1`.

An offline deterministic two-layer hybrid fixture has eight routed experts and one
shared expert. Its BF16 checkpoint values come from a fixed LCG. Independent
Transformers 5.20.2 / torch 2.14.1 CPU F32 execution generates saved intermediate layer
outputs and logits. CPU and fused Metal F32 agree within the 1e-4 relative gate.
`scripts/qwen3_5_moe_reference.py` regenerates the small golden without downloading
weights. The fixture does not stand in for a full-checkpoint quality evaluation.

F32/Q8_0/Q4K grouped GEMM matches scalar F64 accumulation over dequantized weights within
2e-5 relative error, including strided inputs, partial output tiles, different routing
skews, 4,800-assignment multi-block routing, routed (permutation-read) inputs, the fused
shared expert and nonfinite routing rows. Hybrid tests accept a partial speculative prefix
and continue after rollback in all three formats. A separate test measures verification
against one-at-a-time decoding inside and beyond the Q4K GEMV range (see above).

### Expert projection performance gate: open

[Kernel artifacts](baselines/2026-10-23-m1-p8/moe-kernels-grouped.json) use the flagship's
gate/up shape: 256 experts, top-8, input 2048, output 1024. Random scores disperse
assignments; the concentrated case sends every token to the same eight experts. Both are
synthetic. Grouped timings include device routing, with inputs read through the
permutation. The individual baseline already has its selected input rows prepared, so
routing, readback and gather are excluded for it. Each cell is the median of fifteen
completed-work samples with three warmups. Earlier layouts are kept as
`moe-kernels.*` (first grouped version), `moe-kernels-initial.*` and
`moe-kernels-table-search.*`.

| Tokens | Routing | `Q8_0` grouped (first version) | `Q8_0` individual | Q4K grouped (first version) | Q4K individual |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Dispersed | 0.375 (0.584) ms | 0.373 ms | 0.319 (0.458) ms | 0.211 ms |
| 8 | Dispersed | 2.65 (2.83) ms | 2.93 ms | 2.28 ms | 1.69 ms |
| 32 | Dispersed | 7.20 (7.24) ms | 7.67 ms | 7.35 ms | 5.99 ms |
| 128 | Dispersed | 14.96 (17.24) ms | 14.52 ms | 19.14 ms | 18.61 ms |
| 512 | Dispersed | 24.6 (27.3) ms | 22.3 ms | 32.1 ms | 23.4 ms |
| 2048 | Dispersed | 80.5 (93.2) ms | 51.3 ms | 105.9 ms | 57.7 ms |
| 8 | Concentrated | 0.674 (0.988) ms | 0.472 ms | 0.943 ms | 0.788 ms |
| 128 | Concentrated | 4.60 (6.23) ms | 2.56 ms | 6.36 ms | 2.87 ms |
| 2048 | Concentrated | 73.4 (85.2) ms | 55.2 ms | 97.2 ms | 57.2 ms |

Routing (top-k, histograms, permutation and a scatter) now costs 0.02 ms for one token and
0.24 ms for 2,048 tokens; the parenthesized first-version timings include its slower
routing and gather.

**No expert-kernel speedup is claimed.** `Q8_0` decode is at parity with individual
projections and 8–32 dispersed tokens are slightly faster. Q4K decode is about 1.5×
slower than Candle's dense Q4K GEMV. Concentrated groups and prefill-sized passes remain
1.4–1.8× slower: above sixteen rows Candle stages individual groups in F16, while the
grouped kernels accumulate in F32, so those cells also compare different arithmetic.
Two rejected attempts are recorded here: 32×32 register tiles (149 ms at 2,048 dispersed
tokens, from register pressure) and threadgroup-staged 32-wide reduction slices
(239 ms, barrier-bound with lower occupancy). An F16-staged or larger-K tiled grouped GEMM
is the open next step. F32 is measured at 64 experts
([artifact](baselines/2026-10-23-m1-p8/moe-kernels-grouped-f32.json)), since a full F32
bank does not fit beside its host copy here: 1.20 vs 1.29 ms at one token, 9.5 vs 11.8 ms
at 32 dispersed tokens, and 114 vs 61 ms at 2,048. An earlier F32 run was killed because
`PackedLinear::dense` uploaded each per-expert row view with Candle's `to_device`, which
transfers a view's whole underlying storage (the full bank, 64 times). Dense uploads now
copy only the view. The MoE model path remains experimental, with its performance exit gate
open. Trained routing traces and full-model measurements remain necessary before claiming
a useful MoE serving regime.

### Full-checkpoint boundary

Metadata-derived resident **text weight** sizes, including all routed experts:

| Format | Bytes | GiB |
| --- | ---: | ---: |
| F32 | 138,642,442,752 | 129.12 |
| Q8_0 | 37,368,924,672 | 34.80 |
| Q4K | 20,304,005,632 | 18.91 |

These exclude state, snapshots, scratch, runtime overhead, vision and MTP. Even Q4K
weights alone exceed this Mac's 16 GB memory. No full 35B generation, real routing-skew
benchmark, held-out quality score or production-support claim is made. Admission checks
the full resident weight budget rather than the 3B activated parameter count.

## Gemma 4 assessment

The pinned 26B-A4B adapter needs separate model semantics: five sliding-attention layers
per full-attention layer, different local/global head dimensions and KV-head counts,
K-equals-V handling, separate RoPE policies, GELU MLPs, extra feed-forward norms, final
logit softcapping, and a router with input normalization and learned input/per-expert
scales. The shared Qwen expert block's sigmoid gate is not equivalent. Other Gemma 4
variants also expose KV sharing and per-layer inputs that need their own state/config
contracts. See the [pinned implementation](references/p8/modeling_gemma4.py).

Gemma 4 remains explicitly rejected by Qwen configuration parsing. Its future adapter
can reuse the storage and grouped-projection primitives after independent norm, rotary,
attention, routing, state and whole-model conformance tests. No Gemma weights were
downloaded or executed in this package.

## Validation and remaining gates

- Offline workspace nextest: **229 passed, seven checkpoint-dependent tests skipped**
  (after the review fixes; the earlier run in the retained log had 221).
- Strict workspace Clippy, all targets: passed. Workspace checks pass with and without
  default features. Existing `block 0.1.6` future-compatibility notice remains.
- Target-matching equivalence: 256 seeds compare samples, RNG draws, penalties and every
  attention/convolution/recurrent value after rejection and continuation.
- Pipeline tests cover output limits, prefix snapshots, grammar/EOS exclusion, batch
  draft caps, explicit budget rejection, cancellation, resource reclamation, preemption
  of drafting requests (index rebuilt, outputs equal ordinary decoding), a deadline
  expiring under a drafted pass, and decodes drafting beside a prefill.
- Metal sampling with penalties preserves seeded output against ordinary decoding.
- Native MTP execution, full Qwen3.6 conformance/quality, trained routing traces, and
  Gemma 4 execution remain open. Coverage percentage and long-duration thermal/leak
  gates were not measured here.

Reproduction (hardware tests need direct Metal access):

```sh
cargo nextest run --offline --workspace --test-threads 1
cargo clippy --offline --workspace --all-targets -- -D warnings
cargo check --offline --workspace --all-targets --no-default-features
TUR_BENCH_ONLY=moe TUR_BENCH_MOE=q8_0,q4k cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<pinned-snapshot> TUR_BENCH_WORKLOADS=speculation TUR_BENCH_CONCURRENCY=1,4,8 TUR_BENCH_OUTPUT=80 TUR_BENCH_REPEATS=3 cargo bench --offline -p tur-engine --bench serving
# Same command with TUR_BENCH_SAMPLING=0.7,20,0.95 for sampled output.
```
