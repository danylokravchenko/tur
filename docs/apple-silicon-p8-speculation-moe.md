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
earlier continuation. Defaults when enabled: n-grams 1–4, up to four drafts per request
and sixteen across a pass. Limits remain configurable through `SpeculationConfig`.

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
second improve. Unidentified fits explore drafted and ordinary passes; unsuccessful
requests recover toward the shared prior so they can be retried. Mixed prefill/decode
passes are never drafted.

Memory planning charges **all simultaneously possible drafts**, capped by running
sequences and the pass limit. It reserves their recurrent checkpoints and logit rows
before loading weights. Scheduling additionally checks actual page and slot demand,
leaving room for ordinary decodes. Proposals yield to state pressure.

The host n-gram index retains at most two 8K-token windows, pruning periodically. Its
hash-table capacity is charged to the shared memory budget; preemption drops the index
and rebuilds it on resume. Context size cannot create an unbounded proposal index.
The request's complete token history remains available for normal recomputation.

**Default: disabled.** Both adapters expose `--speculative-draft N` (zero by default).
Narrow recurrent state automatically disables drafting: its ordinary per-pass rounding
is not reproduced by the F32 intermediates of a multi-token verification. The validated
speculation path requires F32 recurrent state.
This is an explicit throughput mode: accepted tokens arrive in bursts, with longer
pauses between passes. `PipelineConfig::reproducible` also disables speculation.
The target-matching sampling guarantee does not promise bit-identical logits between
different batch/chunk shapes: existing F16-tile and recurrent rounding differences still
apply. The reproducibility mode fixes those execution choices separately.

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
- Routing, per-expert counts, prefix offsets, permutation construction, token gather,
  grouped matrix-unit GEMM and weighted scatter stay on the device. Expert groups have
  their actual row counts; no uniform routing or per-expert host submission is assumed.
- Q8 groups use direct device-built tile descriptors, a GEMV for at most three rows and
  vector-loaded matrix tiles for larger groups. Tile lookup never searches all experts.
- Scatter accumulates in expert-ID order without floating-point atomics. Invalid router
  scores propagate an invalid result without becoming an unchecked expert index.

Config identifiers, head layout, quantization widths, routing limits, tensor names,
shapes and dtypes are validated. Unsupported routing normalization and sparse-layer
layouts are rejected. The factory, pipeline and adapters share this implementation;
there is no separate dense-model alias pretending to implement MoE.

The checkpoint loader validates every shard header with SafeTensors metadata validation
before reading any payload. It then seeks directly to each needed tensor. Packing a
layer releases its host payload before the next layer is read, avoiding retention of
the complete BF16 checkpoint. Index filenames must remain in the checkpoint directory.
Transient loading memory still scales with the largest tensor being converted and is
not yet measured on the full 35B checkpoint.

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
skews and nonfinite routing rows. Hybrid tests accept a partial speculative prefix and
continue after rollback in all three formats. They preserve the initial chunk boundaries
and keep the Q4K verification window within its F32 GEMV regime: switching between
Candle's F16 tiles and single-token GEMV can change close routing decisions. Earlier
comparisons across those different execution paths differed by about 0.5%; that is not
evidence of a rollback defect or a new numerical tolerance for the kernel.

### Expert projection performance gate: open

[Kernel artifacts](baselines/2026-10-23-m1-p8/moe-kernels.json) use the flagship's actual
gate/up shape: 256 experts, top-8, input 2048, output 1024, Q8_0. Random scores disperse
assignments; the concentrated case sends every token to the same eight experts. Both
are synthetic. The grouped timing includes device routing and gather. The individual
baseline already has its selected input rows prepared, so routing/readback/gather costs
are excluded for that baseline. Each cell is the median of fifteen completed-work
samples with three warmups.

| Tokens | Routing | Grouped, including routing/gather | Individual, prepared inputs |
| ---: | --- | ---: | ---: |
| 1 | Dispersed | 0.584 ms | 0.360 ms |
| 8 | Dispersed | 2.828 ms | 2.784 ms |
| 32 | Dispersed | 7.237 ms | 7.245 ms |
| 128 | Dispersed | 17.235 ms | 12.617 ms |
| 8 | Concentrated | 0.988 ms | 0.438 ms |
| 32 | Concentrated | 1.826 ms | 0.776 ms |
| 128 | Concentrated | 6.234 ms | 2.488 ms |

**No expert-kernel speedup is claimed.** Direct tile indexing removed the repeated
expert-table search and improved the initial prototype substantially, but parity at
8–32 dispersed tokens is not a win beyond noise. Large and concentrated groups still
need work; small groups also pay the metadata dispatch cost. Candle uses F16 staging
for individual groups above sixteen rows, while the custom kernel accumulates in F32,
so those concentrated large-group timings also compare different arithmetic paths.
F32/Q4K grouped math passes correctness tests but has no performance selection evidence.
The new MoE model path remains experimental, with its performance exit gate open.
Rejected layouts are retained as `moe-kernels-initial.*` and
`moe-kernels-table-search.*`. Trained routing traces and full-model measurements remain
necessary before claiming a useful MoE serving regime.

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

- Offline workspace nextest: **221 passed, seven checkpoint-dependent tests skipped**.
  Final targeted checks cover subsequent kernel tuning, streaming-header validation and
  automatic fallback with narrow recurrent state; see the retained test logs.
- Strict workspace Clippy, all targets: passed. Workspace checks pass with and without
  default features. Existing `block 0.1.6` future-compatibility notice remains.
- Target-matching equivalence: 256 seeds compare samples, RNG draws, penalties and every
  attention/convolution/recurrent value after rejection and continuation.
- Pipeline tests cover output limits, prefix snapshots, grammar/EOS exclusion, batch
  draft caps, explicit budget rejection, cancellation and resource reclamation.
- Metal sampling with penalties preserves seeded output against ordinary decoding.
- Native MTP execution, full Qwen3.6 conformance/quality, trained routing traces, and
  Gemma 4 execution remain open. Coverage percentage and long-duration thermal/leak
  gates were not measured here.

Reproduction (hardware tests need direct Metal access):

```sh
cargo nextest run --offline --workspace --test-threads 1
cargo clippy --offline --workspace --all-targets -- -D warnings
cargo check --offline --workspace --all-targets --no-default-features
TUR_BENCH_ONLY=moe cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<pinned-snapshot> TUR_BENCH_WORKLOADS=speculation TUR_BENCH_CONCURRENCY=1,4,8 TUR_BENCH_OUTPUT=80 TUR_BENCH_REPEATS=3 cargo bench --offline -p tur-engine --bench serving
# Same command with TUR_BENCH_SAMPLING=0.7,20,0.95 for sampled output.
```
