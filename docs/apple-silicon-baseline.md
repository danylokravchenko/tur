# P0: Apple Silicon baseline and safe dispatch

P0's evidence and capability gate is complete on the available development Mac.
P1's compatible workspace move is also implemented. This establishes a measured
starting point for the rewrite, not production readiness or hybrid-model support.
The initial [development run](baselines/2026-10-21-m1-dev/manifest.json) is retained
as historical evidence.

## Results

The machine is an Apple M1 with 8 GPU cores and 16 GB unified memory, running
macOS 26.6.2, Xcode 26.4.1 and Rust 1.99.0. The final runs use the optimized
`pre-release` profile; power and compiler details are recorded in each manifest.
Model loading is measured separately from warmed inference. Filesystem caches
were uncontrolled, so the load measurement is not an OS-cold disk measurement.

The checkpoint is Qwen3-0.6B with unchanged upstream GGUF Q4_K_M weights and F32
engine activations. Three measured iterations follow one warmup per prompt length;
each generates exactly 16 tokens. The benchmark deliberately continues through EOS
for fixed work, and records every input/output token ID. Prefix runs reproduce the
cold-cache generated IDs exactly in every measured iteration.

| Prompt tokens | Median prefill + first token, cold prefix | Median with warm prefix | Median decode interval, cold prefix |
| --- | --- | --- | --- |
| 128 | 159.6 ms | 36.1 ms | 26.8 ms |
| 2,048 | 6,400.8 ms | 288.6 ms | 88.8 ms |

The first process model load took 1.003 s. Reported Metal allocation after loading
was 1,198,948,352 bytes; process peak RSS was 1,650,638,848 bytes. The GGUF tensor
payload is 390,753,280 bytes: 168 Q4K tensors, 29 Q6K tensors and 113 F32 tensors.
Packed storage therefore does not imply that total residency equals file payload.
Candle's quantized dispatch and unquantized tensors/scratch remain relevant costs.

The real HTTP server completed 21 measured requests per configuration, across
concurrency 1/2/4 and three rounds. Requests mix short and longer prompts; server
sampling is greedy with repetition penalty 1. The configured server activation
dtype is its existing BF16 default; the engine microbenchmark explicitly uses F32.

| Concurrency | Median successful completion tokens/s, prefix off | Prefix on |
| --- | --- | --- |
| 1 | 32.27 | 32.17 |
| 2 | 27.87 | 29.42 |
| 4 | 33.87 | 32.96 |

All 42 measured HTTP requests completed successfully. These small samples show no
near-linear concurrency scaling. HTTP TTFT includes queueing; recorded streaming
intervals are **text-event intervals**, because an SSE event can contain multiple
tokens. They must not be described as per-token latency. Raw events, usage counts,
latency percentiles, outputs and process resource statistics are retained. The
harness sends SIGINT to its own server after measurement; its recorded `-2` exit
and time(1)'s abnormal-termination text reflect this intentional shutdown.

Artifacts:

- [Primitive capability run](baselines/2026-10-21-m1-p0-primitives/manifest.json)
- [Engine run](baselines/2026-10-21-m1-p0-engine/manifest.json) and [raw measurements](baselines/2026-10-21-m1-p0-engine/run.stdout)
- [Warm-prefix run](baselines/2026-10-21-m1-p0-prefix/manifest.json)
- [HTTP prefix off](baselines/2026-10-21-m1-p0-serving/results.json) and [prefix on](baselines/2026-10-21-m1-p0-serving-prefix/results.json)
- [Pinned files, revisions and SHA-256 hashes](baselines/2026-10-21-m1-p0-validation/checkpoints.json)

## Safe dispatch and capabilities

The custom probe now uses Candle-native buffers, fallible `objc2-metal` compilation,
and Candle's safe typed readback. It verifies lengths 1/31/257/1025, rounded
threadgroups, a four-byte binding offset, and guard elements before and after the
output. Four buffers and the pipeline remain owned across GPU completion. Invalid
MSL and missing functions are rejected as errors. No unsafe Rust was added.

The user approved development-only direct dependencies at existing locked versions:
`candle-metal-kernels` 0.10.2, `objc2-metal` 0.3.2 and `objc2-foundation` 0.3.2.
No production dependency version or public model/state contract changed. Direct
fallible bindings avoid the unwraps in Candle's shader-compilation convenience APIs.
This proves a feasible dispatch/readback path, not a finished production runtime.

| Capability | Observed result |
| --- | --- |
| Candle FP32 affine/matmul | Synchronized outputs match the CPU fixtures exactly |
| Custom MSL compilation/dispatch/readback | Validated, including guard and error checks |
| GPU family queries | Apple7, Metal3 and Metal4 all return true on this Mac/OS |
| Metal 4 host facilities | Transitive objc2 bindings declare allocators, queues, compiler and tensor APIs; direct `metal` 0.33 and Candle wrappers do not expose a complete checked runtime. No Metal 4 execution path was tested |
| Standalone shader compiler | `xcrun metal --version` fails because the optional Metal Toolchain component is absent; runtime shader compilation works |
| Sandbox | Device enumeration is empty inside this sandbox; actual device runs require host GPU access |

## Allocation, command and copy evidence

Instruments Metal System Trace captured the primitive and short model probes.
[Exported summaries](baselines/2026-10-21-m1-p0-traces/p0-metal-model-summary.json)
retain table counts and hashes. Raw traces and XML remain locally in
`target/p0-metal-{primitives,model}.trace` and corresponding `-toc.xml`/`-events.xml`
files. They contain machine/process identifiers and are not published as repository
artifacts. Traced runs are used for structural evidence, not clean timing results.

The model trace reports 947 allocation records, 934 deallocation records, 379
command-buffer IDs in application encoding intervals, and 380 in GPU intervals.
These are whole-probe counts, including load, inference, page inspection and sampling;
they are not counts per generated token. Encoder label components include 2,072
`copy2d` occurrences, 448 `copy_contiguous` occurrences and 165 `blit_to_cpu`
occurrences. Label components are not an exact kernel-dispatch count.

The actual two-request paged probe retains 58,720,256 bytes in physical pages and
another 58,720,256 bytes in full-cache tensors at 128 tokens. After one decode,
those become 66,060,288 and 59,179,008 bytes respectively. All allocator blocks are
released when requests are dropped. This measures the duplicate-cache cost and
partial-page padding; the rewrite must eliminate whole-cache concatenation.

Sampling is shape/strategy dependent. Existing Candle greedy sampling reduces on
GPU and reads back one u32 (4 bytes); temperature/top-p reads back the vocabulary
probabilities (607,744 bytes for this checkpoint). The synchronized standalone
sampling medians were 0.320 ms and 0.952 ms respectively. Active grammar masking
also transfers logits to the host and recreates a tensor. P0 leaves that path intact.
Candle's existing pooled allocations, normalization, elementwise operations and
packed GGUF execution remain useful references. Page-native attention, reduced
submissions/copies and prefill-specific quantized GEMM are rewrite priorities.

## Test evidence and scope

Before relocation, 147 library and 14 server tests passed. The same offline
selection passed after relocation. The final pinned checkpoint run completed
**68 of 69 tests successfully**, with none skipped. Three Rust probe tests and
three Python tooling tests passed. Final workspace checks pass with default Metal
features and with `--no-default-features`; formatting and diff checks pass.

The remaining inherited failure is
`test_guided_json_schema_output_is_valid_json`: output is a valid JSON object
followed by `#`. It remains failing and is recorded in the
[pinned nextest log](baselines/2026-10-21-m1-p0-validation/pinned-nextest.log).
Direct cargo test also exposed repeated global tracing initialization; nextest
isolates those test processes. Initial failure logs are retained alongside the
final run. P0 does not claim these correctness defects are fixed.

Two Granite assertions incorrectly expected SafeTensors from a GGUF fixture; they
now check the actual GGUF representation and tensor names. Template tests use the
same quantized fixture rather than unnecessarily loading full precision weights.
The downloader now uses `ApiBuilder::from_env()` so the isolated `HF_HOME` and
local disabled endpoint are actually honored. Its stale GPU-dequantization log
was corrected. No mathematical inference implementation was changed.

Initial revision was `e58c1c2df467bdd913879ca76449fbbc4d53ad50`; the user's Rust 1.99
change and unrelated files were preserved. Existing unsafe/panic violations,
deprecated constants and the upstream `block` compatibility warning remain.
At the baseline checkpoint, P2 extraction and hybrid public contracts still awaited approval. The user subsequently authorized breaking Rust changes; current contracts and validation are recorded in [the P2 report](apple-silicon-p2-contracts.md).

This baseline does not cover 8K/32K contexts, concurrency 8/16, sustained thermal
behavior, quality budgets, other Macs, cancellation/OOM correctness or production
percentile confidence. Those belong to subsequent implementation/release gates.
A family-support query alone does not prove identical numerical or throughput
capabilities across chips.

## Reproduce selectively

```sh
python3 scripts/p0_checkpoints.py --output target/p0-hf
python3 scripts/apple_silicon_baseline.py --output target/p0-primitives
python3 scripts/apple_silicon_baseline.py --example model_baseline \
  --hf-home target/p0-hf --output target/p0-engine -- \
  --iterations 3 --decode-tokens 16 --prompt-lengths 128,2048
cargo build --locked --offline --profile pre-release -p tur-server --bin server
python3 scripts/p0_serving_baseline.py --binary target/pre-release/server \
  --hf-home target/p0-hf --output target/p0-serving
```

Add `--prefix-cache` to the example arguments or serving script for warm-prefix
runs. Output directories must be new. The checkpoint helper requires the exact
cached revisions listed in its source and does not download or requantize weights.
The recorder saves source/lockfile hashes, hardware/OS/compiler/power information,
raw samples and failed executions. Every inference timer includes completed GPU
work; primitive readback/reference checks are outside the timed operation.

Use targeted checks during the engine rewrite; the full suites above establish
this baseline and need not run after every edit. Profiling can be reproduced with
`xcrun xctrace record --template 'Metal System Trace' --output target/local.trace
--launch -- <probe executable>`, setting the pinned HF_HOME/HF_ENDPOINT for model
runs, then exporting the selected tables with `xcrun xctrace export` and running
`scripts/p0_trace_summary.py`. The model trace used one measured iteration,
128-token prompts and four output tokens, plus the page/sampling probes.
