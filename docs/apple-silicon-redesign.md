# Tur: Apple Silicon inference redesign

Status: P0's development-Mac evidence/capability gate is complete (2026-10-21), and P1's compatible three-member workspace move is implemented. Safe custom MSL dispatch/readback, pinned model/HTTP baselines and Instruments trace summaries are recorded in [the baseline report](apple-silicon-baseline.md). The pinned checkpoint suite has one retained guided-JSON failure (68/69 pass). Production performance, quality and hybrid-model release gates remain open. Public model/state contracts are unchanged.

## Product and platform

Build a Rust inference service for Apple Silicon macOS, optimizing prompt throughput and time to first token alongside sustained token generation and tail latency under concurrency. A single Mac is the initial deployment unit. CPU execution remains useful as a numerical test oracle, not a supported production backend. Intel Macs, CUDA, and Linux are outside the proposed production scope.

Use capability-based Metal dispatch; choose a minimum macOS version after measuring the target machines. Metal 4 exposes newer command and tensor facilities, but their availability does not establish that every chip has identical numerical or throughput capabilities. Benchmark specialized kernels against existing Metal implementations before making them defaults. [Apple Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)

## Repository audit

| Existing implementation | Consequence and replacement |
| --- | --- |
| `models/qwen3.rs` implements ordinary Qwen3 attention; `Qwen35ModelForCausalLM` is only an alias | Renaming it cannot implement Qwen3.8. Keep the old implementation as a reference during migration. |
| `ModelImpl` exports only K/V pairs and accepts per-layer `PagedKvCache` | Introduce explicit hybrid request state before supporting DeltaNet batching and prefix reuse. |
| `PagedKvCache::append` writes blocks and also concatenates full-sequence K/V | Implement attention directly over physical pages; eliminate the duplicate full cache and per-token concatenation. |
| `InferenceEngine::prefill_batch` loops over requests on its paged path | Pack variable-length prompt chunks into one execution plan with per-sequence offsets. |
| Default Qwen GPU attention falls through to standard attention | Add tiled Metal prefill and paged decode kernels, without materializing a complete attention matrix. |
| GGUF linear layers already construct `QMatMul` | Inspect backend dispatch and measure actual resident storage; stale loader logging incorrectly claimed GPU weights were dequantized. |
| `benches/tur.rs` constructs a CPU factory for Qwen3-0.6B | Add reproducible Metal kernel and serving benchmarks before publishing performance claims. |
| Existing loading and CPU kernels contain unsafe code; tests contain unwraps | Remove inherited violations as those paths are replaced; do not introduce new unsafe Rust or panic paths. |

The existing scheduler, API, templates, tokenizer, grammar support, and tests are useful migration assets. Their presence does not prove efficient GPU batching or cache residency.

## Workspace boundaries

First migration step:

```text
crates/
  tur/          # Existing library, integration tests, benchmarks
  tur-cli/      # tur binary
  tur-server/   # server binary
docs/
assets/
Cargo.toml      # Workspace, shared dependency versions and profiles
Cargo.lock
```

During the first move, the server library module stays in `tur` for API compatibility. Build all members with `cargo check --workspace --all-targets`; run `cargo run --bin tur -- ...` or `cargo run --bin server -- ...` from the repository root.

Proposed subsequent extraction, after agreeing the public contracts:

| Crate | Responsibility and dependency direction |
| --- | --- |
| `tur-core` | Extract the existing shared errors and minimal execution/state descriptors; no runtime, HTTP, or model dependencies |
| `tur-metal` | Device capabilities, safe resource ownership, packed buffers, kernel compilation/dispatch; no models or scheduler |
| `tur-models` | Checkpoint validation, model implementations, layer/state descriptions; depends on Metal runtime |
| `tur-engine` | Request lifecycle, cache ownership, scheduling, execution and sampling; depends on models/runtime |
| `tur` | Stable public facade and compatibility exports |
| `tur-cli`, `tur-server` | Product adapters; depend on public facade |

Extract existing functionality rather than copying it into parallel foundational modules. Keep model-specific configuration with its model; avoid an unrelated universal Config replacement. Move existing shared errors into `tur-core` when the extraction is approved, retaining facade re-exports; do not create a second competing error hierarchy. No new third-party dependencies are selected by this proposal.

## Qwen3.8 migration

The first flagship is **Qwen/Qwen3.8-27B**, initially text-only. Its checkpoint declares `qwen3_5` and a nested `text_config`, with a hybrid sequence of recurrent and full-attention layers. Implement the actual configuration, weights and template semantics; do not map every name containing `qwen3` to the existing decoder. [Official configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)

Required work: validated nested config parsing and tensor-name/shape mapping; gated DeltaNet convolution and recurrent state; chunked parallel prefill and recurrent decode; gated full attention; partial rotary embeddings; correct normalization, vocabulary, EOS and output projection; official template handling. Test whole-prompt, chunked and incremental execution against a pinned reference. Treat image/video inputs as unsupported until their encoder and position handling are implemented.

MLX supplies a useful independent implementation reference for the architecture and its separate recurrent/attention caches. Pin a revision and verify Qwen3.8-specific fields rather than assuming any Qwen3.5 implementation is automatically equivalent. [MLX implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)

### Required public contract changes

Replace the assumption that every layer owns K/V with model-declared state kinds: attention page references or recurrent convolution/matrix state. A request owns its state; weights are shared. Execution consumes packed token IDs, sequence boundaries, positions and state handles. State operations must support allocation, release, fork, checkpoint, restore, and speculative commit/rollback.

Prefix entries must associate an exact token boundary with both attention pages and a recurrent snapshot. A recurrent state cannot be truncated as if it were a KV array. Share immutable checkpoints; copy writable recurrent state when branching. Namespace entries by model revision, quantization, adapters, positional settings and tenant policy. Verify tokens on hash hits. Release GPU resources only after all referencing commands complete.

Keep wire-level chat/streaming behavior stable where possible. The affected Rust contracts include `ModelImpl`, `KvCacheImpl`, paged-state arguments, prefix snapshots and engine/model construction. AGENTS.md requires confirmation before these public API changes. Approval of this plan should explicitly include these named contract changes and extraction of shared types; otherwise Sol can complete baseline work and the compatible workspace move first.

## Quantization and kernels

1. Keep weight storage packed. Start with a small validated format set: groupwise 4-bit and 8-bit weights, with explicit packing, group size, scale/zero representation and alignment. Preserve supported GGUF imports and validate architecture metadata. Conversion must be explicit and reproducible; never silently expand the whole model.
2. Separate quantized GEMV/small-batch decode from tiled GEMM prefill. Dequantize tiles in registers/threadgroup memory, accumulate reductions at appropriate precision, and benchmark shape-specific dispatch. Fuse RMSNorm, residuals, activation/gates and rotary transformations where profiling justifies it.
3. Build ragged tiled prefill and page-table decode with direct GQA head mapping. Fuse KV quantization on write and dequantization on read. Measure split-K/reduction overhead for long contexts; avoid expanding repeated KV heads.
4. Start KV validation at 8 bits; evaluate 4-bit K/V independently. Keep a recent higher-precision region if quality tests justify it. KIVI motivates asymmetric key/value quantization, but its published results are not proof of performance or accuracy on this model and Metal. [KIVI](https://arxiv.org/abs/2402.02750)
5. Keep recurrent accumulation in FP32 initially. KV quantization does not justify quantizing recurrent state: its error persists through future updates. Evaluate that separately.
6. Use pooled device buffers, bounded scratch space, cached pipeline states and few command submissions. Overlap host planning with GPU work using explicit completion ownership. Unified memory removes neither synchronization nor bandwidth costs.

FlashInfer's separation of planning and execution and its paged/ragged attention variants are useful design references, not portable CUDA kernels. [FlashInfer](https://arxiv.org/abs/2501.01005)

## Scheduling, caching and serving

Use continuous admission with token, byte and execution-time budgets. Reserve decode work, then fill available time with adaptive prefill chunks; cap queue age to prevent prefill starvation. Charge weights, KV pages, recurrent states, snapshots, scratch and speculative drafts against one memory budget with OS headroom. Never admit based only on a sequence count.

Use reference-counted prefix pages and copy-on-write tails, bounded eviction, request cancellation, deadlines, backpressure and explicit OOM recovery. Consider recomputation versus checkpoint retention using measured costs. Prefix-aware scheduling must include fairness. Asynchronous planning needs versioned request state so cancellation cannot recycle in-flight buffers.

Move common sampling and penalties onto the GPU. Keep per-request RNG independent of batch order. Grammar masks and constrained sampling must remain correct; measure their transfer cost separately. Test batch-size and ordering effects, and provide a reproducibility mode rather than promising bitwise invariance without evidence.

Speculation comes after hybrid rollback correctness. Compare n-gram proposals, a compatible draft model, and checkpoint-supported multi-token prediction. Adapt draft length to acceptance and verification cost, disable it when batching makes it slower, and preserve the target sampling distribution.

## DeepInfer and additional research

DeepInfer targets NVIDIA SM100 and Ascend 950, and its release supports DeepSeek-V4.1-Flash. It is not a Metal backend. Its public release excludes the production KV cache system. [DeepInfer](https://github.com/deepseek-ai/DeepInfer)

| Technique | Tur decision |
| --- | --- |
| Asynchronous scheduling | Adopt the principle; measure overlap on a single Apple GPU. |
| DSpark/confidence-scheduled verification | Research adaptive verification budgets after exact speculative rollback exists. |
| SWA bounded replay | Model-specific; do not apply it to Qwen's full-attention or recurrent layers without equivalent state semantics. |
| Batch-invariant execution and deterministic sampling | Add conformance tests and explicit reproducibility mode. |
| Expert/attention/context parallelism and cluster disaggregation | Defer for the single-Mac target; cluster benchmark numbers are not local performance targets. |

Also evaluate rotation-based low-bit cache quantization such as TurboQuant against simple groupwise KV formats. Count transform, packing, decode and scratch costs on Metal; published accelerator speedups cannot be assumed locally. [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Model shortlist

These are integration candidates, not claims of existing Tur support or a measured quality ranking. Weight estimates below are mathematical 4-bit payload floors: parameter count × 0.5 bytes, before metadata, unquantized tensors, runtime state and OS memory.

| Model | Reason to prioritize | Approximate payload floor |
| --- | --- | --- |
| Qwen3.8-27B | Requested flagship; dense hybrid decoder | 13.5 GB |
| Qwen3.5-0.8B/2B/4B/9B | Smaller hybrid models for tests and constrained Macs | 0.4/1/2/4.5 GB |
| Phi-4-mini-instruct | Compact independent dense-model candidate | 1.9 GB |
| Gemma 4 26B-A4B | Independent sparse architecture candidate after MoE kernels exist | 13 GB |
| Ministral 3 3B/8B/14B | Compact dense alternatives with official GGUF releases | 1.5/4/7 GB nominal language-weight floors; inspect exact checkpoint |

Sources: [Qwen releases](https://github.com/QwenLM/Qwen3.8), [Qwen3.6 MoE](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Phi](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B), [Ministral 3](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF). Keep Granite as a regression model during migration. Do not choose a default checkpoint until target memory and latency requirements are known.

## Implementation order and release gates

1. **Workspace and baseline:** preserve existing behavior, establish Metal benchmarks, record hardware/OS/compiler/revisions. Correct documentation that confuses intended features with measured behavior.
2. **Hybrid state and Qwen:** implement validated loading and text-only reference execution; compare logits, tokenization, templates, EOS and chunk boundaries to a pinned reference. Reject unsupported architectures before loading large weights.
3. **Metal execution:** packed weights, fused primitives, DeltaNet prefill/decode, direct paged attention and quantized KV. Differential tests must cover non-aligned shapes, page boundaries and long contexts.
4. **Serving:** true ragged batches, concurrent cancellation, prefix forks, memory pressure, fairness and overlapping scheduling. Test isolation, resource reclamation and no use of freed in-flight pages.
5. **Advanced optimizations:** speculative decode, adaptive cache precision and shape tuning only when end-to-end measurements justify them.

Compare with pinned MLX-LM and llama.cpp on the same Mac and equivalent weights/quantization, prompts, output lengths, sampling and cache state. Measure prompt lengths 128/2K/8K/32K, concurrency 1/2/4/8/16 where memory permits, cold/warm prefixes, mixed lengths and sustained load. Report TTFT and inter-token latency p50/p95/p99, prompt and decode tokens/s, successful requests/s, peak resident memory, queueing and thermal conditions. Synchronize timing boundaries to include completed GPU work. [MLX-LM](https://github.com/ml-explore/mlx-lm), [llama.cpp](https://github.com/ggml-org/llama.cpp)

Require quantized quality evaluation on held-out language, code, tools and long-context retrieval, not just perplexity. Set numerical tolerances and quality/latency budgets before selecting a kernel or format. Publish benchmark artifacts and rejected configurations. “Ultrafast” is a goal until those measurements exist.

## Detailed work packages for Sol

Complete these sequentially along the dependency graph. Each package should leave a buildable tree, update this document's status, and report tests and measured limitations. Do not mark a model supported merely because its config parses or a synthetic forward pass succeeds.

### P0 — Establish evidence and resolve platform capabilities

Affected files: `Cargo.toml`, `benches/tur.rs`, existing tests, new benchmark fixtures and result artifacts.

- Record the initial git state; preserve the user's Rust 1.99 change and unrelated files.
- Capture current unit/integration results before reorganizing files. Separate offline tests from checkpoint-download tests; never silently skip failures because hardware is unavailable.
- Record machine chip, GPU cores, memory, OS, power mode, Xcode/Metal compiler and Rust versions. Until the user names target machines, use the available Mac for development without presenting it as representative of every Mac.
- Build a manifest for benchmark runs: checkpoint revision and hash, tokenizer/template revisions, quantizer settings, exact token counts and seeds. Record cold model load independently from warmed inference.
- Trace allocations, GPU command submissions, synchronization points, page copies and sampling transfers. Establish which existing Candle operations remain useful.
- Run a minimal custom-MSL compilation/dispatch/readback experiment through available safe bindings. Validate buffer lifetime and completion semantics. Do not hide new unsafe code in another Tur crate. If the available APIs require unsafe calls, document the exact gap and propose an approved safe dependency/upstream API before proceeding with that path.
- Determine which Metal 4 features the existing dependencies expose. Avoid treating shader language support as proof that host APIs are available through the current Rust bindings.

Exit: reproducible baseline artifacts, capability matrix, and a documented feasible safe dispatch path. A benchmark that only times enqueueing GPU work does not pass.

### P1 — Introduce the workspace without changing inference

Move current library/tests/benchmarks to `crates/tur`, CLI to `crates/tur-cli`, and server binary to `crates/tur-server`. Centralize third-party versions in `[workspace.dependencies]` and package metadata where useful; keep root profiles and one lockfile. Preserve binary names and root invocation commands. Resolve assets relative to a deliberate location, not an accidental working directory.

Do not enable all old restriction lints blindly and mix hundreds of unrelated fixes into the move. Inventory inherited violations and enforce restrictions on new code; complete cleanup before declaring the production rewrite finished. Every member eventually opts into workspace lints explicitly.

Exit: workspace checks, offline unit tests and existing server tests pass or match documented baseline failures. No new third-party crate/version is introduced by the move.

### P2 — Extract ownership boundaries and hybrid contracts

Move existing shared functionality into the proposed crates; retain `tur` as a facade. The dependency graph is:

```text
tur-core
   ↑
tur-metal
   ↑
tur-models
   ↑
tur-engine
   ↑
tur (facade)
   ↑
tur-cli / tur-server
```

Higher layers may also directly depend on `tur-core`; HTTP implementation belongs in `tur-server` after compatibility is resolved. Keep weight loading in `tur-models` initially instead of introducing another small crate. Put MSL sources, compilation, packing and device tests in `tur-metal`.

Define these semantics before optimizing them:

| Contract | Required invariants |
| --- | --- |
| Model description | Explicit layer kinds and dimensions, attention state versus recurrent state, supported input modalities, vocabulary and positional limits |
| Request state | One owner per live request; independently tracked sequence length, convolution history, recurrent matrix, attention pages, RNG and grammar state |
| Execution plan | Packed token ranges and sequence offsets, validated non-overlapping state writes, output positions, kernel/workspace requirements |
| Completion | Resources retained until GPU completion; errors/cancellation cannot publish partial state as valid |
| Checkpoint | Position and model/format identity recorded; restore is atomic across every layer kind |
| Speculative transaction | Accepted prefix committed once; rejected tokens rolled back in state, RNG and grammar bookkeeping |

Use checked arithmetic and validate shape/offset/page indices before submission. Keep long-lived model weights immutable. Prefer one execution owner with bounded messages over adding locks to every tensor; optimize synchronization only when measured.

Exit: small deterministic reference tests for state allocation, independent requests, checkpoint/fork/restore, rollback and cancellation. No cyclic crate dependencies or duplicate Config/error systems.

### P3 — Implement Qwen3.8 correctness first

Affected functionality: current `models/qwen3.rs`, factory detection, weight loading, chat templates, tokenizer, model-state tests.

- Add architecture-specific parsing for `qwen3_5` and nested text settings. Retain explicit model identity separate from architecture family. Reject unsupported layer layouts or tensor formats with useful errors.
- Pin Qwen3.8-27B config/tokenizer/template references. Validate dimensions, layer list, gate behavior, normalization conventions, rotary settings and tensor names against weights/reference code.
- Implement a slow, readable reference for each new mathematical primitive using existing dependencies. Use small tensor fixtures for offline CI; use a real small Qwen hybrid for integration before the larger flagship.
- Implement DeltaNet depthwise causal convolution/history, normalized Q/K, recurrence and output gating; support independent batched histories and partial final prefill chunks.
- Implement gated full attention and text position handling. Do not substitute ordinary Qwen3 attention for the newer gate/rotary conventions.
- Handle tokenizer/config EOS IDs and checkpoint templates, including thinking controls. Test tools and multi-turn messages against reference-rendered prompts. Do not append the old `/think` heuristic to every new model.
- Keep the vision encoder disabled explicitly for text-only operation; reject image/video requests rather than silently discarding them.
- Add actual packed-weight execution for supported quantization before advertising the Qwen3.8 integration as production-ready. Full precision reference execution alone is a development milestone.

Exit: reference agreement at intermediate layer outputs and final logits; whole-prefill versus chunked-prefill versus incremental-decode agreement; no state leakage when batch members enter/leave; coherent quantized real-checkpoint generation. Record reference tolerances by dtype and operation.

### P4 — Build the Metal runtime and first fused kernels

Implement capability queries, reusable pipeline compilation, explicit buffer layout validation, allocation pools, scratch planning and completion tracking. Keep intermediate activations on the GPU. Cache compiled variants by device capability, dtype, shape family and layout version.

Kernel order:

1. Reduction/normalization and elementwise primitives, residual addition, activation/gating, rotary application.
2. Packed 4/8-bit matrix-vector and small-batch matrix multiplication for decode; tiled quantized matrix multiplication for prefill.
3. DeltaNet decode update and fused convolution/gates; chunked parallel DeltaNet prefill with numerically tested boundary state.
4. Ragged attention prefill and paged decode, with online softmax and direct grouped-query addressing.
5. Vocabulary projection/sampling optimizations after profiling their share of runtime.

FP16/BF16 selection is capability- and quality-dependent. Keep FP32 softmax/reduction/recurrent accumulation where needed. Packed low-bit formats are storage layouts, not a promise of native low-bit arithmetic. Compare each custom kernel to the existing backend at representative shapes; retaining a faster existing kernel is acceptable.

Exit: device/reference differential tests, allocation bounds, no unexpected CPU fallback or hidden full-weight expansion, and kernel benchmark artifacts. Do not hard-code one threadgroup configuration for all devices.

### P5 — Make memory genuinely paged and quantized

Replace `k_full`/`v_full` concatenation with page-native kernels. Model metadata determines which layers need KV; recurrent layers consume their own state pools. Implement generation-tagged handles or equivalent stale-handle protection, copy-on-write tails, page reclamation after completion and consistent accounting across all pools.

Store packed K/V with explicit scales and optional recent high-precision tokens. Make block size, quantization group size and layout independently validated; tune candidate page sizes such as 16/32/64/128 tokens instead of assuming one optimum. Pin format versions in prefix entries and persisted metadata.

Memory planning uses:

```text
resident = packed weights + attention KV + recurrent state
         + retained checkpoints + activation/scratch peak
         + speculative state + in-flight buffers + runtime overhead
```

For ordinary attention, payload bytes per token are `2 × KV heads × head dimension × bytes per element`, summed only over attention layers, plus scale/packing overhead. Derive recurrent memory from the real state tensors. Never substitute activated MoE parameter count for resident weights.

Exit: no whole-cache reconstruction on decode; bounded allocation growth; quantization roundtrip and attention quality tests; exact full-prefix and partial-prefix restore tests; memory-pressure tests without corruption or unbounded swap reliance.

### P6 — Replace nominal batching with packed execution

Refactor the existing scheduler and batch manager around validated execution plans. Combine projections across variable-length requests, track actual token boundaries, gather only required logits and eliminate padding from the main path. Separate decode and prefill kernels may share one scheduling iteration; a single monolithic kernel is not required.

Start with decode-priority scheduling plus bounded prefill progress. Estimate chunk cost from measured shape/context buckets; bound adaptive changes so noisy timing cannot starve requests. Charge recurrent snapshots and speculative branches to admission. Add maximum queue length, request/token limits, deadlines, cancellation and explicit preemption/recompute policy.

Use a shared device budget across loaded models; the current per-model worker arrangement must not independently promise the same free memory. Begin with one resident model for flagship performance, then test multi-model arbitration separately.

Exit: concurrency stress tests with mixed prompt/context lengths, batch reshuffling, cancellation and OOM injection; bounded fairness; latency/throughput curves that show actual batched execution rather than a loop over requests.

### P7 — Prefix reuse, asynchronous overlap and GPU sampling

Implement prefix lookup over token sequences with collision checks. Decide checkpoint spacing from reuse and snapshot costs; attach DeltaNet snapshots only at valid positions. Persisted caches, if later added, require revision/format validation and atomic writes. SSD spill is optional and must beat recomputation in measured workloads.

Overlap preparing the next execution plan with running the current one. Use completion events and bounded in-flight depth; avoid unbounded submission that worsens cancellation and p99 latency. Report CPU scheduling time and GPU idle gaps.

Implement GPU greedy/top-k/top-p/temperature/penalty operations and per-request RNG. Validate categorical sampling statistically and deterministic modes separately. Integrate grammar constraints without changing the probability semantics. Replace unbounded streaming channels with bounded delivery plus an explicit slow-client policy.

Exit: warmed prefix reuse produces equivalent continuations and measurable saved work; no leaked state after disconnects; reproducibility behavior documented; bounded streaming memory under stalled clients.

### P8 — Speculation and MoE expansion

Add an exact verification path before optimizing proposals. Verify all proposed positions, track accepted length, roll back every layer's state, and apply the correct target/proposal sampling correction. For recurrent layers, retain bounded intermediate checkpoints or recompute the accepted prefix from the pre-draft checkpoint.

Only enable native MTP if the checkpoint actually contains validated usable tensors and the decoding algorithm is implemented. A config field alone is insufficient. Measure accepted tokens per verification, extra memory, draft cost and end-to-end latency across concurrency levels.

For Qwen3.6-35B-A3B, add routing/top-k, shared experts where applicable, grouped expert GEMM, token gather/scatter and quantized expert storage. Preserve routing precision and validate expert weights/layouts. Compare throughput under real routing skew, not uniform synthetic routing alone. Then evaluate Gemma 4 as a separate architecture adapter, not a renamed Qwen model.

Exit: exact rollback tests, sampling equivalence tests, actual speedups in enabled regimes, and independent model conformance. Disable optimizations automatically outside validated beneficial ranges.

### P9 — Production release and positioning

Remove unsupported production CPU/CUDA paths and document the chosen macOS/Apple Silicon capability floor. Preserve reference math only as a development aid. Finish unsafe/panic cleanup, activate workspace lints, and run unit tests with cargo-nextest plus checkpoint-dependent suites on an appropriate Mac.

Add health/readiness reporting, bounded startup loading, failure reporting, request metrics, graceful shutdown, stream cancellation, model revision reporting and diagnostic memory/kernel statistics. Preserve API semantics for usage accounting, stop sequences, tool calls, thinking and structured outputs. Run sustained thermal and leak tests, not only short warmed loops.

Publish the benchmark matrix, quality evaluations, reproducible commands, known limitations and supported model/quantization combinations. Replace unsupported README speedup claims with measured results. Describe performance separately for single-user latency, prefill throughput and concurrent serving throughput.

Exit: release gates below pass and every marketed feature has an end-to-end test or benchmark.

## Quantitative gates to set before optimization

These are proposed project gates, not claims that they have been met:

- **Numerics:** per-operation absolute/relative tolerances derived from FP32 reference ranges; no NaN/Inf; exact indexing/position/state transitions. Do not use one permissive tolerance for all kernels.
- **Quantized quality:** establish held-out baseline scores and predeclare allowed degradation per task before tuning. Include uncertainty intervals; do not pick defaults solely from a short anecdotal prompt set.
- **Kernel selection:** a custom path must improve its intended shape regime beyond run-to-run noise and satisfy correctness. Use fallback dispatch for losing shapes.
- **Serving:** no more than a proposed 5% regression in p95 decode latency when enabling a throughput optimization at the chosen service load, unless an explicitly documented throughput mode accepts that tradeoff.
- **Competitive target:** aim for at least 20% better successful-token throughput than the fastest comparable baseline on the primary workload at the same latency and quality budget. If it is not achieved, report the actual result and continue profiling; do not rename a synthetic microbenchmark win as an engine win.
- **Memory:** measured peak stays inside the configured device/process budget including reserved headroom, and a multi-hour cancellation/churn workload shows no unbounded growth.
- **Coverage:** target the repository's >80% requirement with substantive state/scheduler/kernel tests; include faults and boundary cases, not only implementation-mirroring tests.

## Handoff instructions

Start with P0 and P1. Keep the public-contract decisions visible before P2. Implement one end-to-end small hybrid model path before expanding the supported-model list. Complete Qwen3.8-27B correctness and packed quantized execution before speculative decoding or unrelated model families. Do not spend the first iteration implementing every research idea.

The user has explicitly authorized a `crates/` layout. Breaking Rust contracts and new third-party dependencies still follow AGENTS.md: obtain approval of the concrete contract/dependency proposal where not already authorized. No exception to the no-unsafe rule is proposed. Hardware-specific performance claims require target-Mac measurements, and pending machine preferences must not be replaced by invented requirements.
