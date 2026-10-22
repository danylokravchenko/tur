# Tur: Apple Silicon inference redesign

Status: P0's development-Mac evidence/capability gate and P1's workspace move are complete (2026-10-21). P2's four-crate extraction and explicit hybrid state and packed execution contracts are implemented (revised after review); see [P2 contracts and validation](apple-silicon-p2-contracts.md). Safe dispatch and pinned model/HTTP evidence remain in [the baseline report](apple-silicon-baseline.md). The baseline checkpoint suite retains its guided-JSON failure (68/69 pass); it was not rerun or fixed in P2. The legacy models (ordinary Qwen3, Granite 4.1), their execution/cache path and model runners have been removed; see [Breaking changes and legacy removal](#breaking-changes-and-legacy-removal). P3's model work is complete (2026-10-22): the Qwen3.5 dense hybrid text model matches Transformers on the real Qwen3.5-0.8B checkpoint on CPU and Metal, with Q8_0/Q4K packed execution; Qwen3.8-27B shares that path and is validated against its pinned config, tensor headers and template but has not been executed on this hardware; see [P3 report](apple-silicon-p3-qwen.md). P3's deferred legacy deletion and CLI/server migration are now complete: both adapters use the packed hybrid text pipeline, with checkpoint templates/defaults, independent outcomes, EOS and cancellation. P4's Metal runtime and first fused kernels are implemented (2026-10-22): the fused Qwen3.5 path matches the F32 reference on the real 0.8B checkpoint (≤ 3.8e-6) and is the default on Metal for the engine, CLI and server; chunked DeltaNet prefill remains open; matrix-unit attention prefill is now implemented in the [tiled attention report](apple-silicon-tiled-attention.md); see [P4 report](apple-silicon-p4-metal.md). P5's paged and quantized memory is implemented (2026-10-22): K/V and recurrent state live in persistent device pools read in place through page tables, with generation-tagged leases, copy-on-write tails, completion-gated reclamation, F32/F16/Q8 K/V quantized on write, and a budgeted memory plan with explicit out-of-memory errors; see [P5 report](apple-silicon-p5-memory.md). Production serving hardening, performance and quality gates remain open.

P3–P5 review cleanup (2026-10-22) fixed zero-slot budget accounting, activated inherited
workspace lints, made retained HTTP tests fallible, and removed unused dependencies,
CUDA features and obsolete errors. Validation and the remaining limits are recorded
in the [P5 report](apple-silicon-p5-memory.md#p3p5-review-cleanup-2026-10-22).

## Product and platform

Build a Rust inference service for Apple Silicon macOS, optimizing prompt throughput and time to first token alongside sustained token generation and tail latency under concurrency. A single Mac is the initial deployment unit. CPU execution remains useful as a numerical test oracle, not a supported production backend. Intel Macs, CUDA, and Linux are outside the proposed production scope.

Use capability-based Metal dispatch; choose a minimum macOS version after measuring the target machines. Metal 4 exposes newer command and tensor facilities, but their availability does not establish that every chip has identical numerical or throughput capabilities. Benchmark specialized kernels against existing Metal implementations before making them defaults. [Apple Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)

### ARM64 host optimization

All CPU-side code is optimized for ARM64 (`aarch64-apple-darwin`), not merely compiled for it. GPU kernels carry the model math, but tokenization, scheduling, execution planning, grammar masks, quantization packing/conversion, checkpoint loading, sampling fallbacks and the HTTP path run on the performance and efficiency cores and directly affect TTFT and tail latency.

- **Build targets:** set the release target CPU deliberately (for example an `apple-m1` baseline, which enables NEON, FP16, dot-product and LSE atomics), and measure whether newer per-generation targets are worth separate builds. Do not ship a host-native build as the release artifact. Enable LTO and `codegen-units = 1` for release profiles when measurements confirm a gain.
- **Vectorization without unsafe:** `std::arch` NEON intrinsics are `unsafe`, so the no-unsafe rule applies. Write hot loops in shapes the compiler auto-vectorizes (fixed-width chunks, `chunks_exact`, no bounds checks inside the inner loop, contiguous structure-of-arrays layouts) and check generated assembly for NEON instructions on hot paths. If a safe abstraction cannot reach the needed throughput, document the gap and propose an approved dependency instead of writing intrinsics in Tur.
- **Memory layout:** Apple cores use 128-byte cache lines; pad and align contended atomics and per-worker state to 128 bytes, not the 64 bytes common on x86. Keep host-visible buffers contiguous and page-aligned for zero-copy sharing with the GPU through unified memory.
- **Concurrency:** rely on ARMv8.1 LSE atomics and the weaker ARM memory model. Use the minimal correct `Ordering` instead of defaulting to `SeqCst`, and test lock-free structures under ARM-specific reordering (loom or stress tests on the device), not only on x86 CI.
- **Core topology:** keep latency-critical host threads (scheduler, plan builder, completion handling) on performance cores using QoS classes, and push background work (prefix eviction, logging, metrics) to lower QoS. Size worker pools from the measured performance-core count, not the total logical CPU count.
- **Benchmarks:** host-side microbenchmarks (planning, sampling, grammar, tokenization, packing) run on the target Macs and record compiler target flags alongside the existing benchmark manifest.

## Repository audit

| Existing implementation | Consequence and replacement |
| --- | --- |
| `models/qwen3.rs` implements ordinary Qwen3 attention; `Qwen35ModelForCausalLM` is only an alias | Renaming it cannot implement Qwen3.8. Delete it together with the alias; validate Qwen3.8 against pinned external references instead. |
| `ModelImpl` exports only K/V pairs and accepts per-layer `PagedKvCache` | Introduce explicit hybrid request state before supporting DeltaNet batching and prefix reuse. |
| `PagedKvCache::append` writes blocks and also concatenates full-sequence K/V | Implement attention directly over physical pages; eliminate the duplicate full cache and per-token concatenation. |
| `InferenceEngine::prefill_batch` loops over requests on its paged path | Pack variable-length prompt chunks into one execution plan with per-sequence offsets. |
| Default Qwen GPU attention falls through to standard attention | Add tiled Metal prefill and paged decode kernels, without materializing a complete attention matrix. |
| GGUF linear layers already construct `QMatMul` | Inspect backend dispatch and measure actual resident storage; stale loader logging incorrectly claimed GPU weights were dequantized. |
| `benches/tur.rs` constructs a CPU factory for Qwen3-0.6B | Remove it with the old model; add reproducible Metal kernel, ARM64 host-path and serving benchmarks before publishing performance claims. |
| Existing loading and CPU kernels contain unsafe code; tests contain unwraps | Remove inherited violations as those paths are replaced; do not introduce new unsafe Rust or panic paths. |

The existing scheduler, templates, tokenizer, grammar support, and tests may be reused where they fit the new design, but nothing is preserved for its own sake. Their presence does not prove efficient GPU batching or cache residency. The existing Rust and HTTP APIs are not optimal and have no external users; they are not a compatibility constraint.

## Breaking changes and legacy removal

Breaking changes are fully encouraged and allowed (user decision, 2026-10-21). This covers public Rust APIs, crate boundaries, the HTTP wire format, CLI flags, configuration files, on-disk cache formats and binary names. Prefer the design that is best for Apple Silicon throughput and maintainability over one that eases migration. Do not add compatibility shims, deprecated aliases, re-export layers or dual code paths to keep the old API alive; delete replaced code in the same change that replaces it.

Remove the existing models rather than keeping them as regression references:

- `models/qwen3.rs`, the `Qwen35ModelForCausalLM` alias and `Model`/`Config` re-exports
- `models/granite_41.rs` and `Granite41ModelForCausalLM`
- their factory detection, chat-template special cases (including the `/think` heuristic), loader branches, benchmarks and checkpoint-dependent tests
- legacy contracts that exist only to serve them: `ModelImpl`, `KvCacheImpl`, the full-sequence `PagedKvCache` path and the CPU flash-attention kernels under `models/attention/cpu_flash`

Removal unblocks deleting inherited unsafe code and unwraps instead of fixing them. Until the first hybrid model runs end to end, CLI and server may support no models; that is acceptable. Correctness evidence for new models comes from pinned external references (Transformers, MLX-LM), not from the deleted decoders. The 68/69 baseline checkpoint results become historical evidence only.

Still subject to AGENTS.md: new third-party dependencies need approval, and the no-unsafe/no-panic rules are unchanged.

## Workspace boundaries

Current workspace after P2:

```text
crates/
  tur-core/     # Original shared errors and checked execution descriptors
  tur-metal/    # State storage, completion, Metal runtime, kernels and packed weights
  tur-models/   # Models, checkpoint loading and state declarations
  tur-engine/   # Request lifecycle, execution and sampling
  tur/          # Facade, HTTP library, integration tests, benchmarks
  tur-cli/      # tur binary
  tur-server/   # server binary
docs/
assets/
Cargo.toml      # Workspace, shared dependency versions and profiles
Cargo.lock
```

The server library module remains in `tur` until adapter migration. Root commands select all seven members by default. Build all members with `cargo check --workspace --all-targets`; run `cargo run --bin tur -- ...` or `cargo run --bin server -- ...` from the repository root.

P2 extraction (implemented; breaking Rust changes explicitly authorized on 2026-10-21):

| Crate | Responsibility and dependency direction |
| --- | --- |
| `tur-core` | Extract the existing shared errors and minimal execution/state descriptors; no runtime, HTTP, or model dependencies |
| `tur-metal` | Persistent paged K/V and recurrent state pools (P5), safe completion ownership, the Metal kernel runtime (P4), MSL kernels, packed weights and scratch planning; no models or scheduler |
| `tur-models` | Checkpoint validation, model implementations, layer/state descriptions; depends on Metal runtime |
| `tur-engine` | Request lifecycle, cache ownership, scheduling, execution and sampling; depends on models/runtime |
| `tur` | Public facade; no compatibility exports for the old API |
| `tur-cli`, `tur-server` | Product adapters; depend on public facade |

Extract existing functionality rather than copying it into parallel foundational modules. Keep model-specific configuration with its model; avoid an unrelated universal Config replacement. The existing shared errors are now in `tur-core` with facade re-exports; do not create a second competing error hierarchy. No new third-party dependencies are selected by this proposal.

## Qwen3.8 migration

The first flagship is **Qwen/Qwen3.8-27B**, initially text-only. Its checkpoint declares `qwen3_5` and a nested `text_config`, with a hybrid sequence of recurrent and full-attention layers. Implement the actual configuration, weights and template semantics; do not map every name containing `qwen3` to the existing decoder. [Official configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)

Required work: validated nested config parsing and tensor-name/shape mapping; gated DeltaNet convolution and recurrent state; chunked parallel prefill and recurrent decode; gated full attention; partial rotary embeddings; correct normalization, vocabulary, EOS and output projection; official template handling. Test whole-prompt, chunked and incremental execution against a pinned reference. Treat image/video inputs as unsupported until their encoder and position handling are implemented.

MLX supplies a useful independent implementation reference for the architecture and its separate recurrent/attention caches. Pin a revision and verify Qwen3.8-specific fields rather than assuming any Qwen3.5 implementation is automatically equivalent. [MLX implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)

### Required public contract changes

Replace the assumption that every layer owns K/V with model-declared state kinds: attention page references or recurrent convolution/matrix state. A request owns its state; weights are shared. Execution consumes packed token IDs, sequence boundaries, positions and state handles. State operations must support allocation, release, fork, checkpoint, restore, and speculative commit/rollback.

Prefix entries must associate an exact token boundary with both attention pages and a recurrent snapshot. A recurrent state cannot be truncated as if it were a KV array. Share immutable checkpoints; copy writable recurrent state when branching. Namespace entries by model revision, quantization, adapters, positional settings and tenant policy. Verify tokens on hash hits. Release GPU resources only after all referencing commands complete.

Wire-level chat/streaming behavior may change freely; redesign it where the current shape costs latency or complicates batching. `ModelImpl`, `KvCacheImpl` and the old paged-state arguments are deleted rather than adapted; prefix snapshots and engine/model construction are redesigned around hybrid state. The user explicitly authorized breaking changes on 2026-10-21, satisfying the AGENTS.md confirmation gate. Implemented signatures and ownership rules are in [the P2 contract report](apple-silicon-p2-contracts.md).

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

Sources: [Qwen releases](https://github.com/QwenLM/Qwen3.8), [Qwen3.6 MoE](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Phi](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B), [Ministral 3](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF). Granite and ordinary Qwen3 are removed, not kept as regression models. Do not choose a default checkpoint until target memory and latency requirements are known.

## Implementation order and release gates

1. **Workspace and baseline:** establish Metal and ARM64 host benchmarks, record hardware/OS/compiler/revisions. Correct documentation that confuses intended features with measured behavior.
2. **Hybrid state and Qwen:** delete the existing models and legacy contracts; implement validated loading and text-only reference execution; compare logits, tokenization, templates, EOS and chunk boundaries to a pinned reference. Reject unsupported architectures before loading large weights.
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

Implemented on 2026-10-21. `tur-core` owns the original errors and checked packed
descriptors; `tur-metal` owns extracted caches/memory management plus immutable
hybrid page/recurrent storage and completion fences; `tur-models` owns existing
models/loading and the immutable-weight `HybridModel` contract; `tur-engine` owns
extracted generation code and a single-owner `HybridEngine`. `tur` retains facade
exports and its HTTP library. No third-party package/version was added.

Revised after review on 2026-10-21. `HybridModel::forward` takes one packed batch
(tokens, positions, per-sequence ranges, committed state, requested logits and
recurrent boundaries) and returns one result per sequence. A model can therefore
implement chunked prefill and batched decode directly. The engine validates
descriptors before execution, enqueues one pass, and fences before reading logits.
It then samples, selects the accepted boundary and publishes each request
independently: one request's model error, invalid update, sampling error or finished
grammar never fails the others. Model inputs are zero-copy; outputs are copied once.
Ordinary sequences retain only the final recurrent boundary, and no per-token
snapshots exist. For speculative sequences the engine samples the target at every
position and accepts drafts while they match, so acceptance cannot disagree with
published tokens. State layers declare dtype (F32/F16/BF16), descriptions declare
modalities, and `submit` returns checked payload requirements including model
workspace. Failed-fence quarantine is per engine.

Deterministic offline tests cover packed-versus-incremental agreement, request
reordering, exact page/recurrent values, partial-page forks, atomic restore,
speculative acceptance/rejection, per-request failure isolation, whole-batch failure,
cancellation, stale handles, shape/dtype failures, overflow and invalid plans.
CLI/server used the extracted ordinary attention engine at the P2 exit; P3 has now replaced it with the hybrid pipeline. Exact
signatures, ownership rules, commands and remaining limits are in
[the P2 contract document](apple-silicon-p2-contracts.md).

The implemented boundaries and original package requirements follow. Move existing shared functionality into these crates; retain `tur` as a facade. The dependency graph is:

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

Higher layers may also directly depend on `tur-core`; HTTP implementation moves to `tur-server`; no compatibility layer is required. Keep weight loading in `tur-models` initially instead of introducing another small crate. Put MSL sources, compilation, packing and device tests in `tur-metal`.

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

Completed on 2026-10-22, including the deferred legacy deletion and CLI/server
migration; measurements, pins, adapter behavior and limits are in
[the P3 report](apple-silicon-p3-qwen.md). Both adapters now execute the validated
hybrid model through `HybridEngine`; no ordinary-attention compatibility path remains.
Exit evidence: per-layer and logit agreement with Transformers (relative error
≤ 3.5e-6, against a 1e-4 gate), whole/chunked/incremental agreement, batch
enter/leave isolation, and coherent Q8_0/Q4K generation. Q4K has no quality budget
yet.

Affected functionality: current `models/qwen3.rs` and `models/granite_41.rs`, factory detection, weight loading, chat templates, tokenizer, model-state tests.

- Delete the existing models, their benchmarks/tests and the legacy `ModelImpl`/`KvCacheImpl` engine path first (see [Breaking changes and legacy removal](#breaking-changes-and-legacy-removal)). Rewire CLI/server onto `HybridEngine`; they may serve no model until the first hybrid path passes.
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

Implemented on 2026-10-22; runtime, kernels, tests, measurements and open items are in
[the P4 report](apple-silicon-p4-metal.md). The fused path matches the F32 reference on
the real Qwen3.5-0.8B checkpoint (≤ 3.8e-6, identical greedy tokens) and is the Metal
default. Against the P3 reference it is 3.7–4.5× faster in prefill and 1.6–4.7× in
decode. Kernel order items 1, 2 (decode GEMV; prefill retains Candle's tiled GEMM at
about 2 TFLOPS), 4 and 5 (profiled; no kernel justified) are done. The chunked DeltaNet
prefill in item 3 remains open. Matrix-unit attention prefill is now implemented;
see the [tiled attention report](apple-silicon-tiled-attention.md).

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

Implemented on 2026-10-22; design, tests, measurements and open items are in
[the P5 report](apple-silicon-p5-memory.md). Attention reads persistent page pools in
place (no per-pass concatenation), recurrent layers read and write their own slot pools,
and the model writes state directly into reserved pages and slots. Q8 K/V (group 32)
stays within 1e-4 nats of F32 K/V over 8K held-out tokens with 99.2% top-1 agreement;
quality did not justify a recent high-precision region. Page sizes 16–128 measured
within 3%.

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

CUDA feature wiring and obsolete legacy errors were removed during the P3–P5 cleanup, and every member now inherits workspace lints (unsafe Rust forbidden; unwrap/expect and explicit panic paths denied). Retained HTTP tests propagate setup/task errors; test assertions have a scoped lint allowance. Document the chosen macOS/Apple Silicon capability floor, preserve CPU reference math as a development aid, and run unit tests with cargo-nextest plus checkpoint-dependent suites on an appropriate Mac.

Add health/readiness reporting, bounded startup loading, failure reporting, request metrics, graceful shutdown, stream cancellation, model revision reporting and diagnostic memory/kernel statistics. Support usage accounting, stop sequences, tool calls, thinking and structured outputs; their API shape may differ from the old one. Run sustained thermal and leak tests, not only short warmed loops.

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

The user has explicitly authorized a `crates/` layout and breaking changes of any kind, including Rust, HTTP and CLI APIs; do not spend effort on backwards compatibility. New third-party dependencies still follow AGENTS.md: obtain approval of the concrete dependency proposal. No exception to the no-unsafe rule is proposed. Hardware-specific performance claims require target-Mac measurements, and pending machine preferences must not be replaced by invented requirements.
