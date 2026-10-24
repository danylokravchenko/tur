# Tur: Apple Silicon inference redesign

Status: P0's development-Mac evidence/capability gate and P1's workspace move are complete (2026-10-21). P2's four-crate extraction and explicit hybrid state and packed execution contracts are implemented (revised after review); see [P2 contracts and validation](apple-silicon-p2-contracts.md). Safe dispatch and pinned model/HTTP evidence remain in [the baseline report](apple-silicon-baseline.md). The baseline checkpoint suite retains its guided-JSON failure (68/69 pass); it was not rerun or fixed in P2. The legacy models (ordinary Qwen3, Granite 4.1), their execution/cache path and model runners have been removed; see [Breaking changes and legacy removal](#breaking-changes-and-legacy-removal). P3's model work is complete (2026-10-22): the Qwen3.5 dense hybrid text model matches Transformers on the real Qwen3.5-0.8B checkpoint on CPU and Metal, with Q8_0/Q4K packed execution; Qwen3.8-27B shares that path and is validated against its pinned config, tensor headers and template but has not been executed on this hardware; see [P3 report](apple-silicon-p3-qwen.md). P3's deferred legacy deletion and CLI/server migration are now complete: both adapters use the packed hybrid text pipeline, with checkpoint templates/defaults, independent outcomes, EOS and cancellation. P4's Metal runtime and first fused kernels are implemented (2026-10-22): the fused Qwen3.5 path matches the F32 reference on the real 0.8B checkpoint (≤ 3.8e-6) and is the default on Metal for the engine, CLI and server; chunked DeltaNet prefill remains open; matrix-unit attention prefill is now implemented in the [tiled attention report](apple-silicon-tiled-attention.md); see [P4 report](apple-silicon-p4-metal.md). P5's paged and quantized memory is implemented (2026-10-22): K/V and recurrent state live in persistent device pools read in place through page tables, with generation-tagged leases, copy-on-write tails, completion-gated reclamation, F32/F16/Q8 K/V quantized on write, and a budgeted memory plan with explicit out-of-memory errors; see [P5 report](apple-silicon-p5-memory.md). P6's packed execution and scheduling are implemented (2026-10-22): packed recurrent dispatches over the whole batch, decode-priority scheduling with cost-adaptive bounded prefill, recompute preemption, queue limits, deadlines and a shared device budget; see [P6 report](apple-silicon-p6-scheduling.md). P7 is implemented (2026-10-23): verified zero-copy prefix snapshots at aligned boundaries, GPU penalties and top-k/top-p sampling with per-request RNG, an off-thread completion fence that also returns outputs, and bounded streaming with a slow-client policy. Follow-ups moved grammar masks to the GPU (constrained passes now cost the same as unconstrained ones), added prompt-end snapshots with retained logits (an exact repeat needs no prefill pass), sized snapshot retention by state memory, and measured overlap under streaming arrivals (TTFT p50 −13–14%); see [P7 report](apple-silicon-p7-serving.md). Production serving hardening, performance and quality gates remain open.

P8 is in progress (2026-10-23): exact prompt-lookup verification is connected to serving,
with adaptive draft selection, explicit memory accounting and an opt-in throughput mode.
The Qwen3.6 expert text path has F32 routing, packed expert weights and device-side
grouped GEMM, validated on an independent small Transformers fixture and pinned flagship
headers. Native MTP drafts from the checkpoint's own head (validated against a
Transformers-assembled reference, 71–90% next-next-token accuracy) add 1.16× / 1.11× on
novel greedy text at one and four sequences and are gated off beyond. Q4K small batches
now stay in F32 kernels. Full 35B execution exceeds this 16 GB Mac's capacity; flagship
quality, the flagship's expert MTP head and Gemma 4 execution remain open. See the
[P8 report](apple-silicon-p8-speculation-moe.md).

A kernel performance pass followed P8 (2026-10-23). Q4K uses a dedicated matrix-unit
kernel for 2–16 rows (1.4–1.8× faster), the recurrent decode step is 1.35–1.7× faster,
the token-serial DeltaNet scan was rewritten (2.0–2.7×), Q8 decode attention is
1.4–1.5× faster and Q4K single-token expert projections match Candle. On the 0.8B
checkpoint, 2K prefill is 1.13–1.14× faster and Q4K batched decode 1.11–1.32×; Q8_0
decode gains 0–9%. Chunked DeltaNet prefill and the MoE wide-tier gap remain open. See
the [kernel performance report](apple-silicon-kernel-performance.md).

P10 ran on 2026-10-24, ahead of P9 and measured against the kernel-pass commit on the
M1 only. Chunked DeltaNet prefill was implemented and verified but ran at 0.41–0.66× the
token scan at every shape, so it was removed (preserved as a patch). Decode attention now
pairs query heads over Q8 K/V: 1.06–1.25× attention and up to 1.09× model decode at 8K
context. The recurrent prepare pass writes scales instead of normalised copies, cutting
its scratch by 98% with no model-level speed change. TensorOps run on the M1 without
unsafe bindings but only match Candle's F32 GEMM. MoE and F16/BF16 items are deferred.
See the [P10 report](apple-silicon-p10-kernels.md).

Post-P8 roadmap added on 2026-10-23: P9 establishes a fresh comparative baseline;
P10–P15 evaluate kernel, quantization, execution, speculation and prefix-scheduling
improvements; production release moves from P9 to P16. These packages are planned,
not implemented. Their performance ranges are engineering targets, not measured
speedups; see [Post-P8 performance expectations](#post-p8-performance-expectations).

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

Implementation audit (2026-10-23): TurboQuant and KIVI are absent; attention storage is
F32/F16/symmetric groupwise Q8. P8 adapts verification using historical acceptance and
measured costs, without DSpark's per-position confidence head. P7 overlaps intake and
cancellation with device work, but selects the next pass only after current outputs
are known. Its reproducibility mode serializes requests rather than making concurrent
kernels batch-invariant. P10–P15 below turn these distinctions into explicit experiments.

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

Follow the dependency graph and the post-P8 sequencing below. Each package should leave a buildable tree, update this document's status, and report tests and measured limitations. Do not mark a model supported merely because its config parses or a synthetic forward pass succeeds. A research package may close with a measured rejection; implementing every candidate is not a release requirement.

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

Implemented on 2026-10-22; design, tests, measurements and limits are in
[the P6 report](apple-silicon-p6-scheduling.md). `DeltaNet` convolution and recurrence run
over packed sequences, with step/scan dispatches for decode/prefill (1.2× at batch 8–16 for that work).
The scheduler runs decodes first and gives prefill a measured, damped budget with a
minimum per pass and a queue-age guard. The October 23 review normalizes cost samples,
bounds worker intake, enforces submission deadlines before publication, preserves replay
chunk boundaries, reclaims younger state for aged prefill, and reserves model memory
before loading weights. Typed HTTP errors and sampled-token benchmark timestamps are
covered by regression tests. It preempts the newest request under memory
pressure; recomputation preserves original rounding boundaries and the RNG stream; arbitrary
batch shapes can still differ numerically. Stress tests cover mixed
lengths, reshuffled arrivals, cancellation and pool exhaustion, all matching solo runs,
plus bounded decode gaps and prefill progress. On the mixed 128/2K workload, decoded-text delivery intervals
p95 fall 3.7×; throughput drops 7% and first-text latency p95 rises 14%. These historical
latency measurements timed text fragments; the P6 report distinguishes them from the
corrected sampled-token measurements. Decode-only serving is
unchanged. A follow-up removed most of the per-sequence decode cost:

- Q8_0 projections for 4–16 rows on the matrix units.
- A coalesced recurrent decode step.
- One logits copy per pass and greedy argmax on the device.

The batch-8 model pass falls from 43.4 to 28.6 ms, and greedy serving output at 8 and 16
concurrent requests rises 1.72× and 1.66×. F32 recurrent-state bandwidth is now the
main per-sequence cost.

Recurrent state can also be stored as F16, with stochastic rounding (opt-in,
`--recurrent-state f16`): 6–7% faster batched decode and half the recurrent memory.
Decode KL from F32 state is 5.2e-5 nats on average but grows with context, so F32
remains the default. Projections at 9–16 rows are bound by F32 matrix-unit throughput,
and no precision-preserving variant beat the current kernel.

Refactor the existing scheduler and batch manager around validated execution plans. Combine projections across variable-length requests, track actual token boundaries, gather only required logits and eliminate padding from the main path. Separate decode and prefill kernels may share one scheduling iteration; a single monolithic kernel is not required.

Start with decode-priority scheduling plus bounded prefill progress. Estimate chunk cost from measured shape/context buckets; bound adaptive changes so noisy timing cannot starve requests. Charge recurrent snapshots and speculative branches to admission. Add maximum queue length, request/token limits, deadlines, cancellation and explicit preemption/recompute policy.

Use a shared device budget across loaded models; the current per-model worker arrangement must not independently promise the same free memory. Begin with one resident model for flagship performance, then test multi-model arbitration separately.

Exit: concurrency stress tests with mixed prompt/context lengths, batch reshuffling, cancellation and OOM injection; bounded fairness; latency/throughput curves that show actual batched execution rather than a loop over requests.

### P7 — Prefix reuse, asynchronous overlap and GPU sampling

Implemented on 2026-10-23; design, tests, measurements and limits are in
[the P7 report](apple-silicon-p7-serving.md). Exit evidence:

- **Prefix reuse:** warmed prefixes generate exactly the text of cold runs. Eight users
  behind a 1.5K-token system prompt reach their first token 18× sooner (11.0 → 0.61 s
  p50), with 17× less prefill work. Spacing is 256 tokens, chosen from a 64–512 sweep.
- **No leaks:** cancellation while a pass is in flight, disconnects and stalled streams
  release all request state; snapshots release theirs when evicted.
- **Reproducibility:** run-to-run determinism and a reproducibility mode
  (`PipelineConfig::reproducible`) are tested on Metal. Device sampling passes
  chi-square tests against exact truncated distributions.
- **Streaming memory:** bounded per client: at most `channel_capacity + 1` messages, each
  at most `max_buffered_bytes`.

Sampled serving runs within 1 ms of greedy ITL (1.4× faster than host sampling at 8–16
concurrent requests). CPU scheduling takes 10–30 µs per pass, and the device idles at
most 0.06–0.31 ms between passes (a lower bound). In-flight depth stays one. Gap closure: device grammar masks, prompt-end snapshots with retained logits, state-share snapshot sizing, a streaming-arrival benchmark (overlapped intake: TTFT p50 −13–14%, device idle 9.5% → 5%), and three-run spreads showing greedy throughput unchanged from P6 within noise.

Implement prefix lookup over token sequences with collision checks. Decide checkpoint spacing from reuse and snapshot costs; attach DeltaNet snapshots only at valid positions. Persisted caches, if later added, require revision/format validation and atomic writes. SSD spill is optional and must beat recomputation in measured workloads.

Overlap preparing the next execution plan with running the current one. Use completion events and bounded in-flight depth; avoid unbounded submission that worsens cancellation and p99 latency. Report CPU scheduling time and GPU idle gaps.

Implement GPU greedy/top-k/top-p/temperature/penalty operations and per-request RNG. Validate categorical sampling statistically and deterministic modes separately. Integrate grammar constraints without changing the probability semantics. Replace unbounded streaming channels with bounded delivery plus an explicit slow-client policy.

Exit: warmed prefix reuse produces equivalent continuations and measurable saved work; no leaked state after disconnects; reproducibility behavior documented; bounded streaming memory under stalled clients.

### P8 — Speculation and MoE expansion

In progress; implemented behavior, conformance evidence, measured tradeoffs and remaining
hardware/model gates are in the [P8 report](apple-silicon-p8-speculation-moe.md).

Add an exact verification path before optimizing proposals. Verify all proposed positions, track accepted length, roll back every layer's state, and apply the correct target/proposal sampling correction. For recurrent layers, retain bounded intermediate checkpoints or recompute the accepted prefix from the pre-draft checkpoint.

Only enable native MTP if the checkpoint actually contains validated usable tensors and the decoding algorithm is implemented. A config field alone is insufficient. Measure accepted tokens per verification, extra memory, draft cost and end-to-end latency across concurrency levels.

For Qwen3.6-35B-A3B, add routing/top-k, shared experts where applicable, grouped expert GEMM, token gather/scatter and quantized expert storage. Preserve routing precision and validate expert weights/layouts. Compare throughput under real routing skew, not uniform synthetic routing alone. Then evaluate Gemma 4 as a separate architecture adapter, not a renamed Qwen model.

Exit: exact rollback tests, sampling equivalence tests, actual speedups in enabled regimes, and independent model conformance. Disable optimizations automatically outside validated beneficial ranges.

### P9 — Establish the post-P8 baseline and select experiments

Planned. Freeze an explicitly recorded source revision and worktree diff after relevant
P8 changes settle. Preserve P8's open flagship correctness, MoE performance and hardware
gates; later work does not implicitly close them. Small-model experiments can proceed
while larger-memory hardware is unavailable, with their scope recorded.

1. Re-run the representative benchmark matrix on the development M1, then each available
   target Mac: prompts 128/2K/8K/32K, concurrency 1/2/4/8/16 where memory permits,
   cold/warm prefixes, novel prose/code/tools, mixed arrivals and sustained load. Record
   unavailable cases rather than silently shrinking the matrix.
2. Pin MLX-LM and llama.cpp and compare supported common checkpoints. Match quantization
   exactly where possible; otherwise compare at a declared quality and memory budget,
   identifying format differences. Add vLLM Metal for serving comparisons and BaseRT
   where architecture support permits; no synthetic flagship substitute.
3. Attribute completed time to projections, attention, recurrence, routing, sampling,
   encoding, scheduling and idle gaps. Record weight/KV/recurrent/snapshot/scratch bytes,
   loading peaks, effective bandwidth, dispatch count and allocation activity.
4. Set held-out quality budgets before tuning: language, code, tool/JSON correctness,
   long-context retrieval, and decode-length error growth. Separate calibration data.
   Report repeated-run spreads or confidence intervals and thermal conditions.
5. Rank P10–P15 experiments by measured time or memory share, estimated benefit and
   implementation cost. Record their baseline, acceptance threshold and stop condition.

Exit: a reproducible baseline report and ranked experiment list. At least one real
checkpoint must have matched peer-runtime measurements. Flagship claims remain blocked
until that checkpoint runs. Sources: [MLX-LM](https://github.com/ml-explore/mlx-lm),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[vLLM Metal](https://docs.vllm.ai/projects/vllm-metal/en/stable/),
[BaseRT](https://arxiv.org/html/2607.00501v1).

### P10 — Finish parallel prefill and tune native Metal math

Measured on the M1 (2026-10-24) without a P9 baseline; results, rejected variants and
deferrals are in the [P10 report](apple-silicon-p10-kernels.md). Chunked prefill is a
measured rejection on this device, not an open implementation item; retry it on hardware
with faster matrix units or through TensorOps.

Depends on P9. Work in `tur-metal` kernels and `tur-models`' fused Qwen path, using
the existing `math::delta_chunked` reference and state contracts.

1. Implement chunkwise gated DeltaNet prefill using the WY/UT formulation and matrix
   operations. Sweep chunk sizes such as 32/64/128; retain recurrent scan for decode
   and short inputs. Handle partial chunks and ragged batches without padding the
   entire batch to its longest sequence.
2. Verify final and requested intermediate states against the recurrence. Cover prefix
   snapshot boundaries, speculative rejection, replay, cancellation, extreme decay
   values and long contexts. Do not change state-rounding semantics accidentally.
3. Benchmark GQA kernels that reuse K/V loads across query heads. Tune tiled attention,
   split-K and projection crossovers by device, shape and dtype. Direct GQA addressing
   already exists; the experiment is additional reuse, not first-time GQA support.
4. Close P8's grouped-MoE performance gap with expert-dependent tile selection, weight
   reuse and routing/gather/scatter fusion where profiling justifies it. Use trained
   routing traces and include routing cost in both sides of comparisons.
5. Evaluate Metal TensorOps for projections, attention and chunked recurrence, including
   quantized inputs and cooperative tensors. Check OS/toolchain capabilities and safe
   host binding access first. M5 GPU neural accelerators need separate measurements;
   they are not the separate Apple Neural Engine. Keep existing kernels on unsupported
   or slower configurations. Any dependency change needs a concrete approval proposal.
6. Treat F16/BF16 activation or tile changes as numerical experiments with their own
   quality gates. Keep sensitive reductions and recurrence accumulation in F32 unless
   independently justified; packed storage alone does not imply native low-bit math.

Exit: boundary-state differential tests pass; winning dispatch regimes improve beyond
noise at model/serving level; losing regimes retain their prior kernels. Publish a
capability/dispatch table and rejected variants. References:
[Gated DeltaNet](https://arxiv.org/abs/2412.06464),
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention),
[Apple Metal TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/).

### P11 — Calibrated weights and bounded loading

Depends on P9; coordinate execution formats with P10. Extend existing checkpoint and
packed-weight machinery rather than introducing a competing loader or weight hierarchy.

1. Compare current Q4K/Q8_0 conversion with calibrated AWQ/GPTQ and sensitivity-based
   mixed precision. Evaluate DWQ only if simpler calibration misses the quality budget.
   These are offline preparation options, not dependencies in the serving hot path.
2. Measure sensitivity by tensor/layer, including embeddings/output projection, recurrent
   projections and MoE routing. Allocate precision under a total byte budget; preserve
   routing precision until independently validated.
3. Choose only formats with efficient measured Metal execution. Existing MLX artifacts
   are not automatically GGML-compatible. Document conversion, scales, packing and
   reference numerics before adding a format.
4. Load validated prepacked artifacts directly, with pinned model/quantizer identities,
   format versions and checked shapes. Bound staging memory and avoid whole-checkpoint
   expansion. Use existing safe APIs; no unsafe mmap workaround.
5. Measure cold load time and peak process memory separately from steady-state decode.
   Publish quality-versus-resident-bytes curves, not just nominal bits per weight.

Exit: held-out quality gates and packed-kernel conformance pass; the selected artifact
improves quality at equal bytes or reduces bytes at equal quality. Same-bit calibration
alone is not a speedup claim. Reference:
[MLX-LM learned quantization](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md).

### P12 — Low-bit KV and hybrid-state memory

Depends on P9 and stable P10 attention interfaces. TurboQuant is an experiment in this
package, not a predetermined replacement for Q8.

1. Compare F16/Q8 against simple groupwise Q4, KIVI-style asymmetric keys/values and
   TurboQuant rotation/scalar quantization with the appropriate residual correction
   for inner products. Identify the exact algorithm variant; a plain int4 cache is
   not TurboQuant. Test any recent high-precision region as a separate variable.
2. Fuse packing on write and dequantization/score correction on read into page-native
   kernels. Count rotations, codebooks, scales, residuals, padding and scratch. Do not
   materialize a full dequantized cache or conflate attention-kernel and engine gains.
3. Version state identities and cover page boundaries, copy-on-write, prefix restore,
   speculation, cancellation and completion-gated reclamation for every format.
4. Evaluate language/code/retrieval quality across context lengths and attention scales.
   Include multi-turn reuse and generated continuations, not only teacher-forced logits.
   Published quality neutrality on other architectures is not a Tur guarantee.
5. Reassess F16 recurrent state separately from KV compression: long-decode error,
   snapshot retention and compatibility with speculative rounding. P6 already supplies
   stochastic rounding; P8 currently requires F32 state for speculation. Retain F32
   fallback and do not add low-bit recurrent storage without supporting quality evidence.
6. Select by whole-process memory and useful throughput at a fixed quality/latency
   budget. KV savings may increase context or admission capacity without improving
   single-request decode. Count only attention layers in KV estimates.

Exit: one winning low-bit format or a documented rejection; full quality and lifecycle
tests; memory accounting includes all metadata. References:
[TurboQuant](https://arxiv.org/abs/2504.19874),
[KIVI](https://github.com/jy-yuan/KIVI),
[Open-TQ-Metal](https://arxiv.org/html/2604.16957v1). Open-TQ-Metal's large attention
speedups use a dequantize-then-attend baseline; Tur already reads packed pages directly.

### P13 — Reusable execution plans, overlap and batch invariance

Depends on P9; specialize plans for validated P10/P12 variants. Build on current
completion ownership, scratch planning and device sampling.

1. Instrument host allocations and GPU buffer reuse after warmup. Reuse packed-plan
   vectors, page metadata, sampling buffers and activation scratch for bounded shape
   classes. Account for retained capacity; distinguish pooled allocation from no
   allocation. Preserve dynamic admission and page-native memory.
2. Separate reusable planning from execution, with invalidation on layout, shape,
   capability and kernel changes. Evaluate command/argument reuse only if safe APIs
   and measured encoding costs justify it; do not assume CUDA graph APIs exist on Metal.
3. Overlap next-pass preparation with current execution where dependencies permit.
   Sampled tokens, accepted draft lengths and state ownership still gate final bindings.
   Evaluate device-resident token handoff and a bounded submission depth, initially no
   more than two, only after lifetime/cancellation tests cover the new ordering.
4. Add a distinct concurrent batch-invariance experiment: fixed per-request arithmetic
   and reduction ordering across batch size, ordering and chunk shapes. Test logits,
   tokens, RNG, preemption and speculation. Report its throughput cost separately from
   the existing serial reproducibility mode; do not promise cross-device bit identity.

Exit: measured reduction in allocations/encoding/idle time with no stale state,
unbounded buffering or latency-budget violation. Keep the one-pass path when overlap
does not pay. References: [FlashInfer planning](https://docs.flashinfer.ai/api/attention),
[BaseRT](https://arxiv.org/html/2607.00501v1),
[DeepInfer scheduling](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md).

### P14 — Confidence-aware and learned speculative decoding

Depends on P8 correctness and P9 measurements; re-profile after P10–P13 because their
changes alter verification costs. This package is optional if no compatible drafter pays.

1. Establish the best existing P8 policy per workload, including ordinary decode and
   prompt lookup. Measure native MTP upkeep, accepted tokens per pass, burst gaps and
   total state/logit/draft memory. Resolve native MTP/prefix-cache state compatibility
   only with a precise boundary-state design; disabling prefix reuse has an opportunity cost.
2. Extend verification costs beyond a single linear slope: capture projection-kernel
   crossovers, context, sampling mode and batch occupancy. Evaluate per-position
   acceptance estimates, then a calibrated confidence head when a compatible checkpoint
   exists. Select prefixes by expected accepted tokens per total pass time, with fairness
   and latency limits; historical mean acceptance is not DSpark's confidence mechanism.
3. Evaluate a compatible small draft model and one learned candidate from EAGLE-3,
   DFlash or DSpark. Verify architecture, tokenizer, feature and head compatibility first.
   Public ordinary-Qwen3 drafts do not establish hybrid-Qwen compatibility. Training or
   acquiring a matching draft is a separately sized prerequisite, not assumed available.
4. If stochastic proposals are introduced, implement and statistically validate their
   target/proposal correction. Keep deterministic target matching as its own algorithm.
   Roll back recurrent/attention state, RNG, penalties and grammar together. Add trees
   only after linear verification wins and branch-state costs justify them.
5. Test novel prose/code, repetitive text, sampled and constrained outputs, mixed traffic
   and increasing concurrency. Disable drafting when total throughput or latency loses.
   Test combinations with quantized KV, prefix reuse and recurrent precision explicitly.

Exit: measured improvement over the best applicable P8 baseline at matched quality and
latency, with a bounded memory increase and automatic fallback. Repetitive prompt-lookup
gains are not evidence for learned drafting on novel text. References:
[DSpark](https://arxiv.org/abs/2607.05147),
[DeepSpec](https://github.com/deepseek-ai/DeepSpec),
[DFlash](https://arxiv.org/abs/2602.06036),
[EAGLE-3](https://arxiv.org/abs/2503.01840).

### P15 — Prefix-aware scheduling and retention

Depends on P9 and P7; integrate with P14 only where their state semantics are compatible.

1. Measure shared-prefix locality, lookup cost, saved prefill time and retained state
   bytes on multi-turn, shared-system-prompt and agent workloads. Existing prefix reuse
   is the baseline; do not count its historical 18× TTFT gain again.
2. Evaluate admission ordering that favors reusable prefixes, bounded by deadlines and
   queue-age fairness. Avoid concurrently recomputing the same cold prefix when waiting
   for its in-flight producer is cheaper and cancellation is handled correctly.
3. Compare current never-hit-first/LRU eviction with saved-compute-per-retained-byte
   estimates, and tune checkpoint spacing from reuse and recurrent snapshot costs.
   A radix index is an option only if it beats the existing verified hash lookup.
4. Preserve namespace isolation, collision verification, exact recurrent boundaries and
   copy-on-write behavior. Evaluate cold, low-reuse and adversarial churn workloads too.

Exit: fewer recomputed tokens and improved TTFT/throughput at the same state budget,
without starving unrelated requests or regressing low-reuse traffic beyond its budget.
Reference: [SGLang RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/).

### P16 — Production release and positioning

Formerly P9. Production hardening can proceed alongside experiments; release does not
require every research candidate to win. Each candidate must be accepted with evidence,
rejected, or explicitly deferred, and only validated combinations become defaults.

CUDA feature wiring and obsolete legacy errors were removed during the P3–P5 cleanup, and every member now inherits workspace lints (unsafe Rust forbidden; unwrap/expect and explicit panic paths denied). Retained HTTP tests propagate setup/task errors; test assertions have a scoped lint allowance. Document the chosen macOS/Apple Silicon capability floor, preserve CPU reference math as a development aid, and run unit tests with cargo-nextest plus checkpoint-dependent suites on an appropriate Mac.

Add health/readiness reporting, bounded startup loading, failure reporting, request metrics, graceful shutdown, stream cancellation, model revision reporting and diagnostic memory/kernel statistics. Support usage accounting, stop sequences, tool calls, thinking and structured outputs; their API shape may differ from the old one. Run sustained thermal and leak tests, not only short warmed loops.

Publish the benchmark matrix, quality evaluations, reproducible commands, known limitations and supported model/quantization combinations. Replace unsupported README speedup claims with measured results. Describe performance separately for single-user latency, prefill throughput and concurrent serving throughput.

Exit: release gates below pass and every marketed feature has an end-to-end test or benchmark.

## Post-P8 performance expectations

These are planning hypotheses as of 2026-10-23, not benchmark results or guaranteed
forecasts. Use P9's frozen current implementation as the baseline, not the old P3
reference or a competitor's unoptimized kernel. Compare the same Mac, checkpoint,
workload, quality budget, cache state and service latency target. Report speculative
and ordinary decoding separately. No improvement remains a possible experiment outcome.

| Work and regime | Working target or bound | Basis and limitation |
| --- | --- | --- |
| P10 chunked DeltaNet, long prefill | About 1.06–1.10× whole-model throughput **if** the chunked kernel is 2–4× faster than the current scan | Revised after the 2026-10-23 kernel pass: the rewritten token-serial scan is about 12% of 2K prefill (18 × 10.4 ms of ~1.6 s), down from about 21%; removing it entirely would give at most about 1.14×. **Measured 2026-10-24: the chunked kernel was 0.41–0.66× the scan on M1; rejected** |
| P10–P13 combined, M1 prefill | Aim for 1.10–1.30× prompt tokens/s at unchanged quality | Engineering target; projections already ran near the measured M1 F32 ceiling in P4, limiting gains without changing precision or workload |
| P10–P13 combined, ordinary M1 decode | Aim for 1.05–1.15× tokens/s in selected beneficial regimes | Engineering target; weight/state bandwidth and already-small host scheduling costs limit general gains. Batch-1 improvement may be negligible |
| P12 low-bit KV versus Q8 group 32 | Roughly 1.7–2.4× smaller **KV storage**, subject to measured metadata/padding | Q8 uses 8.5 bits/value including one F16 scale per 32 values. Illustrative 5–3.5 effective-bit candidates give this range; formats and quality still need validation |
| P12 F16 recurrent matrices versus F32 | 2× smaller matrix payload; less than 2× smaller total recurrent state | Already available in P6, not a new gain. Historical serving throughput improved only 4–5% at concurrency 8–16; long-context quality and speculative compatibility remain constraints |
| P11 calibrated weights | Better quality at equal bytes, or fewer bytes at equal quality; no fixed speedup forecast | Same-bit calibration may change no runtime cost. Lower precision helps only if corresponding kernels win and sensitive tensors retain adequate precision |
| P14 learned/confidence speculation, novel text at low concurrency | Exploratory goal: 1.20–1.50× tokens/s over the best applicable P8 policy | Low confidence until a compatible drafter and total cost are measured. At high concurrency the desired result may simply be neutral fallback |
| P15 prefix-aware scheduling | Lower TTFT and fewer recomputed tokens on reuse-heavy traffic; no universal multiplier | P7 already captures large prefix wins. Additional benefit depends on hit rate, queue contention and snapshot memory; no gain is expected with no reuse |
| Metal TensorOps on newer Macs; full flagship MoE | No numerical forecast yet | Requires same-device before/after measurements and real checkpoint execution. Changing hardware is not a software speedup |

The [kernel performance pass](apple-silicon-kernel-performance.md) predates P9 and is
part of its baseline; the targets above are incremental to it, not to the P8 state.

The prefill/decode targets above are incremental **combined** targets, not sums or
products of package estimates. P14's exploratory target is also not a multiplier to
apply to them: faster verification changes draft economics and the best baseline.
P9 must revise these targets using measured bottlenecks, including downward revisions.

For a fraction `f` of baseline runtime accelerated by factor `s`, the whole-pass bound is
`speedup = 1 / ((1 - f) + f / s)`. At the scan's measured `f ≈ 0.12` after the kernel
pass, even eliminating it entirely cannot exceed about 1.14×. Similarly, eliminating a genuinely measured 5% idle fraction alone
cannot exceed 1.053×; it does not establish the size of all encoding or synchronization
costs. These bounds explain why kernel speedups should not become engine promises.

Memory savings also apply to a component, not the entire process. If KV is 20% of resident
memory and its storage halves, total memory falls only 10%, holding other components
fixed. For a hybrid Qwen checkpoint, weights and recurrent matrices can dominate.
More available memory can increase context or concurrency, but not necessarily useful
throughput at the same p95 latency. No compression experiment guarantees that a flagship
fits a 16 GB Mac.

Historical evidence to retain separately: [P8 native MTP](apple-silicon-p8-speculation-moe.md)
measured 1.16× / 1.11× over ordinary decoding on novel greedy text at concurrency 1 / 4;
prompt lookup reached 3.25× on a repetitive single-request workload. Those gains already
exist, involve different baselines, and are not additional post-P8 expectations. The
throughput mode's burst-gap/p95 tradeoff must remain visible.

### Experiment sequencing and deferred research

Run P9 first, then prioritize P10's unfinished prefill work. P11 and P12 follow the
measured weight/state bottleneck; P13 follows measured allocation/encoding/idle costs.
Re-profile before P14. P15 needs representative reuse traffic and may proceed independently
of learned drafting. Carry P8 correctness gates forward; begin P16 hardening without
waiting for optional research to succeed. Existing ARM64 host work remains profile-driven;
audit hot compiler output and QoS only where P9 exposes material host cost.

Keep SnapKV-style token eviction and sparse attention as explicit approximate modes for
future evaluation, not replacements for exact full-attention semantics. FlashAttention-4
and SageAttention supply algorithmic ideas, not directly portable NVIDIA kernel speedups.
DeepInfer SWA bounded replay requires compatible model state semantics. Distributed
prefill/decode, expert parallelism, SSD spill and separate Neural Engine execution remain
deferred until a measured workload justifies their complexity. New attention architectures
such as MLA require matching checkpoints; they are not switches for the current Qwen model.
References: [SnapKV](https://arxiv.org/abs/2404.14469),
[FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/),
[SageAttention](https://arxiv.org/abs/2410.02367),
[DeepInfer](https://github.com/deepseek-ai/DeepInfer).

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

P0–P7 are historical milestones; do not restart the workspace migration or completed
model/runtime work. Read P8's current report, preserve in-progress work and record its
remaining gates. Start post-P8 work with P9 and the experiment sequencing above. Validate
the flagship on suitable hardware before claiming support, while allowing small-model
experiments to proceed with explicit scope. Keep public-contract decisions visible and
do not implement every research idea merely because it appears in this plan.

The user has explicitly authorized a `crates/` layout and breaking changes of any kind, including Rust, HTTP and CLI APIs; do not spend effort on backwards compatibility. New third-party dependencies still follow AGENTS.md: obtain approval of the concrete dependency proposal. No exception to the no-unsafe rule is proposed. Hardware-specific performance claims require target-Mac measurements, and pending machine preferences must not be replaced by invented requirements.
