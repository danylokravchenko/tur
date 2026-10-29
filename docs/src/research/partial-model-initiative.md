# Initiative: preserve large-model capability within a small memory budget

Created: 2026-10-24. Status: research backlog; no implementation or experiments started. Primary target: Qwen/Qwen3.8-Flash-Next, initially text-only on Apple Silicon.

## Objective and hypothesis

Extract a useful subset of an already trained large model, preserving as much of its reasoning and coding ability as possible on a machine that cannot hold the original. Test whether this beats a complete smaller model at the same total memory budget and acceptable latency. "Keep the smartest part" means measured preservation of useful behavior; it does not assume intelligence occupies a single removable component.

The strongest initial hypothesis is task-specific expert selection, combined with calibrated quantization and offloading of sparsely accessed lookup data. A coding specialist may tolerate losing capabilities that a general assistant needs. Neither success on this model nor superiority over smaller models is established yet.

## Target architecture and constraints

The [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) reports:

- A 125B main network, approximately 6B active parameters per token, plus a 51B n-gram embedding table and a separate 4B multi-token prediction (MTP) component.
- 48 layers: three Gated DeltaNet layers followed by one Qwen Sparse Attention layer, repeated twelve times, with MoE feed-forward blocks.
- 512 routed experts per MoE layer; ten selected per token, plus one shared expert.
- Gated residual connections and bigram/trigram embeddings at layer 2.

These are publisher descriptions, not Tur validation. Pin an immutable checkpoint revision, config, tensor inventory and reference implementation before development. Do not infer Flash-Next support from Tur's Qwen3.5/3.8 dense support: QSA, gated residuals and n-gram lookup semantics require an architecture audit and conformance.

Six billion active parameters are not a fixed six-billion-parameter brain: the selected experts change with token, context and layer. Rare experts can be critical. Experts do not necessarily correspond to clean human categories such as "Rust" or "reasoning."

At an ideal four bits per parameter, the 125B main network alone is about 62.5 GB decimal, before scales, metadata, state or embeddings. This is arithmetic, not a checkpoint-size or fit estimate. Ordinary four-bit conversion alone cannot make it resident on a 16–32 GB machine. Quantized lookup-table size depends on the actual artifact; do not assume every component supports the same precision.

## Research and what it establishes

| Source | Relevant finding | Limit for this initiative |
| --- | --- | --- |
| [Not All Experts are Equal (2024)](https://arxiv.org/abs/2402.14800) | Studies task-specific and task-agnostic expert pruning and dynamic expert skipping. Direct starting point for selecting original experts. | Results on other models do not establish Flash-Next's tolerable pruning ratio. Skipping computation saves resident memory only if unused weights are also excluded or offloaded. |
| [Unveiling Super Experts (2025; revised 2026)](https://arxiv.org/abs/2507.23279) | Reports severe degradation after removing just three of 6,144 experts in Qwen3-30B-A3B; studies rare extreme activations. | Necessity is not sufficiency: preserving these experts alone does not preserve the model. Discover and validate critical experts separately for this checkpoint. |
| [ShortGPT (ACL 2025)](https://aclanthology.org/2025.findings-acl.1035.pdf) | Removing ten of forty Llama-2-13B layers reduced MMLU from 55.0 to 52.2 in a reported experiment. Supports measurable layer redundancy. | Multiple-choice performance does not establish generation quality. Its generation variant uses all layers during decoding, so that variant does not remove the need to store/access those layers. |
| [SliceGPT (2024)](https://arxiv.org/abs/2401.15024) | Replaces weight matrices with smaller dense matrices; reports about 99% retained zero-shot benchmark performance for Llama-2-70B at 25% parameter removal. | Benchmark retention is not a percentage of general intelligence; aggressive compression and this hybrid architecture need separate evaluation. |
| [Compact Language Models / Minitron (2024)](https://arxiv.org/abs/2407.14679) | Combines structured pruning with distillation to recover quality and derive smaller models from pretrained larger ones. | Recovery requires training resources and teacher access; preparation may need a larger machine than deployment. |
| [AWQ (2023)](https://arxiv.org/abs/2306.00978) | Activation-aware low-bit weight quantization, including four-bit inference. Establishes an important full-model compression baseline. | Does not select a universal intelligent subset; accuracy and Metal execution depend on supported formats and kernels. |
| [PowerInfer (2023/2024)](https://arxiv.org/abs/2312.12456) | Uses activation locality and predictors to place frequently active neurons on GPU and compute others on CPU. | Benefits depend on activation sparsity; not a drop-in guarantee for Flash-Next. CPU/GPU placement on unified memory does not itself reduce total machine memory. |
| [LLM in a flash (ACL 2024)](https://aclanthology.org/2024.acl-long.678.pdf) | Stores weights in flash and selectively loads them into DRAM, exploiting sparsity and locality. Demonstrates inference beyond resident-memory capacity. | Retains access to the model rather than extracting a permanent subset; SSD transfer cost remains a central constraint. |
| [SGLang Flash-Next implementation](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx) | Documents file-backed NVMe offload of the large n-gram table while the main network stays resident. A concrete implementation reference. | Published measurements are configuration-specific, not Apple Silicon predictions. Pin code revisions and check startup, page-cache and memory behavior. |

The first two sources are the closest match to the original idea. The others provide alternatives and controls. This is a focused reading list, not an exhaustive survey; reported numbers above have not been reproduced in Tur.

## Experimental plan

### 1. Fix the question and establish references

- Record target hardware, usable total memory after OS headroom, SSD capacity/bandwidth, context length, concurrency and maximum acceptable time to first token and decode latency. Use 16/24/32 GB as illustrative budget tiers until hardware is chosen.
- Choose coding as the initial workload, with reasoning, prose, structured tool/JSON output and long-context retrieval as regression domains. Decide acceptable losses before selecting candidates; no numerical quality budget has been agreed yet.
- Pin source and reference revisions, tokenizer/template, generation settings, dataset versions and evaluation harness. Keep calibration, tuning and final tests disjoint.
- Obtain a full-model reference on suitable hardware, or save teacher logits/layer outputs/traces there. A hosted endpoint can provide behavioral comparisons but cannot replace internal routing and layer-output traces.

Deliverable: reproducible baseline manifest, memory accounting and frozen test split.

### 2. Establish correctness and the offload baseline

- Audit and implement only the missing Flash-Next architecture semantics. Compare intermediate outputs, routing and logits with an independent reference on fixtures and a real checkpoint before evaluating compression.
- Keep all layers and experts; first evaluate supported calibrated quantization and file-backed n-gram embedding lookup. An exact lookup offload should preserve values at the same storage precision; quantization changes are a separate experiment.
- If experts still exceed the budget, measure bounded on-demand expert loading with a resident cache. Cache misses fetch the original selected expert; they do not silently substitute another expert in this baseline.
- Measure cold and warm cache behavior. File mapping is not a resident-memory cap: include page cache, staging buffers and completed GPU work in measurements.

Deliverable: quality-preserving placement reference, with transfer bytes and latency.

### 3. Select a permanent subset of original experts

- Collect per-layer routing frequency, routing mass, expert-output contribution, activation outliers and teacher-output reconstruction error on calibration data. Start with frequency as a cheap baseline, not the final importance definition.
- Use ablations and critical-expert checks to protect disproportionately important experts. Evaluate subsets jointly: individually small contributions can interact.
- Initially retain all attention/recurrent layers, shared experts, routers and required residual machinery; vary routed-expert retention at 75%, 50% and 25%. These are experiment points, not promised quality or fit levels. Allocate experts nonuniformly across layers if sensitivity measurements justify it.
- Explicitly define routing after removal: mask deleted expert logits before top-k, preserve at least the original top-k count of candidates, and apply the checkpoint's normalization rules. Simply dropping selected outputs implements a different model.
- Compare pure selection with a later optional recovery/distillation pass. Record preparation cost, teacher requirements and quality gains separately.
- Combine the best subset with quantization/offload only after isolated ablations. Layer removal or smaller replacement blocks are follow-ups if expert pruning fails the budget; gated residuals and hybrid state make arbitrary layer skipping nontrivial.

Deliverable: subset manifests mapping retained IDs to original experts, routing policy, precision and provenance, plus a measured quality/memory/latency frontier.

### 4. Compare against complete smaller models

| Candidate | Purpose |
| --- | --- |
| Full original model on reference hardware | Quality ceiling and teacher; not a local fit claim |
| Full model with quantization and bounded offload | Preserve all components; establish latency cost |
| Expert-pruned model, before and after optional recovery | Test whether a useful original subset survives |
| Complete smaller dense and MoE models that fit | Test the user's central claim at equal total memory |

Use the same workload, context, output limits and memory ceiling. Follow each model's documented prompt template; record thinking settings, output tokens and end-to-end time so longer reasoning is not mistaken for a free accuracy improvement.

Report coding task pass rates, reasoning accuracy, JSON/tool validity and retrieval accuracy, with paired uncertainty estimates where practical. Use teacher-forced KL/perplexity as diagnostics, not substitutes for task results. Include long generated sequences to detect accumulating errors, repetition and collapse.

Report startup peak, steady and peak resident/physical footprint, Metal allocations, KV/recurrent state, staging and cache budgets; do not double-count shared host/device allocations on Apple Silicon. Include TTFT, decode tokens/s, p50/p95 token latency, SSD bytes per token and expert-cache hit rate under cold/warm and sustained workloads. Batch size matters: prefill and concurrent requests may collectively touch many experts.

Decision: proceed only if a candidate fits the declared budget without sustained swap and improves the chosen quality/latency tradeoff over a fitting smaller model. Preserve negative results if selection collapses quality or SSD transfers dominate latency.

## Fit with existing Tur work

- [Metal kernels](../architecture/kernels.md), [Prepacked weights](../guide/quantization.md) and [State memory](../architecture/memory.md): packed experts and grouped Metal dispatch (full 35B quality gates remain open), calibrated quantization, sensitivity allocation, prepacked artifacts, and the single memory budget. Extend that accounting to resident/offloaded weights and staging rather than creating a competing budget system.
- Existing implementation starting points: `crates/tur-models/src/quantization/`, `crates/tur-models/src/models/artifact.rs`, `crates/tur-models/src/models/weights/`, `crates/tur-models/src/models/qwen3_5/model/moe.rs`, `crates/tur-metal/src/packed.rs` and `crates/tur/benches/perf/`.

Current bounded checkpoint loading reduces startup transients; it does not establish runtime expert eviction/offload. Existing calibration records per-expert inputs/counts; it does not yet supply all contribution, router-mass or subset-ablation measurements. Reuse existing components where compatible, with explicit Flash-Next validation. Follow AGENTS.md: no unsafe code, new dependencies/crates or unconfirmed public API changes; preserve current error and concurrency patterns.

## Resume checklist

- [ ] Select hardware, workload, quality-loss allowance and latency ceiling.
- [ ] Read expert pruning and super-expert papers first; inspect their code/protocols.
- [ ] Pin Flash-Next checkpoint/config/reference code and inventory actual tensor bytes.
- [ ] Estimate memory after table offload; establish access to a larger reference machine.
- [ ] Audit architecture gaps against current Tur and finish prerequisite correctness gates.
- [ ] Freeze baselines, data splits and metrics before choosing a retained subset.
- [ ] Run offload, pruning and recovery ablations; publish manifests and raw results.

Open questions: How much of coding/reasoning survives within the actual budget? Which experts are indispensable in this architecture? How stable is selection under domain shift? Can SSD caching meet interactive latency? Does recovery justify its preparation cost? Can the result beat a modern smaller model rather than only an older baseline?
