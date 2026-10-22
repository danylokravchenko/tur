# P3 Qwen3.5/3.8 correctness

Status (2026-10-22): the dense hybrid text model is implemented on the P2 packed
contract and agrees with Hugging Face Transformers on the real Qwen3.5-0.8B checkpoint,
on the CPU and Metal. Qwen3.8-27B uses the same architecture code path. Its config,
all 851 text-tensor names and shapes, its tokenizer differences and its chat template
are validated against the pinned checkpoint, but it has not been executed: its 4-bit
weights alone (~13.5 GB) exceed this Mac's 12.7 GB recommended GPU working set.
The deferred legacy cleanup and CLI/server migration are now complete. Both adapters
use `HybridEngine` through the packed hybrid text pipeline; ordinary Qwen3 and
Granite 4.1 and their execution/cache path have been deleted.

## Pinned references

| Item | Pin |
| --- | --- |
| Integration checkpoint | `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`; config and template byte-identical to `docs/references/p3/small-*` |
| Flagship checkpoint | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; config, template, generation and tokenizer configs in `docs/references/p3/qwen3.8-*`; tensor headers in `qwen3.8-text-tensors.json`, fetched with byte-range requests |
| Reference implementation | transformers 5.20.2 with torch on the CPU in float32. Its `modeling_qwen3_5.py` differs from the pinned copy only in cache plumbing (`has_previous_state`) |
| Golden generator | [`scripts/qwen3_5_reference.py`](../scripts/qwen3_5_reference.py) writes per-layer hidden states, selected full-logit rows, all-position argmax and a 48-token greedy continuation, plus template fixtures |

The 27B `tokenizer.json` matches the 0.8B one in vocabulary and merges. It adds only
the special tokens 248070–248076 (audio and TTS), so the template fixtures, which
were tokenized with the 0.8B tokenizer, hold for both checkpoints.

## Implementation

`tur_models::models::qwen3_5`:

- **Config.** Accepts only `qwen3_5` + `Qwen3_5ForConditionalGeneration` (nested
  `text_config`) or `qwen3_5_text` + `Qwen3_5ForCausalLM`, never a model name. It
  rejects:
  - quantization, rope scaling, MoE, sliding window and MLP-only layers;
  - attention bias, other activations or gates, and non-F32 SSM state;
  - invalid dimensions or head ratios;
  - non-default or non-text-equivalent M-RoPE.

  `generation_config.json` EOS IDs merge with the architecture's, and sampling
  defaults include top-k.
- **Checkpoint.** Every safetensors header is parsed without mmap or `unsafe`, and
  names, shapes and dtypes are checked against the config before any weight data is
  read. Unexpected text-tower tensors are rejected; vision and MTP tensors are ignored.
- **Math.** All of it runs in F32 and follows the reference formulas:
  - zero-centred RMSNorm (`1 + w`, folded in once at load), and the gated RMSNorm
    (`w`, then `· silu(z)`);
  - depthwise causal convolution over a (K−1)-input history;
  - L2-normalised and scaled Q/K, with `repeat_interleave` across value heads;
  - `g = −exp(A_log)·softplus(a + dt_bias)` and `β = σ(b)`;
  - the delta rule: chunked (UT transform, forward-substitution solve, 64-token
    chunks with partial final chunks) for multi-token prefill, recurrent for
    single-token decode and for per-token speculative boundaries, as in the reference;
  - gated GQA attention with per-head interleaved `q|gate`, head-dim Q/K norms, and
    split-half partial rotary over the first `head_dim × 0.25` dimensions in F32;
  - causal attention over committed pages plus new keys, in 256-query blocks.
- **Packed execution.** One embedding lookup, fused projections (`qkv|z|b|a`, `k|v`,
  `gate|up`) and one MLP serve every live sequence in a batch. Only the stateful core
  runs per sequence. Logits are computed only for the requested rows.
- **Errors.** Requests that hold image, video, vision or audio placeholders (from the
  config or the tokenizer's `extra_special_tokens`), or a mismatched state layout, fail
  on their own; device errors fail the batch.
- **Weights.** `WeightFormat::{F32, Q8_0, Q4K}` stores the projection matrices through
  candle `QMatMul` (groupwise 8-bit, 32-element blocks; 4-bit K-quants, 256-element
  super-blocks). Norms, gates, convolution and state stay F32. Quantized formats keep
  the embedding table in BF16. The weight format is part of `ModelIdentity::revision`.
- **Inspection.** `Qwen35::trace` returns the residual stream after each layer for
  reference comparison.

`tur_models::backend::chat_template::ChatTemplate::render` renders a checkpoint's own
template exactly as `apply_chat_template` does:
- Jinja2 `trim_blocks` and `lstrip_blocks`;
- Python `str` methods via minijinja's method callback;
- a `json.dumps`-compatible `tojson`;
- insertion-ordered maps;
- typed messages with `reasoning_content` and `tool_calls`.

`enable_thinking`, `reasoning_effort` and `preserve_thinking` stay undefined when
unset. That matters for Qwen3.8, which thinks by default and adds instructions for
the reasoning-effort levels xhigh, medium and low. No `/think` heuristic is added.
The legacy `format` path and its template preprocessing have been removed; all
serving prompts use `render`.

`tur_engine::hybrid::SamplingOptions` gained `top_k` (top-k, then top-p, as in candle),
which the Qwen3.8 generation defaults need.

## CLI/server migration and legacy removal

The CLI and HTTP worker load the concrete `Qwen35` model with `ModelFactory`, then
run it through `InferencePipeline` and the existing `HybridEngine`. Every iteration
packs one bounded prompt chunk or one decode token per admitted sequence into a
single `PackedExecution`. Intermediate prompt chunks request no logits. Live
requests own independent hybrid state and sampling; completed, failed and cancelled
requests release it. Startup loading is fallible and completes before HTTP binding.

Loading requires an explicit checkpoint identity; remote sources require a fully
qualified repository and immutable commit, with architecture validation before
weight downloads. Local SafeTensors directories work offline. Q8_0 (default) and
Q4K are explicit SafeTensors-to-packed conversions, not legacy GGUF imports. EOS
IDs merge architecture config, generation config and the tokenizer's declared chat
EOS. Sampling parameters inherit checkpoint defaults and HTTP overrides are honored.
Thinking controls remain optional, preserving undefined versus false.

HTTP conversations retain assistant reasoning and tool calls for exact checkpoint
rendering. Complete hybrid XML tool calls use the supplied schema to distinguish
string from JSON-valued arguments. Images, video and audio content parts are
rejected; modality placeholder tokens fail only their request. The worker uses a
bounded intake queue and live-request count and observes disconnected consumers.
The response stream remains unbounded until P7. There is one resident model; no
memory-budget or production concurrency claims are made.

Removed: ordinary Qwen3 and Granite implementations and aliases; `ModelImpl`,
`ModelInput`, `KvCacheImpl`, old K/V storage and memory-pool helpers; the ordinary
engine, scheduler, batch manager and prefix cache; old factory/weight builders and
model-name detection; legacy models' tests, benchmark, baseline runner and download/
serving scripts. Historical P0 reports and measured results remain. The reusable
projection enum, tokenizer streaming, tool definitions, faithful chat rendering and
hybrid grammar contracts remain in use. No third-party dependency/version was added.

Command examples and the intentionally changed flags are in the [README](../README.md).

## Tolerances and measurements

Relative error is the maximum absolute difference divided by the reference's maximum
magnitude. The reference prompt is the pinned template applied to a 162-token user
message, which spans three delta-rule chunks with a partial last one.

| Check | Device | Measured | Gate |
| --- | --- | --- | --- |
| Per-layer residual stream, all 24 layers (F32) | CPU | 2.50e-6 | < 1e-4 |
| | Metal | 2.51e-6 | < 1e-4 |
| Logits at 18 rows (F32) | CPU | 3.33e-6 | < 1e-4 |
| | Metal | 3.52e-6 | < 1e-4 |
| All-position argmax (F32) | CPU, Metal | 162/162 | exact |
| Greedy generation through `HybridEngine` (F32) | Metal | 31/31 tokens to `<\|im_end\|>` | exact |
| Template rendering, 17 cases over both templates | — | byte-identical | exact |
| Tokenization of every rendered case | — | identical IDs | exact |
| Chunked vs recurrent delta rule, n ∈ {1, 63, 64, 65, 150} | CPU | < 1e-4 | < 1e-4 |
| Whole vs 1/37/64/65-token chunked prefill, all positions (tiny model) | CPU | < 1e-4 | < 1e-4 |
| Q8_0 teacher-forced top-1 vs F32 | Metal | 0.994 | ≥ 0.95, set in advance |
| Q8_0 greedy continuation | Metal | identical 31/31, stops | stops |
| Q4K teacher-forced top-1 vs F32 | Metal | 0.821 | none set (see below) |
| Q4K greedy continuation | Metal | 27/31 identical prefix, same sentence, stops | stops; prefix ≥ half |
| Q8_0, tiny random model | CPU | top-1 39/40, relative logit error 4.8e-2 | top-1 ≥ 0.9 |

Q4K has no quality budget yet. A floor of 0.85, guessed before measuring, was missed.
Keeping the output projection at 8 bits did not change agreement (still 0.821), so
that change was reverted. Choosing a 4-bit recipe (mixed precision or importance
weighting) needs the held-out quality evaluation the release gates require; P3 claims
only coherent, terminating Q4K generation.

Offline fixtures also cover:
- requests joining and leaving mid-generation without changing other requests' tokens;
- speculative verification with corrupted drafts reproducing greedy decode;
- per-request rejection of vision placeholders;
- pinned config parsing and rejection of unsupported layouts.

## Validation

```sh
# Workspace and adapter fixtures
cargo nextest run --offline --workspace
cargo test --offline -p tur --test pipeline_integration --test hybrid_serving
# Model/reference fixtures
cargo test --offline -p tur-models --lib qwen3_5
cargo test --offline -p tur-models --test qwen3_5
cargo test --offline --release -p tur-engine --test qwen3_5
# Real checkpoint (TUR_DEVICE=metal for the GPU)
python -I scripts/qwen3_5_reference.py --model "$TUR_QWEN35_0_8B" --golden "$TUR_QWEN35_GOLDEN"
cargo test --offline --release -p tur-models --test qwen3_5 -- --ignored
cargo test --offline --release -p tur-engine --test qwen3_5 -- --ignored
```

Migration validation on 2026-10-22:

- `cargo nextest run --offline --workspace`: **102 passed**, four explicitly ignored
  real-checkpoint tests. The existing mathematical reference results above remain
  the P3 model oracle; the migration does not change that math.
- `cargo check --offline --workspace --all-targets` and the same command with
  `--no-default-features`: pass.
- `cargo clippy --offline --workspace --all-targets -- -D warnings`, formatting and
  diff checks: pass. The retained P0 trace-summary Python test passes.
- Adapter fixtures cover streamed/full text equality, whole/chunked prefill, both
  generation-config and tokenizer EOS, bounded admission, context and sampling
  validation, request join/cancellation/failure isolation, startup errors, HTTP
  sampling/thinking overrides and reasoning/tool history, and media rejection.
- The pinned real **Qwen3.5-0.8B / Q8_0 on Metal** completed through both binaries.
  CLI, concurrent non-streamed HTTP and streamed HTTP produced identical greedy
  text and stopped at EOS. The HTTP modes reported identical usage: 20 prompt and
  21 completion tokens. This used eight-token prefill chunks and two live HTTP
  requests, in a development build; it is a correctness smoke test, not a benchmark.
  [Run manifest](baselines/2026-10-22-m1-p3-serving/manifest.json),
  [HTTP response](baselines/2026-10-22-m1-p3-serving/response.json),
  [SSE output](baselines/2026-10-22-m1-p3-serving/stream.sse) and CLI/server logs
  preserve the evidence. Metal checks ran with GPU access outside the filesystem
  sandbox; the adapters now return a clear error if no Metal device is accessible.

The legacy checkpoint suite, including its historical guided-JSON failure, was
retired with the ordinary models. This is removal of an unsupported execution
path, not a claim that guided JSON serving has been fixed; hybrid grammar state
contracts retain their deterministic tests.

The checkpoint tests are `#[ignore]` and fail, rather than skip, without their
environment variables.

## Limits

- The retained `Execution::Reference` implementation is the CPU path and numerical
  oracle. Its costs include:
  - attention concatenates every committed page on each step;
  - the chunk solver and recurrence loop over candle ops;
  - activations are F32.

  Metal defaults to the fused kernels from [P4](apple-silicon-p4-metal.md), persistent
  page pools from [P5](apple-silicon-p5-memory.md), and [tiled attention
  prefill](apple-silicon-tiled-attention.md). The reference remains for differential
  tests.
- The loader keeps every raw text tensor in memory during load on both paths.
- Speculative per-token recurrent snapshots grow with draft length (P8).
- Image, video and audio inputs are rejected, not supported. MTP heads are not used.
- The CLI and server use the hybrid path. Production scheduling/admission (P6),
  prefix reuse and bounded streaming delivery (P7), and production serving
  hardening (P9) are not part of this migration.
