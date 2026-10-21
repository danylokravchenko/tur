# P2 ownership and hybrid state contracts

Status: implemented on 2026-10-21 and revised the same day after review. The revision
replaces the per-token `step` contract with packed execution, isolates per-request
failures, removes per-token snapshot retention, moves sampling behind the completion
fence, makes the engine decide speculative acceptance, and adds state dtypes, input
modalities and model workspace. Breaking changes are authorized (see
[legacy removal](apple-silicon-redesign.md#breaking-changes-and-legacy-removal)).
This is not Qwen3.8 support.

## Crate ownership

| Owner | Contents |
| --- | --- |
| `tur-core` | `TurError`/`Result` and checked execution descriptors (`execution`) |
| `tur-metal` | Immutable hybrid state and staged updates (`state`), completion fences and per-owner quarantine (`completion`); legacy KV cache and memory pool pending deletion |
| `tur-models` | `HybridModel` packed contract (`hybrid`); legacy models, loading and templates pending deletion |
| `tur-engine` | Single-owner `HybridEngine` (`hybrid`); legacy engine, scheduler and pipeline pending deletion |
| `tur` | Facade and HTTP library until the server adapter moves to `tur-server` |

The graph is `tur-core → tur-metal → tur-models → tur-engine → tur → adapters`.
No new configuration or error hierarchy exists; invalid descriptors and state
transitions use `TurError::InvalidExecution`. No third-party dependency was added.

The legacy `ModelImpl`/`KvCacheImpl` decoders, their re-exports and the CLI/server
paths built on them are not a compatibility commitment. P3 deletes them first and
rewires the adapters onto `HybridEngine`.

## Model description

`ModelDescription` declares identity (`revision`, `state_format`), per-layer
`LayerStateKind`, supported `modalities`, vocabulary and exclusive positional limit.
The revision must identify the exact checkpoint, quantization, adapters, positional
settings and isolation namespace. Attention layers declare KV heads, head dimension,
page token capacity and storage dtype; recurrent layers declare convolution
width/channels, recurrent heads/key/value dimensions and dtype. State dtypes are F32,
F16 or BF16. Packed execution carries text tokens, so a description must include
`Modality::Text`; other modalities are declared but not yet executable. Logits are F32.

## Packed model contract

```rust
pub trait HybridModel: Send + Sync + 'static {
    fn description(&self) -> ModelDescription;
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> { Ok(0) }
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>>;
}
pub struct PackedForward<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [usize],
    pub sequences: &'a [SequenceForward<'a>],
}
pub struct SequenceForward<'a> {
    pub tokens: Range<usize>,      // into the packed arrays
    pub state: &'a [LayerRead],    // committed state, shared storage, read-only
    pub logits: &'a [usize],       // packed positions needing logits
    pub boundaries: Boundaries,    // Final or EveryToken
}
pub struct SequenceStep { pub layers: Vec<LayerUpdate>, pub logits: Tensor }
```

One call covers every request in the batch with no padding, so a model can run
chunked prefill and batched decode as single passes. Per sequence it returns:

- attention: `[tokens, kv_heads, head_dim]` K and V for every new token;
- recurrent: `[b, conv_width, conv_channels]` history and `[b, heads, key, value]` matrix,
  where `b = 1` for `Boundaries::Final` and `b = tokens` for `Boundaries::EveryToken`;
- logits: F32 `[logits.len(), vocab]`, rows in output order.

The outer error fails the whole batch (one shared dispatch). Each inner result is one
sequence, so a model can isolate a per-request failure. `forward` should only enqueue
device work; the engine reads nothing until after the fence.

Inputs are zero-copy: `LayerRead` tensors share storage with live state and
checkpoints, and models must not mutate them in place. Outputs are validated for
kind, shape, dtype, device and position, then copied once into private storage
(`ModelState::stage`). Public inspection through `ModelState::read_layers` returns
detached copies.

## State storage and memory

`ModelState` is an immutable whole-model snapshot at one token boundary. Full pages
are shared through `Arc`. Appending fills the tail page and then opens new pages.
`ModelState::apply(staged, accepted)` builds the snapshot after the first `accepted`
tokens. Attention needs nothing extra for a prefix because pages are append-only.
Recurrent state selects the exact saved boundary and is never truncated like KV.
A partial prefix therefore requires `EveryToken`.

Ordinary sequences request `Final` and retain one recurrent boundary per layer.
Only speculative sequences request `EveryToken`. No snapshot is kept per token:
the base state and the staged outputs are held until the accepted boundary is
materialized once, at `finish`.

`ExecutionRequirements` (returned by `submit`) reports checked packed-input, logits
(requested outputs only), state-update and model-workspace bytes. `total_bytes()` is
the admission figure.

## Engine

```rust
HybridEngine::new(Arc<M>, Device) -> Result<HybridEngine<M>>;
allocate(&mut self, seed: u64, options: SamplingOptions,
    guidance: Option<GuidanceControl>) -> Result<StateHandle>;
metadata(&self, request: Uuid) -> Result<StateMetadata>;     // handle, position, finished
submit(&mut self, plan: PackedExecution) -> Result<ExecutionRequirements>;
finish(&mut self) -> Result<Vec<SequenceResult>>;
checkpoint / fork / restore / cancel / retry_quarantined
```

`PackedExecution { tokens, positions, sequences }` has one position per token.
`SequenceDescriptor { state, tokens, outputs, speculative }` names a request handle,
a nonempty contiguous range and strictly increasing output positions within it.
Ranges partition the input, and each request appears at most once. Positions must
continue from the committed boundary. Stale, cancelled and foreign handles are rejected.
Validation is all-or-nothing and happens before any model work. A speculative
sequence must request an output at every token.

`submit` validates, binds committed state, asks the model for workspace and enqueues
one packed pass. At most one batch is pending. `finish` waits on the fence, then for
each request in order:

1. If the model or output validation failed, return that request's error; its state is unchanged.
2. Otherwise read logits and sample each output in order, applying the grammar mask.
3. Select the accepted boundary, publish model state, RNG and grammar together, and
   bump the revision.

A sampling error part-way through (non-finite logits, a grammar that allows no
token) rolls back that request's RNG counter, sampled history and parser, using a
parser copy taken only when a grammar is active. Other requests still publish.
A fence error keeps the batch pending for retry. A whole-batch model error is
returned once and leaves every request unchanged.

A grammar reaching its terminal state is a normal stop, not an error: later outputs
are not sampled, `SequenceCommit::finished` and `StateMetadata::finished` are set,
and other requests in the batch are unaffected.

### Speculation

A speculative sequence is `[next token, draft_1, …, draft_m]`. The engine samples the
target at each position in order. It stops after the first output that disagrees
with the next draft, and also stops if the grammar finishes. Accepted inputs are the
outputs produced so far; the last output is the correction or bonus token. Sampling
from the target and accepting a draft only if it matches the sample keeps the target
distribution for deterministic proposals (n-gram, greedy draft). Probabilistic
draft-model correction is P8 work. The caller cannot choose an acceptance length,
so model, RNG and grammar state can never disagree with the published tokens.
Rejected drafts never reach the parser.

### RNG

RNG bookkeeping belongs to each request. Each draw derives a SplitMix64 seed from the
request seed and draw counter, then uses the existing Candle temperature/top-p sampler
on the host. Batch order does not enter the stream. Grammar commits and draws happen
only at requested outputs. Admission rejects grammars whose vocabulary differs from
the model's or that need multi-token splices or backtracking.

## Completion

`Completion<T>` owns every buffer referenced by enqueued work: weights, bound
inputs, staged outputs and logits. `Completion::wait` uses `Device::synchronize`,
which waits for all work submitted through that device handle and its clones.
An engine that needs independent completion must own its own `Device`. Extraction is
possible only after a successful wait. Drop also waits. If that wait fails, the
resources go into the owner's `Quarantine`, not a process-wide queue, so one engine's
device fault cannot block other engines. Publishing at `finish` enqueues
materialization work, so its buffers are retained by a trailing completion that the
next `submit` waits on.

## Limits

- Model inputs are shared, so read-only use of `LayerRead` is a contract, not enforced by types.
- Fence granularity is the device queue; per-batch events need APIs Candle does not expose safely.
- Pages are per-request `Arc` tensors, not a pooled physical page table (P5).
- Sampling and grammar run on the host after the fence (GPU sampling is P7).
- Per-boundary recurrent retention for speculation grows with draft length; bounded
  checkpoints or recompute come with P8.
- No Qwen hybrid model, custom kernel or model-quality claim is part of P2.

## Validation

Run from the workspace root with cached dependencies:

```sh
cargo test --offline -p tur-core -p tur-metal -p tur-engine --no-default-features
cargo test --offline -p tur --no-default-features --test server_integration
cargo clippy --offline -p tur-core -p tur-metal -p tur-engine --no-default-features --tests
```

| Selection | Result |
| --- | --- |
| `tur-core` unit tests | 4 passed |
| `tur-metal` unit tests (state staging, packed append, boundary selection, quarantine isolation) | 26 passed |
| `tur-engine` unit tests | 48 passed |
| Hybrid contract fixtures | 19 passed |
| HTTP/server integration | 14 passed |
| Clippy on new modules | No warnings |

The contract fixtures cover:

- packed prefill matching incremental decode;
- batch reordering;
- partial-page forks and atomic restore;
- final-only boundary retention and requirement accounting;
- engine-decided speculative acceptance, rejection and bonus tokens;
- per-request model, update-validation and grammar-terminal isolation;
- whole-batch failure;
- cancellation, stale and foreign handles, and invalid descriptors;
- overflow and positional limits.

New modules forbid unsafe Rust and contain no `unwrap`, `expect` or panics. No
checkpoint, GPU numerical run or benchmark was performed.
