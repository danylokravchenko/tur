# P2 ownership and hybrid state contracts

Status: implemented on 2026-10-21; targeted validation is recorded below. The user
explicitly authorized completely breaking Rust changes on 2026-10-21 after the
initial request. No further approval
is required for this package. Wire behavior and existing decoder mathematics are
outside this contract change. This is not Qwen3.8 support.

## Extraction and compatibility

| Owner | Extracted implementation | Compatibility exports |
| --- | --- | --- |
| `tur-core` | Existing `TurError` and `Result`, unchanged variants and conversions | `tur::{TurError, Result}`, `tur::shared::{errors, TurError, Result}` |
| `tur-metal` | Existing KV caches, block allocator, memory pool; new immutable hybrid storage and completion ownership | `tur::models::kv_cache`, `tur::backend::memory_pool`, `tur::runtime` |
| `tur-models` | Models, layers, attention, weight loading, model factory, templates, tool types and loading progress | `tur::{models, weights, VarBuilderX, ModelFactory, AutoModelFactory, ProgressReporter}`, existing `tur::backend` module paths |
| `tur-engine` | Engine, pipeline, scheduler, batch manager, prefix cache, guidance, tokenizer/sampler and audio interface | `tur::backend`, existing root engine/pipeline exports; new `tur::hybrid` |
| `tur` | Compatibility facade and HTTP library pending later adapter migration | Existing CLI/server imports remain valid |

The graph is `tur-core → tur-metal → tur-models → tur-engine → tur → adapters`.
Higher layers also use core/runtime directly. Templates and progress move with the
model factory because existing constructors depend on them; moving them into the
engine would create an upward dependency. No new configuration or error hierarchy
is introduced. Existing dependency versions remain unchanged.

## Exact execution contracts

The legacy `ModelImpl` and `KvCacheImpl` signatures remain available, including:

```rust
fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor>;
fn forward_batch(&mut self, input: &Tensor, positions: &[usize],
    paged_caches: Option<&mut [Vec<PagedKvCache>]>) -> candle_core::Result<Tensor>;
fn get_kv_cache_state(&self) -> candle_core::Result<Vec<(Tensor, Tensor)>>;
fn set_kv_cache_state(&mut self, state: Vec<(Tensor, Tensor)>) -> candle_core::Result<()>;
```

These are compatibility APIs for ordinary attention decoders, not hybrid state.
Their constructors, cache ownership and sampling behavior are not silently adapted.
New models implement this explicit-state contract in `tur_models::hybrid`:

```rust
pub trait HybridModel: Send + Sync + 'static {
    fn description(&self) -> ModelDescription;
    fn step(&self, token: u32, position: usize, state: &[LayerRead]) -> Result<TokenStep>;
}
pub struct TokenStep { pub layers: Vec<LayerUpdate>, pub logits: Tensor }
```

`ModelDescription` declares identity (`revision`, `state_format`), per-layer
`LayerStateKind`, vocabulary and exclusive positional limit. The revision string
must identify the exact checkpoint, quantization, adapters, positional settings
and isolation namespace; a display model name is insufficient. Attention declares
KV heads, head dimension and page token capacity. Recurrent layers declare
convolution width/channels and recurrent heads/key/value dimensions. P2 supports
text tokens and F32 state/logits only. Model-specific `Config` stays with its model.

`ModelState::allocate(Arc<ModelDescription>, &Device) -> Result<ModelState>`
creates empty attention tables and zero convolution/recurrent tensors.
`ModelState::advance(&self, updates: &[LayerUpdate]) -> Result<ModelState>` checks
every layer's kind, shape, dtype, device and position before constructing a new
whole-model snapshot. Attention updates contain one `[kv_heads, head_dim]` K/V
pair; recurrent updates replace the entire `[conv_width, conv_channels]` history
and `[heads, key_dim, value_dim]` matrix. Recurrent history is never KV-truncated.

The engine owns one live request entry. Weights are held by `Arc<M>` and `step`
receives `&self`; implementations must keep weights immutable and cannot hide
sequence state. Internal snapshot buffers are private and never mutated. Full
attention pages share `Arc`s; appending replaces a partial tail. Model input and
output boundaries use detached tensor copies because Candle supports in-place
mutation through shared tensor references. Checkpoint inspection also returns
copies, so external writes cannot mutate a snapshot.

```rust
HybridEngine::new(Arc<M>, Device) -> Result<HybridEngine<M>>;
allocate(&mut self, seed: u64, options: SamplingOptions,
    guidance: Option<GuidanceControl>) -> Result<StateHandle>;
metadata(&self, request: Uuid) -> Result<StateMetadata>;
submit(&mut self, plan: PackedExecution) -> Result<()>;
finish(&mut self) -> Result<Vec<TokenOutput>>;
commit(&mut self, accepted: &[(Uuid, usize)]) -> Result<Vec<TokenOutput>>;
checkpoint(&self, handle: StateHandle) -> Result<Checkpoint>;
fork(&mut self, checkpoint: &Checkpoint) -> Result<StateHandle>;
restore(&mut self, handle: StateHandle, checkpoint: &Checkpoint) -> Result<StateHandle>;
cancel(&mut self, handle: StateHandle) -> Result<()>;
```

## Validation, completion and publication

`PackedExecution { tokens, positions, sequences }` has one position per packed
token. Each `SequenceDescriptor { state, tokens: Range<usize>, outputs }` names a
request UUID/revision, a nonempty contiguous range and optional output positions
within that range. Ranges partition the input in descriptor order; sequences may
be reordered by repacking. Every request appears at most once. Tokens, output
indices, dimensions, byte products, sequence ends and revisions are validated
with checked arithmetic before model execution. Positions must continue from the
current committed boundary. Handles from other engines, cancelled requests or
prior revisions are rejected. UUIDs are never intentionally reused.

`PackedExecution::validate(self, &ModelDescription, &[StateMetadata]) ->
Result<ValidatedExecution>` returns private immutable input. The engine validates
against its own current metadata at submission, rather than trusting caller-supplied
validation or an old plan. `ValidatedExecution::requirements()` records checked
packed input bytes, retained token logits and a conservative bound for retained
reference state tensors. Layer totals and batch totals are also checked. These
exclude backend temporary allocations and grammar/allocator overhead; they are not
admission budgets. Custom kernel/workspace selection is deferred because P2 does
not dispatch custom packed kernels.

At most one batch is pending. Submission owns weights, base state, exact per-token
boundaries, detached input bindings and model outputs through `Completion<Batch>`.
Live requests remain at the previous committed boundary. Model errors are saved
and reported by `finish`/`commit` after the fence; no partial request state is
published. Cancellation marks an in-flight request and suppresses its publication;
other requests may finish independently. Non-flight cancellation releases immediately.
Restore while that request is in flight is rejected. Checkpointing while in flight
observes the prior committed state.

`Completion::wait(&mut self) -> Result<()>` uses `Device::synchronize`; extraction
is possible only after success. Drop also waits. If drop cannot synchronize,
resources enter a process-wide quarantine and are retried at later submissions
or by `retry_retired()`. New engine submissions refuse a still-unresolved quarantine. This deliberately retains memory on persistent device
failure rather than recycling live bindings. Explicit waits retain their resources
on failure and can be retried. The failure queue is exceptional-path storage,
not a general resource pool. There is no new unsafe Rust.

## Checkpoints and speculation

A checkpoint contains every layer, position, exact model/layout/device identity,
seed and draw counter, sampling options, sampled history and a deep clone of the
existing llguidance parser/stopped state. Fork shares private immutable buffers;
parser branches are independent. Restore validates before publication and bumps
the request revision rather than restoring an old revision. Wrong-model/layout/
device restores leave the original untouched.

Every candidate token boundary has a complete snapshot. `commit` requires exactly
one accepted length per submitted request, from zero through submitted length.
Only the selected boundary and its outputs become visible, once. A zero-length
acceptance restores the base. Rejected model, convolution, RNG and grammar changes
are discarded together. Invalid acceptance descriptors retain the pending batch
for correction. Errors discard the entire candidate batch. A later speculation
algorithm can replace the retained intermediate snapshots with bounded recomputation;
P2 does not implement draft generation or target/proposal correction.

RNG bookkeeping is request-owned. Each draw derives a deterministic SplitMix64 seed
from the request seed/counter, then uses the existing Candle temperature/top-p
sampler. Only the mathematical seed mixer wraps; positions, sizes and counters
are checked. This intentionally defines a new stream, not bitwise compatibility
with the legacy long-lived Candle RNG. Batch ordering does not enter the stream.
Grammar commits and RNG draws occur only for requested output positions.
Admission validates grammar/model vocabulary equality and rejects parser capabilities
requiring multi-token splices/backtracking. This prevents masks or implicit grammar
transitions from escaping the declared token/state boundaries; the legacy guidance
path is unchanged. A terminal grammar refuses another output; EOS policy/stream delivery remains with later serving
integration. `GuidanceControl::fork(&self) -> Self` uses `Constraint::deep_clone`.
`GuidanceControl::validate_token_contract(&self, vocab_size: usize) -> Result<()>`
checks admission compatibility.

## Migration steps and limits

1. Existing adapters can keep importing `tur`; errors and compatibility types are
   re-exports of the same definitions, not wrapper types.
2. New model code implements `HybridModel`, declares exact state dimensions and
   returns complete per-layer token updates. It must not use legacy internal KV.
3. Callers allocate request handles, pack tokens/positions, submit and finish or
   accept a speculative prefix; use returned metadata after every commit/restore.
4. Serving migration must move grammar activation, output delivery and scheduler
   state onto these request handles. The current CLI/server still execute the
   extracted legacy engine and do not acquire hybrid guarantees by re-export.
5. Replace the reference token loop/copies and full per-token checkpoints only
   after P3 math and later runtime/scheduler validation. No Qwen hybrid model,
   custom kernel optimization or model-quality claim is part of P2.

Inherited unsafe/panic paths remain confined to extracted legacy code. P2 adds
safe modules and deterministic offline fixtures; it does not fix the retained
baseline guided-JSON failure. CPU fixtures validate ownership/state semantics,
not Metal numerical accuracy or production memory bounds. Persistent fence-error
quarantine and unbounded per-submission snapshot retention are explicit limitations.

## Targeted validation

Commands run from the workspace root, using only cached dependencies and small
fixtures. No checkpoint suite or benchmark suite was rerun for P2.

```sh
cargo test --offline -p tur-core -p tur-metal -p tur-models -p tur-engine \
  --no-default-features --lib
cargo test --locked --offline -p tur-engine --no-default-features --test hybrid_contracts
cargo test --locked --offline -p tur-metal --no-default-features --lib completion::tests
cargo test --locked --offline -p tur --no-default-features --test server_integration
cargo test --locked --offline -p tur-models --no-default-features --doc
cargo check --locked --offline --workspace --all-targets
cargo check --locked --offline --workspace --all-targets --no-default-features
cargo fmt --all -- --check
git diff --check
```

All listed checks pass:

| Selection | Result |
| --- | --- |
| Extracted library unit tests | 159 passed: core 2, Metal/storage 22, models 88, engine 47 |
| Hybrid contract integration fixtures | 15 passed; includes exact mixed-layer state values and real llguidance branches |
| Completion tests, targeted rerun | 5 passed, including injected failed fences and drop quarantine; included in the 22 above |
| Existing HTTP/server integration tests | 14 passed |
| Model factory doctest | 1 passed |
| Workspace all-targets checks | Passed with default Metal features and with no default features |
| Formatting and diff whitespace | Passed |
| Dependency audit | All third-party lockfile packages, versions and checksums unchanged |

The move preserves 26 source files byte-for-byte; the other moved files only
change module re-exports, factory documentation, or add parser fork/admission
methods. The original errors are moved unchanged. The facade's unused inference
dependencies were removed or moved to development dependencies. Root default
commands now select all seven workspace members, including relocated unit tests.

Existing deprecated constants in CPU attention and the upstream `block`
future-compatibility warning are inherited. New modules forbid unsafe Rust;
source inspection found no newly added unsafe blocks, unwrap/expect calls or
panic macros. P0's retained guided-JSON failure is unchanged and outside this
package. No checkpoint/GPU numerical run or benchmark was performed for P2.
