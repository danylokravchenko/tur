# Execution engine

Two contracts separate the model from the machinery that drives it: `HybridModel`
(`tur-models`) and `HybridEngine` (`tur-engine`).

## The model contract

```rust,ignore
pub trait HybridModel: Send + Sync + 'static {
    fn description(&self) -> ModelDescription;
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>>;
    // workspace_bytes, kernels and draft have defaults
}
```

`ModelDescription` declares everything the engine needs:

- layer kinds and their state (attention K/V, or recurrent convolution and matrix
  state), each with its dtype;
- dimensions, vocabulary and position limit;
- the input modalities the model accepts.

`PackedForward` is one pass over many sequences, without padding:

- `tokens` and `positions` hold every sequence's new tokens back to back;
- each `SequenceForward` gives its range in those arrays, its `SequenceState` (committed
  pages and recurrent slot, plus the pages and slots reserved for this pass), the packed
  positions whose logits are needed, and whether to return hidden states for a drafter;
- `pool` is the persistent state of every sequence.

The model returns one result per sequence: F32 logits for the requested rows. State is
written in place into the reserved pages and slots. A model error for one sequence fails
only that sequence. Model inputs are shared rather than copied, and outputs are copied
once.

## The engine

`HybridEngine` is the single owner of a model's requests, state pool and in-flight work.
No locks are held during execution, and at most one batch is in flight:

1. **`submit(plan)`** validates a `PackedExecution` with checked arithmetic: token
   ranges, positions, state handles, page and slot indices, non-overlapping writes,
   requested logits and draft rows. It reserves pool memory, then encodes one pass and
   hands its fence to a notifier thread. It returns the pass's checked payload and
   workspace requirements.
2. **`finish()`** waits for the fence, then samples, verifies drafts and publishes each
   sequence independently: a `SequenceResult` with its tokens, or its own error.

Between the two calls the owner can do host work: intake, tokenization, cancellation.
The [pipeline](scheduling.md) builds `PackedExecution`s and drives these calls.

### State operations

A request's state is addressed by a generation-tagged `StateHandle`. The engine offers:

- `allocate` and `release_state`;
- `checkpoint`, which takes an immutable `Checkpoint`: model state, RNG position,
  sampled tokens and grammar status at one boundary;
- `fork` and `restore` from a checkpoint, used by prefix reuse and preemption;
- `adopt`, which takes over a `ModelState` (a page table plus recurrent slot);
- `cancel`.

Restoring is atomic across every layer kind.

### Speculative transactions

A sequence can submit drafted tokens with logits requested at every position. The engine
samples the target at each position and accepts drafts while they equal the target's own
samples. On the first mismatch it publishes the target's sample. If every draft matches,
it also publishes the last row's sample as a bonus token. Recurrent layers keep a slot at
each candidate boundary, so rollback is just selecting the accepted boundary. Rejected
slots and pages are released after completion, and RNG and penalty history advance only
for published samples. See [Speculative decoding](speculation.md).

## Completion

Candle commits a command buffer only every few dozen encoders, or when synchronized. If
the owner waited on the fence itself, the tail of each pass would wait for that
synchronization. `Notifier` instead runs the fence on its own thread: `Completion::notify`
hands it over right after encoding, which commits the tail at once, while the owner
keeps working until it consumes the completion event. The host copy of the pass's
outputs is queued after the work and doubles as its fence, so results return without
another round trip to the device.

Resources referenced by a pass (weights, every page and slot it touches, logits) are
held by a `Completion` until it is fenced. A failed fence quarantines them, so they are
never reused while possibly still referenced, and the quarantine is per engine.

## Decisions

- **One packed call per pass.** `HybridModel::forward` takes tokens, positions,
  per-sequence ranges, committed state, requested logits and recurrent boundaries for
  the whole batch. There is no per-request loop, so chunked prefill and batched decode
  share one path.
- **Independent publication.** One request's model, sampling, grammar or state error
  never fails another request; only a failed fence fails the whole pass.
- **Generation-tagged handles.** Stale and foreign handles fail to resolve, and state is
  reclaimed only after every command that references it completes.
- **Double-buffered recurrent state.** Each pass writes new recurrent slots beside the
  committed ones. Rollback is exact, with no truncation or copying, at the cost of two
  recurrent slots per running request.
- **In-flight depth stays one.** The next pass cannot be selected before this pass's
  tokens are known. Host encoding is 13–25% of a pass but already overlaps the GPU, and
  the device idles only 28–38 µs between passes, so a deeper pipeline is bounded at about
  1.003× (see [Rejected experiments](../design/rejected.md)).
