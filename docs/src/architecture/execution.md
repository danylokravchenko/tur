# Execution engine

Two contracts separate the model from the machinery that drives it. `HybridModel` (in `tur-models`) says how to run one batched pass, and `HybridEngine` (in `tur-engine`) owns requests, memory and in-flight GPU work. A new model family only has to implement the first.

## The model contract

`InferencePipeline` owns a `HybridEngine<TextModel>`, where the `TextModel` enum picks Qwen or Gemma at load time. Everything else, including the scheduler, memory pools, completion tracking, prefix snapshots and sampling, is shared.

```rust,ignore
pub trait HybridModel: Send + Sync + 'static {
    fn description(&self) -> ModelDescription;
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>>;
    // workspace_bytes, kernels and draft have defaults
}
```

`ModelDescription` tells the engine what to allocate: each layer's kind and state (attention K/V, or recurrent convolution and matrix state) with its dtype, the dimensions, vocabulary and position limit, and the input types the model accepts. Gemma declares only the layers that produce their own K/V, so its sharing layers cost no extra memory.

`PackedForward` is one pass over many requests, without padding:

- `tokens` and `positions` hold every request's new tokens back to back;
- each `SequenceForward` gives its slice of those arrays, its `SequenceState` (committed pages and recurrent slot, plus the ones reserved for this pass), the positions whose logits it needs, and whether a drafter needs its hidden states;
- `pool` holds the persistent state of every request.

The model returns one result per request: F32 logits for the rows asked for. It writes state in place into the reserved pages and slots. An error in one request fails only that request.

## The engine

`HybridEngine` is the single owner of a model's requests, state pool and in-flight work. Because one owner does everything, no locks are held during execution. At most one batch is in flight at a time:

1. `submit(plan)` checks a `PackedExecution` with checked arithmetic: token ranges, positions, state handles, page and slot indices, that no two requests write the same place, requested logits and draft rows. It reserves memory, encodes the pass, and hands its completion fence to a notifier thread.
2. `finish()` waits for the fence, then samples, verifies drafts and publishes each request independently, as a `SequenceResult` with its tokens or its own error.

Between the two calls, the owner is free to do CPU work such as accepting new requests, tokenizing and cancelling. The [pipeline](scheduling.md) builds each `PackedExecution` and drives these calls.

### State operations

A request's state is addressed by a `StateHandle` tagged with a generation number, so a stale handle can't reach memory that has since been reused. The engine can:

- `allocate` and `release_state`;
- `checkpoint` a request into an immutable `Checkpoint`, holding its model state, random-stream position, sampled tokens and grammar status at one token boundary;
- `fork` or `restore` from a checkpoint, which prefix reuse and preemption use;
- `adopt` an existing `ModelState` (a page table plus a recurrent slot);
- `cancel`.

A restore changes every layer kind at once, or nothing.

### Speculative transactions

A request can submit drafted tokens and ask for logits at every drafted position. The engine samples the model's own token at each position and accepts drafts while they match. At the first mismatch, it publishes the model's token and stops. If every draft matches, it also publishes the token after the last draft as a bonus. Recurrent layers keep a slot at each candidate position, so undoing rejected drafts means picking the right slot, with nothing to recompute. The random stream and penalty history advance only for published tokens. See [Speculative decoding](speculation.md).

## Knowing when the GPU is done

Candle commits a command buffer only every few dozen operations, or when something waits on it. If the engine's thread waited for a pass itself, the tail of the pass would sit uncommitted until the wait started. Instead, `Completion::notify` hands the fence to a separate `Notifier` thread right after encoding. That commits the tail at once, and the engine keeps working until it actually needs the result. The copy of the pass's outputs back to the CPU is queued last and doubles as the fence, so results come back without another trip to the GPU.

A `Completion` holds everything the pass uses (weights, every page and slot, logits) until the fence fires. If a fence fails, those resources are quarantined and never reused, because the GPU might still be touching them.

## Decisions

- One packed call per pass. `HybridModel::forward` takes the tokens, positions, ranges, state, requested logits and recurrent boundaries for the whole batch, with no per-request loop, so prompt chunks and single-token decodes share one path.
- Requests publish independently. A model, sampling, grammar or state error in one request never fails another; only a failed fence fails the whole pass.
- Handles carry generations. Stale and foreign handles fail to resolve, and memory is reclaimed only after every command that uses it completes.
- Recurrent state is double-buffered. Each pass writes new recurrent slots next to the committed ones, so rollback is exact and needs no copying, at the cost of two slots per running request.
- One pass in flight is enough. The next pass can't be chosen before this pass's tokens are known, encoding already overlaps the GPU, and the GPU sits idle only tens of microseconds between passes, so a deeper pipeline could gain at most 0.3% (see [Rejected experiments](../design/rejected.md)).
- Qwen and Gemma share one engine. The model changes the layer equations, while the checked execution and completion contracts stay common, and Gemma's fixtures and F32 chunking, batching and fork tests exercise the same contract.
