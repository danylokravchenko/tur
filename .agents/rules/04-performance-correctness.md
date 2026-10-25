# Performance, concurrency and resources

Adapted from `bob-rs/.bob/rules/04-performance-correctness.md` for an inference
engine with asynchronous Metal execution.

- Measure the affected shapes on the target Mac with the existing `just bench`
  cases before claiming a speedup. Compare equivalent settings and inputs; keep
  correctness checks separate from timing. Record chosen mechanisms and rejected
  experiments in the book as required by `AGENTS.md`.
- Avoid blocking Tokio tasks with file I/O, process waits or heavy host computation.
  Use the existing execution owner or established blocking-work boundary. Do not
  move GPU ownership into new detached tasks to hide latency.
- Every spawned task has an owner and a shutdown path. Make cancellation safe at
  channel and `select!` boundaries: a request, response or resource already taken
  from a queue still needs an owner.
- Use bounded queues and explicit admission or backpressure. Respect existing
  request deadlines and cancellation paths. Avoid holding a lock across an await;
  prefer the current execution owner to additional shared locks.
- Bound request buffers, streaming output, scheduler state and caches through the
  existing budgets. New caches need evidence of benefit, a capacity rule and an
  invalidation rule. Do not add a second memory-accounting mechanism.
- Release CPU-only temporary data when no longer needed. Metal buffers, pages and
  recurrent state must remain owned until commands that use them complete. A
  cancelled request does not imply its GPU work has completed.
- In token and kernel-dispatch paths, avoid repeated allocation, parsing, tensor
  conversion, synchronization and device readback. Borrow or deliberately share
  large data; inspect clones introduced only to satisfy the borrow checker.
- Validate size arithmetic, conversions, page indices and byte ranges before GPU
  submission. Do not use saturation to turn an invalid descriptor into plausible
  work. Preserve byte versus element units and shape/layout invariants.
- Deterministic output must not depend on unordered collection traversal. Retain
  the established numerical and sampling contracts when changing execution order.

Rust, HTTP and CLI changes follow Tur's explicit breaking-change policy. Keep the existing requirement to confirm public API or shared-type changes that the user did not request.
