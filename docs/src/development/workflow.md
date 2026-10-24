# Workflow and rules

## Tasks

The `justfile` holds every routine task:

| Command | Does |
| --- | --- |
| `just check` | `cargo check` of every target, with and without Metal |
| `just lint` | Clippy over every target with `-D warnings` |
| `just fmt` / `just fmt-check` | rustfmt |
| `just test [args]` | The offline nextest suite; arguments go to nextest, e.g. `-E 'test(prefix)'` |
| `just test-checkpoint` | Ignored real-checkpoint tests (see [Testing](testing.md)) |
| `just ci` | `fmt-check`, `lint`, `check` and `test`: run before calling work done |
| `just build` | Release build of `tur` |
| `just run …` / `just serve …` | The CLI against the pinned development checkpoint |
| `just quantize <dir> <out> …` | `tur quantize` |
| `just bench <case>` | One [benchmark](benchmarks.md) case |
| `just docs` / `just docs-build` | Serve or build this book |

## Rules

Workspace lints in `Cargo.toml` and `clippy.toml` enforce these rules for every crate:

- **No unsafe code.** `unsafe_code` is forbidden. This also rules out NEON intrinsics;
  hot host loops are written so they auto-vectorize (`chunks_exact`, no bounds checks
  in inner loops).
- **No panics.** `unwrap`, `expect`, `panic!`, `unreachable!`, `todo!` and
  `process::exit` are denied. Errors propagate with `?` into `TurError`. Tests may assert.
- **No silently discarded results.** `let _ = …` on a `Result` is denied: handle it, or
  `drop(...)` it with a comment saying why nobody needs the outcome.
- **Every suppression explains itself.** `#[allow(...)]` requires `reason = "..."`.
- **Locks.** No lock is held across `.await`. Locks are `parking_lot`; std `Mutex` and
  `RwLock` are disallowed.
- **Modular structure.** Functions stay under 150 lines and cognitive complexity 20;
  split them by responsibility rather than allowing the lint. A module with children
  lives in `name/mod.rs`. `pub` means exported from the crate; otherwise use
  `pub(crate)` or `pub(super)`. A struct's fields are either all public or all private.
- **Clippy pedantic** is on, and lints also apply to public items
  (`avoid-breaking-exported-api = false`).
- **Dependencies.** New third-party crates need approval. Everything comes from the root
  `[workspace.dependencies]`.

## Principles

- **Breaking changes are allowed.** Rust, HTTP and CLI interfaces can break when a better
  design needs it (a project decision of 2026-10-21). Replaced code is deleted in the
  same change, with no compatibility shims or deprecated aliases.
- **Reuse before adding.** Extend existing types instead of creating parallel
  foundational structs, configs or error types.
- **Validate before the GPU.** Check shapes, offsets and page indices with checked
  arithmetic before submission. Release device resources only after the work that uses
  them completes.
- **Measure before claiming.** A new kernel or default must win at its shapes beyond
  run-to-run noise, on the target Mac. Otherwise the existing path stays. A rejected
  experiment gets a row in [Rejected experiments](../design/rejected.md), and the
  relevant chapter's "Decisions" section records what was kept.
- **External references decide correctness.** Do not mark a model supported because its
  config parses. It must match a pinned reference implementation.
- **Comments explain why.** Doc comments state what a public item does and the
  invariants it relies on. Inline comments explain thresholds, orderings and non-obvious
  invariants; they never restate the code.
