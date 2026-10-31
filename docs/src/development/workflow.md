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
| `just ci` | `fmt-check`, `lint`, `check` and `test`; run it before calling work done |
| `just ci-hosted` | The same checks with the reduced `ci` test profile that hosted CI uses (see [Testing](testing.md)) |
| `just build` | Release build of `tur` |
| `just install [args]` | Install `tur` with `install.sh`; arguments go to `cargo install` |
| `just run …` / `just serve …` | The CLI against the pinned development checkpoint |
| `just quantize <dir> <out> …` | `tur quantize` |
| `just bench <case>` | One [benchmark](benchmarks.md) case |
| `just docs` / `just docs-build` | Serve or build this book |

## Rules

Workspace lints in `Cargo.toml` and `clippy.toml` enforce these rules in every crate:

- No unsafe code. `unsafe_code` is forbidden, which also rules out NEON intrinsics, so hot CPU loops are written to auto-vectorize (`chunks_exact`, no bounds checks in inner loops).
- No panics. `unwrap`, `expect`, `panic!`, `unreachable!`, `todo!` and `process::exit` are denied, and errors propagate with `?` into `TurError`. Tests may assert.
- No silently discarded results. `let _ = …` on a `Result` is denied: handle it, or `drop(...)` it with a comment saying why nobody needs the outcome.
- Every suppression explains itself: `#[allow(...)]` needs `reason = "..."`.
- No lock is held across `.await`, and locks are `parking_lot`; std `Mutex` and `RwLock` are disallowed.
- Functions stay under 150 lines and cognitive complexity 20, so split them by responsibility rather than allowing the lint. A module with children lives in `name/mod.rs`. `pub` means exported from the crate; otherwise use `pub(crate)` or `pub(super)`. A struct's fields are either all public or all private.
- Clippy pedantic is on, and lints also apply to public items (`avoid-breaking-exported-api = false`).
- New third-party crates need approval, and every dependency comes from the root `[workspace.dependencies]`.

## Principles

- Breaking changes are allowed. Rust, HTTP and CLI interfaces can change when a better design needs it, and replaced code is deleted in the same change, with no compatibility shims or deprecated aliases.
- Reuse before adding. Extend existing types instead of creating parallel structs, configs or error types.
- Validate before the GPU. Check shapes, offsets and page indices with checked arithmetic before submission, and release GPU resources only after the work that uses them completes.
- Measure before claiming. A new kernel or default must win at its shapes beyond run-to-run noise on the target Mac, or the existing path stays. A rejected experiment gets a row in [Rejected experiments](../design/rejected.md), and the chapter's "Decisions" section records what was kept.
- External references decide correctness. A model isn't supported because its config parses; it must match a pinned reference implementation.
- Comments explain why. Doc comments state what a public item does and the invariants it relies on. Inline comments explain thresholds, orderings and non-obvious invariants, and never restate the code.

## Writing documentation

The book and the README are for people deciding whether Tur fits their needs and learning how it works. Explain what a mechanism does and why it helps before giving its numbers. Keep a number only when the reader needs it to decide something; leave history, such as earlier policies or dates of measurement rounds, in git. Follow the [deslop](https://github.com/danylokravchenko/tur/blob/main/.agents/skills/deslop/SKILL.md) guidance: plain statements, no em dashes or unicode arrows, and no bold labels at the start of list items.

Markdown files never break a line in the middle of a sentence. Each paragraph and each list item is one line, however long; let the editor wrap it. Hard breaks inside sentences have to be re-flowed after every edit, and they drift out of shape when nobody does.
