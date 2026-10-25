# AGENTS.md

Guidance for AI coding agents working on Tur. Read it before changing code. Before
changing kernels, memory, scheduling, prefix reuse, sampling, speculation or weight
formats, read the matching chapter of the book in [docs/src](docs/src/SUMMARY.md). Its
"Decisions" section records what was chosen and measured, and
[docs/src/design/rejected.md](docs/src/design/rejected.md) lists what was rejected.

## Project

Tur is a Rust (edition 2024, Rust 1.99+) inference engine for Apple Silicon. It serves
Qwen3.5-family hybrid and Gemma 4 text models on shared Metal kernels over Candle,
with paged K/V and recurrent state pools, packed continuous batching, prefix reuse,
device sampling and exact speculative decoding. CPU execution exists only as a
numerical reference for tests.

## Layout

```text
crates/
  tur-core/    shared errors (TurError, Result) and checked packed-execution descriptors
  tur-metal/   state pools, completion fences, Metal runtime, MSL kernels, packed weights, sampler
  tur-models/  checkpoint loading and validation, Qwen and Gemma models, chat templates, tool parsing
  tur-engine/  HybridEngine, scheduler, prefix cache, speculation, pipeline, memory plan
  tur/         facade re-exports and the HTTP server library
    tests/integration/   every crate's integration tests, one binary
    tests/fixtures/      pinned configs, templates, tensor headers, golden outputs
    benches/perf/        every benchmark and measurement tool, one binary
  tur-cli/     the `tur` binary: run, serve, quantize
docs/                    mdBook (`just docs`): guide, architecture with decisions, development, design
scripts/                 Python generators for golden references and fixtures
```

Dependencies point down the list (core ← metal ← models ← engine ← tur ← cli). Do not
add crates, cycles, or a second error or `Config` hierarchy. Model-specific
configuration stays with its model.

## Commands

Use the `justfile`:

```sh
just check            # cargo check, with and without Metal
just lint             # clippy --all-targets -D warnings
just fmt              # rustfmt
just test             # offline nextest suite (~30 s); `just test -E 'test(prefix)'` narrows it
just ci               # fmt-check + lint + check + test: run before declaring work done
just run --prompt 'Hello' -n 64          # pinned 0.8B checkpoint unless TUR_MODEL/TUR_REVISION
just serve --listen 127.0.0.1:8080
just bench <case>     # kernels | model | serving | prefix | kv_quality | tensorops | load_weights
just test-checkpoint  # ignored real-checkpoint tests; needs TUR_QWEN35_0_8B and golden files
```

Tests need direct Metal access and fail, rather than skip, without it. Test builds
optimize dependencies, `tur-metal` and `tur-models` (see `Cargo.toml`), so the first
build is slow and later ones are fast.

## Rules

These are enforced by workspace lints in `Cargo.toml` and are not negotiable:

- No `unsafe` (forbidden). This also rules out NEON intrinsics: write hot host loops so
  they auto-vectorize (`chunks_exact`, no bounds checks in inner loops).
- No `unwrap()`, `expect()`, `panic!()`, `unreachable!()` or `std::process::exit`.
  Propagate with `?` into `TurError`.
- No locks held across `.await`, and locks are `parking_lot` (std `Mutex`/`RwLock` are
  disallowed in `clippy.toml`).
- No `let _ = …` on a `Result`: handle it, or `drop(...)` it with a comment saying why
  nobody needs the outcome.
- Every `#[allow(...)]` carries a `reason = "..."`.
- Functions stay under 150 lines and cognitive complexity 20: split by responsibility
  rather than allowing the lint. Modules with children live in `name/mod.rs`. `pub` means
  exported from the crate; use `pub(crate)` or `pub(super)` otherwise.
- No new third-party dependencies unless they are already in the root
  `[workspace.dependencies]` or the user approves them.
- Clippy pedantic is on; `just lint` must pass with `-D warnings`.

Engineering expectations:

- Read the surrounding code and follow its patterns before adding anything. Reuse
  existing types; do not create parallel foundational structs, traits or modules.
- Breaking Rust, HTTP and CLI changes are allowed (user decision, 2026-10-21). Delete
  replaced code in the same change; add no compatibility shims or deprecated aliases. Do
  confirm with the user before changing public APIs or shared types (`TurError`,
  descriptors) that they did not ask about.
- Validate shapes, offsets and page indices with checked arithmetic before submitting
  GPU work. Release device resources only after the commands that use them complete.
- Prefer one execution owner with bounded messages over adding locks. Avoid needless
  clones; cloning an `Arc` is fine but should be deliberate.
- Performance claims need measurements on the target Mac, through `just bench`.
  Default to the existing kernel unless a new one wins at its shapes beyond run-to-run
  noise. A rejected experiment gets a row in `docs/src/design/rejected.md`.
- Model correctness comes from pinned external references (Transformers, MLX-LM). Do not
  mark a model supported because its config parses.

## Tests

- Unit tests go next to the code in `#[cfg(test)]` modules.
- Integration tests are modules of `crates/tur/tests/integration/main.rs`; add a module
  rather than a new test target. `common.rs` holds pipeline and serving fixtures, and
  `model_common.rs` holds model and engine fixtures.
- Name tests `test_<behavior>`. Tests return `Result<()>` and propagate setup errors with
  `?`; `assert!` is allowed in tests.
- Put new benchmarks in `crates/tur/benches/perf/` as a case of the `perf` binary, not as
  a new target.
- Keep the offline suite fast. A test that needs a real checkpoint is `#[ignore]` and
  reads `TUR_QWEN35_0_8B`.

## Documentation

- Doc comments on public items say what the method does and any invariants it relies on.
- Comments explain why: a measured threshold, an ordering constraint, a non-obvious
  invariant. Do not restate the code, and do not cite phase numbers or report files.
- Keep the book current. A changed default, flag, endpoint or mechanism updates its
  chapter in `docs/src`, including the chapter's "Decisions" section. Keep decisions to
  the choice and its evidence, not narrative. `just docs-build` must pass.

## Agent workflows

Read the applicable rules before working; these adapt `bob-rs/.bob/rules/` to Tur:

- [Task scope and completion](.agents/rules/01-overview.md)
- [Code changes and diagnosis](.agents/rules/02-coding.md)
- [Test evidence and failures](.agents/rules/03-testing.md)
- [Performance, concurrency and resources](.agents/rules/04-performance-correctness.md)
- [Workspace and input safety](.agents/rules/05-safety.md)

For test authoring or a requested test audit, use
[test-audit](.agents/skills/test-audit/SKILL.md). Its authoring gate applies to tests
being written or changed; a broader audit runs only when requested.
