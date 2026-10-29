# Code changes and diagnosis

Adapted from `bob-rs/.bob/rules/02-coding.md`; Tur's root rules take precedence.

- Search for an existing helper or type before adding one. Put behavior in the lowest crate that owns it, respecting the dependency order in `AGENTS.md`.
- Delete code and support that the change makes obsolete. Confirm all callers, including tests, feature branches, Metal entry points and runtime kernel names, before deciding something is unused. Report unrelated dead code separately.
- Do not hide unused code by widening visibility, renaming it with an underscore, or suppressing the warning. Split oversized functions by responsibility rather than relaxing lints.
- Use existing checked descriptors and model configuration. Prefer enums and validated values when they express an invariant; avoid parallel wrappers or foundational types that duplicate them.
- Borrow paths, strings and slices when ownership is unnecessary. Keep public visibility deliberate; do not add exports solely to make a test reach an owner.
- Propagate through the existing `TurError` and `Result`. Preserve context and causes, match typed errors where available, and handle each error once. Do not add a per-crate error hierarchy or match message text as a substitute for variants.
- Tur forbids `unsafe`, `unwrap`, `expect` and explicit panics, including in tests. Tests return `Result<()>`; assertion macros remain appropriate for test evidence. An invariant is not an exception to these rules.

For a bug, establish the specific failure before repairing its owner. Use the smallest existing test, CLI invocation or replay that exercises the symptom; distinguish a Metal-access or fixture failure from the reported defect. Change one relevant variable at a time and remove temporary instrumentation afterward.

Do not silently work around nearby races, missing validation or contradictory requirements. Fix those necessary for the task and report unrelated findings with locations, impact and a recommended follow-up. Explain any unverified assumption.
