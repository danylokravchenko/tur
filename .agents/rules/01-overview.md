# Task scope and completion

Adapted from `bob-rs/.bob/rules/01-overview.md`. Root `AGENTS.md` defines Tur's
architecture, commands and engineering requirements.

- Read the relevant owner, callers and book chapter before editing. Use the
  chapter's Decisions and the rejected experiments to avoid repeating settled work.
- Inspect the working tree first. Preserve existing user edits and distinguish
  their failures from failures introduced by the task.
- Complete the requested outcome with the smallest coherent change. Keep unrelated
  refactors, new options, dependency updates and speculative fallbacks out of it.
- Continue authorized, reversible work without repeated confirmation. Resolve
  routine implementation choices from the code and existing project decisions.
  Ask when a missing requirement prevents a correct solution or an action needs
  authorization; keep progressing on independent work where possible.
- Read linked rules when their subject applies. Do not treat a test-authoring task
  as authorization to sweep or prune the wider test suite.

Before reporting completion, review the final diff, run `git diff --check` and
`just ci`, and build the book with `just docs-build` when changing it. Run focused
checks during development; repeat broad checks only after further changes or when
a failure requires it. Follow any explicit user restriction on execution and
report checks it prevents.

Report the resulting behavior, checks actually run and their outcomes, and any
material gaps. Name affected files for unresolved bugs or assumptions. A command
that could not run is missing evidence, not a pass.
