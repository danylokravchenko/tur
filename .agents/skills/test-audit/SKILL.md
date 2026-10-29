---
name: test-audit
description: "Review new or changed Tur tests for independent behavioral coverage; audit existing tests and test-only production seams when requested. Preserve numerical, Metal lifecycle, and external-reference contracts."
license: MIT
---

# Tur test audit

Adapted from OpenClaw's [test-audit skill](https://github.com/openclaw/openclaw/blob/main/.agents/skills/test-audit/SKILL.md) and its [audit](https://github.com/openclaw/openclaw/blob/main/.agents/skills/test-audit/references/audit.md) and [campaign](https://github.com/openclaw/openclaw/blob/main/.agents/skills/test-audit/CAMPAIGN.md) guidance, reviewed on 2026-10-25. Upstream's notice is retained in [LICENSE](LICENSE). This version uses Tur's Rust, Metal and nextest workflow; it has no dependency on OpenClaw skills or tools.

Read the repository's `AGENTS.md` and [testing chapter](../../../docs/src/development/testing.md) first. Resolve repository paths below from the workspace root. For a subsystem audit, read its architecture chapter, Decisions and `docs/src/design/rejected.md` before judging coverage.

Apply the authoring gate to tests in the current change. Start an audit of existing tests only when requested; begin with read-only discovery. Optimize for confidence, not deleted lines. For a whole subsystem, also read [campaign.md](references/campaign.md).

## Authoring gate

Before adding or changing a test, identify:

1. The observable behavior, invariant or independent contract it protects.
2. A credible regression that the assertion would detect.
3. The gap in existing coverage. Another layer needs a distinct numerical, transport, scheduling or lifecycle risk; otherwise extend the existing owner.
4. Whether it requires an export, hook or wrapper unused by production. Prefer testing at the actual boundary or a colocated unit test over adding that seam.

If these answers are missing, investigate before adding a test. A behavior-neutral edit does not inherently need one. For a bug, show the test failing for the intended reason before the fix and passing afterward, or disclose unavailable baseline proof.

Look for expectations derived from the subject itself, assertions on mock-supplied behavior, source inventories, self-comparisons, private call shapes and duplicated contracts. Negative tests must reach the intended validation guard. A test name or successful execution alone does not establish what its assertions prove.

## Tur retention rules

Keep tests that independently protect a contract, including:

- Scalar F64 kernel oracles and pinned Transformers or MLX-LM outputs. Fused versus reference agreement and random tiny checkpoints cover different risks from external model conformance; one does not replace the other.
- Non-aligned shapes, page boundaries, checked offsets, supported K/V and packed weight formats, chunked prefill and long-context numerical behavior.
- Completion fences, resource ownership until GPU completion, failure cleanup, cancellation, concurrent request isolation and bounded resource use.
- Prefix fork/restore exactness, speculative acceptance and rejection, sampling distributions, admission and preemption. A matching greedy output alone may miss corrupted state or an incorrect distribution.
- Chat-template bytes, checkpoint validation, HTTP streaming, CLI output and configuration/default contracts at the boundary that owns them.

Classify each test by the failure it can detect, not its layer or execution time. A narrow unit test may be the only proof of a checked descriptor or error path. Static checks can be valuable independent guards. Do not remove a numerical test because both paths currently agree or loosen tolerances to make a baseline pass.

## Audit evidence and edits

Read each full candidate, its production owner, callers and callees, sibling implementations, overlapping coverage, feature gates and relevant Git history. Inspect dependency code or types when a claim rests on Candle or Metal behavior. Tests across all crates share `crates/tur/tests/integration/main.rs`; do not limit caller or coverage searches to the crate containing the implementation.

Record and report before editing:

| Field | Required evidence |
| --- | --- |
| Candidate | Exact test name and file; concrete failure its assertions detect |
| Owner | Production/support seam and all non-test callers |
| Remaining proof | Specific keeper test and why it catches the same failure, or why no contract exists |
| History | Why the test exists, including any regression it locked down |
| Proposed change | Retain, repair, consolidate or delete; obsolete support it would remove |
| Risk and proof | Coverage that might be lost and the focused validation command |

Missing evidence means retain or investigate. Discovery is read-only; reporting evidence does not require another approval when the user already authorized edits. Keep a read-only audit read-only when that is the requested scope.

Consolidate at the owner, carrying distinct assertions into the keeper before removing duplicates. Remove support made obsolete by the batch after searching all feature configurations and test users. Do not change an unrequested public API or shared type without the confirmation required by `AGENTS.md`. Leave unrelated product defects as named follow-ups; a baseline failure is not a deletion reason.

## Validation and handoff

- Record a baseline before audit edits. Do not edit files while tests or benchmarks are using the checkout. Preserve existing user changes throughout.
- Use `just test -E 'test(<owner>)'` or an appropriate nextest expression for the owner and sibling coverage; confirm the filter selected the intended tests.
- Run `just ci` after edits. It covers formatting, strict linting, both Metal feature configurations and the offline suite. Run `just docs-build` if book content changed. Review `git diff --check` and `git diff --numstat`.
- Tests require direct Metal access. Report an access failure as blocked proof; never skip it or substitute CPU execution for the GPU contract.
- Where coverage depends on a real checkpoint, run the relevant `just test-checkpoint` selection with `TUR_QWEN35_0_8B` and its documented pinned goldens. If unavailable, report the gap; offline random weights do not replace it.
- Use `just bench <case>` on the target Mac for performance claims. Do not claim a speedup from fewer test lines or replace numerical proof with timing.

Review removed coverage against the keepers before finishing. Report changed categories and production seams, valuable tests retained, exact checks and results, missing proof and follow-ups. Separate production, test/support and documentation line changes. Do not commit, push or publish without task authorization.
