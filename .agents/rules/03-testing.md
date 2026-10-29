# Test evidence and failures

Adapted from `bob-rs/.bob/rules/03-testing.md`. Tur's test layout and commands are documented in [Testing](../../docs/src/development/testing.md).

- Protect observable behavior and meaningful invariants at their owning boundary. Use the [test-audit authoring gate](../skills/test-audit/SKILL.md) for new or changed tests. Extend existing coverage when it can catch the regression; avoid adding a test for a behavior-neutral edit or repeating the same proof at layers that add no distinct risk.
- For a regression, demonstrate failure before the fix and success after it when the baseline is available. Record an unavailable baseline explicitly. Do not substitute a mock's behavior for the owner's behavior.
- Reuse `common.rs`, `model_common.rs` and pinned fixtures. Follow Tur's `test_<behavior>` naming and `Result<()>` conventions, and keep integration tests in the existing `integration` binary.
- Keep offline tests independent of network, real user data and checkpoint downloads. Use isolated temporary directories and deterministic inputs. Prefer completion fences, channels or observable events over sleeps for ordering.
- Avoid dependence on execution order, ambient environment or map iteration. Statistical sampling tests require an appropriate statistical criterion rather than a hard-coded generated sequence; numerical tests need justified tolerances.
- Direct Metal access is required. Missing Metal must fail, not silently skip or switch to CPU execution. CPU references establish numerical expectations. Real-checkpoint cases remain `#[ignore]` and use the documented environment and pinned golden artifacts through `just test-checkpoint`.

When a test fails, inspect its actual assertion and inputs first. Repair the owner, or correct a demonstrably wrong expectation and explain why. Do not delete, ignore, weaken a tolerance or add retries just to obtain a pass. Diagnose flakes as races, resource interference or statistical failures before changing timeouts or bounds.

Assert the promised output or state, including relevant error and cancellation paths. `is_ok()`, a substring or a mock call is insufficient when it can pass while that contract is broken. Retain narrow unit tests when broader tests cannot expose the invariant they protect.
