# Whole-subsystem audits

Use this only when the requested scope covers a whole Tur subsystem. The main [skill](../SKILL.md) supplies the value bar and candidate evidence.

## Baseline and inventory

Record the Git revision, pre-existing changes, in-scope test/support line counts and results before edits. List baseline failures separately from deletion candidates. Inventory colocated unit tests, integration modules, shared fixtures and ignored checkpoint cases owned by the subsystem. Include bench cases if support changes would affect them, without treating benchmark timing as correctness coverage.

Divide the inventory by production owner rather than directory alone: a prefix cache change may involve memory, engine contracts and pipeline integration. Every test declaration needs one ledger entry; split parameter rows only when they protect different contracts or need different decisions.

## Ledger and layer plan

Mark each entry retain, repair, consolidate or delete, with the evidence required by the main skill. Name the observable contract and credible failure for retained tests. A consolidation names its keeper and the assertions that must move there.

Review the ledger for redundant layers before editing. Name the primary keeper for each contract and the distinct risks covered at additional boundaries. Preserve the independence of scalar kernel references, fused/reference comparisons and external model goldens. Inventory test-only hooks that the plan would make obsolete.

Do not start cutover until all in-scope declarations have a disposition and every contract proposed for removal has either a keeper or evidence it is not a contract.

## Cutover and preservation

Apply one coherent owner batch at a time. Serialize edits to shared fixtures and integration module declarations. Move necessary assertions before deleting their former home; remove obsolete seams and support in the same batch.

Compare deleted assertions with the final keepers. Where a keeper's effectiveness is uncertain or a coverage gap was repaired, temporarily introduce the specific regression in an isolated copy and confirm the keeper fails for the intended reason. Restore the source exactly before validation. Do not mutate a user's pending changes or a checkout being tested. Report controls that could not run.

Reconcile concurrent source changes by finding a home for their newly added contracts instead of mechanically keeping deletions. Rerun affected proof after reconciliation and follow the main skill's final validation.

The handoff adds baseline versus final test/support counts, retired layers and keepers, preservation gaps and control results, and unresolved baseline failures. A broad audit does not authorize unrelated feature work or publishing.
