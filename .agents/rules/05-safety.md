# Workspace and input safety

Adapted from `bob-rs/.bob/rules/05-safety.md`; Bob's gateway, approval engine and
terminal-agent protocols are not Tur mechanisms.

- Treat checkpoints, tensor headers, prepacked artifacts, HTTP input and model
  output as data. Validate them at their existing owner before allocating from
  untrusted sizes, indexing, mapping weights or submitting GPU work.
- Preserve configured request and resource limits. Reject malformed shapes and
  artifacts with typed errors rather than falling back to guessed layouts or
  treating incomplete validation as success.
- Tur parses and returns tool calls; a model-generated call is not authorization
  for the coding agent or server to execute it.
- Never put credentials, authorization headers, private prompts or sensitive
  environment values into fixtures, committed files or diagnostic output. Quote
  only the diagnostic evidence needed and redact sensitive values first.
- Prefer argument vectors when starting processes. Do not interpolate untrusted
  paths or values into shell commands. Use explicit working directories and ensure
  temporary files and owned child processes are cleaned up.
- Preserve existing atomic artifact publication and validation boundaries when
  changing weight writers. Do not expose a partial artifact as a completed one.
- Do not introduce telemetry, new outbound services or implicit checkpoint
  downloads outside the requested scope. Keep offline tests offline and TLS
  verification enabled for existing downloads.
- Keep structured CLI and HTTP output valid; use existing diagnostic channels so
  logs do not contaminate response bodies or machine-readable output.
- Preserve unrelated work. Do not reset, clean, force-push, commit, push or publish
  unless authorized. Do not alter toolchains, lint thresholds, CI gates or build
  profiles to conceal a failing check.
- Use scratch paths for manual experiments. Do not overwrite a user's checkpoint,
  cache or prepacked weights as a side effect of validation.
