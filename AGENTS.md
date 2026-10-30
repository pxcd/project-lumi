# Project Lumi engineering contract

- Read `.opencode/WORKFLOW.md`, `CONTRIBUTING.md` and the task's original requirements.
- Native role definitions live in `.opencode/agents/*.md`; shared configuration is
  `.opencode/opencode.json`. Arbiter coordinates; SCV implements behavior; probe
  handles presentation only; fresh independent Overlord verifies and never repairs.
- Confirm the absolute repository, branch and current changes. Preserve useful work;
  one implementation writer per checkout and sequential local inference.
- Use the existing project-local Python 3.12 environment. Run the narrowest meaningful
  check first; independent QA executes the relevant existing suites on final source.
- Keep v1.0 and v1.1 tokenizers, configurations and weights correctly paired. Preserve
  seeds, splits, corpus identity, checkpoint compatibility and historical evidence.
- Saved results and model-quality claims need immutable provenance. Never rewrite
  historical outputs, invent approval, or treat a skipped check as a passing check.
- A Git clone excludes weights and training corpora. Do not download data/models,
  install dependencies, train, overwrite artifacts or publish without authorization.
- Cloud source review is authorized for this workflow; do not send credentials,
  training corpora, model weights or private data to cloud context.
- Keep machine-specific provider endpoints and authentication out of Git. Use
  `.opencode/evidence/<task>/` for local task notes, logs and checkpoints.
