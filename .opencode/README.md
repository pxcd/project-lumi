# Repository-local OpenCode setup

Open this checkout in OpenCode and start a fresh session with **Arbiter** selected.
Do not continue an old Build session and assume the default changes its role.

| File | Purpose |
| --- | --- |
| `agents/Arbiter.md` | Cloud GPT-6.1 Sol coordinator |
| `agents/SCV.md` | Local Swift Splash functional builder |
| `agents/probe.md` | Local Qwen3.6 35B A3B MoE presentation builder |
| `agents/Overlord.md` | Fresh independent local Swift Splash QA |
| `opencode.json` | Default agent, shared policy, maintenance models and plugins |
| `WORKFLOW.md` | Lumi requirements, invariants, recovery and verification |
| `WORKTREES.md` | Separate task checkouts when concurrency is justified |

Role definitions and compatibility hooks are ported from the reviewed Latch setup;
Lumi-specific policy stays in AGENTS.md/WORKFLOW.md. No duplicated global roles or
external orchestration framework is required. Compaction/summary use the configured
cloud model, titles use local MoE. Provider endpoints, model registrations, request
settings (including reasoning control), auth and model installation stay machine-local.
A fresh machine must configure those prerequisites before starting real workers.

Before trusting a clone, inspect agent permissions, shared settings and plugins.
From its root, `opencode debug agents --standalone` inspects native resolved roles
without launching inference. `node --test .opencode/plugins/step-limit-compat/normalize.test.mjs`
checks the copied compatibility transformations. These checks do not prove actual
provider hook execution, generation settings or independent repair/re-QA behavior.

Then give Arbiter a bounded requirement or `Work on issue #N`. Keep one coordinator
per issue, sequential local inference and one writer per checkout. Native workers
are not OS sandboxes. Logs/checkpoints go in ignored `evidence/`; do not publish
private data or machine configuration. No Swift/Latch app packaging gates apply.
