# Isolating concurrent implementation work

Use one branch/worktree per concurrently edited task, not one per conversational
agent. Sequential builder/QA sessions can share a frozen issue checkout. Start
with serial execution; worktrees are for demonstrated overlapping work.

Before creating a worktree, confirm the repository, base commit, branch, issue
owner, and current dirty work. Preserve a dirty checkpoint in its original
checkout; a new worktree contains committed source, not uncommitted edits. Do not
create a partial-progress commit without the user's authorization.

Prefer sibling worktrees so a nested checkout does not also load the parent's
project configuration/plugins. Example, from a clean Lumi repository:

```sh
git worktree add -b codex/issue-42-evaluation ../project-lumi-worktrees/issue-42-evaluation main
```

Use a unique branch and directory; never force, reset or reuse an occupied target.
Start OpenCode in the new directory and confirm `git rev-parse --show-toplevel`,
the current branch and base. Verify the native session's recorded directory;
a prompt that says "cd elsewhere" does not establish a separate workspace.
Repository-owned agents/workflow must be present at the selected base revision.

For multiple builders on the same issue, assign disjoint milestones, one owner
per worktree and an integration branch. The coordinator owns integration of
reviewed commits only after Git mutation is authorized; do not copy another
agent's dirty files blindly. New Overlord verifies the final integrated revision,
not merely each builder's branch. Freeze implementation during final QA.

OpenCode's native child calls may inherit the coordinator's workspace. If a child
cannot be launched and verified in a different worktree using supported native
controls, keep the task serial or start an explicitly authorized separate
coordinator session in that checkout. Never claim isolation from role prompts.

Worktrees isolate files/branches, not LM Studio, credentials, processes, network
or Git repository metadata. Keep local inference sequential on Synapse. Observe
the same permissions and training/public-download restrictions everywhere.

After acceptance and authorized integration, inspect `git worktree list` and
the worktree's status before cleanup. Preserve pending commits/evidence. Do not
force-remove dirty worktrees, delete their branches or prune recoverable work.
