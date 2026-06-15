# Agent Instructions

Before editing this repository, read:

- `README.md`: project principles, development commands, validation guidance,
  repository layout, and result publication workflow.
- `CONTRIBUTING.md`: pull request rules, contribution terms, development
  environment, agent guidance, public surface, tests and CI, and generated file
  policy.

The Architecture tab of the console is the living design specification for how
the protocol measures scientific work. Ground each change in it before writing
code, and fold the change back into it. The `.claude/skills/` scaffolds
(`spec-pr`, `review-pr`) encode the author-then-review loop this repository
uses.

If you think a GPU may be available and you need to run benchmark training,
request escalation for the training command so device discovery can use the
host environment. Do not silently accept a CPU fallback when the task depends
on GPU-accelerated training behavior or timing.

`results/` is ignored local benchmark state and may be a symlink to an
external Hugging Face or Git result checkout. Do not delete it during cleanup;
inspect it first with `ls -la results` and, if it is a Git checkout,
`git -C results status`.
