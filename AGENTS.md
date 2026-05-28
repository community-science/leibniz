# Agent Instructions

Before editing this repository, read:

- `README.md`: project principles, development commands, validation guidance,
  repository layout, and result publication workflow.
- `CONTRIBUTING.md`: pull request rules, contribution terms, development
  environment, agent guidance, public surface, tests and CI, and generated file
  policy.

If you think a GPU may be available and you need to run benchmark training,
request escalation for the training command so device discovery can use the
host environment. Do not silently accept a CPU fallback when the task depends
on GPU-accelerated training behavior or timing.
