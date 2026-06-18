---
name: spec-pr
description: Turn a settled design into the spec pull request an implementation is built against. Ground the design in the architecture, scope it, write it as an empty-commit PR with an ordered-commit body, match the template, and hand off. Does not merge.
---

# Writing the spec for an implementation

A repeatable scaffold for turning a settled design into the spec PR an implementation is built against. It encodes the procedure and the discipline that are easy to forget. It does **not** replace the design: the spec's value *is* the design reasoning, and a checklist can only remind you to reason from first principles, not do it for you. The design is the point — treat the steps below as the floor.

## 1. Ground the design in the architecture, and design from the ideal

Before drafting, re-read the living architecture (the console Architecture tab, `src/leibniz/console/web/src/ArchitecturePanel.tsx`). State which root, edge, or roadmap step the work advances — grounded / gap / contradiction, the same opening move as a review. Then design from what *should* exist, reasoning from domain knowledge, rather than by extending whatever is already in the repo. Existing code is a starting point to replace, not a constraint.

## 2. Settle scope and the genuine forks

- One coherent advance, ending in a state consistent with the architecture — the declared target state must not contradict it.
- Decouple a risky or unvalidated bet from an irreversible change rather than bundling them.
- Resolve real design forks explicitly before writing the body. Surface the decisions; do not bury them.

## 3. Write it as an empty-commit PR with an ordered-commit body

Open the PR on an empty commit. Describe the work as an ordered series of commits the implementation lands in sequence: each one independently green, later commits building on earlier ones, and the final commit folding results back into the architecture. Build in observability — for example, diagnostics — so any provisional decision parameter can be judged from data rather than from argument.

## 4. Match the template; write statelessly

- Follow the PR template: every section present and in order, with the contribution-terms section last and verbatim. Validate with `scripts/check_pr_body.py` before opening.
- Write the body statelessly — describe what the PR delivers and the commits that land it, not "this is a spec PR" or "the implementation instruction." That framing goes stale the moment implementation starts.
- Mark unresolved choices explicitly.

## 5. No hidden parameters; open, hand off, do not merge

Surface every surviving decision parameter honestly rather than presenting the work as parameter-free. Open the PR and hand off to the implementation. Do not merge.
