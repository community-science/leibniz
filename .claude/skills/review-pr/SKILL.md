---
name: review-pr
description: Review a single round of an implementation pull request in the spec → implement → review workflow. Ground the change in the architecture, confirm prior findings are closed, run the standing checks, and post one consolidated review comment. Does not merge.
---

# Reviewing a round of an implementation PR

A repeatable scaffold for reviewing one round of an implementation against the spec it was written to. It encodes the procedure and the checks that are easy to forget. It does **not** replace judgment: the findings that matter come from reading the diff carefully, not from a checklist. Treat the steps below as the floor, not the ceiling.

## 1. Ground the change in the architecture

Before reading the diff, re-read the living architecture (the console Architecture tab, `src/leibniz/console/_web_src/src/ArchitecturePanel.tsx`). State which root, edge, or roadmap step the change advances, in the architecture's own vocabulary, and resolve to one of:

- **Grounded** — it advances a named node, edge, or roadmap step, or closes an open rung. Frame the review in that language.
- **Gap** — it needs a concept the architecture does not yet have. The change should propose the addition and land it in its fold-back.
- **Contradiction** — it fights the architecture. Surface that explicitly; either the design or the architecture has to change.

This is the cheapest insurance against re-deriving something already written, or quietly competing with an existing distinction.

## 2. Establish the baseline and read the diff

- Find the last reviewed commit (the previous round's tip). Fetch.
- Use `git log <baseline>..<branch>` and `git diff --stat <baseline>...<branch>` to scope the round, then read the diffs that matter.
- Do not run the full test suite to "verify" the work — assume it was run green. Review the diff instead.

## 3. Confirm prior-round closure

Every finding from the previous round must be addressed. New findings are actions to complete, not advisories — do not label anything "non-blocking." The only legitimate deferral is work explicitly scoped to a named future PR.

## 4. Standing checks

Run each. Each is a place a real defect has hidden behind a green suite.

- **Dead code** — code the change supersedes is deleted or relocated, not left orphaned (and not parked in shipped source marked "unused").
- **Real-path tests** — tests exercise the genuine path end to end, not only synthetic or planted inputs. Discriminating assertions pin absolute values where the quantity matters, not just sign, monotonicity, or "it ran."
- **No silent fallback** — no path swallows a shape or argument mismatch and degenerates while staying green. A mismatch the code cannot handle should raise, not be silently dropped.
- **Architecture consistency** — framed in the architecture's vocabulary; no re-invented taxonomy the architecture already factors differently.
- **Boundary placement** — reusable machinery lives at the right layer: not trapped in one consumer, not leaked into a layer meant to stay agnostic.
- **Owned representations** — the protocol owns its representations and semantics; it does not couple to a specific framework or implementation by name where a declared property or abstraction belongs.
- **Honest parameters** — no claim of "parameter-free" where decision parameters remain; surviving knobs are surfaced (for example, in diagnostics), not hidden.

## 5. Post one consolidated comment; do not merge

- Post a single review comment per round. If several drafts accumulate, consolidate them into one and remove the superseded comments so the author reads one coherent set.
- Lead with what is resolved, then the open items as concrete actions, then a short verdict.
- Do not merge. Take the change to merge-ready and hand off.
