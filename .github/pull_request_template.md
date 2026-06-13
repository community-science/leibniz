<!--
Body conventions, enforced by the Template CI check
(scripts/check_pr_body.py):

- Every section below must be present, in this order, and filled in.
- "## Contribution Terms" must be the final section and match verbatim.
- Extra "## " sections are allowed anywhere before Contribution Terms;
  "### " subsections inside any section are always fine.
- Mark unresolved design choices explicitly (for example a
  "DECISION REQUIRED" subsection under Design Review).
- The body is re-checked on every push; keep its claims current as
  commits land.

Validate locally before opening or editing:

    python scripts/check_pr_body.py --body-file <your-body-file>
-->

## Purpose

What protocol or implementation capability does this add or change?

## Boundary

What contract does this pull request establish?

## Public Surface

List affected modules, schemas, commands, identifiers, generated files, durable
files, or output formats. State whether each change is additive, breaking, or a
pre-`1.0.0` redesign.

## Dependencies

List earlier pull requests, design decisions, data, or external assumptions this
relies on.

## Tests

What was tested, and what contract do those tests prove? List the local checks
run, or explain any skipped check; documentation-only changes may state that no
tests are needed.

## Rationale

Why should this exist in the repository now?

## Design Review

Which design choices were considered, kept, changed, or rejected?

## Contribution Terms

By submitting this pull request, I agree that, if accepted, my contribution will be released under the repository's CC0-1.0 public domain dedication.
