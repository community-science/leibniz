from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_repository_root = Path(__file__).parents[1]
_script_path = _repository_root / "scripts" / "check_pr_body.py"
_spec = importlib.util.spec_from_file_location("check_pr_body", _script_path)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["check_pr_body"] = _module
_spec.loader.exec_module(_module)


def test_pr_body_validator_accepts_completed_template_body() -> None:
    template = _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )

    errors = _module.validate_pr_body(template=template, body=_completed_body())

    assert errors == []


def test_pr_body_validator_rejects_missing_template_heading() -> None:
    template = _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )
    body = _completed_body().replace("## Boundary", "## Scope")

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == [
        "section headings must begin with the pull request template headings in order: "
        "## Purpose, ## Boundary, ## Public Surface, ## Dependencies, ## Tests, "
        "## Rationale, ## Design Review, ## Contribution Terms"
    ]


def test_pr_body_validator_rejects_unchanged_placeholder_text() -> None:
    template_text = (
        _repository_root / ".github" / "pull_request_template.md"
    ).read_text(encoding="utf-8")
    template = _module._sections(template_text)

    errors = _module.validate_pr_body(template=template, body=template_text)

    assert "section ## Purpose still contains only template placeholder text" in errors
    assert "section ## Tests still contains only template placeholder text" in errors


def test_pr_body_validator_requires_verbatim_contribution_terms() -> None:
    template = _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )
    body = _completed_body().replace("By submitting this pull request", "By opening this change")

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == [
        "section ## Contribution Terms must match the template contribution terms exactly"
    ]


def test_pr_body_validator_rejects_extra_contribution_terms_text() -> None:
    template = _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )
    body = _completed_body().replace(
        "public domain dedication.",
        "public domain dedication.\n\nAdditional contribution language.",
    )

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == [
        "section ## Contribution Terms must match the template contribution terms exactly"
    ]


def _completed_body() -> str:
    return """## Purpose

Add a template-driven pull request body check.

## Boundary

The check validates pull request metadata only.

## Public Surface

Adds one workflow and one validation script.

## Dependencies

Uses the checked-in pull request template.

## Tests

Unit tests cover accepted and rejected body shapes.

## Rationale

Keeping validation template-driven avoids policy drift.

## Design Review

Considered hard-coded headings and rejected them.

## Contribution Terms

By submitting this pull request, I agree that, if accepted, my contribution will
be released under the repository's CC0-1.0 public domain dedication.
"""
