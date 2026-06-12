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
_contribution_terms = (
    "By submitting this pull request, I agree that, if accepted, my contribution will be "
    "released under the repository's CC0-1.0 public domain dedication."
)


def test_pr_body_validator_accepts_completed_template_body() -> None:
    template = _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )

    errors = _module.validate_pr_body(template=template, body=_completed_body())

    assert errors == []


def test_pr_body_validator_rejects_missing_template_heading() -> None:
    template = _template_sections()
    body = _completed_body().replace("## Boundary", "## Scope")

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == ["missing required section ## Boundary"]


def test_pr_body_validator_accepts_extra_sections_before_contribution_terms() -> None:
    template = _template_sections()
    body = _completed_body().replace(
        "## Public Surface",
        "## Implementation Plan\n\nStep one, then step two.\n\n## Public Surface",
    ).replace(
        "## Rationale",
        "## Validation Results\n\nAll gates pass on cuda.\n\n## Rationale",
    )

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == []


def test_pr_body_validator_rejects_sections_after_contribution_terms() -> None:
    template = _template_sections()
    body = _completed_body() + "\n## Attribution\n\nGenerated with assistance.\n"

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == [
        "## Contribution Terms must be the final section (found ## Attribution after it)"
    ]


def test_pr_body_validator_rejects_duplicate_required_heading() -> None:
    template = _template_sections()
    body = _completed_body().replace(
        "## Rationale",
        "## Tests\n\nA second tests section.\n\n## Rationale",
    )

    errors = _module.validate_pr_body(template=template, body=body)

    assert "section ## Tests appears more than once" in errors


def test_pr_body_validator_hints_on_case_mismatched_heading() -> None:
    template = _template_sections()
    body = _completed_body().replace("## Public Surface", "## Public surface")

    errors = _module.validate_pr_body(template=template, body=body)

    assert (
        "section ## Public surface does not match required section "
        "## Public Surface; headings are case-sensitive"
    ) in errors
    assert "missing required section ## Public Surface" in errors


def test_pr_body_validator_rejects_out_of_order_required_sections() -> None:
    template = _template_sections()
    body = _completed_body().replace(
        "## Dependencies\n\nUses the checked-in pull request template.\n\n## Tests\n\n"
        "Unit tests cover accepted and rejected body shapes.\n",
        "## Tests\n\nUnit tests cover accepted and rejected body shapes.\n\n"
        "## Dependencies\n\nUses the checked-in pull request template.\n",
    )

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == [
        "required sections out of order: found ## Tests where ## Dependencies was expected"
    ]


def test_pr_body_validator_rejects_empty_extra_section() -> None:
    template = _template_sections()
    body = _completed_body().replace(
        "## Rationale",
        "## Open Questions\n\n## Rationale",
    )

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == ["section ## Open Questions must not be empty"]


def _template_sections() -> tuple[object, ...]:
    return _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )


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


def test_pr_body_validator_accepts_wrapped_contribution_terms() -> None:
    template = _module._sections(
        (_repository_root / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
    )
    body = _completed_body().replace(
        "my contribution will be released",
        "my contribution will\nbe released",
    )

    errors = _module.validate_pr_body(template=template, body=body)

    assert errors == []


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
    return f"""## Purpose

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

{_contribution_terms}
"""
