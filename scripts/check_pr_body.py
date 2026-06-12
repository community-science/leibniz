#!/usr/bin/env python
"""Validate that a pull request body follows the repository template."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_heading_pattern = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class TemplateSection:
    title: str
    body: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=".github/pull_request_template.md", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body-file", type=Path)
    source.add_argument("--event-path", type=Path)
    args = parser.parse_args(argv)

    template = _sections(args.template.read_text(encoding="utf-8"))
    body = _body_from_args(body_file=args.body_file, event_path=args.event_path)
    errors = validate_pr_body(template=template, body=body)
    if errors:
        for error in errors:
            print(f"PR body error: {error}", file=sys.stderr)
        return 1
    return 0


def validate_pr_body(*, template: tuple[TemplateSection, ...], body: str) -> list[str]:
    """Return validation errors for a PR body.

    Rules:

    - Every template section heading appears exactly once.
    - Template headings appear in template order; extra ``##`` sections are
      allowed anywhere before the template's final section.
    - The template's final section (Contribution Terms) is also the body's
      final section and matches the template text exactly.
    - Every section, required or extra, is non-empty; required sections must
      not contain only the template placeholder text.
    """

    if not body.strip():
        return ["body is empty"]
    body_sections = _sections(body)
    body_titles = [section.title for section in body_sections]
    template_titles = [section.title for section in template]
    if not body_sections:
        return [
            "body has no '## ' section headings; required sections: "
            + ", ".join(f"## {title}" for title in template_titles)
        ]
    errors: list[str] = []

    seen_titles: set[str] = set()
    duplicate_titles: list[str] = []
    for title in body_titles:
        if title in seen_titles and title not in duplicate_titles:
            duplicate_titles.append(title)
        seen_titles.add(title)
    errors.extend(
        f"section ## {title} appears more than once" for title in duplicate_titles
    )

    template_title_set = set(template_titles)
    required_by_casefold = {title.casefold(): title for title in template_titles}
    for title in dict.fromkeys(body_titles):
        if title in template_title_set:
            continue
        expected = required_by_casefold.get(title.casefold())
        if expected is not None:
            errors.append(
                f"section ## {title} does not match required section "
                f"## {expected}; headings are case-sensitive"
            )

    missing_titles = [title for title in template_titles if title not in seen_titles]
    errors.extend(f"missing required section ## {title}" for title in missing_titles)

    if not missing_titles and not duplicate_titles:
        ordered_required = [title for title in body_titles if title in template_title_set]
        for found, expected in zip(ordered_required, template_titles, strict=True):
            if found != expected:
                errors.append(
                    "required sections out of order: found "
                    f"## {found} where ## {expected} was expected"
                )
                break

    final_title = template_titles[-1]
    if final_title in seen_titles and body_titles[-1] != final_title:
        errors.append(
            f"## {final_title} must be the final section "
            f"(found ## {body_titles[-1]} after it)"
        )

    template_by_title = {section.title: section for section in template}
    validated_titles: set[str] = set()
    for section in body_sections:
        if section.title in validated_titles:
            continue
        validated_titles.add(section.title)
        content = _normalized_section_body(section.body)
        if not content:
            errors.append(f"section ## {section.title} must not be empty")
            continue
        template_section = template_by_title.get(section.title)
        if template_section is None:
            continue
        template_content = _normalized_section_body(template_section.body)
        if section.title == final_title:
            if _normalized_whitespace(content) != _normalized_whitespace(template_content):
                errors.append(
                    f"section ## {final_title} must match the template "
                    "contribution terms exactly"
                )
            continue
        if content == template_content:
            errors.append(
                f"section ## {section.title} still contains only template placeholder text"
            )
    return errors


def _body_from_args(*, body_file: Path | None, event_path: Path | None) -> str:
    if body_file is not None:
        return body_file.read_text(encoding="utf-8")
    if event_path is None:
        raise ValueError("event_path is required when body_file is absent")
    event = cast(dict[str, Any], json.loads(event_path.read_text(encoding="utf-8")))
    pull_request_value = event.get("pull_request")
    if not isinstance(pull_request_value, dict):
        raise ValueError("event payload does not contain a pull_request object")
    pull_request = cast(dict[str, object], pull_request_value)
    body = pull_request.get("body")
    return "" if body is None else str(body)


def _sections(markdown: str) -> tuple[TemplateSection, ...]:
    matches = tuple(_heading_pattern.finditer(markdown))
    sections: list[TemplateSection] = []
    for index, match in enumerate(matches):
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append(
            TemplateSection(
                title=match.group("title").strip(),
                body=markdown[start:stop].strip(),
            )
        )
    return tuple(sections)


def _normalized_section_body(body: str) -> str:
    return "\n".join(line.rstrip() for line in body.strip().splitlines()).strip()


def _normalized_whitespace(body: str) -> str:
    return " ".join(body.split())


if __name__ == "__main__":
    raise SystemExit(main())
