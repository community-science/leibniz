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
    """Return validation errors for a PR body."""

    errors: list[str] = []
    if not body.strip():
        return ["body is empty"]
    body_sections = _sections(body)
    body_titles = [section.title for section in body_sections]
    template_titles = [section.title for section in template]
    if body_titles[: len(template_titles)] != template_titles:
        errors.append(
            "section headings must begin with the pull request template headings in order: "
            + ", ".join(f"## {title}" for title in template_titles)
        )
        return errors

    body_by_title = {section.title: section for section in body_sections}
    template_by_title = {section.title: section for section in template}
    for title in template_titles:
        section = body_by_title[title]
        content = _normalized_section_body(section.body)
        if not content:
            errors.append(f"section ## {title} must not be empty")
            continue
        template_content = _normalized_section_body(template_by_title[title].body)
        if title == "Contribution Terms":
            if content != template_content:
                errors.append(
                    "section ## Contribution Terms must match the template "
                    "contribution terms exactly"
                )
            continue
        if content == template_content:
            errors.append(f"section ## {title} still contains only template placeholder text")
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


if __name__ == "__main__":
    raise SystemExit(main())
