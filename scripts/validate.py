#!/usr/bin/env python
"""Run the routine validation checks that mirror pull-request CI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    """One routine validation command."""

    label: str
    argv: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One routine validation gate."""

    name: str
    label: str
    commands: tuple[ValidationCommand, ...]


def validation_commands(
    *,
    repository_root: Path,
    include_console: bool,
) -> tuple[ValidationCommand, ...]:
    """Return the routine validation command sequence."""

    return tuple(
        command
        for check in validation_checks(
            repository_root=repository_root,
            include_console=include_console,
        )
        for command in check.commands
    )


def validation_checks(
    *,
    repository_root: Path,
    include_console: bool,
) -> tuple[ValidationCheck, ...]:
    """Return the routine validation gates."""

    python = sys.executable
    commands = [
        ValidationCheck(
            name="tests",
            label="Python tests",
            commands=(
                ValidationCommand(
                    label="Python tests",
                    argv=(python, "-m", "pytest"),
                    cwd=repository_root,
                ),
            ),
        ),
        ValidationCheck(
            name="lint",
            label="Ruff lint",
            commands=(
                ValidationCommand(
                    label="Ruff lint",
                    argv=(python, "-m", "ruff", "check", "."),
                    cwd=repository_root,
                ),
            ),
        ),
        ValidationCheck(
            name="type",
            label="Pyright type check",
            commands=(
                ValidationCommand(
                    label="Pyright type check",
                    argv=(python, "-m", "pyright"),
                    cwd=repository_root,
                ),
            ),
        ),
        ValidationCheck(
            name="repository-policy",
            label="Repository policy",
            commands=(
                ValidationCommand(
                    label="Repository policy",
                    argv=(python, "-m", "leibniz._repository_policy", "."),
                    cwd=repository_root,
                ),
            ),
        ),
        ValidationCheck(
            name="package",
            label="Build and import package",
            commands=(
                ValidationCommand(
                    label="Build wheel and source distribution",
                    argv=(python, "-m", "build", "--no-isolation"),
                    cwd=repository_root,
                ),
                ValidationCommand(
                    label="Install built wheel",
                    argv=(python, "-m", "pip", "install", "--force-reinstall", "__wheel__"),
                    cwd=repository_root,
                ),
                ValidationCommand(
                    label="Import installed package",
                    argv=(python, "-c", "import leibniz; print(leibniz.__version__)"),
                    cwd=repository_root,
                ),
            ),
        ),
    ]
    if include_console:
        commands.append(
            ValidationCheck(
                name="console",
                label="Console build and browser tests",
                commands=(
                    ValidationCommand(
                        label="Console build and browser tests",
                        argv=("npm", "test"),
                        cwd=repository_root / "src" / "leibniz" / "console" / "_web_src",
                    ),
                ),
            )
        )
    return tuple(commands)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-console",
        action="store_true",
        help="skip the console npm test step and report it as intentionally omitted",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=tuple(
            check.name
            for check in validation_checks(repository_root=Path.cwd(), include_console=True)
        ),
        help="run only this named check; may be repeated",
    )
    args = parser.parse_args(argv)

    repository_root = Path(__file__).resolve().parents[1]
    checks = validation_checks(
        repository_root=repository_root,
        include_console=not args.skip_console,
    )
    if args.check is not None:
        requested = set(args.check)
        checks = tuple(check for check in checks if check.name in requested)
        missing = requested.difference(check.name for check in checks)
        if missing:
            print(
                "validation check unavailable: " + ", ".join(sorted(missing)),
                file=sys.stderr,
            )
            return 2
    for check in checks:
        print(f"==> {check.label}", flush=True)
        for command in check.commands:
            completed = _run_command(command, repository_root=repository_root)
            if completed != 0:
                print(
                    f"validation check failed: {command.label} exited with {completed}",
                    file=sys.stderr,
                )
                return completed
    if args.skip_console:
        print("console check skipped by --skip-console", file=sys.stderr)
    return 0


def _run_command(command: ValidationCommand, *, repository_root: Path) -> int:
    argv = _resolve_command_argv(command, repository_root=repository_root)
    print(f"    {' '.join(argv)}", flush=True)
    completed = subprocess.run(argv, cwd=command.cwd, check=False)
    return completed.returncode


def _resolve_command_argv(
    command: ValidationCommand,
    *,
    repository_root: Path,
) -> tuple[str, ...]:
    if "__wheel__" not in command.argv:
        return command.argv
    wheels = sorted(
        (repository_root / "dist").glob("*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not wheels:
        return command.argv
    wheel = str(wheels[-1])
    return tuple(wheel if value == "__wheel__" else value for value in command.argv)


if __name__ == "__main__":
    raise SystemExit(main())
