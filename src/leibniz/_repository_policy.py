"""Repository policy checks for tracked files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_forbidden_names = frozenset(
    {
        ".leibniz",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vite",
        "__pycache__",
        "build",
        "docs",
        "dist",
        "venv",
    }
)
_forbidden_suffixes = (".pyc", ".pyo")
_forbidden_env_files = frozenset({".env"})
_benchmark_artifact_root = PurePosixPath("src/leibniz/benchmarks")


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    """A tracked path that violates a repository policy."""

    path: PurePosixPath
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


class RepositoryPolicy:
    """Repository-level checks for tracked paths."""

    @staticmethod
    def validate_tracked_paths(paths: Iterable[str]) -> tuple[PolicyViolation, ...]:
        return _validate_tracked_paths(paths)


def _validate_tracked_paths(paths: Iterable[str]) -> tuple[PolicyViolation, ...]:
    violations: list[PolicyViolation] = []
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        if _has_forbidden_name(path):
            violations.append(
                PolicyViolation(
                    path=path,
                    message="tracked local, cache, or generated directory",
                )
            )
            continue
        if path.suffix in _forbidden_suffixes:
            violations.append(
                PolicyViolation(
                    path=path,
                    message="tracked Python bytecode output",
                )
            )
            continue
        if path.name in _forbidden_env_files or path.name.startswith(".env."):
            violations.append(
                PolicyViolation(
                    path=path,
                    message="tracked local environment file",
                )
            )
            continue
        if _is_benchmark_interpreter_file(path):
            violations.append(
                PolicyViolation(
                    path=path,
                    message="tracked interpreter file under benchmark artifact tree",
                )
            )
    return tuple(violations)


def _tracked_paths(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    if not result.stdout:
        return ()
    return tuple(path.decode("utf-8") for path in result.stdout.rstrip(b"\0").split(b"\0"))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check tracked repository paths.")
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="repository root to inspect",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    violations = _validate_tracked_paths(_tracked_paths(root))
    if not violations:
        return 0

    for violation in violations:
        print(violation.format(), file=sys.stderr)
    return 1


def _has_forbidden_name(path: PurePosixPath) -> bool:
    return any(part in _forbidden_names for part in path.parts)


def _is_benchmark_interpreter_file(path: PurePosixPath) -> bool:
    if path.suffix != ".py":
        return False
    return path.is_relative_to(_benchmark_artifact_root)


if __name__ == "__main__":
    raise SystemExit(_main())
