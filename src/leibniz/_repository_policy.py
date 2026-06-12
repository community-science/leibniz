"""Repository policy checks for tracked files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_forbidden_names = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "results",
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
_benchmark_implementation_filename = "benchmark.py"
_source_root = PurePosixPath("src/leibniz")
_backend_term_exemptions = frozenset(
    {
        "cost_metrology.py",
        "tensor_runtime.py",
        "_repository_policy.py",
    }
)
_backend_terms = ("torch", "cuda", "cpu", "triton")
_benchmark_runtime_escape_hatches = (
    "tensor_runtime_backend",
    "runtime.device",
    "runtime.torch",
)
_benchmark_hot_path_forbidden_fragments = (
    ".tolist(",
    ".item(",
    ".numpy(",
    ".cpu(",
)


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

    @staticmethod
    def validate_repository(root: Path) -> tuple[PolicyViolation, ...]:
        tracked_paths = _tracked_paths(root)
        return (
            *_validate_tracked_paths(tracked_paths),
            *_validate_source_backend_terms(root=root, paths=tracked_paths),
            *_validate_benchmark_runtime_escape_hatches(root=root, paths=tracked_paths),
        )


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


def _validate_source_backend_terms(
    *,
    root: Path,
    paths: Iterable[str],
) -> tuple[PolicyViolation, ...]:
    violations: list[PolicyViolation] = []
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        if path.suffix != ".py" or not path.is_relative_to(_source_root):
            continue
        if path.relative_to(_source_root).as_posix() in _backend_term_exemptions:
            continue
        try:
            lines = (root / path).read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for term in _backend_terms:
                if re.search(rf"\b{re.escape(term)}", line, flags=re.IGNORECASE):
                    violations.append(
                        PolicyViolation(
                            path=path,
                            message=(
                                "backend implementation term outside tensor runtime "
                                f"at line {line_number}: {term}"
                            ),
                        )
                    )
    return tuple(violations)


def _validate_benchmark_runtime_escape_hatches(
    *,
    root: Path,
    paths: Iterable[str],
) -> tuple[PolicyViolation, ...]:
    violations: list[PolicyViolation] = []
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        if path.name != _benchmark_implementation_filename:
            continue
        if not path.is_relative_to(_benchmark_artifact_root):
            continue
        try:
            lines = (root / path).read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for fragment in _benchmark_runtime_escape_hatches:
                if fragment in line:
                    violations.append(
                        PolicyViolation(
                            path=path,
                            message=(
                                "benchmark implementation runtime escape hatch "
                                f"at line {line_number}: {fragment}"
                            ),
                        )
                    )
            for fragment in _benchmark_hot_path_forbidden_fragments:
                if fragment in line:
                    violations.append(
                        PolicyViolation(
                            path=path,
                            message=(
                                "benchmark implementation hot-path host transfer "
                                f"at line {line_number}: {fragment}"
                            ),
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
    violations = RepositoryPolicy.validate_repository(root)
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
    if not path.is_relative_to(_benchmark_artifact_root):
        return False
    return path.name != _benchmark_implementation_filename


if __name__ == "__main__":
    raise SystemExit(_main())
