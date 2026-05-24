from pathlib import PurePosixPath

from leibniz.repository_policy import PolicyViolation, validate_tracked_paths


def test_repository_policy_accepts_source_and_configuration_paths() -> None:
    violations = validate_tracked_paths(
        [
            ".github/workflows/ci.yml",
            ".gitignore",
            "README.md",
            "pyproject.toml",
            "src/leibniz/__init__.py",
            "src/leibniz/py.typed",
            "tests/test_repository_policy.py",
        ]
    )

    assert violations == ()


def test_repository_policy_rejects_local_state_and_generated_outputs() -> None:
    violations = validate_tracked_paths(
        [
            ".leibniz/measurements.json",
            ".pytest_cache/v/cache/nodeids",
            ".ruff_cache/CACHEDIR.TAG",
            ".venv/pyvenv.cfg",
            "build/lib/leibniz/__init__.py",
            "dist/leibniz-0.0.0.tar.gz",
            "src/leibniz/__pycache__/__init__.cpython-311.pyc",
            "src/leibniz/module.pyc",
            ".env.local",
        ]
    )

    assert violations == (
        PolicyViolation(
            path=PurePosixPath(".leibniz/measurements.json"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".pytest_cache/v/cache/nodeids"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".ruff_cache/CACHEDIR.TAG"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".venv/pyvenv.cfg"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("build/lib/leibniz/__init__.py"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("dist/leibniz-0.0.0.tar.gz"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("src/leibniz/__pycache__/__init__.cpython-311.pyc"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("src/leibniz/module.pyc"),
            message="tracked Python bytecode output",
        ),
        PolicyViolation(
            path=PurePosixPath(".env.local"),
            message="tracked local environment file",
        ),
    )
