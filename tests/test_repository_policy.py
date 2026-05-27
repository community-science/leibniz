import re
from pathlib import Path, PurePosixPath

from leibniz._repository_policy import PolicyViolation, RepositoryPolicy
from leibniz.console.artifact_index import ConsoleArtifactIndexBuilder
from leibniz.documents import load_object_document

_repository_root = Path(__file__).parents[1]


def test_repository_policy_accepts_source_and_configuration_paths() -> None:
    violations = RepositoryPolicy.validate_tracked_paths(
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
    violations = RepositoryPolicy.validate_tracked_paths(
        [
            ".leibniz/measurements.json",
            ".pytest_cache/v/cache/nodeids",
            ".ruff_cache/CACHEDIR.TAG",
            ".runs/measurements/digits/local-run.json",
            ".runs/proposals/digits/proposal_set.json",
            ".runs/views/benchmark_results.json",
            ".venv/pyvenv.cfg",
            ".vite/deps/react.js",
            "build/lib/leibniz/__init__.py",
            "docs/latent-factor-complexity.md",
            "dist/leibniz-0.0.0.tar.gz",
            "src/leibniz/benchmarks/digits/__init__.py",
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
            path=PurePosixPath(".runs/measurements/digits/local-run.json"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".runs/proposals/digits/proposal_set.json"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".runs/views/benchmark_results.json"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".venv/pyvenv.cfg"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath(".vite/deps/react.js"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("build/lib/leibniz/__init__.py"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("docs/latent-factor-complexity.md"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("dist/leibniz-0.0.0.tar.gz"),
            message="tracked local, cache, or generated directory",
        ),
        PolicyViolation(
            path=PurePosixPath("src/leibniz/benchmarks/digits/__init__.py"),
            message="tracked interpreter file under benchmark artifact tree",
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


def test_benchmark_artifact_tree_contains_only_data_files() -> None:
    benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks"

    tracked_files = tuple(path for path in benchmark_root.rglob("*") if path.is_file())

    assert tracked_files
    assert all(path.suffix == ".json" for path in tracked_files)
    assert not any(path.suffix == ".py" for path in tracked_files)


def test_benchmark_artifacts_do_not_declare_architecture_search_space() -> None:
    benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks"

    search_artifacts: list[str] = []
    for path in sorted(benchmark_root.rglob("*.json")):
        record = load_object_document(path.read_bytes(), description=path.as_posix())
        if record.get("format") == "leibniz.architecture-candidate-space":
            search_artifacts.append(path.relative_to(_repository_root).as_posix())

    assert search_artifacts == []


def test_benchmark_names_are_not_hardcoded_outside_benchmark_artifacts() -> None:
    source_root = _repository_root / "src" / "leibniz"
    benchmark_root = source_root / "benchmarks"
    benchmark_names = tuple(path.name for path in benchmark_root.iterdir() if path.is_dir())

    offenders = tuple(
        path.relative_to(_repository_root)
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".ts", ".tsx"}
        and benchmark_root not in path.parents
        and "node_modules" not in path.parts
        and "dist" not in path.parts
        and any(
            re.search(
                rf"(?i)(?<![a-z]){re.escape(benchmark_name)}(?![a-z])",
                path.read_text(encoding="utf-8"),
            )
            for benchmark_name in benchmark_names
        )
    )

    assert offenders == ()


def test_legacy_performance_bundles_are_not_supported() -> None:
    assert "performance-view-bundle" not in ConsoleArtifactIndexBuilder.supported_kinds()
    assert not (_repository_root / "src" / "leibniz" / "performance_bundles.py").exists()
    assert not (
        _repository_root
        / "src"
        / "leibniz"
        / "console"
        / "_web_src"
        / "src"
        / "performanceViews.ts"
    ).exists()
