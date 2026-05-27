from pathlib import Path

from leibniz.console.codegen import generated_console_protocol_module

_repository_root = Path(__file__).parents[1]
_generated_module = (
    _repository_root
    / "src"
    / "leibniz"
    / "console"
    / "_web_src"
    / "src"
    / "generated"
    / "protocolVocabulary.ts"
)
_web_source_root = _repository_root / "src" / "leibniz" / "console" / "_web_src" / "src"


def test_generated_console_protocol_module_is_current() -> None:
    assert _generated_module.read_text(encoding="utf-8") == generated_console_protocol_module()


def test_handwritten_web_source_uses_generated_protocol_formats() -> None:
    migrated_literals = (
        "leibniz.console-data",
        "leibniz.console.artifact-index",
        "leibniz.console.imported-results",
        "leibniz.console.benchmark-results",
        "leibniz.console.work-queue",
        "leibniz.work-queue-item",
    )

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in sorted(_web_source_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".ts", ".tsx"}
        and "generated" not in path.parts
        and any(literal in path.read_text(encoding="utf-8") for literal in migrated_literals)
    )

    assert offenders == ()
