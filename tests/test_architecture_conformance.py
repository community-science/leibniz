"""The Architecture panel must not overclaim.

Every concept the panel marks ``implemented`` carries a machine-checkable
``verify`` block naming the module and symbols that realize it. These tests
assert that those symbols actually resolve in the package, and that the
biconditional holds: a concept is marked implemented if and only if it carries a
verify block. Landing the code for a ``design direction`` concept therefore
forces flipping its status and adding a verify block here, keeping the panel's
claims honest against the codebase.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

_panel_path = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "leibniz"
    / "console"
    / "_web_src"
    / "src"
    / "ArchitecturePanel.tsx"
)

_verify_block = re.compile(
    r"verify:\s*\{\s*module:\s*'([^']+)',\s*symbols:\s*\[([^\]]*)\]\s*\}"
)
_implemented_status = re.compile(r"status:\s*'implemented'")
_symbol = re.compile(r"'([^']+)'")


def _panel_source() -> str:
    return _panel_path.read_text(encoding="utf-8")


def test_implemented_concepts_match_verify_blocks() -> None:
    source = _panel_source()
    implemented = len(_implemented_status.findall(source))
    verified = len(_verify_block.findall(source))
    assert implemented > 0, "expected at least one implemented concept"
    assert verified == implemented, (
        "a concept must be marked implemented if and only if it carries a verify "
        f"block (implemented={implemented}, verify={verified})"
    )


def test_verify_symbols_resolve() -> None:
    source = _panel_source()
    blocks = _verify_block.findall(source)
    assert blocks, "expected at least one verify block"
    for module_name, symbol_blob in blocks:
        module = importlib.import_module(module_name)
        symbols = _symbol.findall(symbol_blob)
        assert symbols, f"verify block for {module_name} lists no symbols"
        for symbol in symbols:
            assert hasattr(module, symbol), (
                f"Architecture panel marks {module_name}.{symbol} implemented, "
                "but the module does not export it"
            )
