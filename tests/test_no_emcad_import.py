"""GUI パッケージが EM-CAD-py（``emcad``）を import していないことを確かめる.

開発環境では ``emcad`` が別リポジトリから import できてしまうため、コピー漏れの
``from emcad ...`` がそのまま動いてしまう。ver3 の GUI は自己完結でなければならない。
"""

from __future__ import annotations

import re
from pathlib import Path

_GUI = Path(__file__).resolve().parents[1] / "src" / "axicavity_fem" / "gui"
_PATTERN = re.compile(r"^\s*(from|import)\s+emcad\b", re.MULTILINE)


def test_gui_does_not_import_emcad():
    offenders = [p.relative_to(_GUI).as_posix() for p in sorted(_GUI.rglob("*.py"))
                 if _PATTERN.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], f"emcad を import している GUI モジュール: {offenders}"
