"""GUI パッケージが matplotlib のバックエンドを切り替える経路を持たないことを確かめる.

``reports/plot_common.py`` は import 時に ``matplotlib.use("Agg")`` を呼ぶので、Qt 埋め込み
キャンバスと混ぜてはいけない（ver2.3 DEVELOPER_GUIDE §4.7）。GUI 側は ``Figure`` と
``FigureCanvasQTAgg`` だけを使い、``pyplot`` / ``plot_common`` / ``matplotlib.use`` / ``wx`` を
import しない。
"""

from __future__ import annotations

import re
from pathlib import Path

_GUI = Path(__file__).resolve().parents[1] / "src" / "axicavity_fem" / "gui"
_PATTERNS = {
    "pyplot": re.compile(
        r"^\s*(from\s+matplotlib\s+import\s+.*\bpyplot\b|import\s+matplotlib\.pyplot\b|"
        r"from\s+matplotlib\.pyplot\s+import)", re.MULTILINE),
    "plot_common": re.compile(
        r"^\s*(from\s+\S*reports\s+import\s+.*\bplot_common\b|from\s+\S*reports\.plot_common\s+import|"
        r"import\s+\S*reports\.plot_common\b)", re.MULTILINE),
    "matplotlib.use": re.compile(r"^\s*matplotlib\.use\(", re.MULTILINE),
    "wx": re.compile(r"^\s*(import\s+wx\b|from\s+wx\b)", re.MULTILINE),
}


def test_gui_has_no_backend_switching_imports():
    offenders: list[str] = []
    for p in sorted(_GUI.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        for name, pattern in _PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{p.relative_to(_GUI).as_posix()}: {name}")
    assert offenders == [], f"GUI で禁止している import: {offenders}"
