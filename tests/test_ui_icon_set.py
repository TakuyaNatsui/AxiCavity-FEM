"""リボンのアイコン（ui/icon_set.py と ui/icons/*.svg）: 対応表の漏れ、SVG の形式、描画、生成スクリプトとの一致.

移植元: EM-CAD-py tests/test_ui_icon_set.py。
"""

import importlib.util
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from axicavity_fem.gui.ui.icon_set import ACTION_ICONS, ICON_DIR, action_icon, has_icon, icon   # noqa: E402

I18N_DIR = Path(__file__).resolve().parent.parent / "src" / "axicavity_fem" / "gui" / "i18n"
GENERATOR = Path(__file__).resolve().parent.parent / "tools" / "make_icons.py"


def _flat_keys(tree: dict, prefix: str = "") -> set[str]:
    keys = set()
    for name, value in tree.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            keys |= _flat_keys(value, key + ".")
        else:
            keys.add(key)
    return keys


def test_every_mapped_icon_has_a_file_and_a_label():
    for lang in ("ja", "en"):
        labels = _flat_keys(json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8")))
        missing = sorted(key for key in ACTION_ICONS if key not in labels)
        assert not missing, f"{lang}.json に無い操作のキー: {missing}"
    assert not sorted(name for name in ACTION_ICONS.values() if not has_icon(name))


def test_svg_files_are_24px_line_art():
    files = sorted(ICON_DIR.glob("*.svg"))
    assert files
    for path in files:
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg"), path.name
        assert root.get("viewBox") == "0 0 24 24", path.name
    # 使われていないファイルを残さない（名前の打ち間違いにも気づける）
    assert {p.stem for p in files} == set(ACTION_ICONS.values())


def test_svg_files_match_the_generator(tmp_path):
    """SVG は tools/make_icons.py の出力そのまま（手で直した・書き出し忘れを見つける）."""
    spec = importlib.util.spec_from_file_location("make_icons", GENERATOR)
    make_icons = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_icons)
    make_icons.write_icons(tmp_path)
    generated = {p.name: p.read_bytes() for p in tmp_path.glob("*.svg")}
    shipped = {p.name: p.read_bytes() for p in ICON_DIR.glob("*.svg")}
    assert generated.keys() == shipped.keys()
    assert [name for name in generated if generated[name] != shipped[name]] == []


@pytest.mark.parametrize("size", [18, 24, 48])
@pytest.mark.parametrize("name", ["run", "bc_pec", "insert_point", "heatmap"])
def test_icon_renders_at_ribbon_sizes(qapp, size, name):
    image = icon(name).pixmap(size, size).toImage()
    assert image.width() == size
    opaque = sum(1 for x in range(size) for y in range(size) if image.pixelColor(x, y).alpha() > 128)
    assert 0.02 * size * size < opaque < 0.9 * size * size


def test_unknown_key_or_file_gives_no_icon(qapp):
    assert action_icon("ribbon.noSuchAction") is None
    assert icon("no_such_icon").isNull()
    assert action_icon("ribbon.new") is not None
