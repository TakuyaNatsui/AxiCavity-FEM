"""ファイルタブの「バージョン情報」と、アプリのアイコン."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pytestqt")

from axicavity_fem.gui import GUI_VERSION  # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language  # noqa: E402
from axicavity_fem.gui.ui.dialogs.about import AboutDialog, license_file  # noqa: E402
from axicavity_fem.gui.ui.icon_set import (  # noqa: E402
    APP_ICON_FILE,
    APP_ICON_SMALL_FILE,
    APP_ICON_SMALL_MAX,
    app_icon,
    render_app_icon,
)


def test_about_dialog_shows_version_details_and_license(qtbot):
    set_language("ja")
    dialog = AboutDialog(details="PARDISO  テスト")
    qtbot.addWidget(dialog)
    assert dialog.details.toPlainText() == "PARDISO  テスト"
    assert license_file() is not None and license_file().name == "LICENSE"      # 開発環境ではリポジトリの LICENSE
    assert dialog.license_button.isEnabled()
    dialog._copy()
    from PySide6 import QtWidgets
    text = QtWidgets.QApplication.clipboard().text()
    assert text.startswith(f"AxiCavity-FEM {GUI_VERSION}") or text == ""      # offscreen ではクリップボードが無いことがある


def test_app_icon_renders(qapp):
    assert APP_ICON_FILE.is_file()
    icon = app_icon()
    assert not icon.isNull()
    image = icon.pixmap(64, 64).toImage()
    opaque = sum(1 for x in range(64) for y in range(64) if image.pixelColor(x, y).alpha() > 200)
    assert opaque > 0.6 * 64 * 64


def _same(a, b) -> bool:
    return a.size() == b.size() and all(a.pixelColor(x, y) == b.pixelColor(x, y)
                                        for x in range(a.width()) for y in range(a.height()))


def test_small_sizes_use_the_icon_with_fewer_lines(qapp):
    """32 px 以下は線を減らした版、それより大きいサイズは 256 px の版を縮小する（EXE の .ico も同じ関数で作る）."""
    from PySide6 import QtGui
    small, large = QtGui.QImage(str(APP_ICON_SMALL_FILE)), QtGui.QImage(str(APP_ICON_FILE))
    assert (small.width(), small.height()) == (APP_ICON_SMALL_MAX, APP_ICON_SMALL_MAX)
    assert (large.width(), large.height()) == (256, 256)
    assert _same(render_app_icon(APP_ICON_SMALL_MAX), small.convertToFormat(QtGui.QImage.Format_ARGB32))
    assert _same(render_app_icon(256), large.convertToFormat(QtGui.QImage.Format_ARGB32))
    assert render_app_icon(16).width() == 16 and render_app_icon(48).width() == 48
    assert not _same(render_app_icon(48).scaled(32, 32), render_app_icon(32))     # 別の絵


def test_icon_images_are_package_data():
    from pathlib import Path
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    for path in (APP_ICON_FILE, APP_ICON_SMALL_FILE):
        assert f'"ui/{path.name}"' in pyproject
