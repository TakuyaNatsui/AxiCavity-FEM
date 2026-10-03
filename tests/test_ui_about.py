"""ファイルタブの「バージョン情報」と、アプリのアイコン."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pytestqt")

from axicavity_fem.gui import GUI_VERSION  # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language  # noqa: E402
from axicavity_fem.gui.ui.dialogs.about import AboutDialog, license_file  # noqa: E402
from axicavity_fem.gui.ui.icon_set import APP_ICON_FILE, app_icon  # noqa: E402


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
