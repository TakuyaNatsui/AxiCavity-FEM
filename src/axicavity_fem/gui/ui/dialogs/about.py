"""ファイルタブの「バージョン情報」: 版・同梱物（PARDISO・メッシュ生成・3D 表示）・ライセンス・リンク."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ... import GUI_VERSION, frozen
from ...app_info import GITHUB_URL, info_text
from ...i18n.tr import tr
from ..icon_set import app_icon


def license_file() -> Optional[Path]:
    """ライセンス表記のファイル（EXE: THIRD_PARTY_NOTICES.txt、開発・pip: リポジトリの LICENSE）."""
    base = frozen.app_dir()
    for name in ("THIRD_PARTY_NOTICES.txt", "LICENSE"):
        if (base / name).is_file():
            return base / name
    return None


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, details: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("about.title"))
        self.setMinimumWidth(560)
        layout = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel()
        logo.setPixmap(app_icon().pixmap(72, 72))
        logo.setAlignment(QtCore.Qt.AlignTop)
        head.addWidget(logo)
        text = QtWidgets.QLabel(
            f"<h2 style='margin:0'>AxiCavity-FEM {GUI_VERSION}</h2>"
            f"<p>{tr('about.description')}</p>"
            f"<p>{tr('about.copyright')}<br><a href='{GITHUB_URL}'>{GITHUB_URL}</a></p>")
        text.setTextFormat(QtCore.Qt.RichText)
        text.setOpenExternalLinks(True)
        text.setWordWrap(True)
        head.addWidget(text, 1)
        layout.addLayout(head)

        note = QtWidgets.QLabel(tr("about.licenseExe") if frozen.is_compiled() else tr("about.licenseSource"))
        note.setWordWrap(True)
        layout.addWidget(note)

        self.details = QtWidgets.QPlainTextEdit(details if details is not None else info_text())
        self.details.setReadOnly(True)
        self.details.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.details.setMinimumHeight(200)
        layout.addWidget(self.details, 1)

        buttons = QtWidgets.QDialogButtonBox()
        self.copy_button = buttons.addButton(tr("about.copy"), QtWidgets.QDialogButtonBox.ActionRole)
        self.license_button = buttons.addButton(tr("about.openLicense"), QtWidgets.QDialogButtonBox.ActionRole)
        self.license_button.setEnabled(license_file() is not None)
        close = buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        close.setText(tr("about.close"))
        self.copy_button.clicked.connect(self._copy)
        self.license_button.clicked.connect(self._open_license)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy(self) -> None:
        QtWidgets.QApplication.clipboard().setText(f"AxiCavity-FEM {GUI_VERSION}\n{self.details.toPlainText()}")

    def _open_license(self) -> None:
        path = license_file()
        if path is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
