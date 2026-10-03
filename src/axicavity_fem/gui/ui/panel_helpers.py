"""右パネル共通の部品（EM-CAD-py ``ui/em_panels.py`` から切り出し）.

- :class:`CommitLineEdit`: Enter / フォーカス移動で確定し、値が変わったときだけ ``committed`` を出す
  （キー入力ごとに Undo 履歴を積まない）
- :class:`ScrollPanel`: 縦スクロールするパネルの土台
- :func:`group_box` / :func:`label` / :func:`fit_table_height` / :func:`chip_icon` / :func:`parse_number`
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

MUTED = "color: #6b7280;"
ERROR = "color: #b91c1c;"
WARNING = "color: #b45309;"
OK = "color: #047857;"


def parse_number(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def fit_table_height(table: QtWidgets.QTableWidget, min_rows: int = 1) -> None:
    """表の高さを行数に合わせる（表の中にスクロールバーを出さない。パネル全体がスクロールする）."""
    header = table.horizontalHeader()
    rows = table.verticalHeader()
    height = 0 if header.isHidden() else header.sizeHint().height()
    height += rows.length() + rows.defaultSectionSize() * max(0, min_rows - table.rowCount())
    table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    table.setFixedHeight(height + 2 * table.frameWidth())


def chip_icon(color: str, size: int = 12) -> QtGui.QIcon:
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtGui.QColor(color))
    return QtGui.QIcon(pix)


def group_box(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(8, 6, 8, 8)
    layout.setSpacing(4)
    return box, layout


def label(text: str = "", style: str = "", wrap: bool = True) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setWordWrap(wrap)
    widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    if style:
        widget.setStyleSheet(style)
    return widget


class CommitLineEdit(QtWidgets.QLineEdit):
    """Enter / フォーカス移動で確定し、値が変わったときだけ committed を出す（Esc で元に戻す）."""

    committed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = ""
        self.editingFinished.connect(self._commit)

    def set_value(self, text: str) -> None:
        self._value = text
        if self.text() != text:
            self.setText(text)

    def _commit(self) -> None:
        text = self.text()
        if text != self._value:
            self._value = text
            self.committed.emit(text)

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.setText(self._value)
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)


class ScrollPanel(QtWidgets.QScrollArea):
    """縦スクロールするパネルの土台."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)   # 横には広げない（縦だけスクロール）
        self.content = QtWidgets.QWidget()
        self.body = QtWidgets.QVBoxLayout(self.content)
        self.body.setContentsMargins(6, 6, 6, 6)
        self.body.setSpacing(8)
        self.setWidget(self.content)
