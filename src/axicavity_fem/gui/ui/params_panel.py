"""パラメータ表（名前・式・値。EM-CAD-py ``ui/sketch_panels.py`` の ParamsPanel を分離し、ver2.3 の変数表に
合わせて値の列を順次評価の結果に、行の並べ替えを追加したもの）.

上の行から順に評価され、上の行を下の行で参照できる（前方参照はエラー）。式は ver2.3 の評価器
（``sin`` / ``sqrt`` / ``pi``、べき乗は ``**``）。
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..app.store import DocumentStore
from ..core.expressions import param_results
from ..i18n.tr import tr


class ParamsPanel(QtWidgets.QWidget):
    def __init__(self, store: DocumentStore, parent=None):
        super().__init__(parent)
        self.store = store
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(f"<b>{tr('params.title')}</b>")
        layout.addWidget(self.title)
        self.hint = QtWidgets.QLabel(tr("params.hint"))
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #6b7280;")
        layout.addWidget(self.hint)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([tr("params.name"), tr("params.expression"), tr("params.value"), ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton(tr("params.add"))
        self.add_button.clicked.connect(self._on_add)
        self.up_button = QtWidgets.QToolButton()
        self.up_button.setText("↑")
        self.up_button.setToolTip(tr("params.moveUp"))
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QtWidgets.QToolButton()
        self.down_button.setText("↓")
        self.down_button.setToolTip(tr("params.moveDown"))
        self.down_button.clicked.connect(lambda: self._move(1))
        buttons.addWidget(self.add_button, 1)
        buttons.addWidget(self.up_button)
        buttons.addWidget(self.down_button)
        layout.addLayout(buttons)
        self._ids: list[str] = []
        self._delete_buttons: list[QtWidgets.QToolButton] = []
        self._updating = False
        store.document_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self._updating = True
        doc = self.store.document
        results = param_results(doc.params)
        self._ids = [p.id for p in doc.params]
        self.table.setRowCount(len(doc.params))
        del self._delete_buttons[len(doc.params):]
        for row, (p, result) in enumerate(zip(doc.params, results)):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(p.name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(p.expression))
            item = QtWidgets.QTableWidgetItem(result)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            if result.startswith("エラー") or result.startswith("Error"):
                item.setForeground(QtCore.Qt.red)
                item.setToolTip(result)
            self.table.setItem(row, 2, item)
            self.table.setCellWidget(row, 3, self._delete_button(row))
        self._updating = False

    def _delete_button(self, row: int) -> QtWidgets.QToolButton:
        # setCellWidget が古いウィジェットを毎回捨てないよう、行ごとに使い回す（EM-CAD-py の知見）
        if row < len(self._delete_buttons):
            return self._delete_buttons[row]
        btn = QtWidgets.QToolButton()
        btn.setText("×")
        btn.setToolTip(tr("props.delete"))
        btn.clicked.connect(lambda _=False, r=row: self._delete_row(r))
        self._delete_buttons.append(btn)
        return btn

    def _delete_row(self, row: int) -> None:
        if row < len(self._ids):
            self.store.remove_param(self._ids[row])

    def _on_add(self) -> None:
        pid = self.store.add_param()
        row = self._ids.index(pid) if pid in self._ids else -1
        if row >= 0:
            self.table.setCurrentCell(row, 0)
            self.table.editItem(self.table.item(row, 0))

    def _move(self, delta: int) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._ids):
            pid = self._ids[row]
            self.store.move_param(pid, delta)
            if pid in self._ids:
                self.table.setCurrentCell(self._ids.index(pid), self.table.currentColumn())

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating or item.row() >= len(self._ids):
            return
        pid = self._ids[item.row()]
        text = item.text().strip()
        if item.column() == 0 and text:
            self.store.update_param(pid, name=text)
        elif item.column() == 1 and text:
            self.store.update_param(pid, expression=text)
        else:
            self.refresh()

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('params.title')}</b>")
        self.hint.setText(tr("params.hint"))
        self.table.setHorizontalHeaderLabels([tr("params.name"), tr("params.expression"), tr("params.value"), ""])
        self.add_button.setText(tr("params.add"))
