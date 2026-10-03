"""メッシュタブの右パネル: メッシュ設定（lc・次数）、生成、統計、スケッチ画面への表示
（EM-CAD-py ``ui/mesh_panel.py`` の転用。ver2.3 の Mesh Output 相当）.

設定は ``document.mesh``（``size`` / ``sizeExpr`` / ``order``。次数はメッシュの幾何次数 = 解析の要素次数）、
生成と統計は :class:`~axicavity_fem.gui.jobs.mesh_controller.MeshController`。
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..app.store import DocumentStore, safe_params
from ..core.expressions import ExpressionError, evaluate_expression, expr_or_none, format_value
from ..i18n.tr import tr
from ..jobs.mesh_controller import MeshController
from .panel_helpers import ERROR, MUTED, OK, WARNING, CommitLineEdit, ScrollPanel, chip_icon, group_box, label
from .sketch_view import BC_COLORS, COLORS

STATE_STYLE = {"none": MUTED, "current": OK, "settings": WARNING, "geometry": WARNING, "running": "color: #1d4ed8;"}


def _num(value, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) else "-"


class MeshPanel(ScrollPanel):
    """メッシュの設定・生成・統計・表示.

    Signals:
        generate_requested(): 「メッシュ生成」が押された（保存先の確認と投入は MainWindow）。
    """

    generate_requested = QtCore.Signal()

    def __init__(self, store: DocumentStore, mesh: MeshController, parent=None):
        super().__init__(parent)
        self.store = store
        self.mesh = mesh
        self._refreshing = False
        body = self.body

        self.title = label(f"<b>{tr('mesh.title')}</b>")
        body.addWidget(self.title)
        self.hint = label(tr("mesh.hint"), MUTED)
        body.addWidget(self.hint)
        self.state_label = label("")
        body.addWidget(self.state_label)
        self.error_label = label("", ERROR)
        self.error_label.hide()
        body.addWidget(self.error_label)

        # --- 設定 ---
        self.settings_box, layout = group_box(tr("mesh.settings"))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        size_row = QtWidgets.QHBoxLayout()
        self.size_edit = CommitLineEdit()
        self.size_edit.setToolTip(tr("mesh.sizeHint"))
        self.size_edit.committed.connect(self._on_size)
        self.size_edit.textEdited.connect(lambda _t: self._update_size_preview())
        self.size_preview = label("", MUTED)
        size_row.addWidget(self.size_edit, 1)
        size_row.addWidget(self.size_preview)
        self.size_label = QtWidgets.QLabel(tr("mesh.sizeUnits", units=store.document.meta.units))
        form.addRow(self.size_label, size_row)
        self.order_combo = QtWidgets.QComboBox()
        self.order_combo.addItem(tr("mesh.order2"), 2)
        self.order_combo.addItem(tr("mesh.order1"), 1)
        self.order_combo.setToolTip(tr("mesh.orderHint"))
        self.order_combo.currentIndexChanged.connect(self._on_order)
        self.order_label = QtWidgets.QLabel(tr("mesh.order"))
        form.addRow(self.order_label, self.order_combo)
        layout.addLayout(form)
        body.addWidget(self.settings_box)

        # --- 生成 ---
        self.generate_box, layout = group_box(tr("mesh.generateTitle"))
        row = QtWidgets.QHBoxLayout()
        self.generate_button = QtWidgets.QPushButton(tr("mesh.generate"))
        font = self.generate_button.font()
        font.setBold(True)
        self.generate_button.setFont(font)
        self.generate_button.clicked.connect(self.generate_requested)
        self.stop_button = QtWidgets.QPushButton(tr("mesh.stop"))
        self.stop_button.clicked.connect(lambda: self.mesh.cancel(force=True))
        self.folder_button = QtWidgets.QPushButton(tr("mesh.openFolder"))
        self.folder_button.clicked.connect(self._open_folder)
        row.addWidget(self.generate_button)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        row.addWidget(self.folder_button)
        layout.addLayout(row)
        self.progress_label = label("", MUTED)
        layout.addWidget(self.progress_label)
        body.addWidget(self.generate_box)

        # --- 統計 ---
        self.stats_box, layout = group_box(tr("mesh.stats"))
        self.stats_label = label("")
        layout.addWidget(self.stats_label)
        self.bc_list = QtWidgets.QListWidget()
        self.bc_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.bc_list)
        body.addWidget(self.stats_box)

        # --- 表示 ---
        self.show_check = QtWidgets.QCheckBox(tr("mesh.show"))
        self.show_check.toggled.connect(lambda on: self._commit(lambda: self.mesh.set_visible(bool(on))))
        body.addWidget(self.show_check)
        body.addStretch(1)

        store.document_changed.connect(self.refresh)
        store.settings_changed.connect(lambda _s: self.refresh())
        mesh.changed.connect(self.refresh)
        self.refresh()

    # ---- 操作 -------------------------------------------------------------

    def _commit(self, fn) -> None:
        if not self._refreshing:
            fn()

    def _parse_size(self, text: str):
        """入力 → (値, 式 or None)。評価できなければ None."""
        expr = expr_or_none(text)
        try:
            if expr is None:
                value = float(text.strip())
            else:
                value = float(evaluate_expression(expr, safe_params(self.store.document)))
        except (ExpressionError, ValueError):
            return None
        return (value, expr) if value > 0 else None

    def _on_size(self, text: str) -> None:
        parsed = self._parse_size(text)
        if parsed is not None:
            value, expr = parsed
            self._commit(lambda: self.store.set_mesh(size=value, sizeExpr=expr))
        self.refresh()

    def _update_size_preview(self) -> None:
        parsed = self._parse_size(self.size_edit.text())
        if parsed is None:
            self.size_preview.setText("= ?" if self.size_edit.text().strip() else "")
        elif parsed[1] is None:
            self.size_preview.setText("")
        else:
            self.size_preview.setText("= " + format_value(parsed[0]))

    def _on_order(self, _index: int) -> None:
        self._commit(lambda: self.store.set_mesh(order=int(self.order_combo.currentData())))

    def _open_folder(self) -> None:
        info = self.mesh.info
        if info is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(info.dir)))

    # ---- 更新 -------------------------------------------------------------

    def refresh(self) -> None:
        self._refreshing = True
        try:
            self._refresh_impl()
        finally:
            self._refreshing = False

    def _refresh_impl(self) -> None:
        store, mesh = self.store, self.mesh
        doc = store.document
        units = doc.meta.units
        self.size_label.setText(tr("mesh.sizeUnits", units=units))
        self.size_edit.set_value(doc.mesh.sizeExpr if doc.mesh.sizeExpr else f"{doc.mesh.size:g}")
        self._update_size_preview()
        self.order_combo.setCurrentIndex(self.order_combo.findData(doc.mesh.order))

        running = mesh.running
        state = "running" if running else mesh.state()
        self.state_label.setText(tr(f"mesh.state.{state}"))
        self.state_label.setStyleSheet(STATE_STYLE[state])
        self.error_label.setText(mesh.last_error or "")
        self.error_label.setVisible(bool(mesh.last_error) and not running)
        self.generate_button.setEnabled(not mesh.session.running and bool(store.profiles))
        self.stop_button.setVisible(running)
        self.folder_button.setEnabled(mesh.info is not None)
        self.progress_label.setText(mesh.job.message if (mesh.job is not None and running) else "")
        self.progress_label.setVisible(running)

        info = mesh.info
        self.stats_box.setVisible(info is not None)
        self.show_check.setEnabled(info is not None)
        self.show_check.setChecked(mesh.visible)
        if info is None:
            return
        s = info.stats
        lines = [tr("mesh.nodesElements", nodes=f"{s.get('nodes', 0):,}", elements=f"{s.get('elements', 0):,}",
                    order=s.get("order", "-")),
                 tr("mesh.meanEdge", value=_num(s.get("meanEdge")), units=s.get("units", units))]
        for r in s.get("regions") or []:
            lines.append(tr("mesh.regionElements", name=r.get("name", r.get("tag", "?")),
                            count=f"{r.get('elements', 0):,}"))
        elapsed = (info.summary.get("elapsedS") or {}).get("meshing")
        lines.append(tr("mesh.created", time=info.created_at.replace("T", " ") or "-", seconds=_num(elapsed, 3)))
        for line in info.summary.get("warnings") or []:
            lines.append(f"<span style='color:#b45309'>{line}</span>")
        self.stats_label.setText("<br>".join(lines))

        self.bc_list.clear()
        for name, count in (s.get("bc") or {}).items():
            item = QtWidgets.QListWidgetItem(tr("mesh.bcEdges", name=name, count=count))
            item.setIcon(chip_icon(BC_COLORS.get(name, "#888888")))
            self.bc_list.addItem(item)
        if s.get("interfaces"):
            item = QtWidgets.QListWidgetItem(tr("mesh.interfaces", count=s["interfaces"]))
            item.setIcon(chip_icon(COLORS["interface"]))
            self.bc_list.addItem(item)
        self.bc_list.setFixedHeight(20 * max(1, self.bc_list.count()) + 6)

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('mesh.title')}</b>")
        self.hint.setText(tr("mesh.hint"))
        self.settings_box.setTitle(tr("mesh.settings"))
        self.order_label.setText(tr("mesh.order"))
        self.order_combo.setItemText(0, tr("mesh.order2"))
        self.order_combo.setItemText(1, tr("mesh.order1"))
        self.generate_box.setTitle(tr("mesh.generateTitle"))
        self.generate_button.setText(tr("mesh.generate"))
        self.stop_button.setText(tr("mesh.stop"))
        self.folder_button.setText(tr("mesh.openFolder"))
        self.stats_box.setTitle(tr("mesh.stats"))
        self.show_check.setText(tr("mesh.show"))
        self.refresh()
