"""物理タブの右パネル: 領域（閉領域）と材料、曲線ごとの境界条件、検証（EM-CAD-py ``PhysicsPanel`` の転用）.

- 領域表: 名前 / 材料タグ / εr / μr / tanδ（εr と tanδ は式可）。閉領域は自動検出（``store.resolve_regions``）で、
  設定の無い閉領域は既定（Vacuum）。行をクリックするとその領域をスケッチ画面で強調する
- 境界条件表: 曲線ごとの有効な境界条件と由来（指定 / 自動: 内部界面 / 自動: 軸 / 既定）。行をクリックすると曲線を
  選択。選択した曲線にコンボの境界条件を「適用」（リボンのボタンと同じ）
- 検証: ``core.convert.check_geometry`` の結果（エラーは赤、警告は橙）
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..app.store import DocumentStore
from ..core.convert import check_geometry, normalize_material_tag
from ..core.document import BC_NAMES, SketchArc, SketchLine
from ..i18n.tr import tr
from .panel_helpers import ERROR, MUTED, WARNING, ScrollPanel, chip_icon, fit_table_height, group_box, label, parse_number
from .sketch_view import BC_COLORS

REGION_COLUMNS = ("name", "tag", "epsR", "muR", "tanDelta")


class PhysicsPanel(ScrollPanel):
    def __init__(self, store: DocumentStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._refreshing = False
        self._region_ids: list[str] = []
        self._curve_ids: list[str] = []
        body = self.body

        self.title = label(f"<b>{tr('physics.title')}</b>")
        body.addWidget(self.title)

        # --- 領域と材料 ---
        self.regions_box, layout = group_box(tr("physics.regions.title"))
        self.regions_hint = label(tr("physics.regions.hint"), MUTED)
        layout.addWidget(self.regions_hint)
        self.regions_empty = label(tr("physics.regions.none"), MUTED)
        layout.addWidget(self.regions_empty)
        self.regions_table = QtWidgets.QTableWidget(0, 5)
        self.regions_table.setHorizontalHeaderLabels(
            [tr("physics.regions.name"), tr("physics.regions.tag"), "εr", "μr", "tanδ"])
        self.regions_table.horizontalHeaderItem(1).setToolTip(tr("physics.regions.tagHint"))
        self.regions_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.regions_table.verticalHeader().hide()
        self.regions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.regions_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.regions_table.itemChanged.connect(self._on_region_item)
        self.regions_table.cellClicked.connect(self._on_region_clicked)
        layout.addWidget(self.regions_table)
        body.addWidget(self.regions_box)

        # --- 境界条件 ---
        self.boundaries_box, layout = group_box(tr("physics.boundaries.title"))
        self.boundaries_hint = label(tr("physics.boundaries.hint"), MUTED)
        layout.addWidget(self.boundaries_hint)
        self.boundaries_empty = label(tr("physics.boundaries.none"), MUTED)
        layout.addWidget(self.boundaries_empty)
        self.boundaries_table = QtWidgets.QTableWidget(0, 3)
        self.boundaries_table.setHorizontalHeaderLabels(
            [tr("physics.boundaries.curve"), tr("physics.boundaries.kind"), tr("physics.boundaries.source")])
        self.boundaries_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.boundaries_table.verticalHeader().hide()
        self.boundaries_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.boundaries_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.boundaries_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.boundaries_table.itemSelectionChanged.connect(self._on_boundary_selection)
        layout.addWidget(self.boundaries_table)
        row = QtWidgets.QHBoxLayout()
        self.selected_label = label("", MUTED)
        self.bc_combo = QtWidgets.QComboBox()
        for bc in BC_NAMES:
            self.bc_combo.addItem(tr(f"physics.bc.{bc}"), bc)
        self.bc_combo.addItem(tr("physics.bc.auto"), None)
        self.apply_button = QtWidgets.QPushButton(tr("physics.boundaries.apply"))
        self.apply_button.clicked.connect(self._apply_bc)
        row.addWidget(self.selected_label, 1)
        row.addWidget(self.bc_combo)
        row.addWidget(self.apply_button)
        layout.addLayout(row)
        body.addWidget(self.boundaries_box)

        self.validation = label("", MUTED)
        body.addWidget(self.validation)
        body.addStretch(1)

        store.document_changed.connect(self.refresh)
        store.settings_changed.connect(lambda _s: self.refresh())
        store.selection_changed.connect(self._refresh_selection)
        store.sketch_mode_changed.connect(lambda _s: self.refresh())
        self.refresh()

    # ---- 領域 ---------------------------------------------------------

    def focus_regions(self) -> None:
        self.regions_table.setFocus()
        if self.regions_table.rowCount():
            self.regions_table.setCurrentCell(0, 0)

    def _on_region_item(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing or item.row() >= len(self._region_ids):
            return
        profile_id = self._region_ids[item.row()]
        col = REGION_COLUMNS[item.column()]
        text = item.text().strip()
        ok = True
        if col == "name":
            ok = bool(text) and self.store.set_region(profile_id, name=text)
        elif col == "tag":
            ok = bool(text) and self.store.set_region(profile_id, materialTag=normalize_material_tag(text))
        elif col == "muR":
            value = parse_number(text)
            ok = value is not None and value > 0 and self.store.set_region(profile_id, muR=value)
        elif col == "epsR":
            ok = bool(text) and self.store.set_region(profile_id, epsR=text)
        elif col == "tanDelta":
            ok = self.store.set_region(profile_id, tanDelta=text or "0")
        if not ok:
            self.refresh()

    def _on_region_clicked(self, row: int, _col: int) -> None:
        if row < len(self._region_ids):
            profile_id = self._region_ids[row]
            self.store.set_selected_region(None if self.store.selected_region_id == profile_id else profile_id)

    # ---- 境界 ---------------------------------------------------------

    def _selected_curve_ids(self) -> list[str]:
        sketch = self.store.active_sketch()
        ids = []
        for sid in self.store.selection:
            e = next((x for x in sketch.entities if x.id == sid), None)
            if isinstance(e, (SketchLine, SketchArc)):
                ids.append(sid)
        return ids

    def _on_boundary_selection(self) -> None:
        if self._refreshing:
            return
        rows = sorted({i.row() for i in self.boundaries_table.selectedIndexes()})
        ids = [self._curve_ids[r] for r in rows if r < len(self._curve_ids)]
        if ids != self._selected_curve_ids():
            self.store.set_selection(ids)

    def _apply_bc(self) -> None:
        ids = self._selected_curve_ids()
        if ids:
            self.store.set_boundary(ids, self.bc_combo.currentData())

    # ---- 更新 ---------------------------------------------------------

    def refresh(self) -> None:
        self._refreshing = True
        try:
            self._refresh_impl()
        finally:
            self._refreshing = False

    def _refresh_selection(self) -> None:
        self._refreshing = True
        try:
            self._refresh_selection_impl()
        finally:
            self._refreshing = False

    def _refresh_selection_impl(self) -> None:
        store = self.store
        selected = set(self._selected_curve_ids())
        table = self.boundaries_table
        table.clearSelection()
        model = table.selectionModel()
        flags = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for row, cid in enumerate(self._curve_ids):
            if cid in selected:
                model.select(table.model().index(row, 0), flags)      # selectRow は ExtendedSelection で置き換えになる
        count = len(selected)
        self.selected_label.setText(tr("physics.boundaries.selected", count=count) if count else "")
        self.apply_button.setEnabled(count > 0)
        if count == 1:
            bc = store.effective_bc(next(iter(selected)))[0]
            self.bc_combo.setCurrentIndex(max(0, self.bc_combo.findData(bc)))
        region = store.selected_region_id
        rt = self.regions_table
        rt.clearSelection()
        if region in self._region_ids:
            rt.selectRow(self._region_ids.index(region))

    def _refresh_impl(self) -> None:
        store = self.store
        doc = store.document
        regions = store.resolve_regions()
        self._region_ids = [r.profile.id for r in regions]
        self.regions_empty.setVisible(not regions)
        self.regions_table.setVisible(bool(regions))
        self.regions_table.setRowCount(len(regions))
        for row, r in enumerate(regions):
            values = [r.name, r.material_tag, r.eps_r_expr or f"{r.eps_r:g}", f"{r.mu_r:g}",
                      r.tan_delta_expr or f"{r.tan_delta:g}"]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 2 and r.eps_r_expr:
                    item.setToolTip(f"{r.eps_r_expr} = {r.eps_r:g}")
                if col == 4 and r.tan_delta_expr:
                    item.setToolTip(f"{r.tan_delta_expr} = {r.tan_delta:g}")
                if r.setting is None:
                    item.setForeground(QtGui.QColor("#6b7280"))
                self.regions_table.setItem(row, col, item)
        fit_table_height(self.regions_table)

        sketch = store.active_sketch()
        curves = [e for e in sketch.entities if isinstance(e, (SketchLine, SketchArc)) and not e.construction]
        self._curve_ids = [c.id for c in curves]
        self.boundaries_empty.setVisible(not curves)
        self.boundaries_table.setVisible(bool(curves))
        self.boundaries_table.setRowCount(len(curves))
        for row, c in enumerate(curves):
            bc, source = store.effective_bc(c.id)
            kind = tr("constraints.tangent") if False else ("Line" if isinstance(c, SketchLine) else "Arc")
            name_item = QtWidgets.QTableWidgetItem(f"{row + 1}  {kind}")
            bc_item = QtWidgets.QTableWidgetItem(bc)
            bc_item.setIcon(chip_icon(BC_COLORS.get(bc, "#888888")))
            source_item = QtWidgets.QTableWidgetItem(tr(f"physics.source.{source}"))
            if source != "explicit":
                source_item.setForeground(QtGui.QColor("#6b7280"))
            for col, item in enumerate((name_item, bc_item, source_item)):
                self.boundaries_table.setItem(row, col, item)
        fit_table_height(self.boundaries_table)
        self._refresh_selection_impl()

        issues = check_geometry(doc)
        if not issues:
            self.validation.setText(tr("physics.validation.ok"))
            self.validation.setStyleSheet(MUTED)
        else:
            parts = []
            for issue in issues:
                color = "#b91c1c" if issue.level == "error" else "#b45309"
                parts.append(f"<span style='color:{color}'>{issue.message}</span>")
            self.validation.setText("<br>".join(parts))
            self.validation.setStyleSheet(ERROR if any(i.level == "error" for i in issues) else WARNING)

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('physics.title')}</b>")
        self.regions_box.setTitle(tr("physics.regions.title"))
        self.regions_hint.setText(tr("physics.regions.hint"))
        self.regions_empty.setText(tr("physics.regions.none"))
        self.regions_table.setHorizontalHeaderLabels(
            [tr("physics.regions.name"), tr("physics.regions.tag"), "εr", "μr", "tanδ"])
        self.boundaries_box.setTitle(tr("physics.boundaries.title"))
        self.boundaries_hint.setText(tr("physics.boundaries.hint"))
        self.boundaries_empty.setText(tr("physics.boundaries.none"))
        self.boundaries_table.setHorizontalHeaderLabels(
            [tr("physics.boundaries.curve"), tr("physics.boundaries.kind"), tr("physics.boundaries.source")])
        for i, bc in enumerate(BC_NAMES):
            self.bc_combo.setItemText(i, tr(f"physics.bc.{bc}"))
        self.bc_combo.setItemText(len(BC_NAMES), tr("physics.bc.auto"))
        self.apply_button.setText(tr("physics.boundaries.apply"))
        self.refresh()
