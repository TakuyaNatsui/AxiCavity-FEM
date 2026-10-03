"""スケッチ編集中のパネル（EM-CAD-py ``emcad/ui/sketch_panels.py`` からの転用、元は TS 版 `PropertyPanel.tsx` /
`ConstraintPanel.tsx`）.

- SketchPropertyPanel: ヒント・数値入力（長さ/半径）・ツールオプション・エンティティ数/閉領域数・選択・
  **選択した点の Z / R（式可、数値プレビュー、r=0 に固定）**・**選択した円弧の中心 Z / R（式可）と半径**・
  自由度・拘束一覧・パラメータ表（トグル）
- ConstraintList: 拘束一覧（自由度・矛盾/冗長・削除・ダブルクリックで値編集）

ver3 の追加は選択点 / 円弧の欄（ver2.3 の Selected Point / Selected Segment に相当）。投影の一覧は無い。
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..app.store import DocumentStore, safe_params
from ..core.document import SketchArc, SketchPoint, is_dimension_constraint
from ..core.expressions import ExpressionError, evaluate_expression, expr_or_none, format_compact, format_value
from ..core.document import is_reference
from ..core.sketch.model import constraint_entity_ids, count_entities, get_entity, point_pos
from ..core.sketch.reference import entity_numbers, reference_label_key
from ..i18n.tr import tr
from .constraint_actions import edit_dimension_value, select_constraint
from .params_panel import ParamsPanel


def _fmt(v: float) -> str:
    return format_compact(v)


class ConstraintList(QtWidgets.QWidget):
    """スケッチの拘束一覧と解の状態."""

    def __init__(self, store: DocumentStore, parent=None):
        super().__init__(parent)
        self.store = store
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(f"<b>{tr('props.constraints')}</b>")
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #b91c1c;")
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("props.constraints"), tr("params.value"), ""])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addWidget(self.message)
        layout.addWidget(self.table, 1)
        self._ids: list[str] = []
        self._delete_buttons: list[QtWidgets.QToolButton] = []   # 行ごとに再利用
        store.document_changed.connect(self.refresh)
        store.solve_info_changed.connect(lambda _i: self.refresh())
        store.selection_changed.connect(self._sync_selection)
        store.sketch_mode_changed.connect(lambda _s: self.refresh())

    def refresh(self) -> None:
        sketch = self.store.active_sketch()
        info = self.store.solve_info
        if sketch is None:
            self._set_row_count(0)
            self.status.setText("")
            self.message.setText("")
            return
        if info is not None:
            text = tr("props.dof", count=info.dof)
            if info.status != "ok":
                text += f" / {tr('solve.status.' + info.status)}"
            self.status.setText(text)
            self.status.setStyleSheet(
                "color: #b91c1c;" if info.status in ("conflict", "failed", "invalid") else "color: #6b7280;")
            msg = tr(info.message) if info.message else ""
            if info.error:
                msg += f": {info.error}"
            self.message.setText(msg)
        conflicting = set(info.conflicting) if info else set()
        redundant = set(info.redundant) if info else set()
        params = safe_params(self.store.document)
        self._ids = [c.id for c in sketch.constraints]
        self._set_row_count(len(sketch.constraints))
        numbers = entity_numbers(sketch)
        for row, c in enumerate(sketch.constraints):
            name = tr(f"constraints.{c.type}")
            try:
                targets = [tr(reference_label_key(i)) if reference_label_key(i) else numbers.get(i, "?")
                           for i in constraint_entity_ids(c)]
            except ValueError:
                targets = []
            if targets:
                name += f"（{' – '.join(targets)}）"          # 何に拘束しているか（P3 – z 軸 など）
            if c.id in redundant:
                name += f" ({tr('solve.status.redundant')})"
            item = QtWidgets.QTableWidgetItem(name)
            if c.id in conflicting:
                item.setForeground(QtGui.QColor("#b91c1c"))
            self.table.setItem(row, 0, item)
            if is_dimension_constraint(c):
                try:
                    value = _fmt(evaluate_expression(c.value, params))
                except Exception:
                    value = tr("params.invalid")
                text = c.value if c.value.strip() == value else f"{c.value} = {value}"
            else:
                text = ""
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(text))
            self.table.setCellWidget(row, 2, self._delete_button(row))
        self._sync_selection()

    def _set_row_count(self, n: int) -> None:
        self.table.setRowCount(n)
        del self._delete_buttons[n:]

    def _delete_button(self, row: int) -> QtWidgets.QToolButton:
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
            self.store.remove_constraint(self._ids[row])

    def _sync_selection(self) -> None:
        sel = self.store.selected_constraint_id
        self.table.blockSignals(True)
        self.table.clearSelection()
        if sel in self._ids:
            self.table.selectRow(self._ids.index(sel))
        self.table.blockSignals(False)

    def _on_row_clicked(self, row: int, _col: int) -> None:
        if row < len(self._ids):
            select_constraint(self.store, self._ids[row])

    def _on_row_double_clicked(self, row: int, _col: int) -> None:
        if row < len(self._ids):
            edit_dimension_value(self.store, self._ids[row], self)


class CoordinateEditor(QtWidgets.QWidget):
    """Z / R の入力欄（式可）と数値プレビュー（ver2.3 の ``= 110.000000``）、適用ボタン.

    ``applied(z_text, r_text)`` を Enter か「適用」で出す。
    """

    applied = QtCore.Signal(str, str)

    def __init__(self, store: DocumentStore, title_key: str, parent=None):
        super().__init__(parent)
        self.store = store
        self.title_key = title_key
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(f"<b>{tr(title_key)}</b>")
        layout.addWidget(self.title)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.z_edit = QtWidgets.QLineEdit()
        self.r_edit = QtWidgets.QLineEdit()
        self.z_preview = QtWidgets.QLabel("")
        self.r_preview = QtWidgets.QLabel("")
        for label in (self.z_preview, self.r_preview):
            label.setStyleSheet("color: #6b7280;")
        self.z_label = QtWidgets.QLabel("Z")
        self.r_label = QtWidgets.QLabel("R")
        grid.addWidget(self.z_label, 0, 0)
        grid.addWidget(self.z_edit, 0, 1)
        grid.addWidget(self.z_preview, 0, 2)
        grid.addWidget(self.r_label, 1, 0)
        grid.addWidget(self.r_edit, 1, 1)
        grid.addWidget(self.r_preview, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.extra = QtWidgets.QLabel("")
        self.extra.setStyleSheet("color: #6b7280;")
        layout.addWidget(self.extra)
        self.buttons = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton(tr("props.apply"))
        self.apply_button.clicked.connect(self._emit)
        self.buttons.addWidget(self.apply_button)
        layout.addLayout(self.buttons)
        for edit in (self.z_edit, self.r_edit):
            edit.returnPressed.connect(self._emit)
            edit.textEdited.connect(lambda _t: self.update_previews())
        self.current: tuple[str, str] = ("", "")        # 直前に set_values で入れた値

    def add_button(self, text: str, tooltip: str = "") -> QtWidgets.QPushButton:
        """「適用」の右に操作ボタンを足す（r=0 に固定など）."""
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tooltip)
        self.buttons.addWidget(button)
        return button

    def set_values(self, z_text: str, r_text: str, extra: str = "") -> None:
        self.current = (z_text, r_text)
        self.z_edit.setText(z_text)
        self.r_edit.setText(r_text)
        self.extra.setText(extra)
        self.extra.setVisible(bool(extra))
        self.update_previews()

    def _preview(self, text: str) -> str:
        expr = expr_or_none(text)
        if expr is None:
            return ""
        try:
            return "= " + format_value(evaluate_expression(expr, safe_params(self.store.document)))
        except (ExpressionError, ValueError):
            return "= ?"

    def update_previews(self) -> None:
        self.z_preview.setText(self._preview(self.z_edit.text()))
        self.r_preview.setText(self._preview(self.r_edit.text()))

    def _emit(self) -> None:
        self.applied.emit(self.z_edit.text(), self.r_edit.text())

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr(self.title_key)}</b>")
        self.apply_button.setText(tr("props.apply"))


class SketchPropertyPanel(QtWidgets.QWidget):
    """モデリングのプロパティ（ヒント・数値入力・オプション・情報・選択点/円弧・拘束・パラメータ）."""

    numeric_entered = QtCore.Signal(str, float)      # ("length" | "radius", 値)

    def __init__(self, store: DocumentStore, parent=None):
        super().__init__(parent)
        self.store = store
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QtWidgets.QLabel(f"<b>{tr('props.sketchTitle')}</b>")
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #1d4ed8;")
        layout.addWidget(self.title)
        layout.addWidget(self.hint)

        form = QtWidgets.QFormLayout()
        self.numeric = QtWidgets.QLineEdit()
        self.numeric.setPlaceholderText(tr("props.numericHint"))
        self.numeric.returnPressed.connect(self._on_numeric)
        self.numeric_label = QtWidgets.QLabel(tr("props.length"))
        form.addRow(self.numeric_label, self.numeric)
        self.snap_grid = QtWidgets.QCheckBox(tr("props.snapToGrid"))
        self.snap_grid.setChecked(store.tool_options.snap_to_grid)
        self.snap_grid.toggled.connect(lambda v: store.set_tool_options(snap_to_grid=bool(v)))
        form.addRow(self.snap_grid)
        self.sides = QtWidgets.QSpinBox()
        self.sides.setRange(3, 64)
        self.sides.setValue(store.tool_options.sides)
        self.sides.valueChanged.connect(lambda v: store.set_tool_options(sides=int(v)))
        self.sides_label = QtWidgets.QLabel(tr("props.sides"))
        form.addRow(self.sides_label, self.sides)
        self.fillet_radius_label = QtWidgets.QLabel(tr("props.filletRadius"))
        self.fillet_radius = QtWidgets.QDoubleSpinBox()
        self.fillet_radius.setDecimals(3)
        self.fillet_radius.setRange(0.001, 1e6)
        self.fillet_radius.setValue(store.tool_options.fillet_radius)
        self.fillet_radius.valueChanged.connect(
            lambda v: store.set_tool_options(fillet_radius=float(v)))
        form.addRow(self.fillet_radius_label, self.fillet_radius)
        layout.addLayout(form)

        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #6b7280;")
        layout.addWidget(self.info)

        # ver3: 選択した点 / 円弧（カーソル座標はステータスバー）
        self.point_editor = CoordinateEditor(store, "props.selectedPoint")
        self.fix_axis_button = self.point_editor.add_button(tr("ribbon.fixAxis"), tr("tip.ribbon.fixAxis"))
        self.fix_axis_button.clicked.connect(lambda: self._fix_selected_point("r"))
        self.fix_z0_button = self.point_editor.add_button(tr("ribbon.fixZ0"), tr("tip.ribbon.fixZ0"))
        self.fix_z0_button.clicked.connect(lambda: self._fix_selected_point("z"))
        self.point_editor.applied.connect(self._on_point_applied)
        self.point_editor.setVisible(False)
        layout.addWidget(self.point_editor)
        self.arc_editor = CoordinateEditor(store, "props.selectedArc")
        self.arc_editor.applied.connect(self._on_arc_applied)
        self.flip_arc_button = self.arc_editor.add_button(tr("props.flipArc"), tr("props.flipArcHint"))
        self.flip_arc_button.clicked.connect(self._on_flip_arc)
        self.arc_editor.setVisible(False)
        layout.addWidget(self.arc_editor)
        self.expr_error = QtWidgets.QLabel("")
        self.expr_error.setWordWrap(True)
        self.expr_error.setStyleSheet("color: #b91c1c;")
        self.expr_error.setVisible(False)
        layout.addWidget(self.expr_error)

        self.constraints = ConstraintList(store)
        layout.addWidget(self.constraints, 1)
        self.params = ParamsPanel(store)
        self.params.setVisible(False)
        layout.addWidget(self.params, 1)
        self._selected_point_id: Optional[str] = None
        self._selected_arc_id: Optional[str] = None
        self._shown_point_id: Optional[str] = None

        store.hint_changed.connect(self._on_hint)
        store.tool_changed.connect(self._on_tool)
        store.document_changed.connect(self._refresh_info)
        store.selection_changed.connect(self._refresh_info)
        store.params_edit_changed.connect(self.params.setVisible)
        store.sketch_mode_changed.connect(lambda _s: self._refresh_info())
        store.tool_options_changed.connect(self._on_tool_options)
        store.solve_info_changed.connect(self._on_solve_info)
        self._on_tool(store.active_tool)

    def _on_hint(self, key) -> None:
        self.hint.setText(tr(key) if key else "")

    def _on_tool(self, tool: str) -> None:
        self.numeric_label.setText(tr("props.radius") if tool in ("circle", "fillet")
                                   else tr("props.length"))
        self.numeric.setEnabled(tool in ("line", "circle", "fillet"))
        # ツール専用の設定はそのツールのときだけ見せる
        for widget in (self.sides_label, self.sides):
            widget.setVisible(tool == "polygon")
        for widget in (self.fillet_radius_label, self.fillet_radius):
            widget.setVisible(tool == "fillet")
        self.fillet_radius_label.setVisible(tool == "fillet")
        self.fillet_radius.setVisible(tool == "fillet")

    def _on_tool_options(self) -> None:
        self.fillet_radius.blockSignals(True)
        self.fillet_radius.setValue(self.store.tool_options.fillet_radius)
        self.fillet_radius.blockSignals(False)

    def _on_numeric(self) -> None:
        text = self.numeric.text().strip()
        try:
            value = float(evaluate_expression(text, safe_params(self.store.document)))
        except Exception:
            return
        name = "radius" if self.store.active_tool in ("circle", "fillet") else "length"
        self.numeric_entered.emit(name, value)
        self.numeric.clear()

    def _on_solve_info(self, info) -> None:
        if info is not None and info.status == "invalid" and info.error:
            self.expr_error.setText(f"{tr('solve.invalid')}: {info.error}")
            self.expr_error.setVisible(True)
        else:
            self.expr_error.setVisible(False)

    # ---- 選択した点 / 円弧 ----

    def _refresh_info(self) -> None:
        sketch = self.store.active_sketch()
        if sketch is None:
            self.info.setText("")
            return
        counts = count_entities(sketch)
        lines = [tr("props.entities", **counts),
                 tr("props.profiles", count=len(self.store.profiles))]
        if self.store.selection:
            lines.append(tr("props.selection", count=len(self.store.selection)))
        self.info.setText("\n".join(lines))
        self._refresh_selected(sketch)

    def _refresh_selected(self, sketch) -> None:
        sel = self.store.selection
        point = get_entity(sketch, sel[0]) if len(sel) == 1 else None
        if isinstance(point, SketchPoint) and not is_reference(point):     # 参照の原点は編集しない
            self._selected_point_id = point.id
            values = (point.xExpr or _fmt(point.x), point.yExpr or _fmt(point.y))
            if point.id == self._shown_point_id and self.point_editor.current == values:
                self.point_editor.update_previews()          # 同じ点・同じ値なら入力中の文字を消さない
            else:
                self.point_editor.set_values(*values)
            self._shown_point_id = point.id
            self.point_editor.setVisible(True)
        else:
            self._selected_point_id = None
            self.point_editor.setVisible(False)
        arc = get_entity(sketch, sel[0]) if len(sel) == 1 else None
        if isinstance(arc, SketchArc):
            self._selected_arc_id = arc.id
            center = get_entity(sketch, arc.center)
            radius = _fmt(((center.x - point_pos(sketch, arc.start)[0]) ** 2
                           + (center.y - point_pos(sketch, arc.start)[1]) ** 2) ** 0.5)
            self.arc_editor.set_values(center.xExpr or _fmt(center.x), center.yExpr or _fmt(center.y),
                                       tr("props.radiusValue", value=radius))
            self.arc_editor.setVisible(True)
        else:
            self._selected_arc_id = None
            self.arc_editor.setVisible(False)

    def _on_point_applied(self, z_text: str, r_text: str) -> None:
        if self._selected_point_id is None:
            return
        self.store.set_point_coords(self._selected_point_id, z_text, r_text)

    def _fix_selected_point(self, coord: str) -> None:
        if self._selected_point_id is not None:
            self.store.fix_points_to_zero([self._selected_point_id], coord)

    def _on_arc_applied(self, z_text: str, r_text: str) -> None:
        if self._selected_arc_id is None:
            return
        self.store.set_arc_center(self._selected_arc_id, z_text, r_text)

    def _on_flip_arc(self) -> None:
        if self._selected_arc_id is not None:
            self.store.flip_arc(self._selected_arc_id)

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('props.sketchTitle')}</b>")
        self.numeric.setPlaceholderText(tr("props.numericHint"))
        self.snap_grid.setText(tr("props.snapToGrid"))
        self.sides_label.setText(tr("props.sides"))
        self.fillet_radius_label.setText(tr("props.filletRadius"))
        self.point_editor.retranslate()
        for button, key in ((self.fix_axis_button, "ribbon.fixAxis"), (self.fix_z0_button, "ribbon.fixZ0")):
            button.setText(tr(key))
            button.setToolTip(tr(f"tip.{key}"))
        self.arc_editor.retranslate()
        self.flip_arc_button.setText(tr("props.flipArc"))
        self.flip_arc_button.setToolTip(tr("props.flipArcHint"))
        self.constraints.title.setText(f"<b>{tr('props.constraints')}</b>")
        self.params.retranslate()
        self._on_tool(self.store.active_tool)
        self._refresh_info()
