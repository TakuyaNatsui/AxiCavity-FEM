"""結果タブの右パネル: 表示中の結果、ジョブの状態、n / モード / 位相 / 時間位相、モード表、表示オプション、場の値
（ver2.3 Result Viewer のコントロールと params 表示の相当）.

状態は :class:`~axicavity_fem.gui.ui.result_model.ResultModel`（選択・オプション）。表の列は post のキー
（``result_renderer.MODE_COLUMNS``）のうち結果にあるものだけ。行を選ぶとそのモードを表示する。
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..app.store import DocumentStore
from ..i18n.tr import tr
from ..jobs.analysis_controller import AnalysisController
from ..project.controller import ProjectController
from .panel_helpers import ERROR, MUTED, OK, WARNING, ScrollPanel, group_box, label
from .result_model import ResultModel
from .result_renderer import MODE_COLUMNS, mode_rows

MONO = "font-family: Consolas, 'Courier New', monospace; font-size: 9pt;"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (abs(value) >= 1e4 or (abs(value) < 1e-2 and value != 0)):
        return f"{value:.{digits}e}"
    return f"{value:.{digits + 1}g}" if isinstance(value, float) else str(value)


class ResultsPanel(ScrollPanel):
    def __init__(self, store: DocumentStore, model: ResultModel, analysis: AnalysisController,
                 project: ProjectController, parent=None):
        super().__init__(parent)
        self.store = store
        self.model = model
        self.analysis = analysis
        self.project = project
        self._refreshing = False
        body = self.body

        self.title = label(f"<b>{tr('results.title')}</b>")
        body.addWidget(self.title)
        self.source_label = label("", MUTED)
        body.addWidget(self.source_label)
        self.relation_label = label("", WARNING)
        self.relation_label.hide()
        body.addWidget(self.relation_label)
        self.error_label = label("", ERROR)
        self.error_label.hide()
        body.addWidget(self.error_label)
        self.job_label = label("", MUTED)
        self.job_label.hide()
        body.addWidget(self.job_label)

        # --- 選択 ---
        self.select_box, layout = group_box(tr("results.selectTitle"))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.n_combo = QtWidgets.QComboBox()
        self.n_combo.currentIndexChanged.connect(
            lambda _i: self._commit(lambda: self.model.set_selection(n=int(self.n_combo.currentData()), mode=0)))
        self.n_label = QtWidgets.QLabel(tr("results.n"))
        form.addRow(self.n_label, self.n_combo)
        self.phase_combo = QtWidgets.QComboBox()
        self.phase_combo.currentIndexChanged.connect(
            lambda _i: self._commit(lambda: self.model.set_selection(phase=float(self.phase_combo.currentData()))))
        self.phase_label = QtWidgets.QLabel(tr("results.phase"))
        form.addRow(self.phase_label, self.phase_combo)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.currentIndexChanged.connect(
            lambda i: self._commit(lambda: self.model.set_selection(mode=max(0, i))))
        self.mode_label = QtWidgets.QLabel(tr("results.mode"))
        form.addRow(self.mode_label, self.mode_combo)
        self.time_phase_spin = QtWidgets.QDoubleSpinBox()
        self.time_phase_spin.setRange(0.0, 360.0)
        self.time_phase_spin.setSingleStep(15.0)
        self.time_phase_spin.setDecimals(1)
        self.time_phase_spin.setSuffix(" °")
        self.time_phase_spin.valueChanged.connect(
            lambda v: self._commit(lambda: self.model.set_selection(time_phase=float(v))))
        self.time_phase_label = QtWidgets.QLabel(tr("results.timePhase"))
        form.addRow(self.time_phase_label, self.time_phase_spin)
        layout.addLayout(form)
        self.freq_label = label("", MONO)
        layout.addWidget(self.freq_label)
        body.addWidget(self.select_box)

        # --- モード表 ---
        self.table_box, layout = group_box(tr("results.tableTitle"))
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(140)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self.table)
        self.table_hint = label(tr("results.noPost"), MUTED)
        layout.addWidget(self.table_hint)
        body.addWidget(self.table_box)

        # --- 表示オプション ---
        self.options_box, layout = group_box(tr("results.optionsTitle"))
        self.color_check = QtWidgets.QCheckBox(tr("results.hColor"))
        self.color_check.toggled.connect(lambda on: self._commit(lambda: self.model.set_options(show_color=bool(on))))
        layout.addWidget(self.color_check)
        row = QtWidgets.QHBoxLayout()
        self.lines_check = QtWidgets.QCheckBox(tr("results.eLines"))
        self.lines_check.toggled.connect(lambda on: self._commit(lambda: self.model.set_options(show_lines=bool(on))))
        self.levels_spin = QtWidgets.QSpinBox()
        self.levels_spin.setRange(2, 200)
        self.levels_spin.valueChanged.connect(lambda v: self._commit(lambda: self.model.set_options(levels=int(v))))
        self.levels_label = QtWidgets.QLabel(tr("results.levels"))
        row.addWidget(self.lines_check)
        row.addStretch(1)
        row.addWidget(self.levels_label)
        row.addWidget(self.levels_spin)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.vectors_check = QtWidgets.QCheckBox(tr("results.vectors"))
        self.vectors_check.toggled.connect(
            lambda on: self._commit(lambda: self.model.set_options(show_vectors=bool(on))))
        self.nz_spin = QtWidgets.QSpinBox()
        self.nz_spin.setRange(2, 200)
        self.nz_spin.valueChanged.connect(lambda v: self._commit(lambda: self.model.set_options(nz=int(v))))
        self.nr_spin = QtWidgets.QSpinBox()
        self.nr_spin.setRange(2, 200)
        self.nr_spin.valueChanged.connect(lambda v: self._commit(lambda: self.model.set_options(nr=int(v))))
        self.nz_label = QtWidgets.QLabel("nz")
        self.nr_label = QtWidgets.QLabel("nr")
        row.addWidget(self.vectors_check)
        row.addStretch(1)
        row.addWidget(self.nz_label)
        row.addWidget(self.nz_spin)
        row.addWidget(self.nr_label)
        row.addWidget(self.nr_spin)
        layout.addLayout(row)
        self.mesh_check = QtWidgets.QCheckBox(tr("results.mesh"))
        self.mesh_check.toggled.connect(lambda on: self._commit(lambda: self.model.set_options(show_mesh=bool(on))))
        layout.addWidget(self.mesh_check)
        self.e_wall_check = QtWidgets.QCheckBox(tr("results.eWall"))
        self.e_wall_check.setToolTip(tr("results.eWallHint"))
        self.e_wall_check.toggled.connect(
            lambda on: self._commit(lambda: self.model.set_options(show_e_wall=bool(on))))
        layout.addWidget(self.e_wall_check)
        body.addWidget(self.options_box)

        # --- 場の値（ダブルクリック） ---
        self.field_box, layout = group_box(tr("results.fieldTitle"))
        self.field_label = label(tr("results.fieldHint"), MUTED + MONO)
        layout.addWidget(self.field_label)
        body.addWidget(self.field_box)
        body.addStretch(1)

        model.loaded.connect(self.refresh)
        model.selection_changed.connect(self.refresh)
        model.options_changed.connect(self.refresh)
        analysis.changed.connect(self.refresh)
        project.changed.connect(self.refresh)
        store.document_changed.connect(self._refresh_relation)
        self.refresh()

    # ---- 操作 -------------------------------------------------------------

    def _commit(self, fn) -> None:
        if not self._refreshing:
            fn()

    def _on_table_selection(self) -> None:
        if self._refreshing:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.model.set_selection(mode=rows[0].row())

    def show_field_values(self, lines: list[str]) -> None:
        self.field_label.setStyleSheet(MONO)
        self.field_label.setText("<br>".join(lines))

    # ---- 更新 -------------------------------------------------------------

    def refresh(self) -> None:
        self._refreshing = True
        try:
            self._refresh_impl()
        finally:
            self._refreshing = False

    def _refresh_relation(self) -> None:
        model = self.model
        relation = self.project.result_relation(model.entry) if model.entry is not None else ""
        if relation in ("geometry", "mesh", "settings"):
            self.relation_label.setText(tr(f"results.relation.{relation}"))
            self.relation_label.show()
        else:
            self.relation_label.hide()

    def _refresh_job(self) -> None:
        analysis = self.analysis
        job = analysis.job
        if job is not None and not job.finished:
            self.job_label.setText(tr("analysis.state.running", message=job.message))
            self.job_label.setStyleSheet("color: #1d4ed8;")
            self.job_label.show()
        elif job is not None and job.status == "error":
            self.job_label.setText(tr("analysis.state.error", message=job.error or job.message))
            self.job_label.setStyleSheet(ERROR)
            self.job_label.show()
        else:
            self.job_label.hide()

    def _refresh_impl(self) -> None:
        model = self.model
        data = model.data
        self._refresh_job()
        self._refresh_relation()
        if model.entry is not None:
            name = model.entry.label or tr(f"project.kind.{model.entry.kind}")
            self.source_label.setText(tr("results.sourceProject", number=model.entry.number, name=name,
                                         file=model.source_name()))
        elif model.path is not None:
            self.source_label.setText(tr("results.sourceFile", path=str(model.path)))
        else:
            self.source_label.setText(tr("results.noResult"))
        self.error_label.setText(model.error)
        self.error_label.setVisible(bool(model.error))
        for box in (self.select_box, self.table_box, self.options_box, self.field_box):
            box.setEnabled(data is not None)
        if data is None:
            self.n_combo.clear()
            self.phase_combo.clear()
            self.mode_combo.clear()
            self.table.setRowCount(0)
            self.freq_label.setText("")
            return

        sel = model.selection
        opts = model.options
        is_hom = data.is_hom
        traveling = data.analysis_type(sel.n) == "traveling"

        self.n_combo.clear()
        for n in data.n_orders:
            self.n_combo.addItem(f"n = {n}", n)
        self.n_combo.setCurrentIndex(max(0, self.n_combo.findData(sel.n)))
        self.n_label.setVisible(is_hom)
        self.n_combo.setVisible(is_hom)

        self.phase_combo.clear()
        for p in data.phases(sel.n):
            self.phase_combo.addItem(f"{p:g}°", p)
        if sel.phase is not None:
            self.phase_combo.setCurrentIndex(max(0, self.phase_combo.findData(sel.phase)))
        self.phase_label.setVisible(traveling)
        self.phase_combo.setVisible(traveling)
        self.time_phase_label.setVisible(traveling)
        self.time_phase_spin.setVisible(traveling)
        self.time_phase_spin.setValue(sel.time_phase)

        freqs = data.frequencies(sel.n, sel.phase)
        self.mode_combo.clear()
        for k, f in enumerate(freqs):
            self.mode_combo.addItem(tr("results.modeItem", index=k, f=f"{f:.6f}"))
        self.mode_combo.setCurrentIndex(min(sel.mode, max(0, len(freqs) - 1)))
        self.freq_label.setText(data.title(sel))

        self._fill_table(data, sel)

        self.color_check.setChecked(opts.show_color)
        self.color_check.setText(tr("results.hColorHom") if is_hom else tr("results.hColor"))
        self.lines_check.setChecked(opts.show_lines)
        self.levels_spin.setValue(opts.levels)
        for w in (self.lines_check, self.levels_label, self.levels_spin, self.e_wall_check):
            w.setVisible(not is_hom)
        self.vectors_check.setChecked(opts.show_vectors)
        self.nz_spin.setValue(opts.nz)
        self.nr_spin.setValue(opts.nr)
        self.mesh_check.setChecked(opts.show_mesh)
        self.e_wall_check.setChecked(opts.show_e_wall)

    def _fill_table(self, data, sel) -> None:
        rows = mode_rows(data, sel.n, sel.phase)
        columns = [(key, name, unit) for key, name, unit in MODE_COLUMNS if any(key in r for r in rows)]
        headers = ["#", "f [GHz]"] + [f"{name} [{unit}]" if unit else name for _k, name, unit in columns]
        table = self.table
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [str(row["index"]), f"{row['f_GHz']:.6f}"] + [_fmt(row.get(key)) for key, _n, _u in columns]
            for j, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if i == sel.mode:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        if rows:
            table.selectRow(min(sel.mode, len(rows) - 1))
        self.table_hint.setVisible(not columns)
        self.table.setMinimumHeight(min(360, 30 + 24 * max(1, len(rows))))

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('results.title')}</b>")
        self.select_box.setTitle(tr("results.selectTitle"))
        self.n_label.setText(tr("results.n"))
        self.phase_label.setText(tr("results.phase"))
        self.mode_label.setText(tr("results.mode"))
        self.time_phase_label.setText(tr("results.timePhase"))
        self.table_box.setTitle(tr("results.tableTitle"))
        self.table_hint.setText(tr("results.noPost"))
        self.options_box.setTitle(tr("results.optionsTitle"))
        self.lines_check.setText(tr("results.eLines"))
        self.levels_label.setText(tr("results.levels"))
        self.vectors_check.setText(tr("results.vectors"))
        self.mesh_check.setText(tr("results.mesh"))
        self.e_wall_check.setText(tr("results.eWall"))
        self.e_wall_check.setToolTip(tr("results.eWallHint"))
        self.field_box.setTitle(tr("results.fieldTitle"))
        self.field_label.setText(tr("results.fieldHint"))
        self.refresh()
