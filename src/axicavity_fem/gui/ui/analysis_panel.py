"""解析タブの右パネル: 解析の種類（TM0 / HOM、方位角次数）、定在波 / 進行波（位相）、モード数、予測周波数、
post（導電率・β）、レポート設定、CLI 相当コマンドのプレビュー、検証、実行中のジョブの状態（ver2.3 の FEM タブ相当）.

設定は ``document.analysis`` / ``post`` / ``report``（保存される）。実行は :class:`AnalysisController`。
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..app.store import DocumentStore
from ..i18n.tr import tr
from ..jobs.analysis_controller import AnalysisController
from .panel_helpers import ERROR, MUTED, OK, WARNING, CommitLineEdit, ScrollPanel, group_box, label, parse_number

MONO = "font-family: Consolas, 'Courier New', monospace; font-size: 9pt;"


class AnalysisPanel(ScrollPanel):
    """Signals:
        run_requested(): 「解析実行」が押された（確認・投入は MainWindow）。
    """

    run_requested = QtCore.Signal()

    def __init__(self, store: DocumentStore, analysis: AnalysisController, parent=None):
        super().__init__(parent)
        self.store = store
        self.analysis = analysis
        self._refreshing = False
        body = self.body

        self.title = label(f"<b>{tr('analysis.title')}</b>")
        body.addWidget(self.title)
        self.hint = label(tr("analysis.hint"), MUTED)
        body.addWidget(self.hint)

        # --- 解析の種類 ---
        self.kind_box, layout = group_box(tr("analysis.kindTitle"))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem(tr("analysis.tm0"), "tm0")
        self.type_combo.addItem(tr("analysis.hom"), "hom")
        self.type_combo.currentIndexChanged.connect(
            lambda _i: self._commit(lambda: self.store.set_analysis(type=self.type_combo.currentData())))
        self.type_label = QtWidgets.QLabel(tr("analysis.type"))
        form.addRow(self.type_label, self.type_combo)
        self.az_edit = CommitLineEdit()
        self.az_edit.setToolTip(tr("analysis.azOrdersHint"))
        self.az_edit.committed.connect(lambda t: self._commit(lambda: self.store.set_analysis(azOrders=t.strip())))
        self.az_label = QtWidgets.QLabel(tr("analysis.azOrders"))
        form.addRow(self.az_label, self.az_edit)
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItem(tr("analysis.standing"), "standing")
        self.wave_combo.addItem(tr("analysis.traveling"), "traveling")
        self.wave_combo.currentIndexChanged.connect(
            lambda _i: self._commit(lambda: self.store.set_analysis(wave=self.wave_combo.currentData())))
        self.wave_label = QtWidgets.QLabel(tr("analysis.wave"))
        form.addRow(self.wave_label, self.wave_combo)
        self.phases_edit = CommitLineEdit()
        self.phases_edit.setToolTip(tr("analysis.phasesHint"))
        self.phases_edit.committed.connect(lambda t: self._commit(lambda: self.store.set_analysis(phases=t.strip())))
        self.phases_label = QtWidgets.QLabel(tr("analysis.phases"))
        form.addRow(self.phases_label, self.phases_edit)
        self.modes_edit = CommitLineEdit()
        self.modes_edit.committed.connect(self._on_modes)
        self.modes_label = QtWidgets.QLabel(tr("analysis.numModes"))
        form.addRow(self.modes_label, self.modes_edit)
        self.target_edit = CommitLineEdit()
        self.target_edit.setPlaceholderText(tr("analysis.targetAuto"))
        self.target_edit.setToolTip(tr("analysis.targetHint"))
        self.target_edit.committed.connect(self._on_target)
        self.target_label = QtWidgets.QLabel(tr("analysis.target"))
        form.addRow(self.target_label, self.target_edit)
        self.order_value = label("", MUTED)
        self.order_label = QtWidgets.QLabel(tr("analysis.elemOrder"))
        form.addRow(self.order_label, self.order_value)
        layout.addLayout(form)
        body.addWidget(self.kind_box)

        # --- post ---
        self.post_box, layout = group_box(tr("analysis.postTitle"))
        self.run_post_check = QtWidgets.QCheckBox(tr("analysis.runPost"))
        self.run_post_check.toggled.connect(lambda on: self._commit(lambda: self.store.set_analysis(runPost=bool(on))))
        layout.addWidget(self.run_post_check)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.cond_edit = CommitLineEdit()
        self.cond_edit.committed.connect(self._on_cond)
        self.cond_label = QtWidgets.QLabel(tr("analysis.cond"))
        form.addRow(self.cond_label, self.cond_edit)
        self.beta_edit = CommitLineEdit()
        self.beta_edit.committed.connect(self._on_beta)
        self.beta_label = QtWidgets.QLabel(tr("analysis.beta"))
        form.addRow(self.beta_label, self.beta_edit)
        layout.addLayout(form)
        body.addWidget(self.post_box)

        # --- レポート ---
        self.report_box, layout = group_box(tr("analysis.reportTitle"))
        self.animate_check = QtWidgets.QCheckBox(tr("analysis.animate"))
        self.animate_check.toggled.connect(lambda on: self._commit(lambda: self.store.set_report(animate=bool(on))))
        layout.addWidget(self.animate_check)
        self.show_mesh_check = QtWidgets.QCheckBox(tr("analysis.showMesh"))
        self.show_mesh_check.toggled.connect(lambda on: self._commit(lambda: self.store.set_report(showMesh=bool(on))))
        layout.addWidget(self.show_mesh_check)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.time_phase_edit = CommitLineEdit()
        self.time_phase_edit.committed.connect(self._on_time_phase)
        self.time_phase_label = QtWidgets.QLabel(tr("analysis.timePhase"))
        form.addRow(self.time_phase_label, self.time_phase_edit)
        self.dpi_edit = CommitLineEdit()
        self.dpi_edit.committed.connect(self._on_dpi)
        self.dpi_label = QtWidgets.QLabel(tr("analysis.dpi"))
        form.addRow(self.dpi_label, self.dpi_edit)
        layout.addLayout(form)
        body.addWidget(self.report_box)

        # --- 実行 ---
        self.run_box, layout = group_box(tr("analysis.runTitle"))
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton(tr("analysis.ribbon.run"))
        font = self.run_button.font()
        font.setBold(True)
        self.run_button.setFont(font)
        self.run_button.clicked.connect(self.run_requested)
        self.cancel_button = QtWidgets.QPushButton(tr("analysis.ribbon.cancel"))
        self.cancel_button.clicked.connect(lambda: self.analysis.cancel())
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.state_label = label("", MUTED)
        layout.addWidget(self.state_label)
        self.error_label = label("", ERROR)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.command_label = label("", MUTED + MONO)
        self.command_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.command_label)
        body.addWidget(self.run_box)

        self.validation = label("", MUTED)
        body.addWidget(self.validation)
        body.addStretch(1)

        store.document_changed.connect(self.refresh)
        store.settings_changed.connect(lambda _s: self.refresh())
        analysis.changed.connect(self.refresh)
        self.refresh()

    # ---- 操作 -------------------------------------------------------------

    def _commit(self, fn) -> None:
        if not self._refreshing:
            fn()

    def _on_modes(self, text: str) -> None:
        value = parse_number(text)
        if value is not None and value >= 1:
            self._commit(lambda: self.store.set_analysis(numModes=int(round(value))))
        self.refresh()

    def _on_target(self, text: str) -> None:
        if not text.strip():
            self._commit(lambda: self.store.set_analysis(targetFreqGHz=None))
        else:
            value = parse_number(text)
            if value is not None and value > 0:
                self._commit(lambda: self.store.set_analysis(targetFreqGHz=value))
        self.refresh()

    def _on_cond(self, text: str) -> None:
        value = parse_number(text)
        if value is not None and value > 0:
            self._commit(lambda: self.store.set_post(cond=value))
        self.refresh()

    def _on_beta(self, text: str) -> None:
        value = parse_number(text)
        if value is not None and 0 < value <= 1:
            self._commit(lambda: self.store.set_post(beta=value))
        self.refresh()

    def _on_time_phase(self, text: str) -> None:
        value = parse_number(text)
        if value is not None:
            self._commit(lambda: self.store.set_report(timePhase=value))
        self.refresh()

    def _on_dpi(self, text: str) -> None:
        value = parse_number(text)
        if value is not None and value >= 30:
            self._commit(lambda: self.store.set_report(dpi=int(round(value))))
        self.refresh()

    # ---- 更新 -------------------------------------------------------------

    def refresh(self) -> None:
        self._refreshing = True
        try:
            self._refresh_impl()
        finally:
            self._refreshing = False

    def _refresh_impl(self) -> None:
        doc = self.store.document
        a, post, report = doc.analysis, doc.post, doc.report
        self.type_combo.setCurrentIndex(self.type_combo.findData(a.type))
        self.az_edit.set_value(a.azOrders)
        self.az_label.setVisible(a.type == "hom")
        self.az_edit.setVisible(a.type == "hom")
        self.wave_combo.setCurrentIndex(self.wave_combo.findData(a.wave))
        self.phases_edit.set_value(a.phases)
        self.phases_label.setVisible(a.wave == "traveling")
        self.phases_edit.setVisible(a.wave == "traveling")
        self.modes_edit.set_value(str(a.numModes))
        self.target_edit.set_value("" if a.targetFreqGHz is None else f"{a.targetFreqGHz:g}")
        self.order_value.setText(tr("analysis.elemOrderValue", order=doc.mesh.order))
        self.run_post_check.setChecked(a.runPost)
        self.cond_edit.set_value(f"{post.cond:g}")
        self.beta_edit.set_value(f"{post.beta:g}")
        self.beta_label.setVisible(a.type == "tm0")
        self.beta_edit.setVisible(a.type == "tm0")
        self.animate_check.setChecked(report.animate)
        self.animate_check.setVisible(a.wave == "traveling")
        self.show_mesh_check.setChecked(report.showMesh)
        self.time_phase_edit.set_value(f"{report.timePhase:g}")
        self.dpi_edit.set_value(str(report.dpi))

        analysis = self.analysis
        running = analysis.running
        self.run_button.setEnabled(not analysis.session.running and bool(self.store.profiles))
        self.cancel_button.setVisible(running)
        if running and analysis.job is not None:
            self.state_label.setText(tr("analysis.state.running", message=analysis.job.message))
            self.state_label.setStyleSheet("color: #1d4ed8;")
        elif analysis.job is not None and analysis.job.finished:
            key = f"analysis.state.{analysis.job.status}"
            self.state_label.setText(tr(key, message=analysis.job.message))
            self.state_label.setStyleSheet(OK if analysis.job.status == "done" else WARNING)
        else:
            self.state_label.setText(tr("analysis.state.idle"))
            self.state_label.setStyleSheet(MUTED)
        self.error_label.setText(analysis.last_error or "")
        self.error_label.setVisible(bool(analysis.last_error) and not running)
        self.command_label.setText("<br>".join(analysis.preview_lines()))

        errors, warnings = analysis.check()
        if not errors and not warnings:
            self.validation.setText(tr("physics.validation.ok"))
            self.validation.setStyleSheet(MUTED)
        else:
            parts = [f"<span style='color:#b91c1c'>{e}</span>" for e in errors]
            parts += [f"<span style='color:#b45309'>{w}</span>" for w in warnings]
            self.validation.setText("<br>".join(parts))
            self.validation.setStyleSheet(ERROR if errors else WARNING)

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('analysis.title')}</b>")
        self.hint.setText(tr("analysis.hint"))
        self.kind_box.setTitle(tr("analysis.kindTitle"))
        self.type_combo.setItemText(0, tr("analysis.tm0"))
        self.type_combo.setItemText(1, tr("analysis.hom"))
        self.wave_combo.setItemText(0, tr("analysis.standing"))
        self.wave_combo.setItemText(1, tr("analysis.traveling"))
        for widget, key in ((self.type_label, "analysis.type"), (self.az_label, "analysis.azOrders"),
                            (self.wave_label, "analysis.wave"), (self.phases_label, "analysis.phases"),
                            (self.modes_label, "analysis.numModes"), (self.target_label, "analysis.target"),
                            (self.order_label, "analysis.elemOrder"), (self.cond_label, "analysis.cond"),
                            (self.beta_label, "analysis.beta"), (self.time_phase_label, "analysis.timePhase"),
                            (self.dpi_label, "analysis.dpi")):
            widget.setText(tr(key))
        self.post_box.setTitle(tr("analysis.postTitle"))
        self.run_post_check.setText(tr("analysis.runPost"))
        self.report_box.setTitle(tr("analysis.reportTitle"))
        self.animate_check.setText(tr("analysis.animate"))
        self.show_mesh_check.setText(tr("analysis.showMesh"))
        self.run_box.setTitle(tr("analysis.runTitle"))
        self.run_button.setText(tr("analysis.ribbon.run"))
        self.cancel_button.setText(tr("analysis.ribbon.cancel"))
        self.refresh()
