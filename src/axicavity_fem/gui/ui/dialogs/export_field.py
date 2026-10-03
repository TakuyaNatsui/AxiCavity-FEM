"""場の書き出し（area / line / axis）のパラメータ入力（ver2.3 ``ExportFieldDialog`` の移植）.

area: Z / R 範囲と格子点数。line: 2 端点と点数。axis: 軸 r=0 上の Z 範囲と点数。
共通: 目標壁損失（scale-to-power、空欄で無効）・倍率・進行波の瞬時値（TM0）・出力形式・出力ベース名。
値は :meth:`ExportFieldDialog.params` が dict で返す（``jobs.commands.export_argv`` の ``params``）。
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtWidgets

from ...i18n.tr import tr
from ..panel_helpers import MUTED, label

SHAPES = ("area", "line", "axis")


class ExportFieldDialog(QtWidgets.QDialog):
    def __init__(self, shape: str, default_output: str, z_bounds: tuple[float, float],
                 r_bounds: tuple[float, float], *, traveling_tm0: bool = False, has_post: bool = True,
                 nz: int = 200, nr: int = 100, npts: int = 500, parent=None):
        super().__init__(parent)
        assert shape in SHAPES
        self.shape = shape
        self.traveling_tm0 = traveling_tm0
        self.setWindowTitle(tr(f"fieldExport.title.{shape}"))
        self.setMinimumWidth(480)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label(tr(f"fieldExport.hint.{shape}"), MUTED))
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        zmin, zmax = z_bounds
        rmin, rmax = r_bounds

        def edit(value: float) -> QtWidgets.QLineEdit:
            w = QtWidgets.QLineEdit(f"{value:.6g}")
            return w

        def spin(lo: int, hi: int, value: int) -> QtWidgets.QSpinBox:
            w = QtWidgets.QSpinBox()
            w.setRange(lo, hi)
            w.setValue(value)
            return w

        self.edits: dict[str, QtWidgets.QLineEdit] = {}
        if shape == "area":
            for key, value, text in (("zmin", zmin, "export.zmin"), ("zmax", zmax, "export.zmax"),
                                     ("rmin", rmin, "export.rmin"), ("rmax", rmax, "export.rmax")):
                self.edits[key] = edit(value)
                form.addRow(tr(text), self.edits[key])
            self.nz_spin = spin(2, 5000, nz)
            self.nr_spin = spin(2, 5000, nr)
            form.addRow(tr("fieldExport.nz"), self.nz_spin)
            form.addRow(tr("fieldExport.nr"), self.nr_spin)
        elif shape == "axis":
            for key, value, text in (("zmin", zmin, "export.zmin"), ("zmax", zmax, "export.zmax")):
                self.edits[key] = edit(value)
                form.addRow(tr(text), self.edits[key])
            self.npts_spin = spin(2, 20000, npts)
            form.addRow(tr("fieldExport.npts"), self.npts_spin)
        else:
            for key, value, text in (("p1z", zmin, "export.p1z"), ("p1r", rmin, "export.p1r"),
                                     ("p2z", zmax, "export.p2z"), ("p2r", rmin, "export.p2r")):
                self.edits[key] = edit(value)
                form.addRow(tr(text), self.edits[key])
            self.npts_spin = spin(2, 20000, npts)
            form.addRow(tr("fieldExport.npts"), self.npts_spin)

        self.power_edit = QtWidgets.QLineEdit("")
        self.power_edit.setPlaceholderText(tr("fieldExport.powerHint"))
        self.power_edit.setEnabled(has_post)
        if not has_post:
            self.power_edit.setToolTip(tr("fieldExport.powerNeedsPost"))
        form.addRow(tr("fieldExport.power"), self.power_edit)
        self.scale_edit = QtWidgets.QLineEdit("1")
        form.addRow(tr("fieldExport.scale"), self.scale_edit)
        self.instant_check = QtWidgets.QCheckBox(tr("fieldExport.instant"))
        self.instant_check.setVisible(traveling_tm0)
        form.addRow("", self.instant_check)
        self.format_combo = QtWidgets.QComboBox()
        for fmt in ("both", "h5", "txt"):
            self.format_combo.addItem(tr(f"fieldExport.format.{fmt}"), fmt)
        form.addRow(tr("fieldExport.formatLabel"), self.format_combo)
        row = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit(default_output)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        form.addRow(tr("fieldExport.output"), row)
        layout.addWidget(label(tr("fieldExport.outputHint"), MUTED))

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(tr("fieldExport.run"))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.error_label = label("", "color: #b91c1c;")
        self.error_label.hide()
        layout.addWidget(self.error_label)

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("fieldExport.output"), self.output_edit.text(),
                                                        tr("fieldExport.filter"))
        if path:
            self.output_edit.setText(os.path.splitext(path)[0])

    def _accept(self) -> None:
        try:
            self.params()
        except ValueError as exc:
            self.error_label.setText(tr("fieldExport.invalid", message=str(exc)))
            self.error_label.show()
            return
        self.accept()

    def _float(self, key: str) -> float:
        text = self.edits[key].text().strip()
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{tr('fieldExport.' + key)}: {text!r}") from None

    def _power(self) -> Optional[float]:
        text = self.power_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{tr('fieldExport.power')}: {text!r}") from None
        if value <= 0:
            raise ValueError(f"{tr('fieldExport.power')}: {text!r}")
        return value

    def params(self) -> dict:
        """``export_argv`` の params（検証込み。不正な値は ValueError）."""
        output = self.output_edit.text().strip()
        if not output:
            raise ValueError(tr("fieldExport.output"))
        try:
            scale = float(self.scale_edit.text().strip() or "1")
        except ValueError:
            raise ValueError(f"{tr('fieldExport.scale')}: {self.scale_edit.text()!r}") from None
        p: dict = {"output": output, "fmt": self.format_combo.currentData(), "scale_to_power": self._power(),
                   "scale": scale, "instant": self.traveling_tm0 and self.instant_check.isChecked()}
        if self.shape == "area":
            p["z_range"] = (self._float("zmin"), self._float("zmax"))
            p["r_range"] = (self._float("rmin"), self._float("rmax"))
            p["nz"] = int(self.nz_spin.value())
            p["nr"] = int(self.nr_spin.value())
        elif self.shape == "axis":
            p["z_range"] = (self._float("zmin"), self._float("zmax"))
            p["npts"] = int(self.npts_spin.value())
        else:
            p["p1"] = (self._float("p1z"), self._float("p1r"))
            p["p2"] = (self._float("p2z"), self._float("p2r"))
            p["npts"] = int(self.npts_spin.value())
        for key in ("z_range", "r_range"):
            if key in p and p[key][1] <= p[key][0]:
                raise ValueError(tr("fieldExport.rangeError"))
        return p
