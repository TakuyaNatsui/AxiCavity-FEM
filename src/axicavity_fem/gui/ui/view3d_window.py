"""3D 表示の別ウィンドウ（pyvistaqt の QtInteractor + 設定パネル）と、結果モデルに追従するコントローラ.

- :func:`is_available`: pyvista / pyvistaqt が入っているか（無ければリボンのボタンを無効にして案内する）
- :class:`View3DController`（QObject）: :class:`~axicavity_fem.gui.ui.result_model.ResultModel` の loaded / selection_changed に
  150 ms のデバウンスで追従し、:class:`~axicavity_fem.gui.ui.view3d_scene.Scene3D` にメッシュと場を渡す。
  表示オプション（:class:`View3DOptions`）、時間位相のアニメーション（QTimer）、PNG / GIF の保存。scene は duck typing
  （set_mesh / set_fields / set_options / set_time_phase / view / render_png / clear / bounds_z）なのでテストは偽物で回せる
- :class:`View3DPanel`（QWidget、VTK 非依存）: 右側の設定
- :class:`View3DWindow`（ToolWindow）: 左に QtInteractor（VTK。offscreen では作らない）、右に View3DPanel

pyvistaqt は qtpy 経由で Qt を選ぶので、import の前に ``QT_API=pyside6`` を環境に入れる（EM-CAD-py と同じ）。
QtInteractor のキー操作は Esc と F 以外を VTK に渡さない（w / e / q などの VTK 既定のキーを殺す）。
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ..core.revolve import mode_fields_from_renderer
from ..i18n.tr import tr
from .panel_helpers import MUTED, ScrollPanel, group_box, label
from .result_model import ResultModel
from .tool_window import ToolWindow
from .view3d_scene import COLOR_FIELDS, VIEWS, View3DOptions, render_gif

FIELD_LABELS = {"|E|": "|E|", "Ez": "E_z", "Er": "E_r", "Ephi": "E_φ", "|H|": "|H|", "Hz": "H_z", "Hr": "H_r",
                "Hphi": "H_φ"}
ANIMATION_FRAMES = 36
ANIMATION_FPS = 12
DEBOUNCE_MS = 150


def is_available() -> bool:
    """pyvista と pyvistaqt が import できるか（VTK のウィンドウは作らない）."""
    os.environ.setdefault("QT_API", "pyside6")
    try:
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
    except Exception:  # noqa: BLE001 — 未導入・壊れた VTK など
        return False
    return True


class View3DController(QtCore.QObject):
    """Signals:
        options_changed(): 表示オプションが変わった。
        state_changed(): 再生 / 停止、結果の有無が変わった。
    """

    options_changed = QtCore.Signal()
    state_changed = QtCore.Signal()

    def __init__(self, model: ResultModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.scene = None
        self.options = View3DOptions()
        self.active = False                   # ウィンドウが見えているときだけ描く
        self.error: str = ""
        self._data_seen = None
        self._fields = None
        self._fields_key = None
        self._time_phase = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / ANIMATION_FPS))
        self._timer.timeout.connect(self._tick)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.refresh)
        model.loaded.connect(self._schedule)
        model.selection_changed.connect(self._schedule)

    # ---- 状態 -------------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    @property
    def time_phase(self) -> float:
        return self._time_phase

    @property
    def can_animate(self) -> bool:
        return self.model.is_traveling and self.model.has_data

    @property
    def is_tm0(self) -> bool:
        return self.model.has_data and not self.model.is_hom

    def attach(self, scene) -> None:
        self.scene = scene
        self._data_seen = None
        self._fields_key = None

    def set_active(self, active: bool) -> None:
        self.active = bool(active)
        if not self.active:
            self.stop()
        else:
            self.refresh()

    # ---- 更新 -------------------------------------------------------------

    def _schedule(self) -> None:
        if self.active:
            self._debounce.start()

    def refresh(self) -> None:
        """モデルの結果・選択を scene に反映する（メッシュが変わっていれば作り直す）."""
        if self.scene is None or not self.active:
            return
        model = self.model
        if model.data is None or model.renderer is None:
            self.scene.clear()
            self._data_seen = None
            self._fields = None
            self._fields_key = None
            self.stop()
            self.state_changed.emit()
            return
        try:
            if model.data is not self._data_seen:
                mesh = model.data.mesh
                self.scene.set_mesh(mesh["vertices"], mesh["simplices"], mesh["elem_order"], self.options)
                self._data_seen = model.data
                self._fields_key = None
            sel = model.selection
            key = (sel.n, sel.phase, sel.mode)
            if key != self._fields_key:
                self._fields = mode_fields_from_renderer(model.renderer, sel)
                self._fields_key = key
            if not self.playing:
                self._time_phase = float(sel.time_phase)
            self.scene.set_fields(self._fields, self._time_phase, self.options)
            self.error = ""
        except Exception as exc:  # noqa: BLE001 — 描けない結果はウィンドウに理由を出す
            self.error = f"{type(exc).__name__}: {exc}"
        if not self.can_animate:
            self.stop()
        self.state_changed.emit()

    def set_options(self, **patch) -> None:
        options = replace(self.options, **patch)
        if options == self.options:
            return
        self.options = options
        if self.scene is not None and self.active and self._data_seen is not None:
            try:
                self.scene.set_options(options)
                self.error = ""
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
        self.options_changed.emit()

    def view(self, name: str) -> None:
        if self.scene is not None:
            self.scene.view(name)

    # ---- アニメーション ---------------------------------------------------

    def play(self) -> bool:
        if not self.active or not self.can_animate or self.scene is None:
            return False
        self._timer.start()
        self.state_changed.emit()
        return True

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.state_changed.emit()

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def _tick(self) -> None:
        if self.scene is None or self._fields is None or not self.active:
            self.stop()
            return
        self._time_phase = (self._time_phase + 360.0 / ANIMATION_FRAMES) % 360.0
        try:
            self.scene.set_time_phase(self._time_phase)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop()
        self.state_changed.emit()

    # ---- 保存 -------------------------------------------------------------

    def save_png(self, path: str | Path) -> None:
        if self.scene is None:
            raise RuntimeError("3D ビューがありません")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.scene.render_png(path)

    def save_gif(self, path: str | Path, n_frames: int, fps: int, size_px: tuple[int, int],
                 camera=None, progress=None) -> bool:
        model = self.model
        if model.data is None or self._fields is None:
            raise RuntimeError("結果がありません")
        mesh = model.data.mesh
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return render_gif(mesh["vertices"], mesh["simplices"], mesh["elem_order"], self._fields, self.options, path,
                          n_frames=n_frames, fps=fps, size_px=size_px, camera=camera, progress=progress)

    def default_output(self, suffix: str) -> str:
        if self.model.path is None:
            return suffix
        return str(self.model.output_base(suffix))


class View3DPanel(ScrollPanel):
    """3D 表示の設定（VTK 非依存）. Signals: png_requested(), gif_requested()."""

    png_requested = QtCore.Signal()
    gif_requested = QtCore.Signal()

    def __init__(self, controller: View3DController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._refreshing = False
        self.setMinimumWidth(360)
        body = self.body

        self.title = label(f"<b>{tr('view3d.title')}</b>")
        body.addWidget(self.title)
        self.hint = label(tr("view3d.hint"), MUTED)
        body.addWidget(self.hint)
        self.status = label("", MUTED)
        body.addWidget(self.status)

        # --- 表示する要素 ---
        self.elements_box, layout = group_box(tr("view3d.elements"))
        self.wall_check = QtWidgets.QCheckBox(tr("view3d.wall"))
        self.wall_check.toggled.connect(lambda on: self._commit(show_wall=bool(on)))
        layout.addWidget(self.wall_check)
        row = QtWidgets.QHBoxLayout()
        self.opacity_label = QtWidgets.QLabel(tr("view3d.wallOpacity"))
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.valueChanged.connect(lambda v: self._commit(wall_opacity=v / 100.0))
        row.addWidget(self.opacity_label)
        row.addWidget(self.opacity_slider, 1)
        layout.addLayout(row)
        self.meridian_check = QtWidgets.QCheckBox(tr("view3d.meridian"))
        self.meridian_check.setToolTip(tr("view3d.meridianHint"))
        self.meridian_check.toggled.connect(lambda on: self._commit(show_meridian=bool(on)))
        layout.addWidget(self.meridian_check)
        self.slice_check = QtWidgets.QCheckBox(tr("view3d.slice"))
        self.slice_check.toggled.connect(lambda on: self._commit(show_slice=bool(on)))
        layout.addWidget(self.slice_check)
        row = QtWidgets.QHBoxLayout()
        self.slice_label = QtWidgets.QLabel(tr("view3d.sliceZ"))
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_slider.setRange(0, 1000)
        self.slice_slider.setValue(500)
        self.slice_slider.valueChanged.connect(self._on_slice_slider)
        self.slice_value = QtWidgets.QLabel("")
        row.addWidget(self.slice_label)
        row.addWidget(self.slice_slider, 1)
        row.addWidget(self.slice_value)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.sector_label = QtWidgets.QLabel(tr("view3d.sector"))
        self.sector_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sector_slider.setRange(30, 360)
        self.sector_slider.setSingleStep(15)
        self.sector_slider.setPageStep(45)
        self.sector_slider.valueChanged.connect(self._on_sector_slider)
        self.sector_value = QtWidgets.QLabel("360°")
        row.addWidget(self.sector_label)
        row.addWidget(self.sector_slider, 1)
        row.addWidget(self.sector_value)
        layout.addLayout(row)
        body.addWidget(self.elements_box)

        # --- 色 ---
        self.color_box, layout = group_box(tr("view3d.colorTitle"))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.color_combo = QtWidgets.QComboBox()
        for name in COLOR_FIELDS:
            self.color_combo.addItem(FIELD_LABELS[name], name)
        self.color_combo.currentIndexChanged.connect(
            lambda _i: self._commit(color_field=str(self.color_combo.currentData())))
        self.color_label = QtWidgets.QLabel(tr("view3d.colorField"))
        form.addRow(self.color_label, self.color_combo)
        self.nphi_spin = QtWidgets.QSpinBox()
        self.nphi_spin.setRange(12, 180)
        self.nphi_spin.setSingleStep(4)
        self.nphi_spin.valueChanged.connect(lambda v: self._commit(n_phi=int(v)))
        self.nphi_label = QtWidgets.QLabel(tr("view3d.nPhi"))
        form.addRow(self.nphi_label, self.nphi_spin)
        self.scalar_bar_check = QtWidgets.QCheckBox(tr("view3d.scalarBar"))
        self.scalar_bar_check.toggled.connect(lambda on: self._commit(show_scalar_bar=bool(on)))
        form.addRow("", self.scalar_bar_check)
        layout.addLayout(form)
        body.addWidget(self.color_box)

        # --- 矢印・電気力線 ---
        self.detail_box, layout = group_box(tr("view3d.detailTitle"))
        row = QtWidgets.QHBoxLayout()
        self.arrows_check = QtWidgets.QCheckBox(tr("view3d.arrows"))
        self.arrows_check.toggled.connect(lambda on: self._commit(show_arrows=bool(on)))
        self.arrow_field_combo = QtWidgets.QComboBox()
        self.arrow_field_combo.addItem("E", "E")
        self.arrow_field_combo.addItem("H", "H")
        self.arrow_field_combo.currentIndexChanged.connect(
            lambda _i: self._commit(arrow_field=str(self.arrow_field_combo.currentData())))
        self.arrow_mode_combo = QtWidgets.QComboBox()
        self.arrow_mode_combo.addItem(tr("view3d.arrowPlanes"), "planes")
        self.arrow_mode_combo.addItem(tr("view3d.arrowVolume"), "volume")
        self.arrow_mode_combo.currentIndexChanged.connect(
            lambda _i: self._commit(arrow_mode=str(self.arrow_mode_combo.currentData())))
        row.addWidget(self.arrows_check)
        row.addStretch(1)
        row.addWidget(self.arrow_field_combo)
        row.addWidget(self.arrow_mode_combo)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.arrow_nz_label = QtWidgets.QLabel(tr("view3d.arrowNz"))
        self.arrow_nz_spin = QtWidgets.QSpinBox()
        self.arrow_nz_spin.setRange(2, 200)
        self.arrow_nz_spin.setFixedWidth(84)
        self.arrow_nz_spin.setToolTip(tr("view3d.arrowNzHint"))
        self.arrow_nz_spin.valueChanged.connect(lambda v: self._commit(arrow_nz=int(v)))
        self.arrow_nxy_label = QtWidgets.QLabel(tr("view3d.arrowNxy"))
        self.arrow_nxy_spin = QtWidgets.QSpinBox()
        self.arrow_nxy_spin.setRange(2, 200)
        self.arrow_nxy_spin.setFixedWidth(84)
        self.arrow_nxy_spin.setToolTip(tr("view3d.arrowNxyHint"))
        self.arrow_nxy_spin.valueChanged.connect(lambda v: self._commit(arrow_nxy=int(v)))
        row.addSpacing(20)
        row.addWidget(self.arrow_nz_label)
        row.addWidget(self.arrow_nz_spin)
        row.addSpacing(12)
        row.addWidget(self.arrow_nxy_label)
        row.addWidget(self.arrow_nxy_spin)
        row.addStretch(1)
        layout.addLayout(row)
        self.elines_check = QtWidgets.QCheckBox(tr("view3d.eLines"))
        self.elines_check.toggled.connect(lambda on: self._commit(show_e_lines=bool(on)))
        layout.addWidget(self.elines_check)
        row = QtWidgets.QHBoxLayout()
        self.eline_levels_label = QtWidgets.QLabel(tr("view3d.eLineLevelsLabel"))
        self.eline_levels_spin = QtWidgets.QSpinBox()
        self.eline_levels_spin.setRange(2, 100)
        self.eline_levels_spin.setFixedWidth(84)
        self.eline_levels_spin.setToolTip(tr("view3d.eLineLevels"))
        self.eline_levels_spin.valueChanged.connect(lambda v: self._commit(e_line_levels=int(v)))
        self.eline_count_label = QtWidgets.QLabel(tr("view3d.eLineCountLabel"))
        self.eline_count_spin = QtWidgets.QSpinBox()
        self.eline_count_spin.setRange(1, 90)
        self.eline_count_spin.setFixedWidth(84)
        self.eline_count_spin.setToolTip(tr("view3d.eLineCount"))
        self.eline_count_spin.valueChanged.connect(lambda v: self._commit(e_line_count=int(v)))
        row.addSpacing(20)
        row.addWidget(self.eline_levels_label)
        row.addWidget(self.eline_levels_spin)
        row.addSpacing(12)
        row.addWidget(self.eline_count_label)
        row.addWidget(self.eline_count_spin)
        row.addStretch(1)
        layout.addLayout(row)
        self.elines_hint = label(tr("view3d.eLinesHint"), MUTED)
        layout.addWidget(self.elines_hint)
        body.addWidget(self.detail_box)

        # --- 視点・アニメーション・保存 ---
        self.actions_box, layout = group_box(tr("view3d.actionsTitle"))
        grid = QtWidgets.QGridLayout()
        self.view_buttons: dict[str, QtWidgets.QPushButton] = {}
        for i, name in enumerate(VIEWS):
            button = QtWidgets.QPushButton(tr(f"view3d.view.{name}"))
            button.clicked.connect(lambda _=False, v=name: self.controller.view(v))
            grid.addWidget(button, i // 2, i % 2)
            self.view_buttons[name] = button
        layout.addLayout(grid)
        row = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton(tr("view3d.play"))
        self.play_button.clicked.connect(self.controller.toggle_play)
        self.phase_label = label("", MUTED)
        row.addWidget(self.play_button)
        row.addWidget(self.phase_label, 1)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.png_button = QtWidgets.QPushButton(tr("results.ribbon.png"))
        self.png_button.clicked.connect(self.png_requested)
        self.gif_button = QtWidgets.QPushButton(tr("results.ribbon.gif"))
        self.gif_button.clicked.connect(self.gif_requested)
        row.addWidget(self.png_button)
        row.addWidget(self.gif_button)
        layout.addLayout(row)
        body.addWidget(self.actions_box)
        body.addStretch(1)

        controller.options_changed.connect(self.refresh)
        controller.state_changed.connect(self.refresh)
        self.refresh()

    # ---- 操作 -------------------------------------------------------------

    def _commit(self, **patch) -> None:
        if not self._refreshing:
            self.controller.set_options(**patch)

    def _bounds_z(self) -> tuple[float, float]:
        scene = self.controller.scene
        if scene is not None and getattr(scene, "has_mesh", False):
            return scene.bounds_z
        data = self.controller.model.data
        if data is not None:
            zmin, zmax, _r0, _r1 = data.bounds
            return zmin, zmax
        return 0.0, 1.0

    def _on_slice_slider(self, value: int) -> None:
        zmin, zmax = self._bounds_z()
        z = zmin + (zmax - zmin) * value / 1000.0
        self.slice_value.setText(f"{z:.4g} m")
        self._commit(slice_z=z)

    def _on_sector_slider(self, value: int) -> None:
        value = int(round(value / 15.0) * 15)
        self.sector_value.setText(f"{value}°")
        self._commit(sector_deg=float(value))

    # ---- 更新 -------------------------------------------------------------

    def refresh(self) -> None:
        self._refreshing = True
        try:
            self._refresh_impl()
        finally:
            self._refreshing = False

    def _refresh_impl(self) -> None:
        c = self.controller
        o = c.options
        model = c.model
        self.wall_check.setChecked(o.show_wall)
        self.opacity_slider.setValue(int(round(o.wall_opacity * 100)))
        self.meridian_check.setChecked(o.show_meridian)
        self.slice_check.setChecked(o.show_slice)
        zmin, zmax = self._bounds_z()
        if o.slice_z is not None and zmax > zmin:
            self.slice_slider.setValue(int(round(1000 * (o.slice_z - zmin) / (zmax - zmin))))
            self.slice_value.setText(f"{o.slice_z:.4g} m")
        else:
            self.slice_value.setText(f"{0.5 * (zmin + zmax):.4g} m")
        self.sector_slider.setValue(int(o.sector_deg))
        self.sector_value.setText(f"{int(o.sector_deg)}°")
        self.color_combo.setCurrentIndex(max(0, self.color_combo.findData(o.color_field)))
        self.nphi_spin.setValue(o.n_phi)
        self.scalar_bar_check.setChecked(o.show_scalar_bar)
        self.arrows_check.setChecked(o.show_arrows)
        self.arrow_field_combo.setCurrentIndex(max(0, self.arrow_field_combo.findData(o.arrow_field)))
        self.arrow_mode_combo.setCurrentIndex(max(0, self.arrow_mode_combo.findData(o.arrow_mode)))
        self.arrow_nz_spin.setValue(o.arrow_nz)
        self.arrow_nxy_spin.setValue(o.arrow_nxy)
        tm0 = c.is_tm0
        for w in (self.elines_check, self.eline_levels_label, self.eline_levels_spin, self.eline_count_label,
                  self.eline_count_spin, self.elines_hint):
            w.setVisible(tm0)
        self.elines_check.setChecked(o.show_e_lines)
        self.eline_levels_spin.setValue(o.e_line_levels)
        self.eline_count_spin.setValue(o.e_line_count)
        self.play_button.setEnabled(c.can_animate)
        self.play_button.setText(tr("view3d.stop") if c.playing else tr("view3d.play"))
        self.phase_label.setText(tr("view3d.timePhase", value=f"{c.time_phase:.0f}") if c.can_animate else "")
        self.png_button.setEnabled(model.has_data)
        self.gif_button.setEnabled(c.can_animate)
        if c.error:
            self.status.setText(c.error)
            self.status.setStyleSheet("color: #b91c1c;")
        elif not model.has_data:
            self.status.setText(tr("view3d.noResult"))
            self.status.setStyleSheet(MUTED)
        else:
            self.status.setText(tr("view3d.showing", title=model.data.title(model.selection)))
            self.status.setStyleSheet(MUTED)

    def retranslate(self) -> None:
        self.title.setText(f"<b>{tr('view3d.title')}</b>")
        self.hint.setText(tr("view3d.hint"))
        self.elements_box.setTitle(tr("view3d.elements"))
        self.wall_check.setText(tr("view3d.wall"))
        self.opacity_label.setText(tr("view3d.wallOpacity"))
        self.meridian_check.setText(tr("view3d.meridian"))
        self.meridian_check.setToolTip(tr("view3d.meridianHint"))
        self.slice_check.setText(tr("view3d.slice"))
        self.slice_label.setText(tr("view3d.sliceZ"))
        self.sector_label.setText(tr("view3d.sector"))
        self.color_box.setTitle(tr("view3d.colorTitle"))
        self.color_label.setText(tr("view3d.colorField"))
        self.nphi_label.setText(tr("view3d.nPhi"))
        self.scalar_bar_check.setText(tr("view3d.scalarBar"))
        self.detail_box.setTitle(tr("view3d.detailTitle"))
        self.arrows_check.setText(tr("view3d.arrows"))
        self.arrow_mode_combo.setItemText(0, tr("view3d.arrowPlanes"))
        self.arrow_mode_combo.setItemText(1, tr("view3d.arrowVolume"))
        self.arrow_nz_label.setText(tr("view3d.arrowNz"))
        self.arrow_nxy_label.setText(tr("view3d.arrowNxy"))
        self.arrow_nz_spin.setToolTip(tr("view3d.arrowNzHint"))
        self.arrow_nxy_spin.setToolTip(tr("view3d.arrowNxyHint"))
        self.elines_check.setText(tr("view3d.eLines"))
        self.eline_levels_spin.setToolTip(tr("view3d.eLineLevels"))
        self.eline_count_spin.setToolTip(tr("view3d.eLineCount"))
        self.eline_levels_label.setText(tr("view3d.eLineLevelsLabel"))
        self.eline_count_label.setText(tr("view3d.eLineCountLabel"))
        self.elines_hint.setText(tr("view3d.eLinesHint"))
        self.actions_box.setTitle(tr("view3d.actionsTitle"))
        for name, button in self.view_buttons.items():
            button.setText(tr(f"view3d.view.{name}"))
        self.png_button.setText(tr("results.ribbon.png"))
        self.gif_button.setText(tr("results.ribbon.gif"))
        self.refresh()


def make_interactor(parent=None):
    """pyvistaqt の QtInteractor（VTK のウィンドウ。offscreen では作らない）."""
    os.environ.setdefault("QT_API", "pyside6")
    from pyvistaqt import QtInteractor

    class View3DInteractor(QtInteractor):
        def keyPressEvent(self, event):                      # noqa: N802 — Qt
            key = event.key()
            if key == QtCore.Qt.Key_F:
                self.reset_camera()
                self.render()
                event.accept()
            elif key == QtCore.Qt.Key_Escape:
                super().keyPressEvent(event)
            else:
                event.accept()                                # VTK 既定のキー（w / e / q …）を渡さない

    return View3DInteractor(parent)


class View3DWindow(ToolWindow):
    """左に 3D ビュー、右に設定パネル。`controller` は MainWindow が持つ."""

    def __init__(self, controller: View3DController, parent=None):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.interactor = make_interactor(content)
        from .view3d_scene import Scene3D

        self.scene = Scene3D(self.interactor)
        self.panel = View3DPanel(controller)
        self.panel.setFixedWidth(400)
        layout.addWidget(self.interactor, 1)
        layout.addWidget(self.panel)
        super().__init__("view3d.title", content, parent, offset=(40, 30), settings_key="view3dWindow")
        self.controller = controller
        controller.attach(self.scene)
        self.panel.png_requested.connect(self.save_png)
        self.panel.gif_requested.connect(self.save_gif)

    def canvas_size(self) -> tuple[int, int]:
        size = self.interactor.size()
        return max(int(size.width()), 400), max(int(size.height()), 300)

    def open_window(self) -> None:
        super().open_window()
        self.controller.set_active(True)

    def hideEvent(self, event) -> None:
        self.controller.set_active(False)
        super().hideEvent(event)

    def save_png(self) -> None:
        if not self.controller.model.has_data:
            return
        default = self.controller.default_output("_3d") + ".png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("results.pngTitle"), default, tr("results.pngFilter"))
        if not path:
            return
        try:
            self.controller.save_png(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, tr("results.pngTitle"), tr("file.exportError", message=str(exc)))
            return
        self.panel.status.setText(tr("results.pngSaved", path=path))

    def save_gif(self) -> None:
        controller = self.controller
        if not controller.can_animate:
            QtWidgets.QMessageBox.information(self, tr("results.ribbon.gif"), tr("results.gifNeedsTraveling"))
            return
        from .dialogs.gif import GifDialog

        dialog = GifDialog(controller.default_output("_3d_anim") + ".gif", parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        params = dialog.params()
        n_frames = params["n_frames"]
        progress = QtWidgets.QProgressDialog(tr("gif.rendering"), tr("gif.cancel"), 0, n_frames, self)
        progress.setWindowTitle(tr("gif.title"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        def step(i: int, n: int) -> bool:
            progress.setValue(i)
            progress.setLabelText(tr("gif.frame", i=i + 1, n=n, theta=f"{360.0 * i / n:.1f}"))
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            ok = controller.save_gif(params["output_path"], n_frames, params["fps"], self.canvas_size(),
                                     camera=self.interactor.camera_position, progress=step)
        except Exception as exc:  # noqa: BLE001
            progress.close()
            QtWidgets.QMessageBox.critical(self, tr("gif.title"), tr("results.gifError", message=str(exc)))
            return
        progress.close()
        self.panel.status.setText(tr("results.gifSaved", path=params["output_path"], frames=n_frames,
                                     fps=params["fps"]) if ok else tr("results.gifCancelled"))
