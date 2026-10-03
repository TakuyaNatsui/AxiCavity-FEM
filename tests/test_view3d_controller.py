"""3D 表示のコントローラとパネル（gui/ui/view3d_window.py。VTK を使わずに offscreen で）:
結果モデルへの追従（デバウンス）、メッシュと場の差し替え、オプション、アニメーション、パネルの結線."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("h5py")
pytest.importorskip("pytestqt")
from PySide6 import QtWidgets                                     # noqa: E402

from axicavity_fem.gui.i18n.tr import set_language, tr           # noqa: E402
from axicavity_fem.gui.ui.result_model import ResultModel          # noqa: E402
from axicavity_fem.gui.ui.view3d_scene import View3DOptions        # noqa: E402
from axicavity_fem.gui.ui.view3d_window import ANIMATION_FRAMES, View3DController, View3DPanel, is_available  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
TM0_TW = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
HOM_SW = SAMPLES / "s-band_1cell_SW_HOM_processed.h5"

pytestmark = pytest.mark.skipif(not (TM0_TW.exists() and HOM_SW.exists()), reason="サンプル結果なし")


class FakeScene:
    """Scene3D の代わり（呼ばれた内容を記録する）."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.has_mesh = False
        self.bounds_z = (0.0, 0.0)
        self.fail = False

    def set_mesh(self, vertices, simplices, elem_order, options=None):
        self.has_mesh = True
        v = np.asarray(vertices)
        self.bounds_z = (float(v[:, 0].min()), float(v[:, 0].max()))
        self.calls.append(("mesh", len(v), elem_order))

    def set_fields(self, fields, time_phase, options=None):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(("fields", fields.n, fields.traveling, time_phase))

    def set_options(self, options):
        self.calls.append(("options", options))

    def set_time_phase(self, time_phase):
        self.calls.append(("phase", time_phase))

    def view(self, name):
        self.calls.append(("view", name))

    def render_png(self, path):
        Path(path).write_bytes(b"png")

    def clear(self):
        self.has_mesh = False
        self.calls.append(("clear",))


@pytest.fixture
def env(qtbot):
    set_language("ja")
    model = ResultModel()
    controller = View3DController(model)
    scene = FakeScene()
    controller.attach(scene)
    panel = View3DPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    return model, controller, scene, panel


def test_is_available_reports_installed_packages():
    assert isinstance(is_available(), bool)


def test_controller_follows_model_when_active(env, qtbot):
    model, controller, scene, panel = env
    assert model.load_file(TM0_TW)
    qtbot.wait(250)
    assert scene.calls == []                                       # ウィンドウが見えていなければ描かない
    controller.set_active(True)
    kinds = [c[0] for c in scene.calls]
    assert kinds == ["mesh", "fields"] and scene.calls[1][1:] == (0, True, 0.0)
    assert controller.can_animate and controller.is_tm0 and not controller.playing
    # 選択が変わるとデバウンス後に場だけ差し替える（メッシュは同じ）
    model.set_selection(mode=1, time_phase=45.0)
    qtbot.waitUntil(lambda: scene.calls[-1] == ("fields", 0, True, 45.0), timeout=3000)
    assert [c[0] for c in scene.calls].count("mesh") == 1
    # 別の結果を読むとメッシュから作り直す。HOM では E 線の項目が消える
    assert model.load_file(HOM_SW)
    qtbot.waitUntil(lambda: [c[0] for c in scene.calls].count("mesh") == 2, timeout=3000)
    assert scene.calls[-1][1:3] == (0, False) and not controller.can_animate and not controller.is_tm0
    assert not panel.elines_check.isVisible() and not panel.play_button.isEnabled()
    # 結果を消すと clear
    model.clear()
    qtbot.waitUntil(lambda: scene.calls[-1] == ("clear",), timeout=3000)
    assert tr("view3d.noResult") in panel.status.text()
    # 描画の失敗はエラー表示（例外にしない）
    assert model.load_file(TM0_TW)
    scene.fail = True
    qtbot.waitUntil(lambda: "boom" in controller.error, timeout=3000)
    assert "boom" in panel.status.text()


def test_options_and_animation(env, qtbot):
    model, controller, scene, panel = env
    assert model.load_file(TM0_TW)
    controller.set_active(True)
    n0 = len(scene.calls)
    controller.set_options(sector_deg=270.0, show_slice=True)
    assert controller.options.sector_deg == 270.0 and scene.calls[-1][0] == "options"
    controller.set_options(sector_deg=270.0)                       # 変わらなければ何もしない
    assert len(scene.calls) == n0 + 1
    # アニメーション: 1 フレームごとに時間位相が進む
    assert controller.play() and controller.playing
    qtbot.waitUntil(lambda: sum(1 for c in scene.calls if c[0] == "phase") >= 2, timeout=3000)
    phases = [c[1] for c in scene.calls if c[0] == "phase"]
    assert phases[0] == pytest.approx(360.0 / ANIMATION_FRAMES) and phases[1] == pytest.approx(2 * 360.0 / ANIMATION_FRAMES)
    controller.stop()
    assert not controller.playing
    controller.set_active(False)                                   # 隠すと止まる・描かない
    controller.play()
    assert not controller.playing
    controller.view("front")
    assert scene.calls[-1] == ("view", "front")


def test_panel_writes_options_and_reads_them_back(env, qtbot):
    model, controller, scene, panel = env
    assert model.load_file(TM0_TW)
    controller.set_active(True)
    panel.wall_check.setChecked(False)
    panel.slice_check.setChecked(True)
    panel.sector_slider.setValue(180)
    panel.color_combo.setCurrentIndex(panel.color_combo.findData("Hphi"))
    panel.arrows_check.setChecked(True)
    panel.arrow_mode_combo.setCurrentIndex(1)
    panel.arrow_nz_spin.setValue(12)
    panel.arrow_nxy_spin.setValue(16)
    panel.elines_check.setChecked(True)
    panel.eline_count_spin.setValue(24)
    panel.nphi_spin.setValue(36)
    o = controller.options
    assert not o.show_wall and o.show_slice and o.sector_deg == 180.0 and o.color_field == "Hphi"
    assert o.show_arrows and o.arrow_mode == "volume" and o.arrow_nz == 12 and o.arrow_nxy == 16
    assert o.show_e_lines and o.e_line_count == 24 and o.n_phi == 36
    panel.slice_slider.setValue(250)
    zmin, zmax = scene.bounds_z
    assert controller.options.slice_z == pytest.approx(zmin + 0.25 * (zmax - zmin))
    # コントローラ側の変更はパネルに戻る
    controller.set_options(show_wall=True, sector_deg=90.0)
    assert panel.wall_check.isChecked() and panel.sector_value.text() == "90°"
    assert panel.play_button.text() == tr("view3d.play") and panel.gif_button.isEnabled()
    panel.retranslate()
    assert panel.wall_check.text() == tr("view3d.wall")
    # PNG（偽の scene が書く）
    out = Path(os.environ.get("TMP", ".")) / "axicavity_view3d_test.png"
    controller.save_png(out)
    assert out.exists()
    out.unlink()
    controller.set_options(**{k: v for k, v in View3DOptions().__dict__.items()})
    assert controller.options == View3DOptions()
