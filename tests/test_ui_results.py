"""結果の UI（offscreen）: ResultModel（読み込み・選択・オプション・追従）、ResultView（描画・エラー・PNG）、
ResultsPanel（n / 位相 / モード / 時間位相、モード表、オプション、HOM で E 線が消える）、書き出し・GIF のダイアログ."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("h5py")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtWidgets                                    # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore            # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language, tr           # noqa: E402
from axicavity_fem.gui.io.axiproj import ResultEntry              # noqa: E402
from axicavity_fem.gui.jobs.analysis_controller import AnalysisController  # noqa: E402
from axicavity_fem.gui.jobs.mesh_controller import MeshController  # noqa: E402
from axicavity_fem.gui.jobs.session import JobSession            # noqa: E402
from axicavity_fem.gui.project.controller import ProjectController  # noqa: E402
from axicavity_fem.gui.ui.dialogs.export_field import ExportFieldDialog  # noqa: E402
from axicavity_fem.gui.ui.dialogs.gif import GifDialog           # noqa: E402
from axicavity_fem.gui.ui.result_model import ResultModel, result_source  # noqa: E402
from axicavity_fem.gui.ui.result_renderer import Selection, ViewOptions  # noqa: E402
from axicavity_fem.gui.ui.result_view import ResultView          # noqa: E402
from axicavity_fem.gui.ui.results_panel import ResultsPanel      # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
TM0_SW = SAMPLES / "diel_simple1_SW_TM0_processed.h5"
TM0_TW = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
HOM_SW = SAMPLES / "s-band_1cell_SW_HOM_processed.h5"
RAW = SAMPLES / "s-band_1cell_SW_TM0.h5"

pytestmark = pytest.mark.skipif(not all(p.exists() for p in (TM0_SW, TM0_TW, HOM_SW, RAW)),
                                reason="samples のサンプル結果なし")


@pytest.fixture
def env(qtbot):
    set_language("ja")
    store = DocumentStore()
    project = ProjectController(store, generator="test")
    project.new()
    session = JobSession()
    analysis = AnalysisController(store, session, project, MeshController(store, session))
    model = ResultModel()
    view = ResultView(model)
    panel = ResultsPanel(store, model, analysis, project)
    qtbot.addWidget(view)
    qtbot.addWidget(panel)
    view.resize(640, 480)
    view.show()
    panel.show()
    qtbot.waitExposed(view)
    yield store, project, model, view, panel
    session.shutdown()


def _entry(tmp_path, source: Path, kind="tm0-sw", status="done") -> ResultEntry:
    folder = tmp_path / f"0001-{kind}"
    folder.mkdir(exist_ok=True)
    raw = folder / "model_SW_TM0.h5"
    processed = folder / "model_SW_TM0_processed.h5"
    raw.write_bytes(source.read_bytes())
    processed.write_bytes(source.read_bytes())
    return ResultEntry(id=folder.name, dir=folder, number=1, kind=kind, status=status, finished_at="t1",
                       files={"raw": raw.name, "processed": processed.name})


def test_model_load_select_options_and_follow(env, tmp_path, qtbot):
    store, project, model, view, panel = env
    assert not model.has_data and view.last_error == "" and tr("results.noResult") in panel.source_label.text()
    loaded, selected, options = [], [], []
    model.loaded.connect(lambda: loaded.append(model.path))
    model.selection_changed.connect(lambda: selected.append(model.selection))
    model.options_changed.connect(lambda: options.append(model.options))

    assert model.load_file(TM0_TW) and model.has_data and model.is_traveling and not model.is_hom
    assert model.selection == Selection(n=0, phase=120.0, mode=0, time_phase=0.0) and loaded == [TM0_TW]
    qtbot.waitUntil(lambda: view.last_title.startswith("Mode 0"), timeout=30_000)
    assert view.last_error == ""
    model.set_selection(mode=1, time_phase=90.0)
    assert selected[-1].mode == 1 and selected[-1].time_phase == 90.0
    model.set_selection(mode=1)                                     # 変わらなければ出さない
    assert len(selected) == 1
    model.set_selection(mode=50, phase=999.0)                       # 丸められる
    assert model.selection.mode == model.data.num_modes() - 1 and model.selection.phase == 120.0
    model.set_options(show_mesh=True, nz=8)
    assert options[-1].show_mesh and options[-1].nz == 8 and options[-1] == model.options
    assert model.output_base("_anim").name == "s-band_1cell_TW_TM0_processed_anim_m1"
    assert model.output_base("").parent == TM0_TW.parent          # 外部 h5 はその隣

    # HOM を読むと E 線 off / ベクトル on（ver2.3 の既定）。プロジェクトの結果は exports/ に
    assert model.load_file(HOM_SW) and model.is_hom and not model.options.show_lines and model.options.show_vectors
    assert model.selection.n == 0 and model.selection.phase is None
    entry = _entry(tmp_path, TM0_SW)
    assert result_source(entry).name == "model_SW_TM0_processed.h5"
    assert model.load_entry(entry) and model.entry is entry and model.path == entry.dir / "model_SW_TM0_processed.h5"
    assert model.output_base("_field_area") == entry.dir / "exports" / "model_SW_TM0_processed_field_area_m0"
    # 追従: 同じ状態なら読み直さない、post 後（finished_at が変わる）は読み直す、消えたら空に
    count = len(loaded)
    model.set_selection(mode=2)
    model.refresh_entry(entry)
    assert len(loaded) == count and model.selection.mode == 2
    newer = ResultEntry(**{**entry.__dict__, "finished_at": "t2"})
    model.refresh_entry(newer)
    assert len(loaded) == count + 1 and model.entry is newer and model.selection.mode == 2   # 選択は保つ
    model.refresh_entry(None)
    assert not model.has_data and model.entry is None and model.path is None
    qtbot.waitUntil(lambda: view.last_title == "", timeout=10_000)

    # 壊れた入力はエラー表示（例外にしない）
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an h5")
    assert not model.load_file(bad) and model.error and not model.has_data
    qtbot.waitUntil(lambda: view.last_error == model.error, timeout=10_000)
    assert panel.error_label.isVisible()
    missing = ResultEntry(id="0002-tm0-sw", dir=tmp_path / "0002-tm0-sw", number=2, kind="tm0-sw")
    assert not model.load_entry(missing) and "h5" in model.error


def test_panel_controls_and_table(env, qtbot):
    store, project, model, view, panel = env
    assert model.load_file(TM0_TW)
    qtbot.waitUntil(lambda: view.last_title != "", timeout=30_000)
    assert not panel.n_combo.isVisible() and panel.phase_combo.isVisible() and panel.time_phase_spin.isVisible()
    assert panel.phase_combo.currentText() == "120°" and panel.mode_combo.count() == model.data.num_modes()
    headers = [panel.table.horizontalHeaderItem(j).text() for j in range(panel.table.columnCount())]
    assert headers[:3] == ["#", "f [GHz]", "Q"] and "R/Q [Ω]" in headers and "v_g [m/s]" in headers
    assert panel.table.rowCount() == model.data.num_modes() and not panel.table_hint.isVisible()
    assert panel.table.item(0, 1).text() == "2.856332"
    # 行を選ぶ → モードが変わる → コンボも追従。時間位相のスピン → 選択
    panel.table.selectRow(1)
    assert model.selection.mode == 1 and panel.mode_combo.currentIndex() == 1
    panel.mode_combo.setCurrentIndex(0)
    assert model.selection.mode == 0 and panel.table.selectionModel().selectedRows()[0].row() == 0
    panel.time_phase_spin.setValue(30.0)
    assert model.selection.time_phase == 30.0
    # オプション
    panel.lines_check.setChecked(False)
    panel.levels_spin.setValue(7)
    panel.vectors_check.setChecked(True)
    panel.nz_spin.setValue(5)
    panel.e_wall_check.setChecked(True)
    assert model.options == ViewOptions(show_color=True, show_lines=False, levels=7, show_vectors=True, nz=5, nr=15,
                                        show_mesh=False, show_e_wall=True)
    model.set_options(show_color=False)
    assert not panel.color_check.isChecked()
    qtbot.waitUntil(lambda: "30" in view.last_title or view.last_title != "", timeout=30_000)
    panel.show_field_values(["z = 1", "Ez = 2"])
    assert "Ez = 2" in panel.field_label.text()

    # post 無し: 表は # と f だけ、案内が出る
    assert model.load_file(RAW)
    assert panel.table.columnCount() == 2 and panel.table_hint.isVisible()
    assert not panel.phase_combo.isVisible() and not panel.time_phase_spin.isVisible()
    # HOM: n が出て E 線 / E-wall が消える
    assert model.load_file(HOM_SW)
    assert panel.n_combo.isVisible() and panel.n_combo.count() == 3
    assert not panel.lines_check.isVisible() and not panel.e_wall_check.isVisible()
    assert panel.color_check.text() == tr("results.hColorHom")
    panel.n_combo.setCurrentIndex(2)
    assert model.selection.n == 2 and model.selection.mode == 0
    assert panel.mode_combo.itemText(0).startswith("Mode 0: 5.353")
    qtbot.waitUntil(lambda: view.last_title.startswith("HOM n=2"), timeout=30_000)


def test_view_png_and_pick(env, tmp_path, qtbot):
    store, project, model, view, panel = env
    picked = []
    view.point_picked.connect(lambda z, r: picked.append((z, r)))
    assert model.load_file(TM0_SW)
    qtbot.waitUntil(lambda: view.last_title != "", timeout=30_000)
    out = tmp_path / "fig.png"
    view.save_png(out, dpi=50)
    assert out.exists() and out.stat().st_size > 1000
    w, h = view.canvas_size()
    assert w >= 400 and h >= 300

    class Event:
        dblclick = True
        xdata = 0.02
        ydata = 0.01
    view._on_click(Event())
    assert picked == [(0.02, 0.01)]
    Event.dblclick = False
    view._on_click(Event())
    assert len(picked) == 1


def test_export_field_dialog(qtbot):
    set_language("ja")
    dialog = ExportFieldDialog("area", "out", (0.0, 0.1), (0.0, 0.05), traveling_tm0=True, has_post=True)
    qtbot.addWidget(dialog)
    dialog.nz_spin.setValue(50)
    dialog.nr_spin.setValue(30)
    dialog.power_edit.setText("1000")
    dialog.instant_check.setChecked(True)
    p = dialog.params()
    assert p["z_range"] == (0.0, 0.1) and p["r_range"] == (0.0, 0.05) and p["nz"] == 50 and p["nr"] == 30
    assert p["scale_to_power"] == 1000.0 and p["fmt"] == "both" and p["output"] == "out" and p["instant"]
    assert p["scale"] == 1.0
    dialog.edits["zmax"].setText("-1")
    with pytest.raises(ValueError):
        dialog.params()
    dialog._accept()                                              # 不正なら閉じずにエラーを出す
    assert not dialog.error_label.isHidden() and dialog.result() != QtWidgets.QDialog.Accepted

    dialog = ExportFieldDialog("line", "ln", (0.0, 0.1), (0.0, 0.05))
    qtbot.addWidget(dialog)
    assert dialog.instant_check.isHidden()
    dialog.edits["p1r"].setText("0.01")
    dialog.edits["p2r"].setText("0.04")
    dialog.npts_spin.setValue(123)
    dialog.format_combo.setCurrentIndex(2)
    p = dialog.params()
    assert p["p1"] == (0.0, 0.01) and p["p2"] == (0.1, 0.04) and p["npts"] == 123 and p["scale_to_power"] is None
    assert p["fmt"] == "txt" and not p["instant"]

    dialog = ExportFieldDialog("axis", "ax", (0.0, 0.1), (0.0, 0.05), has_post=False)
    qtbot.addWidget(dialog)
    assert not dialog.power_edit.isEnabled()
    dialog.edits["zmin"].setText("0.01")
    dialog.scale_edit.setText("2.5")
    p = dialog.params()
    assert p["z_range"] == (0.01, 0.1) and p["npts"] == 500 and "r_range" not in p and p["scale"] == 2.5


def test_gif_dialog(qtbot):
    set_language("ja")
    dialog = GifDialog("a.gif", n_frames=12, fps=6)
    qtbot.addWidget(dialog)
    dialog.frames_spin.setValue(8)
    assert dialog.params() == {"n_frames": 8, "fps": 6, "output_path": "a.gif"}
