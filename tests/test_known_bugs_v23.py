"""ver2.3 の wx GUI にあった既知バグ 14 件が ver3 で再現しないことの回帰テスト（計画 §8.6「既知バグ」の行）.

| # | ver2.3 の症状 | ver3 の対応 |
|---|---|---|
| 1 | Exit が無反応 | 未保存なら 保存 / 破棄 / キャンセル を聞いて終了する |
| 2 | Ctrl+A が Superfish 書き出しと衝突 | Ctrl+A は全選択（入力欄にフォーカスがあるときは奪わない） |
| 3 | Reset で変数が残る | 新規は文書ごと初期化（パラメータ・境界条件・領域も空） |
| 4 | 単位変更がラベルだけ | 換算ダイアログ（数値を換算する / ラベルだけ） |
| 5 | ループの検証が無い | check_geometry（閉領域なし・開いた線・r<0 …） |
| 6 | GEO 書き出しが次数を無視 | mesh_order を引数で渡す（Mesh.ElementOrder） |
| 7 | メッシュ次数と要素次数の不整合 | 単一の設定 document.mesh.order |
| 8 | レポートの場所を推測して探す | result.json の files / 決まった候補 |
| 9 | command.log の置き場が曖昧 | データフォルダ <名前>.axiproj.data/command.log |
| 10 | 進行波で最初の位相の値だけ表示 | 選んだ位相の post 値 |
| 11 | R/Q・V_eff が表示されない | モード表に列がある |
| 12 | E-wall が HOM でも押せる | HOM では隠す（描画も無視） |
| 13 | GIF が止められない | 進捗ダイアログで中止できる |
| 14 | 場の書き出しに instant / scale が無い | ダイアログと argv にある |
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtGui, QtWidgets                          # noqa: E402

from axicavity_fem.gui.core.convert import check_geometry, to_multi_region   # noqa: E402
from axicavity_fem.gui.core.document import SketchLine, SketchPoint          # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_line, point_pos          # noqa: E402
from axicavity_fem.gui.io.axiproj import ResultEntry           # noqa: E402
from axicavity_fem.gui.jobs.analysis_controller import AnalysisController  # noqa: E402
from axicavity_fem.gui.jobs.commands import build_commands, export_argv    # noqa: E402
from axicavity_fem.gui.ui.dialogs.export_field import ExportFieldDialog    # noqa: E402
from axicavity_fem.gui.ui.result_renderer import (              # noqa: E402
    MODE_COLUMNS,
    ResultData,
    ResultRenderer,
    Selection,
    ViewOptions,
    mode_rows,
    render_gif,
)
from test_ui_main_window import SAMPLES, draw_rectangle, win   # noqa: E402, F401  (fixture)
from test_ui_results import env                                 # noqa: E402, F401  (fixture)

TM0_SW = SAMPLES / "diel_simple1_SW_TM0_processed.h5"
TM0_TW = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
HOM_SW = SAMPLES / "s-band_1cell_SW_HOM_processed.h5"


def test_01_exit_asks_to_save(qtbot, win, monkeypatch):
    draw_rectangle(qtbot, win)
    assert win.project.is_dirty()
    answers = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: answers.append(a[2]) or QtWidgets.QMessageBox.Cancel))
    win.close()
    assert answers and win.isVisible()                                 # キャンセル → 閉じない
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Discard))
    win.close()
    assert not win.isVisible()                                         # 破棄 → 閉じる


def test_02_ctrl_a_selects_all(qtbot, win):
    draw_rectangle(qtbot, win)
    store = win.store
    store.set_selection([])
    assert win.select_all_action.shortcut() == QtGui.QKeySequence("Ctrl+A")
    others = [a for a in win.findChildren(QtGui.QAction) if a is not win.select_all_action
              and a.shortcut() == QtGui.QKeySequence("Ctrl+A")]
    assert others == []                                                # 書き出しなどと衝突しない
    win.select_all_action.trigger()
    user = {e.id for e in store.active_sketch().entities if not e.projection}
    assert set(store.selection) == user and len(store.selection) == 7     # 参照の原点・軸は選ばない
    store.set_selection([])
    win.analysis_panel.modes_edit.setFocus()                           # 入力欄で文字を打っているとき
    if QtWidgets.QApplication.focusWidget() is win.analysis_panel.modes_edit:
        win.select_all_action.trigger()
        assert store.selection == []


def test_03_new_project_clears_everything(qtbot, win):
    store = win.store
    draw_rectangle(qtbot, win)
    pid = store.add_param()
    store.update_param(pid, name="a", expression="10")
    lines = [e.id for e in store.active_sketch().entities if isinstance(e, SketchLine) and not e.projection]
    store.set_boundary(lines[:1], "E-short")
    store.set_region(store.profiles[0].id, name="Ceramic", epsR=9.0)
    assert store.document.params and store.document.boundaries and store.document.regions
    win.project.new()
    doc = store.document
    assert doc.params == [] and doc.boundaries == {} and doc.regions == []
    assert [e for e in doc.sketch.entities if not e.projection] == [] and doc.sketch.constraints == []
    assert not store.can_undo() and store.selection == []


def test_04_unit_change_converts_or_relabels(qtbot, win):
    store = win.store
    draw_rectangle(qtbot, win, 0, 0, 40, 20)
    sketch = store.active_sketch()
    xs = sorted({point_pos(sketch, e.id)[0] for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection})
    assert xs == [0, 40] and store.document.meta.units == "mm"
    assert store.set_units("cm", convert_values=True)
    sketch = store.active_sketch()
    xs = sorted({round(point_pos(sketch, e.id)[0], 9) for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection})
    assert xs == [0, 4] and store.document.meta.units == "cm"       # 数値を換算
    assert store.set_units("m", convert_values=False)
    sketch = store.active_sketch()
    xs = sorted({round(point_pos(sketch, e.id)[0], 9) for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection})
    assert xs == [0, 4] and store.document.meta.units == "m"        # ラベルだけ
    assert to_multi_region(store.document).geom.unit == "m"


def test_05_geometry_is_validated(qtbot, win):
    store = win.store
    assert {i.code for i in check_geometry(store.document) if i.level == "error"} == {"no_profile"}
    draw_rectangle(qtbot, win, 0, 0, 40, 20)
    assert check_geometry(store.document) == []
    store.update_active_sketch(lambda s: add_line(s, (60, 5), (80, 15)))   # 開いた線
    codes = {i.code for i in check_geometry(store.document)}
    assert "unused_curve" in codes
    store.update_active_sketch(lambda s: add_line(s, (0, -10), (40, -10)))  # r < 0
    assert "negative_r" in {i.code for i in check_geometry(store.document) if i.level == "error"}
    assert (win.analysis.check()[0] != [])                              # 解析は実行できない


def test_06_geo_export_honours_mesh_order(qtbot, win, monkeypatch, tmp_path):
    draw_rectangle(qtbot, win, 0, 0, 100, 50)
    for order in (2, 1):
        win.store.set_mesh(order=order)
        path = tmp_path / f"o{order}.geo"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
        win._on_export_geo()
        text = path.read_text(encoding="utf-8")
        assert f"Mesh.ElementOrder = {order};" in text


def test_07_single_order_setting_for_mesh_and_solver(qtbot, win):
    store = win.store
    draw_rectangle(qtbot, win, 0, 0, 100, 50)
    for order, index in ((1, 0), (2, 1)):
        store.set_mesh(order=order)
        cmds = build_commands(store.document)
        assert cmds["solve"][cmds["solve"].index("--elem-order") + 1] == str(order)
        assert to_multi_region(store.document).geom.settings["mesh_order"] == index
        assert str(order) in win.analysis_panel.order_value.text()
        assert win.mesh_panel.order_combo.currentData() == order


def test_08_report_is_found_from_result_json(tmp_path):
    folder = tmp_path / "0001-tm0-sw"
    folder.mkdir()
    entry = ResultEntry(id=folder.name, dir=folder, number=1, kind="tm0-sw", status="done",
                        files={"raw": "model_SW_TM0.h5", "processed": "model_SW_TM0_processed.h5"})
    assert AnalysisController.report_path(entry) is None
    report = folder / "model_SW_TM0_processed_report" / "index.html"
    report.parent.mkdir()
    report.write_text("<html></html>", encoding="utf-8")
    assert AnalysisController.report_path(entry) == report                # 決まった候補
    custom = folder / "custom" / "index.html"
    custom.parent.mkdir()
    custom.write_text("<html></html>", encoding="utf-8")
    entry.files["report"] = "custom/index.html"
    assert AnalysisController.report_path(entry) == custom                 # result.json の files が優先
    assert AnalysisController.report_path(None) is None


def test_09_command_log_lives_in_data_folder(qtbot, win, tmp_path):
    project = win.project
    assert project.log_command("axicavity-fem solve") is None            # 未保存なら書かない
    project.save_as(tmp_path / "p.axiproj")
    written = project.log_command("axicavity-fem solve --type tm0")
    assert written == tmp_path / "p.axiproj.data" / "command.log"
    assert "axicavity-fem solve --type tm0" in written.read_text(encoding="utf-8")


def _traveling_data(tmp_path):
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    modeset = {"frequencies": np.array([1.0, 2.0]), "eigenvalues": None, "eigenvectors": np.zeros((2, 3), complex)}
    raw = {"solver_type": "tm0",
           "mesh": {"vertices": vertices, "simplices": np.array([[0, 1, 2]]), "elem_order": 1},
           "results_by_n": {0: {"traveling": {60.0: modeset, 120.0: modeset}}},
           "post_process": {0: {"traveling": {60.0: [{"Q": 60.0}, {"Q": 61.0}], 120.0: [{"Q": 120.0}, {"Q": 121.0}]}}}}
    return ResultData(tmp_path / "fake.h5", raw=raw)


def test_10_traveling_wave_uses_selected_phase(tmp_path):
    data = _traveling_data(tmp_path)
    assert data.phases() == [60.0, 120.0] and data.is_traveling
    assert data.post_params(Selection(phase=60.0, mode=1))["Q"] == 61.0
    assert data.post_params(Selection(phase=120.0, mode=1))["Q"] == 121.0    # 2 番目の位相の値
    assert data.post_params(data.normalize(Selection()))["Q"] == 60.0        # 既定は先頭
    assert [r["Q"] for r in mode_rows(data, 0, 120.0)] == [120.0, 121.0]


def test_11_r_over_q_and_v_eff_are_listed(env, qtbot):
    if not TM0_SW.exists():
        pytest.skip("サンプル結果なし")
    keys = [k for k, _n, _u in MODE_COLUMNS]
    assert "R_over_Q" in keys and "V_eff" in keys
    rows = mode_rows(ResultData(TM0_SW))
    assert rows[0]["R_over_Q"] > 0 and rows[0]["V_eff"] > 0
    store, project, model, view, panel = env
    assert model.load_file(TM0_SW)
    headers = [panel.table.horizontalHeaderItem(j).text() for j in range(panel.table.columnCount())]
    assert "R/Q [Ω]" in headers and "V_eff [V]" in headers


def test_12_e_wall_is_hidden_for_hom(env, qtbot):
    if not (TM0_SW.exists() and HOM_SW.exists()):
        pytest.skip("サンプル結果なし")
    store, project, model, view, panel = env
    assert model.load_file(TM0_SW) and not panel.e_wall_check.isHidden()
    assert model.load_file(HOM_SW) and panel.e_wall_check.isHidden() and panel.lines_check.isHidden()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure()
    FigureCanvasAgg(fig)
    ResultRenderer(model.data).draw(fig, Selection(n=1), ViewOptions(show_e_wall=True))   # HOM では無視される
    assert len(fig.axes) == 4


def test_13_gif_can_be_cancelled(tmp_path, monkeypatch, qtbot):
    if not TM0_TW.exists():
        pytest.skip("サンプル結果なし")
    renderer = ResultRenderer(ResultData(TM0_TW))
    out = tmp_path / "a.gif"
    assert not render_gif(renderer, Selection(), ViewOptions(), out, n_frames=6, progress=lambda i, n: i < 1)
    assert not out.exists()
    from axicavity_fem.gui.ui.dialogs.gif import save_gif_with_progress
    monkeypatch.setattr(QtWidgets.QProgressDialog, "wasCanceled", lambda self: True)
    result = save_gif_with_progress(renderer, Selection(), ViewOptions(),
                                    {"n_frames": 6, "fps": 5, "output_path": str(out)}, (320, 240))
    assert result is None and not out.exists()


def test_14_field_export_has_instant_and_scale(qtbot):
    from axicavity_fem.cli.main import build_parser

    argv = export_argv("tm0", "in.h5", "out", "area", 0, phase=120.0, time_phase=30.0,
                       params={"instant": True, "scale": 2.5, "nz": 4, "nr": 3})
    assert "--instant" in argv and argv[argv.index("--scale") + 1] == "2.5"
    args = build_parser().parse_args(argv)
    assert args.instant and args.scale == 2.5 and args.time_phase == 30.0
    dialog = ExportFieldDialog("axis", "out", (0.0, 0.1), (0.0, 0.05), traveling_tm0=True)
    qtbot.addWidget(dialog)
    dialog.instant_check.setChecked(True)
    dialog.scale_edit.setText("3")
    params = dialog.params()
    assert params["instant"] and params["scale"] == 3.0
    standing = ExportFieldDialog("axis", "out", (0.0, 0.1), (0.0, 0.05), traveling_tm0=False)
    qtbot.addWidget(standing)
    standing.instant_check.setChecked(True)
    assert not standing.params()["instant"]                               # 定在波では常に実数
