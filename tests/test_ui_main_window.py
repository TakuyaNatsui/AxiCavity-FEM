"""MainWindow の結線（offscreen、pytest-qt）: リボン・ショートカット・ツリー・overlay・境界条件・取り込み/書き出し・
単位・軸範囲・言語・配置の保存.

ver3 のメインウィンドウは VTK を含まないので offscreen で開ける（実ウィンドウは _main_window_check.py）。
設定（言語・最近使ったファイル・配置）は一時フォルダの INI に書く（AXICAVITY_SETTINGS_DIR）。
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtCore, QtGui, QtWidgets     # noqa: E402

from axicavity_fem.gui.core.document import SketchLine, SketchPoint   # noqa: E402
from axicavity_fem.gui.core.legacy import load_gmshproj  # noqa: E402
from axicavity_fem.gui.core.sketch.model import count_entities, point_pos  # noqa: E402
from axicavity_fem.gui.io import axiproj                 # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language, tr   # noqa: E402
from axicavity_fem.gui.ui import app_settings            # noqa: E402
from axicavity_fem.gui.ui.main_window import RIBBON_TABS, MainWindow  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("AXICAVITY_SETTINGS_DIR", str(tmp_path / "settings"))
    set_language("ja")
    w = MainWindow()
    # qtbot.addWidget は使わない: pytest-qt は fixture の後始末より先に close し、未保存の確認（モーダル）で止まる
    w.resize(1400, 850)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    QtWidgets.QApplication.processEvents()
    yield w
    w.project._mark_saved()                        # 確認ダイアログを出さずに閉じる
    w.close()
    w.deleteLater()
    QtWidgets.QApplication.processEvents()


def tab(win) -> str:
    return RIBBON_TABS[win.ribbon.currentIndex()]


def px(view, x, y) -> QtCore.QPoint:
    return view.mapFromScene(QtCore.QPointF(x, y))


def click(qtbot, view, x, y):
    p = px(view, x, y)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(p),
                           QtCore.QPointF(view.viewport().mapToGlobal(p)),
                           QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(view.viewport(), ev)
    qtbot.mouseClick(view.viewport(), QtCore.Qt.LeftButton, pos=p)


def draw_rectangle(qtbot, win, x0=0, y0=0, x1=40, y1=20):
    view = win.sketch_view
    view.fit_sketch()
    win.tool_actions["rectangle"].trigger()
    click(qtbot, view, x0, y0)
    click(qtbot, view, x1, y1)
    win.tool_actions["select"].trigger()


def line_between(store, a, b):
    sketch = store.active_sketch()
    return next(e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection
                and {point_pos(sketch, e.p1), point_pos(sketch, e.p2)} == {a, b})


def test_ribbon_tabs_icons_and_tooltips(win):
    assert win.menuWidget() is win.ribbon
    names = [win.ribbon.tabText(i) for i in range(win.ribbon.count())]
    assert names == ["ファイル", "モデリング", "物理", "メッシュ", "解析", "結果", "表示"], names
    assert win.windowTitle().startswith("AxiCavity-FEM v3")
    text_only = []
    for name, bar in win._toolbars.items():
        for act in bar.actions():
            if act.isSeparator() or act is win.language_action or isinstance(act, QtWidgets.QWidgetAction):
                continue
            assert act.toolTip().startswith("<b>"), (name, act.text())
            if act.icon().isNull():
                text_only.append((name, act.text()))
    assert text_only == [], text_only
    # モデリングは 2 段（作図・編集 / 拘束・寸法・パラメータ）
    assert {a.data() for a in win._toolbars["modeling"].actions() if a.data()} >= {
        "ribbon.select", "ribbon.line", "ribbon.insertPoint", "ribbon.fit"}
    assert {a.data() for a in win._toolbars["modeling2"].actions() if a.data()} >= {
        "ribbon.coincident", "ribbon.fixAxis", "ribbon.fixZ0", "ribbon.dimension", "ribbon.params"}
    labels = [w.text() for w in win._toolbars["modeling2"].findChildren(QtWidgets.QLabel)]
    assert tr("ribbon.groupConstraints") in labels and tr("ribbon.groupDimension") in labels   # 見出し
    assert win.bc_actions["PEC"] in win._toolbars["physics"].actions()
    # 取り込み / 書き出しはドロップダウン（ファイルタブのボタンを減らす。2026-09-25 ユーザー決定）
    assert [a.data() for a in win.import_menu.actions()] == ["ribbon.importGmshproj", "ribbon.importSuperfish"]
    export_keys = [a.data() for a in win.export_menu.actions() if not a.isSeparator()]
    assert export_keys == ["ribbon.exportGmshproj", "ribbon.exportSuperfish", "ribbon.exportMsh",
                           "ribbon.exportGeo", "ribbon.exportPython"]
    assert all(not a.icon().isNull() for a in win.export_menu.actions() if not a.isSeparator())
    assert all(not b.icon().isNull() and b.toolTip().startswith("<b>") for b in win._menu_buttons.values())
    file_keys = [a.data() for a in win._toolbars["file"].actions() if a.data()]
    assert "ribbon.exportMsh" not in file_keys and "ribbon.importGmshproj" not in file_keys


def test_draw_undo_redo_and_tree(qtbot, win):
    store = win.store
    draw_rectangle(qtbot, win)
    assert count_entities(store.active_sketch())["line"] == 4 and len(store.profiles) == 1
    shape = win.tree_item(("section", "shape"))
    assert "閉領域 1" in shape.text(0) and "線 4" in shape.text(0), shape.text(0)
    # 下辺は軸の上 → None、残りは PEC
    assert win.tree_item(("bc", "PEC")).text(0) == "PEC × 3"
    assert win.tree_item(("bc", "None")).text(0) == "None × 1"
    assert "Vacuum" in win.tree_item(("section", "regions")).child(0).text(0)
    assert win.isWindowModified()

    # Ctrl+Z / Ctrl+Y はどのタブでも効く
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("analysis"))
    qtbot.keyClick(win, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    assert count_entities(store.active_sketch())["line"] == 0
    qtbot.keyClick(win, QtCore.Qt.Key_Y, QtCore.Qt.ControlModifier)
    assert count_entities(store.active_sketch())["line"] == 4
    assert "閉領域 1" in win.tree_item(("section", "shape")).text(0)

    # Ctrl+A は全選択（wx 版では Superfish 書き出しと衝突していた）
    win.sketch_view.setFocus()
    win.select_all_action.trigger()
    assert len(store.selection) == 7 and store.active_tool == "select"        # 原点の角は参照の原点（選ばない）


def test_tabs_switch_overlay_and_tree_click(qtbot, win):
    store = win.store
    draw_rectangle(qtbot, win)
    assert store.overlay == "sketch"
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("physics"))
    assert store.overlay == "physics" and store.active_tool == "select"
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("mesh"))
    assert store.overlay == "mesh"
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("file"))
    assert store.overlay == "mesh"                       # ファイル / 表示では直前のまま
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("modeling"))
    assert store.overlay == "sketch"
    assert win.props_stack.currentWidget() is win.sketch_panel

    # ツリー: 節のクリックで対応するタブへ。境界条件の項目はその曲線を選択、領域は選択領域
    win._on_tree_clicked(win.tree_item(("section", "mesh")), 0)
    assert tab(win) == "mesh"
    win._on_tree_clicked(win.tree_item(("bc", "PEC")), 0)
    assert tab(win) == "physics" and len(store.selection) == 3
    win._on_tree_clicked(win.tree_item(("region", store.profiles[0].id)), 0)
    assert store.selected_region_id == store.profiles[0].id
    win._on_tree_clicked(win.tree_item(("section", "params")), 0)
    assert tab(win) == "modeling" and store.params_edit


def test_boundary_conditions_from_ribbon(qtbot, win, monkeypatch):
    store = win.store
    draw_rectangle(qtbot, win)
    left = line_between(store, (0, 0), (0, 20))
    assert not win.bc_actions["E-short"].isEnabled()
    store.set_selection([left.id])
    assert win.bc_actions["E-short"].isEnabled()
    assert "PEC" in win.bc_label.text() and "既定" in win.bc_label.text()
    win.bc_actions["E-short"].trigger()
    assert store.effective_bc(left.id) == ("E-short", "explicit")
    assert "E-short" in win.bc_label.text() and "指定" in win.bc_label.text()
    assert win.tree_item(("bc", "PEC")).text(0) == "PEC × 2"
    assert win.tree_item(("bc", "E-short")).text(0) == "E-short × 1"
    win.bc_actions["auto"].trigger()
    assert store.effective_bc(left.id) == ("PEC", "default")
    assert "E-short" not in win.bc_label.text()

    # 検証（問題なし → information）。開いた線を足すと警告
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(("info", a[2]))))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: shown.append(("warn", a[2]))))
    win._on_validate()
    assert shown[-1][0] == "info" and "問題はありません" in shown[-1][1]
    store.set_tool("line")
    click(qtbot, win.sketch_view, 50, 5)
    click(qtbot, win.sketch_view, 60, 25)
    qtbot.keyClick(win.sketch_view.viewport(), QtCore.Qt.Key_Return)
    assert count_entities(store.active_sketch())["line"] == 5
    win._on_validate()
    assert shown[-1][0] == "info" and "警告 1" in shown[-1][1], shown[-1]
    assert len(store.selection) == 1                     # 問題の曲線を選択


def test_edit_actions(qtbot, win):
    store = win.store
    draw_rectangle(qtbot, win, 10, 5, 50, 25)
    # r=0 に固定: 左下の点 → z 軸への拘束。下辺の水平拘束で右下も r=0 に
    sketch = store.active_sketch()
    corner = next(e.id for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection and point_pos(sketch, e.id) == (10, 5))
    store.set_selection([corner])
    assert win.fix_axis_action.isEnabled() and win.fix_z0_action.isEnabled()
    assert not win.insert_point_action.isEnabled()
    win.fix_axis_action.trigger()
    sketch = store.active_sketch()
    assert any(c.type == "pointOnCurve" and c.point == corner and c.curve == "ref-z-axis" for c in sketch.constraints)
    assert line_between(store, (10, 0), (50, 0)) is not None
    assert win.tree_item(("bc", "None")).text(0) == "None × 1"
    # z=0 に固定: 同じ点 → r 軸への拘束。左辺の垂直拘束で左上も z=0 に
    win.fix_z0_action.trigger()
    sketch = store.active_sketch()
    fixed = next(e for e in sketch.entities if e.id == corner)
    assert (fixed.x, fixed.y) == (0, 0)
    assert any(c.type == "pointOnCurve" and c.point == corner and c.curve == "ref-r-axis" for c in sketch.constraints)
    assert line_between(store, (0, 0), (0, 25)) is not None and len(store.profiles) == 1

    top = line_between(store, (0, 25), (50, 25))
    store.set_selection([top.id])
    assert win.insert_point_action.isEnabled() and win.line_to_arc_action.isEnabled()
    win.insert_point_action.trigger()
    sketch = store.active_sketch()
    assert count_entities(sketch)["point"] == 5 and count_entities(sketch)["line"] == 5
    mid = store.selection[0]
    assert point_pos(sketch, mid) == pytest.approx((25, 25))
    win.delete_reconnect_action.trigger()
    sketch = store.active_sketch()
    assert count_entities(sketch)["point"] == 4 and count_entities(sketch)["line"] == 4
    top = line_between(store, (0, 25), (50, 25))
    store.set_selection([top.id])
    win.line_to_arc_action.trigger()                      # 線を選んでいる → すぐ変換（ツールは選択のまま）
    assert count_entities(store.active_sketch())["arc"] == 1
    assert store.active_tool == "select" and win.tool_actions["select"].isChecked()
    assert not win.line_to_arc_action.isChecked()
    assert win.arc_to_line_action.isEnabled()
    win.arc_to_line_action.trigger()
    assert count_entities(store.active_sketch())["arc"] == 0
    assert len(store.profiles) == 1

    # 何も選んでいない → ボタンがツールになり、クリックした線を変換する（2026-09-29 ユーザー要望）
    store.set_selection([])
    assert win.line_to_arc_action.isEnabled() and win.arc_to_line_action.isEnabled()
    win.line_to_arc_action.trigger()
    assert store.active_tool == "lineToArc" and win.line_to_arc_action.isChecked()
    assert store.hint == "hint.pickLineToArc"
    click(qtbot, win.sketch_view, 25, 25)
    sketch = store.active_sketch()
    assert count_entities(sketch)["arc"] == 1 and len(store.profiles) == 1
    arc = next(e for e in sketch.entities if e.id == store.selection[0])
    assert arc.type == "arc"                                # 変換した円弧が選択される（向きの反転がすぐ使える）
    win.arc_to_line_action.trigger()                      # 円弧を選んでいる → すぐ線に戻す
    assert count_entities(store.active_sketch())["arc"] == 0
    assert store.active_tool == "lineToArc" and win.line_to_arc_action.isChecked()
    store.set_selection([])
    win.arc_to_line_action.trigger()
    assert store.active_tool == "arcToLine" and win.arc_to_line_action.isChecked()
    qtbot.keyClick(win.sketch_view.viewport(), QtCore.Qt.Key_Escape)   # Esc → 選択ツール
    assert store.active_tool == "select" and win.tool_actions["select"].isChecked()


def test_import_export_recent_and_new(qtbot, win, monkeypatch, tmp_path):
    store = win.store
    sample = SAMPLES / "diel_test1.gmshproj"
    assert win.open_path(sample)
    assert win.isWindowModified() and "diel_test1" in win.windowTitle()   # 取り込み = 未保存の新規
    regions = win.tree_item(("section", "regions"))
    assert regions.childCount() >= 2
    assert any("tanδ" in regions.child(i).text(0) for i in range(regions.childCount()))
    assert win.tree_item(("section", "params")).text(0).startswith("パラメータ")
    win._rebuild_recent_menu()
    assert win.recent_menu.actions()[0].text() == "diel_test1.gmshproj"

    out = tmp_path / "out.gmshproj"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    assert win._on_export_gmshproj()
    doc, _ = load_gmshproj(out)
    assert count_entities(doc.sketch) == count_entities(store.document.sketch)

    # 変更 → 新規は確認（破棄）→ 空の文書
    store.add_param()
    assert win.isWindowModified()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Discard))
    win._on_new()
    assert count_entities(store.active_sketch())["line"] == 0 and not win.isWindowModified()
    assert "Untitled" in win.windowTitle()


def test_units_axis_range_language_and_layout(qtbot, win, monkeypatch, tmp_path):
    store = win.store
    draw_rectangle(qtbot, win)
    monkeypatch.setattr(MainWindow, "_ask_units", lambda self, current: "cm")
    monkeypatch.setattr(MainWindow, "_ask_unit_conversion", lambda self, old, new: True)
    win._on_units()
    assert store.document.meta.units == "cm"
    sketch = store.active_sketch()
    points = sorted(point_pos(sketch, e.id) for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection)
    assert points == pytest.approx([(0, 2), (4, 0), (4, 2)])               # 原点の角は参照の原点
    assert "cm" in win.tree_item(("section", "mesh")).text(0)

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("-1, 5, 0, 3", True)))
    win._on_axis_range()
    v = store.document.view
    assert (v.zmin, v.zmax, v.rmin, v.rmax) == (-1, 5, 0, 3)
    scene = win.sketch_view.mapToScene(win.sketch_view.viewport().rect()).boundingRect()
    assert scene.left() <= -1 and scene.right() >= 5

    win._on_toggle_language()
    assert win.ribbon.tabText(0) == "File" and app_settings.saved_language() == "en"
    assert win.tree_item(("section", "shape")).text(0).startswith("Shape")
    win._on_toggle_language()
    assert win.ribbon.tabText(0) == "ファイル"

    # 閉じると配置を保存（未保存の変更は破棄で答える）
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Discard))
    win._dock_log.hide()
    assert win.close()
    geometry, state = app_settings.load_window_layout()
    assert geometry is not None and state is not None
    win2 = MainWindow()
    qtbot.addWidget(win2)
    win2.show()
    qtbot.waitExposed(win2)
    assert win2._dock_log.isHidden() and not win2._dock_tree.isHidden()
    win2.reset_layout_action.trigger()
    assert not win2._dock_log.isHidden()
    assert tr("ribbon.resetLayoutDone") in win2.log.toPlainText()
    win2.close()


def test_project_save_open_results_and_close(qtbot, win, monkeypatch, tmp_path):
    store, project = win.store, win.project
    draw_rectangle(qtbot, win)
    assert win.isWindowModified() and project.path is None
    target = tmp_path / "p"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    assert win._on_save()                                    # 初回は名前を付けて保存（拡張子は補完）
    assert project.path == tmp_path / "p.axiproj" and not win.isWindowModified()
    assert "p.axiproj" in win.windowTitle()
    win._rebuild_recent_menu()
    assert win.recent_menu.actions()[0].text() == "p.axiproj"
    store.set_mesh(size=2.0)
    assert win.isWindowModified()
    assert win._on_save() and not win.isWindowModified()     # 2 回目はダイアログ無し

    # 結果フォルダを置くとツリーに出る。クリックで結果タブ + 表示中（太字）。設定を変えると「メッシュ変更前」
    d = project.results_dir / "0001-tm0-sw"
    d.mkdir(parents=True)
    (d / "job.json").write_text('{"kind": "tm0-sw"}', encoding="utf-8")
    (d / "model_SW_TM0.h5").write_bytes(b"h5")
    geom, mesh, model = project.current_hashes()
    axiproj.write_result_meta(d, number=1, kind="tm0-sw", status="done", geometryHash=geom, meshHash=mesh,
                              modelHash=model, summary={"modes": [{"f_GHz": 2.856}, {"f_GHz": 3.1}]})
    project.refresh_results()
    node = win.tree_item(("section", "results"))
    assert node.childCount() == 1 and "#1" in node.child(0).text(0) and "2.8560" in node.child(0).text(0)
    win._on_tree_clicked(node.child(0), 0)
    assert tab(win) == "results" and project.active_result_id == "0001-tm0-sw"
    assert win.tree_item(("result", "0001-tm0-sw")).font(0).bold()
    store.set_mesh(size=3.0)
    assert "メッシュ変更前" in win.tree_item(("result", "0001-tm0-sw")).text(0)

    # 新規 → 未保存の確認で「保存」→ 保存されてから空の文書。開き直すと形状も結果も戻る
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Save))
    win._on_new()
    assert project.path is None and count_entities(store.active_sketch())["line"] == 0
    assert axiproj.read_project(tmp_path / "p.axiproj").document.mesh.size == 3.0
    assert win.open_path(tmp_path / "p.axiproj")
    assert count_entities(store.active_sketch())["line"] == 4 and not win.isWindowModified()
    assert [e.id for e in project.results] == ["0001-tm0-sw"] and project.active_result_id == "0001-tm0-sw"
    assert win.tree_item(("section", "results")).childCount() == 1

    # 開けないファイルはエラー表示で今のまま
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    bad = tmp_path / "bad.axiproj"
    bad.write_text("{", encoding="utf-8")
    assert not win.open_path(bad) and errors and project.path == tmp_path / "p.axiproj"


def test_mesh_generation_from_ribbon(qtbot, win, monkeypatch, tmp_path):
    """メッシュ生成: 未保存なら保存先を聞いてから自動保存 → 子プロセス → 統計・ツリー・スケッチ画面の重ね表示."""
    pytest.importorskip("gmsh")
    store, project, mesh = win.store, win.project, win.mesh
    draw_rectangle(qtbot, win, 0, 0, 100, 50)
    store.set_mesh(size=10.0)
    assert win.mesh_generate_action.isEnabled() and not win.mesh_show_action.isEnabled()
    target = tmp_path / "meshproj.axiproj"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    win.mesh_generate_action.trigger()
    assert project.path == target and infos and "mesh" in infos[0]      # 先に保存先を決める
    assert tab(win) == "mesh" and store.overlay == "mesh" and mesh.running
    assert not win.mesh_generate_action.isVisible() and win.mesh_stop_action.isVisible()
    qtbot.waitUntil(lambda: not mesh.running, timeout=180_000)
    assert mesh.job.status == "done", (mesh.job.error, mesh.job.traceback)
    assert mesh.info is not None and mesh.info.dir.parent == project.mesh_dir
    assert mesh.visible and win.mesh_show_action.isChecked() and win.sketch_view.mesh_preview is not None
    assert len(win.sketch_view.mesh_preview.triangles) == mesh.info.stats["elements"] > 0
    assert "要素" in win.tree_item(("section", "mesh")).text(0)
    assert win.mesh_panel.state_label.text() == tr("mesh.state.current")
    assert not win.isWindowModified()                                   # 自動保存された
    # 形状を変えると重ね表示が消え、ツリーに「要再生成」
    draw_rectangle(qtbot, win, 200, 0, 240, 20)
    assert not mesh.visible and win.sketch_view.mesh_preview is None
    assert "要再生成" in win.tree_item(("section", "mesh")).text(0)
    store.undo()
    win.mesh_show_action.trigger()
    assert mesh.visible and win.sketch_view.mesh_preview is not None
    # 開き直すとメッシュも復元される（メッシュフォルダはプロジェクトの中）
    win.project._mark_saved()
    assert win.open_path(target)
    assert mesh.info is not None and mesh.state() == "current"
    win.session.wait()


def test_analysis_from_ribbon(qtbot, win, monkeypatch, tmp_path):
    """解析実行: 確認ダイアログ → 自動保存 → 子プロセス → 結果ツリー（#1 完了・f・Q）→ レポート."""
    pytest.importorskip("gmsh")
    from axicavity_fem.gui.ui.dialogs import run_confirm

    store, project, analysis = win.store, win.project, win.analysis
    draw_rectangle(qtbot, win, 0, 0, 100, 50)
    store.set_mesh(size=10.0)
    store.set_analysis(numModes=2)
    target = tmp_path / "run.axiproj"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    shown = []

    def fake_exec(dialog):
        shown.append(dialog.summary_text())
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(run_confirm.RunConfirmDialog, "exec", fake_exec)
    assert win.run_action.isEnabled()
    win.run_action.trigger()
    assert shown and "TM0" in shown[0] and "0001-tm0-sw" in shown[0] and "axicavity-fem solve" in shown[0]
    assert project.path == target and analysis.running and tab(win) == "analysis"
    assert win.cancel_action.isVisible() and not win.run_action.isVisible()
    assert "#1" in win.tree_item(("section", "results")).child(0).text(0)
    qtbot.waitUntil(lambda: not analysis.running, timeout=300_000)
    assert analysis.job.status == "done", (analysis.job.error, analysis.job.traceback)
    qtbot.waitUntil(lambda: project.results[0].status == "done", timeout=10_000)
    item = win.tree_item(("section", "results")).child(0)
    assert "#1" in item.text(0) and "GHz" in item.text(0) and "TM0 定在波" in item.text(0)
    assert win.run_action.isVisible() and win.post_action.isEnabled() and win.report_action.isEnabled()
    assert not win.open_report_action.isEnabled()
    assert "完了" in win.analysis_panel.state_label.text()
    # 完了した結果は結果タブに表示される（結果ビュー・モード表）
    model = win.result_model
    qtbot.waitUntil(lambda: model.entry is not None and model.entry.id == "0001-tm0-sw", timeout=10_000)
    assert tab(win) == "results" and win.stack.currentWidget() is win.result_view
    assert win.show_results_action.isChecked() and model.path.name == "model_SW_TM0_processed.h5"
    qtbot.waitUntil(lambda: win.result_view.last_title.startswith("Mode 0"), timeout=60_000)
    panel = win.results_panel
    assert panel.table.rowCount() == 2 and panel.table.horizontalHeaderItem(2).text() == "Q"
    assert "#1" in panel.source_label.text() and not panel.relation_label.isVisible()
    assert win.png_action.isEnabled() and not win.gif_action.isEnabled()          # 定在波は GIF なし
    # 他のタブではスケッチ画面、「結果表示」で戻る
    win._show_tab("modeling")
    assert win.stack.currentWidget() is win.sketch_view and not win.show_results_action.isChecked()
    win.show_results_action.trigger()
    assert tab(win) == "results" and win.stack.currentWidget() is win.result_view
    # post をやり直すと表示中の結果を読み直す
    loads = []
    model.loaded.connect(lambda: loads.append(model.path))
    q_before = model.data.post_params(model.selection)["Q"]
    store.set_post(cond=2.0e7)
    win.post_action.trigger()
    qtbot.waitUntil(lambda: not analysis.running, timeout=120_000)
    qtbot.waitUntil(lambda: bool(loads), timeout=10_000)
    assert model.data.post_params(model.selection)["Q"] < q_before
    # レポート作成 → 開くボタンが有効に
    win.report_action.trigger()
    qtbot.waitUntil(lambda: not analysis.running, timeout=300_000)
    qtbot.waitUntil(lambda: win.open_report_action.isEnabled(), timeout=10_000)
    assert analysis.report_path(project.results[0]).exists()
    # 解析設定を変えると「設定変更前」（ツリーと結果パネル）
    store.set_analysis(numModes=5)
    assert "設定変更前" in win.tree_item(("section", "results")).child(0).text(0)
    assert win.results_panel.relation_label.isVisible()
    win.session.wait()


def test_results_tab_with_sample_files(qtbot, win, monkeypatch, tmp_path):
    """結果タブ: 外部の h5 を開く → 表示オプション → 場の値 → PNG / GIF / 場の書き出し（子プロセス）→ HOM."""
    from axicavity_fem.gui.ui.dialogs import export_field, gif

    tw = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
    hom = SAMPLES / "s-band_1cell_SW_HOM_processed.h5"
    if not (tw.exists() and hom.exists()):
        pytest.skip("サンプル結果なし")
    model, view, panel = win.result_model, win.result_view, win.results_panel
    assert not win.png_action.isEnabled() and not win.gif_action.isEnabled() and not win.view3d_action.isEnabled()
    assert win.open_path(tw)                                                # 起動引数と同じ経路
    assert tab(win) == "results" and model.has_data and model.is_traveling and model.entry is None
    qtbot.waitUntil(lambda: view.last_title.startswith("Mode 0"), timeout=60_000)
    assert win.gif_action.isEnabled() and win.png_action.isEnabled() and win.result_folder_action.isEnabled()
    assert win._menu_buttons["results.ribbon.exportField"].isEnabled()
    from axicavity_fem.gui.ui import view3d_window
    assert win.view3d_action.isCheckable() and win.view3d_action.isEnabled() == view3d_window.is_available()
    assert win.view3d is None                                               # 押すまで VTK は作らない（offscreen）
    # リボンのオプション ⇄ モデル ⇄ パネル
    win.result_option_actions["show_mesh"].trigger()
    assert model.options.show_mesh and panel.mesh_check.isChecked()
    panel.vectors_check.setChecked(True)
    assert win.result_option_actions["show_vectors"].isChecked()
    # 場の値（ダブルクリック相当）
    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    view.point_picked.emit(0.017, 0.02)
    assert infos and "H_theta" in infos[-1] and "Ez" in panel.field_label.text()
    view.point_picked.emit(9.0, 9.0)                                        # 領域外はステータスバーだけ
    assert len(infos) == 1
    # PNG
    png = tmp_path / "out" / "fig.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(png), "")))
    win.png_action.trigger()
    assert png.exists() and png.stat().st_size > 1000
    # GIF（ダイアログは自動で受け付ける）
    gif_path = tmp_path / "anim.gif"

    def gif_exec(dialog):
        dialog.frames_spin.setValue(4)
        dialog.output_edit.setText(str(gif_path))
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(gif.GifDialog, "exec", gif_exec)
    win.gif_action.trigger()
    assert gif_path.exists() and gif_path.stat().st_size > 1000
    # 場の書き出し（子プロセス）: 出力は一時フォルダに
    base = tmp_path / "field" / "area_m0"

    def export_exec(dialog):
        assert dialog.shape == "area" and dialog.traveling_tm0
        dialog.nz_spin.setValue(8)
        dialog.nr_spin.setValue(4)
        dialog.output_edit.setText(str(base))
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(export_field.ExportFieldDialog, "exec", export_exec)
    win.export_field_actions["area"].trigger()
    assert win.analysis.running and win.analysis.job.kind == "export"
    qtbot.waitUntil(lambda: not win.analysis.running, timeout=120_000)
    assert win.analysis.job.status == "done", (win.analysis.job.error, win.analysis.job.traceback)
    qtbot.waitUntil(lambda: base.with_suffix(".h5").exists() and base.with_suffix(".txt").exists(), timeout=10_000)
    assert win._export_field_tmp is None
    # HOM: E 線 / E-wall のボタンが消え、n が選べる
    assert win.open_result_file(hom) and model.is_hom
    assert not win.result_option_actions["show_lines"].isVisible() and not win.gif_action.isEnabled()
    assert panel.n_combo.count() == 3
    qtbot.waitUntil(lambda: view.last_title.startswith("HOM n=0"), timeout=60_000)
    # 壊れたファイルはエラーダイアログ
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a[2])))
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"xx")
    assert not win.open_result_file(bad) and errors
    win.session.wait()
