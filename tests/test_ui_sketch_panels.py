"""モデリングのプロパティパネル（offscreen、pytest-qt）: 選択点の Z/R 式入力と数値プレビュー、r=0 固定、
選択円弧の中心・半径、パラメータ表、拘束一覧."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtCore                     # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore       # noqa: E402
from axicavity_fem.gui.core.document import SketchArc, SketchPoint  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_arc, get_entity, point_pos  # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language, tr       # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402
from axicavity_fem.gui.ui.sketch_panels import SketchPropertyPanel  # noqa: E402


@pytest.fixture
def env(qtbot):
    set_language("ja")
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)
    panel = SketchPropertyPanel(store)
    qtbot.addWidget(panel)
    panel.resize(380, 900)
    panel.show()
    qtbot.waitExposed(panel)
    store.new_document()
    return store, controller, panel


def click(controller, x, y):
    controller.pointer_move((x, y))
    controller.pointer_down((x, y))
    controller.pointer_up((x, y))


def rectangle(store, controller, x0, y0, x1, y1):
    store.set_tool("rectangle")
    click(controller, x0, y0)
    click(controller, x1, y1)
    store.set_tool("select")


def point_at(store, pos):
    sketch = store.active_sketch()
    return next(e for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection and point_pos(sketch, e.id) == pos)


def test_selected_point_expression_input_and_preview(env):
    store, controller, panel = env
    rectangle(store, controller, 0, 0, 40, 20)
    editor = panel.point_editor
    assert not editor.isVisible()
    corner = point_at(store, (40, 20))
    store.set_selection([corner.id])
    assert editor.isVisible()
    assert (editor.z_edit.text(), editor.r_edit.text()) == ("40", "20")
    assert editor.z_preview.text() == ""                      # 数値だけならプレビューは出ない

    pid = store.add_param()
    store.update_param(pid, name="L", expression="60")
    editor.z_edit.setText("L/2")
    editor.z_edit.textEdited.emit("L/2")
    assert editor.z_preview.text() == "= 30.000000"
    editor.z_edit.returnPressed.emit()                       # Enter で適用
    moved = get_entity(store.active_sketch(), corner.id)
    assert (moved.x, moved.xExpr, moved.y, moved.yExpr) == (30, "L/2", 20, None)
    assert (editor.z_edit.text(), editor.r_edit.text()) == ("L/2", "20")
    assert editor.z_preview.text() == "= 30.000000"

    # パラメータを変えると追従（式は残る）
    store.update_param(pid, expression="100")
    moved = get_entity(store.active_sketch(), corner.id)
    assert (moved.x, moved.xExpr) == (50, "L/2")
    assert editor.z_preview.text() == "= 50.000000"

    # 評価できない式は拒否され、エラーが出る
    editor.z_edit.setText("foo+")
    editor.z_edit.textEdited.emit("foo+")
    assert editor.z_preview.text() == "= ?"
    editor.apply_button.click()
    assert get_entity(store.active_sketch(), corner.id).xExpr == "L/2"
    assert panel.expr_error.isVisible()

    # r=0 に固定 → z 軸（r = 0）への「点を曲線上に」の拘束（R の式は付けない）
    panel.fix_axis_button.click()
    fixed = get_entity(store.active_sketch(), corner.id)
    assert (fixed.y, fixed.yExpr) == (0, None)
    assert any(c.type == "pointOnCurve" and c.point == corner.id and c.curve == "ref-z-axis"
               for c in store.active_sketch().constraints)
    # z=0 に固定 → Z の式が "0"（L/2 の式は置き換わる）
    panel.fix_z0_button.click()
    fixed = get_entity(store.active_sketch(), corner.id)
    assert (fixed.x, fixed.y) == (0, 0)
    assert {c.curve for c in store.active_sketch().constraints
            if c.type == "pointOnCurve" and c.point == corner.id} == {"ref-z-axis", "ref-r-axis"}
    assert (editor.z_edit.text(), editor.r_edit.text()) == ("0", "0")
    assert editor.r_edit.text() == "0"
    store.set_selection([])
    assert not editor.isVisible()


def test_selected_arc_center_and_radius(env):
    store, controller, panel = env
    store.update_active_sketch(lambda s: add_arc(s, (0, 0), (10, 0), (0, 10)) is not None)
    arc = next(e for e in store.active_sketch().entities if isinstance(e, SketchArc))
    editor = panel.arc_editor
    assert not editor.isVisible()
    store.set_selection([arc.id])
    assert editor.isVisible()
    assert (editor.z_edit.text(), editor.r_edit.text()) == ("0", "0")
    assert tr("props.radiusValue", value="10") in editor.extra.text()
    editor.r_edit.setText("2")
    editor.apply_button.click()
    sketch = store.active_sketch()
    center = get_entity(sketch, arc.center)
    assert (center.x, center.y) == (0, 2)
    assert get_entity(sketch, arc.id) is not None
    # 向きを反転: 中心が弦の反対側に移り、円弧の ID は変わらない
    editor.z_edit.setText("0")
    editor.r_edit.setText("0")
    editor.apply_button.click()
    start0, end0 = arc.start, arc.end                       # 文書のオブジェクトはその場で書き換わるので控える
    panel.flip_arc_button.click()
    sketch = store.active_sketch()
    flipped = get_entity(sketch, arc.id)
    assert flipped is not None and (flipped.start, flipped.end) == (end0, start0)
    center = get_entity(sketch, flipped.center)
    assert (round(center.x, 9), round(center.y, 9)) == (10, 10)
    assert store.undo() and get_entity(store.active_sketch(), arc.id).start == start0


def test_params_panel_add_edit_move_delete(env):
    store, controller, panel = env
    params = panel.params
    assert not params.isVisible()
    store.set_params_edit(True)
    assert params.isVisible()
    params.add_button.click()
    assert len(store.document.params) == 1 and params.table.rowCount() == 1
    params.table.item(0, 0).setText("R")
    params.table.item(0, 1).setText("2*10")
    assert store.document.params[0].name == "R" and store.document.params[0].expression == "2*10"
    assert params.table.item(0, 2).text() == "20"
    params.add_button.click()
    params.table.item(1, 0).setText("L")
    params.table.item(1, 1).setText("R**2")
    assert params.table.item(1, 2).text() == "400"
    # 上へ移動すると前方参照になりエラー表示、戻すと直る
    params.table.setCurrentCell(1, 1)
    params.up_button.click()
    assert [p.name for p in store.document.params] == ["L", "R"]
    assert params.table.item(0, 2).text().startswith("エラー")
    params.down_button.click()
    assert [p.name for p in store.document.params] == ["R", "L"]
    assert params.table.item(1, 2).text() == "400"
    params._delete_row(1)
    assert [p.name for p in store.document.params] == ["R"]


def test_constraint_list_and_info(env):
    store, controller, panel = env
    rectangle(store, controller, 0, 0, 40, 20)
    assert "閉領域: 1" in panel.info.text()
    assert panel.constraints.table.rowCount() == 4            # 水平 2 + 垂直 2
    assert "自由度" in panel.constraints.status.text()
    panel.constraints._delete_row(0)
    assert panel.constraints.table.rowCount() == 3
    store.set_tool("circle")
    assert panel.numeric_label.text() == tr("props.radius")
    store.set_tool("select")
    assert panel.numeric_label.text() == tr("props.length")
    assert not panel.fillet_radius.isVisible()
    store.set_tool("fillet")
    assert panel.fillet_radius.isVisible()
    QtCore.QCoreApplication.processEvents()


def test_tool_specific_options_are_shown_only_for_their_tool(env):
    """辺の数は多角形のとき、フィレット半径はフィレットのときだけ見せる."""
    store, controller, panel = env
    store.set_tool("line")
    assert panel.sides.isHidden() and panel.sides_label.isHidden() and panel.fillet_radius.isHidden()
    store.set_tool("polygon")
    assert not panel.sides.isHidden() and not panel.sides_label.isHidden() and panel.fillet_radius.isHidden()
    store.set_tool("fillet")
    assert panel.sides.isHidden() and not panel.fillet_radius.isHidden() and not panel.fillet_radius_label.isHidden()
    store.set_tool("select")
    assert panel.sides.isHidden() and panel.fillet_radius.isHidden()
