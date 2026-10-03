"""ver3 の式付き点（ver2.3 の Z/R 式入力の再現）: 保持・パラメータ変更で追従・ドラッグで式破棄・片側固定.

（ver2.3 tests/test_coord_expressions.py の GUI 部分の移植）
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore       # noqa: E402
from axicavity_fem.gui.core.document import SketchArc, SketchConstraint, SketchLine, SketchPoint, new_id  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_point, get_point, point_pos  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402


@pytest.fixture
def env():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)
    store.new_document()
    return store, controller


def click(controller, x, y):
    controller.pointer_move((x, y))
    controller.pointer_down((x, y))
    controller.pointer_up((x, y))


def test_set_point_coords_keeps_expression_and_follows_params(env):
    store, ctl = env
    pid = store.add_param()
    store.update_param(pid, name="a", expression="3")
    store.set_tool("point")
    click(ctl, 2, 2)                       # 原点に置くと参照の原点を指すだけで点はできない
    point = next(e for e in store.active_sketch().entities if isinstance(e, SketchPoint) and not e.projection)
    assert store.set_point_coords(point.id, "a+2", "10")             # 素の数値は式として保存しない
    p = get_point(store.active_sketch(), point.id)
    assert (p.x, p.y, p.xExpr, p.yExpr) == (5, 10, "a+2", None)
    store.update_param(pid, expression="25")                           # 変数を変えると点が動く
    p = get_point(store.active_sketch(), point.id)
    assert p.x == pytest.approx(27) and p.xExpr == "a+2"
    assert store.solve_info.dof == 1                                   # x は式で固定、y だけ自由
    # 評価できない式は拒否（文書は変えない）
    assert store.set_point_coords(point.id, "b+1", None) is False
    assert store.solve_info.status == "invalid" and get_point(store.active_sketch(), point.id).xExpr == "a+2"
    # 空欄は変えない、数値は定数化
    assert store.set_point_coords(point.id, "", "4") is True
    p = get_point(store.active_sketch(), point.id)
    assert (p.x, p.y, p.xExpr, p.yExpr) == (27, 4, "a+2", None)
    assert store.set_point_coords(point.id, "12.5", None) is True
    assert get_point(store.active_sketch(), point.id).xExpr is None
    assert store.undo() and get_point(store.active_sketch(), point.id).xExpr == "a+2"


def test_drag_clears_expression_in_one_undo_step(env):
    store, ctl = env
    pid = store.add_param()
    store.update_param(pid, name="a", expression="5")
    store.update_active_sketch(lambda s: add_point(s, (5, 5), x_expr="a", y_expr="a") and True)
    point = next(e for e in store.active_sketch().entities if isinstance(e, SketchPoint) and not e.projection)
    history = len(store.past)
    store.set_tool("select")
    ctl.pointer_move((5, 5))
    ctl.pointer_down((5, 5))
    ctl.pointer_move((7, 8))
    ctl.pointer_move((8, 12))                                          # グリッド 2 mm に乗る位置
    ctl.pointer_up((8, 12))
    p = get_point(store.active_sketch(), point.id)
    assert (p.xExpr, p.yExpr) == (None, None)
    assert point_pos(store.active_sketch(), point.id) == pytest.approx((8, 12), abs=1e-9)
    assert len(store.past) == history + 1                               # ドラッグは Undo 1 回分
    store.undo()
    p = get_point(store.active_sketch(), point.id)
    assert (p.x, p.y, p.xExpr, p.yExpr) == (5, 5, "a", "a")


def test_expression_point_stays_while_constraints_solve(env):
    store, ctl = env
    pid = store.add_param()
    store.update_param(pid, name="L", expression="30")
    store.set_tool("line")
    click(ctl, 0, 0)
    click(ctl, 20, 10)
    ctl.finish()
    sketch = store.active_sketch()
    line = next(e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection)
    assert line.p1 == "ref-origin"                                      # 原点から描くと参照の原点を共有する
    assert store.set_point_coords(line.p1, "0", "0") is False          # 参照の原点は編集しない
    assert store.set_point_coords(line.p2, "L", None)                   # z = L、r は自由（r = 10 のまま）
    # 参照の原点はもともと固定なので「固定」は付かない
    assert store.add_constraint(SketchConstraint(id=new_id(), type="fix", point=line.p1)) is False
    assert store.add_constraint(SketchConstraint(id=new_id(), type="distance", p1=line.p1, p2=line.p2, value="50"))
    sketch = store.active_sketch()
    x, y = point_pos(sketch, line.p2)
    assert x == pytest.approx(30) and abs(y) == pytest.approx(40, abs=1e-6)   # 30² + 40² = 50²
    store.update_param(pid, expression="14")
    sketch = store.active_sketch()
    x, y = point_pos(sketch, line.p2)
    assert x == pytest.approx(14) and abs(y) == pytest.approx(48, abs=1e-6)   # 14² + 48² = 50²


def test_arc_center_expression(env):
    store, ctl = env
    pid = store.add_param()
    store.update_param(pid, name="c", expression="10")
    store.set_tool("arc")
    click(ctl, 0, 0)
    click(ctl, 10, 0)
    click(ctl, 0, 10)
    arc = next(e for e in store.active_sketch().entities if isinstance(e, SketchArc))
    assert store.set_arc_center(arc.id, "c", "0")
    sketch = store.active_sketch()
    center = get_point(sketch, get_entity_center := arc.center)
    assert (center.x, center.xExpr) == (10, "c")
    assert get_entity_center == arc.center
    assert get_point(sketch, arc.center).x == pytest.approx(10)
    store.update_param(pid, expression="4")
    assert get_point(store.active_sketch(), arc.center).x == pytest.approx(4)
    assert store.expression_errors == []
    store.update_param(pid, expression="bad +")
    assert store.expression_errors and "c" in store.expression_errors[0]
