"""スケッチャのワークフロー（移植元: EM-CAD-py tests/test_sketch_workflow.py）.

ストア + コントローラ + ツール（GUI ウィジェット無し）。ver3 はスケッチが常に 1 枚なので
「スケッチ作成 / 終了」の代わりに new_document / load_document を使う。
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore       # noqa: E402
from axicavity_fem.gui.core.document import SketchCircle, SketchConstraint, SketchLine, new_id  # noqa: E402
from axicavity_fem.gui.core.serialize import parse_document, serialize_document  # noqa: E402
from axicavity_fem.gui.core.sketch.model import count_entities, point_pos  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def env(app):
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)              # 0.1 mm/px → スナップ許容 0.8 mm
    store.new_document()
    return store, controller


def click(controller, x, y, shift=False):
    controller.pointer_move((x, y), shift)
    controller.pointer_down((x, y), shift)
    controller.pointer_up((x, y), shift)


def test_draw_constrain_undo_and_reload(env):
    store, ctl = env
    assert ctl.active and store.document.sketch.name == "Sketch1"

    store.set_tool("rectangle")
    click(ctl, 2, 2)                       # 軸・原点から離す（軸への自動拘束が付かない位置）
    click(ctl, 42.2, 22.1)                 # グリッドへスナップ → (42, 22)
    sketch = store.active_sketch()
    assert count_entities(sketch) == {"point": 4, "line": 4, "circle": 0, "arc": 0}
    assert len(ctl.profiles) == 1 and abs(ctl.profiles[0].area - 800) < 1e-9
    assert ctl.profiles is store.profiles

    store.set_tool("circle")
    click(ctl, 22, 12)
    click(ctl, 28, 12)
    assert count_entities(sketch)["circle"] == 1
    assert len(ctl.profiles) == 2

    assert [c.type for c in sketch.constraints] == ["horizontal", "vertical", "horizontal", "vertical"]
    assert store.solve_info is not None and store.solve_info.status == "ok"
    dof_before = store.solve_info.dof
    store.set_tool("dimension")
    click(ctl, 28.1, 12)                   # 円周をクリック → 直線待ち（円周と直線の距離）
    assert store.hint == "hint.dimCircleNext" and not sketch.constraints[4:]
    ctl.finish()                           # Enter → 半径寸法（線の長さと同じ流儀。2026-09-29 ユーザー決定）
    radius = next(c for c in sketch.constraints if c.type == "radius")
    assert radius.value == "6"
    assert store.selected_constraint_id == radius.id
    assert store.solve_info.dof == dof_before - 1

    assert store.set_constraint_value(radius.id, "8") is True
    circle = next(e for e in sketch.entities if isinstance(e, SketchCircle))
    assert circle.radius == pytest.approx(8)
    store.add_constraint(SketchConstraint(id="bad", type="radius", curve=circle.id, value="9"))
    assert all(c.id != "bad" for c in sketch.constraints)
    assert store.solve_info.status == "conflict"

    assert store.undo()
    assert store.active_sketch().constraints[-1].value == "6"
    assert store.redo()
    sketch = store.active_sketch()
    assert next(e for e in sketch.entities if isinstance(e, SketchCircle)).radius == pytest.approx(8)

    text = serialize_document(store.document)
    restored = parse_document(text)
    assert restored == store.document
    store.load_document(restored)
    assert ctl.active and len(ctl.profiles) == 2 and not store.can_undo()


def test_select_drag_and_delete(env):
    store, ctl = env
    store.set_tool("line")
    click(ctl, 4, 4)                       # 軸・原点から離す
    click(ctl, 34, 4)
    ctl.finish()
    sketch = store.active_sketch()
    line = next(e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection)
    end = line.p2

    store.set_tool("select")
    ctl.pointer_move((34, 4))
    assert store.hover_id == end
    ctl.pointer_down((34, 4))
    ctl.pointer_move((37, 8))
    ctl.pointer_move((40, 12))
    ctl.pointer_up((40, 12))
    assert point_pos(sketch, end) == pytest.approx((40, 12), abs=1e-9)
    assert store.can_undo()
    store.undo()
    sketch = store.active_sketch()
    assert point_pos(sketch, end) == pytest.approx((34, 4), abs=1e-9)

    store.set_tool("horizontal")
    click(ctl, 19, 4.1)
    store.set_tool("dimension")
    click(ctl, 19, 4.1)
    ctl.finish()
    sketch = store.active_sketch()
    assert [c.type for c in sketch.constraints] == ["horizontal", "distance"]
    store.set_tool("select")
    ctl.pointer_move((34, 4))
    ctl.pointer_down((34, 4))
    ctl.pointer_move((29, 10))
    ctl.pointer_up((29, 10))
    a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
    assert a[1] == pytest.approx(b[1], abs=1e-9)
    assert abs(b[0] - a[0]) == pytest.approx(30, abs=1e-9)

    ctl.pointer_move((15, a[1]))
    ctl.pointer_down((15, a[1]))
    ctl.pointer_up((15, a[1]))
    assert store.selection == [line.id]
    ctl.delete_selection()
    assert count_entities(store.active_sketch()) == {"point": 0, "line": 0, "circle": 0, "arc": 0}
    assert store.active_sketch().constraints == []


def test_escape_cancel_and_tool_switch(env):
    store, ctl = env
    store.set_tool("rectangle")
    click(ctl, 0, 0)
    assert ctl.tool.busy()
    ctl.cancel()
    assert not ctl.tool.busy() and store.active_tool == "rectangle"
    ctl.cancel()
    assert store.active_tool == "select"
    assert store.hint == "hint.select"


def test_params_drive_dimensions(env):
    store, ctl = env
    store.set_tool("rectangle")
    click(ctl, 2, 2)                       # 軸・原点から離す
    click(ctl, 22, 12)
    sketch = store.active_sketch()
    bottom = next(e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection
                  and point_pos(sketch, e.p1)[1] == 2 and point_pos(sketch, e.p2)[1] == 2)
    store.set_tool("fix")
    click(ctl, 2, 2)
    assert [c.type for c in sketch.constraints][-1] == "fix" and len(sketch.constraints) == 5
    pid = store.add_param()
    store.update_param(pid, name="W", expression="20")
    assert store.add_constraint(SketchConstraint(id=new_id(), type="distance", p1=bottom.p1,
                                                 p2=bottom.p2, value="W"))
    store.update_param(pid, expression="35")
    sketch = store.active_sketch()
    a, b = point_pos(sketch, bottom.p1), point_pos(sketch, bottom.p2)
    assert abs(b[0] - a[0]) == pytest.approx(35, abs=1e-6)
    assert point_pos(sketch, next(c.point for c in sketch.constraints if c.type == "fix")) == (2, 2)
