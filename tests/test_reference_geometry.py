"""参照ジオメトリ（原点・z 軸・r 軸）と、点を軸に載せる拘束（「r=0 に固定」「z=0 に固定」・作図の自動拘束）.

- 新規・読込の文書には必ず参照要素がある。古いファイルは参照要素を足してもハッシュが変わらない（未保存にならない・
  結果が「形状変更前」にならない）
- 参照要素は消せない・動かない・数に入らない・閉領域に入らない。選択ツールでは選べず、拘束ツールでは選べる
- 固定ボタンは「点を曲線上に（軸）」の拘束を付ける。点は軸に沿って動き、軸から外れない。線を選ぶと両端
- 軸の上に作図した点には同じ拘束が自動で付く（Ctrl で付けない）。両端が同じ軸の上になった線の水平 / 垂直は外す
- 追加した拘束で冗長になった軸の拘束は外す（例: 同心で点が原点に来た）
"""

import copy
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.core.convert import to_multi_region      # noqa: E402
from axicavity_fem.gui.core.document import (                   # noqa: E402
    REFERENCE_ORIGIN,
    REFERENCE_R_AXIS,
    REFERENCE_Z_AXIS,
    SketchCircle,
    SketchLine,
    SketchPoint,
    is_reference,
)
from axicavity_fem.gui.core.hashes import geometry_hash, model_hash  # noqa: E402
from axicavity_fem.gui.core.serialize import document_from_dict, document_to_dict  # noqa: E402
from axicavity_fem.gui.core.sketch.model import count_entities, delete_entities, hit_test, point_pos  # noqa: E402
from axicavity_fem.gui.core.sketch.reference import REFERENCE_IDS, ensure_reference_geometry  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402

AXES = (REFERENCE_Z_AXIS, REFERENCE_R_AXIS)


@pytest.fixture
def env():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)              # グリッド 2 mm、スナップ許容 0.8 mm
    store.new_document()
    return store, controller


def click(controller, x, y, ctrl=False):
    controller.pointer_move((x, y))
    controller.pointer_down((x, y), ctrl=ctrl)
    controller.pointer_up((x, y), ctrl=ctrl)


def draw_rectangle(store, controller, x0, y0, x1, y1, ctrl=False):
    store.set_tool("rectangle")
    click(controller, x0, y0, ctrl)
    click(controller, x1, y1, ctrl)
    store.set_tool("select")


def draw_line(store, controller, a, b, ctrl=False):
    store.set_tool("line")
    click(controller, *a, ctrl=ctrl)
    click(controller, *b, ctrl=ctrl)
    controller.finish()
    store.set_tool("select")
    sketch = store.active_sketch()
    return [e for e in sketch.entities if isinstance(e, SketchLine) and not is_reference(e)][-1]


def user_point_at(store, pos):
    sketch = store.active_sketch()
    return next(e for e in sketch.entities if isinstance(e, SketchPoint) and not is_reference(e)
                and point_pos(sketch, e.id) == pytest.approx(pos, abs=1e-9))


def axis_constraints(sketch):
    """{点 ID: {"r" | "z", ...}}（"r" = z 軸の上 = r が 0、"z" = r 軸の上 = z が 0）."""
    out: dict = {}
    for c in sketch.constraints:
        if c.type == "pointOnCurve" and c.curve in AXES:
            out.setdefault(c.point, set()).add("r" if c.curve == REFERENCE_Z_AXIS else "z")
    return out


def strip_references(doc_dict):
    doc_dict = copy.deepcopy(doc_dict)
    doc_dict["sketch"]["entities"] = [e for e in doc_dict["sketch"]["entities"] if e.get("projection") != "reference"]
    return doc_dict


# ---------------------------------------------------------------------------
# 文書と参照要素
# ---------------------------------------------------------------------------

def test_new_document_has_reference_geometry(env):
    store, _ = env
    sketch = store.active_sketch()
    ids = [e.id for e in sketch.entities]
    assert set(REFERENCE_IDS) <= set(ids)
    assert all(is_reference(e) and e.construction for e in sketch.entities if e.id in REFERENCE_IDS)
    assert point_pos(sketch, REFERENCE_ORIGIN) == (0.0, 0.0)
    assert sum(count_entities(sketch).values()) == 0                   # 参照要素は数えない
    assert store.profiles == []
    assert store.solve_info.status == "ok"


def test_old_document_gets_references_without_changing_hashes(env):
    store, controller = env
    draw_rectangle(store, controller, 0, 0, 100, 50)
    old = strip_references(document_to_dict(store.document))
    doc = document_from_dict(old)
    assert not any(is_reference(e) for e in doc.sketch.entities)
    before = (geometry_hash(doc), model_hash(doc))
    assert ensure_reference_geometry(doc.sketch) is True
    assert (geometry_hash(doc), model_hash(doc)) == before
    assert ensure_reference_geometry(doc.sketch) is False            # 2 回目は何もしない


def test_old_project_opens_clean(env, tmp_path):
    from axicavity_fem.gui.project.controller import ProjectController

    store, controller = env
    project = ProjectController(store, generator="test")
    project.new()
    draw_rectangle(store, controller, 2, 2, 42, 22)
    path = project.save_as(tmp_path / "old.axiproj")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["document"] = strip_references(manifest["document"])     # 参照要素の無い古いファイルを模す
    path.write_text(json.dumps(manifest), encoding="utf-8")

    project.open(path)
    assert set(REFERENCE_IDS) <= {e.id for e in store.active_sketch().entities}
    assert not project.is_dirty()
    assert len(store.profiles) == 1


def test_broken_reference_positions_are_repaired(env):
    store, _ = env
    raw = document_to_dict(store.document)
    for e in raw["sketch"]["entities"]:
        if e["id"] == REFERENCE_ORIGIN:
            e["x"], e["y"] = 5.0, 7.0
    store.load_document(document_from_dict(raw))
    assert point_pos(store.active_sketch(), REFERENCE_ORIGIN) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 消えない・動かない・数えない
# ---------------------------------------------------------------------------

def test_references_cannot_be_deleted_or_edited(env):
    store, _ = env
    sketch = copy.deepcopy(store.active_sketch())
    assert delete_entities(sketch, list(REFERENCE_IDS)) is False
    assert set(REFERENCE_IDS) <= {e.id for e in sketch.entities}
    assert store.set_point_coords(REFERENCE_ORIGIN, "5", "5") is False
    assert point_pos(store.active_sketch(), REFERENCE_ORIGIN) == (0.0, 0.0)


def test_references_excluded_from_counts_profiles_and_conversion(env):
    store, controller = env
    draw_rectangle(store, controller, 0, 0, 100, 50)
    sketch = store.active_sketch()
    counts = count_entities(sketch)
    assert counts["point"] == 3 and counts["line"] == 4                # 原点の角は参照の原点を共有する
    assert len(store.profiles) == 1
    result = to_multi_region(store.document)
    assert len(result.geom.points) == 4 and len(result.geom.segments) == 4
    assert {(round(z, 9), round(r, 9)) for z, r in result.geom.points} == {(0, 0), (100, 0), (100, 50), (0, 50)}


def test_unit_conversion_keeps_references(env):
    store, controller = env
    draw_rectangle(store, controller, 2, 2, 42, 22)
    store.set_units("cm", convert_values=True)
    sketch = store.active_sketch()
    assert point_pos(sketch, REFERENCE_ORIGIN) == (0.0, 0.0)
    assert point_pos(sketch, "ref-z-axis-end") == (1.0, 0.0)
    assert point_pos(sketch, "ref-r-axis-end") == (0.0, 1.0)
    assert user_point_at(store, (4.2, 2.2))


def test_hit_test_reference_axes_as_infinite_lines(env):
    store, controller = env
    sketch = store.active_sketch()
    far = hit_test(sketch, (500.0, 0.2), 0.8)
    assert far is not None and far.id == REFERENCE_Z_AXIS                 # 軸の線分 (0..1) の外でも当たる
    assert hit_test(sketch, (0.3, 300.0), 0.8).id == REFERENCE_R_AXIS
    assert hit_test(sketch, (500.0, 0.2), 0.8, references=False) is None
    assert hit_test(sketch, (0.1, 0.1), 0.8).id == REFERENCE_ORIGIN
    assert hit_test(sketch, (0.0, 0.0), 0.8, references=False) is None
    assert hit_test(sketch, (1.0, 0.0), 0.8).id != "ref-z-axis-end"      # 軸を決めるだけの点は選べない
    line = draw_line(store, controller, (10, 0), (50, 0), ctrl=True)
    assert hit_test(store.active_sketch(), (30.0, 0.1), 0.8).id == line.id   # 利用者の線が優先


def test_select_tool_does_not_pick_or_drag_references(env):
    store, controller = env
    store.set_tool("select")
    click(controller, 300, 0)
    click(controller, 0, 0)
    assert store.selection == [] or not any(i in REFERENCE_IDS for i in store.selection)
    controller.pointer_move((0, 0))
    controller.pointer_down((0, 0))
    controller.pointer_move((20, 20))
    controller.pointer_up((20, 20))
    assert point_pos(store.active_sketch(), REFERENCE_ORIGIN) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 「r=0 に固定」「z=0 に固定」
# ---------------------------------------------------------------------------

def test_fix_r0_constrains_line_endpoints_and_they_slide_along_the_axis(env):
    store, controller = env
    line = draw_line(store, controller, (10, 6), (50, 20))
    assert store.fix_points_to_zero([line.id], "r") is True
    sketch = store.active_sketch()
    assert point_pos(sketch, line.p1) == pytest.approx((10.0, 0.0))       # z はそのまま、r だけ 0 に
    assert point_pos(sketch, line.p2) == pytest.approx((50.0, 0.0))
    assert axis_constraints(sketch) == {line.p1: {"r"}, line.p2: {"r"}}
    assert store.solve_info.status == "ok"

    ok = store.drag_solve(copy.deepcopy(sketch), {line.p2: (70.0, 15.0)}, history=True)
    assert ok
    assert point_pos(store.active_sketch(), line.p2) == pytest.approx((70.0, 0.0))   # 軸に沿って滑る

    store.undo()
    store.undo()
    assert axis_constraints(store.active_sketch()) == {}
    assert point_pos(store.active_sketch(), line.p1) == pytest.approx((10.0, 6.0))


def test_fix_z0_puts_point_on_r_axis_and_clears_expression(env):
    store, controller = env
    store.set_tool("point")
    click(controller, 6, 30)
    store.set_tool("select")
    point = user_point_at(store, (6, 30))
    assert store.set_point_coords(point.id, "3*2", None)
    assert store.fix_points_to_zero([point.id], "z") is True
    sketch = store.active_sketch()
    p = next(e for e in sketch.entities if e.id == point.id)
    assert (p.x, p.y) == pytest.approx((0.0, 30.0)) and p.xExpr is None
    assert axis_constraints(sketch) == {point.id: {"z"}}
    assert store.fix_points_to_zero([point.id], "z") is False            # 2 回目は何もしない


def test_typing_a_coordinate_releases_the_axis_constraint(env):
    store, controller = env
    line = draw_line(store, controller, (10, 0), (50, 0))
    assert axis_constraints(store.active_sketch()) == {line.p1: {"r"}, line.p2: {"r"}}
    assert store.set_point_coords(line.p2, "60", None)                   # z だけ変える: 軸の上のまま
    assert axis_constraints(store.active_sketch())[line.p2] == {"r"}
    assert store.set_point_coords(line.p2, None, "8")                    # r に 0 以外: 軸から離す
    sketch = store.active_sketch()
    assert point_pos(sketch, line.p2) == pytest.approx((60.0, 8.0))
    assert line.p2 not in axis_constraints(sketch)
    pid = store.add_param()
    store.update_param(pid, name="h", expression="3")
    assert store.set_point_coords(line.p1, None, "h")                     # 式でも離す
    sketch = store.active_sketch()
    assert point_pos(sketch, line.p1) == pytest.approx((10.0, 3.0)) and axis_constraints(sketch) == {}
    assert store.solve_info.status == "ok"


def test_fix_drops_redundant_horizontal(env):
    store, controller = env
    from axicavity_fem.gui.core.document import SketchConstraint, new_id
    line = draw_line(store, controller, (10, 6), (50, 6))
    assert store.add_constraint(SketchConstraint(id=new_id(), type="horizontal", line=line.id))
    assert store.fix_points_to_zero([line.id], "r")
    sketch = store.active_sketch()
    assert sorted(c.type for c in sketch.constraints) == ["pointOnCurve", "pointOnCurve"]
    assert store.solve_info.status == "ok"


# ---------------------------------------------------------------------------
# 作図の自動拘束
# ---------------------------------------------------------------------------

def test_rectangle_from_origin_is_constrained_to_the_axes(env):
    store, controller = env
    draw_rectangle(store, controller, 0, 0, 100, 50)
    sketch = store.active_sketch()
    user_lines = [e for e in sketch.entities if isinstance(e, SketchLine) and not is_reference(e)]
    assert sum(REFERENCE_ORIGIN in (e.p1, e.p2) for e in user_lines) == 2      # 角が原点を共有
    on_z, on_r = user_point_at(store, (100, 0)), user_point_at(store, (0, 50))
    assert axis_constraints(sketch) == {on_z.id: {"r"}, on_r.id: {"z"}}
    types = sorted(c.type for c in sketch.constraints)
    assert types == ["horizontal", "pointOnCurve", "pointOnCurve", "vertical"]  # 軸の上の辺の水平 / 垂直は外れる
    assert store.solve_info.status == "ok" and store.solve_info.dof == 2

    corner = user_point_at(store, (100, 50))
    assert store.drag_solve(copy.deepcopy(sketch), {corner.id: (120.0, 60.0)}, history=True)
    sketch = store.active_sketch()
    assert point_pos(sketch, on_z.id) == pytest.approx((120.0, 0.0))
    assert point_pos(sketch, on_r.id) == pytest.approx((0.0, 60.0))
    assert point_pos(sketch, REFERENCE_ORIGIN) == (0.0, 0.0)


def test_line_on_axis_gets_axis_constraints_not_horizontal(env):
    store, controller = env
    line = draw_line(store, controller, (10, 0), (50, 0))
    sketch = store.active_sketch()
    assert axis_constraints(sketch) == {line.p1: {"r"}, line.p2: {"r"}}
    assert [c.type for c in sketch.constraints] == ["pointOnCurve", "pointOnCurve"]
    assert store.solve_info.status == "ok"


def test_ctrl_suppresses_auto_axis_constraints(env):
    store, controller = env
    draw_line(store, controller, (10, 0), (50, 0), ctrl=True)
    assert store.active_sketch().constraints == []
    draw_rectangle(store, controller, 0, 10, 20, 30, ctrl=True)
    assert axis_constraints(store.active_sketch()) == {}


def test_point_off_axis_gets_no_constraint(env):
    store, controller = env
    draw_line(store, controller, (10, 6), (50, 20))
    assert axis_constraints(store.active_sketch()) == {}


def test_expression_point_is_not_auto_constrained(env):
    store, controller = env
    store.set_tool("point")
    click(controller, 20, 10)
    store.set_tool("select")
    point = user_point_at(store, (20, 10))
    store.set_point_coords(point.id, None, "0")                          # r を式で 0 に
    assert axis_constraints(store.active_sketch()) == {}                  # 式は式で決まる（二重に決めない）


# ---------------------------------------------------------------------------
# 拘束ツールで軸・原点を選ぶ
# ---------------------------------------------------------------------------

def test_point_on_curve_tool_picks_axis(env):
    store, controller = env
    store.set_tool("point")
    click(controller, 20, 10, ctrl=True)
    point = user_point_at(store, (20, 10))
    store.set_tool("pointOnCurve")
    click(controller, 20, 10)
    click(controller, 300, 0.2)                                           # 軸の線分の外
    sketch = store.active_sketch()
    assert axis_constraints(sketch) == {point.id: {"r"}}
    assert point_pos(sketch, point.id)[1] == pytest.approx(0.0, abs=1e-9)


def test_coincident_with_origin_merges_into_the_reference(env):
    store, controller = env
    line = draw_line(store, controller, (10, 0), (50, 0))
    store.set_tool("coincident")
    click(controller, 10, 0)
    click(controller, 0, 0)
    sketch = store.active_sketch()
    merged = next(e for e in sketch.entities if e.id == line.id)
    assert REFERENCE_ORIGIN in (merged.p1, merged.p2)
    assert point_pos(sketch, REFERENCE_ORIGIN) == (0.0, 0.0)
    # 原点の「軸の上」は自明なので残らない
    assert all(c.point != REFERENCE_ORIGIN for c in sketch.constraints if c.type == "pointOnCurve")
    assert store.solve_info.status == "ok"


def test_constraint_between_references_only_is_rejected(env):
    store, controller = env
    from axicavity_fem.gui.core.document import SketchConstraint, new_id
    assert store.add_constraint(SketchConstraint(id=new_id(), type="horizontal", line=REFERENCE_Z_AXIS)) is False
    assert store.active_sketch().constraints == []
    hints = []
    store.hint_changed.connect(hints.append)
    store.set_tool("horizontal")
    click(controller, 300, 0.2)                                           # z 軸に水平 → 付かない
    assert store.active_sketch().constraints == []
    assert hints[-1] == "hint.referenceGeometry"


def test_equal_with_an_axis_is_refused(env):
    store, controller = env
    line = draw_line(store, controller, (10, 6), (50, 20))
    store.set_tool("equal")
    click(controller, 30, 13)
    click(controller, 300, 0.2)
    sketch = store.active_sketch()
    assert sketch.constraints == []
    assert point_pos(sketch, line.p2) == pytest.approx((50.0, 20.0))


def test_dimension_from_point_to_axis_and_tangent_to_axis(env):
    store, controller = env
    store.set_tool("point")
    click(controller, 20, 10, ctrl=True)
    point = user_point_at(store, (20, 10))
    store.set_tool("dimension")
    click(controller, 20, 10)
    click(controller, 300, 0.2)                                           # 点と z 軸の距離 = r 座標
    sketch = store.active_sketch()
    dim = sketch.constraints[-1]
    assert (dim.type, dim.point, dim.line, dim.value) == ("pointLineDistance", point.id, REFERENCE_Z_AXIS, "10")
    assert store.set_constraint_value(dim.id, "12")
    assert point_pos(store.active_sketch(), point.id)[1] == pytest.approx(12.0)

    store.set_tool("circle")
    click(controller, 60, 20, ctrl=True)
    click(controller, 66, 20, ctrl=True)
    circle = next(e for e in store.active_sketch().entities if isinstance(e, SketchCircle))
    store.set_tool("tangent")
    click(controller, 66.1, 20)
    click(controller, 300, 0.2)
    sketch = store.active_sketch()
    assert sketch.constraints[-1].type == "tangent"
    center = point_pos(sketch, circle.center)
    circle = next(e for e in sketch.entities if e.id == circle.id)
    assert abs(center[1]) == pytest.approx(circle.radius, abs=1e-6)       # 円が z 軸に接する


def test_symmetric_about_the_z_axis(env):
    store, controller = env
    store.set_tool("point")
    click(controller, 20, 10, ctrl=True)
    click(controller, 30, -8, ctrl=True)
    a, b = user_point_at(store, (20, 10)), user_point_at(store, (30, -8))
    store.set_tool("symmetric")
    click(controller, 20, 10)
    click(controller, 30, -8)
    click(controller, 300, 0.2)                                           # 対称軸 = z 軸
    sketch = store.active_sketch()
    pa, pb = point_pos(sketch, a.id), point_pos(sketch, b.id)
    assert pa[0] == pytest.approx(pb[0], abs=1e-6) and pa[1] == pytest.approx(-pb[1], abs=1e-6)
    assert store.solve_info.status == "ok"


def test_concentric_drops_implied_axis_constraint(env):
    store, controller = env
    store.set_tool("circle")
    click(controller, 0, 0)
    click(controller, 10, 0)
    click(controller, 30, 0)
    click(controller, 36, 0)
    sketch = store.active_sketch()
    c1, c2 = [e for e in sketch.entities if isinstance(e, SketchCircle)]
    assert c1.center == REFERENCE_ORIGIN
    assert axis_constraints(sketch).get(c2.center) == {"r"}
    store.set_tool("concentric")
    click(controller, 0, 10.1)
    click(controller, 36.1, 0)
    sketch = store.active_sketch()
    assert [c.type for c in sketch.constraints] == ["concentric"]          # c2 中心の軸の拘束は同心から従う
    assert store.solve_info.status == "ok"
    assert point_pos(sketch, c2.center) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_point_on_axis_twice_replaces_the_old_constraint(env):
    store, controller = env
    line = draw_line(store, controller, (10, 0), (50, 0))
    store.set_tool("pointOnCurve")
    click(controller, 50, 0)
    click(controller, 300, 0.2)
    sketch = store.active_sketch()
    assert [c.point for c in sketch.constraints if c.type == "pointOnCurve"].count(line.p2) == 1
    assert store.solve_info.status == "ok"


# ---------------------------------------------------------------------------
# 拘束の一覧
# ---------------------------------------------------------------------------

def test_constraint_list_names_the_axis(env, qtbot):
    from axicavity_fem.gui.i18n.tr import set_language
    from axicavity_fem.gui.ui.sketch_panels import SketchPropertyPanel

    set_language("ja")
    store, controller = env
    panel = SketchPropertyPanel(store)
    qtbot.addWidget(panel)
    draw_line(store, controller, (10, 0), (50, 6))
    table = panel.constraints.table
    names = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert names == ["点を曲線上に（P1 – z 軸）"]
