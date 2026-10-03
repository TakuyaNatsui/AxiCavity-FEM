"""スケッチツール（移植元: EM-CAD-py tests/test_sketch_tools.py）.

ポインタ入力を SketchController に直接与える（Qt のウィジェット無し。QObject のシグナルは同期で届く）。
"""

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore       # noqa: E402
from axicavity_fem.gui.core.document import SketchArc, SketchCircle, SketchLine, SketchPoint  # noqa: E402
from axicavity_fem.gui.core.sketch.model import arc_params, count_entities, point_pos  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402


@pytest.fixture
def env():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)              # グリッド 2 mm、スナップ許容 0.8 mm
    store.new_document()
    return store, controller


def click(controller, x, y, shift=False):
    controller.pointer_move((x, y), shift)
    controller.pointer_down((x, y), shift)
    controller.pointer_up((x, y), shift)


REFERENCE_AXES = ("ref-z-axis", "ref-r-axis")


def user_constraints(sketch):
    """参照の軸への拘束（作図で自動に付く）を除いた拘束の種類."""
    return [c.type for c in sketch.constraints if c.curve not in REFERENCE_AXES]


def lines(sketch):
    return [e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection]


def test_arc_tool(env):
    store, ctl = env
    store.set_tool("arc")
    click(ctl, 0, 0)                       # 中心
    click(ctl, 10, 0)                      # 始点
    assert ctl.preview is not None or ctl.tool.busy()
    click(ctl, 0, 10)                      # 終点方向（反時計回りに 90°）
    sketch = store.active_sketch()
    arc = next(e for e in sketch.entities if isinstance(e, SketchArc))
    p = arc_params(sketch, arc)
    assert p.radius == pytest.approx(10)
    assert p.sweep == pytest.approx(math.pi / 2)
    assert count_entities(sketch)["point"] == 2            # 中心は参照の原点を共有する
    assert arc.center == "ref-origin"
    assert not ctl.tool.busy()

    store.set_tool("line")
    click(ctl, 0, 10.3)
    click(ctl, 0, 0.2)
    ctl.finish()
    sketch = store.active_sketch()
    assert count_entities(sketch)["point"] == 2            # 線の端は円弧の端点と原点を共有
    assert len(ctl.profiles) == 0          # 円弧 + 線 + 中心を結ぶ線がまだ無い
    click(ctl, 0, 0)
    click(ctl, 10, 0)
    ctl.finish()
    assert len(ctl.profiles) == 1          # 扇形（面積は円弧の折れ線近似）
    assert ctl.profiles[0].area == pytest.approx(math.pi * 100 / 4, rel=3e-3)


def test_polygon_and_point_tools(env):
    store, ctl = env
    store.set_tool_options(sides=5)
    store.set_tool("polygon")
    click(ctl, 0, 0)
    click(ctl, 20, 0)
    sketch = store.active_sketch()
    assert count_entities(sketch) == {"point": 5, "line": 5, "circle": 0, "arc": 0}
    assert len(ctl.profiles) == 1
    store.set_tool("point")
    click(ctl, 40, 40)
    assert count_entities(store.active_sketch())["point"] == 6
    assert any(isinstance(e, SketchPoint) and not e.projection and (e.x, e.y) == (40, 40) for e in store.active_sketch().entities)


def test_construction_toggle_excludes_profiles(env):
    store, ctl = env
    store.set_tool("rectangle")
    click(ctl, 0, 0)
    click(ctl, 20, 10)
    assert len(ctl.profiles) == 1
    store.set_tool("select")
    click(ctl, 10, 0.2)                    # 下辺を選択
    line_id = store.selection[0]
    store.toggle_construction([line_id])
    sketch = store.active_sketch()
    assert next(e for e in lines(sketch) if e.id == line_id).construction is True
    assert len(ctl.profiles) == 0          # 構築線は閉領域に使わない
    store.toggle_construction([line_id])
    assert len(ctl.profiles) == 1


def test_two_pick_constraints(env):
    store, ctl = env
    store.set_tool("line")
    click(ctl, 0, 0)
    click(ctl, 20, 2)
    ctl.finish()
    click(ctl, 0, 10)
    click(ctl, 20, 14)
    ctl.finish()
    sketch = store.active_sketch()
    l1, l2 = lines(sketch)

    store.set_tool("parallel")
    click(ctl, 10, 1)
    assert store.selection == [l1.id] and ctl.tool.busy()
    click(ctl, 10, 12)
    sketch = store.active_sketch()
    assert user_constraints(sketch) == ["parallel"]      # 軸の上の点に付いた自動の拘束は除く
    assert not ctl.tool.busy() and store.selection == []
    d1 = point_pos(sketch, l1.p2)[1] - point_pos(sketch, l1.p1)[1]
    d2 = point_pos(sketch, l2.p2)[1] - point_pos(sketch, l2.p1)[1]
    x1 = point_pos(sketch, l1.p2)[0] - point_pos(sketch, l1.p1)[0]
    x2 = point_pos(sketch, l2.p2)[0] - point_pos(sketch, l2.p1)[0]
    assert d1 * x2 == pytest.approx(d2 * x1, abs=1e-6)

    store.set_tool("equal")
    click(ctl, 10, point_pos(sketch, l1.p1)[1] + 0.5 * d1)
    click(ctl, 10, point_pos(sketch, l2.p1)[1] + 0.5 * d2)
    sketch = store.active_sketch()
    assert user_constraints(sketch) == ["parallel", "equal"]
    len1 = math.dist(point_pos(sketch, l1.p1), point_pos(sketch, l1.p2))
    len2 = math.dist(point_pos(sketch, l2.p1), point_pos(sketch, l2.p2))
    assert len1 == pytest.approx(len2, abs=1e-6)

    store.set_tool("coincident")
    click(ctl, *point_pos(sketch, l1.p2))
    click(ctl, *point_pos(sketch, l2.p1))
    sketch = store.active_sketch()
    assert count_entities(sketch)["point"] == 2            # 1 本目の始点は参照の原点（数えない）
    assert len(lines(sketch)) == 2
    assert store.solve_info.status == "ok"


def test_tangent_concentric_symmetric(env):
    store, ctl = env
    store.set_tool("circle")
    click(ctl, 0, 0)
    click(ctl, 10, 0)
    click(ctl, 30, 0)
    click(ctl, 36, 0)
    store.set_tool("line")
    click(ctl, -20, 14)
    click(ctl, 40, 14)
    ctl.finish()
    sketch = store.active_sketch()
    c1, c2 = [e for e in sketch.entities if isinstance(e, SketchCircle)]
    line = lines(sketch)[0]

    store.set_tool("tangent")
    click(ctl, 0, 10.2)                    # 円 1 の円周
    click(ctl, 10, 14.1)                   # 線
    sketch = store.active_sketch()
    assert user_constraints(sketch) == ["tangent"]
    c1 = next(e for e in sketch.entities if e.id == c1.id)
    center = point_pos(sketch, c1.center)
    a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
    dx, dy = b[0] - a[0], b[1] - a[1]
    dist = abs(dx * (a[1] - center[1]) - dy * (a[0] - center[0])) / math.hypot(dx, dy)
    assert dist == pytest.approx(c1.radius, abs=1e-6)

    store.set_tool("concentric")
    c2 = next(e for e in sketch.entities if e.id == c2.id)
    click(ctl, point_pos(sketch, c1.center)[0], point_pos(sketch, c1.center)[1] + c1.radius + 0.1)
    click(ctl, point_pos(sketch, c2.center)[0] + c2.radius + 0.1, point_pos(sketch, c2.center)[1])
    sketch = store.active_sketch()
    assert user_constraints(sketch) == ["tangent", "concentric"]
    c1 = next(e for e in sketch.entities if e.id == c1.id)
    c2 = next(e for e in sketch.entities if e.id == c2.id)
    assert point_pos(sketch, c1.center) == pytest.approx(point_pos(sketch, c2.center), abs=1e-6)

    store.set_tool("line")
    click(ctl, 60, -20)
    click(ctl, 60, 40)
    ctl.finish()
    store.set_tool("point")
    click(ctl, 50, 2)                      # 軸（r = 0）の上だと軸への拘束が自動で付くので離す
    click(ctl, 74, 6)
    sketch = store.active_sketch()
    axis = lines(sketch)[-1]
    p1 = next(e for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection and (e.x, e.y) == (50, 2))
    p2 = next(e for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection and (e.x, e.y) == (74, 6))
    store.set_tool("symmetric")
    click(ctl, 50, 2)
    click(ctl, 74, 6)
    click(ctl, 60.1, 20)
    sketch = store.active_sketch()
    assert [c.type for c in sketch.constraints][-1] == "symmetric"
    q1, q2 = point_pos(sketch, p1.id), point_pos(sketch, p2.id)
    a, b = point_pos(sketch, axis.p1), point_pos(sketch, axis.p2)
    ux, uy = b[0] - a[0], b[1] - a[1]
    mx, my = (q1[0] + q2[0]) / 2, (q1[1] + q2[1]) / 2
    assert ux * (my - a[1]) - uy * (mx - a[0]) == pytest.approx(0, abs=1e-6)
    assert ux * (q2[0] - q1[0]) + uy * (q2[1] - q1[1]) == pytest.approx(0, abs=1e-6)
    assert store.solve_info.status == "ok"


def test_numeric_input_line_and_circle(env):
    store, ctl = env
    store.set_tool("line")
    click(ctl, 0, 0)
    ctl.pointer_move((7, 0.4))
    ctl.numeric("length", 25)
    sketch = store.active_sketch()
    line = lines(sketch)[0]
    assert point_pos(sketch, line.p2) == pytest.approx((25, 0), abs=1e-9)
    assert ctl.tool.busy()
    ctl.finish()

    store.set_tool("circle")
    click(ctl, 40, 40)
    ctl.numeric("radius", 7.5)
    sketch = store.active_sketch()
    circle = next(e for e in sketch.entities if isinstance(e, SketchCircle))
    assert circle.radius == pytest.approx(7.5)
    assert not ctl.tool.busy()


def test_point_line_and_parallel_line_distances(env):
    store, ctl = env
    store.set_tool("rectangle")
    click(ctl, 0, 0)
    click(ctl, 20, 10)
    store.set_tool("point")
    click(ctl, 10, 30)
    sketch = store.active_sketch()
    point = next(e for e in sketch.entities if isinstance(e, SketchPoint) and not e.projection and (e.x, e.y) == (10, 30))

    def top_bottom():
        s = store.active_sketch()
        ys = sorted((point_pos(s, ln.p1)[1] + point_pos(s, ln.p2)[1]) / 2 for ln in lines(s)
                    if abs(point_pos(s, ln.p1)[1] - point_pos(s, ln.p2)[1]) < 1e-9)
        return ys[-1], ys[0]

    store.set_tool("dimension")
    click(ctl, 10, 0.2)
    click(ctl, 10, 9.8)
    c = store.active_sketch().constraints[-1]
    assert c.type == "pointLineDistance" and float(c.value) == pytest.approx(10)
    assert store.set_constraint_value(c.id, "14")
    top, bottom = top_bottom()
    assert top - bottom == pytest.approx(14, abs=1e-6)
    assert store.solve_info.status == "ok"

    click(ctl, 10, 30)
    click(ctl, 10, top - 0.2)
    c = store.active_sketch().constraints[-1]
    assert c.type == "pointLineDistance" and c.point == point.id
    assert float(c.value) == pytest.approx(30 - top, abs=1e-3)
    assert store.set_constraint_value(c.id, "5")
    s = store.active_sketch()
    top, _ = top_bottom()
    assert point_pos(s, point.id)[1] - top == pytest.approx(5, abs=1e-6)

    click(ctl, 19.8, (top + bottom) / 2)
    click(ctl, *point_pos(s, point.id))
    c = store.active_sketch().constraints[-1]
    assert c.type == "pointLineDistance" and c.point == point.id
    assert float(c.value) == pytest.approx(10, abs=1e-3)


def test_point_line_distance_annotations(env):
    from axicavity_fem.gui.sketch.annotations import build_annotations
    store, ctl = env
    store.set_tool("rectangle")                  # 軸から離して描く（原点を共有しない）
    click(ctl, 2, 2)
    click(ctl, 42, 22)
    store.set_tool("point")
    click(ctl, 32, 42)
    store.set_tool("dimension")
    click(ctl, 22, 2.2)
    click(ctl, 22, 21.8)
    click(ctl, 32, 42)
    click(ctl, 12, 21.8)
    sketch = store.active_sketch()
    dims = [c for c in sketch.constraints if c.type == "pointLineDistance"]
    assert len(dims) == 2
    ann = build_annotations(sketch, {}, None, 0.1)
    segments = {tuple(sorted((round(x, 6), round(y, 6)) for x, y in seg)) for seg in ann.lines}
    assert ((22.0, 2.0), (22.0, 22.0)) in segments
    assert ((32.0, 22.0), (32.0, 42.0)) in segments
    assert sorted(label.text for label in ann.labels if label.constraintId in {c.id for c in dims}) == ["20", "20"]


def test_angle_and_distance_dimensions(env):
    store, ctl = env
    store.set_tool("line")
    click(ctl, 0, 0)
    click(ctl, 30, 0)
    ctl.finish()
    click(ctl, 0, 0)
    click(ctl, 20, 20)
    ctl.finish()
    sketch = store.active_sketch()
    l1, l2 = lines(sketch)

    store.set_tool("dimension")
    click(ctl, 15, 0.2)
    click(ctl, 10, 10.2)
    sketch = store.active_sketch()
    ang = next(c for c in sketch.constraints if c.type == "angle")
    assert float(ang.value) == pytest.approx(45)
    assert store.set_constraint_value(ang.id, "60")
    sketch = store.active_sketch()

    def direction(line):
        a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
        return math.atan2(b[1] - a[1], b[0] - a[0])
    assert math.degrees(direction(l2) - direction(l1)) == pytest.approx(60, abs=1e-6)

    click(ctl, *point_pos(sketch, l1.p2))
    click(ctl, *point_pos(sketch, l2.p2))
    sketch = store.active_sketch()
    dist = next(c for c in sketch.constraints if c.type == "distance")
    expected = math.dist(point_pos(sketch, l1.p2), point_pos(sketch, l2.p2))
    assert float(dist.value) == pytest.approx(expected, abs=1e-3)
    assert store.solve_info.status == "ok"


def test_line_to_arc_and_arc_to_line_tools(env):
    """ボタンを押してから曲線をクリックして変換するツール（2026-09-29 ユーザー要望）. 確定後の平面化・軸の自動拘束はしない
    （選択してからボタンを押したときと同じ結果）."""
    store, ctl = env
    store.set_tool("line")
    click(ctl, 10, 6)
    click(ctl, 10, -6)
    ctl.finish()
    click(ctl, 30, 6)
    click(ctl, 50, 6)
    ctl.finish()
    first, second = lines(store.active_sketch())
    store.set_tool("lineToArc")
    assert store.hint == "hint.pickLineToArc"
    ctl.pointer_move((20, 20))
    assert store.hover_id is None
    ctl.pointer_move((10, 2))
    assert store.hover_id == first.id
    click(ctl, 10, 2)
    sketch = store.active_sketch()
    arc = next(e for e in sketch.entities if isinstance(e, SketchArc))
    assert store.selection == [arc.id] and store.active_tool == "lineToArc"
    assert point_pos(sketch, arc.center) == pytest.approx((16.0, 0.0))   # 中心は軸の上だが自動拘束は付けない
    assert all(c.point != arc.center for c in sketch.constraints)
    click(ctl, 40, 6)                                     # 続けてもう 1 本
    sketch = store.active_sketch()
    assert count_entities(sketch)["arc"] == 2 and count_entities(sketch)["line"] == 0
    assert all(e.id != second.id for e in sketch.entities)

    store.set_tool("arcToLine")
    assert store.hint == "hint.pickArcToLine"
    click(ctl, 16 - 6 * math.sqrt(2), 0)                  # 1 本目の円弧（中心 (16,0)、半径 6√2、135°→225°）の最も左
    sketch = store.active_sketch()
    assert count_entities(sketch)["arc"] == 1 and count_entities(sketch)["line"] == 1
    assert store.solve_info.status == "ok"
    store.undo()
    assert count_entities(store.active_sketch())["arc"] == 2        # 1 回の変換 = Undo 1 回
