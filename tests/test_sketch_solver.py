"""拘束ソルバ（移植元: EM-CAD-py tests/test_sketch_solver.py、元は TS 版 tests/sketch/solver.test.ts）.

3D（押し出し）と投影の項目は除き、ver3 の式付き点（座標の式 = 固定パラメータ）の項目を足す。
"""

import math

import pytest

pytest.importorskip("planegcs")

from axicavity_fem.gui.core.document import (       # noqa: E402
    SketchConstraint,
    SketchLine,
    SketchPoint,
    create_empty_document,
    new_id,
)
from axicavity_fem.gui.core.serialize import parse_document, serialize_document  # noqa: E402
from axicavity_fem.gui.core.sketch.model import (   # noqa: E402
    add_arc,
    add_circle,
    add_line,
    add_point,
    add_rectangle,
    create_sketch_feature,
    delete_entities,
    get_entity,
    merge_points,
    point_pos,
)
from axicavity_fem.gui.sketch.solver import SketchSolver, measure_dimension   # noqa: E402

solver = SketchSolver()


def base():
    return create_sketch_feature("Sketch1")


def constrain(sketch, *specs):
    for spec in specs:
        sketch.constraints.append(SketchConstraint(id=new_id(), **spec))
    return sketch


def lines(sketch):
    return [e for e in sketch.entities if isinstance(e, SketchLine)]


def test_resizes_rectangle_by_distance_constraints():
    s = base()
    (l0, l1, l2, l3), (p0, p1, _p2, p3) = add_rectangle(s, (0, 0), (10, 5))
    constrain(s, dict(type="fix", point=p0), dict(type="horizontal", line=l0),
              dict(type="vertical", line=l1), dict(type="horizontal", line=l2),
              dict(type="vertical", line=l3),
              dict(type="distance", p1=p0, p2=p1, value="20"),
              dict(type="distance", p1=p0, p2=p3, value="8"))
    out = solver.solve(s, {})
    assert out.ok and out.status == "ok" and out.dof == 0
    assert point_pos(s, p1) == pytest.approx((20, 0))
    assert point_pos(s, p3)[1] == pytest.approx(8, abs=1e-9)
    assert point_pos(s, p0) == (0, 0)
    assert out.changed is True


def test_point_line_distance():
    s = base()
    line, a, b = add_line(s, (0, 0), (10, 0))
    point = add_circle(s, (5, 3), 1.0)[1]          # 円の中心を点として使う
    constrain(s, dict(type="fix", point=a), dict(type="fix", point=b),
              dict(type="pointLineDistance", point=point, line=line, value="7"))
    c = s.constraints[-1]
    assert measure_dimension(s, c) == pytest.approx(3)
    out = solver.solve(s, {})
    assert out.ok and out.status == "ok"
    assert abs(point_pos(s, point)[1]) == pytest.approx(7, abs=1e-9)
    assert measure_dimension(s, c) == pytest.approx(7)
    doc = create_empty_document("t")
    doc.sketch = s
    loaded = parse_document(serialize_document(doc)).sketch.constraints[-1]
    assert (loaded.type, loaded.point, loaded.line, loaded.value) == ("pointLineDistance", point, line, "7")
    delete_entities(s, [line])
    assert not any(k.type == "pointLineDistance" for k in s.constraints)


def test_drives_dimensions_with_document_parameters():
    s = base()
    _, p0, p1 = add_line(s, (0, 0), (10, 0))
    constrain(s, dict(type="fix", point=p0), dict(type="horizontal", line=lines(s)[0].id),
              dict(type="distance", p1=p0, p2=p1, value="2 * L + 1"))
    out = solver.solve(s, {"L": 7})
    assert out.ok
    assert point_pos(s, p1)[0] == pytest.approx(15, abs=1e-9)
    bad = solver.solve(s, {})
    assert not bad.ok and bad.status == "invalid" and "L" in bad.error


def test_reports_conflicting_constraints():
    s = base()
    _, p0, p1 = add_line(s, (0, 0), (10, 0))
    constrain(s, dict(type="fix", point=p0),
              dict(type="distance", p1=p0, p2=p1, value="10"),
              dict(type="distance", p1=p0, p2=p1, value="12"))
    before = point_pos(s, p1)
    out = solver.solve(s, {})
    assert not out.ok and out.status == "conflict" and out.conflicting
    assert point_pos(s, p1) == before          # 失敗時はスケッチを変えない


def test_radius_angle_parallel_perpendicular_equal():
    s = base()
    circle_id, _ = add_circle(s, (0, 0), 3)
    add_line(s, (0, 10), (10, 10))
    add_line(s, (0, 20), (10, 21))
    add_line(s, (20, 0), (21, 10))
    a, b, c = (l.id for l in lines(s))
    constrain(s, dict(type="radius", curve=circle_id, value="5"),
              dict(type="horizontal", line=a), dict(type="parallel", l1=a, l2=b),
              dict(type="perpendicular", l1=a, l2=c), dict(type="equal", e1=a, e2=c))
    out = solver.solve(s, {})
    assert out.ok, out
    assert get_entity(s, circle_id).radius == pytest.approx(5, abs=1e-9)
    bl = get_entity(s, b)
    assert point_pos(s, bl.p1)[1] == pytest.approx(point_pos(s, bl.p2)[1], abs=1e-9)
    cl = get_entity(s, c)
    assert point_pos(s, cl.p1)[0] == pytest.approx(point_pos(s, cl.p2)[0], abs=1e-9)

    def length(line_id):
        l = get_entity(s, line_id)
        return math.dist(point_pos(s, l.p1), point_pos(s, l.p2))

    assert length(a) == pytest.approx(length(c), abs=1e-9)

    s2 = base()
    add_line(s2, (0, 10), (10, 10))
    add_line(s2, (20, 0), (21, 10))
    a2, c2 = (l.id for l in lines(s2))
    constrain(s2, dict(type="fix", point=lines(s2)[0].p1),
              dict(type="angle", l1=a2, l2=c2, value="45"))
    out2 = solver.solve(s2, {})
    assert out2.ok, out2
    now = measure_dimension(s2, SketchConstraint(id="m", type="angle", l1=a2, l2=c2, value="0"))
    assert now == pytest.approx(45, abs=1e-6)


def test_keeps_line_tangent_to_arc_while_dragging():
    s = base()
    arc_id = add_arc(s, (0, 0), (10, 0), (0, 10))
    arc = get_entity(s, arc_id)
    line_id, _, p2 = add_line(s, arc.end, (-10, 10))     # 円弧の終点から水平に伸びる線 → 接線
    constrain(s, dict(type="fix", point=arc.center),
              dict(type="tangent", e1=line_id, e2=arc_id),
              dict(type="radius", curve=arc_id, value="10"))
    first = solver.solve(s, {})
    assert first.ok, first
    dragged = solver.solve(s, {}, fixed_points={p2: (-10, 5)})
    assert dragged.ok, dragged
    end = point_pos(s, arc.end)
    q = point_pos(s, p2)
    direction = (q[0] - end[0], q[1] - end[1])
    radial = end
    assert math.hypot(*radial) == pytest.approx(10, abs=1e-3)
    cos_angle = (direction[0] * radial[0] + direction[1] * radial[1]) / (
        math.hypot(*direction) * math.hypot(*radial))
    assert abs(cos_angle) < 1e-3        # 冗長拘束を含む系は planegcs が緩い収束で解く


def test_expression_points_are_fixed_per_coordinate():
    """ver3: 式のある座標だけ固定される（z だけ式なら r は自由）."""
    s = base()
    p_fixed = add_point(s, (10, 5), x_expr="L")            # z = L は固定、r は自由（r=0 だと距離の微分が 0 で特異）
    p_free = add_point(s, (0, 0))
    line_id, _, _ = add_line(s, p_fixed, p_free)
    constrain(s, dict(type="fix", point=p_free), dict(type="distance", p1=p_fixed, p2=p_free, value="26"))
    out = solver.solve(s, {"L": 10})
    assert out.ok, out
    x, y = point_pos(s, p_fixed)
    assert x == pytest.approx(10) and abs(y) == pytest.approx(24, abs=1e-6)     # 10² + 24² = 26²
    assert out.dof == 0
    # 両方の座標に式があれば固定点: 2 つの固定点から等距離の自由点だけが動く
    both = base()
    a = add_point(both, (0, 0), x_expr="0", y_expr="0")
    b = add_point(both, (10, 0), x_expr="L", y_expr="0")
    c = add_point(both, (5, 5))
    add_line(both, a, c)
    add_line(both, b, c)
    constrain(both, dict(type="distance", p1=a, p2=c, value="13"),
              dict(type="distance", p1=b, p2=c, value="13"))
    out = solver.solve(both, {"L": 10})
    assert out.ok and out.dof == 0
    assert point_pos(both, a) == (0, 0) and point_pos(both, b) == (10, 0)
    assert point_pos(both, c) == pytest.approx((5, 12), abs=1e-6)                   # 5² + 12² = 13²


def test_constraint_bookkeeping_in_model():
    s = base()
    line_ids, _ = add_rectangle(s, (0, 0), (10, 5))
    constrain(s, dict(type="horizontal", line=line_ids[0]), dict(type="vertical", line=line_ids[1]))
    delete_entities(s, [line_ids[0]])
    assert len(s.constraints) == 1 and s.constraints[0].type == "vertical"

    s2 = base()
    a_id, _, a_p2 = add_line(s2, (0, 0), (10, 0))
    b_id, b_p1, b_p2 = add_line(s2, (10, 0.5), (10, 5))
    assert merge_points(s2, a_p2, b_p1) is True
    assert sum(isinstance(e, SketchPoint) for e in s2.entities) == 3
    bl = get_entity(s2, b_id)
    assert bl.p1 == a_p2 and bl.p2 == b_p2
