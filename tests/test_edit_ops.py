"""edit_ops: ver2.3 の Multi-Region Editor にあった編集（点挿入・点削除再接続・線⇄円弧・端点結合）."""

import math

import pytest

from axicavity_fem.gui.core.document import SketchArc, SketchConstraint, SketchLine, SketchPoint, new_id
from axicavity_fem.gui.core.sketch.edit_ops import (
    arc_center_from_chord,
    arc_to_line,
    close_polyline,
    delete_points_reconnect,
    insert_point_on_curve,
    flip_arc,
    line_to_arc,
    open_endpoints,
    set_arc_center,
)
from axicavity_fem.gui.core.sketch.model import (
    add_arc,
    add_circle,
    add_line,
    add_polyline,
    add_rectangle,
    arc_params,
    count_entities,
    create_sketch_feature,
    get_entity,
    get_point,
    point_pos,
)
from axicavity_fem.gui.core.sketch.profiles import detect_profiles


def rect():
    s = create_sketch_feature("s")
    line_ids, point_ids = add_rectangle(s, (0, 0), (10, 5))
    return s, line_ids, point_ids


# --- 点の挿入 ---

def test_insert_point_on_line_splits_it_and_keeps_the_region():
    s, line_ids, _ = rect()
    bottom = line_ids[0]
    s.constraints.append(SketchConstraint(id=new_id(), type="horizontal", line=bottom))
    pid, id_map = insert_point_on_curve(s, bottom, (4, -0.3))
    assert pid is not None and point_pos(s, pid) == (4, 0)
    assert count_entities(s) == {"point": 5, "line": 5, "circle": 0, "arc": 0}
    new_ids = id_map[bottom]
    assert len(new_ids) == 2 and get_entity(s, bottom) is None
    assert {c.line for c in s.constraints if c.type == "horizontal"} == set(new_ids)   # 水平拘束を複製
    profiles = detect_profiles(s)
    assert len(profiles) == 1 and profiles[0].area == pytest.approx(50)
    # エンティティの順序: 元の線の位置に新しい 2 本
    ids = [e.id for e in s.entities if isinstance(e, SketchLine)]
    assert ids[:2] == new_ids


def test_insert_point_at_endpoint_or_on_circle_does_nothing():
    s, line_ids, _ = rect()
    assert insert_point_on_curve(s, line_ids[0], (0, 0)) == (None, {})
    circle_id, _ = add_circle(s, (20, 20), 3)
    assert insert_point_on_curve(s, circle_id, (23, 20)) == (None, {})
    assert count_entities(s)["line"] == 4


def test_insert_point_on_arc_splits_it():
    s = create_sketch_feature("s")
    arc_id = add_arc(s, (0, 0), (10, 0), (-10, 0))         # 上半円（CCW: 0° → 180°）
    add_line(s, (-10, 0), (10, 0), tolerance=1e-6)
    pid, id_map = insert_point_on_curve(s, arc_id, (0, 12))
    assert pid is not None
    assert point_pos(s, pid) == pytest.approx((0, 10), abs=1e-9)    # クリック角度（90°）の円弧上に置く
    arcs = [e for e in s.entities if isinstance(e, SketchArc)]
    assert len(arcs) == 2 and [a.id for a in arcs] == id_map[arc_id]
    for a in arcs:
        assert arc_params(s, a).sweep == pytest.approx(math.pi / 2)
    profiles = detect_profiles(s, arc_step=0.5 * math.pi / 180)
    assert len(profiles) == 1 and profiles[0].area == pytest.approx(math.pi * 50, abs=0.5)
    # 弧の外（角度範囲外）は何もしない
    assert insert_point_on_curve(s, arcs[0].id, (0, -5)) == (None, {})


# --- 点の削除（再接続） ---

def test_delete_inserted_point_reconnects_neighbours():
    s, line_ids, _ = rect()
    pid, id_map = insert_point_on_curve(s, line_ids[0], (5, 0))
    first, second = id_map[line_ids[0]]
    changed, id_map2 = delete_points_reconnect(s, [pid])
    assert changed is True
    assert count_entities(s) == {"point": 4, "line": 4, "circle": 0, "arc": 0}
    assert id_map2[first] == id_map2[second] and len(id_map2[first]) == 1
    new_line = get_entity(s, id_map2[first][0])
    assert {point_pos(s, new_line.p1), point_pos(s, new_line.p2)} == {(0, 0), (10, 0)}
    assert len(detect_profiles(s)) == 1


def test_delete_corner_keeps_loop_as_triangle():
    s, _, point_ids = rect()
    changed, id_map = delete_points_reconnect(s, [point_ids[0]])
    assert changed and len(id_map) == 2
    assert count_entities(s) == {"point": 3, "line": 3, "circle": 0, "arc": 0}
    assert detect_profiles(s)[0].area == pytest.approx(25)


def test_delete_point_with_three_users_is_a_plain_delete():
    s, _, point_ids = rect()
    add_line(s, point_ids[0], point_ids[2])                  # 対角線 → 角の点は 3 本に使われる
    changed, id_map = delete_points_reconnect(s, [point_ids[0]])
    assert changed and id_map == {}
    assert count_entities(s) == {"point": 3, "line": 2, "circle": 0, "arc": 0}


def test_delete_arc_endpoint_reconnects_and_drops_orphan_center():
    s = create_sketch_feature("s")
    arc_id = add_arc(s, (0, 0), (10, 0), (0, 10))
    arc = get_entity(s, arc_id)
    line_a, _, _ = add_line(s, arc.end, (-10, 0))
    line_b, _, _ = add_line(s, (-10, 0), arc.start, tolerance=1e-6)
    changed, id_map = delete_points_reconnect(s, [arc.end])
    # 結び直す先の 2 点は既に line_b で結ばれているので、新しい線は作らず line_b に集約される
    assert changed and id_map == {arc_id: [line_b], line_a: [line_b]}
    assert count_entities(s) == {"point": 2, "line": 1, "circle": 0, "arc": 0}   # 中心点も消える


# --- 線 ⇄ 円弧 ---

def test_arc_center_from_chord_matches_ver23_convention():
    center = arc_center_from_chord((0, 0), (10, 0))
    assert center == pytest.approx((5, 5))
    assert arc_center_from_chord((0, 0), (0, 0)) is None


def test_line_to_arc_and_back():
    s, line_ids, point_ids = rect()
    right = line_ids[1]                                      # (10,0) → (10,5)
    arc_id, id_map = line_to_arc(s, right)
    assert id_map == {right: [arc_id]}
    arc = get_entity(s, arc_id)
    assert isinstance(arc, SketchArc)
    assert point_pos(s, arc.center) == pytest.approx((7.5, 2.5))
    p = arc_params(s, arc)
    assert p.sweep == pytest.approx(math.pi / 2) and p.radius == pytest.approx(math.hypot(2.5, 2.5))
    assert count_entities(s) == {"point": 5, "line": 3, "circle": 0, "arc": 1}
    profiles = detect_profiles(s)
    assert len(profiles) == 1 and profiles[0].area > 50      # 外側に膨らむ
    line_id, id_map2 = arc_to_line(s, arc_id)
    assert id_map2 == {arc_id: [line_id]}
    assert count_entities(s) == {"point": 4, "line": 4, "circle": 0, "arc": 0}   # 中心点は消える
    assert detect_profiles(s)[0].area == pytest.approx(50)
    assert line_to_arc(s, "nope") == (None, {}) and arc_to_line(s, line_id) == (None, {})


def test_flip_arc_bulges_to_the_other_side_and_keeps_id():
    s, line_ids, point_ids = rect()
    arc_id, _ = line_to_arc(s, line_ids[1])                 # (10,0) → (10,5)、外側（右）に膨らむ
    arc = get_entity(s, arc_id)
    start, end = arc.start, arc.end
    assert point_pos(s, arc.center) == pytest.approx((7.5, 2.5))
    assert detect_profiles(s)[0].area > 50
    assert flip_arc(s, arc_id)
    arc = get_entity(s, arc_id)                             # ID は同じ
    assert (arc.start, arc.end) == (end, start)
    assert point_pos(s, arc.center) == pytest.approx((12.5, 2.5))   # 弦 x=10 の反対側
    p = arc_params(s, arc)
    assert p.sweep == pytest.approx(math.pi / 2) and p.radius == pytest.approx(math.hypot(2.5, 2.5))
    assert detect_profiles(s)[0].area < 50                  # 内側にへこむ
    assert flip_arc(s, arc_id) and point_pos(s, arc.center) == pytest.approx((7.5, 2.5))
    assert not flip_arc(s, line_ids[0]) and not flip_arc(s, "nope")
    # 中心を共有していれば専用の中心点を新設する
    from axicavity_fem.gui.core.sketch.model import add_circle
    center = arc.center
    add_circle(s, point_pos(s, center), 1.0)
    circle = [e for e in s.entities if e.type == "circle"][-1]
    circle.center = center
    assert flip_arc(s, arc_id)
    assert get_entity(s, arc_id).center != center and get_entity(s, center) is not None


def test_set_arc_center_moves_center_or_splits_shared_center():
    s = create_sketch_feature("s")
    arc_id = add_arc(s, (0, 0), (10, 0), (0, 10))
    assert set_arc_center(s, arc_id, (2, 2), x_expr="a", y_expr=None) is True
    arc = get_entity(s, arc_id)
    cp = get_point(s, arc.center)
    assert (cp.x, cp.y, cp.xExpr, cp.yExpr) == (2, 2, "a", None)
    assert arc.radius == pytest.approx(math.hypot(8, 2))
    # 中心を共有する 2 本目の円弧 → 専用の中心点を新設し、1 本目は動かない
    arc2_id = add_arc(s, arc.center, (2, 12), (-8, 2))
    assert set_arc_center(s, arc2_id, (0, 0)) is True
    arc2 = get_entity(s, arc2_id)
    assert arc2.center != arc.center and point_pos(s, arc2.center) == (0, 0)
    assert point_pos(s, arc.center) == (2, 2)
    assert set_arc_center(s, "nope", (0, 0)) is False


# --- 端点を結ぶ ---

def test_close_polyline_joins_the_two_open_ends():
    s = create_sketch_feature("s")
    add_polyline(s, [(0, 0), (10, 0), (10, 5)], False)
    assert len(open_endpoints(s)) == 2
    line_id, _ = close_polyline(s)
    assert line_id is not None
    assert open_endpoints(s) == [] and len(detect_profiles(s)) == 1
    assert close_polyline(s) == (None, {})                   # 閉じていれば何もしない
    add_line(s, (20, 20), (30, 30))                           # 孤立した 1 本の線 → 結ぶものが無い
    assert close_polyline(s) == (None, {})
    add_line(s, (40, 40), (50, 50))                           # 開いた端点が 4 つ → 何もしない
    assert close_polyline(s) == (None, {})


def test_close_polyline_ignores_construction_and_points():
    s = create_sketch_feature("s")
    add_polyline(s, [(0, 0), (10, 0), (10, 5)], False)
    add_line(s, (50, 50), (60, 60), construction=True)
    s.entities.append(SketchPoint(id=new_id(), x=99, y=99))
    assert len(open_endpoints(s)) == 2
    assert close_polyline(s)[0] is not None
