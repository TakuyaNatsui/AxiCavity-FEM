"""スケッチのコア（移植元: EM-CAD-py tests/test_core_sketch.py、元は TS 版 tests/core/*.test.ts）.

3D 固有の項目（平面フレーム・面/エッジの参照）は除き、スケッチモデル・スナップ・閉領域検出と
ver3 のドキュメントでの往復を確かめる。
"""

import math

import pytest

from axicavity_fem.gui.core.document import (
    PlaneRef,
    SketchLine,
    SketchPoint,
    create_empty_document,
)
from axicavity_fem.gui.core.serialize import parse_document, serialize_document
from axicavity_fem.gui.core.sketch.geometry import distance
from axicavity_fem.gui.core.sketch.model import (
    add_arc,
    add_circle,
    add_line,
    add_point,
    add_polyline,
    add_rectangle,
    add_regular_polygon,
    arc_params,
    count_entities,
    create_sketch_feature,
    delete_entities,
    get_entity,
    get_point,
    hit_test,
    merge_points,
    point_pos,
    translate_entities,
)
from axicavity_fem.gui.core.sketch.profiles import detect_profiles, profile_at
from axicavity_fem.gui.core.sketch.snap import choose_grid_spacing, snap_point


def base():
    return create_sketch_feature("Sketch1", PlaneRef.origin("XY"))


# --- sketch model ---

def test_rectangle_has_4_shared_points_and_4_lines():
    s = base()
    line_ids, point_ids = add_rectangle(s, (0, 0), (10, 5))
    assert len(line_ids) == 4 and len(point_ids) == 4
    assert count_entities(s) == {"point": 4, "line": 4, "circle": 0, "arc": 0}


def test_reuses_existing_points_within_tolerance():
    s = base()
    add_line(s, (0, 0), (10, 0))
    add_line(s, (10, 1e-7), (10, 5), tolerance=1e-6)
    assert count_entities(s)["point"] == 3


def test_delete_line_keeps_shared_points_and_drops_orphans():
    s = base()
    line_ids, _ = add_rectangle(s, (0, 0), (10, 5))
    assert delete_entities(s, [line_ids[0]]) is True
    assert count_entities(s) == {"point": 4, "line": 3, "circle": 0, "arc": 0}
    delete_entities(s, [line_ids[1]])
    assert count_entities(s) == {"point": 3, "line": 2, "circle": 0, "arc": 0}
    assert delete_entities(s, []) is False


def test_deleting_a_point_cascades_to_curves():
    s = base()
    _, point_ids = add_rectangle(s, (0, 0), (10, 5))
    delete_entities(s, [point_ids[0]])
    assert count_entities(s) == {"point": 3, "line": 2, "circle": 0, "arc": 0}


def test_translate_keeps_arcs_consistent():
    s = base()
    arc_id = add_arc(s, (0, 0), (10, 0), (0, 10))
    arc = get_entity(s, arc_id)
    assert translate_entities(s, [arc_id], (5, 5)) is True
    assert point_pos(s, arc.center) == (5, 5)
    p = arc_params(s, arc)
    assert p.radius == pytest.approx(10, abs=1e-9)
    assert p.sweep == pytest.approx(math.pi / 2, abs=1e-9)


def test_hit_test_points_before_curves():
    s = base()
    add_rectangle(s, (0, 0), (10, 5))
    add_circle(s, (20, 20), 3)
    assert hit_test(s, (0.1, 0.1), 0.5).kind == "point"
    assert hit_test(s, (5, 0.1), 0.5).kind == "line"
    assert hit_test(s, (23.2, 20), 0.5).kind == "circle"
    assert hit_test(s, (50, 50), 0.5) is None


def test_point_expressions_survive_merge_and_round_trip():
    """ver3: 座標式付きの点は統合・JSON 往復で式を保つ."""
    s = base()
    pid = add_point(s, (10, 0), x_expr="L", y_expr=None)
    other = add_point(s, (10, 0))
    add_line(s, pid, (0, 0))
    add_line(s, other, (10, 5))
    assert merge_points(s, pid, other) is True
    assert get_point(s, pid).xExpr == "L" and get_point(s, pid).yExpr is None
    doc = create_empty_document("s")
    doc.sketch = s
    restored = parse_document(serialize_document(doc))
    assert restored == doc
    assert get_point(restored.sketch, pid).xExpr == "L"


def test_sketch_round_trips_through_document_serializer():
    s = base()
    add_rectangle(s, (0, 0), (10, 5))
    add_circle(s, (5, 2.5), 1)
    doc = create_empty_document("s")
    doc.sketch = s
    restored = parse_document(serialize_document(doc))
    assert restored == doc


# --- snap ---

def sketch_with_line():
    s = create_sketch_feature("s", PlaneRef.origin("XY"))
    _, p1, p2 = add_line(s, (0, 0), (10, 0))
    return s, p1, p2


def test_snap_prefers_existing_points():
    s, _p1, p2 = sketch_with_line()
    r = snap_point(s, (10.2, 0.1), tolerance=0.5, grid_spacing=1)
    assert r.kind == "point" and r.pointId == p2 and r.pos == (10, 0)


def test_snap_to_line_midpoints():
    s, _, _ = sketch_with_line()
    r = snap_point(s, (5.2, 0.2), tolerance=0.5, grid_spacing=1)
    assert r.kind == "midpoint" and r.pos == (5, 0)


def test_snap_align_and_grid():
    s, _, _ = sketch_with_line()
    aligned = snap_point(s, (3.3, 7.1), tolerance=0.5, grid_spacing=1, reference=(3.2, 0))
    assert aligned.kind == "align" and aligned.alignX == 3.2 and aligned.pos == (3.2, 7)
    grid = snap_point(s, (3.3, 7.1), tolerance=0.5, grid_spacing=1)
    assert grid.kind == "grid" and grid.pos == (3, 7)
    free = snap_point(s, (3.3, 7.1), tolerance=0.5, grid_spacing=None)
    assert free.kind == "none" and free.pos == (3.3, 7.1)


def test_snap_excludes_dragged_points():
    s, p1, _ = sketch_with_line()
    r = snap_point(s, (0.1, 0.1), tolerance=0.5, grid_spacing=None, exclude={p1})
    assert r.kind == "none"


def test_grid_spacing_1_2_5():
    assert choose_grid_spacing(0.1) == 2
    assert choose_grid_spacing(0.5) == 10
    assert choose_grid_spacing(0.004) == 0.1


# --- profiles ---

def test_profiles_single_rectangle():
    s = base()
    add_rectangle(s, (0, 0), (10, 5))
    profiles = detect_profiles(s)
    assert len(profiles) == 1
    p = profiles[0]
    assert p.area == pytest.approx(50, abs=1e-9)
    assert len(p.outer) == 4 and p.holes == []
    assert 0 < p.anchor[0] < 10


def test_profiles_circle_inside_rectangle():
    s = base()
    add_rectangle(s, (0, 0), (10, 5))
    add_circle(s, (5, 2.5), 1)
    profiles = detect_profiles(s)
    assert len(profiles) == 2
    annulus, disc = profiles
    assert len(annulus.holes) == 1
    assert annulus.area == pytest.approx(50 - math.pi, abs=0.05)
    assert disc.holes == [] and disc.area == pytest.approx(math.pi, abs=0.05)
    assert distance(annulus.anchor, (5, 2.5)) > 1


def test_profiles_adjacent_rectangles_share_edge():
    s = base()
    add_rectangle(s, (0, 0), (10, 5))
    add_rectangle(s, (10, 0), (20, 5), tolerance=1e-6)
    profiles = detect_profiles(s)
    assert len(profiles) == 2
    for p in profiles:
        assert p.area == pytest.approx(50, abs=1e-9)
    assert sum(isinstance(e, SketchLine) for e in s.entities) == 7
    assert sum(isinstance(e, SketchPoint) for e in s.entities) == 6


def test_profiles_ignore_open_polylines_and_dangling_lines():
    s = base()
    add_polyline(s, [(0, 0), (10, 0), (10, 5)], False)
    assert detect_profiles(s) == []
    s = base()
    add_rectangle(s, (0, 0), (10, 5))
    add_line(s, (2, 2), (4, 3))
    profiles = detect_profiles(s)
    assert len(profiles) == 1 and profiles[0].area == pytest.approx(50, abs=1e-9)


def test_profiles_half_disc():
    s = base()
    add_arc(s, (0, 0), (10, 0), (-10, 0))
    add_line(s, (-10, 0), (10, 0), tolerance=1e-6)
    profiles = detect_profiles(s, arc_step=0.5 * math.pi / 180)
    assert len(profiles) == 1
    assert profiles[0].area == pytest.approx(math.pi * 100 / 2, abs=0.5)
    assert len(profiles[0].outer) == 2


def test_profiles_concentric_circles():
    s = base()
    add_circle(s, (0, 0), 5)
    add_circle(s, (0, 0), 2, tolerance=1e-6)
    profiles = detect_profiles(s, arc_step=0.5 * math.pi / 180)
    assert len(profiles) == 2
    assert len(profiles[0].holes) == 1
    assert profiles[0].area == pytest.approx(math.pi * 21, abs=0.5)
    assert profiles[1].area == pytest.approx(math.pi * 4, abs=0.5)


def test_profiles_merge_nearly_coincident_endpoints():
    s = base()
    add_line(s, (0, 0), (10, 0))
    add_line(s, (10, 1e-8), (10, 5))
    add_line(s, (10, 5), (0, 5))
    add_line(s, (0, 5 - 1e-8), (0, 0))
    assert len(detect_profiles(s, tolerance=1e-6)) == 1


def test_profiles_regular_polygon_and_profile_at():
    s = base()
    add_regular_polygon(s, (0, 0), (10, 0), 6)
    profiles = detect_profiles(s)
    assert len(profiles) == 1
    assert profiles[0].area == pytest.approx(3 * math.sqrt(3) * 100 / 2, abs=1e-6)
    assert profile_at(profiles, (0, 0)) is profiles[0]
    assert profile_at(profiles, (50, 50)) is None
