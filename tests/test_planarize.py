"""平面化（core/sketch/planarize.py）: T 字・交差・重なりでの分割、重複の削除、円弧、冪等性、境界条件の引き継ぎ、
作図時の自動分割（2026-09-25 ユーザー報告: 長方形 2 つの辺が一部重なると小さい方だけがメッシュになる）."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from axicavity_fem.gui.core.convert import check_geometry, sketch_profiles, to_multi_region  # noqa: E402
from axicavity_fem.gui.core.document import SketchArc, SketchLine, SketchPoint, create_empty_document  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_arc, add_line, add_rectangle, count_entities, point_pos  # noqa: E402
from axicavity_fem.gui.core.sketch.planarize import count_conflicts, planarize  # noqa: E402


def _lines(sketch):
    return [e for e in sketch.entities if isinstance(e, SketchLine)]


def _segments(sketch):
    return sorted(tuple(sorted((point_pos(sketch, e.p1), point_pos(sketch, e.p2)))) for e in _lines(sketch))


def test_overlapping_rectangles_become_two_regions():
    doc = create_empty_document("rect2")
    add_rectangle(doc.sketch, (0, 0), (50, 50))
    add_rectangle(doc.sketch, (25, 25), (50, 40))
    assert [len(p.holes) for p in sketch_profiles(doc)] == [1, 0]          # 分割前: 小さい方が穴
    assert count_conflicts(doc.sketch) > 0
    assert any(i.code == "overlap" for i in check_geometry(doc))
    n, id_map = planarize(doc.sketch)
    assert n == 3                                                            # T 字 2 + 重複 1
    assert count_entities(doc.sketch) == {"point": 8, "line": 9, "circle": 0, "arc": 0}
    assert ((50, 25), (50, 40)) in _segments(doc.sketch)
    profiles = sketch_profiles(doc)
    assert [len(p.holes) for p in profiles] == [0, 0] and len(profiles) == 2
    assert [round(p.area) for p in profiles] == [2125, 375]
    assert len(profiles[0].outer) == 8 and len(profiles[1].outer) == 4
    geom = to_multi_region(doc).geom
    assert len(geom.regions) == 2 and len(geom.segments) == 9
    assert planarize(doc.sketch) == (0, {})                                   # 冪等
    assert not any(i.code == "overlap" for i in check_geometry(doc))


def test_crossing_lines_and_collinear_overlap():
    doc = create_empty_document("x")
    a, _, _ = add_line(doc.sketch, (0, 0), (40, 40))
    b, _, _ = add_line(doc.sketch, (0, 40), (40, 0))
    n, id_map = planarize(doc.sketch)
    assert n == 1 and count_entities(doc.sketch)["line"] == 4 and count_entities(doc.sketch)["point"] == 5
    assert set(id_map) == {a, b} and all(len(v) == 2 for v in id_map.values())
    mid = next(e for e in doc.sketch.entities if isinstance(e, SketchPoint) and (e.x, e.y) == (20, 20))
    assert sum(1 for e in _lines(doc.sketch) if mid.id in (e.p1, e.p2)) == 4

    doc = create_empty_document("overlap")
    add_line(doc.sketch, (0, 0), (100, 0))
    add_line(doc.sketch, (50, 0), (150, 0))
    n, _ = planarize(doc.sketch)
    assert n == 3
    assert _segments(doc.sketch) == [((0, 0), (50, 0)), ((50, 0), (100, 0)), ((100, 0), (150, 0))]
    assert count_entities(doc.sketch)["point"] == 4


def test_arcs_t_junction_and_crossing():
    doc = create_empty_document("arc")
    add_arc(doc.sketch, (0, 0), (30, 0), (0, 30))                            # 第 1 象限の 90° 円弧
    tip = next(e for e in doc.sketch.entities if isinstance(e, SketchPoint) and (e.x, e.y) == (0, 0))
    q = (30 * 2 ** -0.5, 30 * 2 ** -0.5)
    add_line(doc.sketch, tip.id, q)                                          # 中心から円弧の途中へ（T 字）
    n, _ = planarize(doc.sketch)
    assert n == 1 and count_entities(doc.sketch) == {"point": 4, "line": 1, "circle": 0, "arc": 2}
    arcs = [e for e in doc.sketch.entities if isinstance(e, SketchArc)]
    assert {round(a.radius, 6) for a in arcs} == {30.0}

    doc = create_empty_document("cross")
    add_arc(doc.sketch, (0, 0), (30, 0), (0, 30))
    add_line(doc.sketch, (0, 10), (40, 10))                                  # 円弧と交差（x = sqrt(800)）
    n, _ = planarize(doc.sketch)
    assert n == 1 and count_entities(doc.sketch) == {"point": 6, "line": 2, "circle": 0, "arc": 2}
    cut = next(e for e in doc.sketch.entities if isinstance(e, SketchPoint) and abs(e.y - 10) < 1e-9 and 0 < e.x < 40)
    assert cut.x == pytest.approx(800 ** 0.5)


def test_store_auto_split_and_boundary_inheritance():
    pytest.importorskip("planegcs")
    from PySide6 import QtWidgets

    from axicavity_fem.gui.app.store import DocumentStore
    from axicavity_fem.gui.sketch.controller import SketchController

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = DocumentStore()
    ctl = SketchController(store)
    ctl.set_scale(0.1)
    store.new_document()

    def click(x, y):
        ctl.pointer_move((x, y))
        ctl.pointer_down((x, y))
        ctl.pointer_up((x, y))

    store.set_tool("rectangle")
    click(0, 0)
    click(50, 50)
    right = next(e for e in _lines(store.active_sketch())
                 if point_pos(store.active_sketch(), e.p1)[0] == 50 and point_pos(store.active_sketch(), e.p2)[0] == 50)
    store.set_boundary([right.id], "E-short")
    click(25, 25)
    click(50, 40)                                                            # 確定と同時に自動分割
    sketch = store.active_sketch()
    assert count_entities(sketch)["line"] == 9 and len(store.profiles) == 2
    assert all(not p.holes for p in store.profiles)
    pieces = [e for e in _lines(sketch) if point_pos(sketch, e.p1)[0] == 50 and point_pos(sketch, e.p2)[0] == 50]
    assert len(pieces) == 3 and all(store.effective_bc(e.id) == ("E-short", "explicit") for e in pieces)
    assert {store.effective_bc(e.id)[0] for e in _lines(sketch)} == {"E-short", "PEC", "None"}
    piece_ids = {e.id for e in pieces}                                       # 垂直の拘束は分けた線に複製される
    assert sum(1 for c in sketch.constraints if c.type == "vertical" and c.line in piece_ids) == 3
    # Undo 1 回で長方形ごと戻る（分割は同じ履歴にまとまる）
    store.undo()
    assert count_entities(store.active_sketch())["line"] == 4
    store.redo()
    assert count_entities(store.active_sketch())["line"] == 9
    # 自動分割をオフにすると穴のまま。「交差で分割」（store.planarize）で分かれる
    store.auto_planarize = False
    store.set_tool("rectangle")
    click(5, 5)
    click(20, 20)
    click(20, 10)
    click(40, 15)
    assert any(i.code == "overlap" for i in check_geometry(store.document))
    assert store.planarize() == 3 and not any(i.code == "overlap" for i in check_geometry(store.document))
    assert store.planarize() == 0
