"""寸法「円周と直線の距離」（circleLineDistance、2026-09-29 ユーザー要望）.

- 寸法ツールで「円 / 円弧 → 直線」または「直線 → 円 / 円弧」: 値は |中心と直線の距離 − 半径|（planegcs の C2L / A2L 距離）
- 円 / 円弧を先に選んで Enter / 空白クリック → 半径（線の長さと同じ流儀。以前は円のクリックですぐ半径だった）
- 半径を変えても円周と直線の隙間は保たれる。参照の z 軸 / r 軸も直線として選べる
- 寸法線は円周上で直線に最も近い点から直線への垂線。円弧の外なら円周を延ばして見せる
"""

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.core.document import REFERENCE_Z_AXIS, SketchArc, SketchCircle, SketchLine  # noqa: E402
from axicavity_fem.gui.core.serialize import document_from_dict, document_to_dict  # noqa: E402
from axicavity_fem.gui.core.sketch.model import arc_params, point_pos  # noqa: E402
from axicavity_fem.gui.sketch.annotations import build_annotations  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402
from axicavity_fem.gui.sketch.solver import measure_dimension  # noqa: E402


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


def circle(store, controller, center, radius):
    store.set_tool("circle")
    click(controller, *center, ctrl=True)
    click(controller, center[0] + radius, center[1], ctrl=True)
    return [e for e in store.active_sketch().entities if isinstance(e, SketchCircle)][-1]


def entity(store, eid):
    return next(e for e in store.active_sketch().entities if e.id == eid)


def dims(store, kind):
    return [c for c in store.active_sketch().constraints if c.type == kind]


def test_circle_then_axis_distance_keeps_the_gap_when_the_radius_changes(env):
    store, ctl = env
    c = circle(store, ctl, (30, 30), 6)
    store.set_tool("dimension")
    click(ctl, 36.1, 30)                                  # 円 → 空白クリック = 半径
    click(ctl, 80, 80)
    [r] = dims(store, "radius")
    assert r.value == "6"
    click(ctl, 36.1, 30)                                  # 円周
    assert store.hint == "hint.dimCircleNext" and len(store.active_sketch().constraints) == 1
    click(ctl, 100, 0.2)                                  # z 軸（参照）
    [d] = dims(store, "circleLineDistance")
    assert (d.curve, d.line, d.value) == (c.id, REFERENCE_Z_AXIS, "24")   # 30 − 6
    assert store.selected_constraint_id == d.id and store.hint == "hint.dimStart"

    assert store.set_constraint_value(d.id, "10")
    assert point_pos(store.active_sketch(), c.center)[1] == pytest.approx(16.0)
    assert store.solve_info.status == "ok"
    assert store.set_constraint_value(r.id, "8")
    sketch = store.active_sketch()
    assert entity(store, c.id).radius == pytest.approx(8.0)
    assert point_pos(sketch, c.center)[1] == pytest.approx(18.0)          # 隙間 10 は保たれる
    assert store.solve_info.status == "ok" and store.solve_info.dof == 1  # 残りは中心の z だけ


def test_line_then_arc_distance(env):
    store, ctl = env
    store.set_tool("line")
    click(ctl, 10, 50, ctrl=True)
    click(ctl, 70, 50, ctrl=True)
    ctl.finish()
    store.set_tool("arc")
    click(ctl, 40, 20, ctrl=True)                         # 中心
    click(ctl, 50, 20, ctrl=True)                         # 始点
    click(ctl, 40, 30, ctrl=True)                         # 終点（90°）
    arc = next(e for e in store.active_sketch().entities if isinstance(e, SketchArc))
    line = next(e for e in store.active_sketch().entities if isinstance(e, SketchLine) and not e.projection)

    store.set_tool("dimension")
    click(ctl, 40, 50.2)                                  # 線
    click(ctl, 40 + 10 * math.cos(math.pi / 4), 20 + 10 * math.sin(math.pi / 4))   # 円弧
    [d] = dims(store, "circleLineDistance")
    assert (d.curve, d.line, d.value) == (arc.id, line.id, "20")          # 50 − 20 − 10
    assert store.set_constraint_value(d.id, "15")
    sketch = store.active_sketch()
    assert measure_dimension(sketch, d) == pytest.approx(15.0, abs=1e-6)
    assert store.solve_info.status == "ok"


def test_line_crossing_the_circle_measures_the_inner_gap(env):
    store, ctl = env
    c = circle(store, ctl, (30, 4), 10)
    store.set_tool("dimension")
    click(ctl, 40.1, 4)
    click(ctl, 100, 0.2)
    [d] = dims(store, "circleLineDistance")
    assert d.value == "6"                                  # |4 − 10|
    assert store.set_constraint_value(d.id, "2")           # 半径は自由: 解けて隙間が 2 になる（半径が変わってもよい）
    assert measure_dimension(store.active_sketch(), d) == pytest.approx(2.0, abs=1e-6)
    assert store.set_constraint_value(d.id, "6")
    click(ctl, point_pos(store.active_sketch(), c.center)[0] + entity(store, c.id).radius + 0.1,
          point_pos(store.active_sketch(), c.center)[1])
    ctl.finish()                                           # 半径を固定してから
    r = entity(store, c.id).radius
    assert store.set_constraint_value(d.id, "3")
    center = point_pos(store.active_sketch(), c.center)
    assert entity(store, c.id).radius == pytest.approx(r)
    assert r - abs(center[1]) == pytest.approx(3.0, abs=1e-6)   # 直線は円を横切ったまま（内側の隙間）
    assert store.solve_info.status == "ok"


def test_circle_first_enter_gives_radius_and_escape_cancels(env):
    store, ctl = env
    c1 = circle(store, ctl, (30, 30), 6)
    c2 = circle(store, ctl, (60, 30), 4)
    store.set_tool("dimension")
    click(ctl, 36.1, 30)
    ctl.cancel()                                           # Esc: 何も付けない
    assert store.active_sketch().constraints == [] and store.active_tool == "dimension"
    click(ctl, 36.1, 30)
    ctl.finish()                                           # Enter / 右クリック: 半径
    click(ctl, 64.1, 30)
    click(ctl, 60, 60)                                     # 空白クリック: 半径
    assert sorted((c.curve, c.value) for c in dims(store, "radius")) == sorted([(c1.id, "6"), (c2.id, "4")])
    click(ctl, 36.1, 30)
    click(ctl, 64.1, 30)                                   # 円の次に円は選べない（直線だけ）
    assert len(store.active_sketch().constraints) == 2 and store.hint == "hint.dimCircleNext"


def test_annotation_from_circumference_to_line(env):
    store, ctl = env
    c = circle(store, ctl, (30, 30), 6)
    store.set_tool("arc")
    click(ctl, 60, 20, ctrl=True)
    click(ctl, 70, 20, ctrl=True)
    click(ctl, 60, 30, ctrl=True)
    arc = next(e for e in store.active_sketch().entities if isinstance(e, SketchArc))
    store.set_tool("dimension")
    click(ctl, 36.1, 30)
    click(ctl, 100, 0.2)
    click(ctl, 60 + 10 * math.cos(math.pi / 4), 20 + 10 * math.sin(math.pi / 4))
    click(ctl, 100, 0.2)
    sketch = store.active_sketch()
    ann = build_annotations(sketch, {}, None, 0.1)
    ends = {tuple(sorted(((round(pl[0][0], 6), round(pl[0][1], 6)), (round(pl[-1][0], 6), round(pl[-1][1], 6)))))
            for pl in ann.lines}
    assert ((30.0, 0.0), (30.0, 24.0)) in ends             # 円周の最下点 → z 軸
    assert ((60.0, 0.0), (60.0, 10.0)) in ends             # 円弧を含む円の最下点 → z 軸
    assert ((60.0, 10.0), (70.0, 20.0)) in ends            # 円弧（0°〜90°）の外なので始点から最下点まで円周を延ばす
    assert not any((1.0, 0.0) in pair for pair in ends)    # 参照の軸の端点から延長線を引かない
    labels = sorted(label.text for label in ann.labels if label.dimension)
    assert labels == ["10", "24"]
    p = arc_params(sketch, arc)
    assert p.sweep == pytest.approx(math.pi / 2)


def test_serialize_round_trip_and_constraint_list(env, qtbot):
    from axicavity_fem.gui.i18n.tr import set_language
    from axicavity_fem.gui.ui.sketch_panels import SketchPropertyPanel

    set_language("ja")
    store, ctl = env
    panel = SketchPropertyPanel(store)
    qtbot.addWidget(panel)
    circle(store, ctl, (30, 30), 6)
    store.set_tool("dimension")
    click(ctl, 36.1, 30)
    click(ctl, 100, 0.2)
    table = panel.constraints.table
    assert [table.item(r, 0).text() for r in range(table.rowCount())] == ["距離（円周と直線）（C1 – z 軸）"]
    assert table.item(0, 1).text() == "24"
    doc = document_from_dict(document_to_dict(store.document))
    [d] = [c for c in doc.sketch.constraints if c.type == "circleLineDistance"]
    assert (d.line, d.value) == (REFERENCE_Z_AXIS, "24")
