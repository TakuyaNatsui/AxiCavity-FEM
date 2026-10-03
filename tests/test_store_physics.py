"""ver3 の物理（領域の材料・境界条件）と編集操作の store 側: キーの再対応付け・ID の付け替え・Undo."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore       # noqa: E402
from axicavity_fem.gui.core.convert import to_multi_region  # noqa: E402
from axicavity_fem.gui.core.document import SketchArc, SketchLine  # noqa: E402
from axicavity_fem.gui.core.sketch.model import count_entities, point_pos  # noqa: E402
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


def two_boxes(store, ctl):
    store.set_tool("rectangle")
    click(ctl, 0, 0)
    click(ctl, 50, 50)
    click(ctl, 50, 0)
    click(ctl, 100, 50)
    assert len(store.profiles) == 2
    left = next(p for p in store.profiles if p.anchor[0] < 50)
    right = next(p for p in store.profiles if p.anchor[0] > 50)
    return left, right


def test_region_settings_follow_edits(env):
    store, ctl = env
    left, right = two_boxes(store, ctl)
    assert store.set_region(right.id, name="Ceramic", materialTag="ceramic", epsR="9.64", tanDelta="1e-4")
    setting = store.region_setting(right.id)
    assert (setting.name, setting.epsR, setting.epsRExpr, setting.tanDelta) == ("Ceramic", 9.64, None, 1e-4)
    # 式で入れると式が残る
    store.add_param()
    store.update_param(store.document.params[0].id, name="er", expression="4")
    assert store.set_region(right.id, epsR="er")
    setting = store.region_setting(right.id)
    assert (setting.epsR, setting.epsRExpr) == (4.0, "er")
    assert store.set_region(right.id, epsR="er +") is False           # 評価できない
    # 右の領域の辺に点を挿入 → 閉領域の ID が変わっても設定が追従する
    right_edge = next(e for e in store.active_sketch().entities if isinstance(e, SketchLine)
                      and point_pos(store.active_sketch(), e.p1)[0] == 100
                      and point_pos(store.active_sketch(), e.p2)[0] == 100)
    pid = store.insert_point_on_curve(right_edge.id, (100, 25))
    assert pid is not None and len(store.profiles) == 2
    right2 = next(p for p in store.profiles if p.anchor[0] > 50)
    assert right2.id != right.id
    assert store.region_setting(right2.id) is setting and setting.key == right2.id
    resolved = {r.profile.id: r for r in store.resolve_regions()}
    assert resolved[right2.id].name == "Ceramic" and resolved[right2.id].eps_r == 4.0
    assert resolved[next(p.id for p in store.profiles if p.anchor[0] < 50)].name == "Vacuum"
    # 点をドラッグして領域を動かしても追従（アンカーは領域の中）
    store.set_tool("select")
    ctl.pointer_move((100, 50))
    ctl.pointer_down((100, 50))
    ctl.pointer_move((110, 60))
    ctl.pointer_up((110, 60))
    right3 = next(p for p in store.profiles if p.anchor[0] > 50)
    assert store.region_setting(right3.id) is setting
    # Undo で戻る
    while store.can_undo():
        store.undo()
    assert store.document.regions == [] and len(store.profiles) == 0


def test_boundary_assignments_follow_split_and_reconnect(env):
    store, ctl = env
    left, right = two_boxes(store, ctl)
    sketch = store.active_sketch()
    shared = next(e for e in sketch.entities if isinstance(e, SketchLine)
                  and point_pos(sketch, e.p1)[0] == 50 and point_pos(sketch, e.p2)[0] == 50)
    top_left = next(e for e in sketch.entities if isinstance(e, SketchLine)
                    and point_pos(sketch, e.p1)[1] == 50 and point_pos(sketch, e.p2)[1] == 50
                    and max(point_pos(sketch, e.p1)[0], point_pos(sketch, e.p2)[0]) <= 50)
    assert store.effective_bc(shared.id) == ("None", "interface")
    assert store.effective_bc(top_left.id) == ("PEC", "default")
    assert store.set_boundary([top_left.id], "M-short")
    assert store.effective_bc(top_left.id) == ("M-short", "explicit")
    assert store.set_boundary([top_left.id], "M-short") is False        # 変化なし
    # 分割 → 両方の新しい線に引き継ぐ
    pid = store.insert_point_on_curve(top_left.id, (25, 50))
    sketch = store.active_sketch()
    halves = [e for e in sketch.entities if isinstance(e, SketchLine) and pid in (e.p1, e.p2)]
    assert len(halves) == 2 and all(store.effective_bc(e.id) == ("M-short", "explicit") for e in halves)
    assert top_left.id not in store.document.boundaries
    # 再接続（点を消す）→ 新しい 1 本に引き継ぐ
    assert store.delete_points_reconnect([pid])
    sketch = store.active_sketch()
    merged = next(e for e in sketch.entities if isinstance(e, SketchLine)
                  and point_pos(sketch, e.p1)[1] == 50 and point_pos(sketch, e.p2)[1] == 50
                  and max(point_pos(sketch, e.p1)[0], point_pos(sketch, e.p2)[0]) <= 50)
    assert store.effective_bc(merged.id) == ("M-short", "explicit")
    # 線 → 円弧 → 線 でも引き継ぐ
    arc_id = store.convert_line_to_arc(merged.id)
    assert arc_id is not None and store.effective_bc(arc_id) == ("M-short", "explicit")
    assert isinstance(next(e for e in store.active_sketch().entities if e.id == arc_id), SketchArc)
    line_id = store.convert_arc_to_line(arc_id)
    assert store.effective_bc(line_id) == ("M-short", "explicit")
    # 自動に戻す
    assert store.set_boundary([line_id], None)
    assert store.effective_bc(line_id) == ("PEC", "default")
    geom = to_multi_region(store.document).geom
    assert geom.validate() == [] and len(geom.regions) == 2


def test_close_polyline_and_settings_signals(env):
    store, ctl = env
    store.set_tool("line")
    click(ctl, 0, 0)
    click(ctl, 40, 0)
    click(ctl, 40, 20)
    ctl.finish()
    assert len(store.profiles) == 0
    assert store.close_polyline() is not None
    assert len(store.profiles) == 1
    received = []
    store.settings_changed.connect(received.append)
    assert store.set_mesh(size=2.5, order=1) and store.document.mesh.order == 1
    assert store.set_analysis(type="hom", wave="traveling") and store.document.analysis.type == "hom"
    assert store.set_post(cond=1e7) and store.set_report(animate=True)
    assert received == ["mesh", "analysis", "post", "report"]
    assert store.set_units("cm", convert_values=True)
    assert store.document.meta.units == "cm" and store.document.mesh.size == pytest.approx(0.25)
    assert point_pos(store.active_sketch(), next(e.id for e in store.active_sketch().entities
                                                    if e.type == "point" and e.x > 3)) == pytest.approx((4, 0))
    assert count_entities(store.active_sketch())["line"] == 3
    with pytest.raises(KeyError):
        store.set_mesh(bogus=1)
