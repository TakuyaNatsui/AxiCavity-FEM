"""SketchView（QGraphicsView）の描画と入力の結線（offscreen、pytest-qt。移植元: EM-CAD-py tests/test_ui_sketch_view.py）.

SketchView 単体を offscreen で開き、実マウス/キーイベントとレンダリング結果（ピクセル色）を確かめる。
ver3 の追加: 軸 r=0 の一点鎖線・r<0 の網掛け・式付きの点（四角）・ダブルクリックの点挿入・物理/メッシュの overlay。
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtCore, QtGui, QtWidgets     # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore        # noqa: E402
from axicavity_fem.gui.core.document import SketchLine, SketchPoint  # noqa: E402
from axicavity_fem.gui.core.sketch.model import count_entities, point_pos  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402
from axicavity_fem.gui.ui.sketch_view import BC_COLORS, COLORS, MeshPreview, SketchView   # noqa: E402


@pytest.fixture
def view(qtbot):
    store = DocumentStore()
    controller = SketchController(store)
    w = SketchView(store, controller)
    w.resize(600, 500)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    store.new_document()
    w.fit_sketch()
    return w


def px(view, x, y) -> QtCore.QPoint:
    return view.mapFromScene(QtCore.QPointF(x, y))


def move(view, x, y, buttons=QtCore.Qt.NoButton):
    p = px(view, x, y)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(p),
                           QtCore.QPointF(view.viewport().mapToGlobal(p)),
                           QtCore.Qt.NoButton, buttons, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(view.viewport(), ev)


def click(qtbot, view, x, y, button=QtCore.Qt.LeftButton):
    move(view, x, y)
    qtbot.mouseClick(view.viewport(), button, pos=px(view, x, y))


def color_at(view, x, y) -> str:
    img = view.grab().toImage()
    p = px(view, x, y)
    return QtGui.QColor(img.pixel(p)).name()


def has_color(view, name: str, x0, y0, x1, y1, tol: int = 60) -> bool:
    """画面領域（スケッチ座標の矩形、±2 px 拡張）に指定色に近いピクセルがあるか
    （アンチエイリアスで端の色は薄まるので許容差つき）."""
    img = view.grab().toImage()
    a, b = px(view, x0, y0), px(view, x1, y1)
    t = QtGui.QColor(name)
    for yy in range(min(a.y(), b.y()) - 2, max(a.y(), b.y()) + 3):
        for xx in range(min(a.x(), b.x()) - 2, max(a.x(), b.x()) + 3):
            c = QtGui.QColor(img.pixel(xx, yy))
            if max(abs(c.red() - t.red()), abs(c.green() - t.green()), abs(c.blue() - t.blue())) <= tol:
                return True
    return False


def rectangle(qtbot, view, x0, y0, x1, y1):
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_R)
    click(qtbot, view, x0, y0)
    click(qtbot, view, x1, y1)
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_S)


def test_grid_scale_axis_and_negative_region(view):
    assert view.viewport().hasMouseTracking()
    wpp = view.world_per_pixel()
    assert 0.2 < wpp < 0.3                 # 600x500 px に z ±60 mm
    assert view.controller.grid_spacing in (5.0, 10.0)
    # 背景色、z = 0 の線、軸 r = 0 の一点鎖線、r < 0 の網掛け
    assert color_at(view, 37.3, 41.7) == COLORS["background"]
    assert has_color(view, COLORS["grid_axis"], 0, 20, 0, 20)
    assert has_color(view, COLORS["axis"], 3, 0, 23, 0, tol=70)      # 一点鎖線（アンチエイリアスで薄まる）
    assert color_at(view, 17.3, -4.2) == COLORS["negative"]


def test_draw_rectangle_with_mouse_and_keys(qtbot, view):
    store = view.store
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_R)
    assert store.active_tool == "rectangle"
    click(qtbot, view, -20, 10)
    move(view, 18, 28)                     # グリッド → (20, 30)
    assert view.controller.preview is not None
    assert has_color(view, COLORS["preview"], -20, 30, 20, 30)     # プレビュー枠の上辺
    click(qtbot, view, 20, 30)
    sketch = store.active_sketch()
    assert count_entities(sketch)["line"] == 4
    assert has_color(view, COLORS["curve"], -20, 30, 20, 30)       # 上辺
    assert has_color(view, COLORS["point"], -20, 10, -20, 10)      # 頂点マーカー
    assert len(store.profiles) == 1

    # 閉領域の塗り（選択ツールでホバーすると色が変わる）
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_S)
    move(view, 0, 20)
    assert store.hover_profile_id is not None
    assert color_at(view, 3, 23) != COLORS["background"]

    # 右クリック = 確定（Enter）、Esc で選択ツールへ
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_L)
    click(qtbot, view, -40, 5)
    click(qtbot, view, -40, 45)
    click(qtbot, view, -30, 25, button=QtCore.Qt.RightButton)    # 折れ線を確定
    assert not view.controller.tool.busy()
    assert count_entities(store.active_sketch())["line"] == 5
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Escape)
    assert store.active_tool == "select"

    # 選択 → Delete
    click(qtbot, view, -40, 25)
    assert len(store.selection) == 1
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Delete)
    assert count_entities(store.active_sketch())["line"] == 4


def test_drag_point_and_wheel_zoom(qtbot, view):
    store = view.store
    store.set_tool("line")
    click(qtbot, view, 10, 10)                     # 軸・原点から離す（画面のグリッド 5 に乗る位置）
    click(qtbot, view, 40, 10)
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Return)
    store.set_tool("select")
    line = next(e for e in store.active_sketch().entities if isinstance(e, SketchLine) and not e.projection)
    move(view, 40, 10)
    assert store.hover_id == line.p2
    qtbot.mousePress(view.viewport(), QtCore.Qt.LeftButton, pos=px(view, 40, 10))
    move(view, 43, 15, QtCore.Qt.LeftButton)
    move(view, 48, 19, QtCore.Qt.LeftButton)
    qtbot.mouseRelease(view.viewport(), QtCore.Qt.LeftButton, pos=px(view, 48, 19))
    assert point_pos(store.active_sketch(), line.p2) == pytest.approx((50, 20), abs=1e-9)  # グリッドへ

    before = view.world_per_pixel()
    p = px(view, 0, 0)
    ev = QtGui.QWheelEvent(QtCore.QPointF(p), QtCore.QPointF(view.viewport().mapToGlobal(p)),
                           QtCore.QPoint(0, 0), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
                           QtCore.Qt.NoModifier, QtCore.Qt.ScrollUpdate, False)
    QtWidgets.QApplication.sendEvent(view.viewport(), ev)
    assert view.world_per_pixel() < before                         # 拡大
    assert (px(view, 0, 0) - p).manhattanLength() <= 1             # カーソル位置を保つ
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_F)               # 全体表示

    # 寸法ラベルの描画
    store.set_tool("dimension")
    click(qtbot, view, 30, 15.2)                                   # 線 (10,10)-(50,20) の上
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Return)          # 線の長さ
    assert [c.type for c in store.active_sketch().constraints] == ["distance"]
    assert has_color(view, COLORS["label_selected"], -10, -20, 50, 30, tol=90) or \
        has_color(view, COLORS["dimension"], -10, -20, 50, 30, tol=90)


def test_dimension_label_click_and_double_click(qtbot, view, monkeypatch):
    """寸法ラベル: クリックで拘束を選択、ダブルクリックで値編集ダイアログ."""
    store = view.store
    store.set_tool("line")
    click(qtbot, view, 10, 10)                     # 軸・原点から離す
    click(qtbot, view, 50, 10)
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Return)
    store.set_tool("dimension")
    click(qtbot, view, 30, 10.3)
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Return)          # 線の長さ 40
    sketch = store.active_sketch()
    dim = next(c for c in sketch.constraints if c.type == "distance")
    assert dim.value == "40"
    label = next(l for l in view.controller.annotations.labels if l.constraintId == dim.id)
    label_px = view.mapFromScene(QtCore.QPointF(*label.pos))

    # 選択ツールでラベルをクリック → 拘束と関係する点が選択される（ツールには渡らない）
    store.set_tool("select")
    store.set_selected_constraint(None)
    store.set_selection([])
    qtbot.mouseClick(view.viewport(), QtCore.Qt.LeftButton, pos=label_px)
    assert store.selected_constraint_id == dim.id
    assert set(store.selection) == {dim.p1, dim.p2}

    # ダブルクリック → ダイアログ（QInputDialog.getText を差し替え）→ 値が変わり形状が更新
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("55", True)))
    qtbot.mouseDClick(view.viewport(), QtCore.Qt.LeftButton, pos=label_px)
    sketch = store.active_sketch()
    assert next(c for c in sketch.constraints if c.id == dim.id).value == "55"
    line = next(e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection)
    a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
    assert ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 == pytest.approx(55, abs=1e-6)

    # 解けない値は拒否され警告が出る（QMessageBox.warning を差し替え）
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a)))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("-5", True)))
    qtbot.mouseDClick(view.viewport(), QtCore.Qt.LeftButton, pos=label_px)
    assert next(c for c in store.active_sketch().constraints if c.id == dim.id).value == "55"
    assert len(warnings) == 1


def test_double_click_on_curve_inserts_point(qtbot, view):
    """ver2.3 の操作: 選択ツールで線・円弧をダブルクリックするとその位置に点が入り 2 本に分かれる."""
    store = view.store
    store.set_tool("line")
    click(qtbot, view, 10, 10)                     # 軸・原点から離す
    click(qtbot, view, 50, 10)
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Return)
    store.set_tool("select")
    inserted = []
    view.point_inserted.connect(inserted.append)
    qtbot.mouseDClick(view.viewport(), QtCore.Qt.LeftButton, pos=px(view, 30, 10.2))
    sketch = store.active_sketch()
    assert count_entities(sketch) == {"point": 3, "line": 2, "circle": 0, "arc": 0}
    assert len(inserted) == 1 and store.selection == inserted
    assert point_pos(sketch, inserted[0]) == pytest.approx((30, 10), abs=0.3)  # クリック位置（画素の丸め）
    # 何も無いところのダブルクリックは通常のクリック（選択解除）
    qtbot.mouseDClick(view.viewport(), QtCore.Qt.LeftButton, pos=px(view, 30, 40))
    assert store.selection == []
    assert count_entities(store.active_sketch())["point"] == 3


def test_expression_points_and_point_labels(qtbot, view):
    """式付きの点は四角で描く。点番号の表示を切り替えても描ける."""
    store = view.store
    rectangle(qtbot, view, 0, 0, 40, 20)
    pid = store.add_param()
    store.update_param(pid, name="L", expression="60")
    corner = next(e for e in store.active_sketch().entities
                  if isinstance(e, SketchPoint) and not e.projection and point_pos(store.active_sketch(), e.id) == (40, 20))
    assert store.set_point_coords(corner.id, "L", None)
    moved = next(e for e in store.active_sketch().entities if e.id == corner.id)
    assert (moved.x, moved.xExpr) == (60, "L")
    assert has_color(view, COLORS["expr_point"], 60, 20, 60, 20, tol=40)
    view.set_show_point_labels(True)
    assert has_color(view, COLORS["point_label"], 60, 20, 66, 24, tol=80)
    view.set_show_point_labels(False)


def test_physics_overlay_draws_boundary_colors_and_selects_regions(qtbot, view):
    store = view.store
    rectangle(qtbot, view, 0, 0, 40, 20)
    store.set_overlay("physics")
    assert has_color(view, BC_COLORS["PEC"], 5, 20, 35, 20, tol=30)         # 上辺 = PEC（既定）
    sketch = store.active_sketch()
    left = next(e for e in sketch.entities if isinstance(e, SketchLine) and not e.projection
                and {point_pos(sketch, e.p1), point_pos(sketch, e.p2)} == {(0, 0), (0, 20)})
    store.set_boundary([left.id], "E-short")
    assert has_color(view, BC_COLORS["E-short"], 0, 5, 0, 15, tol=30)
    # 曲線に当たらないクリックで閉領域を選ぶ（Esc で解除）
    click(qtbot, view, 20, 10)
    assert store.selected_region_id == store.profiles[0].id
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key_Escape)
    assert store.selected_region_id is None
    store.set_overlay("sketch")
    assert not has_color(view, BC_COLORS["E-short"], 0, 5, 0, 15, tol=20)
    view.set_show_bc_colors(True)                                            # モデリングでも BC の色
    assert has_color(view, BC_COLORS["E-short"], 0, 5, 0, 15, tol=30)


def test_mesh_overlay_draws_preview(qtbot, view):
    store = view.store
    rectangle(qtbot, view, 0, 0, 40, 20)
    preview = MeshPreview(triangles=[[(0, 0), (40, 0), (40, 20)], [(0, 0), (40, 20), (0, 20)]],
                          bc_segments={"PEC": [[(0, 20), (40, 20)]]}, interfaces=[[(20, 0), (20, 20)]])
    view.set_mesh_preview(preview)
    store.set_overlay("mesh")
    assert has_color(view, COLORS["mesh"], 11, 6, 19, 9, tol=40)             # 対角線（グリッド線を避けた範囲）
    assert has_color(view, COLORS["interface"], 20, 5, 20, 15, tol=40)
    assert has_color(view, BC_COLORS["PEC"], 5, 20, 35, 20, tol=30)
    view.set_mesh_preview(None)
    assert not has_color(view, COLORS["interface"], 20, 5, 20, 15, tol=20)
