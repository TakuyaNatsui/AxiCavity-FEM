"""2D スケッチエディタ（QGraphicsView。EM-CAD-py ``emcad/ui/sketch_view.py`` からの転用、元は TS 版
`src/viewport/SketchOverlay.ts` の描画と `SketchController.ts` の入力処理）.

シーン座標 = スケッチ座標 (z, r)（文書の単位）。ビュー変換で y を反転して上向きを正にする。描画はアイテムを
持たず、`drawBackground`（グリッド・r<0 の網掛け・軸）と `drawForeground`（閉領域・曲線・点・プレビュー・
スナップ・注釈）で毎回描く。ピクセル一定の要素（点・記号・文字）は変換をリセットしてデバイス座標で描く。

ver3 の追加:
    - 軸 r=0 を一点鎖線で、r<0 を網掛けで描く（軸対称なので r ≥ 0 に描く）。軸のラベルに単位
    - overlay: "sketch"（通常）/ "physics"（曲線を有効な境界条件の色で、閉領域を材料で塗る）/
      "mesh"（メッシュのプレビューを重ねる。``set_mesh_preview``）
    - 式付きの点は四角、点番号の表示（``show_point_labels``）
    - 選択ツールで線・円弧をダブルクリックすると点を挿入（ver2.3 の操作）。物理では領域のクリックで領域を選ぶ

マウス: 左 = ツール操作、中ドラッグ = パン、ホイール = ズーム、右クリック = 確定（Enter）。
拘束記号・寸法ラベル: クリックで拘束を選択、寸法はダブルクリックで値の編集ダイアログ。
キー: Esc / Enter / Delete / L R C A P S / F。
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..app.store import DocumentStore
from ..core.document import REFERENCE_ORIGIN, SketchArc, SketchCircle, SketchLine, SketchPoint, Vec2, is_reference
from ..core.sketch import geometry as g
from ..core.sketch.model import arc_params, hit_test, point_pos
from ..core.sketch.profiles import profile_at
from ..sketch.annotations import LabelAnnotation
from ..sketch.controller import SketchController
from .constraint_actions import edit_dimension_value, select_constraint
from .focus import text_entry_has_focus

COLORS = {
    "grid_minor": "#dfe5eb", "grid_major": "#c3ccd6", "grid_axis": "#8f9aa6",
    "curve": "#1d4ed8", "construction": "#7c3aed", "projected": "#0f766e",
    "selected": "#f59e0b", "hover": "#0ea5e9", "point": "#1e3a8a", "preview": "#059669",
    "profile": "#3b82f6", "profile_hover": "#38bdf8", "snap": "#dc2626", "guide": "#9ca3af",
    "dimension": "#6b7280", "label": "#374151", "label_selected": "#b45309",
    "background": "#ffffff", "axis": "#111827", "negative": "#f3f4f6", "axis_label": "#4b5563",
    "expr_point": "#7c2d12", "point_label": "#6b7280", "mesh": "#9ca3af", "interface": "#d946ef",
}
# 境界条件の色（ver2.3 の Multi-Region Editor と同じ）
BC_COLORS = {"PEC": "#cc6600", "E-short": "#1f77b4", "M-short": "#2ca02c", "None": "#444444"}
# 材料の塗り（物理）: 真空は薄い青、誘電体は薄い橙
MATERIAL_FILL = {"vacuum": "#bfdbfe", "dielectric": "#fed7aa", "selected": "#fbbf24"}
POINT_RADIUS_PX = 3.0
POINT_HILITE_PX = 4.5
SNAP_RADIUS_PX = 5.5
ZOOM_STEP = 1.15
LABEL_POINT_SIZE = 9.0


def _pen(color: str, width: float = 1.5, style=QtCore.Qt.SolidLine) -> QtGui.QPen:
    pen = QtGui.QPen(QtGui.QColor(color), width, style)
    pen.setCosmetic(True)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    return pen


def _polygon(points) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])


class MeshPreview:
    """メッシュのプレビュー（子プロセスが書いた ``mesh_preview.npz`` の中身。単位はスケッチ座標）.

    ``triangles``: [(x, y), (x, y), (x, y)] の列、``bc_segments``: {BC 名: [[(x, y), ...], ...]}、
    ``interfaces``: 誘電体界面の折れ線。描画用の QPainterPath は 1 回だけ作る。
    """

    def __init__(self, triangles, bc_segments=None, interfaces=None):
        self.triangles = triangles
        self.bc_segments = bc_segments or {}
        self.interfaces = interfaces or []
        self._path: Optional[QtGui.QPainterPath] = None

    @property
    def path(self) -> QtGui.QPainterPath:
        if self._path is None:
            path = QtGui.QPainterPath()
            for tri in self.triangles:
                path.addPolygon(_polygon([*tri, tri[0]]))
            self._path = path
        return self._path


class SketchView(QtWidgets.QGraphicsView):
    point_inserted = QtCore.Signal(str)               # ダブルクリックで点を挿入した（点の ID）

    def __init__(self, store: DocumentStore, controller: SketchController, parent=None):
        super().__init__(parent)
        self.store = store
        self.controller = controller
        self.setScene(QtWidgets.QGraphicsScene(-1e5, -1e5, 2e5, 2e5, self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        # QGraphicsView はアイテムが無いとビューポートのマウス追跡を有効にしないので明示する
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QColor(COLORS["background"]))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._pan_last: Optional[QtCore.QPoint] = None
        self._right_press: Optional[QtCore.QPoint] = None
        self._label_pressed = False        # 左ボタンがラベル上で押された（ツールに渡さない）
        self._fitted = False
        self.show_grid = True
        self.show_point_labels = False
        self.show_bc_colors = False        # モデリングでも境界条件の色で描く（表示タブ）
        self.mesh_preview: Optional[MeshPreview] = None
        controller.display_changed.connect(lambda: self.viewport().update())
        store.overlay_changed.connect(lambda _m: self.viewport().update())
        store.document_changed.connect(lambda: self.viewport().update())
        store.settings_changed.connect(lambda _s: self.viewport().update())

    # ------------------------------------------------------------------
    # 座標・ズーム
    # ------------------------------------------------------------------

    def world_per_pixel(self) -> float:
        m11 = abs(self.transform().m11())
        return 1.0 / m11 if m11 > 0 else 1.0

    def _notify_scale(self) -> None:
        self.controller.set_scale(self.world_per_pixel())
        self.viewport().update()

    def fit(self, half_extent: float = 60.0) -> None:
        """原点付近（z: ±half_extent、r: 0〜half_extent）が収まるように表示する."""
        self.fit_rect(-half_extent, half_extent, -0.15 * half_extent, half_extent)

    def fit_rect(self, x0: float, x1: float, y0: float, y1: float, margin: float = 1.1) -> None:
        """スケッチ座標の矩形が収まるように表示する."""
        w, h = max(self.viewport().width(), 1), max(self.viewport().height(), 1)
        sx = w / (max(x1 - x0, 1e-9) * margin)
        sy = h / (max(y1 - y0, 1e-9) * margin)
        s = min(sx, sy)
        self.setTransform(QtGui.QTransform(s, 0, 0, -s, 0, 0))
        self.centerOn((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self._fitted = True
        self._notify_scale()

    def fit_sketch(self) -> None:
        """スケッチ全体（と軸）が収まるように表示する（空なら ±60）."""
        sketch = self.store.active_sketch()
        pts = [(e.x, e.y) for e in sketch.entities
               if isinstance(e, SketchPoint) and not is_reference(e)] if sketch else []
        if len(pts) < 2:
            self.fit()
            return
        (x0, y0), (x1, y1) = g.polygon_bounds(pts)
        y0 = min(y0, 0.0)
        span = max(x1 - x0, y1 - y0, 1.0)
        pad = 0.08 * span
        self.fit_rect(x0 - pad, x1 + pad, y0 - pad, y1 + pad, margin=1.0)

    def fit_view_settings(self) -> bool:
        """文書の表示範囲（ver2.3 の Draw Area）があればそれに合わせる."""
        v = self.store.document.view
        if None in (v.zmin, v.zmax, v.rmin, v.rmax) or v.zmax <= v.zmin or v.rmax <= v.rmin:
            return False
        self.fit_rect(v.zmin, v.zmax, v.rmin, v.rmax, margin=1.02)
        return True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            if not self.fit_view_settings():
                self.fit_sketch()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._notify_scale()

    def _scene_pos(self, event) -> Vec2:
        p = self.mapToScene(event.position().toPoint())
        return (float(p.x()), float(p.y()))

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP ** (delta / 120.0)
        anchor = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        after = self.mapToScene(event.position().toPoint())
        d = after - anchor
        self.translate(d.x(), d.y())
        self._notify_scale()
        event.accept()

    # ------------------------------------------------------------------
    # マウス・キー
    # ------------------------------------------------------------------

    @staticmethod
    def _mods(event) -> tuple[bool, bool]:
        m = event.modifiers()
        return bool(m & QtCore.Qt.ShiftModifier), bool(m & QtCore.Qt.ControlModifier)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setFocus()
        if event.button() == QtCore.Qt.MiddleButton:
            self._pan_last = event.position().toPoint()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        elif event.button() == QtCore.Qt.LeftButton:
            label = self._label_at(event.position())
            if label is not None:
                self._label_pressed = True
                select_constraint(self.store, label.constraintId)
                self.viewport().update()
            else:
                pos = self._scene_pos(event)
                if self.store.overlay == "physics" and self.store.active_tool == "select":
                    self._physics_click(pos)
                self.controller.pointer_down(pos, *self._mods(event))
        elif event.button() == QtCore.Qt.RightButton:
            self._right_press = event.position().toPoint()
        event.accept()

    def _physics_click(self, pos: Vec2) -> None:
        """物理: 曲線に当たらないクリックは閉領域の選択（曲線の選択はツールに任せる）."""
        sketch = self.store.active_sketch()
        hit = hit_test(sketch, pos, self.controller.tolerance(), references=False)
        if hit is None:
            prof = profile_at(self.controller.profiles, pos)
            self.store.set_selected_region(prof.id if prof else None)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            label = self._label_at(event.position())
            if label is not None:
                self._label_pressed = True
                if label.dimension:
                    edit_dimension_value(self.store, label.constraintId, self)
                event.accept()
                return
            if self.store.active_tool == "select":
                # ver2.3: 線・円弧のダブルクリックでその位置に点を挿入
                pos = self._scene_pos(event)
                hit = hit_test(self.store.active_sketch(), pos, self.controller.tolerance(), ("line", "arc"),
                               references=False)
                if hit is not None:
                    self._label_pressed = True            # 続く release をツールに渡さない
                    pid = self.store.insert_point_on_curve(hit.id, pos)
                    if pid is not None:
                        self.point_inserted.emit(pid)
                    event.accept()
                    return
        # ラベル以外はクリックと同じ扱い（QWidget の既定と同じ）
        self.mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pan_last is not None:
            pos = event.position().toPoint()
            d = pos - self._pan_last
            self._pan_last = pos
            wpp = self.world_per_pixel()
            self.translate(d.x() * wpp, -d.y() * wpp)
            self.viewport().update()
        else:
            self.controller.pointer_move(self._scene_pos(event), *self._mods(event))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._pan_last = None
            self.unsetCursor()
        elif event.button() == QtCore.Qt.LeftButton:
            if self._label_pressed:
                self._label_pressed = False
            else:
                self.controller.pointer_up(self._scene_pos(event), *self._mods(event))
        elif event.button() == QtCore.Qt.RightButton:
            if self._right_press is not None and \
                    (event.position().toPoint() - self._right_press).manhattanLength() < 4:
                self.controller.finish()
            self._right_press = None
        event.accept()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        if not text_entry_has_focus() and self.window().isActiveWindow():
            self.setFocus(QtCore.Qt.MouseFocusReason)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.controller.pointer_leave()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        ctrl = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        if key == QtCore.Qt.Key_Escape:
            self.controller.cancel()
            self.store.set_selected_region(None)
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.controller.finish()
        elif key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.controller.delete_selection()
        elif key == QtCore.Qt.Key_F:
            self.fit_sketch()
        elif not ctrl and key in _TOOL_KEYS:
            self.store.set_tool(_TOOL_KEYS[key])
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        painter.fillRect(rect, QtGui.QColor(COLORS["background"]))
        # r < 0 は軸対称の外（網掛け）
        if rect.top() < 0:
            painter.fillRect(QtCore.QRectF(rect.left(), rect.top(), rect.width(), min(rect.height(), -rect.top())),
                             QtGui.QColor(COLORS["negative"]))
        spacing = self.controller.grid_spacing
        if self.show_grid and spacing > 0:
            major = spacing * 5
            x0 = math.floor(rect.left() / spacing) * spacing
            y0 = math.floor(rect.top() / spacing) * spacing
            n = (rect.width() / spacing) + (rect.height() / spacing)
            if n <= 2000:
                minor_pen, major_pen = _pen(COLORS["grid_minor"], 1.0), _pen(COLORS["grid_major"], 1.0)
                x = x0
                while x <= rect.right():
                    painter.setPen(major_pen if abs(round(x / major) * major - x) < 1e-9 else minor_pen)
                    painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
                    x += spacing
                y = y0
                while y <= rect.bottom():
                    painter.setPen(major_pen if abs(round(y / major) * major - y) < 1e-9 else minor_pen)
                    painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
                    y += spacing
        painter.setPen(_pen(COLORS["grid_axis"], 1.5))
        painter.drawLine(QtCore.QPointF(0, rect.top()), QtCore.QPointF(0, rect.bottom()))   # z = 0
        painter.setPen(_pen(COLORS["axis"], 1.6, QtCore.Qt.DashDotLine))
        painter.drawLine(QtCore.QPointF(rect.left(), 0), QtCore.QPointF(rect.right(), 0))   # 軸 r = 0

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        sketch = self.store.active_sketch()
        if sketch is None or not self.controller.active:
            return
        selection = set(self.store.selection)
        hover = self.store.hover_id
        overlay = self.store.overlay
        if overlay == "mesh" and self.mesh_preview is not None:
            self._draw_mesh(painter)
        else:
            self._draw_profiles(painter, physics=(overlay == "physics"))
        self._draw_curves(painter, sketch, selection, hover,
                          physics=(overlay in ("physics", "mesh") or self.show_bc_colors))
        self._draw_reference_lines(painter, sketch, selection, hover, rect)
        self._draw_annotation_lines(painter)
        self._draw_preview(painter)
        # 以降はピクセル一定の要素（デバイス座標）
        painter.save()
        painter.resetTransform()
        self._draw_points(painter, sketch, selection, hover)
        self._draw_snap(painter)
        self._draw_labels(painter)
        self._draw_axis_labels(painter)
        painter.restore()

    def _region_fill(self, prof) -> QtGui.QColor:
        """物理: 閉領域の材料の色（真空 / 誘電体、選択中は濃く）."""
        setting = self.store.region_setting(prof.id)
        dielectric = setting is not None and (setting.epsR != 1.0 or setting.epsRExpr or setting.tanDelta > 0)
        if prof.id == self.store.selected_region_id:
            color = QtGui.QColor(MATERIAL_FILL["selected"])
            color.setAlphaF(0.55)
        else:
            color = QtGui.QColor(MATERIAL_FILL["dielectric" if dielectric else "vacuum"])
            color.setAlphaF(0.5 if prof.id == self.store.hover_profile_id else 0.35)
        return color

    def _draw_profiles(self, painter: QtGui.QPainter, physics: bool) -> None:
        hover_id = self.store.hover_profile_id
        painter.setPen(QtCore.Qt.NoPen)
        for prof in self.controller.profiles:
            path = QtGui.QPainterPath()
            path.setFillRule(QtCore.Qt.OddEvenFill)
            path.addPolygon(_polygon(prof.outerPolygon))
            for hole in prof.holePolygons:
                path.addPolygon(_polygon(hole))
            if physics:
                color = self._region_fill(prof)
            else:
                color = QtGui.QColor(COLORS["profile_hover" if prof.id == hover_id else "profile"])
                color.setAlphaF(0.35 if prof.id == hover_id else 0.15)
            painter.fillPath(path, color)

    def _draw_mesh(self, painter: QtGui.QPainter) -> None:
        preview = self.mesh_preview
        painter.setPen(_pen(COLORS["mesh"], 1.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(preview.path)
        for bc, polylines in preview.bc_segments.items():
            painter.setPen(_pen(BC_COLORS.get(bc, COLORS["curve"]), 2.4))
            for pl in polylines:
                painter.drawPolyline(_polygon(pl))
        painter.setPen(_pen(COLORS["interface"], 1.8, QtCore.Qt.DashLine))
        for pl in preview.interfaces:
            painter.drawPolyline(_polygon(pl))

    def _curve_pen(self, e, selection, hover, physics: bool) -> QtGui.QPen:
        if e.id in selection:
            return _pen(COLORS["selected"], 3.0 if physics else 2.5)
        if e.id == hover:
            return _pen(COLORS["hover"], 2.5)
        if e.construction:
            return _pen(COLORS["construction"], 1.2, QtCore.Qt.DashLine)
        if physics:
            bc = self.store.effective_bc(e.id)[0]
            return _pen(BC_COLORS.get(bc, COLORS["curve"]), 2.6,
                        QtCore.Qt.DashLine if bc == "None" else QtCore.Qt.SolidLine)
        return _pen(COLORS["curve"], 1.8)

    def _draw_reference_lines(self, painter, sketch, selection, hover, rect: QtCore.QRectF) -> None:
        """参照の軸（z 軸 r = 0 / r 軸 z = 0）: ふだんは背景の軸線のまま。カーソルを合わせた・選んだときだけ強調する."""
        for e in sketch.entities:
            if not (isinstance(e, SketchLine) and is_reference(e)) or (e.id not in selection and e.id != hover):
                continue
            painter.setPen(_pen(COLORS["selected" if e.id in selection else "hover"], 2.5, QtCore.Qt.DashDotLine))
            a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
            if abs(b[1] - a[1]) < abs(b[0] - a[0]):          # z 軸（水平）
                painter.drawLine(QtCore.QPointF(rect.left(), a[1]), QtCore.QPointF(rect.right(), a[1]))
            else:                                            # r 軸（垂直）
                painter.drawLine(QtCore.QPointF(a[0], rect.top()), QtCore.QPointF(a[0], rect.bottom()))

    def _draw_curves(self, painter, sketch, selection, hover, physics: bool) -> None:
        for e in sketch.entities:
            if isinstance(e, SketchPoint) or is_reference(e):
                continue
            painter.setPen(self._curve_pen(e, selection, hover, physics))
            painter.setBrush(QtCore.Qt.NoBrush)
            if isinstance(e, SketchLine):
                a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
                painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
            elif isinstance(e, SketchCircle):
                c = point_pos(sketch, e.center)
                painter.drawEllipse(QtCore.QPointF(*c), e.radius, e.radius)
            elif isinstance(e, SketchArc):
                p = arc_params(sketch, e)
                pts = g.sample_arc(p.center, p.radius, p.startAngle, p.sweep, 2 * g.DEG)
                painter.drawPolyline(_polygon(pts))

    def _draw_annotation_lines(self, painter) -> None:
        painter.setPen(_pen(COLORS["dimension"], 1.0))
        for pl in self.controller.annotations.lines:
            painter.drawPolyline(_polygon(pl))

    def _draw_preview(self, painter) -> None:
        preview = self.controller.preview
        if preview is None:
            return
        painter.setPen(_pen(COLORS["preview"], 1.5, QtCore.Qt.DashLine))
        painter.setBrush(QtCore.Qt.NoBrush)
        for pl in preview.lines:
            if len(pl) >= 2:
                painter.drawPolyline(_polygon(pl))

    def _device(self, p: Vec2) -> QtCore.QPointF:
        return QtCore.QPointF(self.mapFromScene(QtCore.QPointF(p[0], p[1])))

    def _draw_points(self, painter, sketch, selection, hover) -> None:
        painter.setPen(QtCore.Qt.NoPen)
        font = self._label_font()
        painter.setFont(font)
        index = 0
        for e in sketch.entities:
            if not isinstance(e, SketchPoint):
                continue
            if is_reference(e):
                if e.id == REFERENCE_ORIGIN:
                    self._draw_origin(painter, e, selection, hover)
                continue                                     # 軸を決めるだけの点は描かない
            index += 1
            d = self._device((e.x, e.y))
            expr = bool(e.xExpr or e.yExpr)
            if e.id in selection:
                color, r = COLORS["selected"], POINT_HILITE_PX
            elif e.id == hover:
                color, r = COLORS["hover"], POINT_HILITE_PX
            elif expr:
                color, r = COLORS["expr_point"], POINT_RADIUS_PX + 0.5
            else:
                color, r = COLORS["point"], POINT_RADIUS_PX
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(color))
            if expr:
                painter.drawRect(QtCore.QRectF(d.x() - r, d.y() - r, 2 * r, 2 * r))   # 式付きの点は四角
            else:
                painter.drawEllipse(d, r, r)
            if self.show_point_labels:
                painter.setPen(QtGui.QColor(COLORS["point_label"]))
                painter.drawText(QtCore.QPointF(d.x() + r + 2, d.y() - r - 1), str(index))
        for c in sketch.constraints:            # 固定点は輪郭を付ける
            if c.type == "fix":
                try:
                    d = self._device(point_pos(sketch, c.point))
                except KeyError:
                    continue
                painter.setPen(_pen(COLORS["point"], 1.0))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(d, POINT_RADIUS_PX + 3, POINT_RADIUS_PX + 3)
                painter.setPen(QtCore.Qt.NoPen)

    def _draw_origin(self, painter, e, selection, hover) -> None:
        """参照の原点: 青緑の輪（選べる・スナップできる・曲線の端点として共有できる）."""
        d = self._device((e.x, e.y))
        color = COLORS["selected"] if e.id in selection else COLORS["hover"] if e.id == hover else COLORS["projected"]
        painter.setPen(_pen(color, 1.8))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(d, POINT_HILITE_PX + 1.0, POINT_HILITE_PX + 1.0)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(d, 1.6, 1.6)

    def _draw_snap(self, painter) -> None:
        snap = self.controller.snap
        if snap is None:
            return
        d = self._device(snap.pos)
        vp = self.viewport().rect()
        if snap.alignX is not None or snap.alignY is not None:
            painter.setPen(_pen(COLORS["guide"], 1.0, QtCore.Qt.DotLine))
            if snap.alignX is not None:
                painter.drawLine(QtCore.QPointF(d.x(), vp.top()), QtCore.QPointF(d.x(), vp.bottom()))
            if snap.alignY is not None:
                painter.drawLine(QtCore.QPointF(vp.left(), d.y()), QtCore.QPointF(vp.right(), d.y()))
        if snap.kind in ("point", "midpoint", "align", "grid"):
            painter.setPen(_pen(COLORS["snap"], 1.5))
            painter.setBrush(QtCore.Qt.NoBrush)
            r = SNAP_RADIUS_PX
            if snap.kind == "point":
                painter.drawEllipse(d, r, r)
            elif snap.kind == "midpoint":
                painter.drawPolygon(QtGui.QPolygonF([
                    QtCore.QPointF(d.x(), d.y() - r), QtCore.QPointF(d.x() + r, d.y() + r),
                    QtCore.QPointF(d.x() - r, d.y() + r)]))
            else:
                painter.drawLine(QtCore.QPointF(d.x() - r, d.y()), QtCore.QPointF(d.x() + r, d.y()))
                painter.drawLine(QtCore.QPointF(d.x(), d.y() - r), QtCore.QPointF(d.x(), d.y() + r))

    def _draw_axis_labels(self, painter) -> None:
        """軸のラベル（単位付き）: 右下に z、左上に r."""
        units = self.store.document.meta.units
        vp = self.viewport().rect()
        painter.setFont(self._label_font())
        painter.setPen(QtGui.QColor(COLORS["axis_label"]))
        painter.drawText(QtCore.QPointF(vp.right() - 60, vp.bottom() - 8), f"z [{units}]")
        painter.drawText(QtCore.QPointF(vp.left() + 8, vp.top() + 16), f"r [{units}]")

    def _label_font(self) -> QtGui.QFont:
        font = QtGui.QFont(self.viewport().font())
        font.setPointSizeF(LABEL_POINT_SIZE)
        return font

    def _label_box(self, label: LabelAnnotation, metrics: QtGui.QFontMetricsF) -> QtCore.QRectF:
        d = self._device(label.pos)
        w = metrics.horizontalAdvance(label.text) + 6
        h = metrics.height() + 2
        return QtCore.QRectF(d.x() - w / 2, d.y() - h / 2, w, h)

    def _label_at(self, view_pos: QtCore.QPointF) -> Optional[LabelAnnotation]:
        if not self.controller.active or self.controller.tool.busy():
            return None
        metrics = QtGui.QFontMetricsF(self._label_font())
        for label in reversed(self.controller.annotations.labels):
            if self._label_box(label, metrics).contains(view_pos):
                return label
        return None

    def _draw_labels(self, painter) -> None:
        font = self._label_font()
        painter.setFont(font)
        metrics = QtGui.QFontMetricsF(font)
        for label in self.controller.annotations.labels:
            box = self._label_box(label, metrics)
            bg = QtGui.QColor("#fff7ed" if label.selected else "#ffffff")
            bg.setAlphaF(0.85)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(box, 3, 3)
            painter.setPen(QtGui.QColor(COLORS["label_selected" if label.selected else "label"]))
            painter.drawText(box, QtCore.Qt.AlignCenter, label.text)

    # ------------------------------------------------------------------
    # 表示の設定
    # ------------------------------------------------------------------

    def set_mesh_preview(self, preview: Optional[MeshPreview]) -> None:
        self.mesh_preview = preview
        self.viewport().update()

    def set_show_grid(self, on: bool) -> None:
        self.show_grid = bool(on)
        self.viewport().update()

    def set_show_point_labels(self, on: bool) -> None:
        self.show_point_labels = bool(on)
        self.viewport().update()

    def set_show_bc_colors(self, on: bool) -> None:
        self.show_bc_colors = bool(on)
        self.viewport().update()

    def retranslate(self) -> None:
        self.viewport().update()


_TOOL_KEYS = {
    QtCore.Qt.Key_L: "line", QtCore.Qt.Key_R: "rectangle", QtCore.Qt.Key_C: "circle",
    QtCore.Qt.Key_A: "arc", QtCore.Qt.Key_P: "polygon", QtCore.Qt.Key_S: "select",
}
