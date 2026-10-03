"""スケッチツール（状態機械、移植元: EM-CAD-py `emcad/sketch/tools.py`、元は TS 版 `src/sketch/tools.ts`）.

ver3 の変更: 選択ツールのドラッグは、動かす点の座標式（``xExpr`` / ``yExpr``）を最初の移動で破棄して定数にする
（ver2.3 の「式付きの点をドラッグすると式は捨てられる」と同じ）。

座標は全てスケッチ平面のローカル 2D。ツールはコントローラが用意する
:class:`ToolContext` を通してスケッチを更新し、プレビュー・ヒント・選択を操作する。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Union

from ..core.document import Id, SketchConstraint, SketchFeature, SketchLine, SketchPoint, Vec2, is_reference, new_id
from ..core.sketch import geometry as g
from ..core.sketch.model import (
    HitResult,
    add_arc,
    add_circle,
    add_line,
    add_point,
    add_rectangle,
    add_regular_polygon,
    delete_entities,
    entity_point_ids,
    fillet_corner,
    get_entity,
    hit_test,
    point_pos,
    point_users,
    translate_entities,
)
from ..core.sketch.snap import SnapResult
from .solver import measure_dimension

MERGE_TOLERANCE = 1e-6
PointInput = Union[Vec2, Id]


@dataclass
class PreviewGeometry:
    lines: list[list[Vec2]] = field(default_factory=list)
    points: list[Vec2] = field(default_factory=list)


@dataclass
class PointerInfo:
    raw: Vec2                      # 平面上の生の座標
    snap: SnapResult
    shift: bool = False
    ctrl: bool = False


class ToolContext(Protocol):
    """ツールがコントローラ経由で使える機能."""

    def sketch(self) -> SketchFeature: ...
    def update(self, fn, history: bool = True) -> None: ...
    def restore(self, snapshot: SketchFeature, history: bool = True) -> None: ...
    def tolerance(self) -> float: ...
    def sides(self) -> int: ...
    def fillet_radius(self) -> float: ...
    def set_fillet_radius(self, value: float) -> None: ...
    def set_preview(self, preview: Optional[PreviewGeometry]) -> None: ...
    def set_hint(self, key: Optional[str]) -> None: ...
    def selection(self) -> list[Id]: ...
    def set_selection(self, ids: list[Id]) -> None: ...
    def set_hover(self, id: Optional[Id]) -> None: ...
    def add_constraint(self, constraint: SketchConstraint) -> bool: ...
    def merge_points(self, keep: Id, remove: Id) -> None: ...
    def drag_solve(self, snapshot: SketchFeature, fixed: dict[Id, Vec2],
                   history: bool = False) -> bool: ...
    def has_constraints(self) -> bool: ...
    def refresh_solve(self) -> None: ...
    def select_constraint(self, id: Optional[Id]) -> None: ...
    def clear_point_expressions(self, ids: list[Id]) -> None: ...     # ver3
    def convert_line_to_arc(self, line_id: Id) -> Optional[Id]: ...  # ver3
    def convert_arc_to_line(self, arc_id: Id) -> Optional[Id]: ...  # ver3


class BaseTool:
    id = "base"
    # 確定後に平面化（交差で分割）と軸の自動拘束を行うか。その場で変換するだけのツールは行わない（ボタンと同じ結果）
    auto_postprocess = True

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def activate(self) -> None:
        self.reset()

    def deactivate(self) -> None:
        self.reset()
        self.ctx.set_preview(None)
        self.ctx.set_hint(None)

    def snap_reference(self) -> Optional[Vec2]:
        return None

    def snap_exclude(self) -> Optional[set[Id]]:
        return None

    def on_move(self, p: PointerInfo) -> None:
        pass

    def on_down(self, p: PointerInfo) -> None:
        pass

    def on_up(self, p: PointerInfo) -> None:
        pass

    def on_finish(self) -> None:
        """Enter / 右クリック: 進行中の作図を確定して終える（ツールは維持）."""
        self.reset()

    def on_cancel(self) -> None:
        """Escape: 進行中の作図を破棄する."""
        self.reset()

    def on_numeric(self, name: str, value: float) -> None:
        pass

    def busy(self) -> bool:
        return False

    def reset(self) -> None:
        self.ctx.set_preview(None)

    def _as_input(self, p: PointerInfo) -> PointInput:
        return p.snap.pointId if p.snap.pointId is not None else p.snap.pos


# ---------------------------------------------------------------- select

@dataclass
class _Drag:
    ids: list[Id]
    point_ids: set[Id]
    start_sketch: SketchFeature          # ドラッグ開始時のスナップショット
    start_raw: Vec2
    point_start: Optional[Vec2]          # 単一点ドラッグ時の点の元位置
    start_positions: dict[Id, Vec2]
    moved: bool = False


class SelectTool(BaseTool):
    id = "select"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.drag: Optional[_Drag] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.select")

    def snap_exclude(self) -> Optional[set[Id]]:
        return self.drag.point_ids if self.drag else None

    def on_move(self, p: PointerInfo) -> None:
        drag = self.drag
        if drag is None:
            hit = hit_test(self.ctx.sketch(), p.raw, self.ctx.tolerance(), references=False)
            self.ctx.set_hover(hit.id if hit else None)
            return
        if drag.point_start is not None:
            delta = g.sub(p.snap.pos, drag.point_start)     # 単一点はスナップ先へ吸着させる
        else:
            delta = g.sub(p.raw, drag.start_raw)
        if not drag.moved and g.length(delta) < self.ctx.tolerance() * 0.5:
            return
        first = not drag.moved
        drag.moved = True
        if first:
            # ver3: 式付きの点は動かした時点で定数にする。スナップショット（ドラッグの基準）の式を消しておけば、
            # 下の restore / drag_solve がその内容を文書に書き、Undo 用の deepcopy には式付きの状態が残る
            for e in drag.start_sketch.entities:
                if isinstance(e, SketchPoint) and e.id in drag.point_ids:
                    e.xExpr = e.yExpr = None
        if self.ctx.has_constraints():
            # 拘束を保ったまま動かす: 動かす点を固定点として解く
            fixed = {pid: g.add(start, delta) for pid, start in drag.start_positions.items()}
            if not self.ctx.drag_solve(drag.start_sketch, fixed, history=first):
                return
        else:
            self.ctx.restore(drag.start_sketch, history=first)
            self.ctx.update(lambda s: translate_entities(s, drag.ids, delta), history=False)
        self.ctx.set_hint("hint.selectDrag")

    def on_down(self, p: PointerInfo) -> None:
        sketch = self.ctx.sketch()
        hit = hit_test(sketch, p.raw, self.ctx.tolerance(), references=False)     # 参照の原点・軸は選択しない
        selection = self.ctx.selection()
        if hit is None:
            if not p.shift:
                self.ctx.set_selection([])
                self.ctx.select_constraint(None)
            return
        if p.shift:
            ids = [i for i in selection if i != hit.id] if hit.id in selection \
                else [*selection, hit.id]
            self.ctx.set_selection(ids)
            return
        ids = selection if hit.id in selection else [hit.id]
        self.ctx.set_selection(ids)
        self.ctx.select_constraint(None)

        point_ids: set[Id] = set()
        for e in sketch.entities:
            if e.id not in ids:
                continue
            if isinstance(e, SketchPoint):
                point_ids.add(e.id)
            else:
                point_ids.update(entity_point_ids(e))
        # 固定拘束の点と投影点は動かせない
        for c in sketch.constraints:
            if c.type == "fix":
                point_ids.discard(c.point)
        for e in sketch.entities:
            if isinstance(e, SketchPoint) and e.projection:
                point_ids.discard(e.id)
        start_positions = {e.id: (e.x, e.y) for e in sketch.entities
                           if isinstance(e, SketchPoint) and e.id in point_ids}
        single = get_entity(sketch, hit.id) if len(ids) == 1 and hit.kind == "point" else None
        self.drag = _Drag(
            ids=list(ids), point_ids=point_ids, start_sketch=copy.deepcopy(sketch),
            start_raw=p.raw,
            point_start=(single.x, single.y) if isinstance(single, SketchPoint) else None,
            start_positions=start_positions)

    def on_up(self, p: PointerInfo) -> None:
        self.drag = None                 # 履歴は最初の移動で積んである
        self.ctx.set_hint("hint.select")

    def on_cancel(self) -> None:
        if self.drag is not None and self.drag.moved:
            self.ctx.restore(self.drag.start_sketch, history=False)
        self.drag = None
        self.ctx.set_selection([])
        self.ctx.select_constraint(None)

    def delete_selection(self) -> None:
        ids = self.ctx.selection()
        if not ids:
            return
        self.ctx.update(lambda s: delete_entities(s, ids))
        self.ctx.set_selection([])
        self.ctx.set_hover(None)

    def busy(self) -> bool:
        return self.drag is not None


# ---------------------------------------------------------------- line (polyline)

class LineTool(BaseTool):
    id = "line"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.start: Optional[PointInput] = None
        self.last_id: Optional[Id] = None
        self.last_pos: Optional[Vec2] = None
        self.first_id: Optional[Id] = None
        self.cursor: Optional[Vec2] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.lineStart")

    def snap_reference(self) -> Optional[Vec2]:
        return self.last_pos

    def on_move(self, p: PointerInfo) -> None:
        self.cursor = p.snap.pos
        if self.last_pos is not None:
            self.ctx.set_preview(PreviewGeometry(lines=[[self.last_pos, p.snap.pos]]))

    def on_down(self, p: PointerInfo) -> None:
        self._place(p.snap.pos, p.snap.pointId)

    def _place(self, pos: Vec2, point_id: Optional[Id]) -> None:
        if self.last_pos is None:
            self.start = point_id if point_id is not None else pos
            self.last_id = point_id
            self.last_pos = pos
            self.ctx.set_hint("hint.lineNext")
            return
        if g.distance(pos, self.last_pos) < MERGE_TOLERANCE:
            return
        src: PointInput = self.last_id if self.last_id is not None else self.start
        dst: PointInput = point_id if point_id is not None else pos
        result: dict = {}

        def apply(s: SketchFeature) -> bool:
            _, result["p1"], result["p2"] = add_line(s, src, dst)
            return True
        self.ctx.update(apply)
        if self.first_id is None:
            self.first_id = result["p1"]
        if point_id is not None and point_id == self.first_id:
            self.reset()                    # 始点に戻ったので閉じて終了
            self.ctx.set_hint("hint.lineStart")
            return
        self.last_id = result["p2"]
        self.last_pos = pos
        self.ctx.set_preview(None)

    def on_numeric(self, name: str, value: float) -> None:
        if name != "length" or self.last_pos is None or self.cursor is None or value <= 0:
            return
        direction = g.normalize(g.sub(self.cursor, self.last_pos))
        if direction == (0.0, 0.0):
            return
        self._place(g.add(self.last_pos, g.scale(direction, value)), None)

    def on_finish(self) -> None:
        self.reset()
        self.ctx.set_hint("hint.lineStart")

    def on_cancel(self) -> None:
        self.on_finish()

    def busy(self) -> bool:
        return self.last_pos is not None

    def reset(self) -> None:
        super().reset()
        self.start = self.last_id = self.last_pos = self.first_id = None


# ---------------------------------------------------------------- rectangle

class RectangleTool(BaseTool):
    id = "rectangle"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.c1: Optional[Vec2] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.rectStart")

    def snap_reference(self) -> Optional[Vec2]:
        return self.c1

    def on_move(self, p: PointerInfo) -> None:
        if self.c1 is None:
            return
        (x1, y1), (x2, y2) = self.c1, p.snap.pos
        self.ctx.set_preview(PreviewGeometry(
            lines=[[(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]]))

    def on_down(self, p: PointerInfo) -> None:
        if self.c1 is None:
            self.c1 = p.snap.pos
            self.ctx.set_hint("hint.rectEnd")
            return
        c1, c2 = self.c1, p.snap.pos
        if abs(c2[0] - c1[0]) < MERGE_TOLERANCE or abs(c2[1] - c1[1]) < MERGE_TOLERANCE:
            return
        def apply(s: SketchFeature) -> bool:
            ids_a, ids_b = add_rectangle(s, c1, c2, tolerance=MERGE_TOLERANCE)
            line_ids = ids_a if all(isinstance(get_entity(s, i), SketchLine) for i in ids_a) else ids_b
            # 辺は c1 → (x2, y1) → c2 → (x1, y2) の順: 偶数番が水平、奇数番が垂直
            for i, line_id in enumerate(line_ids):
                s.constraints.append(SketchConstraint(
                    id=new_id(), type="horizontal" if i % 2 == 0 else "vertical", line=line_id))
            return True
        self.ctx.update(apply)
        self.ctx.refresh_solve()
        self.reset()
        self.ctx.set_hint("hint.rectStart")

    def busy(self) -> bool:
        return self.c1 is not None

    def reset(self) -> None:
        super().reset()
        self.c1 = None


# ---------------------------------------------------------------- circle

class CircleTool(BaseTool):
    id = "circle"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.center: Optional[PointInput] = None
        self.center_pos: Optional[Vec2] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.circleCenter")

    def snap_reference(self) -> Optional[Vec2]:
        return self.center_pos

    def on_move(self, p: PointerInfo) -> None:
        if self.center_pos is None:
            return
        r = g.distance(self.center_pos, p.snap.pos)
        self.ctx.set_preview(PreviewGeometry(
            lines=[_closed_circle(self.center_pos, r), [self.center_pos, p.snap.pos]]))

    def on_down(self, p: PointerInfo) -> None:
        if self.center_pos is None:
            self.center = self._as_input(p)
            self.center_pos = p.snap.pos
            self.ctx.set_hint("hint.circleRadius")
            return
        self._commit(g.distance(self.center_pos, p.snap.pos))

    def on_numeric(self, name: str, value: float) -> None:
        if name == "radius" and self.center_pos is not None and value > 0:
            self._commit(value)

    def _commit(self, radius: float) -> None:
        if radius < MERGE_TOLERANCE or self.center is None:
            return
        center = self.center
        self.ctx.update(lambda s: add_circle(s, center, radius) and True)
        self.reset()
        self.ctx.set_hint("hint.circleCenter")

    def busy(self) -> bool:
        return self.center_pos is not None

    def reset(self) -> None:
        super().reset()
        self.center = self.center_pos = None


# ---------------------------------------------------------------- arc (center, start, end)

class ArcTool(BaseTool):
    id = "arc"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.center: Optional[PointInput] = None
        self.center_pos: Optional[Vec2] = None
        self.start: Optional[PointInput] = None
        self.start_pos: Optional[Vec2] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.arcCenter")

    def snap_reference(self) -> Optional[Vec2]:
        return self.center_pos

    def on_move(self, p: PointerInfo) -> None:
        if self.center_pos is None:
            return
        if self.start_pos is None:
            self.ctx.set_preview(PreviewGeometry(lines=[[self.center_pos, p.snap.pos]]))
            return
        self.ctx.set_preview(PreviewGeometry(lines=self._arc_preview(p.snap.pos)))

    def _arc_preview(self, cursor: Vec2) -> list[list[Vec2]]:
        c, s = self.center_pos, self.start_pos
        radius = g.distance(c, s)
        a0 = g.angle_of(g.sub(s, c))
        a1 = g.angle_of(g.sub(cursor, c))
        sweep = g.ccw_sweep(a0, a1)
        end = g.point_at_angle(c, radius, a1)
        return [g.sample_arc(c, radius, a0, sweep, 3 * g.DEG), [c, s], [c, end]]

    def on_down(self, p: PointerInfo) -> None:
        if self.center_pos is None:
            self.center = self._as_input(p)
            self.center_pos = p.snap.pos
            self.ctx.set_hint("hint.arcStart")
            return
        if self.start_pos is None:
            if g.distance(self.center_pos, p.snap.pos) < MERGE_TOLERANCE:
                return
            self.start = self._as_input(p)
            self.start_pos = p.snap.pos
            self.ctx.set_hint("hint.arcEnd")
            return
        if g.distance(self.center_pos, p.snap.pos) < MERGE_TOLERANCE:
            return
        center, start, end = self.center, self.start, self._as_input(p)
        self.ctx.update(lambda s: add_arc(s, center, start, end) and True)
        self.reset()
        self.ctx.set_hint("hint.arcCenter")

    def busy(self) -> bool:
        return self.center_pos is not None

    def reset(self) -> None:
        super().reset()
        self.center = self.center_pos = self.start = self.start_pos = None


# ---------------------------------------------------------------- regular polygon

class PolygonTool(BaseTool):
    id = "polygon"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.center: Optional[Vec2] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.polygonCenter")

    def snap_reference(self) -> Optional[Vec2]:
        return self.center

    def on_move(self, p: PointerInfo) -> None:
        if self.center is None:
            return
        pts = g.regular_polygon(self.center, p.snap.pos, self.ctx.sides())
        pts.append(pts[0])
        self.ctx.set_preview(PreviewGeometry(lines=[pts, [self.center, p.snap.pos]]))

    def on_down(self, p: PointerInfo) -> None:
        if self.center is None:
            self.center = p.snap.pos
            self.ctx.set_hint("hint.polygonVertex")
            return
        center, vertex = self.center, p.snap.pos
        if g.distance(center, vertex) < MERGE_TOLERANCE:
            return
        sides = self.ctx.sides()
        self.ctx.update(lambda s: add_regular_polygon(s, center, vertex, sides,
                                                      tolerance=MERGE_TOLERANCE) and True)
        self.reset()
        self.ctx.set_hint("hint.polygonCenter")

    def busy(self) -> bool:
        return self.center is not None

    def reset(self) -> None:
        super().reset()
        self.center = None


# ---------------------------------------------------------------- point

class PointTool(BaseTool):
    id = "point"

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.point")

    def on_down(self, p: PointerInfo) -> None:
        if p.snap.pointId is not None:
            return
        pos = p.snap.pos
        self.ctx.update(lambda s: add_point(s, pos) and True)


# ---------------------------------------------------------------- fillet

class FilletTool(BaseTool):
    """角の点（線 2 本が接する点）か線 2 本をクリックして、角を円弧に置き換える.

    半径はツールオプション（プロパティの「フィレット半径」）または数値入力で指定する。
    """

    id = "fillet"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.first: Optional[Id] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.filletPick")

    def _hit(self, p: PointerInfo) -> Optional[HitResult]:
        hit = hit_test(self.ctx.sketch(), p.raw, self.ctx.tolerance(), ("point", "line"), references=False)
        return hit if hit is not None and hit.id != self.first else None

    def on_move(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        self.ctx.set_hover(hit.id if hit else None)

    def on_down(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        if hit is None:
            return
        sketch = self.ctx.sketch()
        if hit.kind == "point":
            users = [e for e in point_users(sketch, hit.id) if isinstance(e, SketchLine)]
            if len(users) == 2:
                self._apply(hit.id, users[0].id, users[1].id)
            self.reset()
            self.ctx.set_hint("hint.filletPick")
            return
        if self.first is None:
            self.first = hit.id
            self.ctx.set_selection([hit.id])
            self.ctx.set_hint("hint.filletSecond")
            return
        la, lb = get_entity(sketch, self.first), get_entity(sketch, hit.id)
        shared = [pid for pid in (la.p1, la.p2) if pid in (lb.p1, lb.p2)]
        if shared:
            self._apply(shared[0], la.id, lb.id)
        self.reset()
        self.ctx.set_hint("hint.filletPick")

    def _apply(self, corner: Id, line_a: Id, line_b: Id) -> None:
        radius = self.ctx.fillet_radius()
        self.ctx.update(lambda s: fillet_corner(s, corner, line_a, line_b, radius) is not None)
        self.ctx.refresh_solve()

    def on_numeric(self, name: str, value: float) -> None:
        if value > 0:
            self.ctx.set_fillet_radius(value)

    def on_finish(self) -> None:
        self.reset()
        self.ctx.set_hint("hint.filletPick")

    def on_cancel(self) -> None:
        self.on_finish()

    def busy(self) -> bool:
        return self.first is not None

    def reset(self) -> None:
        super().reset()
        self.first = None
        self.ctx.set_selection([])
        self.ctx.set_hover(None)


# ---------------------------------------------------------------- constraints

LINE = ("line",)
POINT = ("point",)
CURVE = ("line", "circle", "arc")
ROUND = ("circle", "arc")


def _build_constraint(tool_id: str, sketch: SketchFeature, picks: list[Id]):
    """選び終えたときに作る拘束。("merge", a, b) は一致 = 点の統合."""
    if tool_id == "coincident":
        if get_entity(sketch, picks[1]).type == "point":
            return ("merge", picks[0], picks[1])
        return _point_on_curve(sketch, picks[0], picks[1])
    if tool_id == "pointOnCurve":
        return _point_on_curve(sketch, picks[0], picks[1])
    if tool_id in ("horizontal", "vertical"):
        return SketchConstraint(id=new_id(), type=tool_id, line=picks[0])
    if tool_id in ("parallel", "perpendicular"):
        return SketchConstraint(id=new_id(), type=tool_id, l1=picks[0], l2=picks[1])
    if tool_id == "tangent":
        ka, kb = get_entity(sketch, picks[0]).type, get_entity(sketch, picks[1]).type
        if ka == "line" and kb == "line":
            return None
        return SketchConstraint(id=new_id(), type="tangent", e1=picks[0], e2=picks[1])
    if tool_id == "equal":
        ka, kb = get_entity(sketch, picks[0]).type, get_entity(sketch, picks[1]).type
        if (ka == "line") != (kb == "line"):
            return None
        if any(is_reference(get_entity(sketch, i)) for i in picks):
            return None                    # 参照の軸は長さを持たない（線分 0..1 の長さに合わせてしまう）
        return SketchConstraint(id=new_id(), type="equal", e1=picks[0], e2=picks[1])
    if tool_id == "concentric":
        return SketchConstraint(id=new_id(), type="concentric", c1=picks[0], c2=picks[1])
    if tool_id == "fix":
        return SketchConstraint(id=new_id(), type="fix", point=picks[0])
    if tool_id == "symmetric":
        return SketchConstraint(id=new_id(), type="symmetric", p1=picks[0], p2=picks[1],
                                line=picks[2])
    return None


def _point_on_curve(sketch: SketchFeature, point_id: Id, curve_id: Id):
    """点を曲線上に。点が曲線自身の端点・中心なら意味が無いので None."""
    curve = get_entity(sketch, curve_id)
    if curve is None or point_id in entity_point_ids(curve):
        return None
    return SketchConstraint(id=new_id(), type="pointOnCurve", point=point_id, curve=curve_id)


CONSTRAINT_STEPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "coincident": (POINT, POINT + CURVE), "pointOnCurve": (POINT, CURVE), "horizontal": (LINE,), "vertical": (LINE,),
    "parallel": (LINE, LINE), "perpendicular": (LINE, LINE), "tangent": (CURVE, CURVE),
    "equal": (CURVE, CURVE), "concentric": (ROUND, ROUND), "fix": (POINT,),
    "symmetric": (POINT, POINT, LINE),
}


def _hint_for_kinds(kinds: tuple[str, ...]) -> str:
    if kinds == POINT:
        return "hint.pickPoint"
    if "point" in kinds and "line" in kinds:
        return "hint.pickPointOrCurve"
    if kinds == LINE:
        return "hint.pickLine"
    if "line" in kinds:
        return "hint.pickCurve"
    return "hint.pickCircle"


class ConstraintTool(BaseTool):
    """必要なエンティティを順にクリックして拘束を追加する汎用ツール."""

    def __init__(self, ctx: ToolContext, tool_id: str):
        super().__init__(ctx)
        self.id = tool_id
        self.steps = CONSTRAINT_STEPS[tool_id]
        self.picks: list[Id] = []

    def activate(self) -> None:
        super().activate()
        self._update_hint()

    def _allowed(self) -> tuple[str, ...]:
        return self.steps[len(self.picks)] if len(self.picks) < len(self.steps) else ()

    def _update_hint(self) -> None:
        self.ctx.set_hint(_hint_for_kinds(self._allowed()))

    def _hit(self, p: PointerInfo) -> Optional[HitResult]:
        hit = hit_test(self.ctx.sketch(), p.raw, self.ctx.tolerance(), self._allowed())
        return hit if hit is not None and hit.id not in self.picks else None

    def on_move(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        self.ctx.set_hover(hit.id if hit else None)

    def on_down(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        if hit is None:
            return
        self.picks.append(hit.id)
        self.ctx.set_selection(list(self.picks))
        if len(self.picks) >= len(self.steps):
            sketch = self.ctx.sketch()
            picks = list(self.picks)
            result = _build_constraint(self.id, sketch, picks)
            refused = result is None
            if isinstance(result, tuple):
                self.ctx.merge_points(result[1], result[2])
            elif result is not None:
                refused = not self.ctx.add_constraint(result)
            self.reset()
            with_reference = [is_reference(get_entity(sketch, i)) for i in picks]
            if refused and any(with_reference) and (result is None or all(with_reference)):
                self.ctx.set_hint("hint.referenceGeometry")      # 参照要素（原点・軸）には付けられない組み合わせ
                return
        self._update_hint()

    def on_finish(self) -> None:
        self.reset()
        self._update_hint()

    def on_cancel(self) -> None:
        self.on_finish()

    def busy(self) -> bool:
        return bool(self.picks)

    def reset(self) -> None:
        super().reset()
        self.picks = []
        self.ctx.set_selection([])
        self.ctx.set_hover(None)


# ---------------------------------------------------------------- dimension

def _format_measured(v: float) -> str:
    rounded = round(v * 10000) / 10000
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


class DimensionTool(BaseTool):
    """寸法ツール: 点 + 点 → 距離、線 → 長さ（Enter/空クリック）または線 + 線 → 角度（平行なら 2 直線の間の距離）、
    点 + 線 / 線 + 点 → 点と直線の距離（平行線と点・直線の距離は Python 版の拡張 pointLineDistance。2026-09-23
    ユーザー要望）、円/円弧 → 半径（Enter/空クリック。線と同じ流儀）または円/円弧 + 線 / 線 + 円/円弧 → 円周と直線の
    距離 circleLineDistance（2026-09-29 ユーザー要望）."""

    PARALLEL_TOLERANCE_DEG = 0.5            # これより小さい角度の 2 直線は平行とみなして距離にする

    id = "dimension"

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self.first: Optional[HitResult] = None

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint("hint.dimStart")

    def _allowed(self) -> tuple[str, ...]:
        if self.first is None:
            return ("point", "line", "circle", "arc")
        if self.first.kind == "point":
            return ("point", "line")
        if self.first.kind == "line":
            return ("point", "line", "circle", "arc")
        return ("line",)                   # 円 / 円弧の次は直線（円周と直線の距離）

    def _hit(self, p: PointerInfo) -> Optional[HitResult]:
        hit = hit_test(self.ctx.sketch(), p.raw, self.ctx.tolerance(), self._allowed())
        return hit if hit is not None and (self.first is None or hit.id != self.first.id) else None

    def on_move(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        self.ctx.set_hover(hit.id if hit else None)

    def on_down(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        if self.first is None:
            if hit is None:
                return
            self.first = hit
            self.ctx.set_selection([hit.id])
            self.ctx.set_hint({"point": "hint.dimSecondPoint", "line": "hint.dimLineNext"}.get(
                hit.kind, "hint.dimCircleNext"))
            return
        if hit is None:
            # 空白（または同じ要素）のクリック: 線なら長さ、円 / 円弧なら半径。選べない要素の上なら何もしない
            under = hit_test(self.ctx.sketch(), p.raw, self.ctx.tolerance(), references=False)
            if under is None or under.id == self.first.id:
                self._single()
            return
        kinds = (self.first.kind, hit.kind)
        round_kinds = ("circle", "arc")
        if kinds[0] in round_kinds or kinds[1] in round_kinds:
            curve, line = (self.first.id, hit.id) if kinds[0] in round_kinds else (hit.id, self.first.id)
            self._add_dimension(SketchConstraint(id=new_id(), type="circleLineDistance",
                                                 curve=curve, line=line, value="0"))
        elif kinds == ("point", "point"):
            self._add_dimension(SketchConstraint(id=new_id(), type="distance",
                                                 p1=self.first.id, p2=hit.id, value="0"))
        elif kinds == ("line", "line"):
            if self._parallel(self.first.id, hit.id):
                # 2 直線の間の距離 = 1 本目の始点から 2 本目の直線まで（平行は別の拘束で保つ）
                start = get_entity(self.ctx.sketch(), self.first.id).p1
                self._add_dimension(SketchConstraint(id=new_id(), type="pointLineDistance",
                                                     point=start, line=hit.id, value="0"))
            else:
                self._add_dimension(SketchConstraint(id=new_id(), type="angle",
                                                     l1=self.first.id, l2=hit.id, value="0"))
        else:
            point, line = (self.first.id, hit.id) if kinds == ("point", "line") else (hit.id, self.first.id)
            ln = get_entity(self.ctx.sketch(), line)
            if isinstance(ln, SketchLine) and point not in (ln.p1, ln.p2):     # 自分の端点は距離 0
                self._add_dimension(SketchConstraint(id=new_id(), type="pointLineDistance",
                                                     point=point, line=line, value="0"))

    def _parallel(self, l1: Id, l2: Id) -> bool:
        sketch = self.ctx.sketch()
        a, b = get_entity(sketch, l1), get_entity(sketch, l2)
        if not isinstance(a, SketchLine) or not isinstance(b, SketchLine):
            return False
        d1 = g.sub(point_pos(sketch, a.p2), point_pos(sketch, a.p1))
        d2 = g.sub(point_pos(sketch, b.p2), point_pos(sketch, b.p1))
        n1, n2 = math.hypot(*d1), math.hypot(*d2)
        if n1 == 0 or n2 == 0:
            return False
        return abs(g.cross(d1, d2)) / (n1 * n2) < math.sin(math.radians(self.PARALLEL_TOLERANCE_DEG))

    def _single(self) -> bool:
        """1 つだけ選んだ要素の寸法（線: 長さ、円 / 円弧: 半径）を付ける. 付けたら True（点だけでは付けない）."""
        if self.first is None:
            return False
        if self.first.kind == "line":
            self._line_length()
            return True
        if self.first.kind in ("circle", "arc"):
            self._add_dimension(SketchConstraint(id=new_id(), type="radius", curve=self.first.id, value="0"))
            return True
        return False

    def _line_length(self) -> None:
        line = get_entity(self.ctx.sketch(), self.first.id) if self.first else None
        if not isinstance(line, SketchLine):
            self.reset()
            return
        self._add_dimension(SketchConstraint(id=new_id(), type="distance",
                                             p1=line.p1, p2=line.p2, value="0"))

    def _add_dimension(self, c: SketchConstraint) -> None:
        measured = measure_dimension(self.ctx.sketch(), c)
        c.value = _format_measured(measured if measured is not None else 0.0)
        if self.ctx.add_constraint(c):
            self.ctx.select_constraint(c.id)
        self.reset()
        self.ctx.set_hint("hint.dimStart")

    def on_finish(self) -> None:
        if not self._single():
            self.reset()
        self.ctx.set_hint("hint.dimStart")

    def on_cancel(self) -> None:
        self.reset()
        self.ctx.set_hint("hint.dimStart")

    def busy(self) -> bool:
        return self.first is not None

    def reset(self) -> None:
        super().reset()
        self.first = None
        self.ctx.set_selection([])
        self.ctx.set_hover(None)


# ---------------------------------------------------------------- 線 ⇄ 円弧

class ConvertCurveTool(BaseTool):
    """線→円弧 / 円弧→線: クリックした線（円弧）をその場で変換する（続けて何本でも。Esc で選択ツールへ）.

    リボンのボタンは、線（円弧）を選んでいればすぐ変換し、選んでいなければこのツールにする（2026-09-29 ユーザー要望）。
    変換後の曲線は選択状態になる（円弧ならプロパティの「向きを反転」がすぐ使える）。
    """

    auto_postprocess = False

    def __init__(self, ctx: ToolContext, tool_id: str, kind: str, hint: str):
        super().__init__(ctx)
        self.id = tool_id
        self.kind = kind
        self.hint = hint

    def activate(self) -> None:
        super().activate()
        self.ctx.set_hint(self.hint)

    def _hit(self, p: PointerInfo) -> Optional[HitResult]:
        return hit_test(self.ctx.sketch(), p.raw, self.ctx.tolerance(), (self.kind,), references=False)

    def on_move(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        self.ctx.set_hover(hit.id if hit else None)

    def on_down(self, p: PointerInfo) -> None:
        hit = self._hit(p)
        if hit is None:
            return
        if self.kind == "line":
            self.ctx.convert_line_to_arc(hit.id)
        else:
            self.ctx.convert_arc_to_line(hit.id)
        self.ctx.set_hover(None)
        self.ctx.set_hint(self.hint)


def _closed_circle(center: Vec2, radius: float) -> list[Vec2]:
    pts = g.sample_circle(center, radius, 96)
    pts.append(pts[0])
    return pts


def create_tools(ctx: ToolContext) -> dict[str, BaseTool]:
    tools: dict[str, BaseTool] = {
        "select": SelectTool(ctx), "line": LineTool(ctx), "rectangle": RectangleTool(ctx),
        "circle": CircleTool(ctx), "arc": ArcTool(ctx), "polygon": PolygonTool(ctx),
        "point": PointTool(ctx), "dimension": DimensionTool(ctx), "fillet": FilletTool(ctx),
    }
    for tool_id in CONSTRAINT_STEPS:
        tools[tool_id] = ConstraintTool(ctx, tool_id)
    tools["lineToArc"] = ConvertCurveTool(ctx, "lineToArc", "line", "hint.pickLineToArc")
    tools["arcToLine"] = ConvertCurveTool(ctx, "arcToLine", "arc", "hint.pickArcToLine")
    return tools
