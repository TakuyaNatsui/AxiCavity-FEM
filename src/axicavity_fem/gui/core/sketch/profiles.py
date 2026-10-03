"""閉領域（プロファイル）検出（移植元: EM-CAD-py `emcad/core/sketch/profiles.py`、元は TS 版 `src/core/sketch/profiles.ts`）.

曲線の端点（共有点）を頂点とする平面グラフを作り、半辺を角度順にたどって面を
列挙する。得られた面の入れ子関係から「外周 + 穴」の領域を組み立てる。
円は単独で 1 つの面になる。交差しているが頂点を共有しない曲線は分割しない
（将来の拡張）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..document import Id, SketchArc, SketchCircle, SketchFeature, SketchLine, SketchPoint, Vec2
from . import geometry as g
from .model import arc_params, point_pos


@dataclass(frozen=True)
class LoopSegment:
    entityId: Id
    reversed: bool          # True なら曲線の定義方向と逆向きにたどる


@dataclass
class Profile:
    id: str                          # 外周を構成するエンティティから決まる安定な ID
    outer: list[LoopSegment]
    holes: list[list[LoopSegment]]
    outerPolygon: list[Vec2]         # 反時計回りの折れ線（表示・内外判定用）
    holePolygons: list[list[Vec2]]
    area: float                      # 穴を除いた面積
    anchor: Vec2                     # 領域内部の代表点


@dataclass
class _HalfEdge:
    index: int
    from_: int
    to: int
    twin: int
    entityId: Id
    reversed: bool
    angle: float                     # from を出るときの接線方向
    points: list[Vec2]               # from から to までの折れ線（両端を含む）


@dataclass
class _Face:
    loop: list[LoopSegment]
    polygon: list[Vec2]
    area: float
    anchor: Vec2
    parent: Optional[int] = None
    children: list[int] = field(default_factory=list)


def detect_profiles(sketch: SketchFeature, tolerance: float = 1e-6,
                    arc_step: float = 5 * g.DEG) -> list[Profile]:
    """閉領域を列挙する（面積の大きい順）.

    Args:
        tolerance: 点を同一頂点とみなす距離。
        arc_step: 円弧・円を折れ線化する角度ステップ。
    """
    # 1. 頂点（近接点をマージ）
    vertex_pos: list[Vec2] = []
    point_vertex: dict[Id, int] = {}
    for e in sketch.entities:
        if not isinstance(e, SketchPoint):
            continue
        pos = (e.x, e.y)
        found = -1
        for i, v in enumerate(vertex_pos):
            if g.distance(v, pos) <= tolerance:
                found = i
                break
        if found < 0:
            found = len(vertex_pos)
            vertex_pos.append(pos)
        point_vertex[e.id] = found

    # 2. 半辺
    half_edges: list[_HalfEdge] = []

    def add_pair(entity_id: Id, a: int, b: int, forward_points: list[Vec2],
                 forward_angle: float, reverse_angle: float) -> None:
        i = len(half_edges)
        half_edges.append(_HalfEdge(i, a, b, i + 1, entity_id, False,
                                    forward_angle, forward_points))
        half_edges.append(_HalfEdge(i + 1, b, a, i, entity_id, True,
                                    reverse_angle, list(reversed(forward_points))))

    faces: list[_Face] = []
    seen_lines: set[tuple[int, int]] = set()
    for e in sketch.entities:
        if getattr(e, "construction", None):
            continue
        if isinstance(e, SketchLine):
            a, b = point_vertex.get(e.p1), point_vertex.get(e.p2)
            if a is None or b is None or a == b:
                continue
            key = (a, b) if a < b else (b, a)   # 同じ頂点対の二重辺は 1 本だけ
            if key in seen_lines:
                continue
            seen_lines.add(key)
            pa, pb = vertex_pos[a], vertex_pos[b]
            direction = g.sub(pb, pa)
            add_pair(e.id, a, b, [pa, pb], g.angle_of(direction),
                     g.angle_of(g.scale(direction, -1.0)))
        elif isinstance(e, SketchArc):
            a, b = point_vertex.get(e.start), point_vertex.get(e.end)
            if a is None or b is None or a == b:
                continue
            p = arc_params(sketch, e)
            if p.radius <= tolerance:
                continue
            pts = g.sample_arc(p.center, p.radius, p.startAngle, p.sweep, arc_step)
            pts[0] = vertex_pos[a]                 # 端点は頂点位置に揃える
            pts[-1] = vertex_pos[b]
            add_pair(e.id, a, b, pts, p.startAngle + math.pi / 2,
                     p.startAngle + p.sweep - math.pi / 2)
        elif isinstance(e, SketchCircle):
            if e.radius <= tolerance:
                continue
            center = point_pos(sketch, e.center)
            polygon = g.sample_circle(center, e.radius,
                                      max(16, math.ceil(g.TWO_PI / arc_step)))
            faces.append(_Face([LoopSegment(e.id, False)], polygon,
                               g.signed_area(polygon), center))

    # 3. 各頂点の出る半辺を角度順に並べる
    outgoing: dict[int, list[_HalfEdge]] = {}
    for h in half_edges:
        outgoing.setdefault(h.from_, []).append(h)
    for lst in outgoing.values():
        lst.sort(key=lambda h: g.normalize_angle(h.angle))

    # 面積のしきい値（スケッチのスケールに相対）
    diag = 0.0
    if vertex_pos:
        lo, hi = g.polygon_bounds(vertex_pos)
        diag = g.distance(lo, hi)
    area_eps = max(1e-12, 1e-9 * diag * diag)

    # 4. 面のトレース: 到着した半辺の逆向きから時計回りに次の半辺を選ぶ（面は左側）
    visited = [False] * len(half_edges)
    for start in half_edges:
        if visited[start.index]:
            continue
        loop: list[LoopSegment] = []
        polygon: list[Vec2] = []
        current = start
        guard = 0
        while True:
            visited[current.index] = True
            loop.append(LoopSegment(current.entityId, current.reversed))
            polygon.extend(current.points[:-1])
            lst = outgoing[current.to]
            twin_index = next(i for i, h in enumerate(lst) if h.index == current.twin)
            current = lst[(twin_index - 1) % len(lst)]
            guard += 1
            if current.index == start.index or guard > len(half_edges) + 1:
                break
        area = g.signed_area(polygon)
        if area > area_eps:
            faces.append(_Face(loop, polygon, area, g.interior_point(polygon)))

    # 5. 入れ子: 自分を含む最小の面を親とする
    order = sorted(range(len(faces)), key=lambda i: faces[i].area)
    for k, fi in enumerate(order):
        face = faces[fi]
        for gi in order[k + 1:]:
            other = faces[gi]
            if other.area <= face.area:
                continue
            if g.point_in_polygon(face.anchor, other.polygon):
                face.parent = gi
                other.children.append(fi)
                break

    # 6. 領域 = 面 − 直下の子
    profiles: list[Profile] = []
    for face in faces:
        children = [faces[ci] for ci in face.children]
        hole_polygons = [c.polygon for c in children]
        area = face.area - sum(c.area for c in children)
        anchor = face.anchor if not children else g.interior_point(face.polygon, hole_polygons)
        profiles.append(Profile(
            id="|".join(sorted(s.entityId for s in face.loop)),
            outer=face.loop, holes=[c.loop for c in children],
            outerPolygon=face.polygon, holePolygons=hole_polygons,
            area=area, anchor=anchor))
    profiles.sort(key=lambda p: -p.area)
    return profiles


def profile_at(profiles: list[Profile], p: Vec2) -> Optional[Profile]:
    """点がどのプロファイル内部にあるか（最も面積の小さいもの）."""
    best: Optional[Profile] = None
    for prof in profiles:
        if not g.point_in_polygon(p, prof.outerPolygon):
            continue
        if any(g.point_in_polygon(p, h) for h in prof.holePolygons):
            continue
        if best is None or prof.area < best.area:
            best = prof
    return best
