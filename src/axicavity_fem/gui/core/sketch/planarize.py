"""スケッチの平面化（T 字・交差・重なりでの分割。純関数）.

閉領域検出（``profiles.py``）は「頂点を共有する曲線」だけをつなぐので、点が別の曲線の途中に乗っていたり
（T 字）、曲線同士が交差していたり、線分が重なっていると、領域が穴として扱われたり検出されなかったりする
（2026-09-25 ユーザー報告: 長方形 2 つの辺が一部重なると小さい方だけがメッシュになる）。

:func:`planarize` はスケッチを次の規則で直し、変えた箇所の数と ID 対応表（``{古い曲線 ID: [新しい曲線 ID …]}``）を返す:

1. 点が別の曲線（線・円弧）の途中に乗っている → その点で曲線を分割
2. 曲線同士が途中で交差している → 交点を作って両方を分割
3. 同じ端点を持つ同じ曲線が 2 本ある（重なった線分は 1 で端点ごとに分割された後にこうなる）→ 後の方を消す

変化が無くなるまで繰り返す。構築線は対象外。分割は ``edit_ops.insert_point_on_curve``（水平 / 垂直の拘束を複製）。
作図の確定時に自動で行い（store.auto_planarize）、リボンの「交差で分割」でも実行する。
"""

from __future__ import annotations

import math
from typing import Optional

from ..document import Id, SketchArc, SketchFeature, SketchLine, SketchPoint, Vec2, is_reference
from . import geometry as g
from .edit_ops import IdMap, insert_point_on_curve
from .model import arc_params, delete_entities, point_pos

MAX_ROUNDS = 50


def default_tolerance(sketch: SketchFeature) -> float:
    """点が曲線に「乗っている」とみなす距離（スケッチの大きさに相対）."""
    pts = [(e.x, e.y) for e in sketch.entities if isinstance(e, SketchPoint) and not is_reference(e)]
    if len(pts) < 2:
        return 1e-9
    lo, hi = g.polygon_bounds(pts)
    return max(1e-9, 1e-7 * g.distance(lo, hi))


def _curves(sketch: SketchFeature) -> list:
    return [e for e in sketch.entities if isinstance(e, (SketchLine, SketchArc)) and not e.construction]


def _endpoint_ids(curve) -> tuple[Id, Id]:
    return (curve.p1, curve.p2) if isinstance(curve, SketchLine) else (curve.start, curve.end)


def _merge_map(id_map: IdMap, old: Id, news: list[Id]) -> None:
    """``old`` が ``news`` に置き換わったことを対応表に反映する（既存の対応の中の ``old`` も置き換える）."""
    hit = False
    for key, ids in id_map.items():
        if old in ids:
            id_map[key] = [i for i in ids if i != old] + [n for n in news if n not in ids]
            hit = True
    if not hit:
        id_map[old] = list(news)


# ---------------------------------------------------------------------------
# 幾何
# ---------------------------------------------------------------------------

def _on_line_interior(p: Vec2, a: Vec2, b: Vec2, tol: float) -> bool:
    if g.distance_to_segment(p, a, b) > tol:
        return False
    return g.distance(p, a) > tol and g.distance(p, b) > tol


def _on_arc_interior(p: Vec2, center: Vec2, radius: float, start: float, sweep: float, tol: float) -> bool:
    if abs(g.distance(p, center) - radius) > tol:
        return False
    rel = g.normalize_angle(g.angle_of(g.sub(p, center)) - start)
    ang_tol = tol / max(radius, tol)
    return ang_tol < rel < sweep - ang_tol


def _line_line(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2, tol: float) -> Optional[Vec2]:
    """2 線分の内部同士の交点（平行なら None）."""
    d1, d2 = g.sub(a2, a1), g.sub(b2, b1)
    denom = g.cross(d1, d2)
    if abs(denom) <= 1e-12 * max(g.length(d1) * g.length(d2), 1e-300):
        return None
    w = g.sub(b1, a1)
    t = g.cross(w, d2) / denom
    u = g.cross(w, d1) / denom
    lt = g.length(d1)
    lu = g.length(d2)
    if lt <= tol or lu <= tol:
        return None
    et, eu = tol / lt, tol / lu
    if et < t < 1 - et and eu < u < 1 - eu:
        return g.add(a1, g.scale(d1, t))
    return None


def _circle_line(center: Vec2, radius: float, a: Vec2, b: Vec2) -> list[Vec2]:
    d = g.sub(b, a)
    f = g.sub(a, center)
    qa = g.dot(d, d)
    if qa <= 0:
        return []
    qb = 2 * g.dot(f, d)
    qc = g.dot(f, f) - radius * radius
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return []
    root = math.sqrt(disc)
    return [g.add(a, g.scale(d, t)) for t in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa))]


def _circle_circle(c1: Vec2, r1: float, c2: Vec2, r2: float) -> list[Vec2]:
    d = g.distance(c1, c2)
    if d <= 1e-12 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return []
    h = math.sqrt(max(h2, 0.0))
    u = g.scale(g.sub(c2, c1), 1.0 / d)
    m = g.add(c1, g.scale(u, a))
    n = g.perp(u)
    return [g.add(m, g.scale(n, h)), g.sub(m, g.scale(n, h))]


# ---------------------------------------------------------------------------
# 1 ラウンド
# ---------------------------------------------------------------------------

def _split_at_existing_points(sketch: SketchFeature, tol: float, id_map: IdMap) -> int:
    """規則 1: 点が乗っている曲線をその点で分割する（見つけたら 1 つ分割して戻る）.

    対象は曲線の端点になっている点だけ（円弧の中心や孤立点が線の上にあっても分けない）。
    """
    curves = _curves(sketch)
    graph_points = {pid for c in curves for pid in _endpoint_ids(c)}
    points = [e for e in sketch.entities if isinstance(e, SketchPoint) and e.id in graph_points]
    for curve in curves:
        ends = set(_endpoint_ids(curve))
        if isinstance(curve, SketchLine):
            a, b = point_pos(sketch, curve.p1), point_pos(sketch, curve.p2)
            for p in points:
                if p.id in ends:
                    continue
                if _on_line_interior((p.x, p.y), a, b, tol):
                    pid, m = insert_point_on_curve(sketch, curve.id, (p.x, p.y), point_id=p.id)
                    if pid is not None:
                        _merge_map(id_map, curve.id, m[curve.id])
                        return 1
        else:
            ends.add(curve.center)
            ap = arc_params(sketch, curve)
            if ap.radius <= tol:
                continue
            for p in points:
                if p.id in ends:
                    continue
                if _on_arc_interior((p.x, p.y), ap.center, ap.radius, ap.startAngle, ap.sweep, tol):
                    pid, m = insert_point_on_curve(sketch, curve.id, (p.x, p.y), point_id=p.id)
                    if pid is not None:
                        _merge_map(id_map, curve.id, m[curve.id])
                        return 1
    return 0


def _crossing(sketch: SketchFeature, c1, c2, tol: float) -> Optional[Vec2]:
    """2 曲線の内部同士の交点（無ければ None）."""
    if isinstance(c1, SketchLine) and isinstance(c2, SketchLine):
        return _line_line(point_pos(sketch, c1.p1), point_pos(sketch, c1.p2),
                          point_pos(sketch, c2.p1), point_pos(sketch, c2.p2), tol)
    if isinstance(c1, SketchArc) and isinstance(c2, SketchLine):
        c1, c2 = c2, c1
    if isinstance(c1, SketchLine):
        a, b = point_pos(sketch, c1.p1), point_pos(sketch, c1.p2)
        ap = arc_params(sketch, c2)
        for q in _circle_line(ap.center, ap.radius, a, b):
            if _on_line_interior(q, a, b, tol) and \
                    _on_arc_interior(q, ap.center, ap.radius, ap.startAngle, ap.sweep, tol):
                return q
        return None
    a1, a2 = arc_params(sketch, c1), arc_params(sketch, c2)
    for q in _circle_circle(a1.center, a1.radius, a2.center, a2.radius):
        if _on_arc_interior(q, a1.center, a1.radius, a1.startAngle, a1.sweep, tol) and \
                _on_arc_interior(q, a2.center, a2.radius, a2.startAngle, a2.sweep, tol):
            return q
    return None


def _split_crossings(sketch: SketchFeature, tol: float, id_map: IdMap) -> int:
    """規則 2: 交差している 2 曲線を交点で分割する（見つけたら 1 組分割して戻る）."""
    curves = _curves(sketch)
    for i, c1 in enumerate(curves):
        for c2 in curves[i + 1:]:
            q = _crossing(sketch, c1, c2, tol)
            if q is None:
                continue
            pid, m1 = insert_point_on_curve(sketch, c1.id, q)
            if pid is None:
                continue
            _merge_map(id_map, c1.id, m1[c1.id])
            _, m2 = insert_point_on_curve(sketch, c2.id, q, point_id=pid)
            if c2.id in m2:
                _merge_map(id_map, c2.id, m2[c2.id])
            return 1
    return 0


def _same_curve(sketch: SketchFeature, c1, c2, tol: float) -> bool:
    e1 = [point_pos(sketch, i) for i in _endpoint_ids(c1)]
    e2 = [point_pos(sketch, i) for i in _endpoint_ids(c2)]
    same = (g.distance(e1[0], e2[0]) <= tol and g.distance(e1[1], e2[1]) <= tol)
    flipped = (g.distance(e1[0], e2[1]) <= tol and g.distance(e1[1], e2[0]) <= tol)
    if not (same or flipped):
        return False
    if isinstance(c1, SketchLine) and isinstance(c2, SketchLine):
        return True
    if isinstance(c1, SketchArc) and isinstance(c2, SketchArc):
        a1, a2 = arc_params(sketch, c1), arc_params(sketch, c2)
        return g.distance(a1.center, a2.center) <= tol and abs(a1.radius - a2.radius) <= tol
    return False


def _remove_duplicates(sketch: SketchFeature, tol: float, id_map: IdMap) -> int:
    """規則 3: 同じ曲線が 2 本あれば後の方を消す（見つけたら 1 本消して戻る）."""
    curves = _curves(sketch)
    for i, c1 in enumerate(curves):
        for c2 in curves[i + 1:]:
            if _same_curve(sketch, c1, c2, tol):
                delete_entities(sketch, [c2.id])
                _merge_map(id_map, c2.id, [c1.id])
                return 1
    return 0


def planarize(sketch: SketchFeature, tolerance: Optional[float] = None) -> tuple[int, IdMap]:
    """T 字・交差・重なりで曲線を分割し重複を消す。(変えた箇所の数, ID 対応表) を返す."""
    tol = default_tolerance(sketch) if tolerance is None else tolerance
    id_map: IdMap = {}
    count = 0
    for _ in range(MAX_ROUNDS * 20):
        n = _split_at_existing_points(sketch, tol, id_map) or _split_crossings(sketch, tol, id_map) \
            or _remove_duplicates(sketch, tol, id_map)
        if not n:
            break
        count += n
    return count, id_map


def count_conflicts(sketch: SketchFeature, tolerance: Optional[float] = None) -> int:
    """分割が必要な箇所の数（スケッチは変えない。検証の警告用）."""
    import copy

    return planarize(copy.deepcopy(sketch), tolerance)[0]
