"""ver2.3 の Multi-Region Editor にあった編集操作を、EM-CAD-py 由来のスケッチモデルで再現する（純関数）.

移植元: ver2.3 ``gui/multi_region_editor.py`` の ``add_point_on_segment`` / ``delete_selected_point``
（前後を結び直す）/ ``convert_selected_to_arc`` / ``convert_selected_to_line`` / ``update_arc_center`` /
``finalize_simple_loop_with_vacuum``（最後の点と最初の点を結ぶ部分）。

どの関数もスケッチを直接変更し、曲線の ID が変わる操作は **ID 対応表**
``{古い曲線 ID: [新しい曲線 ID, ...]}`` を返す。呼び出し側（store）はこれで境界条件の指定と
領域のキーを付け替える。Undo はドキュメントのスナップショットで行う（EM-CAD-py と同じ）。
"""

from __future__ import annotations

from typing import Optional

from ..document import Id, SketchArc, SketchConstraint, SketchFeature, SketchLine, SketchPoint, Vec2, new_id
from . import geometry as g
from .model import (
    add_line,
    add_point,
    arc_params,
    delete_entities,
    get_entity,
    get_point,
    normalize_arcs,
    point_pos,
    point_users,
)

IdMap = dict[Id, list[Id]]
_EPS = 1e-9


def _endpoints(entity) -> list[Id]:
    """線・円弧の端点（円弧の中心は含まない）."""
    if isinstance(entity, SketchLine):
        return [entity.p1, entity.p2]
    if isinstance(entity, SketchArc):
        return [entity.start, entity.end]
    return []


def _other_end(entity, point_id: Id) -> Id:
    ends = _endpoints(entity)
    return ends[1] if ends[0] == point_id else ends[0]


def _insert_after(sketch: SketchFeature, anchor_id: Id, *entities) -> None:
    """``anchor_id`` のエンティティの直後に挿入する（エンティティの順序 = 書き出し順を保つ）."""
    index = next((i for i, e in enumerate(sketch.entities) if e.id == anchor_id), len(sketch.entities) - 1)
    for k, e in enumerate(entities):
        sketch.entities.insert(index + 1 + k, e)


def _construction(*entities) -> Optional[bool]:
    return True if all(getattr(e, "construction", None) for e in entities) else None


# ---------------------------------------------------------------------------
# 点の挿入（ver2.3: 線上ダブルクリック）
# ---------------------------------------------------------------------------

def insert_point_on_curve(sketch: SketchFeature, curve_id: Id, pos: Vec2,
                          point_id: Optional[Id] = None) -> tuple[Optional[Id], IdMap]:
    """線または円弧の上（``pos`` に最も近い位置）に点を挿入して曲線を 2 本に分ける.

    端点そのものに当たるときは何もしない（None）。``point_id`` を渡すとその既存の点で分ける（平面化用）。
    分けた曲線には元の水平 / 垂直の拘束を複製し、
    元の曲線を参照するそれ以外の拘束は消える。戻り値は (新しい点の ID, ID 対応表)。
    """
    e = get_entity(sketch, curve_id)
    if isinstance(e, SketchLine):
        a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
        q = g.closest_point_on_segment(pos, a, b)
        tol = _EPS * max(1.0, g.distance(a, b))
        if g.distance(q, a) <= tol or g.distance(q, b) <= tol:
            return None, {}
        pid = point_id if point_id is not None else add_point(sketch, q)
        first = SketchLine(id=new_id(), p1=e.p1, p2=pid, construction=e.construction)
        second = SketchLine(id=new_id(), p1=pid, p2=e.p2, construction=e.construction)
        kept = [c for c in sketch.constraints if c.type in ("horizontal", "vertical") and c.line == e.id]
        _insert_after(sketch, e.id, first, second)
        delete_entities(sketch, [e.id])
        for c in kept:
            for line in (first, second):
                sketch.constraints.append(SketchConstraint(id=new_id(), type=c.type, line=line.id))
        return pid, {e.id: [first.id, second.id]}
    if isinstance(e, SketchArc):
        p = arc_params(sketch, e)
        if p.radius <= _EPS:
            return None, {}
        rel = g.normalize_angle(g.angle_of(g.sub(pos, p.center)) - p.startAngle)
        if rel <= 1e-6 or rel >= p.sweep - 1e-6:
            return None, {}
        pid = point_id if point_id is not None else \
            add_point(sketch, g.point_at_angle(p.center, p.radius, p.startAngle + rel))
        first = SketchArc(id=new_id(), center=e.center, start=e.start, end=pid, radius=p.radius,
                          construction=e.construction)
        second = SketchArc(id=new_id(), center=e.center, start=pid, end=e.end, radius=p.radius,
                           construction=e.construction)
        _insert_after(sketch, e.id, first, second)
        delete_entities(sketch, [e.id])
        return pid, {e.id: [first.id, second.id]}
    return None, {}


# ---------------------------------------------------------------------------
# 点の削除（ver2.3: ちょうど 2 本の線が集まる点を消すと前後の点を結び直す）
# ---------------------------------------------------------------------------

def delete_points_reconnect(sketch: SketchFeature, ids) -> tuple[bool, IdMap]:
    """点を削除する。ちょうど 2 本の線 / 円弧の端点になっている点は、両隣の点を直線で結び直す.

    それ以外（3 本以上が集まる点、円 / 円弧の中心、孤立点）は :func:`delete_entities` と同じ
    （その点を使う曲線ごと消える）。戻り値は (変更があったか, ID 対応表)。
    """
    changed = False
    id_map: IdMap = {}
    for pid in list(ids):
        e = get_entity(sketch, pid)
        if not isinstance(e, SketchPoint):
            continue
        users = point_users(sketch, pid)
        ends = [u for u in users if pid in _endpoints(u)]
        if len(users) == 2 and len(ends) == 2:
            ua, ub = ends
            far_a, far_b = _other_end(ua, pid), _other_end(ub, pid)
            if far_a != far_b:
                line_id, _, _ = add_line(sketch, far_a, far_b, construction=bool(_construction(ua, ub)))
                delete_entities(sketch, [pid])          # ua / ub は消え、far_a / far_b は新しい線が使う
                id_map[ua.id] = [line_id]
                id_map[ub.id] = [line_id]
                changed = True
                continue
        changed = delete_entities(sketch, [pid]) or changed
    return changed, id_map


# ---------------------------------------------------------------------------
# 線 ⇄ 円弧（ver2.3: Convert to Arc / Convert to Line / Update Center）
# ---------------------------------------------------------------------------

def arc_center_from_chord(a: Vec2, b: Vec2) -> Optional[Vec2]:
    """ver2.3 ``arc_params_from_two_points`` の中心: 弦の中点から左向きの垂線方向へ |ab|/2.

    この中心なら a から b へ反時計回りに進む弧が劣弧（90°）になる。
    """
    d = g.sub(b, a)
    dist = g.length(d)
    if dist < 1e-12:
        return None
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    perp = (-d[1] / dist, d[0] / dist)
    return g.add(mid, g.scale(perp, dist / 2.0))


def line_to_arc(sketch: SketchFeature, line_id: Id) -> tuple[Optional[Id], IdMap]:
    """線を、同じ 2 端点を結ぶ 90° の劣弧（中心点は新設）に置き換える."""
    e = get_entity(sketch, line_id)
    if not isinstance(e, SketchLine):
        return None, {}
    a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
    center = arc_center_from_chord(a, b)
    if center is None:
        return None, {}
    c_id = add_point(sketch, center)
    arc = SketchArc(id=new_id(), center=c_id, start=e.p1, end=e.p2, radius=g.distance(center, a),
                    construction=e.construction)
    _insert_after(sketch, e.id, arc)
    delete_entities(sketch, [e.id])
    return arc.id, {e.id: [arc.id]}


def flip_arc(sketch: SketchFeature, arc_id: Id) -> bool:
    """円弧の膨らむ向きを反対側にする（中心を弦に対して鏡映し、始点と終点を入れ替える）.

    円弧は「中心のまわりを反時計回りに start → end」なので、中心を弦の反対側に移すだけでは同じ側を通る優弧になる。
    端点を入れ替えることで、反対側を通る同じ角度の弧になる。中心点を他の円 / 円弧と共有していれば専用の中心点を
    新設する（式は引き継がない）。円弧の ID は変わらない。
    """
    e = get_entity(sketch, arc_id)
    if not isinstance(e, SketchArc):
        return False
    a, b = point_pos(sketch, e.start), point_pos(sketch, e.end)
    c = point_pos(sketch, e.center)
    d = g.sub(b, a)
    length = g.length(d)
    if length < 1e-12:
        return False
    # 中心を弦 ab に対して鏡映する
    t = g.dot(g.sub(c, a), d) / (length * length)
    foot = g.add(a, g.scale(d, t))
    mirrored = g.sub(g.scale(foot, 2.0), c)
    shared = (any(u.id != e.id for u in point_users(sketch, e.center))
              or get_entity(sketch, e.center).projection is not None)     # 参照の原点は動かさない
    if shared:
        e.center = add_point(sketch, mirrored)
    else:
        center = get_entity(sketch, e.center)
        center.x, center.y = mirrored
        center.xExpr = center.yExpr = None
    e.start, e.end = e.end, e.start
    normalize_arcs(sketch)
    return True


def arc_to_line(sketch: SketchFeature, arc_id: Id) -> tuple[Optional[Id], IdMap]:
    """円弧を、その 2 端点を結ぶ線に戻す（中心点は他で使われていなければ消える）."""
    e = get_entity(sketch, arc_id)
    if not isinstance(e, SketchArc):
        return None, {}
    line = SketchLine(id=new_id(), p1=e.start, p2=e.end, construction=e.construction)
    _insert_after(sketch, e.id, line)
    delete_entities(sketch, [e.id])
    return line.id, {e.id: [line.id]}


def set_arc_center(sketch: SketchFeature, arc_id: Id, center: Vec2,
                   x_expr: Optional[str] = None, y_expr: Optional[str] = None) -> bool:
    """円弧の中心を動かす（半径は端点との距離に合わせる）.

    中心点を他の円 / 円弧と共有しているときは、この円弧専用の中心点を新設する。
    """
    e = get_entity(sketch, arc_id)
    if not isinstance(e, SketchArc):
        return False
    shared = (any(u.id != e.id for u in point_users(sketch, e.center))
              or get_entity(sketch, e.center).projection is not None)     # 参照の原点は動かさない
    if shared:
        e.center = add_point(sketch, center, x_expr=x_expr, y_expr=y_expr)
    else:
        cp = get_point(sketch, e.center)
        cp.x, cp.y = float(center[0]), float(center[1])
        cp.xExpr, cp.yExpr = x_expr, y_expr
    normalize_arcs(sketch)
    return True


# ---------------------------------------------------------------------------
# 端点を結ぶ（ver2.3: Close Loop）
# ---------------------------------------------------------------------------

def open_endpoints(sketch: SketchFeature) -> list[Id]:
    """非構築の線 / 円弧の端点として 1 回しか使われていない点（開いた端点）."""
    degree: dict[Id, int] = {}
    for e in sketch.entities:
        if isinstance(e, (SketchLine, SketchArc)) and not e.construction:
            for pid in _endpoints(e):
                degree[pid] = degree.get(pid, 0) + 1
    return [pid for pid, n in degree.items() if n == 1]


def close_polyline(sketch: SketchFeature) -> tuple[Optional[Id], IdMap]:
    """開いた端点がちょうど 2 つなら、それらを線で結んで閉じる（ver2.3 の Close Loop）."""
    ends = open_endpoints(sketch)
    if len(ends) != 2:
        return None, {}
    # 2 つの開いた端点が同じ 1 本の曲線の両端なら（孤立した線）、結ぶものが無い
    for e in sketch.entities:
        if isinstance(e, (SketchLine, SketchArc)) and set(_endpoints(e)) == set(ends):
            return None, {}
    line_id, _, _ = add_line(sketch, ends[0], ends[1])
    return line_id, {}
