"""スケッチの編集操作と問い合わせ（移植元: EM-CAD-py `emcad/core/sketch/model.py`、
さらにその移植元は TS 版 `src/core/sketch/model.ts`）.

ver3 での変更は ``add_point`` の座標式引数（``x_expr`` / ``y_expr``）だけ。投影（``projection``）
関連の分岐はコピー元との差分を小さく保つため残してあるが、ver3 のスケッチは投影を持たない。

曲線の端点は SketchPoint を ID で共有し、共有によって「つながり」を表す。
TS 版は不変更新（新しいスケッチを返す）だが、Python 版は **スケッチを直接
変更する**（Undo はドキュメントのスナップショットで行う）。変化の有無が
必要な関数は bool を返す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Union

from ..document import (
    Id,
    PlaneRef,
    SketchArc,
    SketchCircle,
    SketchConstraint,
    SketchEntity,
    SketchFeature,
    SketchLine,
    SketchPoint,
    Vec2,
    is_reference,
    new_id,
    REFERENCE_R_END,
    REFERENCE_Z_END,
)
from . import geometry as g

_HIDDEN_REFERENCE_POINTS = (REFERENCE_Z_END, REFERENCE_R_END)   # 軸を決めるためだけの点（選べない・再利用しない）

PointInput = Union[Vec2, Id]      # 既存点の ID か、新しい座標


def create_sketch_feature(name: str, plane: PlaneRef | None = None, id: Id | None = None
                          ) -> SketchFeature:
    """スケッチを作る。ver3 の平面は原点 XY（x = z, y = r）だけなので ``plane`` は省略できる."""
    return SketchFeature(id=id or new_id(), name=name, plane=plane or PlaneRef.origin())


def get_entity(sketch: SketchFeature, id: Id) -> Optional[SketchEntity]:
    for e in sketch.entities:
        if e.id == id:
            return e
    return None


def get_point(sketch: SketchFeature, id: Id) -> SketchPoint:
    e = get_entity(sketch, id)
    if not isinstance(e, SketchPoint):
        raise KeyError(f"Sketch point not found: {id}")
    return e


def point_pos(sketch: SketchFeature, id: Id) -> Vec2:
    p = get_point(sketch, id)
    return (p.x, p.y)


def find_point_near(sketch: SketchFeature, pos: Vec2, tolerance: float,
                    exclude: Optional[set[Id]] = None) -> Optional[SketchPoint]:
    best = None
    best_dist = tolerance
    for e in sketch.entities:
        if not isinstance(e, SketchPoint) or (exclude and e.id in exclude) or e.id in _HIDDEN_REFERENCE_POINTS:
            continue
        d = math.hypot(e.x - pos[0], e.y - pos[1])
        if d <= best_dist:
            best, best_dist = e, d
    return best


def add_point(sketch: SketchFeature, pos: Vec2, construction: bool = False,
              x_expr: str | None = None, y_expr: str | None = None) -> Id:
    """点を追加する。``x_expr`` / ``y_expr`` は座標の式（ver3: パラメータで決まる固定点）."""
    point = SketchPoint(id=new_id(), x=float(pos[0]), y=float(pos[1]),
                        construction=True if construction else None,
                        xExpr=x_expr, yExpr=y_expr)
    sketch.entities.append(point)
    return point.id


def ensure_point(sketch: SketchFeature, input: PointInput, tolerance: float = 0.0) -> Id:
    """既存点 ID ならそのまま、座標なら tolerance 内の既存点を再利用し、無ければ作る."""
    if isinstance(input, str):
        get_point(sketch, input)
        return input
    if tolerance > 0:
        existing = find_point_near(sketch, input, tolerance)
        if existing is not None:
            return existing.id
    return add_point(sketch, input)


def add_line(sketch: SketchFeature, a: PointInput, b: PointInput,
             tolerance: float = 0.0, construction: bool = False
             ) -> tuple[Id, Id, Id]:
    """線分を追加して (line_id, p1, p2) を返す。同じ 2 点を結ぶ線は重複させない."""
    p1 = ensure_point(sketch, a, tolerance)
    p2 = ensure_point(sketch, b, tolerance)
    for e in sketch.entities:
        if isinstance(e, SketchLine) and {e.p1, e.p2} == {p1, p2}:
            return e.id, p1, p2
    line = SketchLine(id=new_id(), p1=p1, p2=p2,
                      construction=True if construction else None)
    sketch.entities.append(line)
    return line.id, p1, p2


def add_polyline(sketch: SketchFeature, points: list[PointInput], close: bool,
                 tolerance: float = 0.0, construction: bool = False
                 ) -> tuple[list[Id], list[Id]]:
    """折れ線を追加して (line_ids, point_ids) を返す."""
    point_ids = [ensure_point(sketch, p, tolerance) for p in points]
    line_ids: list[Id] = []
    n = len(point_ids)
    count = n if close else n - 1
    for i in range(count):
        a, b = point_ids[i], point_ids[(i + 1) % n]
        if a == b:
            continue
        line_id, _, _ = add_line(sketch, a, b, construction=construction)
        line_ids.append(line_id)
    return line_ids, point_ids


def add_rectangle(sketch: SketchFeature, c1: Vec2, c2: Vec2,
                  tolerance: float = 0.0, construction: bool = False
                  ) -> tuple[list[Id], list[Id]]:
    corners = [(c1[0], c1[1]), (c2[0], c1[1]), (c2[0], c2[1]), (c1[0], c2[1])]
    return add_polyline(sketch, corners, True, tolerance, construction)


def add_circle(sketch: SketchFeature, center: PointInput, radius: float,
               tolerance: float = 0.0, construction: bool = False
               ) -> tuple[Id, Id]:
    """円を追加して (circle_id, center_id) を返す."""
    c = ensure_point(sketch, center, tolerance)
    circle = SketchCircle(id=new_id(), center=c, radius=float(radius),
                          construction=True if construction else None)
    sketch.entities.append(circle)
    return circle.id, c


def add_arc(sketch: SketchFeature, center: PointInput, start: PointInput,
            end: PointInput, tolerance: float = 0.0, construction: bool = False) -> Id:
    """中心・始点・終点から円弧を作る（始点から終点へ反時計回り）.

    終点が座標で与えられた場合は円周上に投影する。
    """
    c_id = ensure_point(sketch, center, tolerance)
    s_id = ensure_point(sketch, start, tolerance)
    c = point_pos(sketch, c_id)
    s = point_pos(sketch, s_id)
    radius = g.distance(c, s)
    end_input: PointInput = end
    if not isinstance(end, str):
        direction = g.normalize(g.sub(end, c))
        end_input = end if direction == (0.0, 0.0) else g.add(c, g.scale(direction, radius))
    e_id = ensure_point(sketch, end_input, tolerance)
    arc = SketchArc(id=new_id(), center=c_id, start=s_id, end=e_id, radius=radius,
                    construction=True if construction else None)
    sketch.entities.append(arc)
    return arc.id


def add_regular_polygon(sketch: SketchFeature, center: Vec2, vertex: Vec2, sides: int,
                        tolerance: float = 0.0, construction: bool = False
                        ) -> tuple[list[Id], list[Id]]:
    n = max(3, int(sides))
    return add_polyline(sketch, g.regular_polygon(center, vertex, n), True,
                        tolerance, construction)


def entity_point_ids(entity: SketchEntity) -> list[Id]:
    """曲線が参照する点の ID."""
    if isinstance(entity, SketchLine):
        return [entity.p1, entity.p2]
    if isinstance(entity, SketchCircle):
        return [entity.center]
    if isinstance(entity, SketchArc):
        return [entity.center, entity.start, entity.end]
    return []


def point_users(sketch: SketchFeature, point_id: Id) -> list[SketchEntity]:
    return [e for e in sketch.entities
            if not isinstance(e, SketchPoint) and point_id in entity_point_ids(e)]


def constraint_entity_ids(c: SketchConstraint) -> list[Id]:
    """拘束が参照するエンティティ ID."""
    t = c.type
    if t in ("coincident", "distance"):
        return [c.p1, c.p2]
    if t in ("horizontal", "vertical"):
        return [c.line]
    if t in ("parallel", "perpendicular", "angle"):
        return [c.l1, c.l2]
    if t in ("tangent", "equal"):
        return [c.e1, c.e2]
    if t == "concentric":
        return [c.c1, c.c2]
    if t == "fix":
        return [c.point]
    if t == "symmetric":
        return [c.p1, c.p2, c.line]
    if t in ("radius", "diameter"):
        return [c.curve]
    if t == "pointOnCurve":
        return [c.point, c.curve]
    if t == "pointLineDistance":
        return [c.point, c.line]
    if t == "circleLineDistance":
        return [c.curve, c.line]
    raise ValueError(f"Unknown constraint type: {t}")


def delete_entities(sketch: SketchFeature, ids: Iterable[Id],
                    keep_projections: bool = False) -> bool:
    """エンティティを削除する（変更があれば True）.

    - 点を削除するとそれを参照する曲線も削除する
    - 削除した曲線の端点は、他に使う曲線が無ければ一緒に削除する（投影点は残す）
    - 削除したエンティティを参照する拘束も削除する
    - 投影のエンティティを削除するとその投影全体を削除する（keep_projections=True
      なら投影の記録は残す）。投影点を使っている自作の曲線があれば、
      その点は投影から切り離した通常の点として残す
    """
    by_id = {e.id: e for e in sketch.entities}
    to_delete = {i for i in ids if not is_reference(by_id.get(i))}     # ver3: 参照ジオメトリは消せない
    if not to_delete:
        return False
    removed_projections: set[Id] = set()
    if not keep_projections:
        for e in sketch.entities:
            if e.projection and e.id in to_delete:
                removed_projections.add(e.projection)
        for e in sketch.entities:
            if e.projection and e.projection in removed_projections:
                to_delete.add(e.id)
    # 投影点を使う投影外の曲線があれば、その点は削除せず通常の点にする
    released: set[Id] = set()
    for e in sketch.entities:
        if isinstance(e, SketchPoint) or e.id in to_delete:
            continue
        for pid in entity_point_ids(e):
            p = by_id.get(pid)
            if isinstance(p, SketchPoint) and p.projection and pid in to_delete:
                to_delete.discard(pid)
                released.add(pid)
    for e in sketch.entities:
        if not isinstance(e, SketchPoint) and any(
                pid in to_delete for pid in entity_point_ids(e)):
            to_delete.add(e.id)
    candidates: set[Id] = set()
    for e in sketch.entities:
        if e.id in to_delete and not isinstance(e, SketchPoint):
            candidates.update(entity_point_ids(e))
    remaining = [e for e in sketch.entities if e.id not in to_delete]
    used: set[Id] = set()
    for e in remaining:
        if not isinstance(e, SketchPoint):
            used.update(entity_point_ids(e))
    entities: list[SketchEntity] = []
    for e in remaining:
        if isinstance(e, SketchPoint) and e.id in candidates and e.id not in used \
                and not e.projection:
            continue
        if e.id in released:
            e.projection = None
        entities.append(e)
    remaining_ids = {e.id for e in entities}
    constraints = [c for c in sketch.constraints
                   if all(i in remaining_ids for i in constraint_entity_ids(c))]
    changed = (len(entities) != len(sketch.entities)
               or len(constraints) != len(sketch.constraints) or bool(released))
    sketch.entities = entities
    sketch.constraints = constraints
    if removed_projections:
        sketch.projections = [p for p in sketch.projections
                              if p.id not in removed_projections]
        changed = True
    return changed


def is_projected(entity: Optional[SketchEntity]) -> bool:
    return entity is not None and entity.projection is not None


def merge_points(sketch: SketchFeature, keep: Id, remove: Id) -> bool:
    """2 つの点を統合する（一致拘束）。remove を参照する曲線・拘束は keep を参照する.

    投影点は動かせないので残す側にする。投影点同士は統合できない（False）。
    """
    if keep == remove:
        return False
    keep_point = get_point(sketch, keep)
    remove_point = get_point(sketch, remove)
    if remove_point.projection:
        if keep_point.projection:
            return False
        return merge_points(sketch, remove, keep)

    def remap(id: Id) -> Id:
        return keep if id == remove else id

    entities: list[SketchEntity] = []
    for e in sketch.entities:
        if isinstance(e, SketchPoint):
            if e.id != remove:
                entities.append(e)
        elif isinstance(e, SketchLine):
            if remap(e.p1) == remap(e.p2):
                continue                       # 退化した線は消す
            e.p1, e.p2 = remap(e.p1), remap(e.p2)
            entities.append(e)
        elif isinstance(e, SketchCircle):
            e.center = remap(e.center)
            entities.append(e)
        elif isinstance(e, SketchArc):
            e.center, e.start, e.end = remap(e.center), remap(e.start), remap(e.end)
            entities.append(e)
    projected = {e.id for e in entities if e.projection}
    constraints: list[SketchConstraint] = []
    seen: set = set()
    for c in sketch.constraints:
        mapped = replace(c, **{name: remap(getattr(c, name))
                               for name in ("p1", "p2", "line", "l1", "l2", "e1",
                                            "e2", "c1", "c2", "point", "curve")
                               if getattr(c, name) is not None})
        ids = constraint_entity_ids(mapped)
        # 統合で自明になった拘束（同じ点同士の一致など）は捨てる
        if len(set(ids)) != len(ids) and mapped.type in ("coincident", "distance"):
            continue
        # 動かない要素（投影・参照）だけの拘束（原点に統合した点の「軸の上」など）と、統合で重複した拘束も捨てる
        if ids and all(i in projected for i in ids):
            continue
        signature = (mapped.type, *[(k, v) for k, v in vars(mapped).items() if k != "id"])
        if signature in seen:
            continue
        seen.add(signature)
        constraints.append(mapped)
    sketch.entities = entities
    sketch.constraints = constraints
    normalize_arcs(sketch)
    return True


def translate_entities(sketch: SketchFeature, ids: Iterable[Id], delta: Vec2) -> bool:
    """指定エンティティ（と、それが参照する点）を平行移動する。投影点は動かない."""
    id_set = set(ids)
    point_ids: set[Id] = set()
    for e in sketch.entities:
        if e.id not in id_set:
            continue
        if isinstance(e, SketchPoint):
            point_ids.add(e.id)
        else:
            point_ids.update(entity_point_ids(e))
    if not point_ids:
        return False
    for e in sketch.entities:
        if isinstance(e, SketchPoint) and e.id in point_ids and not e.projection:
            e.x += delta[0]
            e.y += delta[1]
    normalize_arcs(sketch)
    return True


def set_construction(sketch: SketchFeature, ids: Iterable[Id], construction: bool) -> bool:
    """指定した曲線を構築ジオメトリにする / 通常に戻す（点は対象外）."""
    id_set = set(ids)
    changed = False
    for e in sketch.entities:
        if isinstance(e, SketchPoint) or e.id not in id_set or is_reference(e):
            continue
        if bool(e.construction) == construction:
            continue
        e.construction = True if construction else None
        changed = True
    return changed


def normalize_arcs(sketch: SketchFeature) -> bool:
    """円弧の半径を中心と始点の距離に合わせる（点が個別に動いた後の整合）."""
    changed = False
    for e in sketch.entities:
        if not isinstance(e, SketchArc):
            continue
        radius = g.distance(point_pos(sketch, e.center), point_pos(sketch, e.start))
        if abs(radius - e.radius) < 1e-12:
            continue
        e.radius = radius
        changed = True
    return changed


@dataclass(frozen=True)
class ArcParams:
    center: Vec2
    radius: float
    startAngle: float
    sweep: float


def arc_params(sketch: SketchFeature, arc: SketchArc) -> ArcParams:
    center = point_pos(sketch, arc.center)
    start = point_pos(sketch, arc.start)
    end = point_pos(sketch, arc.end)
    radius = g.distance(center, start) or arc.radius
    start_angle = g.angle_of(g.sub(start, center))
    end_angle = g.angle_of(g.sub(end, center))
    return ArcParams(center, radius, start_angle, g.ccw_sweep(start_angle, end_angle))


def circle_params(sketch: SketchFeature, circle: SketchCircle) -> tuple[Vec2, float]:
    return point_pos(sketch, circle.center), circle.radius


@dataclass(frozen=True)
class HitResult:
    id: Id
    kind: str            # point / line / circle / arc
    distance: float


def hit_test(sketch: SketchFeature, pos: Vec2, tolerance: float,
             kinds: Optional[Iterable[str]] = None, references: bool = True) -> Optional[HitResult]:
    """点を優先し、次に最も近い曲線を返す。kinds を与えるとその種別だけを対象にする.

    ``references=False`` なら参照ジオメトリ（原点・軸）を対象にしない（選択ツール・フィレット・物理のクリック）。
    参照の軸は利用者の曲線に当たらなかったときだけ、無限の直線として当たる。
    """
    allowed = set(kinds) if kinds is not None else None

    def ok(kind: str) -> bool:
        return allowed is None or kind in allowed

    best: Optional[HitResult] = None
    if ok("point"):
        for e in sketch.entities:
            if (not isinstance(e, SketchPoint) or e.id in _HIDDEN_REFERENCE_POINTS
                    or (not references and is_reference(e))):
                continue
            d = math.hypot(e.x - pos[0], e.y - pos[1])
            if d <= tolerance and (best is None or d < best.distance):
                best = HitResult(e.id, "point", d)
    if best is not None:
        return best
    reference_lines = []
    for e in sketch.entities:
        if isinstance(e, SketchPoint) or not ok(e.type):
            continue
        if is_reference(e):
            if references:
                reference_lines.append(e)        # 参照の軸は利用者の曲線が無いときだけ（下）
            continue
        if isinstance(e, SketchLine):
            d = g.distance_to_segment(pos, point_pos(sketch, e.p1), point_pos(sketch, e.p2))
        elif isinstance(e, SketchCircle):
            d = g.distance_to_circle(pos, point_pos(sketch, e.center), e.radius)
        else:
            a = arc_params(sketch, e)
            d = g.distance_to_arc(pos, a.center, a.radius, a.startAngle, a.sweep)
        if d <= tolerance and (best is None or d < best.distance):
            best = HitResult(e.id, e.type, d)
    if best is None:
        for e in reference_lines:                # 軸は無限の直線として当たる
            a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
            direction = g.sub(b, a)
            length = g.length(direction) or 1.0
            d = abs(direction[0] * (pos[1] - a[1]) - direction[1] * (pos[0] - a[0])) / length
            if d <= tolerance and (best is None or d < best.distance):
                best = HitResult(e.id, e.type, d)
    return best


def count_entities(sketch: SketchFeature) -> dict[str, int]:
    counts = {"point": 0, "line": 0, "circle": 0, "arc": 0}
    for e in sketch.entities:
        if not is_reference(e):              # 参照ジオメトリ（原点・軸）は数えない
            counts[e.type] += 1
    return counts


# ---------------------------------------------------------------------------
# 2D フィレット
# ---------------------------------------------------------------------------

def fillet_corner(sketch: SketchFeature, corner: Id, line_a: Id, line_b: Id,
                  radius: float) -> Optional[Id]:
    """角の点 corner を共有する 2 本の線の角を、半径 radius の円弧に置き換える.

    2 本の線は接点まで短縮し（corner の点は線 a の接点へ動かし、線 b には新しい点を与える）、
    接点の間に円弧（中心点は新規）を作って線それぞれとの接線拘束を付ける。寸法拘束は付けない。

    接点が線のもう一方の端に達する半径（線を使い切る）まで許す。使い切った線は削除し、
    円弧はその端点（隣の要素と共有）に直接つながる（3D フィレットの「エッジまで」と同じ扱い）。
    使い切った線との接線拘束は付けない。

    corner が他のエンティティや拘束に使われている、線が平行、接点が線の端を越える、
    投影された要素、のときは何もせず None。作った円弧の ID を返す。
    """
    from ..document import SketchArc, SketchConstraint, new_id

    la, lb = get_entity(sketch, line_a), get_entity(sketch, line_b)
    if not isinstance(la, SketchLine) or not isinstance(lb, SketchLine) or la.id == lb.id:
        return None
    if corner not in (la.p1, la.p2) or corner not in (lb.p1, lb.p2):
        return None
    if is_projected(la) or is_projected(lb) or is_projected(get_entity(sketch, corner)):
        return None
    if any(e.id not in (la.id, lb.id) for e in point_users(sketch, corner)):
        return None
    if any(corner in constraint_entity_ids(c) for c in sketch.constraints):
        return None
    p = point_pos(sketch, corner)
    a_far = la.p2 if la.p1 == corner else la.p1
    b_far = lb.p2 if lb.p1 == corner else lb.p1
    u, v = g.sub(point_pos(sketch, a_far), p), g.sub(point_pos(sketch, b_far), p)
    len_a, len_b = g.length(u), g.length(v)
    if radius <= 0 or len_a < 1e-12 or len_b < 1e-12:
        return None
    u, v = g.scale(u, 1.0 / len_a), g.scale(v, 1.0 / len_b)
    theta = math.acos(max(-1.0, min(1.0, g.dot(u, v))))     # 2 線のなす角
    if theta < 1e-6 or math.pi - theta < 1e-6:
        return None
    t = radius / math.tan(theta / 2.0)                        # 角から接点までの距離
    eps = 1e-9 * max(1.0, len_a, len_b)
    if t > len_a + eps or t > len_b + eps:
        return None                                           # 線の端を越える
    consume_a = t >= len_a - eps                              # 接点が線の端に達する → 線を使い切る
    consume_b = t >= len_b - eps
    ta = point_pos(sketch, a_far) if consume_a else g.add(p, g.scale(u, t))
    tb = point_pos(sketch, b_far) if consume_b else g.add(p, g.scale(v, t))
    center = g.add(p, g.scale(g.normalize(g.add(u, v)), radius / math.sin(theta / 2.0)))

    # 円弧の端点: 使い切る線は元の端点（隣の要素と共有）、残る線は接点
    cp = get_point(sketch, corner)
    if consume_a:
        pa = a_far
    else:
        pa = corner                                           # 角の点を線 a の接点へ動かす
        cp.x, cp.y = ta
    if consume_b:
        pb = b_far
    elif consume_a:
        pb = corner                                           # 線 a を使い切るなら角の点を線 b の接点に流用
        cp.x, cp.y = tb
    else:
        pb = add_point(sketch, tb)
        if lb.p1 == corner:
            lb.p1 = pb
        else:
            lb.p2 = pb
    c_id = add_point(sketch, center)
    a0, a1 = g.angle_of(g.sub(ta, center)), g.angle_of(g.sub(tb, center))
    start, end = (pa, pb) if g.ccw_sweep(a0, a1) <= math.pi else (pb, pa)
    construction = True if (la.construction and lb.construction) else None
    arc = SketchArc(id=new_id(), center=c_id, start=start, end=end, radius=float(radius),
                    construction=construction)
    sketch.entities.append(arc)
    if not consume_a:
        sketch.constraints.append(SketchConstraint(id=new_id(), type="tangent", e1=la.id, e2=arc.id))
    if not consume_b:
        sketch.constraints.append(SketchConstraint(id=new_id(), type="tangent", e1=lb.id, e2=arc.id))
    consumed = [line.id for line, flag in ((la, consume_a), (lb, consume_b)) if flag]
    if consumed:
        # 使い切った線とその拘束（長方形の水平/垂直など）を消す。端点は円弧が使うので残る
        delete_entities(sketch, consumed, keep_projections=True)
    return arc.id
