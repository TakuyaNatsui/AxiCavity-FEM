"""スケッチ拘束ソルバ（planegcs = FreeCAD PlaneGCS の Python バインディング）
（移植元: EM-CAD-py `emcad/sketch/solver.py`、元は TS 版 `src/sketch/solver.ts`）.

ver3 の変更: 式評価器を ver2.3 の ``expression_eval`` に（``core.expressions``）。座標の式を持つ点
（``SketchPoint.xExpr`` / ``yExpr``、ver2.3 の点座標の式入力）は、式のある座標だけを固定パラメータにして解く
（``add_point_from_params``。両方に式があれば固定点）。座標の値は store が式から評価済みにしておく。

スケッチのエンティティ/拘束を planegcs のプリミティブに変換して解き、結果を
スケッチに書き戻す。ドラッグ中は動かす点を固定点として解くので、拘束を
保ったまま追従する。

TS 版との違い: 解けたときはスケッチを**直接更新**する（失敗時は変更しない）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from planegcs import Sketch as GcsSketch
from planegcs import SolveStatus

from ..core.document import (
    Id,
    SketchArc,
    SketchCircle,
    SketchConstraint,
    SketchEntity,
    SketchFeature,
    SketchLine,
    SketchPoint,
    Vec2,
    is_dimension_constraint,
)
from ..core.expressions import evaluate_expression
from ..core.sketch import geometry as g
from ..core.sketch.model import arc_params, get_entity, point_pos

_NEAR = 1e-12


class SolverInputError(ValueError):
    pass


@dataclass
class SolveOutcome:
    """Attributes:
        ok: 解けたか（スケッチは ok のときだけ更新される）。
        status: ok / redundant / conflict / failed / invalid。
        conflicting, redundant: 該当する拘束 ID。
        dof: 自由度（invalid のときは 0）。
        error: 式の評価エラーなど。
        changed: 書き戻しでエンティティが動いたか。
    """

    ok: bool
    status: str
    conflicting: list[Id] = field(default_factory=list)
    redundant: list[Id] = field(default_factory=list)
    dof: int = 0
    error: Optional[str] = None
    changed: bool = False


def _kind_of(sketch: SketchFeature, id: Id) -> str:
    e = get_entity(sketch, id)
    if e is None:
        raise SolverInputError(f"entity not found: {id}")
    return e.type


def _endpoints(sketch: SketchFeature, id: Id) -> list[Id]:
    e = get_entity(sketch, id)
    if isinstance(e, SketchLine):
        return [e.p1, e.p2]
    if isinstance(e, SketchArc):
        return [e.start, e.end]
    return []


def _shared_endpoint(sketch: SketchFeature, e1: Id, e2: Id) -> Optional[Id]:
    """2 つの曲線（線・円弧）が共有する端点（無ければ None）."""
    ends = _endpoints(sketch, e2)
    for pid in _endpoints(sketch, e1):
        if pid in ends:
            return pid
    return None


def _gcs_normal(sketch: SketchFeature, id: Id, p: Vec2) -> Vec2:
    """planegcs（GCS）と同じ規約の法線: 線は方向 p1→p2 を反時計回りに 90° 回したもの、
    円/円弧は中心 − p."""
    e = get_entity(sketch, id)
    if isinstance(e, SketchLine):
        a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
        return (-(b[1] - a[1]), b[0] - a[0])
    center = point_pos(sketch, e.center)
    return (center[0] - p[0], center[1] - p[1])


def _tangent_angle_at(sketch: SketchFeature, e1: Id, e2: Id, point: Id) -> float:
    """共有点での接線を angle_via_point で表すときの角度（0 か π。現在の向きに近い方）."""
    p = point_pos(sketch, point)
    n1, n2 = _gcs_normal(sketch, e1, p), _gcs_normal(sketch, e2, p)
    return 0.0 if n1[0] * n2[0] + n1[1] * n2[1] >= 0.0 else math.pi


def _require_kind(sketch: SketchFeature, id: Id, *kinds: str) -> str:
    kind = _kind_of(sketch, id)
    if kind not in kinds:
        raise SolverInputError(f"unsupported entity kind {kind} for constraint")
    return kind


def _center_of(sketch: SketchFeature, id: Id) -> Id:
    e = get_entity(sketch, id)
    if not isinstance(e, (SketchCircle, SketchArc)):
        raise SolverInputError("not a circle/arc")
    return e.center


class SketchSolver:
    """拘束を満たすようにスケッチを解く（呼び出しごとに planegcs Sketch を作り直す）."""

    def solve(self, sketch: SketchFeature, params: Mapping[str, float] | None = None,
              fixed_points: Mapping[Id, Vec2] | None = None) -> SolveOutcome:
        """Args:
            params: ドキュメントのパラメータ値（寸法式の参照先）。
            fixed_points: ドラッグ中などに位置を固定する点 {ID: 位置}。
        """
        params = params or {}
        fixed_points = fixed_points or {}
        try:
            gcs, ids, tags = self._build(sketch, params, fixed_points)
        except (SolverInputError, ValueError) as exc:
            return SolveOutcome(False, "invalid", error=str(exc))
        if not ids:
            return SolveOutcome(True, "ok")

        try:
            status = gcs.solve()
            diag = gcs.diagnose()
        except Exception as exc:                        # planegcs 内部エラー
            return SolveOutcome(False, "invalid", error=str(exc))
        conflicting = [tags[t] for t in diag.conflicting if t in tags]
        redundant = [tags[t] for t in list(diag.redundant) + list(diag.partially_redundant)
                     if t in tags]
        # 同じ拘束が複数タグに現れても 1 回だけ
        conflicting = list(dict.fromkeys(conflicting))
        redundant = list(dict.fromkeys(redundant))
        dof = int(diag.dof)
        if conflicting:
            return SolveOutcome(False, "conflict", conflicting, redundant, dof)
        if status in (SolveStatus.Failed, SolveStatus.SuccessfulSolutionInvalid):
            return SolveOutcome(False, "failed", conflicting, redundant, dof)
        changed = self._read_back(sketch, gcs, ids)
        return SolveOutcome(True, "redundant" if redundant else "ok",
                            conflicting, redundant, dof, changed=changed)

    # ------------------------------------------------------------------

    def _build(self, sketch: SketchFeature, params: Mapping[str, float],
               fixed_points: Mapping[Id, Vec2]):
        """スケッチ → planegcs（点 → 曲線 → 円弧規則 → 拘束の順）."""
        gcs = GcsSketch()
        ids: dict[Id, object] = {}            # エンティティ ID → planegcs ID
        self._radius_params: dict[Id, object] = {}   # 円 ID → 半径のパラメータ（円周と直線の距離で使う）
        tags: dict[object, Id] = {}           # 拘束タグ → 拘束 ID
        fixed_by_constraint = {c.point for c in sketch.constraints if c.type == "fix"}

        for e in sketch.entities:
            if not isinstance(e, SketchPoint):
                continue
            # 投影点は評価結果で位置が決まるので常に固定（ドラッグもされない）
            dragged = None if e.projection else fixed_points.get(e.id)
            x, y = (dragged if dragged is not None else (e.x, e.y))
            fixed = (dragged is not None or e.id in fixed_by_constraint
                     or e.fixed is True or e.projection is not None)
            if fixed:
                ids[e.id] = gcs.add_fixed_point(x, y)
            elif e.xExpr or e.yExpr:
                # ver3: 式のある座標だけ固定（パラメータで決まる）。もう一方は自由
                px = gcs.add_fixed_param(x) if e.xExpr else gcs.add_param(x)
                py = gcs.add_fixed_param(y) if e.yExpr else gcs.add_param(y)
                ids[e.id] = gcs.add_point_from_params(px, py)
            else:
                ids[e.id] = gcs.add_point(x, y)

        arcs: list[Id] = []
        for e in sketch.entities:
            if isinstance(e, SketchLine):
                ids[e.id] = gcs.add_line(ids[e.p1], ids[e.p2])
            elif isinstance(e, SketchCircle):
                # 投影した円の半径は変えられない（固定パラメータ）
                radius = gcs.add_fixed_param(e.radius) if e.projection \
                    else gcs.add_param(e.radius)
                ids[e.id] = gcs.add_circle(ids[e.center], radius)
                self._radius_params[e.id] = radius
            elif isinstance(e, SketchArc):
                p = arc_params(sketch, e)
                ids[e.id] = gcs.add_arc_cse(ids[e.center], ids[e.start], ids[e.end],
                                            p.radius, p.startAngle,
                                            p.startAngle + p.sweep)
                arcs.append(e.id)
        for arc_id in arcs:
            gcs.arc_rules(ids[arc_id])

        for c in sketch.constraints:
            tag = self._add_constraint(gcs, sketch, c, params, ids)
            for t in (tag if isinstance(tag, tuple) else (tag,)):   # 複数の planegcs 拘束で表す拘束もある
                if t is not None:
                    tags[t] = c.id
        return gcs, ids, tags

    def _add_constraint(self, gcs: GcsSketch, sketch: SketchFeature,
                        c: SketchConstraint, params: Mapping[str, float],
                        ids: dict[Id, object]):
        def value() -> float:
            return evaluate_expression(c.value, params)

        t = c.type
        if t == "coincident":
            return gcs.coincident(ids[c.p1], ids[c.p2])
        if t == "horizontal":
            _require_kind(sketch, c.line, "line")
            return gcs.horizontal(ids[c.line])
        if t == "vertical":
            _require_kind(sketch, c.line, "line")
            return gcs.vertical(ids[c.line])
        if t == "parallel":
            _require_kind(sketch, c.l1, "line")
            _require_kind(sketch, c.l2, "line")
            return gcs.parallel(ids[c.l1], ids[c.l2])
        if t == "perpendicular":
            _require_kind(sketch, c.l1, "line")
            _require_kind(sketch, c.l2, "line")
            return gcs.perpendicular(ids[c.l1], ids[c.l2])
        if t == "angle":
            _require_kind(sketch, c.l1, "line")
            _require_kind(sketch, c.l2, "line")
            return gcs.set_l2l_angle(ids[c.l1], ids[c.l2], value() * g.DEG)
        if t == "tangent":
            k1, k2 = _kind_of(sketch, c.e1), _kind_of(sketch, c.e2)
            pair = (k1, k2)
            a, b = ids[c.e1], ids[c.e2]
            shared = _shared_endpoint(sketch, c.e1, c.e2)
            if shared is not None:
                # 端点を共有する接線は「点経由の角度」で表す（FreeCAD の TangentViaPoint）。
                # planegcs の tangent_line_arc は端点共有だと円弧規則と冗長扱いになり、
                # 自由度も正しく数えられない
                angle = _tangent_angle_at(sketch, c.e1, c.e2, shared)
                return gcs.set_angle_via_point(a, b, ids[shared], angle)
            if pair == ("line", "circle"):
                return gcs.tangent_line_circle(a, b)
            if pair == ("circle", "line"):
                return gcs.tangent_line_circle(b, a)
            if pair == ("line", "arc"):
                return gcs.tangent_line_arc(a, b)
            if pair == ("arc", "line"):
                return gcs.tangent_line_arc(b, a)
            if pair == ("circle", "circle"):
                return gcs.tangent_circle_circle(a, b)
            if pair == ("circle", "arc"):
                return gcs.tangent_circle_arc(a, b)
            if pair == ("arc", "circle"):
                return gcs.tangent_circle_arc(b, a)
            if pair == ("arc", "arc"):
                return gcs.tangent_arc_arc(a, b)
            raise SolverInputError(f"tangent not supported for {k1}-{k2}")
        if t == "pointOnCurve":
            _require_kind(sketch, c.point, "point")
            kind = _require_kind(sketch, c.curve, "line", "circle", "arc")
            if kind == "line":
                return gcs.point_on_line(ids[c.point], ids[c.curve])
            if kind == "circle":
                return gcs.point_on_circle(ids[c.point], ids[c.curve])
            return gcs.point_on_arc(ids[c.point], ids[c.curve])
        if t == "equal":
            k1, k2 = _kind_of(sketch, c.e1), _kind_of(sketch, c.e2)
            pair = (k1, k2)
            a, b = ids[c.e1], ids[c.e2]
            if pair == ("line", "line"):
                return gcs.equal_length(a, b)
            if pair == ("circle", "circle"):
                return gcs.equal_radius_cc(a, b)
            if pair == ("circle", "arc"):
                return gcs.equal_radius_ca(a, b)
            if pair == ("arc", "circle"):
                return gcs.equal_radius_ca(b, a)
            if pair == ("arc", "arc"):
                return gcs.equal_radius_aa(a, b)
            raise SolverInputError(f"equal not supported for {k1}-{k2}")
        if t == "concentric":
            p1, p2 = _center_of(sketch, c.c1), _center_of(sketch, c.c2)
            if p1 == p2:
                return None
            return gcs.coincident(ids[p1], ids[p2])
        if t == "fix":
            return None                     # 点の固定フラグで表現
        if t == "symmetric":
            _require_kind(sketch, c.p1, "point")
            _require_kind(sketch, c.p2, "point")
            _require_kind(sketch, c.line, "line")
            return gcs.symmetric_line(ids[c.p1], ids[c.p2], ids[c.line])
        if t == "distance":
            _require_kind(sketch, c.p1, "point")
            _require_kind(sketch, c.p2, "point")
            return gcs.set_p2p_distance(ids[c.p1], ids[c.p2], value())
        if t == "pointLineDistance":
            _require_kind(sketch, c.point, "point")
            _require_kind(sketch, c.line, "line")
            return gcs.set_p2l_distance(ids[c.point], ids[c.line], value())
        if t == "circleLineDistance":
            return self._circle_line_distance(gcs, sketch, c, value(), ids)
        if t == "radius":
            kind = _require_kind(sketch, c.curve, "circle", "arc")
            return gcs.set_circle_radius(ids[c.curve], value()) if kind == "circle" \
                else gcs.set_arc_radius(ids[c.curve], value())
        if t == "diameter":
            kind = _require_kind(sketch, c.curve, "circle", "arc")
            return gcs.set_circle_diameter(ids[c.curve], value()) if kind == "circle" \
                else gcs.set_arc_diameter(ids[c.curve], value())
        raise SolverInputError(f"unknown constraint type: {t}")

    def _circle_line_distance(self, gcs: GcsSketch, sketch: SketchFeature, c: SketchConstraint,
                              gap: float, ids: dict[Id, object]):
        """円周と直線の距離 = |中心と直線の距離 − 半径|（円弧は円弧を含む円で測る）.

        planegcs の C2L / A2L 距離は、半径が自由で直線が円を横切ると解けない（2026-09-29 実測）。そこで
        「中心と直線の距離 D」を補助パラメータにし、今の形が外側（D ≥ r）なら D − r = gap、内側なら r − D = gap
        （``difference``）と、点と直線の距離 D（``p2l_distance``）で表す。円弧の半径は ``add_arc_cse`` の中にあって
        取り出せないので、中心と始点の距離 R（補助パラメータ）を半径として使う。
        """
        e = get_entity(sketch, c.curve)
        _require_kind(sketch, c.curve, "circle", "arc")
        line = get_entity(sketch, c.line)
        _require_kind(sketch, c.line, "line")
        if gap < 0:
            raise SolverInputError("circle-line distance must be >= 0")
        tags = []
        if isinstance(e, SketchCircle):
            radius_value = e.radius
            radius = self._radius_params[e.id]
        else:
            radius_value = arc_params(sketch, e).radius
            radius = gcs.add_param(radius_value)
            tags.append(gcs.p2p_distance(ids[e.center], ids[e.start], radius))
        center_distance = point_line_distance(sketch, e.center, line)
        distance = gcs.add_param(center_distance)
        fixed_gap = gcs.add_fixed_param(gap)
        if center_distance >= radius_value:
            tags.append(gcs.difference(radius, distance, fixed_gap))      # D − r = gap（直線は円の外）
        else:
            tags.append(gcs.difference(distance, radius, fixed_gap))      # r − D = gap（直線が円を横切る）
        tags.append(gcs.p2l_distance(ids[e.center], ids[c.line], distance))
        return tuple(tags)

    @staticmethod
    def _read_back(sketch: SketchFeature, gcs: GcsSketch, ids: dict[Id, object]) -> bool:
        """planegcs の解をスケッチに書き戻す（動いたものがあれば True）."""
        changed = False
        for e in sketch.entities:
            gid = ids.get(e.id)
            if gid is None:
                continue
            if isinstance(e, SketchPoint):
                x, y = gcs.get_point(gid)
                if abs(x - e.x) >= _NEAR or abs(y - e.y) >= _NEAR:
                    e.x, e.y = float(x), float(y)
                    changed = True
            elif isinstance(e, SketchCircle):
                r = float(gcs.get_circle(gid).radius)
                if abs(r - e.radius) >= _NEAR:
                    e.radius = r
                    changed = True
            elif isinstance(e, SketchArc):
                r = float(gcs.get_arc(gid).radius)
                if abs(r - e.radius) >= _NEAR:
                    e.radius = r
                    changed = True
        return changed


def point_line_distance(sketch: SketchFeature, point: Id, line: SketchLine) -> float:
    """点から直線（線分の延長を含む）までの距離."""
    a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
    d = g.sub(b, a)
    length = math.hypot(d[0], d[1])
    p = point_pos(sketch, point)
    return abs(g.cross(d, g.sub(p, a))) / length if length > 0 else g.distance(p, a)


def measure_dimension(sketch: SketchFeature, c: SketchConstraint) -> Optional[float]:
    """寸法拘束の現在の実測値（拘束を追加するときの初期値に使う）."""
    if not is_dimension_constraint(c):
        return None
    if c.type == "distance":
        return g.distance(point_pos(sketch, c.p1), point_pos(sketch, c.p2))
    if c.type == "pointLineDistance":
        line = get_entity(sketch, c.line)
        if not isinstance(line, SketchLine):
            return None
        return point_line_distance(sketch, c.point, line)
    if c.type == "circleLineDistance":
        line, e = get_entity(sketch, c.line), get_entity(sketch, c.curve)
        if not isinstance(line, SketchLine) or not isinstance(e, (SketchCircle, SketchArc)):
            return None
        r = e.radius if isinstance(e, SketchCircle) else arc_params(sketch, e).radius
        return abs(point_line_distance(sketch, e.center, line) - r)
    if c.type in ("radius", "diameter"):
        e = get_entity(sketch, c.curve)
        if isinstance(e, SketchCircle):
            r = e.radius
        elif isinstance(e, SketchArc):
            r = arc_params(sketch, e).radius
        else:
            return None
        return r if c.type == "radius" else 2 * r
    if c.type == "angle":
        l1, l2 = get_entity(sketch, c.l1), get_entity(sketch, c.l2)
        if not isinstance(l1, SketchLine) or not isinstance(l2, SketchLine):
            return None
        d1 = g.sub(point_pos(sketch, l1.p2), point_pos(sketch, l1.p1))
        d2 = g.sub(point_pos(sketch, l2.p2), point_pos(sketch, l2.p1))
        return math.atan2(g.cross(d1, d2), g.dot(d1, d2)) / g.DEG
    return None
