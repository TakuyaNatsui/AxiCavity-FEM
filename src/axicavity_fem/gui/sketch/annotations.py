"""拘束記号と寸法表示の配置計算（純粋関数、移植元: EM-CAD-py `emcad/sketch/annotations.py`、元は TS 版
`src/sketch/annotations.ts`）.

座標はスケッチ平面のローカル 2D。ピクセル単位の寸法は world_per_pixel で
スケッチ座標に換算する（ズームしても画面上の大きさが変わらない）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.document import (
    Id,
    SketchArc,
    SketchCircle,
    SketchConstraint,
    SketchFeature,
    SketchLine,
    SketchPoint,
    Vec2,
    is_dimension_constraint,
    is_reference,
)
from ..core.expressions import evaluate_expression
from ..core.sketch import geometry as g
from ..core.sketch.model import arc_params, get_entity, point_pos

GLYPHS = {
    "horizontal": "H", "vertical": "V", "parallel": "∥", "perpendicular": "⊥",
    "tangent": "T", "equal": "=", "concentric": "◎", "fix": "⚓", "symmetric": "S",
    "coincident": "•", "pointOnCurve": "◦",
}

_NUMBER = re.compile(r"^[-+]?\d*\.?\d+(e[-+]?\d+)?$", re.IGNORECASE)


@dataclass
class LabelAnnotation:
    text: str
    pos: Vec2
    constraintId: Id
    selected: bool
    dimension: bool


@dataclass
class Annotations:
    labels: list[LabelAnnotation] = field(default_factory=list)
    lines: list[list[Vec2]] = field(default_factory=list)     # 寸法線・引出線


def format_value(v: float) -> str:
    rounded = round(v * 1000) / 1000
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def _anchor_of(sketch: SketchFeature, id: Id) -> Optional[tuple[Vec2, Vec2]]:
    """エンティティの「記号を置く代表点」と、記号をずらす方向."""
    e = get_entity(sketch, id)
    if isinstance(e, SketchPoint):
        return (e.x, e.y), (1.0, 0.0)
    if isinstance(e, SketchLine):
        a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        n = g.normalize(g.perp(g.sub(b, a)))
        return mid, ((0.0, 1.0) if n == (0.0, 0.0) else n)
    if isinstance(e, SketchCircle):
        c = point_pos(sketch, e.center)
        d = g.normalize((1.0, 1.0))
        return g.add(c, g.scale(d, e.radius)), d
    if isinstance(e, SketchArc):
        p = arc_params(sketch, e)
        a = p.startAngle + p.sweep / 2
        return g.point_at_angle(p.center, p.radius, a), (math.cos(a), math.sin(a))
    return None


def _parallel_overlap_mid(sketch: SketchFeature, point: Id, line: SketchLine, a: Vec2, d: Vec2,
                          length2: float) -> Optional[float]:
    """点と直線の寸法で、点が直線と平行な線の端点なら（平行な 2 直線の寸法）、2 本が重なる範囲の中央の
    直線上のパラメータ t（a + t·d）。そうでなければ、または重ならなければ None."""
    if length2 <= 0:
        return None
    for e in sketch.entities:
        if not isinstance(e, SketchLine) or e.id == line.id or point not in (e.p1, e.p2):
            continue
        p1, p2 = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
        other = g.sub(p2, p1)
        norm = math.sqrt(length2) * math.hypot(other[0], other[1])
        if norm <= 0 or abs(d[0] * other[1] - d[1] * other[0]) > 1e-6 * norm:
            continue                                            # 平行でない
        t1, t2 = (g.dot(g.sub(q, a), d) / length2 for q in (p1, p2))
        lo, hi = max(0.0, min(t1, t2)), min(1.0, max(t1, t2))
        if lo <= hi:
            return (lo + hi) / 2
    return None


def _targets(c: SketchConstraint) -> list[Id]:
    t = c.type
    if t == "coincident":
        return [c.p1]
    if t in ("horizontal", "vertical"):
        return [c.line]
    if t in ("parallel", "perpendicular"):
        return [c.l1, c.l2]
    if t in ("tangent", "equal"):
        return [c.e1, c.e2]
    if t == "concentric":
        return [c.c1]
    if t == "fix":
        return [c.point]
    if t == "symmetric":
        return [c.p1, c.p2]
    if t == "pointOnCurve":
        return [c.point]
    return []


def _extend_arc_to(out: Annotations, sketch: SketchFeature, arc: SketchArc, angle: float) -> None:
    """寸法の起点（円周上の角度 angle）が円弧の外なら、近いほうの端から起点まで円周を延ばして見せる."""
    p = arc_params(sketch, arc)
    rel = g.normalize_angle(angle - p.startAngle)
    if rel <= p.sweep + 1e-9:
        return
    after_end, before_start = rel - p.sweep, 2 * math.pi - rel
    if after_end <= before_start:
        start, span = p.startAngle + p.sweep, after_end
    else:
        start, span = angle, before_start
    n = max(2, int(span / (math.pi / 36)) + 1)
    out.lines.append([g.point_at_angle(p.center, p.radius, start + span * i / (n - 1)) for i in range(n)])


def build_annotations(sketch: SketchFeature, params: dict[str, float],
                      selected_id: Optional[Id], world_per_pixel: float) -> Annotations:
    out = Annotations()
    px = lambda n: n * world_per_pixel      # noqa: E731
    stack: dict[Id, int] = {}               # 同じエンティティに複数の記号が付く場合にずらす

    def glyph_at(entity_id: Id, text: str, constraint_id: Id) -> None:
        anchor = _anchor_of(sketch, entity_id)
        if anchor is None:
            return
        pos0, direction = anchor
        n = stack.get(entity_id, 0)
        stack[entity_id] = n + 1
        along = g.perp(direction)
        pos = g.add(g.add(pos0, g.scale(direction, px(12))), g.scale(along, px(14 * n)))
        out.labels.append(LabelAnnotation(text, pos, constraint_id,
                                          constraint_id == selected_id, False))

    for c in sketch.constraints:
        if not is_dimension_constraint(c):
            glyph = GLYPHS.get(c.type, "?")
            for id in _targets(c):
                glyph_at(id, glyph, c.id)
            continue
        try:
            value_text = format_value(evaluate_expression(c.value, params))
        except Exception:
            value_text = c.value
        is_expression = not _NUMBER.match(c.value.strip())
        text = f"{c.value} = {value_text}" if is_expression else value_text
        selected = c.id == selected_id

        if c.type == "distance":
            a, b = point_pos(sketch, c.p1), point_pos(sketch, c.p2)
            d = g.sub(b, a)
            n = g.normalize(g.perp(d))
            off = g.scale((0.0, 1.0) if n == (0.0, 0.0) else n, px(18))
            a2, b2 = g.add(a, off), g.add(b, off)
            out.lines.append([a, g.add(a2, g.scale(off, 0.2))])
            out.lines.append([b, g.add(b2, g.scale(off, 0.2))])
            out.lines.append([a2, b2])
            out.labels.append(LabelAnnotation(
                text, g.add(((a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2), g.scale(off, 0.45)),
                c.id, selected, True))
        elif c.type == "pointLineDistance":
            # 点から直線への垂線（足が線分の外なら、近い端点から足まで直線を延ばして見せる）
            line = get_entity(sketch, c.line)
            if not isinstance(line, SketchLine):
                continue
            p = point_pos(sketch, c.point)
            a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
            d = g.sub(b, a)
            length2 = d[0] * d[0] + d[1] * d[1]
            t = g.dot(g.sub(p, a), d) / length2 if length2 > 0 else 0.0
            foot = g.add(a, g.scale(d, t))
            # 平行な 2 直線の寸法（点が、直線と平行な線の端点）は、2 本が重なる範囲の中央に描く（端点に描くと
            # 長方形の辺などに重なって見分けにくい。距離はどこでも同じ）
            mid = _parallel_overlap_mid(sketch, c.point, line, a, d, length2)
            if mid is not None:
                offset = g.sub(p, foot)
                foot = g.add(a, g.scale(d, mid))
                p, t = g.add(foot, offset), mid
            if (t < 0 or t > 1) and not is_reference(line):      # 参照の軸は無限の直線（延長は描かない）
                out.lines.append([a if t < 0 else b, foot])
            out.lines.append([p, foot])
            along = g.normalize(d) if length2 > 0 else (1.0, 0.0)
            out.labels.append(LabelAnnotation(
                text, g.add(((p[0] + foot[0]) / 2, (p[1] + foot[1]) / 2), g.scale(along, px(14))),
                c.id, selected, True))
        elif c.type == "circleLineDistance":
            # 円周上で直線にいちばん近い点（中心から直線への垂線と円の交点）から直線への垂線
            line, e = get_entity(sketch, c.line), get_entity(sketch, c.curve)
            if not isinstance(line, SketchLine) or not isinstance(e, (SketchCircle, SketchArc)):
                continue
            center = point_pos(sketch, e.center)
            radius = e.radius if isinstance(e, SketchCircle) else arc_params(sketch, e).radius
            a, b = point_pos(sketch, line.p1), point_pos(sketch, line.p2)
            d = g.sub(b, a)
            length2 = d[0] * d[0] + d[1] * d[1]
            t = g.dot(g.sub(center, a), d) / length2 if length2 > 0 else 0.0
            foot = g.add(a, g.scale(d, t))
            toward = g.normalize(g.sub(foot, center))
            if toward == (0.0, 0.0):                          # 中心が直線の上
                toward = g.normalize(g.perp(d)) if length2 > 0 else (0.0, 1.0)
            on_circle = g.add(center, g.scale(toward, radius))
            if (t < 0 or t > 1) and not is_reference(line):      # 参照の軸は無限の直線（延長は描かない）
                out.lines.append([a if t < 0 else b, foot])
            if isinstance(e, SketchArc):
                _extend_arc_to(out, sketch, e, g.angle_of(toward))
            out.lines.append([on_circle, foot])
            along = g.normalize(d) if length2 > 0 else (1.0, 0.0)
            out.labels.append(LabelAnnotation(
                text, g.add(((on_circle[0] + foot[0]) / 2, (on_circle[1] + foot[1]) / 2), g.scale(along, px(14))),
                c.id, selected, True))
        elif c.type in ("radius", "diameter"):
            e = get_entity(sketch, c.curve)
            if not isinstance(e, (SketchCircle, SketchArc)):
                continue
            center = point_pos(sketch, e.center)
            if isinstance(e, SketchCircle):
                radius, angle = e.radius, math.pi / 4
            else:
                p = arc_params(sketch, e)
                radius, angle = p.radius, p.startAngle + p.sweep / 2
            on_curve = g.point_at_angle(center, radius, angle)
            direction = (math.cos(angle), math.sin(angle))
            label_pos = g.add(on_curve, g.scale(direction, px(16)))
            start = center if c.type == "radius" else g.point_at_angle(center, radius, angle + math.pi)
            out.lines.append([start, label_pos])
            out.labels.append(LabelAnnotation(("R" if c.type == "radius" else "⌀") + text,
                                              label_pos, c.id, selected, True))
        elif c.type == "angle":
            l1, l2 = get_entity(sketch, c.l1), get_entity(sketch, c.l2)
            if not isinstance(l1, SketchLine) or not isinstance(l2, SketchLine):
                continue
            a1, b1 = point_pos(sketch, l1.p1), point_pos(sketch, l1.p2)
            a2, b2 = point_pos(sketch, l2.p1), point_pos(sketch, l2.p2)
            d1, d2 = g.sub(b1, a1), g.sub(b2, a2)
            den = g.cross(d1, d2)
            if abs(den) < 1e-12:
                vertex = ((a1[0] + b1[0] + a2[0] + b2[0]) / 4, (a1[1] + b1[1] + a2[1] + b2[1]) / 4)
            else:
                t = g.cross(g.sub(a2, a1), d2) / den
                vertex = g.add(a1, g.scale(d1, t))
            bis = g.normalize(g.add(g.normalize(d1), g.normalize(d2)))
            label_pos = g.add(vertex, g.scale((1.0, 0.0) if bis == (0.0, 0.0) else bis, px(28)))
            out.lines.append([vertex, label_pos])
            out.labels.append(LabelAnnotation(f"{text}°", label_pos, c.id, selected, True))
    return out
