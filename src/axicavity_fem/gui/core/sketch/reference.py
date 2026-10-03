"""参照ジオメトリ（原点・z 軸・r 軸）と、点を軸に拘束するための関数（純 Python）.

スケッチには常に次の要素がある（``projection = REFERENCE_PROJECTION``、構築ジオメトリ扱い）:

- 原点 ``ref-origin``（0, 0）: 見えて、スナップ・選択できる。曲線の端点として共有できる（共有すると端点は原点に固定される）
- z 軸 ``ref-z-axis``（直線 r = 0）と r 軸 ``ref-r-axis``（直線 z = 0）: 画面では無限の直線。拘束ツールで選べる
- 軸のもう一方の端点（(1, 0) / (0, 1)）: 直線を決めるためだけの点。見えない・選べない・スナップしない

planegcs の「点を直線上に」は無限の直線として効くので、軸の線分が短くても軸上のどこでも拘束できる。
参照要素はソルバーで固定点（EM-CAD-py の投影と同じ扱い）、閉領域・変換・メッシュ・ハッシュには入らない。

「r=0 に固定」は点に「点を曲線上に（z 軸）」、「z=0 に固定」は「点を曲線上に（r 軸）」の拘束を付ける。作図で新しく
できた点が軸の上にあれば同じ拘束を自動で付ける（:func:`axis_constraints_for`）。
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..document import (
    REFERENCE_ORIGIN,
    REFERENCE_PROJECTION,
    REFERENCE_R_AXIS,
    REFERENCE_R_END,
    REFERENCE_Z_AXIS,
    REFERENCE_Z_END,
    Id,
    SketchConstraint,
    SketchFeature,
    SketchLine,
    SketchPoint,
    is_reference,
    new_id,
)
from . import geometry as g

REFERENCE_IDS = (REFERENCE_ORIGIN, REFERENCE_Z_END, REFERENCE_R_END, REFERENCE_Z_AXIS, REFERENCE_R_AXIS)
HIDDEN_REFERENCE_POINTS = frozenset({REFERENCE_Z_END, REFERENCE_R_END})
REFERENCE_LINES = (REFERENCE_Z_AXIS, REFERENCE_R_AXIS)
# 座標（"r" = r を 0 に = z 軸の上、"z" = z を 0 に = r 軸の上）→ 軸
AXIS_OF = {"r": REFERENCE_Z_AXIS, "z": REFERENCE_R_AXIS}


def _reference_entities() -> list:
    return [
        SketchPoint(id=REFERENCE_ORIGIN, x=0.0, y=0.0, construction=True, projection=REFERENCE_PROJECTION),
        SketchPoint(id=REFERENCE_Z_END, x=1.0, y=0.0, construction=True, projection=REFERENCE_PROJECTION),
        SketchPoint(id=REFERENCE_R_END, x=0.0, y=1.0, construction=True, projection=REFERENCE_PROJECTION),
        SketchLine(id=REFERENCE_Z_AXIS, p1=REFERENCE_ORIGIN, p2=REFERENCE_Z_END, construction=True,
                   projection=REFERENCE_PROJECTION),
        SketchLine(id=REFERENCE_R_AXIS, p1=REFERENCE_ORIGIN, p2=REFERENCE_R_END, construction=True,
                   projection=REFERENCE_PROJECTION),
    ]


def ensure_reference_geometry(sketch: SketchFeature) -> bool:
    """参照要素が無ければ先頭に足す（あれば位置と印を正す）. 変えたら True."""
    wanted = {e.id: e for e in _reference_entities()}
    present = {e.id: e for e in sketch.entities if e.id in wanted}
    changed = False
    for eid, e in present.items():
        ref = wanted[eid]
        if isinstance(ref, SketchPoint) and ((e.x, e.y) != (ref.x, ref.y) or e.xExpr or e.yExpr):
            e.x, e.y, e.xExpr, e.yExpr = ref.x, ref.y, None, None
            changed = True
        if e.projection != REFERENCE_PROJECTION or not e.construction:
            e.projection, e.construction = REFERENCE_PROJECTION, True
            changed = True
    missing = [e for eid, e in wanted.items() if eid not in present]
    if missing:
        sketch.entities[:0] = missing
        changed = True
    return changed


def user_entities(sketch: SketchFeature) -> list:
    """参照要素を除いたエンティティ."""
    return [e for e in sketch.entities if not is_reference(e)]


def user_points(sketch: SketchFeature) -> list[SketchPoint]:
    return [e for e in sketch.entities if isinstance(e, SketchPoint) and not is_reference(e)]


def is_hidden_reference_point(entity_or_id) -> bool:
    eid = entity_or_id if isinstance(entity_or_id, str) else getattr(entity_or_id, "id", None)
    return eid in HIDDEN_REFERENCE_POINTS


def on_axis(sketch: SketchFeature, point_id: Id, coord: str) -> bool:
    """点が既にその軸に拘束されているか（原点そのもの、または「点を曲線上に」がある）."""
    if point_id == REFERENCE_ORIGIN:
        return True
    axis = AXIS_OF[coord]
    return any(c.type == "pointOnCurve" and c.point == point_id and c.curve == axis for c in sketch.constraints)


def axis_constraint(point_id: Id, coord: str) -> SketchConstraint:
    return SketchConstraint(id=new_id(), type="pointOnCurve", point=point_id, curve=AXIS_OF[coord])


def implied_redundant(sketch: SketchFeature, constraints: Iterable[SketchConstraint]) -> list[Id]:
    """新しい軸の拘束で冗長になる水平 / 垂直拘束（両端が z 軸の上の線の「水平」、r 軸の上の線の「垂直」）の ID."""
    new = list(constraints)
    on = {"r": set(), "z": set()}
    for coord in ("r", "z"):
        on[coord].add(REFERENCE_ORIGIN)
        for c in [*sketch.constraints, *new]:
            if c.type == "pointOnCurve" and c.curve == AXIS_OF[coord]:
                on[coord].add(c.point)
    lines = {e.id: e for e in sketch.entities if isinstance(e, SketchLine) and not is_reference(e)}
    drop = []
    for c in sketch.constraints:
        line = lines.get(c.line) if c.type in ("horizontal", "vertical") else None
        if line is None:
            continue
        coord = "r" if c.type == "horizontal" else "z"
        if line.p1 in on[coord] and line.p2 in on[coord]:
            drop.append(c.id)
    return drop


def axis_constraints_for(sketch: SketchFeature, point_ids: Iterable[Id], tolerance: float) -> list[SketchConstraint]:
    """点のうち軸の上（|r| ≤ tol なら z 軸、|z| ≤ tol なら r 軸）にあって、まだ拘束されていないものへの拘束.

    参照要素・原点・その座標に式を持つ点は対象外（式は式で決まる）。
    """
    by_id = {e.id: e for e in sketch.entities if isinstance(e, SketchPoint)}
    out: list[SketchConstraint] = []
    for pid in point_ids:
        p = by_id.get(pid)
        if p is None or is_reference(p):
            continue
        if abs(p.y) <= tolerance and not p.yExpr and not on_axis(sketch, pid, "r"):
            out.append(axis_constraint(pid, "r"))
        if abs(p.x) <= tolerance and not p.xExpr and not on_axis(sketch, pid, "z"):
            out.append(axis_constraint(pid, "z"))
    return out


def reference_label_key(entity_id: Id) -> Optional[str]:
    """参照要素の表示名の i18n キー（参照要素でなければ None）."""
    return {REFERENCE_ORIGIN: "reference.origin", REFERENCE_Z_AXIS: "reference.zAxis",
            REFERENCE_R_AXIS: "reference.rAxis"}.get(entity_id)


def infinite_line_distance(p, a, b) -> float:
    """点 p から、a と b を通る無限の直線までの距離."""
    d = g.sub(b, a)
    length = g.length(d)
    if length < 1e-15:
        return g.distance(p, a)
    return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / length


def entity_numbers(sketch: SketchFeature) -> dict[Id, str]:
    """画面の点番号と同じ規則のラベル: 点は "P1"…、曲線は "線 1" などではなく種別ごとの番号 ("L1", "A1", "C1")."""
    counters = {"point": 0, "line": 0, "arc": 0, "circle": 0}
    prefix = {"point": "P", "line": "L", "arc": "A", "circle": "C"}
    out: dict[Id, str] = {}
    for e in sketch.entities:
        if is_reference(e):
            continue
        counters[e.type] += 1
        out[e.id] = f"{prefix[e.type]}{counters[e.type]}"
    return out
