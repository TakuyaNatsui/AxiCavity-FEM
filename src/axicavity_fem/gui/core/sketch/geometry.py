"""スケッチ平面上の 2D 幾何ユーティリティ（純粋関数）
（移植元: EM-CAD-py `emcad/core/sketch/geometry.py`、元は TS 版 `src/core/sketch/geometry.ts`）.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..document import Vec2

TWO_PI = math.pi * 2
DEG = math.pi / 180


def polygon_crosses_line(points: Sequence[Vec2], p: Vec2, d: Vec2, tol: float = 1e-9) -> bool:
    """多角形の頂点が直線（点 p・方向 d）の両側にあるか（直線上の点は許す）.

    回転軸が閉領域を横切っていないかの判定に使う。
    """
    neg = pos = False
    for q in points:
        s = d[0] * (q[1] - p[1]) - d[1] * (q[0] - p[0])
        if s > tol:
            pos = True
        elif s < -tol:
            neg = True
        if neg and pos:
            return True
    return False


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def length(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def perp(a: Vec2) -> Vec2:
    """反時計回りに 90° 回転."""
    return (-a[1], a[0])


def angle_of(a: Vec2) -> float:
    return math.atan2(a[1], a[0])


def normalize(a: Vec2) -> Vec2:
    l = length(a)
    return (0.0, 0.0) if l < 1e-15 else (a[0] / l, a[1] / l)


def normalize_angle(a: float) -> float:
    """[0, 2π) に正規化."""
    r = math.fmod(a, TWO_PI)
    return r + TWO_PI if r < 0 else r


def ccw_sweep(start_angle: float, end_angle: float) -> float:
    """start から end へ反時計回りに進む角度 (0, 2π]。一致する場合は全周とみなす."""
    s = normalize_angle(end_angle - start_angle)
    return TWO_PI if s < 1e-12 else s


def point_at_angle(center: Vec2, radius: float, angle: float) -> Vec2:
    return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))


def closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2:
    ab = sub(b, a)
    l2 = dot(ab, ab)
    if l2 < 1e-24:
        return a
    t = min(1.0, max(0.0, dot(sub(p, a), ab) / l2))
    return add(a, scale(ab, t))


def distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> float:
    return distance(p, closest_point_on_segment(p, a, b))


def angle_in_sweep(angle: float, start_angle: float, sweep: float) -> bool:
    return normalize_angle(angle - start_angle) <= sweep + 1e-9


def distance_to_arc(p: Vec2, center: Vec2, radius: float, start_angle: float,
                    sweep: float) -> float:
    a = angle_of(sub(p, center))
    if angle_in_sweep(a, start_angle, sweep):
        return abs(distance(p, center) - radius)
    return min(distance(p, point_at_angle(center, radius, start_angle)),
               distance(p, point_at_angle(center, radius, start_angle + sweep)))


def distance_to_circle(p: Vec2, center: Vec2, radius: float) -> float:
    return abs(distance(p, center) - radius)


def sample_arc(center: Vec2, radius: float, start_angle: float, sweep: float,
               max_step: float = 5 * DEG) -> list[Vec2]:
    """円弧を折れ線に分割する（両端点を含む）."""
    n = max(2, math.ceil(sweep / max_step))
    return [point_at_angle(center, radius, start_angle + sweep * i / n)
            for i in range(n + 1)]


def sample_circle(center: Vec2, radius: float, segments: int = 64) -> list[Vec2]:
    """円を折れ線に分割する（閉じない: 最後の点は最初の点と異なる）."""
    return [point_at_angle(center, radius, TWO_PI * i / segments)
            for i in range(segments)]


def signed_area(poly: Sequence[Vec2]) -> float:
    """符号付き面積（反時計回りで正）."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        s += cross(poly[i], poly[(i + 1) % n])
    return s / 2


def centroid(poly: Sequence[Vec2]) -> Vec2:
    n = max(1, len(poly))
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def point_in_polygon(p: Vec2, poly: Sequence[Vec2]) -> bool:
    """レイキャスティングによる内外判定（境界上は不定）."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        a, b = poly[i], poly[j]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if p[0] < x:
                inside = not inside
        j = i
    return inside


def polygon_area(poly: Sequence[Vec2]) -> float:
    """多角形の符号付き面積（反時計回りが正、靴紐公式）."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def polygon_bounds(poly: Sequence[Vec2]) -> tuple[Vec2, Vec2]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys)), (max(xs), max(ys))


def interior_point(outer: Sequence[Vec2],
                   holes: Sequence[Sequence[Vec2]] = ()) -> Vec2:
    """外周と穴で囲まれた領域の内部にある点を返す（走査線法）.

    外周と穴を合わせた偶奇規則で内側区間を探し、その中点を採用する。
    """
    lo, hi = polygon_bounds(outer)
    polygons = [outer, *holes]
    height = hi[1] - lo[1]
    eps = 1e-9 * max(1.0, hi[0] - lo[0])
    for f in (0.5, 0.37, 0.63, 0.25, 0.75, 0.13, 0.87, 0.05, 0.95):
        y = lo[1] + height * f
        xs: list[float] = []
        for poly in polygons:
            n = len(poly)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                if (a[1] > y) != (b[1] > y):
                    xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            x0, x1 = xs[i], xs[i + 1]
            if x1 - x0 > eps:
                return ((x0 + x1) / 2, y)
    return centroid(outer)


def regular_polygon(center: Vec2, vertex: Vec2, sides: int) -> list[Vec2]:
    """中心と 1 頂点から正多角形の頂点列（反時計回り）を作る."""
    r = distance(center, vertex)
    a0 = angle_of(sub(vertex, center))
    return [point_at_angle(center, r, a0 + TWO_PI * i / sides) for i in range(sides)]
