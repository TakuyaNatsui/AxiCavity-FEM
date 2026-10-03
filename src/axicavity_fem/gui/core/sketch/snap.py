"""スナップ（純粋関数）。優先順位: 既存点 > 線の中点 > 水平/垂直整列 + グリッド
（移植元: EM-CAD-py `emcad/core/sketch/snap.py`、元は TS 版 `src/core/sketch/snap.ts`）.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..document import REFERENCE_R_END, REFERENCE_Z_END, Id, SketchFeature, SketchLine, SketchPoint, Vec2, is_reference
from .model import point_pos

_HIDDEN = (REFERENCE_Z_END, REFERENCE_R_END)       # 軸を決めるためだけの点（スナップしない）


@dataclass(frozen=True)
class SnapResult:
    pos: Vec2
    kind: str                          # point / midpoint / align / grid / none
    pointId: Optional[Id] = None       # 既存点にスナップした場合はその ID
    alignX: Optional[float] = None     # 整列ガイド（基準点と同じ x / y）
    alignY: Optional[float] = None


def snap_point(sketch: SketchFeature, raw: Vec2, tolerance: float,
               grid_spacing: Optional[float], reference: Optional[Vec2] = None,
               exclude: Optional[set[Id]] = None) -> SnapResult:
    """Args:
        tolerance: スケッチ座標系での許容距離。
        grid_spacing: None ならグリッドスナップ無し。
        reference: 水平/垂直整列の基準点（作図中の直前の点など）。
        exclude: スナップ対象から除外する点（ドラッグ中の点など）。
    """
    best_point = None
    for e in sketch.entities:
        if not isinstance(e, SketchPoint) or (exclude and e.id in exclude) or e.id in _HIDDEN:
            continue
        d = math.hypot(e.x - raw[0], e.y - raw[1])
        if d <= tolerance and (best_point is None or d < best_point[2]):
            best_point = (e.id, (e.x, e.y), d)
    if best_point is not None:
        return SnapResult(pos=best_point[1], kind="point", pointId=best_point[0])

    best_mid = None
    for e in sketch.entities:
        if not isinstance(e, SketchLine) or is_reference(e):    # 軸の「中点」には吸着しない
            continue
        if exclude and (e.p1 in exclude or e.p2 in exclude):
            continue
        a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        d = math.hypot(mid[0] - raw[0], mid[1] - raw[1])
        if d <= tolerance and (best_mid is None or d < best_mid[1]):
            best_mid = (mid, d)
    if best_mid is not None:
        return SnapResult(pos=best_mid[0], kind="midpoint")

    x, y = raw
    kind = "none"
    if grid_spacing and grid_spacing > 0:
        # JS の Math.round と同じ「0.5 は切り上げ」（Python の round は偶数丸め）
        x = math.floor(x / grid_spacing + 0.5) * grid_spacing
        y = math.floor(y / grid_spacing + 0.5) * grid_spacing
        kind = "grid"
    align_x = align_y = None
    if reference is not None:
        if abs(raw[0] - reference[0]) <= tolerance:
            align_x = x = reference[0]
            kind = "align"
        if abs(raw[1] - reference[1]) <= tolerance:
            align_y = y = reference[1]
            kind = "align"
    return SnapResult(pos=(x, y), kind=kind, alignX=align_x, alignY=align_y)


_GRID_CANDIDATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100,
                    200, 500, 1000)


def choose_grid_spacing(world_per_pixel: float, min_pixels: float = 20) -> float:
    """画面上でおよそ min_pixels 以上の間隔になる 1-2-5 系列のグリッド間隔."""
    target = world_per_pixel * min_pixels
    for c in _GRID_CANDIDATES:
        if c >= target:
            return float(c)
    return 1000.0
