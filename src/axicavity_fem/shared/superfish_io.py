"""Superfish (.af) 入出力 (GUI 非依存)。

ver2.2 まで旧 GUI (単一領域エディタ) にハードコードされていた Superfish
インポート/エクスポートを、MultiRegionGeometry ベースの純関数として
shared 層に移植したもの。数値挙動 (単位換算・劣弧選択・閉合判定) は
旧実装と同一。

Superfish は Poisson/Superfish (LANL) の 2D 空洞計算コードで、境界を
$po 行 (点列) で記述する .af ファイルを入力に取る。内部単位は cm。

対応書式 (旧実装と同一):
    直線:  $po x=<z>, y=<r> $
    円弧:  $po nt=2, r=<radius>, theta=<終点角 deg>, x0=<cz>, y0=<cr> $

制約:
    - .af は単一領域境界のみ表現できるため、エクスポートは
      「Region が 1 つ・穴なし・閉ループ」の場合のみ許可する
      (validate_superfish_exportable で事前チェック)。
    - BC 情報は .af に対応概念がなく保存されない。インポート時は全 segment
      が DEFAULT_BC ("PEC") になる。
    - 円弧は劣弧 (<=180 度) として解釈する。ちょうど 180 度は向きが
      曖昧なため非推奨 (Multi-Region Editor 自体も劣弧のみ生成する)。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .multi_region_model import (
    DEFAULT_BC,
    Loop,
    MultiRegionGeometry,
    Segment,
)


# Superfish の内部単位は cm。unit -> cm の換算係数 (エクスポートで乗算、
# インポートでは逆数を使う)。旧実装の換算表と数値一致。
CM_PER_UNIT = {"m": 100.0, "cm": 1.0, "mm": 0.1, "inch": 2.54}


class SuperfishImportError(ValueError):
    """Superfish (.af) の読み込みに失敗した。"""


class SuperfishExportError(ValueError):
    """Superfish (.af) へ出力できない幾何、または書き込み失敗。"""


# $po 行のパターン (旧実装と同一)
_LINE_RE = re.compile(
    r"^\$po\s+x=([+\-\d.eE]+)\s*,\s*y=([+\-\d.eE]+)\s*\$", re.IGNORECASE
)
_ARC_RE = re.compile(
    r"^\$po\s+nt=2\s*,\s*r=([+\-\d.eE]+)\s*,\s*theta=([+\-\d.eE]+)\s*,"
    r"\s*x0=([+\-\d.eE]+)\s*,\s*y0=([+\-\d.eE]+)\s*\$",
    re.IGNORECASE,
)


def _points_close(a: tuple[float, float], b: tuple[float, float],
                  atol: float, rtol: float = 1e-5) -> bool:
    """np.allclose(a, b, atol=atol) と同値の 2 点比較 (numpy 非依存)。"""
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# インポート
# ---------------------------------------------------------------------------
def import_superfish(path: str | Path, unit: str = "mm") -> MultiRegionGeometry:
    """Superfish (.af) を読み込んで MultiRegionGeometry を返す。

    Args:
        path: .af ファイルパス。
        unit: 変換先の単位 (GUI の Unit 選択)。ファイル内の cm 値を
            この単位へ換算する。

    Returns:
        閉ループを検出した場合は 1 Loop + Vacuum 1 Region を持つ幾何
        (from_legacy_single_loop 経由)。閉じていない場合は loops/regions が
        空の幾何 (GUI 側で Close Loop を促す)。

    Raises:
        SuperfishImportError: 単位が未対応・ファイル読込失敗・有効な
            $po 行が 1 つも無い場合。
    """
    if unit not in CM_PER_UNIT:
        raise SuperfishImportError(
            f"未対応の単位です: {unit!r} (対応: {tuple(CM_PER_UNIT)})"
        )
    scale = 1.0 / CM_PER_UNIT[unit]  # cm -> unit

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise SuperfishImportError(f"ファイルを読み込めません: {e}") from e

    points: list[tuple[float, float]] = []
    seg_dicts: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.lower().startswith("$po"):
            continue
        lm = _LINE_RE.match(line)
        am = _ARC_RE.match(line)
        # 最初の点が確定するまで円弧行は無視する (旧実装と同一)
        if not points and lm:
            points.append((float(lm.group(1)) * scale,
                           float(lm.group(2)) * scale))
            continue
        if not points:
            continue
        p1_idx = len(points) - 1
        p1 = points[p1_idx]
        if am:
            # --- 円弧: 劣弧選択 (旧実装と同一のロジック) ---
            r_val = float(am.group(1)) * scale
            theta_deg_end = float(am.group(2))
            cx = float(am.group(3)) * scale
            cy = float(am.group(4)) * scale
            # 終点の座標
            tr_end = math.radians(theta_deg_end)
            px = cx + r_val * math.cos(tr_end)
            py = cy + r_val * math.sin(tr_end)
            # 始点角 a1 と終点角 a2 の差を (-180, 180] に正規化し、
            # 短い方の弧を theta1 < theta2 (matplotlib の CCW 規約) で持つ
            a1 = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
            a2 = theta_deg_end
            da = a2 - a1
            while da <= -180.0:
                da += 360.0
            while da > 180.0:
                da -= 360.0
            if da >= 0:
                theta1, theta2 = a1, a1 + da
            else:
                theta1, theta2 = a2, a2 + abs(da)
            if theta2 <= theta1:
                theta2 += 360.0
            points.append((px, py))
            seg_dicts.append({
                "type": "arc",
                "points": [p1_idx, len(points) - 1],
                "physical_name": DEFAULT_BC,
                "center": [cx, cy],
                "radius": r_val,
                "theta1": theta1,
                "theta2": theta2,
            })
        elif lm:
            px = float(lm.group(1)) * scale
            py = float(lm.group(2)) * scale
            points.append((px, py))
            seg_dicts.append({
                "type": "line",
                "points": [p1_idx, len(points) - 1],
                "physical_name": DEFAULT_BC,
            })

    if not points:
        raise SuperfishImportError(
            "有効な $po 行が見つかりませんでした。Superfish (.af) 形式か確認してください。"
        )

    # 閉ループ判定 (旧実装の np.allclose(atol=1e-7*scale) と同値)
    closed = False
    if len(points) > 2 and seg_dicts:
        if _points_close(points[0], points[-1], atol=1e-7 * scale):
            seg_dicts[-1]["points"][1] = 0
            points.pop()
            closed = True

    if closed:
        # テスト済みの legacy 変換 (1 Loop + Vacuum Region 自動生成) を再利用
        legacy = {
            "points": [list(p) for p in points],
            "segments": seg_dicts,
            "loop_closed": True,
            "settings": {"unit": unit},
        }
        return MultiRegionGeometry.from_legacy_single_loop(legacy)

    # 非閉合: loops/regions を作らず返す (GUI 側で Close Loop を促す)
    segments: list[Segment] = []
    for i, s in enumerate(seg_dicts):
        kw: dict[str, Any] = {
            "id": i,
            "type": s["type"],
            "point_indices": list(s["points"]),
            "bc_name": DEFAULT_BC,
        }
        if s["type"] == "arc":
            kw.update(center=tuple(s["center"]), radius=s["radius"],
                      theta1=s["theta1"], theta2=s["theta2"])
        segments.append(Segment(**kw))
    return MultiRegionGeometry(points=points, segments=segments,
                               loops=[], regions=[], unit=unit)


# ---------------------------------------------------------------------------
# エクスポート
# ---------------------------------------------------------------------------
def _walk_loop_oriented(geom: MultiRegionGeometry,
                        loop: Loop) -> list[tuple[Segment, bool]]:
    """ループの segment 列を「現在点始まり」の向きに解決して返す。

    MR の Loop は逆向き接続 (point_indices が [p2, p1] 順) を許容するため、
    各 segment を (segment, is_reversed) のタプルで返す。先頭 segment の
    向きは 2 番目の segment と共有する端点で確定する。

    Raises:
        SuperfishExportError: ループが連結していない、または閉じていない。
    """
    segs = [geom.segment_by_id(sid) for sid in loop.segment_ids]
    if len(segs) < 2:
        raise SuperfishExportError("ループの segment 数が不足しています。")
    a0, b0 = segs[0].point_indices
    second = set(segs[1].point_indices)
    if b0 in second:
        first_reversed = False
    elif a0 in second:
        first_reversed = True
    else:
        raise SuperfishExportError(
            f"ループが連結していません (segment id={segs[0].id} と "
            f"id={segs[1].id} が端点を共有していません)。"
        )
    oriented: list[tuple[Segment, bool]] = [(segs[0], first_reversed)]
    start = b0 if first_reversed else a0
    current = a0 if first_reversed else b0
    for seg in segs[1:]:
        pa, pb = seg.point_indices
        if pa == current:
            oriented.append((seg, False))
            current = pb
        elif pb == current:
            oriented.append((seg, True))
            current = pa
        else:
            raise SuperfishExportError(
                f"ループが連結していません (segment id={seg.id})。"
            )
    if current != start:
        raise SuperfishExportError("ループが閉じていません。")
    return oriented


def validate_superfish_exportable(geom: MultiRegionGeometry) -> list[str]:
    """Superfish (.af) に出力可能か検証し、問題を文字列リストで返す。

    空リストなら出力可能。条件: 単位が対応表にある・Region がちょうど 1 つ・
    穴なし・外周ループが連結した閉ループ・segment 数 3 以上。
    """
    errors: list[str] = []
    if geom.unit not in CM_PER_UNIT:
        errors.append(
            f"未対応の単位です: {geom.unit!r} (対応: {tuple(CM_PER_UNIT)})"
        )
    if len(geom.regions) != 1:
        errors.append(
            f"Region が {len(geom.regions)} 個あります。"
            "Superfish (.af) 出力は単一領域のみ対応です。"
        )
        return errors
    region = geom.regions[0]
    if region.hole_loop_ids:
        errors.append("穴 (hole) 付き領域は Superfish (.af) に出力できません。")
    try:
        loop = geom.loop_by_id(region.outer_loop_id)
    except KeyError:
        errors.append(f"外周ループ (id={region.outer_loop_id}) が見つかりません。")
        return errors
    if len(loop.segment_ids) < 3:
        errors.append("ループの segment 数が 3 未満です。")
        return errors
    try:
        _walk_loop_oriented(geom, loop)
    except SuperfishExportError as e:
        errors.append(str(e))
    return errors


def export_superfish(geom: MultiRegionGeometry, path: str | Path,
                     mesh_size: float | None = None,
                     freq_mhz: float = 2856.0) -> None:
    """MultiRegionGeometry を Superfish (.af) に書き出す。

    出力書式 (ヘッダ・$po 行) は旧実装と同一。座標は geom.unit から cm へ
    換算する。開始点はループ walk の始点 (loop.segment_ids[0] の解決済み
    始点) であり、旧実装 (常に points[0]) とファイル上の開始点が異なる
    ことがあるが、表す形状は同一。

    Args:
        geom: 単一領域 (穴なし・閉ループ) の幾何。
        path: 出力先 .af パス。
        mesh_size: $reg 行の dx (単位は geom.unit)。None なら geom.mesh_size。
        freq_mhz: $reg 行の freq (MHz)。既定は旧実装と同じ 2856.0。

    Raises:
        SuperfishExportError: validate_superfish_exportable が問題を返した
            場合、または書き込みに失敗した場合。
    """
    errors = validate_superfish_exportable(geom)
    if errors:
        raise SuperfishExportError("\n".join(errors))

    scale = CM_PER_UNIT[geom.unit]  # unit -> cm
    if mesh_size is None:
        mesh_size = geom.mesh_size
    region = geom.regions[0]
    loop = geom.loop_by_id(region.outer_loop_id)
    oriented = _walk_loop_oriented(geom, loop)

    # walk 順の頂点列 (始点 + 各 segment の終点)。xdri/ydri 算出用。
    first_seg, first_rev = oriented[0]
    start_idx = first_seg.point_indices[1] if first_rev else first_seg.point_indices[0]
    vert_indices = [start_idx]
    for seg, rev in oriented:
        vert_indices.append(seg.point_indices[0] if rev else seg.point_indices[1])
    xs = [geom.points[i][0] * scale for i in vert_indices]
    ys = [geom.points[i][1] * scale for i in vert_indices]
    xdri = (min(xs) + max(xs)) / 2.0
    ydri = (min(ys) + max(ys)) / 2.0

    lines = ["Gmsh2Fish"]
    lines.append(
        f"$reg kprob=1, dx={mesh_size * scale:.6e}, "
        f"xdri={xdri:.6e}, ydri={ydri:.6e}, "
        f"nbsup=1, nbslo=0, nbslf=0, nbsrt=0, "
        f"freq={freq_mhz}, kmethod=1, beta=1.0 $"
    )
    lines.append("")
    z0, r0 = geom.points[start_idx]
    lines.append(f"$po x={z0 * scale:.6e}, y={r0 * scale:.6e} $")
    for seg, rev in oriented:
        end_idx = seg.point_indices[0] if rev else seg.point_indices[1]
        ez, er = geom.points[end_idx]
        if seg.type == "line":
            lines.append(f"$po x={ez * scale:.6e}, y={er * scale:.6e} $")
        else:
            cz, cr = seg.center
            th = math.degrees(math.atan2(er - cr, ez - cz))
            lines.append(
                f"$po nt=2, r={seg.radius * scale:.6e}, theta={th:.6e}, "
                f"x0={cz * scale:.6e}, y0={cr * scale:.6e} $"
            )

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        raise SuperfishExportError(f"ファイルを書き込めません: {e}") from e
