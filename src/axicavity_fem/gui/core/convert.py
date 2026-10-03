"""スケッチ + 設定 ⇄ ver2.3 の ``MultiRegionGeometry``（純 Python）.

ver3 の GUI は「閉領域 = 材料領域、曲線 = 境界条件」のスケッチを編集し、メッシュ生成・Superfish・
旧 ``.gmshproj``・GEO / Python 出力はすべて ver2.3 のコア（``shared/multi_region_model.py`` と
``shared/gmsh_export_occ.py``、無変更）に、この変換層を通して渡す。

順変換 :func:`to_multi_region`（計画 §6.1）:
    1. パラメータを評価し、式付きの点座標を確定する
    2. :func:`detect_profiles` で閉領域を求める。閉領域の境界に使われる曲線だけを書き出す
    3. 点プール（許容差内の点は統合）→ Segment（線 / 円弧。180° 以上の弧は 2 分割、円は 90° の弧 4 本）
    4. Loop（閉領域の外周と穴。``LoopSegment.reversed`` なら副セグメント列を逆順に。OCC が向きを補正する）
    5. Region（閉領域ごとの材料設定。無ければ既定の Vacuum）
    6. 境界条件は :func:`effective_bc`（明示指定 → 内部界面 None → 軸 None → 既定 PEC）
逆変換 :func:`from_multi_region`（§6.2）は旧形式の取り込みに使う。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from ...shared.multi_region_model import (
    BC_NONE,
    DEFAULT_BC,
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)
from . import document as d
from .document import AxiDocument, Id, RegionSetting, SketchArc, SketchCircle, SketchLine, SketchPoint, Vec2, is_reference
from .expressions import ExpressionError, evaluate_expression, param_results, resolve_params
from .sketch import geometry as g
from .sketch.model import arc_params, create_sketch_feature, get_entity, point_pos
from .sketch.profiles import Profile, detect_profiles, profile_at
from .sketch.planarize import count_conflicts

# コアの軸検出しきい値（shared.mesh_loader / fem_tm0.boundary: r < 1e-6 m）を文書単位に換算して使う
AXIS_TOL_M = 1e-6
UNIT_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "inch": 0.0254}
# BC 分類と衝突する材料タグ（大文字小文字を無視）
_RESERVED_TAGS = {"pec", "e-short", "e_short", "eshort", "m-short", "m_short", "mshort", "dirichlet", "none"}
_TAG_RE = re.compile(r"[^a-z0-9_]+")
_DEG = 180.0 / math.pi
_ARC_SAMPLE_STEP = 5 * g.DEG
# 外周エンティティ集合の類似度（Jaccard）でも領域設定を対応付ける下限。4 辺の領域の 1 辺を分割すると
# 3/6 = 0.5 になるので「以上」で拾う
_JACCARD_MIN = 0.5


def axis_tolerance(units: str) -> float:
    """軸 r=0 とみなす |r| のしきい値（文書単位）."""
    return AXIS_TOL_M / UNIT_TO_M.get(units, 1e-3)


def normalize_material_tag(name: str) -> str:
    """材料タグを ``[a-z0-9_]`` に正規化する（gmsh の Physical Surface 名 / materials.json のキー）."""
    tag = _TAG_RE.sub("_", (name or "").strip().lower()).strip("_")
    if not tag:
        tag = "region"
    if tag in _RESERVED_TAGS or tag.replace("_", "-") in _RESERVED_TAGS:
        tag += "_mat"
    return tag


# ---------------------------------------------------------------------------
# 検証の結果
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    level: str                 # "error" / "warning"
    code: str
    message: str
    entity_ids: list = field(default_factory=list)


class ConvertError(ValueError):
    """書き出せない（error レベルの Issue がある）."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(i.message for i in issues if i.level == "error") or "変換できません")


# ---------------------------------------------------------------------------
# 式の適用
# ---------------------------------------------------------------------------

def document_params(doc: AxiDocument) -> dict[str, float]:
    """パラメータの評価（評価できない行は飛ばす）."""
    return resolve_params(doc.params, strict=False)


def apply_point_expressions(doc: AxiDocument, params: Optional[dict[str, float]] = None) -> list[str]:
    """式付きの点座標をパラメータで評価して座標に書き戻す。評価できなかった式のメッセージを返す."""
    params = document_params(doc) if params is None else params
    errors: list[str] = []
    for e in doc.sketch.entities:
        if not isinstance(e, SketchPoint):
            continue
        for attr, expr in (("x", e.xExpr), ("y", e.yExpr)):
            if not expr:
                continue
            try:
                setattr(e, attr, float(evaluate_expression(expr, params)))
            except ExpressionError as exc:
                errors.append(f"点の {'Z' if attr == 'x' else 'R'} の式 '{expr}': {exc}")
    return errors


def apply_setting_expressions(doc: AxiDocument, params: Optional[dict[str, float]] = None) -> list[str]:
    """式付きの設定（メッシュサイズ、領域の εr・tanδ）をパラメータで評価して値に書き戻す。エラーの一覧を返す."""
    params = document_params(doc) if params is None else params
    errors: list[str] = []
    doc.mesh.size = _eval_optional(doc.mesh.sizeExpr, doc.mesh.size, params, "メッシュサイズ", errors)
    for r in doc.regions:
        r.epsR = _eval_optional(r.epsRExpr, r.epsR, params, f"領域 {r.name} の εr", errors)
        r.tanDelta = _eval_optional(r.tanDeltaExpr, r.tanDelta, params, f"領域 {r.name} の tanδ", errors)
    return errors


def apply_document_expressions(doc: AxiDocument, params: Optional[dict[str, float]] = None) -> list[str]:
    """文書中の式（点座標・メッシュサイズ・材料）をすべて評価し直す（パラメータの変更時など）."""
    params = document_params(doc) if params is None else params
    return apply_point_expressions(doc, params) + apply_setting_expressions(doc, params)


def _eval_optional(expr: Optional[str], fallback: float, params: dict[str, float],
                   label: str, errors: list[str]) -> float:
    if not expr:
        return fallback
    try:
        return float(evaluate_expression(expr, params))
    except ExpressionError as exc:
        errors.append(f"{label} の式 '{expr}': {exc}")
        return fallback


# ---------------------------------------------------------------------------
# 閉領域と曲線の使われ方
# ---------------------------------------------------------------------------

def profile_tolerance(doc: AxiDocument) -> float:
    pts = [(e.x, e.y) for e in doc.sketch.entities if isinstance(e, SketchPoint) and not is_reference(e)]
    if len(pts) < 2:
        return 1e-9
    lo, hi = g.polygon_bounds(pts)
    return max(1e-9, 1e-9 * g.distance(lo, hi))


def sketch_profiles(doc: AxiDocument) -> list[Profile]:
    return detect_profiles(doc.sketch, tolerance=profile_tolerance(doc))


def curve_use_counts(profiles: list[Profile]) -> dict[Id, int]:
    """曲線ごとに、その曲線を境界に持つ閉領域の数（外周 + 穴）."""
    counts: dict[Id, int] = {}
    for p in profiles:
        for loop in [p.outer, *p.holes]:
            for seg in loop:
                counts[seg.entityId] = counts.get(seg.entityId, 0) + 1
    return counts


def _is_on_axis(doc: AxiDocument, entity, tol: float) -> bool:
    if not isinstance(entity, SketchLine):
        return False
    a, b = point_pos(doc.sketch, entity.p1), point_pos(doc.sketch, entity.p2)
    return abs(a[1]) <= tol and abs(b[1]) <= tol


def auto_bc(doc: AxiDocument, curve_id: Id, profiles: Optional[list[Profile]] = None,
            counts: Optional[dict[Id, int]] = None) -> tuple[str, str]:
    """明示指定を無視した自動判定: (BC 名, 由来 "interface" / "axis" / "default")."""
    if counts is None:
        counts = curve_use_counts(sketch_profiles(doc) if profiles is None else profiles)
    if counts.get(curve_id, 0) >= 2:
        return BC_NONE, "interface"
    if _is_on_axis(doc, get_entity(doc.sketch, curve_id), axis_tolerance(doc.meta.units)):
        return BC_NONE, "axis"
    return DEFAULT_BC, "default"


def effective_bc(doc: AxiDocument, curve_id: Id, profiles: Optional[list[Profile]] = None,
                 counts: Optional[dict[Id, int]] = None) -> tuple[str, str]:
    """曲線の有効な境界条件と由来: 明示指定 "explicit" → 内部界面 → 軸 → 既定 PEC."""
    explicit = doc.boundaries.get(curve_id)
    if explicit in d.BC_NAMES:
        return explicit, "explicit"
    return auto_bc(doc, curve_id, profiles, counts)


def simplify_boundaries(doc: AxiDocument, profiles: Optional[list[Profile]] = None) -> int:
    """自動判定と同じ明示指定を取り除く（取り込み直後の整理）。取り除いた数を返す."""
    counts = curve_use_counts(sketch_profiles(doc) if profiles is None else profiles)
    removed = 0
    for curve_id, bc in list(doc.boundaries.items()):
        if get_entity(doc.sketch, curve_id) is None or auto_bc(doc, curve_id, counts=counts)[0] == bc:
            del doc.boundaries[curve_id]
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# 領域の解決（閉領域 ↔ RegionSetting）
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRegion:
    profile: Profile
    setting: Optional[RegionSetting]      # None なら既定（設定なし）
    name: str
    material_tag: str
    eps_r: float = 1.0
    mu_r: float = 1.0
    tan_delta: float = 0.0
    eps_r_expr: Optional[str] = None
    tan_delta_expr: Optional[str] = None


def _key_ids(key: str) -> set[str]:
    return {k for k in key.split("|") if k}


def match_regions(doc: AxiDocument, profiles: list[Profile]) -> dict[str, RegionSetting]:
    """閉領域 ID → 設定。key 一致 → アンカー点 → 外周エンティティ集合の類似度（Jaccard ≥ 0.5）の順."""
    by_key = {s.key: s for s in doc.regions}
    matched: dict[str, RegionSetting] = {}
    used: set[int] = set()
    for p in profiles:                                            # 1. key
        s = by_key.get(p.id)
        if s is not None and id(s) not in used:
            matched[p.id] = s
            used.add(id(s))
    for p in profiles:                                            # 2. anchor
        if p.id in matched:
            continue
        for s in doc.regions:
            if id(s) in used:
                continue
            if profile_at([p], s.anchor) is p:
                matched[p.id] = s
                used.add(id(s))
                break
    for p in profiles:                                            # 3. Jaccard
        if p.id in matched:
            continue
        ids = _key_ids(p.id)
        best, best_score = None, 0.0
        for s in doc.regions:
            if id(s) in used:
                continue
            other = _key_ids(s.key)
            if not other:
                continue
            score = len(ids & other) / len(ids | other)
            if score >= _JACCARD_MIN and score > best_score:
                best, best_score = s, score
        if best is not None:
            matched[p.id] = best
            used.add(id(best))
    return matched


def _unique_name(base: str, taken: set[str]) -> str:
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}{n}"
    taken.add(name)
    return name


def resolve_regions(doc: AxiDocument, profiles: Optional[list[Profile]] = None,
                    params: Optional[dict[str, float]] = None,
                    errors: Optional[list[str]] = None) -> list[ResolvedRegion]:
    """閉領域ごとの材料（設定が無ければ既定の Vacuum）。名前と材料タグは全体で一意にする."""
    profiles = sketch_profiles(doc) if profiles is None else profiles
    params = document_params(doc) if params is None else params
    errors = [] if errors is None else errors
    matched = match_regions(doc, profiles)
    names: set[str] = set()
    tags: set[str] = set()
    out: list[ResolvedRegion] = []
    for p in profiles:
        s = matched.get(p.id)
        if s is None:
            name = _unique_name(d.DEFAULT_REGION_NAME, names)
            out.append(ResolvedRegion(p, None, name, _unique_name(normalize_material_tag(name), tags)))
            continue
        name = _unique_name(s.name.strip() or d.DEFAULT_REGION_NAME, names)
        tag = _unique_name(normalize_material_tag(s.materialTag or name), tags)
        eps_r = _eval_optional(s.epsRExpr, s.epsR, params, f"領域 {name} の ε_r", errors)
        tan_delta = _eval_optional(s.tanDeltaExpr, s.tanDelta, params, f"領域 {name} の tanδ", errors)
        out.append(ResolvedRegion(p, s, name, tag, eps_r=eps_r, mu_r=s.muR, tan_delta=tan_delta,
                                  eps_r_expr=s.epsRExpr, tan_delta_expr=s.tanDeltaExpr))
    return out


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------

def _arc_min_r(sketch, arc: SketchArc) -> float:
    p = arc_params(sketch, arc)
    lo = p.center[1] - p.radius
    if g.angle_in_sweep(-math.pi / 2, p.startAngle, p.sweep):
        return lo
    return min(g.point_at_angle(p.center, p.radius, p.startAngle)[1],
               g.point_at_angle(p.center, p.radius, p.startAngle + p.sweep)[1])


def check_geometry(doc: AxiDocument) -> list[Issue]:
    """書き出し・メッシュ生成・解析の前の検証。error があれば止め、warning は確認ダイアログに出す.

    式付きの点座標は評価済みとして扱う（呼び出し側は先に :func:`apply_point_expressions`）。
    """
    issues: list[Issue] = []
    sketch = doc.sketch
    tol = axis_tolerance(doc.meta.units)
    params = document_params(doc)
    for (name, _), result in zip(((p.name, p.expression) for p in doc.params), param_results(doc.params)):
        if result.startswith("エラー"):
            issues.append(Issue("error", "param_error", f"パラメータ {name}: {result}"))
    try:
        lc = float(evaluate_expression(doc.mesh.sizeExpr, params)) if doc.mesh.sizeExpr else float(doc.mesh.size)
    except ExpressionError as exc:
        lc = 0.0
        issues.append(Issue("error", "mesh_size", f"メッシュサイズの式 '{doc.mesh.sizeExpr}': {exc}"))
    if lc <= 0:
        issues.append(Issue("error", "mesh_size", "メッシュサイズ lc は正の値にしてください"))

    negative = [e.id for e in sketch.entities if isinstance(e, SketchPoint) and e.y < -tol
                and not e.construction]
    if negative:
        issues.append(Issue("error", "negative_r", f"r < 0 の点が {len(negative)} 個あります（軸対称では r ≥ 0）",
                            negative))
    profiles = sketch_profiles(doc)
    conflicts = count_conflicts(doc.sketch)
    if conflicts:
        issues.append(Issue("warning", "overlap",
                            f"曲線が別の曲線の途中に乗っている・交差している・重なっている所が {conflicts} 箇所あります"
                            "（「交差で分割」で分けてください。領域が穴として扱われます）"))
    if not profiles:
        issues.append(Issue("error", "no_profile", "閉じた領域がありません（線で囲まれた領域を作ってください）"))
        return issues
    counts = curve_use_counts(profiles)
    unused = [e.id for e in sketch.entities
              if isinstance(e, (SketchLine, SketchArc, SketchCircle)) and not e.construction
              and e.id not in counts]
    if unused:
        issues.append(Issue("warning", "unused_curve",
                            f"閉領域の境界になっていない曲線が {len(unused)} 本あります（書き出しません）", unused))
    for e in sketch.entities:
        if isinstance(e, SketchArc) and e.id in counts and _arc_min_r(sketch, e) < -tol:
            issues.append(Issue("error", "arc_crosses_axis", "円弧が軸 r=0 を横切っています", [e.id]))

    # 境界条件
    on_axis: list[Id] = []
    for curve_id in counts:
        e = get_entity(sketch, curve_id)
        bc, source = effective_bc(doc, curve_id, counts=counts)
        auto, auto_source = auto_bc(doc, curve_id, counts=counts)
        if source == "explicit" and auto_source == "interface" and bc != BC_NONE:
            issues.append(Issue("warning", "interface_wall_bc",
                                f"内部の界面に境界条件 {bc} が指定されています（通常は None）", [curve_id]))
        if source == "explicit" and auto_source == "axis" and bc != BC_NONE:
            issues.append(Issue("warning", "axis_wall_bc",
                                f"軸 r=0 上の線に境界条件 {bc} が指定されています（軸はコアが自動判定します）", [curve_id]))
        if source == "explicit" and auto_source == "default" and bc == BC_NONE:
            issues.append(Issue("warning", "outer_none",
                                "外周の曲線に None が指定されています（境界グループが付かず、TM0 では自然境界になります）",
                                [curve_id]))
        if auto_source == "axis":
            on_axis.append(curve_id)
    # 進行波: z_min / z_max の端面が PEC だと壁損失に数えられる
    if doc.analysis.wave == "traveling":
        verts = [point_pos(sketch, pid) for e in sketch.entities if isinstance(e, (SketchLine, SketchArc))
                 and e.id in counts for pid in ((e.p1, e.p2) if isinstance(e, SketchLine) else (e.start, e.end))]
        if verts:
            zmin, zmax = min(v[0] for v in verts), max(v[0] for v in verts)
            span = max(zmax - zmin, 1.0)
            ends: list[Id] = []
            for curve_id in counts:
                e = get_entity(sketch, curve_id)
                if not isinstance(e, SketchLine):
                    continue
                a, b = point_pos(sketch, e.p1), point_pos(sketch, e.p2)
                for z_end in (zmin, zmax):
                    if abs(a[0] - z_end) <= 1e-9 * span and abs(b[0] - z_end) <= 1e-9 * span \
                            and effective_bc(doc, curve_id, counts=counts)[0] == "PEC":
                        ends.append(curve_id)
            if ends:
                issues.append(Issue("warning", "tw_end_pec",
                                    "進行波解析ですが z_min / z_max の端面が PEC です（周期境界として対にされますが、"
                                    "壁損失に数えられます。E-short / M-short / None を推奨）", ends))
    # 材料
    resolved = resolve_regions(doc, profiles, params, errors := [])
    for msg in errors:
        issues.append(Issue("error", "expr_error", msg))
    for r in resolved:
        if r.setting is not None and normalize_material_tag(r.setting.materialTag) != (r.setting.materialTag or ""):
            issues.append(Issue("warning", "tag_renamed",
                                f"領域 {r.name} の材料タグ '{r.setting.materialTag}' は '{r.material_tag}' として書き出します"))
        if r.eps_r <= 0:
            issues.append(Issue("error", "eps_r", f"領域 {r.name} の ε_r は正の値にしてください"))
        if r.tan_delta < 0:
            issues.append(Issue("error", "tan_delta", f"領域 {r.name} の tanδ は 0 以上にしてください"))
        if r.eps_r != 1.0 and any(seg.entityId in on_axis for seg in r.profile.outer):
            issues.append(Issue("warning", "axis_region_eps",
                                f"軸に接する領域 {r.name} の ε_r ≠ 1 です（コアの既知制約: V_eff・R/Q が不正確になります）"))
    return issues


# ---------------------------------------------------------------------------
# 順変換: スケッチ → MultiRegionGeometry
# ---------------------------------------------------------------------------

@dataclass
class ConvertResult:
    geom: MultiRegionGeometry
    segments_of_curve: dict[Id, list[int]] = field(default_factory=dict)   # 曲線 ID → Segment ID 列（CCW 順）
    region_of_profile: dict[str, int] = field(default_factory=dict)        # 閉領域 ID → Region ID
    point_index: dict[Id, int] = field(default_factory=dict)              # 点 ID → 点プールの番号
    issues: list[Issue] = field(default_factory=list)                     # warning のみ（error は例外）

    @property
    def warnings(self) -> list[str]:
        return [i.message for i in self.issues if i.level == "warning"]


class _PointPool:
    """MultiRegionGeometry.points（許容差内の点は統合）."""

    def __init__(self, tol: float):
        self.tol = tol
        self.points: list[Vec2] = []
        self.exprs: list[tuple[Optional[str], Optional[str]]] = []
        self.index: dict[Id, int] = {}
        self.merged = 0

    def add_sketch_point(self, p: SketchPoint) -> int:
        if p.id in self.index:
            return self.index[p.id]
        for i, q in enumerate(self.points):
            if g.distance(q, (p.x, p.y)) <= self.tol:
                self.index[p.id] = i
                self.merged += 1
                if (p.xExpr or p.yExpr) and self.exprs[i] == (None, None):
                    self.exprs[i] = (p.xExpr, p.yExpr)
                return i
        self.index[p.id] = len(self.points)
        self.points.append((float(p.x), float(p.y)))
        self.exprs.append((p.xExpr, p.yExpr))
        return self.index[p.id]

    def add_coord(self, pos: Vec2) -> int:
        for i, q in enumerate(self.points):
            if g.distance(q, pos) <= self.tol:
                return i
        self.points.append((float(pos[0]), float(pos[1])))
        self.exprs.append((None, None))
        return len(self.points) - 1


def _arc_segments(sketch, arc: SketchArc, pool: _PointPool, next_id, bc: str, center_expr) -> list[Segment]:
    """円弧 → Segment 列（CCW 順。180° 以上なら 2 分割）."""
    p = arc_params(sketch, arc)
    start_idx = pool.add_sketch_point(get_entity(sketch, arc.start))
    end_idx = pool.add_sketch_point(get_entity(sketch, arc.end))
    pieces = 2 if p.sweep >= math.pi - 1e-9 else 1
    step = p.sweep / pieces
    indices = [start_idx]
    for k in range(1, pieces):
        indices.append(pool.add_coord(g.point_at_angle(p.center, p.radius, p.startAngle + step * k)))
    indices.append(end_idx)
    theta1 = math.degrees(p.startAngle)
    if theta1 > 180.0:
        theta1 -= 360.0
    segments = []
    for k in range(pieces):
        t1 = theta1 + math.degrees(step) * k
        segments.append(Segment(id=next_id(), type="arc", point_indices=[indices[k], indices[k + 1]], bc_name=bc,
                                center=(float(p.center[0]), float(p.center[1])), radius=float(p.radius),
                                theta1=t1, theta2=t1 + math.degrees(step), center_expr=center_expr))
    return segments


def _circle_segments(sketch, circle: SketchCircle, pool: _PointPool, next_id, bc: str, center_expr) -> list[Segment]:
    """円 → 90° の円弧 4 本（角度 0 / 90 / 180 / 270° の点を追加）."""
    center = point_pos(sketch, circle.center)
    r = float(circle.radius)
    indices = [pool.add_coord(g.point_at_angle(center, r, k * math.pi / 2)) for k in range(4)]
    segments = []
    for k in range(4):
        segments.append(Segment(id=next_id(), type="arc", point_indices=[indices[k], indices[(k + 1) % 4]],
                                bc_name=bc, center=(float(center[0]), float(center[1])), radius=r,
                                theta1=90.0 * k, theta2=90.0 * (k + 1), center_expr=center_expr))
    return segments


def to_multi_region(doc: AxiDocument, strict: bool = True) -> ConvertResult:
    """ドキュメントを ``MultiRegionGeometry`` にする。error があれば :class:`ConvertError`."""
    issues = check_geometry(doc)
    if strict and any(i.level == "error" for i in issues):
        raise ConvertError(issues)
    sketch = doc.sketch
    params = document_params(doc)
    profiles = sketch_profiles(doc)
    counts = curve_use_counts(profiles)
    pool = _PointPool(profile_tolerance(doc))
    counter = {"n": 0}

    def next_id() -> int:
        counter["n"] += 1
        return counter["n"] - 1

    segments: list[Segment] = []
    segments_of_curve: dict[Id, list[int]] = {}
    for e in sketch.entities:
        if e.id not in counts:
            continue
        bc = effective_bc(doc, e.id, counts=counts)[0]
        if isinstance(e, SketchLine):
            a = pool.add_sketch_point(get_entity(sketch, e.p1))
            b = pool.add_sketch_point(get_entity(sketch, e.p2))
            segs = [Segment(id=next_id(), type="line", point_indices=[a, b], bc_name=bc)]
        elif isinstance(e, SketchArc):
            cp = get_entity(sketch, e.center)
            center_expr = (cp.xExpr, cp.yExpr) if (cp.xExpr or cp.yExpr) else None
            segs = _arc_segments(sketch, e, pool, next_id, bc, center_expr)
        elif isinstance(e, SketchCircle):
            cp = get_entity(sketch, e.center)
            center_expr = (cp.xExpr, cp.yExpr) if (cp.xExpr or cp.yExpr) else None
            segs = _circle_segments(sketch, e, pool, next_id, bc, center_expr)
        else:
            continue
        segments.extend(segs)
        segments_of_curve[e.id] = [s.id for s in segs]
    if pool.merged:
        issues.append(Issue("warning", "merged_points", f"近接する点 {pool.merged} 組を 1 点に統合しました"))

    loops: list[Loop] = []
    loop_by_key: dict[frozenset, int] = {}

    def loop_id_for(loop_segments) -> int:
        ids: list[int] = []
        for seg in loop_segments:
            part = segments_of_curve[seg.entityId]
            ids.extend(reversed(part) if seg.reversed else part)
        key = frozenset(ids)
        if key in loop_by_key:
            return loop_by_key[key]
        loop = Loop(id=len(loops), segment_ids=ids, orientation="CCW")
        loops.append(loop)
        loop_by_key[key] = loop.id
        return loop.id

    regions: list[Region] = []
    region_of_profile: dict[str, int] = {}
    for r in resolve_regions(doc, profiles, params):
        outer = loop_id_for(r.profile.outer)
        holes = [loop_id_for(h) for h in r.profile.holes]
        region = Region(id=len(regions), name=r.name, outer_loop_id=outer, hole_loop_ids=holes,
                        material_tag=r.material_tag, eps_r=r.eps_r, mu_r=r.mu_r,
                        eps_r_expr=r.eps_r_expr, tan_delta=r.tan_delta, tan_delta_expr=r.tan_delta_expr)
        regions.append(region)
        region_of_profile[r.profile.id] = region.id

    try:
        mesh_size = float(evaluate_expression(doc.mesh.sizeExpr, params)) if doc.mesh.sizeExpr else float(doc.mesh.size)
    except ExpressionError:
        mesh_size = float(doc.mesh.size)
    settings: dict = {"mesh_order": 1 if doc.mesh.order == 2 else 0}
    view = doc.view
    if None not in (view.zmin, view.zmax, view.rmin, view.rmax):
        settings.update(xmin=str(view.zmin), xmax=str(view.zmax), ymin=str(view.rmin), ymax=str(view.rmax))
    geom = MultiRegionGeometry(points=list(pool.points), segments=segments, loops=loops, regions=regions,
                               unit=doc.meta.units, mesh_size=mesh_size, settings=settings,
                               variables=[(p.name, p.expression) for p in doc.params],
                               point_exprs=list(pool.exprs), mesh_size_expr=doc.mesh.sizeExpr)
    errors = geom.validate()
    if errors:
        raise ConvertError([Issue("error", "validate", m) for m in errors])
    return ConvertResult(geom=geom, segments_of_curve=segments_of_curve, region_of_profile=region_of_profile,
                         point_index=dict(pool.index), issues=[i for i in issues if i.level == "warning"])


# ---------------------------------------------------------------------------
# 逆変換: MultiRegionGeometry → ドキュメント（旧形式の取り込み）
# ---------------------------------------------------------------------------

def _loop_polygon(geom: MultiRegionGeometry, loop: Loop) -> list[Vec2]:
    """ループを折れ線にする（円弧は 5° 刻み。向きは segment 列の順にたどる）."""
    poly: list[Vec2] = []
    current: Optional[int] = None
    for sid in loop.segment_ids:
        seg = geom.segment_by_id(sid)
        a, b = seg.point_indices
        if current is None:
            # 次の segment と共有しない方の端点から始める
            nxt = geom.segment_by_id(loop.segment_ids[1]) if len(loop.segment_ids) > 1 else None
            if nxt is not None and a in nxt.point_indices and b not in nxt.point_indices:
                a, b = b, a
        elif b == current:
            a, b = b, a
        pts = [geom.points[a], geom.points[b]]
        if seg.type == "arc":
            t1, t2 = math.radians(seg.theta1), math.radians(seg.theta2)
            samples = g.sample_arc(seg.center, seg.radius, t1, t2 - t1, _ARC_SAMPLE_STEP)
            # theta1 側の端点が先頭。たどる向きが逆なら反転
            if g.distance(samples[0], geom.points[a]) > g.distance(samples[-1], geom.points[a]):
                samples.reverse()
            pts = samples
        poly.extend(pts[:-1])                 # 終点は次の segment の始点（閉ループ）
        current = b
    return poly


def from_multi_region(geom: MultiRegionGeometry) -> tuple[AxiDocument, list[str]]:
    """ver2.3 の ``MultiRegionGeometry`` から ver3 のドキュメントを作る。戻り値は (文書, 警告)."""
    warnings: list[str] = []
    doc = d.create_empty_document(units=geom.unit if geom.unit in d.UNITS else "mm")
    sketch = doc.sketch = create_sketch_feature("Sketch1")
    doc.params = [d.Param(id=d.new_id(), name=name, expression=expr) for name, expr in geom.variables]
    exprs = list(geom.point_exprs) + [(None, None)] * (len(geom.points) - len(geom.point_exprs))
    point_ids: list[Id] = []
    for (z, r), (ze, re_) in zip(geom.points, exprs):
        p = SketchPoint(id=d.new_id(), x=float(z), y=float(r), xExpr=ze or None, yExpr=re_ or None)
        sketch.entities.append(p)
        point_ids.append(p.id)
    tol = max(1e-9, 1e-9 * (g.distance(*g.polygon_bounds(geom.points)) if len(geom.points) >= 2 else 1.0))
    centers: list[tuple[Vec2, Id, tuple]] = []      # (座標, 点 ID, 式) — 同じ座標でも式が違えば別の中心点
    curve_of_segment: dict[int, Id] = {}
    for seg in geom.segments:
        a, b = point_ids[seg.point_indices[0]], point_ids[seg.point_indices[1]]
        if seg.type == "line":
            e = SketchLine(id=d.new_id(), p1=a, p2=b)
        else:
            center = (float(seg.center[0]), float(seg.center[1]))
            ce = tuple(x or None for x in (seg.center_expr or (None, None)))
            c_id = next((cid for pos, cid, exprs in centers
                         if g.distance(pos, center) <= tol and (exprs == ce or ce == (None, None))), None)
            if c_id is None:
                cp = SketchPoint(id=d.new_id(), x=center[0], y=center[1], xExpr=ce[0], yExpr=ce[1])
                sketch.entities.append(cp)
                c_id = cp.id
                centers.append((center, c_id, ce))
            # theta1 側の端点が start（point_indices の順とは無関係）
            pa = geom.points[seg.point_indices[0]]
            ang = math.degrees(math.atan2(pa[1] - center[1], pa[0] - center[0]))
            rel = (ang - float(seg.theta1)) % 360.0
            start, end = (a, b) if min(rel, 360.0 - rel) < 1e-3 else (b, a)
            e = SketchArc(id=d.new_id(), center=c_id, start=start, end=end,
                          radius=g.distance(center, point_pos(sketch, start)))
        sketch.entities.append(e)
        curve_of_segment[seg.id] = e.id
        if seg.bc_name in d.BC_NAMES:
            doc.boundaries[e.id] = seg.bc_name

    profiles = sketch_profiles(doc)
    for region in geom.regions:
        try:
            outer_poly = _loop_polygon(geom, geom.loop_by_id(region.outer_loop_id))
            hole_polys = [_loop_polygon(geom, geom.loop_by_id(h)) for h in region.hole_loop_ids]
        except KeyError as exc:
            warnings.append(f"領域 {region.name}: ループが見つかりません（{exc}）")
            continue
        if len(outer_poly) < 3:
            warnings.append(f"領域 {region.name}: 外周ループが閉じていません")
            continue
        anchor = g.interior_point(outer_poly, hole_polys)
        prof = profile_at(profiles, anchor)
        setting = RegionSetting(key=prof.id if prof is not None else "", anchor=prof.anchor if prof is not None else anchor,
                                name=region.name, materialTag=region.material_tag, epsR=float(region.eps_r),
                                epsRExpr=region.eps_r_expr or None, muR=float(region.mu_r),
                                tanDelta=float(region.tan_delta), tanDeltaExpr=region.tan_delta_expr or None)
        doc.regions.append(setting)
        if prof is None:
            warnings.append(f"領域 {region.name} に対応する閉領域が見つかりません（設定は孤立として保持）")
    simplify_boundaries(doc, profiles)

    s = geom.settings or {}
    doc.mesh = d.MeshSettings(size=float(geom.mesh_size), sizeExpr=geom.mesh_size_expr or None,
                              order=1 if s.get("mesh_order") == 0 else 2)
    params = document_params(doc)
    view: dict[str, Optional[float]] = {}
    for src, dst in (("xmin", "zmin"), ("xmax", "zmax"), ("ymin", "rmin"), ("ymax", "rmax")):
        raw = s.get(src)
        try:
            view[dst] = float(evaluate_expression(str(raw), params)) if raw not in (None, "") else None
        except ExpressionError:
            view[dst] = None
    doc.view = d.ViewSettings(**view)
    if not geom.regions:
        warnings.append("領域がありません（輪郭が閉じていない場合は、線ツールまたは「端点を結ぶ」で閉じてください）")
    return doc, warnings
