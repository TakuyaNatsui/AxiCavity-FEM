"""ver2.1 多領域メッシュのデータモデル。

ver2 までの単一閉プロファイル (PointLineEditor / .gmshproj) を拡張し、
複数領域・穴あき(ドーナツ)領域・領域ごとの材質(eps_r)を表現する。

主な要素:
    Point        : (z, r) 座標
    Segment      : 直線 or 円弧の 1 セグメント + BC 名
    Loop         : Segment 列で閉じる輪郭 (外周 CCW / 穴 CW)
    Region       : 1 外周ループ + 0..N 穴ループ + 材質 (eps_r)
    MultiRegionGeometry : 上記をまとめた幾何全体

設計方針:
    - 点プールは全領域共通。共有境界の節点は同じ点を参照する。
    - segment は ID で管理し、loop は segment_ids で参照する。
    - BC は segment 単位で付与する (PEC / E-short / M-short)。
    - JSON シリアライズ可能な dict/list/数値のみで構成。
    - 既存単一領域 .gmshproj (PointLineEditor 出力) を vacuum 1 領域に変換する
      互換読込 `from_legacy_single_loop` を提供。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json
import math


SCHEMA_VERSION = "2.3"
# .gmshproj の読み込みを許可するスキーマ版 (後方互換: 2.1/2.2 も MR 形式として読む)
SUPPORTED_SCHEMA_VERSIONS = ("2.1", "2.2", "2.3")

# 標準 BC 名 (BC_NAMING.md と一致)
BC_NAMES = ("PEC", "E-short", "M-short")
DEFAULT_BC = "PEC"

# "None": Physical Curve グループを付けない境界を表す特別な BC 名。
#   - 領域間の共有(内部)境界: 場が連続するため壁を付けない
#   - 軸 r=0 の線: ソルバが幾何的に自動判定するため明示 BC 不要
# gmsh エクスポートでは "None" の segment を Physical Curve から除外する。
BC_NONE = "None"
# Segment.bc_name に許可される全値
ALLOWED_BC_NAMES = BC_NAMES + (BC_NONE,)

# 既定材質タグ
VACUUM_TAG = "vacuum"


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """直線または円弧のセグメント。

    Attributes:
        id: 全 segment 中で一意の整数 ID。
        type: "line" または "arc"。
        point_indices: [p_start_idx, p_end_idx] (MultiRegionGeometry.points への
            インデックス)。
        bc_name: BC 名 ("PEC" / "E-short" / "M-short")。ver2 の "physical_name"
            と等価。
        center: arc のみ。円弧の中心 (cz, cr)。
        radius: arc のみ。
        theta1: arc のみ (deg)。
        theta2: arc のみ (deg)。
    """

    id: int
    type: str
    point_indices: list[int]
    bc_name: str = DEFAULT_BC
    center: tuple[float, float] | None = None
    radius: float | None = None
    theta1: float | None = None
    theta2: float | None = None
    # arc 中心をユーザが明示入力した式 (cz_expr, cr_expr)。自動算出時は None (ver2.2)
    center_expr: tuple[str | None, str | None] | None = None

    def __post_init__(self) -> None:
        if self.type not in ("line", "arc"):
            raise ValueError(f"Segment.type must be 'line' or 'arc', got {self.type!r}")
        if len(self.point_indices) != 2:
            raise ValueError(
                f"Segment.point_indices must have length 2, got {self.point_indices!r}"
            )
        if self.bc_name not in ALLOWED_BC_NAMES:
            raise ValueError(
                f"Segment.bc_name must be one of {ALLOWED_BC_NAMES}, "
                f"got {self.bc_name!r}"
            )
        if self.type == "arc":
            for attr in ("center", "radius", "theta1", "theta2"):
                if getattr(self, attr) is None:
                    raise ValueError(f"arc Segment requires '{attr}'")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "point_indices": list(self.point_indices),
            "bc_name": self.bc_name,
        }
        if self.type == "arc":
            d["center"] = [float(self.center[0]), float(self.center[1])]
            d["radius"] = float(self.radius)
            d["theta1"] = float(self.theta1)
            d["theta2"] = float(self.theta2)
            if self.center_expr is not None:
                d["center_expr"] = [self.center_expr[0], self.center_expr[1]]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        center = tuple(d["center"]) if d.get("center") is not None else None
        ce = d.get("center_expr")
        center_expr = (ce[0], ce[1]) if ce is not None else None
        return cls(
            id=int(d["id"]),
            type=str(d["type"]),
            point_indices=[int(i) for i in d["point_indices"]],
            bc_name=str(d.get("bc_name", DEFAULT_BC)),
            center=center,
            radius=d.get("radius"),
            theta1=d.get("theta1"),
            theta2=d.get("theta2"),
            center_expr=center_expr,
        )


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
@dataclass
class Loop:
    """閉ループ。Region の外周 or 穴として使われる。

    Attributes:
        id: ループ ID。
        segment_ids: ループを構成する Segment.id の順序付きリスト
            (連続して端点を共有する)。
        orientation: "CCW" (外周) または "CW" (穴)。表示・検証時に参照。
            幾何的な向きは segments の順序と各 segment の point_indices で決まる。
            ここは「意図された向き」のラベルとして保持。
            注: ver2.1 の Gmsh エクスポートは OpenCASCADE モードで行うため、
            この向きラベルは実際のメッシュ生成では無視される（OCC が自動補正）。
            データ後方互換のためフィールドは残置。
    """

    id: int
    segment_ids: list[int]
    orientation: str = "CCW"

    def __post_init__(self) -> None:
        if self.orientation not in ("CCW", "CW"):
            raise ValueError(
                f"Loop.orientation must be 'CCW' or 'CW', got {self.orientation!r}"
            )
        if len(self.segment_ids) < 1:
            raise ValueError("Loop.segment_ids must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "segment_ids": list(self.segment_ids),
            "orientation": self.orientation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Loop:
        return cls(
            id=int(d["id"]),
            segment_ids=[int(s) for s in d["segment_ids"]],
            orientation=str(d.get("orientation", "CCW")),
        )


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------
@dataclass
class Region:
    """1 つの材質領域 (1 外周ループ + 0..N 穴ループ)。

    Attributes:
        id: 領域 ID (Gmsh の 2D Physical Group タグにも対応させる)。
        name: 表示用の名前 (例 "Vacuum", "Dielectric_1")。
        outer_loop_id: 外周ループ ID。
        hole_loop_ids: 穴ループ ID のリスト (空なら穴なし)。
        material_tag: HDF5 と Gmsh Physical Surface 名に使われる文字列タグ。
            英数+アンダースコアのみ推奨 (例 "vacuum", "dielectric_1")。
        eps_r: 比誘電率 (実数、>0)。
        mu_r: 比透磁率 (実数、>0)。ver2.1 では使用しないが将来拡張のため保持。
        tan_delta: 誘電正接 tanδ (実数、>=0)。0 で無損失 (ver2.3)。
    """

    id: int
    name: str
    outer_loop_id: int
    hole_loop_ids: list[int] = field(default_factory=list)
    material_tag: str = VACUUM_TAG
    eps_r: float = 1.0
    mu_r: float = 1.0
    eps_r_expr: str | None = None  # eps_r を式で入力した場合の式 (ver2.2)
    tan_delta: float = 0.0
    tan_delta_expr: str | None = None  # tan_delta を式で入力した場合の式 (ver2.3)

    def __post_init__(self) -> None:
        if self.eps_r <= 0:
            raise ValueError(f"Region.eps_r must be > 0, got {self.eps_r}")
        if self.mu_r <= 0:
            raise ValueError(f"Region.mu_r must be > 0, got {self.mu_r}")
        if self.tan_delta < 0:
            raise ValueError(
                f"Region.tan_delta must be >= 0, got {self.tan_delta}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "outer_loop_id": self.outer_loop_id,
            "hole_loop_ids": list(self.hole_loop_ids),
            "material_tag": self.material_tag,
            "eps_r": float(self.eps_r),
            "mu_r": float(self.mu_r),
            "tan_delta": float(self.tan_delta),
        }
        if self.eps_r_expr:
            d["eps_r_expr"] = self.eps_r_expr
        if self.tan_delta_expr:
            d["tan_delta_expr"] = self.tan_delta_expr
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Region:
        return cls(
            id=int(d["id"]),
            name=str(d["name"]),
            outer_loop_id=int(d["outer_loop_id"]),
            hole_loop_ids=[int(h) for h in d.get("hole_loop_ids", [])],
            material_tag=str(d.get("material_tag", VACUUM_TAG)),
            eps_r=float(d.get("eps_r", 1.0)),
            mu_r=float(d.get("mu_r", 1.0)),
            eps_r_expr=d.get("eps_r_expr"),
            tan_delta=float(d.get("tan_delta", 0.0)),
            tan_delta_expr=d.get("tan_delta_expr"),
        )


# ---------------------------------------------------------------------------
# MultiRegionGeometry
# ---------------------------------------------------------------------------
@dataclass
class MultiRegionGeometry:
    """多領域幾何モデル全体。

    Attributes:
        points: [(z0, r0), (z1, r1), ...] 全領域共通の点プール。
        segments: Segment のリスト。id は重複なし。
        loops: Loop のリスト。
        regions: Region のリスト。
        unit: 座標単位 ("mm", "cm", "m" など)。Gmsh 出力時のスケール変換に使う。
        mesh_size: Gmsh の characteristic length (単位は `unit` と同じ)。
        settings: GUI 状態など追加メタデータ (xmin/xmax/ymin/ymax 軸範囲、
            mesh_order 等を入れる)。
        variables: ユーザ定義変数 [(name, expr), ...]。順序＝評価順で、上の行の
            変数を下の行で参照できる。座標・寸法・材料の入力欄で数式に使う (ver2.2)。
    """

    points: list[tuple[float, float]] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    unit: str = "mm"
    mesh_size: float = 5.0
    settings: dict[str, Any] = field(default_factory=dict)
    variables: list[tuple[str, str]] = field(default_factory=list)
    # 各点の座標式 (z_expr, r_expr)。points と常に同長。None=式なし=定数 (ver2.2)
    point_exprs: list[tuple[str | None, str | None]] = field(default_factory=list)
    mesh_size_expr: str | None = None  # mesh_size を式で入力した場合 (ver2.2)

    def __post_init__(self) -> None:
        # point_exprs を points と同長に整える (load/legacy/直接構築すべてに効く)
        self._ensure_point_exprs_length()

    # ---- 点＋式の同期ヘルパ (points と point_exprs を常に同長に保つ) -----
    def _ensure_point_exprs_length(self) -> None:
        n = len(self.points)
        m = len(self.point_exprs)
        if m < n:
            self.point_exprs.extend([(None, None)] * (n - m))
        elif m > n:
            del self.point_exprs[n:]

    def append_point(self, z: float, r: float,
                     z_expr: str | None = None, r_expr: str | None = None) -> int:
        """点を追加し、対応する式エントリも追加する。新規点のインデックスを返す。"""
        self.points.append((float(z), float(r)))
        self.point_exprs.append((z_expr, r_expr))
        return len(self.points) - 1

    def set_point(self, idx: int, z: float, r: float,
                  z_expr: str | None = None, r_expr: str | None = None) -> None:
        """既存点の座標と式を更新する。"""
        self._ensure_point_exprs_length()
        self.points[idx] = (float(z), float(r))
        self.point_exprs[idx] = (z_expr, r_expr)

    def pop_point(self, idx: int) -> None:
        """点とその式エントリを同時に削除する (segment の再インデックスは呼び出し側)。"""
        self.points.pop(idx)
        if 0 <= idx < len(self.point_exprs):
            self.point_exprs.pop(idx)
        self._ensure_point_exprs_length()

    # ---- ID 採番ヘルパ -------------------------------------------------
    def next_segment_id(self) -> int:
        return max((s.id for s in self.segments), default=-1) + 1

    def next_loop_id(self) -> int:
        return max((l.id for l in self.loops), default=-1) + 1

    def next_region_id(self) -> int:
        return max((r.id for r in self.regions), default=-1) + 1

    # ---- ルックアップ --------------------------------------------------
    def segment_by_id(self, sid: int) -> Segment:
        for s in self.segments:
            if s.id == sid:
                return s
        raise KeyError(f"segment id {sid} not found")

    def loop_by_id(self, lid: int) -> Loop:
        for l in self.loops:
            if l.id == lid:
                return l
        raise KeyError(f"loop id {lid} not found")

    def region_by_id(self, rid: int) -> Region:
        for r in self.regions:
            if r.id == rid:
                return r
        raise KeyError(f"region id {rid} not found")

    # ---- 妥当性チェック ------------------------------------------------
    def validate(self) -> list[str]:
        """整合性を検証し、問題を文字列リストで返す。空リストなら OK。"""
        errors: list[str] = []
        # 点インデックス境界
        n_points = len(self.points)
        # ver2.2: point_exprs は points と同長でなければならない
        if len(self.point_exprs) != n_points:
            errors.append(
                f"point_exprs length {len(self.point_exprs)} != points length {n_points}"
            )
        for s in self.segments:
            for pi in s.point_indices:
                if not (0 <= pi < n_points):
                    errors.append(
                        f"segment {s.id}: point_indices {s.point_indices} out of range "
                        f"[0, {n_points})"
                    )
        # ID の重複
        for kind, items in (("segment", self.segments),
                            ("loop", self.loops),
                            ("region", self.regions)):
            ids = [x.id for x in items]
            if len(set(ids)) != len(ids):
                errors.append(f"{kind} ids contain duplicates: {ids}")
        # loop が segment id を持っているか
        seg_ids = {s.id for s in self.segments}
        for l in self.loops:
            for sid in l.segment_ids:
                if sid not in seg_ids:
                    errors.append(f"loop {l.id} refers to unknown segment {sid}")
        # region が loop id を持っているか
        loop_ids = {l.id for l in self.loops}
        for r in self.regions:
            if r.outer_loop_id not in loop_ids:
                errors.append(
                    f"region {r.id}: outer_loop_id {r.outer_loop_id} not found"
                )
            for hid in r.hole_loop_ids:
                if hid not in loop_ids:
                    errors.append(
                        f"region {r.id}: hole_loop_id {hid} not found"
                    )
        # material_tag の重複は許容しない (HDF5/Gmsh の Physical Group 名に使うため)
        tags = [r.material_tag for r in self.regions]
        if len(set(tags)) != len(tags):
            errors.append(f"region material_tags contain duplicates: {tags}")
        return errors

    # ---- 幾何ヘルパ ----------------------------------------------------
    def loop_signed_area(self, lid: int) -> float:
        """ループの符号付き面積 (CCW なら +、CW なら -)。

        円弧は弦近似 (端点を結ぶ直線) で計算する。厳密ではないが
        向き判定 (正負) には十分。
        """
        loop = self.loop_by_id(lid)
        verts = self._loop_vertex_sequence(loop)
        n = len(verts)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    def _loop_vertex_sequence(self, loop: Loop) -> list[tuple[float, float]]:
        """ループの segment 列を端点で連結した頂点列を返す。

        各 segment の point_indices[0] が前 segment の point_indices[1] と
        一致する前提だが、逆向き接続も許容する。
        """
        verts: list[tuple[float, float]] = []
        prev_end: int | None = None
        for sid in loop.segment_ids:
            seg = self.segment_by_id(sid)
            p_a, p_b = seg.point_indices
            if prev_end is None:
                verts.append(self.points[p_a])
                verts.append(self.points[p_b])
                prev_end = p_b
            else:
                if p_a == prev_end:
                    verts.append(self.points[p_b])
                    prev_end = p_b
                elif p_b == prev_end:
                    verts.append(self.points[p_a])
                    prev_end = p_a
                else:
                    # 不連続。とりあえず append (validate で別途検出)
                    verts.append(self.points[p_a])
                    verts.append(self.points[p_b])
                    prev_end = p_b
        # 末尾が最初と一致するなら除去 (閉ループ)
        if len(verts) > 1 and verts[-1] == verts[0]:
            verts.pop()
        return verts

    # ---- シリアライズ --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "points": [[float(p[0]), float(p[1])] for p in self.points],
            "segments": [s.to_dict() for s in self.segments],
            "loops": [l.to_dict() for l in self.loops],
            "regions": [r.to_dict() for r in self.regions],
            "unit": self.unit,
            "mesh_size": float(self.mesh_size),
            "settings": dict(self.settings),
            "variables": [[str(n), str(e)] for n, e in self.variables],
            "point_exprs": [[ze, re] for (ze, re) in self.point_exprs],
            "mesh_size_expr": self.mesh_size_expr,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiRegionGeometry:
        version = d.get("schema_version")
        if version is None:
            # schema_version が無ければ ver2 形式 (legacy) として読む
            return cls.from_legacy_single_loop(d)
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {version!r} "
                f"(expected one of {SUPPORTED_SCHEMA_VERSIONS!r})"
            )
        return cls(
            points=[tuple(p) for p in d.get("points", [])],
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            loops=[Loop.from_dict(l) for l in d.get("loops", [])],
            regions=[Region.from_dict(r) for r in d.get("regions", [])],
            unit=str(d.get("unit", "mm")),
            mesh_size=float(d.get("mesh_size", 5.0)),
            settings=dict(d.get("settings", {})),
            variables=[(str(v[0]), str(v[1])) for v in d.get("variables", [])],
            point_exprs=[(pe[0], pe[1]) for pe in d.get("point_exprs", [])],
            mesh_size_expr=d.get("mesh_size_expr"),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> MultiRegionGeometry:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ---- ver2 単一領域 JSON との互換 ----------------------------------
    @classmethod
    def from_legacy_single_loop(cls, d: dict[str, Any]) -> MultiRegionGeometry:
        """ver2 (PointLineEditor) の .gmshproj 形式を 1 領域 (vacuum) として読む。

        ver2 形式:
            {
              "points": [[x, y], ...],
              "segments": [
                {"type": "line"|"arc", "points": [p1, p2], "physical_name": "...",
                 "center": [...], "radius": ..., "theta1": ..., "theta2": ...},
                ...
              ],
              "loop_closed": true/false,
              "settings": {"unit": "mm", "xmin": "0", ..., "mesh_size": "5",
                           "mesh_order": 0 or 1}
            }
        """
        points: list[tuple[float, float]] = [tuple(p) for p in d.get("points", [])]
        legacy_segs = d.get("segments", [])
        segments: list[Segment] = []
        for i, s in enumerate(legacy_segs):
            stype = s.get("type", "line")
            bc = s.get("physical_name") or s.get("bc_name") or DEFAULT_BC
            if bc == "None" or bc is None:
                bc = DEFAULT_BC
            kw: dict[str, Any] = {
                "id": i,
                "type": stype,
                "point_indices": [int(x) for x in s["points"]],
                "bc_name": bc,
            }
            if stype == "arc":
                kw.update(
                    center=tuple(s["center"]),
                    radius=float(s["radius"]),
                    theta1=float(s["theta1"]),
                    theta2=float(s["theta2"]),
                )
            segments.append(Segment(**kw))

        loop = Loop(id=0, segment_ids=[s.id for s in segments], orientation="CCW")
        region = Region(
            id=0,
            name="Vacuum",
            outer_loop_id=0,
            hole_loop_ids=[],
            material_tag=VACUUM_TAG,
            eps_r=1.0,
            mu_r=1.0,
        )

        legacy_settings = d.get("settings", {}) or {}
        unit = str(legacy_settings.get("unit", "mm"))
        try:
            mesh_size = float(legacy_settings.get("mesh_size", 5.0))
        except (TypeError, ValueError):
            mesh_size = 5.0

        # 残りの settings (xmin/xmax/ymin/ymax/mesh_order 等) はそのまま保持
        settings = {k: v for k, v in legacy_settings.items()
                    if k not in ("unit", "mesh_size")}

        geom = cls(
            points=points,
            segments=segments,
            loops=[loop],
            regions=[region],
            unit=unit,
            mesh_size=mesh_size,
            settings=settings,
        )

        # ver2.1: OCC モードでは向きは自動補正されるため常に CCW ラベルに固定。
        # (旧 ver2 の "CW なら反転" ロジックは廃止)
        loop.orientation = "CCW"
        return geom
