"""ver3 のドキュメントモデル（純 Python dataclass）.

スケッチ系の dataclass（``SketchPoint`` … ``SketchConstraint`` / ``SketchFeature`` / ``Param``）は
EM-CAD-py ``emcad/core/document.py`` からの転用で、フィールド名は JSON 互換のため camelCase
（転用したモジュール群がそのまま動くよう名前を変えない）。ver3 の設定 dataclass も同じ流儀で
camelCase にそろえる。関数名・変数名は snake_case。

ver3 の追加:
    - ``SketchPoint.xExpr`` / ``yExpr``: 座標の式（ver2.3 の点座標の式入力。式を持つ座標は
      パラメータで決まる固定座標として扱う）
    - ``AxiDocument``: スケッチ 1 枚（x = z, y = r）＋ 領域（閉領域ごとの材料）＋ 境界条件
      （曲線ごとの明示指定）＋ メッシュ / 解析 / post / レポート / 表示範囲の設定
座標の単位は ``meta.units``（m / cm / mm / inch。ver2.3 の ``MultiRegionGeometry.unit`` と同じ）。
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union

Id = str
Vec2 = tuple[float, float]

SCHEMA_VERSION = 1
UNITS = ("m", "cm", "mm", "inch")
# 境界条件名（ver2.3 shared.multi_region_model.ALLOWED_BC_NAMES と一致）
BC_NAMES = ("PEC", "E-short", "M-short", "None")
ANALYSIS_TYPES = ("tm0", "hom")
WAVE_TYPES = ("standing", "traveling")
DEFAULT_REGION_NAME = "Vacuum"
DEFAULT_MATERIAL_TAG = "vacuum"
SIGMA_COPPER = 5.8e7


def new_id() -> Id:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# スケッチ（EM-CAD-py から転用）
# ---------------------------------------------------------------------------
@dataclass
class PlaneRef:
    """スケッチ平面。ver3 は原点平面 XY（x = z, y = r）だけを使う（転用コードとの互換のため残す）."""

    kind: str = "origin"
    plane: Optional[str] = "XY"

    @staticmethod
    def origin(plane: str = "XY") -> "PlaneRef":
        return PlaneRef(kind="origin", plane=plane)


# 参照ジオメトリ（原点・z 軸 r = 0・r 軸 z = 0）: ``projection`` にこの値を持つ要素。動かせない・消せない・閉領域に入らない。
# 作るのは ``core/sketch/reference.py``（文書を開くとき ``DocumentStore.load_document`` が足す）
REFERENCE_PROJECTION = "reference"
REFERENCE_ORIGIN = "ref-origin"
REFERENCE_Z_AXIS = "ref-z-axis"          # z 軸 = 直線 r = 0（原点 → (1, 0)）
REFERENCE_R_AXIS = "ref-r-axis"          # r 軸 = 直線 z = 0（原点 → (0, 1)）
REFERENCE_Z_END = "ref-z-axis-end"
REFERENCE_R_END = "ref-r-axis-end"


def is_reference(entity) -> bool:
    """参照ジオメトリの要素か."""
    return entity is not None and getattr(entity, "projection", None) == REFERENCE_PROJECTION


@dataclass
class SketchPoint:
    id: Id
    x: float
    y: float
    fixed: Optional[bool] = None
    construction: Optional[bool] = None
    projection: Optional[Id] = None    # ver3: 参照ジオメトリなら REFERENCE_PROJECTION（それ以外は None）
    xExpr: Optional[str] = None        # ver3: x（= z）の式。None なら定数
    yExpr: Optional[str] = None        # ver3: y（= r）の式
    type: ClassVar[str] = "point"


@dataclass
class SketchLine:
    id: Id
    p1: Id
    p2: Id
    construction: Optional[bool] = None
    projection: Optional[Id] = None
    type: ClassVar[str] = "line"


@dataclass
class SketchCircle:
    id: Id
    center: Id
    radius: float
    construction: Optional[bool] = None
    projection: Optional[Id] = None
    type: ClassVar[str] = "circle"


@dataclass
class SketchArc:
    """中心・始点・終点を持ち、反時計回りに start から end へ進む."""

    id: Id
    center: Id
    start: Id
    end: Id
    radius: float
    construction: Optional[bool] = None
    projection: Optional[Id] = None
    type: ClassVar[str] = "arc"


SketchEntity = Union[SketchPoint, SketchLine, SketchCircle, SketchArc]

CONSTRAINT_TYPES = (
    "coincident", "horizontal", "vertical", "parallel", "perpendicular",
    "tangent", "equal", "concentric", "fix", "symmetric", "pointOnCurve",
    "distance", "radius", "diameter", "angle", "pointLineDistance", "circleLineDistance",
)
DIMENSION_TYPES = ("distance", "radius", "diameter", "angle", "pointLineDistance", "circleLineDistance")


@dataclass
class SketchConstraint:
    """スケッチ拘束（EM-CAD-py と同じ 1 dataclass 方式）.

    種別ごとに使うフィールド:
        coincident: p1, p2 / horizontal, vertical: line /
        parallel, perpendicular, angle(value): l1, l2 /
        tangent, equal: e1, e2 / concentric: c1, c2 / fix: point /
        symmetric: p1, p2, line / pointOnCurve: point, curve /
        distance(value): p1, p2 / pointLineDistance(value): point, line /
        radius, diameter(value): curve /
        circleLineDistance(value): curve（円 / 円弧）, line — 円周と直線の距離 |中心と直線の距離 − 半径|（ver3）
    寸法拘束の value は式文字列（ドキュメントのパラメータを参照できる）。
    """

    id: Id
    type: str
    p1: Optional[Id] = None
    p2: Optional[Id] = None
    line: Optional[Id] = None
    l1: Optional[Id] = None
    l2: Optional[Id] = None
    e1: Optional[Id] = None
    e2: Optional[Id] = None
    c1: Optional[Id] = None
    c2: Optional[Id] = None
    point: Optional[Id] = None
    curve: Optional[Id] = None
    value: Optional[str] = None


def is_dimension_constraint(c: SketchConstraint) -> bool:
    return c.value is not None


@dataclass
class SketchFeature:
    id: Id
    name: str
    plane: PlaneRef = field(default_factory=PlaneRef.origin)
    suppressed: bool = False
    entities: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    projections: list = field(default_factory=list)   # ver3 では常に空
    type: ClassVar[str] = "sketch"


ENTITY_CLASSES: dict[str, type] = {
    cls.type: cls for cls in (SketchPoint, SketchLine, SketchCircle, SketchArc)
}


@dataclass
class Param:
    """パラメータ（ver2.3 の変数表の 1 行）。上から順に評価し、上の行を下の行で参照できる."""

    id: Id
    name: str
    expression: str


# ---------------------------------------------------------------------------
# ver3 の設定
# ---------------------------------------------------------------------------
@dataclass
class RegionSetting:
    """閉領域 1 つの材料設定.

    ``key`` は閉領域の ID（外周エンティティ ID の連結、``profiles.Profile.id``）、``anchor`` は
    領域内の点。文書が変わるたびに store が現在の閉領域へ対応付け直す（key → anchor → 類似度）。
    """

    key: str
    anchor: Vec2
    name: str = DEFAULT_REGION_NAME
    materialTag: str = DEFAULT_MATERIAL_TAG
    epsR: float = 1.0
    epsRExpr: Optional[str] = None
    muR: float = 1.0
    tanDelta: float = 0.0
    tanDeltaExpr: Optional[str] = None


@dataclass
class MeshSettings:
    size: float = 5.0                  # 節点間隔 lc（meta.units の単位）
    sizeExpr: Optional[str] = None
    order: int = 2                     # メッシュ幾何次数 = 要素次数（1 / 2）


@dataclass
class AnalysisSettings:
    type: str = "tm0"                  # tm0 / hom
    wave: str = "standing"             # standing / traveling
    phases: str = "120"                # 進行波の位相 [deg]: "120" / "60,90,120" / "0:180:20"
    numModes: int = 10
    azOrders: str = "0 1"              # HOM の方位角次数（空白区切り）
    targetFreqGHz: Optional[float] = None
    runPost: bool = True               # solve の後に post を続けて実行する


@dataclass
class PostSettings:
    cond: float = SIGMA_COPPER         # 壁の導電率 [S/m]
    beta: float = 1.0                  # 粒子の β（TM0 の V_eff）


@dataclass
class ReportSettings:
    animate: bool = False
    timePhase: float = 0.0
    showMesh: bool = True
    dpi: int = 120


@dataclass
class ViewSettings:
    """表示範囲（ver2.3 の Draw Area。None なら全体表示）."""

    zmin: Optional[float] = None
    zmax: Optional[float] = None
    rmin: Optional[float] = None
    rmax: Optional[float] = None


@dataclass
class DocumentMeta:
    name: str = "Untitled"
    units: str = "mm"
    createdAt: str = ""
    modifiedAt: str = ""


@dataclass
class AxiDocument:
    meta: DocumentMeta = field(default_factory=DocumentMeta)
    params: list = field(default_factory=list)                 # list[Param]
    sketch: SketchFeature = field(default_factory=lambda: SketchFeature(id=new_id(), name="Sketch1"))
    regions: list = field(default_factory=list)                # list[RegionSetting]
    boundaries: dict = field(default_factory=dict)             # 曲線 ID → BC 名（明示指定のみ）
    mesh: MeshSettings = field(default_factory=MeshSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    post: PostSettings = field(default_factory=PostSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    view: ViewSettings = field(default_factory=ViewSettings)
    schemaVersion: int = SCHEMA_VERSION

    def feature(self, feature_id: Id):
        """転用コード（``store.find_sketch`` など）との互換: ID がスケッチのものならそれを返す."""
        return self.sketch if self.sketch.id == feature_id else None

    def param(self, param_id: Id) -> Optional[Param]:
        for p in self.params:
            if p.id == param_id:
                return p
        return None


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def create_empty_document(name: str = "Untitled", units: str = "mm") -> AxiDocument:
    now = _now_iso()
    return AxiDocument(meta=DocumentMeta(name=name, units=units, createdAt=now, modifiedAt=now))
