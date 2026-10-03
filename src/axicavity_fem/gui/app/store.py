"""ドキュメントと編集状態の管理（EM-CAD-py ``emcad/app/store.py`` の 2D 部分を ver3 向けに整えたもの）.

Qt のシグナルで UI に変更を知らせる。ドキュメントは**直接変更**し、Undo はドキュメント全体の
deepcopy を積む（EM-CAD-py と同じ）::

    store.update_document(lambda doc: ..., history=True)
    store.update_active_sketch(lambda sketch: ..., history=True)

fn はドキュメント / スケッチを変更して「変わったか」を返す（None は変わった扱い）。

ver3 ではスケッチは常に 1 枚（``document.sketch``）で「スケッチ作成 / 終了」の概念が無い。転用した
コントローラ・パネルは ``sketch_mode_changed(id)`` で編集を始めるので、``load_document`` /
``new_document`` の後に 1 回だけ出す。文書更新のたびに:

    1. 式付きの点座標をパラメータで評価して書き戻す（``convert.apply_document_expressions``）
    2. 閉領域を求め、領域設定を対応付け直す（``convert.match_regions``: key → アンカー → 類似度）
    3. 存在しない曲線の境界条件の指定を消す

を行い、``document_changed`` を出す。曲線の ID が変わる編集（点挿入・再接続・線⇄円弧）は
``edit_ops`` の ID 対応表で境界条件の指定と領域のキーを付け替える。
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass, field
from typing import Callable, Optional

from PySide6 import QtCore

from ..core import convert
from ..core.convert import ResolvedRegion, apply_document_expressions, match_regions, sketch_profiles
from ..core.document import (
    BC_NAMES,
    UNITS,
    AxiDocument,
    Id,
    Param,
    RegionSetting,
    SketchArc,
    SketchCircle,
    SketchConstraint,
    SketchFeature,
    SketchLine,
    SketchPoint,
    Vec2,
    create_empty_document,
    is_dimension_constraint,
    is_reference,
    new_id,
)
from ..core.expressions import ExpressionError, evaluate_expression, expr_or_none, resolve_params
from ..core.sketch import edit_ops
from ..core.sketch.model import constraint_entity_ids
from ..core.sketch.model import merge_points as merge_sketch_points
from ..core.sketch.model import set_construction as set_sketch_construction
from ..core.sketch.profiles import Profile
from ..core.sketch.planarize import planarize as planarize_sketch
from ..core.sketch.reference import (
    AXIS_OF,
    axis_constraint,
    axis_constraints_for,
    ensure_reference_geometry,
    implied_redundant,
    on_axis,
)
from ..sketch.solver import SketchSolver, SolveOutcome

MAX_HISTORY = 200

SKETCH_TOOLS = ("select", "line", "rectangle", "circle", "arc", "polygon", "point", "fillet")
CONSTRAINT_TOOLS = ("coincident", "horizontal", "vertical", "parallel", "perpendicular",
                    "tangent", "equal", "concentric", "fix", "symmetric", "pointOnCurve")
OVERLAY_MODES = ("sketch", "physics", "mesh")


@dataclass
class SolveInfo:
    status: str                       # ok / redundant / conflict / failed / invalid
    dof: int
    conflicting: list = field(default_factory=list)
    redundant: list = field(default_factory=list)
    message: Optional[str] = None     # 直前の操作が拒否された理由（i18n キー）
    error: Optional[str] = None


@dataclass
class ToolOptions:
    sides: int = 6
    snap_to_grid: bool = True
    fillet_radius: float = 5.0       # 2D フィレットの半径（文書単位）


def safe_params(doc: AxiDocument) -> dict[str, float]:
    """パラメータの評価（評価できない行は飛ばす。EM-CAD-py の safe_params 相当）."""
    try:
        return resolve_params(doc.params, strict=False)
    except Exception:
        return {}


def _failure_info(out: SolveOutcome) -> SolveInfo:
    message = {"conflict": "solve.conflict", "invalid": "solve.invalid"}.get(out.status, "solve.failed")
    return SolveInfo(out.status, out.dof, list(out.conflicting), list(out.redundant), message, out.error)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


class DocumentStore(QtCore.QObject):
    """ドキュメント + 編集状態。UI はシグナルで追従する."""

    document_changed = QtCore.Signal()              # 内容が変わった（Undo/Redo 含む）
    sketch_mode_changed = QtCore.Signal(object)     # スケッチの ID（文書を読み込んだとき）
    tool_changed = QtCore.Signal(str)
    selection_changed = QtCore.Signal()             # 選択・ホバー・閉領域ホバー・選択拘束・選択領域
    hint_changed = QtCore.Signal(object)
    cursor_changed = QtCore.Signal(object)
    solve_info_changed = QtCore.Signal(object)
    params_edit_changed = QtCore.Signal(bool)
    tool_options_changed = QtCore.Signal()
    settings_changed = QtCore.Signal(str)           # ver3: "mesh" / "analysis" / "post" / "report" / "units"
    overlay_changed = QtCore.Signal(str)            # ver3: sketch / physics / mesh

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document: AxiDocument = create_empty_document()
        ensure_reference_geometry(self.document.sketch)
        self.past: list[AxiDocument] = []
        self.future: list[AxiDocument] = []
        self.active_tool = "select"
        self.selection: list[Id] = []
        self.hover_id: Optional[Id] = None
        self.hover_profile_id: Optional[str] = None
        self.profiles: list[Profile] = []
        self.tool_options = ToolOptions()
        self.hint: Optional[str] = None
        self.cursor: Optional[Vec2] = None
        self.solve_info: Optional[SolveInfo] = None
        self.selected_constraint_id: Optional[Id] = None
        self.selected_region_id: Optional[str] = None
        self.params_edit = False
        self.overlay = "sketch"
        self.auto_planarize = True                   # 作図の確定・ドラッグ終了で T 字・交差・重なりを分割
        self.expression_errors: list[str] = []       # 直前の文書更新で評価できなかった式
        self.solver = SketchSolver()
        self._sketch_version = 0

    # ------------------------------------------------------------------
    # 問い合わせ
    # ------------------------------------------------------------------

    @property
    def sketch_id(self) -> Id:
        return self.document.sketch.id

    def find_sketch(self, sketch_id: Optional[Id]) -> Optional[SketchFeature]:
        return self.document.sketch if sketch_id == self.document.sketch.id else None

    def active_sketch(self) -> Optional[SketchFeature]:
        return self.document.sketch

    @property
    def sketch_version(self) -> int:
        """スケッチが変わるたびに増える（表示側の再計算判定用）."""
        return self._sketch_version

    def can_undo(self) -> bool:
        return bool(self.past)

    def can_redo(self) -> bool:
        return bool(self.future)

    def params(self) -> dict[str, float]:
        return safe_params(self.document)

    # ------------------------------------------------------------------
    # ドキュメント更新・Undo
    # ------------------------------------------------------------------

    def update_document(self, fn: Callable[[AxiDocument], Optional[bool]], history: bool = True) -> bool:
        snapshot = copy.deepcopy(self.document) if history else None
        changed = fn(self.document)
        if changed is False:
            return False
        self.document.meta.modifiedAt = _now_iso()
        if history:
            self.past.append(snapshot)
            del self.past[:-MAX_HISTORY]
            self.future.clear()
        self._after_document_change()
        return True

    def update_active_sketch(self, fn: Callable[[SketchFeature], Optional[bool]], history: bool = True) -> bool:
        sketch = self.document.sketch
        return self.update_document(lambda _doc: fn(sketch), history=history)

    def replace_active_sketch(self, snapshot: SketchFeature, history: bool = True) -> bool:
        """スケッチの内容をスナップショット（deepcopy）で置き換える（ドラッグの戻し用）."""
        def apply(sketch: SketchFeature) -> bool:
            sketch.entities = copy.deepcopy(snapshot.entities)
            sketch.constraints = copy.deepcopy(snapshot.constraints)
            return True
        return self.update_active_sketch(apply, history=history)

    def undo(self) -> bool:
        if not self.past:
            return False
        self.future.insert(0, self.document)
        self.document = self.past.pop()
        self._after_document_change()
        return True

    def redo(self) -> bool:
        if not self.future:
            return False
        self.past.append(self.document)
        self.document = self.future.pop(0)
        self._after_document_change()
        return True

    def new_document(self, name: str = "Untitled", units: str = "mm") -> None:
        self.load_document(create_empty_document(name, units))

    def load_document(self, document: AxiDocument) -> None:
        ensure_reference_geometry(document.sketch)          # 原点・z 軸・r 軸（古いファイルにも足す）
        self.document = document
        self.past.clear()
        self.future.clear()
        self._clean_modes()
        self._after_document_change()
        self.sketch_mode_changed.emit(self.document.sketch.id)
        self.tool_changed.emit(self.active_tool)
        self.refresh_solve_info()

    def _after_document_change(self) -> None:
        doc = self.document
        self.expression_errors = apply_document_expressions(doc, safe_params(doc))
        self._reconcile_regions()
        self._prune_boundaries()
        self._reconcile()
        self._sketch_version += 1
        self.document_changed.emit()

    def _reconcile_regions(self) -> None:
        """閉領域を求め直し、領域設定の key / anchor を現在の閉領域に合わせる."""
        doc = self.document
        self.profiles = sketch_profiles(doc)
        by_id = {p.id: p for p in self.profiles}
        for profile_id, setting in match_regions(doc, self.profiles).items():
            profile = by_id[profile_id]
            setting.key, setting.anchor = profile.id, profile.anchor

    def _prune_boundaries(self) -> None:
        ids = {e.id for e in self.document.sketch.entities}
        for curve_id in [k for k in self.document.boundaries if k not in ids]:
            del self.document.boundaries[curve_id]

    def _reconcile(self) -> None:
        """Undo/Redo 後に選択がドキュメントと矛盾しないよう整える."""
        sketch = self.document.sketch
        ids = {e.id for e in sketch.entities}
        self.selection = [i for i in self.selection if i in ids]
        if self.hover_id is not None and self.hover_id not in ids:
            self.hover_id = None
        cids = {c.id for c in sketch.constraints}
        if self.selected_constraint_id not in cids:
            self.selected_constraint_id = None
        if self.selected_region_id is not None and \
                self.selected_region_id not in {p.id for p in self.profiles}:
            self.selected_region_id = None
        self.selection_changed.emit()

    def _clean_modes(self) -> None:
        self.active_tool = "select"
        self.selection = []
        self.hover_id = None
        self.hover_profile_id = None
        self.hint = None
        self.cursor = None
        self.solve_info = None
        self.selected_constraint_id = None
        self.selected_region_id = None
        self.params_edit = False

    # ------------------------------------------------------------------
    # ツール・選択・ヒント
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        if tool == self.active_tool:
            return
        self.active_tool = tool
        self.hover_id = None
        self.tool_changed.emit(tool)
        self.selection_changed.emit()

    def set_selection(self, ids: list[Id]) -> None:
        self.selection = list(ids)
        self.selection_changed.emit()

    def set_hover(self, id: Optional[Id]) -> None:
        if self.hover_id != id:
            self.hover_id = id
            self.selection_changed.emit()

    def set_hover_profile(self, id: Optional[str]) -> None:
        if self.hover_profile_id != id:
            self.hover_profile_id = id
            self.selection_changed.emit()

    def set_selected_region(self, profile_id: Optional[str]) -> None:
        if self.selected_region_id != profile_id:
            self.selected_region_id = profile_id
            self.selection_changed.emit()

    def set_tool_options(self, **patch) -> None:
        for key, value in patch.items():
            setattr(self.tool_options, key, value)
        self.tool_options_changed.emit()

    def set_hint(self, hint: Optional[str]) -> None:
        if self.hint != hint:
            self.hint = hint
            self.hint_changed.emit(hint)

    def set_cursor(self, cursor: Optional[Vec2]) -> None:
        self.cursor = cursor
        self.cursor_changed.emit(cursor)

    def set_params_edit(self, editing: bool) -> None:
        self.params_edit = editing
        self.params_edit_changed.emit(editing)

    def set_overlay(self, mode: str) -> None:
        if mode not in OVERLAY_MODES:
            raise ValueError(mode)
        if self.overlay != mode:
            self.overlay = mode
            self.overlay_changed.emit(mode)

    # ------------------------------------------------------------------
    # 拘束
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: SketchConstraint) -> bool:
        """拘束を追加する（矛盾するときは追加せず False）.

        参照要素（原点・軸）だけの拘束は意味が無いので追加しない。点を軸に載せる拘束なら、それで従う水平 / 垂直
        （両端が同じ軸の上になる線）を外す。追加で冗長になった軸への拘束も外す（:meth:`_drop_implied_axis_constraints`）。
        """
        sketch = self.document.sketch
        by_id = {e.id: e for e in sketch.entities}
        try:
            ids = constraint_entity_ids(constraint)
        except ValueError:
            ids = []
        if ids and all(is_reference(by_id.get(i)) for i in ids):
            return False
        signature = {k: v for k, v in vars(constraint).items() if k != "id"}
        if any({k: v for k, v in vars(c).items() if k != "id"} == signature for c in sketch.constraints):
            return False                                # 同じ拘束がもうある（例: 自動で付いた軸の拘束）
        trial = copy.deepcopy(sketch)
        if constraint.type == "pointOnCurve" and constraint.curve in AXIS_OF.values():
            drop = set(implied_redundant(trial, [constraint]))
            trial.constraints = [c for c in trial.constraints if c.id not in drop]
        trial.constraints.append(constraint)
        out = self.solver.solve(trial, safe_params(self.document))
        if not out.ok:
            self.solve_info = _failure_info(out)
            self.solve_info_changed.emit(self.solve_info)
            return False
        if out.status == "redundant":
            trial, out = self._drop_implied_axis_constraints(trial, out, keep={constraint.id})

        def apply(s: SketchFeature) -> bool:
            s.entities = trial.entities
            s.constraints = trial.constraints
            return True
        self.update_active_sketch(apply)
        self.solve_info = SolveInfo(out.status, out.dof, out.conflicting, out.redundant,
                                    "solve.redundant" if out.status == "redundant" else None)
        self.solve_info_changed.emit(self.solve_info)
        return True

    def remove_constraint(self, constraint_id: Id) -> None:
        def apply(s: SketchFeature) -> bool:
            before = len(s.constraints)
            s.constraints = [c for c in s.constraints if c.id != constraint_id]
            return len(s.constraints) != before
        self.update_active_sketch(apply)
        if self.selected_constraint_id == constraint_id:
            self.set_selected_constraint(None)
        self.refresh_solve_info()

    def set_constraint_value(self, constraint_id: Id, value: str) -> bool:
        sketch = self.document.sketch
        trial = copy.deepcopy(sketch)
        for c in trial.constraints:
            if c.id == constraint_id and is_dimension_constraint(c):
                c.value = value
        out = self.solver.solve(trial, safe_params(self.document))
        if not out.ok:
            self.solve_info = _failure_info(out)
            self.solve_info_changed.emit(self.solve_info)
            return False

        def apply(s: SketchFeature) -> bool:
            s.entities = trial.entities
            s.constraints = trial.constraints
            return True
        self.update_active_sketch(apply)
        self.solve_info = SolveInfo(out.status, out.dof, out.conflicting, out.redundant)
        self.solve_info_changed.emit(self.solve_info)
        return True

    def merge_points(self, keep: Id, remove: Id) -> None:
        def apply(s: SketchFeature) -> bool:
            if not merge_sketch_points(s, keep, remove):
                return False
            drop = set(implied_redundant(s, []))       # 例: 原点に一致させて両端が軸の上になった線の水平
            if drop:
                s.constraints = [c for c in s.constraints if c.id not in drop]
            return True
        self.update_active_sketch(apply)
        self.refresh_solve_info()

    def _drop_implied_axis_constraints(self, trial: SketchFeature, out, keep=()):
        """冗長と判定された軸への拘束を 1 本ずつ外す. 外しても自由度が増えないもの（ほかの拘束から従うもの）だけ.

        例: 軸の上の円の中心に、原点の円との同心を付けた。自動で付いた軸の拘束は同心から従うので要らない。
        """
        params = safe_params(self.document)
        candidates = [c.id for c in reversed(trial.constraints)          # 新しく付けたものから
                      if c.type == "pointOnCurve" and c.curve in AXIS_OF.values() and c.id not in keep]
        for cid in candidates:
            if out.status != "redundant" or cid not in out.redundant:
                continue
            test = copy.deepcopy(trial)
            test.constraints = [c for c in test.constraints if c.id != cid]
            res = self.solver.solve(test, params)
            if res.ok and res.dof == out.dof:
                trial, out = test, res
        return trial, out

    def set_selected_constraint(self, constraint_id: Optional[Id]) -> None:
        if self.selected_constraint_id != constraint_id:
            self.selected_constraint_id = constraint_id
            self.selection_changed.emit()

    def toggle_construction(self, ids: list[Id]) -> None:
        sketch = self.document.sketch
        if not ids:
            return
        curves = [e for e in sketch.entities if e.type != "point" and e.id in ids]
        if not curves:
            return
        construction = not all(e.construction for e in curves)   # 全て構築なら通常へ
        self.update_active_sketch(lambda s: set_sketch_construction(s, ids, construction))

    def drag_solve(self, snapshot: SketchFeature, fixed: dict[Id, Vec2], history: bool = False) -> bool:
        """ドラッグ: スナップショットに戻してから固定点つきで解き、成功なら反映する
        （history は最初の移動だけ True にして 1 回分の Undo にする）。失敗なら False."""
        trial = copy.deepcopy(snapshot)
        # 軸に拘束した点はカーソルを軸の上に射影して動かす（軸に沿って滑る。そのまま固定すると拘束とぶつかる）
        on_z = {c.point for c in trial.constraints if c.type == "pointOnCurve" and c.curve == AXIS_OF["r"]}
        on_r = {c.point for c in trial.constraints if c.type == "pointOnCurve" and c.curve == AXIS_OF["z"]}
        fixed = {pid: (0.0 if pid in on_r else pos[0], 0.0 if pid in on_z else pos[1]) for pid, pos in fixed.items()}
        out = self.solver.solve(trial, safe_params(self.document), fixed_points=fixed)
        if not out.ok:
            return False

        def apply(s: SketchFeature) -> bool:
            s.entities = trial.entities
            s.constraints = trial.constraints
            return True
        self.update_active_sketch(apply, history=history)
        return True

    def has_constraints(self) -> bool:
        s = self.document.sketch
        return bool(s.constraints) or any(e.type == "point" and e.fixed for e in s.entities)

    def refresh_solve_info(self) -> None:
        """スケッチを解いて自由度などの情報だけ更新する（ドキュメントは変えない）."""
        sketch = self.document.sketch
        if not sketch.constraints and not any(e.type == "point" and e.fixed for e in sketch.entities):
            # 拘束が無ければ自由度は点ごとに 2（式のある座標を除く）、円/円弧ごとに半径 1
            dof = 0
            for e in sketch.entities:
                if e.projection:                             # 参照ジオメトリは固定
                    continue
                if isinstance(e, SketchPoint):
                    dof += (0 if e.xExpr else 1) + (0 if e.yExpr else 1)
                elif e.type in ("circle", "arc"):
                    dof += 1
            self.solve_info = SolveInfo("ok", dof)
            self.solve_info_changed.emit(self.solve_info)
            return
        trial = copy.deepcopy(sketch)
        out = self.solver.solve(trial, safe_params(self.document))
        message = self.solve_info.message if self.solve_info else None
        self.solve_info = SolveInfo(out.status, out.dof, list(out.conflicting), list(out.redundant),
                                    message, out.error)
        self.solve_info_changed.emit(self.solve_info)

    def _resolve_constraints(self, doc: AxiDocument) -> None:
        """拘束を持つスケッチを現在のパラメータ・座標で解き直す（失敗ならそのまま）."""
        if doc.sketch.constraints:
            self.solver.solve(doc.sketch, safe_params(doc))

    # ------------------------------------------------------------------
    # パラメータ
    # ------------------------------------------------------------------

    def add_param(self) -> Id:
        used = {p.name for p in self.document.params}
        n = 1
        while f"p{n}" in used:
            n += 1
        param = Param(id=new_id(), name=f"p{n}", expression="10")

        def apply(doc: AxiDocument) -> bool:
            doc.params.append(param)
            return True
        self.update_document(apply)
        return param.id

    def update_param(self, param_id: Id, name: Optional[str] = None, expression: Optional[str] = None) -> None:
        def apply(doc: AxiDocument) -> bool:
            for p in doc.params:
                if p.id == param_id:
                    if name is not None:
                        p.name = name
                    if expression is not None:
                        p.expression = expression
            apply_document_expressions(doc, safe_params(doc))
            self._resolve_constraints(doc)
            return True
        self.update_document(apply)
        self.refresh_solve_info()

    def remove_param(self, param_id: Id) -> None:
        def apply(doc: AxiDocument) -> bool:
            doc.params = [p for p in doc.params if p.id != param_id]
            apply_document_expressions(doc, safe_params(doc))
            self._resolve_constraints(doc)
            return True
        self.update_document(apply)
        self.refresh_solve_info()

    def move_param(self, param_id: Id, delta: int) -> None:
        """パラメータ表の行を上下に動かす（評価順が変わる）."""
        def apply(doc: AxiDocument) -> bool:
            ids = [p.id for p in doc.params]
            if param_id not in ids:
                return False
            i = ids.index(param_id)
            j = min(max(i + delta, 0), len(ids) - 1)
            if i == j:
                return False
            doc.params.insert(j, doc.params.pop(i))
            apply_document_expressions(doc, safe_params(doc))
            self._resolve_constraints(doc)
            return True
        self.update_document(apply)
        self.refresh_solve_info()

    # ------------------------------------------------------------------
    # 点・曲線の編集（ver2.3 の Multi-Region Editor にあった操作）
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_ids(doc: AxiDocument, id_map: dict) -> None:
        """曲線の ID が変わったとき、境界条件の指定と領域のキーを新しい ID に付け替える."""
        for old, news in id_map.items():
            bc = doc.boundaries.pop(old, None)
            if bc is not None:
                for new in news:
                    doc.boundaries.setdefault(new, bc)
            for setting in doc.regions:
                ids = setting.key.split("|") if setting.key else []
                if old in ids:
                    ids = [i for i in ids if i != old] + list(news)
                    setting.key = "|".join(sorted(ids))

    def _parse_coord(self, text: Optional[str], current: float, current_expr: Optional[str]
                     ) -> tuple[float, Optional[str]]:
        """入力欄の文字列 → (値, 式)。空なら変えない。数値なら定数（式は消える）。式なら評価して保持."""
        if text is None or not text.strip():
            return current, current_expr
        expr = expr_or_none(text)
        if expr is None:
            return float(text.strip()), None
        return float(evaluate_expression(expr, safe_params(self.document))), expr

    def set_point_coords(self, point_id: Id, z_text: Optional[str], r_text: Optional[str]) -> bool:
        """選択点の Z / R（数値または式）を設定する。式が評価できなければ False（文書は変えない）."""
        sketch = self.document.sketch
        point = next((e for e in sketch.entities if isinstance(e, SketchPoint) and e.id == point_id), None)
        if point is None or point.projection:               # 参照の原点・軸の点は動かせない
            return False
        try:
            x, x_expr = self._parse_coord(z_text, point.x, point.xExpr)
            y, y_expr = self._parse_coord(r_text, point.y, point.yExpr)
        except (ExpressionError, ValueError) as exc:
            self.solve_info = SolveInfo("invalid", 0, message="solve.invalid", error=str(exc))
            self.solve_info_changed.emit(self.solve_info)
            return False
        if (x, y, x_expr, y_expr) == (point.x, point.y, point.xExpr, point.yExpr):
            return False
        # 軸に拘束した座標に 0 以外の値や式を入れたら、その軸の拘束を外す（入れた値で軸から離す。式と拘束はぶつかる）
        release = {AXIS_OF[coord] for coord, text, value, expr in (("z", z_text, x, x_expr), ("r", r_text, y, y_expr))
                   if text is not None and text.strip() and (expr or value != 0.0)}

        def apply(doc: AxiDocument) -> bool:
            p = next(e for e in doc.sketch.entities if e.id == point_id)
            p.x, p.y, p.xExpr, p.yExpr = x, y, x_expr, y_expr
            doc.sketch.constraints = [c for c in doc.sketch.constraints
                                      if not (c.type == "pointOnCurve" and c.point == point_id and c.curve in release)]
            self._resolve_constraints(doc)
            return True
        self.update_document(apply)
        self.refresh_solve_info()
        return True

    def clear_point_expressions(self, ids: list[Id], history: bool = True) -> bool:
        def apply(s: SketchFeature) -> bool:
            changed = False
            for e in s.entities:
                if isinstance(e, SketchPoint) and e.id in ids and not e.projection and (e.xExpr or e.yExpr):
                    e.xExpr = e.yExpr = None
                    changed = True
            return changed
        return self.update_active_sketch(apply, history=history)

    def fix_points_to_zero(self, ids: list[Id], coord: str = "r") -> bool:
        """点を軸に拘束する（coord = "r": z 軸 r = 0 の上、"z": r 軸 z = 0 の上）. 線を渡すと両端の点.

        「点を曲線上に（参照の軸）」の拘束を付けて解く（点は軸に沿っては動ける。ドラッグしても軸から外れない）。
        その座標に式があれば外す（式と拘束がぶつかるため）。両端が同じ軸に乗った線の水平 / 垂直拘束は冗長なので外す。
        解けなければ何も変えず False（理由は solve_info）。
        """
        if coord not in ("z", "r"):
            raise ValueError(coord)
        sketch = self.document.sketch
        by_id = {e.id: e for e in sketch.entities}
        points: list[Id] = []
        for i in ids:
            e = by_id.get(i)
            if e is None or e.projection:
                continue
            for pid in ([e.id] if isinstance(e, SketchPoint) else [e.p1, e.p2] if isinstance(e, SketchLine) else []):
                if pid not in points and not by_id[pid].projection and not on_axis(sketch, pid, coord):
                    points.append(pid)
        if not points:
            return False
        return self._add_axis_constraints([axis_constraint(pid, coord) for pid in points], history=True,
                                          clear_expr=coord)

    def add_axis_constraints_for(self, point_ids: list[Id], tolerance: float, history: bool = False) -> int:
        """作図の自動拘束: 点のうち軸の上にあるものを軸に拘束する（付けた数。解けなければ 0 で何も変えない）."""
        constraints = axis_constraints_for(self.document.sketch, point_ids, tolerance)
        if not constraints:
            return 0
        return len(constraints) if self._add_axis_constraints(constraints, history=history) else 0

    def _add_axis_constraints(self, constraints: list[SketchConstraint], history: bool,
                              clear_expr: Optional[str] = None) -> bool:
        sketch = self.document.sketch
        trial = copy.deepcopy(sketch)
        if clear_expr is not None:
            targets = {c.point for c in constraints}
            for e in trial.entities:
                if isinstance(e, SketchPoint) and e.id in targets:
                    if clear_expr == "r":                   # 先に軸の上へ（解くときにもう一方の座標を動かさない）
                        e.y, e.yExpr = 0.0, None
                    else:
                        e.x, e.xExpr = 0.0, None
        drop = set(implied_redundant(trial, constraints))
        trial.constraints = [c for c in trial.constraints if c.id not in drop] + list(constraints)
        out = self.solver.solve(trial, safe_params(self.document))
        if not out.ok:
            self.solve_info = _failure_info(out)
            self.solve_info_changed.emit(self.solve_info)
            return False
        if out.status == "redundant":                   # ほかの拘束から既に軸の上に決まっている点の分は付けない
            trial, out = self._drop_implied_axis_constraints(trial, out)

        def apply(s: SketchFeature) -> bool:
            s.entities = trial.entities
            s.constraints = trial.constraints
            return True
        self.update_active_sketch(apply, history=history)
        self.refresh_solve_info()
        return True

    def fix_points_to_axis(self, ids: list[Id]) -> bool:
        """点を軸 r=0 に固定する（= ``fix_points_to_zero(ids, "r")``）."""
        return self.fix_points_to_zero(ids, "r")

    def insert_point_on_curve(self, curve_id: Id, pos: Vec2) -> Optional[Id]:
        holder: dict = {}

        def apply(doc: AxiDocument) -> bool:
            pid, id_map = edit_ops.insert_point_on_curve(doc.sketch, curve_id, pos)
            if pid is None:
                return False
            holder["id"] = pid
            self._remap_ids(doc, id_map)
            return True
        self.update_document(apply)
        if "id" in holder:
            self.set_selection([holder["id"]])
            self.refresh_solve_info()
        return holder.get("id")

    def planarize(self, history: bool = True) -> int:
        """T 字・交差・重なりで曲線を分割し重複を消す（リボンの「交差で分割」と作図の自動分割）. 変えた箇所の数."""
        holder: dict = {"n": 0}

        def apply(doc: AxiDocument) -> bool:
            n, id_map = planarize_sketch(doc.sketch)
            holder["n"] = n
            self._remap_ids(doc, id_map)
            return n > 0
        if self.update_document(apply, history=history):
            self.refresh_solve_info()
        return holder["n"]

    def delete_points_reconnect(self, ids: list[Id]) -> bool:
        def apply(doc: AxiDocument) -> bool:
            changed, id_map = edit_ops.delete_points_reconnect(doc.sketch, ids)
            self._remap_ids(doc, id_map)
            return changed
        changed = self.update_document(apply)
        if changed:
            self.set_selection([])
            self.set_hover(None)
            self.refresh_solve_info()
        return changed

    def convert_line_to_arc(self, line_id: Id) -> Optional[Id]:
        holder: dict = {}

        def apply(doc: AxiDocument) -> bool:
            arc_id, id_map = edit_ops.line_to_arc(doc.sketch, line_id)
            if arc_id is None:
                return False
            holder["id"] = arc_id
            self._remap_ids(doc, id_map)
            return True
        self.update_document(apply)
        if "id" in holder:
            self.set_selection([holder["id"]])
            self.refresh_solve_info()
        return holder.get("id")

    def convert_arc_to_line(self, arc_id: Id) -> Optional[Id]:
        holder: dict = {}

        def apply(doc: AxiDocument) -> bool:
            line_id, id_map = edit_ops.arc_to_line(doc.sketch, arc_id)
            if line_id is None:
                return False
            holder["id"] = line_id
            self._remap_ids(doc, id_map)
            return True
        self.update_document(apply)
        if "id" in holder:
            self.set_selection([holder["id"]])
            self.refresh_solve_info()
        return holder.get("id")

    def flip_arc(self, arc_id: Id) -> bool:
        """選択円弧の膨らむ向きを反対側にする（境界条件・領域設定はそのまま）."""
        return self.update_document(lambda doc: edit_ops.flip_arc(doc.sketch, arc_id))

    def set_arc_center(self, arc_id: Id, z_text: Optional[str], r_text: Optional[str]) -> bool:
        """選択円弧の中心 Z / R（数値または式）を設定する."""
        sketch = self.document.sketch
        arc = next((e for e in sketch.entities if isinstance(e, SketchArc) and e.id == arc_id), None)
        if arc is None:
            return False
        center = next(e for e in sketch.entities if e.id == arc.center)
        try:
            x, x_expr = self._parse_coord(z_text, center.x, center.xExpr)
            y, y_expr = self._parse_coord(r_text, center.y, center.yExpr)
        except (ExpressionError, ValueError) as exc:
            self.solve_info = SolveInfo("invalid", 0, message="solve.invalid", error=str(exc))
            self.solve_info_changed.emit(self.solve_info)
            return False

        def apply(doc: AxiDocument) -> bool:
            if not edit_ops.set_arc_center(doc.sketch, arc_id, (x, y), x_expr, y_expr):
                return False
            self._resolve_constraints(doc)
            return True
        changed = self.update_document(apply)
        if changed:
            self.refresh_solve_info()
        return changed

    def close_polyline(self) -> Optional[Id]:
        holder: dict = {}

        def apply(doc: AxiDocument) -> bool:
            line_id, _ = edit_ops.close_polyline(doc.sketch)
            if line_id is None:
                return False
            holder["id"] = line_id
            return True
        self.update_document(apply)
        return holder.get("id")

    # ------------------------------------------------------------------
    # 物理（領域の材料・境界条件）
    # ------------------------------------------------------------------

    def region_setting(self, profile_id: str) -> Optional[RegionSetting]:
        return next((s for s in self.document.regions if s.key == profile_id), None)

    def resolve_regions(self) -> list[ResolvedRegion]:
        return convert.resolve_regions(self.document, self.profiles, safe_params(self.document))

    def set_region(self, profile_id: str, **patch) -> bool:
        """閉領域の材料設定（name / materialTag / epsR / epsRExpr / muR / tanDelta / tanDeltaExpr）を更新する
        （無ければ作る）。``epsR`` / ``tanDelta`` に文字列を渡すと式として扱う."""
        allowed = ("name", "materialTag", "epsR", "epsRExpr", "muR", "tanDelta", "tanDeltaExpr")
        for key in patch:
            if key not in allowed:
                raise KeyError(key)
        profile = next((p for p in self.profiles if p.id == profile_id), None)
        if profile is None:
            return False
        values = dict(patch)
        try:
            for key, expr_key in (("epsR", "epsRExpr"), ("tanDelta", "tanDeltaExpr")):
                if isinstance(values.get(key), str):
                    text = values[key]
                    expr = expr_or_none(text)
                    values[key] = float(evaluate_expression(text, safe_params(self.document))) if expr \
                        else float(text)
                    values[expr_key] = expr
        except (ExpressionError, ValueError) as exc:
            self.solve_info = SolveInfo("invalid", 0, message="solve.invalid", error=str(exc))
            self.solve_info_changed.emit(self.solve_info)
            return False

        def apply(doc: AxiDocument) -> bool:
            existing = next((s for s in doc.regions if s.key == profile_id), None)
            changed = existing is None
            if existing is None:
                names = {s.name for s in doc.regions}
                base, n, name = "Vacuum", 1, "Vacuum"
                while name in names:
                    n += 1
                    name = f"{base}{n}"
                existing = RegionSetting(key=profile.id, anchor=profile.anchor, name=name,
                                         materialTag=convert.normalize_material_tag(name))
                doc.regions.append(existing)
            for key, value in values.items():
                if getattr(existing, key) != value:
                    setattr(existing, key, value)
                    changed = True
            return changed
        return self.update_document(apply)

    def remove_region_setting(self, key: str) -> bool:
        def apply(doc: AxiDocument) -> bool:
            before = len(doc.regions)
            doc.regions = [s for s in doc.regions if s.key != key]
            return len(doc.regions) != before
        return self.update_document(apply)

    def effective_bc(self, curve_id: Id) -> tuple[str, str]:
        return convert.effective_bc(self.document, curve_id, self.profiles)

    def set_boundary(self, curve_ids: list[Id], bc: Optional[str]) -> bool:
        """曲線に境界条件を明示指定する（``bc=None`` は自動判定に戻す）."""
        if bc is not None and bc not in BC_NAMES:
            raise ValueError(bc)
        curves = {e.id for e in self.document.sketch.entities
                  if isinstance(e, (SketchLine, SketchArc, SketchCircle))}
        ids = [i for i in curve_ids if i in curves]
        if not ids:
            return False

        def apply(doc: AxiDocument) -> bool:
            changed = False
            for i in ids:
                if bc is None:
                    changed |= doc.boundaries.pop(i, None) is not None
                elif doc.boundaries.get(i) != bc:
                    doc.boundaries[i] = bc
                    changed = True
            return changed
        return self.update_document(apply)

    # ------------------------------------------------------------------
    # 設定
    # ------------------------------------------------------------------

    def _patch_settings(self, section: str, patch: dict) -> bool:
        def apply(doc: AxiDocument) -> bool:
            target = getattr(doc, section)
            changed = False
            for key, value in patch.items():
                if not hasattr(target, key):
                    raise KeyError(key)
                if getattr(target, key) != value:
                    setattr(target, key, value)
                    changed = True
            return changed
        changed = self.update_document(apply)
        if changed:
            self.settings_changed.emit(section)
        return changed

    def set_mesh(self, **patch) -> bool:
        return self._patch_settings("mesh", patch)

    def set_analysis(self, **patch) -> bool:
        return self._patch_settings("analysis", patch)

    def set_post(self, **patch) -> bool:
        return self._patch_settings("post", patch)

    def set_report(self, **patch) -> bool:
        return self._patch_settings("report", patch)

    def set_view(self, **patch) -> bool:
        return self._patch_settings("view", patch)

    def set_units(self, units: str, convert_values: bool = False) -> bool:
        """単位を変える。``convert_values`` なら座標・半径・メッシュサイズを換算する（式は換算しない）."""
        if units not in UNITS:
            raise ValueError(units)
        if units == self.document.meta.units:
            return False
        factor = convert.UNIT_TO_M[self.document.meta.units] / convert.UNIT_TO_M[units]

        def apply(doc: AxiDocument) -> bool:
            doc.meta.units = units
            if convert_values:
                for e in doc.sketch.entities:
                    if isinstance(e, SketchPoint) and not e.projection:      # 参照の原点・軸は換算しない
                        e.x *= factor
                        e.y *= factor
                    elif isinstance(e, (SketchCircle, SketchArc)):
                        e.radius *= factor
                doc.mesh.size *= factor
                for key in ("zmin", "zmax", "rmin", "rmax"):
                    value = getattr(doc.view, key)
                    if value is not None:
                        setattr(doc.view, key, value * factor)
                for s in doc.regions:
                    s.anchor = (s.anchor[0] * factor, s.anchor[1] * factor)
            return True
        self.update_document(apply)
        self.settings_changed.emit("units")
        return True
