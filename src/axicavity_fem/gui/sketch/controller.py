"""スケッチのコントローラ（移植元: EM-CAD-py `emcad/sketch/controller.py`、元は TS 版
`src/sketch/SketchController.ts`）.

ver3 の変更: 閉領域は store が純 Python の ``core.sketch.profiles`` で求めたもの（``store.profiles``）を使う
（EM-CAD-py は OCCT 版 ``kernel.regions``）。スケッチ平面のフレーム（3D 用）は無い。

ストア（DocumentStore）の状態変化を監視して表示状態（閉領域・注釈）を更新し、
2D エディタ（ui.sketch_view.SketchView）からのポインタ/キー入力をツールへ配る。
ツールが使う :class:`~axicavity_fem.gui.sketch.tools.ToolContext` の実装でもある。
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore

from ..app.store import DocumentStore, safe_params
from ..core.document import Id, SketchFeature, SketchPoint, Vec2
from ..core.sketch.planarize import default_tolerance
from ..core.sketch.profiles import Profile, profile_at
from ..core.sketch.snap import SnapResult, choose_grid_spacing, snap_point
from .annotations import Annotations, build_annotations
from .tools import BaseTool, PointerInfo, PreviewGeometry, SelectTool, create_tools

SNAP_PIXELS = 8


class SketchController(QtCore.QObject):
    """Signals:
        display_changed: 表示（エンティティ・プレビュー・スナップ・注釈）を描き直す
    """

    display_changed = QtCore.Signal()

    def __init__(self, store: DocumentStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.tools = create_tools(self)
        self.tool: BaseTool = self.tools["select"]
        self.active = False
        self.profiles: list[Profile] = []
        self.annotations = Annotations()
        self.preview: Optional[PreviewGeometry] = None
        self.snap: Optional[SnapResult] = None
        self.world_per_pixel = 1.0
        self.grid_spacing = 10.0
        self._profiles_version = -1
        self._version_at_down = -1
        self._points_at_down: set[Id] = set()
        self._ctrl_at_down = False
        store.sketch_mode_changed.connect(self._on_sketch_mode)
        store.tool_changed.connect(self._on_tool_changed)
        store.document_changed.connect(self._on_document_changed)
        store.selection_changed.connect(self._on_selection_changed)
        store.tool_options_changed.connect(lambda: self.display_changed.emit())

    # ------------------------------------------------------------------
    # モードの出入り
    # ------------------------------------------------------------------

    def _on_sketch_mode(self, sketch_id) -> None:
        if sketch_id is None:
            if self.active:
                self._exit()
            return
        if self.active:
            self._exit()
        self._enter(sketch_id)

    def _enter(self, sketch_id: Id) -> None:
        sketch = self.store.find_sketch(sketch_id)
        if sketch is None:
            return
        self.active = True
        self._profiles_version = -1
        self.tool = self.tools[self.store.active_tool]
        self.tool.activate()
        self._refresh_profiles()
        self.display_changed.emit()

    def _exit(self) -> None:
        self.active = False
        self.tool.deactivate()
        self.preview = None
        self.snap = None
        self.profiles = []
        self.annotations = Annotations()
        self.display_changed.emit()

    # ------------------------------------------------------------------
    # ストアの変化
    # ------------------------------------------------------------------

    def _on_tool_changed(self, tool_id: str) -> None:
        if not self.active or self.tool.id == tool_id:
            return
        previous = self.tool
        self.tool = self.tools[tool_id]
        previous.deactivate()
        self.tool.activate()
        self.snap = None
        self.display_changed.emit()

    def _on_document_changed(self) -> None:
        if not self.active:
            return
        if self.store.active_sketch() is None:
            return
        self._refresh_profiles()
        self.display_changed.emit()

    def _on_selection_changed(self) -> None:
        if self.active:
            self._update_annotations()
            self.display_changed.emit()

    def _refresh_profiles(self) -> None:
        sketch = self.store.active_sketch()
        if sketch is None:
            return
        if self._profiles_version != self.store.sketch_version:
            self._profiles_version = self.store.sketch_version
            self.profiles = self.store.profiles          # store が文書更新のたびに求めている
            self.store.refresh_solve_info()
        self._update_annotations()

    def _update_annotations(self) -> None:
        sketch = self.store.active_sketch()
        if sketch is None:
            return
        self.annotations = build_annotations(
            sketch, safe_params(self.store.document), self.store.selected_constraint_id,
            self.world_per_pixel)

    # ------------------------------------------------------------------
    # ビューからの入力
    # ------------------------------------------------------------------

    def set_scale(self, world_per_pixel: float) -> None:
        """ビューのズームが変わったときに呼ぶ（スナップ許容差・グリッド・注釈に影響）."""
        if world_per_pixel <= 0:
            return
        changed = abs(world_per_pixel - self.world_per_pixel) / self.world_per_pixel > 1e-6
        self.world_per_pixel = world_per_pixel
        self.grid_spacing = choose_grid_spacing(world_per_pixel)
        if changed and self.active:
            self._update_annotations()

    def _pointer_info(self, raw: Vec2, shift: bool, ctrl: bool) -> PointerInfo:
        sketch = self.store.active_sketch()
        if sketch is None:
            return PointerInfo(raw, SnapResult(pos=raw, kind="none"), shift, ctrl)
        grid = self.grid_spacing if self.store.tool_options.snap_to_grid else None
        snap = snap_point(sketch, raw, tolerance=self.tolerance(), grid_spacing=grid,
                          reference=self.tool.snap_reference(),
                          exclude=self.tool.snap_exclude())
        return PointerInfo(raw, snap, shift, ctrl)

    def pointer_move(self, raw: Vec2, shift: bool = False, ctrl: bool = False) -> None:
        if not self.active:
            return
        info = self._pointer_info(raw, shift, ctrl)
        self.store.set_cursor(info.snap.pos)
        self.tool.on_move(info)
        if self.tool.id == "select" and not self.tool.busy() and self.store.hover_id is None:
            prof = profile_at(self.profiles, raw)
            self.store.set_hover_profile(prof.id if prof else None)
        else:
            self.store.set_hover_profile(None)
        self.snap = info.snap if self.tool.id != "select" else None
        self.display_changed.emit()

    def pointer_down(self, raw: Vec2, shift: bool = False, ctrl: bool = False) -> None:
        if not self.active:
            return
        self._version_at_down = self.store.sketch_version
        self._points_at_down = self._point_ids()
        self._ctrl_at_down = ctrl
        self.tool.on_down(self._pointer_info(raw, shift, ctrl))
        self.display_changed.emit()

    def pointer_up(self, raw: Vec2, shift: bool = False, ctrl: bool = False) -> None:
        if not self.active:
            return
        self.tool.on_up(self._pointer_info(raw, shift, ctrl))
        self._auto_planarize(self._version_at_down)
        self._auto_axis_constraints(self._points_at_down, self._ctrl_at_down or ctrl)
        self.display_changed.emit()

    def _auto_planarize(self, version_before: int) -> None:
        """作図の確定・ドラッグ終了でスケッチが変わっていたら T 字・交差・重なりを分割する（同じ Undo 1 回にまとめる）."""
        if not self.tool.auto_postprocess:
            return
        if self.store.auto_planarize and self.store.sketch_version != version_before:
            self.store.planarize(history=False)

    def _point_ids(self) -> set[Id]:
        sketch = self.store.active_sketch()
        return {e.id for e in sketch.entities if isinstance(e, SketchPoint)} if sketch is not None else set()

    def _auto_axis_constraints(self, points_before: set[Id], ctrl: bool) -> None:
        """作図で新しくできた点が軸（r = 0 / z = 0）の上にあれば軸に拘束する（Ctrl を押していれば付けない。
        選択ツールのドラッグは対象外）. 作図と同じ Undo 1 回にまとめる."""
        if ctrl or self.tool.id == "select" or not self.tool.auto_postprocess:
            return
        new_points = [pid for pid in self._point_ids() if pid not in points_before]
        if new_points:
            self.store.add_axis_constraints_for(new_points, default_tolerance(self.sketch()), history=False)

    def pointer_leave(self) -> None:
        self.store.set_cursor(None)
        self.store.set_hover_profile(None)
        self.snap = None
        self.display_changed.emit()

    def finish(self) -> None:
        """Enter / 右クリック."""
        if self.active:
            version = self.store.sketch_version
            points = self._point_ids()
            self.tool.on_finish()
            self._auto_planarize(version)
            self._auto_axis_constraints(points, False)
            self.display_changed.emit()

    def cancel(self) -> None:
        """Escape: 作図中なら破棄、そうでなければ選択ツールへ、選択ツールなら選択解除."""
        if not self.active:
            return
        if self.tool.busy():
            self.tool.on_cancel()
        elif self.tool.id != "select":
            self.store.set_tool("select")
        else:
            self.store.set_selection([])
        self.display_changed.emit()

    def delete_selection(self) -> None:
        if self.active and isinstance(self.tool, SelectTool):
            self.tool.delete_selection()

    def numeric(self, name: str, value: float) -> None:
        """プロパティパネルの数値入力（length / radius）をツールへ届ける."""
        if self.active:
            version = self.store.sketch_version
            points = self._point_ids()
            self.tool.on_numeric(name, value)
            self._auto_planarize(version)
            self._auto_axis_constraints(points, False)
            self.display_changed.emit()

    # ------------------------------------------------------------------
    # ToolContext
    # ------------------------------------------------------------------

    def sketch(self) -> SketchFeature:
        s = self.store.active_sketch()
        if s is None:
            raise RuntimeError("No active sketch")
        return s

    def update(self, fn, history: bool = True) -> None:
        self.store.update_active_sketch(fn, history=history)

    def restore(self, snapshot: SketchFeature, history: bool = True) -> None:
        self.store.replace_active_sketch(snapshot, history=history)

    def tolerance(self) -> float:
        return self.world_per_pixel * SNAP_PIXELS

    def sides(self) -> int:
        return self.store.tool_options.sides

    def fillet_radius(self) -> float:
        return self.store.tool_options.fillet_radius

    def set_fillet_radius(self, value: float) -> None:
        self.store.set_tool_options(fillet_radius=float(value))

    def refresh_solve(self) -> None:
        self.store.refresh_solve_info()

    def set_preview(self, preview: Optional[PreviewGeometry]) -> None:
        self.preview = preview

    def set_hint(self, key: Optional[str]) -> None:
        self.store.set_hint(key)

    def selection(self) -> list[Id]:
        return list(self.store.selection)

    def set_selection(self, ids: list[Id]) -> None:
        self.store.set_selection(ids)

    def set_hover(self, id: Optional[Id]) -> None:
        self.store.set_hover(id)

    def add_constraint(self, constraint) -> bool:
        return self.store.add_constraint(constraint)

    def merge_points(self, keep: Id, remove: Id) -> None:
        self.store.merge_points(keep, remove)

    def drag_solve(self, snapshot: SketchFeature, fixed: dict[Id, Vec2],
                   history: bool = False) -> bool:
        return self.store.drag_solve(snapshot, fixed, history=history)

    def has_constraints(self) -> bool:
        return self.store.has_constraints()

    def select_constraint(self, id: Optional[Id]) -> None:
        self.store.set_selected_constraint(id)

    def clear_point_expressions(self, ids: list[Id]) -> None:
        self.store.clear_point_expressions(ids, history=False)

    def convert_line_to_arc(self, line_id: Id) -> Optional[Id]:
        return self.store.convert_line_to_arc(line_id)

    def convert_arc_to_line(self, arc_id: Id) -> Optional[Id]:
        return self.store.convert_arc_to_line(arc_id)
