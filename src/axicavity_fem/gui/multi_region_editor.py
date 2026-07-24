"""ver2.1 段階3: 多領域メッシュ編集パネル (MVP)。

PointLineEditor と並列の wx.Panel。MultiRegionGeometry を直接編集対象とし、
matplotlib キャンバスで点・セグメント・ループ・領域を可視化する。

MVP スコープ:
    - 点の追加 / ドラッグ移動 / 削除 (Add Points, Edit Points)
    - 直線 segment の追加 / 削除 / BC 変更 (Add Segments, Edit BC)
    - ループの構築 (Build Loop) と外周/穴の向きラベル
    - 領域定義 (右ペインのフォーム経由)
    - Gmsh 出力 (shared.gmsh_export_occ 経由)

未対応 (後続): 円弧、ループ反転、Superfish インポート、領域塗りつぶし。
"""

from __future__ import annotations

import math
import os
from typing import Optional

import wx
import matplotlib
matplotlib.use("WXAgg")
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

from ..shared.multi_region_model import (
    ALLOWED_BC_NAMES,
    BC_NAMES,
    BC_NONE,
    DEFAULT_BC,
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)
from ..shared.expression_eval import eval_scalar, EvalError


# モード定数
MODE_AUTO_POINT_SEGMENT = "auto_point_segment"  # Simple mode の既定モード
MODE_ADD_POINTS = "add_points"
MODE_EDIT_POINTS = "edit_points"
MODE_ADD_SEGMENTS = "add_segments"
MODE_BUILD_LOOP = "build_loop"
MODE_EDIT_BC = "edit_bc"

MODES = (
    MODE_AUTO_POINT_SEGMENT,
    MODE_ADD_POINTS,
    MODE_EDIT_POINTS,
    MODE_ADD_SEGMENTS,
    MODE_BUILD_LOOP,
    MODE_EDIT_BC,
)

# BC 表示色
BC_COLORS = {
    "PEC": "#cc6600",      # 濃いオレンジ
    "E-short": "#1f77b4",  # 青
    "M-short": "#2ca02c",  # 緑
    "None": "#444444",     # 黒よりの灰色
}
# BC ごとの線幅: すべて同一基準
BC_LINEWIDTHS = {
    "PEC": 2.5,
    "E-short": 2.5,
    "M-short": 2.5,
    "None": 2.5,
}
# 未知 BC のフォールバック
_DEFAULT_BC_COLOR = "#444444"
_DEFAULT_BC_LW = 2.5
# 選択時に加算する線幅 (色は変えない)
_SELECTED_LW_DELTA = 2.0

# ループ表示色 (建設中のループ)
COLOR_PENDING_LOOP = "#ff7f0e"
COLOR_SELECTED = "#bcbd22"
COLOR_POINT = "red"
COLOR_POINT_SELECTED = "green"

# ヒット判定 (ピクセル単位の二乗距離)
POINT_HIT_PX_SQ = 100.0


def _point_segment_distance_sq(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq < 1e-12:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = ((px - x1) * dx + (py - y1) * dy) / lsq
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2


def _point_segment_dist_t(px, py, x1, y1, x2, y2):
    """点と線分の最短距離の二乗と、線分上のパラメータ t (0..1) を返す。"""
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq < 1e-12:
        return (px - x1) ** 2 + (py - y1) ** 2, 0.0
    t = ((px - x1) * dx + (py - y1) * dy) / lsq
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2, t


# ---------------------------------------------------------------------------
# 円弧 (arc) の幾何ヘルパ (PointLineEditor から移植)
# ---------------------------------------------------------------------------
def _minor_arc_angles(p1, p2, center):
    """中心 center に対する端点 p1,p2 の角度から、劣弧 (minor arc) になる
    (theta1, theta2) を度数法で返す。PointLineEditor と完全一致のロジック。

    端点→中心の角度 a1,a2 を求め、角度差を (-180, 180] に正規化して短い方の弧を
    選ぶ。matplotlib patches.Arc は theta1 から theta2 へ反時計回りに描く。
    """
    a1 = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
    a2 = math.degrees(math.atan2(p2[1] - center[1], p2[0] - center[0]))
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
    return theta1, theta2


def arc_params_from_two_points(p1, p2):
    """2 端点 p1=(z,r), p2=(z,r) から劣弧の (center, radius, theta1, theta2)。

    PointLineEditor 互換: 中点から弦に垂直 (-dy, dx) 方向へ |p1p2|/2 だけ
    オフセットした点を中心とし、radius は中心と端点の実距離 (= |p1p2|/√2)。
    """
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-12:
        raise ValueError("2 端点が一致しているため円弧を作れません。")
    # 弦に垂直なベクトル (-dy, dx) を正規化し、|p1p2|/2 だけオフセット
    perp_x, perp_y = -dy / dist, dx / dist
    center = (mx + perp_x * (dist / 2.0), my + perp_y * (dist / 2.0))
    radius = math.hypot(p1[0] - center[0], p1[1] - center[1])
    theta1, theta2 = _minor_arc_angles(p1, p2, center)
    return center, radius, theta1, theta2


def recompute_arc_params(p1, p2, center):
    """新しい中心 center と 2 端点から (radius, theta1, theta2) を再計算する。"""
    radius = math.hypot(p1[0] - center[0], p1[1] - center[1])
    theta1, theta2 = _minor_arc_angles(p1, p2, center)
    return radius, theta1, theta2


def _arc_distance_sq(px, py, seg, points):
    """点 (px,py) と arc segment の最短距離の二乗 (PointLineEditor 移植)。

    劣弧の角度範囲内なら ||p-center| - r|^2、範囲外なら端点までの最短距離。
    """
    cz, cr = seg.center
    radius = seg.radius
    t1, t2 = seg.theta1, seg.theta2
    angle = math.degrees(math.atan2(py - cr, px - cz))
    # angle を [t1, t1+360) に正規化
    while angle < t1:
        angle += 360.0
    while angle >= t1 + 360.0:
        angle -= 360.0
    if t1 <= angle <= t2:
        return (math.hypot(px - cz, py - cr) - radius) ** 2
    p1 = points[seg.point_indices[0]]
    p2 = points[seg.point_indices[1]]
    d1 = (px - p1[0]) ** 2 + (py - p1[1]) ** 2
    d2 = (px - p2[0]) ** 2 + (py - p2[1]) ** 2
    return min(d1, d2)


class MultiRegionEditorPanel(wx.Panel):
    """多領域メッシュ編集の左ペイン (matplotlib キャンバス本体)。"""

    def __init__(self, parent, parent_frame=None, statusbar=None):
        super().__init__(parent)
        self.parent_frame = parent_frame or wx.GetTopLevelParent(self)
        self.statusbar = statusbar

        self.geom = MultiRegionGeometry()
        # 既定で 1 つの "Vacuum" 領域定義の枠を確保しない (空状態)。
        # ループ・領域はユーザ操作で作成する。

        self.mode = MODE_ADD_POINTS
        self.selected_point_index: Optional[int] = None
        self.selected_segment_id: Optional[int] = None
        # Build Loop モードで構築中の segment ID 列
        self.pending_loop_segments: list[int] = []
        # 直前に Add Segments で選択された始点
        self.pending_segment_start: Optional[int] = None

        self.dragging_point = False
        self.is_dirty = False
        # Simple モードのみ: クリック対象に応じて Point Edit ⇄ Line Edit を自動切替
        self.auto_mode_switch = False

        # matplotlib artists のキャッシュ (index/id -> artist)
        self._point_artists: dict[int, "matplotlib.lines.Line2D"] = {}
        self._point_labels: dict[int, "matplotlib.text.Text"] = {}
        self._segment_artists: dict[int, "matplotlib.lines.Line2D"] = {}

        self._init_ui()
        self.redraw()
        self._update_statusbar()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _init_ui(self):
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 0)
        sizer.Add(self.toolbar, 0, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(sizer)

    # ------------------------------------------------------------------
    # 通知系
    # ------------------------------------------------------------------
    def mark_dirty(self):
        self.is_dirty = True
        if self.parent_frame and hasattr(self.parent_frame, "on_mr_editor_changed"):
            self.parent_frame.on_mr_editor_changed()

    def notify_selection_changed(self):
        if self.parent_frame and hasattr(self.parent_frame, "on_mr_selection_changed"):
            self.parent_frame.on_mr_selection_changed()

    def notify_mode_changed(self):
        """自動モード切替を frame に通知し、Simple radiobox 等を同期させる。"""
        if self.parent_frame and hasattr(self.parent_frame, "on_mr_mode_autoswitched"):
            self.parent_frame.on_mr_mode_autoswitched(self.mode)

    def _switch_mode_notify(self, new_mode: str):
        """内部からモードを切替え、frame に通知する (radiobox 同期用)。"""
        if self.mode != new_mode:
            self.mode = new_mode
            self.notify_mode_changed()

    def _update_statusbar(self):
        if not self.statusbar:
            return
        parts = [f"Mode: {self.mode}"]
        if self.selected_point_index is not None:
            parts.append(f"Point: {self.selected_point_index}")
        if self.selected_segment_id is not None:
            parts.append(f"Segment: {self.selected_segment_id}")
        if self.pending_segment_start is not None:
            parts.append(f"Pending start: {self.pending_segment_start}")
        if self.pending_loop_segments:
            parts.append(f"Loop seg-chain: {self.pending_loop_segments}")
        self.statusbar.SetStatusText(" | ".join(parts))

    # ------------------------------------------------------------------
    # 公開 API: モード切替
    # ------------------------------------------------------------------
    def set_mode(self, new_mode: str):
        if new_mode not in MODES:
            raise ValueError(f"unknown mode: {new_mode}")
        if self.mode == new_mode:
            return
        # モード切替で建設中の状態を解除
        self.pending_segment_start = None
        self.pending_loop_segments = []
        self.deselect_point()
        self.deselect_segment()
        self.mode = new_mode
        self.redraw()
        self._update_statusbar()
        self.notify_selection_changed()

    # ------------------------------------------------------------------
    # 公開 API: ジオメトリ設定
    # ------------------------------------------------------------------
    def set_geometry(self, geom: MultiRegionGeometry):
        """既存ジオメトリを置き換える (load 時に使う)。"""
        self.geom = geom
        self.selected_point_index = None
        self.selected_segment_id = None
        self.pending_segment_start = None
        self.pending_loop_segments = []
        self.is_dirty = False
        self.redraw()
        self._update_statusbar()
        self.notify_selection_changed()

    def get_geometry(self) -> MultiRegionGeometry:
        return self.geom

    def reset(self):
        self.set_geometry(MultiRegionGeometry())
        self.mark_dirty()

    def set_axes_limits(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.axes.set_xlim(xmin, xmax)
        self.axes.set_ylim(ymin, ymax)
        self.canvas.draw_idle()

    def set_axes_labels(self, unit: str):
        self.axes.set_xlabel(f"Z [{unit}]")
        self.axes.set_ylabel(f"R [{unit}]")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 検索ヘルパ
    # ------------------------------------------------------------------
    def _find_point_at_pixel(self, px, py) -> Optional[int]:
        for i, (z, r) in enumerate(self.geom.points):
            mpx, mpy = self.axes.transData.transform((z, r))
            if (px - mpx) ** 2 + (py - mpy) ** 2 < POINT_HIT_PX_SQ:
                return i
        return None

    def _find_segment_at(self, x, y) -> Optional[int]:
        xlim = self.axes.get_xlim()
        th_sq = ((xlim[1] - xlim[0]) * 0.02) ** 2
        best_d = float("inf")
        best_sid: Optional[int] = None
        for seg in self.geom.segments:
            p_a, p_b = seg.point_indices
            if seg.type == "arc" and seg.center is not None:
                d = _arc_distance_sq(x, y, seg, self.geom.points)
            else:
                x1, y1 = self.geom.points[p_a]
                x2, y2 = self.geom.points[p_b]
                d = _point_segment_distance_sq(x, y, x1, y1, x2, y2)
            if d < best_d:
                best_d = d
                best_sid = seg.id
        if best_sid is not None and best_d < th_sq:
            return best_sid
        return None

    # ------------------------------------------------------------------
    # キャンバス操作 (クリック / ドラッグ)
    # ------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.axes:
            return
        if self.toolbar.mode:
            # zoom/pan 中はキャンバス操作を抑制
            return
        x, y = event.xdata, event.ydata
        px, py = event.x, event.y

        if self.mode == MODE_AUTO_POINT_SEGMENT:
            # Simple mode の "Edit Points":
            #  - 既存点クリック → 選択 + ドラッグ
            #  - ライン上ダブルクリック → その位置に点を挿入 (Mesh Geometry 同様)
            #  - (auto_mode_switch) ライン上シングルクリック → Line Edit へ自動遷移し選択
            #  - 構築中 (Region 未作成) に空クリック → 末尾点 + 自動 segment 追加
            #  - 構築後 (Region 作成済み) に空クリック → 選択解除
            hit = self._find_point_at_pixel(px, py)
            if hit is not None:
                self._select_point(hit)
                self.dragging_point = True
                self.notify_selection_changed()
                return
            if getattr(event, "dblclick", False):
                # ライン上ダブルクリックで点を挿入
                if self.add_point_on_segment(x, y):
                    return
            # ライン上シングルクリック → Line Edit へ自動遷移 (Simple のみ)
            if self.auto_mode_switch:
                sid = self._find_segment_at(x, y)
                if sid is not None:
                    self.selected_segment_id = sid
                    self.selected_point_index = None
                    self._switch_mode_notify(MODE_EDIT_BC)
                    self.redraw()
                    self._update_statusbar()
                    self.notify_selection_changed()
                    return
            if self.geom.regions:
                # 構築後: 空クリックは選択解除のみ
                self.deselect_point()
                self.deselect_segment()
                self.notify_selection_changed()
                return
            # 構築中: 新規点 + 直前点との自動 segment
            new_idx = self.geom.append_point(x, y)
            if new_idx >= 1:
                sid = self.geom.next_segment_id()
                seg = Segment(id=sid, type="line",
                              point_indices=[new_idx - 1, new_idx],
                              bc_name=DEFAULT_BC)
                self.geom.segments.append(seg)
            self.mark_dirty()
            self.redraw()
            self.notify_selection_changed()

        elif self.mode == MODE_ADD_POINTS:
            hit = self._find_point_at_pixel(px, py)
            if hit is not None:
                self._select_point(hit)
            else:
                self.geom.append_point(x, y)
                self.mark_dirty()
                self.redraw()
            self.notify_selection_changed()

        elif self.mode == MODE_EDIT_POINTS:
            hit = self._find_point_at_pixel(px, py)
            if hit is not None:
                self._select_point(hit)
                self.dragging_point = True
            self.notify_selection_changed()

        elif self.mode == MODE_ADD_SEGMENTS:
            hit = self._find_point_at_pixel(px, py)
            if hit is None:
                return
            if self.pending_segment_start is None:
                self.pending_segment_start = hit
                self._select_point(hit)
            else:
                a, b = self.pending_segment_start, hit
                if a == b:
                    # 同一点の選択解除
                    self.pending_segment_start = None
                    self.deselect_point()
                else:
                    sid = self.geom.next_segment_id()
                    seg = Segment(id=sid, type="line", point_indices=[a, b], bc_name=DEFAULT_BC)
                    self.geom.segments.append(seg)
                    self.pending_segment_start = None
                    self.deselect_point()
                    self.selected_segment_id = sid
                    self.mark_dirty()
                    self.redraw()
            self._update_statusbar()
            self.notify_selection_changed()

        elif self.mode == MODE_BUILD_LOOP:
            sid = self._find_segment_at(x, y)
            if sid is None:
                return
            if sid in self.pending_loop_segments:
                # クリックでトグル: 末尾と一致なら除去
                if self.pending_loop_segments[-1] == sid:
                    self.pending_loop_segments.pop()
            else:
                self.pending_loop_segments.append(sid)
            self.redraw()
            self._update_statusbar()
            self.notify_selection_changed()

        elif self.mode == MODE_EDIT_BC:
            # Simple mode の "Edit Lines" (auto_mode_switch) では:
            #  - 点の近くをクリック → Point Edit へ自動遷移し点を選択 + ドラッグ
            #  - ライン上ダブルクリック → 点を挿入して Point Edit へ遷移
            #  - ライン上シングルクリック → segment 選択 (現状維持)
            if self.auto_mode_switch:
                hit = self._find_point_at_pixel(px, py)
                if hit is not None:
                    self.selected_segment_id = None
                    self._select_point(hit)
                    self.dragging_point = True
                    self._switch_mode_notify(MODE_AUTO_POINT_SEGMENT)
                    self.notify_selection_changed()
                    return
                if getattr(event, "dblclick", False):
                    if self.add_point_on_segment(x, y):
                        # 挿入された点を選択した状態で Point Edit へ
                        self._switch_mode_notify(MODE_AUTO_POINT_SEGMENT)
                        return
            sid = self._find_segment_at(x, y)
            if sid is not None:
                self.selected_segment_id = sid
            else:
                self.selected_segment_id = None
            self.redraw()
            self.notify_selection_changed()

    def on_motion(self, event):
        if not self.dragging_point:
            return
        if event.inaxes != self.axes or self.selected_point_index is None:
            return
        idx = self.selected_point_index
        # ドラッグは式を破棄して定数化 (ver2.2 の確定仕様)
        self.geom.set_point(idx, event.xdata, event.ydata, None, None)
        # 端点が動いた arc は直線に戻す (PointLineEditor と同じ挙動)
        self._revert_arcs_touching_point(idx)
        self.redraw()

    def on_release(self, event):
        if self.dragging_point:
            self.dragging_point = False
            self.mark_dirty()

    # ------------------------------------------------------------------
    # 選択操作
    # ------------------------------------------------------------------
    def _select_point(self, idx: int):
        self.selected_point_index = idx
        self.selected_segment_id = None
        self.redraw()
        self._update_statusbar()

    def deselect_point(self):
        if self.selected_point_index is not None:
            self.selected_point_index = None
            self.redraw()

    def deselect_segment(self):
        if self.selected_segment_id is not None:
            self.selected_segment_id = None
            self.redraw()

    # ------------------------------------------------------------------
    # 編集 API (右ペインから呼ばれる)
    # ------------------------------------------------------------------
    def update_point_coords(self, idx: int, z: float, r: float,
                            z_expr: str | None = None, r_expr: str | None = None):
        if 0 <= idx < len(self.geom.points):
            self.geom.set_point(idx, z, r, z_expr, r_expr)
            self.mark_dirty()
            self.redraw()

    def _reindex_after_point_removal(self, idx: int):
        """点 idx を削除した後、全 segment の point_indices を詰める。"""
        for seg in self.geom.segments:
            seg.point_indices = [pi - 1 if pi > idx else pi
                                 for pi in seg.point_indices]

    def _replace_two_segments_in_loops(self, old_a: int, old_b: int, new_id: int):
        """loop の segment_ids 内の old_a, old_b を new_id 1 つに置換する。

        new_id は最初に現れた位置に挿入し、ループの順序を保つ。
        """
        for lp in self.geom.loops:
            if old_a not in lp.segment_ids and old_b not in lp.segment_ids:
                continue
            new_list: list[int] = []
            inserted = False
            for sid in lp.segment_ids:
                if sid in (old_a, old_b):
                    if not inserted:
                        new_list.append(new_id)
                        inserted = True
                else:
                    new_list.append(sid)
            lp.segment_ids = new_list

    def delete_selected_point(self):
        """選択点を削除する。

        点がちょうど 2 つの segment に接続している場合 (= ループ境界上の点)、
        Mesh Geometry Edit と同様に、前後の点を結ぶ直線 segment を新規生成して
        ループを保つ。それ以外 (端点・分岐点・孤立点) は接続 segment を削除する。
        """
        idx = self.selected_point_index
        if idx is None:
            return
        touching = [s for s in self.geom.segments if idx in s.point_indices]

        # --- ループ境界上の点: 前後を繋いでループ維持 ---
        if len(touching) == 2:
            seg_a, seg_b = touching

            def _other(seg, i):
                a, b = seg.point_indices
                return a if b == i else b

            prev_pt = _other(seg_a, idx)
            next_pt = _other(seg_b, idx)
            old_a, old_b = seg_a.id, seg_b.id
            bc = seg_a.bc_name

            # 2 segment を削除
            self.geom.segments = [s for s in self.geom.segments
                                  if s.id not in (old_a, old_b)]
            # prev と next が異なる点なら橋渡し segment を生成
            if prev_pt != next_pt:
                new_id = self.geom.next_segment_id()
                self.geom.segments.append(Segment(
                    id=new_id, type="line",
                    point_indices=[prev_pt, next_pt], bc_name=bc))
                self._replace_two_segments_in_loops(old_a, old_b, new_id)
            else:
                # 退化ケース (2 点ループ等): 単に loop から両 segment を除去
                for lp in self.geom.loops:
                    lp.segment_ids = [s for s in lp.segment_ids
                                      if s not in (old_a, old_b)]
            # 点削除 + reindex (橋渡し segment も含めて詰める)
            self.geom.pop_point(idx)
            self._reindex_after_point_removal(idx)
            self.selected_point_index = None
            self.selected_segment_id = None
            self.mark_dirty()
            self.redraw()
            self.notify_selection_changed()
            return

        # --- それ以外: 接続 segment を削除する (従来挙動) ---
        if touching:
            msg = (f"点 {idx} は {len(touching)} 個の segment に参照されています。"
                   "それらも一緒に削除してよいですか？")
            if wx.MessageBox(msg, "Confirm", wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
                return
            removed_sids = {s.id for s in touching}
            self.geom.segments = [s for s in self.geom.segments
                                  if s.id not in removed_sids]
            for lp in self.geom.loops:
                lp.segment_ids = [sid for sid in lp.segment_ids
                                  if sid not in removed_sids]
        self.geom.pop_point(idx)
        self._reindex_after_point_removal(idx)
        self.selected_point_index = None
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()

    def delete_selected_segment(self):
        sid = self.selected_segment_id
        if sid is None:
            return
        self.geom.segments = [s for s in self.geom.segments if s.id != sid]
        for lp in self.geom.loops:
            lp.segment_ids = [s for s in lp.segment_ids if s != sid]
        self.selected_segment_id = None
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()

    def set_selected_segment_bc(self, bc_name: str):
        if self.selected_segment_id is None:
            return
        if bc_name not in ALLOWED_BC_NAMES:
            return
        for seg in self.geom.segments:
            if seg.id == self.selected_segment_id:
                seg.bc_name = bc_name
                self.mark_dirty()
                self.redraw()
                break

    # ---- 円弧 (arc) 編集 ------------------------------------------------
    def _selected_segment(self) -> Optional[Segment]:
        if self.selected_segment_id is None:
            return None
        for seg in self.geom.segments:
            if seg.id == self.selected_segment_id:
                return seg
        return None

    def convert_selected_to_arc(self) -> bool:
        """選択中の line segment を劣弧 arc に変換する。"""
        seg = self._selected_segment()
        if seg is None or seg.type == "arc":
            return False
        p1 = self.geom.points[seg.point_indices[0]]
        p2 = self.geom.points[seg.point_indices[1]]
        try:
            center, radius, t1, t2 = arc_params_from_two_points(p1, p2)
        except ValueError:
            return False
        seg.type = "arc"
        seg.center = (float(center[0]), float(center[1]))
        seg.radius = float(radius)
        seg.theta1 = float(t1)
        seg.theta2 = float(t2)
        seg.center_expr = None  # 自動算出中心は式なし (端点追従で再算出される)
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()
        return True

    def convert_selected_to_line(self) -> bool:
        """選択中の arc segment を直線に戻す。"""
        seg = self._selected_segment()
        if seg is None or seg.type == "line":
            return False
        seg.type = "line"
        seg.center = None
        seg.radius = None
        seg.theta1 = None
        seg.theta2 = None
        seg.center_expr = None
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()
        return True

    def update_arc_center(self, cz: float, cr: float,
                          cz_expr: str | None = None, cr_expr: str | None = None) -> bool:
        """選択中 arc の中心を (cz, cr) に変更し radius/theta を再計算する。

        cz_expr/cr_expr を渡すと中心式として保存し、変数変更時に再評価される。
        いずれも None なら「明示中心なし (自動算出扱い)」として center_expr=None。
        """
        seg = self._selected_segment()
        if seg is None or seg.type != "arc":
            return False
        p1 = self.geom.points[seg.point_indices[0]]
        p2 = self.geom.points[seg.point_indices[1]]
        radius, t1, t2 = recompute_arc_params(p1, p2, (cz, cr))
        seg.center = (float(cz), float(cr))
        seg.radius = float(radius)
        seg.theta1 = float(t1)
        seg.theta2 = float(t2)
        seg.center_expr = (cz_expr, cr_expr) if (cz_expr or cr_expr) else None
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()
        return True

    def recompute_from_variables(self, variables: dict) -> list[str]:
        """変数辞書で、式を持つ座標・寸法・材料・arc を再評価し再描画する (ver2.2)。

        - 各点の point_exprs（非 None）を再評価して points[i] を更新（値のみ、式は保持）。
        - mesh_size_expr / 各 Region.eps_r_expr / arc の中心式を再評価。
        - arc は端点追従で radius/theta を再算出（明示中心式があれば中心も再評価、
          無ければ 2 端点から自動算出）。直線化はしない。
        失敗した式は前の値を維持し、エラー文字列のリストを返す。
        """
        errors: list[str] = []
        geom = self.geom

        def _ev(expr, label):
            try:
                return float(eval_scalar(expr, variables)), None
            except EvalError as e:
                return None, f"{label}: {e}"

        # 点座標
        for i, (ze, re_) in enumerate(geom.point_exprs):
            if ze is None and re_ is None:
                continue
            z, r = geom.points[i]
            if ze is not None:
                v, err = _ev(ze, f"point[{i}].z '{ze}'")
                if err:
                    errors.append(err)
                else:
                    z = v
            if re_ is not None:
                v, err = _ev(re_, f"point[{i}].r '{re_}'")
                if err:
                    errors.append(err)
                else:
                    r = v
            geom.points[i] = (float(z), float(r))

        # mesh_size
        if geom.mesh_size_expr:
            v, err = _ev(geom.mesh_size_expr, f"mesh_size '{geom.mesh_size_expr}'")
            if err:
                errors.append(err)
            elif v > 0:
                geom.mesh_size = v
            else:
                errors.append("mesh_size must be > 0")

        # eps_r
        for rg in geom.regions:
            if rg.eps_r_expr:
                v, err = _ev(rg.eps_r_expr, f"region {rg.id} eps_r '{rg.eps_r_expr}'")
                if err:
                    errors.append(err)
                elif v > 0:
                    rg.eps_r = v
                else:
                    errors.append(f"region {rg.id} eps_r must be > 0")

        # tan_delta (ver2.3: 0 を許容するので >= 0)
        for rg in geom.regions:
            if rg.tan_delta_expr:
                v, err = _ev(rg.tan_delta_expr,
                             f"region {rg.id} tan_delta '{rg.tan_delta_expr}'")
                if err:
                    errors.append(err)
                elif v >= 0:
                    rg.tan_delta = v
                else:
                    errors.append(f"region {rg.id} tan_delta must be >= 0")

        # arc: 中心式を再評価し、端点追従で radius/theta を再算出
        for seg in geom.segments:
            if seg.type != "arc":
                continue
            p1 = geom.points[seg.point_indices[0]]
            p2 = geom.points[seg.point_indices[1]]
            if seg.center_expr is not None:
                cz, cr = seg.center if seg.center else (0.0, 0.0)
                cze, cre = seg.center_expr
                if cze is not None:
                    v, err = _ev(cze, f"arc {seg.id} center.z '{cze}'")
                    if err:
                        errors.append(err)
                    else:
                        cz = v
                if cre is not None:
                    v, err = _ev(cre, f"arc {seg.id} center.r '{cre}'")
                    if err:
                        errors.append(err)
                    else:
                        cr = v
                try:
                    radius, t1, t2 = recompute_arc_params(p1, p2, (cz, cr))
                    seg.center = (float(cz), float(cr))
                    seg.radius = float(radius)
                    seg.theta1 = float(t1)
                    seg.theta2 = float(t2)
                except Exception:
                    pass
            else:
                try:
                    center, radius, t1, t2 = arc_params_from_two_points(p1, p2)
                    seg.center = (float(center[0]), float(center[1]))
                    seg.radius = float(radius)
                    seg.theta1 = float(t1)
                    seg.theta2 = float(t2)
                except ValueError:
                    pass

        self.redraw()
        return errors

    def _revert_arcs_touching_point(self, idx: int):
        """点 idx を端点に持つ arc segment を直線に戻す (ドラッグ時用)。"""
        for seg in self.geom.segments:
            if seg.type == "arc" and idx in seg.point_indices:
                seg.type = "line"
                seg.center = None
                seg.radius = None
                seg.theta1 = None
                seg.theta2 = None
                seg.center_expr = None

    def add_point_on_segment(self, x: float, y: float) -> bool:
        """(x,y) に最も近い直線 segment を分割し、その射影位置に点を挿入する。

        Mesh Geometry Edit の add_point_on_segment 相当。挿入により分割された
        segment は 2 本に置き換わり、その segment を参照する loop の
        segment_ids も更新される。BC 名は元 segment から引き継ぐ。

        Returns:
            挿入できたら True、近傍に直線 segment が無ければ False。
        """
        xlim = self.axes.get_xlim()
        th_sq = ((xlim[1] - xlim[0]) * 0.05) ** 2
        best_d = float("inf")
        best_seg = None
        best_t = 0.0
        for seg in self.geom.segments:
            if seg.type != "line":
                continue  # 円弧上への挿入は非対応 (Mesh Geometry も直線のみ)
            p1 = self.geom.points[seg.point_indices[0]]
            p2 = self.geom.points[seg.point_indices[1]]
            d, t = _point_segment_dist_t(x, y, p1[0], p1[1], p2[0], p2[1])
            if d < best_d:
                best_d = d
                best_seg = seg
                best_t = t
        if best_seg is None or best_d >= th_sq:
            return False

        p1_i, p2_i = best_seg.point_indices
        p1 = self.geom.points[p1_i]
        p2 = self.geom.points[p2_i]
        nx = p1[0] + best_t * (p2[0] - p1[0])
        ny = p1[1] + best_t * (p2[1] - p1[1])
        new_idx = self.geom.append_point(nx, ny)

        bc = best_seg.bc_name
        old_id = best_seg.id
        # 元 segment を削除し、分割した 2 本を追加
        self.geom.segments = [s for s in self.geom.segments if s.id != old_id]
        sid1 = self.geom.next_segment_id()
        seg1 = Segment(id=sid1, type="line", point_indices=[p1_i, new_idx],
                       bc_name=bc)
        self.geom.segments.append(seg1)
        sid2 = self.geom.next_segment_id()
        seg2 = Segment(id=sid2, type="line", point_indices=[new_idx, p2_i],
                       bc_name=bc)
        self.geom.segments.append(seg2)

        # loop の segment_ids 内の old_id を [sid1, sid2] に置換
        for lp in self.geom.loops:
            if old_id in lp.segment_ids:
                pos = lp.segment_ids.index(old_id)
                lp.segment_ids[pos:pos + 1] = [sid1, sid2]

        self.selected_point_index = new_idx
        self.selected_segment_id = None
        self.mark_dirty()
        self.redraw()
        self._update_statusbar()
        self.notify_selection_changed()
        return True

    def finalize_simple_loop_with_vacuum(self) -> Optional[int]:
        """Simple mode 用: 全 segment を 1 つの Loop にし、Vacuum Region を自動作成。

        - 末尾点と先頭点が segment で繋がっていなければ「閉じ用 segment」を追加。
        - 全 segment_ids (順序は segments のリスト順) を 1 つの Loop にまとめる。
        - Region(name="Vacuum", material_tag="vacuum", eps_r=1.0) を 1 つ追加。

        既に loop または region が存在する場合は False 系（None 返却）。
        Reset でやり直す運用とする。

        Returns:
            作成された region.id（成功時）または None（失敗時）。
        """
        if self.geom.loops or self.geom.regions:
            wx.MessageBox(
                "既に Loop または Region が存在します。"
                "Simple モードで作り直すには先に Reset してください。",
                "Info", wx.OK | wx.ICON_INFORMATION)
            return None
        if len(self.geom.points) < 3:
            wx.MessageBox(
                "閉ループには最低 3 点が必要です。",
                "Info", wx.OK | wx.ICON_INFORMATION)
            return None

        # 末尾点 → 先頭点 の segment が無ければ追加（閉じる）
        n_pts = len(self.geom.points)
        last_idx = n_pts - 1
        first_idx = 0
        closing_segment_needed = True
        for seg in self.geom.segments:
            pi = set(seg.point_indices)
            if pi == {last_idx, first_idx}:
                closing_segment_needed = False
                break
        if closing_segment_needed:
            sid = self.geom.next_segment_id()
            self.geom.segments.append(Segment(
                id=sid, type="line", point_indices=[last_idx, first_idx],
                bc_name=DEFAULT_BC,
            ))

        # 全 segment を 1 つの Loop に
        lid = self.geom.next_loop_id()
        loop = Loop(id=lid,
                    segment_ids=[s.id for s in self.geom.segments],
                    orientation="CCW")
        self.geom.loops.append(loop)

        # Vacuum Region を 1 つ
        rid = self.geom.next_region_id()
        region = Region(
            id=rid, name="Vacuum", outer_loop_id=lid,
            hole_loop_ids=[], material_tag="vacuum",
            eps_r=1.0, mu_r=1.0,
        )
        self.geom.regions.append(region)

        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()
        return rid

    def finalize_pending_loop(self, orientation: str = "CCW") -> Optional[int]:
        if not self.pending_loop_segments:
            return None
        lid = self.geom.next_loop_id()
        loop = Loop(id=lid, segment_ids=list(self.pending_loop_segments), orientation=orientation)
        self.geom.loops.append(loop)
        self.pending_loop_segments = []
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()
        return lid

    def cancel_pending_loop(self):
        if self.pending_loop_segments:
            self.pending_loop_segments = []
            self.redraw()
            self.notify_selection_changed()

    def delete_loop(self, lid: int):
        # 領域参照を持つループは削除拒否
        for r in self.geom.regions:
            if r.outer_loop_id == lid or lid in r.hole_loop_ids:
                wx.MessageBox(f"Loop {lid} は領域 {r.id}({r.name}) に参照されています。"
                              "先に領域を削除してください。", "Error",
                              wx.OK | wx.ICON_ERROR)
                return
        self.geom.loops = [lp for lp in self.geom.loops if lp.id != lid]
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()

    def add_region(self, name: str, outer_loop_id: int, hole_loop_ids: list[int],
                   material_tag: str, eps_r: float,
                   eps_r_expr: str | None = None,
                   tan_delta: float = 0.0,
                   tan_delta_expr: str | None = None) -> int:
        rid = self.geom.next_region_id()
        region = Region(
            id=rid, name=name, outer_loop_id=outer_loop_id,
            hole_loop_ids=list(hole_loop_ids),
            material_tag=material_tag, eps_r=float(eps_r),
            eps_r_expr=eps_r_expr,
            tan_delta=float(tan_delta), tan_delta_expr=tan_delta_expr,
        )
        self.geom.regions.append(region)
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()
        return rid

    def update_region(self, rid: int, name: str, outer_loop_id: int,
                      hole_loop_ids: list[int], material_tag: str, eps_r: float,
                      eps_r_expr: str | None = None,
                      tan_delta: float = 0.0,
                      tan_delta_expr: str | None = None):
        for r in self.geom.regions:
            if r.id == rid:
                r.name = name
                r.outer_loop_id = outer_loop_id
                r.hole_loop_ids = list(hole_loop_ids)
                r.material_tag = material_tag
                r.eps_r = float(eps_r)
                r.eps_r_expr = eps_r_expr
                r.tan_delta = float(tan_delta)
                r.tan_delta_expr = tan_delta_expr
                self.mark_dirty()
                self.redraw()
                self.notify_selection_changed()
                return

    def delete_region(self, rid: int):
        self.geom.regions = [r for r in self.geom.regions if r.id != rid]
        self.mark_dirty()
        self.redraw()
        self.notify_selection_changed()

    # ------------------------------------------------------------------
    # メッシュ出力
    # ------------------------------------------------------------------
    def export_geo(self, out_path: str) -> str:
        """`geom` を gmsh で OCC モデル化し、`.geo` (unrolled) で書き出す。"""
        from ..shared.gmsh_export_occ import export_geo_multi_region
        errs = self.geom.validate()
        if errs:
            raise ValueError("Geometry validation failed:\n  - " + "\n  - ".join(errs))
        if not self.geom.regions:
            raise ValueError("少なくとも 1 つの Region が必要です。")
        export_geo_multi_region(self.geom, out_path)
        return out_path

    def export_python_script(self, out_path: str, mesh_order: int = 1) -> str:
        """`geom` を再構築する自己完結 Python スクリプトを書き出す。"""
        from ..shared.gmsh_export_occ import export_python_script_multi_region
        errs = self.geom.validate()
        if errs:
            raise ValueError("Geometry validation failed:\n  - " + "\n  - ".join(errs))
        if not self.geom.regions:
            raise ValueError("少なくとも 1 つの Region が必要です。")
        export_python_script_multi_region(self.geom, out_path,
                                          mesh_order=mesh_order)
        return out_path

    def export_msh(self, out_path: str, mesh_order: int = 1) -> str:
        """`.msh` をエクスポートし、同名の `.materials.json` も書き出す。

        サイドカー JSON は ``axicavity-fem solve`` が自動読込し、
        TM0 ソルバへ material_table として渡される。
        """
        import json as _json
        from ..shared.gmsh_export_occ import export_msh_multi_region
        from ..shared.material_resolver import build_material_table_from_geometry

        errs = self.geom.validate()
        if errs:
            raise ValueError("Geometry validation failed:\n  - " + "\n  - ".join(errs))
        if not self.geom.regions:
            raise ValueError("少なくとも 1 つの Region が必要です。")
        export_msh_multi_region(self.geom, out_path, mesh_order=mesh_order)

        # サイドカー materials.json
        materials_path = os.path.splitext(out_path)[0] + ".materials.json"
        table = build_material_table_from_geometry(self.geom)
        with open(materials_path, "w", encoding="utf-8") as f:
            _json.dump({
                "schema": "axicavity-fem-v21.materials/1",
                "mesh_file": os.path.basename(out_path),
                "materials": table,
            }, f, indent=2, ensure_ascii=False)
        return out_path

    # ------------------------------------------------------------------
    # 描画 (フル再描画 — MVP のため簡素優先)
    # ------------------------------------------------------------------
    def redraw(self):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.cla()
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True)

        # segments
        for seg in self.geom.segments:
            p_a, p_b = seg.point_indices
            if not (0 <= p_a < len(self.geom.points) and 0 <= p_b < len(self.geom.points)):
                continue
            x1, y1 = self.geom.points[p_a]
            x2, y2 = self.geom.points[p_b]
            color = BC_COLORS.get(seg.bc_name, _DEFAULT_BC_COLOR)
            lw = BC_LINEWIDTHS.get(seg.bc_name, _DEFAULT_BC_LW)
            if seg.id == self.selected_segment_id:
                # 選択時は色を変えず、線を太くするのみ
                lw = lw + _SELECTED_LW_DELTA
            elif seg.id in self.pending_loop_segments:
                color = COLOR_PENDING_LOOP
                lw = lw + 1.0
            if seg.type == "arc" and seg.center is not None:
                arc = patches.Arc(
                    (seg.center[0], seg.center[1]),
                    width=2.0 * seg.radius, height=2.0 * seg.radius,
                    angle=0.0, theta1=seg.theta1, theta2=seg.theta2,
                    color=color, linewidth=lw, zorder=2)
                self.axes.add_patch(arc)
            else:
                self.axes.plot([x1, x2], [y1, y2], color=color,
                               linewidth=lw, zorder=2)

        # points
        for i, (z, r) in enumerate(self.geom.points):
            color = COLOR_POINT_SELECTED if i == self.selected_point_index else COLOR_POINT
            size = 8 if i == self.selected_point_index else 5
            self.axes.plot(z, r, "o", color=color, markersize=size, zorder=3)
            self.axes.annotate(str(i), xy=(z, r), xytext=(4, 4),
                                textcoords="offset points", fontsize=8, color="#444444",
                                zorder=4)

        # Pending segment start ハイライト
        if self.pending_segment_start is not None and \
                0 <= self.pending_segment_start < len(self.geom.points):
            z, r = self.geom.points[self.pending_segment_start]
            self.axes.plot(z, r, "o", color=COLOR_PENDING_LOOP, markersize=10,
                            markerfacecolor="none", markeredgewidth=2, zorder=5)

        self.canvas.draw_idle()


# Convenience: load .gmshproj into a MultiRegionGeometry (v2.1 形式 or legacy)
def load_gmshproj_as_multi_region(path: str) -> MultiRegionGeometry:
    """`.gmshproj` を読み、schema_version で v2.1 / legacy を分岐して返す。"""
    import json
    data = json.loads(open(path, "r", encoding="utf-8").read())
    return MultiRegionGeometry.from_dict(data)
