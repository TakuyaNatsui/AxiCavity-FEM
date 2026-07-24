"""ver2.1: Multi-Region Editor の Simple/Advanced モード切替テスト.

Simple モードでは Loops/Regions/Edit Mode UI が隠れ、auto-point-segment +
Close Loop で Vacuum 領域 (eps_r=1) が自動生成されることを確認する。
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
wx = pytest.importorskip("wx")

from axicavity_fem.gui.multi_region_editor import (  # noqa: E402
    MODE_AUTO_POINT_SEGMENT,
)


@pytest.fixture
def frame():
    from axicavity_fem.gui.main_frame import MyFrame
    try:
        app = wx.App(False)
        frm = MyFrame(None, wx.ID_ANY, "")
    except Exception as e:
        pytest.skip(f"wx GUI を構築できません: {e}")
    yield frm
    frm.Destroy()
    app.Destroy()


def test_default_is_simple_mode(frame):
    assert frame.mr_view_mode_radiobox.GetSelection() == 0
    assert frame.mr_view_mode_radiobox.GetStringSelection() == "Simple"
    # エディタは auto-point-segment が既定
    assert frame.mr_editor_panel.mode == MODE_AUTO_POINT_SEGMENT


def test_advanced_widgets_hidden_in_simple(frame):
    # Simple 既定で Advanced 用 UI 群が非表示 (panel_advanced_mode が Hide)
    assert hasattr(frame, "panel_advanced_mode")
    assert not frame.panel_advanced_mode.IsShown()
    # Simple 専用の Close Loop ボタンは表示
    assert frame.mr_simple_close_loop_button.IsShown()


def test_switch_to_advanced_shows_widgets(frame):
    frame.mr_view_mode_radiobox.SetSelection(1)  # Advanced
    frame._apply_view_mode()
    assert frame.panel_advanced_mode.IsShown()


def test_simple_close_loop_creates_vacuum_region(frame):
    """auto-point-segment で点を 4 つ追加 → Close Loop で Vacuum 領域が自動生成。"""
    p = frame.mr_editor_panel
    assert p.mode == MODE_AUTO_POINT_SEGMENT
    geom = p.get_geometry()

    # 直接 geom を編集して auto-point-segment 相当の状態を作る
    # (キャンバスクリックは GUI イベント経由で煩雑なので、
    #  ロジックを直接呼ぶ)
    from axicavity_fem.shared.multi_region_model import Segment
    geom.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    # 直前点との segment を 3 本 (4点→3segment)
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b],
                                      bc_name="PEC"))

    rid = p.finalize_simple_loop_with_vacuum()
    assert rid == 0
    geom = p.get_geometry()
    # 閉じ用 segment が追加されているはず (4 segments total)
    assert len(geom.segments) == 4
    # 全 segment が 1 つの Loop に
    assert len(geom.loops) == 1
    assert set(geom.loops[0].segment_ids) == {s.id for s in geom.segments}
    # Vacuum Region 1 つ
    assert len(geom.regions) == 1
    r = geom.regions[0]
    assert r.name == "Vacuum"
    assert r.material_tag == "vacuum"
    assert r.eps_r == 1.0
    assert r.outer_loop_id == geom.loops[0].id
    assert r.hole_loop_ids == []


def test_simple_close_loop_warns_when_already_has_loop(frame):
    from axicavity_fem.shared.multi_region_model import Segment, Loop, Region
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    geom.points.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    geom.loops.append(Loop(id=0, segment_ids=[0, 1, 2]))
    geom.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                                material_tag="vacuum", eps_r=1.0))
    # 既に loop/region があるので None で拒否されるはず
    # (MessageBox は GUI 表示なので、ヘッドレスで自動 OK されないよう
    #  ここでは戻り値だけ確認)
    import unittest.mock as mock
    with mock.patch("wx.MessageBox"):
        rid = p.finalize_simple_loop_with_vacuum()
    assert rid is None


def test_simple_close_loop_button_handler(frame):
    """Simple モード専用ボタンのハンドラ OnBtnMrCloseLoop が Vacuum 領域を作る。"""
    import unittest.mock as mock
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    geom.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    with mock.patch("wx.MessageBox"):
        frame.OnBtnMrCloseLoop(None)
    geom = p.get_geometry()
    assert len(geom.regions) == 1
    assert geom.regions[0].material_tag == "vacuum"
    assert len(geom.segments) == 4  # 閉じ用 segment 追加済み


def test_advanced_to_simple_keeps_internal_state(frame):
    """Advanced で複雑な状態を作って Simple へ戻しても geom は失われない。"""
    from axicavity_fem.shared.multi_region_model import Segment, Loop, Region
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    geom.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    geom.loops.append(Loop(id=0, segment_ids=[0, 1, 2, 3]))
    geom.regions.append(Region(id=0, name="Dielectric", outer_loop_id=0,
                                material_tag="dielectric_1", eps_r=4.0))
    frame.mr_view_mode_radiobox.SetSelection(1)  # Advanced
    frame._apply_view_mode()
    # Simple へ戻す (警告が出るはず) - MessageBox をモックして OK 扱い
    import unittest.mock as mock
    with mock.patch("wx.MessageBox"):
        frame.mr_view_mode_radiobox.SetSelection(0)
        frame.OnMrViewModeChange(None)
    # geom は保持されている
    assert len(p.get_geometry().regions) == 1
    assert p.get_geometry().regions[0].eps_r == 4.0


# ---------------------------------------------------------------------------
# 円弧 UI 配線 (Step2)
# ---------------------------------------------------------------------------
def test_arc_widgets_exist_and_visible_both_modes(frame):
    """Selected Segment / 円弧操作ボタンが両モードで表示される。"""
    for name in ("mr_bc_choice", "mr_delete_segment_button",
                 "mr_convert_arc_button", "mr_convert_line_button",
                 "mr_arc_center_z_ctrl", "mr_arc_center_r_ctrl",
                 "mr_update_center_button", "mr_selected_segment_label"):
        assert hasattr(frame, name), f"{name} が存在しない"
    # Simple モード (既定) でも Selected Segment 関連は表示
    assert frame.mr_convert_arc_button.IsShown()
    assert frame.mr_bc_choice.IsShown()
    # Advanced でも表示
    frame.mr_view_mode_radiobox.SetSelection(1)
    frame._apply_view_mode()
    assert frame.mr_convert_arc_button.IsShown()


def test_arc_controls_disabled_without_selection(frame):
    """segment 未選択時は円弧ボタン・中心欄が無効。"""
    frame.on_mr_selection_changed()
    assert not frame.mr_convert_arc_button.IsEnabled()
    assert not frame.mr_convert_line_button.IsEnabled()
    assert not frame.mr_update_center_button.IsEnabled()


def test_convert_to_arc_via_handler(frame):
    """segment 選択 → OnMrConvertToArc で arc になり、中心欄が埋まる。"""
    import unittest.mock as mock
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    p.selected_segment_id = 0
    frame.on_mr_selection_changed()
    # line 選択時: Convert to Arc 有効, Convert to Line 無効
    assert frame.mr_convert_arc_button.IsEnabled()
    assert not frame.mr_convert_line_button.IsEnabled()
    # 円弧化
    with mock.patch("wx.MessageBox"):
        frame.OnMrConvertToArc(None)
    seg = g.segment_by_id(0)
    assert seg.type == "arc"
    # selection 同期で中心欄が埋まり、Convert to Line/Update Center 有効
    assert frame.mr_convert_line_button.IsEnabled()
    assert frame.mr_update_center_button.IsEnabled()
    assert frame.mr_arc_center_z_ctrl.GetValue() != ""


def test_update_center_via_handler(frame):
    import unittest.mock as mock
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    p.selected_segment_id = 0
    with mock.patch("wx.MessageBox"):
        frame.OnMrConvertToArc(None)
    frame.mr_arc_center_z_ctrl.SetValue("5.0")
    frame.mr_arc_center_r_ctrl.SetValue("-8.0")
    with mock.patch("wx.MessageBox"):
        frame.OnMrUpdateCenter(None)
    assert g.segment_by_id(0).center == (5.0, -8.0)


def test_simple_segment_selection_after_close_loop(frame):
    """Simple モードの "Edit Lines" (MODE_EDIT_BC) で segment を選択できる。"""
    import unittest.mock as mock
    from axicavity_fem.gui.multi_region_editor import MODE_EDIT_BC
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    with mock.patch("wx.MessageBox"):
        frame.OnBtnMrCloseLoop(None)  # Region 作成
    assert len(g.regions) == 1
    # Edit Lines に切替 (radiobox index 1)
    frame.mr_modeSimple_radiobox.SetSelection(1)
    frame._apply_mr_simple_mode()
    assert p.mode == MODE_EDIT_BC
    # 軸範囲を既知にして、segment 1 (10,0)-(10,5) の中点付近をクリック
    p.set_axes_limits(-5, 15, -5, 10)
    seg1 = g.segment_by_id(1)
    pa, pb = seg1.point_indices
    mzx = (g.points[pa][0] + g.points[pb][0]) / 2.0
    mzy = (g.points[pa][1] + g.points[pb][1]) / 2.0

    class _Evt:
        inaxes = p.axes
        xdata = mzx
        ydata = mzy
        x = 0
        y = 0
    _Evt.x, _Evt.y = p.axes.transData.transform((mzx, mzy))
    p.toolbar.mode = ""
    p.on_click(_Evt)
    assert p.selected_segment_id is not None
    assert len(g.points) == 4  # 点は増えていない


def test_simple_radiobox_switches_mode(frame):
    """mr_modeSimple_radiobox で MR エディタのモードが切り替わる。"""
    from axicavity_fem.gui.multi_region_editor import (
        MODE_AUTO_POINT_SEGMENT, MODE_EDIT_BC,
    )

    class _Evt:
        def __init__(self, obj):
            self._o = obj
        def GetEventObject(self):
            return self._o
        def Skip(self):
            pass
    frame.mr_modeSimple_radiobox.SetSelection(1)
    frame.on_mode_change(_Evt(frame.mr_modeSimple_radiobox))
    assert frame.mr_editor_panel.mode == MODE_EDIT_BC
    frame.mr_modeSimple_radiobox.SetSelection(0)
    frame.on_mode_change(_Evt(frame.mr_modeSimple_radiobox))
    assert frame.mr_editor_panel.mode == MODE_AUTO_POINT_SEGMENT


def test_simple_radiobox_enabled_only_with_points(frame):
    """点が無いうちは Simple radiobox は無効、点があれば有効。"""
    assert not frame.mr_modeSimple_radiobox.IsEnabled()
    from axicavity_fem.shared.multi_region_model import Segment
    g = frame.mr_editor_panel.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0)])
    frame.on_mr_editor_changed()
    assert frame.mr_modeSimple_radiobox.IsEnabled()


def test_add_point_on_segment_inserts_and_updates_loop(frame):
    """ライン分割で点が挿入され、loop の segment_ids も更新される。"""
    from axicavity_fem.shared.multi_region_model import Segment, Loop, Region
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    g.loops.append(Loop(id=0, segment_ids=[0, 1, 2, 3]))
    g.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                            material_tag="vacuum", eps_r=1.0))
    p.set_axes_limits(-5, 15, -5, 10)
    n_seg0 = len(g.segments)
    # segment 0 (0,0)-(10,0) の中点 (5,0) 付近に挿入
    ok = p.add_point_on_segment(5.0, 0.1)
    assert ok is True
    assert len(g.points) == 5  # 点が 1 つ増えた
    assert len(g.segments) == n_seg0 + 1  # 1 本 → 2 本
    # loop は 4 → 5 segment に
    assert len(g.loops[0].segment_ids) == 5
    # loop の全 segment_id が実在する
    seg_ids = {s.id for s in g.segments}
    assert all(sid in seg_ids for sid in g.loops[0].segment_ids)


def _build_closed_square(frame):
    """4 点正方形を Close Loop して Region を作り、Edit Points 状態にする。"""
    import unittest.mock as mock
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    with mock.patch("wx.MessageBox"):
        frame.OnBtnMrCloseLoop(None)
    p.set_axes_limits(-5, 15, -5, 10)
    return p, g


def _click(p, zx, zy, dblclick=False):
    class _E:
        inaxes = p.axes
        xdata = zx
        ydata = zy
    _E.dblclick = dblclick
    _E.x, _E.y = p.axes.transData.transform((zx, zy))
    p.toolbar.mode = ""
    p.on_click(_E)


def test_autoswitch_point_edit_to_line_edit_on_line_click(frame):
    """Point Edit でライン上をクリックすると Line Edit に自動遷移し segment 選択。"""
    from axicavity_fem.gui.multi_region_editor import (
        MODE_AUTO_POINT_SEGMENT, MODE_EDIT_BC,
    )
    p, g = _build_closed_square(frame)
    assert p.mode == MODE_AUTO_POINT_SEGMENT
    # segment 1 = (10,0)-(10,5), 中点 (10,2.5)
    _click(p, 10.0, 2.5)
    assert p.mode == MODE_EDIT_BC
    assert p.selected_segment_id == 1
    assert frame.mr_modeSimple_radiobox.GetSelection() == 1


def test_autoswitch_line_edit_to_point_edit_on_point_click(frame):
    """Line Edit で点の近くをクリックすると Point Edit に自動遷移し点を選択。"""
    from axicavity_fem.gui.multi_region_editor import (
        MODE_AUTO_POINT_SEGMENT, MODE_EDIT_BC,
    )
    p, g = _build_closed_square(frame)
    # まず Line Edit にする
    frame.mr_modeSimple_radiobox.SetSelection(1)
    frame._apply_mr_simple_mode()
    assert p.mode == MODE_EDIT_BC
    # 点 0 (0,0) 付近クリック
    _click(p, 0.0, 0.0)
    assert p.mode == MODE_AUTO_POINT_SEGMENT
    assert p.selected_point_index == 0
    assert frame.mr_modeSimple_radiobox.GetSelection() == 0


def test_autoswitch_line_edit_dblclick_inserts_and_returns_point_edit(frame):
    """Line Edit でライン上ダブルクリック → 点挿入して Point Edit に遷移。"""
    from axicavity_fem.gui.multi_region_editor import MODE_AUTO_POINT_SEGMENT
    p, g = _build_closed_square(frame)
    frame.mr_modeSimple_radiobox.SetSelection(1)
    frame._apply_mr_simple_mode()
    n_pts = len(g.points)
    # segment 0 = (0,0)-(10,0) 上 (5,0) をダブルクリック
    _click(p, 5.0, 0.0, dblclick=True)
    assert len(g.points) == n_pts + 1
    assert p.mode == MODE_AUTO_POINT_SEGMENT
    assert frame.mr_modeSimple_radiobox.GetSelection() == 0


def test_autoswitch_disabled_in_advanced(frame):
    """Advanced モードでは auto_mode_switch が無効。"""
    frame.mr_view_mode_radiobox.SetSelection(1)
    frame._apply_view_mode()
    assert frame.mr_editor_panel.auto_mode_switch is False


def test_set_bc_none_via_handler(frame):
    """Boundary Condition で None を選ぶと segment.bc_name が "None" になる。"""
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0)])
    g.segments.append(Segment(id=0, type="line", point_indices=[0, 1], bc_name="PEC"))
    p.selected_segment_id = 0
    frame.mr_bc_choice.SetStringSelection("None")
    frame.OnMrBcChoice(None)
    assert g.segment_by_id(0).bc_name == "None"
    # 選択同期で choice が "None" を指す
    frame.on_mr_selection_changed()
    assert frame.mr_bc_choice.GetStringSelection() == "None"


def test_delete_point_reconnects_loop(frame):
    """ループ境界上の点を削除すると、前後点を繋ぐ segment が生成されループが保たれる。"""
    from axicavity_fem.shared.multi_region_model import Segment, Loop, Region
    p = frame.mr_editor_panel
    g = p.get_geometry()
    # 4 点の正方形ループ
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    g.loops.append(Loop(id=0, segment_ids=[0, 1, 2, 3]))
    g.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                            material_tag="vacuum", eps_r=1.0))
    # 点 1 (10,0) を削除 → 点 0 と点 2 を繋ぐ segment が出来る
    p.selected_point_index = 1
    p.delete_selected_point()
    g = p.get_geometry()
    assert len(g.points) == 3  # 1 点減
    # segment 数は 4 → 2 削除 + 1 追加 = 3
    assert len(g.segments) == 3
    # ループは 3 segment で構成され、全て実在
    assert len(g.loops[0].segment_ids) == 3
    seg_ids = {s.id for s in g.segments}
    assert all(sid in seg_ids for sid in g.loops[0].segment_ids)
    # ループの連結性: 各 segment が端点を共有して閉じている
    # (point_indices を辿って 3 点の閉路になっているか)
    pts_in_loop = set()
    for sid in g.loops[0].segment_ids:
        s = g.segment_by_id(sid)
        pts_in_loop.update(s.point_indices)
    assert len(pts_in_loop) == 3


def test_delete_segment_button_disabled_in_simple(frame):
    """Simple モードでは Delete Segment ボタンが無効。Advanced では有効。"""
    # Simple (既定)
    assert not frame.mr_delete_segment_button.IsEnabled()
    # Advanced に切替
    frame.mr_view_mode_radiobox.SetSelection(1)
    frame._apply_view_mode()
    assert frame.mr_delete_segment_button.IsEnabled()
    # Simple に戻す
    frame.mr_view_mode_radiobox.SetSelection(0)
    frame._apply_view_mode()
    assert not frame.mr_delete_segment_button.IsEnabled()


def test_dblclick_inserts_point_in_edit_points(frame):
    """Edit Points (auto-point-segment) でライン上ダブルクリックすると点が挿入される。"""
    import unittest.mock as mock
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    g = p.get_geometry()
    g.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        g.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    with mock.patch("wx.MessageBox"):
        frame.OnBtnMrCloseLoop(None)  # Region 作成 + Edit Points モード
    p.set_axes_limits(-5, 15, -5, 10)
    n_pts = len(g.points)
    mzx, mzy = 5.0, 0.0  # segment 0 上

    class _Evt:
        inaxes = p.axes
        xdata = mzx
        ydata = 0.1
        dblclick = True
    _Evt.x, _Evt.y = p.axes.transData.transform((mzx, 0.1))
    p.toolbar.mode = ""
    p.on_click(_Evt)
    assert len(g.points) == n_pts + 1
