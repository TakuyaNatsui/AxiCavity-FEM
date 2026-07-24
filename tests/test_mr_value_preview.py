"""ver2.3: Multi-Region Editor の座標欄・数値プレビューのテスト (ヘッドレス)。

Point Z/R・Arc Center Z/R の入力欄に式を入れると、評価済み数値が隣のラベルに
`= <値>` として表示される。素の数値では非表示、未定義変数では `= ?`。
表示位置・見た目の最終確認は手動。
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
wx = pytest.importorskip("wx")


@pytest.fixture
def frame():
    from axicavity_fem.gui.main_frame import MyFrame
    try:
        app = wx.App(False)
        frm = MyFrame(None, wx.ID_ANY, "")
    except Exception as e:  # pragma: no cover - 環境依存
        pytest.skip(f"wx GUI を構築できません: {e}")
    yield frm
    frm.Destroy()
    app.Destroy()


def _set_var(frame, row, name, expr):
    g = frame.mr_variables_grid
    g.SetCellValue(row, 0, name)
    g.SetCellValue(row, 1, expr)
    frame._sync_mr_variables_from_grid()


def test_preview_labels_exist(frame):
    """プレビューラベル 4 個が (wxGlade 生成 or フォールバックで) 用意される。"""
    for name in ("mr_point_z_value_label", "mr_point_r_value_label",
                 "mr_arc_center_z_value_label", "mr_arc_center_r_value_label"):
        assert getattr(frame, name, None) is not None


def test_format_value_preview_rules(frame):
    _set_var(frame, 0, "a", "100")
    _set_var(frame, 1, "b", "10")
    # 式 → = 値
    assert frame._format_value_preview("a+b") == "= 110.000000"
    # 素の数値 → 空 (併記しない)
    assert frame._format_value_preview("12.5") == ""
    # 空 → 空
    assert frame._format_value_preview("") == ""
    assert frame._format_value_preview("   ") == ""
    # 評価不能 (未定義変数) → = ?
    assert frame._format_value_preview("q+1") == "= ?"


def test_point_preview_on_selection(frame):
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    _set_var(frame, 0, "a", "100")
    _set_var(frame, 1, "b", "10")
    # z は式、r は素の数値
    idx = geom.append_point(110.0, 5.0, "a+b", None)
    p.selected_point_index = idx
    frame.on_mr_selection_changed()
    assert frame.mr_point_z_value_label.GetLabel() == "= 110.000000"
    assert frame.mr_point_r_value_label.GetLabel() == ""


def test_point_preview_follows_variable_change(frame):
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    _set_var(frame, 0, "a", "100")
    _set_var(frame, 1, "b", "10")
    idx = geom.append_point(110.0, 0.0, "a+b", None)
    p.selected_point_index = idx
    frame.on_mr_selection_changed()
    assert frame.mr_point_z_value_label.GetLabel() == "= 110.000000"
    # a を 200 に変更 → プレビューが追従
    frame.mr_variables_grid.SetCellValue(0, 1, "200")
    frame.OnMrVariableCellChanged(None)
    assert frame.mr_point_z_value_label.GetLabel() == "= 210.000000"


def test_point_preview_updates_after_update_button(frame):
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    _set_var(frame, 0, "a", "100")
    idx = geom.append_point(0.0, 0.0)
    p.selected_point_index = idx
    frame.mr_point_z_ctrl.SetValue("a*2")
    frame.mr_point_r_ctrl.SetValue("5")
    frame.OnMrUpdatePoint(None)
    assert frame.mr_point_z_value_label.GetLabel() == "= 200.000000"
    assert frame.mr_point_r_value_label.GetLabel() == ""


def test_preview_empty_when_no_selection(frame):
    frame.on_mr_selection_changed()  # 何も選択されていない
    assert frame.mr_point_z_value_label.GetLabel() == ""
    assert frame.mr_point_r_value_label.GetLabel() == ""


def test_arc_center_preview(frame):
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    _set_var(frame, 0, "c", "3")
    # 直線 segment を作って arc に変換
    geom.points.extend([(0.0, 0.0), (10.0, 0.0)])
    geom.segments.append(Segment(id=0, type="line", point_indices=[0, 1]))
    p.selected_segment_id = 0
    assert p.convert_selected_to_arc() is True
    # 中心 Z を式で更新
    frame.mr_arc_center_z_ctrl.SetValue("c+2")
    frame.mr_arc_center_r_ctrl.SetValue("1")
    frame.OnMrUpdateCenter(None)
    assert frame.mr_arc_center_z_value_label.GetLabel() == "= 5.000000"
    assert frame.mr_arc_center_r_value_label.GetLabel() == ""


def test_live_preview_on_typing(frame):
    """EVT_TEXT により SetValue でプレビューが即時更新される。"""
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    _set_var(frame, 0, "a", "7")
    idx = geom.append_point(0.0, 0.0)
    p.selected_point_index = idx
    # SetValue は EVT_TEXT を発火する
    frame.mr_point_z_ctrl.SetValue("a*3")
    assert frame.mr_point_z_value_label.GetLabel() == "= 21.000000"
