"""ver2.2: 変数定義テーブル (wxGrid) と座標欄の数式評価のスモークテスト.

ヘッドレスで MyFrame を構築し、変数グリッドの評価・geom.variables 連携・
座標入力欄が数式を評価することを確認する。表示の最終確認は手動。
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


def test_variables_grid_exists(frame):
    g = frame.mr_variables_grid
    assert g is not None
    assert g.GetNumberCols() == 3
    assert g.GetColLabelValue(0) == "Name"
    assert g.GetColLabelValue(1) == "Expression"
    assert g.GetColLabelValue(2) == "Value"


def test_variables_grid_evaluation(frame):
    g = frame.mr_variables_grid
    g.SetCellValue(0, 0, "a"); g.SetCellValue(0, 1, "100")
    g.SetCellValue(1, 0, "b"); g.SetCellValue(1, 1, "10")
    g.SetCellValue(2, 0, "c"); g.SetCellValue(2, 1, "a + b")
    frame._sync_mr_variables_from_grid()
    # Value 列に評価結果
    assert g.GetCellValue(0, 2) == "100"
    assert g.GetCellValue(1, 2) == "10"
    assert g.GetCellValue(2, 2) == "110"
    # geom.variables に (name, expr) が保存される
    assert frame.mr_editor_panel.geom.variables == [
        ("a", "100"), ("b", "10"), ("c", "a + b"),
    ]
    # get_mr_variables は評価済み辞書
    assert frame.get_mr_variables() == {"a": 100.0, "b": 10.0, "c": 110.0}


def test_variables_grid_error_display(frame):
    g = frame.mr_variables_grid
    g.SetCellValue(0, 0, "x"); g.SetCellValue(0, 1, "unknown + 1")
    frame._sync_mr_variables_from_grid()
    assert g.GetCellValue(0, 2).startswith("エラー")
    assert "x" not in frame.get_mr_variables()


def test_set_variables_rows_populates_grid(frame):
    frame.set_mr_variables_rows([("w", "50"), ("h", "w/2")])
    g = frame.mr_variables_grid
    assert g.GetCellValue(0, 0) == "w"
    assert g.GetCellValue(0, 1) == "50"
    assert g.GetCellValue(1, 0) == "h"
    assert g.GetCellValue(1, 1) == "w/2"
    assert frame.get_mr_variables() == {"w": 50.0, "h": 25.0}


def test_point_coord_uses_variables(frame):
    p = frame.mr_editor_panel
    p.get_geometry().points.append((1.0, 2.0))
    p.selected_point_index = 0
    # 変数 a=5 を定義
    g = frame.mr_variables_grid
    g.SetCellValue(0, 0, "a"); g.SetCellValue(0, 1, "5")
    frame._sync_mr_variables_from_grid()
    # 座標欄に数式を入力して更新
    frame.mr_point_z_ctrl.SetValue("a * 2")
    frame.mr_point_r_ctrl.SetValue("a + 1")
    frame.OnMrUpdatePoint(None)
    assert p.get_geometry().points[0] == (10.0, 6.0)


def test_point_coord_invalid_expr_no_update(frame, monkeypatch):
    p = frame.mr_editor_panel
    p.get_geometry().points.append((1.0, 2.0))
    p.selected_point_index = 0
    # モーダルダイアログを抑止
    monkeypatch.setattr(wx, "MessageBox", lambda *a, **k: None)
    frame.mr_point_z_ctrl.SetValue("nonexistent + 1")
    frame.mr_point_r_ctrl.SetValue("3")
    frame.OnMrUpdatePoint(None)
    # 評価失敗により座標は更新されない
    assert p.get_geometry().points[0] == (1.0, 2.0)
