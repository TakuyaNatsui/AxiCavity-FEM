"""ver2.2 第2弾: 座標式の保持・再計算・Export 反映のテスト.

- モデル: point_exprs/center_expr/mesh_size_expr/eps_r_expr の往復・後方互換・
  長さ不変条件、同期ヘルパ (append/set/pop_point)。
- 再計算 (ヘッドレス GUI): 変数変更で式付き座標が動く、ドラッグで式破棄、arc 追従。
- Export Python: 変数定義＋式が出力され、生成物が compile() 可能。
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)
from axicavity_fem.shared.gmsh_export_occ import export_python_script_multi_region


# ---------------------------------------------------------------------------
# モデル: 同期ヘルパと長さ不変条件
# ---------------------------------------------------------------------------
def test_point_helpers_keep_parallel_length():
    g = MultiRegionGeometry()
    i0 = g.append_point(1.0, 2.0)
    i1 = g.append_point(3.0, 4.0, "a", "b")
    assert (i0, i1) == (0, 1)
    assert len(g.points) == len(g.point_exprs) == 2
    assert g.point_exprs[1] == ("a", "b")
    g.set_point(0, 5.0, 6.0, "c", None)
    assert g.points[0] == (5.0, 6.0)
    assert g.point_exprs[0] == ("c", None)
    g.pop_point(0)
    assert len(g.points) == len(g.point_exprs) == 1
    assert g.point_exprs[0] == ("a", "b")  # idx1 が詰められた
    assert g.validate() == []


def test_ensure_length_on_construct_and_backward_compat():
    # point_exprs を渡さず points だけ与えても __post_init__ で None 埋め
    g = MultiRegionGeometry(points=[(0.0, 0.0), (1.0, 1.0)])
    assert g.point_exprs == [(None, None), (None, None)]
    assert g.validate() == []


def test_expr_fields_roundtrip():
    g = MultiRegionGeometry(unit="mm", mesh_size=5.0)
    g.variables = [("a", "10")]
    g.append_point(0.0, 0.0)
    g.append_point(10.0, 0.0, "a", None)
    g.mesh_size_expr = "a/2"
    d = g.to_dict()
    assert d["point_exprs"] == [[None, None], ["a", None]]
    assert d["mesh_size_expr"] == "a/2"
    g2 = MultiRegionGeometry.from_dict(d)
    assert g2.point_exprs == [(None, None), ("a", None)]
    assert g2.mesh_size_expr == "a/2"
    assert len(g2.point_exprs) == len(g2.points)


def test_segment_center_expr_roundtrip():
    seg = Segment(id=0, type="arc", point_indices=[0, 1],
                  center=(1.0, 2.0), radius=1.0, theta1=0.0, theta2=90.0,
                  center_expr=("a", "b"))
    d = seg.to_dict()
    assert d["center_expr"] == ["a", "b"]
    seg2 = Segment.from_dict(d)
    assert seg2.center_expr == ("a", "b")


def test_region_eps_r_expr_roundtrip():
    r = Region(id=0, name="D", outer_loop_id=0, eps_r=4.0, eps_r_expr="er")
    d = r.to_dict()
    assert d["eps_r_expr"] == "er"
    r2 = Region.from_dict(d)
    assert r2.eps_r_expr == "er"


def test_legacy_dict_without_expr_fields_loads():
    # schema 2.1 相当 (式フィールドなし) でも読め、式は空/None
    g = MultiRegionGeometry(points=[(0.0, 0.0), (1.0, 1.0)])
    d = g.to_dict()
    d["schema_version"] = "2.1"
    for k in ("point_exprs", "mesh_size_expr"):
        d.pop(k, None)
    g2 = MultiRegionGeometry.from_dict(d)
    assert g2.point_exprs == [(None, None), (None, None)]
    assert g2.mesh_size_expr is None


# ---------------------------------------------------------------------------
# Export Python: 変数定義＋式の反映と compile()
# ---------------------------------------------------------------------------
def _rect_geom_with_exprs() -> MultiRegionGeometry:
    g = MultiRegionGeometry(unit="mm", mesh_size=5.0)
    g.variables = [("w", "100"), ("h", "50")]
    g.append_point(0.0, 0.0)                 # idx0 定数
    g.append_point(100.0, 0.0, "w", None)    # idx1 z=w
    g.append_point(100.0, 50.0, "w", "h")    # idx2 z=w, r=h
    g.append_point(0.0, 50.0, None, "h")     # idx3 r=h
    for i in range(4):
        g.segments.append(Segment(id=i, type="line",
                                  point_indices=[i, (i + 1) % 4]))
    g.loops.append(Loop(id=0, segment_ids=[0, 1, 2, 3]))
    g.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                            material_tag="vacuum", eps_r=1.0))
    g.mesh_size_expr = "w/20"
    return g


def test_export_python_emits_variables_and_expressions(tmp_path):
    g = _rect_geom_with_exprs()
    out = tmp_path / "gen.py"
    export_python_script_multi_region(g, out)
    text = out.read_text(encoding="utf-8")
    assert "from math import *" in text
    # 変数は user_vars で上書き可能な形で出力される (ver2.3)
    assert "w = user_vars.get('w', 100)" in text
    assert "h = user_vars.get('h', 50)" in text
    assert "def build_model(" in text
    assert "if __name__ == '__main__':" in text
    assert "(w) * _SCALE" in text          # 式の点
    assert "(h) * _SCALE" in text
    assert "(w/20) * _SCALE" in text        # _LC が式
    # 生成スクリプトが構文的に妥当 (実行は gmsh 依存のため compile のみ)
    compile(text, str(out), "exec")


def test_export_python_constant_points_stay_numeric(tmp_path):
    g = _rect_geom_with_exprs()
    out = tmp_path / "gen2.py"
    export_python_script_multi_region(g, out)
    text = out.read_text(encoding="utf-8")
    # idx0 は定数なので数値 repr で出る (0.0)
    assert "0.0 * _SCALE, 0.0 * _SCALE" in text


def test_export_python_build_model_accepts_overrides(tmp_path):
    """生成された build_model(**user_vars) が変数を上書きして座標に反映する。

    gmsh をモックに差し替え、addPoint に渡る z 座標を記録して検証する
    (実 gmsh 非依存)。
    """
    import sys
    from unittest import mock

    g = _rect_geom_with_exprs()  # unit=mm (scale=1e-3), 変数 w=100, 点に z=w
    out = tmp_path / "gen3.py"
    export_python_script_multi_region(g, out, msh_output="x.msh")
    text = out.read_text(encoding="utf-8")

    recorded_z: list[float] = []
    counter = {"n": 0}

    def _next_tag(*_a, **_k):
        counter["n"] += 1
        return counter["n"]

    def _add_point(z, r, zc, lc):
        recorded_z.append(z)
        return _next_tag()

    fake = mock.MagicMock()
    fake.model.occ.addPoint.side_effect = _add_point
    for name in ("addLine", "addCircleArc", "addCurveLoop", "addPlaneSurface"):
        getattr(fake.model.occ, name).side_effect = _next_tag

    ns = {"__name__": "gen_module"}  # __main__ ガードを回避
    with mock.patch.dict(sys.modules, {"gmsh": fake}):
        exec(compile(text, str(out), "exec"), ns)
        assert callable(ns["build_model"])
        recorded_z.clear()
        ns["build_model"](w=250)   # w を 100 → 250 に上書き

    # z=w の点が 250mm = 0.25m として反映される (scale mm→m = 1e-3)
    assert max(recorded_z) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# ヘッドレス GUI: 更新で式保存・変数変更で再計算・ドラッグで式破棄
# ---------------------------------------------------------------------------
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


def test_update_point_stores_and_shows_expression(frame):
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    idx = geom.append_point(0.0, 0.0)
    p.selected_point_index = idx
    _set_var(frame, 0, "a", "3")
    frame.mr_point_z_ctrl.SetValue("a+2")
    frame.mr_point_r_ctrl.SetValue("10")   # 素の数値 → 式保存しない
    frame.OnMrUpdatePoint(None)
    assert geom.points[idx] == pytest.approx((5.0, 10.0))
    assert geom.point_exprs[idx] == ("a+2", None)
    # 再選択で欄に式が戻る (z=式, r=数値)
    frame.on_mr_selection_changed()
    assert frame.mr_point_z_ctrl.GetValue() == "a+2"
    assert "10" in frame.mr_point_r_ctrl.GetValue()


def test_variable_change_recomputes_point(frame):
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    idx = geom.append_point(10.0, 0.0, "a", None)
    _set_var(frame, 0, "a", "10")
    frame.OnMrVariableCellChanged(None)
    assert geom.points[idx][0] == pytest.approx(10.0)
    # a を 25 に変更 → 点が動く
    frame.mr_variables_grid.SetCellValue(0, 1, "25")
    frame.OnMrVariableCellChanged(None)
    assert geom.points[idx][0] == pytest.approx(25.0)


def test_drag_clears_expression(frame):
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    idx = geom.append_point(5.0, 5.0, "a", "b")
    p.selected_point_index = idx
    p.dragging_point = True

    class _Evt:
        xdata = 7.0
        ydata = 8.0
    evt = _Evt()
    evt.inaxes = p.axes
    p.on_motion(evt)
    assert geom.point_exprs[idx] == (None, None)
    assert geom.points[idx] == pytest.approx((7.0, 8.0))
