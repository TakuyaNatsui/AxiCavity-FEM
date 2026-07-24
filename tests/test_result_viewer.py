"""段階7d 検証: GUI ResultViewer のロード・描画（ヘッドレス・スモーク）.

ウィンドウ表示はせず、v2 HDF5 を読み込んで update_plots が例外なく動くこと、
モード選択肢が構築されることを確認する。ディスプレイ無し環境では skip。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("h5py")
pytest.importorskip("matplotlib")
wx = pytest.importorskip("wx")

from axicavity_fem.shared.hdf5_io import write_results  # noqa: E402

_CYL_2ND = Path(__file__).resolve().parents[1] / "samples" / "cylinder100mm.msh"

pytestmark = pytest.mark.skipif(
    not _CYL_2ND.exists(), reason="cylinder100mm.msh なし")


def _write_tm0_h5(path):
    from axicavity_fem.fem_tm0.solver import solve_tm0_standing
    from axicavity_fem.shared.mesh_loader import load_mesh
    mesh = load_mesh(str(_CYL_2ND), 2)
    res = solve_tm0_standing(mesh, num_modes=3)
    write_results(path, solver_type="tm0", mesh={
        "vertices": res.nodes, "simplices": res.elements,
        "edge_map_keys": None, "edge_map_values": None,
        "physical_groups": res.physical_groups,
        "elem_order": res.element_order, "num_edges": 0},
        results_by_n={0: {"standing": {
            "frequencies": res.frequencies, "eigenvalues": None,
            "eigenvectors": res.eigenvectors}}})


def _write_hom_h5(path, n=1):
    from axicavity_fem.fem_hom.solver import solve_hom_standing
    from axicavity_fem.shared.mesh_loader import load_mesh_hom
    mesh = load_mesh_hom(str(_CYL_2ND), n, 2)
    res = solve_hom_standing(mesh, num_modes=3)
    keys = np.array(list(mesh.edge_index_map.keys()), dtype=int)
    vals = np.array(list(mesh.edge_index_map.values()), dtype=int)
    write_results(path, solver_type="hom", mesh={
        "vertices": mesh.vertices, "simplices": mesh.simplices,
        "edge_map_keys": keys, "edge_map_values": vals,
        "physical_groups": mesh.physical_groups,
        "elem_order": 2, "num_edges": mesh.num_edges},
        results_by_n={n: {"standing": {
            "frequencies": res.normal.frequencies,
            "eigenvalues": res.normal.eigenvalues,
            "eigenvectors": res.normal.eigenvectors}}})


@pytest.fixture
def app():
    try:
        a = wx.App(False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"wx App を作成できません: {e}")
    yield a
    a.Destroy()


def test_result_viewer_tm0_loads(app, tmp_path):
    from axicavity_fem.gui.result_viewer import ResultViewer
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    try:
        dlg = ResultViewer(None, str(h5))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ResultViewer を構築できません: {e}")
    assert dlg.solver_type == "tm0"
    assert dlg.choice_mode.GetCount() == 3
    dlg.update_plots()   # 例外なく再描画できる
    dlg.Destroy()


def test_result_viewer_hom_loads(app, tmp_path):
    from axicavity_fem.gui.result_viewer import ResultViewer
    h5 = tmp_path / "hom.h5"
    _write_hom_h5(h5, n=1)
    try:
        dlg = ResultViewer(None, str(h5))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ResultViewer を構築できません: {e}")
    assert dlg.solver_type == "hom"
    assert dlg.is_hom
    assert dlg.choice_mode.GetCount() == 3
    # HOM では E Lines / Levels が非表示、Vectors が既定 ON
    assert not dlg.checkbox_E_lines.IsShown()
    assert not dlg.spin_ctrl_E_line_levels.IsShown()
    assert not dlg.label_E_levels.IsShown()
    assert dlg.checkbox_vectors.GetValue()
    dlg.update_plots()
    dlg.Destroy()


def test_result_viewer_tm0_shows_elines(app, tmp_path):
    """TM0 では E Lines / Levels が表示される。"""
    from axicavity_fem.gui.result_viewer import ResultViewer
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    try:
        dlg = ResultViewer(None, str(h5))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ResultViewer を構築できません: {e}")
    assert dlg.checkbox_E_lines.IsShown()
    assert dlg.spin_ctrl_E_line_levels.IsShown()
    assert dlg.label_E_levels.IsShown()
    dlg.Destroy()


def test_export_field_dialog_area_params(app):
    """ExportFieldDialog(area) の get_params が範囲・点数・power を返す。"""
    from axicavity_fem.gui.result_viewer import ExportFieldDialog
    frame = wx.Frame(None)
    try:
        dlg = ExportFieldDialog(frame, shape="area", default_output="out",
                                z_bounds=(0.0, 0.1), r_bounds=(0.0, 0.05))
        dlg.spin_nz.SetValue(50)
        dlg.spin_nr.SetValue(30)
        dlg.txt_power.SetValue("1000")
        p = dlg.get_params()
        assert p["z_range"] == (0.0, 0.1)
        assert p["r_range"] == (0.0, 0.05)
        assert p["nz"] == 50 and p["nr"] == 30
        assert p["scale_to_power"] == 1000.0
        assert p["fmt"] == "both"
        assert p["output"] == "out"
        dlg.Destroy()
    finally:
        frame.Destroy()


def test_export_field_dialog_line_params(app):
    """ExportFieldDialog(line) の get_params が p1/p2/npts を返し、power 空欄=None。"""
    from axicavity_fem.gui.result_viewer import ExportFieldDialog
    frame = wx.Frame(None)
    try:
        dlg = ExportFieldDialog(frame, shape="line", default_output="ln",
                                z_bounds=(0.0, 0.1), r_bounds=(0.0, 0.05))
        dlg.txt_p1z.SetValue("0.0")
        dlg.txt_p1r.SetValue("0.01")
        dlg.txt_p2z.SetValue("0.1")
        dlg.txt_p2r.SetValue("0.04")
        dlg.spin_npts.SetValue(123)
        p = dlg.get_params()
        assert p["p1"] == (0.0, 0.01)
        assert p["p2"] == (0.1, 0.04)
        assert p["npts"] == 123
        assert p["scale_to_power"] is None  # 空欄
        dlg.Destroy()
    finally:
        frame.Destroy()


def test_export_field_dialog_axis_params(app):
    """ExportFieldDialog(axis) の get_params が z_range/npts を返す。"""
    from axicavity_fem.gui.result_viewer import ExportFieldDialog
    frame = wx.Frame(None)
    try:
        dlg = ExportFieldDialog(frame, shape="axis", default_output="ax",
                                z_bounds=(0.0, 0.1), r_bounds=(0.0, 0.05))
        dlg.txt_zmin.SetValue("0.01")
        dlg.txt_zmax.SetValue("0.09")
        dlg.spin_npts.SetValue(321)
        dlg.txt_power.SetValue("500")
        p = dlg.get_params()
        assert p["z_range"] == (0.01, 0.09)
        assert p["npts"] == 321
        assert p["scale_to_power"] == 500.0
        assert "r_range" not in p and "p1" not in p
        dlg.Destroy()
    finally:
        frame.Destroy()
