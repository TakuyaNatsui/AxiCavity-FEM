"""段階7e 検証: reports.report_builder の HTML レポート生成（スモーク）.

index.html と場マップ PNG が生成されることを確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

from axicavity_fem.shared.hdf5_io import append_post_process, write_results  # noqa: E402

_CYL_2ND = Path(__file__).resolve().parents[1] / "samples" / "cylinder100mm.msh"

pytestmark = pytest.mark.skipif(
    not _CYL_2ND.exists(), reason="cylinder100mm.msh なし")


def _write_tm0_processed(path):
    from axicavity_fem.fem_tm0.post_process import compute_tm0_parameters
    from axicavity_fem.fem_tm0.solver import solve_tm0_standing
    from axicavity_fem.shared.boundary_groups import classify_boundaries
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
    cls = classify_boundaries(mesh.physical_groups, mesh.nodes)
    params = compute_tm0_parameters(res, cls).per_phase[0.0]
    append_post_process(path, {0: {"standing": [
        {"Q": p.q_factor, "U_stored": p.stored_energy, "P_loss": p.p_loss,
         "R_over_Q": p.rq_eff, "V_eff": p.v_eff, "frequency_GHz": p.frequency_ghz}
        for p in params]}})


def test_build_report_tm0(tmp_path):
    from axicavity_fem.reports.report_builder import build_report
    h5 = tmp_path / "tm0.h5"
    _write_tm0_processed(h5)
    out = tmp_path / "rep"
    index = build_report(str(h5), str(out), dpi=60)
    assert Path(index).exists()
    files = list(out.iterdir())
    assert any(f.name == "index.html" for f in files)
    assert any(f.suffix == ".png" for f in files)
    text = Path(index).read_text(encoding="utf-8")
    assert "AxiCavity-FEM Analysis Report" in text
    assert "Q" in text


def test_build_report_via_cli(tmp_path):
    from axicavity_fem.cli.main import cli_main
    h5 = tmp_path / "tm0.h5"
    _write_tm0_processed(h5)
    out = tmp_path / "rep_cli"
    rc = cli_main(["report", "--type", "tm0", "-i", str(h5), "-o", str(out),
                   "--dpi", "60"])
    assert rc == 0 and (out / "index.html").exists()


def test_build_report_hom(tmp_path):
    from axicavity_fem.fem_hom.solver import solve_hom_standing
    from axicavity_fem.reports.report_builder import build_report
    from axicavity_fem.shared.mesh_loader import load_mesh_hom
    mesh = load_mesh_hom(str(_CYL_2ND), 1, 2)
    res = solve_hom_standing(mesh, num_modes=3)
    keys = np.array(list(mesh.edge_index_map.keys()), dtype=int)
    vals = np.array(list(mesh.edge_index_map.values()), dtype=int)
    h5 = tmp_path / "hom.h5"
    write_results(h5, solver_type="hom", mesh={
        "vertices": mesh.vertices, "simplices": mesh.simplices,
        "edge_map_keys": keys, "edge_map_values": vals,
        "physical_groups": mesh.physical_groups,
        "elem_order": 2, "num_edges": mesh.num_edges},
        results_by_n={1: {"standing": {
            "frequencies": res.normal.frequencies,
            "eigenvalues": res.normal.eigenvalues,
            "eigenvectors": res.normal.eigenvectors}}})
    out = tmp_path / "hrep"
    index = build_report(str(h5), str(out), dpi=60)
    assert Path(index).exists()
    assert any(f.suffix == ".png" for f in out.iterdir())
