"""段階7b 検証: reports.export_fields の場データ出力.

export がサンプリングした場の値が、同じ固有ベクトルから直接再構成した値
（field_recon）と一致することを確認する。eigsh の不安定性を避けるため、
1 回だけ解いた結果を H5 に書き出してから export する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("h5py")
import h5py  # noqa: E402

from axicavity_fem.reports.export_fields import export_fields  # noqa: E402
from axicavity_fem.shared.hdf5_io import read_results, write_results  # noqa: E402

_CYL_2ND = Path(__file__).resolve().parents[1] / "samples" / "cylinder100mm.msh"

pytestmark = pytest.mark.skipif(
    not _CYL_2ND.exists(), reason="cylinder100mm.msh なし")


def _write_tm0_h5(path):
    from axicavity_fem.fem_tm0.solver import solve_tm0_standing
    from axicavity_fem.shared.mesh_loader import load_mesh
    mesh = load_mesh(str(_CYL_2ND), 2)
    res = solve_tm0_standing(mesh, num_modes=4)
    write_results(path, solver_type="tm0", mesh={
        "vertices": res.nodes, "simplices": res.elements,
        "edge_map_keys": None, "edge_map_values": None,
        "physical_groups": res.physical_groups,
        "elem_order": res.element_order, "num_edges": 0},
        results_by_n={0: {"standing": {
            "frequencies": res.frequencies, "eigenvalues": None,
            "eigenvectors": res.eigenvectors}}})
    return res


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
    return mesh, res


def test_export_tm0_area_matches_field_recon(tmp_path):
    from axicavity_fem.fem_tm0.field_recon import TM0FieldReconstructor

    h5 = tmp_path / "tm0.h5"
    res = _write_tm0_h5(h5)

    data = export_fields(str(h5), str(tmp_path / "area"), solver_type="tm0",
                         mode=0, shape="area", nz=8, nr=5, fmt="h5")

    recon = TM0FieldReconstructor(res.nodes, res.elements,
                                  res.element_order, "standing")
    eigvec, freq = res.eigenvectors[0], res.frequencies[0]
    checked = 0
    for i, z in enumerate(data["z_vec"]):
        for j, r in enumerate(data["r_vec"]):
            if not data["mask"][i, j]:
                continue
            f = recon.calculate_fields(eigvec, freq, z, r)
            assert np.isclose(data["Ez"][i, j], f["Ez"], rtol=1e-9, atol=1e-6)
            assert np.isclose(data["H_theta"][i, j], f["H_theta"],
                              rtol=1e-9, atol=1e-9)
            checked += 1
    assert checked > 0

    # H5 が読めて構造が正しいこと
    with h5py.File(tmp_path / "area.h5", "r") as g:
        assert g.attrs["shape"] == "area"
        assert g["Ez"].shape == (8, 5)


def test_export_tm0_axis_and_txt(tmp_path):
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    data = export_fields(str(h5), str(tmp_path / "axis"), solver_type="tm0",
                         mode=0, shape="axis", npts=30, fmt="both")
    assert len(data["Ez"]) == 30
    assert (tmp_path / "axis.h5").exists()
    assert (tmp_path / "axis.txt").exists()
    # 軸上 (r=0) では E_r = 0
    assert np.allclose(np.real(data["Er"]), 0.0, atol=1e-6)


def test_export_tm0_line(tmp_path):
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    data = export_fields(str(h5), str(tmp_path / "line"), solver_type="tm0",
                         mode=0, shape="line", p1=(0.0, 0.0), p2=(0.0, 0.04),
                         npts=20, fmt="h5")
    assert len(data["Ez"]) == 20
    assert np.isclose(data["length"], 0.04, rtol=1e-9)


def test_export_hom_area_matches_field_recon(tmp_path):
    from axicavity_fem.fem_hom.field_recon import HOMFieldReconstructor

    h5 = tmp_path / "hom.h5"
    mesh, res = _write_hom_h5(h5, n=1)

    data = export_fields(str(h5), str(tmp_path / "harea"), solver_type="hom",
                         mode=0, n=1, shape="area", nz=6, nr=4, fmt="h5")

    recon = HOMFieldReconstructor(mesh.simplices, mesh.vertices,
                                  mesh.edge_index_map, 2, 1, "standing")
    recon.load_mode(res.normal.eigenvectors[0], res.normal.frequencies[0])
    checked = 0
    for i, z in enumerate(data["z_vec"]):
        for j, r in enumerate(data["r_vec"]):
            if not data["mask"][i, j]:
                continue
            f = recon.calculate_fields(z, r)
            assert np.isclose(data["Ez"][i, j], f["Ez"], rtol=1e-9, atol=1e-6)
            assert np.isclose(data["E_theta"][i, j], f["E_theta"],
                              rtol=1e-9, atol=1e-6)
            checked += 1
    assert checked > 0


def test_export_via_cli(tmp_path):
    from axicavity_fem.cli.main import cli_main
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    rc = cli_main(["export", "--type", "tm0", "-i", str(h5),
                   "-o", str(tmp_path / "out"), "--shape", "axis",
                   "--npts", "20", "--format", "h5"])
    assert rc == 0
    assert (tmp_path / "out.h5").exists()
