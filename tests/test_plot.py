"""段階7c 検証: reports.plot_* の場マップ PNG 生成（スモーク）.

画像の見た目は自動検証できないため、PNG が生成され非空であることを確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

from axicavity_fem.shared.hdf5_io import read_results, write_results  # noqa: E402

_CYL_2ND: Path       # samples/cylinder100mm.gmshproj（r = 50 mm, L = 100 mm）の 2 次メッシュ


@pytest.fixture(autouse=True)
def _meshes(sample_mesh):
    global _CYL_2ND
    _CYL_2ND = sample_mesh("cylinder100mm", 2)


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


def test_plot_tm0_png(tmp_path):
    from axicavity_fem.reports.plot_tm0 import plot_tm0_mode
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    out = tmp_path / "tm0.png"
    plot_tm0_mode(read_results(h5), 0, str(out), dpi=60)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_hom_png(tmp_path):
    from axicavity_fem.reports.plot_hom import plot_hom_mode
    h5 = tmp_path / "hom.h5"
    _write_hom_h5(h5, n=1)
    out = tmp_path / "hom.png"
    plot_hom_mode(read_results(h5), 0, str(out), n=1, dpi=60)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_via_cli(tmp_path):
    from axicavity_fem.cli.main import cli_main
    h5 = tmp_path / "tm0.h5"
    _write_tm0_h5(h5)
    out = tmp_path / "cli.png"
    rc = cli_main(["plot", "--type", "tm0", "-i", str(h5), "-m", "0",
                   "-o", str(out), "--dpi", "60"])
    assert rc == 0 and out.exists() and out.stat().st_size > 1000
