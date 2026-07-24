"""ver2.1 段階9 配線: 多領域メッシュ + sidecar materials.json で CLI が
material_table を渡しているかの end-to-end 検証.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from axicavity_fem.cli import cmd_solve
from axicavity_fem.fem_tm0.assembly import assemble_global_matrices
from axicavity_fem.fem_tm0.boundary import build_dirichlet_nodes
from axicavity_fem.shared.boundary_groups import classify_boundaries
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.material_resolver import (
    build_eps_r_per_element,
    build_material_table_from_geometry,
)
from axicavity_fem.shared.mesh_loader import load_mesh
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)


def _two_region_pillbox(eps_r_left: float, eps_r_right: float) -> MultiRegionGeometry:
    """z=L/2 で 2 領域に分けた pillbox。各領域に eps_r を指定。"""
    a = 50.0
    L = 100.0
    Lh = L / 2.0
    pts = [
        (0.0, 0.0), (Lh, 0.0), (Lh, a), (0.0, a),
        (L, 0.0), (L, a),
    ]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="E-short"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
    ]
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3]),
        Loop(id=1, segment_ids=[4, 5, 6, 1]),
    ]
    regions = [
        Region(id=0, name="Left", outer_loop_id=0,
               material_tag="vacuum", eps_r=eps_r_left),
        Region(id=1, name="Right", outer_loop_id=1,
               material_tag="dielectric_1", eps_r=eps_r_right),
    ]
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=regions,
        unit="mm", mesh_size=10.0,
    )


def _make_msh_with_sidecar(tmp_path: Path, eps_r_left: float, eps_r_right: float,
                            *, write_sidecar: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    geom = _two_region_pillbox(eps_r_left, eps_r_right)
    msh_path = tmp_path / "two_region.msh"
    export_msh_multi_region(geom, msh_path, mesh_order=1, verbose=0)
    if write_sidecar:
        side = tmp_path / "two_region.materials.json"
        side.write_text(json.dumps({
            "schema": "axicavity-fem-v21.materials/1",
            "materials": build_material_table_from_geometry(geom),
        }), encoding="utf-8")
    return msh_path


def _solve_via_cli(mesh_path: Path, out_path: Path, *,
                   materials_file: str | None = None,
                   num_modes: int = 3) -> Path:
    """cmd_solve.run() を Namespace で直接呼ぶ。"""
    args = Namespace(
        mesh_file=str(mesh_path),
        elem_order=1,
        num_modes=num_modes,
        phase="0.0",
        output_file=str(out_path),
        type="tm0",
        az_order=[0],
        materials_file=materials_file,
    )
    rc = cmd_solve.run(args)
    assert rc == 0, f"cmd_solve.run() returned {rc}"
    return out_path


def _read_standing_frequencies(h5_path: Path) -> np.ndarray:
    """ver2 HDF5 スキーマ: results/n0/standing/mode_<i>/@frequency_GHz."""
    freqs: list[float] = []
    with h5py.File(h5_path, "r") as f:
        grp = f["results/n0/standing"]
        mode_keys = sorted(k for k in grp.keys() if k.startswith("mode_"))
        for k in mode_keys:
            freqs.append(float(grp[k].attrs["frequency_GHz"]))
    return np.asarray(freqs)


def _smallest_lambda_dense(mesh, eps_r) -> float:
    """dense eigh で最小一般化固有値を返す (shift-invert を避ける)。"""
    from scipy.linalg import eigh
    K, M = assemble_global_matrices(mesh, eps_r)
    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)
    free = np.setdiff1d(np.arange(mesh.num_nodes),
                        np.asarray(dirichlet_nodes, dtype=int))
    Kd = K.toarray()[np.ix_(free, free)]
    Md = M.toarray()[np.ix_(free, free)]
    eigvals = eigh(Kd, Md, eigvals_only=True)
    pos = eigvals[eigvals > 1e-6]
    return float(np.min(pos))


# ---------------------------------------------------------------------------
# 1) sidecar 自動読込 — eps_r=1 の sidecar は無いケースと一致
# ---------------------------------------------------------------------------
def test_cli_autodetects_sidecar_materials(tmp_path):
    """両領域 eps_r=1 の sidecar を作っても、無いケースと固有周波数が一致する。"""
    mesh_path = _make_msh_with_sidecar(tmp_path, 1.0, 1.0, write_sidecar=True)

    h5_with = _solve_via_cli(mesh_path, tmp_path / "with.h5")
    freqs_with = _read_standing_frequencies(h5_with)

    (tmp_path / "two_region.materials.json").unlink()
    h5_without = _solve_via_cli(mesh_path, tmp_path / "without.h5")
    freqs_without = _read_standing_frequencies(h5_without)

    np.testing.assert_allclose(freqs_with, freqs_without, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2) --materials 指定で eps_r が反映される (dense eigh で照合)
# ---------------------------------------------------------------------------
def test_cli_materials_arg_changes_lowest_eigenvalue(tmp_path):
    """全領域 eps_r=4 にすると、最小固有値が 1/4 倍 (周波数 1/2 倍) になる。

    eigsh shift-invert は ε_r が大きいと最低モードを見逃す既知問題があるため、
    dense eigh で λ_min を直接照合する。
    """
    # vacuum
    mesh_vac = _make_msh_with_sidecar(tmp_path / "vac", 1.0, 1.0,
                                       write_sidecar=False)
    mesh = load_mesh(mesh_vac, element_order=1)
    eps_r_vac = build_eps_r_per_element(mesh, None)
    if eps_r_vac is None:
        eps_r_vac = np.ones(mesh.num_elements)
    lam_vac = _smallest_lambda_dense(mesh, eps_r_vac)

    # eps_r=4 全域
    mesh_diel = _make_msh_with_sidecar(tmp_path / "diel", 4.0, 4.0,
                                        write_sidecar=True)
    mesh = load_mesh(mesh_diel, element_order=1)
    materials_path = tmp_path / "diel" / "two_region.materials.json"
    with open(materials_path) as f:
        table = json.load(f)["materials"]
    eps_r_diel = build_eps_r_per_element(mesh, table)
    lam_diel = _smallest_lambda_dense(mesh, eps_r_diel)

    # λ ∝ 1/eps_r → 1/4 倍
    assert lam_vac / lam_diel == pytest.approx(4.0, rel=1e-10), \
        f"lam_vac={lam_vac}, lam_diel={lam_diel}, ratio={lam_vac/lam_diel}"
