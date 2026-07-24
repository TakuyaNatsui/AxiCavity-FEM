"""material_resolver.check_port_and_axis_are_vacuum と
solver.solve_tm0_standing の警告動作テスト。"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from axicavity_fem.fem_tm0.solver import solve_tm0_standing
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.material_resolver import (
    build_eps_r_per_element,
    build_material_table_from_geometry,
    check_port_and_axis_are_vacuum,
)
from axicavity_fem.shared.mesh_loader import load_mesh
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)


def _pillbox(eps_r: float, material_tag: str = VACUUM_TAG) -> MultiRegionGeometry:
    a, L = 50.0, 100.0
    pts = [(0.0, 0.0), (L, 0.0), (L, a), (0.0, a)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="R", outer_loop_id=0,
                    material_tag=material_tag, eps_r=eps_r)
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=[loop], regions=[region],
        unit="mm", mesh_size=10.0,
    )


def test_sanity_returns_empty_for_vacuum(tmp_path):
    geom = _pillbox(1.0)
    out = tmp_path / "x.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)
    eps_r = build_eps_r_per_element(mesh, build_material_table_from_geometry(geom))
    msgs = check_port_and_axis_are_vacuum(mesh, eps_r)
    assert msgs == []


def test_sanity_returns_empty_when_none(tmp_path):
    geom = _pillbox(1.0)
    out = tmp_path / "x.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)
    assert check_port_and_axis_are_vacuum(mesh, None) == []


def test_sanity_flags_dielectric_on_axis(tmp_path):
    """全領域 eps_r=4 にすると、軸とポート上に dielectric が乗るため
    両方の警告が出る。"""
    geom = _pillbox(4.0, material_tag="dielectric_full")
    out = tmp_path / "x.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)
    eps_r = build_eps_r_per_element(
        mesh, {"dielectric_full": {"eps_r": 4.0, "mu_r": 1.0}},
    )
    msgs = check_port_and_axis_are_vacuum(mesh, eps_r)
    # 軸とポートの両方で違反
    assert any("axis" in m for m in msgs)
    assert any("ポート" in m for m in msgs)


def test_solver_warns_for_dielectric_on_axis(tmp_path):
    """solve_tm0_standing が ver2.1 制約警告を出す。"""
    geom = _pillbox(4.0, material_tag="dielectric_full")
    out = tmp_path / "x.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        solve_tm0_standing(
            mesh, num_modes=3,
            material_table={"dielectric_full": {"eps_r": 4.0, "mu_r": 1.0}},
        )
    msgs = [str(w.message) for w in captured]
    assert any("ver2.1" in m for m in msgs), msgs


def test_solver_no_warning_for_vacuum(tmp_path):
    """ε_r=1 一様なら警告無し。"""
    geom = _pillbox(1.0)
    out = tmp_path / "x.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        solve_tm0_standing(mesh, num_modes=3, material_table=None)
    msgs = [str(w.message) for w in captured]
    # ver2.1 制約警告は出ない (他の matplotlib/numpy の deprecation は無視)
    assert not any("ver2.1" in m for m in msgs)
