"""mesh_loader の領域マッピング拡張のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from axicavity_fem.shared.mesh_loader import load_mesh
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.material_resolver import (
    build_eps_r_per_element,
    build_material_table_from_geometry,
)
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)


# ---------------------------------------------------------------------------
# ジオメトリ
# ---------------------------------------------------------------------------
def _single_region_geom() -> MultiRegionGeometry:
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Vacuum", outer_loop_id=0,
                    material_tag=VACUUM_TAG, eps_r=1.0)
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=[loop], regions=[region],
        unit="mm", mesh_size=2.0,
    )


def _two_region_geom(eps_r_left=1.0, eps_r_right=4.0) -> MultiRegionGeometry:
    pts = [
        (0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0),
        (20.0, 0.0), (20.0, 5.0),
    ]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="PEC"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
        Segment(id=7, type="line", point_indices=[2, 1], bc_name="PEC"),
    ]
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3]),
        Loop(id=1, segment_ids=[4, 5, 6, 7]),
    ]
    regions = [
        Region(id=0, name="Vacuum", outer_loop_id=0,
               material_tag="vacuum", eps_r=eps_r_left),
        Region(id=1, name="Dielectric", outer_loop_id=1,
               material_tag="dielectric_1", eps_r=eps_r_right),
    ]
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=regions,
        unit="mm", mesh_size=2.0,
    )


# ---------------------------------------------------------------------------
# mesh_loader の拡張
# ---------------------------------------------------------------------------
def test_load_mesh_accepts_pathlib(tmp_path):
    geom = _single_region_geom()
    out = tmp_path / "x.msh"
    export_msh_multi_region(geom, out, verbose=0)
    mesh = load_mesh(out, element_order=1)  # Path 直接
    assert mesh.num_elements > 0


def test_load_mesh_two_regions_element_region_ids(tmp_path):
    geom = _two_region_geom()
    out = tmp_path / "two.msh"
    export_msh_multi_region(geom, out, verbose=0)

    mesh = load_mesh(out, element_order=1)
    assert mesh.element_region_ids is not None
    assert mesh.region_names is not None
    # 2 つの material_tag があるはず
    tags = set(mesh.region_names.values())
    assert tags == {"vacuum", "dielectric_1"}
    # 全要素が -1 でなく、いずれかの region に属する
    assert int(np.min(mesh.element_region_ids)) >= 0


def test_load_mesh_assigns_correct_region_by_position(tmp_path):
    """左 (z<10mm) の要素は vacuum, 右 (z>10mm) は dielectric_1 に属する。"""
    geom = _two_region_geom()
    out = tmp_path / "two.msh"
    export_msh_multi_region(geom, out, verbose=0)

    mesh = load_mesh(out, element_order=1)
    region_names = mesh.region_names

    # 各要素の重心 z を出す (元単位は mm → m なので 10mm=0.010)
    tri_verts = mesh.nodes[mesh.elements[:, :3]]
    centroids = tri_verts.mean(axis=1)

    for i, rid in enumerate(mesh.element_region_ids):
        name = region_names[int(rid)]
        if name == "vacuum":
            assert centroids[i, 0] < 0.010 + 1e-9, (
                f"vacuum element at z={centroids[i, 0]} expected < 0.010"
            )
        elif name == "dielectric_1":
            assert centroids[i, 0] > 0.010 - 1e-9, (
                f"dielectric element at z={centroids[i, 0]} expected > 0.010"
            )


def test_load_mesh_legacy_no_2d_physical_group(tmp_path):
    """ver2 形式 (2D Physical Group なし) のメッシュは
    element_region_ids = None になる。
    """
    # 単一領域メッシュを v2.1 以降で生成しても 2D Physical Group が 1 つは付くため、None にはならない。
    # なので Physical Group 無しの最小メッシュをここで自作する。
    import gmsh
    out = tmp_path / "nophys.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0.0)
        gmsh.model.add("nophys")
        p1 = gmsh.model.geo.addPoint(0, 0, 0, 0.5)
        p2 = gmsh.model.geo.addPoint(1, 0, 0, 0.5)
        p3 = gmsh.model.geo.addPoint(1, 1, 0, 0.5)
        p4 = gmsh.model.geo.addPoint(0, 1, 0, 0.5)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out))
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

    mesh = load_mesh(out, element_order=1)
    assert mesh.element_region_ids is None
    assert mesh.region_names is None


# ---------------------------------------------------------------------------
# material_resolver
# ---------------------------------------------------------------------------
def test_material_table_from_geometry():
    geom = _two_region_geom(eps_r_left=1.0, eps_r_right=4.0)
    table = build_material_table_from_geometry(geom)
    assert table == {
        "vacuum": {"eps_r": 1.0, "mu_r": 1.0, "tan_delta": 0.0},
        "dielectric_1": {"eps_r": 4.0, "mu_r": 1.0, "tan_delta": 0.0},
    }


def test_build_eps_r_per_element(tmp_path):
    geom = _two_region_geom(eps_r_left=1.0, eps_r_right=4.0)
    out = tmp_path / "two.msh"
    export_msh_multi_region(geom, out, verbose=0)
    mesh = load_mesh(out, element_order=1)
    table = build_material_table_from_geometry(geom)
    eps_r = build_eps_r_per_element(mesh, table)

    assert eps_r is not None
    assert eps_r.shape == (mesh.num_elements,)
    # 値は 1.0 か 4.0 のいずれか
    assert set(np.unique(eps_r).tolist()) <= {1.0, 4.0}
    # 各値が少なくとも 1 要素以上に出ている
    assert (eps_r == 1.0).any()
    assert (eps_r == 4.0).any()


def test_build_eps_r_per_element_legacy_returns_none(tmp_path):
    """ver2 形式 (region_ids なし) の mesh では None を返す。"""
    # nophys mesh を作る (上の helper を流用)
    import gmsh
    out = tmp_path / "nophys.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0.0)
        gmsh.model.add("x")
        p1 = gmsh.model.geo.addPoint(0, 0, 0, 0.5)
        p2 = gmsh.model.geo.addPoint(1, 0, 0, 0.5)
        p3 = gmsh.model.geo.addPoint(1, 1, 0, 0.5)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p1)
        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3])
        gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out))
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
    mesh = load_mesh(out, element_order=1)
    assert build_eps_r_per_element(mesh, {}) is None


def test_build_eps_r_unknown_region_uses_default(tmp_path):
    geom = _two_region_geom()
    out = tmp_path / "two.msh"
    export_msh_multi_region(geom, out, verbose=0)
    mesh = load_mesh(out, element_order=1)
    # 空のテーブル → 全要素 default_eps_r
    eps_r = build_eps_r_per_element(mesh, {}, default_eps_r=2.5)
    assert eps_r is not None
    assert np.allclose(eps_r, 2.5)
