"""gmsh_export_occ: OCC + fragment ベースのエクスポートテスト。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)
from axicavity_fem.shared.gmsh_export_occ import (
    export_msh_multi_region,
    inspect_msh_physical_groups,
)
from axicavity_fem.shared.mesh_loader import load_mesh


# ---------------------------------------------------------------------------
# ジオメトリ生成ヘルパ
# ---------------------------------------------------------------------------
def _single_square_geom(side: float = 10.0) -> MultiRegionGeometry:
    pts = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
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


def _two_side_by_side_geom() -> MultiRegionGeometry:
    """左 [0,10]x[0,5] 真空、右 [10,20]x[0,5] 誘電体、共有境界 x=10。

    各領域は別の segment 群で描き、x=10 線は両領域で 2 度引かれている
    （fragment で共通化される）。
    """
    # 共有点: (10, 0) と (10, 5) を 1 つずつに
    pts = [
        (0.0, 0.0),   # 0  左下
        (10.0, 0.0),  # 1  中下
        (10.0, 5.0),  # 2  中上
        (0.0, 5.0),   # 3  左上
        (20.0, 0.0),  # 4  右下
        (20.0, 5.0),  # 5  右上
    ]
    segs = [
        # 左領域 (vacuum, CCW): 0->1->2->3->0
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),  # 共有
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        # 右領域 (dielectric, CCW): 1->4->5->2->1
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="PEC"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
        Segment(id=7, type="line", point_indices=[2, 1], bc_name="PEC"),  # 共有
    ]
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3], orientation="CCW"),
        Loop(id=1, segment_ids=[4, 5, 6, 7], orientation="CCW"),
    ]
    regions = [
        Region(id=0, name="Vacuum", outer_loop_id=0,
               material_tag="vacuum", eps_r=1.0),
        Region(id=1, name="Dielectric", outer_loop_id=1,
               material_tag="dielectric_1", eps_r=4.0),
    ]
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=regions,
        unit="mm", mesh_size=2.0,
    )


def _square_with_hole_geom() -> MultiRegionGeometry:
    """外側 [0,20]x[0,20] の中に [5,15]x[5,15] の穴を持つ 1 領域。"""
    pts = [
        (0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0),     # 0-3 外
        (5.0, 5.0), (5.0, 15.0), (15.0, 15.0), (15.0, 5.0),     # 4-7 穴 (CW: 4->5->6->7)
    ]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        # 穴は外周から見て CW (反転向き)
        Segment(id=4, type="line", point_indices=[4, 5], bc_name="PEC"),
        Segment(id=5, type="line", point_indices=[5, 6], bc_name="PEC"),
        Segment(id=6, type="line", point_indices=[6, 7], bc_name="PEC"),
        Segment(id=7, type="line", point_indices=[7, 4], bc_name="PEC"),
    ]
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3], orientation="CCW"),
        Loop(id=1, segment_ids=[4, 5, 6, 7], orientation="CW"),
    ]
    region = Region(id=0, name="Vacuum", outer_loop_id=0, hole_loop_ids=[1],
                    material_tag="vacuum", eps_r=1.0)
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=[region],
        unit="mm", mesh_size=2.0,
    )


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------
def test_export_single_region(tmp_path):
    geom = _single_square_geom()
    out = tmp_path / "single.msh"
    result = export_msh_multi_region(geom, out, verbose=0)

    assert out.exists()
    assert out.stat().st_size > 0
    # 1 領域 → 1 surface
    assert len(result.surface_tags_by_region_id[0]) == 1
    # Physical Surface 名 = "vacuum"
    assert "vacuum" in result.physical_groups[2]
    # BC 名 3 種類が登場
    assert set(result.physical_groups[1].keys()) >= {"PEC", "M-short", "E-short"}


def test_inspect_single_region(tmp_path):
    geom = _single_square_geom()
    out = tmp_path / "single.msh"
    export_msh_multi_region(geom, out, verbose=0)
    pg = inspect_msh_physical_groups(out)
    assert "vacuum" in pg[2]
    assert {"PEC", "M-short", "E-short"} <= set(pg[1].keys())


def test_export_two_regions_with_shared_boundary(tmp_path):
    geom = _two_side_by_side_geom()
    out = tmp_path / "two_regions.msh"
    result = export_msh_multi_region(geom, out, verbose=0)

    assert out.exists()
    assert "vacuum" in result.physical_groups[2]
    assert "dielectric_1" in result.physical_groups[2]
    # 各領域 1 surface
    assert len(result.surface_tags_by_region_id[0]) == 1
    assert len(result.surface_tags_by_region_id[1]) == 1


def test_two_regions_share_nodes_on_interface(tmp_path):
    """fragment で共有境界の節点が一致していることを確認する。

    左領域の節点集合と右領域の節点集合の x=0.01 (=10mm→m) 上の点が、
    両領域で完全に共有されていればよい。
    """
    geom = _two_side_by_side_geom()
    out = tmp_path / "two_regions.msh"
    export_msh_multi_region(geom, out, verbose=0)

    mesh = load_mesh(str(out))
    nodes = mesh.nodes  # (N, 2) (z, r) [m]

    # 単一節点配列なので、共有境界 (x=10mm = 0.01 m) 上の節点が存在する
    on_interface = np.where(np.isclose(nodes[:, 0], 0.010, atol=1e-9))[0]
    assert len(on_interface) >= 2, (
        f"interface should have shared nodes; got {len(on_interface)}"
    )

    # 重複節点が無いことを確認 (fragment で同じ位置の二重節点が出ないこと)
    coords = np.round(nodes, 12)
    unique_coords = {tuple(c) for c in coords}
    assert len(unique_coords) == len(coords), (
        "duplicate nodes detected — fragment did not merge interface nodes"
    )


def test_export_with_hole(tmp_path):
    geom = _square_with_hole_geom()
    out = tmp_path / "donut.msh"
    result = export_msh_multi_region(geom, out, verbose=0)

    assert out.exists()
    assert "vacuum" in result.physical_groups[2]

    # 穴が空いているか: メッシュの (z, r) 重心が穴の中（5<z<15, 5<r<15）に
    # 入る要素が 1 つも無いことを確認
    mesh = load_mesh(str(out))
    nodes = mesh.nodes
    elements = mesh.elements  # (Ne, 3) or (Ne, 6)
    # 1 次三角形の頂点 (最初の 3 列) で重心を計算
    tri_verts = nodes[elements[:, :3]]
    centroids = tri_verts.mean(axis=1)
    # メッシュ単位は m に変換されているので 5mm=0.005, 15mm=0.015
    in_hole = (
        (centroids[:, 0] > 0.005 + 1e-9)
        & (centroids[:, 0] < 0.015 - 1e-9)
        & (centroids[:, 1] > 0.005 + 1e-9)
        & (centroids[:, 1] < 0.015 - 1e-9)
    )
    assert not in_hole.any(), (
        f"{int(in_hole.sum())} element centroids are inside the hole"
    )


def test_export_mesh_order_2(tmp_path):
    geom = _single_square_geom()
    out = tmp_path / "single_o2.msh"
    export_msh_multi_region(geom, out, mesh_order=2, verbose=0)
    mesh = load_mesh(str(out))
    # 2 次三角形なら elements は (Ne, 6)
    assert mesh.elements.shape[1] == 6


def test_export_invalid_mesh_order(tmp_path):
    geom = _single_square_geom()
    out = tmp_path / "x.msh"
    with pytest.raises(ValueError):
        export_msh_multi_region(geom, out, mesh_order=3, verbose=0)


def test_export_validation_failure(tmp_path):
    geom = _single_square_geom()
    geom.segments[0].point_indices = [0, 99]  # 範囲外
    out = tmp_path / "x.msh"
    with pytest.raises(ValueError, match="validation failed"):
        export_msh_multi_region(geom, out, verbose=0)


def test_duplicate_material_tag_rejected(tmp_path):
    geom = _two_side_by_side_geom()
    geom.regions[1].material_tag = geom.regions[0].material_tag  # 重複
    out = tmp_path / "x.msh"
    with pytest.raises(ValueError, match="material_tag"):
        export_msh_multi_region(geom, out, verbose=0)
