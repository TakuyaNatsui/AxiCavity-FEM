"""ver2.3 誘電体損失 tanδ（摂動法 Q_diel）の検証テスト.

1. データモデル/リゾルバ/materials.json/HDF5 の tan_delta 配管。
2. 物理: 一様充填で Q_diel = 1/tanδ が（比の構成により）機械精度で成立、
   合成則 1/Q = 1/Q_wall + 1/Q_diel、部分充填では電気的充填率と整合。
3. 回帰: tanδ=0（None/全ゼロ）で従来出力とビット同一。
4. CLI e2e: sidecar materials.json → solve → post で Q_diel/Q_wall/P_diel が
   HDF5 と .txt サマリに出る。
"""

from __future__ import annotations

import json
import warnings
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from axicavity_fem.cli import cmd_post, cmd_solve
from axicavity_fem.cli.cmd_solve import _load_material_table
from axicavity_fem.fem_hom.post_process import compute_hom_parameters
from axicavity_fem.fem_hom.solver import solve_hom_standing
from axicavity_fem.fem_tm0.post_process import (
    _dielectric_weight_integrals,
    compute_tm0_parameters,
)
from axicavity_fem.fem_tm0.solver import solve_tm0_standing
from axicavity_fem.shared.boundary_groups import classify_boundaries
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.hdf5_io import read_results, write_results
from axicavity_fem.shared.material_resolver import (
    build_eps_r_per_element,
    build_material_table_from_geometry,
    build_tan_delta_per_element,
)
from axicavity_fem.shared.mesh_loader import load_mesh, load_mesh_hom
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)

EPS_R = 2.1
TAND = 1e-3


# ---------------------------------------------------------------------------
# ジオメトリ（test_fem_tm0_dielectric.py と同型の pillbox）
# ---------------------------------------------------------------------------
def _pillbox_geom(eps_r: float, tan_delta: float,
                  material_tag: str = "diel") -> MultiRegionGeometry:
    a, L = 50.0, 100.0
    pts = [(0.0, 0.0), (L, 0.0), (L, a), (0.0, a)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Diel", outer_loop_id=0,
                    material_tag=material_tag, eps_r=eps_r,
                    tan_delta=tan_delta)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[loop],
                               regions=[region], unit="mm", mesh_size=10.0)


def _two_region_pillbox(tand_right: float) -> MultiRegionGeometry:
    """z=L/2 で 2 領域に分け、右半分のみ tanδ を付ける（eps_r は両方 1）。"""
    a, L = 50.0, 100.0
    Lh = L / 2.0
    pts = [(0.0, 0.0), (Lh, 0.0), (Lh, a), (0.0, a), (L, 0.0), (L, a)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="E-short"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
    ]
    loops = [Loop(id=0, segment_ids=[0, 1, 2, 3]),
             Loop(id=1, segment_ids=[4, 5, 6, 1])]
    regions = [
        Region(id=0, name="Left", outer_loop_id=0, material_tag="vacuum"),
        Region(id=1, name="Right", outer_loop_id=1, material_tag="lossy",
               tan_delta=tand_right),
    ]
    return MultiRegionGeometry(points=pts, segments=segs, loops=loops,
                               regions=regions, unit="mm", mesh_size=10.0)


def _solve_tm0_with_materials(tmp_path, geom, elem_order=2, num_modes=3):
    msh = tmp_path / "pill.msh"
    export_msh_multi_region(geom, msh, mesh_order=elem_order, verbose=0)
    table = build_material_table_from_geometry(geom)
    mesh = load_mesh(msh, element_order=elem_order)
    eps_e = build_eps_r_per_element(mesh, table)
    tand_e = build_tan_delta_per_element(mesh, table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_tm0_standing(mesh, num_modes, material_table=table)
    cls = classify_boundaries(mesh.physical_groups, mesh.nodes)
    return res, cls, eps_e, tand_e


# ---------------------------------------------------------------------------
# 1) データモデル
# ---------------------------------------------------------------------------
def test_region_tan_delta_roundtrip():
    r = Region(id=0, name="D", outer_loop_id=0, material_tag="d",
               eps_r=9.0, tan_delta=1e-4, tan_delta_expr="a*2")
    d = r.to_dict()
    assert d["tan_delta"] == 1e-4
    assert d["tan_delta_expr"] == "a*2"
    r2 = Region.from_dict(d)
    assert r2.tan_delta == 1e-4
    assert r2.tan_delta_expr == "a*2"


def test_region_tan_delta_defaults_and_validation():
    # キー欠如（旧ファイル）→ 0.0
    r = Region.from_dict({"id": 0, "name": "V", "outer_loop_id": 0})
    assert r.tan_delta == 0.0
    assert r.tan_delta_expr is None
    # tan_delta=0 の to_dict に tan_delta_expr キーは無い
    assert "tan_delta_expr" not in r.to_dict()
    # 負値は ValueError
    with pytest.raises(ValueError):
        Region(id=0, name="X", outer_loop_id=0, tan_delta=-0.1)


def test_geometry_schema_22_file_still_loads():
    """tan_delta キーの無い schema 2.2 相当の dict も読める（0.0 扱い）。"""
    geom = _pillbox_geom(1.0, 0.0)
    d = geom.to_dict()
    d["schema_version"] = "2.2"
    for rd in d["regions"]:
        rd.pop("tan_delta", None)
        rd.pop("tan_delta_expr", None)
    geom2 = MultiRegionGeometry.from_dict(d)
    assert geom2.regions[0].tan_delta == 0.0


# ---------------------------------------------------------------------------
# 2) リゾルバ / materials.json / HDF5
# ---------------------------------------------------------------------------
def test_build_tan_delta_per_element(tmp_path):
    geom = _two_region_pillbox(tand_right=5e-3)
    msh = tmp_path / "two.msh"
    export_msh_multi_region(geom, msh, mesh_order=1, verbose=0)
    table = build_material_table_from_geometry(geom)
    assert table["lossy"]["tan_delta"] == 5e-3
    assert table["vacuum"]["tan_delta"] == 0.0

    mesh = load_mesh(msh, element_order=1)
    tand_e = build_tan_delta_per_element(mesh, table)
    assert tand_e is not None and len(tand_e) == mesh.num_elements
    assert set(np.unique(tand_e)) == {0.0, 5e-3}
    # 右半分 (z > L/2) の要素のみ tanδ > 0（メッシュ座標は m 単位に変換済み）
    centroids_z = mesh.nodes[mesh.elements[:, :3], 0].mean(axis=1)
    z_mid = 0.5 * (mesh.nodes[:, 0].min() + mesh.nodes[:, 0].max())
    assert np.all(tand_e[centroids_z > z_mid] == 5e-3)
    assert np.all(tand_e[centroids_z < z_mid] == 0.0)


def test_material_interface_geometry(tmp_path):
    """誘電体界面（材質が変わる内部辺）の抽出: 2 領域 pillbox の z=L/2 面。"""
    from axicavity_fem.shared.boundary_groups import material_interface_geometry
    geom = _two_region_pillbox(tand_right=5e-3)
    msh = tmp_path / "two.msh"
    export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
    table = build_material_table_from_geometry(geom)
    mesh = load_mesh(msh, element_order=2)
    segs, edges = material_interface_geometry(
        mesh.elements, mesh.nodes, mesh.element_order,
        eps_r_per_element=build_eps_r_per_element(mesh, table),
        tan_delta_per_element=build_tan_delta_per_element(mesh, table))
    assert segs and len(segs) == len(edges)
    # 界面辺はすべて z=L/2 平面上（eps_r は両領域 1 なので tanδ 差で検出）
    z_mid = 0.5 * (mesh.nodes[:, 0].min() + mesh.nodes[:, 0].max())
    for s in segs:
        assert np.allclose(np.asarray(s)[:, 0], z_mid, atol=1e-12)
    # 材質情報なし（真空メッシュ）→ 空
    assert material_interface_geometry(
        mesh.elements, mesh.nodes, mesh.element_order) == ([], [])


def test_build_tan_delta_legacy_mesh_returns_none():
    from axicavity_fem.shared.mesh_loader import MeshData
    md = MeshData(nodes=np.zeros((3, 2)), elements=np.array([[0, 1, 2]]),
                  element_order=1, physical_groups={})
    assert build_tan_delta_per_element(md, {"vacuum": {"tan_delta": 1.0}}) is None


def test_load_material_table_tan_delta(tmp_path):
    side = tmp_path / "m.materials.json"
    side.write_text(json.dumps({
        "schema": "axicavity-fem-v21.materials/1",
        "materials": {"d1": {"eps_r": 9.0, "tan_delta": 2e-4},
                      "vac": {"eps_r": 1.0}},
    }), encoding="utf-8")
    table = _load_material_table("dummy.msh", str(side))
    assert table["d1"]["tan_delta"] == 2e-4
    assert table["vac"]["tan_delta"] == 0.0  # キー欠如 → 0.0


def test_hdf5_tan_delta_roundtrip(tmp_path):
    mesh_dict = {
        "vertices": np.zeros((3, 2)), "simplices": np.array([[0, 1, 2]]),
        "edge_map_keys": None, "edge_map_values": None,
        "physical_groups": {}, "elem_order": 1, "num_edges": 0,
        "eps_r_per_element": np.array([9.0]),
        "tan_delta_per_element": np.array([1e-4]),
    }
    rec = {"standing": {"frequencies": np.array([1.0]), "eigenvalues": None,
                        "eigenvectors": np.array([[1.0, 2.0, 3.0]])}}
    out = tmp_path / "t.h5"
    write_results(out, solver_type="tm0", mesh=mesh_dict,
                  results_by_n={0: rec},
                  materials={"d1": {"eps_r": 9.0, "tan_delta": 1e-4}})
    data = read_results(out)
    np.testing.assert_allclose(data["mesh"]["tan_delta_per_element"], [1e-4])
    assert data["materials"]["d1"]["tan_delta"] == 1e-4

    # 書かなければ None / 0.0（旧ファイル互換）
    mesh_dict2 = dict(mesh_dict)
    mesh_dict2["tan_delta_per_element"] = None
    out2 = tmp_path / "t2.h5"
    write_results(out2, solver_type="tm0", mesh=mesh_dict2,
                  results_by_n={0: rec}, materials={"d1": {"eps_r": 9.0}})
    data2 = read_results(out2)
    assert data2["mesh"]["tan_delta_per_element"] is None
    assert data2["materials"]["d1"]["tan_delta"] == 0.0


# ---------------------------------------------------------------------------
# 3) 物理: TM0 一様充填 → Q_diel = 1/tanδ
# ---------------------------------------------------------------------------
def test_tm0_uniform_fill_q_diel(tmp_path):
    res, cls, eps_e, tand_e = _solve_tm0_with_materials(
        tmp_path, _pillbox_geom(EPS_R, TAND))
    params = compute_tm0_parameters(res, cls, eps_r_per_element=eps_e,
                                    tan_delta_per_element=tand_e)
    ref = compute_tm0_parameters(res, cls, eps_r_per_element=eps_e)
    for mp, mp_ref in zip(params.per_phase[0.0], ref.per_phase[0.0]):
        # 一様充填: 比の構成により厳密に 1/tanδ
        assert mp.q_diel == pytest.approx(1.0 / TAND, rel=1e-9)
        # q_wall は tanδ 無しの従来 q_factor と厳密一致
        assert mp.q_wall == mp_ref.q_factor
        # 合成則 1/Q = 1/Q_wall + 1/Q_diel
        assert 1.0 / mp.q_factor == pytest.approx(
            1.0 / mp.q_wall + 1.0 / mp.q_diel, rel=1e-12)
        # P_diel = ωU/Q_diel
        omega = 2 * np.pi * mp.frequency_ghz * 1e9
        assert mp.p_diel == pytest.approx(
            omega * mp.stored_energy * TAND, rel=1e-9)


# ---------------------------------------------------------------------------
# 4) 物理: HOM 一様充填 → Q_diel = 1/tanδ
# ---------------------------------------------------------------------------
def test_hom_uniform_fill_q_diel(tmp_path):
    geom = _pillbox_geom(EPS_R, TAND)
    msh = tmp_path / "pill.msh"
    export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
    table = build_material_table_from_geometry(geom)
    mesh = load_mesh_hom(msh, 1, 2)
    eps_e = build_eps_r_per_element(mesh, table)
    tand_e = build_tan_delta_per_element(mesh, table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_hom_standing(mesh, 3, material_table=table)
    params = compute_hom_parameters(res, eps_r_per_element=eps_e,
                                    tan_delta_per_element=tand_e)
    ref = compute_hom_parameters(res, eps_r_per_element=eps_e)
    for mp, mp_ref in zip(params.per_phase[0.0], ref.per_phase[0.0]):
        assert mp.q_diel == pytest.approx(1.0 / TAND, rel=1e-9)
        assert mp.q_wall == mp_ref.q_factor
        assert 1.0 / mp.q_factor == pytest.approx(
            1.0 / mp.q_wall + 1.0 / mp.q_diel, rel=1e-12)
        omega = 2 * np.pi * mp.frequency_ghz * 1e9
        assert mp.p_diel == pytest.approx(
            omega * mp.stored_energy * TAND, rel=1e-9)


# ---------------------------------------------------------------------------
# 5) 物理: 部分充填（右半分のみ tanδ）→ 電気的充填率と整合
# ---------------------------------------------------------------------------
def test_tm0_partial_fill_q_diel(tmp_path):
    tand_right = 1e-3
    res, cls, eps_e, tand_e = _solve_tm0_with_materials(
        tmp_path, _two_region_pillbox(tand_right))
    assert eps_e is not None and np.all(eps_e == 1.0)
    params = compute_tm0_parameters(res, cls, eps_r_per_element=eps_e,
                                    tan_delta_per_element=tand_e)
    nodes = np.asarray(res.nodes)
    elements = np.asarray(res.elements)
    for m, mp in enumerate(params.per_phase[0.0]):
        # 0 < 1/Q_diel < tanδ（部分充填なので必ず全充填より小さい損失）
        assert 0.0 < 1.0 / mp.q_diel < tand_right
        # 独立クロスチェック: 1/Q_diel = tanδ × 電気的充填率
        u = params.normalized_eigenvectors[0.0][m]
        w_total, w_right = _dielectric_weight_integrals(
            nodes, elements, res.element_order, u, eps_e,
            (tand_e > 0).astype(float))
        fill = w_right / w_total
        assert 0.0 < fill < 1.0
        assert 1.0 / mp.q_diel == pytest.approx(tand_right * fill, rel=1e-12)


# ---------------------------------------------------------------------------
# 6) 回帰: tanδ=0（None/全ゼロ）で従来出力とビット同一
# ---------------------------------------------------------------------------
def test_tm0_zero_tan_delta_bit_identical(tmp_path):
    res, cls, eps_e, _ = _solve_tm0_with_materials(
        tmp_path, _pillbox_geom(EPS_R, 0.0))
    ref = compute_tm0_parameters(res, cls, eps_r_per_element=eps_e)
    zero = compute_tm0_parameters(res, cls, eps_r_per_element=eps_e,
                                  tan_delta_per_element=np.zeros_like(eps_e))
    for a, b in zip(ref.per_phase[0.0], zero.per_phase[0.0]):
        assert a.q_factor == b.q_factor
        assert a.p_loss == b.p_loss
        assert a.stored_energy == b.stored_energy
        assert a.v_eff == b.v_eff
        assert a.rq_eff == b.rq_eff
        assert b.q_diel == 0.0 and b.p_diel == 0.0
        assert b.q_wall == b.q_factor  # 無損失時は q_wall = q_factor


# ---------------------------------------------------------------------------
# 7) CLI e2e: sidecar → solve → post → HDF5 attrs + .txt サマリ
# ---------------------------------------------------------------------------
def test_cli_e2e_tan_delta(tmp_path):
    geom = _pillbox_geom(EPS_R, TAND)
    msh = tmp_path / "pill.msh"
    export_msh_multi_region(geom, msh, mesh_order=1, verbose=0)
    side = tmp_path / "pill.materials.json"
    side.write_text(json.dumps({
        "schema": "axicavity-fem-v21.materials/1",
        "materials": build_material_table_from_geometry(geom),
    }), encoding="utf-8")

    h5 = tmp_path / "pill_tm0.h5"
    args = Namespace(mesh_file=str(msh), elem_order=1, num_modes=2,
                     phase="0.0", output_file=str(h5), type="tm0",
                     az_order=[0], materials_file=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cmd_solve.run(args) == 0

    post_args = Namespace(input_file=str(h5), output_file=None,
                          cond=5.8e7, beta=1.0, type="tm0")
    assert cmd_post.run(post_args) == 0

    with h5py.File(h5, "r") as f:
        # solve 段階で tanδ 配列と材料属性が保存される
        np.testing.assert_allclose(
            np.unique(f["mesh/tan_delta_per_element"][:]), [TAND])
        assert f["materials/diel"].attrs["tan_delta"] == TAND
        # post 段階で Q_wall/Q_diel/P_diel が追記される
        attrs = f["post_process/n0/standing/mode_0"].attrs
        assert attrs["Q_diel"] == pytest.approx(1.0 / TAND, rel=1e-9)
        assert attrs["Q_wall"] > attrs["Q"] > 0
        assert attrs["P_diel"] > 0
        assert 1.0 / attrs["Q"] == pytest.approx(
            1.0 / attrs["Q_wall"] + 1.0 / attrs["Q_diel"], rel=1e-12)

    txt = h5.with_suffix(".txt").read_text(encoding="utf-8")
    assert "Q_diel" in txt and "Q_wall" in txt and "P_diel" in txt
