"""HOM 誘電体対応の物理検証テスト。

HOM は **E 場定式化**なので、誘電体では質量行列 M に ε_r を乗じる
（TM0 の H_φ 定式化が K に 1/ε_r を乗じるのと逆）。

1. 組立レベル: M_eps == ε_r·M_vac かつ K_eps == K_vac（K 不変）。
   ← 定式化の核心（M 側 / K 側のどちらに ε_r が入るかを直接検証）
2. 一様 ε_r=ε で最小固有値 λ_min が 1/ε 倍（dense eigh で照合、shift-invert 回避）。
3. material_table=None と eps_r=1 一様 material_table が完全一致（ver1 互換維持）。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from axicavity_fem.fem_hom.assembly import assemble_global_matrices
from axicavity_fem.fem_hom.boundary import (
    apply_bc_transformation,
    create_transformation_matrix,
    get_pec_dof_indices_1st,
    get_pec_dof_indices_2nd,
)
from axicavity_fem.fem_hom.solver import solve_hom_standing
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.material_resolver import (
    build_material_table_from_geometry,
)
from axicavity_fem.shared.mesh_loader import load_mesh_hom
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
def _pillbox_geom(eps_r: float = 1.0, material_tag: str = VACUUM_TAG):
    """半径 a=50mm, 長さ L=100mm の単一領域 pillbox。"""
    a, L = 50.0, 100.0
    pts = [(0.0, 0.0), (L, 0.0), (L, a), (0.0, a)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),  # 軸 r=0
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),      # z=L
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),      # r=a
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="PEC"),      # z=0
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Region", outer_loop_id=0,
                    material_tag=material_tag, eps_r=eps_r)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[loop],
                               regions=[region], unit="mm", mesh_size=15.0)


def _make_hom_mesh(tmp_path, n, element_order=1):
    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / f"hom_pillbox_o{element_order}.msh"
    export_msh_multi_region(geom, out, mesh_order=element_order, verbose=0)
    return load_mesh_hom(str(out), n, element_order)


def _standing_boundary_dofs(mesh):
    if mesh.element_order == 2:
        return get_pec_dof_indices_2nd(
            mesh.num_edges, len(mesh.simplices), len(mesh.vertices), mesh.n,
            mesh.boundary_edge_indices, mesh.boundary_vertices_indices)
    return get_pec_dof_indices_1st(
        mesh.num_edges, mesh.n,
        mesh.boundary_edge_indices, mesh.boundary_vertices_indices)


def _smallest_lambda_dense(mesh, eps_r_per_element):
    """Dirichlet 縮約後、dense 一般化固有値で最小の物理固有値を返す。"""
    K, M = assemble_global_matrices(mesh, eps_r_per_element=eps_r_per_element)
    boundary_dofs = _standing_boundary_dofs(mesh)
    T, _ = create_transformation_matrix(K.shape[0], boundary_dofs)
    Kr, Mr = apply_bc_transformation(K, M, T)
    eigvals = scipy.linalg.eigh(Kr.toarray(), Mr.toarray(), eigvals_only=True)
    pos = eigvals[eigvals > 1e-6]
    return float(np.min(pos))


# ---------------------------------------------------------------------------
# 1) 組立レベル: M に ε_r、K は不変
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("order", [1, 2])
def test_assembly_scales_M_not_K(tmp_path, n, order):
    mesh = _make_hom_mesh(tmp_path, n, order)
    E = len(mesh.simplices)
    eps = 4.0

    K0, M0 = assemble_global_matrices(mesh, eps_r_per_element=None)
    K4, M4 = assemble_global_matrices(
        mesh, eps_r_per_element=np.full(E, eps))

    # K は不変
    dK = (K4 - K0)
    assert np.max(np.abs(dK.toarray())) < 1e-12 * max(1.0, np.max(np.abs(K0.toarray())))
    # M は ε_r 倍
    dM = (M4 - eps * M0)
    assert np.max(np.abs(dM.toarray())) < 1e-12 * max(1.0, np.max(np.abs(M0.toarray())))


# ---------------------------------------------------------------------------
# 2) 一様 ε_r で最小固有値が 1/ε_r 倍
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1])
def test_uniform_eps_scales_eigenvalue(tmp_path, n):
    mesh = _make_hom_mesh(tmp_path, n, element_order=1)
    E = len(mesh.simplices)
    lam1 = _smallest_lambda_dense(mesh, np.ones(E))
    lam4 = _smallest_lambda_dense(mesh, np.full(E, 4.0))
    # λ = (ω/c)^2 ∝ 1/ε_r → 1/4 倍（周波数は 1/2 倍）
    assert lam1 / lam4 == pytest.approx(4.0, rel=1e-9), (
        f"n={n}: lam1={lam1}, lam4={lam4}, ratio={lam1/lam4}")


# ---------------------------------------------------------------------------
# 3) material_table=None と eps_r=1 一様 table が完全一致（ver1 互換）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1])
def test_none_matches_vacuum_table(tmp_path, n):
    mesh = _make_hom_mesh(tmp_path, n, element_order=1)
    res_none = solve_hom_standing(mesh, num_modes=3, material_table=None)
    res_vac = solve_hom_standing(
        mesh, num_modes=3,
        material_table={VACUUM_TAG: {"eps_r": 1.0, "mu_r": 1.0}})
    np.testing.assert_allclose(
        res_none.normal.frequencies, res_vac.normal.frequencies,
        rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 4) material_table 経由で誘電体が周波数を下げる（end-to-end, dense 照合と整合）
# ---------------------------------------------------------------------------
def test_material_table_lowers_frequency(tmp_path):
    """全領域 eps_r=4 の material_table で最小固有値が 1/4 になる（dense eigh 照合）。

    solver の eigsh は shift-invert sigma が ε_r 非追従で最低モードを見逃す
    ことがあるため、ここでは dense eigh で確認する（TM0 と同じ方針）。
    """
    n = 1
    geom_vac = _pillbox_geom(eps_r=1.0)
    geom_diel = _pillbox_geom(eps_r=4.0, material_tag="dielectric_1")
    tbl_diel = build_material_table_from_geometry(geom_diel)

    out_v = tmp_path / "v.msh"
    out_d = tmp_path / "d.msh"
    export_msh_multi_region(geom_vac, out_v, mesh_order=1, verbose=0)
    export_msh_multi_region(geom_diel, out_d, mesh_order=1, verbose=0)
    mesh_v = load_mesh_hom(str(out_v), n, 1)
    mesh_d = load_mesh_hom(str(out_d), n, 1)

    from axicavity_fem.shared.material_resolver import build_eps_r_per_element
    eps_v = build_eps_r_per_element(mesh_v, None)
    if eps_v is None:
        eps_v = np.ones(len(mesh_v.simplices))
    eps_d = build_eps_r_per_element(mesh_d, tbl_diel)
    assert eps_d is not None and np.allclose(eps_d, 4.0)

    lam_v = _smallest_lambda_dense(mesh_v, eps_v)
    lam_d = _smallest_lambda_dense(mesh_d, eps_d)
    assert lam_v / lam_d == pytest.approx(4.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 5) 蓄積エネルギー U の ε_r 重み付け（_stored_energy）
# ---------------------------------------------------------------------------
def test_stored_energy_scales_with_eps_r(tmp_path):
    """同じ固有ベクトルで、一様 ε_r=4 の U は ε_r=None（真空）の 4 倍になる。

    U = u_coeff·2π·∫ε₀·ε_r·|E|²·r dA なので、ε_r を一様 4 倍すると U も 4 倍。
    （eigsh の M 正規化と切り離すため、固定の固有ベクトルで直接 _stored_energy を比較）
    """
    from axicavity_fem.fem_hom.post_process import _stored_energy
    from axicavity_fem.fem_hom.field_recon import split_hom_dofs
    from axicavity_fem.shared.constants import EPS0

    n = 1
    mesh = _make_hom_mesh(tmp_path, n, element_order=1)
    res = solve_hom_standing(mesh, num_modes=1)
    evec = res.normal.eigenvectors[0]
    num_edges = mesh.num_edges
    num_elements = len(mesh.simplices)
    num_nodes = len(mesh.vertices)
    ev, ev_lt, face, E_theta = split_hom_dofs(
        evec, num_edges, num_elements, num_nodes, 1, n,
        mesh.vertices, mesh.simplices)

    args = (mesh.simplices, mesh.vertices, mesh.edge_index_map, 1,
            ev, ev_lt, face, E_theta, 1.0, EPS0)
    U_vac = _stored_energy(*args, eps_r_per_element=None)
    U_eps = _stored_energy(*args,
                           eps_r_per_element=np.full(num_elements, 4.0))
    assert U_eps == pytest.approx(4.0 * U_vac, rel=1e-12)
    # None と eps_r=1 一様が一致
    U_one = _stored_energy(*args, eps_r_per_element=np.ones(num_elements))
    assert U_one == pytest.approx(U_vac, rel=1e-12)
