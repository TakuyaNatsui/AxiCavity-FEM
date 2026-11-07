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
from scipy.special import jn_zeros, jnp_zeros

from axicavity_fem.fem_hom.assembly import assemble_global_matrices
from axicavity_fem.fem_hom.boundary import (
    apply_bc_transformation,
    create_transformation_matrix,
    get_pec_dof_indices_1st,
    get_pec_dof_indices_2nd,
)
from axicavity_fem.fem_hom.post_process import compute_hom_parameters
from axicavity_fem.fem_hom.solver import solve_hom_standing, solve_hom_traveling
from axicavity_fem.shared.constants import C0, EPS0, MU0
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.material_resolver import (
    build_eps_r_per_element,
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


# ---------------------------------------------------------------------------
# 6) 誘電体を入れた導波管セル: 後処理は ε_r を含めて正しい（solve の警告の文の根拠）
#    媒質中の k_d = √ε_r·ω/c, η_d = η/√ε_r として真空の式と同じ:
#      k_d² = k_c² + β², v_g/c = β/(√ε_r·k_d), Q = ω/(2αv_g)
#      TM: α = R_s·k_d/(R·η_d·β), TE: α = R_s/(R·k_d·η_d·β)·(k_c² + k_d²n²/(x² − n²))
# ---------------------------------------------------------------------------
_L_MM, _R_MM, _SIGMA, _THETA = 35.0, 40.0, 5.8e7, 120.0
_EPS_R = 4.0


def _cell_geom():
    pts = [(0.0, 0.0), (_L_MM, 0.0), (_L_MM, _R_MM), (0.0, _R_MM)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="None")]
    region = Region(id=0, name="Cell", outer_loop_id=0, material_tag="diel", eps_r=_EPS_R)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[region], unit="mm", mesh_size=2.0)


def _layered_cell():
    """r < 20 mm は真空、r > 20 mm は誘電体（ポート断面に誘電体があり、軸は真空）."""
    mid = _R_MM / 2.0
    pts = [(0.0, 0.0), (_L_MM, 0.0), (_L_MM, mid), (0.0, mid), (_L_MM, _R_MM), (0.0, _R_MM)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="None"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="None"),
            Segment(id=4, type="line", point_indices=[2, 4], bc_name="None"),
            Segment(id=5, type="line", point_indices=[4, 5], bc_name="PEC"),
            Segment(id=6, type="line", point_indices=[5, 3], bc_name="None")]
    regions = [Region(id=0, name="Core", outer_loop_id=0, material_tag=VACUUM_TAG),
               Region(id=1, name="Layer", outer_loop_id=1, material_tag="diel", eps_r=_EPS_R)]
    return MultiRegionGeometry(points=pts, segments=segs,
                               loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 2])],
                               regions=regions, unit="mm", mesh_size=2.0)


def _solve_traveling(tmp_path, geom, n, phases, **kw):
    path = tmp_path / f"cell_{n}.msh"
    export_msh_multi_region(geom, path, mesh_order=2, verbose=0)
    mesh = load_mesh_hom(str(path), n, 2)
    table = build_material_table_from_geometry(geom)
    with pytest.warns(UserWarning, match="シフト値"):
        res = solve_hom_traveling(mesh, num_modes=4, phase_shifts=phases, material_table=table, **kw)
    return res, compute_hom_parameters(res, conductivity=_SIGMA, eps_r_per_element=build_eps_r_per_element(mesh, table))


@pytest.mark.parametrize("kind, n", [("TM", 0), ("TM", 1), ("TE", 1), ("TE", 2)], ids=["TM01", "TM11", "TE11", "TE21"])
def test_filled_waveguide_cell_matches_analytic(tmp_path, kind, n):
    R, L = _R_MM * 1e-3, _L_MM * 1e-3
    x = float(jn_zeros(n, 1)[0] if kind == "TM" else jnp_zeros(n, 1)[0])
    kc, beta = x / R, np.radians(_THETA) / L
    k_d = np.hypot(kc, beta)
    sqrt_eps = np.sqrt(_EPS_R)
    omega = k_d * C0 / sqrt_eps
    f_ghz = omega / (2e9 * np.pi)
    rs = np.sqrt(omega * MU0 / (2.0 * _SIGMA))
    eta_d = np.sqrt(MU0 / EPS0) / sqrt_eps
    if kind == "TM":
        alpha = rs * k_d / (R * eta_d * beta)
    else:
        alpha = rs / (R * k_d * eta_d * beta) * (kc ** 2 + k_d ** 2 * n ** 2 / (x ** 2 - n ** 2))
    vg_c = beta / (sqrt_eps * k_d)

    res, params = _solve_traveling(tmp_path, _cell_geom(), n, (_THETA,), target_freq_ghz=f_ghz * 1.01)
    freqs = res.periodic[_THETA].frequencies
    mode = params.per_phase[_THETA][int(np.argmin(np.abs(freqs - f_ghz)))]
    assert mode.frequency_ghz == pytest.approx(f_ghz, rel=1e-3)
    assert mode.v_group_c == pytest.approx(vg_c, rel=0.005, abs=0)
    assert mode.attenuation == pytest.approx(alpha, rel=0.005, abs=0)
    assert mode.q_factor == pytest.approx(omega / (2.0 * alpha * vg_c * C0), rel=0.005, abs=0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_group_velocity_with_dielectric_at_the_port(tmp_path, n):
    """ポート断面に誘電体: 解析解は無いので v_g = L dω/dθ（固有周波数だけ）と比べる."""
    res, params = _solve_traveling(tmp_path, _layered_cell(), n, (_THETA - 1.0, _THETA, _THETA + 1.0))
    low, high = res.periodic[_THETA - 1.0].frequencies, res.periodic[_THETA + 1.0].frequencies
    center = res.periodic[_THETA].frequencies
    # 番号で差分を取るので、θ = 120° の近くで交差するモードの組は除く（n = 2 に 1 組ある）
    gaps = np.abs(center[:, None] - center[None, :]) + np.eye(len(center))
    isolated = gaps.min(axis=1) > 3.0 * np.abs(high - low).max()
    assert isolated.sum() >= 2
    fd = (high - low) * 2e9 * np.pi * (_L_MM * 1e-3) / np.radians(2.0) / C0
    vg = np.array([mode.v_group_c for mode in params.per_phase[_THETA]])
    assert vg[isolated] == pytest.approx(fd[isolated], abs=2e-3)


def test_cli_post_matches_the_api_with_dielectric(tmp_path):
    """CLI の post も ε_r を渡す: h5 の U が API（解析解で確かめた）と同じで、ε_r なしとは違う."""
    from axicavity_fem.cli.main import cli_main
    from axicavity_fem.gui.jobs.pipeline import write_materials_json
    from axicavity_fem.shared.hdf5_io import read_results

    geom = _pillbox_geom(eps_r=4.0, material_tag="diel")
    msh, raw, proc = tmp_path / "m.msh", tmp_path / "raw.h5", tmp_path / "proc.h5"
    export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
    write_materials_json(geom, msh)
    assert cli_main(["solve", "--type", "hom", "-m", str(msh), "--elem-order", "2", "--num-modes", "1", "--az-order", "1",
                     "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "hom", "-i", str(raw), "-o", str(proc)]) == 0
    stored = read_results(proc)["post_process"][1]["standing"][0]["U_stored"]

    mesh = load_mesh_hom(str(msh), 1, 2)
    table = build_material_table_from_geometry(geom)
    res = solve_hom_standing(mesh, num_modes=1, material_table=table)
    with_eps = compute_hom_parameters(res, eps_r_per_element=build_eps_r_per_element(mesh, table)).per_phase[0.0][0]
    no_eps = compute_hom_parameters(res).per_phase[0.0][0]
    assert stored == pytest.approx(with_eps.stored_energy, rel=1e-6, abs=0)
    assert abs(with_eps.stored_energy / no_eps.stored_energy - 1.0) > 0.5
