"""TM0 誘電体対応の物理検証テスト。

1. eps_r 一様 = 1 のとき、ver2 と同じ結果（K, M, 固有値）を返す。
2. eps_r 一様 ≠ 1 のとき、固有値 λ = k² が 1/ε_r 倍 (周波数 f は 1/√ε_r 倍)
   になる（K → K/ε_r、M 不変 → λ → λ/ε_r）。
3. 多領域 (vacuum + dielectric) で sanity check: 両者の中間の固有値になる。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import quad, trapezoid
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros, y0, y1

from axicavity_fem.fem_tm0.assembly import assemble_global_matrices
from axicavity_fem.fem_tm0.post_process import compute_tm0_parameters
from axicavity_fem.fem_tm0.solver import solve_tm0_standing, solve_tm0_traveling
from axicavity_fem.shared.boundary_groups import classify_boundaries
from axicavity_fem.shared.constants import C0, EPS0, MU0
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
    VACUUM_TAG,
)


# ---------------------------------------------------------------------------
# ジオメトリ
# ---------------------------------------------------------------------------
def _pillbox_geom(eps_r: float = 1.0, material_tag: str = VACUUM_TAG) -> MultiRegionGeometry:
    """半径 a=50mm, 長さ L=100mm の単一領域 pillbox 空洞 (r=0 軸対称)。

    BC:
        z=0  → E-short (TM 対称面、E_z 通過)
        z=L  → E-short (同上)
        r=a  → PEC
        r=0  → 軸（軸対称コード側で自動処理）
    """
    a = 50.0   # mm
    L = 100.0  # mm
    pts = [
        (0.0, 0.0), (L, 0.0), (L, a), (0.0, a),
    ]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),  # 軸 r=0 (自動処理)
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),  # z=L
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),      # r=a
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),  # z=0
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Region", outer_loop_id=0,
                    material_tag=material_tag, eps_r=eps_r)
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=[loop], regions=[region],
        unit="mm", mesh_size=10.0,
    )


def _two_region_pillbox(eps_r_left: float, eps_r_right: float) -> MultiRegionGeometry:
    """同じ pillbox を z=L/2 で 2 領域に分け、それぞれ eps_r を指定する。"""
    a = 50.0
    L = 100.0
    Lh = L / 2.0
    pts = [
        (0.0, 0.0), (Lh, 0.0), (Lh, a), (0.0, a),
        (L, 0.0), (L, a),
    ]
    segs = [
        # 左領域
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),  # 共有境界 (内部)
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        # 右領域
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="E-short"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
        Segment(id=7, type="line", point_indices=[2, 1], bc_name="PEC"),  # 共有境界 (内部)
    ]
    # 注: 内部の共有境界 segment 1, 7 は fragment で共通化されるが、BC 名は PEC
    # としていても M-short と同等の自然境界として扱われる (PEC 重複指定は無害)。
    # ただし、ここでは物理的に「2 領域の界面は内部の連続面で、本来 PEC でない」
    # ので bc_name を空 (or None) にしたい所だが、データモデル上 BC 必須。
    # 計算結果に影響しないことは sanity 確認の通り。
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3]),
        Loop(id=1, segment_ids=[4, 5, 6, 7]),
    ]
    regions = [
        Region(id=0, name="Left", outer_loop_id=0,
               material_tag="left_mat", eps_r=eps_r_left),
        Region(id=1, name="Right", outer_loop_id=1,
               material_tag="right_mat", eps_r=eps_r_right),
    ]
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=regions,
        unit="mm", mesh_size=10.0,
    )


# ---------------------------------------------------------------------------
# Test 1: 後方互換 (eps_r=None → ver2 と同じ K, M)
# ---------------------------------------------------------------------------
def test_assembly_eps_r_none_matches_legacy(tmp_path):
    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / "pill.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    K_legacy, M_legacy = assemble_global_matrices(mesh)
    K_ones, M_ones = assemble_global_matrices(
        mesh, np.ones(mesh.num_elements),
    )
    # eps_r=1 一様は eps_r=None と完全一致
    assert (K_legacy - K_ones).count_nonzero() == 0 or \
        np.allclose(K_legacy.toarray(), K_ones.toarray(), atol=1e-15)
    assert (M_legacy - M_ones).count_nonzero() == 0 or \
        np.allclose(M_legacy.toarray(), M_ones.toarray(), atol=1e-15)


# ---------------------------------------------------------------------------
# Test 2: 一様 eps_r ≠ 1 で K → K/ε_r、固有値 → λ/ε_r
# ---------------------------------------------------------------------------
def test_assembly_eps_r_uniform_scales_K(tmp_path):
    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / "pill.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    K_base, M_base = assemble_global_matrices(mesh)
    eps_r = np.full(mesh.num_elements, 4.0)
    K_eps, M_eps = assemble_global_matrices(mesh, eps_r)

    # K_eps == K_base / 4
    diff = (K_eps - K_base / 4.0)
    assert np.max(np.abs(diff.toarray())) < 1e-15
    # M_eps == M_base (eps_r 非依存)
    assert (M_eps - M_base).count_nonzero() == 0


def _solve_smallest_dense(K_global, M_global, dirichlet_nodes, n_total):
    """疎行列 (K, M) を dense 化し、Dirichlet 縮約後の最小固有値を返す。

    solver.py の shift-invert sigma が ε_r に依存するため、ε_r をまたぐ厳密比較
    では信頼できない。ここでは dense 一般化固有値で最小値だけを取る。
    """
    free = np.setdiff1d(np.arange(n_total), np.asarray(dirichlet_nodes, dtype=int))
    Kd = K_global.toarray()[np.ix_(free, free)]
    Md = M_global.toarray()[np.ix_(free, free)]
    # 一般化固有値 K u = λ M u を解く
    from scipy.linalg import eigh
    eigvals = eigh(Kd, Md, eigvals_only=True)
    # 正の最小固有値
    pos = eigvals[eigvals > 1e-6]
    return float(np.min(pos))


def test_solve_uniform_eps_r_scales_frequency(tmp_path):
    """全領域 eps_r=ε にすると、最小固有値 λ_min は 1/ε 倍になる
    (周波数 f は 1/√ε 倍)。

    Note: ソルバ全体 (solve_tm0_standing) は shift-invert sigma が rmax のみで
    決まり ε_r に追従しないため、ε_r=4 で最低モードを見逃す。ここでは
    dense eigh で素直に最小固有値だけを取り、スケール則を確認する。
    """
    from axicavity_fem.shared.boundary_groups import classify_boundaries
    from axicavity_fem.fem_tm0.boundary import build_dirichlet_nodes

    out = tmp_path / "pill.msh"
    geom = _pillbox_geom(eps_r=1.0)
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)

    K1, M1 = assemble_global_matrices(mesh, np.ones(mesh.num_elements))
    K4, M4 = assemble_global_matrices(mesh, np.full(mesh.num_elements, 4.0))

    lam1 = _solve_smallest_dense(K1, M1, dirichlet_nodes, mesh.num_nodes)
    lam4 = _solve_smallest_dense(K4, M4, dirichlet_nodes, mesh.num_nodes)

    # λ_4 = λ_1 / 4 (K だけが 1/4 倍されるため、M 不変なら λ も 1/4 倍)
    assert lam1 / lam4 == pytest.approx(4.0, rel=1e-10), (
        f"lam1={lam1}, lam4={lam4}, ratio={lam1/lam4}"
    )


def test_legacy_path_matches_eps1_path(tmp_path):
    """material_table=None と eps_r=1 一様の material_table を渡した場合で、
    固有値が完全一致する。"""
    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / "pill.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    res_none = solve_tm0_standing(mesh, num_modes=3, material_table=None)
    res_eps1 = solve_tm0_standing(
        mesh, num_modes=3,
        material_table={VACUUM_TAG: {"eps_r": 1.0, "mu_r": 1.0}},
    )
    np.testing.assert_allclose(res_none.frequencies, res_eps1.frequencies,
                               rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Test 3: 多領域でも一様な eps_r を入れたら 1/√ε 倍になる
# ---------------------------------------------------------------------------
def test_two_region_uniform_eps_matches_single_region(tmp_path):
    """2 領域 (両方 eps_r=4) と 1 領域 (eps_r=4) は同じ周波数を与えるべき。"""
    out1 = tmp_path / "single.msh"
    out2 = tmp_path / "two.msh"

    g_single = _pillbox_geom(eps_r=4.0)
    export_msh_multi_region(g_single, out1, mesh_order=1, verbose=0)
    mesh_single = load_mesh(out1, element_order=1)
    tbl_single = build_material_table_from_geometry(g_single)
    res_single = solve_tm0_standing(mesh_single, num_modes=3,
                                    material_table=tbl_single)

    g_two = _two_region_pillbox(4.0, 4.0)
    export_msh_multi_region(g_two, out2, mesh_order=1, verbose=0)
    mesh_two = load_mesh(out2, element_order=1)
    tbl_two = build_material_table_from_geometry(g_two)
    res_two = solve_tm0_standing(mesh_two, num_modes=3,
                                 material_table=tbl_two)

    # メッシュが違うので完全一致は望めないが、相対誤差 < 1% を期待
    np.testing.assert_allclose(res_single.frequencies, res_two.frequencies,
                               rtol=2e-2)


# ---------------------------------------------------------------------------
# Test 4: 不均一誘電体ロード空洞 (sanity check)
# ---------------------------------------------------------------------------
def test_partial_dielectric_frequency_in_between(tmp_path):
    """左 vacuum + 右 eps_r=4 の組み合わせの最低固有値は、
    完全真空と完全 eps_r=4 の間にあるべき。

    （shift-invert の影響を避けるため dense eigh で最小固有値を比較）"""
    from axicavity_fem.shared.boundary_groups import classify_boundaries
    from axicavity_fem.fem_tm0.boundary import build_dirichlet_nodes

    def lam_min(geom: MultiRegionGeometry, out_path) -> float:
        export_msh_multi_region(geom, out_path, mesh_order=1, verbose=0)
        mesh = load_mesh(out_path, element_order=1)
        tbl = build_material_table_from_geometry(geom)
        eps_r = build_eps_r_per_element(mesh, tbl)
        K, M = assemble_global_matrices(mesh, eps_r)
        classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
        dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)
        return _solve_smallest_dense(K, M, dirichlet_nodes, mesh.num_nodes)

    lam_vac = lam_min(_two_region_pillbox(1.0, 1.0), tmp_path / "vac.msh")
    lam_eps = lam_min(_two_region_pillbox(4.0, 4.0), tmp_path / "eps.msh")
    lam_mix = lam_min(_two_region_pillbox(1.0, 4.0), tmp_path / "mix.msh")

    # λ ∝ k² なので λ_eps = λ_vac / 4
    assert lam_eps < lam_mix < lam_vac, (
        f"lam_eps={lam_eps}, lam_mix={lam_mix}, lam_vac={lam_vac}"
    )
    # 同じメッシュ・形状なので λ_vac / λ_eps = 4
    assert lam_vac / lam_eps == pytest.approx(4.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Test 5: H→E 換算の ε_r 補正（field_recon）
# ---------------------------------------------------------------------------
def test_field_recon_divides_E_by_eps_r(tmp_path):
    """同じ H_phi 固有ベクトルから、誘電体要素内の E が ε_r で割られる。

    eps_r_per_element=None（真空）と eps_r=4 一様で calculate_fields を比較すると、
    全評価点で E が 1/4 になる。
    """
    from axicavity_fem.fem_tm0.field_recon import TM0FieldReconstructor

    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / "pill.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    # 適当な滑らかな H_phi 場（軸対称・r 依存）を固有ベクトルとして使う
    rng = np.random.default_rng(0)
    eigvec = rng.standard_normal(mesh.num_nodes)
    freq = 1.5  # GHz（係数の絶対値はテストの比に影響しない）

    recon_vac = TM0FieldReconstructor(
        mesh.nodes, mesh.elements, 1, "standing", eps_r_per_element=None)
    recon_eps = TM0FieldReconstructor(
        mesh.nodes, mesh.elements, 1, "standing",
        eps_r_per_element=np.full(mesh.num_elements, 4.0))

    # メッシュ内部の複数点で比較（メッシュの座標は m）
    pts = [(0.025, 0.010), (0.050, 0.025), (0.075, 0.030), (0.040, 0.005), (0.050, 0.0)]
    for z, r in pts:
        fv = recon_vac.calculate_fields(eigvec, freq, z, r, return_complex=True)
        fe = recon_eps.calculate_fields(eigvec, freq, z, r, return_complex=True)
        assert fv is not None and fe is not None, (z, r)
        # E は 1/4、H_theta は不変
        assert abs(fv["Ez"]) > 0.0
        assert fe["Ez"] == pytest.approx(fv["Ez"] / 4.0, rel=1e-12)
        assert fe["Er"] == pytest.approx(fv["Er"] / 4.0, rel=1e-12)
        assert fe["H_theta"] == pytest.approx(fv["H_theta"], rel=1e-12)

    # 場の図・3D・レポートの図が使う節点の場も同じ
    node_vac = recon_vac.calculate_all_node_fields(eigvec, freq)
    node_eps = recon_eps.calculate_all_node_fields(eigvec, freq)
    assert np.max(np.abs(node_vac["Ez"])) > 0.0
    assert node_eps["Ez"] == pytest.approx(node_vac["Ez"] / 4.0, rel=1e-12, abs=0)
    assert node_eps["Er"] == pytest.approx(node_vac["Er"] / 4.0, rel=1e-12, abs=0)


def test_field_recon_none_matches_eps1(tmp_path):
    """eps_r_per_element=None と eps_r=1 一様で calculate_fields が完全一致。"""
    from axicavity_fem.fem_tm0.field_recon import TM0FieldReconstructor

    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / "pill.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)
    eigvec = np.random.default_rng(1).standard_normal(mesh.num_nodes)

    r_none = TM0FieldReconstructor(mesh.nodes, mesh.elements, 1, "standing")
    r_one = TM0FieldReconstructor(
        mesh.nodes, mesh.elements, 1, "standing",
        eps_r_per_element=np.ones(mesh.num_elements))
    node_none = r_none.calculate_all_node_fields(eigvec, 1.5)
    node_one = r_one.calculate_all_node_fields(eigvec, 1.5)
    np.testing.assert_allclose(node_none["Ez"], node_one["Ez"], rtol=1e-12)
    np.testing.assert_allclose(node_none["Er"], node_one["Er"], rtol=1e-12)


def test_cli_solve_tm0_stores_eps_r(tmp_path):
    """CLI solve --type tm0 --materials で H5 に eps_r_per_element が保存される。"""
    import json
    from argparse import Namespace
    from axicavity_fem.cli import cmd_solve
    from axicavity_fem.shared.hdf5_io import read_results

    geom = _pillbox_geom(eps_r=4.0, material_tag="dielectric_1")
    msh = tmp_path / "diel.msh"
    export_msh_multi_region(geom, msh, mesh_order=1, verbose=0)
    side = tmp_path / "diel.materials.json"
    side.write_text(json.dumps({
        "materials": build_material_table_from_geometry(geom)}), encoding="utf-8")

    out = tmp_path / "diel.h5"
    args = Namespace(mesh_file=str(msh), elem_order=1, num_modes=3, phase="0.0",
                     output_file=str(out), type="tm0", az_order=[0],
                     materials_file=None)
    assert cmd_solve.run(args) == 0
    data = read_results(out)
    assert data["schema_version"] == "2.2"
    eps = data["mesh"]["eps_r_per_element"]
    assert eps is not None and np.allclose(eps, 4.0)
    assert "dielectric_1" in data["materials"]


# ---------------------------------------------------------------------------
# Test 6: P_flow の ε_r 補正（断面エッジを 1/ε_r で割る）
# ---------------------------------------------------------------------------
def test_p_flow_scales_with_inv_eps_r(tmp_path):
    """同じ進行波固有ベクトルで、一様 ε_r=4 の P_flow は真空の 1/4 になる。

    P_flow = -π/(ωε₀ε_r)∫Im[(∂ψ/∂z)ψ*]r dr。E_r ∝ 1/ε_r なので断面が一様 ε_r=4
    なら P_flow は 1/4。（eigsh の正規化と切り離すため固定固有ベクトルで比較）
    """
    from axicavity_fem.fem_tm0.post_process import tm0_p_flow_zmin

    geom = _pillbox_geom(eps_r=1.0)
    out = tmp_path / "pill.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    mesh = load_mesh(out, element_order=1)

    rng = np.random.default_rng(3)
    # 複素固有ベクトル（進行波想定: Im 成分が必要）
    psi = (rng.standard_normal(mesh.num_nodes)
           + 1j * rng.standard_normal(mesh.num_nodes))
    omega = 2 * np.pi * 2.0e9

    p_vac = tm0_p_flow_zmin(psi, mesh.nodes, mesh.elements, 1, omega,
                            eps_r_per_element=None)
    p_eps = tm0_p_flow_zmin(psi, mesh.nodes, mesh.elements, 1, omega,
                            eps_r_per_element=np.full(mesh.num_elements, 4.0))
    assert p_eps == pytest.approx(p_vac / 4.0, rel=1e-12)
    # None と eps_r=1 一様が一致
    p_one = tm0_p_flow_zmin(psi, mesh.nodes, mesh.elements, 1, omega,
                            eps_r_per_element=np.ones(mesh.num_elements))
    assert p_one == pytest.approx(p_vac, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 7: 誘電体の後処理と解析解（軸に誘電体があるときの E_z も ε_r で割る。3.0.0 までは真空の式だった）
#   媒質中の k_d = √ε_r·ω/c, η_d = η/√ε_r として TM01 の導波管の式: k_d² = k_c² + β²,
#   v_g/c = β/(√ε_r·k_d), α = R_s·k_d/(R·η_d·β), Q = ω/(2αv_g)
#   max|E_z| = 1 の U = ½ε₀ε_r·πLR²·J1(x)²·k_d²/k_c²、V_acc = |∫e^{-jβz}dz|、V_eff = |∫e^{-jβz}e^{jωz/c}dz|（β_particle = 1）
# ---------------------------------------------------------------------------
_L_MM, _R_MM, _SIGMA, _THETA = 35.0, 40.0, 5.8e7, 120.0


def _uniform_cell(eps_r):
    pts = [(0.0, 0.0), (_L_MM, 0.0), (_L_MM, _R_MM), (0.0, _R_MM)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="None")]
    region = Region(id=0, name="Cell", outer_loop_id=0, material_tag="diel", eps_r=eps_r)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[region], unit="mm", mesh_size=2.0)


def _layered_cell(eps_r):
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
               Region(id=1, name="Layer", outer_loop_id=1, material_tag="diel", eps_r=eps_r)]
    return MultiRegionGeometry(points=pts, segments=segs,
                               loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 2])],
                               regions=regions, unit="mm", mesh_size=2.0)


def _traveling_tm0(tmp_path, geom, phases, **kw):
    path = tmp_path / "cell.msh"
    export_msh_multi_region(geom, path, mesh_order=2, verbose=0)
    mesh = load_mesh(path, element_order=2)
    table = build_material_table_from_geometry(geom)
    res = solve_tm0_traveling(mesh, num_modes=3, phase_shifts=phases, material_table=table, **kw)
    return res, compute_tm0_parameters(res, classify_boundaries(mesh.physical_groups, mesh.nodes), conductivity=_SIGMA,
                                       eps_r_per_element=build_eps_r_per_element(mesh, table))


@pytest.mark.parametrize("eps_r", [1.0, 4.0])
def test_filled_waveguide_cell_matches_analytic(tmp_path, eps_r):
    R, L = _R_MM * 1e-3, _L_MM * 1e-3
    x = float(jn_zeros(0, 1)[0])
    kc, beta = x / R, np.radians(_THETA) / L
    k_d = np.hypot(kc, beta)
    sqrt_eps = np.sqrt(eps_r)
    omega = k_d * C0 / sqrt_eps
    f_ghz = omega / (2e9 * np.pi)
    rs = np.sqrt(omega * MU0 / (2.0 * _SIGMA))
    alpha = rs * k_d / (R * (np.sqrt(MU0 / EPS0) / sqrt_eps) * beta)
    vg_c = beta / (sqrt_eps * k_d)
    u = 0.5 * EPS0 * eps_r * np.pi * L * R ** 2 * j1(x) ** 2 * (k_d / kc) ** 2
    v_acc = 2.0 * abs(np.sin(beta * L / 2.0)) / beta
    dk = omega / C0 - beta
    v_eff = 2.0 * abs(np.sin(dk * L / 2.0)) / abs(dk)

    res, params = _traveling_tm0(tmp_path, _uniform_cell(eps_r), (_THETA,), target_freq_ghz=f_ghz * 1.01)
    freqs = res.phase_results[_THETA]["frequencies"]
    mode = params.per_phase[_THETA][int(np.argmin(np.abs(freqs - f_ghz)))]
    assert mode.frequency_ghz == pytest.approx(f_ghz, rel=1e-3)
    assert mode.v_group_c == pytest.approx(vg_c, rel=0.005, abs=0)
    assert mode.attenuation == pytest.approx(alpha, rel=0.005, abs=0)
    assert mode.q_factor == pytest.approx(omega / (2.0 * alpha * vg_c * C0), rel=0.005, abs=0)
    # 3.0.0 まで U は ε_r = 4 で 1/16、R/Q は 16 倍。U の許容は軸上 E_z の離散化誤差の分（実測 0.55%。ε_r = 1 と 4 で同じ）
    assert mode.stored_energy == pytest.approx(u, rel=0.01, abs=0)
    assert mode.rq_apparent == pytest.approx(v_acc ** 2 / (omega * u), rel=0.01, abs=0)
    assert mode.rq_eff == pytest.approx(v_eff ** 2 / (omega * u), rel=0.01, abs=0)
    assert mode.p_flow == pytest.approx(vg_c * C0 * u / L, rel=0.015, abs=0)     # 規格化がそろっている
    assert mode.p_loss == pytest.approx(2.0 * alpha * L * mode.p_flow, rel=0.01, abs=0)


# 軸を横切る誘電体: 境界で E_z が跳ぶ（D_z が連続）ピルボックス。半径 a、長さ L、端面は E-short（E_r = 0）
_LAYER_A, _LAYER_L, _LAYER_D, _LAYER_EPS = 50e-3, 100e-3, 40e-3, 4.0


def _slab_pillbox():
    """z < d が誘電体、z > d が真空の 2 層."""
    a, length, d = _LAYER_A * 1e3, _LAYER_L * 1e3, _LAYER_D * 1e3
    pts = [(0.0, 0.0), (d, 0.0), (d, a), (0.0, a), (length, 0.0), (length, a)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
            Segment(id=4, type="line", point_indices=[1, 4], bc_name="None"),
            Segment(id=5, type="line", point_indices=[4, 5], bc_name="E-short"),
            Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC")]
    regions = [Region(id=0, name="Slab", outer_loop_id=0, material_tag="diel", eps_r=_LAYER_EPS),
               Region(id=1, name="Vac", outer_loop_id=1, material_tag=VACUUM_TAG)]
    return MultiRegionGeometry(points=pts, segments=segs,
                               loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 1])],
                               regions=regions, unit="mm", mesh_size=2.0)


def _slab_ref():
    """H_φ = J1(k_c r)Z(z), Z'' + (ε k₀² − k_c²)Z = 0, Z'(0) = Z'(L) = 0, 界面で Z と Z'/ε が連続. 最低次のモード.

    E_z(0, z) = k_c Z/(ωε₀ε)（大きさ）、U = ½μ₀·2π·(a²/2)J1(x)²·∫Z²dz.
    """
    x = float(jn_zeros(0, 1)[0])
    kc = x / _LAYER_A
    d, rest = _LAYER_D, _LAYER_L - _LAYER_D

    def q_of(k0, eps):
        return np.sqrt(complex(eps * k0 ** 2 - kc ** 2))

    def equation(k0):
        q1, q2 = q_of(k0, _LAYER_EPS), q_of(k0, 1.0)
        return ((q1 / _LAYER_EPS) * np.sin(q1 * d) * np.cos(q2 * rest) + q2 * np.sin(q2 * rest) * np.cos(q1 * d)).real

    grid = np.linspace(0.6 * kc / np.sqrt(_LAYER_EPS), 1.5 * kc, 4000)
    values = [equation(k) for k in grid]
    i = next(i for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0)
    k0 = brentq(equation, grid[i], grid[i + 1], xtol=1e-12)
    omega = k0 * C0
    q1, q2 = q_of(k0, _LAYER_EPS), q_of(k0, 1.0)
    z1, z2 = np.linspace(0.0, d, 20001), np.linspace(d, _LAYER_L, 20001)
    b = np.cos(q1 * d) / np.cos(q2 * rest)
    h1, h2 = np.cos(q1 * z1).real, (b * np.cos(q2 * (_LAYER_L - z2))).real
    e1, e2 = kc * h1 / (omega * EPS0 * _LAYER_EPS), kc * h2 / (omega * EPS0)
    scale = 1.0 / max(np.max(np.abs(e1)), np.max(np.abs(e2)))
    u = 0.5 * MU0 * 2.0 * np.pi * (_LAYER_A ** 2 / 2.0) * j1(x) ** 2 * (trapezoid(h1 ** 2, z1) + trapezoid(h2 ** 2, z2)) * scale ** 2
    v_acc = abs(trapezoid(e1, z1) + trapezoid(e2, z2)) * scale
    transit = (trapezoid(e1 * np.exp(1j * k0 * z1), z1) + trapezoid(e2 * np.exp(1j * k0 * z2), z2))
    return dict(f_ghz=omega / (2e9 * np.pi), omega=omega, u=u, v_acc=v_acc, v_eff=abs(transit) * scale,
                jump=np.max(np.abs(e1)) * scale)


def test_slab_across_the_axis_matches_analytic(tmp_path):
    """軸を横切る誘電体の板: E_z は界面で 1/ε_r に跳ぶ. U・V・R/Q を解析解と比べる."""
    ref = _slab_ref()
    res, mode = _standing_tm0(tmp_path, _slab_pillbox(), ref["f_ghz"])
    assert mode.frequency_ghz == pytest.approx(ref["f_ghz"], rel=1e-4)
    # 実測の誤差（h = 2 mm）: U −0.21%、V −0.05%、R/Q +0.11%。h² で減る
    assert mode.stored_energy == pytest.approx(ref["u"], rel=0.005, abs=0)
    assert mode.v_acc == pytest.approx(ref["v_acc"], rel=0.002, abs=0)
    assert mode.v_eff == pytest.approx(ref["v_eff"], rel=0.002, abs=0)
    assert mode.rq_apparent == pytest.approx(ref["v_acc"] ** 2 / (ref["omega"] * ref["u"]), rel=0.005, abs=0)
    assert mode.rq_eff == pytest.approx(ref["v_eff"] ** 2 / (ref["omega"] * ref["u"]), rel=0.005, abs=0)


# 軸上の誘電体棒: E_z は界面で連続。半径 b の棒（ε_r）、半径 a の空洞、長さ L、端面 E-short（E_z は z によらない）
_ROD_B, _ROD_A, _ROD_L, _ROD_EPS = 15e-3, 50e-3, 60e-3, 4.0


def _rod_pillbox():
    b, a, length = _ROD_B * 1e3, _ROD_A * 1e3, _ROD_L * 1e3
    pts = [(0.0, 0.0), (length, 0.0), (length, b), (0.0, b), (length, a), (0.0, a)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="None"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
            Segment(id=4, type="line", point_indices=[2, 4], bc_name="E-short"),
            Segment(id=5, type="line", point_indices=[4, 5], bc_name="PEC"),
            Segment(id=6, type="line", point_indices=[5, 3], bc_name="E-short")]
    regions = [Region(id=0, name="Rod", outer_loop_id=0, material_tag="diel", eps_r=_ROD_EPS),
               Region(id=1, name="Vac", outer_loop_id=1, material_tag=VACUUM_TAG)]
    return MultiRegionGeometry(points=pts, segments=segs,
                               loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 2])],
                               regions=regions, unit="mm", mesh_size=2.0)


def _rod_ref():
    """E_z = J0(k₁r) (r < b)、C[J0(k₂r)Y0(k₂a) − Y0(k₂r)J0(k₂a)] (r > b). E_z と ∂E_z/∂r が界面で連続. 最低次のモード."""
    b, a = _ROD_B, _ROD_A

    def outer(r, k2):
        return j0(k2 * r) * y0(k2 * a) - y0(k2 * r) * j0(k2 * a)

    def outer_dr(r, k2):
        return -k2 * (j1(k2 * r) * y0(k2 * a) - y1(k2 * r) * j0(k2 * a))

    def equation(k0):
        k1, k2 = np.sqrt(_ROD_EPS) * k0, k0
        return j0(k1 * b) * outer_dr(b, k2) + k1 * j1(k1 * b) * outer(b, k2)

    grid = np.linspace(5.0, 60.0, 4000)
    values = [equation(k) for k in grid]
    i = next(i for i in range(len(grid) - 1) if values[i] * values[i + 1] < 0)
    k0 = brentq(equation, grid[i], grid[i + 1], xtol=1e-12)
    k1, k2 = np.sqrt(_ROD_EPS) * k0, k0
    c = j0(k1 * b) / outer(b, k2)
    omega = k0 * C0
    inner_int = quad(lambda r: j0(k1 * r) ** 2 * r, 0.0, b)[0]
    outer_int = quad(lambda r: (c * outer(r, k2)) ** 2 * r, b, a)[0]
    u = 0.5 * EPS0 * 2.0 * np.pi * _ROD_L * (_ROD_EPS * inner_int + outer_int)      # 軸上で max|E_z| = J0(0) = 1
    h_wall = abs(c * outer_dr(a, k2)) / (omega * MU0)
    rs = np.sqrt(omega * MU0 / (2.0 * _SIGMA))
    p_loss = 0.5 * rs * 2.0 * np.pi * a * _ROD_L * h_wall ** 2                      # 側面だけ（端面は E-short）
    return dict(f_ghz=omega / (2e9 * np.pi), omega=omega, u=u, v_acc=_ROD_L, p_loss=p_loss)


def test_rod_on_the_axis_matches_analytic(tmp_path):
    """軸上の誘電体棒: U・V・R/Q・P_loss の絶対値を解析解と比べる."""
    ref = _rod_ref()
    res, mode = _standing_tm0(tmp_path, _rod_pillbox(), ref["f_ghz"])
    assert mode.frequency_ghz == pytest.approx(ref["f_ghz"], rel=1e-4)
    # 実測の誤差（h = 2 mm）: U −0.26%、V −0.05%、R/Q +0.16%、P_loss −0.26%。h² で減る
    assert mode.stored_energy == pytest.approx(ref["u"], rel=0.005, abs=0)
    assert mode.v_acc == pytest.approx(ref["v_acc"], rel=0.002, abs=0)
    assert mode.rq_apparent == pytest.approx(ref["v_acc"] ** 2 / (ref["omega"] * ref["u"]), rel=0.005, abs=0)
    assert mode.p_loss == pytest.approx(ref["p_loss"], rel=0.005, abs=0)
    assert mode.q_wall == pytest.approx(ref["omega"] * ref["u"] / ref["p_loss"], rel=0.005, abs=0)


def _standing_tm0(tmp_path, geom, f_ghz):
    path = tmp_path / "standing.msh"
    export_msh_multi_region(geom, path, mesh_order=2, verbose=0)
    mesh = load_mesh(path, element_order=2)
    table = build_material_table_from_geometry(geom)
    res = solve_tm0_standing(mesh, num_modes=1, material_table=table, target_freq_ghz=f_ghz)
    params = compute_tm0_parameters(res, classify_boundaries(mesh.physical_groups, mesh.nodes), conductivity=_SIGMA,
                                    eps_r_per_element=build_eps_r_per_element(mesh, table))
    return res, params.per_phase[0.0][0]


def test_uniform_pillbox_rq_scales_with_inverse_sqrt_eps(tmp_path):
    """一様充填: 同じモード形で ω が 1/√ε_r、U = ½ε₀ε_r∫|E|²dV なので R/Q（V_acc）は 1/√ε_r 倍（3.0.0 までは ε_r^{3/2} 倍）."""
    f_tm010 = float(jn_zeros(0, 1)[0]) * C0 / (2e9 * np.pi * 50e-3)
    rq = {}
    for eps_r, tag in ((1.0, VACUUM_TAG), (4.0, "diel")):
        _, mode = _standing_tm0(tmp_path, _pillbox_geom(eps_r=eps_r, material_tag=tag), f_tm010 / np.sqrt(eps_r))
        rq[eps_r] = mode.rq_apparent
    assert rq[4.0] / rq[1.0] == pytest.approx(0.5, rel=1e-5)


@pytest.mark.parametrize("axis_side", [True, False], ids=["axis", "port"])
def test_solver_does_not_warn_about_post_for_a_dielectric(tmp_path, axis_side):
    """軸やポート断面に誘電体があっても後処理の警告はない（要素ごとの ε_r で計算する）. 見逃しの警告は test_tm0_low_modes.py."""
    geom = _rod_pillbox() if axis_side else _layered_cell(4.0)
    path = tmp_path / "m.msh"
    export_msh_multi_region(geom, path, mesh_order=1, verbose=0)
    mesh = load_mesh(path, element_order=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve_tm0_standing(mesh, num_modes=1, material_table=build_material_table_from_geometry(geom))
    assert not [w for w in caught if "ε_r" in str(w.message) or "eps_r" in str(w.message)]


def test_group_velocity_with_dielectric_at_the_port(tmp_path):
    """ポート断面に誘電体（軸は真空）: 解析解は無いので v_g = L dω/dθ（固有周波数だけ）と比べる."""
    res, params = _traveling_tm0(tmp_path, _layered_cell(4.0), (_THETA - 1.0, _THETA, _THETA + 1.0))
    low = res.phase_results[_THETA - 1.0]["frequencies"]
    high = res.phase_results[_THETA + 1.0]["frequencies"]
    fd = (high - low) * 2e9 * np.pi * (_L_MM * 1e-3) / np.radians(2.0) / C0
    vg = np.array([mode.v_group_c for mode in params.per_phase[_THETA]])
    assert vg == pytest.approx(fd, abs=2e-3)


def test_wall_q_of_a_filled_pillbox_scales_as_eps_to_minus_one_quarter(tmp_path):
    """同じモード形で ω が ε_r^(−1/2) 倍: Q_wall ∝ ω/R_s ∝ √ω なので ε_r^(−1/4) 倍（一様 ε_r = 4 で 0.70711）."""
    f_tm010 = float(jn_zeros(0, 1)[0]) * C0 / (2e9 * np.pi * 50e-3)      # 半径 50 mm
    q = {}
    for eps_r, tag in ((1.0, VACUUM_TAG), (4.0, "diel")):
        geom = _pillbox_geom(eps_r=eps_r, material_tag=tag)
        path = tmp_path / f"p{eps_r:g}.msh"
        export_msh_multi_region(geom, path, mesh_order=2, verbose=0)
        mesh = load_mesh(path, element_order=2)
        table = build_material_table_from_geometry(geom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = solve_tm0_standing(mesh, num_modes=1, material_table=table, target_freq_ghz=f_tm010 / np.sqrt(eps_r))
        assert res.frequencies[0] == pytest.approx(f_tm010 / np.sqrt(eps_r), rel=1e-3)
        params = compute_tm0_parameters(res, classify_boundaries(mesh.physical_groups, mesh.nodes), conductivity=_SIGMA,
                                        eps_r_per_element=build_eps_r_per_element(mesh, table))
        q[eps_r] = params.per_phase[0.0][0].q_wall
    assert q[4.0] / q[1.0] == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-5)


# ---------------------------------------------------------------------------
# CLI の post も ε_r を渡す: h5 の値が API（解析解で確かめた）と同じ
# ---------------------------------------------------------------------------
def _cli_filled_cell(tmp_path, analysis):
    from axicavity_fem.cli.main import cli_main
    from axicavity_fem.gui.jobs.pipeline import write_materials_json
    from axicavity_fem.shared.hdf5_io import read_results

    geom = _uniform_cell(4.0) if analysis == "traveling" else _pillbox_geom(eps_r=4.0, material_tag="diel")
    msh, raw, proc = tmp_path / "m.msh", tmp_path / "raw.h5", tmp_path / "proc.h5"
    export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
    write_materials_json(geom, msh)
    phase = ["-p", str(_THETA)] if analysis == "traveling" else []
    assert cli_main(["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "2", "--num-modes", "1", *phase, "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(proc), "--cond", str(_SIGMA)]) == 0
    post = read_results(proc)["post_process"][0]
    return geom, msh, (post["traveling"][_THETA] if analysis == "traveling" else post["standing"])


@pytest.mark.parametrize("analysis", ["standing", "traveling"])
def test_cli_post_matches_the_api_with_dielectric_on_the_axis(tmp_path, analysis):
    geom, msh, stored = _cli_filled_cell(tmp_path, analysis)
    mesh = load_mesh(msh, element_order=2)
    table = build_material_table_from_geometry(geom)
    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    if analysis == "traveling":
        res = solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(_THETA,), material_table=table)
    else:
        res = solve_tm0_standing(mesh, num_modes=1, material_table=table)
    eps = build_eps_r_per_element(mesh, table)
    with_eps = next(iter(compute_tm0_parameters(res, classification, conductivity=_SIGMA, eps_r_per_element=eps).per_phase.values()))[0]
    no_eps = next(iter(compute_tm0_parameters(res, classification, conductivity=_SIGMA).per_phase.values()))[0]
    assert stored[0]["U_stored"] == pytest.approx(with_eps.stored_energy, rel=1e-6, abs=0)
    assert stored[0]["R_over_Q"] == pytest.approx(with_eps.rq_eff, rel=1e-6, abs=0)
    assert abs(with_eps.rq_eff / no_eps.rq_eff - 1.0) > 0.5        # ε_r を落とすと 16 倍なので区別できる


def test_plots_of_a_tm0_result_use_the_dielectric(tmp_path):
    """場の図（GUI とレポート）の再構成にも ε_r が渡る（節点の E は ε_r で割る）."""
    pytest.importorskip("matplotlib")
    from axicavity_fem.gui.ui.result_renderer import ResultData, ResultRenderer
    from axicavity_fem.reports.plot_tm0 import _tm0_setup
    from axicavity_fem.shared.hdf5_io import read_results

    _cli_filled_cell(tmp_path, "standing")
    proc = tmp_path / "proc.h5"
    data = read_results(proc)
    assert _tm0_setup(data, 0, None)["recon"].eps_r_per_element == pytest.approx(4.0)
    assert ResultRenderer(ResultData(proc)).recon(0, "standing").eps_r_per_element == pytest.approx(4.0)


def test_region_tag_missing_from_the_material_table_warns(tmp_path):
    """真空として解くことを黙らない（別の材料ファイルを渡したとき）."""
    geom = _pillbox_geom(eps_r=4.0, material_tag="diel")
    path = tmp_path / "m.msh"
    export_msh_multi_region(geom, path, mesh_order=1, verbose=0)
    mesh = load_mesh(path, element_order=1)
    with pytest.warns(UserWarning, match="材料タグ 'diel'"):
        build_eps_r_per_element(mesh, {"other": {"eps_r": 2.0}})
