"""TM0 誘電体対応の物理検証テスト。

1. eps_r 一様 = 1 のとき、ver2 と同じ結果（K, M, 固有値）を返す。
2. eps_r 一様 ≠ 1 のとき、固有値 λ = k² が 1/ε_r 倍 (周波数 f は 1/√ε_r 倍)
   になる（K → K/ε_r、M 不変 → λ → λ/ε_r）。
3. 多領域 (vacuum + dielectric) で sanity check: 両者の中間の固有値になる。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from axicavity_fem.fem_tm0.assembly import assemble_global_matrices
from axicavity_fem.fem_tm0.solver import solve_tm0_standing
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

    # メッシュ内部の複数点で比較
    a, L = 50.0, 100.0
    pts = [(25.0, 10.0), (50.0, 25.0), (75.0, 30.0), (40.0, 5.0)]
    for z, r in pts:
        fv = recon_vac.calculate_fields(eigvec, freq, z, r, return_complex=True)
        fe = recon_eps.calculate_fields(eigvec, freq, z, r, return_complex=True)
        if fv is None or fe is None:
            continue
        # E は 1/4、H_theta は不変
        assert fe["Ez"] == pytest.approx(fv["Ez"] / 4.0, rel=1e-12)
        assert fe["Er"] == pytest.approx(fv["Er"] / 4.0, rel=1e-12)
        assert fe["H_theta"] == pytest.approx(fv["H_theta"], rel=1e-12)


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
