"""HOM standing / traveling 駆動.

ver1 ``FEM_HOM_code/run_analysis.py`` の ``run_standing_wave`` / ``run_traveling_wave``
に対応する。メッシュ読込・全体行列組立・境界縮小・固有値解析を束ねて、
HDF5 非依存の :class:`HOMResult` を返す。

  - 定在波: 実対称 ``eigsh`` (shift-invert)、k^2 > sigma/100 を物理モードとして採用
  - 進行波: 複素エルミート ``eigsh`` (shift-invert)、k^2 > sigma/1000 を採用

固有値・全体行列は ver1 と同一になるよう、係数・DOF 体系・ソルバ設定を合わせている。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..shared.constants import C0, k2_from_freq_ghz
from ..shared.eigensolver import solve_eigenmodes_eigsh
from ..shared.material_resolver import build_eps_r_per_element
from ..shared.mesh_loader import HOMMeshData, load_mesh_hom
from .assembly import assemble_global_matrices, matrix_size_hom
from .boundary import (
    apply_bc_transformation,
    apply_bc_transformation_hermitian,
    create_complex_transformation_matrix,
    create_complex_transformation_matrix_2nd,
    create_transformation_matrix,
    find_periodic_boundary_pairs,
    get_pec_dof_indices_1st,
    get_pec_dof_indices_2nd,
    reconstruct_eigenvector_transformation,
)


@dataclass
class HOMModeSet:
    """1 つの解析（定在波 or 1 位相）の固有モード集合."""

    frequencies: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray   # (num_modes, N_original)


@dataclass
class HOMResult:
    """HOM 解析結果（方位角次数 n 固定）.

    Attributes:
        n, element_order: モード次数・要素次数。
        simplices, vertices, num_edges, edge_index_map, physical_groups: メッシュ。
        analysis_type: "standing" / "traveling"。
        normal: 定在波の :class:`HOMModeSet`（standing のみ）。
        periodic: ``{theta_deg: HOMModeSet}``（traveling のみ）。
        M_global: 全体質量行列（規格化・エネルギー計算用）。
    """

    n: int
    element_order: int
    simplices: np.ndarray
    vertices: np.ndarray
    num_edges: int
    edge_index_map: dict
    physical_groups: dict
    analysis_type: str
    normal: HOMModeSet | None = None
    periodic: dict = field(default_factory=dict)
    M_global: object = None
    pec_loss_edge_indices: list = field(default_factory=list)


def _sigma(vertices) -> float:
    """ver1 と同じシフト値 sigma = (2π·(c/(2 r_max))/c)^2 = (π/r_max)^2 を返す."""
    r_max = np.max(vertices[:, 1])
    freq_expct = C0 / (2 * r_max)
    return (2 * np.pi * freq_expct / C0) ** 2


def _resolve_sigma(vertices, target_freq_ghz):
    """固有値解析のシフト値とモード選択基準を返す (ver2.3).

    ``target_freq_ghz`` が指定されていれば sigma = k^2(f)、未指定なら
    従来どおり r_max からの自動推定 :func:`_sigma` を使う。

    Returns:
        ``(sigma, sigma_nearest)``。``sigma_nearest`` は指定時のみ sigma
        （:func:`_select_modes` が最低次からではなく指定周波数に近い順に
        モードを採用するためのフラグ兼基準値）、未指定なら ``None``。

    Note:
        偽モード除去の閾値 (sigma/100, sigma/1000) にこの値を使わないこと。
        閾値は k^2≈0 の非物理モードを落とすためのもので、常に自動 sigma を
        基準にする。
    """
    if target_freq_ghz is not None and float(target_freq_ghz) > 0.0:
        sigma = k2_from_freq_ghz(target_freq_ghz)
        print(f"[HOM] 指定周波数 {float(target_freq_ghz):.6g} GHz 近傍を探索 "
              f"(sigma = {sigma:.6g} 1/m^2)")
        return sigma, sigma
    return _sigma(vertices), None


def _to_ghz(eigenvalue) -> float:
    return (C0 / (2 * np.pi)) * np.sqrt(np.abs(eigenvalue)) / 1e9


def _num_to_solve(num_modes, reduced_size):
    k = num_modes * 3
    if k >= reduced_size:
        k = max(1, reduced_size - 1)
    return k


def _select_modes(eigenvalues, eigenvectors_reduced, T, num_modes, threshold,
                  sigma_nearest=None):
    """物理モード（k^2 > threshold）を num_modes 個まで復元して返す.

    既定では入力順（λ 昇順）に先頭から採用する＝最低次から num_modes 本。
    ``sigma_nearest`` を与えると（ver2.3 探索周波数指定）、まず |λ-σ| の
    小さい順に num_modes 本を選び、その中を λ 昇順に並べて返す。
    """
    order = range(len(eigenvalues))
    if sigma_nearest is not None:
        cand = [i for i in order if eigenvalues[i] > threshold]
        cand.sort(key=lambda i: abs(eigenvalues[i] - sigma_nearest))
        order = sorted(cand[:num_modes], key=lambda i: eigenvalues[i])
    freqs, evals, evecs = [], [], []
    for i in order:
        ev = eigenvalues[i]
        if ev > threshold:
            if len(freqs) >= num_modes:
                break
            evecs.append(np.asarray(T @ eigenvectors_reduced[:, i]).flatten())
            evals.append(ev)
            freqs.append(_to_ghz(ev))
    return (np.array(freqs), np.array(evals),
            np.array(evecs) if evecs else np.empty((0, T.shape[0])))


def _resolve_eps_r(mesh: HOMMeshData, material_table):
    """material_table から要素ごと ε_r 配列を作り、誘電体があれば警告する.

    Returns:
        eps_r_per_element (E,) または None（真空・従来互換）。
    """
    eps_r = build_eps_r_per_element(mesh, material_table)
    if eps_r is not None and np.any(np.abs(eps_r - 1.0) > 1e-12):
        warnings.warn(
            "[ver2.1 HOM] 誘電体 (eps_r != 1) が指定されました。"
            "HOM は E 場定式化のため M に eps_r を乗じます（共振周波数は正確）。"
            "ただし後処理 (U, P_loss, Q 等) は真空前提のため不正確になる場合があります。",
            UserWarning, stacklevel=2,
        )
    return eps_r


def _standing_boundary_dofs(mesh: HOMMeshData):
    if mesh.element_order == 2:
        return get_pec_dof_indices_2nd(
            mesh.num_edges, len(mesh.simplices), len(mesh.vertices), mesh.n,
            mesh.boundary_edge_indices, mesh.boundary_vertices_indices)
    return get_pec_dof_indices_1st(
        mesh.num_edges, mesh.n,
        mesh.boundary_edge_indices, mesh.boundary_vertices_indices)


def solve_hom_standing(mesh: HOMMeshData, num_modes: int = 10,
                       material_table: dict | None = None,
                       target_freq_ghz: float | None = None) -> HOMResult:
    """定在波 HOM 解析を実行する.

    Args:
        mesh: :class:`~axicavity_fem.shared.mesh_loader.HOMMeshData`。
        num_modes: 採用する物理モード数。
        material_table: ``{material_tag: {"eps_r": float, "mu_r": float}}``。
            誘電体のとき M に ε_r を乗じる（E 場定式化）。``None`` で真空。
        target_freq_ghz: 探索したい周波数 [GHz] (ver2.3)。指定するとシフト値
            sigma = k^2(f) となり、その周波数に近いモードが得られる。
            ``None`` なら従来どおり r_max からの自動推定。

    Returns:
        :class:`HOMResult`（analysis_type="standing"）。
    """
    eps_r = _resolve_eps_r(mesh, material_table)
    K_global, M_global = assemble_global_matrices(mesh, eps_r_per_element=eps_r)
    size = matrix_size_hom(mesh)

    boundary_dofs = _standing_boundary_dofs(mesh)
    T, _ = create_transformation_matrix(size, boundary_dofs)
    K_reduced, M_reduced = apply_bc_transformation(K_global, M_global, T)

    sigma, sigma_nearest = _resolve_sigma(mesh.vertices, target_freq_ghz)
    k_solve = _num_to_solve(num_modes, K_reduced.shape[0])
    eigenvalues, eigvecs_reduced = solve_eigenmodes_eigsh(
        K_reduced, M_reduced, num_eigenmodes=k_solve, sigma=sigma)

    # 偽モード除去の閾値は常に自動 sigma 基準（指定周波数でスケールしない）
    freqs, evals, evecs = _select_modes(
        eigenvalues, eigvecs_reduced, T, num_modes, _sigma(mesh.vertices) / 100,
        sigma_nearest=sigma_nearest)

    return HOMResult(
        n=mesh.n, element_order=mesh.element_order,
        simplices=mesh.simplices, vertices=mesh.vertices,
        num_edges=mesh.num_edges, edge_index_map=mesh.edge_index_map,
        physical_groups=mesh.physical_groups, analysis_type="standing",
        normal=HOMModeSet(freqs, evals, evecs), M_global=M_global,
        pec_loss_edge_indices=mesh.pec_loss_edge_indices)


def solve_hom_traveling(mesh: HOMMeshData, num_modes: int = 10,
                        phase_shifts=(120.0,),
                        material_table: dict | None = None,
                        target_freq_ghz: float | None = None) -> HOMResult:
    """進行波 HOM 解析を実行する（周期境界条件・複素）.

    Args:
        mesh: :class:`HOMMeshData`。
        num_modes: 採用する物理モード数。
        phase_shifts: 位相 θ [度] のリスト。
        material_table: ``solve_hom_standing`` 参照。誘電体で M に ε_r を乗じる。
        target_freq_ghz: ``solve_hom_standing`` 参照 (ver2.3)。

    Returns:
        :class:`HOMResult`（analysis_type="traveling"）。
    """
    eps_r = _resolve_eps_r(mesh, material_table)
    K_global, M_global = assemble_global_matrices(mesh, eps_r_per_element=eps_r)
    num_nodes = len(mesh.vertices)
    num_elements = len(mesh.simplices)
    sigma, sigma_nearest = _resolve_sigma(mesh.vertices, target_freq_ghz)
    # 偽モード除去の閾値は常に自動 sigma 基準（指定周波数でスケールしない）
    spurious_threshold = _sigma(mesh.vertices) / 1000

    periodic = {}
    for theta_deg in phase_shifts:
        theta_rad = np.deg2rad(theta_deg)
        edge_pairs, node_pairs, _, _ = find_periodic_boundary_pairs(
            mesh.vertices, mesh.simplices, mesh.edge_index_map,
            mesh.num_edges, mesh.n, tol=1e-5)

        if mesh.element_order == 2:
            T, _ = create_complex_transformation_matrix_2nd(
                num_nodes, mesh.num_edges, num_elements, mesh.n, theta_rad,
                edge_pairs, node_pairs,
                mesh.boundary_edge_indices, mesh.boundary_vertices_indices)
        else:
            T, _ = create_complex_transformation_matrix(
                num_nodes, mesh.num_edges, mesh.n, theta_rad,
                edge_pairs, node_pairs,
                mesh.boundary_edge_indices, mesh.boundary_vertices_indices)

        K_reduced, M_reduced = apply_bc_transformation_hermitian(
            K_global, M_global, T)
        k_solve = _num_to_solve(num_modes, K_reduced.shape[0])
        eigenvalues, eigvecs_reduced = solve_eigenmodes_eigsh(
            K_reduced, M_reduced, num_eigenmodes=k_solve, sigma=sigma)

        freqs, evals, evecs = _select_modes(
            eigenvalues, eigvecs_reduced, T, num_modes, spurious_threshold,
            sigma_nearest=sigma_nearest)
        periodic[theta_deg] = HOMModeSet(freqs, evals, evecs)

    return HOMResult(
        n=mesh.n, element_order=mesh.element_order,
        simplices=mesh.simplices, vertices=mesh.vertices,
        num_edges=mesh.num_edges, edge_index_map=mesh.edge_index_map,
        physical_groups=mesh.physical_groups, analysis_type="traveling",
        periodic=periodic, M_global=M_global,
        pec_loss_edge_indices=mesh.pec_loss_edge_indices)


def solve_hom(mesh_file: str, n: int, element_order: int = 1,
              num_modes: int = 10, phase_shifts=None,
              material_table: dict | None = None,
              target_freq_ghz: float | None = None) -> HOMResult:
    """メッシュファイルから方位角次数 n の HOM 解析を実行する便利関数.

    ``phase_shifts`` を与えると進行波、省略すると定在波解析を行う。
    ``material_table`` を与えると誘電体（M に ε_r）として解く。
    ``target_freq_ghz`` を与えるとその周波数近傍のモードを探索する。
    """
    mesh = load_mesh_hom(mesh_file, n, element_order)
    if phase_shifts is None:
        return solve_hom_standing(mesh, num_modes,
                                  material_table=material_table,
                                  target_freq_ghz=target_freq_ghz)
    return solve_hom_traveling(mesh, num_modes, phase_shifts,
                               material_table=material_table,
                               target_freq_ghz=target_freq_ghz)
