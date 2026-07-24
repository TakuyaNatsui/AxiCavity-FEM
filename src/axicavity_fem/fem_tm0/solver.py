"""TM0 standing / traveling 駆動.

ver1 ``run_fem_analysis_standingTM0`` / ``run_fem_analysis_travelingTM0`` に対応する。
メッシュ読込・BC 分類・全体行列組立・縮小・固有値解析を束ねて結果を返す。

  - 定在波 (standing): 実対称 ``eigsh`` (which='LA', shift-invert)
  - 進行波 (traveling): 複素・非エルミート ``eigs`` (which='LM', shift-invert)、
    周期境界の位相を複数まとめて処理

ver1 と同じ係数・ソルバ設定を用いることで数値一致を保つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import warnings

from ..shared.boundary_groups import classify_boundaries
from ..shared.constants import C0, k2_from_freq_ghz
from ..shared.eigensolver import solve_eigenmodes_eigs, solve_eigenmodes_eigsh
from ..shared.material_resolver import (
    build_eps_r_per_element,
    check_port_and_axis_are_vacuum,
)
from ..shared.mesh_loader import MeshData, load_mesh
from .assembly import assemble_global_matrices
from .boundary import (
    apply_bc_transformation,
    build_dirichlet_nodes,
    create_transformation_matrix,
    find_nodes_on_r0_boundary,
    identify_periodic_boundaries,
)


@dataclass
class TM0Result:
    """TM0 解析結果.

    Attributes:
        nodes, elements, element_order: メッシュ情報。
        analysis_type: "standing" または "traveling"。
        physical_groups: PhysicalGroup 辞書。
        r0_nodes: 軸 r=0 上の節点インデックス。
        M_global: 全体質量行列（規格化・エネルギー計算用）。
        frequencies: 定在波の周波数 [GHz]（standing のみ）。
        eigenvectors: 定在波の固有ベクトル (num_modes, N)（standing のみ）。
        phase_results: 進行波の ``{phase_deg: {...}}``（traveling のみ）。
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_order: int
    analysis_type: str
    physical_groups: dict = field(default_factory=dict)
    r0_nodes: list = field(default_factory=list)
    M_global: object = None
    frequencies: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    phase_results: dict | None = None


def _sigma_from_rmax(nodes: np.ndarray) -> float:
    """ver1 と同じシフト値 sigma = (2π (c/(4 r_max)) / c)^2 を返す."""
    r_max = np.max(nodes[:, 1])
    return (2 * np.pi * (C0 / (4 * r_max)) / C0) ** 2


def _resolve_sigma(nodes: np.ndarray, target_freq_ghz, which_default: str):
    """シフト値 sigma と ARPACK の選択基準 which を決める (ver2.3).

    ``target_freq_ghz`` が指定されていれば sigma = k^2(f) とし、その周波数に
    最も近いモードを取るため which='LM' に切り替える。未指定 (None / 非正)
    なら従来どおり r_max からの自動推定と ``which_default`` を使う。

    Returns:
        (sigma, which)。
    """
    if target_freq_ghz is not None and float(target_freq_ghz) > 0.0:
        sigma = k2_from_freq_ghz(target_freq_ghz)
        print(f"[TM0] 指定周波数 {float(target_freq_ghz):.6g} GHz 近傍を探索 "
              f"(sigma = {sigma:.6g} 1/m^2)")
        return sigma, "LM"
    return _sigma_from_rmax(nodes), which_default


def _to_ghz(eigenvalues) -> np.ndarray:
    """固有値 λ=k^2 [1/m^2] を周波数 [GHz] に変換する."""
    return (C0 / (2 * np.pi)) * np.sqrt(np.abs(eigenvalues)) / 1e9


def solve_tm0_standing(
    mesh: MeshData,
    num_modes: int = 10,
    material_table: dict | None = None,
    n_quad: int = 7,
    target_freq_ghz: float | None = None,
) -> TM0Result:
    """定在波 TM0 解析を実行する（実数ベース）.

    Args:
        mesh: :class:`~axicavity_fem.shared.mesh_loader.MeshData`。
        num_modes: 計算する固有モード数。
        material_table: ``{material_tag: {"eps_r": float, "mu_r": float}}``。
            ``mesh.region_names`` の各タグを引いて要素ごと eps_r を決める。
            ``None`` または mesh が region 情報を持たない場合は真空計算
            (ver2 互換)。
        n_quad: 要素行列の三角形ガウス求積点数 (既定 7)。ver1 一致は 4。
        target_freq_ghz: 探索したい周波数 [GHz] (ver2.3)。指定するとシフト値
            sigma = k^2(f) とし、その周波数に近いモードから順に返す。
            ``None`` なら従来どおり r_max から自動推定した sigma で
            最低次から返す。

    Returns:
        :class:`TM0Result`（analysis_type="standing"）。
    """
    eps_r_per_element = build_eps_r_per_element(mesh, material_table)
    _warn_if_dielectric_on_axis_or_port(mesh, eps_r_per_element)
    K_global, M_global = assemble_global_matrices(
        mesh, eps_r_per_element, n_quad=n_quad)

    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)

    T, _ = create_transformation_matrix(mesh.num_nodes, dirichlet_nodes)
    T = T.real   # 定在波は実数として扱う

    K_reduced = T.T @ K_global @ T
    M_reduced = T.T @ M_global @ T

    sigma, which = _resolve_sigma(mesh.nodes, target_freq_ghz, "LA")
    eigenvalues, eigenvectors_reduced = solve_eigenmodes_eigsh(
        K_reduced, M_reduced, num_eigenmodes=num_modes,
        sigma=sigma, which=which, tol=1e-9)

    eigenvectors_full = np.array(
        [np.asarray(T @ eigenvectors_reduced[:, i]).flatten()
         for i in range(len(eigenvalues))])

    return TM0Result(
        nodes=mesh.nodes,
        elements=mesh.elements,
        element_order=mesh.element_order,
        analysis_type="standing",
        physical_groups=mesh.physical_groups,
        r0_nodes=find_nodes_on_r0_boundary(mesh.nodes),
        M_global=M_global,
        frequencies=_to_ghz(eigenvalues),
        eigenvectors=eigenvectors_full,
    )


def solve_tm0_traveling(
    mesh: MeshData,
    num_modes: int = 10,
    phase_shifts=(120.0,),
    material_table: dict | None = None,
    n_quad: int = 7,
    target_freq_ghz: float | None = None,
) -> TM0Result:
    """進行波 TM0 解析を実行する（複素数ベース, 周期境界条件）.

    全体行列の組立を 1 回で済ませ、複数の位相シフトを連続計算する。

    Args:
        mesh: :class:`~axicavity_fem.shared.mesh_loader.MeshData`。
        num_modes: 計算する固有モード数。
        phase_shifts: 位相シフト [度] のリスト。
        material_table: ``solve_tm0_standing`` 参照。
        n_quad: 要素行列の三角形ガウス求積点数 (既定 7)。ver1 一致は 4。
        target_freq_ghz: 探索したい周波数 [GHz] (ver2.3)。``solve_tm0_standing``
            参照。進行波はもともと sigma 最近接 (which='LM') で探索するため、
            シフト値のみが置き換わる。

    Returns:
        :class:`TM0Result`（analysis_type="traveling"）。
    """
    eps_r_per_element = build_eps_r_per_element(mesh, material_table)
    _warn_if_dielectric_on_axis_or_port(mesh, eps_r_per_element)
    K_global, M_global = assemble_global_matrices(
        mesh, eps_r_per_element, n_quad=n_quad)

    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)

    periodic_pairs = identify_periodic_boundaries(mesh.nodes)
    if periodic_pairs:
        r0_set = set(find_nodes_on_r0_boundary(mesh.nodes))
        dirichlet_set = set(dirichlet_nodes)
        filtered = []
        for p_min, p_max in periodic_pairs:
            if p_min in r0_set or p_max in r0_set:
                continue
            dirichlet_set.discard(p_min)
            dirichlet_set.discard(p_max)
            filtered.append((p_min, p_max))
        periodic_pairs = filtered
        dirichlet_nodes = sorted(dirichlet_set)

    sigma, which = _resolve_sigma(mesh.nodes, target_freq_ghz, "LM")

    phase_results = {}
    for phase in phase_shifts:
        T, _ = create_transformation_matrix(
            mesh.num_nodes, dirichlet_nodes, periodic_pairs, phase)
        K_reduced, M_reduced = apply_bc_transformation(
            K_global, M_global, T, hermitian=True)

        eigenvalues, eigenvectors_reduced = solve_eigenmodes_eigs(
            K_reduced, M_reduced, num_eigenmodes=num_modes,
            sigma=sigma, which=which, tol=1e-9)

        eigenvectors_full = np.array(
            [np.asarray(T @ eigenvectors_reduced[:, i]).flatten()
             for i in range(len(eigenvalues))])

        phase_results[phase] = {
            "frequencies": _to_ghz(eigenvalues),
            "eigenvectors": eigenvectors_full,
        }

    return TM0Result(
        nodes=mesh.nodes,
        elements=mesh.elements,
        element_order=mesh.element_order,
        analysis_type="traveling",
        physical_groups=mesh.physical_groups,
        r0_nodes=find_nodes_on_r0_boundary(mesh.nodes),
        M_global=M_global,
        phase_results=phase_results,
    )


def solve_tm0(
    mesh_file: str,
    element_order: int = 2,
    num_modes: int = 10,
    phase_shifts=None,
    material_table: dict | None = None,
    n_quad: int = 7,
    target_freq_ghz: float | None = None,
) -> TM0Result:
    """メッシュファイルから TM0 解析を実行する便利関数.

    ``phase_shifts`` を与えると進行波、省略すると定在波解析を行う。

    Args:
        mesh_file: メッシュファイルパス (.msh)。
        element_order: 要素次数 (1 または 2)。
        num_modes: 固有モード数。
        phase_shifts: 進行波の位相シフト [度] のリスト。None で定在波。
        material_table: ``solve_tm0_standing`` 参照。
        n_quad: 要素行列の三角形ガウス求積点数 (既定 7)。ver1 一致は 4。
        target_freq_ghz: ``solve_tm0_standing`` 参照。

    Returns:
        :class:`TM0Result`。
    """
    mesh = load_mesh(mesh_file, element_order)
    if phase_shifts is None:
        return solve_tm0_standing(mesh, num_modes,
                                  material_table=material_table, n_quad=n_quad,
                                  target_freq_ghz=target_freq_ghz)
    return solve_tm0_traveling(mesh, num_modes, phase_shifts,
                               material_table=material_table, n_quad=n_quad,
                               target_freq_ghz=target_freq_ghz)


def _warn_if_dielectric_on_axis_or_port(
    mesh: MeshData, eps_r_per_element,
) -> None:
    """軸/ポート上に誘電体があれば後処理が不正確になるため警告する (ver2.1)。"""
    msgs = check_port_and_axis_are_vacuum(mesh, eps_r_per_element)
    for m in msgs:
        warnings.warn(
            f"[ver2.1 制約] {m} "
            "工学パラメータ (U, P_flow, V/V_eff, Q, R/Q) は不正確になる可能性があります。",
            stacklevel=3,
        )
