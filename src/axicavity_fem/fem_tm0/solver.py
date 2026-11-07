"""TM0 standing / traveling 駆動.

ver1 ``run_fem_analysis_standingTM0`` / ``run_fem_analysis_travelingTM0`` に対応する。
メッシュ読込・BC 分類・全体行列組立・縮小・固有値解析を束ねて結果を返す。

  - 定在波 (standing): 実対称 ``eigsh`` (which='LA', shift-invert)
  - 進行波 (traveling): 複素・非エルミート ``eigs`` (which='LM', shift-invert)、
    周期境界の位相を複数まとめて処理

ver1 と同じ係数・ソルバ設定を用いることで数値一致を保つ。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import connected_components

from ..shared.boundary_groups import classify_boundaries
from ..shared.constants import C0, k2_from_freq_ghz
from ..shared.eigensolver import count_eigenvalues_below, solve_eigenmodes_eigs, solve_eigenmodes_eigsh
from ..shared.material_resolver import build_eps_r_per_element
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
    physical_edges: dict | None = None


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


# これより大きい問題では見逃しの本数を数えない（数えるのに固有値解析と同じくらいの時間とメモリがかかる）
_COUNT_MAX_DOF = 300_000
_COUNT_ALL_PHASES_DOF = 50_000


def _static_components(K_global, dirichlet_nodes, pairs=(), phase=None) -> list[np.ndarray]:
    """周波数 0 の静的な解 H_φ ∝ 1/r を持つ連結成分の節点の配列.

    H_φ の Dirichlet が無く（軸も M-short も無い）、定在波か、周期条件が θ = 0 かペアなしの成分にある。
    離散の固有値は 0 にならないので（1/r を要素で表せない）、周波数では判定しない。
    """
    count, labels = connected_components(K_global.tocsr(), directed=False)
    blocked = {int(labels[i]) for i in dirichlet_nodes}
    if phase is not None and abs((float(phase) + 180.0) % 360.0 - 180.0) > 1e-9:
        blocked |= {int(labels[i]) for pair in pairs for i in pair}
    components = [np.nonzero(labels == c)[0] for c in range(count) if c not in blocked]
    return [nodes for nodes in components if len(nodes) > 1]       # 要素に属さない孤立節点は除く


def _count_missed_modes(K, M, eigenvalues, static_count) -> int | None:
    """求めた最低の固有値より下に残っている（静的な解を除く）モードの本数. 数えられなければ None."""
    if len(eigenvalues) == 0:
        return None
    if K.shape[0] > _COUNT_MAX_DOF:
        print(f"情報: 問題が大きい（{K.shape[0]} 自由度）ので、見逃したモードの本数は数えません")
        return None
    limit = float(np.min(np.real(eigenvalues))) * (1.0 - 1e-3)
    try:
        below = count_eigenvalues_below(K, M, limit)
    except Exception as exc:  # noqa: BLE001  情報を出すだけの処理で解を失わない
        print(f"情報: 見逃したモードの本数を数えられませんでした: {exc}")
        return None
    if below is None:
        return None
    return max(below - static_count, 0)


def _warn_about_missed_modes(count, lowest_ghz, nodes, label="") -> None:
    if not count:
        return
    warnings.warn(
        f"[TM0] 予測周波数を指定していないので、低いモードを見逃しています{label}: 求めた最低の "
        f"{lowest_ghz:.4g} GHz より低いモードが {count} 本あります"
        f"（自動のシフト {_to_ghz(_sigma_from_rmax(nodes)):.4g} GHz の近くから探すため）。"
        "予測周波数（--target-freq）に低めの値を指定してください。",
        UserWarning, stacklevel=3)


def _warn_about_static_modes(eigenvectors, frequencies, M_global, nodes, components, label="") -> None:
    """返したモードが静的な解 H_φ ∝ 1/r に近ければ知らせる（細い内導体や粗いメッシュで実モードと混ざる）."""
    for component in components:
        v = np.zeros(len(nodes))
        v[component] = 1.0 / nodes[component, 1]
        mv = M_global @ v
        for k, x in enumerate(eigenvectors):
            norm = np.real(np.conj(x) @ (M_global @ x)) * np.real(v @ mv)
            if norm > 0.0 and abs(np.conj(x) @ mv) ** 2 / norm > 0.25:     # 振幅の重なり 0.5 以上
                warnings.warn(
                    f"[TM0] モード {k}（{float(frequencies[k]):.4g} GHz）{label}は、周波数 0 の静的な解 H_φ ∝ 1/r と混ざっています"
                    "（軸が無く H_φ の固定が無い形状で、内導体をメッシュで解像できていません）。"
                    "メッシュを細かくして周波数が 0 に向かうなら、実在の共振モードではありません。",
                    UserWarning, stacklevel=3)


def _warn_if_end_planes_do_not_pair(nodes, pairs, wall_nodes, tol=1e-8) -> None:
    """進行波で端面の節点が対にならないとき知らせる. 対のない節点は自然境界（壁）のまま残る."""
    z, r = nodes[:, 0], nodes[:, 1]
    low = np.nonzero(np.abs(z - np.min(z)) < tol)[0]
    high = np.nonzero(np.abs(z - np.max(z)) < tol)[0]
    if not pairs:
        warnings.warn(
            "[TM0] z_min と z_max の端面の節点が 1 つも対になりません。周期条件が掛からないので、結果は位相によりません"
            "（端面が直線か、節点がそろっているか確かめてください）。",
            UserWarning, stacklevel=3)
        return
    paired = {int(i) for pair in pairs for i in pair}
    r_lo, r_hi = max(np.min(r[low]), np.min(r[high])), min(np.max(r[low]), np.max(r[high]))
    # 壁のラベルの節点は、反対の端面の範囲の外なら対が無くてよい（片方の端面だけの壁）
    stray = [i for i in np.concatenate([low, high]) if int(i) not in paired and r[i] > 1e-6
             and (int(i) not in wall_nodes or r_lo + tol < r[i] < r_hi - tol)]
    if stray:
        warnings.warn(
            f"[TM0] z_min と z_max の端面の節点のうち {len(stray)} 個が対になりません。対のない節点は周期条件が掛からず壁のままになり、"
            "周波数が正しくありません。両方の端面を同じ r の位置で分けるか、端面が z 軸に垂直で r の値がそろっているか確かめてください。",
            UserWarning, stacklevel=3)


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
            ``None`` なら従来どおり r_max から自動推定した sigma より上のモードを
            低い順に返す（それより低いモードが残っていれば警告する）。

    Returns:
        :class:`TM0Result`（analysis_type="standing"）。
    """
    eps_r_per_element = build_eps_r_per_element(mesh, material_table)
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
    frequencies = _to_ghz(eigenvalues)

    static = _static_components(K_global, dirichlet_nodes)
    if target_freq_ghz is None or float(target_freq_ghz) <= 0.0:
        _warn_about_missed_modes(_count_missed_modes(K_reduced, M_reduced, eigenvalues, len(static)),
                                 float(np.min(frequencies)) if len(frequencies) else 0.0, mesh.nodes)
    _warn_about_static_modes(eigenvectors_full, frequencies, M_global, mesh.nodes, static)

    return TM0Result(
        nodes=mesh.nodes,
        elements=mesh.elements,
        element_order=mesh.element_order,
        analysis_type="standing",
        physical_groups=mesh.physical_groups,
        r0_nodes=find_nodes_on_r0_boundary(mesh.nodes),
        M_global=M_global,
        frequencies=frequencies,
        eigenvectors=eigenvectors_full,
        physical_edges=mesh.physical_edges,
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
    K_global, M_global = assemble_global_matrices(
        mesh, eps_r_per_element, n_quad=n_quad)

    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)

    periodic_pairs = identify_periodic_boundaries(mesh.nodes)
    _warn_if_end_planes_do_not_pair(mesh.nodes, periodic_pairs,
                                    {int(i) for i in classification.pec_nodes} | {int(i) for i in classification.eshort_nodes})
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
    phase_shifts = list(phase_shifts)
    # 大きな問題は分解が重いので、最初と最後と 180° に近い位相だけ数える
    checked_phases = ((phase_shifts[0], phase_shifts[-1],
                       min(phase_shifts, key=lambda p: abs((float(p) % 360.0) - 180.0))) if phase_shifts else ())
    missed = None
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
        frequencies = _to_ghz(eigenvalues)

        static = _static_components(K_global, dirichlet_nodes, periodic_pairs, phase)
        counted = K_reduced.shape[0] <= _COUNT_ALL_PHASES_DOF or phase in checked_phases
        if counted and (target_freq_ghz is None or float(target_freq_ghz) <= 0.0):
            count = _count_missed_modes(K_reduced, M_reduced, eigenvalues, len(static))
            if count and (missed is None or count > missed[0]):
                missed = (count, float(np.min(frequencies)), phase)
        _warn_about_static_modes(eigenvectors_full, frequencies, M_global, mesh.nodes, static, label=f"（θ = {phase:g}°）")

        phase_results[phase] = {
            "frequencies": frequencies,
            "eigenvectors": eigenvectors_full,
        }

    if missed is not None:
        _warn_about_missed_modes(missed[0], missed[1], mesh.nodes, label=f"（θ = {missed[2]:g}°）")

    return TM0Result(
        nodes=mesh.nodes,
        elements=mesh.elements,
        element_order=mesh.element_order,
        analysis_type="traveling",
        physical_groups=mesh.physical_groups,
        r0_nodes=find_nodes_on_r0_boundary(mesh.nodes),
        M_global=M_global,
        phase_results=phase_results,
        physical_edges=mesh.physical_edges,
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
