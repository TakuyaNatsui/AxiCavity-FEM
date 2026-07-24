"""TM0 境界条件適用.

DOF が H_phi スカラー節点のため、各 BC の FEM 行列上の扱いは:

  PEC, E-short → Neumann（自然境界、何もしない）
  M-short       → Dirichlet (H_phi = 0)
  対称軸 r=0    → Dirichlet (H_phi = 0)（軸上で H_phi は消える）

Dirichlet 条件と周期境界条件（進行波）を変換行列 T に統合し、縮小行列を作る。
進行波の位相因子は規約セーフティ層 :mod:`axicavity_fem.physics_core.pbc_phase`
経由で取得する（``x_max = e^{-jθ} · x_min`` を一元管理）。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import KDTree

from ..physics_core.pbc_phase import pbc_phase_factor_deg
from ..shared.boundary_groups import BoundaryClassification


def find_nodes_on_r0_boundary(nodes: np.ndarray, r_threshold: float = 1e-6):
    """対称軸 r=0 上の節点インデックスのリストを返す."""
    return [i for i, p in enumerate(nodes) if abs(p[1]) < r_threshold]


def build_dirichlet_nodes(
    classification: BoundaryClassification,
    nodes: np.ndarray,
) -> list[int]:
    """TM0 の Dirichlet 条件節点（M-short ∪ 軸 r=0）をソート済みで返す.

    PEC / E-short は自然境界（Neumann）なので含めない。
    """
    dirichlet = set(int(i) for i in classification.mshort_nodes)
    dirichlet |= set(find_nodes_on_r0_boundary(nodes))
    return sorted(dirichlet)


def identify_periodic_boundaries(nodes: np.ndarray, tol: float = 1e-8):
    """z 軸に垂直な直線端 (z_min, z_max) で同一 r を持つ周期境界ペアを抽出する.

    Args:
        nodes: 節点座標 (N, 2). 各行は ``[z, r]``。
        tol: 座標比較の許容誤差。

    Returns:
        ``(node_idx_zmin, node_idx_zmax)`` のリスト（r=0 軸上の節点は除外）。
    """
    z_coords = nodes[:, 0]
    r_coords = nodes[:, 1]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    if (z_max - z_min) < tol:
        return []

    z_min_indices = np.where(np.abs(z_coords - z_min) < tol)[0]
    z_max_indices = np.where(np.abs(z_coords - z_max) < tol)[0]

    periodic_pairs = []
    if len(z_min_indices) > 0 and len(z_max_indices) > 0:
        tree = KDTree(r_coords[z_max_indices].reshape(-1, 1))
        for idx_min in z_min_indices:
            r_val = r_coords[idx_min]
            if r_val < tol:   # 軸上は Dirichlet を優先するためペアから除外
                continue
            dist, idx_in_max = tree.query([[r_val]])
            if dist[0] < tol:
                periodic_pairs.append((idx_min, z_max_indices[idx_in_max[0]]))

    periodic_pairs.sort(key=lambda x: r_coords[x[0]])
    return periodic_pairs


def create_transformation_matrix(
    N: int,
    dirichlet_nodes,
    periodic_pairs=None,
    phase_shift_deg: float = 0.0,
) -> tuple[csr_matrix, np.ndarray]:
    """独立自由度への変換行列 T を生成する (x = T @ x_i).

    周期境界条件とディリクレ境界条件を統合する。従属側 (idx_max) は独立側 (idx_min)
    に位相因子 ``e^{-jθ}`` を掛けて拘束する。

    Args:
        N: 元の自由度数。
        dirichlet_nodes: u=0 を課す節点インデックス列。
        periodic_pairs: ``(idx_min, idx_max)`` ペアのリスト。idx_max が従属側。
        phase_shift_deg: 位相差 [度]。``x_max = e^{-jθ} · x_min``。

    Returns:
        (T, internal_indices): T は (N, M) の複素 CSR、internal_indices は独立自由度の
        元ノードインデックス。
    """
    if periodic_pairs is None:
        periodic_pairs = []

    dirichlet_set = set(dirichlet_nodes)
    periodic_max_to_min = {p_max: p_min for p_min, p_max in periodic_pairs}
    periodic_max_set = set(periodic_max_to_min)

    internal_indices = []
    node_to_internal_idx = {}
    curr_idx = 0
    for i in range(N):
        if i not in dirichlet_set and i not in periodic_max_set:
            internal_indices.append(i)
            node_to_internal_idx[i] = curr_idx
            curr_idx += 1
    internal_indices = np.array(internal_indices)
    M = len(internal_indices)

    # 規約セーフティ層経由で位相因子を取得（e^{-jθ}）
    phase_factor = pbc_phase_factor_deg(phase_shift_deg)

    rows, cols, data = [], [], []
    for i in range(N):
        if i in dirichlet_set:
            continue
        if i in periodic_max_set:
            idx_min = periodic_max_to_min[i]
            if idx_min in node_to_internal_idx:
                rows.append(i)
                cols.append(node_to_internal_idx[idx_min])
                data.append(phase_factor)
        else:
            if i in node_to_internal_idx:
                rows.append(i)
                cols.append(node_to_internal_idx[i])
                data.append(1.0 + 0.0j)

    T = coo_matrix((data, (rows, cols)), shape=(N, M),
                   dtype=np.complex128).tocsr()
    return T, internal_indices


def apply_bc_transformation(K_global, M_global, T, hermitian: bool = False):
    """変換行列 T で縮小行列 ``T^H K T`` / ``T^H M T`` を返す.

    Args:
        K_global, M_global: 全体行列。
        T: 変換行列 (N, M)。
        hermitian: True なら共役転置 (進行波・複素)、False なら転置 (定在波・実数)。

    Returns:
        (K_reduced, M_reduced)。
    """
    if not isinstance(K_global, csr_matrix):
        K_global = K_global.tocsr()
    if not isinstance(M_global, csr_matrix):
        M_global = M_global.tocsr()
    if not isinstance(T, csr_matrix):
        T = T.tocsr()

    Tl = T.conjugate().transpose() if hermitian else T.transpose()
    K_reduced = Tl @ K_global @ T
    M_reduced = Tl @ M_global @ T
    return K_reduced, M_reduced
