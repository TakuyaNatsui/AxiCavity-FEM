"""HOM 境界条件適用.

DOF が Nédélec エッジ + 節点スカラーのため:
  PEC, E-short → Dirichlet (E_tan = 0)
  M-short       → Neumann（自然境界）
  軸 r=0, n≥1  → Dirichlet 強制

ver1 ``FEM_HOM_code/boundary_conditions.py`` から、run_analysis が実際に使う
関数のみを移植したもの（未使用の 2N×2N 実数展開・一般拘束ソルバは除外）。
進行波の周期境界位相は規約セーフティ層
:mod:`axicavity_fem.physics_core.pbc_phase` 経由で取得する（``e^{-jθ}`` を一元管理）。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import KDTree

from ..physics_core.pbc_phase import pbc_phase_factor
from .assembly import get_edge_orientation_sign


# ==========================================================================
# 定在波: ディリクレ境界条件
# ==========================================================================
def create_transformation_matrix(N, boundary_indices):
    """独立自由度への変換行列 T と独立 DOF インデックスを返す (x = T @ x_i).

    Args:
        N: 元の自由度数。
        boundary_indices: ディリクレ条件を課す DOF インデックス。

    Returns:
        (T (csr, N×M), internal_indices)。
    """
    boundary_set = set(int(i) for i in boundary_indices)
    internal_indices = np.array(
        sorted(set(range(N)) - boundary_set), dtype=int)
    M = len(internal_indices)
    if M == 0:
        return csr_matrix((N, 0)), internal_indices

    T = coo_matrix(
        (np.ones(M), (internal_indices, np.arange(M))),
        shape=(N, M)).tocsr()
    return T, internal_indices


def apply_bc_transformation(K_global, M_global, T):
    """実変換 ``T^T K T`` / ``T^T M T`` で縮小行列を返す（定在波）."""
    if not isinstance(K_global, csr_matrix):
        K_global = K_global.tocsr()
    if not isinstance(M_global, csr_matrix):
        M_global = M_global.tocsr()
    if not isinstance(T, csr_matrix):
        T = T.tocsr()
    return T.transpose() @ K_global @ T, T.transpose() @ M_global @ T


def apply_bc_transformation_hermitian(K_global, M_global, T):
    """複素変換 ``T† K T`` / ``T† M T`` で縮小行列を返す（進行波）."""
    Th = T.conj().T
    return Th @ K_global @ T, Th @ M_global @ T


def reconstruct_eigenvector_transformation(eigenvector_reduced, T):
    """縮小固有ベクトルを元サイズに復元する (x = T @ x_i)."""
    if T.shape[1] != len(eigenvector_reduced):
        raise ValueError(
            f"Shape mismatch: T columns {T.shape[1]} != "
            f"eigenvector length {len(eigenvector_reduced)}")
    return T @ eigenvector_reduced


# ==========================================================================
# 進行波: 周期境界ペアの検出
# ==========================================================================
def find_periodic_boundary_pairs(vertices, simplices, edge_index_map,
                                 num_edges, n, tol=1e-6):
    """z 方向周期境界に対応する DOF ペアを検出する.

    Returns:
        (edge_pairs [(i, j, sign)], node_pairs [(i, j)], z_min, z_max)。
        エッジペアは方向整合のための符号 ``sign_factor`` を含む。
        節点ペアは n>0 のみ（外壁コーナー r=r_max は PEC に委ねて除外）。
    """
    num_nodes = len(vertices)
    if num_nodes == 0:
        return [], [], None, None

    z_coords = vertices[:, 0]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    if abs(z_max - z_min) < tol:
        return [], [], z_min, z_max

    # エッジ → 要素・ローカル端点の対応
    edge_elements = [[] for _ in range(num_edges)]
    for elem_idx, simplex in enumerate(simplices):
        nodes = simplex
        for n1, n2 in [(nodes[0], nodes[1]), (nodes[1], nodes[2]),
                       (nodes[2], nodes[0])]:
            key = tuple(sorted((n1, n2)))
            if key in edge_index_map:
                edge_elements[edge_index_map[key]].append((elem_idx, n1, n2))

    edge_dofs = []
    for edge_nodes_sorted, edge_idx in edge_index_map.items():
        z1, r1 = vertices[edge_nodes_sorted[0]]
        z2, r2 = vertices[edge_nodes_sorted[1]]
        edge_dofs.append({"original_index": edge_idx,
                          "z": (z1 + z2) / 2.0, "r": (r1 + r2) / 2.0})

    node_dofs = []
    if n > 0:
        for node_idx in range(num_nodes):
            z, r = vertices[node_idx]
            node_dofs.append({"original_index": node_idx, "z": z, "r": r})

    def _pair_by_kdtree(dofs, kind):
        min_side = [d for d in dofs if abs(d["z"] - z_min) < tol]
        max_side = [d for d in dofs if abs(d["z"] - z_max) < tol]
        if not min_side or not max_side:
            if min_side or max_side:
                raise RuntimeError(
                    f"{kind} DOF counts mismatch on periodic boundary: "
                    f"z_min={len(min_side)}, z_max={len(max_side)}")
            return []
        if len(min_side) != len(max_side):
            raise RuntimeError(
                f"{kind} DOF counts mismatch on periodic boundary: "
                f"z_min={len(min_side)}, z_max={len(max_side)}")
        max_tree = KDTree(np.array([[d["r"]] for d in max_side]))
        dists, idxs = max_tree.query(np.array([[d["r"]] for d in min_side]), k=1)
        pairs_local = []
        used_max = set()
        for i_min, dof_i in enumerate(min_side):
            j = idxs[i_min]
            if dists[i_min] >= tol:
                raise RuntimeError(
                    f"Could not pair {kind} DOF "
                    f"(orig_idx={dof_i['original_index']}, r={dof_i['r']:.6e}).")
            if j in used_max:
                raise RuntimeError(
                    f"{kind} DOF on z_max matched twice "
                    f"(orig_idx={max_side[j]['original_index']}).")
            used_max.add(j)
            pairs_local.append((dof_i, max_side[j]))
        return pairs_local

    edge_pairs = []
    for dof_i, dof_j in _pair_by_kdtree(edge_dofs, "edge"):
        idx_i = dof_i["original_index"]
        idx_j = dof_j["original_index"]
        elements_i = edge_elements[idx_i]
        elements_j = edge_elements[idx_j]
        if len(elements_i) != 1 or len(elements_j) != 1:
            sign_factor = 1.0
        else:
            _, n1_i, n2_i = elements_i[0]
            _, n1_j, n2_j = elements_j[0]
            sign_factor = (-1.0 * get_edge_orientation_sign(n1_i, n2_i)
                           * get_edge_orientation_sign(n1_j, n2_j))
        edge_pairs.append((idx_i, idx_j, sign_factor))

    node_pairs = []
    if n > 0:
        r_max_domain = np.max(vertices[:, 1])
        for dof_i, dof_j in _pair_by_kdtree(node_dofs, "node"):
            idx_i = dof_i["original_index"]
            idx_j = dof_j["original_index"]
            r_i = vertices[idx_i][1]
            r_j = vertices[idx_j][1]
            if abs(r_i - r_max_domain) < tol or abs(r_j - r_max_domain) < tol:
                continue   # 外壁コーナーは PEC Dirichlet に委ねる
            node_pairs.append((idx_i, idx_j))

    return edge_pairs, node_pairs, z_min, z_max


# ==========================================================================
# 進行波: PEC + PBC 複素変換行列
# ==========================================================================
def create_complex_transformation_matrix(
        num_nodes, num_edges, n, theta,
        edge_pairs, node_pairs,
        boundary_edges_pec, boundary_nodes_pec):
    """1 次 DOF 体系の PEC + PBC 複素変換行列 (N×M, complex128) を構築する.

    PBC: ``x_j = sign · e^{-jθ} · x_i``（physics_core 経由）。PBC を PEC より優先。

    Returns:
        (T (csr complex), internal_indices)。
    """
    N_original = num_edges + (num_nodes if n > 0 else 0)
    phase_factor = pbc_phase_factor(theta)
    node_offset = num_edges

    dependent_dofs = set()
    pbc_related_dofs = set()
    pbc_constraints = {}

    for idx_i, idx_j, sign_factor in edge_pairs:
        pbc_constraints[idx_j] = (idx_i, sign_factor * phase_factor)
        dependent_dofs.add(idx_j)
        pbc_related_dofs.update([idx_i, idx_j])

    if n > 0:
        for idx_i, idx_j in node_pairs:
            dof_i, dof_j = node_offset + idx_i, node_offset + idx_j
            pbc_constraints[dof_j] = (dof_i, phase_factor)
            dependent_dofs.add(dof_j)
            pbc_related_dofs.update([dof_i, dof_j])

    for k in boundary_edges_pec:
        if k not in pbc_related_dofs:
            dependent_dofs.add(k)
    if n > 0:
        for k in boundary_nodes_pec:
            dof_k = node_offset + k
            if dof_k not in pbc_related_dofs:
                dependent_dofs.add(dof_k)

    independent_dofs = sorted(set(range(N_original)) - dependent_dofs)
    dof_to_col = {dof: j for j, dof in enumerate(independent_dofs)}

    rows, cols, data = [], [], []
    for dof in independent_dofs:
        rows.append(dof)
        cols.append(dof_to_col[dof])
        data.append(1.0 + 0.0j)
    for dep_dof, (indep_dof, coeff) in pbc_constraints.items():
        if indep_dof in dof_to_col:
            rows.append(dep_dof)
            cols.append(dof_to_col[indep_dof])
            data.append(coeff)

    T = coo_matrix((data, (rows, cols)),
                   shape=(N_original, len(independent_dofs)),
                   dtype=np.complex128).tocsr()
    return T, np.array(independent_dofs, dtype=int)


def create_complex_transformation_matrix_2nd(
        num_nodes, num_edges, num_elements, n, theta,
        edge_pairs, node_pairs,
        boundary_edges_pec, boundary_nodes_pec):
    """2 次 DOF 体系の PEC + PBC 複素変換行列 (N×M, complex128) を構築する.

    PBC: CT/LN は ``sign · e^{-jθ}``、LT/LN・node は ``e^{-jθ}``。face DOF はペアなし。

    Returns:
        (T (csr complex), internal_indices)。
    """
    node_offset = 2 * num_edges + 2 * num_elements
    N_original = (2 * num_edges + 2 * num_elements
                  + (num_nodes if n > 0 else 0))
    phase_factor = pbc_phase_factor(theta)

    dependent_dofs = set()
    pbc_related_dofs = set()
    pbc_constraints = {}

    for idx_i, idx_j, sign_factor in edge_pairs:
        ct_i, lt_i = 2 * idx_i, 2 * idx_i + 1
        ct_j, lt_j = 2 * idx_j, 2 * idx_j + 1
        pbc_constraints[ct_j] = (ct_i, sign_factor * phase_factor)
        dependent_dofs.add(ct_j)
        pbc_related_dofs.update([ct_i, ct_j])
        pbc_constraints[lt_j] = (lt_i, phase_factor)
        dependent_dofs.add(lt_j)
        pbc_related_dofs.update([lt_i, lt_j])

    if n > 0:
        for idx_i, idx_j in node_pairs:
            dof_i, dof_j = node_offset + idx_i, node_offset + idx_j
            pbc_constraints[dof_j] = (dof_i, phase_factor)
            dependent_dofs.add(dof_j)
            pbc_related_dofs.update([dof_i, dof_j])

    for e in boundary_edges_pec:
        for dof in (2 * e, 2 * e + 1):
            if dof not in pbc_related_dofs:
                dependent_dofs.add(dof)
    if n > 0:
        for k in boundary_nodes_pec:
            dof_k = node_offset + k
            if dof_k not in pbc_related_dofs:
                dependent_dofs.add(dof_k)

    independent_dofs = sorted(set(range(N_original)) - dependent_dofs)
    dof_to_col = {dof: j for j, dof in enumerate(independent_dofs)}

    rows, cols, data = [], [], []
    for dof in independent_dofs:
        rows.append(dof)
        cols.append(dof_to_col[dof])
        data.append(1.0 + 0.0j)
    for dep_dof, (indep_dof, coeff) in pbc_constraints.items():
        if indep_dof in dof_to_col:
            rows.append(dep_dof)
            cols.append(dof_to_col[indep_dof])
            data.append(coeff)

    T = coo_matrix((data, (rows, cols)),
                   shape=(N_original, len(independent_dofs)),
                   dtype=np.complex128).tocsr()
    return T, np.array(independent_dofs, dtype=int)


# ==========================================================================
# 2次要素: PEC ディリクレ DOF インデックス
# ==========================================================================
def get_pec_dof_indices_2nd(num_edges, num_elements, num_nodes, n,
                            boundary_edge_indices, boundary_node_indices):
    """2 次 DOF 体系での PEC ディリクレ境界 DOF インデックスを返す.

    境界エッジの CT/LN・LT/LN 両方と、境界節点 DOF（n>0）を Dirichlet にする。
    face DOF は要素内部なので拘束しない。
    """
    node_offset = 2 * num_edges + 2 * num_elements
    pec_dofs = []
    for e in boundary_edge_indices:
        pec_dofs.append(2 * e)
        pec_dofs.append(2 * e + 1)
    if n > 0:
        for i in boundary_node_indices:
            pec_dofs.append(node_offset + i)
    return sorted(set(pec_dofs))


def get_pec_dof_indices_1st(num_edges, n,
                            boundary_edge_indices, boundary_node_indices):
    """1 次 DOF 体系での PEC ディリクレ境界 DOF インデックスを返す."""
    dofs = list(boundary_edge_indices)
    if n > 0:
        dofs += [num_edges + i for i in boundary_node_indices]
    return sorted(set(dofs))
