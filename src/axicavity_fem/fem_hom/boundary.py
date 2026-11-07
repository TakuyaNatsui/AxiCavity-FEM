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
def periodic_tolerance(vertices, edge_index_map):
    """周期面の判定に使う許容 [m] を返す: 周期面の一番短い辺の 1e-2 倍（1e-5 m とセル長の 1e-3 倍を超えない）.

    絶対値の許容だと、許容に近い大きさのメッシュで内部の辺を周期面の辺と取り違える。
    周期面と関係のない細かい部分には引きずられない。座標の丸めを吸収するので、面の辺より十分小さければ狭くしない。
    """
    if not edge_index_map:
        return 1e-5
    z = vertices[:, 0]
    z_lo, z_hi = np.min(z), np.max(z)
    band = 1e-3 * (z_hi - z_lo)
    ends = np.array(list(edge_index_map.keys()))
    on_low = (np.abs(z[ends[:, 0]] - z_lo) < band) & (np.abs(z[ends[:, 1]] - z_lo) < band)
    on_high = (np.abs(z[ends[:, 0]] - z_hi) < band) & (np.abs(z[ends[:, 1]] - z_hi) < band)
    plane = ends[on_low | on_high]
    if len(plane) == 0:
        return min(1e-5, band)
    lengths = np.linalg.norm(vertices[plane[:, 1]] - vertices[plane[:, 0]], axis=1)
    return min(1e-5, band, 1e-2 * float(np.min(lengths)))


def _check_stray_end_nodes(vertices, simplices, edge_index_map, wall_edge_indices, z_min, z_max, tol):
    """壁でない外周の辺の節点が周期面のすぐ外にあれば止める（対にならない節点が自由のまま残り、値が黙って誤る）.

    壁の点や内部の節点は周期面に近くてもよい（端面に斜めに達する壁、薄い壁）。
    """
    uses: dict[tuple[int, int], int] = {}
    for simplex in np.asarray(simplices)[:, :3]:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(simplex[a]), int(simplex[b]))))
            uses[key] = uses.get(key, 0) + 1
    walls = set(int(k) for k in wall_edge_indices)
    z = vertices[:, 0]
    for edge, count in uses.items():
        if count != 1 or edge_index_map[edge] in walls:
            continue
        for node in edge:
            for z_end in (z_min, z_max):
                offset = abs(z[node] - z_end)
                if tol <= offset < 5.0 * tol:
                    raise RuntimeError(
                        f"周期面の節点の z 座標がそろっていません（{offset:.3g} m のずれ、許容 {tol:.3g} m）。"
                        "メッシュの端面を確かめてください。")


def find_periodic_boundary_pairs(vertices, edge_index_map, n, pec_edge_indices, tol=1e-6, simplices=None,
                                 magnetic_edge_indices=()):
    """z 方向周期境界に対応する DOF ペアを検出する.

    Args:
        pec_edge_indices: PEC / E-short の壁の辺（周期面上のものを含んでよい。
            :func:`~axicavity_fem.shared.mesh_loader.compute_hom_boundary` が内部の辺を除いたもの）。
        simplices: 与えると、周期面のすぐ外に外れた端面の節点を検出する（解くときは与える）。
        magnetic_edge_indices: M-short の壁の辺（検出で壁として扱う。周期条件には関係しない）。

    Returns:
        (edge_pairs [(i, j, sign)], node_pairs [(i, j)], z_min, z_max, wall_images)。
        ``sign`` はエッジの基底の向き（番号の小さい節点 → 大きい節点）が r 方向で揃うなら +1、逆なら -1。
        節点ペアは n>0 のみ。周期面以外の PEC 壁に接する節点（外壁の角、アイリスの先端、
        同軸の内導体、内部の PEC シートの端）は E_θ = 0 なので、周期条件でつながず PEC に委ねる。
        壁が片方の周期面だけに触れるとき、もう片方の面の対の節点も壁の像なので E_θ = 0。
        ``wall_images`` はその節点（呼び出し側が PEC 節点に加える）。
    """
    num_nodes = len(vertices)
    if num_nodes == 0:
        return [], [], None, None, []

    z_coords = vertices[:, 0]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    if abs(z_max - z_min) < tol:
        return [], [], z_min, z_max, []
    edge_dofs = []
    for edge_nodes_sorted, edge_idx in edge_index_map.items():
        z1, r1 = vertices[edge_nodes_sorted[0]]
        z2, r2 = vertices[edge_nodes_sorted[1]]
        middle = (z1 + z2) / 2.0
        on_plane = False
        for z_end in (z_min, z_max):
            first, second = abs(z1 - z_end) < tol, abs(z2 - z_end) < tol
            on_plane = on_plane or (first and second)
            if first != second and abs(middle - z_end) < tol:
                raise RuntimeError(
                    f"周期面のすぐそばに節点があります（辺の両端の z の差 {abs(z2 - z1):.3g} m、許容 {tol:.3g} m）。"
                    "端面に対して壁がほぼ平行に近づく形状か、メッシュの座標のずれです。セルの切り方かメッシュを変えてください。")
        if on_plane:                 # 周期面の辺は両端が面の上にある（中点の z だけでは決めない）
            edge_dofs.append({"original_index": edge_idx, "z": middle, "r": (r1 + r2) / 2.0,
                              "r_direction": 1.0 if r2 > r1 else -1.0})
    if simplices is not None:
        _check_stray_end_nodes(vertices, simplices, edge_index_map, list(pec_edge_indices) + list(magnetic_edge_indices),
                               z_min, z_max, tol)

    node_dofs = []
    if n > 0:
        for node_idx in range(num_nodes):
            z, r = vertices[node_idx]
            node_dofs.append({"original_index": node_idx, "z": z, "r": r})

    def _pair_by_kdtree(dofs, kind):
        min_side = [d for d in dofs if abs(d["z"] - z_min) < tol]
        max_side = [d for d in dofs if abs(d["z"] - z_max) < tol]
        if len(min_side) != len(max_side):
            raise RuntimeError(
                f"{kind} DOF counts mismatch on periodic boundary: "
                f"z_min={len(min_side)}, z_max={len(max_side)}. "
                "進行波の HOM は z_min と z_max の端面が同じ r の範囲で同じメッシュである必要があります。"
                "片方の端面だけに壁があるときは、壁が端面に来ないようにセルの切り方を変えてください。")
        if not min_side:
            return []
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
        edge_pairs.append((dof_i["original_index"], dof_j["original_index"],
                           dof_i["r_direction"] * dof_j["r_direction"]))

    node_pairs = []
    wall_images = []
    if n > 0:
        periodic_edges = {k for pair in edge_pairs for k in pair[:2]}
        pec_edges = set(int(k) for k in pec_edge_indices)
        wall_nodes = set()
        for edge_nodes_sorted, edge_idx in edge_index_map.items():
            if edge_idx in pec_edges and edge_idx not in periodic_edges:
                wall_nodes.update(int(v) for v in edge_nodes_sorted)
        for dof_i, dof_j in _pair_by_kdtree(node_dofs, "node"):
            idx_i = dof_i["original_index"]
            idx_j = dof_j["original_index"]
            if idx_i in wall_nodes or idx_j in wall_nodes:   # 壁の点は PEC Dirichlet に委ねる
                wall_images.extend(k for k in (idx_i, idx_j) if k not in wall_nodes)
                continue
            node_pairs.append((idx_i, idx_j))

    return edge_pairs, node_pairs, z_min, z_max, wall_images


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
