"""
boundary_conditions.py
境界条件の適用: 変換行列法によるディリクレBC・周期境界条件の処理

元の FEM_helmholtz_calclation.py から分離:
  - create_transformation_matrix         (ディリクレBC用)
  - apply_bc_transformation              (行列の縮小, 実数 T)
  - reconstruct_eigenvector_transformation (固有ベクトル復元)
  - find_periodic_boundary_pairs         (周期境界ペア検出)
  - create_transformation_matrix_from_constraints (一般拘束条件)
  - create_combined_transformation_matrix (PEC + PBC 統合, 2N×2N 実数展開)
  - create_complex_transformation_matrix (PEC + PBC 統合, N×N 複素数直接)
  - apply_bc_transformation_hermitian    (行列の縮小, 複素 T†)
"""

import time
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.spatial import KDTree

from element_assembly import get_edge_orientation_sign


# ==========================================================================
# ディリクレ境界条件 (定在波モード)
# ==========================================================================
def create_transformation_matrix(N, boundary_indices):
    """独立自由度への変換行列 T と独立自由度のインデックスを生成する

    x = T @ x_i  (x: 元のベクトル, x_i: 独立自由度ベクトル)

    Args:
        N: 元の自由度数
        boundary_indices: ディリクレ境界条件を適用する自由度のインデックスリスト

    Returns:
        T: 変換行列 CSR (N x M)
        internal_indices: 独立自由度の元のインデックス (ソート済み)
    """
    all_indices = np.arange(N)
    boundary_indices_set = set(boundary_indices)

    internal_indices = np.array(
        sorted(list(set(all_indices) - boundary_indices_set)), dtype=int)
    M = len(internal_indices)

    if M == 0:
        print("Warning: No independent degrees of freedom remain.")
        return csr_matrix((N, 0)), internal_indices

    row_indices = internal_indices
    col_indices = np.arange(M)
    data = np.ones(M, dtype=float)

    T = coo_matrix((data, (row_indices, col_indices)), shape=(N, M)).tocsr()

    print(f"Original matrix size (N): {N}")
    print(f"Number of boundary DOFs (N-M): {N - M}")
    print(f"Number of independent DOFs (M): {M}")
    print(f"Transformation matrix T shape: {T.shape}")

    return T, internal_indices


def apply_bc_transformation(K_global, M_global, T):
    """変換行列 T を用いて縮小された行列を計算する

    K_reduced = T^T @ K_global @ T
    M_reduced = T^T @ M_global @ T

    Args:
        K_global: 全体剛性行列 (CSR)
        M_global: 全体質量行列 (CSR)
        T: 変換行列 (CSR, N x M)

    Returns:
        K_reduced, M_reduced: 縮小された行列 (M x M)
    """
    if not isinstance(K_global, csr_matrix):
        K_global = K_global.tocsr()
    if not isinstance(M_global, csr_matrix):
        M_global = M_global.tocsr()
    if not isinstance(T, csr_matrix):
        T = T.tocsr()

    K_reduced = T.transpose() @ K_global @ T
    M_reduced = T.transpose() @ M_global @ T

    print(f"Reduced matrix K shape: {K_reduced.shape}")
    print(f"Reduced matrix M shape: {M_reduced.shape}")

    return K_reduced, M_reduced


def reconstruct_eigenvector_transformation(eigenvector_reduced, T):
    """変換行列 T を用いて、縮小された固有ベクトルを元のサイズに復元する

    x = T @ x_i

    Args:
        eigenvector_reduced: 縮小された固有ベクトル (サイズ M)
        T: 変換行列 (N x M)

    Returns:
        eigenvector_full: 元のサイズの固有ベクトル (サイズ N)
    """
    if T.shape[1] != len(eigenvector_reduced):
        raise ValueError(
            f"Shape mismatch: T columns {T.shape[1]} != "
            f"eigenvector length {len(eigenvector_reduced)}")

    eigenvector_full = T @ eigenvector_reduced
    return eigenvector_full


# ==========================================================================
# 周期境界条件 (進行波モード)
# ==========================================================================
def find_periodic_boundary_pairs(vertices, simplices, edge_index_map,
                                  num_edges, n, tol=1e-6):
    """z方向の周期境界に対応する自由度のペアを見つける

    エッジペアには方向整合性のための sign_factor を含める。
    sign_factor は要素情報と get_edge_orientation_sign を使って決定する。

    Args:
        vertices: 節点座標 (num_nodes x 2) [z, r]
        simplices: 要素リスト (num_elements x 3)
        edge_index_map: エッジインデックスマップ
        num_edges: エッジ自由度の数
        n: 方位角モード番号
        tol: 座標比較の許容誤差

    Returns:
        edge_pairs: [(idx_i, idx_j, sign_factor)] (idx: 0..num_edges-1)
        node_pairs: [(idx_i, idx_j)] (idx: 0..num_nodes-1)
        z_min_found: 検出された z_min 座標
        z_max_found: 検出された z_max 座標
    """
    num_nodes = len(vertices)
    num_elements = len(simplices)

    if num_nodes == 0:
        return [], [], None, None

    z_coords = vertices[:, 0]
    z_min_found = np.min(z_coords)
    z_max_found = np.max(z_coords)
    print(f"Auto-detected z_min: {z_min_found:.6f}, z_max: {z_max_found:.6f}")
    if abs(z_max_found - z_min_found) < tol:
        return [], [], z_min_found, z_max_found

    # --- 境界要素と境界エッジのマッピングを作成 ---
    edge_elements = [[] for _ in range(num_edges)]
    for elem_idx, simplex in enumerate(simplices):
        nodes = simplex
        local_edges_nodes = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[0]),
        ]
        for n1, n2 in local_edges_nodes:
            global_edge_key = tuple(sorted((n1, n2)))
            if global_edge_key in edge_index_map:
                global_edge_idx = edge_index_map[global_edge_key]
                edge_elements[global_edge_idx].append((elem_idx, n1, n2))
            else:
                print(f"Warning: Edge ({n1}, {n2}) in element {elem_idx} "
                      f"not found in edge_index_map.")

    # --- 自由度の情報を DOF 種別ごとに分けて構築 ---
    # 注意: 2次要素では「エッジ中点」と「辺上中点ノード」が物理的に同位置にある。
    #       両者を同じ KDTree でペアリングすると種別を取り違えて未拘束 DOF が
    #       生じ、進行波 n>0 で偽モードを生成する。種別別に処理する必要がある。
    edge_dofs = []
    edge_list = list(edge_index_map.keys())
    edge_indices_map_val = list(edge_index_map.values())
    for i, edge_nodes_sorted in enumerate(edge_list):
        edge_idx = edge_indices_map_val[i]
        node1_idx, node2_idx = edge_nodes_sorted
        z1, r1 = vertices[node1_idx]
        z2, r2 = vertices[node2_idx]
        edge_dofs.append({
            'original_index': edge_idx,
            'z': (z1 + z2) / 2.0,
            'r': (r1 + r2) / 2.0,
        })

    node_dofs = []
    if n > 0:
        for node_idx in range(num_nodes):
            z, r = vertices[node_idx]
            node_dofs.append({
                'original_index': node_idx,
                'z': z, 'r': r,
            })

    edge_pairs = []
    node_pairs = []

    def _pair_by_kdtree(dofs, kind):
        """DOF 種別 (kind: 'edge' or 'node') ごとに z_min ↔ z_max をペアリング"""
        min_side = [d for d in dofs if abs(d['z'] - z_min_found) < tol]
        max_side = [d for d in dofs if abs(d['z'] - z_max_found) < tol]
        print(f"  [{kind}] z=z_min: {len(min_side)} DOFs, "
              f"z=z_max: {len(max_side)} DOFs")

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

        min_r = np.array([[d['r']] for d in min_side])
        max_r = np.array([[d['r']] for d in max_side])
        max_tree = KDTree(max_r)
        dists, idxs = max_tree.query(min_r, k=1)

        pairs_local = []
        used_max = set()
        for i_min, dof_i in enumerate(min_side):
            j = idxs[i_min]
            if dists[i_min] >= tol:
                raise RuntimeError(
                    f"Could not pair {kind} DOF (orig_idx="
                    f"{dof_i['original_index']}, r={dof_i['r']:.6e}) "
                    f"with any DOF on z_max (best dist={dists[i_min]:.3e}).")
            if j in used_max:
                raise RuntimeError(
                    f"{kind} DOF on z_max (orig_idx="
                    f"{max_side[j]['original_index']}, r={max_side[j]['r']:.6e}) "
                    f"matched twice. Mesh r-coordinates may be degenerate.")
            used_max.add(j)
            pairs_local.append((dof_i, max_side[j]))
        return pairs_local

    # --- エッジ DOF のペアリング ---
    for dof_i, dof_j in _pair_by_kdtree(edge_dofs, 'edge'):
        idx_i = dof_i['original_index']
        idx_j = dof_j['original_index']

        elements_i = edge_elements[idx_i]
        elements_j = edge_elements[idx_j]

        if len(elements_i) != 1 or len(elements_j) != 1:
            print(f"Warning: Boundary edge {idx_i} or {idx_j} belongs to "
                  f"{len(elements_i)}/{len(elements_j)} elements. "
                  f"Assuming sign_factor = +1.")
            sign_factor = 1.0
        else:
            _, n1_i, n2_i = elements_i[0]
            _, n1_j, n2_j = elements_j[0]
            sign_orient1 = get_edge_orientation_sign(n1_i, n2_i)
            sign_orient2 = get_edge_orientation_sign(n1_j, n2_j)
            sign_factor = -1.0 * sign_orient1 * sign_orient2
            if sign_factor == -1.0:
                print(f"  Info: Sign factor -1 for edge pair ({idx_i}, {idx_j})")

        edge_pairs.append((idx_i, idx_j, sign_factor))

    # --- 節点 DOF のペアリング (n>0 のみ) ---
    # r≈r_max の角ノード (z_min/z_max, r_max) は PBC ペア対象外とする。
    # これらのノードは境界ノードセット (boundary_nodes_pec) に含まれるため、
    # 変換行列構築時の PEC Dirichlet 処理で正しく 0 に拘束される。
    # （メッシュの "PEC" グループが z境界面も含む場合、全節点が bnd_nodes に入るため
    #   r 座標で判定するのが唯一の確実な方法）
    if n > 0:
        r_max_domain = np.max(vertices[:, 1])
        for dof_i, dof_j in _pair_by_kdtree(node_dofs, 'node'):
            idx_i = dof_i['original_index']
            idx_j = dof_j['original_index']
            r_i = vertices[idx_i][1]
            r_j = vertices[idx_j][1]
            if abs(r_i - r_max_domain) < tol or abs(r_j - r_max_domain) < tol:
                # 外壁コーナー: PBC ではなく PEC Dirichlet に任せる
                print(f"  Info: Skipping PBC for r_max corner node pair "
                      f"(idx={idx_i}, r={r_i:.4f}) <-> (idx={idx_j}, r={r_j:.4f})")
            else:
                node_pairs.append((idx_i, idx_j))

    print(f"Generated {len(edge_pairs)} edge pairs (with sign) "
          f"and {len(node_pairs)} node pairs.")
    return edge_pairs, node_pairs, z_min_found, z_max_found


# ==========================================================================
# 一般拘束条件からの変換行列 (依存関係解決付き)
# ==========================================================================
def create_transformation_matrix_from_constraints(N_total, dirichlet_indices,
                                                   linear_constraints):
    """ディリクレ拘束と線形拘束から変換行列 T を構築する

    Args:
        N_total: 全自由度数
        dirichlet_indices: ディリクレ条件 X[k]=0 となるインデックス
        linear_constraints: [(dep, [(ind, coeff), ...])] 線形拘束

    Returns:
        T: 変換行列 CSR (N_total x M)
        internal_indices: 独立自由度の元のインデックス (ソート済み)
    """
    start_time = time.time()
    print("Starting transformation matrix creation...")

    # 1. 全従属自由度の特定
    dirichlet_set = set(dirichlet_indices)
    linear_constraints_dict = {}
    linear_dependent_dofs = set()

    for dep, constraints in linear_constraints:
        if dep in linear_constraints_dict:
            raise ValueError(
                f"DOF {dep} is defined multiple times in linear constraints.")
        if dep in dirichlet_set:
            raise ValueError(
                f"DOF {dep} is specified as both Dirichlet and Linear dependent.")
        linear_constraints_dict[dep] = constraints
        linear_dependent_dofs.add(dep)

    dependent_dofs = dirichlet_set.union(linear_dependent_dofs)
    print(f"  Total DOFs: {N_total}")
    print(f"  Dirichlet DOFs: {len(dirichlet_set)}")
    print(f"  Linearly Dependent DOFs: {len(linear_dependent_dofs)}")
    print(f"  Total Dependent DOFs: {len(dependent_dofs)}")

    # 2. 独立自由度の特定
    all_indices = np.arange(N_total)
    internal_indices = np.array(
        sorted(list(set(all_indices) - dependent_dofs)), dtype=int)
    M = len(internal_indices)
    print(f"  Independent DOFs (M): {M}")

    if M == 0:
        print("Warning: No independent degrees of freedom remain.")
        return csr_matrix((N_total, 0)), internal_indices

    # 3. 変換行列 T の構築 (LIL形式)
    T = lil_matrix((N_total, M), dtype=float)

    # 独立自由度 → 単位行列部分
    independent_map = {idx: j for j, idx in enumerate(internal_indices)}
    for idx, j in independent_map.items():
        T[idx, j] = 1.0

    # 4. 線形従属自由度の行を計算 (依存関係を反復的に解決)
    calculated_deps = set(internal_indices)
    remaining_deps = list(linear_dependent_dofs)
    max_iters = len(remaining_deps) + 2
    iters = 0

    print(f"  Calculating rows for {len(remaining_deps)} linearly dependent DOFs...")
    while remaining_deps and iters < max_iters:
        iters += 1
        processed_in_this_iter = []
        next_remaining = []

        for dep in remaining_deps:
            constraints = linear_constraints_dict[dep]
            can_calculate = True
            dependencies_rows = []

            for ind, coeff in constraints:
                if ind not in calculated_deps:
                    can_calculate = False
                    break
                dependencies_rows.append((T[ind, :], coeff))

            if can_calculate:
                T_row_new = lil_matrix((1, M), dtype=float)
                for row, coeff in dependencies_rows:
                    T_row_new += coeff * row

                T[dep, :] = T_row_new
                calculated_deps.add(dep)
                processed_in_this_iter.append(dep)
            else:
                next_remaining.append(dep)

        remaining_deps = next_remaining

        if not processed_in_this_iter and remaining_deps:
            print(f"Error: Could not resolve dependencies after {iters} iterations.")
            print(f"  Remaining unresolved dependent DOFs: {remaining_deps}")
            for unresolved_dep in remaining_deps:
                deps = linear_constraints_dict[unresolved_dep]
                uncalculated = [ind for ind, _ in deps if ind not in calculated_deps]
                print(f"    DOF {unresolved_dep} depends on uncalculated DOFs: {uncalculated}")
            raise RuntimeError(
                "Failed to construct T matrix due to unresolved dependencies.")

    if remaining_deps:
        raise RuntimeError(
            f"Failed to construct T matrix within {max_iters} iterations.")

    # 5. CSR形式に変換
    T_csr = T.tocsr()
    end_time = time.time()
    print(f"Transformation matrix T created. Shape: {T_csr.shape}. "
          f"Time: {end_time - start_time:.3f} sec")

    return T_csr, internal_indices


# ==========================================================================
# PEC + PBC 統合変換行列 (進行波・2N x 2N 系)
# ==========================================================================
def create_combined_transformation_matrix(
        num_nodes, num_edges, n, theta,
        edge_pairs, node_pairs,
        boundary_edges_pec, boundary_nodes_pec):
    """PEC と PBC を組み合わせて 2N x 2N 系用の変換行列 T を作成する

    周期境界条件: x_j = e^{j*theta} * x_i を実部・虚部に分離して適用。
    PBC を PEC より優先する。

    Args:
        num_nodes: 節点数
        num_edges: エッジ数
        n: 方位角モード番号
        theta: 周期境界の位相差 [rad]
        edge_pairs: PBC エッジペア [(idx_i, idx_j, sign_factor)]
        node_pairs: PBC 節点ペア [(idx_i, idx_j)]
        boundary_edges_pec: PEC エッジインデックス
        boundary_nodes_pec: PEC 節点インデックス

    Returns:
        T: 変換行列 CSR (2N x M)
        internal_indices: 独立自由度のインデックス
    """
    # 0. N_original と N_total の計算
    N_original = num_edges
    if n > 0:
        N_original += num_nodes
    elif n < 0:
        raise ValueError("n (azimuthal mode number) cannot be negative.")
    N_total = 2 * N_original  # 2N x 2N 系

    # 1. 拘束条件リストの初期化
    dirichlet_indices_list = []
    linear_constraints = []
    pbc_related_dofs_set = set()

    # 物理規約 (e^{+jωt}): x_max = e^{-jθ} · x_min = (cos θ - j sin θ) · x_min
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)   # Im 成分は -sin(θ) を使う
    node_offset = num_edges

    print("\n--- Generating Constraints for Transformation Matrix ---")
    print(f"  N_original = {N_original}, N_total (2N) = {N_total}")
    print(f"  Phase theta = {theta:.4f} rad  (PBC: x_max = e^{{-jθ}} x_min)")

    # 2. PBC 条件処理 (優先)
    print("  Processing Periodic Boundary Conditions (Priority)...")
    for idx_i, idx_j, sign_factor in edge_pairs:
        idx_re_i = idx_i
        idx_im_i = idx_i + N_original
        idx_re_j = idx_j
        idx_im_j = idx_j + N_original

        # Im(x[i]) = 0
        dirichlet_indices_list.append(idx_im_i)
        # Re(x[j]) = sign_factor * cos(theta) * Re(x[i])
        linear_constraints.append(
            (idx_re_j, [(idx_re_i, sign_factor * cos_theta)]))
        # Im(x[j]) = sign_factor * (-sin(theta)) * Re(x[i])  ← e^{-jθ} の虚部は -sinθ
        linear_constraints.append(
            (idx_im_j, [(idx_re_i, sign_factor * (-sin_theta))]))

        pbc_related_dofs_set.update([idx_re_i, idx_im_i, idx_re_j, idx_im_j])
    print(f"    Added constraints from {len(edge_pairs)} edge pairs.")

    # 節点ペア (n > 0)
    if n > 0:
        for idx_i, idx_j in node_pairs:
            dof_idx_i = node_offset + idx_i
            dof_idx_j = node_offset + idx_j
            idx_re_i = dof_idx_i
            idx_im_i = dof_idx_i + N_original
            idx_re_j = dof_idx_j
            idx_im_j = dof_idx_j + N_original

            dirichlet_indices_list.append(idx_im_i)
            linear_constraints.append(
                (idx_re_j, [(idx_re_i, cos_theta)]))
            linear_constraints.append(
                (idx_im_j, [(idx_re_i, -sin_theta)]))

            pbc_related_dofs_set.update(
                [idx_re_i, idx_im_i, idx_re_j, idx_im_j])
        print(f"    Added constraints from {len(node_pairs)} node pairs.")
    else:
        print("    Node pairs not applicable (n=0).")

    # 3. PEC 条件処理
    print("  Processing PEC Boundary Conditions...")
    pec_count = 0
    for k in boundary_edges_pec:
        idx_re_k = k
        idx_im_k = k + N_original
        if idx_re_k not in pbc_related_dofs_set:
            dirichlet_indices_list.append(idx_re_k)
            pec_count += 1
        if idx_im_k not in pbc_related_dofs_set:
            dirichlet_indices_list.append(idx_im_k)
            pec_count += 1
    print(f"    Added {pec_count} Dirichlet from {len(boundary_edges_pec)} PEC edges.")

    pec_node_count = 0
    if n > 0:
        for k in boundary_nodes_pec:
            dof_idx_k = node_offset + k
            idx_re_k = dof_idx_k
            idx_im_k = dof_idx_k + N_original
            if idx_re_k not in pbc_related_dofs_set:
                dirichlet_indices_list.append(idx_re_k)
                pec_node_count += 1
            if idx_im_k not in pbc_related_dofs_set:
                dirichlet_indices_list.append(idx_im_k)
                pec_node_count += 1
        print(f"    Added {pec_node_count} Dirichlet from "
              f"{len(boundary_nodes_pec)} PEC nodes.")

    # 4. 拘束条件を最終化
    unique_dirichlet_indices = sorted(list(set(dirichlet_indices_list)))
    print(f"  Total unique Dirichlet indices: {len(unique_dirichlet_indices)}")
    print(f"  Total linear constraints: {len(linear_constraints)}")

    # 5. 変換行列 T の生成
    T, internal_indices = create_transformation_matrix_from_constraints(
        N_total, unique_dirichlet_indices, linear_constraints)

    return T, internal_indices


# ==========================================================================
# PEC + PBC 複素変換行列 (進行波・N × N 系, 複素数直接)
# ==========================================================================
def create_complex_transformation_matrix(
        num_nodes, num_edges, n, theta,
        edge_pairs, node_pairs,
        boundary_edges_pec, boundary_nodes_pec):
    """PEC + PBC 変換行列を複素数で直接構築する (N×M, complex128)

    2N×2N 実数展開版 (create_combined_transformation_matrix) の代替。
    周期境界条件 x_j = sign * exp(jθ) * x_i を複素数係数で直接設定し、
    行列サイズを半減させる。

    Args:
        num_nodes: 節点数
        num_edges: エッジ数
        n: 方位角モード番号
        theta: 周期境界の位相差 [rad]
        edge_pairs: PBC エッジペア [(idx_i, idx_j, sign_factor)]
        node_pairs: PBC 節点ペア [(idx_i, idx_j)]
        boundary_edges_pec: PEC エッジインデックス
        boundary_nodes_pec: PEC 節点インデックス

    Returns:
        T: 変換行列 CSR (N_original x M, complex128)
        internal_indices: 独立自由度のインデックス配列
    """
    N_original = num_edges
    if n > 0:
        N_original += num_nodes
    elif n < 0:
        raise ValueError("n (azimuthal mode number) cannot be negative.")

    # 物理規約 (e^{+jωt}) に合わせた PBC: x_max = e^{-jθ} · x_min
    # +z 方向進行波の空間依存は e^{-jkz} → z_max での位相は e^{-jkL} = e^{-jθ}
    phase_factor = np.exp(-1j * theta)
    node_offset = num_edges

    print("\n--- Generating Complex Transformation Matrix ---")
    print(f"  N_original = {N_original}")
    print(f"  Phase theta = {theta:.4f} rad, exp(-jθ) = {phase_factor:.4f}")

    # 1. 従属 DOF の収集
    dependent_dofs = set()       # ディリクレ + PBC 従属
    pbc_related_dofs = set()     # PBC に関わる全 DOF（PEC 除外判定用）
    # PBC 制約: {dependent_dof: (independent_dof, complex_coefficient)}
    pbc_constraints = {}

    # エッジペア
    for idx_i, idx_j, sign_factor in edge_pairs:
        pbc_constraints[idx_j] = (idx_i, sign_factor * phase_factor)
        dependent_dofs.add(idx_j)
        pbc_related_dofs.update([idx_i, idx_j])
    print(f"  PBC edge pairs: {len(edge_pairs)}")

    # 節点ペア (n > 0)
    if n > 0:
        for idx_i, idx_j in node_pairs:
            dof_i = node_offset + idx_i
            dof_j = node_offset + idx_j
            pbc_constraints[dof_j] = (dof_i, phase_factor)
            dependent_dofs.add(dof_j)
            pbc_related_dofs.update([dof_i, dof_j])
        print(f"  PBC node pairs: {len(node_pairs)}")

    # PEC ディリクレ条件（PBC で使用済みの DOF は除外）
    pec_count = 0
    for k in boundary_edges_pec:
        if k not in pbc_related_dofs:
            dependent_dofs.add(k)
            pec_count += 1

    if n > 0:
        for k in boundary_nodes_pec:
            dof_k = node_offset + k
            if dof_k not in pbc_related_dofs:
                dependent_dofs.add(dof_k)
                pec_count += 1
    print(f"  PEC Dirichlet DOFs (excl. PBC): {pec_count}")

    # 2. 独立 DOF の決定
    all_dofs = set(range(N_original))
    independent_dofs = sorted(all_dofs - dependent_dofs)
    M = len(independent_dofs)
    print(f"  Independent DOFs (M): {M}")

    # 独立 DOF → 縮小行列内のインデックスマッピング
    dof_to_col = {dof: j for j, dof in enumerate(independent_dofs)}

    # 3. COO 形式で T を構築
    rows = []
    cols = []
    data = []

    # 独立 DOF: T[dof, j] = 1.0
    for dof in independent_dofs:
        rows.append(dof)
        cols.append(dof_to_col[dof])
        data.append(1.0 + 0.0j)

    # PBC 従属 DOF: T[dep, col(indep)] = coefficient
    for dep_dof, (indep_dof, coeff) in pbc_constraints.items():
        if indep_dof in dof_to_col:
            rows.append(dep_dof)
            cols.append(dof_to_col[indep_dof])
            data.append(coeff)

    T = coo_matrix((data, (rows, cols)),
                    shape=(N_original, M),
                    dtype=np.complex128).tocsr()

    internal_indices = np.array(independent_dofs, dtype=int)
    print(f"  T shape: {T.shape}, nnz: {T.nnz}")

    return T, internal_indices


# ==========================================================================
# 2次要素用: PEC ディリクレ境界 DOF インデックス取得
# ==========================================================================
def get_pec_dof_indices_2nd(num_edges, num_elements, num_nodes, n,
                             boundary_edge_indices, boundary_node_indices):
    """2次DOF体系における PEC ディリクレ境界 DOF のインデックスリストを返す。

    DOF 番号体系 (assemble_global_matrices_2nd と同一):
        CT/LN DOF of edge e : 2*e
        LT/LN DOF of edge e : 2*e + 1
        face DOF (N7) of elem k: 2*num_edges + 2*k
        face DOF (N8) of elem k: 2*num_edges + 2*k + 1
        node DOF of node i (n>0): 2*num_edges + 2*num_elements + i

    PEC 境界では:
        - 境界エッジの CT/LN, LT/LN 両方 → Dirichlet = 0
        - 面内 DOF は要素内部 → 常に自由 (境界条件不要)
        - 境界ノードの DOF (n>0) → Dirichlet = 0

    Args:
        num_edges (int): エッジ数
        num_elements (int): 要素数
        num_nodes (int): 節点数 (コーナー + 中点)
        n (int): 方位角モード次数
        boundary_edge_indices (list[int]): PEC 境界エッジのインデックス
        boundary_node_indices (list[int]): PEC 境界節点のインデックス (n>0 用)

    Returns:
        list[int]: Dirichlet 境界 DOF のインデックスリスト
    """
    node_offset = 2 * num_edges + 2 * num_elements
    pec_dofs = []

    # エッジ DOF: CT/LN (偶数) と LT/LN (奇数) の両方
    for e in boundary_edge_indices:
        pec_dofs.append(2 * e)       # CT/LN
        pec_dofs.append(2 * e + 1)   # LT/LN

    # ノード DOF (n > 0 のみ): G1-G6 (コーナー + 辺上中点)
    if n > 0:
        for i in boundary_node_indices:
            pec_dofs.append(node_offset + i)

    return sorted(set(pec_dofs))


# ==========================================================================
# 2次要素用: PEC + PBC 複素変換行列 (進行波・N×N 複素数直接)
# ==========================================================================
def create_complex_transformation_matrix_2nd(
        num_nodes, num_edges, num_elements, n, theta,
        edge_pairs, node_pairs,
        boundary_edges_pec, boundary_nodes_pec):
    """2次DOF体系における PEC + PBC 複素変換行列を構築する (N×M, complex128)。

    1次版 create_complex_transformation_matrix の 2次要素拡張。
    DOF 体系は assemble_global_matrices_2nd と同一。

    PBC 制約:
        CT/LN: x_j = sign_factor * exp(jθ) * x_i  (向き符号あり)
        LT/LN: x_j =              exp(jθ) * x_i  (向き符号なし、LT/LN は対称)
        face DOF: 周期ペアなし (要素内部 DOF)
        node DOF: x_j =          exp(jθ) * x_i

    Args:
        num_nodes (int): 節点数 (コーナー + 中点)
        num_edges (int): エッジ数
        num_elements (int): 要素数
        n (int): 方位角モード次数
        theta (float): 周期境界の位相差 [rad]
        edge_pairs (list): [(idx_i, idx_j, sign_factor)] PBC エッジペア
        node_pairs (list): [(idx_i, idx_j)] PBC 節点ペア
        boundary_edges_pec (list[int]): PEC 境界エッジインデックス
        boundary_nodes_pec (list[int]): PEC 境界節点インデックス

    Returns:
        T (csr_matrix): 変換行列 (N_original × M, complex128)
        internal_indices (ndarray): 独立 DOF のインデックス配列
    """
    face_offset = 2 * num_edges
    node_offset = 2 * num_edges + 2 * num_elements

    N_original = (2 * num_edges          # CT/LN + LT/LN
                  + 2 * num_elements     # face N7, N8
                  + (num_nodes if n > 0 else 0))  # node G1-G6

    # 物理規約 (e^{+jωt}) に合わせた PBC: x_max = e^{-jθ} · x_min
    phase_factor = np.exp(-1j * theta)

    print("\n--- Generating 2nd Order Complex Transformation Matrix ---")
    print(f"  N_original = {N_original}  (edges×2={2*num_edges}, "
          f"face×2={2*num_elements}, nodes={num_nodes if n > 0 else 0})")
    print(f"  Phase theta = {theta:.4f} rad, exp(-jθ) = {phase_factor:.4f}")

    # 1. 従属 DOF の収集
    dependent_dofs = set()
    pbc_related_dofs = set()
    pbc_constraints = {}   # {dep_dof: (indep_dof, complex_coeff)}

    # --- PBC エッジペア ---
    for idx_i, idx_j, sign_factor in edge_pairs:
        ct_i, lt_i = 2 * idx_i, 2 * idx_i + 1
        ct_j, lt_j = 2 * idx_j, 2 * idx_j + 1

        # CT/LN: 向き符号あり
        pbc_constraints[ct_j] = (ct_i, sign_factor * phase_factor)
        dependent_dofs.add(ct_j)
        pbc_related_dofs.update([ct_i, ct_j])

        # LT/LN: 対称形なので向き符号なし
        pbc_constraints[lt_j] = (lt_i, phase_factor)
        dependent_dofs.add(lt_j)
        pbc_related_dofs.update([lt_i, lt_j])

    print(f"  PBC edge pairs: {len(edge_pairs)} → {2*len(edge_pairs)} DOF pairs")

    # face DOF は要素内部 → 周期ペア不要

    # --- PBC 節点ペア (n > 0) ---
    if n > 0:
        for idx_i, idx_j in node_pairs:
            dof_i = node_offset + idx_i
            dof_j = node_offset + idx_j
            pbc_constraints[dof_j] = (dof_i, phase_factor)
            dependent_dofs.add(dof_j)
            pbc_related_dofs.update([dof_i, dof_j])
        print(f"  PBC node pairs: {len(node_pairs)}")

    # --- PEC ディリクレ条件 (PBC 使用済み DOF を除外) ---
    pec_count = 0
    for e in boundary_edges_pec:
        for dof in (2 * e, 2 * e + 1):   # CT/LN, LT/LN
            if dof not in pbc_related_dofs:
                dependent_dofs.add(dof)
                pec_count += 1

    if n > 0:
        for k in boundary_nodes_pec:
            dof_k = node_offset + k
            if dof_k not in pbc_related_dofs:
                dependent_dofs.add(dof_k)
                pec_count += 1

    print(f"  PEC Dirichlet DOFs (excl. PBC): {pec_count}")

    # 2. 独立 DOF の決定
    independent_dofs = sorted(set(range(N_original)) - dependent_dofs)
    M = len(independent_dofs)
    print(f"  Independent DOFs (M): {M}")

    dof_to_col = {dof: j for j, dof in enumerate(independent_dofs)}

    # 3. COO 形式で T を構築
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
                   shape=(N_original, M),
                   dtype=np.complex128).tocsr()

    internal_indices = np.array(independent_dofs, dtype=int)
    print(f"  T shape: {T.shape}, nnz: {T.nnz}")

    return T, internal_indices


def apply_bc_transformation_hermitian(K_global, M_global, T):
    """エルミート随伴転置 T† を使った行列縮小

    複素変換行列 T に対して K_reduced = T† K T, M_reduced = T† M T を計算する。
    K, M が実対称 → K_reduced, M_reduced は複素エルミート行列になる。

    Args:
        K_global: 全体剛性行列 (N x N, 実数)
        M_global: 全体質量行列 (N x N, 実数)
        T: 変換行列 (N x M, complex128)

    Returns:
        K_reduced, M_reduced: 縮小行列 (M x M, complex128, エルミート)
    """
    Th = T.conj().T
    K_reduced = Th @ K_global @ T
    M_reduced = Th @ M_global @ T
    return K_reduced, M_reduced
