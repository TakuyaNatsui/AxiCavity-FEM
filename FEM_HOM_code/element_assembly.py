"""
element_assembly.py
要素行列（剛性行列K・質量行列M）の組み立てと全体行列アセンブリ

1次要素:
  - assemble_stiffness_matrix_element   : 要素剛性行列 (scalar, forループ)
  - assemble_mass_matrix_element         : 要素質量行列 (scalar, forループ)
  - assemble_element_matrices_1st_batch  : 要素行列 (NumPy ベクトル化, 全要素一括)  ★推奨
  - assemble_global_matrices_vectorized  : 全体行列 (ベクトル化版, 高速)            ★推奨
  - assemble_global_matrices             : 全体行列 (forループ版, 参照用)

2次要素:
  - assemble_element_matrices_2nd        : 要素行列 (scalar, forループ)
  - assemble_global_matrices_2nd         : 全体行列 (forループ版, 参照用)
  - assemble_element_matrices_2nd_batch  : 要素行列 (NumPy ベクトル化, 全要素一括)
  - assemble_global_matrices_2nd_vectorized : 全体行列 (ベクトル化版, 高速・曲線要素対応)
    → 等パラメトリック写像で Jacobian を正確に計算するため曲線要素にも対応
    → 直線要素では assemble_global_matrices_2nd と機械精度で一致
    → 速度: forループ版の約 20-40倍 (メッシュ規模依存)
"""

import numpy as np
from scipy.sparse import coo_matrix

from FEM_element_function import (
    calculate_triangle_area_double,
    calculate_area_coordinates,
    grad_area_coordinates,
    calculate_edge_shape_functions,
    calculate_edge_shape_functions_2nd,
    calculate_curl_edge_shape_functions_2nd,
    calculate_quadratic_nodal_shape_functions,
    grad_quadratic_nodal_shape_functions,
)
from gaussian_quadrature_triangle import (
    gaussian_quadrature_triangle,
    integration_points_triangle,
    calculate_triangle_area,
)


def get_edge_orientation_sign(global1, global2):
    """エッジの向きを決定する符号を返す（グローバルインデックスの昇順なら+1）"""
    return 1 if global1 < global2 else -1


# ==========================================================================
# 要素剛性行列 K_e
# ==========================================================================
def assemble_stiffness_matrix_element(vertices, n):
    """要素剛性行列を組み立てる

    PDF式(54): n=0 の場合 k_ij = rc/Ae
    PDF式(64): n>0 の場合 ハイブリッド要素の6x6行列
    (係数2は省略: PDFの記法に従う)

    Args:
        vertices: 三角形要素の頂点座標 (3x2 array) [z, r]
        n: 方位角モード次数

    Returns:
        K_e: 要素剛性行列 (n=0: 3x3, n>0: 6x6)
    """
    r1, r2, r3 = vertices
    r1_r, r2_r, r3_r = r1[1], r2[1], r3[1]
    rc = (r1_r + r2_r + r3_r) / 3
    A2 = calculate_triangle_area_double(vertices)
    A = A2 / 2
    grad_L = grad_area_coordinates(vertices)

    if n == 0:
        # PDF式(54): k_ij = rc / Ae (係数2省略)
        K_e = np.full((3, 3), rc / A)

    elif n > 0:
        # --- 被積分関数の定義 ---
        def func_rover_Ni_Nj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            r = L1 * rv1[1] + L2 * rv2[1] + L3 * rv3[1]
            z = L1 * rv1[0] + L2 * rv2[0] + L3 * rv3[0]
            rv = np.array([z, r])
            N = calculate_edge_shape_functions(rv, vertices)
            return (1.0 / r) * np.dot(N[i], N[j])

        def func_rover_Ni_dot_gradLj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            r = L1 * rv1[1] + L2 * rv2[1] + L3 * rv3[1]
            z = L1 * rv1[0] + L2 * rv2[0] + L3 * rv3[0]
            rv = np.array([z, r])
            N = calculate_edge_shape_functions(rv, vertices)
            grad_L_local = grad_area_coordinates(vertices)
            return np.dot(N[i], grad_L_local[j]) / r

        def func_rover_1(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            r = L1 * rv1[1] + L2 * rv2[1] + L3 * rv3[1]
            return 1.0 / r

        K_e = np.zeros((6, 6))

        # edge-edge ブロック (i<=3, j<=3)
        # PDF式(64): k_ij = m^2 ∫∫ Ni·Nj/r dzdr + rc/Ae
        for i in range(3):
            for j in range(i, 3):
                term1 = n * n * gaussian_quadrature_triangle(func_rover_Ni_Nj, vertices, i, j)
                term2 = rc / A
                K_e[i, j] = K_e[j, i] = term1 + term2

            # edge-node ブロック (i<=3, j>3)
            # PDF式(64): k_ij = -m ∫∫ Ni·∇L(j-3) / r dzdr
            for j in range(3, 6):
                K_e[i, j] = K_e[j, i] = -n * gaussian_quadrature_triangle(
                    func_rover_Ni_dot_gradLj, vertices, i, j - 3)

        # node-node ブロック (i>3, j>3)
        # PDF式(64): k_ij = ∇L(i-3)·∇L(j-3) ∫∫ 1/r dzdr
        for i in range(3, 6):
            for j in range(i, 6):
                K_e[i, j] = K_e[j, i] = np.dot(grad_L[i - 3], grad_L[j - 3]) * \
                    gaussian_quadrature_triangle(func_rover_1, vertices, i - 3, j - 3)

    return K_e


# ==========================================================================
# 要素質量行列 M_e
# ==========================================================================
def assemble_mass_matrix_element(vertices, n):
    """要素質量行列を組み立てる

    PDF式(55): n=0 の場合 m_ij = ∫∫ Ni·Nj r dzdr
    PDF式(64): n>0 の場合 ハイブリッド要素の6x6行列
    (係数2は省略: PDFの記法に従う)

    Args:
        vertices: 三角形要素の頂点座標 (3x2 array) [z, r]
        n: 方位角モード次数

    Returns:
        M_e: 要素質量行列 (n=0: 3x3, n>0: 6x6)
    """
    def func_r_NiNj(L1, L2, L3, vertices, i, j):
        rv1, rv2, rv3 = vertices
        r = L1 * rv1[1] + L2 * rv2[1] + L3 * rv3[1]
        z = L1 * rv1[0] + L2 * rv2[0] + L3 * rv3[0]
        rv = np.array([z, r])
        N = calculate_edge_shape_functions(rv, vertices)
        return r * np.dot(N[i], N[j])

    def func_rover_LiLj(L1, L2, L3, vertices, i, j):
        L = [L1, L2, L3]
        rv1, rv2, rv3 = vertices
        r = L1 * rv1[1] + L2 * rv2[1] + L3 * rv3[1]
        return L[i] * L[j] / r

    if n == 0:
        # PDF式(55): m_ij = ∫∫ Ni·Nj r dzdr
        M_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                M_e[i, j] = M_e[j, i] = gaussian_quadrature_triangle(func_r_NiNj, vertices, i, j)

    elif n > 0:
        M_e = np.zeros((6, 6))

        # edge-edge ブロック: m_ij = ∫∫ Ni·Nj r dzdr
        for i in range(3):
            for j in range(i, 3):
                M_e[i, j] = M_e[j, i] = gaussian_quadrature_triangle(func_r_NiNj, vertices, i, j)

        # node-node ブロック: m_ij = ∫∫ L(i-3)L(j-3)/r dzdr
        # (rE_theta を未知数としているため 1/r の重み: PDF式(64))
        for i in range(3, 6):
            for j in range(i, 6):
                M_e[i, j] = M_e[j, i] = gaussian_quadrature_triangle(
                    func_rover_LiLj, vertices, i - 3, j - 3)

        # edge-node ブロック: m_ij = 0 (PDF式(64))
        # 既に np.zeros で初期化済み

    return M_e


# ==========================================================================
# 2次要素: 要素行列 (K_e, M_e) の同時組み立て
# ==========================================================================
def assemble_element_matrices_2nd(vertices, n):
    """2次エッジ要素の要素剛性行列 K_e と質量行列 M_e を同時に組み立てる。

    7点ガウス積分 (Dunavant, 5次精度) を使用。

    DOF の対応 (ローカル):
        0-2:  CT/LN (Whitney) エッジ関数 N1-N3  → 向き符号あり
        3-5:  LT/LN           エッジ関数 N4-N6  → 向き符号なし (curl=0)
        6-7:  face            内部関数 N7-N8    → 向き符号なし
        8-13: node            節点関数 G1-G6    → 向き符号なし (n>0 のみ)

    行列の構造 (PDF 式(70)):
        edge-edge (8×8):
            K_ij = ∫∫ (∇×Ni)(∇×Nj)·r dzdr  +  n² ∫∫ Ni·Nj/r dzdr
            M_ij = ∫∫ Ni·Nj·r dzdr
        edge-node (8×6, n>0):
            K_ij = -n ∫∫ Ni·∇G(j)/r dzdr,  M_ij = 0
        node-node (6×6, n>0):
            K_ij = ∫∫ ∇Gi·∇Gj/r dzdr
            M_ij = ∫∫ Gi·Gj/r dzdr

    Args:
        vertices (numpy.ndarray): コーナー頂点座標 (3×2) [z, r]
        n (int): 方位角モード次数

    Returns:
        tuple: (K_e, M_e)  — それぞれ (8×8) または (14×14) の ndarray
    """
    size = 8 if n == 0 else 14
    K_e = np.zeros((size, size))
    M_e = np.zeros((size, size))

    area = calculate_triangle_area(vertices)
    if np.isclose(area, 0.0):
        return K_e, M_e

    pts_data = integration_points_triangle[7]
    coords_L = pts_data['coords']   # (7, 3)
    weights  = pts_data['weights']  # (7,)

    # grad_L は要素内定数 — ループ外で計算
    grad_L = grad_area_coordinates(vertices)

    for pt in range(7):
        L1, L2, L3 = coords_L[pt]
        wa = weights[pt] * area   # 重み × 面積

        # 積分点の物理座標
        z  = L1*vertices[0,0] + L2*vertices[1,0] + L3*vertices[2,0]
        r  = L1*vertices[0,1] + L2*vertices[1,1] + L3*vertices[2,1]
        rv = np.array([z, r])

        # 2次エッジ形状関数とカール (各8個)
        N      = calculate_edge_shape_functions_2nd(rv, vertices)
        curl_N = calculate_curl_edge_shape_functions_2nd(rv, vertices)

        # ---- edge-edge ブロック (8×8) ----
        for i in range(8):
            for j in range(i, 8):
                # K: (∇×Ni)(∇×Nj)·r
                k_cc = curl_N[i] * curl_N[j] * r
                # K: n²·Ni·Nj/r  (n>0 のみ)
                k_mn = n * n * np.dot(N[i], N[j]) / r if n > 0 else 0.0
                k_val = wa * (k_cc + k_mn)
                K_e[i, j] += k_val
                if i != j:
                    K_e[j, i] += k_val

                # M: Ni·Nj·r
                m_val = wa * np.dot(N[i], N[j]) * r
                M_e[i, j] += m_val
                if i != j:
                    M_e[j, i] += m_val

        # ---- node ブロック (n>0 のみ) ----
        if n > 0:
            L  = np.array([L1, L2, L3])
            G     = calculate_quadratic_nodal_shape_functions(L)       # (6,)
            gradG = grad_quadratic_nodal_shape_functions(L, grad_L)    # (6,2)

            # edge-node ブロック (8×6): K_ij = -n ∫∫ Ni·∇Gj/r dzdr
            for i in range(8):
                for j in range(6):
                    k_val = wa * (-n * np.dot(N[i], gradG[j]) / r)
                    K_e[i,   8+j] += k_val
                    K_e[8+j, i  ] += k_val   # 対称性

            # node-node ブロック (6×6)
            for i in range(6):
                for j in range(i, 6):
                    # K: ∇Gi·∇Gj/r
                    k_val = wa * np.dot(gradG[i], gradG[j]) / r
                    K_e[8+i, 8+j] += k_val
                    if i != j:
                        K_e[8+j, 8+i] += k_val

                    # M: Gi·Gj/r
                    m_val = wa * G[i] * G[j] / r
                    M_e[8+i, 8+j] += m_val
                    if i != j:
                        M_e[8+j, 8+i] += m_val

    return K_e, M_e


# ==========================================================================
# 2次要素: 全体行列アセンブリ (COO形式)
# ==========================================================================
def assemble_global_matrices_2nd(simplices, vertices, edge_index_map, num_edges, n):
    """2次エッジ要素の全体剛性行列 K と質量行列 M を組み立てる (COO→CSR)。

    DOF 番号体系:
        [0,              2*num_edges):              エッジDOF
                                                      偶数 2e   → CT/LN (N1-N3 型)
                                                      奇数 2e+1 → LT/LN (N4-N6 型)
        [2*num_edges,    2*num_edges+2*num_elem):   面内DOF
                                                      2*num_edges+2k   → N7
                                                      2*num_edges+2k+1 → N8
        [2*num_edges+2*num_elem, ...):              ノードDOF (n>0 のみ)
                                                      node_offset + global_node_idx

    符号ルール:
        CT/LN (ローカル 0-2): エッジ向きに応じて ±1
        LT/LN (ローカル 3-5): 常に +1
        face  (ローカル 6-7): 常に +1
        node  (ローカル 8-13): 常に +1

    Args:
        simplices (numpy.ndarray): 要素リスト
                  1次メッシュ: (num_elem, 3)  — corner nodes のみ
                  2次メッシュ: (num_elem, 6)  — corner + midside nodes
        vertices (numpy.ndarray): 節点座標 (num_nodes, 2) [z, r]
        edge_index_map (dict): {tuple(sorted(corner_nodes)): edge_index}
        num_edges (int): 総エッジ数
        n (int): 方位角モード次数

    Returns:
        tuple: (K_global, M_global) — CSR 形式の疎行列
    """
    num_elements = len(simplices)
    num_nodes    = len(vertices)

    face_offset = 2 * num_edges
    node_offset = 2 * num_edges + 2 * num_elements
    matrix_size = node_offset + (num_nodes if n > 0 else 0)

    k_rows, k_cols, k_data = [], [], []
    m_rows, m_cols, m_data = [], [], []

    local_size = 8 if n == 0 else 14

    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        vertices_elem = vertices[corner]

        K_e, M_e = assemble_element_matrices_2nd(vertices_elem, n)

        # エッジのグローバルインデックスと向き符号
        edge_pairs  = [(corner[0], corner[1]),
                       (corner[1], corner[2]),
                       (corner[2], corner[0])]
        edge_global = [edge_index_map[tuple(sorted(ep))] for ep in edge_pairs]
        edge_signs  = [get_edge_orientation_sign(ep[0], ep[1]) for ep in edge_pairs]

        # ローカルDOF → グローバルDOF・符号の配列を構築
        g_dofs  = np.empty(local_size, dtype=int)
        g_signs = np.ones(local_size, dtype=float)

        for k in range(3):
            g_dofs[k]   = 2 * edge_global[k]        # CT/LN
            g_dofs[k+3] = 2 * edge_global[k] + 1    # LT/LN
            g_signs[k]  = edge_signs[k]              # CT/LN のみ符号あり

        g_dofs[6] = face_offset + 2 * elem_idx      # N7
        g_dofs[7] = face_offset + 2 * elem_idx + 1  # N8

        if n > 0:
            if len(simplex) < 6:
                raise ValueError(
                    "2次節点形状関数 G4-G6 (辺上中点) にはメッシュの6節点三角形が必要です。"
                    "mesh_reader で element_order=2 を指定してください。"
                )
            for k in range(6):
                g_dofs[8+k] = node_offset + simplex[k]  # G1-G6

        # COO フォーマットにアセンブル
        for i in range(local_size):
            gi = g_dofs[i]
            si = g_signs[i]
            for j in range(local_size):
                gj = g_dofs[j]
                sj = g_signs[j]
                sign = si * sj

                k_val = sign * K_e[i, j]
                m_val = sign * M_e[i, j]

                if k_val != 0.0:
                    k_rows.append(gi); k_cols.append(gj); k_data.append(k_val)
                if m_val != 0.0:
                    m_rows.append(gi); m_cols.append(gj); m_data.append(m_val)

    K_global = coo_matrix((k_data, (k_rows, k_cols)),
                          shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()
    M_global = coo_matrix((m_data, (m_rows, m_cols)),
                          shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()

    return K_global, M_global


# ==========================================================================
# 2次要素: 全体行列アセンブリ ベクトル化版 (高速・曲線要素対応)
# ==========================================================================

def _precompute_G_gradG_ref():
    """7点 Dunavant 積分点での 2次形状関数値 G と参照勾配 gradG_ref を事前計算する。

    Returns:
        G_ref      : (7, 6) — 各積分点での G1-G6 の値
        gradG_ref  : (7, 6, 2) — 参照座標系での ∂G/∂L1, ∂G/∂L2
        L_points   : (7, 3) — 積分点の面積座標
        weights    : (7,) — 積分重み
    """
    pts_data = integration_points_triangle[7]
    L_points = pts_data['coords']   # (7, 3)
    weights  = pts_data['weights']  # (7,)
    num_pts  = len(weights)

    G_ref      = np.zeros((num_pts, 6))
    gradG_ref  = np.zeros((num_pts, 6, 2))

    for p in range(num_pts):
        L1, L2, L3 = L_points[p]

        # 形状関数値
        G_ref[p, 0] = (2*L1 - 1) * L1
        G_ref[p, 1] = (2*L2 - 1) * L2
        G_ref[p, 2] = (2*L3 - 1) * L3
        G_ref[p, 3] = 4 * L1 * L2
        G_ref[p, 4] = 4 * L2 * L3
        G_ref[p, 5] = 4 * L3 * L1

        # 参照勾配 dG/dLk (L3 = 1 - L1 - L2 を考慮)
        dG_dL = np.zeros((6, 3))
        dG_dL[0, 0] = 4*L1 - 1
        dG_dL[1, 1] = 4*L2 - 1
        dG_dL[2, 2] = 4*L3 - 1
        dG_dL[3, 0], dG_dL[3, 1] = 4*L2, 4*L1
        dG_dL[4, 1], dG_dL[4, 2] = 4*L3, 4*L2
        dG_dL[5, 0], dG_dL[5, 2] = 4*L3, 4*L1

        gradG_ref[p, :, 0] = dG_dL[:, 0] - dG_dL[:, 2]  # dG/dL1
        gradG_ref[p, :, 1] = dG_dL[:, 1] - dG_dL[:, 2]  # dG/dL2

    return G_ref, gradG_ref, L_points, weights


def assemble_element_matrices_2nd_batch(P, n):
    """2次要素の要素行列を全要素一括で計算する（ベクトル化・曲線要素対応）。

    等パラメトリック写像によるヤコビアンを使用するため、曲線要素にも対応。
    直線要素の場合はヤコビアンが各要素内で定数になる（精度は同等）。

    Args:
        P (numpy.ndarray): 全要素の節点座標 (E, 6, 2) [z, r]
                           各要素の全6節点（コーナー3 + 辺中点3）の座標
        n (int): 方位角モード次数

    Returns:
        tuple: (K_e_all, M_e_all) — それぞれ (E, D, D) の ndarray
               D = 8 (n=0), D = 14 (n>0)
    """
    E = len(P)
    D = 8 if n == 0 else 14

    K_e_all = np.zeros((E, D, D))
    M_e_all = np.zeros((E, D, D))

    # 7点積分点での形状関数を事前計算
    G_ref, gradG_ref, L_points, weights = _precompute_G_gradG_ref()

    for p in range(7):
        L1p, L2p, L3p = L_points[p]
        wp = weights[p]

        # --- ヤコビアン計算 (等パラメトリック写像) ---
        # P: (E, 6, 2),  gradG_ref[p]: (6, 2)
        # Jac[e, α, β] = Σ_i P[e, i, α] * gradG_ref[p, i, β]
        # shape: (E, 2, 6) @ (6, 2) → (E, 2, 2)
        Jac = np.einsum('eij,jk->eik', P.transpose(0, 2, 1), gradG_ref[p])

        detJ     = Jac[:, 0, 0] * Jac[:, 1, 1] - Jac[:, 0, 1] * Jac[:, 1, 0]
        abs_detJ = np.abs(detJ)

        # 逆ヤコビアン J^{-1}
        invJ = np.zeros_like(Jac)
        invJ[:, 0, 0] =  Jac[:, 1, 1] / detJ
        invJ[:, 0, 1] = -Jac[:, 0, 1] / detJ
        invJ[:, 1, 0] = -Jac[:, 1, 0] / detJ
        invJ[:, 1, 1] =  Jac[:, 0, 0] / detJ

        # 面積座標の物理勾配 (E, 3, 2)
        # invJ の行が ∇L1, ∇L2 に対応；∇L3 = -∇L1 - ∇L2
        gL1 = invJ[:, 0, :]                     # (E, 2)
        gL2 = invJ[:, 1, :]                     # (E, 2)
        gL3 = -gL1 - gL2                        # (E, 2)

        # 積分点の物理r座標 (E,)
        r_p = P[:, :, 1] @ G_ref[p]            # (E,)

        # 体積要素
        # 体積要素: |det J| は (L1,L2) 参照三角形上の積分で面積要素 dL1 dL2 を与える。
        # 参照三角形 (面積 0.5) 上で Dunavant 重みは Σw=1 に正規化されているため、
        # 物理領域の積分 ∫f dA = ∫f |detJ| dL1dL2 ≈ 0.5 * Σ w_i f(L_i) * |detJ(L_i)|
        # = Σ w_i * f(L_i) * (|detJ|/2) となる。
        # よって既存の scalar 実装 (wa = w * area = w * |detJ|/2) と整合させるため 0.5 を掛ける。
        dV = abs_detJ * wp * 0.5                # (E,)

        # --- ベクトルエッジ形状関数 N1-N8 の構築 (E, 8, 2) ---
        # CT/LN: N1=L1*gL2-L2*gL1, N2=L2*gL3-L3*gL2, N3=L3*gL1-L1*gL3
        # LT/LN: N4=L1*gL2+L2*gL1, N5=L2*gL3+L3*gL2, N6=L3*gL1+L1*gL3
        # face:  N7=L3*(L1*gL2-L2*gL1), N8=L1*(L2*gL3-L3*gL2)
        N = np.zeros((E, 8, 2))
        N[:, 0] = L1p * gL2 - L2p * gL1        # N1 (CT/LN, 辺1-2)
        N[:, 1] = L2p * gL3 - L3p * gL2        # N2 (CT/LN, 辺2-3)
        N[:, 2] = L3p * gL1 - L1p * gL3        # N3 (CT/LN, 辺3-1)
        N[:, 3] = L1p * gL2 + L2p * gL1        # N4 (LT/LN, 辺1-2)
        N[:, 4] = L2p * gL3 + L3p * gL2        # N5 (LT/LN, 辺2-3)
        N[:, 5] = L3p * gL1 + L1p * gL3        # N6 (LT/LN, 辺3-1)
        N[:, 6] = L3p * (L1p * gL2 - L2p * gL1)  # N7 (face, F3)
        N[:, 7] = L1p * (L2p * gL3 - L3p * gL2)  # N8 (face, F1)

        # --- カール ∇×N の計算 (E, 8) ---
        # cross2D(a, b) = a_z * b_r - a_r * b_z  (2D 回転)
        # 符号注意: z→index0, r→index1
        def cross2D(a, b):
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        c12 = cross2D(gL1, gL2)   # (E,)  = 1/(2Ae) for straight elements
        c23 = cross2D(gL2, gL3)
        c31 = cross2D(gL3, gL1)
        # c13 = -c31, c32 = -c23

        curl_N = np.zeros((E, 8))
        curl_N[:, 0] = 2 * c12                   # ∇×N1
        curl_N[:, 1] = 2 * c23                   # ∇×N2
        curl_N[:, 2] = 2 * c31                   # ∇×N3
        # curl_N[:, 3-5] = 0 (LT/LN)
        # ∇×N7 = L1*cross(gL3,gL2) + L2*cross(gL1,gL3) + 2*L3*cross(gL1,gL2)
        curl_N[:, 6] = (L1p * (-c23) + L2p * (-c31) + 2 * L3p * c12)
        # ∇×N8 = L2*cross(gL1,gL3) + L3*cross(gL2,gL1) + 2*L1*cross(gL2,gL3)
        curl_N[:, 7] = (L2p * (-c31) + L3p * (-c12) + 2 * L1p * c23)

        # --- edge-edge ブロック (8×8) ---
        # K: curl_outer * r_p  + n² * N_dot / r_p
        # M: N_dot * r_p
        curl_outer = np.einsum('ei,ej->eij', curl_N, curl_N)   # (E, 8, 8)
        N_dot      = np.einsum('eik,ejk->eij', N, N)           # (E, 8, 8)
        dV_3d = dV[:, np.newaxis, np.newaxis]
        rp_3d = r_p[:, np.newaxis, np.newaxis]

        K_e_all[:, :8, :8] += (curl_outer * r_p[:, np.newaxis, np.newaxis]
                                + (n * n * N_dot / r_p[:, np.newaxis, np.newaxis]
                                   if n > 0 else 0.0)) * dV_3d
        M_e_all[:, :8, :8] += N_dot * rp_3d * dV_3d

        # --- node ブロック (n>0 のみ) ---
        if n > 0:
            # 物理座標系での節点形状関数勾配 (E, 6, 2)
            # gradG_ref[p]: (6, 2),  invJ: (E, 2, 2)
            # gradG_phys[e, i, k] = Σ_l gradG_ref[p, i, l] * invJ[e, l, k]
            gradG_phys = np.einsum('jk,ekl->ejl', gradG_ref[p], invJ)  # (E, 6, 2)

            # edge-node ブロック (8×6): K_ij = -n * Ni·∇Gj / r
            EN_dot = np.einsum('eik,ejk->eij', N, gradG_phys)  # (E, 8, 6)
            contrib = -n * EN_dot / rp_3d * dV_3d
            K_e_all[:, :8, 8:] += contrib
            K_e_all[:, 8:, :8] += contrib.transpose(0, 2, 1)   # 対称

            # node-node ブロック (6×6)
            NN_dot = np.einsum('eik,ejk->eij', gradG_phys, gradG_phys)  # (E, 6, 6)
            G_outer = np.outer(G_ref[p], G_ref[p])                       # (6, 6)

            K_e_all[:, 8:, 8:] += NN_dot / rp_3d * dV_3d
            M_e_all[:, 8:, 8:] += (G_outer[np.newaxis, :, :] / rp_3d) * dV_3d

    return K_e_all, M_e_all


def assemble_global_matrices_2nd_vectorized(simplices, vertices, edge_index_map,
                                             num_edges, n):
    """2次エッジ要素の全体行列をベクトル化で高速に組み立てる。

    等パラメトリック写像（アイソパラメトリック）を使用するため、
    曲線要素（辺中点が曲線上にある場合）にも正確に対応する。
    直線要素では判定不要で一律に処理でき、精度は同等。

    DOF番号体系・符号ルールは assemble_global_matrices_2nd と同一。

    Args:
        simplices (numpy.ndarray): 要素リスト (num_elem, 6) — corner + midside nodes
        vertices (numpy.ndarray): 節点座標 (num_nodes, 2) [z, r]
        edge_index_map (dict): {tuple(sorted(corner_nodes)): edge_index}
        num_edges (int): 総エッジ数
        n (int): 方位角モード次数

    Returns:
        tuple: (K_global, M_global) — CSR 形式の疎行列
    """
    E         = len(simplices)
    num_nodes = len(vertices)

    face_offset = 2 * num_edges
    node_offset = 2 * num_edges + 2 * E
    matrix_size = node_offset + (num_nodes if n > 0 else 0)

    D = 8 if n == 0 else 14

    # --- グローバルDOFインデックスと符号の事前計算 ---
    g_dofs_all = np.zeros((E, D), dtype=int)
    sign_all   = np.ones((E, 8), dtype=float)  # CT/LN (0-2) のみ符号あり

    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        edge_pairs = [(corner[0], corner[1]),
                      (corner[1], corner[2]),
                      (corner[2], corner[0])]

        for k, (n1, n2) in enumerate(edge_pairs):
            eg = edge_index_map[tuple(sorted((n1, n2)))]
            g_dofs_all[elem_idx, k]   = 2 * eg          # CT/LN
            g_dofs_all[elem_idx, k+3] = 2 * eg + 1      # LT/LN
            sign_all[elem_idx, k] = 1 if n1 < n2 else -1

        g_dofs_all[elem_idx, 6] = face_offset + 2 * elem_idx      # N7
        g_dofs_all[elem_idx, 7] = face_offset + 2 * elem_idx + 1  # N8

        if n > 0:
            for k in range(6):
                g_dofs_all[elem_idx, 8+k] = node_offset + simplex[k]

    # --- バッチ要素行列計算 ---
    P = vertices[simplices]   # (E, 6, 2)
    K_e_all, M_e_all = assemble_element_matrices_2nd_batch(P, n)

    # --- 符号行列の適用 (CT/LN DOF のみ) ---
    # sign_mat[e, i, j] = sign_all[e, i] * sign_all[e, j]  (8×8)
    sign_mat = np.einsum('ei,ej->eij', sign_all, sign_all)   # (E, 8, 8)
    K_e_all[:, :8, :8] *= sign_mat
    M_e_all[:, :8, :8] *= sign_mat

    if n > 0:
        # edge-node ブロック: CT/LN 行 (0-2) のみ符号あり
        K_e_all[:, :3, 8:] *= sign_all[:, :3, np.newaxis]
        K_e_all[:, 8:, :3] *= sign_all[:, np.newaxis, :3]
        # LT/LN 行 (3-5), face 行 (6-7): sign = 1 のまま

    # --- COO → CSR 疎行列アセンブリ ---
    # rows, cols: 各要素の (D×D) 成分のグローバルインデックス
    rows = np.repeat(g_dofs_all, D, axis=1).flatten()   # (E*D*D,)
    cols = np.tile(  g_dofs_all, (1, D)).flatten()       # (E*D*D,)

    K_global = coo_matrix(
        (K_e_all.flatten(), (rows, cols)),
        shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()
    M_global = coo_matrix(
        (M_e_all.flatten(), (rows, cols)),
        shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()

    return K_global, M_global


# ==========================================================================
# 1次要素: 要素行列バッチ計算 + 全体行列アセンブリ ベクトル化版 (高速)
# ==========================================================================

def assemble_element_matrices_1st_batch(P3, n):
    """1次Whitney エッジ要素の要素行列を全要素一括で計算する（ベクトル化）。

    K curl-curl 項は解析的に計算（= rc/A）。それ以外の積分項には
    7点 Dunavant 求積法 (Σw=1) を使用。

    DOF の対応 (ローカル):
        0-2:  Whitney エッジ関数 N1-N3  → 向き符号あり
        3-5:  節点関数 L1-L3           → 向き符号なし (n>0 のみ)

    行列の構造 (1次要素):
        edge-edge (3×3):
            K_ij = ∫∫ (∇×Ni)(∇×Nj)·r dzdr  [= rc/A 解析的]
                 + n² ∫∫ Ni·Nj/r dzdr         (n>0)
            M_ij = ∫∫ Ni·Nj·r dzdr
        edge-node (3×3, n>0):
            K_ij = -n ∫∫ Ni·∇L(j)/r dzdr,  M_ij = 0
        node-node (3×3, n>0):
            K_ij = ∫∫ ∇Li·∇Lj/r dzdr
            M_ij = ∫∫ Li·Lj/r dzdr

    Args:
        P3 (numpy.ndarray): 全要素のコーナー節点座標 (E, 3, 2) [z, r]
        n (int): 方位角モード次数

    Returns:
        tuple: (K_e_all, M_e_all) — それぞれ (E, D, D) の ndarray
               D = 3 (n=0), D = 6 (n>0)
    """
    E = len(P3)
    D = 3 if n == 0 else 6

    K_e_all = np.zeros((E, D, D))
    M_e_all = np.zeros((E, D, D))

    z1, r1 = P3[:, 0, 0], P3[:, 0, 1]
    z2, r2 = P3[:, 1, 0], P3[:, 1, 1]
    z3, r3 = P3[:, 2, 0], P3[:, 2, 1]

    # 符号付き2倍面積 A2 = (z2-z1)(r3-r1) - (z3-z1)(r2-r1)
    A2 = (z2 - z1) * (r3 - r1) - (z3 - z1) * (r2 - r1)   # (E,)
    A  = np.abs(A2) * 0.5                                   # (E,) 面積
    rc = (r1 + r2 + r3) / 3.0                              # (E,) 重心r座標

    # 面積座標の勾配 gL (E, 3, 2) = [∂L/∂z, ∂L/∂r]
    gL = np.zeros((E, 3, 2))
    gL[:, 0, 0] = (r2 - r3) / A2;  gL[:, 0, 1] = (z3 - z2) / A2   # ∇L1
    gL[:, 1, 0] = (r3 - r1) / A2;  gL[:, 1, 1] = (z1 - z3) / A2   # ∇L2
    gL[:, 2, 0] = (r1 - r2) / A2;  gL[:, 2, 1] = (z2 - z1) / A2   # ∇L3

    # K curl-curl 項 (解析的): ∇×Ni = 2/A2 (全辺共通) → K_ij = rc / A
    K_e_all[:, :3, :3] = (rc / A)[:, np.newaxis, np.newaxis]

    # 7点 Dunavant 求積 (残余積分項)
    pts_data = integration_points_triangle[7]
    L_points = pts_data['coords']   # (7, 3)
    weights  = pts_data['weights']  # (7,)  Σw=1

    for p in range(7):
        L1p, L2p, L3p = L_points[p]
        wp = weights[p]

        # 積分点の物理r座標 (E,)
        r_p = L1p * r1 + L2p * r2 + L3p * r3

        # 体積要素 dV = A * wp (Σwp=1, 物理領域での積分=A*Σwp*f(Lp))
        dV = A * wp   # (E,)

        # Whitney エッジ形状関数 N (E, 3, 2)
        # N1 = L1∇L2 - L2∇L1, N2 = L2∇L3 - L3∇L2, N3 = L3∇L1 - L1∇L3
        N = np.zeros((E, 3, 2))
        N[:, 0] = L1p * gL[:, 1] - L2p * gL[:, 0]   # N1
        N[:, 1] = L2p * gL[:, 2] - L3p * gL[:, 1]   # N2
        N[:, 2] = L3p * gL[:, 0] - L1p * gL[:, 2]   # N3

        N_dot  = np.einsum('eik,ejk->eij', N, N)   # (E, 3, 3)  Ni·Nj
        dV_3d  = dV[:, np.newaxis, np.newaxis]
        rp_3d  = r_p[:, np.newaxis, np.newaxis]

        # M edge-edge: ∫ Ni·Nj * r dA
        M_e_all[:, :3, :3] += N_dot * rp_3d * dV_3d

        if n > 0:
            # K edge-edge n² 項: n² ∫ Ni·Nj / r dA
            K_e_all[:, :3, :3] += (n * n) * N_dot / rp_3d * dV_3d

            # K edge-node: -n ∫ Ni·∇Lj / r dA (対称性あり)
            N_dot_gL = np.einsum('eik,ejk->eij', N, gL)   # (E, 3, 3)
            contrib  = -n * N_dot_gL / rp_3d * dV_3d
            K_e_all[:, :3, 3:] += contrib
            K_e_all[:, 3:, :3] += contrib.transpose(0, 2, 1)

            # K node-node: ∫ ∇Li·∇Lj / r dA
            gL_dot = np.einsum('eik,ejk->eij', gL, gL)   # (E, 3, 3)
            K_e_all[:, 3:, 3:] += gL_dot / rp_3d * dV_3d

            # M node-node: ∫ Li*Lj / r dA
            L_outer = np.outer([L1p, L2p, L3p], [L1p, L2p, L3p])   # (3, 3)
            M_e_all[:, 3:, 3:] += L_outer[np.newaxis, :, :] / rp_3d * dV_3d

    return K_e_all, M_e_all


def assemble_global_matrices_vectorized(simplices, vertices, edge_index_map, num_edges, n):
    """1次エッジ要素の全体行列をベクトル化で高速に組み立てる。

    DOF 番号体系:
        [0, num_edges):          エッジ DOF (CT/LN Whitney 関数)
        [num_edges, ...):        節点 DOF (rE_theta, n>0 のみ)

    符号ルール:
        エッジ DOF (ローカル 0-2): エッジ向きに応じて ±1
        節点 DOF (ローカル 3-5): 常に +1

    Args:
        simplices (numpy.ndarray): 要素リスト (num_elem, 3 or 6) — コーナー3節点を使用
        vertices (numpy.ndarray): 節点座標 (num_nodes, 2) [z, r]
        edge_index_map (dict): {tuple(sorted(corner_nodes)): edge_index}
        num_edges (int): 総エッジ数
        n (int): 方位角モード次数

    Returns:
        tuple: (K_global, M_global) — CSR 形式の疎行列
    """
    E         = len(simplices)
    num_nodes = len(vertices)
    matrix_size = num_edges + (num_nodes if n > 0 else 0)
    D = 3 if n == 0 else 6

    # --- グローバル DOF インデックスと符号の事前計算 ---
    g_dofs_all = np.zeros((E, D), dtype=int)
    sign_all   = np.ones((E, 3), dtype=float)   # エッジ DOF (0-2) のみ符号あり

    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        edge_pairs = [(corner[0], corner[1]),
                      (corner[1], corner[2]),
                      (corner[2], corner[0])]

        for k, (n1, n2) in enumerate(edge_pairs):
            eg = edge_index_map[tuple(sorted((n1, n2)))]
            g_dofs_all[elem_idx, k] = eg
            sign_all[elem_idx, k]   = 1 if n1 < n2 else -1

        if n > 0:
            for k in range(3):
                g_dofs_all[elem_idx, 3 + k] = num_edges + corner[k]

    # --- バッチ要素行列計算 ---
    P3 = vertices[simplices[:, :3]]   # (E, 3, 2)
    K_e_all, M_e_all = assemble_element_matrices_1st_batch(P3, n)

    # --- 符号行列の適用 (エッジ DOF のみ) ---
    sign_mat = np.einsum('ei,ej->eij', sign_all, sign_all)   # (E, 3, 3)
    K_e_all[:, :3, :3] *= sign_mat
    M_e_all[:, :3, :3] *= sign_mat

    if n > 0:
        # edge-node ブロック: エッジ行/列にのみ符号を適用
        K_e_all[:, :3, 3:] *= sign_all[:, :, np.newaxis]
        K_e_all[:, 3:, :3] *= sign_all[:, np.newaxis, :]
        # M edge-node = 0 のため処理不要

    # --- COO → CSR 疎行列アセンブリ ---
    rows = np.repeat(g_dofs_all, D, axis=1).flatten()   # (E*D*D,)
    cols = np.tile(  g_dofs_all, (1, D)).flatten()       # (E*D*D,)

    K_global = coo_matrix(
        (K_e_all.flatten(), (rows, cols)),
        shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()
    M_global = coo_matrix(
        (M_e_all.flatten(), (rows, cols)),
        shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()

    return K_global, M_global


# ==========================================================================
# 1次要素: 全体行列アセンブリ (COO形式, forループ版, 参照用)
# ==========================================================================
def assemble_global_matrices(simplices, vertices, edge_index_map, num_edges, n):
    """全体剛性行列Kと質量行列Mを組み立てる（COO形式→CSR変換）

    Args:
        simplices: 要素リスト (num_elements x 3)
        vertices: 節点座標 (num_nodes x 2) [z, r]
        edge_index_map: エッジインデックスマップ {tuple(sorted(nodes)): edge_index}
        num_edges: エッジ数
        n: 方位角モード次数

    Returns:
        K_global_sparse, M_global_sparse: CSR形式の疎行列
    """
    num_nodes = len(vertices)
    matrix_size = num_edges + num_nodes if n > 0 else num_edges

    # COOフォーマット用リスト
    k_rows, k_cols, k_data = [], [], []
    m_rows, m_cols, m_data = [], [], []

    for simplex in simplices:
        # 2次メッシュ (6節点) の場合もコーナー3節点のみ使用
        vertices_element = vertices[simplex[:3]]
        K_element = assemble_stiffness_matrix_element(vertices_element, n)
        M_element = assemble_mass_matrix_element(vertices_element, n)

        # エッジのグローバルインデックス
        edge_indices_global = []
        edge_vertices_local = [
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[2], simplex[0]),
        ]
        for ev in edge_vertices_local:
            edge_indices_global.append(edge_index_map[tuple(sorted(ev))])


        # エッジ-エッジ ブロック
        for i in range(3):
            for j in range(3):
                gi = edge_indices_global[i]
                gj = edge_indices_global[j]
                sign = get_edge_orientation_sign(simplex[i], simplex[(i + 1) % 3]) * \
                       get_edge_orientation_sign(simplex[j], simplex[(j + 1) % 3])

                k_val = sign * K_element[i, j]
                m_val = sign * M_element[i, j]

                if k_val != 0:
                    k_rows.append(gi); k_cols.append(gj); k_data.append(k_val)
                if m_val != 0:
                    m_rows.append(gi); m_cols.append(gj); m_data.append(m_val)

        # 高次モードの追加ブロック
        if n > 0:
            # エッジ-ノード ブロック
            for i in range(3):
                for j in range(3, 6):
                    sign = get_edge_orientation_sign(simplex[i], simplex[(i + 1) % 3])
                    gi = edge_indices_global[i]
                    gj = num_edges + simplex[j - 3]

                    k_val = sign * K_element[i, j]
                    m_val = sign * M_element[i, j]

                    if k_val != 0:
                        k_rows.append(gi); k_cols.append(gj); k_data.append(k_val)
                        k_rows.append(gj); k_cols.append(gi); k_data.append(k_val)
                    if m_val != 0:
                        m_rows.append(gi); m_cols.append(gj); m_data.append(m_val)
                        m_rows.append(gj); m_cols.append(gi); m_data.append(m_val)

            # ノード-ノード ブロック
            for i in range(3, 6):
                for j in range(3, 6):
                    gi = num_edges + simplex[i - 3]
                    gj = num_edges + simplex[j - 3]

                    k_val = K_element[i, j]
                    m_val = M_element[i, j]

                    if k_val != 0:
                        k_rows.append(gi); k_cols.append(gj); k_data.append(k_val)
                    if m_val != 0:
                        m_rows.append(gi); m_cols.append(gj); m_data.append(m_val)

    K_global = coo_matrix((k_data, (k_rows, k_cols)),
                          shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()
    M_global = coo_matrix((m_data, (m_rows, m_cols)),
                          shape=(matrix_size, matrix_size), dtype=np.float64).tocsr()

    return K_global, M_global
