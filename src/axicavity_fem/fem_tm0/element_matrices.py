"""TM0 要素行列 K_e, M_e（1次/2次要素, ベクトル化）.

DOF は H_phi スカラー節点。軸対称ヘルムホルツ方程式の弱形式から導かれる
剛性行列・質量行列を、全要素一括（NumPy ベクトル化）で計算する。

ver1 ``FEM_code/FEM_helmholtz_TM0_calclation.py`` の
``assemble_global_matrix_vectorized_1st`` / ``assemble_global_matrix_vectorized_2nd``
の **要素行列計算部** を分離移植したもの。全体行列への組立 (COO) は
:mod:`axicavity_fem.fem_tm0.assembly` が担当する。

ver1 との数値一致のため、係数・積分点・演算順序は元コードと厳密に同一にしている
（1 次要素の係数 2 倍は「後のエネルギー計算のため省略しない」という ver1 の方針を踏襲）。
"""

from __future__ import annotations

import numpy as np

from ..shared.quadrature import integration_points_triangle


def element_matrices_1st(nodes: np.ndarray, elements: np.ndarray,
                         n_quad: int = 7):
    """1 次三角形要素の要素行列 K_e, M_e を返す.

    Args:
        nodes: 節点座標 (N, 2). 各行は ``[z, r]``。
        elements: 要素節点インデックス (E, 3)。
        n_quad: 三角形ガウス求積の積分点数 (既定 7)。1/r 項や質量項の
            数値積分に使う。ver2.3 で既定を 7 点 (Dunavant 5 次) に変更
            （偽モード抑制のため。examples/quadrature_convergence 参照）。
            ver1 とビット一致させたいときは ``n_quad=4`` を指定する。

    Returns:
        (K_e, M_e): それぞれ (E, 3, 3) の ndarray。
    """
    N_elem = len(elements)

    P = nodes[elements]          # (E, 3, 2)
    z = P[:, :, 0]
    r = P[:, :, 1]

    # 符号付き 2A
    twoA = (z[:, 0] * (r[:, 1] - r[:, 2])
            + z[:, 1] * (r[:, 2] - r[:, 0])
            + z[:, 2] * (r[:, 0] - r[:, 1]))
    A = np.abs(twoA) / 2.0

    # 形状関数勾配 grad_L (E, 3, 2) = [dz, dr]
    b = np.zeros((N_elem, 3))
    c = np.zeros((N_elem, 3))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        b[:, i] = r[:, j] - r[:, k]
        c[:, i] = z[:, k] - z[:, j]
    sign = np.sign(twoA)[:, np.newaxis]
    grad_L = np.stack([c, b], axis=2) / (
        2.0 * A[:, np.newaxis, np.newaxis] * sign[:, :, np.newaxis])

    rc = np.mean(r, axis=1)      # 重心 r

    points_data = integration_points_triangle[n_quad]
    coords_L = points_data["coords"]   # (n_quad, 3)
    weights = points_data["weights"]   # (n_quad,)

    r_p = r @ coords_L.T               # (E, n_quad)

    K_e = np.zeros((N_elem, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_dot = (grad_L[:, i, 0] * grad_L[:, j, 0]
                        + grad_L[:, i, 1] * grad_L[:, j, 1])
            K_e[:, i, j] += 2 * A * rc * grad_dot
            K_e[:, i, j] += 2 * (A / 3.0) * (grad_L[:, i, 0] + grad_L[:, j, 0])

            term2 = np.zeros(N_elem)
            for p in range(len(weights)):
                term2 += weights[p] * (coords_L[p, i] * coords_L[p, j] / r_p[:, p])
            K_e[:, i, j] += 2 * A * term2

    M_e = np.zeros((N_elem, 3, 3))
    for i in range(3):
        for j in range(3):
            term_m = np.zeros(N_elem)
            for p in range(len(weights)):
                term_m += weights[p] * (r_p[:, p] * coords_L[p, i] * coords_L[p, j])
            M_e[:, i, j] += 2 * A * term_m

    return K_e, M_e


def element_matrices_2nd(nodes: np.ndarray, elements: np.ndarray,
                         n_quad: int = 7):
    """2 次三角形要素（曲線要素対応）の要素行列 K_e, M_e を返す.

    アイソパラメトリック写像で全要素を一括処理する（曲線エッジのヤコビアンも含む）。

    Args:
        nodes: 節点座標 (N, 2). 各行は ``[z, r]``。
        elements: 要素節点インデックス (E, 6)。中間節点を含む。
        n_quad: 三角形ガウス求積の積分点数 (既定 7)。2 次要素の質量項
            r·G_i·G_j は 5 次多項式のため 4 点則 (3 次精度) では過小積分に
            なり、質量行列が数値的に不定化して偽モードの原因になる。
            ver2.3 で既定を 7 点則 (Dunavant 5 次, 厳密積分) に変更し偽モードを
            抑制した (examples/quadrature_convergence 参照)。ver1 とビット一致
            させたいときは ``n_quad=4`` を指定する。

    Returns:
        (K_e, M_e): それぞれ (E, 6, 6) の ndarray。
    """
    N_elem = len(elements)

    P = nodes[elements]          # (E, 6, 2), P[:, i, 0]=z, P[:, i, 1]=r

    n_points = n_quad
    points_data = integration_points_triangle[n_points]
    L_points = points_data["coords"]   # (P, 3)
    weights = points_data["weights"]   # (P,)
    num_pts = len(weights)

    # 各積分点での節点形状関数 G と基準座標系勾配 gradG_ref
    G = np.zeros((num_pts, 6))
    gradG_ref = np.zeros((num_pts, 6, 2))
    for p in range(num_pts):
        L1, L2, L3 = L_points[p]
        G[p, 0] = (2 * L1 - 1) * L1
        G[p, 1] = (2 * L2 - 1) * L2
        G[p, 2] = (2 * L3 - 1) * L3
        G[p, 3] = 4 * L1 * L2
        G[p, 4] = 4 * L2 * L3
        G[p, 5] = 4 * L3 * L1

        dG_dL = np.zeros((6, 3))
        dG_dL[0, 0] = 4 * L1 - 1
        dG_dL[1, 1] = 4 * L2 - 1
        dG_dL[2, 2] = 4 * L3 - 1
        dG_dL[3, 0], dG_dL[3, 1] = 4 * L2, 4 * L1
        dG_dL[4, 1], dG_dL[4, 2] = 4 * L3, 4 * L2
        dG_dL[5, 0], dG_dL[5, 2] = 4 * L3, 4 * L1
        gradG_ref[p, :, 0] = dG_dL[:, 0] - dG_dL[:, 2]   # dG/dL1
        gradG_ref[p, :, 1] = dG_dL[:, 1] - dG_dL[:, 2]   # dG/dL2

    K_e = np.zeros((N_elem, 6, 6))
    M_e = np.zeros((N_elem, 6, 6))

    for p in range(num_pts):
        # ヤコビ行列 J = P^T @ gradG_ref (E, 2, 2)
        Jac = np.einsum("eij,jk->eik", P.transpose(0, 2, 1), gradG_ref[p])
        detJ = Jac[:, 0, 0] * Jac[:, 1, 1] - Jac[:, 0, 1] * Jac[:, 1, 0]
        abs_detJ = np.abs(detJ)

        invJ = np.zeros_like(Jac)
        invJ[:, 0, 0] = Jac[:, 1, 1] / detJ
        invJ[:, 0, 1] = -Jac[:, 0, 1] / detJ
        invJ[:, 1, 0] = -Jac[:, 1, 0] / detJ
        invJ[:, 1, 1] = Jac[:, 0, 0] / detJ

        # グローバル勾配 gradG_glob = gradG_ref @ J^-1 (E, 6, 2)
        gradG_glob = np.einsum("jk,ekl->ejl", gradG_ref[p], invJ)

        r_p = P[:, :, 1] @ G[p]    # (E,)

        GiGj = np.outer(G[p], G[p])                            # (6, 6)
        grad_dot = np.einsum("eik,ejk->eij", gradG_glob, gradG_glob)
        grad_r = gradG_glob[:, :, 1]
        term_ij = G[p][np.newaxis, :, np.newaxis] * grad_r[:, np.newaxis, :]
        term_ji = G[p][np.newaxis, np.newaxis, :] * grad_r[:, :, np.newaxis]
        grad_r_sum = term_ij + term_ji

        dV = abs_detJ * weights[p]   # (E,)

        K_e += (r_p[:, np.newaxis, np.newaxis] * grad_dot
                + GiGj[np.newaxis, :, :] / r_p[:, np.newaxis, np.newaxis]
                + grad_r_sum) * dV[:, np.newaxis, np.newaxis]
        M_e += (r_p[:, np.newaxis, np.newaxis]
                * GiGj[np.newaxis, :, :]) * dV[:, np.newaxis, np.newaxis]

    return K_e, M_e
