"""HOM 要素行列（エッジ+節点ハイブリッド、1次/2次, ベクトル化）.

DOF はベクトル界 E（Nédélec エッジ基底）+ 方位角成分 rE_theta（節点基底, n>0）。
ver1 ``FEM_HOM_code/element_assembly.py`` のベクトル化バッチ計算
（``assemble_element_matrices_1st_batch`` / ``assemble_element_matrices_2nd_batch``）を
移植したもの。全体行列への組立は :mod:`axicavity_fem.fem_hom.assembly` が担当する。

積分は 7 点 Dunavant (degree 5) 求積（shared.quadrature の ``integration_points_triangle[7]``、
重み Σ=1 正規化）。2 次要素はアイソパラメトリック写像で曲線要素にも対応する。
"""

from __future__ import annotations

import numpy as np

from ..shared.quadrature import integration_points_triangle


def _precompute_G_gradG_ref():
    """7 点積分点での 2 次形状関数値 G と参照勾配 gradG_ref を事前計算する.

    Returns:
        (G_ref (7,6), gradG_ref (7,6,2), L_points (7,3), weights (7,))。
    """
    pts = integration_points_triangle[7]
    L_points = pts["coords"]
    weights = pts["weights"]
    num_pts = len(weights)

    G_ref = np.zeros((num_pts, 6))
    gradG_ref = np.zeros((num_pts, 6, 2))
    for p in range(num_pts):
        L1, L2, L3 = L_points[p]
        G_ref[p, 0] = (2 * L1 - 1) * L1
        G_ref[p, 1] = (2 * L2 - 1) * L2
        G_ref[p, 2] = (2 * L3 - 1) * L3
        G_ref[p, 3] = 4 * L1 * L2
        G_ref[p, 4] = 4 * L2 * L3
        G_ref[p, 5] = 4 * L3 * L1

        dG_dL = np.zeros((6, 3))
        dG_dL[0, 0] = 4 * L1 - 1
        dG_dL[1, 1] = 4 * L2 - 1
        dG_dL[2, 2] = 4 * L3 - 1
        dG_dL[3, 0], dG_dL[3, 1] = 4 * L2, 4 * L1
        dG_dL[4, 1], dG_dL[4, 2] = 4 * L3, 4 * L2
        dG_dL[5, 0], dG_dL[5, 2] = 4 * L3, 4 * L1
        gradG_ref[p, :, 0] = dG_dL[:, 0] - dG_dL[:, 2]
        gradG_ref[p, :, 1] = dG_dL[:, 1] - dG_dL[:, 2]

    return G_ref, gradG_ref, L_points, weights


def element_matrices_1st_batch(P3: np.ndarray, n: int):
    """1 次 Whitney エッジ要素の要素行列を全要素一括で計算する.

    curl-curl 項は解析的 (= rc/A)、その他は 7 点求積。

    DOF: 0-2 = Whitney エッジ N1-N3（向き符号あり）、3-5 = 節点 L1-L3（n>0）。

    Args:
        P3: 全要素のコーナー節点座標 (E, 3, 2) = [z, r]。
        n: 方位角モード次数。

    Returns:
        (K_e_all, M_e_all): 各 (E, D, D)。D = 3 (n=0) / 6 (n>0)。
    """
    E = len(P3)
    D = 3 if n == 0 else 6
    K_e_all = np.zeros((E, D, D))
    M_e_all = np.zeros((E, D, D))

    z1, r1 = P3[:, 0, 0], P3[:, 0, 1]
    z2, r2 = P3[:, 1, 0], P3[:, 1, 1]
    z3, r3 = P3[:, 2, 0], P3[:, 2, 1]

    A2 = (z2 - z1) * (r3 - r1) - (z3 - z1) * (r2 - r1)
    A = np.abs(A2) * 0.5
    rc = (r1 + r2 + r3) / 3.0

    gL = np.zeros((E, 3, 2))
    gL[:, 0, 0] = (r2 - r3) / A2;  gL[:, 0, 1] = (z3 - z2) / A2
    gL[:, 1, 0] = (r3 - r1) / A2;  gL[:, 1, 1] = (z1 - z3) / A2
    gL[:, 2, 0] = (r1 - r2) / A2;  gL[:, 2, 1] = (z2 - z1) / A2

    K_e_all[:, :3, :3] = (rc / A)[:, np.newaxis, np.newaxis]

    pts = integration_points_triangle[7]
    L_points = pts["coords"]
    weights = pts["weights"]

    for p in range(len(weights)):
        L1p, L2p, L3p = L_points[p]
        wp = weights[p]
        r_p = L1p * r1 + L2p * r2 + L3p * r3
        dV = A * wp

        N = np.zeros((E, 3, 2))
        N[:, 0] = L1p * gL[:, 1] - L2p * gL[:, 0]
        N[:, 1] = L2p * gL[:, 2] - L3p * gL[:, 1]
        N[:, 2] = L3p * gL[:, 0] - L1p * gL[:, 2]

        N_dot = np.einsum("eik,ejk->eij", N, N)
        dV_3d = dV[:, np.newaxis, np.newaxis]
        rp_3d = r_p[:, np.newaxis, np.newaxis]

        M_e_all[:, :3, :3] += N_dot * rp_3d * dV_3d

        if n > 0:
            K_e_all[:, :3, :3] += (n * n) * N_dot / rp_3d * dV_3d
            N_dot_gL = np.einsum("eik,ejk->eij", N, gL)
            contrib = -n * N_dot_gL / rp_3d * dV_3d
            K_e_all[:, :3, 3:] += contrib
            K_e_all[:, 3:, :3] += contrib.transpose(0, 2, 1)
            gL_dot = np.einsum("eik,ejk->eij", gL, gL)
            K_e_all[:, 3:, 3:] += gL_dot / rp_3d * dV_3d
            L_outer = np.outer([L1p, L2p, L3p], [L1p, L2p, L3p])
            M_e_all[:, 3:, 3:] += L_outer[np.newaxis, :, :] / rp_3d * dV_3d

    return K_e_all, M_e_all


def element_matrices_2nd_batch(P: np.ndarray, n: int):
    """2 次エッジ要素（曲線対応）の要素行列を全要素一括で計算する.

    アイソパラメトリック写像でヤコビアンを正確に評価する。

    DOF: 0-2 CT/LN, 3-5 LT/LN, 6-7 face(N7,N8), 8-13 node G1-G6（n>0）。

    Args:
        P: 全要素の節点座標 (E, 6, 2) = [z, r]（コーナー3 + 辺中点3）。
        n: 方位角モード次数。

    Returns:
        (K_e_all, M_e_all): 各 (E, D, D)。D = 8 (n=0) / 14 (n>0)。
    """
    E = len(P)
    D = 8 if n == 0 else 14
    K_e_all = np.zeros((E, D, D))
    M_e_all = np.zeros((E, D, D))

    G_ref, gradG_ref, L_points, weights = _precompute_G_gradG_ref()

    for p in range(len(weights)):
        L1p, L2p, L3p = L_points[p]
        wp = weights[p]

        Jac = np.einsum("eij,jk->eik", P.transpose(0, 2, 1), gradG_ref[p])
        detJ = Jac[:, 0, 0] * Jac[:, 1, 1] - Jac[:, 0, 1] * Jac[:, 1, 0]
        abs_detJ = np.abs(detJ)

        invJ = np.zeros_like(Jac)
        invJ[:, 0, 0] = Jac[:, 1, 1] / detJ
        invJ[:, 0, 1] = -Jac[:, 0, 1] / detJ
        invJ[:, 1, 0] = -Jac[:, 1, 0] / detJ
        invJ[:, 1, 1] = Jac[:, 0, 0] / detJ

        gL1 = invJ[:, 0, :]
        gL2 = invJ[:, 1, :]
        gL3 = -gL1 - gL2

        r_p = P[:, :, 1] @ G_ref[p]
        dV = abs_detJ * wp * 0.5

        N = np.zeros((E, 8, 2))
        N[:, 0] = L1p * gL2 - L2p * gL1
        N[:, 1] = L2p * gL3 - L3p * gL2
        N[:, 2] = L3p * gL1 - L1p * gL3
        N[:, 3] = L1p * gL2 + L2p * gL1
        N[:, 4] = L2p * gL3 + L3p * gL2
        N[:, 5] = L3p * gL1 + L1p * gL3
        N[:, 6] = L3p * (L1p * gL2 - L2p * gL1)
        N[:, 7] = L1p * (L2p * gL3 - L3p * gL2)

        def cross2D(a, b):
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        c12 = cross2D(gL1, gL2)
        c23 = cross2D(gL2, gL3)
        c31 = cross2D(gL3, gL1)

        curl_N = np.zeros((E, 8))
        curl_N[:, 0] = 2 * c12
        curl_N[:, 1] = 2 * c23
        curl_N[:, 2] = 2 * c31
        curl_N[:, 6] = L1p * (-c23) + L2p * (-c31) + 2 * L3p * c12
        curl_N[:, 7] = L2p * (-c31) + L3p * (-c12) + 2 * L1p * c23

        curl_outer = np.einsum("ei,ej->eij", curl_N, curl_N)
        N_dot = np.einsum("eik,ejk->eij", N, N)
        dV_3d = dV[:, np.newaxis, np.newaxis]
        rp_3d = r_p[:, np.newaxis, np.newaxis]

        K_e_all[:, :8, :8] += (
            curl_outer * rp_3d
            + (n * n * N_dot / rp_3d if n > 0 else 0.0)
        ) * dV_3d
        M_e_all[:, :8, :8] += N_dot * rp_3d * dV_3d

        if n > 0:
            gradG_phys = np.einsum("jk,ekl->ejl", gradG_ref[p], invJ)
            EN_dot = np.einsum("eik,ejk->eij", N, gradG_phys)
            contrib = -n * EN_dot / rp_3d * dV_3d
            K_e_all[:, :8, 8:] += contrib
            K_e_all[:, 8:, :8] += contrib.transpose(0, 2, 1)

            NN_dot = np.einsum("eik,ejk->eij", gradG_phys, gradG_phys)
            G_outer = np.outer(G_ref[p], G_ref[p])
            K_e_all[:, 8:, 8:] += NN_dot / rp_3d * dV_3d
            M_e_all[:, 8:, 8:] += (G_outer[np.newaxis, :, :] / rp_3d) * dV_3d

    return K_e_all, M_e_all
