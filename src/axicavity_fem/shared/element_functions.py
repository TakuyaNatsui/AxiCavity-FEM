"""三角形要素の面積座標・形状関数（節点形状関数 + Whitney/Webb 階層エッジ基底）.

ver1 ``FEM_HOM_code/FEM_element_function.py`` を移植したもの。
TM0 / HOM の両ソルバから共通に使用する。

公開関数:
    calculate_triangle_area_double(vertices)
    calculate_area_coordinates(r, vertices)
    grad_area_coordinates(vertices)
    calculate_quadratic_nodal_shape_functions(L)
    grad_quadratic_nodal_shape_functions(L, grad_L)
    calculate_edge_shape_functions(r, vertices)
    calculate_edge_shape_functions_2nd(r, vertices)
    calculate_curl_edge_shape_functions_2nd(r, vertices)
"""

import numpy as np


def calculate_triangle_area_double(vertices):
    """三角形の符号付き面積の2倍 (2 * Ae) を返す.

    Args:
        vertices (numpy.ndarray): 三角形の頂点座標 (3 x 2 array)

    Returns:
        float: 符号付き 2*Ae（反時計回りで正）
    """
    r1, r2, r3 = vertices
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)


def calculate_area_coordinates(r, vertices):
    """点 r における三角形の面積座標 (L1, L2, L3) を返す."""
    r1, r2, r3 = vertices
    x, y = r
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3
    A2 = calculate_triangle_area_double(vertices)
    L1 = ((x2 * y3 - x3 * y2) + (y2 - y3) * x + (x3 - x2) * y) / A2
    L2 = ((x3 * y1 - x1 * y3) + (y3 - y1) * x + (x1 - x3) * y) / A2
    L3 = ((x1 * y2 - x2 * y1) + (y1 - y2) * x + (x2 - x1) * y) / A2
    return np.array([L1, L2, L3])


def grad_area_coordinates(vertices):
    """面積座標 L1, L2, L3 の勾配ベクトルを返す (要素内で定数)."""
    r1, r2, r3 = vertices
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3
    A2 = calculate_triangle_area_double(vertices)
    grad_L1 = np.array([(y2 - y3) / A2, (x3 - x2) / A2])
    grad_L2 = np.array([(y3 - y1) / A2, (x1 - x3) / A2])
    grad_L3 = np.array([(y1 - y2) / A2, (x2 - x1) / A2])
    return grad_L1, grad_L2, grad_L3


def calculate_quadratic_nodal_shape_functions(L):
    """三角形2次要素の節点形状関数 G1-G6 を返す.

    節点番号: G1-G3 が頂点、G4=辺(1-2)中点、G5=辺(2-3)中点、G6=辺(3-1)中点
    """
    L1, L2, L3 = L
    G = np.zeros(6)
    G[0] = (2 * L1 - 1) * L1
    G[1] = (2 * L2 - 1) * L2
    G[2] = (2 * L3 - 1) * L3
    G[3] = 4 * L1 * L2
    G[4] = 4 * L2 * L3
    G[5] = 4 * L3 * L1
    return G


def grad_quadratic_nodal_shape_functions(L, grad_L):
    """三角形2次要素の節点形状関数の勾配 grad(G1)-grad(G6) を返す."""
    L1, L2, L3 = L
    gL1, gL2, gL3 = grad_L
    gradG = np.zeros((6, 2))
    gradG[0] = (4 * L1 - 1) * gL1
    gradG[1] = (4 * L2 - 1) * gL2
    gradG[2] = (4 * L3 - 1) * gL3
    gradG[3] = 4 * (L2 * gL1 + L1 * gL2)
    gradG[4] = 4 * (L3 * gL2 + L2 * gL3)
    gradG[5] = 4 * (L1 * gL3 + L3 * gL1)
    return gradG


def calculate_edge_shape_functions(r, vertices):
    """1次 Whitney エッジ形状関数 N1, N2, N3 を返す.

    辺の順序: N1 = 辺(1-2), N2 = 辺(2-3), N3 = 辺(3-1)
    """
    L1, L2, L3 = calculate_area_coordinates(r, vertices)
    grad_L1, grad_L2, grad_L3 = grad_area_coordinates(vertices)
    N1 = L1 * grad_L2 - L2 * grad_L1
    N2 = L2 * grad_L3 - L3 * grad_L2
    N3 = L3 * grad_L1 - L1 * grad_L3
    return N1, N2, N3


def calculate_edge_shape_functions_2nd(r, vertices):
    """2次要素のエッジ形状関数 N1-N8 (Webb の階層的ベクトル基底) を返す.

    DOF の対応:
        N1-N3: CT/LN 型（1次 Whitney と同じ）— 辺の符号反転が必要
        N4-N6: LT/LN 型（2次追加、curl=0）— 対称形のため符号反転不要
        N7-N8: face/interior DOF — 要素固有、符号反転不要

    Returns:
        list[np.ndarray]: 各要素は 2D ベクトル [Nz, Nr]
    """
    L = calculate_area_coordinates(r, vertices)
    gL = grad_area_coordinates(vertices)
    L1, L2, L3 = L
    gL1, gL2, gL3 = gL

    # CT/LN 型 (1次 Whitney 関数と同じ)
    N1 = L1 * gL2 - L2 * gL1   # 辺 1-2
    N2 = L2 * gL3 - L3 * gL2   # 辺 2-3
    N3 = L3 * gL1 - L1 * gL3   # 辺 3-1

    # LT/LN 型 (2次追加エッジ関数, curl = 0)
    N4 = L1 * gL2 + L2 * gL1
    N5 = L2 * gL3 + L3 * gL2
    N6 = L3 * gL1 + L1 * gL3

    # face/interior 関数 (PDF の定義に従い F3, F1 を選択)
    N7 = L3 * (L1 * gL2 - L2 * gL1)
    N8 = L1 * (L2 * gL3 - L3 * gL2)

    return [N1, N2, N3, N4, N5, N6, N7, N8]


def calculate_curl_edge_shape_functions_2nd(r, vertices):
    """2次要素エッジ形状関数のカール ∇×N を返す.

    2D (z, r) 平面のスカラー curl. A2 = 2*Ae を使用：
        curl(N1-N3) = 1/Ae = 2/A2 (要素内定数)
        curl(N4-N6) = 0 (LT/LN は対称形のため)
        curl(N7)   = (3L3 - 1) / A2
        curl(N8)   = (3L1 - 1) / A2

    Returns:
        list[float]: スカラー curl 値 8 個
    """
    L = calculate_area_coordinates(r, vertices)
    L1, L2, L3 = L
    A2 = calculate_triangle_area_double(vertices)

    curl_1st = 2.0 / A2
    return [
        curl_1st,                       # N1
        curl_1st,                       # N2
        curl_1st,                       # N3
        0.0,                            # N4
        0.0,                            # N5
        0.0,                            # N6
        (3.0 * L3 - 1.0) / A2,          # N7
        (3.0 * L1 - 1.0) / A2,          # N8
    ]
