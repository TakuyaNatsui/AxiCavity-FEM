import numpy as np

def calculate_triangle_area_double(vertices):
    """三角形の符号付き面積の2倍を計算する関数

    Args:
        vertices (numpy.ndarray): 三角形の頂点座標 (3x2 array)

    Returns:
        float: 符号付き面積の2倍
    """
    r1, r2, r3 = vertices
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3
    signed_area_double = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    return signed_area_double

def calculate_area_coordinates(r, vertices):
    """重心座標 (面積座標) を計算する関数

    Args:
        r (numpy.ndarray): 座標 (1x2 array)
        vertices (numpy.ndarray): 三角形の頂点座標 (3x2 array)

    Returns:
        numpy.ndarray: 重心座標 (L1, L2, L3)
    """
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
    """重心座標の勾配ベクトルを計算する関数

    Args:
        vertices (numpy.ndarray): 三角形の頂点座標 (3x2 array)

    Returns:
        tuple: 重心座標 L1, L2, L3 の勾配ベクトル (それぞれ 1x2 array)
    """
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
    """
    三角形2次要素の節点形状関数 (G1-G6) を計算する。
    資料 Equation (15) に基づく。

    Args:
        L (numpy.ndarray): 重心座標 (L1, L2, L3)

    Returns:
        numpy.ndarray: 形状関数 G1-G6 (サイズ 6)
    """
    L1, L2, L3 = L
    G = np.zeros(6)
    
    # 頂点節点 (i <= 3)
    G[0] = (2*L1 - 1) * L1
    G[1] = (2*L2 - 1) * L2
    G[2] = (2*L3 - 1) * L3
    
    # 辺上中点節点 (i > 3)
    # 節点4: 1-2間, 節点5: 2-3間, 節点6: 3-1間 (資料の図2に準拠)
    G[3] = 4 * L1 * L2
    G[4] = 4 * L2 * L3
    G[5] = 4 * L3 * L1
    
    return G

def grad_quadratic_nodal_shape_functions(L, grad_L):
    """
    三角形2次要素の節点形状関数の勾配 (grad G1 - grad G6) を計算する。
    資料 Equation (16) に基づく。

    Args:
        L (numpy.ndarray): 重心座標 (L1, L2, L3)
        grad_L (tuple): 重心座標の勾配ベクトル (grad_L1, grad_L2, grad_L3)

    Returns:
        numpy.ndarray: 勾配ベクトル grad G1-G6 (サイズ 6x2)
    """
    L1, L2, L3 = L
    gL1, gL2, gL3 = grad_L
    gradG = np.zeros((6, 2))
    
    # 頂点節点 (i <= 3)
    gradG[0] = (4*L1 - 1) * gL1
    gradG[1] = (4*L2 - 1) * gL2
    gradG[2] = (4*L3 - 1) * gL3
    
    # 辺上中点節点 (i > 3)
    gradG[3] = 4 * (L2 * gL1 + L1 * gL2)
    gradG[4] = 4 * (L3 * gL2 + L2 * gL3)
    gradG[5] = 4 * (L1 * gL3 + L3 * gL1)
    
    return gradG


def calculate_edge_shape_functions(r, vertices):
    """
    電磁場解析におけるエッジ要素（vector basis function）の形状関数を計算します。
    形状関数は、三角形要素内の任意の点 r におけるエッジに沿ったベクトル場の基底関数となります。

    Args:
        r (numpy.ndarray): 座標 (1x2 array)
        vertices (numpy.ndarray): 三角形の頂点座標 (3x2 array)

    Returns:
        tuple: 形状関数 N1, N2, N3 (それぞれ 1x2 array)
    """
    L1, L2, L3 = calculate_area_coordinates(r, vertices) 
    grad_L1, grad_L2, grad_L3 = grad_area_coordinates(vertices) 
    N1 = L1 * grad_L2 - L2 * grad_L1 
    N2 = L2 * grad_L3 - L3 * grad_L2 
    N3 = L3 * grad_L1 - L1 * grad_L3 

    return N1, N2, N3


def calculate_edge_shape_functions_2nd(r, vertices):
    """
    2次要素のエッジ形状関数 N1-N8 (Webb の階層的ベクトル基底) を計算する。

    DOF の対応:
        N1-N3: 1次 Whitney 関数 (CT/LN type) — 辺 (1-2), (2-3), (3-1) 各1つ
        N4-N6: 2次エッジ関数 (LT/LN type) — 同じ辺の2番目 N_{i+3} = Li∇Lj + Lj∇Li
        N7-N8: 面内 (face/interior) DOF — 要素に固有

    辺の番号付け (PDF の図に準拠):
        辺1: ノード1-2,  辺2: ノード2-3,  辺3: ノード3-1

    符号ルール:
        N1-N3 (CT/LN): 反対称形 → エッジ向きに応じて符号反転が必要
        N4-N6 (LT/LN): 対称形   → 符号反転不要 (curl = 0)
        N7-N8 (face):  要素固有  → 符号反転不要

    Args:
        r (numpy.ndarray): 積分点の座標 [z, r]
        vertices (numpy.ndarray): 三角形コーナー頂点座標 (3x2) [z, r]

    Returns:
        list: 形状関数 [N1, N2, ..., N8]
              各要素は 2次元ベクトル numpy.ndarray([Nz, Nr])
    """
    L = calculate_area_coordinates(r, vertices)
    gL = grad_area_coordinates(vertices)
    L1, L2, L3 = L
    gL1, gL2, gL3 = gL

    # --- CT/LN 型 (1次 Whitney 関数と同じ) ---
    N1 = L1 * gL2 - L2 * gL1   # 辺 1-2
    N2 = L2 * gL3 - L3 * gL2   # 辺 2-3
    N3 = L3 * gL1 - L1 * gL3   # 辺 3-1

    # --- LT/LN 型 (2次追加エッジ関数, curl = 0) ---
    N4 = L1 * gL2 + L2 * gL1   # 辺 1-2 の2番目
    N5 = L2 * gL3 + L3 * gL2   # 辺 2-3 の2番目
    N6 = L3 * gL1 + L1 * gL3   # 辺 3-1 の2番目

    # --- 面内 (face/interior) 関数 ---
    # 3候補 Fi = Li(Lj∇Lk - Lk∇Lj) のうち縮退性から2つを選択 (PDFの定義に従う)
    # ∇×N7 = (3L3-1)/(2Ae), ∇×N8 = (3L1-1)/(2Ae)
    N7 = L3 * (L1 * gL2 - L2 * gL1)   # F3
    N8 = L1 * (L2 * gL3 - L3 * gL2)   # F1

    return [N1, N2, N3, N4, N5, N6, N7, N8]


def calculate_curl_edge_shape_functions_2nd(r, vertices):
    """
    2次要素のエッジ形状関数のカール (∇×N) を計算する。

    2D (z, r) 平面での回転: ∇×N = ∂Nr/∂z - ∂Nz/∂r

    解析的な値 (A2 = 2*Ae を使用):
        N1-N3 (CT/LN): ∇×Ni = 1/Ae = 2/A2  (要素内一様, 定数)
        N4-N6 (LT/LN): ∇×Ni = 0             (対称形のためカールはゼロ)
        N7 (face, F3):  ∇×N7 = (3L3-1)/(2Ae) = (3L3-1)/A2
        N8 (face, F1):  ∇×N8 = (3L1-1)/(2Ae) = (3L1-1)/A2

    導出:
        cross2D(gLi, gLj) = 1/A2 (i,j が巡回順), -1/A2 (逆順)
        N1 = L1∇L2 - L2∇L1 → ∇×N1 = 2*cross2D(gL1,gL2) = 2/A2
        N4 = L1∇L2 + L2∇L1 → ∇×N4 = 0
        N7 = L3(L1∇L2 - L2∇L1) →
             ∇×N7 = L1*cross2D(gL3,gL2) - L2*cross2D(gL3,gL1) + 2L3*cross2D(gL1,gL2)
                  = (-L1 - L2 + 2L3)/A2 = (3L3-1)/A2
        N8 = L1(L2∇L3 - L3∇L2) →
             ∇×N8 = (-L2 - L3 + 2L1)/A2 = (3L1-1)/A2

    Args:
        r (numpy.ndarray): 積分点の座標 [z, r]
        vertices (numpy.ndarray): 三角形コーナー頂点座標 (3x2)

    Returns:
        list: カール [curlN1, curlN2, ..., curlN8] (各要素はスカラー)
    """
    L = calculate_area_coordinates(r, vertices)
    L1, L2, L3 = L
    A2 = calculate_triangle_area_double(vertices)   # 符号付き 2*Ae

    # N1-N3 (CT/LN): ∇×(Li∇Lj - Lj∇Li) = 2/A2 = 1/Ae (要素内定数)
    curl_1st = 2.0 / A2
    curlN1 = curl_1st
    curlN2 = curl_1st
    curlN3 = curl_1st

    # N4-N6 (LT/LN): ∇×(Li∇Lj + Lj∇Li) = 0
    curlN4 = 0.0
    curlN5 = 0.0
    curlN6 = 0.0

    # N7 = L3(L1∇L2 - L2∇L1): ∇×N7 = (3L3-1)/A2
    curlN7 = (3.0 * L3 - 1.0) / A2

    # N8 = L1(L2∇L3 - L3∇L2): ∇×N8 = (3L1-1)/A2
    curlN8 = (3.0 * L1 - 1.0) / A2

    return [curlN1, curlN2, curlN3, curlN4, curlN5, curlN6, curlN7, curlN8]
