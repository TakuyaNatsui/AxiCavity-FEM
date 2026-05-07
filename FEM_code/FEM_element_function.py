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

def calculate_boundary_integral_quadratic(edge_nodes_coords, edge_values):
    """
    2次エッジ要素（3節点）上での f(l) = H^2 * r の積分を計算する。
    
    Args:
        edge_nodes_coords (numpy.ndarray): 3つの節点の座標 (3x2: [z, r])
        edge_values (numpy.ndarray): 3つの節点における物理量の値 (H_theta)
        
    Returns:
        float: 積分値 \int H^2 * r dl
    """
    p1, p2, p3 = edge_nodes_coords # 始点, 終点, 中点
    v1, v2, v3 = edge_values
    
    # エッジの長さ (直線近似)
    L_total = np.linalg.norm(p2 - p1)
    
    # 積分点（無次元座標 xi: -1 to 1）
    # 3点ガウス求積
    xi = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    wi = np.array([5/9, 8/9, 5/9])
    
    integral = 0.0
    for i in range(len(xi)):
        x = xi[i]
        # 2次形状関数 (1D)
        n1 = 0.5 * x * (x - 1)
        n2 = 0.5 * x * (x + 1)
        n3 = 1 - x**2
        
        # 補間された H と r
        h_interp = v1*n1 + v2*n2 + v3*n3
        r_interp = p1[1]*n1 + p2[1]*n2 + p3[1]*n3
        
        integral += wi[i] * (h_interp**2 * r_interp)
        
    return integral * (L_total / 2.0)
