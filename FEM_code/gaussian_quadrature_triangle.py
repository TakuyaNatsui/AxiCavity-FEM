import numpy as np

# --- 7点積分スキーム用の定数計算（5次精度） ---
# 解析的に正確な値を用いることで、浮動小数点の丸め誤差を防ぎます
_sqrt15 = np.sqrt(15)
_a = (6 - _sqrt15) / 21
_b = (6 + _sqrt15) / 21
_wa = (155 - _sqrt15) / 1200
_wb = (155 + _sqrt15) / 1200

# --- 積分点と重みの定義 ---
integration_points_triangle = {
    1: { # 1次精度
        'coords': np.array([[1/3, 1/3, 1/3]]),
        'weights': np.array([1.0])
    },
    3: { # 2次精度
        'coords': np.array([
            [1/2, 1/2, 0.0],
            [0.0, 1/2, 1/2],[1/2, 0.0, 1/2]
        ]),
        'weights': np.array([1/3, 1/3, 1/3])
    },
    4: { # 3次精度
        'coords': np.array([[1/3, 1/3, 1/3],
            [3/5, 1/5, 1/5], # 0.6 を分数で厳密に表現
            [1/5, 3/5, 1/5],
            [1/5, 1/5, 3/5]
        ]),
        'weights': np.array([-27/48, 25/48, 25/48, 25/48])
    },
    7: { # 5次精度 (Cowperの公式など)
         # 非線形問題や高次要素（2次要素など）で必要になる高精度スキーム
        'coords': np.array([[1/3, 1/3, 1/3],
            [_a, _a, 1 - 2*_a],[_a, 1 - 2*_a, _a],[1 - 2*_a, _a, _a],
            [_b, _b, 1 - 2*_b],[_b, 1 - 2*_b, _b],[1 - 2*_b, _b, _b]
        ]),
        'weights': np.array([
            9/40,  # 重心
            _wa, _wa, _wa,
            _wb, _wb, _wb
        ])
    }
}

def calculate_triangle_area(nodes):
    """3つの頂点座標から三角形の面積を計算する"""
    p1, p2, p3 = nodes
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return area

# 引数に *args, **kwargs を追加し、どんな追加パラメータ(ie, je等)にも対応可能に改良
def gaussian_quadrature_triangle(func, vertices, n_points=4, *args, **kwargs):
    """
    三角形要素上の関数の積分をガウス求積法で計算する。
    """
    if n_points not in integration_points_triangle:
        raise ValueError(f"未定義の積分点数: {n_points}. "
                         f"利用可能な点数: {list(integration_points_triangle.keys())}")

    points_data = integration_points_triangle[n_points]
    coords_L = points_data['coords']
    weights = points_data['weights']

    area = calculate_triangle_area(vertices[:3])
    if np.isclose(area, 0.0):
        return 0.0

    integral_value = 0.0

    # 積分点のループ計算
    for i in range(n_points):
        L1i, L2i, L3i = coords_L[i]
        wi = weights[i]

        # *args, **kwargs で ie, je 等の任意の変数を関数に渡せるように修正
        #g_val = func(L1i, L2i, L3i, *args, **kwargs)
        g_val = func(L1i, L2i, L3i, vertices, *args, **kwargs)

        integral_value += wi * g_val

    return area * integral_value

# --- 使用例 ---
if __name__ == '__main__':
    nodes_ex1 = [(0, 0), (1, 0), (0, 1)] # 面積 0.5
    area_exact = 0.5

    # 例1: 定数の積分
    def func_const(L1, L2, L3): return 1.0
    print(f"例1: f = 1 の積分 (面積理論値={area_exact})")
    for n_p in [1, 3, 4, 7]:
        integral = gaussian_quadrature_triangle(func_const, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - area_exact):.2e})")

    # 例2: L1 (1次式) の積分
    def func_L1(L1, L2, L3): return L1
    integral_exact_L1 = area_exact / 3.0
    print(f"\n例2: f = L1 の積分 (理論値={integral_exact_L1:.8f})")
    for n_p in [1, 3, 4, 7]:
        integral = gaussian_quadrature_triangle(func_L1, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_L1):.2e})")

    # 例3: L1^2 * L2 * L3 (4次式) の積分
    # 理論値: ディリクレの積分公式より A / 180
    def func_L1_sq_L2_L3(L1, L2, L3): return (L1**2) * L2 * L3
    integral_exact_ex3 = area_exact / 180.0
    print(f"\n例3: f = L1^2 * L2 * L3 (4次式) の積分 (理論値={integral_exact_ex3:.8f})")
    for n_p in[1, 3, 4, 7]:
        integral = gaussian_quadrature_triangle(func_L1_sq_L2_L3, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_ex3):.2e})")