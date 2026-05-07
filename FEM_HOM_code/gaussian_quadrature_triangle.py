import numpy as np

# --- 積分点と重みの定義 ---
# 事前に定義済みの積分点データ (面積座標 L1, L2, L3 と 重み w)
# ここでは代表的な 1点, 3点, 4点 の例を挙げます。
# 重みは A * Σ[wi * g(Li)] の形式で使うため、Σwi = 1 となるように選ぶか、
# 文献の重み (Σwi = 0.5 など) を使う場合は最後に掛ける面積 A を調整します。
# ここでは Σwi = 1 となるように調整した値を使います（一般的な定義の一つ）。
# (注意: 文献によって重みの定義が異なる場合があるので確認が必要です)


# 7点スキームの係数 (Dunavant, 1985, degree 5)
_sqrt15 = np.sqrt(15.0)
_a  = (6.0 - _sqrt15) / 21.0          # ≈ 0.10128650732
_b  = (6.0 + _sqrt15) / 21.0          # ≈ 0.47014206411
_w1 = 9.0 / 40.0                       # = 0.225   (重心)
_w2 = (155.0 + _sqrt15) / 1200.0       # ≈ 0.13239415279  (b側3点)
_w3 = (155.0 - _sqrt15) / 1200.0       # ≈ 0.12593918054  (a側3点)

integration_points_triangle = {
    1: {
        # 1次多項式まで正確 (重心1点)
        'coords': np.array([[1/3, 1/3, 1/3]]),
        'weights': np.array([1.0])              # 重み合計 1.0
    },
    3: {
        # 2次多項式まで正確 (3点スキーム)
        'coords': np.array([
            [2/3, 1/6, 1/6],
            [1/6, 2/3, 1/6],
            [1/6, 1/6, 2/3],
        ]),
        'weights': np.array([1/3, 1/3, 1/3])   # 重み合計 1.0
    },
    4: {
        # 3次多項式まで正確 (4点スキーム)
        'coords': np.array([
            [1/3, 1/3, 1/3],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]),
        'weights': np.array([
            -27/48,  # 重心
             25/48,  # 他3点
             25/48,
             25/48,
        ])  # 重み合計: -27/48 + 3*(25/48) = 1.0
    },
    7: {
        # 5次多項式まで正確 (7点スキーム, Dunavant 1985)
        # 2次エッジ要素の被積分関数 (最高4次) に対して十分な精度
        # 重心1点 + aを含む3点 + bを含む3点 (Σw = 1.0)
        'coords': np.array([
            [1/3,    1/3,    1/3   ],   # 重心
            [_a,     _a,     1-2*_a],   # a側3点 (巡回置換)
            [_a,     1-2*_a, _a    ],
            [1-2*_a, _a,     _a    ],
            [_b,     _b,     1-2*_b],   # b側3点 (巡回置換)
            [_b,     1-2*_b, _b    ],
            [1-2*_b, _b,     _b    ],
        ]),
        'weights': np.array([_w1, _w3, _w3, _w3, _w2, _w2, _w2]),
        # Σw = _w1 + 3*_w3 + 3*_w2 = 9/40 + 3*(155-√15)/1200 + 3*(155+√15)/1200
        #    = 9/40 + 3*310/1200 = 9/40 + 930/1200 = 9/40 + 31/40 = 1.0 ✓
    },
}

def calculate_triangle_area(nodes):
    """3つの頂点座標から三角形の面積を計算する (2D)"""
    p1, p2, p3 = nodes
    # 行列式/外積を使用 (座標は(x,y)または(r,z)を想定)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # Area = 0.5 * |(x1(y2-y3) + x2(y3-y1) + x3(y1-y2))|
    # または Area = 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return area

def gaussian_quadrature_triangle(func, vertices, ie, je, n_points=4, ):
    """
    三角形要素上の関数の積分をガウス求積法で計算する。

    Args:
        func (callable): 被積分関数。引数として面積座標 (L1, L2, L3) を受け取る。
                         例: def my_func(L1, L2, L3): return L1*L1 + L2
        nodes (list or tuple): 要素の3つの頂点座標のリストまたはタプル。
                               [(x1, y1), (x2, y2), (x3, y3)] または
                               [(r1, z1), (r2, z2), (r3, z3)]
        n_points (int): 使用する積分点の数 (integration_points_triangleで定義済みのもの)。
        ie (int): 要素内のインデックスi
        je (int): 要素内のインデックスj
        n_points (int): 使用する積分点の数 (integration_points_triangleで定義済みのもの)。
    Returns:
        float: 積分の近似値。
               軸対称の場合: ∫[A] func(L1,L2,L3) * 2 * pi * r dA
               非軸対称の場合: ∫[A] func(L1,L2,L3) dA
    """
    if n_points not in integration_points_triangle:
        raise ValueError(f"未定義の積分点数: {n_points}. "
                         f"利用可能な点数: {list(integration_points_triangle.keys())}")

    points_data = integration_points_triangle[n_points]
    coords_L = points_data['coords']  # (n, 3) array of L1, L2, L3
    weights = points_data['weights'] # (n,) array of weights

    # 要素の面積を計算
    area = calculate_triangle_area(vertices)
    if np.isclose(area, 0.0):
        return 0.0 # 面積ゼロの要素は積分値ゼロ

    integral_value = 0.0
    #vertices_array = np.array(vertices) # (3, 2) array

    for i in range(n_points):
        L1i, L2i, L3i = coords_L[i]
        wi = weights[i]

        # 被積分関数 func を積分点の面積座標で評価
        g_val = func(L1i, L2i, L3i, vertices, ie, je)

        # 積分点での値に重みを掛ける (共通部分)
        term_value = wi * g_val

        # 計算された値を累積
        integral_value += term_value

    # 最後に面積 A を掛ける (∫g dA ≈ A * Σ[wi * g(Li)])
    # (∫g * 2πr dA ≈ A * Σ[wi * g(Li) * 2πri])
    return area * integral_value

# --- 使用例 ---
if __name__ == '__main__':
    # 例1: f(L1, L2, L3) = 1 を積分 (結果は面積A)
    def func_const(L1, L2, L3):
        return 1.0

    nodes_ex1 = [(0, 0), (1, 0), (0, 1)] # 面積 0.5
    area_exact = 0.5
    print(f"例1: f = 1 の積分 (面積={area_exact})")
    for n_p in [1, 3, 4]:
        integral = gaussian_quadrature_triangle(func_const, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - area_exact):.2e})")

    # 例2: f(L1, L2, L3) = L1 を積分 (結果は A/3)
    def func_L1(L1, L2, L3):
        return L1
    integral_exact_L1 = area_exact / 3.0
    print(f"\n例2: f = L1 の積分 (理論値={integral_exact_L1:.8f})")
    for n_p in [1, 3, 4]:
        integral = gaussian_quadrature_triangle(func_L1, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_L1):.2e})")

    # 例3: f(L1, L2, L3) = L1*L2 を積分 (結果は A/12)
    def func_L1L2(L1, L2, L3):
        return L1 * L2
    integral_exact_L1L2 = area_exact / 12.0
    print(f"\n例3: f = L1*L2 の積分 (理論値={integral_exact_L1L2:.8f})")
    for n_p in [1, 3, 4]: # 1点では不正確なはず
        integral = gaussian_quadrature_triangle(func_L1L2, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_L1L2):.2e})")

    # 例4: 軸対称 f = 1 (∫ 2πr dA = 2π * A * r_centroid)
    nodes_ex2 = [(1, 0), (3, 0), (2, 2)] # r1=1, r2=3, r3=2; z1=0, z2=0, z3=2
    area_ex2 = calculate_triangle_area(nodes_ex2) # 0.5 * |(3-1)(2-0)-(2-1)(0-0)| = 0.5 * 4 = 2.0
    r_centroid = (1 + 3 + 2) / 3.0 #= 6.0 / 3.0 = 2.0
    integral_exact_axisym1 = 2 * np.pi * area_ex2 * r_centroid # 2 * pi * 2.0 * 2.0 = 8 * pi
    print(f"\n例4: 軸対称 f=1 (理論値={integral_exact_axisym1:.8f})")
    for n_p in [1, 3, 4]:
        integral = gaussian_quadrature_triangle(func_const, nodes_ex2, n_p, axisymmetric=True, axis_coord_index=0)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_axisym1):.2e})")

    # 例5: 軸対称 f = L1 (∫ L1 * 2πr dA = 2π * A/12 * (2*r1 + r2 + r3))
    r1, r2, r3 = 1, 3, 2
    integral_exact_axisym2 = 2 * np.pi * area_ex2 / 12.0 * (2*r1 + r2 + r3)
    # = 2 * pi * 2.0 / 12.0 * (2*1 + 3 + 2) = 4 * pi / 12.0 * (7) = 7 * pi / 3
    print(f"\n例5: 軸対称 f=L1 (理論値={integral_exact_axisym2:.8f})")
    for n_p in [1, 3, 4]: # 2次式の積分になるので3点以上で精度が良くなるはず
        integral = gaussian_quadrature_triangle(func_L1, nodes_ex2, n_p, axisymmetric=True, axis_coord_index=0)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_axisym2):.2e})")

    # 例6: 軸対称 f = r (∫ r * 2πr dA = ∫ 2πr^2 dA)
    # r = L1*r1 + L2*r2 + L3*r3
    # 被積分関数 g = 2π * (L1*r1 + L2*r2 + L3*r3)^2
    def func_r(L1, L2, L3):
        # この関数自体は r を直接使わない。外部で 2πr が掛けられる。
        # なので、f=r を積分する場合、func としては r/ (2πr) = 1/(2π) ？
        # いや、 func が g そのものを返す設計ではない。
        # gaussian_quadrature_triangle は ∫[A] func(L) * (2πr if axisymmetric else 1) dA を計算する。
        # なので、∫ 2πr^2 dA を計算したいなら、 func(L1,L2,L3) は r = L1*r1+L2*r2+L3*r3 を返すようにすればよい。
        r1, r2, r3 = nodes_ex2[0][0], nodes_ex2[1][0], nodes_ex2[2][0]
        return L1*r1 + L2*r2 + L3*r3

    # 理論値 ∫ 2πr^2 dA = 2π ∫ (L1r1+L2r2+L3r3)^2 dA
    # = 2π * A * (r1^2/12 + r2^2/12 + r3^2/12 + r1r2/12 + r2r3/12 + r3r1/12) * 2? (公式確認)
    # ∫ L1^2 dA = A/6, ∫ L1L2 dA = A/12 (係数2Aを掛ける前の公式) -> 2A * a!b!c! / (a+b+c+2)!
    # ∫ L1^2 dA = 2A * 2! / 4! = 4A/24 = A/6
    # ∫ L1*L2 dA = 2A * 1!1! / 4! = 2A/24 = A/12
    # ∫ r^2 dA = ∫ (L1r1+L2r2+L3r3)^2 dA
    # = r1^2 ∫L1^2 + r2^2 ∫L2^2 + r3^2 ∫L3^2 + 2r1r2 ∫L1L2 + 2r2r3 ∫L2L3 + 2r3r1 ∫L3L1
    # = A/6 * (r1^2+r2^2+r3^2) + 2*A/12 * (r1r2+r2r3+r3r1)
    # = A/6 * (r1^2+r2^2+r3^2 + r1r2+r2r3+r3r1)
    integral_r_sq = area_ex2 / 6.0 * (r1**2 + r2**2 + r3**2 + r1*r2 + r2*r3 + r3*r1)
    integral_exact_axisym3 = 2 * np.pi * integral_r_sq
    print(f"\n例6: 軸対称 f=r (∫ 2πr^2 dA) (理論値={integral_exact_axisym3:.8f})")
    for n_p in [1, 3, 4]: # 3次式の積分になるので4点積分が必要
        integral = gaussian_quadrature_triangle(func_r, nodes_ex2, n_p, axisymmetric=True, axis_coord_index=0)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_axisym3):.2e})")


    def func_L1_5(L1, L2, L3):
        return L1**5
    integral_exact = area_ex2* (1.0/42)
    print(f"\n例7: L1^5 (∫ L1^5 dA) (理論値={integral_exact:.8f})")
    for n_p in [1, 3, 4]: # 3次式の積分になるので4点積分が必要
        integral = gaussian_quadrature_triangle(func_L1_5, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact):.2e})")
    
    # 例7: f = L1**2 を積分 (理論値 A/6)
    # 被積分関数は2次式。3点(2次精度)以上で正確になるはず。
    def func_L1sq(L1, L2, L3):
        return L1**2
    integral_exact_L1sq = area_exact / 6.0
    print(f"\n例7: f = L1^2 の積分 (理論値={integral_exact_L1sq:.8f})")
    for n_p in [1, 3, 4]:
        integral = gaussian_quadrature_triangle(func_L1sq, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_L1sq):.2e})")

    # 例8: f = L1**3 を積分 (理論値 A/10)
    # 被積分関数は3次式。4点(3次精度)で正確になるはず。
    def func_L1cb(L1, L2, L3):
        return L1**3
    integral_exact_L1cb = area_exact / 10.0
    print(f"\n例8: f = L1^3 の積分 (理論値={integral_exact_L1cb:.8f})")
    for n_p in [1, 3, 4]:
        integral = gaussian_quadrature_triangle(func_L1cb, nodes_ex1, n_p)
        print(f"  積分点 {n_p}個: 結果 = {integral:.8f} (誤差 = {abs(integral - integral_exact_L1cb):.2e})")
