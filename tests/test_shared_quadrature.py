"""shared/quadrature.py の単体テスト.

ディリクレの積分公式: ∫ L1^a L2^b L3^c dA = 2 * A * a! b! c! / (a+b+c+2)!

これに基づき、各積分点数での精度（多項式次数の上限）を検証する。
"""

import math

import numpy as np
import pytest

from axicavity_fem.shared.quadrature import (
    calculate_triangle_area,
    gaussian_quadrature_triangle,
    integration_points_triangle,
)

TOL = 1e-12


@pytest.fixture
def tri():
    # 単位直角三角形 (面積 1/2)
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def tri_general():
    return np.array([[1.0, 0.5], [3.0, 1.0], [2.0, 3.0]])


def dirichlet_integral(a, b, c, area):
    """∫ L1^a L2^b L3^c dA の解析値."""
    return (
        2.0 * area * math.factorial(a) * math.factorial(b) * math.factorial(c)
        / math.factorial(a + b + c + 2)
    )


# ============================================================
# 積分点・重みの整合性
# ============================================================
@pytest.mark.parametrize("n_points", [1, 3, 4, 7, "3_tm0_midpoint"])
def test_weights_sum_to_one(n_points):
    """全ての積分公式で Σ w_i = 1."""
    rule = integration_points_triangle[n_points]
    assert abs(rule["weights"].sum() - 1.0) < TOL


@pytest.mark.parametrize("n_points", [1, 3, 4, 7, "3_tm0_midpoint"])
def test_barycentric_sum_to_one(n_points):
    """全ての積分点で L1 + L2 + L3 = 1."""
    rule = integration_points_triangle[n_points]
    for L in rule["coords"]:
        assert abs(L.sum() - 1.0) < TOL


# ============================================================
# 定数関数: f = 1 → 面積
# ============================================================
@pytest.mark.parametrize("n_points", [1, 3, 4, 7, "3_tm0_midpoint"])
def test_constant_function(tri, n_points):
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: 1.0, tri, n_points=n_points
    )
    assert abs(result - 0.5) < TOL


# ============================================================
# 1次多項式: f = L1 → A/3
# ============================================================
@pytest.mark.parametrize("n_points", [1, 3, 4, 7, "3_tm0_midpoint"])
def test_linear_L1(tri, n_points):
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: L1, tri, n_points=n_points
    )
    expected = dirichlet_integral(1, 0, 0, area=0.5)  # = 1/6
    assert abs(result - expected) < TOL


# ============================================================
# 2次多項式: f = L1 * L2 → A/12（3 点以上で正確）
# ============================================================
@pytest.mark.parametrize("n_points", [3, 4, 7, "3_tm0_midpoint"])
def test_quadratic_L1L2(tri, n_points):
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: L1 * L2, tri, n_points=n_points
    )
    expected = dirichlet_integral(1, 1, 0, area=0.5)  # = 1/24
    assert abs(result - expected) < TOL


@pytest.mark.parametrize("n_points", [3, 4, 7, "3_tm0_midpoint"])
def test_quadratic_L1sq(tri, n_points):
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: L1 ** 2, tri, n_points=n_points
    )
    expected = dirichlet_integral(2, 0, 0, area=0.5)  # = 1/12
    assert abs(result - expected) < TOL


def test_quadratic_inexact_with_1point(tri):
    """1 点積分では 2 次多項式は不正確（粗いが許容範囲）."""
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: L1 * L2, tri, n_points=1
    )
    expected = dirichlet_integral(1, 1, 0, area=0.5)
    # 1点積分: L1=L2=L3=1/3 → result = 0.5 * (1/3)*(1/3) = 1/18 ≠ 1/24
    assert abs(result - expected) > 1e-3


# ============================================================
# 3次多項式: f = L1^3 → A/10（4 点以上で正確）
# ============================================================
@pytest.mark.parametrize("n_points", [4, 7])
def test_cubic_L1cb(tri, n_points):
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: L1 ** 3, tri, n_points=n_points
    )
    expected = dirichlet_integral(3, 0, 0, area=0.5)  # = 1/20
    assert abs(result - expected) < TOL


# ============================================================
# 5次多項式: f = L1^5 → 7 点積分のみで正確
# ============================================================
def test_quintic_L1_5_with_7points(tri):
    result = gaussian_quadrature_triangle(
        lambda L1, L2, L3: L1 ** 5, tri, n_points=7
    )
    expected = dirichlet_integral(5, 0, 0, area=0.5)  # = 1/42
    assert abs(result - expected) < TOL


# ============================================================
# 軸対称積分 ∫ r dA（クロージャで vertices をバインド）
# ============================================================
def test_axisymmetric_r_integral(tri_general):
    """∫ r dA = A * r_centroid （重心の r 座標）."""
    verts = tri_general
    area = calculate_triangle_area(verts)
    r_centroid = verts[:, 1].mean()
    expected = area * r_centroid

    def integrand(L1, L2, L3):
        return L1 * verts[0, 1] + L2 * verts[1, 1] + L3 * verts[2, 1]

    for n_points in [3, 4, 7]:
        result = gaussian_quadrature_triangle(integrand, verts, n_points=n_points)
        assert abs(result - expected) < TOL


def test_closure_binds_indices():
    """クロージャ方式で要素インデックスをバインドする使い方が機能する."""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def make_integrand(i, j):
        Ls_fn = [lambda L1, L2, L3: L1, lambda L1, L2, L3: L2, lambda L1, L2, L3: L3]
        return lambda L1, L2, L3: Ls_fn[i](L1, L2, L3) * Ls_fn[j](L1, L2, L3)

    # ∫ L1*L2 dA = 1/24
    result = gaussian_quadrature_triangle(make_integrand(0, 1), tri, n_points=4)
    assert abs(result - 1 / 24) < TOL


# ============================================================
# エラーハンドリング
# ============================================================
def test_unknown_npoints_raises():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="未定義の積分点数"):
        gaussian_quadrature_triangle(lambda L1, L2, L3: 1.0, tri, n_points=5)


def test_zero_area_returns_zero():
    """退化した三角形は積分値 0 を返す."""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # 共線
    result = gaussian_quadrature_triangle(lambda L1, L2, L3: 1.0, tri, n_points=4)
    assert result == 0.0
