"""P_flow ラインインテグラル公式の回帰テスト.

ver1 (2026 年 4 月以前) で発生していたバグ:
    旧実装は P_flow を「全要素 2D 域積分 + 1/r 被積分関数」で計算
    正しくは z=z_min 断面の **ラインインテグラル** (1D)

本テストは ``physics_core/power_flow.py`` の数値積分プリミティブの正しさを
検証する。TM0 の本実装（メッシュエッジ走査）は段階 3 で fem_tm0/ 側に
構築される。
"""

import math

import numpy as np
import pytest

from axicavity_fem.physics_core.power_flow import (
    edge_line_integral_3point,
    gauss_3point_unit_interval,
    tm0_p_flow_coefficient,
    tm0_p_flow_zmin,
)
from axicavity_fem.shared.constants import EPS0

TOL = 1e-12


# ============================================================
# Gauss 3 点求積の整合性
# ============================================================
def test_gauss_weights_sum_to_one():
    """単位区間 [0, 1] 上で重み合計が 1."""
    _, w = gauss_3point_unit_interval()
    assert abs(w.sum() - 1.0) < TOL


def test_gauss_nodes_in_unit_interval():
    """積分点が [0, 1] 内に収まる."""
    t, _ = gauss_3point_unit_interval()
    assert (t > 0).all() and (t < 1).all()


def test_gauss_nodes_symmetric_around_half():
    """3 点が 0.5 を中心に対称."""
    t, _ = gauss_3point_unit_interval()
    assert abs(t[0] + t[2] - 1.0) < TOL  # 両端
    assert abs(t[1] - 0.5) < TOL  # 中央


# ============================================================
# edge_line_integral_3point: 多項式精度
# ============================================================
@pytest.mark.parametrize("L", [1.0, 2.5, 7.0])
def test_constant_integrand(L):
    """∫₀¹ 1 dt · L = L."""
    result = edge_line_integral_3point(lambda t: 1.0, L)
    assert abs(result - L) < TOL


def test_linear_integrand():
    """∫₀¹ t dt · L = L/2."""
    L = 3.0
    result = edge_line_integral_3point(lambda t: t, L)
    assert abs(result - L / 2) < TOL


def test_quadratic_integrand():
    """∫₀¹ t² dt · L = L/3."""
    L = 2.0
    result = edge_line_integral_3point(lambda t: t ** 2, L)
    assert abs(result - L / 3) < TOL


def test_cubic_integrand():
    """∫₀¹ t³ dt · L = L/4."""
    L = 1.5
    result = edge_line_integral_3point(lambda t: t ** 3, L)
    assert abs(result - L / 4) < TOL


def test_quintic_integrand_exact():
    """3 点 Gauss-Legendre は 5 次多項式まで正確: ∫₀¹ t⁵ dt · L = L/6."""
    L = 1.0
    result = edge_line_integral_3point(lambda t: t ** 5, L)
    assert abs(result - 1.0 / 6.0) < TOL


def test_sextic_integrand_inexact():
    """3 点 Gauss は 6 次多項式以上では誤差が出る (それを把握しておく)."""
    L = 1.0
    result = edge_line_integral_3point(lambda t: t ** 6, L)
    expected = 1.0 / 7.0
    assert abs(result - expected) > 1e-6   # 機械精度ではない
    assert abs(result - expected) < 1e-3   # でもそれほど大きくない


# ============================================================
# edge_line_integral_3point: 物理的な使い方（複素数積分）
# ============================================================
def test_complex_integrand():
    """複素数を返す integrand も扱える."""
    L = 1.0
    # ∫₀¹ (1 + jt) dt · L = (1 + j/2) · L
    result = edge_line_integral_3point(lambda t: 1.0 + 1j * t, L)
    expected = (1.0 + 0.5j) * L
    assert abs(result - expected) < TOL


def test_axisymmetric_r_integral_along_edge():
    """軸対称: エッジ (r_a, z) → (r_b, z) で ∫ r dr = (r_b² − r_a²) / 2.

    パラメータ化: r = r_a + t (r_b − r_a), dr/dt = r_b − r_a
    被積分関数: r(t) を返し、edge_length = r_b - r_a として渡す。
    """
    r_a, r_b = 0.0, 0.05
    L = r_b - r_a  # = dr/dt

    def integrand(t):
        return r_a + t * (r_b - r_a)

    result = edge_line_integral_3point(integrand, L)
    expected = 0.5 * (r_b ** 2 - r_a ** 2)
    assert abs(result - expected) < TOL


# ============================================================
# tm0_p_flow_coefficient: 物理量の係数チェック
# ============================================================
def test_tm0_p_flow_coefficient_sign_and_magnitude():
    """−π / (ω ε₀) の符号と大きさ."""
    omega = 2.0 * math.pi * 3.0e9  # 3 GHz
    coef = tm0_p_flow_coefficient(omega)

    # 符号: 負
    assert coef < 0

    # 大きさ
    expected = -math.pi / (omega * EPS0)
    assert abs(coef - expected) < TOL * abs(expected)


def test_tm0_p_flow_coefficient_custom_eps0():
    """eps0 をカスタム指定できる."""
    omega = 1.0
    coef = tm0_p_flow_coefficient(omega, eps0=2.0)
    assert abs(coef - (-math.pi / 2.0)) < TOL


# ============================================================
# tm0_p_flow_zmin: 本実装は段階 3 で行う
# ============================================================
def test_tm0_p_flow_zmin_not_implemented_yet():
    """段階 3 の本実装まで NotImplementedError を投げることを確認."""
    with pytest.raises(NotImplementedError, match="fem_tm0/post_process.py"):
        tm0_p_flow_zmin()


# ============================================================
# 回帰: 公式は LINE 積分（1D）であり、VOLUME 積分（2D）ではない
# ============================================================
def test_regression_uses_line_integral_not_volume():
    """edge_line_integral_3point は 1 エッジ (1D) を扱う。

    入力に edge_length（スカラー）を取る設計は、2D 体積積分用の
    要素面積を取る設計とは構造的に異なる。これは旧バグ
    （2D 体積積分 + 1/r）と区別される設計を保証する。
    """
    # 単位エッジで f = 1 → 結果はちょうど edge_length = 1.0
    result = edge_line_integral_3point(lambda t: 1.0, 1.0)
    assert abs(result - 1.0) < TOL

    # edge_length を変えると線形にスケール（面積比例ではない）
    result_2 = edge_line_integral_3point(lambda t: 1.0, 2.0)
    assert abs(result_2 - 2.0) < TOL    # 2 倍 ← 線形
    assert abs(result_2 - 4.0) > 1.0    # 4 倍にはならない (面積比例なら 4.0)
