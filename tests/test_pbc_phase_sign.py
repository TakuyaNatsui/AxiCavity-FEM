"""進行波 PBC 位相符号の回帰テスト.

ver1 (2025 年 4 月以前) で発生していたバグ:
    旧実装は ``x_max = e^{+jθ} * x_min`` を使用
    正しくは物理規約 e^{+jωt} のもとで ``x_max = e^{-jθ} * x_min``

本テストは ``physics_core/pbc_phase.py`` が正しい符号 (e^{-jθ}) を返すこと、
旧バグの形 (e^{+jθ}) と区別されることを担保する。
"""

import math

import numpy as np
import pytest

from axicavity_fem.physics_core.pbc_phase import (
    apply_pbc_complex,
    pbc_phase_factor,
    pbc_phase_factor_deg,
    real_imag_pbc_matrix,
)

TOL = 1e-12


# ============================================================
# 基本値: e^{-jθ} の各代表点
# ============================================================
def test_phase_factor_zero():
    """θ = 0 → 1."""
    assert abs(pbc_phase_factor(0.0) - 1.0) < TOL


def test_phase_factor_pi_half_is_minus_j():
    """θ = π/2 → -j （バグ版なら +j になる）."""
    val = pbc_phase_factor(math.pi / 2)
    assert abs(val - (-1j)) < TOL
    # バグ版との明確な区別
    assert abs(val - (+1j)) > 1.0


def test_phase_factor_pi():
    """θ = π → -1."""
    assert abs(pbc_phase_factor(math.pi) - (-1.0)) < TOL


def test_phase_factor_minus_pi_half():
    """θ = -π/2 → +j."""
    assert abs(pbc_phase_factor(-math.pi / 2) - 1j) < TOL


# ============================================================
# 度数版
# ============================================================
@pytest.mark.parametrize("deg,expected", [
    (0.0, 1.0 + 0j),
    (90.0, -1j),
    (180.0, -1.0 + 0j),
    (270.0, 1j),
    (360.0, 1.0 + 0j),
])
def test_phase_factor_deg(deg, expected):
    val = pbc_phase_factor_deg(deg)
    assert abs(val - expected) < TOL


# ============================================================
# apply_pbc_complex
# ============================================================
def test_apply_pbc_real_input():
    """実数 x_min = 1 に θ = 60° を適用すると e^{-j 60°}."""
    val = apply_pbc_complex(1.0, math.radians(60.0))
    expected = complex(math.cos(math.radians(60.0)), -math.sin(math.radians(60.0)))
    assert abs(val - expected) < TOL


def test_apply_pbc_array():
    """配列入力に対しても正しく動作."""
    x = np.array([1.0, 2.0, 3.0])
    result = apply_pbc_complex(x, math.pi / 2)
    expected = -1j * x
    np.testing.assert_allclose(result, expected, atol=TOL)


# ============================================================
# real_imag_pbc_matrix
# ============================================================
def test_real_imag_pbc_matrix_rotation():
    """[Re; Im] 表現と複素計算が一致する."""
    theta = math.radians(75.0)
    M = real_imag_pbc_matrix(theta)

    # 任意の x_min = a + jb をテスト
    x_min = 1.7 + 2.3j
    re_im = M @ np.array([x_min.real, x_min.imag])
    x_max_via_matrix = complex(re_im[0], re_im[1])

    x_max_via_complex = apply_pbc_complex(x_min, theta)

    assert abs(x_max_via_matrix - x_max_via_complex) < TOL


def test_real_imag_pbc_matrix_determinant_one():
    """回転行列なので determinant = 1."""
    for theta in [0.1, 1.0, math.pi / 3, -0.7]:
        M = real_imag_pbc_matrix(theta)
        assert abs(np.linalg.det(M) - 1.0) < TOL


# ============================================================
# 回帰: バグ版 e^{+jθ} と区別される (重要)
# ============================================================
def test_regression_not_plus_j_sign():
    """過去のバグ (e^{+jθ}) と確実に区別されることを検証.

    バグ版: e^{+jπ/2} = +j
    正しい: e^{-jπ/2} = -j
    """
    val = pbc_phase_factor(math.pi / 2)
    # 正しい: -j
    assert val.imag < 0
    # バグ版なら imag > 0 になる
    assert not (val.imag > 0)


def test_regression_phase_direction_for_plus_z_propagation():
    """+z 方向伝搬 (θ > 0) で x_max は x_min から **負方向** に位相回転する.

    物理規約 e^{+jωt} のもとで +z 伝搬する進行波は
    空間依存が e^{-jkz} のため、+z 方向に進むと位相が遅れる
    (= 位相角が減少する)。
    """
    theta = math.radians(30.0)  # +z 伝搬
    x_min = 1.0 + 0j
    x_max = apply_pbc_complex(x_min, theta)

    # x_max の位相角が x_min より小さい (負の方向に回転している)
    assert math.degrees(np.angle(x_max)) < 0
