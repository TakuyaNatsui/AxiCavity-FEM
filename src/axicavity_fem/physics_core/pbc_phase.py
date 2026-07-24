"""進行波の周期境界条件 (Bloch) における位相因子を一元管理する.

本モジュールは AxiCavity-FEM の **規約セーフティ層** である。
TM0 / HOM 双方のソルバから、PBC に関する位相計算は **必ず本モジュール経由で**
行うこと。直接 ``cmath.exp(1j * theta)`` 等を書かないこと。

物理規約と過去のバグ
--------------------
本コードは時間規約 ``exp(+j ω t)`` を採用する。+z 方向へ伝搬する進行波の Bloch
条件は次式となる::

    x_max = e^{-j θ} * x_min        (θ = k * L_cell > 0 で +z 伝搬)

ver1 では 2025 年 4 月以前まで誤って ``e^{+j θ}`` を使っていた。これにより
進行波アニメーションが −z 方向を向くという症状が発生していた。本モジュールに
固定実装することで、再発を防止する。

公開 API
--------
- :func:`pbc_phase_factor` (rad)
- :func:`pbc_phase_factor_deg` (degree)
- :func:`apply_pbc_complex`     — complex 振幅 x_min から x_max を算出
- :func:`real_imag_pbc_matrix`  — [Re; Im] 表現の 2x2 実数変換行列
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def pbc_phase_factor(theta_rad: float) -> complex:
    """進行波 PBC の位相因子 ``e^{-jθ}`` を返す.

    Args:
        theta_rad: 位相アドバンス θ [rad]. θ > 0 で +z 方向伝搬。

    Returns:
        複素数 ``e^{-jθ} = cos θ − j sin θ``.
    """
    return cmath.exp(-1j * theta_rad)


def pbc_phase_factor_deg(theta_deg: float) -> complex:
    """進行波 PBC の位相因子（度数指定版）."""
    return pbc_phase_factor(math.radians(theta_deg))


def apply_pbc_complex(x_min, theta_rad: float):
    """``x_max = e^{-jθ} * x_min`` を返す.

    Args:
        x_min: 複素振幅（スカラー、または配列）。
        theta_rad: 位相アドバンス [rad].

    Returns:
        ``x_max`` （入力と同じ型）。
    """
    return pbc_phase_factor(theta_rad) * np.asarray(x_min)


def real_imag_pbc_matrix(theta_rad: float) -> np.ndarray:
    """[Re; Im] 表現での PBC 変換行列を返す.

    複素 ``x_max = (cos θ − j sin θ) * x_min`` を [Re, Im] 実数ベクトル表現
    で書くと::

        [Re(x_max)]   [ cos θ,  sin θ] [Re(x_min)]
        [Im(x_max)] = [−sin θ,  cos θ] [Im(x_min)]

    Args:
        theta_rad: 位相アドバンス [rad].

    Returns:
        (2, 2) ndarray.
    """
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, s], [-s, c]])
