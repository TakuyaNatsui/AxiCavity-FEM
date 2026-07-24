"""ポインティングベクトル P_flow の正しい計算公式と数値ライブラリ.

本モジュールは AxiCavity-FEM の **規約セーフティ層** である。
P_flow に関する計算は **必ず本モジュールの関数経由で** 行うこと。

物理規約と過去のバグ
--------------------
ver1 では 2026 年 4 月以前まで、P_flow を「全要素 2D 域積分かつ被積分関数に
1/r を含む」誤った形で計算していた。次元的にも物理的にも誤りで、群速度の
符号は合っていたが値が誤っていた。

正しい P_flow は **z = z_min 断面のラインインテグラル** として求める。

TM0 モード（軸対称、DOF: ψ = H_φ）の場合の解析式::

    S_z = ½ Re[E_r H_φ^*] = −1/(2 ω ε₀) · Im[(∂ψ/∂z) ψ^*]
    P_flow = ∫∫ S_z dA = −π / (ω ε₀) · ∫₀^{r_max} Im[(∂ψ/∂z) ψ^*] r dr

（方位角積分 ∫₀^{2π} dφ = 2π を含む。詳細は PHYSICS_AND_CONVENTIONS.md。）

公開 API
--------
- :func:`gauss_3point_unit_interval` — 単位区間 [0, 1] の 3 点ガウス求積データ
- :func:`edge_line_integral_3point` — 1 エッジ上の 3 点ガウス線積分
- :func:`tm0_p_flow_coefficient` — TM0 公式の前置係数 −π/(ω ε₀)

注意
----
TM0 の P_flow 本実装（メッシュエッジ走査 + ∂ψ/∂z 計算）は ``fem_tm0/
post_process.py`` 側で本モジュールのプリミティブを使って構築する。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..shared.constants import EPS0


# ガウス・ルジャンドル 3 点求積（標準区間 [-1, 1]）
# ξ = ±sqrt(3/5), 0; w = 5/9, 8/9, 5/9
_GL3_NODES_M1P1 = np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)])
_GL3_WEIGHTS_M1P1 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def gauss_3point_unit_interval() -> tuple[np.ndarray, np.ndarray]:
    """単位区間 ``t ∈ [0, 1]`` 上の 3 点ガウス求積点と重みを返す.

    標準区間 [-1, 1] からの線形変換 ``t = (1 + ξ) / 2``,  ``dt = dξ / 2`` により
    重みは半分になる::

        nodes:   (1 ± √(3/5)) / 2,  1/2
        weights: 5/18, 8/18, 5/18    (Σ = 1)

    Returns:
        ``(t_nodes, w_nodes)`` 形状 ``(3,)`` ずつの配列。
    """
    t_nodes = 0.5 * (1.0 + _GL3_NODES_M1P1)
    w_nodes = 0.5 * _GL3_WEIGHTS_M1P1
    return t_nodes, w_nodes


def edge_line_integral_3point(
    integrand: Callable[[float], float],
    edge_length: float,
) -> float:
    """1 エッジ上の 3 点ガウス線積分 ``∫₀¹ integrand(t) · L dt`` を返す.

    Args:
        integrand: パラメータ ``t ∈ [0, 1]`` を受ける純関数（複素数を返してもよい）。
            エッジ上の被積分関数値（例: ``Im[∂ψ/∂z · ψ*] · r``）を返す。
        edge_length: エッジの物理長 (Jacobian)。z=z_min 断面で z 一定なら ``|r_b - r_a|``。

    Returns:
        積分値（``integrand`` の戻り値の型に従う）。
    """
    t_nodes, w_nodes = gauss_3point_unit_interval()
    total = 0.0
    for t, w in zip(t_nodes, w_nodes):
        total += w * integrand(float(t))
    return total * edge_length


def tm0_p_flow_coefficient(omega: float, eps0: float = EPS0) -> float:
    """TM0 モードの P_flow 公式の前置係数 ``−π / (ω ε₀)`` を返す.

    Args:
        omega: 角周波数 [rad/s]。
        eps0: 真空誘電率 [F/m]。デフォルトは ``shared.constants.EPS0``。

    Returns:
        前置係数 ``−π / (ω ε₀)``.

    Note (ver2.1 制約):
        ε₀ は z=z_min ポート断面で一定であることを前提とする。誘電体が port
        断面を跨ぐ場合は ε₀ → ε(r)·ε₀ で区分的に積分する必要があるが、
        ver2.1 では未対応。ポートは vacuum 領域内に置くことを推奨し、
        ``fem_tm0.post_process.tm0_p_flow_zmin`` 側で sanity check を行う。
    """
    return -math.pi / (omega * eps0)


# ------------------------------------------------------------------
# 以下、TM0 の本実装は段階 3 で fem_tm0/post_process.py に詰める。
# ここではシグネチャだけ確保しておき、誤った P_flow 公式が他の場所で
# 書かれるのを防ぐ。
# ------------------------------------------------------------------
def tm0_p_flow_zmin(*args, **kwargs):
    """TM0 進行波の P_flow を z=z_min 断面のラインインテグラルで計算する.

    Args:
        ψ_nodes (np.ndarray): H_φ 節点値の複素配列。
        mesh_data: メッシュデータ（節点座標・要素・要素次数）。
        omega (float): 角周波数 [rad/s]。
        eps0 (float): 真空誘電率 [F/m]（省略時 EPS0）。
        n_gauss (int): エッジ積分の点数（標準 3）。

    Returns:
        float: P_flow [W]。

    Raises:
        NotImplementedError: 本実装は段階 3 (fem_tm0/post_process.py) で行う。
    """
    raise NotImplementedError(
        "TM0 P_flow 本実装は fem_tm0/post_process.py で行います。"
        " 公式は physics_core/power_flow.py の docstring を参照してください。"
    )
