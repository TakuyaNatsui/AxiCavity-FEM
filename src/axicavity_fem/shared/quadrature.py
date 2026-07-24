"""三角形上のガウス求積.

ver1 ``gaussian_quadrature_triangle.py`` (TM0/HOM 両版) を統合・API 統一したもの。

API 統一の方針:
    呼び出し側がクロージャで頂点座標 ``vertices`` や要素インデックスをバインドし、
    被積分関数 ``integrand`` は面積座標 ``(L1, L2, L3)`` のみを受け取る純関数とする。

    例 (軸対称 ∫ r * f dA を計算する場合)::

        def make_integrand(i, j, verts):
            def integrand(L1, L2, L3):
                r = L1 * verts[0, 1] + L2 * verts[1, 1] + L3 * verts[2, 1]
                return r * L1 * L2  # 例
            return integrand

        value = gaussian_quadrature_triangle(make_integrand(i, j, V), V, n_points=4)

積分点・重み:
    - n_points=1: 重心1点 (1次精度)
    - n_points=3: Dunavant degree 2 (ver1 HOM 版を採用; 2次精度)
    - n_points=4: 重心 + 3点 (3次精度)
    - n_points=7: Dunavant degree 5 (5次精度、2次エッジ要素向け)
    - n_points="3_tm0_midpoint": ver1 TM0 旧スキーム (辺中点ベース、2次精度)
      ※ 後方互換用。新規コードでは 3 (Dunavant) を推奨。
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np


# --- 7 点積分（Dunavant 1985, degree 5）の係数 ---
_sqrt15 = np.sqrt(15.0)
_a = (6.0 - _sqrt15) / 21.0
_b = (6.0 + _sqrt15) / 21.0
_w1 = 9.0 / 40.0
_w2 = (155.0 + _sqrt15) / 1200.0
_w3 = (155.0 - _sqrt15) / 1200.0


# --- 積分点と重み (重み合計 = 1.0 に正規化) ---
integration_points_triangle: dict[Union[int, str], dict[str, np.ndarray]] = {
    1: {
        "coords": np.array([[1 / 3, 1 / 3, 1 / 3]]),
        "weights": np.array([1.0]),
    },
    3: {
        # Dunavant degree 2 (ver1 HOM 版)
        "coords": np.array([
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 2 / 3, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
        ]),
        "weights": np.array([1 / 3, 1 / 3, 1 / 3]),
    },
    4: {
        "coords": np.array([
            [1 / 3, 1 / 3, 1 / 3],
            [3 / 5, 1 / 5, 1 / 5],
            [1 / 5, 3 / 5, 1 / 5],
            [1 / 5, 1 / 5, 3 / 5],
        ]),
        "weights": np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48]),
    },
    7: {
        # Dunavant degree 5
        "coords": np.array([
            [1 / 3, 1 / 3, 1 / 3],
            [_a, _a, 1 - 2 * _a],
            [_a, 1 - 2 * _a, _a],
            [1 - 2 * _a, _a, _a],
            [_b, _b, 1 - 2 * _b],
            [_b, 1 - 2 * _b, _b],
            [1 - 2 * _b, _b, _b],
        ]),
        "weights": np.array([_w1, _w3, _w3, _w3, _w2, _w2, _w2]),
    },
    "3_tm0_midpoint": {
        # ver1 TM0 旧スキーム（辺中点ベース、後方互換用）
        "coords": np.array([
            [1 / 2, 1 / 2, 0.0],
            [0.0, 1 / 2, 1 / 2],
            [1 / 2, 0.0, 1 / 2],
        ]),
        "weights": np.array([1 / 3, 1 / 3, 1 / 3]),
    },
}


def calculate_triangle_area(vertices) -> float:
    """3 頂点から三角形の面積（絶対値）を返す."""
    p1, p2, p3 = vertices[0], vertices[1], vertices[2]
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def gaussian_quadrature_triangle(
    integrand: Callable[[float, float, float], float],
    vertices: np.ndarray,
    n_points: Union[int, str] = 4,
) -> float:
    """三角形上の積分 ``∫_Ae integrand(L1, L2, L3) dA`` を返す.

    Args:
        integrand: 面積座標 ``(L1, L2, L3)`` を受ける純関数（または ndarray を返す関数）。
            頂点や要素インデックスはクロージャでバインドすること。
        vertices: 三角形コーナー頂点 (3, 2). 2次要素の場合は ``vertices[:3]`` を渡す。
        n_points: 1, 3, 4, 7 のいずれか、または "3_tm0_midpoint"。

    Returns:
        積分値 ``Ae * Σ w_i integrand(L^(i))``。軸対称の ``2πr`` 係数等は ``integrand``
        の中で扱うこと。
    """
    if n_points not in integration_points_triangle:
        raise ValueError(
            f"未定義の積分点数: {n_points!r}. "
            f"利用可能: {list(integration_points_triangle.keys())}"
        )

    rule = integration_points_triangle[n_points]
    coords_L = rule["coords"]
    weights = rule["weights"]

    area = calculate_triangle_area(vertices[:3])
    if np.isclose(area, 0.0):
        return 0.0

    total = 0.0
    for k in range(len(weights)):
        L1, L2, L3 = coords_L[k]
        total += weights[k] * integrand(L1, L2, L3)

    return area * total
