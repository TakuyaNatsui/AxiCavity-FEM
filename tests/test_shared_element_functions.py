"""shared/element_functions.py の単体テスト."""

import numpy as np
import pytest

from axicavity_fem.shared.element_functions import (
    calculate_area_coordinates,
    calculate_curl_edge_shape_functions_2nd,
    calculate_edge_shape_functions,
    calculate_edge_shape_functions_2nd,
    calculate_quadratic_nodal_shape_functions,
    calculate_triangle_area_double,
    grad_area_coordinates,
    grad_quadratic_nodal_shape_functions,
)

TOL = 1e-12


# ------------------------------------------------------------
# テスト用三角形
# 反時計回り (CCW): A = 2.0 → A2 = 4.0
# ------------------------------------------------------------
@pytest.fixture
def tri_ccw():
    return np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])


@pytest.fixture
def tri_general():
    # 一般的な三角形（重心が原点でない、軸非整列）
    return np.array([[1.0, 0.5], [3.0, 1.0], [2.0, 3.0]])


# ============================================================
# calculate_triangle_area_double
# ============================================================
def test_area_double_ccw(tri_ccw):
    assert abs(calculate_triangle_area_double(tri_ccw) - 4.0) < TOL


def test_area_double_cw_sign():
    """時計回りでは符号が反転する."""
    tri_cw = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    assert abs(calculate_triangle_area_double(tri_cw) + 4.0) < TOL


# ============================================================
# calculate_area_coordinates
# ============================================================
def test_area_coords_vertex(tri_general):
    """頂点で L は (1, 0, 0), (0, 1, 0), (0, 0, 1) の単位ベクトル."""
    for i, expected in enumerate(np.eye(3)):
        L = calculate_area_coordinates(tri_general[i], tri_general)
        np.testing.assert_allclose(L, expected, atol=TOL)


def test_area_coords_centroid(tri_general):
    """重心で L = (1/3, 1/3, 1/3)."""
    centroid = tri_general.mean(axis=0)
    L = calculate_area_coordinates(centroid, tri_general)
    np.testing.assert_allclose(L, [1 / 3, 1 / 3, 1 / 3], atol=TOL)


def test_area_coords_sum_to_one(tri_general):
    """任意の点で L1 + L2 + L3 = 1."""
    for r in [[1.5, 1.5], [2.0, 2.0], [1.2, 0.8]]:
        L = calculate_area_coordinates(np.array(r), tri_general)
        assert abs(L.sum() - 1.0) < TOL


# ============================================================
# grad_area_coordinates
# ============================================================
def test_grad_area_coords_sum_to_zero(tri_general):
    """∇L1 + ∇L2 + ∇L3 = 0 (Σ Li = 1 の勾配は 0)."""
    gL1, gL2, gL3 = grad_area_coordinates(tri_general)
    np.testing.assert_allclose(gL1 + gL2 + gL3, [0.0, 0.0], atol=TOL)


# ============================================================
# calculate_quadratic_nodal_shape_functions (G1-G6)
# ============================================================
def test_quad_nodal_partition_of_unity():
    """任意の面積座標で Σ Gi = 1."""
    for L in [
        [1.0, 0.0, 0.0],         # 頂点1
        [0.0, 1.0, 0.0],         # 頂点2
        [0.5, 0.5, 0.0],         # 辺中点
        [1 / 3, 1 / 3, 1 / 3],   # 重心
        [0.6, 0.3, 0.1],         # ランダム
    ]:
        G = calculate_quadratic_nodal_shape_functions(np.array(L))
        assert abs(G.sum() - 1.0) < TOL


def test_quad_nodal_kronecker_delta():
    """6 節点位置で δ_ij が成立 (G_i(node_j) = δ_ij)."""
    # G1-G6 に対応するノード位置の面積座標
    node_L = [
        [1.0, 0.0, 0.0],   # 節点 1 (頂点 1)
        [0.0, 1.0, 0.0],   # 節点 2 (頂点 2)
        [0.0, 0.0, 1.0],   # 節点 3 (頂点 3)
        [0.5, 0.5, 0.0],   # 節点 4 (辺 1-2 中点)
        [0.0, 0.5, 0.5],   # 節点 5 (辺 2-3 中点)
        [0.5, 0.0, 0.5],   # 節点 6 (辺 3-1 中点)
    ]
    for j, L in enumerate(node_L):
        G = calculate_quadratic_nodal_shape_functions(np.array(L))
        expected = np.zeros(6)
        expected[j] = 1.0
        np.testing.assert_allclose(G, expected, atol=TOL)


# ============================================================
# grad_quadratic_nodal_shape_functions
# ============================================================
def test_grad_quad_nodal_sum_to_zero(tri_general):
    """Σ Gi = 1 ⇒ Σ ∇Gi = 0 (任意の点で)."""
    gL = grad_area_coordinates(tri_general)
    for L in [[0.5, 0.3, 0.2], [1 / 3, 1 / 3, 1 / 3]]:
        gradG = grad_quadratic_nodal_shape_functions(np.array(L), gL)
        np.testing.assert_allclose(gradG.sum(axis=0), [0.0, 0.0], atol=TOL)


# ============================================================
# calculate_edge_shape_functions (1次 Whitney)
# ============================================================
def test_edge_shape_whitney_tangential_unit(tri_ccw):
    """Whitney 基底はエッジに沿う単位接線成分を持つ.

    エッジ i-j の中点で、N_k(辺ij) を辺ベクトル e_ij に内積すると：
    対応するエッジでは ∫ N · t dl = 1（長さ 1 単位接線積分）。
    ここでは弱い性質として、N1 の中点での向きが e_{12} と一致することを確認する。
    """
    # 辺 1-2 の中点
    mid12 = (tri_ccw[0] + tri_ccw[1]) / 2
    N1, N2, N3 = calculate_edge_shape_functions(mid12, tri_ccw)
    edge12 = tri_ccw[1] - tri_ccw[0]
    # N1 は辺 1-2 に対応する Whitney 関数。N1 と edge12 の内積は正
    assert np.dot(N1, edge12) > 0


# ============================================================
# calculate_curl_edge_shape_functions_2nd
# ============================================================
def test_curl_2nd_constant_for_ctln(tri_ccw):
    """N1-N3 (CT/LN) のカールは 1/Ae で要素内一定."""
    A2 = calculate_triangle_area_double(tri_ccw)  # 4.0
    expected = 2.0 / A2  # = 0.5

    for L in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3]]:
        # 面積座標から座標 r を逆算
        r = (
            L[0] * tri_ccw[0]
            + L[1] * tri_ccw[1]
            + L[2] * tri_ccw[2]
        )
        curls = calculate_curl_edge_shape_functions_2nd(r, tri_ccw)
        assert abs(curls[0] - expected) < TOL
        assert abs(curls[1] - expected) < TOL
        assert abs(curls[2] - expected) < TOL


def test_curl_2nd_zero_for_ltln(tri_general):
    """N4-N6 (LT/LN) のカールは 0."""
    curls = calculate_curl_edge_shape_functions_2nd(
        tri_general.mean(axis=0), tri_general
    )
    assert abs(curls[3]) < TOL
    assert abs(curls[4]) < TOL
    assert abs(curls[5]) < TOL


def test_curl_2nd_face_formula(tri_ccw):
    """N7 = (3 L3 - 1) / A2, N8 = (3 L1 - 1) / A2."""
    A2 = calculate_triangle_area_double(tri_ccw)
    # 重心で L = (1/3, 1/3, 1/3) → N7 = 0, N8 = 0
    centroid = tri_ccw.mean(axis=0)
    curls = calculate_curl_edge_shape_functions_2nd(centroid, tri_ccw)
    assert abs(curls[6]) < TOL
    assert abs(curls[7]) < TOL

    # 頂点3 で L3 = 1 → N7 = 2/A2
    curls = calculate_curl_edge_shape_functions_2nd(tri_ccw[2], tri_ccw)
    assert abs(curls[6] - 2.0 / A2) < TOL


# ============================================================
# calculate_edge_shape_functions_2nd
# ============================================================
def test_edge_shape_2nd_returns_8_vectors(tri_general):
    Ns = calculate_edge_shape_functions_2nd(
        tri_general.mean(axis=0), tri_general
    )
    assert len(Ns) == 8
    for N in Ns:
        assert N.shape == (2,)
