"""shared/edge_index.py の単体テスト."""

import numpy as np

from axicavity_fem.shared.edge_index import create_edge_index_map


def test_single_triangle():
    """1 つの三角形は 3 エッジ."""
    simplices = np.array([[0, 1, 2]])
    edge_map, n_edges = create_edge_index_map(simplices)
    assert n_edges == 3
    assert (0, 1) in edge_map
    assert (1, 2) in edge_map
    assert (0, 2) in edge_map


def test_shared_edge_deduplicated():
    """共有エッジを持つ 2 三角形は 5 エッジ (1 つ重複)."""
    simplices = np.array([
        [0, 1, 2],
        [1, 2, 3],
    ])
    edge_map, n_edges = create_edge_index_map(simplices)
    assert n_edges == 5
    # (1, 2) が両要素で共有される
    assert (1, 2) in edge_map


def test_keys_are_sorted_tuples():
    """キーは常に sorted tuple で正規化される."""
    simplices = np.array([[5, 2, 8]])
    edge_map, _ = create_edge_index_map(simplices)
    # 順序を逆にしてもキーが見つかる
    assert (2, 5) in edge_map
    assert (2, 8) in edge_map
    assert (5, 8) in edge_map


def test_indices_are_sequential():
    """エッジインデックスは 0 から連続."""
    simplices = np.array([[0, 1, 2], [2, 3, 4]])
    edge_map, n_edges = create_edge_index_map(simplices)
    indices = sorted(edge_map.values())
    assert indices == list(range(n_edges))


def test_works_with_6node_corners_only():
    """2 次要素のコーナーノードだけを渡せばよい (midside ノードは含めない)."""
    # 2次要素 (6 nodes per element): corners are first 3
    simplices_full = np.array([
        [0, 1, 2, 10, 11, 12],
        [1, 2, 3, 11, 13, 14],
    ])
    edge_map, n_edges = create_edge_index_map(simplices_full[:, :3])
    assert n_edges == 5
