"""要素のコーナーノードからエッジインデックスマップを構築する.

ver1 ``FEM_HOM_code/mesh_reader.py:create_edge_index_map`` を移植したもの。
1 次（3 節点）・2 次（6 節点）要素どちらにも使用可能。
2 次要素の場合はコーナーのみを渡すこと: ``create_edge_index_map(simplices[:, :3])``
"""

from __future__ import annotations


def create_edge_index_map(
    simplices,
) -> tuple[dict[tuple[int, int], int], int]:
    """要素コーナーノードリストから一意なエッジインデックスを割り当てる.

    Args:
        simplices: 要素コーナーノードリスト (num_elements x 3)。
            各行は ``[n0, n1, n2]`` のコーナーノードインデックス。
            2 次要素の場合は ``simplices[:, :3]`` を渡す。

    Returns:
        (edge_index_map, edge_count) のタプル。
        ``edge_index_map`` は ``{tuple(sorted((n_i, n_j))): edge_index}`` の辞書。
    """
    edge_index_map: dict[tuple[int, int], int] = {}
    edge_count = 0
    for simplex in simplices:
        edges_local = [
            tuple(sorted((int(simplex[0]), int(simplex[1])))),
            tuple(sorted((int(simplex[1]), int(simplex[2])))),
            tuple(sorted((int(simplex[2]), int(simplex[0])))),
        ]
        for edge in edges_local:
            if edge not in edge_index_map:
                edge_index_map[edge] = edge_count
                edge_count += 1
    return edge_index_map, edge_count
