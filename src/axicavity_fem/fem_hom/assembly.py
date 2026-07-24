"""HOM 全体行列のアセンブリ（ベクトル化版）.

ver1 ``FEM_HOM_code/element_assembly.py`` の
``assemble_global_matrices_vectorized`` / ``assemble_global_matrices_2nd_vectorized``
を移植したもの。DOF 番号体系・エッジ向き符号ルールは ver1 と同一。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..shared.mesh_loader import HOMMeshData
from .element_matrices import element_matrices_1st_batch, element_matrices_2nd_batch


def get_edge_orientation_sign(global1, global2) -> int:
    """エッジの向き符号（グローバルインデックス昇順なら +1）."""
    return 1 if global1 < global2 else -1


def assemble_global_matrices_1st(simplices, vertices, edge_index_map,
                                 num_edges, n, eps_r_per_element=None):
    """1 次エッジ要素の全体行列 (K, M) を組み立てる.

    DOF 体系: ``[0, num_edges)`` エッジ（Whitney）、``[num_edges, ...)`` 節点（n>0）。
    符号: エッジ DOF はエッジ向きに応じて ±1、節点 DOF は +1。

    Args:
        eps_r_per_element: (E,) 要素ごと比誘電率 (実数, >0)。HOM は E 場定式化
            のため、誘電体では **M_e に ε_r を乗じる**（TM0 の H_φ 定式化が K に
            1/ε_r を乗じるのと逆）。``None`` なら真空 (ε_r=1) で従来互換。

    Returns:
        (K_global, M_global): CSR。
    """
    E = len(simplices)
    num_nodes = len(vertices)
    matrix_size = num_edges + (num_nodes if n > 0 else 0)
    D = 3 if n == 0 else 6

    g_dofs_all = np.zeros((E, D), dtype=int)
    sign_all = np.ones((E, 3), dtype=float)

    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        edge_pairs = [(corner[0], corner[1]),
                      (corner[1], corner[2]),
                      (corner[2], corner[0])]
        for k, (n1, n2) in enumerate(edge_pairs):
            eg = edge_index_map[tuple(sorted((n1, n2)))]
            g_dofs_all[elem_idx, k] = eg
            sign_all[elem_idx, k] = 1 if n1 < n2 else -1
        if n > 0:
            for k in range(3):
                g_dofs_all[elem_idx, 3 + k] = num_edges + corner[k]

    P3 = vertices[simplices[:, :3]]
    K_e_all, M_e_all = element_matrices_1st_batch(P3, n)

    # 誘電体: 質量行列 M_e に ε_r を乗じる（E 場定式化。全ブロック一律、K は不変）
    if eps_r_per_element is not None:
        M_e_all = M_e_all * np.asarray(eps_r_per_element)[:, None, None]

    sign_mat = np.einsum("ei,ej->eij", sign_all, sign_all)
    K_e_all[:, :3, :3] *= sign_mat
    M_e_all[:, :3, :3] *= sign_mat
    if n > 0:
        K_e_all[:, :3, 3:] *= sign_all[:, :, np.newaxis]
        K_e_all[:, 3:, :3] *= sign_all[:, np.newaxis, :]

    rows = np.repeat(g_dofs_all, D, axis=1).flatten()
    cols = np.tile(g_dofs_all, (1, D)).flatten()
    K_global = coo_matrix((K_e_all.flatten(), (rows, cols)),
                          shape=(matrix_size, matrix_size),
                          dtype=np.float64).tocsr()
    M_global = coo_matrix((M_e_all.flatten(), (rows, cols)),
                          shape=(matrix_size, matrix_size),
                          dtype=np.float64).tocsr()
    return K_global, M_global


def assemble_global_matrices_2nd(simplices, vertices, edge_index_map,
                                 num_edges, n, eps_r_per_element=None):
    """2 次エッジ要素の全体行列 (K, M) を組み立てる.

    DOF 体系:
      - ``2*e``    : エッジ e の CT/LN
      - ``2*e+1``  : エッジ e の LT/LN
      - ``2*num_edges + 2*k`` / ``+1`` : 要素 k の face DOF (N7, N8)
      - ``2*num_edges + 2*num_elem + i`` : 節点 i（n>0）
    符号: CT/LN のみエッジ向きに応じて ±1、それ以外 +1。

    Args:
        eps_r_per_element: (E,) 要素ごと比誘電率。誘電体では M_e に ε_r を乗じる
            （:func:`assemble_global_matrices_1st` 参照）。``None`` で従来互換。

    Returns:
        (K_global, M_global): CSR。
    """
    E = len(simplices)
    num_nodes = len(vertices)
    face_offset = 2 * num_edges
    node_offset = 2 * num_edges + 2 * E
    matrix_size = node_offset + (num_nodes if n > 0 else 0)
    D = 8 if n == 0 else 14

    g_dofs_all = np.zeros((E, D), dtype=int)
    sign_all = np.ones((E, 8), dtype=float)

    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        edge_pairs = [(corner[0], corner[1]),
                      (corner[1], corner[2]),
                      (corner[2], corner[0])]
        for k, (n1, n2) in enumerate(edge_pairs):
            eg = edge_index_map[tuple(sorted((n1, n2)))]
            g_dofs_all[elem_idx, k] = 2 * eg
            g_dofs_all[elem_idx, k + 3] = 2 * eg + 1
            sign_all[elem_idx, k] = 1 if n1 < n2 else -1
        g_dofs_all[elem_idx, 6] = face_offset + 2 * elem_idx
        g_dofs_all[elem_idx, 7] = face_offset + 2 * elem_idx + 1
        if n > 0:
            for k in range(6):
                g_dofs_all[elem_idx, 8 + k] = node_offset + simplex[k]

    P = vertices[simplices]
    K_e_all, M_e_all = element_matrices_2nd_batch(P, n)

    # 誘電体: 質量行列 M_e に ε_r を乗じる（E 場定式化。全ブロック一律、K は不変）
    if eps_r_per_element is not None:
        M_e_all = M_e_all * np.asarray(eps_r_per_element)[:, None, None]

    sign_mat = np.einsum("ei,ej->eij", sign_all, sign_all)
    K_e_all[:, :8, :8] *= sign_mat
    M_e_all[:, :8, :8] *= sign_mat
    if n > 0:
        K_e_all[:, :3, 8:] *= sign_all[:, :3, np.newaxis]
        K_e_all[:, 8:, :3] *= sign_all[:, np.newaxis, :3]

    rows = np.repeat(g_dofs_all, D, axis=1).flatten()
    cols = np.tile(g_dofs_all, (1, D)).flatten()
    K_global = coo_matrix((K_e_all.flatten(), (rows, cols)),
                          shape=(matrix_size, matrix_size),
                          dtype=np.float64).tocsr()
    M_global = coo_matrix((M_e_all.flatten(), (rows, cols)),
                          shape=(matrix_size, matrix_size),
                          dtype=np.float64).tocsr()
    return K_global, M_global


def assemble_global_matrices(mesh: HOMMeshData, eps_r_per_element=None
                             ) -> tuple[csr_matrix, csr_matrix]:
    """:class:`HOMMeshData` から要素次数に応じて全体行列を組み立てる便利関数.

    Args:
        eps_r_per_element: (E,) 要素ごと比誘電率。``None`` で真空（従来互換）。
            誘電体では M に ε_r を乗じる（HOM は E 場定式化）。
    """
    if mesh.element_order == 2:
        return assemble_global_matrices_2nd(
            mesh.simplices, mesh.vertices, mesh.edge_index_map,
            mesh.num_edges, mesh.n, eps_r_per_element=eps_r_per_element)
    return assemble_global_matrices_1st(
        mesh.simplices, mesh.vertices, mesh.edge_index_map,
        mesh.num_edges, mesh.n, eps_r_per_element=eps_r_per_element)


def matrix_size_hom(mesh: HOMMeshData) -> int:
    """HOM 全体行列のサイズ（DOF 数）を返す."""
    num_nodes = len(mesh.vertices)
    num_elements = len(mesh.simplices)
    if mesh.element_order == 2:
        return (2 * mesh.num_edges + 2 * num_elements
                + (num_nodes if mesh.n > 0 else 0))
    return mesh.num_edges + (num_nodes if mesh.n > 0 else 0)
