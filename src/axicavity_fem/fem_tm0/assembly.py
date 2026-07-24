"""TM0 全体行列のアセンブリ.

要素行列 (E, n, n) を COO 形式で全体疎行列に組み立てる。
ver1 ``assemble_global_matrix_vectorized_1st/2nd`` の COO 組立部に対応する。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..shared.mesh_loader import MeshData
from .element_matrices import element_matrices_1st, element_matrices_2nd


def assemble_global_matrices(
    mesh: MeshData,
    eps_r_per_element: np.ndarray | None = None,
    n_quad: int = 7,
) -> tuple[csr_matrix, csr_matrix]:
    """メッシュから全体剛性行列 K と質量行列 M を組み立てる.

    Args:
        mesh: :class:`~axicavity_fem.shared.mesh_loader.MeshData`。
        eps_r_per_element: (Ne,) 形状の要素ごと比誘電率 (実数, >0)。
            H_φ 弱形式 ∫(1/ε)(∇w·∇H + wH/r²)r dz dr = ω²∫μwH·r dz dr に
            おいて K に 1/ε、M に μ が入る。ver2 では ε=ε₀, μ=μ₀ を消した
            形になっているため、ver2.1 では K_e に **1/ε_r** を、M_e は μ_r
            のみ (TM0 は μ_r=1 を前提) を乗じる。
            ``None`` のときは ver2 と同じ真空 (eps_r=1) 計算で完全互換。

            数値検証: 半径 a の円筒空洞 TM010 を ε_r 一様にすると、固有値 k²
            は 1/ε_r 倍 → 共振周波数 f は 1/√ε_r 倍になる。
        n_quad: 要素行列の三角形ガウス求積点数 (既定 7, ver2.3 で偽モード抑制の
            ため 4→7 に変更)。ver1 とビット一致させたいときは ``n_quad=4``。

    Returns:
        (K_global, M_global): CSR 形式の疎行列。

    Raises:
        ValueError: ``element_order`` が 1, 2 以外、または
            ``eps_r_per_element`` の長さが要素数と一致しないとき。
    """
    nodes = mesh.nodes
    elements = mesh.elements
    N_nodes = len(nodes)

    if mesh.element_order == 1:
        K_e, M_e = element_matrices_1st(nodes, elements, n_quad=n_quad)
        n_local = 3
    elif mesh.element_order == 2:
        K_e, M_e = element_matrices_2nd(nodes, elements, n_quad=n_quad)
        n_local = 6
    else:
        raise ValueError("element_order は 1 か 2 を指定してください。")

    if eps_r_per_element is not None:
        if eps_r_per_element.shape != (len(elements),):
            raise ValueError(
                f"eps_r_per_element shape {eps_r_per_element.shape} != "
                f"(num_elements={len(elements)},)"
            )
        # K_e は (E, n_local, n_local)。1/ε_r を (E, 1, 1) にブロードキャスト。
        # M_e は eps_r 非依存。
        inv_eps_r = (1.0 / eps_r_per_element).astype(K_e.dtype, copy=False)
        K_e = K_e * inv_eps_r[:, None, None]

    rows = np.repeat(elements, n_local, axis=1).flatten()
    cols = np.tile(elements, (1, n_local)).flatten()

    K_global = coo_matrix(
        (K_e.flatten(), (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
    M_global = coo_matrix(
        (M_e.flatten(), (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()

    return K_global, M_global
