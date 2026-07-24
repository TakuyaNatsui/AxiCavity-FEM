"""Gmsh API でメッシュを読み込み、MeshData dataclass を返す.

ver1 ``FEM_code/FEM_helmholtz_TM0_calclation.py:load_gmsh_mesh`` と
``FEM_HOM_code/mesh_reader.py:load_gmsh_mesh_hom`` のメッシュ読込部を統合した版。

境界条件分類 (PEC/E-short/M-short) はここでは行わず、節点・要素・PhysicalGroup
だけを返す。BC 分類は :mod:`axicavity_fem.shared.boundary_groups` が担当する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from .boundary_groups import classify_boundaries
from .edge_index import create_edge_index_map

# 要素次数 → (gmsh 要素タイプ, 1 要素あたり節点数)
_ELEM_TYPE = {
    1: (2, 3),   # 3 節点三角形
    2: (9, 6),   # 6 節点三角形
}


@dataclass
class MeshData:
    """軸対称 2D メッシュデータ.

    Attributes:
        nodes: 節点座標 (N, 2). 各行は ``[z, r]``。
        elements: 要素節点インデックス (E, nodes_per_elem)。
            1 次要素は 3 列、2 次要素は中間節点を含む 6 列。
        element_order: 要素次数 (1 または 2)。
        physical_groups: ``{name: np.ndarray(node_indices)}``。
        element_region_ids: (Ne,) int 配列。各要素が属する 2D Physical Group の
            tag (= region_id)。2D Physical Group が無いメッシュ（ver2 形式）では
            ``None``。
        region_names: ``{region_id: material_tag}``。``element_region_ids`` が
            ``None`` のときは ``None``。
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_order: int
    physical_groups: dict[str, np.ndarray] = field(default_factory=dict)
    element_region_ids: np.ndarray | None = None
    region_names: dict[int, str] | None = None

    @property
    def simplices(self) -> np.ndarray:
        """要素のコーナー節点のみ (E, 3)。エッジ生成・境界判定に使う。"""
        return self.elements[:, :3]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)


def load_mesh(filename: str | Path, element_order: int = 2) -> MeshData:
    """Gmsh メッシュファイルを読み込み :class:`MeshData` を返す.

    節点の並び順は gmsh ``getNodes()`` の返り順に従う（ver1 と同一）。
    全体行列のインデックスはこの並び順に一致するため、ver1 との数値一致が保たれる。

    Args:
        filename: メッシュファイルパス (.msh)。
        element_order: 要素次数 (1: 3 節点三角形, 2: 6 節点三角形)。

    Returns:
        :class:`MeshData`。

    Raises:
        ValueError: ``element_order`` が 1, 2 以外のとき。
    """
    if element_order not in _ELEM_TYPE:
        raise ValueError("element_order は 1 か 2 を指定してください。")
    elem_type, nodes_per_elem = _ELEM_TYPE[element_order]

    # 毎回フレッシュな gmsh 状態で読む（前段の状態が 2 次曲線メッシュの
    # 再パラメータ化に影響して結果が変わるのを防ぐ）。
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.open(str(filename))

        # 節点 (2 次要素の中間節点も自動で取得される)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(node_coords).reshape(-1, 3)[:, :2]
        tag2idx = {tag: i for i, tag in enumerate(node_tags)}

        # 要素
        elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
        elements = np.array(
            [tag2idx[tag] for tag in elem_node_tags], dtype=int
        ).reshape(-1, nodes_per_elem)

        # PhysicalGroup → 節点インデックス集合
        physical_groups: dict[str, np.ndarray] = {}
        for dim, p_tag in gmsh.model.getPhysicalGroups():
            name = (gmsh.model.getPhysicalName(dim, p_tag)
                    or f"PhysicalGroup_{dim}D_{p_tag}")
            group_node_tags: list[int] = []
            for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(dim, p_tag):
                n_tags, _, _ = gmsh.model.mesh.getNodes(
                    dim, entity_tag, includeBoundary=True)
                group_node_tags.extend(n_tags)

            group_node_tags = np.unique(group_node_tags)
            group_node_idx = np.array(
                [tag2idx[t] for t in group_node_tags if t in tag2idx], dtype=int)

            if name in physical_groups:
                physical_groups[name] = np.unique(
                    np.concatenate([physical_groups[name], group_node_idx]))
            else:
                physical_groups[name] = group_node_idx

        # 2D Physical Group → element ごとの region_id マッピング
        element_region_ids, region_names = _build_element_region_map(
            elem_tags, elem_type,
        )
    finally:
        gmsh.finalize()

    return MeshData(
        nodes=nodes,
        elements=elements,
        element_order=element_order,
        physical_groups=physical_groups,
        element_region_ids=element_region_ids,
        region_names=region_names,
    )


def _build_element_region_map(
    elem_tags: np.ndarray, elem_type: int,
) -> tuple[np.ndarray | None, dict[int, str] | None]:
    """各要素の所属 2D Physical Group を解決する。

    2D Physical Group が 1 つも無ければ ``(None, None)`` を返す（ver2 形式の
    単一領域メッシュとの後方互換）。

    Args:
        elem_tags: ``getElementsByType(elem_type)`` で得た全要素タグ列。
        elem_type: gmsh 要素タイプ整数。

    Returns:
        (element_region_ids, region_names)
            - element_region_ids: (Ne,) int 配列。属する Physical Group tag。
              所属無しの要素は -1。
            - region_names: ``{region_id: material_tag}``。
    """
    p_groups_2d = gmsh.model.getPhysicalGroups(dim=2)
    if not p_groups_2d:
        return None, None

    elem_tag_to_pgroup: dict[int, int] = {}
    region_names: dict[int, str] = {}
    for _dim, p_tag in p_groups_2d:
        name = (gmsh.model.getPhysicalName(2, p_tag)
                or f"PhysicalGroup_2D_{p_tag}")
        region_names[int(p_tag)] = name
        for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(2, p_tag):
            sub_tags, _ = gmsh.model.mesh.getElementsByType(elem_type, entity_tag)
            for et in sub_tags:
                elem_tag_to_pgroup[int(et)] = int(p_tag)

    element_region_ids = np.full(len(elem_tags), -1, dtype=int)
    for i, et in enumerate(elem_tags):
        pg = elem_tag_to_pgroup.get(int(et))
        if pg is not None:
            element_region_ids[i] = pg

    return element_region_ids, region_names


@dataclass
class HOMMeshData:
    """HOM ソルバ用のメッシュデータ（エッジ + 境界 DOF 分類込み）.

    Attributes:
        simplices: 要素節点インデックス (E, 3 or 6)。コーナーは ``[:, :3]``。
        vertices: 節点座標 (N, 2) = [z, r]。
        edge_index_map: ``{tuple(sorted(corner_nodes)): edge_index}``。
        num_edges: 総エッジ数。
        boundary_edge_indices: PEC/E-short 境界エッジ（FEM Dirichlet 用）。
        boundary_vertices_indices: PEC/E-short + 軸 の境界節点（n>0 の Dirichlet 用）。
        physical_groups: PhysicalGroup 辞書。
        pec_loss_edge_indices: 物理 PEC 壁のみの境界エッジ（P_loss 用、E-short 除外）。
        element_order: 要素次数 (1 or 2)。
        n: 方位角モード次数。
        element_region_ids: (Ne,) 各要素が属する 2D Physical Group tag。多領域
            （誘電体）メッシュのみ。単一領域メッシュでは ``None``（真空扱い）。
        region_names: ``{region_id: material_tag}``。``None`` なら真空扱い。
    """

    simplices: np.ndarray
    vertices: np.ndarray
    edge_index_map: dict
    num_edges: int
    boundary_edge_indices: list
    boundary_vertices_indices: list
    physical_groups: dict
    pec_loss_edge_indices: list
    element_order: int
    n: int
    element_region_ids: np.ndarray | None = None
    region_names: dict[int, str] | None = None


def compute_hom_boundary(vertices, simplices, edge_index_map, physical_groups,
                         n):
    """HOM の境界 DOF 分類を行う（メッシュ読込と独立、H5 からの再計算にも使う）.

    Returns:
        (boundary_edge_indices, boundary_vertices_indices,
         pec_loss_edge_indices)。
    """
    classification = classify_boundaries(physical_groups, vertices)
    # HOM の FEM Dirichlet: PEC ∪ E-short（PEC は Dirichlet エイリアスを含む）
    pec_nodes_set = set(int(i) for i in classification.pec_nodes)
    pec_nodes_set |= set(int(i) for i in classification.eshort_nodes)
    # P_loss 用: 物理 PEC 壁のみ（E-short 対称境界を除く）
    pec_loss_nodes_set = set(int(i) for i in classification.pec_nodes)

    # 軸 r=0 ノードの処理（ver1 と同じ閾値・両端コーナー除外）
    z_axis_nodes = {i for i, v in enumerate(vertices) if v[1] < 1e-10}
    if z_axis_nodes:
        axis_list = list(z_axis_nodes)
        zmin_node = zmax_node = axis_list[0]
        z_min = z_max = vertices[axis_list[0]][0]
        for i in z_axis_nodes:
            if vertices[i][0] < z_min:
                z_min, zmin_node = vertices[i][0], i
            if vertices[i][0] > z_max:
                z_max, zmax_node = vertices[i][0], i
        z_axis_nodes.discard(zmin_node)
        z_axis_nodes.discard(zmax_node)
        if n == 0:
            pec_nodes_set.difference_update(z_axis_nodes)
        else:
            pec_nodes_set.update(z_axis_nodes)

    boundary_edge_indices = []
    pec_loss_edge_indices = []
    for (v1, v2), edge_idx in edge_index_map.items():
        if v1 in pec_nodes_set and v2 in pec_nodes_set:
            boundary_edge_indices.append(edge_idx)
        if v1 in pec_loss_nodes_set and v2 in pec_loss_nodes_set:
            pec_loss_edge_indices.append(edge_idx)

    boundary_vertices_indices = sorted(pec_nodes_set) if n > 0 else []
    return (boundary_edge_indices, boundary_vertices_indices,
            pec_loss_edge_indices)


def load_mesh_hom(filename: str, n: int, element_order: int = 1) -> HOMMeshData:
    """HOM 計算用にメッシュを読み込み、エッジ・境界 DOF 分類を行う.

    ver1 ``FEM_HOM_code/mesh_reader.py:load_gmsh_mesh_hom`` を共通基盤
    （:func:`load_mesh` + :func:`create_edge_index_map` + :func:`classify_boundaries`）
    の上に再構成した版。

    HOM の境界規約（DOF: Nédélec エッジ + 節点）:
      - PEC / E-short（+ Dirichlet エイリアス）→ エッジ・節点ともに Dirichlet (=0)
      - M-short → 自然境界（何もしない）
      - 軸 r=0, n=0 → 自然境界（PEC 指定されていても除外）
      - 軸 r=0, n≥1 → 強制 Dirichlet

    Args:
        filename: メッシュファイルパス (.msh)。
        n: 方位角モード次数。
        element_order: 要素次数 (1 or 2)。

    Returns:
        :class:`HOMMeshData`。
    """
    mesh = load_mesh(filename, element_order)
    vertices = mesh.nodes
    simplices = mesh.elements
    physical_groups = mesh.physical_groups

    edge_index_map, num_edges = create_edge_index_map(simplices[:, :3])

    boundary_edge_indices, boundary_vertices_indices, pec_loss_edge_indices = \
        compute_hom_boundary(vertices, simplices, edge_index_map,
                             physical_groups, n)

    return HOMMeshData(
        simplices=simplices,
        vertices=vertices,
        edge_index_map=edge_index_map,
        num_edges=num_edges,
        boundary_edge_indices=boundary_edge_indices,
        boundary_vertices_indices=boundary_vertices_indices,
        physical_groups=physical_groups,
        pec_loss_edge_indices=pec_loss_edge_indices,
        element_order=element_order,
        n=n,
        element_region_ids=mesh.element_region_ids,
        region_names=mesh.region_names,
    )
