"""
mesh_reader.py
gmsh Python API を用いたメッシュファイルの読み込みとエッジインデックスマップの作成

PhysicalGroup 名による境界条件の分類:
  PEC / E-short (Dirichlet): エッジDOF・節点DOFにディリクレ条件 (=0) を課す
  M-short (PMC)            : 自然境界条件 (何もしない)
  z軸 r=0, n=0             : 常に自然境界条件（PhysicalGroupで誤ってPEC指定されていても除外）
  z軸 r=0, n≥1             : 自動ディリクレ条件（PhysicalGroup指定に関わらず強制付加）
"""

import numpy as np
import gmsh

# ディリクレ条件を課す PhysicalGroup 名のセット (FEM 行列構築用)
PEC_GROUP_NAMES = {"PEC", "Dirichlet", "E-short"}

# P_loss 積分に使う物理的 PEC 壁のみ (E-short は対称境界なので除外)
PEC_LOSS_GROUP_NAMES = {"PEC", "Dirichlet"}


def create_edge_index_map(simplices):
    """要素コーナーノードリストからエッジインデックスマップを作成する

    1次（3節点）・2次（6節点）要素どちらにも使用可能。
    2次要素の場合は simplices[:, :3] を渡してコーナーノードのみを処理すること。

    Args:
        simplices: 要素コーナーノードリスト (num_elements x 3)
                   各行は [n0, n1, n2] のコーナーノードインデックス

    Returns:
        edge_index_map: {tuple(sorted(n_i, n_j)): edge_index}
        edge_count: 総エッジ数
    """
    edge_index_map = {}
    edge_count = 0
    for simplex in simplices:
        edges_local = [
            tuple(sorted((simplex[0], simplex[1]))),
            tuple(sorted((simplex[1], simplex[2]))),
            tuple(sorted((simplex[2], simplex[0]))),
        ]
        for edge in edges_local:
            if edge not in edge_index_map:
                edge_index_map[edge] = edge_count
                edge_count += 1
    return edge_index_map, edge_count


def load_gmsh_mesh_hom(msh_file, n, element_order=1):
    """gmsh API を使ってメッシュを読み込み、HOM FEM 計算用データを返す

    PhysicalGroup 名で境界条件を分類する:
    - PEC / Dirichlet / E-short → ディリクレ条件
    - M-short → 自然境界条件（リストに含めない）
    - z軸 (r=0), n≥1 → 自動ディリクレ条件

    Args:
        msh_file: Gmsh メッシュファイルパス (.msh)
        n: 方位角モード次数
        element_order: 要素次数 (1: 3節点三角形, 2: 6節点三角形)

    Returns:
        simplices:                 要素コーナーノード (num_elements x 3) ← エッジ生成用
        vertices:                  節点座標 (num_nodes x 2) [z, r]
        edge_index_map:            {(n_i, n_j): edge_idx}
        num_edges:                 エッジ数
        boundary_edge_indices:     PEC/E-short 境界エッジインデックスリスト (FEM行列用)
        boundary_vertices_indices: PEC/E-short + z軸 境界節点インデックスリスト (n>0)
        physical_groups:           {name: np.ndarray(node_indices)} 辞書（参照・保存用）
        pec_loss_edge_indices:     物理PEC壁のみの境界エッジインデックスリスト (P_loss積分用)
    """
    # 要素タイプと節点数の設定
    if element_order == 1:
        elem_type = 2       # 3節点三角形
        nodes_per_elem = 3
    elif element_order == 2:
        elem_type = 9       # 6節点三角形
        nodes_per_elem = 6
    else:
        raise ValueError("element_order は 1 か 2 を指定してください。")

    if not gmsh.isInitialized():
        gmsh.initialize()

    try:
        gmsh.open(msh_file)

        # 1. 節点の取得
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        vertices = np.array(nodeCoords).reshape(-1, 3)[:, :2]  # [z, r] のみ
        tag2idx = {tag: i for i, tag in enumerate(nodeTags)}

        # 2. 要素の取得
        _, elemNodeTags = gmsh.model.mesh.getElementsByType(elem_type)
        elements_full = np.array(
            [tag2idx[tag] for tag in elemNodeTags], dtype=int
        ).reshape(-1, nodes_per_elem)

        # simplices: 1次要素は (n_elem, 3)、2次要素は (n_elem, 6)
        # 2次要素のときは midside ノード (列 3-5) も含めて返す
        # エッジ生成・境界判定では simplex[:3] (コーナーのみ) を使用すること
        simplices = elements_full

        # 3. PhysicalGroup の取得
        physical_groups = {}
        for dim, p_tag in gmsh.model.getPhysicalGroups():
            name = (gmsh.model.getPhysicalName(dim, p_tag)
                    or f"PhysicalGroup_{dim}D_{p_tag}")
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, p_tag)

            group_node_tags = []
            for entity_tag in entities:
                nTags, _, _ = gmsh.model.mesh.getNodes(
                    dim, entity_tag, includeBoundary=True)
                group_node_tags.extend(nTags)

            group_node_tags = np.unique(group_node_tags)
            group_node_idx = np.array(
                [tag2idx[tag] for tag in group_node_tags if tag in tag2idx],
                dtype=int)

            if name in physical_groups:
                physical_groups[name] = np.unique(
                    np.concatenate([physical_groups[name], group_node_idx]))
            else:
                physical_groups[name] = group_node_idx

    finally:
        gmsh.finalize()

    # 4. エッジインデックスマップの作成
    edge_index_map, num_edges = create_edge_index_map(simplices)

    # 5. ディリクレ条件ノードセットの構築
    pec_nodes_set = set()
    for name in PEC_GROUP_NAMES:
        if name in physical_groups:
            pec_nodes_set.update(physical_groups[name].tolist())

    # P_loss 用: 物理 PEC 壁のみ (E-short 対称境界を除く)
    pec_loss_nodes_set = set()
    for name in PEC_LOSS_GROUP_NAMES:
        if name in physical_groups:
            pec_loss_nodes_set.update(physical_groups[name].tolist())

    # z軸 (r=0) ノードを識別（n の値に関わらず常に実施）
    z_axis_nodes = {i for i, v in enumerate(vertices) if v[1] < 1e-10}

    z_axis_zmin = z_axis_zmax = vertices[list(z_axis_nodes)[0]][0]
    z_axis_zmin_node = z_axis_zmax_node = list(z_axis_nodes)[0]
    for i in z_axis_nodes :
        if z_axis_zmin > vertices[i][0] :
            z_axis_zmin = vertices[i][0]
            z_axis_zmin_node = i
        if z_axis_zmax < vertices[i][0] :
            z_axis_zmax = vertices[i][0]
            z_axis_zmax_node = i
    
    z_axis_nodes.remove(z_axis_zmin_node)
    z_axis_nodes.remove(z_axis_zmax_node)


    if n == 0:
        # n=0: z軸は自然境界条件
        # PhysicalGroup で誤って PEC/Dirichlet 指定されていても除外する
 
        removed = pec_nodes_set & z_axis_nodes
        if removed:
            print(f"  警告: n=0 では z軸 (r=0) の {len(removed)} ノードを"
                  f"ディリクレ条件から除外します（自然境界条件として扱います）。")
        pec_nodes_set.difference_update(z_axis_nodes)

    else:
        # n≥1: z軸は強制的にディリクレ条件（PhysicalGroup 指定に関わらず）
        pec_nodes_set.update(z_axis_nodes)

    # 6. 境界エッジの判定: 両端がディリクレ条件ノードセットに含まれるエッジ
    boundary_edge_indices = []
    pec_loss_edge_indices = []
    for (v1, v2), edge_idx in edge_index_map.items():
        if v1 in pec_nodes_set and v2 in pec_nodes_set:
            boundary_edge_indices.append(edge_idx)
        if v1 in pec_loss_nodes_set and v2 in pec_loss_nodes_set:
            pec_loss_edge_indices.append(edge_idx)

    # 7. 境界節点の判定 (n > 0 のみ: E_theta の Dirichlet 条件用)
    boundary_vertices_indices = []
    if n > 0:
        boundary_vertices_indices = sorted(list(pec_nodes_set))

    print(f"  Nodes: {len(vertices)}, Elements: {len(simplices)}, "
          f"Edges: {num_edges}")
    print(f"  Physical groups: {list(physical_groups.keys())}")
    print(f"  PEC/Dirichlet boundary edges (FEM): {len(boundary_edge_indices)}")
    print(f"  PEC loss boundary edges (P_loss):   {len(pec_loss_edge_indices)}")
    if n > 0:
        print(f"  E_theta boundary nodes: {len(boundary_vertices_indices)}")

    return (simplices, vertices, edge_index_map, num_edges,
            boundary_edge_indices, boundary_vertices_indices,
            physical_groups, pec_loss_edge_indices)
