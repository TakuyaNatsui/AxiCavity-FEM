import numpy as np

def calculate_E_theta_from_rE_theta(rE_theta_values, vertices, n, simplices):
    """
    r*E_theta の自由度ベクトルから物理的な E_theta を計算する。
    n=1 の場合は r=0 の節点で補間を行う。
    """
    num_nodes = len(vertices)
    # 入力が None またはサイズが不一致の場合のエラーチェックを追加
    if rE_theta_values is None or len(rE_theta_values) != num_nodes:
        # n=0 の場合など、節点自由度がない場合は None が渡される可能性がある
        # またはサイズ不一致エラー
        # この場合は E_theta も存在しないか計算できない
        if n > 0 : # n>0 なのに rE_theta がないのはおかしい
             raise ValueError("rE_theta_values size mismatch or None for n > 0.")
        else: # n=0なら妥当
             return None # E_theta は存在しない

    # dtype を入力に合わせる (実数 or 複素数)
    dtype = float if np.isrealobj(rE_theta_values) else complex
    E_theta_values = np.zeros(num_nodes, dtype=dtype)
    r_coords = vertices[:, 1]

    # r > 0 の節点では単純に割る
    non_zero_r = np.where(r_coords > 1e-9)[0]
    # Check if rE_theta_values[non_zero_r] is actually subscriptable if it could be None
    if rE_theta_values is not None:
        E_theta_values[non_zero_r] = rE_theta_values[non_zero_r] / r_coords[non_zero_r]

    # n=1 の場合のみ r=0 で補間
    if n == 1:
        axis_node_indices = np.where(np.abs(r_coords) < 1e-9)[0]
        if len(axis_node_indices) > 0:
            # node_to_elements マップ作成
            node_to_elements = [[] for _ in range(num_nodes)]
            for elem_idx, simplex in enumerate(simplices):
                for node_idx in simplex: node_to_elements[node_idx].append(elem_idx)

            # 補間実行
            for i in axis_node_indices:
                neighbor_vals = []
                neighbor_nodes = set()
                for elem_idx in node_to_elements[i]:
                    simplex = simplices[elem_idx]
                    for k in simplex:
                        # 割り算後の E_theta 値を使う
                        if k != i and abs(vertices[k, 1]) > 1e-9:
                             if k not in neighbor_nodes:
                                 neighbor_nodes.add(k)
                                 neighbor_vals.append(E_theta_values[k]) # 割った後の値
                if neighbor_vals:
                    E_theta_values[i] = np.mean(np.array(neighbor_vals), axis=0)
                # else: 補間できない場合は 0 のまま

    return E_theta_values



def save_mode_to_hdf5(f, mode_path, mode_index, eigenvalue, x_vector, n,
                      vertices, simplices, num_edges, is_periodic,
                      elem_order=1):
    """
    単一モードの計算結果を指定されたHDF5パスに保存する。

    elem_order=1 (1次要素):
        DOF配列: [edge(num_edges), node(num_nodes, n>0のみ)]
        保存データセット:
            edge_vectors[_re/_im]  : エッジDOF
            E_theta[_re/_im]       : E_theta 節点値 (n>0)

    elem_order=2 (2次要素):
        DOF配列: [CT/LN(2e), LT/LN(2e+1) × num_edges, face × 2*num_elem, node(n>0)]
        保存データセット:
            edge_vectors[_re/_im]    : CT/LN エッジDOF (num_edges値)
            edge_vectors_lt[_re/_im] : LT/LN エッジDOF (num_edges値)
            face_vectors[_re/_im]    : 面内DOF (2*num_elements値)
            E_theta[_re/_im]         : E_theta 節点値 (n>0)
        属性: elem_order=2, num_elements

    Args:
        f (h5py.File): 開いているHDF5ファイルオブジェクト
        mode_path (str): HDF5内のグループパス
        mode_index (int): モード番号 (0, 1, 2, ...)
        eigenvalue (float): 固有値 (k^2)
        x_vector (np.ndarray): 固有ベクトル
        n (int): 方位角モード番号
        vertices (np.ndarray): 節点座標
        simplices (np.ndarray): 要素リスト (1次: n×3, 2次: n×6)
        num_edges (int): エッジ数
        is_periodic (bool): 周期境界条件計算かどうか
        elem_order (int): 要素次数 (1 or 2, デフォルト 1)
    """
    num_nodes    = len(vertices)
    num_elements = len(simplices)

    # DOF 構造の定義
    if elem_order == 2:
        face_offset = 2 * num_edges
        node_offset = 2 * num_edges + 2 * num_elements
        N_original  = node_offset + (num_nodes if n > 0 else 0)
    else:
        N_original  = num_edges + (num_nodes if n > 0 else 0)

    mode_group_name = f"mode_{mode_index}"
    mode_group_path = f"{mode_path}/{mode_group_name}"
    print(f"  Saving {mode_group_path} ...")

    f.require_group(mode_path)
    if mode_group_path in f:
        print(f"  Warning: {mode_group_path} already exists. Overwriting.")
    mode_group = f.require_group(mode_group_path)

    # 共通属性
    c0 = 299792458.0
    freq_GHz = (np.sqrt(eigenvalue) * c0 / (2 * np.pi) / 1e9
                if eigenvalue >= 0 else 0.0)
    mode_group.attrs["eigenvalue_k2"]  = eigenvalue
    mode_group.attrs["frequency_GHz"]  = freq_GHz
    mode_group.attrs["elem_order"]     = elem_order
    if elem_order == 2:
        mode_group.attrs["num_elements"] = num_elements

    def _extract_dofs(vec):
        """DOF配列を (CT/LN エッジ, LT/LN エッジ, face, node) に分解する。"""
        if elem_order == 2:
            ct    = vec[0:face_offset:2]          # CT/LN, num_edges 個
            lt    = vec[1:face_offset:2]          # LT/LN, num_edges 個
            face  = vec[face_offset:node_offset]  # 2*num_elements 個
            node  = vec[node_offset:] if n > 0 else None
        else:
            ct    = vec[:num_edges]
            lt    = None
            face  = None
            node  = vec[num_edges:] if n > 0 else None
        return ct, lt, face, node

    def _save_datasets(mg, suffix, ct, lt, face, rE_theta_vec):
        """データセットを保存するヘルパー。suffix は '' / '_re' / '_im'。"""
        mg.create_dataset(f"edge_vectors{suffix}", data=ct, compression="gzip")
        if lt is not None:
            mg.create_dataset(f"edge_vectors_lt{suffix}", data=lt,
                              compression="gzip")
        if face is not None:
            mg.create_dataset(f"face_vectors{suffix}", data=face,
                              compression="gzip")
        if n > 0 and rE_theta_vec is not None:
            E_theta = calculate_E_theta_from_rE_theta(
                rE_theta_vec, vertices, n, simplices)
            mg.create_dataset(f"E_theta{suffix}", data=E_theta,
                              compression="gzip")

    # --- データセット保存 ---
    if is_periodic:
        # 周期境界: 複素ベクトル (新方式) または 2N実数ベクトル (旧方式)
        if np.iscomplexobj(x_vector):
            if len(x_vector) != N_original:
                raise ValueError(
                    f"Complex periodic eigenvector size mismatch: "
                    f"len={len(x_vector)}, expected={N_original}")
            ct_re, lt_re, face_re, node_re = _extract_dofs(x_vector.real)
            ct_im, lt_im, face_im, node_im = _extract_dofs(x_vector.imag)
        else:
            # 旧方式 (互換性維持): 2N次元実数ベクトル [Re, Im]
            N_old = num_edges + (num_nodes if n > 0 else 0)
            if len(x_vector) != 2 * N_old:
                raise ValueError(
                    f"Periodic eigenvector size mismatch (legacy 2N): "
                    f"len={len(x_vector)}, expected={2*N_old}")
            ct_re = x_vector[:num_edges]
            ct_im = x_vector[N_old:N_old + num_edges]
            lt_re = lt_im = face_re = face_im = None
            node_re = x_vector[num_edges:N_old] if n > 0 else None
            node_im = x_vector[N_old+num_edges:] if n > 0 else None

        _save_datasets(mode_group, "_re", ct_re, lt_re, face_re, node_re)
        _save_datasets(mode_group, "_im", ct_im, lt_im, face_im, node_im)

    else:
        # 定在波: 実数ベクトル
        if len(x_vector) != N_original:
            raise ValueError(
                f"Normal eigenvector size mismatch: "
                f"len={len(x_vector)}, expected={N_original}")
        ct, lt, face, node = _extract_dofs(x_vector)
        _save_datasets(mode_group, "", ct, lt, face, node)


def save_mesh_and_params_to_hdf5(f, vertices, simplices, edge_index_map, params_dict):
     """メッシュ情報と計算パラメータをHDF5ファイルに保存する。"""
     # メッシュ保存
     mesh_group = f.require_group("mesh")
     mesh_group.create_dataset("vertices", data=vertices)
     mesh_group.create_dataset("simplices", data=simplices)
     # edge_map はキーと値に分割して保存
     edge_map_keys_array = np.array(list(edge_index_map.keys()), dtype=int)
     edge_map_values_array = np.array(list(edge_index_map.values()), dtype=int)
     mesh_group.create_dataset("edge_map_keys", data=edge_map_keys_array)
     mesh_group.create_dataset("edge_map_values", data=edge_map_values_array)
     mesh_group.attrs["num_nodes"] = len(vertices)
     mesh_group.attrs["num_edges"] = len(edge_index_map)
     print("Mesh data saved.")

     # パラメータ保存 (属性として)
     param_group = f.require_group("parameters")
     for key, value in params_dict.items():
         # HDF5属性はリストや複雑な型を直接保存できない場合がある
         if isinstance(value, (list, tuple)) and all(isinstance(i, (int, float, str, bool)) for i in value):
            # 単純なリストは NumPy 配列にしてデータセットとして保存も可能
            # param_group.create_dataset(key, data=np.array(value))
            # または、属性として保存するなら文字列にするなど工夫が必要
            param_group.attrs[key] = str(value) # 例: 文字列化
         elif isinstance(value, (int, float, str, bool, np.number)):
             param_group.attrs[key] = value
         else:
             print(f"Warning: Cannot save parameter '{key}' of type {type(value)} as HDF5 attribute. Converting to string.")
             param_group.attrs[key] = str(value)
     print("Parameters saved.")

def save_boundaries_to_hdf5(f, bc_path, pec_edges, pec_nodes=None, pbc_edges=None, pbc_nodes=None,
                            pec_loss_edges=None):
    """境界条件情報を指定されたパスに保存する"""
    bc_group = f.require_group(bc_path)
    bc_group.create_dataset("boundary_edges_pec", data=np.array(pec_edges, dtype=int))
    if pec_nodes is not None:
        bc_group.create_dataset("boundary_nodes_pec", data=np.array(pec_nodes, dtype=int))
    if pbc_edges is not None:
        bc_group.create_dataset("pbc_edge_pairs", data=np.array(pbc_edges, dtype=float))
    if pbc_nodes is not None:
        bc_group.create_dataset("pbc_node_pairs", data=np.array(pbc_nodes, dtype=int))
    if pec_loss_edges is not None:
        bc_group.create_dataset("boundary_edges_pec_loss", data=np.array(pec_loss_edges, dtype=int))
    print(f"Boundary conditions saved to {bc_path}")