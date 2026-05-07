import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker # ticker モジュールをインポート

import os
import imageio

from FEM_element_function import calculate_triangle_area_double, calculate_area_coordinates, calculate_edge_shape_functions

# プロジェクトルートを sys.path に追加して plot_common を解決
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from plot_common import STYLE  # noqa: E402

# 必要な場合 get_edge_orientation_sign
def get_edge_orientation_sign(global1, global2):
    if global1 < global2 : return 1
    else : return -1

# --- 新規: 要素中心座標を計算する関数 ---
def calculate_element_center(vertices_element):
    """三角形要素の中心座標を計算する関数

    Args:
        vertices_element (numpy.ndarray): 三角形要素の頂点座標 (3x2 array)

    Returns:
        numpy.ndarray: 要素中心座標 (1x2 array)
    """
    return np.mean(vertices_element, axis=0) # 各頂点座標の平均


# --- visualize_eigenmode 関数 (境界辺ハイライト表示追加版) ---
def visualize_eigenmode(eigenvector, simplices, vertices, edge_index_map, boundary_edge_indices_pec, eigenmode_index, eigenvalue):
    """固有モードをベクトルで可視化する関数 (要素中心でベクトル表示, 境界辺ハイライト表示)

    Args:
        eigenvector (numpy.ndarray): 固有ベクトル (1次元配列)
        vertices (numpy.ndarray): 頂点座標 (Px2 array)
        simplices (numpy.ndarray): 三角形要素の頂点インデックス (Mx3 array)
        edge_index_map (dict): 辺インデックスマップ (辞書型)
        boundary_edge_indices_pec (list): PEC境界辺のグローバルインデックスのリスト
        eigenmode_index (int): 固有モードのインデックス (グラフタイトル用)
        eigenvalue (complex): 固有値 (グラフタイトル用)
    """

    #print(simplices) ######################
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.triplot(vertices[:, 0], vertices[:, 1], simplices)
    #plt.triplot(vertices[:, 0], vertices[:, 1], simplices, linewidth=0.5, color='black', label='Mesh Edges') # メッシュの辺 (凡例用)

    # PEC境界辺を黒色で強調表示 (TM0 スタイル統一)
    for edge_index in boundary_edge_indices_pec:
        for edge, index in edge_index_map.items(): # 辺インデックスマップから辺情報を検索
            if index == edge_index:
                vertex_indices = edge # 境界辺の頂点インデックス
                vertex1 = vertices[vertex_indices[0]] # 頂点1座標
                vertex2 = vertices[vertex_indices[1]] # 頂点2座標
                plt.plot([vertex1[0], vertex2[0]], [vertex1[1], vertex2[1]],
                         color=STYLE.BORDER_COLOR, linewidth=STYLE.BORDER_LW_STRONG, zorder=10,
                         label='PEC Boundary Edges' if edge_index == boundary_edge_indices_pec[0] else "")

    # 要素中心にベクトルを表示 (以前と同じ)
    element_centers = []
    element_vectors = []
    for simplex in simplices:
        vertices_element = vertices[simplex]
        element_center = calculate_element_center(vertices_element)
        element_centers.append(element_center)
        E_element = np.zeros(3)
        edge_vertices_local = [
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[2], simplex[0]),
        ]
        for i, edge_vertex_local in enumerate(edge_vertices_local):
            edge_global_index = edge_index_map[tuple(sorted(edge_vertex_local))]
            #orientation_sign = get_edge_orientation_sign(simplex, edge_vertex_local[0], edge_vertex_local[1])
            orientation_sign = get_edge_orientation_sign(simplex[i], simplex[(i+1)%3])
            E_element[i] = orientation_sign * eigenvector[edge_global_index].real
        N1, N2, N3 = calculate_edge_shape_functions(element_center, vertices_element)
        element_vector = E_element[0] * N1 + E_element[1] * N2 + E_element[2] * N3
        element_vectors.append(element_vector)

    element_centers_array = np.array(element_centers)
    element_vectors_array = np.array(element_vectors)*0.5
    plt.quiver(
        element_centers_array[:, 0], element_centers_array[:, 1],
        element_vectors_array[:, 0], element_vectors_array[:, 1],
        angles='xy', scale_units='xy', scale=5,
        color=STYLE.QUIVER_COLOR, alpha=STYLE.QUIVER_ALPHA,
        label='Eigenmode Vector Field'
    )

    # --- 頂点番号をプロット ---
    for i, vertex in enumerate(vertices):
        plt.text(vertex[0], vertex[1], str(i), color='black', fontsize=8, ha='left', va='bottom') # 頂点座標に番号をテキストで表示

    frequency = np.sqrt(eigenvalue.real) *299792458 /(2*np.pi)
    freqGHz = frequency/1e9
    #plt.title(f'Resonance Mode {eigenmode_index+1} (Eigenvalue: {eigenvalue.real:.4f}) - Vector Field with PEC Boundary') # タイトル変更
    plt.title(f'Resonance Mode {eigenmode_index+1} (frequency[GHz]: {freqGHz:.4f}) - Vector Field with PEC Boundary') # タイトル変更
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend() # 凡例表示
    plt.show()




# --- 新しい可視化関数 (実部・虚部分離表示) ---
def visualize_complex_eigenmode_re_im(
    x_complex,           # 複素固有ベクトル (サイズ N_original)
    eigenvalue,          # 固有値 (k^2)
    theta_pbc,           # 周期境界の位相差 theta
    eigenmode_index,     # モード番号 (プロットタイトル用)
    n,                   # 方位角モード番号
    vertices,            # 節点座標 [z, r] (num_nodes x 2)
    simplices,           # 要素リスト (num_elements x 3)
    edge_index_map,      # エッジマップ
    num_edges,           # エッジ数
    boundary_edges_pec,  # PEC エッジインデックスリスト
    # boundary_nodes_pec, # PEC 節点インデックスリスト (E_theta 用、必要なら追加)
    vec_scale=5.0,       # ベクトル表示のスケール
    vec_width=0.003,     # ベクトル表示の幅
    vec_pivot='mid',     # ベクトルの基点 ('mid', 'tail')
    cmap_scalar='viridis' # スカラー場表示用カラーマップ
):
    """
    複素固有モードを可視化する (進行波用) - 実部と虚部を分離表示。
    Er, Ez ベクトル場と、(n>0の場合) E_theta スカラー場を表示する。
    """
    N_original = len(x_complex)
    num_nodes = len(vertices)
    if N_original != num_edges + (num_nodes if n > 0 else 0):
         raise ValueError("Size of x_complex does not match num_edges and num_nodes.")

    # --- データの分離 ---
    edge_dofs_complex = x_complex[:num_edges]
    node_dofs_complex = None
    E_theta_complex_nodes = None
    if n > 0:
        node_dofs_complex = x_complex[num_edges:] # rE_theta
        r_coords = vertices[:, 1]
        E_theta_complex_nodes = np.zeros_like(node_dofs_complex, dtype=complex)
        non_zero_r_indices = np.where(r_coords > 1e-9)[0]
        E_theta_complex_nodes[non_zero_r_indices] = node_dofs_complex[non_zero_r_indices] / r_coords[non_zero_r_indices]

    # --- 電場ベクトルの計算 (要素中心) ---
    element_centers = []
    E_vec_complex_elements = [] # [Ez, Er]
    for elem_idx, simplex in enumerate(simplices):
        vertices_element = vertices[simplex]
        element_center = calculate_element_center(vertices_element)
        element_centers.append(element_center)
        E_element_complex = np.zeros(3, dtype=complex)
        edge_vertices_local_nodes = [
            (simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])
        ]
        for i, (n1, n2) in enumerate(edge_vertices_local_nodes):
            global_edge_key = tuple(sorted((n1, n2)))
            if global_edge_key in edge_index_map:
                edge_global_index = edge_index_map[global_edge_key]
                orientation_sign = get_edge_orientation_sign(n1, n2)
                E_element_complex[i] = orientation_sign * edge_dofs_complex[edge_global_index]
            else:
                 print(f"Warning: Edge ({n1},{n2}) not in edge_index_map for element {elem_idx}")
        N1, N2, N3 = calculate_edge_shape_functions(element_center, vertices_element)
        vec_complex = E_element_complex[0] * N1 + E_element_complex[1] * N2 + E_element_complex[2] * N3
        E_vec_complex_elements.append(vec_complex)

    element_centers_array = np.array(element_centers)
    E_vec_complex_array = np.array(E_vec_complex_elements)
    Ez_complex = E_vec_complex_array[:, 0]
    Er_complex = E_vec_complex_array[:, 1]

    # --- プロット設定 ---
    num_plots_per_component = 2 # Real and Imaginary
    num_components = 1 + (1 if n > 0 else 0) # Er/Ez + E_theta
    fig, axes = plt.subplots(num_components, num_plots_per_component,
                             figsize=(6 * num_plots_per_component, 5 * num_components),
                             squeeze=False)

    # --- 共通描画関数 ---
    def draw_mesh_and_pec(ax):
        tri = Triangulation(vertices[:, 0], vertices[:, 1], simplices)
        ax.set_aspect('equal', adjustable='box')
        ax.triplot(tri, linewidth=STYLE.MESH_LW,
                   color=STYLE.MESH_COLOR, alpha=STYLE.MESH_ALPHA)
        pec_label_added = False
        for edge_index in boundary_edges_pec:
            for edge, index in edge_index_map.items():
                if index == edge_index:
                    v_idx = edge
                    label = 'PEC Boundary' if not pec_label_added else ""
                    ax.plot(vertices[v_idx, 0], vertices[v_idx, 1],
                            color=STYLE.BORDER_COLOR,
                            linewidth=STYLE.BORDER_LW_STRONG,
                            zorder=10, label=label)
                    pec_label_added = True
                    break
        ax.set_xlabel(STYLE.LABEL_Z)
        ax.set_ylabel(STYLE.LABEL_R)
        ax.grid(True, linestyle=STYLE.GRID_LS, alpha=STYLE.GRID_ALPHA)

    # --- Er, Ez ベクトル場の描画 ---
    ax_re_vec = axes[0, 0]
    ax_im_vec = axes[0, 1]

    # 実部
    draw_mesh_and_pec(ax_re_vec)
    # 実部のベクトルスケールのためのキーを追加（オプション）
    max_re_val = np.max(np.sqrt(np.real(Ez_complex)**2 + np.real(Er_complex)**2))

    q_re = ax_re_vec.quiver(element_centers_array[:, 0], element_centers_array[:, 1],
                      np.real(Ez_complex), np.real(Er_complex), angles='xy', scale_units='xy',
                      scale=vec_scale*max_re_val, width=vec_width, pivot=vec_pivot,
                      color=STYLE.QUIVER_COLOR, alpha=STYLE.QUIVER_ALPHA)
    
    ax_re_vec.set_title('Re(Ez, Er)')
    print('max_re_val:', max_re_val, 'vec_scale:', vec_scale)
    #if max_re_val > 1e-9 : # Avoid plotting if field is zero
    #    scale_key = 1/(max_re_val * vec_scale) / 5.0 # Example scale key length
    #    ax_re_vec.quiverkey(q_re, 0.8, 0.9, scale_key, f'{scale_key:.2e}', labelpos='E', coordinates='axes')



    # 虚部
    # 虚部のベクトルスケールのためのキーを追加（オプション）
    max_im_val = np.max(np.sqrt(np.imag(Ez_complex)**2 + np.imag(Er_complex)**2))
    draw_mesh_and_pec(ax_im_vec)
    q_im = ax_im_vec.quiver(element_centers_array[:, 0], element_centers_array[:, 1],
                      np.imag(Ez_complex), np.imag(Er_complex), angles='xy', scale_units='xy',
                      scale=vec_scale*max_im_val, width=vec_width, pivot=vec_pivot,
                      color=STYLE.QUIVER_IMAG_COLOR, alpha=STYLE.QUIVER_ALPHA)
    ax_im_vec.set_title('Im(Ez, Er)')
    #if max_im_val > 1e-9 :
    #    scale_key_im = 1/(max_im_val * vec_scale) / 5.0
    #    ax_im_vec.quiverkey(q_im, 0.8, 0.9, scale_key_im, f'{scale_key_im:.2e}', labelpos='E', coordinates='axes')


    # --- r=0 節点における E_theta 値の補間 n=1の場合のみ行う (後処理) ---
    if n == 1 and E_theta_complex_nodes is not None:
        print("Interpolating E_theta at r=0 nodes...")
        axis_node_indices = np.where(np.abs(vertices[:, 1]) < 1e-9)[0]
        print(f"  Found {len(axis_node_indices)} nodes on r=0 axis.")

        # 節点iがどの要素に含まれるかのリストを作成 (事前計算推奨だがここで作成)
        node_to_elements = [[] for _ in range(num_nodes)]
        for elem_idx, simplex in enumerate(simplices):
            for node_idx in simplex:
                node_to_elements[node_idx].append(elem_idx)

        E_theta_interpolated_values = {} # 補間値を一時保存

        for i in axis_node_indices:
            neighbor_e_theta_values = []
            neighbor_node_indices = set() # 重複カウントを防ぐ

            # 節点iが属する全要素を調べる
            for elem_idx in node_to_elements[i]:
                simplex = simplices[elem_idx]
                # 要素内の他の節点 (k) を調べる
                for k in simplex:
                    if k != i and abs(vertices[k, 1]) > 1e-9: # r>0 の隣接節点
                        if k not in neighbor_node_indices:
                             neighbor_node_indices.add(k)
                             # E_theta_complex_nodes[k] は既に計算済み
                             neighbor_e_theta_values.append(E_theta_complex_nodes[k])

            if neighbor_e_theta_values: # 隣接する r>0 の節点が見つかった場合
                 # 複素数の平均値を計算
                 mean_e_theta = np.mean(np.array(neighbor_e_theta_values), axis=0)
                 E_theta_interpolated_values[i] = mean_e_theta
                 # print(f"  Node {i} (r=0): Interpolated E_theta from {len(neighbor_e_theta_values)} neighbors.")
            else:
                 # 孤立した r=0 節点など、隣接 r>0 節点がない場合
                 # print(f"  Node {i} (r=0): No r>0 neighbors found for interpolation. Keeping E_theta=0.")
                 E_theta_interpolated_values[i] = 0.0 + 0.0j # 0のままにする

        # 計算した補間値を E_theta_complex_nodes に反映
        for node_idx, value in E_theta_interpolated_values.items():
            E_theta_complex_nodes[node_idx] = value
        print("  Interpolation finished.")


    # --- E_theta 成分の描画 (n > 0) ---
    if n > 0 and E_theta_complex_nodes is not None:
        ax_re_theta = axes[1, 0]
        ax_im_theta = axes[1, 1]
        tri = Triangulation(vertices[:, 0], vertices[:, 1], simplices)

        # 実部
        draw_mesh_and_pec(ax_re_theta)
        E_theta_real = np.real(E_theta_complex_nodes)
        # カラーマップの範囲を決定（実部と虚部で合わせると比較しやすい）
        vmax = max(abs(np.min(E_theta_real)), abs(np.max(E_theta_real)), 1e-9) # Avoid zero range
        vmin = -vmax 
        norm_scalar = mcolors.Normalize(vmin=vmin, vmax=vmax)
        contour_re = ax_re_theta.tricontourf(tri, E_theta_real, cmap='jet', norm=norm_scalar, levels=100) #cmap='jet' or 'RdBu_r'
        cbar_re = fig.colorbar(contour_re, ax=ax_re_theta, fraction=0.046, pad=0.04)
        cbar_re.set_label(r'Re($E_\theta$)')
        ax_re_theta.set_title(r'Re($E_\theta$)')

        # 虚部
        draw_mesh_and_pec(ax_im_theta)
        E_theta_imag = np.imag(E_theta_complex_nodes)
        vmax_im = max(abs(np.min(E_theta_imag)), abs(np.max(E_theta_imag)), 1e-9)
        vmax_both = max(vmax, vmax_im) # Make ranges symmetric and equal if desired
        vmin_both = -vmax_both
        norm_scalar_both = mcolors.Normalize(vmin=vmin_both, vmax=vmax_both)
        # Redraw real part with potentially updated range
        for c in contour_re.collections: c.remove() # Remove old contour
        contour_re = ax_re_theta.tricontourf(tri, E_theta_real, cmap='jet', norm=norm_scalar_both, levels=100)
        cbar_re.mappable.set_norm(norm_scalar_both) # Update colorbar norm

        # Draw imaginary part
        contour_im = ax_im_theta.tricontourf(tri, E_theta_imag, cmap='jet', norm=norm_scalar_both, levels=100)
        cbar_im = fig.colorbar(contour_im, ax=ax_im_theta, fraction=0.046, pad=0.04)
        cbar_im.set_label(r'Im($E_\theta$)')
        ax_im_theta.set_title(r'Im($E_\theta$)')


    # --- 全体のタイトル ---
    frequency = np.sqrt(eigenvalue.real) * 299792458 / (2 * np.pi)
    freqGHz = frequency / 1e9
    betaL = theta_pbc # Use theta directly as Beta*L
    L_pbc = np.max(vertices[:,0]) - np.min(vertices[:,0])
    beta = betaL / L_pbc if L_pbc > 1e-9 else 0
    fig.suptitle(f'Mode {eigenmode_index+1} (n={n}), Freq: {freqGHz:.4f} GHz, k^2: {eigenvalue.real:.4f}, Beta*L: {betaL:.3f} (beta={beta:.3f})',
                 fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# --- 共通描画関数 (再利用) ---
def draw_mesh_and_pec(ax, vertices, simplices, edge_index_map, boundary_edges_pec):
    """共通スタイルでメッシュと PEC 境界を描画する (TM0 スタイル統一)。"""
    tri = Triangulation(vertices[:, 0], vertices[:, 1], simplices)
    ax.set_aspect('equal', adjustable='box')
    ax.triplot(tri, linewidth=STYLE.MESH_LW,
               color=STYLE.MESH_COLOR, alpha=STYLE.MESH_ALPHA)
    pec_label_added = False
    edge_nodes_map = {v: k for k, v in edge_index_map.items()}
    for edge_index in boundary_edges_pec:
         nodes = edge_nodes_map.get(edge_index)
         if nodes:
             label = 'PEC Boundary' if not pec_label_added else ""
             ax.plot(vertices[list(nodes), 0], vertices[list(nodes), 1],
                     color=STYLE.BORDER_COLOR,
                     linewidth=STYLE.BORDER_LW_STRONG,
                     zorder=10, label=label)
             pec_label_added = True
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.grid(True, linestyle=STYLE.GRID_LS, alpha=STYLE.GRID_ALPHA)
    return tri # Triangulationオブジェクトを返す

# --- アニメーションフレーム生成関数 ---
def generate_animation_frame(
    x_complex, phi, frame_index, output_dir,
    eigenvalue, theta_pbc, eigenmode_index, n,
    vertices, simplices, edge_index_map, num_edges, boundary_edges_pec,
    vec_scale, vec_width, vec_pivot, cmap_scalar, vmin_theta, vmax_theta, # 固定スケール/範囲
    quiver_key_val=None, # ベクトルスケールのキー値
    theta_cbar_format='%.2e'
):
    """指定された位相 phi における電場の実部を計算し、フレーム画像を保存する。"""
    N_original = len(x_complex)
    num_nodes = len(vertices)
    x_at_phi_real = (x_complex * np.exp(-1j * phi)).real # 位相phiでの実部

    edge_dofs_real = x_at_phi_real[:num_edges]
    node_dofs_real = None
    E_theta_real_nodes = None

    # E_theta (実部) の計算と補間 (n=1 のみ)
    if n > 0:
        node_dofs_real = x_at_phi_real[num_edges:] # r*E_theta の実部
        r_coords = vertices[:, 1]
        E_theta_real_nodes = np.zeros(num_nodes)
        non_zero_r = np.where(r_coords > 1e-9)[0]
        E_theta_real_nodes[non_zero_r] = node_dofs_real[non_zero_r] / r_coords[non_zero_r]
        if n == 1:
            # r=0 補間 (実数値で行う)
            axis_node_indices = np.where(np.abs(vertices[:, 1]) < 1e-9)[0]
            # node_to_elements マップ作成（事前計算が望ましい）
            node_to_elements = [[] for _ in range(num_nodes)]
            for elem_idx, simplex in enumerate(simplices):
                for node_idx in simplex: node_to_elements[node_idx].append(elem_idx)
            # 補間実行
            E_theta_interpolated_values = {}
            for i in axis_node_indices:
                neighbor_vals = []
                neighbor_nodes = set()
                for elem_idx in node_to_elements[i]:
                    simplex = simplices[elem_idx]
                    for k in simplex:
                        if k != i and abs(vertices[k, 1]) > 1e-9:
                             if k not in neighbor_nodes:
                                 neighbor_nodes.add(k); neighbor_vals.append(E_theta_real_nodes[k])
                if neighbor_vals: E_theta_interpolated_values[i] = np.mean(neighbor_vals)
                else: E_theta_interpolated_values[i] = 0.0
            for node_idx, value in E_theta_interpolated_values.items():
                 E_theta_real_nodes[node_idx] = value

    # Er, Ez (実部) の計算 (要素中心)
    element_centers = []
    E_vec_real_elements = []
    for elem_idx, simplex in enumerate(simplices):
        vertices_element = vertices[simplex]
        element_center = calculate_element_center(vertices_element)
        element_centers.append(element_center)
        E_element_real = np.zeros(3)
        edge_vertices_local_nodes = [
            (simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])
        ]
        for i, (n1, n2) in enumerate(edge_vertices_local_nodes):
            global_edge_key = tuple(sorted((n1, n2)))
            if global_edge_key in edge_index_map:
                edge_global_index = edge_index_map[global_edge_key]
                orientation_sign = get_edge_orientation_sign(n1, n2)
                E_element_real[i] = orientation_sign * edge_dofs_real[edge_global_index]
        N1, N2, N3 = calculate_edge_shape_functions(element_center, vertices_element)
        vec_real = E_element_real[0] * N1 + E_element_real[1] * N2 + E_element_real[2] * N3
        E_vec_real_elements.append(vec_real)
    element_centers_array = np.array(element_centers)
    E_vec_real_array = np.array(E_vec_real_elements)
    Ez_real = E_vec_real_array[:, 0]
    Er_real = E_vec_real_array[:, 1]

    # --- プロット ---
    num_subplots_h = 1 + (1 if n > 0 else 0)
    fig, axes = plt.subplots(1, num_subplots_h, figsize=(8 * num_subplots_h, 7), squeeze=False)
    ax_vec = axes[0, 0]

    # Er, Ez ベクトル
    tri = draw_mesh_and_pec(ax_vec, vertices, simplices, edge_index_map, boundary_edges_pec) # 共通描画
    q = ax_vec.quiver(element_centers_array[:, 0], element_centers_array[:, 1],
                      Ez_real, Er_real, angles='xy', scale_units='xy',
                      scale=vec_scale, width=vec_width, pivot=vec_pivot,
                      color=STYLE.QUIVER_COLOR, alpha=STYLE.QUIVER_ALPHA)
    if quiver_key_val is not None: # quiverkey を表示 (値は外部で決定)
        ax_vec.quiverkey(q, 0.8, 0.9, quiver_key_val, f'{quiver_key_val:.2e}', labelpos='E', coordinates='axes')
    ax_vec.set_title(f'Re(Ez, Er)') # 位相情報は全体タイトルへ

    # E_theta スカラー (n > 0)
    if n > 0 and E_theta_real_nodes is not None:
        ax_theta = axes[0, 1]
        tri_theta = draw_mesh_and_pec(ax_theta, vertices, simplices, edge_index_map, boundary_edges_pec)
        norm_scalar = mcolors.Normalize(vmin=vmin_theta, vmax=vmax_theta) # 固定範囲

        
        num_levels = 101
        levels = np.linspace(vmin_theta, vmax_theta, num_levels)
        #print('levels:', levels)

        contour = ax_theta.tricontourf(tri_theta, E_theta_real_nodes, cmap='jet', norm=norm_scalar, levels=levels)
        cbar = fig.colorbar(contour, ax=ax_theta, fraction=0.046, pad=0.04)
        cbar.set_label(r'Re($E_\theta$)')
        # --- ここでカラーバーのフォーマットを指定 ---
        try:
            formatter = mticker.FormatStrFormatter(theta_cbar_format) # 引数を使用
            cbar.ax.yaxis.set_major_formatter(formatter)
        except ValueError:
            print(f"Warning: Invalid format string '{theta_cbar_format}' for colorbar. Using default format.")
        # ------------------------------------------
        # --- 目盛りとフォーマットを固定 ---
        try:
            num_ticks = 5 # 例: 表示する目盛りの数
            ticks = np.linspace(vmin_theta, vmax_theta, num_ticks)
            cbar.set_ticks(ticks) # 目盛り位置を設定
            formatter = mticker.FormatStrFormatter(theta_cbar_format) # フォーマットを作成
            cbar.ax.yaxis.set_major_formatter(formatter) # フォーマットを適用
            print(f"  Frame {frame_index}: Set {num_ticks} ticks from {vmin_theta:.3e} to {vmax_theta:.3e} with format '{theta_cbar_format}'") # デバッグ用出力
        except Exception as e:
             print(f"Warning: Could not set fixed ticks/format for E_theta colorbar: {e}")
        # ------------------------------
        
        ax_theta.set_title(r'Re($E_\theta$)')

    # 全体タイトル
    frequency = np.sqrt(eigenvalue.real) * 299792458 / (2 * np.pi)
    freqGHz = frequency / 1e9
    fig.suptitle(f'Mode {eigenmode_index+1} (n={n}), Freq: {freqGHz:.4f} GHz, Phase phi={phi:.3f} rad', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # ファイル保存
    filepath = os.path.join(output_dir, f"frame_{frame_index:03d}.png")
    try:
        plt.savefig(filepath, dpi=100) # dpi調整可
    except Exception as e:
        print(f"Error saving frame {frame_index}: {e}")
    finally:
        plt.close(fig) # メモリを解放


# --- アニメーション作成関数 ---
def create_mode_animation(
    x_complex, eigenvalue, theta_pbc, eigenmode_index, n,
    vertices, simplices, edge_index_map, num_edges, boundary_edges_pec,
    num_frames=30, output_dir="animation_frames", output_gif="mode_animation.gif",
    theta_cbar_format='%.2e', # カラーバーのフォーマット文字列を引数に追加
    duration=100, # GIFのフレーム間隔(ms)
    manual_vec_scale=None,    # <<<--- 手動スケール指定用引数
    vec_length_factor=0.1    # <<<--- 自動スケール調整用引数

):
    """指定されたモードの電場時間変化アニメーションを作成する。"""
    print(f"\nCreating animation for Mode {eigenmode_index+1} (n={n})...")
    N_original = len(x_complex)
    num_nodes = len(vertices)

    # --- プロット範囲とスケールの決定 (複素振幅に基づく) ---
    print(" Determining plot scales and ranges based on complex amplitudes...")
    max_vec_amp = 0.0
    max_theta_amp = 0.0

    # 1. Er, Ez ベクトルの複素振幅の最大値を計算
    edge_dofs_complex = x_complex[:num_edges] # 元の複素自由度
    # 要素中心での複素ベクトル E_vec_complex を計算
    E_vec_complex_elements_for_scale = []
    for elem_idx, simplex in enumerate(simplices):
        vertices_element = vertices[simplex]
        element_center = calculate_element_center(vertices_element)
        E_element_complex = np.zeros(3, dtype=complex)
        edge_vertices_local_nodes = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
        for i, (n1, n2) in enumerate(edge_vertices_local_nodes):
            global_edge_key = tuple(sorted((n1, n2)))
            if global_edge_key in edge_index_map:
                edge_global_index = edge_index_map[global_edge_key]
                orientation_sign = get_edge_orientation_sign(n1, n2)
                E_element_complex[i] = orientation_sign * edge_dofs_complex[edge_global_index] # 複素値を使う
        N1, N2, N3 = calculate_edge_shape_functions(element_center, vertices_element)
        vec_complex = E_element_complex[0] * N1 + E_element_complex[1] * N2 + E_element_complex[2] * N3
        E_vec_complex_elements_for_scale.append(vec_complex)
    E_vec_complex_array_for_scale = np.array(E_vec_complex_elements_for_scale)
    # ベクトルの振幅 (|Ez|^2 + |Er|^2)^(1/2) ではなく、各瞬間の最大値に対応するため、
    # Re(Ez)^2 + Re(Er)^2 の最大値を近似的に使うか、より単純には複素振幅を使う。
    # ここでは単純に sqrt(|Ez|^2 + |Er|^2) の最大値を使う。
    vec_amplitude = np.sqrt(np.abs(E_vec_complex_array_for_scale[:, 0])**2 + np.abs(E_vec_complex_array_for_scale[:, 1])**2)
    max_vec_amp = np.max(vec_amplitude) if vec_amplitude.size > 0 else 0.0

    # 2. E_theta の複素振幅の最大値を計算 (n > 0 の場合)
    if n > 0:
        node_dofs_complex = x_complex[num_edges:] # rE_theta (複素数)
        r_coords = vertices[:, 1]
        E_theta_complex_nodes = np.zeros(num_nodes, dtype=complex)
        non_zero_r = np.where(r_coords > 1e-9)[0]
        # E_theta = (rE_theta) / r を複素数で計算
        E_theta_complex_nodes[non_zero_r] = node_dofs_complex[non_zero_r] / r_coords[non_zero_r]
        # E_theta の振幅 |E_theta| を計算
        theta_amplitude = np.abs(E_theta_complex_nodes)
        max_theta_amp = np.max(theta_amplitude) if theta_amplitude.size > 0 else 0.0
        # 注意: n=1 の r=0 補間は各フレームで行うため、その影響はここでの
        #       最大振幅計算には直接含まれない。補間は主にr=0での表示を滑らかにするため。

    # --- スケール決定 (修正部分) ---
    quiver_key_val = max_vec_amp # quiverkey は最大振幅を表示
    vmin_theta = -max_theta_amp if max_theta_amp > 1e-9 else -1.0
    vmax_theta = max_theta_amp if max_theta_amp > 1e-9 else 1.0

    if manual_vec_scale is not None:
        # 手動指定があればそれを使用
        vec_scale = manual_vec_scale
        print(f"  Using manual vec_scale = {vec_scale:.3e}")
    else:
        # 自動計算ロジック
        print(f"  Using automatic vec_scale calculation with vec_length_factor = {vec_length_factor}")
        # プロット領域の対角線の長さを計算 (データ単位)
        z_min, z_max = np.min(vertices[:,0]), np.max(vertices[:,0])
        r_min, r_max = np.min(vertices[:,1]), np.max(vertices[:,1])
        diag_len_data = np.sqrt((z_max - z_min)**2 + (r_max - r_min)**2)
        if diag_len_data < 1e-9: diag_len_data = 1.0 # Fallback

        # vec_scale を計算 (最大振幅の矢印が対角線の factor 分の長さになるように)
        # scale = データ単位 / 矢印長(データ単位)
        target_arrow_length = diag_len_data * vec_length_factor
        vec_scale = max_vec_amp / target_arrow_length if max_vec_amp > 1e-9 and target_arrow_length > 1e-9 else 1.0
        print(f"  Max vector amplitude (|E|): {max_vec_amp:.3e}")
        print(f"  Plot diagonal length: {diag_len_data:.3e}")
        print(f"  Target arrow length (factor={vec_length_factor}): {target_arrow_length:.3e}")
        print(f"  Calculated vec_scale = {vec_scale:.3e}")

    print(f"  Max E_theta amplitude (|E_theta|): {max_theta_amp:.3e} -> colorbar range=({vmin_theta:.3e}, {vmax_theta:.3e})")
    # --- 出力ディレクトリ作成 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        # 古いファイルを削除（オプション）
        for f in os.listdir(output_dir):
            if f.startswith("frame_") and f.endswith(".png"):
                os.remove(os.path.join(output_dir, f))
        print(f"Cleaned output directory: {output_dir}")


    # --- フレーム生成ループ ---
    print(f" Generating {num_frames} frames...")
    phases = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    for i, phi in enumerate(phases):
        print(f"  Generating frame {i+1}/{num_frames} (phi={phi:.3f})...")
        generate_animation_frame(
            x_complex, phi, i, output_dir,
            eigenvalue, theta_pbc, eigenmode_index, n,
            vertices, simplices, edge_index_map, num_edges, boundary_edges_pec,
            vec_scale, 0.003, 'mid', 'viridis', 
            vmin_theta=vmin_theta, vmax_theta=vmax_theta, # 固定スケール/範囲を渡す
            quiver_key_val=quiver_key_val,
            theta_cbar_format=theta_cbar_format # 指定されたフォーマットを渡す
        )

    # --- GIFアニメーション作成 ---
    print(f" Creating GIF animation: {output_gif} ...")
    images = []
    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".png")])
    if not frame_files:
        print("Error: No frames found to create GIF.")
        return

    for filename in frame_files:
        try:
            images.append(imageio.imread(filename))
        except Exception as e:
            print(f"Error reading frame {filename}: {e}")

    if images:
        try:
            imageio.mimsave(output_gif, images, duration=duration, loop=0) # loop=0で無限ループ
            print(f"Animation successfully saved to {output_gif}")
        except Exception as e:
            print(f"Error creating GIF: {e}")
    else:
        print("No valid frames were read, GIF not created.")