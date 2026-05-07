import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import os

from FEM_element_function import calculate_triangle_area_double, calculate_area_coordinates, grad_area_coordinates, calculate_edge_shape_functions

import re # 正規表現を使ってグループ名をパースするため

# プロジェクトルートを sys.path に追加して plot_common を解決
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from plot_common import STYLE  # noqa: E402

def print_hdf5_summary(filepath):
    """
    指定された HDF5 ファイルの構造と保存されているモード情報を要約して表示する。
    """
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return

    print(f"\n--- HDF5 File Summary: {filepath} ---")

    try:
        with h5py.File(filepath, 'r') as f:
            # --- ルート直下の確認 ---
            print("Root groups:", list(f.keys()))
            if "results" not in f:
                print("'/results' group not found.")
                return

            results_group = f["results"]

            # --- n ごとのループ ---
            n_groups = sorted([g for g in results_group if g.startswith('n')], key=lambda x: int(x[1:]))
            if not n_groups:
                print("No results found under '/results'.")
                return

            print("\nStored Results Overview:")
            for n_group_name in n_groups:
                try:
                    n_mode = int(n_group_name[1:]) # 'n1' -> 1
                    n_group = results_group[n_group_name]
                    print(f"\n  n = {n_mode}:")

                    # --- Calculation Mode (Normal/Periodic) ---
                    for calc_mode_name in n_group:
                        calc_mode_group = n_group[calc_mode_name]

                        if calc_mode_name == "Normal":
                            print(f"    Calculation: Normal (Standing Wave)")
                            # Normal モードをリスト
                            mode_list = sorted([m for m in calc_mode_group if m.startswith('mode_')], key=lambda x: int(x.split('_')[1]))
                            if not mode_list:
                                print("      No modes found.")
                                continue
                            print(f"      Found {len(mode_list)} modes:")
                            # 最初の数モードの情報を表示
                            for i, mode_name in enumerate(mode_list[:5]): # Display first 5 modes
                                mode_group = calc_mode_group[mode_name]
                                freq = mode_group.attrs.get('frequency_GHz', np.nan)
                                k2 = mode_group.attrs.get('eigenvalue_k2', np.nan)
                                print(f"        {mode_name}: Freq = {freq:.4f} GHz, k^2 = {k2:.4e}")
                            if len(mode_list) > 5: print("        ...")

                        elif calc_mode_name == "Periodic":
                            print(f"    Calculation: Periodic (Traveling Wave)")
                            # Theta ごとのループ
                            theta_groups = sorted(calc_mode_group.keys())
                            if not theta_groups:
                                print("      No phase angles (theta) found.")
                                continue

                            for theta_group_name in theta_groups:
                                # グループ名から Theta を抽出 (例: PB_Phase_030_0_deg)
                                match = re.match(r"PB_Phase_(\d+)_(\d+)_deg", theta_group_name)
                                if match:
                                    theta_deg = float(f"{match.group(1)}.{match.group(2)}")
                                    print(f"      Theta = {theta_deg:.1f} deg:")
                                else:
                                    print(f"      Theta Group: {theta_group_name} (Could not parse angle)")
                                    theta_deg = np.nan # パース失敗

                                theta_group = calc_mode_group[theta_group_name]
                                # Periodic/Theta 下のモードをリスト
                                mode_list = sorted([m for m in theta_group if m.startswith('mode_')], key=lambda x: int(x.split('_')[1]))
                                if not mode_list:
                                    print("        No modes found for this phase.")
                                    continue
                                print(f"        Found {len(mode_list)} modes:")
                                # 最初の数モードの情報を表示
                                for i, mode_name in enumerate(mode_list[:5]): # Display first 5 modes
                                    mode_group = theta_group[mode_name]
                                    freq = mode_group.attrs.get('frequency_GHz', np.nan)
                                    k2 = mode_group.attrs.get('eigenvalue_k2', np.nan)
                                    beta = mode_group.attrs.get('beta', np.nan) # beta もあれば表示
                                    print(f"          {mode_name}: Freq = {freq:.4f} GHz, k^2 = {k2:.4e}, beta = {beta:.3f}")
                                if len(mode_list) > 5: print("          ...")
                        else:
                            print(f"    Unknown calculation type: {calc_mode_name}")

                except Exception as e:
                    print(f"  Error processing group {n_group_name}: {e}")

    except FileNotFoundError:
         # 開始時にチェック済みだが念のため
         print(f"エラー: ファイルが見つかりません: {filepath}")
    except Exception as e:
        print(f"エラー: HDF5ファイルの読み込み/解析中にエラーが発生しました: {e}")

    print("\n--- Summary End ---")


def get_edge_orientation_sign(global1, global2):
    if global1< global2 : return 1
    else : return -1

# --- メッシュとPEC境界を描画する関数 --- <<<--- ★★★ ここに追加 ★★★
def draw_mesh_and_pec(ax, vertices, simplices, edge_index_map, boundary_edges_pec):
    """与えられた Axes にメッシュと PEC 境界を描画する (TM0 スタイル統一)。"""
    tri = Triangulation(vertices[:, 0], vertices[:, 1], simplices)
    ax.set_aspect('equal', adjustable='box')
    ax.triplot(tri, linewidth=STYLE.MESH_LW,
               color=STYLE.MESH_COLOR, alpha=STYLE.MESH_ALPHA)
    pec_label_added = False
    # 逆引きマップを作成
    edge_nodes_map = {v: k for k, v in edge_index_map.items()}
    if boundary_edges_pec is not None and len(boundary_edges_pec) > 0:
        for edge_index in boundary_edges_pec:
             # HDF5から読み込んだデータがintでない可能性を考慮
             if not isinstance(edge_index, (int, np.integer)): continue
             nodes = edge_nodes_map.get(int(edge_index))
             if nodes:
                 label = 'PEC Boundary' if not pec_label_added else ""
                 ax.plot(vertices[list(nodes), 0], vertices[list(nodes), 1],
                         color=STYLE.BORDER_COLOR,
                         linewidth=STYLE.BORDER_LW,
                         zorder=10, label=label)
                 pec_label_added = True
    if pec_label_added:
         ax.legend(loc='upper right', fontsize=STYLE.LEGEND_FONTSIZE)
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.grid(True, linestyle=STYLE.GRID_LS, alpha=STYLE.GRID_ALPHA)
    return tri # Triangulationオブジェクトを返す (contourfなどで利用)
# --- ★★★ 追加ここまで ★★★ ---

def load_mode_data_from_hdf5(filepath, result_path, mode_index):
    """
    HDF5ファイルから指定されたモードのデータを読み込む。

    Args:
        filepath (str): HDF5ファイルパス
        result_path (str): 計算結果グループへのパス (例: "/results/n1/Periodic/PB_Phase_030_0_deg")
        mode_index (int): 読み込むモードのインデックス (0, 1, ...)

    Returns:
        dict: モードデータを含む辞書、またはエラー時に None
              {
                  'eigenvalue_k2': float,
                  'frequency_GHz': float,
                  'beta': float (optional),
                  'edge_vectors_re': np.ndarray or None,
                  'edge_vectors_im': np.ndarray or None,
                  'E_theta_re': np.ndarray or None,
                  'E_theta_im': np.ndarray or None,
                  'edge_vectors': np.ndarray or None, # Normal case
                  'E_theta': np.ndarray or None,      # Normal case
                  'is_periodic': bool,
                  # メッシュとパラメータ情報も追加で返す
                  'vertices': np.ndarray,
                  'simplices': np.ndarray,
                  'edge_map_keys': np.ndarray,
                  'edge_map_values': np.ndarray,
                  'num_edges': int,
                  'num_nodes': int,
                  'n_mode': int,
                  'theta_deg': float (optional),
                  'theta_rad': float (optional),
                  'boundary_edges_pec': np.ndarray,
                  # ... 他の必要なパラメータや境界情報
              }
    """
    data = {'is_periodic': "Periodic" in result_path}
    try:
        with h5py.File(filepath, 'r') as f:
            # --- メッシュ情報読み込み ---
            if "/mesh" not in f: raise KeyError("Group '/mesh' not found.")
            mesh_group = f["/mesh"]
            data['vertices'] = mesh_group["vertices"][:]
            data['simplices'] = mesh_group["simplices"][:]
            data['edge_map_keys'] = mesh_group["edge_map_keys"][:]
            data['edge_map_values'] = mesh_group["edge_map_values"][:]
            data['num_nodes'] = mesh_group.attrs["num_nodes"]
            data['num_edges'] = mesh_group.attrs["num_edges"]

            # --- パラメータ読み込み (n_modeなど) ---
            # result_path から n を抽出 (例: "/results/n1/...")
            path_parts = result_path.strip('/').split('/')
            if len(path_parts) < 2 or not path_parts[1].startswith('n'):
                raise ValueError(f"Could not determine n_mode from path: {result_path}")
            data['n_mode'] = int(path_parts[1][1:]) # 'n1' -> 1

            # thetaもパスから取得 (Periodicの場合)
            if data['is_periodic']:
                 if len(path_parts) < 4 or not path_parts[3].startswith('PB_Phase_'):
                      print(f"Warning: Could not parse theta from path: {result_path}")
                      data['theta_deg'] = None
                      data['theta_rad'] = None
                 else:
                     try:
                        # 例: PB_Phase_030_0_deg -> 30.0
                        theta_str = path_parts[3].split('_')[2] + '.' + path_parts[3].split('_')[3]
                        data['theta_deg'] = float(theta_str)
                        data['theta_rad'] = np.deg2rad(data['theta_deg'])
                     except:
                         print(f"Warning: Error parsing theta from group name: {path_parts[3]}")
                         data['theta_deg'] = None
                         data['theta_rad'] = None


            # --- モードデータ読み込み ---
            mode_group_path = f"{result_path}/mode_{mode_index}"
            if mode_group_path not in f:
                raise KeyError(f"Mode group '{mode_group_path}' not found.")
            mode_group = f[mode_group_path]

            data['eigenvalue_k2'] = mode_group.attrs["eigenvalue_k2"]
            data['frequency_GHz'] = mode_group.attrs["frequency_GHz"]
            if 'beta' in mode_group.attrs: data['beta'] = mode_group.attrs["beta"]
            if 'beta_target' in mode_group.attrs: data['beta'] = mode_group.attrs["beta_target"] # 保存方法による


            if data['is_periodic']:
                data['edge_vectors_re'] = mode_group["edge_vectors_re"][:]
                data['edge_vectors_im'] = mode_group["edge_vectors_im"][:]
                data['E_theta_re'] = mode_group.get("E_theta_re", None) # n=0では存在しない
                if data['E_theta_re'] is not None: data['E_theta_re'] = data['E_theta_re'][:]
                data['E_theta_im'] = mode_group.get("E_theta_im", None)
                if data['E_theta_im'] is not None: data['E_theta_im'] = data['E_theta_im'][:]
                # Normal 用のキーは None に設定
                data['edge_vectors'] = None
                data['E_theta'] = None
            else: # Normal
                data['edge_vectors'] = mode_group["edge_vectors"][:]
                data['E_theta'] = mode_group.get("E_theta", None)
                if data['E_theta'] is not None: data['E_theta'] = data['E_theta'][:]
                # Periodic 用のキーは None に設定
                data['edge_vectors_re'] = None; data['edge_vectors_im'] = None
                data['E_theta_re'] = None; data['E_theta_im'] = None

            # --- 境界条件読み込み (パラメータグループから) ---
            param_path = f"{result_path}/parameters"
            if param_path in f:
                 param_group = f[param_path]
                 data['boundary_edges_pec'] = param_group['boundary_edges_pec'][:]
                 if 'boundary_nodes_pec' in param_group:
                     data['boundary_nodes_pec'] = param_group['boundary_nodes_pec'][:]
                 # 必要ならPBCペアも読み込む
                 # data['pbc_edge_pairs'] = param_group.get('pbc_edge_pairs')[:]
                 # data['pbc_node_pairs'] = param_group.get('pbc_node_pairs')[:]

            else:
                 print(f"Warning: Parameters group not found at {param_path}. Using default PEC from mesh group if available.")
                 # フォールバックとして mesh グループ下のパラメータ等を参照する可能性も？
                 # ここではとりあえず空リストなどを設定
                 data['boundary_edges_pec'] = np.array([], dtype=int)


    except FileNotFoundError:
        print(f"エラー: HDF5ファイルが見つかりません: {filepath}")
        return None
    except KeyError as e:
        print(f"エラー: HDF5ファイルに必要なデータ/グループが見つかりません: {e}")
        return None
    except Exception as e:
        print(f"エラー: HDF5ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return None

    return data

# --- 必要なヘルパー関数 ---
# calculate_area_coordinates, grad_area_coordinates, calculate_edge_shape_functions
# get_edge_orientation_sign が必要 (前の回答からコピー)

def interpolate_fields_on_grid(
    grid_z, grid_r, # 2D grid coordinates from np.meshgrid
    vertices, simplices, # Mesh info
    edge_index_map, # <<<--- 引数に追加
    n_mode, # Azimuthal mode number
    # Field data (provide either complex or real/imag pairs)
    edge_vec_re=None, edge_vec_im=None, E_theta_re=None, E_theta_im=None, # Periodic
    edge_vec=None, E_theta=None # Normal
):
    """格子点上で電場 (Ez, Er, E_theta) を補間する"""
    print("Interpolating fields onto grid...")
    if grid_z.shape != grid_r.shape:
        raise ValueError("grid_z and grid_r must have the same shape.")
    grid_shape = grid_z.shape
    points_to_interpolate = np.vstack((grid_z.ravel(), grid_r.ravel())).T # (N_points, 2)

    num_points = len(points_to_interpolate)
    num_edges = len(edge_index_map) # edge_index_map から取得
    num_nodes = len(vertices)

    # 出力配列を初期化
    interp_Ez_re = np.full(num_points, np.nan)
    interp_Er_re = np.full(num_points, np.nan)
    interp_Ez_im = np.full(num_points, np.nan)
    interp_Er_im = np.full(num_points, np.nan)
    interp_Etheta_re = np.full(num_points, np.nan)
    interp_Etheta_im = np.full(num_points, np.nan)

    is_periodic = edge_vec_re is not None # Periodic かどうかの判定

    # --- 効率化のための準備 ---
    # edge_index -> sorted_nodes の逆マップ
    #edge_index_to_nodes = {val: tuple(key) for key, val in zip(edge_map_keys, edge_map_values)}
    # 要素検索のための Triangulation と trifinder
    triangulation = Triangulation(vertices[:, 0], vertices[:, 1], simplices)
    trifinder = triangulation.get_trifinder()
    print(" Finding elements for grid points...")
    containing_elements = trifinder(points_to_interpolate[:, 0], points_to_interpolate[:, 1])

    print(" Interpolating fields...")
    # --- 各格子点でループ (ベクトル化も可能だが、まずはループで) ---
    for p_idx in range(num_points):
        elem_idx = containing_elements[p_idx]
        if elem_idx == -1: continue # 要素外の点

        point = points_to_interpolate[p_idx] # [z, r]
        simplex = simplices[elem_idx] # [n0, n1, n2]
        vertices_element = vertices[simplex]

        # 要素内エッジ自由度の取得
        E_element_re = np.zeros(3)
        E_element_im = np.zeros(3)
        local_edges_nodes = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
        valid_edge_data = True
        for i, (n1, n2) in enumerate(local_edges_nodes):
            global_edge_key = tuple(sorted((n1, n2)))
            global_edge_idx = edge_index_map.get(global_edge_key, -1) # 存在しない場合の処理
            if global_edge_idx == -1:
                valid_edge_data = False; break # エッジが見つからない
            orientation_sign = get_edge_orientation_sign(n1, n2)
            if is_periodic:
                if edge_vec_re is None or edge_vec_im is None or global_edge_idx >= len(edge_vec_re):
                     valid_edge_data = False; break # データがない
                E_element_re[i] = orientation_sign * edge_vec_re[global_edge_idx]
                E_element_im[i] = orientation_sign * edge_vec_im[global_edge_idx]
            else:
                if edge_vec is None or global_edge_idx >= len(edge_vec):
                     valid_edge_data = False; break
                E_element_re[i] = orientation_sign * edge_vec[global_edge_idx]
                # E_element_im は 0 のまま
        if not valid_edge_data: continue # この要素では補間できない

        # エッジ形状関数による補間 (Ez, Er)
        N1, N2, N3 = calculate_edge_shape_functions(point, vertices_element)
        interp_Ez_re[p_idx] = E_element_re[0] * N1[0] + E_element_re[1] * N2[0] + E_element_re[2] * N3[0]
        interp_Er_re[p_idx] = E_element_re[0] * N1[1] + E_element_re[1] * N2[1] + E_element_re[2] * N3[1]
        if is_periodic:
            interp_Ez_im[p_idx] = E_element_im[0] * N1[0] + E_element_im[1] * N2[0] + E_element_im[2] * N3[0]
            interp_Er_im[p_idx] = E_element_im[0] * N1[1] + E_element_im[1] * N2[1] + E_element_im[2] * N3[1]

        # 節点形状関数による補間 (E_theta) (n > 0)
        if n_mode > 0:
            L = calculate_area_coordinates(point, vertices_element) # 重心座標=ラグランジュ基底値
            theta_val_re = 0.0
            theta_val_im = 0.0
            valid_theta_data = True
            for i in range(3):
                node_idx = simplex[i]
                if is_periodic:
                    if E_theta_re is None or E_theta_im is None or node_idx >= len(E_theta_re):
                        valid_theta_data = False; break
                    theta_val_re += E_theta_re[node_idx] * L[i]
                    theta_val_im += E_theta_im[node_idx] * L[i]
                else:
                    if E_theta is None or node_idx >= len(E_theta):
                         valid_theta_data = False; break
                    theta_val_re += E_theta[node_idx] * L[i]
            if valid_theta_data:
                 interp_Etheta_re[p_idx] = theta_val_re
                 if is_periodic: interp_Etheta_im[p_idx] = theta_val_im

    # NaN を 0 に置き換える (オプション)
    interp_Ez_re = np.nan_to_num(interp_Ez_re).reshape(grid_shape)
    interp_Er_re = np.nan_to_num(interp_Er_re).reshape(grid_shape)
    interp_Etheta_re = np.nan_to_num(interp_Etheta_re).reshape(grid_shape)
    if is_periodic:
        interp_Ez_im = np.nan_to_num(interp_Ez_im).reshape(grid_shape)
        interp_Er_im = np.nan_to_num(interp_Er_im).reshape(grid_shape)
        interp_Etheta_im = np.nan_to_num(interp_Etheta_im).reshape(grid_shape)
        print("Interpolation finished.")
        return interp_Ez_re, interp_Er_re, interp_Etheta_re, interp_Ez_im, interp_Er_im, interp_Etheta_im
    else:
        print("Interpolation finished.")
        return interp_Ez_re, interp_Er_re, interp_Etheta_re, None, None, None
    

def plot_mode_on_grid_noTricontourf(
    grid_z, grid_r, # Grid coordinates
    interp_Ez_re, interp_Er_re, interp_Etheta_re,
    interp_Ez_im, interp_Er_im, interp_Etheta_im, # Can be None for Normal case
    mode_data, # Dictionary from load_mode_data_from_hdf5
    mode_index, # <<<--- 引数に追加
    # Plotting options
    plot_type='real_imag', # 'real_imag', 'amp_phase' (not implemented yet), 'real_only'
    vec_scale_factor=1, # vector scale factor 大きくするとベクトルが大きくなる
    vec_density=1, # Plot every 'vec_density' grid points
    cmap_scalar= 'jet', # or 'RdBu_r' # Colormap for E_theta 
):
    """補間されたグリッドデータを使ってモードをプロットする"""

    is_periodic = mode_data['is_periodic']
    n_mode = mode_data['n_mode']
    vertices = mode_data['vertices']
    simplices = mode_data['simplices']
    # edge_map の再構築 (draw_mesh_and_pec用)
    edge_index_map = {tuple(key): val for key, val in zip(mode_data['edge_map_keys'], mode_data['edge_map_values'])}
    boundary_edges_pec = mode_data['boundary_edges_pec']
    eigenvalue = mode_data['eigenvalue_k2']
    freqGHz = mode_data['frequency_GHz']

    # --- プロット設定 ---
    num_cols = 2 if plot_type == 'real_imag' and is_periodic else 1
    num_rows = 1 + (1 if n_mode > 0 else 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 6*num_rows), squeeze=False)

    # --- グリッドの間引き (Quiver用) ---
    skip = (slice(None, None, vec_density), slice(None, None, vec_density))
    grid_z_q = grid_z[skip]
    grid_r_q = grid_r[skip]
    Ez_re_q = interp_Ez_re[skip]
    Er_re_q = interp_Er_re[skip]
    Ez_im_q = interp_Ez_im[skip] if is_periodic else None
    Er_im_q = interp_Er_im[skip] if is_periodic else None

    # --- メッシュ内マスク作成 (ゼロ矢印ドット防止) ---
    _tri_finder = Triangulation(vertices[:, 0], vertices[:, 1], simplices).get_trifinder()
    _inside = (_tri_finder(grid_z_q.ravel(), grid_r_q.ravel()) >= 0).reshape(grid_z_q.shape)

    # --- スケール決定  ---
    max_re_mag = np.max(np.sqrt(Ez_re_q**2 + Er_re_q**2))
    max_im_mag = np.max(np.sqrt(Ez_im_q**2 + Er_im_q**2)) if is_periodic else 0.0
    max_mag = max(max_re_mag, max_im_mag)
    #vec_scale = scale_factor_auto / max_mag if max_mag > 1e-9 else 1.0
    gfac = abs(grid_z[1][0] -grid_z[0][0])
    vec_scale = (max_mag/gfac) /vec_scale_factor #値が大きいほどベクトルは小さくなる
    print(f"Auto-calculated vec_scale: {vec_scale:.2e} (based on max_mag={max_mag:.2e} and grid facor={gfac:.2e})")

    # --- Real Part Plot ---
    ax_re_vec = axes[0, 0]
    tri = draw_mesh_and_pec(ax_re_vec, vertices, simplices, edge_index_map, boundary_edges_pec)
    q_re = ax_re_vec.quiver(grid_z_q[_inside], grid_r_q[_inside],
                            Ez_re_q[_inside], Er_re_q[_inside], angles='xy',
                            scale=vec_scale, scale_units='xy', pivot='mid',
                            color=STYLE.QUIVER_COLOR, alpha=STYLE.QUIVER_ALPHA, width=0.003)
    ax_re_vec.set_title('Re(Ez, Er)', fontsize=STYLE.TITLE_FONTSIZE)
    if max_re_mag > 1e-9:
         ax_re_vec.quiverkey(q_re, 0.85, 0.92, max_re_mag, f'{max_re_mag:.2e}', labelpos='E', coordinates='axes')

    if n_mode > 0:
        ax_re_theta = axes[1, 0]
        draw_mesh_and_pec(ax_re_theta, vertices, simplices, edge_index_map, boundary_edges_pec)
        vmax = max(np.abs(interp_Etheta_re).max(), 1e-9)
        vmin = -vmax
        norm_re = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cont_re = ax_re_theta.contourf(grid_z, grid_r, interp_Etheta_re, cmap=cmap_scalar, norm=norm_re, levels=50, extend='both')
        cbar_re = fig.colorbar(cont_re, ax=ax_re_theta, fraction=0.046, pad=0.04)
        cbar_re.set_label(r'Re($E_\theta$)')
        cbar_re.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
        ax_re_theta.set_title(r'Re($E_\theta$)')


    # --- Imaginary Part Plot (Periodic) ---
    if is_periodic and plot_type == 'real_imag':
        ax_im_vec = axes[0, 1]
        draw_mesh_and_pec(ax_im_vec, vertices, simplices, edge_index_map, boundary_edges_pec)
        q_im = ax_im_vec.quiver(grid_z_q[_inside], grid_r_q[_inside],
                                Ez_im_q[_inside], Er_im_q[_inside], angles='xy',
                                scale=vec_scale, scale_units='xy', pivot='mid',
                                color=STYLE.QUIVER_IMAG_COLOR, alpha=STYLE.QUIVER_ALPHA, width=0.003)
        ax_im_vec.set_title('Im(Ez, Er)', fontsize=STYLE.TITLE_FONTSIZE)
        if max_im_mag > 1e-9:
            ax_im_vec.quiverkey(q_im, 0.85, 0.92, max_im_mag, f'{max_im_mag:.2e}', labelpos='E', coordinates='axes')

        if n_mode > 0:
            ax_im_theta = axes[1, 1]
            draw_mesh_and_pec(ax_im_theta, vertices, simplices, edge_index_map, boundary_edges_pec)
            # 実部と同じカラースケールを使う
            cont_im = ax_im_theta.contourf(grid_z, grid_r, interp_Etheta_im, cmap=cmap_scalar, norm=norm_re, levels=50, extend='both')
            cbar_im = fig.colorbar(cont_im, ax=ax_im_theta, fraction=0.046, pad=0.04)
            cbar_im.set_label(r'Im($E_\theta$)')
            cbar_im.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
            ax_im_theta.set_title(r'Im($E_\theta$)')

    # --- 全体タイトル ---
    title = f'Mode Index: {mode_index}, Freq: {freqGHz:.4f} GHz, n={n_mode}'
    if is_periodic and mode_data.get('theta_deg') is not None:
         title += f', Theta={mode_data["theta_deg"]:.1f} deg'
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# plot_mode_on_grid 関数の修正版 (ベクトルスケールとE_theta対応)
def plot_mode_on_grid(
    grid_z, grid_r, # Grid coordinates for vector field
    interp_Ez_re, interp_Er_re, # Interpolated vectors (real)
    interp_Ez_im, interp_Er_im, # Interpolated vectors (imag, can be None)
    # No E_theta args here, get from mode_data
    mode_data, # Dictionary from load_mode_data_from_hdf5
    mode_index, # Current mode index
    # Plotting options
    plot_type='real_imag',
    vec_scale_factor=1.0, # <<<--- ユーザー定義のスケールファクター
    vec_density=1,
    cmap_scalar='jet',
):
    """
    モードをプロットする。Er/Ez はグリッド補間値、E_theta は節点データを使用。
    vec_scale_factor でベクトルサイズ調整。Normal/Periodic の E_theta を正しく処理。
    """
    is_periodic = mode_data['is_periodic']
    n_mode = mode_data['n_mode']
    vertices = mode_data['vertices']
    simplices = mode_data['simplices']
    edge_index_map = {tuple(map(int, key)): int(val) for key, val in zip(mode_data['edge_map_keys'], mode_data['edge_map_values'])}
    boundary_edges_pec = mode_data['boundary_edges_pec']
    eigenvalue = mode_data['eigenvalue_k2']
    freqGHz = mode_data['frequency_GHz']

    # --- E_theta データを取得 ---
    E_theta_re_nodes = None
    E_theta_im_nodes = None
    if n_mode > 0:
        if is_periodic:
            E_theta_re_nodes = mode_data.get('E_theta_re')
            E_theta_im_nodes = mode_data.get('E_theta_im')
        else: # Normal
            E_theta_re_nodes = mode_data.get('E_theta') # Normal時は実数データのみ

    # --- プロット設定 ---
    num_cols = 2 if plot_type == 'real_imag' and is_periodic else 1
    num_rows = 1 + (1 if n_mode > 0 else 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 6*num_rows), squeeze=False)

    # --- グリッドの間引き ---
    skip = (slice(None, None, vec_density), slice(None, None, vec_density))
    grid_z_q = grid_z[skip]; grid_r_q = grid_r[skip]
    Ez_re_q = interp_Ez_re[skip]; Er_re_q = interp_Er_re[skip]
    Ez_im_q = interp_Ez_im[skip] if is_periodic and interp_Ez_im is not None else None
    Er_im_q = interp_Er_im[skip] if is_periodic and interp_Er_im is not None else None

    # --- メッシュ内マスク作成 (ゼロ矢印ドット防止) ---
    _tri_finder = Triangulation(vertices[:, 0], vertices[:, 1], simplices).get_trifinder()
    _inside = (_tri_finder(grid_z_q.ravel(), grid_r_q.ravel()) >= 0).reshape(grid_z_q.shape)

    # --- スケール決定 (ユーザー修正版ロジック) ---
    max_re_mag = np.max(np.sqrt(Ez_re_q**2 + Er_re_q**2)) if Ez_re_q.size > 0 else 0.0
    max_im_mag = np.max(np.sqrt(Ez_im_q**2 + Er_im_q**2)) if is_periodic and Ez_im_q is not None and Ez_im_q.size > 0 else 0.0
    max_mag = max(max_re_mag, max_im_mag, 1e-9)
    gfac = abs(grid_z[1,0] - grid_z[0,0]) if grid_z.shape[0] > 1 else 1.0
    vec_scale = (max_mag / gfac) / vec_scale_factor if gfac > 1e-9 else max_mag / vec_scale_factor # <<<--- ユーザー定義ファクター使用
    quiver_key_val = max_mag # キーには最大振幅を表示
    print(f"vec_scale set to: {vec_scale:.2e} (max_mag={max_mag:.2e}, gfac={gfac:.2e}, factor={vec_scale_factor})")

    # --- Real Part Plot ---
    ax_re_vec = axes[0, 0]
    tri = draw_mesh_and_pec(ax_re_vec, vertices, simplices, edge_index_map, boundary_edges_pec)
    if Ez_re_q.size > 0:
        q_re = ax_re_vec.quiver(grid_z_q[_inside], grid_r_q[_inside],
                                Ez_re_q[_inside], Er_re_q[_inside], angles='xy',
                                scale=vec_scale, scale_units='xy', pivot='mid',
                                color=STYLE.QUIVER_COLOR, alpha=STYLE.QUIVER_ALPHA, width=0.003)
        ax_re_vec.quiverkey(q_re, 0.85, 0.92, quiver_key_val, f'{quiver_key_val:.2e}', labelpos='E', coordinates='axes') # キー値修正
    ax_re_vec.set_title('Re(Ez, Er) on Grid', fontsize=STYLE.TITLE_FONTSIZE)

    if n_mode > 0 and E_theta_re_nodes is not None: # <<<--- E_theta_re_nodes をチェック
        ax_re_theta = axes[1, 0]
        tri_theta = draw_mesh_and_pec(ax_re_theta, vertices, simplices, edge_index_map, boundary_edges_pec)
        # 実部の E_theta データを使用
        vmax_re = max(np.abs(E_theta_re_nodes).max(), 1e-9)
        vmin_re = -vmax_re
        # 虚部がある場合、それと範囲を合わせる
        vmax_im = 0.0
        if is_periodic and E_theta_im_nodes is not None:
             vmax_im = max(np.abs(E_theta_im_nodes).max(), 1e-9)
        vmax_both = max(vmax_re, vmax_im)
        vmin_both = -vmax_both

        norm_re = mcolors.Normalize(vmin=vmin_both, vmax=vmax_both) # <<<--- 共通の範囲を使う
        levels = np.linspace(vmin_both, vmax_both, 51)
        cont_re = ax_re_theta.tricontourf(tri_theta, E_theta_re_nodes, cmap=cmap_scalar, norm=norm_re, levels=levels, extend='both')
        cbar_re = fig.colorbar(cont_re, ax=ax_re_theta, fraction=0.046, pad=0.04)
        cbar_re.set_label(r'Re($E_\theta$)')
        cbar_re.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
        ax_re_theta.set_title(r'Re($E_\theta$) on Mesh')

    # --- Imaginary Part Plot (Periodic) ---
    if is_periodic and plot_type == 'real_imag':
        ax_im_vec = axes[0, 1]
        tri_im = draw_mesh_and_pec(ax_im_vec, vertices, simplices, edge_index_map, boundary_edges_pec)
        if Ez_im_q is not None and Ez_im_q.size > 0:
            q_im = ax_im_vec.quiver(grid_z_q[_inside], grid_r_q[_inside],
                                    Ez_im_q[_inside], Er_im_q[_inside], angles='xy',
                                    scale=vec_scale, scale_units='xy', pivot='mid',
                                    color=STYLE.QUIVER_IMAG_COLOR, alpha=STYLE.QUIVER_ALPHA, width=0.003)
            ax_im_vec.quiverkey(q_im, 0.85, 0.92, quiver_key_val, f'{quiver_key_val:.2e}', labelpos='E', coordinates='axes') # キー値修正
        ax_im_vec.set_title('Im(Ez, Er) on Grid', fontsize=STYLE.TITLE_FONTSIZE)

        if n_mode > 0 and E_theta_im_nodes is not None: # <<<--- E_theta_im_nodes をチェック
            ax_im_theta = axes[1, 1]
            tri_im_theta = draw_mesh_and_pec(ax_im_theta, vertices, simplices, edge_index_map, boundary_edges_pec)
            # 実部と同じカラースケール norm_re を使う
            levels_im = np.linspace(vmin_both, vmax_both, 51) # 実部と同じレベル
            cont_im = ax_im_theta.tricontourf(tri_im_theta, E_theta_im_nodes, cmap=cmap_scalar, norm=norm_re, levels=levels_im, extend='both')
            cbar_im = fig.colorbar(cont_im, ax=ax_im_theta, fraction=0.046, pad=0.04)
            cbar_im.set_label(r'Im($E_\theta$)')
            cbar_im.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
            ax_im_theta.set_title(r'Im($E_\theta$) on Mesh')

    # --- 全体タイトル (変更なし) ---
    title = f'Mode Index: {mode_index}, Freq: {freqGHz:.4f} GHz, n={n_mode}'
    if is_periodic and mode_data.get('theta_deg') is not None:
         title += f', Theta={mode_data["theta_deg"]:.1f} deg'
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- メインの可視化スクリプト ---
if __name__ == "__main__":
    h5_filepath = "result.h5" # 保存したファイル
    print_hdf5_summary(h5_filepath)

    # 可視化したいモードへのパスを指定
    result_path = "/results/n0/Normal"
    mode_idx_to_plot = 1

    # 1. データ読み込み
    mode_data = load_mode_data_from_hdf5(h5_filepath, result_path, mode_idx_to_plot)

    if mode_data:
        vertices = mode_data['vertices']
        simplices = mode_data['simplices']
        n_mode = mode_data['n_mode']

        # --- edge_index_map をここで再構築 ---
        edge_index_map = {tuple(map(int, key)): int(val) for key, val in zip(mode_data['edge_map_keys'], mode_data['edge_map_values'])}


        # 2. グリッド生成
        num_grid_points_z = 30 # Z方向のグリッド数
        num_grid_points_r = 50 # R方向のグリッド数
        z_min, z_max = vertices[:, 0].min(), vertices[:, 0].max()
        r_min, r_max = vertices[:, 1].min(), vertices[:, 1].max()
        # 少しだけ内側にグリッドを取ると補間が安定する場合がある
        grid_z_1d = np.linspace(z_min + 1e-6, z_max - 1e-6, num_grid_points_z)
        grid_r_1d = np.linspace(r_min + 1e-6, r_max - 1e-6, num_grid_points_r)
        grid_z, grid_r = np.meshgrid(grid_z_1d, grid_r_1d, indexing='ij')

        # 3. 電場補間
        interp_results = interpolate_fields_on_grid(
            grid_z, grid_r,
            mode_data['vertices'], mode_data['simplices'],
            edge_index_map, # <<<--- 再構築したものを渡す
            n_mode,
            # 周期/定在波に応じてデータを渡す
            edge_vec_re=mode_data['edge_vectors_re'], edge_vec_im=mode_data['edge_vectors_im'],
            E_theta_re=mode_data['E_theta_re'], E_theta_im=mode_data['E_theta_im'],
            edge_vec=mode_data['edge_vectors'], E_theta=mode_data['E_theta']
        )
        interp_Ez_re, interp_Er_re, interp_Etheta_re, \
        interp_Ez_im, interp_Er_im, interp_Etheta_im = interp_results

        # 4. プロット
        plot_mode_on_grid(
            grid_z, grid_r,
            interp_Ez_re, interp_Er_re, interp_Etheta_re,
            interp_Ez_im, interp_Er_im, interp_Etheta_im,
            mode_data,
            mode_idx_to_plot, # <<<--- モードインデックスを渡す
            vec_scale_factor = 1,
            #vec_density=2 # 2点ごとにベクトル表示 (間引き)
            # vec_scale=1e-5 # 必要なら手動指定
        )