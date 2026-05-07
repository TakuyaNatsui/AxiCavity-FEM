import argparse
import numpy as np
import h5py
import os
import sys
import shutil

# 同一階層のモジュールをインポート
sys.path.append(os.path.dirname(__file__))
from field_calculator import FieldCalculator
from FEM_element_function import calculate_boundary_integral_quadratic
from FEM_element_function import (
    grad_area_coordinates,
    calculate_quadratic_nodal_shape_functions,
    grad_quadratic_nodal_shape_functions
)

def load_sparse_matrix(group, name):
    from scipy.sparse import csr_matrix
    sparse_grp = group[name]
    data = sparse_grp['data'][:]
    indices = sparse_grp['indices'][:]
    indptr = sparse_grp['indptr'][:]
    shape = sparse_grp.attrs['shape']
    return csr_matrix((data, indices, indptr), shape=shape)

def calc_p_flow(elements, nodes, eigenvectors_mode, mesh_order, omega, cell_length):
    """z=z_min断面でのポインティングベクトルz成分の積分から電力流を計算する。

    P_flow = -(pi / omega*eps0) * integral_0^inf Im[dpsi/dz * psi*] r dr

    ここで psi = H_phi（DOF）、積分はz=z_min境界面のラインインテグラル。
    """
    eps0 = 8.8541878128e-12

    # z=z_min境界ノードの特定
    z_min = np.min(nodes[:, 0])
    z_tol = (np.max(nodes[:, 0]) - z_min) * 1e-6
    z_min_set = set(np.where(np.abs(nodes[:, 0] - z_min) < z_tol)[0])

    # 辺ID付きエッジ定義（辺IDで重心座標マッピングを決定）
    if mesh_order == 2:
        edge_defs = [(0, [0,1,3]), (1, [1,2,4]), (2, [2,0,5])]
    else:
        edge_defs = [(0, [0,1]),   (1, [1,2]),   (2, [2,0])]

    def bary(eid, t):
        """辺ID と t∈[0,1] から重心座標 (L1,L2,L3) を返す"""
        if eid == 0: return np.array([1-t, t,   0.0])
        if eid == 1: return np.array([0.0, 1-t, t  ])
        return             np.array([t,   0.0, 1-t])

    # z=z_min境界エッジを収集（重複防止のためsetで管理）
    seen_edges = set()
    boundary = []
    for elem in elements:
        for eid, idxs in edge_defs:
            global_nodes = tuple(elem[idxs])
            if all(n in z_min_set for n in global_nodes):
                key = tuple(sorted(global_nodes))
                if key not in seen_edges:
                    seen_edges.add(key)
                    boundary.append((eid, global_nodes, elem))

    # 3点ガウス求積
    xi_pts = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    weights = np.array([5.0/9, 8.0/9, 5.0/9])

    p_flow_total = 0.0

    for eid, edge_nodes, elem in boundary:
        vtx_coords = nodes[list(elem[:3])]       # 頂点のみ (3×2)
        psi_e = eigenvectors_mode[list(elem)]
        grad_L = grad_area_coordinates(vtx_coords)

        # エッジのJacobian（r方向の長さ、z=一定なので r成分のみ）
        r_end1 = nodes[edge_nodes[0], 1]
        r_end2 = nodes[edge_nodes[1], 1]
        L_edge = abs(r_end2 - r_end1)
        if L_edge < 1e-15:
            continue

        edge_sum = 0.0
        for xi, w in zip(xi_pts, weights):
            t = (xi + 1.0) / 2.0
            L_coords = bary(eid, t)

            if mesh_order == 1:
                G = L_coords
                grad_G = np.array(grad_L)          # (3, 2)
            else:
                G = calculate_quadratic_nodal_shape_functions(L_coords)
                grad_G = grad_quadratic_nodal_shape_functions(L_coords, grad_L)  # (6, 2)

            psi_pt  = np.dot(G, psi_e)
            dpsi_dz = np.dot(grad_G[:, 0], psi_e)
            r       = np.dot(L_coords, vtx_coords[:, 1])

            edge_sum += w * np.imag(dpsi_dz * np.conj(psi_pt)) * r

        p_flow_total += edge_sum * L_edge / 2.0

    return -(np.pi / (omega * eps0)) * p_flow_total


# ============================================================================
# Stage 1: パラメータ計算 + HDF5保存 (可視化なし)
# ============================================================================
def run_parameter_calculation(input_file, output_file, conductivity=5.8e7, n_scan=2000, beta=1.0):
    """
    FEM解析結果からQ値、R/Q、蓄積エネルギー等の工学パラメータを計算し、
    規格化した固有ベクトルとともにHDF5ファイルに保存する。
    可視化・レポート生成は行わない。
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found."); return

    # raw HDF5 をコピーして、追記モードで開く
    if os.path.abspath(input_file) != os.path.abspath(output_file):
        shutil.copy2(input_file, output_file)

    calc = FieldCalculator(input_file)
    analysis_type = calc.analysis_type
    print(f'Detected Analysis Type: {analysis_type}')

    mu0, c_light = 4.0 * np.pi * 1e-7, 299792458.0

    with h5py.File(input_file, 'r') as f:
        M_global = load_sparse_matrix(f, 'matrices/M_global')
        phase_shifts = f.attrs.get('phase_shifts', [0.0])

        pec_nodes = []
        if 'mesh/physical_groups' in f:
            for name in f['mesh/physical_groups'].keys():
                if 'PEC' in name.upper() or 'WALL' in name.upper():
                    pec_nodes = f[f'mesh/physical_groups/{name}'][:]
                    break
        elements = f['mesh/elements'][:]

    pec_set = set(pec_nodes)
    pec_edges = []
    edge_indices = [[0, 1, 3], [1, 2, 4], [2, 0, 5]] if calc.mesh_order == 2 else [[0, 1], [1, 2], [2, 0]]
    for elem in elements:
        for idxs in edge_indices:
            if all(n in pec_set for n in elem[idxs]): pec_edges.append(elem[idxs])
    pec_edges = sorted(list(set(tuple(sorted(e)) for e in pec_edges)))

    z_min, z_max = np.min(calc.nodes[:, 0]), np.max(calc.nodes[:, 0])
    cell_length = z_max - z_min
    z_scan = np.linspace(z_min, z_max, n_scan)
    dz = z_scan[1] - z_scan[0]

    # 結果格納用
    all_phase_results = {}
    all_normalized_eigvecs = {}
    all_norm_factors = {}
    all_ez_axial = {}

    for ph in phase_shifts:
        print(f"Processing Phase: {ph} deg...")
        calc.set_phase(ph)
        n_modes = len(calc.frequencies)
        res_ph = {k: [] for k in ['stored_energy', 'p_loss', 'q_factor', 'v_eff', 'rq_eff',
                                    'v_acc', 'rq_apparent',
                                    'r_shunt_eff', 'r_shunt_apparent',
                                    'r_shunt_eff_m', 'r_shunt_apparent_m',
                                    'p_flow', 'v_group_c', 'v_phase_c', 'attenuation', 'r_shunt_m']}
        norm_factors = []
        ez_axial_list = []
        normalized_eigvecs = np.zeros_like(calc.eigenvectors, dtype=complex)

        for m in range(n_modes):
            freq = calc.frequencies[m] * 1e9
            omega = 2 * np.pi * freq

            # Axial Field & Normalization
            ez_axial = []
            for zp in z_scan:
                f_pt = calc.calculate_fields(zp, 0.0, mode_index=m)
                ez_axial.append(f_pt['Ez'] if f_pt else 0.0)
            ez_axial = np.array(ez_axial)
            norm_factor = 1.0 / np.max(np.abs(ez_axial)) if np.max(np.abs(ez_axial)) > 0 else 1.0
            norm_factors.append(norm_factor)
            ez_axial_list.append(ez_axial * norm_factor)

            # Stored Energy U
            u_norm = (calc.eigenvectors[m] * norm_factor).astype(complex)
            u_vec = u_norm.reshape(-1, 1)
            uMu = np.vdot(u_vec, M_global @ u_vec)
            U = np.pi * mu0 * np.real(uMu)/2  #マトリクス構成のときに省略されたpiをかける

            # 規格化した固有ベクトルを保存
            normalized_eigvecs[m] = u_norm

            # 可視化用にセット (calculate_fields で使うため)
            if calc.eigenvectors.dtype != complex:
                calc.eigenvectors = calc.eigenvectors.astype(complex)
            calc.eigenvectors[m] = u_norm

            # Effective Voltage V_eff (with transit time factor)
            phase_factor = np.exp(1j * omega * z_scan / (beta * c_light))
            v_eff = np.abs(np.trapz(ez_axial * norm_factor * phase_factor, dx=dz))
            rq_eff = (v_eff**2) / (omega * U) if U > 0 else 0.0

            # Accelerating Voltage V_acc (without transit time factor)
            v_acc = np.abs(np.trapz(ez_axial * norm_factor, dx=dz))
            rq_apparent = (v_acc**2) / (omega * U) if U > 0 else 0.0

            # Wall Loss P_loss
            rs_ohm = np.sqrt(omega * mu0 / (2.0 * conductivity))
            integral_h2r = 0.0
            for edge in pec_edges:
                coords = calc.nodes[list(edge)]
                vals_abs = np.abs(u_norm[list(edge)])
                if calc.mesh_order == 2:
                    dist = [np.linalg.norm(coords[i]-coords[j]) for i,j in [(0,1), (0,2), (1,2)]]
                    max_idx = np.argmax(dist)
                    p_idxs = [(0,1,2), (0,2,1), (1,2,0)][max_idx]
                    integral_h2r += calculate_boundary_integral_quadratic(coords[list(p_idxs)], vals_abs[list(p_idxs)])
                else:
                    p1, p2 = coords; v1, v2 = vals_abs
                    integral_h2r += 0.5 * (v1**2 * p1[1] + v2**2 * p2[1]) * np.linalg.norm(p2 - p1)
            p_loss = np.pi * rs_ohm * integral_h2r
            q_factor = (omega * U / p_loss) if p_loss > 0 else 0.0

            # Shunt Impedance
            p_flow, v_group_c, v_phase_c, atten, r_shunt_m = 0, 0, 0, 0, 0
            r_shunt_eff, r_shunt_apparent = 0.0, 0.0
            r_shunt_eff_m, r_shunt_apparent_m = 0.0, 0.0
            if analysis_type == 'traveling_multi':
                p_flow = calc_p_flow(elements, calc.nodes, u_norm, calc.mesh_order, omega, cell_length)
                v_group_c = (p_flow * cell_length / U) / c_light if U > 0 else 0
                theta_rad = np.deg2rad(ph)
                v_phase_c = (omega * cell_length) / (theta_rad * c_light) if theta_rad > 1e-10 else 0.0
                atten = p_loss / (2 * p_flow * cell_length) if p_flow != 0 else 0
                r_shunt_m = (v_eff**2 / p_loss) / cell_length if p_loss > 0 else 0
            else:
                # Standing wave: shunt impedances
                r_shunt_eff = (v_eff**2 / p_loss) if p_loss > 0 else 0.0
                r_shunt_apparent = (v_acc**2 / p_loss) if p_loss > 0 else 0.0
                r_shunt_eff_m = r_shunt_eff / cell_length if cell_length > 0 else 0.0
                r_shunt_apparent_m = r_shunt_apparent / cell_length if cell_length > 0 else 0.0

            # Store results
            res_ph['stored_energy'].append(U)
            res_ph['p_loss'].append(p_loss)
            res_ph['q_factor'].append(q_factor)
            res_ph['v_eff'].append(v_eff)
            res_ph['rq_eff'].append(rq_eff)
            res_ph['v_acc'].append(v_acc)
            res_ph['rq_apparent'].append(rq_apparent)
            res_ph['r_shunt_eff'].append(r_shunt_eff)
            res_ph['r_shunt_apparent'].append(r_shunt_apparent)
            res_ph['r_shunt_eff_m'].append(r_shunt_eff_m)
            res_ph['r_shunt_apparent_m'].append(r_shunt_apparent_m)
            res_ph['p_flow'].append(p_flow)
            res_ph['v_group_c'].append(v_group_c)
            res_ph['v_phase_c'].append(v_phase_c)
            res_ph['attenuation'].append(atten)
            res_ph['r_shunt_m'].append(r_shunt_m)

            if analysis_type == 'standing':
                print(f"  Mode {m}: f={calc.frequencies[m]:.6f} GHz, Q={q_factor:.5e}, R/Q_eff={rq_eff:.4f}, Rs_eff={r_shunt_eff:.4e}, Rs_app={r_shunt_apparent:.4e} Ohm")
            else:
                print(f"  Mode {m}: f={calc.frequencies[m]:.6f} GHz, Q={q_factor:.5e}, R/Q={rq_eff:.4f} Ohm")

        all_phase_results[ph] = res_ph
        all_normalized_eigvecs[ph] = normalized_eigvecs
        all_norm_factors[ph] = np.array(norm_factors)
        all_ez_axial[ph] = np.array(ez_axial_list)

    # HDF5に結果を保存
    with h5py.File(output_file, 'a') as f:
        f.attrs['is_processed'] = True
        f.attrs['conductivity'] = conductivity
        f.attrs['beta'] = beta

        # post_processed グループ (既存なら削除して再作成)
        if 'post_processed' in f:
            del f['post_processed']
        pp_grp = f.create_group('post_processed')
        pp_grp.attrs['n_scan'] = n_scan
        pp_grp.attrs['cell_length'] = cell_length

        for ph in phase_shifts:
            ph_grp = pp_grp.create_group(f'phase_{ph}')
            ph_grp.create_dataset('normalized_eigenvectors', data=all_normalized_eigvecs[ph])
            ph_grp.create_dataset('norm_factors', data=all_norm_factors[ph])
            ph_grp.create_dataset('ez_axial', data=all_ez_axial[ph])
            ph_grp.create_dataset('z_scan', data=z_scan)

            eng_grp = ph_grp.create_group('engineering_parameters')
            res_ph = all_phase_results[ph]
            for key in ['stored_energy', 'p_loss', 'q_factor', 'v_eff', 'rq_eff',
                         'v_acc', 'rq_apparent',
                         'r_shunt_eff', 'r_shunt_apparent',
                         'r_shunt_eff_m', 'r_shunt_apparent_m',
                         'p_flow', 'v_group_c', 'v_phase_c', 'attenuation', 'r_shunt_m']:
                eng_grp.create_dataset(key, data=np.array(res_ph[key]))

            # 規格化した固有ベクトルで results 内の eigenvectors を上書き
            eigvec_path = f'results/phase_{ph}/eigenvectors'
            if eigvec_path in f:
                del f[eigvec_path]
                f[f'results/phase_{ph}'].create_dataset('eigenvectors', data=all_normalized_eigvecs[ph])
            elif 'results/eigenvectors' in f:
                del f['results/eigenvectors']
                f['results'].create_dataset('eigenvectors', data=all_normalized_eigvecs[ph])

    # サマリー表示とテキストファイルへの保存
    param_txt = os.path.splitext(output_file)[0] + "_parameters.txt"
    with open(param_txt, 'w') as ptxt:
        ptxt.write(f"Post-Process Parameters - {analysis_type.upper()}\n")
        ptxt.write("=" * 90 + "\n")
        ptxt.write(f"Conductivity: {conductivity:.2e} S/m\n")
        ptxt.write(f"Beta (v/c): {beta:.3f}\n\n")

        for ph in phase_shifts:
            res_ph = all_phase_results[ph]
            n_modes = len(res_ph['stored_energy'])

            with h5py.File(output_file, 'r') as f:
                freq_key = f'results/phase_{ph}/frequencies' if f'results/phase_{ph}/frequencies' in f else 'results/frequencies'
                freqs = f[freq_key][:]

            if analysis_type == 'standing':
                ptxt.write(f"Phase: {ph} deg\n")
                ptxt.write(f"{'Mode':>4} {'Freq [GHz]':>12} {'Q Factor':>14} {'V_acc [V]':>14} {'V_eff [V]':>14} "
                           f"{'R/Q_app [Ohm]':>14} {'R/Q_eff [Ohm]':>14} "
                           f"{'Rs_app [Ohm]':>14} {'Rs_eff [Ohm]':>14} "
                           f"{'Rs_app [O/m]':>14} {'Rs_eff [O/m]':>14} "
                           f"{'U [J]':>14} {'P_loss [W]':>14}\n")
                ptxt.write("-" * 200 + "\n")
                for m in range(n_modes):
                    ptxt.write(f"{m:4d} {freqs[m]:12.6f} {res_ph['q_factor'][m]:14.5e} "
                               f"{res_ph['v_acc'][m]:14.5e} {res_ph['v_eff'][m]:14.5e} "
                               f"{res_ph['rq_apparent'][m]:14.4f} {res_ph['rq_eff'][m]:14.4f} "
                               f"{res_ph['r_shunt_apparent'][m]:14.5e} {res_ph['r_shunt_eff'][m]:14.5e} "
                               f"{res_ph['r_shunt_apparent_m'][m]:14.5e} {res_ph['r_shunt_eff_m'][m]:14.5e} "
                               f"{res_ph['stored_energy'][m]:14.5e} {res_ph['p_loss'][m]:14.5e}\n")
                ptxt.write("\n")
            else:
                ptxt.write(f"Phase: {ph} deg\n")
                ptxt.write(f"{'Mode':>4} {'Freq [GHz]':>12} {'Vg [c]':>10} {'Vp [c]':>10} {'Atten [Np/m]':>14} {'U [J]':>14} {'P_loss [W]':>14} {'r_shunt [Ω/m]':>16}\n")
                ptxt.write("-" * 122 + "\n")
                for m in range(n_modes):
                    ptxt.write(f"{m:4d} {freqs[m]:12.6f} {res_ph['v_group_c'][m]:10.4f} {res_ph['v_phase_c'][m]:10.4f} {res_ph['attenuation'][m]:14.4e} {res_ph['stored_energy'][m]:14.4e} {res_ph['p_loss'][m]:14.4e} {res_ph['r_shunt_m'][m]:16.4e}\n")
                ptxt.write("\n")

    print(f"\n{'='*70}")
    print(f"Parameter Calculation Complete: {output_file}")
    print(f"Parameters saved to: {param_txt}")
    print(f"{'='*70}")
    for ph in phase_shifts:
        res_ph = all_phase_results[ph]
        n_modes = len(res_ph['stored_energy'])
        print(f"\n  Phase: {ph} deg")
        if analysis_type == 'standing':
            print(f"  {'Mode':>4} {'Freq [GHz]':>12} {'Q Factor':>14} {'R/Q_eff [Ohm]':>14} "
                  f"{'Rs_eff [Ohm]':>14} {'Rs_app [Ohm]':>14} "
                  f"{'Rs_eff [O/m]':>14} {'Rs_app [O/m]':>14} "
                  f"{'U [J]':>14} {'P_loss [W]':>14}")
            print(f"  {'-'*4} {'-'*12} {'-'*14} {'-'*14} "
                  f"{'-'*14} {'-'*14} "
                  f"{'-'*14} {'-'*14} "
                  f"{'-'*14} {'-'*14}")
            for m in range(n_modes):
                with h5py.File(output_file, 'r') as f:
                    freq_key = f'results/phase_{ph}/frequencies' if f'results/phase_{ph}/frequencies' in f else 'results/frequencies'
                    freq = f[freq_key][m]
                print(f"  {m:4d} {freq:12.6f} {res_ph['q_factor'][m]:14.5e} {res_ph['rq_eff'][m]:14.4f} "
                      f"{res_ph['r_shunt_eff'][m]:14.5e} {res_ph['r_shunt_apparent'][m]:14.5e} "
                      f"{res_ph['r_shunt_eff_m'][m]:14.5e} {res_ph['r_shunt_apparent_m'][m]:14.5e} "
                      f"{res_ph['stored_energy'][m]:14.5e} {res_ph['p_loss'][m]:14.5e}")
        else:
            print(f"  {'Mode':>4} {'Freq [GHz]':>12} {'Vg [c]':>10} {'Vp [c]':>10} {'Atten [Np/m]':>14} {'U [J]':>14} {'P_loss [W]':>14}")
            print(f"  {'-'*4} {'-'*12} {'-'*10} {'-'*10} {'-'*14} {'-'*14} {'-'*14}")
            for m in range(n_modes):
                with h5py.File(output_file, 'r') as f:
                    freq_key = f'results/phase_{ph}/frequencies' if f'results/phase_{ph}/frequencies' in f else 'results/frequencies'
                    freq = f[freq_key][m]
                print(f"  {m:4d} {freq:12.6f} {res_ph['v_group_c'][m]:10.4f} {res_ph['v_phase_c'][m]:10.4f} {res_ph['attenuation'][m]:14.4e} {res_ph['stored_energy'][m]:14.4e} {res_ph['p_loss'][m]:14.4e}")


# ============================================================================
# Stage 2: レポート生成 (可視化 + HTML)
# ============================================================================
def run_report_generation(processed_file, skip_anim=False):
    """
    処理済みHDF5ファイルからレポート（PNG/GIF画像 + HTML）を生成する。
    """
    import matplotlib.pyplot as plt
    from plot_utils import FEMPlotter

    if not os.path.exists(processed_file):
        print(f"Error: {processed_file} not found."); return

    # processed フラグの確認
    with h5py.File(processed_file, 'r') as f:
        if not f.attrs.get('is_processed', False):
            print("Error: This file has not been processed. Run parameter calculation first.")
            return
        conductivity = f.attrs.get('conductivity', 5.8e7)
        beta = f.attrs.get('beta', 1.0)
        phase_shifts = f.attrs.get('phase_shifts', [0.0])

        # PEC ノードの読み込み
        pec_nodes = []
        if 'mesh/physical_groups' in f:
            for name in f['mesh/physical_groups'].keys():
                if 'PEC' in name.upper() or 'WALL' in name.upper():
                    pec_nodes = f[f'mesh/physical_groups/{name}'][:]
                    break

    calc = FieldCalculator(processed_file)
    analysis_type = calc.analysis_type
    print(f'Detected Analysis Type: {analysis_type}')

    plotter = FEMPlotter(calc)
    report_dir = os.path.splitext(processed_file)[0] + "_report"
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    # メッシュ概要図
    fig_m, ax_m = plotter.setup_axes()
    ax_m.triplot(plotter.triang_raw, color='gray', linewidth=0.2, alpha=0.5)
    if len(pec_nodes) > 0: ax_m.plot(calc.nodes[pec_nodes, 0], calc.nodes[pec_nodes, 1], 'o', color='orange', markersize=2, label='PEC')
    ax_m.set_title(f"Mesh Overview: {len(calc.nodes)} nodes")
    fig_m.savefig(os.path.join(report_dir, "mesh_overview.png"), dpi=120)
    plt.close(fig_m)

    # 工学パラメータの読み込みと結果辞書の構築
    results_all = {}

    for ph in phase_shifts:
        print(f"Processing Phase: {ph} deg...")
        calc.set_phase(ph)
        n_modes = len(calc.frequencies)

        # HDF5から工学パラメータを読み込む
        with h5py.File(processed_file, 'r') as f:
            eng_path = f'post_processed/phase_{ph}/engineering_parameters'
            eng_grp = f[eng_path]
            res_ph = {}
            for key in ['stored_energy', 'p_loss', 'q_factor', 'v_eff', 'rq_eff',
                         'v_acc', 'rq_apparent',
                         'r_shunt_eff', 'r_shunt_apparent',
                         'r_shunt_eff_m', 'r_shunt_apparent_m',
                         'p_flow', 'v_group_c', 'attenuation', 'r_shunt_m']:
                if key in eng_grp:
                    res_ph[key] = eng_grp[key][:].tolist()
                else:
                    res_ph[key] = [0.0] * n_modes

        res_ph['images'] = []

        for m in range(n_modes):
            print(f"  Visualizing Mode {m}...")
            prefix = f"phase_{ph}_"

            if analysis_type == 'standing':
                imgs = plotter.plot_standing(m, output_dir=report_dir, prefix=prefix)
            else:
                imgs = plotter.plot_complex_components(m, output_dir=report_dir, prefix=prefix)

            # 軸上電場プロット
            imgs['axial'] = plotter.plot_axial_field(m, output_dir=report_dir, prefix=prefix)

            # 進行波のみアニメーション
            if analysis_type == 'traveling_multi' and not skip_anim:
                try:
                    imgs['anim'] = plotter.create_animation(m, os.path.join(report_dir, f"{prefix}mode_{m:02d}_anim.gif"))
                except Exception as e:
                    print(f"  [Warning] Animation for mode {m} failed: {e}")

            res_ph['images'].append(imgs)

        results_all[ph] = res_ph

    # HTML レポート生成
    report_info = {
        'input_file': processed_file,
        'mesh_order': calc.mesh_order,
        'conductivity': conductivity,
        'beta': beta,
        'average_mesh_size': calc.average_mesh_size
    }
    generate_html_report(report_dir, calc, results_all, analysis_type, report_info)
    print(f"\nReport generation completed. Report: {os.path.join(report_dir, 'index.html')}")


# ============================================================================
# 従来互換ラッパー: パラメータ計算 + レポート生成を一括実行
# ============================================================================
def run_post_process(input_file, output_file, conductivity=5.8e7, n_scan=2000, beta=1.0, skip_anim=False):
    """従来互換: パラメータ計算とレポート生成を一括実行する。"""
    run_parameter_calculation(input_file, output_file, conductivity, n_scan, beta)
    run_report_generation(output_file, skip_anim=skip_anim)


# ============================================================================
# HTML レポート生成
# ============================================================================
def generate_html_report(output_dir, calc, results_all, analysis_type, report_info):
    """
    統合されたHTMLレポートを生成する
    """
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FEM Analysis Report - {analysis_type.capitalize()}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f4f7f6; color: #333; }}
        .header {{ text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 10px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
        th {{ background-color: #2c3e50; color: white; text-align: center; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .mode-card {{ margin-top: 50px; border-top: 2px solid #3498db; padding-top: 20px; }}
        .image-grid {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; }}
        .img-box {{ flex: 1; min-width: 350px; text-align: center; }}
        .img-box img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .anim-box {{ max-width: 600px; margin: 20px auto; }}
        .tag {{ padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
        .tag-standing {{ background: #e74c3c; color: white; }}
        .tag-traveling {{ background: #f39c12; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FEM Analysis Report</h1>
        <p>Generated on: {now} | Type: <span class="tag tag-{analysis_type}">{analysis_type.upper()}</span></p>
    </div>

    <div class="card">
        <h2>Mesh & Analysis Info</h2>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p><strong>Input File:</strong> {report_info['input_file']}</p>
                <p><strong>Nodes:</strong> {len(calc.nodes)} | <strong>Elements:</strong> {len(calc.elements)} ({report_info['mesh_order']}-order)</p>
                <p><strong>Conductivity:</strong> {report_info['conductivity']:.2e} S/m</p>
                <p><strong>Particle Beta:</strong> {report_info['beta']:.3f}</p>
                <p><strong>Average mesh size:</strong> {report_info['average_mesh_size']:.3e}</p>
            </div>
            <div style="width: 40%;"><img src="mesh_overview.png" style="width: 100%; border: 1px solid #ccc;"></div>
        </div>
    </div>

    <h2>Results Summary</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Mode</th><th>Freq [GHz]</th>
                    {"<th>Q Factor</th><th>R/Q eff [&Omega;]</th><th>Rs eff [&Omega;]</th><th>Rs app [&Omega;]</th><th>Rs eff [&Omega;/m]</th><th>Rs app [&Omega;/m]</th>" if analysis_type == 'standing' else "<th>Vg [c]</th><th>Atten [Np/m]</th>"}
                    <th>Stored Energy [J]</th><th>Wall Loss [W]</th>
                </tr>
            </thead>
            <tbody>"""

    # 最初の位相（または単一結果）のサマリーを表示
    ph_first = list(results_all.keys())[0]
    res_list = results_all[ph_first]
    for i in range(len(calc.frequencies)):
        if analysis_type == 'standing':
            html += (f"<tr><td>{i}</td><td>{calc.frequencies[i]:.6f}</td>"
                     f"<td>{res_list['q_factor'][i]:.5e}</td>"
                     f"<td>{res_list['rq_eff'][i]:.4f}</td>"
                     f"<td>{res_list['r_shunt_eff'][i]:.4e}</td>"
                     f"<td>{res_list['r_shunt_apparent'][i]:.4e}</td>"
                     f"<td>{res_list['r_shunt_eff_m'][i]:.4e}</td>"
                     f"<td>{res_list['r_shunt_apparent_m'][i]:.4e}</td>"
                     f"<td>{res_list['stored_energy'][i]:.5e}</td>"
                     f"<td>{res_list['p_loss'][i]:.5e}</td></tr>")
        else:
            html += f"<tr><td>{i}</td><td>{calc.frequencies[i]:.6f}</td><td>{res_list['v_group_c'][i]:.4f}</td><td>{res_list['attenuation'][i]:.4e}</td><td>{res_list['stored_energy'][i]:.4e}</td><td>{res_list['p_loss'][i]:.4e}</td></tr>"

    html += """</tbody></table></div>"""

    for ph, res_data in results_all.items():
        html += f"<h2 id='phase_{ph}'>Detailed Distributions (Phase Shift: {ph}°)</h2>"
        for m in range(len(calc.frequencies)):
            imgs = res_data['images'][m]
            html += f"""
            <div class="card mode-card">
                <h3>Mode {m}: {calc.frequencies[m]:.6f} GHz</h3>
                <div class="image-grid">"""

            if analysis_type == 'standing':
                html += f'<div class="img-box"><img src="{imgs["standing"]}"></div>'
            else:
                html += f'<div class="img-box"><p>Real Part</p><img src="{imgs["real"]}"></div>'
                html += f'<div class="img-box"><p>Imaginary Part</p><img src="{imgs["imag"]}"></div>'
                html += f'<div class="img-box"><p>Magnitude</p><img src="{imgs["abs"]}"></div>'

            # 軸上電場プロットを追加
            if imgs.get('axial'):
                html += f'<div class="img-box"><p>Axial Ez</p><img src="{imgs["axial"]}"></div>'

            html += "</div>"

            if imgs.get('anim'):
                html += f'<div class="anim-box"><p style="text-align:center"><b>Time Evolution Animation</b></p><img src="{imgs["anim"]}" style="width:100%"></div>'

            html += f"""<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-top:20px;">
                <div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Stored Energy:</b> {res_data['stored_energy'][m]:.5e} J</div>
                <div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Wall Loss:</b> {res_data['p_loss'][m]:.5e} W</div>"""

            if analysis_type == 'standing':
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Q Factor:</b> {res_data["q_factor"][m]:.5e}</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>R/Q effective:</b> {res_data["rq_eff"][m]:.4f} &Omega;</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>R/Q apparent:</b> {res_data["rq_apparent"][m]:.4f} &Omega;</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Rs effective:</b> {res_data["r_shunt_eff"][m]:.4e} &Omega;</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Rs apparent:</b> {res_data["r_shunt_apparent"][m]:.4e} &Omega;</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Rs eff/m:</b> {res_data["r_shunt_eff_m"][m]:.4e} &Omega;/m</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Rs app/m:</b> {res_data["r_shunt_apparent_m"][m]:.4e} &Omega;/m</div>'
            else:
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Group Velocity:</b> {res_data["v_group_c"][m]:.4f} c</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Attenuation:</b> {res_data["attenuation"][m]:.4e} Np/m</div>'
                html += f'<div class="card" style="box-shadow:none; border:1px solid #eee;"><b>Shunt Imp. r:</b> {res_data["r_shunt_m"][m]:.4e} &Omega;/m</div>'

            html += "</div></div>"

    html += "</body></html>"
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f: f.write(html)


# ============================================================================
# CLI エントリポイント
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified FEM Post-Processor')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input HDF5 file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output HDF5 file (for calc mode)')
    parser.add_argument('-c', '--cond', type=float, default=5.8e7, help='Conductivity [S/m]')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='Particle beta (v/c)')
    parser.add_argument('--no-anim', action='store_true', help='Skip animation generation')
    parser.add_argument('--mode', choices=['all', 'calc', 'report'], default='all',
                        help='all=both stages, calc=parameters only, report=report only')
    args = parser.parse_args()

    if args.mode == 'calc':
        if not args.output:
            args.output = os.path.splitext(args.input)[0] + "_processed.h5"
        run_parameter_calculation(args.input, args.output, args.cond, beta=args.beta)
    elif args.mode == 'report':
        run_report_generation(args.input, skip_anim=args.no_anim)
    else:
        if not args.output:
            args.output = os.path.splitext(args.input)[0] + "_processed.h5"
        run_post_process(args.input, args.output, args.cond, beta=args.beta, skip_anim=args.no_anim)
