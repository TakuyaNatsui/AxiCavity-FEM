"""
run_analysis.py
HOM (Higher-Order Mode) 解析のメインスクリプト

元の FEM_helmholtz_calclation.py の __main__ ブロックを整理し、
分離された各モジュールを呼び出す構成に変更。

使用例:
  python run_analysis.py -m cavity.msh --az-order 0 1 2 --num-modes 10 -o result.h5
  python run_analysis.py -m cavity.msh --az-order 0 -p 60
  python run_analysis.py -m cavity.msh --az-order 1 -p "0:180:20"
"""

import sys
import numpy as np
import h5py

from calclation_parser import parse_calclation_args
from mesh_reader import load_gmsh_mesh_hom
from element_assembly import (assemble_global_matrices, assemble_global_matrices_2nd,
                              assemble_global_matrices_2nd_vectorized,
                              assemble_global_matrices_vectorized)
from boundary_conditions import (
    create_transformation_matrix,
    apply_bc_transformation,
    reconstruct_eigenvector_transformation,
    find_periodic_boundary_pairs,
    create_complex_transformation_matrix,
    create_complex_transformation_matrix_2nd,
    apply_bc_transformation_hermitian,
    get_pec_dof_indices_2nd,
)
from eigensolver import solve_eigenmodes_eigsh
from save_function import (
    save_mode_to_hdf5,
    save_mesh_and_params_to_hdf5,
    save_boundaries_to_hdf5,
)


def run_standing_wave(hf, n, simplices, vertices, edge_index_map, num_edges,
                      boundary_edges_pec, boundary_nodes_pec,
                      K_global, M_global, args, result_base_path,
                      pec_loss_edges=None):
    """定在波モードの計算と保存

    Args:
        hf: 開いている HDF5 ファイルオブジェクト
        n: 方位角モード次数
        その他: メッシュ・行列・境界条件情報

    Returns:
        frequencies_list: (mode_index, eigenvalue, frequency_GHz) のリスト
    """
    print("--- Calculating Normal (Standing Wave) Modes ---")
    elem_order   = args.elem_order
    num_elements = len(simplices)
    num_nodes    = len(vertices)
    calc_mode_path = f"{result_base_path}/Normal"
    bc_path = f"{calc_mode_path}/parameters"
    frequencies_list = []

    save_boundaries_to_hdf5(
        hf, bc_path, boundary_edges_pec,
        boundary_nodes_pec if n > 0 else None,
        pec_loss_edges=pec_loss_edges)

    # 要素次数に応じた境界DOFインデックスと全体DOF数
    if elem_order == 2:
        boundary_dofs = get_pec_dof_indices_2nd(
            num_edges, num_elements, num_nodes, n,
            boundary_edges_pec, boundary_nodes_pec)
        matrix_size_original = (2 * num_edges + 2 * num_elements
                                 + (num_nodes if n > 0 else 0))
    else:
        boundary_dofs = list(boundary_edges_pec)
        if n > 0:
            boundary_dofs += [num_edges + idx for idx in boundary_nodes_pec]
            boundary_dofs = sorted(set(boundary_dofs))
        matrix_size_original = num_edges + (num_nodes if n > 0 else 0)
    print(f"境界自由度: 計 {len(boundary_dofs)} 個 (elem_order={elem_order})")

    # 変換行列 T
    T, internal_indices = create_transformation_matrix(
        matrix_size_original, boundary_dofs)

    # 行列の縮小
    K_reduced, M_reduced = apply_bc_transformation(K_global, M_global, T)
    matrix_size_reduced = K_reduced.shape[0]

    if matrix_size_reduced == 0:
        print("Reduced matrix size is zero. Cannot solve.")
        return frequencies_list

    # 固有値問題を解く
    num_eigenmodes_to_solve = args.num_modes * 3
    if num_eigenmodes_to_solve >= matrix_size_reduced:
        num_eigenmodes_to_solve = max(1, matrix_size_reduced - 1)

    r_max = np.max(vertices[:, 1])
    freq_expct = 299792458 / (2 * r_max)
    sigma = (2 * np.pi * freq_expct / 299792458) ** 2
    print(f"sigma = {sigma:.4e}, freq_expct = {freq_expct * 1e-9:.3f} GHz "
          f"(r_max: {r_max:.6f})")

    try:
        eigenvalues, eigenvectors_reduced = solve_eigenmodes_eigsh(
            K_reduced, M_reduced,
            num_eigenmodes=num_eigenmodes_to_solve, sigma=sigma)

        if eigenvalues is None:
            print("Eigenvalue solver returned None.")
            return frequencies_list

        print("Eigenvalues (k^2):", eigenvalues)

        # 固有モード保存
        mode_count_saved = 0
        for i in range(len(eigenvalues)):
            eigenvalue = eigenvalues[i]
            if eigenvalue > sigma / 100:  # 物理的な解を選択
                if mode_count_saved < args.num_modes:
                    print(f"  Processing mode {i} (k^2={eigenvalue:.4e})...")
                    eigenvector_full = reconstruct_eigenvector_transformation(
                        eigenvectors_reduced[:, i], T)
                    save_mode_to_hdf5(
                        hf, calc_mode_path, mode_count_saved, eigenvalue,
                        eigenvector_full, n, vertices, simplices, num_edges,
                        is_periodic=False, elem_order=elem_order)

                    # 周波数を計算 (GHz単位)
                    k = np.sqrt(eigenvalue)
                    frequency_hz = (299792458 / (2 * np.pi)) * k
                    frequency_ghz = frequency_hz / 1e9
                    frequencies_list.append((mode_count_saved, eigenvalue, frequency_ghz))

                    mode_count_saved += 1
                else:
                    break

    except Exception as e:
        print(f"Eigenvalue solution failed: {e}")

    return frequencies_list


def run_traveling_wave(hf, n, simplices, vertices, edge_index_map, num_edges,
                       boundary_edges_pec, boundary_nodes_pec,
                       K_global, M_global, args, result_base_path, phase_list,
                       pec_loss_edges=None):
    """進行波（周期境界条件）モードの計算と保存

    複素変換行列 T (N×M, complex128) を用いて K_reduced = T†KT を eigsh で解く。

    Returns:
        frequencies_dict: {theta_deg: [(mode_index, eigenvalue, frequency_GHz), ...]}
    """
    theta_deg_array = np.array(phase_list)
    print(f"Theta [deg] to be calculated: {theta_deg_array}")
    elem_order   = args.elem_order
    num_nodes    = len(vertices)
    num_elements = len(simplices)
    frequencies_dict = {}

    for theta_deg in theta_deg_array:
        theta_rad = np.deg2rad(theta_deg)
        theta_group_name = f"PB_Phase_{theta_deg:05.1f}_deg".replace('.', '_')
        calc_mode_path = f"{result_base_path}/Periodic/{theta_group_name}"
        print(f"\n--- Processing theta = {theta_deg:.1f} deg "
              f"({theta_rad:.3f} rad) ---")
        print(f"  Saving results to: {calc_mode_path}")

        # 周期境界ペアの検出
        edge_pairs, node_pairs, z_min, z_max = find_periodic_boundary_pairs(
            vertices, simplices, edge_index_map, num_edges, n, tol=1e-5)

        # 境界条件の保存
        bc_path = f"{calc_mode_path}/parameters"
        save_boundaries_to_hdf5(
            hf, bc_path, boundary_edges_pec,
            boundary_nodes_pec if n > 0 else None,
            edge_pairs, node_pairs if n > 0 else None,
            pec_loss_edges=pec_loss_edges)

        L_pbc = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        beta = theta_rad / L_pbc if L_pbc > 1e-9 else 0
        hf[bc_path].attrs["beta_target"] = beta
        hf[bc_path].attrs["theta_rad"] = theta_rad
        hf[bc_path].attrs["theta_deg"] = theta_deg

        # 複素変換行列 T (N×M, complex128) — 要素次数に応じて切り替え
        if elem_order == 2:
            T, internal_indices = create_complex_transformation_matrix_2nd(
                num_nodes, num_edges, num_elements, n, theta_rad,
                edge_pairs, node_pairs,
                boundary_edges_pec, boundary_nodes_pec)
        else:
            T, internal_indices = create_complex_transformation_matrix(
                num_nodes, num_edges, n, theta_rad,
                edge_pairs, node_pairs,
                boundary_edges_pec, boundary_nodes_pec)

        # 行列の縮小 (T† K T → 複素エルミート行列)
        K_reduced, M_reduced = apply_bc_transformation_hermitian(
            K_global, M_global, T)
        matrix_size_reduced = K_reduced.shape[0]

        if matrix_size_reduced == 0:
            print("Reduced matrix size is zero. Cannot solve.")
            continue

        # 固有値問題を解く (eigsh は複素エルミート対応)
        num_eigenmodes_to_solve = args.num_modes * 3
        if num_eigenmodes_to_solve >= matrix_size_reduced:
            num_eigenmodes_to_solve = max(1, matrix_size_reduced - 1)

        r_max = np.max(vertices[:, 1])
        freq_expct = 299792458 / (2 * r_max)
        sigma = (2 * np.pi * freq_expct / 299792458) ** 2

        try:
            eigenvalues, eigenvectors_reduced = solve_eigenmodes_eigsh(
                K_reduced, M_reduced,
                num_eigenmodes=num_eigenmodes_to_solve, sigma=sigma)

            if eigenvalues is None:
                print("Eigenvalue solver returned None.")
                continue

            print("Eigenvalues (k^2):", eigenvalues)

            # 固有モード保存
            mode_count_saved = 0
            theta_frequencies = []
            for i in range(len(eigenvalues)):
                eigenvalue = eigenvalues[i]
                if eigenvalue > sigma / 1000:
                    if mode_count_saved < args.num_modes:
                        print(f"  Processing mode {i} "
                              f"(k^2={eigenvalue:.4e})...")
                        # N次元の複素固有ベクトルを復元
                        eigenvector_full = T @ eigenvectors_reduced[:, i]

                        save_mode_to_hdf5(
                            hf, calc_mode_path, mode_count_saved, eigenvalue,
                            eigenvector_full, n, vertices, simplices,
                            num_edges, is_periodic=True,
                            elem_order=elem_order)

                        mode_group_path = \
                            f"{calc_mode_path}/mode_{mode_count_saved}"
                        if mode_group_path in hf:
                            hf[mode_group_path].attrs["beta_target"] = beta

                        # 周波数を計算 (GHz単位)
                        k = np.sqrt(eigenvalue)
                        frequency_hz = (299792458 / (2 * np.pi)) * k
                        frequency_ghz = frequency_hz / 1e9
                        theta_frequencies.append((mode_count_saved, eigenvalue, frequency_ghz))

                        mode_count_saved += 1
                #else :
                #    print(f"sprias mode i = {i}")

            frequencies_dict[theta_deg] = theta_frequencies

        except Exception as e:
            print(f"Eigenvalue solution failed: {e}")

    return frequencies_dict


# ==========================================================================
# 周波数保存ヘルパー関数
# ==========================================================================
def _save_frequencies_to_txt(all_frequencies, output_hdf5_file):
    """周波数情報をテキストファイルに保存

    Args:
        all_frequencies: {n: {"Normal"|"Periodic": frequencies_list_or_dict}}
        output_hdf5_file: HDF5 出力ファイルのパス
    """
    # HDF5ファイルの拡張子を.txtに置き換える
    output_txt_file = output_hdf5_file.replace('.h5', '_frequencies.txt')
    if output_txt_file.endswith('.hdf5'):
        output_txt_file = output_txt_file.replace('.hdf5', '_frequencies.txt')
    elif not output_txt_file.endswith('_frequencies.txt'):
        output_txt_file = output_hdf5_file + '_frequencies.txt'

    try:
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            f.write("Frequency Analysis Results\n")
            f.write("=" * 80 + "\n\n")

            for n in sorted(all_frequencies.keys()):
                mode_data = all_frequencies[n]
                f.write(f"Azimuthal Mode Order n = {n}\n")
                f.write("-" * 80 + "\n")

                if "Normal" in mode_data:
                    # 定在波モード
                    frequencies_list = mode_data["Normal"]
                    f.write("Standing Wave (Normal) Modes:\n")
                    f.write(f"{'Mode #':<8} {'k^2 (eigenvalue)':<25} {'Frequency (GHz)':<20}\n")
                    f.write("-" * 80 + "\n")

                    for mode_idx, eigenvalue, freq_ghz in frequencies_list:
                        f.write(f"{mode_idx:<8} {eigenvalue:<25.15e} {freq_ghz:<20.15g}\n")

                elif "Periodic" in mode_data:
                    # 進行波モード（位相ごと）
                    frequencies_dict = mode_data["Periodic"]
                    f.write("Traveling Wave (Periodic) Modes:\n\n")

                    for theta_deg in sorted(frequencies_dict.keys()):
                        frequencies_list = frequencies_dict[theta_deg]
                        f.write(f"  Phase theta = {theta_deg:.1f} deg:\n")
                        f.write(f"  {'Mode #':<8} {'k^2 (eigenvalue)':<25} {'Frequency (GHz)':<20}\n")
                        f.write("  " + "-" * 78 + "\n")

                        for mode_idx, eigenvalue, freq_ghz in frequencies_list:
                            f.write(f"  {mode_idx:<8} {eigenvalue:<25.15e} {freq_ghz:<20.15g}\n")

                        f.write("\n")

                f.write("\n")

        print(f"周波数をテキストファイルに保存しました: {output_txt_file}")

    except Exception as e:
        print(f"警告: 周波数ファイルの保存に失敗しました: {e}", file=sys.stderr)


# ==========================================================================
# メイン処理
# ==========================================================================
def main():
    try:
        args = parse_calclation_args()
        print("\n--- 解析条件 ---")
        params_to_save = vars(args).copy()
        for key, value in params_to_save.items():
            print(f"{key}: {value}")
        print("---------------")
    except Exception as e:
        print(f"\n引数解析エラー: {e}", file=sys.stderr)
        sys.exit(1)

    msh_file = args.mesh_file
    output_hdf5_file = args.output_file
    phase_list = args.phase_list
    is_traveling = not (len(phase_list) == 1 and phase_list[0] == 0.0)
    print(f"結果保存ファイル: {output_hdf5_file}")
    print(f"波動タイプ: {'進行波 (Traveling)' if is_traveling else '定在波 (Standing)'}")

    try:
        # 周波数情報を収集するための辞書
        all_frequencies = {}

        with h5py.File(output_hdf5_file, 'w') as hf:
            # メッシュとパラメータの保存 (n=0 で代表的に読み込み)
            print("メッシュ情報を読み込み中 (保存用)...")
            simplices_mesh, vertices_mesh, edge_index_map_mesh, \
                num_edges_mesh, _, _, _, _ = load_gmsh_mesh_hom(
                    msh_file, 0, element_order=args.elem_order)

            save_mesh_and_params_to_hdf5(
                hf, vertices_mesh, simplices_mesh,
                edge_index_map_mesh, params_to_save)
            del simplices_mesh, vertices_mesh, edge_index_map_mesh
            print("-" * 40)

            # --- モード次数ループ ---
            for n in args.az_order:
                print(f"\n{'=' * 60}")
                print(f"  方位角モード次数 n = {n}")
                print(f"{'=' * 60}")

                # 1. メッシュ読み込み
                print(f"Mesh読み込み: {msh_file}")
                (simplices, vertices, edge_index_map, num_edges,
                 boundary_edges_pec, boundary_nodes_pec, physical_groups,
                 pec_loss_edges
                 ) = load_gmsh_mesh_hom(msh_file, n, element_order=args.elem_order)

                num_elements = len(simplices)
                print(f"  num_edges: {num_edges}")
                print(f"  num_nodes: {len(vertices)}")
                print(f"  num_elements: {num_elements}")

                # 2. 全体マトリクスアセンブリ (要素次数に応じて切り替え)
                print(f"--- Matrix Assembri mode:{n} ---")
                if args.elem_order == 2:
                    K_global, M_global = assemble_global_matrices_2nd_vectorized(
                        simplices, vertices, edge_index_map, num_edges, n)
                    matrix_size = (2 * num_edges + 2 * num_elements
                                   + (len(vertices) if n > 0 else 0))
                else:
                    K_global, M_global = assemble_global_matrices_vectorized(
                        simplices, vertices, edge_index_map, num_edges, n)
                    matrix_size = num_edges + (len(vertices) if n > 0 else 0)
                print(f"  matrix_size: {matrix_size}")

                result_base_path = f"/results/n{n}"

                # 3. 定在波 or 進行波
                if not is_traveling:
                    frequencies = run_standing_wave(
                        hf, n, simplices, vertices, edge_index_map,
                        num_edges, boundary_edges_pec, boundary_nodes_pec,
                        K_global, M_global, args, result_base_path,
                        pec_loss_edges=pec_loss_edges)
                    all_frequencies[n] = {"Normal": frequencies}
                else:
                    frequencies_dict = run_traveling_wave(
                        hf, n, simplices, vertices, edge_index_map,
                        num_edges, boundary_edges_pec, boundary_nodes_pec,
                        K_global, M_global, args, result_base_path,
                        phase_list=phase_list,
                        pec_loss_edges=pec_loss_edges)
                    all_frequencies[n] = {"Periodic": frequencies_dict}

        # HDF5ファイルを閉じた後、周波数をテキストファイルに保存
        _save_frequencies_to_txt(all_frequencies, output_hdf5_file)

    except FileNotFoundError:
        print(f"エラー: メッシュファイル '{msh_file}' が見つかりません。",
              file=sys.stderr)
        sys.exit(1)
    except ImportError:
        print("エラー: 必要なライブラリがインストールされていません。",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
