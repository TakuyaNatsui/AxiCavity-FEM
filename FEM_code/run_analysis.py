import argparse
import numpy as np
import os
import h5py
from FEM_helmholtz_TM0_calclation import run_fem_analysis_standingTM0, run_fem_analysis_travelingTM0

def parse_phase_arg(phase_str):
    phases = []
    for part in phase_str.split(','):
        part = part.strip()
        if not part: continue
        if ':' in part:
            comps = part.split(':')
            if len(comps) == 3:
                start, end, step = map(float, comps)
                phases.extend(np.arange(start, end + step*1e-5, step).tolist())
            elif len(comps) == 2:
                start, end = map(float, comps)
                phases.extend([start, end])
        else:
            phases.append(float(part))
    return sorted(list(set(phases)))

def save_sparse_matrix(group, matrix, name):
    """疎行列を HDF5 グループ内に保存するヘルパー関数"""
    sparse_grp = group.create_group(name)
    # CSR形式で保存
    csr = matrix.tocsr()
    sparse_grp.create_dataset('data', data=csr.data)
    sparse_grp.create_dataset('indices', data=csr.indices)
    sparse_grp.create_dataset('indptr', data=csr.indptr)
    sparse_grp.attrs['shape'] = csr.shape

def main():
    parser = argparse.ArgumentParser(description='FEM Helmholtz TM0 Analysis Runner')
    parser.add_argument('-m', '--mesh', type=str, default='./mesh_data/sphere_2nd_order_01.msh',
                        dest='mesh_file',
                        help='Path to the mesh file (.msh)')
    parser.add_argument('--elem-order', type=int, default=2, choices=[1, 2],
                        dest='elem_order',
                        help='FEM element order (1 for linear, 2 for quadratic)')
    parser.add_argument('--num-modes', type=int, default=10,
                        dest='num_modes',
                        help='Number of eigenmodes to calculate')
    parser.add_argument('-o', '--output', type=str, default='analysis_result.h5',
                        dest='output_file',
                        help='Output filename to save results (.h5)')
    parser.add_argument('-p', '--phase', type=str, default='0.0',
                        help='Phase shift(s) in deg (e.g. "120", "60,90", "0:180:20")')

    args = parser.parse_args()
    phase_list = parse_phase_arg(args.phase)

    if not os.path.exists(args.mesh_file):
        print(f"Error: Mesh file not found: {args.mesh_file}")
        return

    print(f"Starting analysis: {args.mesh_file} (elem-order: {args.elem_order}, num-modes: {args.num_modes})")

    try:
        if len(phase_list) == 1 and phase_list[0] == 0.0:
            print("Mode: Standing Wave Analysis")
            results = run_fem_analysis_standingTM0(args.mesh_file, mesh_order=args.elem_order, num_modes=args.num_modes)
        else:
            print(f"Mode: Traveling Wave Analysis (Phase Shifts: {phase_list})")
            results = run_fem_analysis_travelingTM0(args.mesh_file, mesh_order=args.elem_order, num_modes=args.num_modes, phase_shifts=phase_list)

        # 結果の保存 (HDF5形式)
        print(f"Saving results to: {args.output_file}")
        with h5py.File(args.output_file, 'w') as f:
            # メッシュ情報グループ
            grp_mesh = f.create_group('mesh')
            grp_mesh.create_dataset('nodes', data=results["nodes"])
            grp_mesh.create_dataset('elements', data=results["elements"])
            grp_mesh.create_dataset('r0_nodes', data=results["r0_nodes"])
            grp_mesh.attrs['order'] = results["mesh_order"]
            
            # 物理グループの保存
            if "physical_groups" in results:
                grp_phys = grp_mesh.create_group('physical_groups')
                for name, node_indices in results["physical_groups"].items():
                    grp_phys.create_dataset(name, data=node_indices)
            
            # 疎行列の保存 (蓄積エネルギー計算用)
            if "M_global" in results:
                save_sparse_matrix(f, results["M_global"], 'matrices/M_global')
            
            # 解析結果グループ
            grp_res = f.create_group('results')
            
            if results["analysis_type"] == "standing":
                grp_res.create_dataset('frequencies', data=results["frequencies"])
                grp_res.create_dataset('eigenvectors', data=results["eigenvectors"])
                f.attrs['phase_shift'] = 0.0

                print(f"\nCalculated Frequencies [GHz]:")
                for i, freq in enumerate(results["frequencies"]):
                    print(f"  Mode {i:2d}: {freq:12.8f} GHz")

                # 周波数をテキストファイルに保存
                freq_txt = os.path.splitext(args.output_file)[0] + "_frequencies.txt"
                with open(freq_txt, 'w') as ftxt:
                    ftxt.write("Mode\tFrequency [GHz]\n")
                    ftxt.write("-" * 30 + "\n")
                    for i, freq in enumerate(results["frequencies"]):
                        ftxt.write(f"{i:4d}\t{freq:16.8f}\n")
                print(f"\nFrequencies saved to: {freq_txt}")
            else:
                phase_shifts_saved = []
                for ph, res_p in results["phase_results"].items():
                    sub_grp = grp_res.create_group(f"phase_{ph}")
                    sub_grp.create_dataset('frequencies', data=res_p["frequencies"])
                    sub_grp.create_dataset('eigenvectors', data=res_p["eigenvectors"])
                    sub_grp.attrs['eigenvalue_solve_time'] = res_p["eigenvalue_solve_time"]
                    phase_shifts_saved.append(ph)
                    print(f"\n--- Phase Shift {ph} deg Frequencies [GHz] ---")
                    for i, freq in enumerate(res_p["frequencies"]):
                        print(f"  Mode {i:2d}: {freq:12.8f} GHz")
                f.attrs['phase_shifts'] = phase_shifts_saved

                # 周波数をテキストファイルに保存（最初の位相）
                freq_txt = os.path.splitext(args.output_file)[0] + "_frequencies.txt"
                with open(freq_txt, 'w') as ftxt:
                    ftxt.write("Traveling Wave Analysis - Frequencies [GHz]\n")
                    ftxt.write("=" * 50 + "\n\n")
                    for ph in phase_shifts_saved:
                        res_p = results["phase_results"][ph]
                        ftxt.write(f"Phase Shift: {ph} deg\n")
                        ftxt.write("Mode\tFrequency [GHz]\n")
                        ftxt.write("-" * 30 + "\n")
                        for i, freq in enumerate(res_p["frequencies"]):
                            ftxt.write(f"{i:4d}\t{freq:16.8f}\n")
                        ftxt.write("\n")
                print(f"\nFrequencies saved to: {freq_txt}")

            # メタデータ
            f.attrs['description'] = 'Helmholtz TM0 Analysis Results'
            f.attrs['created_by'] = 'FEM_helmholtz_TM0_calclation.py'
            f.attrs['matrix_assembly_time'] = results.get("matrix_assembly_time", 0.0)
            f.attrs['analysis_type'] = results.get("analysis_type", "unknown")

        print(f"\nAnalysis completed successfully.")
            
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
