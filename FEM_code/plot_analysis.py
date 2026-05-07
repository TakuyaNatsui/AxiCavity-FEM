import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from field_calculator import FieldCalculator

def plot_mode(data_file, mode_index=0, show_vector=False):
    if not os.path.exists(data_file):
        print(f"Error: Result file not found: {data_file}")
        return

    # 1. データの読み込み (HDF5)
    try:
        with h5py.File(data_file, 'r') as f:
            nodes = f['mesh/nodes'][:]
            elements = f['mesh/elements'][:]
            r0_nodes = f['mesh/r0_nodes'][:]
            mesh_order = f['mesh'].attrs['order']
            
            frequencies = f['results/frequencies'][:]
            eigenvectors = f['results/eigenvectors'][:]
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return

    if mode_index >= len(frequencies):
        print(f"Error: Mode index {mode_index} is out of range. (Available: 0 to {len(frequencies)-1})")
        return

    freq_ghz = frequencies[mode_index]
    print(f"Visualizing Mode {mode_index}: {freq_ghz:.10f} GHz (Order: {mesh_order})")

    z_coords, r_coords = nodes[:, 0], nodes[:, 1]
    H_theta = eigenvectors[mode_index]
    phi = r_coords * H_theta # ポテンシャル Phi = r * H_theta

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_aspect("equal")

    # 可視化用のメッシュ分割 (2次要素の場合)
    if mesh_order == 1:
        elements_vis = elements
    else:
        elements_vis = np.vstack([
            elements[:, [0, 3, 5]],
            elements[:, [3, 1, 4]],
            elements[:, [5, 4, 2]],
            elements[:, [3, 4, 5]]
        ])

    # A. 背景カラーコンター: H_theta (磁場)
    num_contours = 50
    cf = plt.tricontourf(z_coords, r_coords, elements_vis, H_theta, levels=num_contours, cmap='jet')
    plt.colorbar(cf, label='$H_\\theta$ [A/m]')

    # B. 電気力線: Phi = r*H_theta (ポテンシャル) の等高線
    # 全域に等間隔で 25 本程度の黒い実線を引く
    phi_levels = np.linspace(phi.min(), phi.max(), 25)
    plt.tricontour(z_coords, r_coords, elements_vis, phi, levels=phi_levels, colors='black', linewidths=0.7, alpha=0.6)

    # C. 高精度電場ベクトル (Quiver) の計算
    if show_vector:
        print("Calculating high-precision field map for vectors...")
        calc = FieldCalculator(data_file)
        
        # 領域内に均一なグリッドを生成 (メッシュに依存しない美しい配置)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        r_min, r_max = np.min(r_coords), np.max(r_coords)
        
        nz_v, nr_v = 40, 25 # ベクトルの密度
        zv = np.linspace(z_min, z_max, nz_v)
        rv = np.linspace(r_min, r_max, nr_v)
        ZV, RV = np.meshgrid(zv, rv)
        
        ez_v = np.zeros_like(ZV)
        er_v = np.zeros_like(ZV)
        mask = np.zeros_like(ZV, dtype=bool)

        for i in range(nr_v):
            for j in range(nz_v):
                res = calc.calculate_fields(ZV[i, j], RV[i, j], mode_index)
                if res:
                    ez_v[i, j] = res['Ez']
                    er_v[i, j] = res['Er']
                    mask[i, j] = True
        
        # マスク内のベクトルのみを表示
        ez_plot = ez_v[mask]
        er_plot = er_v[mask]
        zv_plot = ZV[mask]
        rv_plot = RV[mask]
        
        if len(ez_plot) > 0:
            # 向きと強さを可視化 (正規化)
            e_abs = np.sqrt(ez_plot**2 + er_plot**2)
            max_e = np.max(e_abs) if np.max(e_abs) > 0 else 1.0
            
            plt.quiver(zv_plot, rv_plot, ez_plot, er_plot, 
                       color='white', pivot='mid', scale=max_e*15, 
                       width=0.003, headwidth=4, alpha=0.9)

    # D. 装飾
    plt.triplot(z_coords, r_coords, elements_vis, linewidth=0.05, color='white', alpha=0.2)
    plt.plot(z_coords[r0_nodes], r_coords[r0_nodes], 'ro', markersize=1, alpha=0.4, label='r=0 Boundary')

    plt.title(f'Mode {mode_index}: {freq_ghz:.6f} GHz (Order: {mesh_order})\nBackground: $H_\\theta$, Lines: Electric Field Lines ($r H_\\theta$), Vectors: E-Field')
    plt.xlabel('z [m]')
    plt.ylabel('r [m]')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='High-Quality FEM Analysis Visualizer')
    parser.add_argument('-f', '--file', type=str, default='analysis_result.h5',
                        help='Path to the analysis result file (.h5)')
    parser.add_argument('-m', '--mode', type=int, default=0,
                        help='Index of the mode to visualize (default: 0)')
    parser.add_argument('-v', '--vector', action='store_true',
                        help='Show high-precision electric field vectors')

    args = parser.parse_args()
    plot_mode(args.file, args.mode, args.vector)

if __name__ == "__main__":
    main()
