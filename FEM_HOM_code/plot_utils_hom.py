import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation, PillowWriter

# プロジェクトルートを sys.path に追加して plot_common を解決
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from plot_common import STYLE  # noqa: E402

class FEMPlotterHOM:
    def __init__(self, calc):
        self.calc = calc

    def plot_mode(self, mode_idx, ax_left, ax_right, theta=0.0, 
                  show_colormaps=True, show_vectors=True, 
                  show_mesh=False, E_lines=20, v_steps=(20, 20)):
        """
        左のAxes(ax_left)に電場E、右のAxes(ax_right)に磁場Hを描画する。
        - theta: 時間位相
        """
        # Ensure data is loaded
        if self.calc.modes[mode_idx][0] != mode_idx:
            # simple check, but actual mode index might be offset.
            # calc.load_mode_data uses the real mode number from self.modes
            pass 
            
        real_mode_idx = self.calc.modes[mode_idx][0]
        self.calc.load_mode_data(real_mode_idx)
        
        # 三角形メッシュの生成
        vertices = self.calc.vertices
        simplices = self.calc.simplices[:, :3]
        tri = Triangulation(vertices[:, 0], vertices[:, 1], simplices)
        
        # コンター用の節点ごとの値を取得
        nodes_res = self.calc.calculate_all_node_fields(real_mode_idx, theta_time=theta)
        
        # E_theta と H_theta をカラーマップとして描画
        if show_colormaps:
            E_t = nodes_res['E_theta']
            H_t = nodes_res['H_theta']
            
            vmax_E = np.max(np.abs(E_t))
            if vmax_E < 1e-30: vmax_E = 1.0
            levels_E = np.linspace(-vmax_E, vmax_E, max(3, E_lines))
            cf_E = ax_left.tricontourf(tri, E_t, levels=levels_E, cmap=STYLE.DEFAULT_CMAP)
            # 簡便のためカラーバーは省略するか、呼び出し側で追加する
            # plt.colorbar(cf_E, ax=ax_left, fraction=0.046, pad=0.04, label=r'$E_\theta$')

            vmax_H = np.max(np.abs(H_t))
            if vmax_H < 1e-30: vmax_H = 1.0
            levels_H = np.linspace(-vmax_H, vmax_H, max(3, E_lines))
            cf_H = ax_right.tricontourf(tri, H_t, levels=levels_H, cmap=STYLE.DEFAULT_CMAP)

        # メッシュの描画
        if show_mesh:
            ax_left.triplot(tri, color=STYLE.MESH_COLOR,
                            linewidth=STYLE.MESH_LW, alpha=STYLE.MESH_ALPHA)
            ax_right.triplot(tri, color=STYLE.MESH_COLOR,
                             linewidth=STYLE.MESH_LW, alpha=STYLE.MESH_ALPHA)

        # ベクトルの描画
        if show_vectors:
            z_min, z_max = vertices[:, 0].min(), vertices[:, 0].max()
            r_min, r_max = vertices[:, 1].min(), vertices[:, 1].max()
            num_z, num_r = v_steps
            z_grid = np.linspace(z_min, z_max, max(2, num_z))
            r_grid = np.linspace(min(r_min, 1e-6), r_max, max(2, num_r))
            
            Z_grid, R_grid = np.meshgrid(z_grid, r_grid)
            Ez_grid = np.zeros_like(Z_grid)
            Er_grid = np.zeros_like(R_grid)
            Hz_grid = np.zeros_like(Z_grid)
            Hr_grid = np.zeros_like(R_grid)
            valid_mask = np.zeros(Z_grid.shape, dtype=bool)

            for i in range(num_r):
                for j in range(num_z):
                    z = Z_grid[i, j]
                    r = R_grid[i, j]
                    res = self.calc.calculate_fields(z, r, real_mode_idx, theta_time=theta)
                    if res is not None:
                        Ez_grid[i, j] = res['Ez']
                        Er_grid[i, j] = res['Er']
                        Hz_grid[i, j] = res['Hz']
                        Hr_grid[i, j] = res['Hr']
                        valid_mask[i, j] = True

            # normalize for quiver (メッシュ内の点のみ使用)
            mag_E = np.sqrt(Ez_grid**2 + Er_grid**2)
            max_mag_E = np.max(mag_E[valid_mask]) if valid_mask.any() else 1.0
            if max_mag_E < 1e-30: max_mag_E = 1.0

            mag_H = np.sqrt(Hz_grid**2 + Hr_grid**2)
            max_mag_H = np.max(mag_H[valid_mask]) if valid_mask.any() else 1.0
            if max_mag_H < 1e-30: max_mag_H = 1.0

            ax_left.quiver(Z_grid[valid_mask], R_grid[valid_mask],
                           Ez_grid[valid_mask] / max_mag_E,
                           Er_grid[valid_mask] / max_mag_E,
                           color=STYLE.QUIVER_COLOR, pivot='mid', scale=40,
                           alpha=STYLE.QUIVER_ALPHA)

            ax_right.quiver(Z_grid[valid_mask], R_grid[valid_mask],
                            Hz_grid[valid_mask] / max_mag_H,
                            Hr_grid[valid_mask] / max_mag_H,
                            color=STYLE.QUIVER_COLOR, pivot='mid', scale=40,
                            alpha=STYLE.QUIVER_ALPHA)

        # Axesの調整
        ax_left.set_aspect('equal')
        ax_left.set_xlabel('z [m]')
        ax_left.set_ylabel('r [m]')
        ax_left.set_title(r"E-field (colormap: $E_\theta$, vectors: $E_z, E_r$)")
        
        ax_right.set_aspect('equal')
        ax_right.set_xlabel('z [m]')
        ax_right.set_ylabel('r [m]')
        ax_right.set_title(r"H-field (colormap: $H_\theta$, vectors: $H_z, H_r$)")

    # ------------------------------------------------------------------
    # レポート生成用ヘルパ
    # ------------------------------------------------------------------
    def plot_mode_to_file(self, mode_idx, output_dir, filename, theta=0.0,
                          show_mesh=True, show_vectors=True, E_lines=20,
                          v_steps=(20, 20), title_prefix=""):
        """E-field と H-field を左右に並べて 1 枚の PNG として保存する。"""
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        self.plot_mode(mode_idx, ax_left=ax_left, ax_right=ax_right,
                       theta=theta, show_colormaps=True,
                       show_vectors=show_vectors, show_mesh=show_mesh,
                       E_lines=E_lines, v_steps=v_steps)
        if title_prefix:
            fig.suptitle(title_prefix)
        fig.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return filename

    def plot_axial_field(self, mode_idx, output_dir, filename,
                         analysis_type='standing'):
        """軸上 (r=0) 付近の Ez を 1D プロットして保存する。"""
        real_mode_idx = self.calc.modes[mode_idx][0]
        self.calc.load_mode_data(real_mode_idx)

        z_min = float(np.min(self.calc.vertices[:, 0]))
        z_max = float(np.max(self.calc.vertices[:, 0]))
        z_scan = np.linspace(z_min, z_max, 500)
        # n>=1 では r=0 上で E_z が定義されないため、わずかに離れた r で評価
        r_eval = max(1e-10, (np.max(self.calc.vertices[:, 1]) -
                             np.min(self.calc.vertices[:, 1])) * 1e-6)

        fig, ax = plt.subplots(figsize=(10, 4))
        if analysis_type == 'standing':
            ez = []
            for z in z_scan:
                res = self.calc.calculate_fields(z, r_eval, real_mode_idx,
                                                 theta_time=0.0)
                ez.append(res['Ez'] if res else 0.0)
            ax.plot(z_scan, ez, color=STYLE.AXIAL_REAL_COLOR, lw=2,
                    label='Ez')
        else:
            ez0, ez90 = [], []
            for z in z_scan:
                res0 = self.calc.calculate_fields(z, r_eval, real_mode_idx,
                                                  theta_time=0.0)
                res90 = self.calc.calculate_fields(z, r_eval, real_mode_idx,
                                                   theta_time=90.0)
                ez0.append(res0['Ez'] if res0 else 0.0)
                ez90.append(res90['Ez'] if res90 else 0.0)
            ax.plot(z_scan, ez0, color=STYLE.AXIAL_REAL_COLOR, lw=2,
                    label='Ez (θ=0°)')
            ax.plot(z_scan, ez90, color=STYLE.AXIAL_IMAG_COLOR, lw=1.5,
                    linestyle='--', label='Ez (θ=90°)')
            ax.legend(fontsize=STYLE.LEGEND_FONTSIZE)

        ax.set_title(f"Axial Electric Field (Mode {mode_idx})")
        ax.set_xlabel(STYLE.LABEL_Z)
        ax.set_ylabel("Ez [V/m]")
        ax.grid(True, linestyle=STYLE.GRID_LS, alpha=0.6)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return filename

    def create_animation(self, mode_idx, output_path, fps=10, n_frames=36,
                         show_vectors=True, show_mesh=False, E_lines=20,
                         v_steps=(20, 20), title_prefix=""):
        """進行波モード用の GIF アニメーションを保存する。"""
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

        def update(frame):
            ax_left.clear()
            ax_right.clear()
            theta_deg = (frame / n_frames) * 360
            self.plot_mode(mode_idx, ax_left=ax_left, ax_right=ax_right,
                           theta=theta_deg, show_colormaps=True,
                           show_vectors=show_vectors, show_mesh=show_mesh,
                           E_lines=E_lines, v_steps=v_steps)
            if title_prefix:
                fig.suptitle(f"{title_prefix}  θ={theta_deg:.1f}°")
            else:
                fig.suptitle(f"θ={theta_deg:.1f}°")

        ani = FuncAnimation(fig, update, frames=n_frames)
        ani.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return os.path.basename(output_path)
