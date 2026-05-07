"""
plot_common.py

TM0 / HOM 双方の可視化コードで共有するスタイル定数と描画ユーティリティ。

使い方:
    from plot_common import STYLE, setup_axes, draw_mesh_overlay, draw_pec_boundary
    from plot_common import BaseFEMPlotter, TM0Plotter

設計方針:
  * スタイル定数は ``STYLE`` にまとめ、TM0 / HOM でカラー・線幅・grid 設定を
    揃える。カラーマップは TM0 準拠で 'jet' を既定とする。
  * ``BaseFEMPlotter`` は FEM_code/plot_utils.py の旧 FEMPlotter を移設した
    ものをベースクラスとする。TM0 固有の描画は ``TM0Plotter`` に集約。
  * HOM 側のデータ構造（edge DOF + node DOF）は TM0 と大きく異なるため、
    HOM 用の plotter クラスは将来実装の余地を残し、本モジュールではまず
    スタイル・共通ヘルパを提供して HOM 既存スクリプトから参照させる。
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# スタイル定数 (TM0 準拠に統一)
# ---------------------------------------------------------------------------
class STYLE:
    # カラーマップ
    DEFAULT_CMAP = 'jet'          # 双極性スカラー場 (H_theta, E_theta 実部など)
    PEAK_CMAP = 'hot_r'           # 絶対値・振幅表示

    # 色
    PSI_COLOR = 'black'           # 電気力線 (Psi 等高線)
    QUIVER_COLOR = 'darkmagenta'  # 実部ベクトル (TM0 E, HOM (Ez,Er))
    QUIVER_IMAG_COLOR = 'crimson' # 進行波の虚部ベクトル
    AXIAL_REAL_COLOR = 'darkblue'
    AXIAL_IMAG_COLOR = 'crimson'

    # メッシュ
    MESH_COLOR = 'gray'
    MESH_LW = 0.3
    MESH_ALPHA = 0.5

    # 境界線 (PEC 等)
    BORDER_COLOR = 'black'
    BORDER_LW = 1.0
    BORDER_LW_STRONG = 1.2

    # Psi 等高線
    PSI_LW = 0.5
    PSI_ALPHA = 0.6

    # Quiver
    QUIVER_ALPHA = 0.8
    QUIVER_PIVOT = 'middle'

    # グリッド
    GRID_LS = ':'
    GRID_ALPHA = 0.5

    # フォント
    TITLE_FONTSIZE = 11
    LEGEND_FONTSIZE = 8

    # 軸ラベル
    LABEL_Z = 'z [m]'
    LABEL_R = 'r [m]'


# ---------------------------------------------------------------------------
# 共通ヘルパ関数
# ---------------------------------------------------------------------------
def setup_axes(ax=None, figsize=(10, 8)):
    """TM0/HOM 共通の Axes 設定を行う。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal')
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.grid(True, linestyle=STYLE.GRID_LS, alpha=STYLE.GRID_ALPHA)
    return fig, ax


def draw_mesh_overlay(ax, triang_or_vertices, simplices=None):
    """メッシュ線を共通スタイルで重ねる。

    ``triang_or_vertices`` に ``matplotlib.tri.Triangulation`` を渡すか、
    ``(vertices, simplices)`` の形で座標と要素リストを渡す。
    """
    if simplices is None:
        ax.triplot(triang_or_vertices,
                   color=STYLE.MESH_COLOR,
                   linewidth=STYLE.MESH_LW,
                   alpha=STYLE.MESH_ALPHA)
    else:
        vertices = triang_or_vertices
        ax.triplot(vertices[:, 0], vertices[:, 1], simplices,
                   color=STYLE.MESH_COLOR,
                   linewidth=STYLE.MESH_LW,
                   alpha=STYLE.MESH_ALPHA)


def draw_pec_boundary(ax, segments, strong=False, label=None):
    """PEC 境界を共通スタイル (黒線) で描画する。

    Args:
        ax: matplotlib Axes
        segments: [[(z1,r1),(z2,r2)], ...] の線分リスト、または
                  NumPy 配列 (N, 2, 2)
        strong: True なら少し太めの線幅を使う (アニメーション等)
        label: 凡例ラベル (指定時のみ)
    """
    if segments is None or len(segments) == 0:
        return None
    lw = STYLE.BORDER_LW_STRONG if strong else STYLE.BORDER_LW
    kw = dict(colors=STYLE.BORDER_COLOR, linewidths=lw, zorder=10)
    if label:
        kw['label'] = label
    lc = LineCollection(segments, **kw)
    ax.add_collection(lc)
    return lc


def plot_bipolar_contour(ax, triang, scalar, cmap=None, levels=51, vmax=None,
                         label=None, fig=None):
    """対称レンジ (vmin=-vmax, vmax=+vmax) でスカラー場を tricontourf する。

    ``vmax`` が None の場合は ``|scalar|.max()`` を使用 (0 の場合は 1.0)。
    TM0 の H_theta・HOM の E_theta 実部などの双極性スカラー場向け。
    """
    cmap = cmap or STYLE.DEFAULT_CMAP
    if vmax is None:
        vmax = float(np.max(np.abs(scalar)))
        if vmax < 1e-30:
            vmax = 1.0
    lv = np.linspace(-vmax, vmax, levels)
    cf = ax.tricontourf(triang, scalar, levels=lv, cmap=cmap,
                        vmin=-vmax, vmax=vmax)
    if fig is not None and label is not None:
        fig.colorbar(cf, ax=ax, label=label)
    return cf


def plot_quiver_styled(ax, x, y, u, v, imag=False, **kwargs):
    """共通 quiver スタイル。``imag=True`` で進行波虚部用の色に切替。"""
    color = STYLE.QUIVER_IMAG_COLOR if imag else STYLE.QUIVER_COLOR
    params = dict(color=color,
                  pivot=STYLE.QUIVER_PIVOT,
                  alpha=STYLE.QUIVER_ALPHA,
                  scale_units='xy')
    params.update(kwargs)
    return ax.quiver(x, y, u, v, **params)


# ---------------------------------------------------------------------------
# BaseFEMPlotter (旧 FEM_code/plot_utils.py の FEMPlotter を移設)
# ---------------------------------------------------------------------------
class BaseFEMPlotter:
    """TM0 / HOM 共通のプロッタ基底クラス。

    ``FieldCalculator`` インタフェース (TM0 由来) を前提に、メッシュ
    triangulation と境界セグメントを準備する。mesh_order=2 の場合は
    コンター塗りつぶし用に 2 次要素を 4 つのサブ三角形へ分割する。
    """

    def __init__(self, field_calc):
        self.calc = field_calc
        self.nodes = field_calc.nodes
        self.elements = field_calc.elements
        self.mesh_order = field_calc.mesh_order

        # 線描画用 (三角形の 3 頂点のみ)
        self.elements_tri = self.elements[:, :3]

        # コンター用 (2 次要素は 4 分割)
        if self.mesh_order == 1:
            self.elements_vis = self.elements
        else:
            self.elements_vis = np.vstack([
                self.elements[:, [0, 3, 5]],
                self.elements[:, [3, 1, 4]],
                self.elements[:, [5, 4, 2]],
                self.elements[:, [3, 4, 5]],
            ])

        self.z = self.nodes[:, 0]
        self.r = self.nodes[:, 1]
        self.triang_vis = tri.Triangulation(self.z, self.r, self.elements_vis)
        self.triang_raw = tri.Triangulation(self.z, self.r, self.elements_tri)

        edges = self.calc.get_boundary_edges()
        self.boundary_segments = [self.nodes[list(e)] for e in edges]

    # 共通スタイルのショートカット
    def setup_axes(self, ax=None, figsize=(10, 8)):
        return setup_axes(ax=ax, figsize=figsize)

    def _draw_border(self, ax, strong=False):
        return draw_pec_boundary(ax, self.boundary_segments, strong=strong)

    def _draw_mesh(self, ax):
        draw_mesh_overlay(ax, self.triang_raw)


# ---------------------------------------------------------------------------
# TM0Plotter — TM0 固有描画 (H_theta 背景 + Psi + E quiver)
# ---------------------------------------------------------------------------
class TM0Plotter(BaseFEMPlotter):
    """TM0 モード用プロッタ。描画スタイルは ``STYLE`` に従う。"""

    def plot_mode_snapshot(self, mode_idx, theta=0, ax=None,
                           show_h_theta=True, show_psi=True, show_vectors=False,
                           show_mesh=False, E_lines=51, v_steps=(40, 40),
                           title=None, cmap=None, peak_mode=False):
        """瞬時位相 ``theta`` におけるモード分布を描画する。

        ``peak_mode=True`` の場合は位相を無視し振幅ピークを表示する。
        """
        cmap = cmap or STYLE.DEFAULT_CMAP
        fig, ax = self.setup_axes(ax)

        if peak_mode:
            fields = self.calc.calculate_peak_fields(mode_idx)
            theta_str = "Peak"
        else:
            theta_rad = np.deg2rad(theta) if isinstance(theta, (int, float)) else theta
            fields = self.calc.calculate_all_node_fields(mode_idx, theta=theta_rad)
            theta_str = f"{theta:.1f}°"

        h_complex = self.calc.eigenvectors[mode_idx]
        h_max = np.max(np.abs(h_complex))
        if h_max == 0:
            h_max = 1.0

        # 1. H_theta コンター
        if show_h_theta:
            h_data = fields['H_theta']
            if peak_mode:
                levels = np.linspace(0, h_max, 51)
                cf = ax.tricontourf(self.triang_vis, h_data, levels=levels,
                                    cmap=STYLE.PEAK_CMAP, vmin=0, vmax=h_max)
                fig.colorbar(cf, ax=ax, label=r'$|H_\theta|$ [A/m]')
            else:
                plot_bipolar_contour(ax, self.triang_vis, h_data, cmap=cmap,
                                     vmax=h_max,
                                     label=r'$H_\theta$ [A/m]', fig=fig)

        # 2. Psi 電気力線
        if show_psi:
            psi_data = fields['Psi']
            psi_max = np.max(np.abs(self.r * h_complex))
            if psi_max > 1e-20:
                psi_levels = np.linspace(0 if peak_mode else -psi_max, psi_max, E_lines)
                ax.tricontour(self.triang_vis, psi_data, levels=psi_levels,
                              colors=STYLE.PSI_COLOR,
                              linewidths=STYLE.PSI_LW,
                              alpha=STYLE.PSI_ALPHA,
                              linestyles='solid')

        # 3. 電界ベクトル
        if show_vectors:
            z_steps, r_steps = v_steps
            gf = self.calc.calculate_grid_fields(
                mode_idx, z_steps=z_steps, r_steps=r_steps,
                theta=np.deg2rad(theta) if not peak_mode else np.pi / 2)
            mask = gf['mask']
            plot_quiver_styled(ax, gf['zz'][mask], gf['rr'][mask],
                               gf['Ez'][mask], gf['Er'][mask])

        # 4. メッシュ
        if show_mesh:
            self._draw_mesh(ax)

        # 5. 境界
        self._draw_border(ax)

        if title is None:
            freq = self.calc.frequencies[mode_idx]
            title = f"Mode {mode_idx}: {freq:.6f} GHz, Phase: {theta_str}"
        ax.set_title(title)
        return fig, ax

    def plot_mode_standing(self, mode_idx, ax=None,
                           show_h_theta=True, show_psi=True, show_vectors=True,
                           show_mesh=False, E_lines=51, v_steps=(40, 40),
                           title=None, cmap=None):
        """定在波描画。"""
        cmap = cmap or STYLE.DEFAULT_CMAP
        fig, ax = self.setup_axes(ax)
        fields = self.calc.calculate_all_node_fields(mode_idx, theta=0.0)

        if show_h_theta:
            h_data = fields['H_theta']
            cf = ax.tricontourf(self.triang_vis, h_data, levels=51, cmap=cmap)
            fig.colorbar(cf, ax=ax, label=r'$H_\theta$ [A/m]')

        if show_psi:
            psi_data = fields['Psi']
            ax.tricontour(self.triang_vis, psi_data, levels=E_lines,
                          colors=STYLE.PSI_COLOR,
                          linewidths=STYLE.PSI_LW,
                          alpha=STYLE.PSI_ALPHA,
                          linestyles='solid')

        if show_vectors:
            z_steps, r_steps = v_steps
            gf = self.calc.calculate_grid_fields(
                mode_idx, z_steps=z_steps, r_steps=r_steps)
            mask = gf['mask']
            plot_quiver_styled(ax, gf['zz'][mask], gf['rr'][mask],
                               gf['Ez'][mask], gf['Er'][mask])

        if show_mesh:
            self._draw_mesh(ax)

        self._draw_border(ax)

        if title is None:
            freq = self.calc.frequencies[mode_idx]
            title = f"Mode {mode_idx}: {freq:.6f} GHz"
        ax.set_title(title)
        return fig, ax

    def plot_standing(self, mode_idx, output_dir=None, prefix=""):
        """定在波レポート用画像を生成・保存。"""
        fig, ax = self.setup_axes()
        self.plot_mode_standing(mode_idx, ax, show_mesh=True)
        name = 'standing'
        saved_files = {}
        if output_dir:
            fname = f"{prefix}mode_{mode_idx:02d}_{name}.png"
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            saved_files[name] = fname
            plt.close(fig)
        return saved_files

    def plot_axial_field(self, mode_idx, output_dir=None, prefix=""):
        """軸上電場 Ez を 1D でプロット。進行波は実部/虚部併記。"""
        z_min, z_max = np.min(self.z), np.max(self.z)
        z_scan = np.linspace(z_min, z_max, 500)

        analysis_type = getattr(self.calc, 'analysis_type', 'standing')

        fig, ax = plt.subplots(figsize=(10, 4))

        if analysis_type == 'standing':
            ez_axial = []
            for z_pt in z_scan:
                res = self.calc.calculate_fields(z_pt, 0.0, mode_idx, theta=0.0)
                ez_axial.append(res['Ez'] if res else 0.0)
            ax.plot(z_scan, ez_axial,
                    color=STYLE.AXIAL_REAL_COLOR, linewidth=2, label='Ez')
        else:
            ez_real = []
            ez_imag = []
            for z_pt in z_scan:
                res = self.calc.calculate_fields(z_pt, 0.0, mode_idx,
                                                 theta=0.0, return_complex=True)
                if res:
                    ez_real.append(np.real(res['Ez']))
                    ez_imag.append(np.imag(res['Ez']))
                else:
                    ez_real.append(0.0)
                    ez_imag.append(0.0)
            ax.plot(z_scan, ez_real,
                    color=STYLE.AXIAL_REAL_COLOR, linewidth=2,
                    label='Ez (Real, 0°)')
            ax.plot(z_scan, ez_imag,
                    color=STYLE.AXIAL_IMAG_COLOR, linewidth=1.5,
                    linestyle='--', label='Ez (Imag, 90°)')
            ax.legend(fontsize=STYLE.LEGEND_FONTSIZE)

        ax.set_title(f"Axial Electric Field (Mode {mode_idx})")
        ax.set_xlabel(STYLE.LABEL_Z)
        ax.set_ylabel("Ez [V/m]")
        ax.grid(True, linestyle=STYLE.GRID_LS, alpha=0.6)

        fname = ""
        if output_dir:
            fname = f"{prefix}mode_{mode_idx:02d}_axial.png"
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)
        return fname

    def plot_complex_components(self, mode_idx, output_dir=None, prefix=""):
        components = [
            ('real', 0, r'Real Part of $H_\theta$', STYLE.DEFAULT_CMAP),
            ('imag', -90, r'Imaginary Part of $H_\theta$', STYLE.DEFAULT_CMAP),
            ('abs', None, r'Magnitude of $H_\theta$', STYLE.PEAK_CMAP),
        ]
        saved_files = {}
        for name, theta, label, cmap in components:
            fig, ax = self.setup_axes()
            if name == 'abs':
                h_complex = self.calc.eigenvectors[mode_idx]
                h_data = np.abs(h_complex)
                cf = ax.tricontourf(self.triang_vis, h_data, levels=40, cmap=cmap)
                fig.colorbar(cf, ax=ax, label=r'$|H_\theta|$ [A/m]')
                ax.set_title(f"Mode {mode_idx}: Magnitude")
            else:
                self.plot_mode_snapshot(mode_idx, theta=theta, ax=ax,
                                        show_vectors=True, show_mesh=True,
                                        title=f"Mode {mode_idx}: {label}",
                                        cmap=cmap)
            if output_dir:
                fname = f"{prefix}mode_{mode_idx:02d}_{name}.png"
                fig.savefig(os.path.join(output_dir, fname), dpi=150)
                saved_files[name] = fname
                plt.close(fig)
        return saved_files

    def create_animation(self, mode_idx, output_path, fps=10, n_frames=36):
        fig, ax = self.setup_axes()
        h_complex = self.calc.eigenvectors[mode_idx]
        h_max = np.max(np.abs(h_complex))
        psi_max = np.max(np.abs(self.r * h_complex))

        def update(frame):
            ax.clear()
            ax.set_aspect('equal')
            theta_deg = (frame / n_frames) * 360
            fields = self.calc.calculate_all_node_fields(
                mode_idx, theta=np.deg2rad(theta_deg))
            h_t, psi_t = fields['H_theta'], fields['Psi']
            levels = np.linspace(-h_max, h_max, 41)
            ax.tricontourf(self.triang_vis, h_t, levels=levels,
                           cmap=STYLE.DEFAULT_CMAP, vmin=-h_max, vmax=h_max)
            if psi_max > 1e-20:
                psi_levels = np.linspace(-psi_max, psi_max, 21)
                ax.tricontour(self.triang_vis, psi_t, levels=psi_levels,
                              colors=STYLE.PSI_COLOR,
                              linewidths=STYLE.PSI_LW,
                              alpha=STYLE.PSI_ALPHA)
            draw_pec_boundary(ax, self.boundary_segments, strong=True)
            ax.set_title(f"Mode {mode_idx} Phase Evolution: {theta_deg:.1f}°")
            ax.set_xlabel(STYLE.LABEL_Z)
            ax.set_ylabel(STYLE.LABEL_R)

        ani = FuncAnimation(fig, update, frames=n_frames)
        ani.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return os.path.basename(output_path)


# ---------------------------------------------------------------------------
# HOMPlotter — スタイル統一のためのヘルパを集約
# ---------------------------------------------------------------------------
class HOMPlotter:
    """HOM 用プロッタ。HOM のデータ構造は TM0 と異なるため、既存スクリプト
    から呼ばれる共通描画ヘルパ群を集約する薄いクラス。

    ``draw_mesh_and_pec(ax, vertices, simplices, edge_index_map, pec_indices)``
    のように関数形式でも使えるよう、クラスメソッドとしても同等の処理を提供
    する。
    """

    @staticmethod
    def draw_mesh_and_pec_segments(ax, vertices, simplices, pec_segments,
                                   mesh=True, border=True, xlabel=None, ylabel=None):
        """HOM 用の共通下地描画: メッシュ + PEC 境界 + 軸ラベル統一。

        ``pec_segments`` は [[(z1,r1),(z2,r2)], ...] の線分リスト。
        """
        ax.set_aspect('equal', adjustable='box')
        if mesh:
            draw_mesh_overlay(ax, vertices, simplices)
        if border and pec_segments:
            draw_pec_boundary(ax, pec_segments, label='PEC')
        ax.set_xlabel(xlabel or STYLE.LABEL_Z)
        ax.set_ylabel(ylabel or STYLE.LABEL_R)
        ax.grid(True, linestyle=STYLE.GRID_LS, alpha=STYLE.GRID_ALPHA)

    @staticmethod
    def pec_segments_from_edge_map(edge_index_map, pec_indices, vertices):
        """edge_index_map (dict: (n1,n2)->idx) と PEC グローバル辺インデックス
        リストから、LineCollection 用の線分配列を生成する。
        """
        pec_set = set(int(i) for i in pec_indices)
        inv = {int(v): k for k, v in edge_index_map.items()}
        segs = []
        for idx in pec_set:
            key = inv.get(idx)
            if key is None:
                continue
            n1, n2 = int(key[0]), int(key[1])
            segs.append([(vertices[n1, 0], vertices[n1, 1]),
                         (vertices[n2, 0], vertices[n2, 1])])
        return segs


__all__ = [
    'STYLE',
    'setup_axes',
    'draw_mesh_overlay',
    'draw_pec_boundary',
    'plot_bipolar_contour',
    'plot_quiver_styled',
    'BaseFEMPlotter',
    'TM0Plotter',
    'HOMPlotter',
]
