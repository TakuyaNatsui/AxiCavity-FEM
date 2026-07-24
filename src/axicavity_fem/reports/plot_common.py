"""Matplotlib 共通ユーティリティ（カラーマップ、レイアウト）.

ver1 ``plot_common.py`` の STYLE 規約に準拠した、場マップ描画の最小ヘルパ。
CLI から使うため既定で非対話バックエンド (Agg) を選ぶ。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # CLI 既定（GUI 側は wxagg を別途設定）

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402


class STYLE:
    DEFAULT_CMAP = "jet"      # 双極性スカラー場（成分）
    PEAK_CMAP = "hot_r"       # 絶対値・振幅
    MESH_COLOR = "gray"
    MESH_LW = 0.3
    MESH_ALPHA = 0.4
    LABEL_Z = "z [m]"
    LABEL_R = "r [m]"
    TITLE_FONTSIZE = 11


def make_triangulation(nodes, simplices) -> mtri.Triangulation:
    """節点座標 (N,2)[z,r] と要素コーナー (E,3) から三角形分割を作る."""
    s = np.asarray(simplices)[:, :3]
    return mtri.Triangulation(np.asarray(nodes)[:, 0], np.asarray(nodes)[:, 1], s)


def make_triangulation_vis(nodes, simplices, elem_order) -> mtri.Triangulation:
    """塗り/等高線用 triangulation。2 次要素は中点ノードで 4 分割する。

    1 次要素や中点情報が無い場合はコーナー 3 頂点をそのまま使う。
    """
    nodes = np.asarray(nodes)
    s = np.asarray(simplices)
    if elem_order == 2 and s.shape[1] >= 6:
        tris = np.vstack([
            s[:, [0, 3, 5]],
            s[:, [3, 1, 4]],
            s[:, [5, 4, 2]],
            s[:, [3, 4, 5]],
        ])
    else:
        tris = s[:, :3]
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)


def plot_scalar_field(ax, triang, values, title, *, magnitude=False,
                      show_mesh=True):
    """三角形分割上のスカラー場を塗りつぶし等高線で描画する.

    Args:
        ax: 描画先 Axes。
        triang: :class:`matplotlib.tri.Triangulation`。
        values: 節点値 (N,)。複素数なら実部を使う（magnitude のときは絶対値）。
        title: パネルタイトル。
        magnitude: True で絶対値・PEAK_CMAP、False で実部・DEFAULT_CMAP。
        show_mesh: メッシュを薄く重ねる。
    """
    v = np.abs(values) if magnitude else np.real(values)
    if magnitude:
        cmap, vmin, vmax = STYLE.PEAK_CMAP, 0.0, (np.max(v) if v.size else 1.0)
    else:
        amax = np.max(np.abs(v)) if v.size else 1.0
        amax = amax if amax > 0 else 1.0
        cmap, vmin, vmax = STYLE.DEFAULT_CMAP, -amax, amax

    tcf = ax.tricontourf(triang, v, levels=40, cmap=cmap, vmin=vmin, vmax=vmax)
    if show_mesh:
        ax.triplot(triang, color=STYLE.MESH_COLOR, lw=STYLE.MESH_LW,
                   alpha=STYLE.MESH_ALPHA)
    ax.set_aspect("equal")
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.set_title(title, fontsize=STYLE.TITLE_FONTSIZE)
    ax.get_figure().colorbar(tcf, ax=ax, shrink=0.8)
    return tcf


def save_figure(fig, path, dpi=120):
    """図を保存して閉じる."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_gif(draw_frame, output, *, n_frames=36, fps=12,
               figsize=(8, 6), dpi=120) -> str:
    """時間位相 0→360° を周回する GIF を生成・保存する.

    各フレームは固定サイズのオフスクリーン Agg figure に描画するため、
    全フレームが同一寸法になり GIF のがたつきを防ぐ。

    Args:
        draw_frame: ``draw_frame(fig, time_phase_deg)`` で 1 フレームを描く関数。
            figure は呼び出し前に毎回クリアされる。
        output: 出力 GIF パス。
        n_frames: 1 ループのフレーム数。
        fps: フレームレート。
        figsize: figure サイズ [inch]。
        dpi: 解像度。

    Returns:
        出力 GIF パス。
    """
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    frames = []
    for theta in np.linspace(0.0, 360.0, n_frames, endpoint=False):
        fig.clf()
        draw_frame(fig, float(theta))
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
    if frames:
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       loop=0, duration=int(1000 / fps), optimize=False)
    return output


def new_panel_grid(n_panels, suptitle=None):
    """n_panels 用の 2 列グリッド figure を返す."""
    ncol = 2
    nrow = (n_panels + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    if suptitle:
        fig.suptitle(suptitle, fontsize=STYLE.TITLE_FONTSIZE + 2)
    return fig, axes
