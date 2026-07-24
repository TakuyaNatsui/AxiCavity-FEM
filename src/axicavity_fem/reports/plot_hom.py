"""HOM 場マップ描画（Result Viewer と同じ E / H の 2 パネル: カラー＋ベクトル）.

左パネル = E 場（カラー: E_theta、ベクトル: (E_z, E_r)）、
右パネル = H 場（カラー: H_theta、ベクトル: (H_z, H_r)）。
PEC 境界は黒太線で常時描画。進行波は GIF アニメーションも生成できる
（:func:`animate_hom_mode`）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..fem_hom.field_recon import HOMFieldReconstructor
from ..shared.boundary_groups import (material_interface_geometry,
                                      pec_boundary_geometry)
from .plot_common import (
    STYLE,
    make_triangulation,
    make_triangulation_vis,
    render_gif,
    save_figure,
)

_QUIVER_COLOR = "darkmagenta"


def _select(data, n, phase):
    results = data["results_by_n"]
    if n is None:
        n = sorted(results)[0]
    rec = results[n]
    if "standing" in rec:
        return rec["standing"], "standing", n, None
    trav = rec["traveling"]
    if phase is None:
        phase = sorted(trav)[0]
    return trav[phase], "traveling", n, phase


def _hom_setup(data, mode, n, phase):
    """時間位相に依存しないセットアップ（recon・三角形分割・PEC 幾何）を作る."""
    modeset, analysis_type, n, phase = _select(data, n, phase)
    if mode >= len(modeset["frequencies"]):
        raise IndexError(f"mode {mode} は範囲外（モード数 {len(modeset['frequencies'])}）")
    mesh = data["mesh"]
    recon = HOMFieldReconstructor(mesh["simplices"], mesh["vertices"],
                                  mesh["edge_index_map"], mesh["elem_order"],
                                  n, analysis_type)
    recon.load_mode(modeset["eigenvectors"][mode],
                    float(modeset["frequencies"][mode]))
    segments, _ = pec_boundary_geometry(mesh["simplices"], mesh["vertices"],
                                        mesh.get("physical_groups"),
                                        mesh["elem_order"])
    mat_segments, _ = material_interface_geometry(
        mesh["simplices"], mesh["vertices"], mesh["elem_order"],
        eps_r_per_element=mesh.get("eps_r_per_element"),
        tan_delta_per_element=mesh.get("tan_delta_per_element"))
    return {
        "recon": recon,
        "freq": float(modeset["frequencies"][mode]),
        "mesh": mesh,
        "triang_vis": make_triangulation_vis(mesh["vertices"],
                                             mesh["simplices"],
                                             mesh["elem_order"]),
        "triang_raw": make_triangulation(mesh["vertices"], mesh["simplices"]),
        "segments": segments,
        "mat_segments": mat_segments,
        "analysis_type": analysis_type,
        "n": n,
        "phase": phase,
        "mode": mode,
    }


def _suptitle(ctx):
    s = (f"HOM n={ctx['n']} mode {ctx['mode']}  f = {ctx['freq']:.6f} GHz  "
         f"({ctx['analysis_type']})")
    if ctx["analysis_type"] == "traveling":
        s += f"  phase={ctx['phase']}deg"
    return s


def _hom_panel(ax, triang_fill, triang_mesh, values, title, show_mesh, segments,
               mat_segments=None):
    """1 パネル: 双極カラーマップ（vis 分割）＋ メッシュ ＋ PEC 黒太線＋誘電体界面."""
    v = np.real(values)
    amax = float(np.max(np.abs(v))) if v.size else 1.0
    amax = amax if amax > 0 else 1.0
    tcf = ax.tricontourf(triang_fill, v, levels=40, cmap=STYLE.DEFAULT_CMAP,
                         vmin=-amax, vmax=amax)
    if show_mesh:
        ax.triplot(triang_mesh, color=STYLE.MESH_COLOR, lw=STYLE.MESH_LW,
                   alpha=STYLE.MESH_ALPHA)
    if segments:
        ax.add_collection(LineCollection(segments, colors="black",
                                         linewidths=1.2, zorder=10))
    # ver2.3: 誘電体界面（ε_r / tanδ が変わる内部辺）をマゼンタ破線で
    if mat_segments:
        ax.add_collection(LineCollection(mat_segments, colors="magenta",
                                         linewidths=1.5, linestyles="--",
                                         zorder=9))
    ax.set_aspect("equal")
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.set_title(title, fontsize=STYLE.TITLE_FONTSIZE)
    ax.get_figure().colorbar(tcf, ax=ax, shrink=0.8)


def _hom_vectors(ax_e, ax_h, recon, vertices, time_phase, vec_steps):
    """E 場パネルに (E_z, E_r)、H 場パネルに (H_z, H_r) をグリッド quiver で描く."""
    verts = np.asarray(vertices)
    nz = max(2, vec_steps[0])
    nr = max(2, vec_steps[1])
    zg = np.linspace(verts[:, 0].min(), verts[:, 0].max(), nz)
    rg = np.linspace(verts[:, 1].min(), verts[:, 1].max(), nr)
    Z, R, Ez, Er, Hz, Hr = [], [], [], [], [], []
    for z in zg:
        for r in rg:
            res = recon.calculate_fields(z, max(r, 1e-10), theta_time=time_phase)
            if res is None:
                continue
            Z.append(z); R.append(r)
            Ez.append(res["Ez"]); Er.append(res["Er"])
            Hz.append(res["Hz"]); Hr.append(res["Hr"])
    if Z:
        ax_e.quiver(Z, R, Ez, Er, color=_QUIVER_COLOR, alpha=0.8, pivot="middle")
        ax_h.quiver(Z, R, Hz, Hr, color=_QUIVER_COLOR, alpha=0.8, pivot="middle")


def _draw_hom_panels(axl, axr, ctx, time_phase, *, show_mesh, show_vectors,
                     vec_steps):
    """E / H の 2 パネル（カラー＋ベクトル＋PEC線）を描く."""
    recon = ctx["recon"]
    node = recon.calculate_all_node_fields(theta_time=time_phase)
    _hom_panel(axl, ctx["triang_vis"], ctx["triang_raw"], node["E_theta"],
               r"E-field (color $E_\theta$, vec $E_z,E_r$)", show_mesh,
               ctx["segments"], ctx["mat_segments"])
    _hom_panel(axr, ctx["triang_vis"], ctx["triang_raw"], node["H_theta"],
               r"H-field (color $H_\theta$, vec $H_z,H_r$)", show_mesh,
               ctx["segments"], ctx["mat_segments"])
    if show_vectors:
        _hom_vectors(axl, axr, recon, ctx["mesh"]["vertices"], time_phase,
                     vec_steps)


def plot_hom_mode(data, mode, output, *, n=None, phase=None, time_phase=0.0,
                  show_mesh=True, show_vectors=True, vec_steps=(20, 20),
                  dpi=120) -> str:
    """HOM の 1 モードを E / H の 2 パネル（カラー＋ベクトル）で PNG 出力する."""
    ctx = _hom_setup(data, mode, n, phase)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(_suptitle(ctx), fontsize=STYLE.TITLE_FONTSIZE + 2)
    _draw_hom_panels(axl, axr, ctx, time_phase, show_mesh=show_mesh,
                     show_vectors=show_vectors, vec_steps=vec_steps)
    save_figure(fig, output, dpi=dpi)
    return output


def animate_hom_mode(data, mode, output, *, n=None, phase=None, n_frames=36,
                     fps=12, show_mesh=True, show_vectors=True,
                     vec_steps=(20, 20), dpi=120) -> str:
    """HOM 進行波の 1 モードの E / H パネルを時間位相で動画化した GIF を出力する."""
    ctx = _hom_setup(data, mode, n, phase)
    base = _suptitle(ctx)

    def draw(fig, tp):
        axl = fig.add_subplot(121)
        axr = fig.add_subplot(122)
        _draw_hom_panels(axl, axr, ctx, tp, show_mesh=show_mesh,
                         show_vectors=show_vectors, vec_steps=vec_steps)
        fig.suptitle(f"{base}   θ={tp:.0f}°", fontsize=STYLE.TITLE_FONTSIZE + 1)

    return render_gif(draw, output, n_frames=n_frames, fps=fps,
                      figsize=(12, 6), dpi=dpi)
