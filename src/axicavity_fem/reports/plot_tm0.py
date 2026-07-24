"""TM0 場マップ描画（Result Viewer 準拠: H_phi カラー＋E-Lines＋ベクトル＋PEC線）.

静止画 PNG は上下 2 段（上=場マップ、下=軸上 E_z(z)）。進行波は GIF アニメーション
（場マップを時間位相で動画化）も生成できる（:func:`animate_tm0_mode`）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..fem_tm0.field_recon import TM0FieldReconstructor
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
_AXIAL_RE_COLOR = "darkblue"
_AXIAL_IM_COLOR = "crimson"


def _select(data, phase):
    rec = data["results_by_n"][0]
    if "standing" in rec:
        return rec["standing"], "standing", None
    trav = rec["traveling"]
    if phase is None:
        phase = sorted(trav)[0]
    return trav[phase], "traveling", phase


def _tm0_setup(data, mode, phase):
    """時間位相に依存しないセットアップ（recon・三角形分割・PEC 幾何）を作る."""
    modeset, analysis_type, phase = _select(data, phase)
    if mode >= len(modeset["frequencies"]):
        raise IndexError(f"mode {mode} は範囲外（モード数 {len(modeset['frequencies'])}）")
    mesh = data["mesh"]
    recon = TM0FieldReconstructor(mesh["vertices"], mesh["simplices"],
                                  mesh["elem_order"], analysis_type,
                                  eps_r_per_element=mesh.get("eps_r_per_element"))
    segments, _ = pec_boundary_geometry(mesh["simplices"], mesh["vertices"],
                                        mesh.get("physical_groups"),
                                        mesh["elem_order"])
    mat_segments, _ = material_interface_geometry(
        mesh["simplices"], mesh["vertices"], mesh["elem_order"],
        eps_r_per_element=mesh.get("eps_r_per_element"),
        tan_delta_per_element=mesh.get("tan_delta_per_element"))
    return {
        "recon": recon,
        "eigvec": np.asarray(modeset["eigenvectors"][mode]),
        "freq": float(modeset["frequencies"][mode]),
        "mesh": mesh,
        "triang_vis": make_triangulation_vis(mesh["vertices"],
                                             mesh["simplices"],
                                             mesh["elem_order"]),
        "triang_raw": make_triangulation(mesh["vertices"], mesh["simplices"]),
        "segments": segments,
        "mat_segments": mat_segments,
        "is_traveling": analysis_type == "traveling",
        "analysis_type": analysis_type,
        "phase": phase,
        "mode": mode,
    }


def _suptitle(ctx):
    s = (f"TM0 mode {ctx['mode']}  f = {ctx['freq']:.6f} GHz  "
         f"({ctx['analysis_type']})")
    if ctx["is_traveling"]:
        s += f"  phase={ctx['phase']}deg"
    return s


def _tm0_vectors(ax, recon, eigvec, freq, vertices, time_phase, vec_steps):
    """場マップに (E_z, E_r) のグリッド quiver（瞬時値）を描く."""
    verts = np.asarray(vertices)
    nz = max(2, vec_steps[0])
    nr = max(2, vec_steps[1])
    zg = np.linspace(verts[:, 0].min(), verts[:, 0].max(), nz)
    rg = np.linspace(verts[:, 1].min(), verts[:, 1].max(), nr)
    Z, R, U, V = [], [], [], []
    for z in zg:
        for r in rg:
            res = recon.calculate_fields(eigvec, freq, z, r,
                                         theta=np.deg2rad(time_phase))
            if res is None:
                continue
            Z.append(z); R.append(r)
            U.append(np.real(res["Ez"])); V.append(np.real(res["Er"]))
    if Z:
        ax.quiver(Z, R, U, V, color=_QUIVER_COLOR, alpha=0.8, pivot="middle")


def _draw_tm0_map(fig, ax, ctx, time_phase, *, show_mesh, show_vectors,
                  vec_steps, e_line_levels):
    """場マップ（H_phi カラー＋電気力線＋ベクトル＋PEC線）を ax に描く."""
    recon, eigvec, freq = ctx["recon"], ctx["eigvec"], ctx["freq"]
    mesh = ctx["mesh"]
    node = recon.calculate_all_node_fields(
        eigvec, freq, theta=np.deg2rad(time_phase),
        return_complex=ctx["is_traveling"])

    hmax = float(np.max(np.abs(eigvec))) or 1.0
    tcf = ax.tricontourf(ctx["triang_vis"], np.real(node["H_theta"]),
                         levels=40, cmap=STYLE.DEFAULT_CMAP,
                         vmin=-hmax, vmax=hmax)
    fig.colorbar(tcf, ax=ax, shrink=0.8)

    psi = np.real(node["Psi"])
    if ctx["is_traveling"]:
        psi_max = float(np.max(np.abs(mesh["vertices"][:, 1] * eigvec)))
        lv = (np.linspace(-psi_max, psi_max, e_line_levels)
              if psi_max > 1e-20 and e_line_levels >= 2 else e_line_levels)
    else:
        lv = e_line_levels
    # E-Lines: 青紫の実線（負レベルが破線になる既定を linestyles で抑制）
    ax.tricontour(ctx["triang_vis"], psi, levels=lv, colors="blueviolet",
                  linewidths=1.5, linestyles="solid", alpha=0.9)

    if show_mesh:
        ax.triplot(ctx["triang_raw"], color=STYLE.MESH_COLOR, lw=STYLE.MESH_LW,
                   alpha=STYLE.MESH_ALPHA)
    if show_vectors:
        _tm0_vectors(ax, recon, eigvec, freq, mesh["vertices"], time_phase,
                     vec_steps)
    if ctx["segments"]:
        ax.add_collection(LineCollection(ctx["segments"], colors="black",
                                         linewidths=1.2, zorder=10))
    # ver2.3: 誘電体界面（ε_r / tanδ が変わる内部辺）をマゼンタ破線で
    if ctx["mat_segments"]:
        ax.add_collection(LineCollection(ctx["mat_segments"], colors="magenta",
                                         linewidths=1.5, linestyles="--",
                                         zorder=9))
    ax.set_aspect("equal")
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.set_title(r"$H_\phi$ (color) + E-lines + E-vectors",
                 fontsize=STYLE.TITLE_FONTSIZE)


def _plot_axial(ax, recon, eigvec, freq, vertices, is_traveling, n=400):
    """軸上 (r=0) の E_z を 1D プロットする（traveling は実部/虚部併記）."""
    verts = np.asarray(vertices)
    z0, z1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    zs = np.linspace(z0, z1, n)
    ez = np.array([
        (recon.calculate_fields(eigvec, freq, z, 0.0, theta=0.0,
                                return_complex=True) or {"Ez": 0.0})["Ez"]
        for z in zs
    ])
    if is_traveling:
        ax.plot(zs, np.real(ez), color=_AXIAL_RE_COLOR, lw=2,
                label="Ez (Re, 0°)")
        ax.plot(zs, np.imag(ez), color=_AXIAL_IM_COLOR, lw=1.5, ls="--",
                label="Ez (Im, 90°)")
        ax.legend(fontsize=8)
    else:
        ax.plot(zs, np.real(ez), color=_AXIAL_RE_COLOR, lw=2, label="Ez")
    ax.set_xlim(z0, z1)
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel("Ez on axis [a.u.]")
    ax.set_title("Axial electric field (r=0)", fontsize=STYLE.TITLE_FONTSIZE)
    ax.grid(True, ls=":", alpha=0.5)


def plot_tm0_mode(data, mode, output, *, phase=None, time_phase=0.0,
                  show_mesh=True, show_vectors=True, vec_steps=(20, 20),
                  e_line_levels=20, dpi=120) -> str:
    """TM0 の 1 モードを場マップ＋軸上電場で PNG 出力する（Result Viewer 準拠）."""
    ctx = _tm0_setup(data, mode, phase)

    fig = plt.figure(figsize=(8, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_map = fig.add_subplot(gs[0])
    ax_axial = fig.add_subplot(gs[1])
    fig.suptitle(_suptitle(ctx), fontsize=STYLE.TITLE_FONTSIZE + 2)

    _draw_tm0_map(fig, ax_map, ctx, time_phase, show_mesh=show_mesh,
                  show_vectors=show_vectors, vec_steps=vec_steps,
                  e_line_levels=e_line_levels)
    _plot_axial(ax_axial, ctx["recon"], ctx["eigvec"], ctx["freq"],
                ctx["mesh"]["vertices"], ctx["is_traveling"])
    save_figure(fig, output, dpi=dpi)
    return output


def animate_tm0_mode(data, mode, output, *, phase=None, n_frames=36, fps=12,
                     show_mesh=True, show_vectors=True, vec_steps=(20, 20),
                     e_line_levels=20, dpi=120) -> str:
    """TM0 進行波の 1 モードの場マップを時間位相で動画化した GIF を出力する."""
    ctx = _tm0_setup(data, mode, phase)
    base = _suptitle(ctx)

    def draw(fig, tp):
        ax = fig.add_subplot(111)
        _draw_tm0_map(fig, ax, ctx, tp, show_mesh=show_mesh,
                      show_vectors=show_vectors, vec_steps=vec_steps,
                      e_line_levels=e_line_levels)
        fig.suptitle(f"{base}   θ={tp:.0f}°", fontsize=STYLE.TITLE_FONTSIZE + 1)

    return render_gif(draw, output, n_frames=n_frames, fps=fps,
                      figsize=(8, 6.5), dpi=dpi)
