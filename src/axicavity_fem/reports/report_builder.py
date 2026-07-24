"""HTML レポート生成.

solve/post 済みの v2 HDF5 から、場マップ PNG（:mod:`reports.plot_tm0` /
:mod:`reports.plot_hom`）と工学パラメータ表（``/post_process``）をまとめた
``index.html`` を生成する。ver1 の ``generate_html_report`` /
``generate_hom_html_report`` を ver2 のデータ構造に合わせて再構成した版。
"""

from __future__ import annotations

import datetime
import html
import os

from ..shared.hdf5_io import is_v2, load_v1_legacy, read_results
from .plot_common import STYLE, make_triangulation, save_figure


# 境界条件エッジの色分け（メッシュ概要図）。
# "None" は BC 未指定の外周辺（対称軸 r=0 など）＝黒。
_BC_EDGE_COLORS = {"PEC": "orange", "E-short": "blue", "M-short": "green",
                   "None": "black"}


def _mesh_overview(data, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from ..shared.boundary_groups import (bc_boundary_segments,
                                          material_interface_geometry,
                                          mean_edge_length)

    mesh = data["mesh"]
    triang = make_triangulation(mesh["vertices"], mesh["simplices"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.triplot(triang, color=STYLE.MESH_COLOR, lw=0.3, alpha=0.6)

    # 境界条件エッジを色分けして重ね描き
    # （PEC=オレンジ / E-short=青 / M-short=緑 / None=黒）
    bc_segs = bc_boundary_segments(mesh["simplices"], mesh["vertices"],
                                   mesh.get("physical_groups"),
                                   mesh["elem_order"])
    has_bc = False
    for name, color in _BC_EDGE_COLORS.items():
        segs = bc_segs.get(name)
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=2.0,
                                             label=name, zorder=10))
            has_bc = True

    # ver2.3: 誘電体界面（隣接要素で ε_r / tanδ が変わる内部辺）をマゼンタ破線で
    mat_segs, _ = material_interface_geometry(
        mesh["simplices"], mesh["vertices"], mesh["elem_order"],
        eps_r_per_element=mesh.get("eps_r_per_element"),
        tan_delta_per_element=mesh.get("tan_delta_per_element"))
    if mat_segs:
        ax.add_collection(LineCollection(mat_segs, colors="magenta",
                                         linewidths=2.0, linestyles="--",
                                         label="dielectric interface",
                                         zorder=11))
        has_bc = True
    if has_bc:
        ax.legend(loc="best", fontsize=8)

    h_mean = mean_edge_length(mesh["simplices"], mesh["vertices"])
    ax.set_aspect("equal")
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.set_title(f"Mesh: {len(mesh['vertices'])} nodes, "
                 f"{len(mesh['simplices'])} elements, "
                 f"mean edge = {h_mean:.4g} m")
    path = os.path.join(output_dir, "mesh_overview.png")
    save_figure(fig, path)
    return "mesh_overview.png"


def _iter_groups(rec):
    """rec から (analysis_type, phase, modeset) を順に返す."""
    if "standing" in rec:
        yield "standing", None, rec["standing"]
    if "traveling" in rec:
        for ph in sorted(rec["traveling"]):
            yield "traveling", ph, rec["traveling"][ph]


def _post_modes(data, n, analysis_type, phase):
    pp = data.get("post_process")
    if not pp or n not in pp:
        return None
    rec = pp[n]
    if analysis_type == "standing":
        return rec.get("standing")
    return rec.get("traveling", {}).get(phase)


def _param_keys(solver_type):
    if solver_type == "tm0":
        return [("frequency_GHz", "f [GHz]"), ("Q", "Q"),
                ("Q_wall", "Q_wall"), ("Q_diel", "Q_diel"),
                ("R_over_Q", "R/Q [Ohm]"), ("V_eff", "V_eff [V]"),
                ("U_stored", "U [J]"), ("P_loss", "P_loss [W]"),
                ("P_diel", "P_diel [W]"),
                ("group_velocity", "v_g [m/s]")]
    return [("frequency_GHz", "f [GHz]"), ("Q", "Q"),
            ("Q_wall", "Q_wall"), ("Q_diel", "Q_diel"),
            ("U_stored", "U [J]"), ("P_loss", "P_loss [W]"),
            ("P_diel", "P_diel [W]"),
            ("P_flow_zmin", "P_flow [W]"), ("group_velocity", "v_g [m/s]")]


def build_report(input_h5, output_dir=None, *, solver_type=None,
                 dpi=120, time_phase=0.0, show_mesh=True, animate=False) -> str:
    """v2 HDF5 から HTML レポート（index.html + PNG）を生成する.

    Args:
        input_h5: 入力 HDF5（solve / post 済み）。
        output_dir: 出力ディレクトリ（省略で ``<input>_report``）。
        solver_type: "tm0"/"hom"。None で H5 の solver_type。
        dpi: 画像解像度。
        time_phase: 進行波の時間位相 [度]。
        show_mesh: 場マップにメッシュを重ねる。
        animate: True かつ進行波のとき、各モードに GIF アニメーションを追加する。

    Returns:
        生成した index.html のパス。
    """
    data = read_results(input_h5) if is_v2(input_h5) else load_v1_legacy(input_h5)
    solver_type = solver_type or data["solver_type"]
    if output_dir is None:
        output_dir = os.path.splitext(input_h5)[0] + "_report"
    os.makedirs(output_dir, exist_ok=True)

    if solver_type == "tm0":
        from .plot_tm0 import animate_tm0_mode, plot_tm0_mode
    else:
        from .plot_hom import animate_hom_mode, plot_hom_mode

    mesh_png = _mesh_overview(data, output_dir)
    keys = _param_keys(solver_type)
    sections = []

    for n in sorted(data["results_by_n"]):
        rec = data["results_by_n"][n]
        for analysis_type, phase, modeset in _iter_groups(rec):
            post = _post_modes(data, n, analysis_type, phase)
            rows = []
            for m in range(len(modeset["frequencies"])):
                tag = f"n{n}_{analysis_type}"
                if phase is not None:
                    tag += f"_ph{phase:g}"
                png = f"{tag}_mode{m}.png"
                png_path = os.path.join(output_dir, png)
                if solver_type == "tm0":
                    plot_tm0_mode(data, m, png_path, phase=phase,
                                  time_phase=time_phase, show_mesh=show_mesh,
                                  dpi=dpi)
                else:
                    plot_hom_mode(data, m, png_path, n=n, phase=phase,
                                  time_phase=time_phase, show_mesh=show_mesh,
                                  dpi=dpi)
                gif = None
                if animate and analysis_type == "traveling":
                    gif = f"{tag}_mode{m}.gif"
                    gif_path = os.path.join(output_dir, gif)
                    if solver_type == "tm0":
                        animate_tm0_mode(data, m, gif_path, phase=phase,
                                         show_mesh=show_mesh, dpi=dpi)
                    else:
                        animate_hom_mode(data, m, gif_path, n=n, phase=phase,
                                         show_mesh=show_mesh, dpi=dpi)
                params = post[m] if (post and m < len(post)) else {}
                freq = float(modeset["frequencies"][m])
                rows.append((m, freq, params, png, gif))
            sections.append((n, analysis_type, phase, rows))

    index_path = os.path.join(output_dir, "index.html")
    _write_html(index_path, solver_type, data, mesh_png, sections, keys)
    return index_path


def _fmt(v):
    try:
        return f"{float(v):.4e}"
    except (TypeError, ValueError):
        return html.escape(str(v))


def _write_html(path, solver_type, data, mesh_png, sections, keys):
    from ..shared.boundary_groups import mean_edge_length

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mesh = data["mesh"]
    h_mean = mean_edge_length(mesh["simplices"], mesh["vertices"])
    parts = [f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>AxiCavity-FEM Report ({solver_type.upper()})</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background:#f4f7f6; color:#333; }}
.header {{ text-align:center; border-bottom:3px solid #2c3e50; padding-bottom:10px; }}
.card {{ background:#fff; padding:20px; border-radius:8px;
         box-shadow:0 2px 10px rgba(0,0,0,0.1); margin-bottom:25px; }}
table {{ border-collapse:collapse; width:100%; margin-top:10px; }}
th,td {{ border:1px solid #ddd; padding:8px; text-align:right; }}
th {{ background:#2c3e50; color:#fff; text-align:center; }}
tr:nth-child(even) {{ background:#f9f9f9; }}
img {{ max-width:100%; border:1px solid #ddd; border-radius:4px; }}
.mode {{ border-top:2px solid #3498db; padding-top:12px; margin-top:24px; }}
</style></head><body>
<div class="header"><h1>AxiCavity-FEM Analysis Report</h1>
<p>{now} &nbsp;|&nbsp; solver: <b>{solver_type.upper()}</b></p></div>
<div class="card"><h2>Mesh</h2>
<p>nodes: {len(mesh['vertices'])} &nbsp; elements: {len(mesh['simplices'])} &nbsp;
order: {mesh.get('elem_order')} &nbsp; mean edge length: {h_mean:.4g} m</p>
<p style="font-size:0.9em;color:#666">
boundary colors: <b style="color:orange">PEC</b> /
<b style="color:blue">E-short</b> / <b style="color:green">M-short</b> /
<b style="color:black">None</b> (no BC: symmetry axis r=0 etc.)</p>
<img src="{mesh_png}" style="max-width:60%"></div>
"""]

    for n, analysis_type, phase, rows in sections:
        ph = f", phase={phase:g}deg" if phase is not None else ""
        parts.append(f'<div class="card"><h2>n = {n} &nbsp; '
                     f'{analysis_type}{ph}</h2>')
        # サマリー表
        parts.append("<table><thead><tr><th>Mode</th>"
                     + "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in keys)
                     + "</tr></thead><tbody>")
        for m, freq, params, _png, _gif in rows:
            cells = [f"<td>{m}</td>"]
            for key, _lbl in keys:
                if key == "frequency_GHz":
                    cells.append(f"<td>{freq:.6f}</td>")
                else:
                    cells.append(f"<td>{_fmt(params.get(key, 'N/A'))}</td>")
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</tbody></table>")
        # 各モード画像（GIF があればアニメーションも併記）
        for m, freq, _params, png, gif in rows:
            parts.append(f'<div class="mode"><h3>Mode {m}: {freq:.6f} GHz</h3>'
                         f'<img src="{png}">')
            if gif:
                parts.append(f'<h4>Animation</h4><img src="{gif}">')
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
