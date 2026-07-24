"""Plain-text result summary for solve / post.

Writes a `.txt` next to the HDF5 output (`foo.h5` -> `foo.txt`) so the analysis
mode, mesh file and resonant frequencies (and Q etc. for post) can be checked
quickly in a text editor. The GUI Run Solver / Run Post-Process buttons get these
for free because they shell out to the same CLI.

The file content is intentionally English-only. Frequencies are printed with ~15
significant digits (the eigenvalue frequency itself carries >10 significant
figures of precision).
"""

from __future__ import annotations

import datetime
import os

# 周波数の表示: 15 有効桁 (周波数は 10 桁以上の精度を持つ)
_FREQ_FMT = "{:.15g}"


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def txt_path_for(h5_path: str) -> str:
    """`foo.h5` -> `foo.txt` (replace the extension with .txt)."""
    return os.path.splitext(h5_path)[0] + ".txt"


def _header_lines(title: str, parameters: dict) -> list[str]:
    p = parameters or {}
    solver = str(p.get("solver_type", "?")).upper()
    return [
        f"AxiCavity-FEM result summary ({title})",
        f"generated: {_now()}",
        "=" * 50,
        f"analysis type : {solver}",
        f"mesh file     : {p.get('mesh_file', '?')}",
        f"element order : {p.get('elem_order', '?')}",
        f"num modes     : {p.get('num_modes', '?')}",
        f"phase         : {p.get('phase', '?')}",
    ]


def _iter_groups(rec):
    """Yield (label, list) from rec (results_by_n[n] or post_by_n[n]),
    flattening standing / traveling(per-phase)."""
    if "standing" in rec:
        yield "standing", rec["standing"]
    if "traveling" in rec:
        for ph in sorted(rec["traveling"].keys()):
            yield f"traveling phase={ph:g}deg", rec["traveling"][ph]


def render_solve_summary(parameters: dict, results_by_n: dict) -> str:
    """Render the solve result (frequencies only) as text."""
    lines = _header_lines("solve", parameters)
    for n in sorted(results_by_n.keys()):
        rec = results_by_n[n]
        for label, modeset in _iter_groups(rec):
            lines.append("-" * 50)
            lines.append(f"[n={n}] {label}")
            for k, f in enumerate(list(modeset["frequencies"])):
                lines.append(f"  mode {k:2d}: f = "
                             + _FREQ_FMT.format(float(f)) + " GHz")
    lines.append("")
    return "\n".join(lines)


# post parameter dict: (key, label, format). Frequency uses ~15 sig digits.
_POST_FIELDS = [
    ("frequency_GHz", "f", _FREQ_FMT + " GHz"),
    ("Q", "Q", "{:.6e}"),
    ("Q_wall", "Q_wall", "{:.6e}"),
    ("Q_diel", "Q_diel", "{:.6e}"),
    ("R_over_Q", "R/Q", "{:.6e} ohm"),
    ("V_eff", "V_eff", "{:.6e} V"),
    ("U_stored", "U", "{:.6e} J"),
    ("P_loss", "P_loss", "{:.6e} W"),
    ("P_diel", "P_diel", "{:.6e} W"),
    ("P_flow_zmin", "P_flow", "{:.6e} W"),
    ("group_velocity", "v_group", "{:.6e} m/s"),
    ("attenuation", "alpha", "{:.6e}"),
]


def render_post_summary(parameters: dict, post_by_n: dict) -> str:
    """Render the post result (frequency + Q etc.) as text."""
    lines = _header_lines("post", parameters)
    for n in sorted(post_by_n.keys()):
        rec = post_by_n[n]
        for label, plist in _iter_groups(rec):
            lines.append("-" * 50)
            lines.append(f"[n={n}] {label}")
            for k, params in enumerate(plist):
                lines.append(f"  mode {k}:")
                for key, name, fmt in _POST_FIELDS:
                    if key in params and params[key] is not None:
                        lines.append(f"      {name:<8}= "
                                     + fmt.format(float(params[key])))
    lines.append("")
    return "\n".join(lines)


def write_solve_summary(h5_path: str, parameters: dict, results_by_n: dict) -> str:
    """Write the solve summary to `<h5>.txt` and return its path."""
    path = txt_path_for(h5_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_solve_summary(parameters, results_by_n))
    return path


def write_post_summary(h5_path: str, parameters: dict, post_by_n: dict) -> str:
    """Write the post summary to `<h5>.txt` and return its path."""
    path = txt_path_for(h5_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_post_summary(parameters, post_by_n))
    return path
