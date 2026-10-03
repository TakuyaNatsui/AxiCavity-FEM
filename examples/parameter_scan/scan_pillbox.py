"""パラメータスキャンの例: .axiproj のパラメータを Python から変えて解析し、周波数を読む.

    python examples/parameter_scan/scan_pillbox.py            # ピルボックスを作って半径 a を振る
    python examples/parameter_scan/scan_pillbox.py cav.axiproj a 40 42 44 46   # 手持ちのプロジェクトで

ピルボックス（長さ L、半径 a）の TM010 は f = j01 c / (2π a) で a に反比例する。解析値と比べて表にする。
プロジェクトは GUI で作ったものでよい（パラメータ表の名前で指定する。式付きの点と寸法拘束が追従する）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from axicavity_fem.gui.batch import Project
from axicavity_fem.gui.core.document import Param, create_empty_document
from axicavity_fem.gui.core.sketch.model import add_rectangle, point_pos

C_LIGHT = 299792458.0
J01 = 2.404825557695773


def make_pillbox(path: Path, length_mm: float = 100.0, radius_mm: float = 50.0) -> Project:
    """パラメータ L, a を持つピルボックスのプロジェクトを作って保存する（GUI で作るのと同じ文書）."""
    doc = create_empty_document("pillbox")
    doc.params = [Param(id="p_L", name="L", expression=f"{length_mm:g}"),
                  Param(id="p_a", name="a", expression=f"{radius_mm:g}")]
    _lines, points = add_rectangle(doc.sketch, (0, 0), (length_mm, radius_mm))
    for pid in points:
        point = next(e for e in doc.sketch.entities if e.id == pid)
        x, y = point_pos(doc.sketch, pid)
        if x > 0:
            point.xExpr = "L"                 # 右の辺は z = L
        if y > 0:
            point.yExpr = "a"                 # 上の辺は r = a
    doc.mesh.size = 5.0
    doc.analysis.numModes = 2
    project = Project.from_document(doc)
    project.save_as(path)
    return project


def scan(project: Project, name: str, values, post: bool = False) -> list[tuple[float, np.ndarray, Path]]:
    rows = []
    for value in values:
        project.set_params(**{name: value})
        result = project.run(post=post)
        rows.append((float(value), result.frequencies(), result.dir))
    return rows


def main(argv: list[str]) -> int:
    if argv:
        project = Project.open(argv[0])
        name = argv[1]
        values = [float(v) for v in argv[2:]]
    else:
        project = make_pillbox(Path("build") / "pillbox_scan" / "pillbox.axiproj")
        name, values = "a", [40.0, 45.0, 50.0, 55.0, 60.0]
    print(f"project: {project.path}   scan {name} = {values}")
    print(f"{'a [mm]':>8s} {'f0 [GHz]':>10s} {'f1 [GHz]':>10s}   {'TM010 analytic':>14s}   {'diff':>8s}")
    for value, freqs, folder in scan(project, name, values):
        analytic = J01 * C_LIGHT / (2 * np.pi * value * 1e-3) / 1e9 if name == "a" else float("nan")
        f1 = f"{freqs[1]:10.6f}" if len(freqs) > 1 else " " * 10
        diff = f"{(freqs[0] - analytic) / analytic * 100:7.3f}%" if np.isfinite(analytic) else ""
        print(f"{value:8.2f} {freqs[0]:10.6f} {f1}   {analytic:14.6f}   {diff}    {folder.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
