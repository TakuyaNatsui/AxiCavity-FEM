"""アプリのアイコン（`src/axicavity_fem/gui/ui/app_icon.png`・`app_icon_small.png`）を計算結果から描く.

サンプル `samples/s-band_1cell.gmshproj` の TM0 進行波（位相 120°、2π/3 モード）を解き、結果画面と同じ量
（色 = H_φ、紫の線 = 電気力線 r·H_φ の等高線、黒線 = PEC 壁）で 1 セルを描く。z は正方形になるよう
セルの両側へ少し広げる（隣のセルは位相を 120° ずらした同じ場）。

- `app_icon.png`（256 px）: 48 px 以上で使う。線は細かく 22 本
- `app_icon_small.png`（32 px）: 32 px 以下で使う。同じ構図で線を 9 本に減らして太くする

場の全体位相は固有値ソルバー次第なので、時間位相は ver2.3 の結果画面のスクリーンショット（時間位相 80°）の
色に合わせて選ぶ（何度実行しても同じ絵になる）。色の範囲は複素振幅の 1.2 倍（橙〜黄〜黄緑の色調）。

    python tools/make_app_icon.py                   # gui/ui/ に書き出す（要 gmsh・matplotlib）
    python tools/make_app_icon.py --sheet a.png     # 確認用の一覧（256 / 64 / 48 / 32 / 24 / 16 px）
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "s-band_1cell.gmshproj"
UI_DIR = ROOT / "src" / "axicavity_fem" / "gui" / "ui"
LARGE_FILE, SMALL_FILE = UI_DIR / "app_icon.png", UI_DIR / "app_icon_small.png"
LARGE_SIZE, SMALL_SIZE = 256, 32
RENDER = 512                         # 描く大きさ（px）。ここから縮小する
CORNER = 56                          # 角の丸み（RENDER に対して）
PHASE = 120.0                        # セル間の位相 [deg]
COLOR_SCALE = 1.2                    # 色の範囲 = ±振幅 × これ
LINE_COLOR, WALL_COLOR = "#8a3fd1", "#1a1a1a"
# ver2.3 のスクリーンショットから読んだ H_φ（振幅を 100 として）: (z, r) [mm] → 値
REFERENCE = {(17.5, 44.0): 72.0, (18.0, 30.0): 60.0, (2.0, 14.0): -12.0, (33.0, 14.0): 30.0, (17.5, 8.0): 30.0}


def solve():
    """サンプルを解いて (nodes [mm], 2 次三角形 (E, 6), 固有ベクトル（振幅 100）, PEC の線 [mm]) を返す."""
    from axicavity_fem.fem_tm0.solver import solve_tm0_traveling
    from axicavity_fem.shared.boundary_groups import pec_boundary_geometry
    from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
    from axicavity_fem.shared.mesh_loader import load_mesh
    from axicavity_fem.shared.multi_region_model import MultiRegionGeometry

    geom = MultiRegionGeometry.load_json(SAMPLE)
    geom.mesh_size = 0.8
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        msh = Path(tmp) / "cell.msh"
        export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
        mesh = load_mesh(msh, element_order=2)
        result = solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(PHASE,))
    vec = np.asarray(result.phase_results[PHASE]["eigenvectors"][0])
    nodes = np.asarray(mesh.nodes) * 1e3
    segments, _ = pec_boundary_geometry(mesh.elements, mesh.nodes, mesh.physical_groups, 2)
    # 端面（周期面の開口）は壁ではないので描かない（サンプルの端面は単独で解くための PEC ラベル付き）
    z_ends = (float(nodes[:, 0].min()), float(nodes[:, 0].max()))
    walls = [seg for seg in (np.asarray(s) * 1e3 for s in segments)
             if not any(np.all(np.abs(seg[:, 0] - z) < 1e-6) for z in z_ends)]
    return nodes, np.asarray(mesh.elements), vec / np.max(np.abs(vec)) * 100.0, walls


def match_time_phase(nodes, vec) -> float:
    """REFERENCE の値に最も近くなる時間位相 [deg]."""
    idx = [int(np.argmin(np.hypot(nodes[:, 0] - z, nodes[:, 1] - r))) for z, r in REFERENCE]
    want = np.array(list(REFERENCE.values()))
    phases = np.arange(0.0, 360.0, 0.05)
    err = [np.sum((np.real(vec[idx] * np.exp(1j * np.radians(t))) - want) ** 2) for t in phases]
    return float(phases[int(np.argmin(err))])


def draw(nodes, elements, vec, segments, t, *, lines: int, line_width: float, wall_width: float):
    """1 セル（両側に隣のセルを少し）を RENDER px の正方形に描いた PIL 画像（角は丸く透明）."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from PIL import Image, ImageDraw

    tris = np.vstack([elements[:, [0, 3, 5]], elements[:, [3, 1, 4]],
                      elements[:, [5, 4, 2]], elements[:, [3, 4, 5]]])
    length, radius = float(nodes[:, 0].max()), float(nodes[:, 1].max())
    cells = (-1, 0, 1)
    vmax = 100.0 * COLOR_SCALE
    fig = plt.figure(figsize=(RENDER / 100, RENDER / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(length / 2 - radius / 2, length / 2 + radius / 2)
    ax.set_ylim(0.0, radius)
    fields = {k: np.real(vec * np.exp(1j * np.radians(t - PHASE * k))) for k in cells}
    psi_max = max(np.max(np.abs(f * nodes[:, 1])) for f in fields.values())
    for k, f in fields.items():
        tri = mtri.Triangulation(nodes[:, 0] + k * length, nodes[:, 1], tris)
        ax.tricontourf(tri, f, levels=np.linspace(-100, 100, 81), cmap="jet", vmin=-vmax, vmax=vmax)
        ax.tricontour(tri, f * nodes[:, 1], levels=np.linspace(-psi_max, psi_max, lines + 2)[1:-1],
                      colors=LINE_COLOR, linewidths=line_width, linestyles="solid")
        for seg in segments:
            ax.plot(seg[:, 0] + k * length, seg[:, 1], color=WALL_COLOR, lw=wall_width, solid_capstyle="round")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor="white", format="png")
    plt.close(fig)
    image = Image.open(buf).convert("RGBA")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, RENDER - 1, RENDER - 1], radius=CORNER, fill=255)
    image.putalpha(mask)
    return image


def make_images():
    """(大きいアイコン 256 px, 小さいアイコン 32 px) の PIL 画像."""
    from PIL import Image

    nodes, elements, vec, segments = solve()
    t = match_time_phase(nodes, vec)
    large = draw(nodes, elements, vec, segments, t, lines=22, line_width=2.4, wall_width=3.2)
    small = draw(nodes, elements, vec, segments, t, lines=9, line_width=5.0, wall_width=6.0)
    return large.resize((LARGE_SIZE, LARGE_SIZE), Image.LANCZOS), small.resize((SMALL_SIZE, SMALL_SIZE), Image.LANCZOS)


def write_sheet(large, small, out: Path) -> None:
    from PIL import Image

    sheet = Image.new("RGBA", (560, 300), (240, 240, 240, 255))
    sheet.alpha_composite(large, (12, 22))
    x = 290
    for size in (64, 48, 32, 24, 16):
        src = large if size > SMALL_SIZE else small
        img = src if size == src.size[0] else src.resize((size, size), Image.LANCZOS)
        sheet.alpha_composite(img, (x, 22))
        x += size + 8
    sheet.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", type=Path, help="確認用の一覧画像だけを書き出す")
    args = parser.parse_args()
    large, small = make_images()
    if args.sheet:
        write_sheet(large, small, args.sheet)
        print(f"書き出し: {args.sheet}")
        return
    large.save(LARGE_FILE, optimize=True)
    small.save(SMALL_FILE, optimize=True)
    print(f"書き出し: {LARGE_FILE.name}, {SMALL_FILE.name}")


if __name__ == "__main__":
    main()
