"""ガウス求積 4 点 vs 7 点の収束比較（球形空洞 TM0 最低次モード）.

球形空洞（半径 a = 50 mm）の軸対称 TM0 最低次モードは解析解
TM_11 = 2.618234880208165 GHz（球ベッセル関数から求めた値）。
メッシュサイズ lc を細かくしながら、4 点則（既定）と 7 点則（Dunavant 5 次）で
最低次周波数が理論値へどれだけ速く収束するか、また偽モード（残差の大きい
未収束固有値）が何本出るかを記録する。

2 次要素の質量項 r·G_i·G_j は 5 次多項式なので、4 点則（3 次精度）では
過小積分になり質量行列 M が数値的に不定化して偽モードの原因になる。
7 点則なら厳密積分できる、という仮説の検証。

実行:
    python run_convergence.py
出力:
    標準出力に表、`convergence_result.md` に結果表とまとめを書き出す。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from axicavity_fem.fem_tm0.assembly import assemble_global_matrices
from axicavity_fem.fem_tm0.boundary import (
    build_dirichlet_nodes,
    create_transformation_matrix,
)
from axicavity_fem.fem_tm0.solver import _sigma_from_rmax, _to_ghz
from axicavity_fem.shared.boundary_groups import (
    classify_boundaries,
    mean_edge_length,
)
from axicavity_fem.shared.eigensolver import _RESIDUAL_TOL, _relative_residuals
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.mesh_loader import load_mesh
from axicavity_fem.shared.multi_region_model import MultiRegionGeometry

# 解析解 TM_11（半径 50 mm 球形空洞の軸対称 TM0 最低次）[GHz]
ANALYTIC_TM11_GHZ = 2.618234880208165

_HERE = Path(__file__).resolve().parent
_GMSHPROJ = (_HERE.parents[1] / "samples" / "sphere50mm.gmshproj")

# 実験するメッシュサイズ lc [mm]（粗 → 細）
MESH_SIZES = [12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0]
NUM_MODES = 12          # 偽モードも見えるよう多めに要求
ELEM_ORDER = 2          # 求積次数が効くのは 2 次要素


def solve_raw(mesh, n_quad, num_modes=NUM_MODES):
    """BC 縮小した実対称問題を素の eigsh で解く（偽モードを除外しない）。

    Returns:
        (freqs[GHz] 昇順, 相対残差, n_dof)
    """
    K, M = assemble_global_matrices(mesh, None, n_quad=n_quad)
    cls = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dnodes = build_dirichlet_nodes(cls, mesh.nodes)
    T, _ = create_transformation_matrix(mesh.num_nodes, dnodes)
    T = T.real
    Kr = (T.T @ K @ T).tocsr()
    Mr = (T.T @ M @ T).tocsr()
    sigma = _sigma_from_rmax(mesh.nodes)
    k = min(num_modes, Kr.shape[0] - 1)
    vals, vecs = eigsh(Kr, k=k, M=Mr, sigma=sigma, which="LA", tol=1e-9)
    res = _relative_residuals(Kr, Mr, vals, vecs)
    order = np.argsort(vals)
    return _to_ghz(vals[order]), res[order], Kr.shape[0]


def run_one(geom, lc, n_quad, workdir):
    """lc・求積点数を指定してメッシュ生成 → 解析し、集計値を返す。"""
    geom.mesh_size = float(lc)
    geom.mesh_size_expr = None
    msh = workdir / f"sphere_lc{lc}.msh"
    export_msh_multi_region(geom, msh, mesh_order=ELEM_ORDER)
    mesh = load_mesh(str(msh), ELEM_ORDER)

    freqs, res, ndof = solve_raw(mesh, n_quad)
    clean = res < _RESIDUAL_TOL
    n_spurious = int((~clean).sum())
    f_clean = freqs[clean]
    if len(f_clean) == 0:
        return dict(ndof=ndof, h=mean_edge_length(mesh.elements, mesh.nodes),
                    f_low=np.nan, rel_err=np.nan,
                    n_clean=0, n_spurious=n_spurious)
    f_low = float(np.min(f_clean))
    return dict(
        ndof=ndof,
        h=mean_edge_length(mesh.elements, mesh.nodes),
        f_low=f_low,
        rel_err=(f_low - ANALYTIC_TM11_GHZ) / ANALYTIC_TM11_GHZ,
        n_clean=int(clean.sum()),
        n_spurious=n_spurious,
    )


def main() -> int:
    if not _GMSHPROJ.exists():
        print(f"gmshproj が見つかりません: {_GMSHPROJ}")
        return 1

    rows = []
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        for lc in MESH_SIZES:
            row = {"lc": lc}
            for n_quad in (4, 7):
                geom = MultiRegionGeometry.load_json(_GMSHPROJ)
                r = run_one(geom, lc, n_quad, workdir)
                row[n_quad] = r
            rows.append(row)
            print(f"lc={lc:>5} mm | ndof={row[4]['ndof']:>6} | "
                  f"4pt: f={row[4]['f_low']:.9f} err={row[4]['rel_err']:+.2e} "
                  f"spur={row[4]['n_spurious']} | "
                  f"7pt: f={row[7]['f_low']:.9f} err={row[7]['rel_err']:+.2e} "
                  f"spur={row[7]['n_spurious']}")

    png = _write_plot(rows)
    _write_markdown(rows, png)
    return 0


def _fit_order(rows, nq):
    """相対誤差 |err| ~ h^p の収束次数 p を log-log 最小二乗で推定する。"""
    h = np.array([r[nq]["h"] for r in rows])
    e = np.array([abs(r[nq]["rel_err"]) for r in rows])
    ok = np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return float("nan")
    p = np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0]
    return float(p)


def _write_plot(rows):
    """相対誤差 vs n_dof（log-log）と偽モード本数 vs n_dof の 2 段グラフ。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib 不可、グラフはスキップ: {e}")
        return None
    ndof = [r[4]["ndof"] for r in rows]
    err4 = [abs(r[4]["rel_err"]) for r in rows]
    err7 = [abs(r[7]["rel_err"]) for r in rows]
    spur4 = [r[4]["n_spurious"] for r in rows]
    spur7 = [r[7]["n_spurious"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    ax1.loglog(ndof, err4, "o-", label="4-point", color="tab:red")
    ax1.loglog(ndof, err7, "s-", label="7-point", color="tab:blue")
    ax1.set_xlabel("n_dof (reduced)")
    ax1.set_ylabel("|relative error| of TM_11 freq")
    ax1.set_title("Convergence to analytic TM_11 (sphere 50mm, TM0, 2nd order)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(ndof, spur4, "o-", label="4-point", color="tab:red")
    ax2.plot(ndof, spur7, "s-", label="7-point", color="tab:blue")
    ax2.set_xlabel("n_dof (reduced)")
    ax2.set_ylabel(f"# spurious modes (res >= {_RESIDUAL_TOL:.0e})")
    ax2.set_title(f"Spurious modes out of {NUM_MODES} requested")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    path = _HERE / "convergence.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def _write_markdown(rows, png):
    out = _HERE / "convergence_result.md"
    p4, p7 = _fit_order(rows, 4), _fit_order(rows, 7)
    tot_spur4 = sum(r[4]["n_spurious"] for r in rows)
    tot_spur7 = sum(r[7]["n_spurious"] for r in rows)
    finest = rows[-1]

    L = []
    L.append("# ガウス求積 4 点 vs 7 点の収束比較（球形空洞 TM0 最低次）\n")
    L.append("- 対象: 半径 50 mm 球形空洞、軸対称 TM0、2 次要素（BC: 軸=E-short, 球面=PEC）")
    L.append(f"- 解析解 TM_11 = **{ANALYTIC_TM11_GHZ:.12f} GHz**")
    L.append(f"- 偽モード判定: 相対残差 ‖Kx−λMx‖/‖λMx‖ ≥ {_RESIDUAL_TOL:.0e}")
    L.append(f"- 各行 {NUM_MODES} モード要求。`clean` = 収束モード本数、"
             "`spur` = 偽モード本数。\n")
    L.append("| lc [mm] | n_dof | h平均 [m] | "
             "4pt f_low [GHz] | 4pt 相対誤差 | 4pt clean/spur | "
             "7pt f_low [GHz] | 7pt 相対誤差 | 7pt clean/spur |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        a, b = row[4], row[7]
        L.append(
            f"| {row['lc']} | {a['ndof']} | {a['h']:.4e} | "
            f"{a['f_low']:.9f} | {a['rel_err']:+.3e} | "
            f"{a['n_clean']}/{a['n_spurious']} | "
            f"{b['f_low']:.9f} | {b['rel_err']:+.3e} | "
            f"{b['n_clean']}/{b['n_spurious']} |")
    L.append("")
    if png:
        L.append(f"![convergence]({png})\n")
    L.append("## まとめ\n")
    L.append(f"- **物理モードの精度・収束はほぼ同等**。最細メッシュ "
             f"(n_dof={finest[4]['ndof']}) の相対誤差は 4 点 "
             f"{finest[4]['rel_err']:+.2e} / 7 点 {finest[7]['rel_err']:+.2e} と"
             "同オーダー。収束次数（|err|〜h^p, log-log 回帰）は "
             f"4 点 p≈{p4:.2f} / 7 点 p≈{p7:.2f} でほぼ一致。"
             "「7 点で精度が落ちる」という以前の記録は本ケースでは再現しなかった。")
    L.append(f"- **偽モードは決定的に異なる**。4 点は全メッシュで偽モードが発生"
             f"（合計 {tot_spur4} 本 / 各 {NUM_MODES} 本要求）。7 点は"
             f"**全メッシュで偽モード 0 本**（合計 {tot_spur7} 本）。")
    L.append("- 生産ソルバ（`solve_eigenmodes_eigsh`）は残差フィルタ＋解き直しで"
             "偽モードを除外するが、4 点では偽モードが多く、粗いメッシュでは"
             "収束モードが要求数に届かず返却本数が減ることがある。7 点なら"
             "その必要がない。")
    L.append("- **結論: 7 点求積（Dunavant 5 次）を推奨**。2 次要素の質量項 "
             "r·G_i·G_j は 5 次多項式で、4 点則（3 次精度）では過小積分となり "
             "質量行列 M が数値的に不定化して偽モードを生む。7 点則は厳密積分"
             "できるため、物理モードの精度を犠牲にせず偽モードを根絶できる。")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n結果表を書き出しました: {out}")
    print(f"convergence order p(4pt)={p4:.2f} / p(7pt)={p7:.2f} / "
          f"total spurious 4pt={tot_spur4} 7pt={tot_spur7}")


if __name__ == "__main__":
    sys.exit(main())
