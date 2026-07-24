"""統一 CLI のトップレベル argparse とディスパッチ.

サブコマンド::

    axicavity-fem solve  --type {tm0,hom} -m MESH [--az-order N ...]
                         [--elem-order {1,2}] [--num-modes K] [-p PHASE] [-o OUT]
    axicavity-fem post   --type {tm0,hom} -i RAW.h5 [-o OUT] [--cond S/m] [--beta β]
    axicavity-fem info   -i FILE.h5
    axicavity-fem export ...   (reports レイヤ実装後に有効)
    axicavity-fem plot   ...   (reports レイヤ実装後に有効)
"""

from __future__ import annotations

import argparse
import sys

from axicavity_fem import __version__

from ..shared.cli_common import add_common_solver_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axicavity-fem",
        description="軸対称空洞共振モードの 2D FEM 解析 (TM0 / HOM 統合)",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # --- solve ---
    p_solve = sub.add_parser("solve", help="FEM 固有値解析を実行")
    p_solve.add_argument("--type", choices=["tm0", "hom"], required=True,
                         help="ソルバ種別")
    add_common_solver_args(p_solve)
    p_solve.add_argument("--az-order", type=int, nargs="+", default=[0],
                         dest="az_order",
                         help="HOM の方位角モード次数 n（例: 0 1 2）")
    p_solve.add_argument("--materials", dest="materials_file", default=None,
                         help="material_table を含む JSON ファイル "
                              "(material_tag → {eps_r, mu_r})。省略時は "
                              "<mesh>.materials.json を自動探索。TM0 のみ有効。")

    # --- post ---
    p_post = sub.add_parser("post", help="ポストプロセス（パラメータ計算）")
    p_post.add_argument("--type", choices=["tm0", "hom"], default=None,
                        help="ソルバ種別（省略時は H5 の solver_type を使用）")
    p_post.add_argument("-i", "--input", dest="input_file", required=True,
                        help="入力 HDF5（solve の出力）")
    p_post.add_argument("-o", "--output", dest="output_file", default=None,
                        help="出力 HDF5（省略時は入力に追記）")
    p_post.add_argument("--cond", type=float, default=5.8e7,
                        help="壁の導電率 [S/m]")
    p_post.add_argument("--beta", type=float, default=1.0,
                        help="粒子 β = v/c（TM0 の走行時間係数）")

    # --- info ---
    p_info = sub.add_parser("info", help="HDF5 ファイルの構造ダンプ")
    p_info.add_argument("-i", "--input", dest="input_file", required=True,
                        help="入力 HDF5")

    # --- export ---
    p_export = sub.add_parser("export", help="電磁場マップ出力 (area/line/axis)")
    p_export.add_argument("--type", choices=["tm0", "hom"], default=None,
                          help="ソルバ種別（省略時は H5 の solver_type）")
    p_export.add_argument("-i", "--input", dest="input_file", required=True,
                          help="入力 HDF5（solve/post の出力）")
    p_export.add_argument("-o", "--output", dest="output_file", required=True,
                          help="出力ベースパス（.h5/.txt が自動付与）")
    p_export.add_argument("-m", "--mode", type=int, default=0, help="モード番号")
    p_export.add_argument("--n", type=int, default=None,
                          help="方位角次数 n（HOM, 省略で自動）")
    p_export.add_argument("--shape", choices=["area", "line", "axis"],
                          default="area", help="出力形状")
    p_export.add_argument("--z-range", default=None,
                          help='z 範囲 "zmin,zmax" [m]（省略で全体）')
    p_export.add_argument("--r-range", default=None,
                          help='r 範囲 "rmin,rmax" [m]（省略で全体, area）')
    p_export.add_argument("--nz", type=int, default=200, help="z 方向点数 (area)")
    p_export.add_argument("--nr", type=int, default=100, help="r 方向点数 (area)")
    p_export.add_argument("--p1", default=None, help='直線始点 "z,r" (line)')
    p_export.add_argument("--p2", default=None, help='直線終点 "z,r" (line)')
    p_export.add_argument("--npts", type=int, default=500, help="直線の点数")
    p_export.add_argument("--phase", type=float, default=None,
                          help="進行波の位相 [度]（省略で先頭）")
    p_export.add_argument("--time-phase", type=float, default=0.0,
                          dest="time_phase", help="瞬時値の時間位相 [度]")
    p_export.add_argument("--scale", type=float, default=1.0, help="場の倍率")
    p_export.add_argument("--scale-to-power", type=float, default=None,
                          dest="scale_to_power",
                          help="目標壁損失 [W]（post 済みファイルが必要）")
    p_export.add_argument("--instant", action="store_true",
                          help="進行波で瞬時実数値を出力 (TM0)")
    p_export.add_argument("--format", choices=["h5", "txt", "both"],
                          default="both", dest="fmt", help="出力フォーマット")

    # --- plot ---
    p_plot = sub.add_parser("plot", help="場マップを PNG で可視化")
    p_plot.add_argument("--type", choices=["tm0", "hom"], default=None,
                        help="ソルバ種別（省略時は H5 の solver_type）")
    p_plot.add_argument("-i", "--input", dest="input_file", required=True,
                        help="入力 HDF5（solve/post の出力）")
    p_plot.add_argument("-o", "--output", dest="output_file", default=None,
                        help="出力 PNG パス（省略で入力名から自動）")
    p_plot.add_argument("-m", "--mode", type=int, default=0, help="モード番号")
    p_plot.add_argument("--n", type=int, default=None,
                        help="方位角次数 n（HOM, 省略で自動）")
    p_plot.add_argument("--phase", type=float, default=None,
                        help="進行波の位相 [度]（省略で先頭）")
    p_plot.add_argument("--time-phase", type=float, default=0.0,
                        dest="time_phase", help="瞬時値の時間位相 [度]")
    p_plot.add_argument("--no-mesh", action="store_true", dest="no_mesh",
                        help="メッシュの重ね描きを省略")
    p_plot.add_argument("--dpi", type=int, default=120, help="出力解像度")

    # --- report ---
    p_report = sub.add_parser("report", help="HTML レポート生成")
    p_report.add_argument("--type", choices=["tm0", "hom"], default=None,
                          help="ソルバ種別（省略時は H5 の solver_type）")
    p_report.add_argument("-i", "--input", dest="input_file", required=True,
                          help="入力 HDF5（solve / post 済み）")
    p_report.add_argument("-o", "--output", dest="output_dir", default=None,
                          help="出力ディレクトリ（省略で <input>_report）")
    p_report.add_argument("--time-phase", type=float, default=0.0,
                          dest="time_phase", help="進行波の時間位相 [度]")
    p_report.add_argument("--no-mesh", action="store_true", dest="no_mesh",
                          help="場マップにメッシュを重ねない")
    p_report.add_argument("--animate", action="store_true",
                          help="進行波モードに GIF アニメーションを追加する")
    p_report.add_argument("--dpi", type=int, default=120, help="画像解像度")

    return parser


def cli_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        from . import cmd_solve
        return cmd_solve.run(args)
    if args.command == "post":
        from . import cmd_post
        return cmd_post.run(args)
    if args.command == "info":
        from . import cmd_info
        return cmd_info.run(args)
    if args.command == "export":
        from . import cmd_export
        return cmd_export.run(args)
    if args.command == "plot":
        from . import cmd_plot
        return cmd_plot.run(args)
    if args.command == "report":
        from . import cmd_report
        return cmd_report.run(args)
    parser.error(f"未知のコマンド: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(cli_main())
