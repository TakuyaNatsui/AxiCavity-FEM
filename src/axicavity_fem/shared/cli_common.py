"""TM0/HOM 共通の CLI 引数定義と位相文字列パーサ.

``parse_phase_list("0:180:20")`` → ``[0.0, 20.0, ..., 180.0]``。
``add_common_solver_args(parser)`` で -m/--elem-order/--num-modes/-o/-p を一括追加。

ver1 の TM0 ``parse_phase_arg`` と HOM ``calclation_parser.parse_phase_arg`` を統合。
"""

from __future__ import annotations

import argparse

import numpy as np


def parse_phase_list(phase_str: str) -> list[float]:
    """位相指定文字列を位相値（度）のソート済みリストへ変換する.

    対応形式:
      - 単一値:        ``"120"``
      - カンマ区切り:  ``"60,90,120"``
      - レンジ(3要素): ``"0:180:20"`` → start:end:step (np.arange)
      - レンジ(2要素): ``"0:180"`` → [0.0, 180.0]
      - 混合:          ``"0:180:20,270"``

    Returns:
        ソート済み・重複排除済みの位相リスト [度]。
    """
    phases: list[float] = []
    for part in phase_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            comps = part.split(":")
            if len(comps) == 3:
                start, end, step = map(float, comps)
                phases.extend(np.arange(start, end + step * 1e-5, step).tolist())
            elif len(comps) == 2:
                start, end = map(float, comps)
                phases.extend([start, end])
        else:
            phases.append(float(part))
    return sorted(set(phases))


def add_common_solver_args(parser: argparse.ArgumentParser) -> None:
    """solve サブコマンド共通の引数を parser に追加する."""
    parser.add_argument("-m", "--mesh", dest="mesh_file", required=True,
                        help="入力メッシュファイル (.msh)")
    parser.add_argument("--elem-order", type=int, default=2, choices=[1, 2],
                        dest="elem_order", help="FEM 要素次数 (1 or 2)")
    parser.add_argument("--num-modes", type=int, default=10, dest="num_modes",
                        help="出力する固有モード数")
    parser.add_argument("-o", "--output", dest="output_file", default=None,
                        help="出力 HDF5 ファイル名 (.h5)")
    parser.add_argument("-p", "--phase", dest="phase", default="0.0",
                        help="位相シフト [度]。0.0 で定在波、それ以外で進行波。"
                             "単一値 '120'、スキャン '0:180:20'")
    parser.add_argument("--target-freq", type=float, default=None,
                        dest="target_freq",
                        help="探索したい周波数 [GHz] (ver2.3)。指定すると "
                             "shift-invert のシフト値 sigma=(2πf/c)^2 として "
                             "この周波数に近いモードを返す。省略時は "
                             "メッシュの r_max から自動推定（最低次から）。")
