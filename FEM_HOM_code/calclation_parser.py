import argparse
import os
import sys
import numpy as np


def parse_phase_arg(phase_str):
    """位相文字列を解析してリストに変換する。

    対応形式:
      - 単一値:         "120"
      - カンマ区切り:   "60,90,120"
      - レンジ(3要素):  "0:180:20"  → start:end:step (np.arange)
      - レンジ(2要素):  "0:180"     → [0.0, 180.0]
      - 混合:           "0:180:20,270"

    Returns:
        list[float]: ソート済み・重複排除済みの位相値リスト（度）
    """
    phases = []
    for part in phase_str.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            comps = part.split(':')
            if len(comps) == 3:
                start, end, step = map(float, comps)
                phases.extend(np.arange(start, end + step * 1e-5, step).tolist())
            elif len(comps) == 2:
                start, end = map(float, comps)
                phases.extend([start, end])
        else:
            phases.append(float(part))
    return sorted(list(set(phases)))


def parse_calclation_args(argv=None):
    """
    共振空洞解析のコマンドライン引数を解析する関数
    argv: テスト用に引数リストを渡す場合に使用 (通常はNone)
    """
    parser = argparse.ArgumentParser(
        description="2D(軸対称) 共振空洞 共振モード解析ソルバー (HOM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 必須入力 ---
    parser.add_argument('-m', '--mesh', dest='mesh_file', required=True,
                        help='入力メッシュファイル (.msh など)')

    # --- 解析設定 ---
    analysis_group = parser.add_argument_group('解析設定')
    analysis_group.add_argument("--axisymmetric", action="store_true", default=True,
                                dest="axisymmetric",
                                help="軸対称問題として解析する (デフォルト)")
    analysis_group.add_argument("--no-axisymmetric", action="store_false",
                                dest="axisymmetric",
                                help="2D直交座標系の問題として解析する")
    analysis_group.add_argument("--az-order", type=int, nargs='+', default=[0],
                                dest='az_order',
                                help="解析する方位角モード次数 n (例: 0 1 2)")
    analysis_group.add_argument("--elem-order", type=int, default=1, choices=[1, 2],
                                dest='elem_order',
                                help="FEM 要素次数 (1: 1次要素, 2: 2次要素)")

    # --- 境界条件 ---
    bc_group = parser.add_argument_group('境界条件')
    bc_group.add_argument("--left-bc", default=None,
                          choices=["PEC", "Open", "Dirichlet", None],
                          help="左端の境界条件 (定在波時に有効)")
    bc_group.add_argument("--right-bc", default=None,
                          choices=["PEC", "Open", "Dirichlet", None],
                          help="右端の境界条件 (定在波時に有効)")
    bc_group.add_argument("-p", "--phase", type=str, default='0.0',
                          dest='phase',
                          help="周期境界の位相シフト量 [度]。"
                               "0.0 のとき定在波、それ以外は進行波（周期境界）。"
                               "単一値: '120', スキャン: '0:180:20' (start:end:step)")

    # --- 出力設定 ---
    output_group = parser.add_argument_group('出力設定')
    output_group.add_argument("-o", "--output", dest='output_file', default=None,
                              help="結果を出力するファイル名 (.h5)")
    output_group.add_argument("--num-modes", type=int, default=10, dest='num_modes',
                              help="計算結果から出力する共振モードの数 (周波数が低い方から)")

    # --- 引数をパース ---
    args = parser.parse_args(argv)

    # --- 出力ファイル名のデフォルト設定 ---
    if args.output_file is None:
        base_name = os.path.splitext(os.path.basename(args.mesh_file))[0]
        args.output_file = base_name + ".h5"
        print(f"情報: 出力ファイル名が指定されなかったため、デフォルト値 '{args.output_file}' を使用します。")
    elif not args.output_file.lower().endswith('.h5'):
        print(f"警告: 出力ファイル名に .h5 拡張子を追加します: '{args.output_file}.h5'")
        args.output_file += ".h5"

    # --- 位相リストの生成 ---
    args.phase_list = parse_phase_arg(args.phase)

    return args


# --- メインの処理 ---
if __name__ == "__main__":
    try:
        args = parse_calclation_args()
        print("\n--- 解析条件 ---")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("---------------")
    except argparse.ArgumentError:
        sys.exit(1)
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
