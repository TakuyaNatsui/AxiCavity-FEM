"""plot サブコマンド: 場マップを PNG で可視化する."""

from __future__ import annotations

import os

from ..shared.hdf5_io import is_v2, load_v1_legacy, read_results


def run(args) -> int:
    """plot サブコマンドの本体."""
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1

    data = read_results(args.input_file) if is_v2(args.input_file) \
        else load_v1_legacy(args.input_file)
    solver_type = args.type or data["solver_type"]

    output = args.output_file
    if not output:
        base = os.path.splitext(args.input_file)[0]
        output = f"{base}_{solver_type}_mode{args.mode}.png"

    show_mesh = not args.no_mesh
    try:
        if solver_type == "tm0":
            from ..reports.plot_tm0 import plot_tm0_mode
            plot_tm0_mode(data, args.mode, output, phase=args.phase,
                          time_phase=args.time_phase, show_mesh=show_mesh,
                          dpi=args.dpi)
        else:
            from ..reports.plot_hom import plot_hom_mode
            plot_hom_mode(data, args.mode, output, n=args.n, phase=args.phase,
                          time_phase=args.time_phase, show_mesh=show_mesh,
                          dpi=args.dpi)
    except (ValueError, IndexError) as e:
        print(f"エラー: {e}")
        return 1

    print(f"[plot] type={solver_type} mode={args.mode} -> {output}")
    return 0
