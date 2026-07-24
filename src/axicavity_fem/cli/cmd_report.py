"""report サブコマンド: HTML レポートを生成する."""

from __future__ import annotations

import os


def run(args) -> int:
    """report サブコマンドの本体."""
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1

    from ..reports.report_builder import build_report

    try:
        index = build_report(
            args.input_file, args.output_dir, solver_type=args.type,
            dpi=args.dpi, time_phase=args.time_phase,
            show_mesh=not args.no_mesh, animate=args.animate)
    except (ValueError, IndexError) as e:
        print(f"エラー: {e}")
        return 1

    print(f"[report] 生成完了: {index}")
    return 0
