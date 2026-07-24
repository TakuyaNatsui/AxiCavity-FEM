"""info サブコマンド: HDF5 ファイルの構造をダンプする."""

from __future__ import annotations

import os

from ..shared.hdf5_io import dump_info, is_v2


def run(args) -> int:
    """info サブコマンドの本体."""
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1

    schema = "v2" if is_v2(args.input_file) else "v1 (legacy)"
    print(f"[info] {args.input_file}  schema: {schema}")
    print(dump_info(args.input_file))
    return 0
