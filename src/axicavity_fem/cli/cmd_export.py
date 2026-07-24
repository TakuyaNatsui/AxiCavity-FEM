"""export サブコマンド: 電磁場マップを HDF5/TXT で出力する."""

from __future__ import annotations

import os


def _parse_pair(s):
    if s is None:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"'z,r' または 'min,max' 形式で指定してください: {s}")
    return float(parts[0]), float(parts[1])


def run(args) -> int:
    """export サブコマンドの本体."""
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1

    from ..reports.export_fields import export_fields

    try:
        data = export_fields(
            args.input_file, args.output_file,
            solver_type=args.type, mode=args.mode, n=args.n,
            phase=args.phase, shape=args.shape,
            z_range=_parse_pair(args.z_range),
            r_range=_parse_pair(args.r_range),
            nz=args.nz, nr=args.nr,
            p1=_parse_pair(args.p1), p2=_parse_pair(args.p2),
            npts=args.npts, time_phase=args.time_phase,
            scale=args.scale, scale_to_power=args.scale_to_power,
            instant=args.instant, fmt=args.fmt,
        )
    except (ValueError, IndexError) as e:
        print(f"エラー: {e}")
        return 1

    meta = data["_meta"]
    base, _ = os.path.splitext(args.output_file)
    print(f"[export] type={meta['solver_type']} shape={meta['shape']} "
          f"mode={meta['mode_index']} f={meta['frequency_GHz']:.6f} GHz "
          f"scale={meta['scale_factor']:.4e}")
    if args.fmt in ("h5", "both"):
        print(f"  HDF5: {base}.h5")
    if args.fmt in ("txt", "both"):
        print(f"  TEXT: {base}.txt")
    return 0
