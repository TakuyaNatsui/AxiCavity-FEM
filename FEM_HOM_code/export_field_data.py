"""
HOM 解析結果 (raw .h5 / processed .h5) から、
矩形領域 (Area) または直線上 (Line) の電磁場データを HDF5 と TXT で出力する。

主な機能:
- 矩形領域指定 (--shape area --z-range --r-range --nz --nr)
- 直線指定     (--shape line --p1 z,r --p2 z,r --npts)
- 軸上専用簡易指定 (--shape axis --z-range --npts)
- 入力電力スケーリング (--scale-to-power P_target_W)
- 瞬時値出力        (--time-phase DEG, default 0)
- 任意の方位角次数 n とモード番号を指定

processed.h5 を入力に与えると、p_loss / U / Q がメタデータとして付与される。
"""

import argparse
import os
import sys
import numpy as np
import h5py

# パス設定 (ルートディレクトリも含める)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from FEM_HOM_code.field_calculator_hom import HOMFieldCalculator


# ------------------------------------------------------------------
# パース補助
# ------------------------------------------------------------------
def _parse_pair(s, name):
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError(f"{name} は 'z,r' 形式で指定してください: {s}")
    return float(parts[0]), float(parts[1])


def _parse_range(s, name):
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError(f"{name} は 'min,max' 形式で指定してください: {s}")
    return float(parts[0]), float(parts[1])


# ------------------------------------------------------------------
# 場の計算
# ------------------------------------------------------------------
_FIELDS = ('Ez', 'Er', 'E_theta', 'Hz', 'Hr', 'H_theta')


def _empty_arrays(shape, dtype=float):
    return {k: np.zeros(shape, dtype=dtype) for k in _FIELDS}


def _calc_point_with_mode(calc, z, r, mode_idx, theta_deg, return_complex):
    """1 点の場を返す。
    return_complex=True なら 2 スナップショット (theta=0, 90°) から
    複素振幅 F̃ = F(0) − j·F(π/2) を再構築する。
    return_complex=False なら指定 theta_deg での実瞬時値。
    None (領域外) は伝播。
    """
    if not return_complex:
        return calc.calculate_fields(z, r, mode_idx, theta_time=theta_deg)

    res0 = calc.calculate_fields(z, r, mode_idx, theta_time=0.0)
    if res0 is None:
        return None
    res90 = calc.calculate_fields(z, r, mode_idx, theta_time=90.0)
    if res90 is None:
        return None
    out = {}
    for k in _FIELDS:
        out[k] = complex(res0[k]) - 1j * complex(res90[k])
    out['E_abs'] = float(np.sqrt(abs(out['Ez']) ** 2
                                 + abs(out['Er']) ** 2
                                 + abs(out['E_theta']) ** 2))
    out['H_abs'] = float(np.sqrt(abs(out['Hz']) ** 2
                                 + abs(out['Hr']) ** 2
                                 + abs(out['H_theta']) ** 2))
    return out


def calc_area_fields(calc, mode_idx, z_range, r_range, nz, nr,
                     theta_deg=0.0, scale=1.0, return_complex=False):
    z_min, z_max = z_range
    r_min, r_max = r_range
    z_vec = np.linspace(z_min, z_max, nz)
    r_vec = np.linspace(r_min, r_max, nr)

    dtype = complex if return_complex else float
    arrays = _empty_arrays((nz, nr), dtype=dtype)
    E_abs = np.zeros((nz, nr), dtype=float)
    H_abs = np.zeros((nz, nr), dtype=float)
    mask = np.zeros((nz, nr), dtype=bool)

    for i in range(nz):
        for j in range(nr):
            res = _calc_point_with_mode(calc, z_vec[i], r_vec[j], mode_idx,
                                        theta_deg, return_complex)
            if res is None:
                continue
            for k in _FIELDS:
                arrays[k][i, j] = res[k] * scale
            E_abs[i, j] = res['E_abs'] * scale
            H_abs[i, j] = res['H_abs'] * scale
            mask[i, j] = True

    return {
        'z_vec': z_vec, 'r_vec': r_vec,
        'mask': mask,
        'E_abs': E_abs, 'H_abs': H_abs,
        **arrays,
    }


def calc_line_fields(calc, mode_idx, p1, p2, npts, theta_deg=0.0, scale=1.0,
                     return_complex=False):
    z1, r1 = p1
    z2, r2 = p2
    s_vec = np.linspace(0.0, 1.0, npts)
    z_pts = z1 + (z2 - z1) * s_vec
    r_pts = r1 + (r2 - r1) * s_vec
    length = float(np.hypot(z2 - z1, r2 - r1))
    distance = s_vec * length

    dtype = complex if return_complex else float
    arrays = _empty_arrays(npts, dtype=dtype)
    E_abs = np.zeros(npts, dtype=float)
    H_abs = np.zeros(npts, dtype=float)
    mask = np.zeros(npts, dtype=bool)

    for i in range(npts):
        res = _calc_point_with_mode(calc, z_pts[i], r_pts[i], mode_idx,
                                    theta_deg, return_complex)
        if res is None:
            continue
        for k in _FIELDS:
            arrays[k][i] = res[k] * scale
        E_abs[i] = res['E_abs'] * scale
        H_abs[i] = res['H_abs'] * scale
        mask[i] = True

    return {
        's': s_vec, 'distance': distance,
        'z': z_pts, 'r': r_pts,
        'mask': mask,
        'E_abs': E_abs, 'H_abs': H_abs,
        'length': length,
        **arrays,
    }


# ------------------------------------------------------------------
# HDF5 / TXT 書き出し
# ------------------------------------------------------------------
def _write_attrs(grp, meta):
    for k, v in meta.items():
        if v is None:
            grp.attrs[k] = 'N/A'
        elif isinstance(v, (str, bool, int, float, np.integer, np.floating)):
            grp.attrs[k] = v
        else:
            grp.attrs[k] = str(v)


def write_h5(out_path, data, meta, shape):
    with h5py.File(out_path, 'w') as f:
        f.attrs['shape'] = shape
        f.attrs['code_type'] = 'HOM'
        _write_attrs(f, meta)
        if shape == 'area':
            keys = ('z_vec', 'r_vec', 'mask',
                    'Ez', 'Er', 'E_theta', 'E_abs',
                    'Hz', 'Hr', 'H_theta', 'H_abs')
        else:
            keys = ('s', 'distance', 'z', 'r', 'mask',
                    'Ez', 'Er', 'E_theta', 'E_abs',
                    'Hz', 'Hr', 'H_theta', 'H_abs')
        for k in keys:
            f.create_dataset(k, data=data[k])


def _fmt_cx(v):
    return f"{v.real:.6e} {v.imag:.6e}"


def write_txt_area(out_path, data, meta, is_complex):
    z_vec = data['z_vec']
    r_vec = data['r_vec']
    nz = len(z_vec)
    nr = len(r_vec)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# HOM Field Map (Area)\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        f.write(f"# nz = {nz}\n")
        f.write(f"# nr = {nr}\n")
        if is_complex:
            f.write("# Columns: z[m] r[m] "
                    "Re(Ez) Im(Ez) Re(Er) Im(Er) Re(E_theta) Im(E_theta) |E| "
                    "Re(Hz) Im(Hz) Re(Hr) Im(Hr) Re(H_theta) Im(H_theta) |H| "
                    "mask\n")
        else:
            f.write("# Columns: z[m] r[m] "
                    "Ez[V/m] Er[V/m] E_theta[V/m] |E|[V/m] "
                    "Hz[A/m] Hr[A/m] H_theta[A/m] |H|[A/m] mask\n")
        for i in range(nz):
            for j in range(nr):
                base = f"{z_vec[i]:.9e} {r_vec[j]:.9e} "
                if is_complex:
                    f.write(base
                            + f"{_fmt_cx(data['Ez'][i, j])} "
                            + f"{_fmt_cx(data['Er'][i, j])} "
                            + f"{_fmt_cx(data['E_theta'][i, j])} "
                            + f"{data['E_abs'][i, j]:.6e} "
                            + f"{_fmt_cx(data['Hz'][i, j])} "
                            + f"{_fmt_cx(data['Hr'][i, j])} "
                            + f"{_fmt_cx(data['H_theta'][i, j])} "
                            + f"{data['H_abs'][i, j]:.6e} "
                            + f"{int(data['mask'][i, j])}\n")
                else:
                    f.write(base
                            + f"{data['Ez'][i, j]:.6e} "
                            + f"{data['Er'][i, j]:.6e} "
                            + f"{data['E_theta'][i, j]:.6e} "
                            + f"{data['E_abs'][i, j]:.6e} "
                            + f"{data['Hz'][i, j]:.6e} "
                            + f"{data['Hr'][i, j]:.6e} "
                            + f"{data['H_theta'][i, j]:.6e} "
                            + f"{data['H_abs'][i, j]:.6e} "
                            + f"{int(data['mask'][i, j])}\n")


def write_txt_line(out_path, data, meta, is_complex):
    npts = len(data['s'])
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# HOM Field on Line\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        f.write(f"# npts = {npts}\n")
        f.write(f"# length = {data['length']:.9e} m\n")
        if is_complex:
            f.write("# Columns: s distance[m] z[m] r[m] "
                    "Re(Ez) Im(Ez) Re(Er) Im(Er) Re(E_theta) Im(E_theta) |E| "
                    "Re(Hz) Im(Hz) Re(Hr) Im(Hr) Re(H_theta) Im(H_theta) |H| "
                    "mask\n")
        else:
            f.write("# Columns: s distance[m] z[m] r[m] "
                    "Ez[V/m] Er[V/m] E_theta[V/m] |E|[V/m] "
                    "Hz[A/m] Hr[A/m] H_theta[A/m] |H|[A/m] mask\n")
        for i in range(npts):
            base = (f"{data['s'][i]:.6f} {data['distance'][i]:.9e} "
                    f"{data['z'][i]:.9e} {data['r'][i]:.9e} ")
            if is_complex:
                f.write(base
                        + f"{_fmt_cx(data['Ez'][i])} "
                        + f"{_fmt_cx(data['Er'][i])} "
                        + f"{_fmt_cx(data['E_theta'][i])} "
                        + f"{data['E_abs'][i]:.6e} "
                        + f"{_fmt_cx(data['Hz'][i])} "
                        + f"{_fmt_cx(data['Hr'][i])} "
                        + f"{_fmt_cx(data['H_theta'][i])} "
                        + f"{data['H_abs'][i]:.6e} "
                        + f"{int(data['mask'][i])}\n")
            else:
                f.write(base
                        + f"{data['Ez'][i]:.6e} "
                        + f"{data['Er'][i]:.6e} "
                        + f"{data['E_theta'][i]:.6e} "
                        + f"{data['E_abs'][i]:.6e} "
                        + f"{data['Hz'][i]:.6e} "
                        + f"{data['Hr'][i]:.6e} "
                        + f"{data['H_theta'][i]:.6e} "
                        + f"{data['H_abs'][i]:.6e} "
                        + f"{int(data['mask'][i])}\n")


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------
def run_export(args):
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    calc = HOMFieldCalculator(args.input)
    if not calc.available_ns:
        print("Error: 結果ファイルから方位角次数 n が見つかりません。")
        return 1

    if args.n not in calc.available_ns:
        print(f"Error: n={args.n} は結果ファイルに含まれていません。"
              f"利用可能: {calc.available_ns}")
        return 1

    if not calc.set_n_and_phase(args.n, args.phase):
        print(f"Error: n={args.n}, phase={args.phase} の読み込みに失敗しました。")
        return 1

    # phase 引数が phase_shifts に無い場合の警告
    if (calc.analysis_type == 'traveling' and len(calc.phase_shifts) > 0
            and args.phase not in calc.phase_shifts):
        print(f"Warning: 指定の phase ({args.phase}) はファイル内に無いため、"
              f"自動で {calc.phase_shift} を使用しました。"
              f"利用可能: {calc.phase_shifts}")

    # モード番号確認
    real_mode_indices = [m[0] for m in calc.modes]
    if args.mode not in real_mode_indices:
        print(f"Error: mode={args.mode} は読み込まれた結果に存在しません。"
              f"利用可能: {real_mode_indices}")
        return 1

    calc.load_mode_data(args.mode)

    # エンジニアリング parameters
    eng = calc.get_mode_parameters(args.n, args.mode)
    p_loss_curr = eng['p_loss'] if eng is not None else None

    # スケーリング
    scale = 1.0
    if args.scale_to_power is not None:
        if p_loss_curr is None or p_loss_curr <= 0:
            print("Error: --scale-to-power 指定には processed.h5 (p_loss > 0) "
                  "が必要です。")
            return 1
        scale = float(np.sqrt(args.scale_to_power / p_loss_curr))

    if eng is not None:
        p_loss_out = eng['p_loss'] * (scale ** 2)
        u_out = eng['stored_energy'] * (scale ** 2)
        q_out = eng['q_factor']
    else:
        p_loss_out = None
        u_out = None
        q_out = None

    # 進行波の出力モード:
    #   default → 複素振幅 (Re/Im) を 2 スナップショットから再構築
    #   --instant 指定時 → 指定 time-phase での実瞬時値
    is_traveling = (calc.analysis_type != 'standing')
    return_complex = is_traveling and (not args.instant)

    # 形状ごとに分岐
    if args.shape == 'area':
        z_range = args.z_range or (float(np.min(calc.vertices[:, 0])),
                                   float(np.max(calc.vertices[:, 0])))
        r_range = args.r_range or (float(np.min(calc.vertices[:, 1])),
                                   float(np.max(calc.vertices[:, 1])))
        if args.z_range is None:
            print(f"Info: z-range auto = [{z_range[0]:.6g}, {z_range[1]:.6g}] m")
        if args.r_range is None:
            print(f"Info: r-range auto = [{r_range[0]:.6g}, {r_range[1]:.6g}] m")

        data = calc_area_fields(calc, args.mode, z_range, r_range,
                                args.nz, args.nr,
                                theta_deg=args.time_phase, scale=scale,
                                return_complex=return_complex)
    elif args.shape in ('line', 'axis'):
        if args.shape == 'axis':
            z_min = (args.z_range[0] if args.z_range else
                     float(np.min(calc.vertices[:, 0])))
            z_max = (args.z_range[1] if args.z_range else
                     float(np.max(calc.vertices[:, 0])))
            p1 = (z_min, 0.0)
            p2 = (z_max, 0.0)
        else:
            if args.p1 is None or args.p2 is None:
                print("Error: --shape line では --p1 と --p2 が必要です。")
                return 1
            p1 = args.p1
            p2 = args.p2
        data = calc_line_fields(calc, args.mode, p1, p2, args.npts,
                                theta_deg=args.time_phase, scale=scale,
                                return_complex=return_complex)
    else:
        print(f"Error: unknown shape: {args.shape}")
        return 1

    # メタデータ
    meta = {
        'input_file': os.path.abspath(args.input),
        'azimuthal_n': int(args.n),
        'mode_index': int(args.mode),
        'frequency_GHz': float(calc.freq_GHz),
        'analysis_type': str(calc.analysis_type),
        'phase_shift_deg': float(calc.phase_shift),
        'time_phase_deg': float(args.time_phase),
        'scale_factor': float(scale),
        'target_power_W': (float(args.scale_to_power)
                           if args.scale_to_power is not None else 'N/A'),
        'p_loss_original_W': (float(p_loss_curr)
                              if p_loss_curr is not None else 'N/A'),
        'p_loss_W': (float(p_loss_out) if p_loss_out is not None else 'N/A'),
        'stored_energy_J': (float(u_out) if u_out is not None else 'N/A'),
        'q_factor': (float(q_out) if q_out is not None else 'N/A'),
        'is_complex': bool(return_complex),
        'shape': args.shape,
    }
    if args.shape == 'area':
        meta['z_min'] = float(data['z_vec'][0])
        meta['z_max'] = float(data['z_vec'][-1])
        meta['r_min'] = float(data['r_vec'][0])
        meta['r_max'] = float(data['r_vec'][-1])
        meta['nz'] = int(args.nz)
        meta['nr'] = int(args.nr)
    else:
        meta['p1_z'] = float(data['z'][0])
        meta['p1_r'] = float(data['r'][0])
        meta['p2_z'] = float(data['z'][-1])
        meta['p2_r'] = float(data['r'][-1])
        meta['npts'] = int(args.npts)
        meta['line_length_m'] = float(data['length'])

    # 出力
    base, _ = os.path.splitext(args.output)
    h5_path = base + '.h5'
    txt_path = base + '.txt'

    do_h5 = args.format in ('h5', 'both')
    do_txt = args.format in ('txt', 'both')

    if do_h5:
        write_h5(h5_path, data, meta, args.shape)
    if do_txt:
        if args.shape == 'area':
            write_txt_area(txt_path, data, meta, return_complex)
        else:
            write_txt_line(txt_path, data, meta, return_complex)

    print(f"\n=== Export complete ===")
    if do_h5:
        print(f"  HDF5 : {h5_path}")
    if do_txt:
        print(f"  TEXT : {txt_path}")
    print(f"  n = {args.n}, mode = {args.mode}, "
          f"f = {calc.freq_GHz:.6f} GHz")
    if p_loss_curr is not None:
        print(f"  P_loss (original)   : {p_loss_curr:.4e} W")
    if args.scale_to_power is not None:
        print(f"  P_loss (target)     : {args.scale_to_power:.4e} W")
        print(f"  Field scale factor  : {scale:.6e}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Export HOM electromagnetic field data (area or line).')
    parser.add_argument('-i', '--input', required=True,
                        help='Input HDF5 result (raw or processed)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output base path (拡張子は自動で .h5 / .txt)')
    parser.add_argument('--n', type=int, default=0,
                        help='Azimuthal mode order n')
    parser.add_argument('-m', '--mode', type=int, default=0,
                        help='Mode index (within the chosen n)')
    parser.add_argument('--shape', choices=['area', 'line', 'axis'],
                        default='area',
                        help='Export shape')
    parser.add_argument('--z-range', type=lambda s: _parse_range(s, 'z-range'),
                        default=None,
                        help='Z range "zmin,zmax" [m]')
    parser.add_argument('--r-range', type=lambda s: _parse_range(s, 'r-range'),
                        default=None,
                        help='R range "rmin,rmax" [m] (area のみ)')
    parser.add_argument('--nz', type=int, default=200)
    parser.add_argument('--nr', type=int, default=100)
    parser.add_argument('--p1', type=lambda s: _parse_pair(s, 'p1'),
                        default=None,
                        help='Line start "z,r" [m] (shape=line)')
    parser.add_argument('--p2', type=lambda s: _parse_pair(s, 'p2'),
                        default=None,
                        help='Line end "z,r" [m] (shape=line)')
    parser.add_argument('--npts', type=int, default=500)
    parser.add_argument('--phase', type=float, default=0.0,
                        help='Phase shift [deg] for traveling wave')
    parser.add_argument('--time-phase', type=float, default=0.0,
                        help='Time phase [deg] for instantaneous output')
    parser.add_argument('--scale-to-power', type=float, default=None,
                        help='Target wall loss [W]; '
                             'fields are rescaled so P_loss matches '
                             '(processed.h5 が必要)')
    parser.add_argument('--instant', action='store_true',
                        help='進行波で実瞬時値を出力する '
                             '(default: 進行波は複素振幅 Re/Im を 2 スナップショットから再構築)')
    parser.add_argument('--format', choices=['h5', 'txt', 'both'],
                        default='both',
                        help='出力フォーマット (default: both)')
    args = parser.parse_args()

    return run_export(args)


if __name__ == '__main__':
    sys.exit(main())
