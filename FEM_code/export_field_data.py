"""
TM0 解析結果 (raw .h5 / processed .h5) から、
矩形領域 (Area) または直線上 (Line) の電磁場データを HDF5 と TXT で出力する。

主な機能:
- 矩形領域指定 (--shape area --z-range --r-range --nz --nr)
- 直線指定     (--shape line --p1 z,r --p2 z,r --npts)
- 軸上専用簡易指定 (--shape axis --z-range --npts)
- 入力電力スケーリング (--scale-to-power P_target_W)
- 瞬時値出力        (--time-phase DEG, default 0)

processed.h5 を入力に与えると、p_loss / U / Q がメタデータとして付与される。
"""

import argparse
import os
import sys
import numpy as np
import h5py

# 自分自身のディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from field_calculator import FieldCalculator


# ------------------------------------------------------------------
# パース補助
# ------------------------------------------------------------------
def _parse_pair(s, name):
    """'z,r' を (float, float) に変換する。"""
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError(f"{name} は 'z,r' 形式で指定してください: {s}")
    return float(parts[0]), float(parts[1])


def _parse_range(s, name):
    """'min,max' を (float, float) に変換する。"""
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError(f"{name} は 'min,max' 形式で指定してください: {s}")
    return float(parts[0]), float(parts[1])


# ------------------------------------------------------------------
# Engineering parameter 取得 (processed.h5)
# ------------------------------------------------------------------
def get_engineering_params(h5_path, mode_index, phase_shift):
    """processed.h5 から該当モードの p_loss, U, Q を取得する。
    raw.h5 の場合は None を返す。
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            candidates = [
                f'post_processed/phase_{phase_shift}/engineering_parameters',
                f'results/phase_{phase_shift}/engineering_parameters',
                'engineering_parameters',
            ]
            for path in candidates:
                if path in f and 'p_loss' in f[path]:
                    grp = f[path]
                    n_modes = len(grp['p_loss'])
                    if mode_index >= n_modes:
                        return None
                    return {
                        'p_loss': float(grp['p_loss'][mode_index]),
                        'stored_energy': float(grp['stored_energy'][mode_index]),
                        'q_factor': float(grp['q_factor'][mode_index]),
                    }
    except Exception:
        return None
    return None


# ------------------------------------------------------------------
# 場の計算 (Area / Line)
# ------------------------------------------------------------------
def _calc_point(calc, z, r, mode_index, theta_rad, return_complex):
    res = calc.calculate_fields(z, r, mode_index, theta=theta_rad,
                                return_complex=return_complex)
    return res


def calc_area_fields(calc, mode_index, z_range, r_range, nz, nr,
                     theta_deg=0.0, scale=1.0, return_complex=False):
    z_min, z_max = z_range
    r_min, r_max = r_range
    z_vec = np.linspace(z_min, z_max, nz)
    r_vec = np.linspace(r_min, r_max, nr)

    dtype = complex if return_complex else float
    H_theta = np.zeros((nz, nr), dtype=dtype)
    Ez = np.zeros((nz, nr), dtype=dtype)
    Er = np.zeros((nz, nr), dtype=dtype)
    E_abs = np.zeros((nz, nr), dtype=float)
    mask = np.zeros((nz, nr), dtype=bool)

    theta_rad = np.deg2rad(theta_deg)
    for i in range(nz):
        for j in range(nr):
            res = _calc_point(calc, z_vec[i], r_vec[j], mode_index,
                              theta_rad, return_complex)
            if res is None:
                continue
            H_theta[i, j] = res['H_theta'] * scale
            Ez[i, j] = res['Ez'] * scale
            Er[i, j] = res['Er'] * scale
            E_abs[i, j] = float(res['E_abs']) * scale
            mask[i, j] = True

    return {
        'z_vec': z_vec, 'r_vec': r_vec,
        'H_theta': H_theta, 'Ez': Ez, 'Er': Er,
        'E_abs': E_abs, 'mask': mask,
    }


def calc_line_fields(calc, mode_index, p1, p2, npts,
                     theta_deg=0.0, scale=1.0, return_complex=False):
    z1, r1 = p1
    z2, r2 = p2
    s_vec = np.linspace(0.0, 1.0, npts)
    z_pts = z1 + (z2 - z1) * s_vec
    r_pts = r1 + (r2 - r1) * s_vec
    length = float(np.hypot(z2 - z1, r2 - r1))
    distance = s_vec * length

    dtype = complex if return_complex else float
    H_theta = np.zeros(npts, dtype=dtype)
    Ez = np.zeros(npts, dtype=dtype)
    Er = np.zeros(npts, dtype=dtype)
    E_abs = np.zeros(npts, dtype=float)
    mask = np.zeros(npts, dtype=bool)

    theta_rad = np.deg2rad(theta_deg)
    for i in range(npts):
        res = _calc_point(calc, z_pts[i], r_pts[i], mode_index,
                          theta_rad, return_complex)
        if res is None:
            continue
        H_theta[i] = res['H_theta'] * scale
        Ez[i] = res['Ez'] * scale
        Er[i] = res['Er'] * scale
        E_abs[i] = float(res['E_abs']) * scale
        mask[i] = True

    return {
        's': s_vec, 'distance': distance,
        'z': z_pts, 'r': r_pts,
        'H_theta': H_theta, 'Ez': Ez, 'Er': Er,
        'E_abs': E_abs, 'mask': mask,
        'length': length,
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
        f.attrs['code_type'] = 'TM0'
        _write_attrs(f, meta)
        if shape == 'area':
            for k in ('z_vec', 'r_vec', 'H_theta', 'Ez', 'Er', 'E_abs', 'mask'):
                f.create_dataset(k, data=data[k])
        else:  # line / axis
            for k in ('s', 'distance', 'z', 'r',
                      'H_theta', 'Ez', 'Er', 'E_abs', 'mask'):
                f.create_dataset(k, data=data[k])


def _fmt_complex(v):
    return f"{v.real:.6e} {v.imag:.6e}"


def write_txt_area(out_path, data, meta, is_complex):
    z_vec = data['z_vec']
    r_vec = data['r_vec']
    nz = len(z_vec)
    nr = len(r_vec)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# TM0 Field Map (Area)\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        f.write(f"# nz = {nz}\n")
        f.write(f"# nr = {nr}\n")
        if is_complex:
            f.write("# Columns: z[m] r[m] Re(H_theta)[A/m] Im(H_theta) "
                    "Re(Ez)[V/m] Im(Ez) Re(Er)[V/m] Im(Er) |E|[V/m] mask\n")
        else:
            f.write("# Columns: z[m] r[m] H_theta[A/m] Ez[V/m] Er[V/m] "
                    "|E|[V/m] mask\n")
        for i in range(nz):
            for j in range(nr):
                z = z_vec[i]
                r = r_vec[j]
                ht = data['H_theta'][i, j]
                ez = data['Ez'][i, j]
                er = data['Er'][i, j]
                ea = data['E_abs'][i, j]
                ms = int(data['mask'][i, j])
                if is_complex:
                    f.write(f"{z:.9e} {r:.9e} "
                            f"{_fmt_complex(ht)} "
                            f"{_fmt_complex(ez)} "
                            f"{_fmt_complex(er)} "
                            f"{ea:.6e} {ms}\n")
                else:
                    f.write(f"{z:.9e} {r:.9e} "
                            f"{ht:.6e} {ez:.6e} {er:.6e} "
                            f"{ea:.6e} {ms}\n")


def write_txt_line(out_path, data, meta, is_complex):
    npts = len(data['s'])
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# TM0 Field on Line\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        f.write(f"# npts = {npts}\n")
        f.write(f"# length = {data['length']:.9e} m\n")
        if is_complex:
            f.write("# Columns: s distance[m] z[m] r[m] "
                    "Re(H_theta)[A/m] Im(H_theta) "
                    "Re(Ez)[V/m] Im(Ez) Re(Er)[V/m] Im(Er) |E|[V/m] mask\n")
        else:
            f.write("# Columns: s distance[m] z[m] r[m] "
                    "H_theta[A/m] Ez[V/m] Er[V/m] |E|[V/m] mask\n")
        for i in range(npts):
            ht = data['H_theta'][i]
            ez = data['Ez'][i]
            er = data['Er'][i]
            if is_complex:
                f.write(f"{data['s'][i]:.6f} {data['distance'][i]:.9e} "
                        f"{data['z'][i]:.9e} {data['r'][i]:.9e} "
                        f"{_fmt_complex(ht)} "
                        f"{_fmt_complex(ez)} "
                        f"{_fmt_complex(er)} "
                        f"{data['E_abs'][i]:.6e} {int(data['mask'][i])}\n")
            else:
                f.write(f"{data['s'][i]:.6f} {data['distance'][i]:.9e} "
                        f"{data['z'][i]:.9e} {data['r'][i]:.9e} "
                        f"{ht:.6e} {ez:.6e} {er:.6e} "
                        f"{data['E_abs'][i]:.6e} {int(data['mask'][i])}\n")


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------
def run_export(args):
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    calc = FieldCalculator(args.input)

    # 位相切替
    if args.phase is not None and len(calc.phase_shifts) > 0:
        if args.phase not in list(calc.phase_shifts):
            print(f"Warning: Phase {args.phase} not in result file. "
                  f"Available: {list(calc.phase_shifts)}")
        else:
            calc.set_phase(args.phase)

    if args.mode >= len(calc.frequencies):
        print(f"Error: mode {args.mode} out of range "
              f"(have {len(calc.frequencies)} modes)")
        return 1

    # メタ取得
    eng = get_engineering_params(args.input, args.mode, calc.phase_shift)
    p_loss_curr = eng['p_loss'] if eng is not None else None

    # スケーリング
    scale = 1.0
    if args.scale_to_power is not None:
        if p_loss_curr is None or p_loss_curr <= 0:
            print("Error: --scale-to-power 指定には processed.h5 (p_loss > 0) "
                  "が必要です。")
            return 1
        scale = float(np.sqrt(args.scale_to_power / p_loss_curr))

    # 出力場のスケール後 p_loss/U
    if eng is not None:
        p_loss_out = eng['p_loss'] * (scale ** 2)
        u_out = eng['stored_energy'] * (scale ** 2)
        q_out = eng['q_factor']
    else:
        p_loss_out = None
        u_out = None
        q_out = None

    # 進行波の出力モード:
    #   default → 複素振幅 (Re/Im) を出力
    #   --instant 指定時 → 指定 time-phase での実瞬時値
    # 定在波は常に実数値 (peak)
    is_traveling = (calc.analysis_type != 'standing')
    return_complex = is_traveling and (not args.instant)

    # 形状ごとに分岐
    if args.shape == 'area':
        z_range = args.z_range or (np.min(calc.nodes[:, 0]),
                                   np.max(calc.nodes[:, 0]))
        r_range = args.r_range or (np.min(calc.nodes[:, 1]),
                                   np.max(calc.nodes[:, 1]))
        if args.z_range is None:
            print(f"Info: z-range auto = [{z_range[0]:.6g}, {z_range[1]:.6g}] m")
        if args.r_range is None:
            print(f"Info: r-range auto = [{r_range[0]:.6g}, {r_range[1]:.6g}] m")

        data = calc_area_fields(calc, args.mode, z_range, r_range,
                                args.nz, args.nr,
                                theta_deg=args.time_phase,
                                scale=scale,
                                return_complex=return_complex)
    elif args.shape in ('line', 'axis'):
        if args.shape == 'axis':
            z_min = args.z_range[0] if args.z_range else float(np.min(calc.nodes[:, 0]))
            z_max = args.z_range[1] if args.z_range else float(np.max(calc.nodes[:, 0]))
            p1 = (z_min, 0.0)
            p2 = (z_max, 0.0)
        else:
            if args.p1 is None or args.p2 is None:
                print("Error: --shape line では --p1 と --p2 が必要です。")
                return 1
            p1 = args.p1
            p2 = args.p2
        data = calc_line_fields(calc, args.mode, p1, p2, args.npts,
                                theta_deg=args.time_phase,
                                scale=scale,
                                return_complex=return_complex)
    else:
        print(f"Error: unknown shape: {args.shape}")
        return 1

    # メタデータ
    meta = {
        'input_file': os.path.abspath(args.input),
        'mode_index': int(args.mode),
        'frequency_GHz': float(calc.frequencies[args.mode]),
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
    print(f"  Mode : {args.mode}, f = {calc.frequencies[args.mode]:.6f} GHz")
    if p_loss_curr is not None:
        print(f"  P_loss (original)   : {p_loss_curr:.4e} W")
    if args.scale_to_power is not None:
        print(f"  P_loss (target)     : {args.scale_to_power:.4e} W")
        print(f"  Field scale factor  : {scale:.6e}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Export TM0 electromagnetic field data (area or line).')
    parser.add_argument('-i', '--input', required=True,
                        help='Input HDF5 result (raw or processed)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output base path (拡張子は自動で .h5 / .txt)')
    parser.add_argument('-m', '--mode', type=int, default=0,
                        help='Mode index')
    parser.add_argument('--shape', choices=['area', 'line', 'axis'],
                        default='area',
                        help='Export shape: area / line / axis (line on r=0)')
    parser.add_argument('--z-range', type=lambda s: _parse_range(s, 'z-range'),
                        default=None,
                        help='Z range "zmin,zmax" [m] (省略時は領域全体)')
    parser.add_argument('--r-range', type=lambda s: _parse_range(s, 'r-range'),
                        default=None,
                        help='R range "rmin,rmax" [m] (省略時は領域全体, area のみ)')
    parser.add_argument('--nz', type=int, default=200,
                        help='Number of grid points along z (area)')
    parser.add_argument('--nr', type=int, default=100,
                        help='Number of grid points along r (area)')
    parser.add_argument('--p1', type=lambda s: _parse_pair(s, 'p1'),
                        default=None,
                        help='Line start "z,r" [m] (shape=line)')
    parser.add_argument('--p2', type=lambda s: _parse_pair(s, 'p2'),
                        default=None,
                        help='Line end "z,r" [m] (shape=line)')
    parser.add_argument('--npts', type=int, default=500,
                        help='Number of sampling points along line')
    parser.add_argument('--phase', type=float, default=None,
                        help='Phase shift [deg] (進行波・複数位相結果から選択)')
    parser.add_argument('--time-phase', type=float, default=0.0,
                        help='Time phase [deg] for instantaneous output (default 0)')
    parser.add_argument('--scale-to-power', type=float, default=None,
                        help='Target wall loss [W]; '
                             'fields are rescaled so P_loss matches this value '
                             '(processed.h5 が必要)')
    parser.add_argument('--instant', action='store_true',
                        help='進行波で実瞬時値を出力する '
                             '(default: 進行波は複素振幅 Re/Im を出力)')
    parser.add_argument('--format', choices=['h5', 'txt', 'both'],
                        default='both',
                        help='出力フォーマット (default: both)')
    args = parser.parse_args()

    return run_export(args)


if __name__ == '__main__':
    sys.exit(main())
