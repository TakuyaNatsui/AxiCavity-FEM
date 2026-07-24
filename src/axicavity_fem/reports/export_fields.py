"""電磁場マップを HDF5/TXT で出力（TM0/HOM 統合, --type で分岐）.

ver1 ``FEM_code/export_field_data.py`` / ``FEM_HOM_code/export_field_data.py`` を
ver2 の場再構成（:mod:`axicavity_fem.fem_tm0.field_recon` /
:mod:`axicavity_fem.fem_hom.field_recon`）の上に統合・移植したもの。

形状 (shape):
    - ``area``: 矩形領域 (nz×nr) のグリッド
    - ``line``: 2 点 p1→p2 を結ぶ直線 (npts 点)
    - ``axis``: 対称軸 r=0 上の直線 (npts 点)

定在波は実数値（peak）。進行波は既定で複素振幅（TM0 のみ Re/Im 分離出力）、
``instant=True`` で指定 ``time_phase`` の瞬時実数値。HOM は瞬時実数値のみ。
"""

from __future__ import annotations

import os

import numpy as np

from ..shared.hdf5_io import is_v2, load_v1_legacy, read_results

_TM0_FIELDS = ["H_theta", "Ez", "Er", "E_abs"]
_HOM_FIELDS = ["Ez", "Er", "E_theta", "E_abs", "Hz", "Hr", "H_theta", "H_abs"]
_ABS_FIELDS = {"E_abs", "H_abs"}


# ---------------------------------------------------------------------------
# 入力 H5 → 場サンプリング用クロージャ
# ---------------------------------------------------------------------------
def _select_modeset(data, solver_type, n, phase):
    """results_by_n から (modeset, analysis_type, n, phase) を決定する."""
    results = data["results_by_n"]
    if n is None:
        n = 0 if solver_type == "tm0" else sorted(results.keys())[0]
    if n not in results:
        raise ValueError(f"n={n} の結果がありません（利用可能: {sorted(results)}）")
    rec = results[n]
    if "standing" in rec:
        return rec["standing"], "standing", n, None
    trav = rec["traveling"]
    if phase is None:
        phase = sorted(trav.keys())[0]
    if phase not in trav:
        raise ValueError(f"位相 {phase} がありません（利用可能: {sorted(trav)}）")
    return trav[phase], "traveling", n, phase


def _build_field_fn(data, solver_type, mode, n, analysis_type, modeset,
                    time_phase, return_complex):
    """点 (z, r) の場 dict を返すクロージャと、場名リストを返す."""
    mesh = data["mesh"]
    eigvec = modeset["eigenvectors"][mode]
    freq = float(modeset["frequencies"][mode])

    if solver_type == "tm0":
        from ..fem_tm0.field_recon import TM0FieldReconstructor
        recon = TM0FieldReconstructor(
            mesh["vertices"], mesh["simplices"], mesh["elem_order"],
            analysis_type,
            eps_r_per_element=mesh.get("eps_r_per_element"))
        theta_rad = np.deg2rad(time_phase)

        def field_at(z, r):
            return recon.calculate_fields(eigvec, freq, z, r,
                                          theta=theta_rad,
                                          return_complex=return_complex)
        return field_at, _TM0_FIELDS

    from ..fem_hom.field_recon import HOMFieldReconstructor
    recon = HOMFieldReconstructor(
        mesh["simplices"], mesh["vertices"], mesh["edge_index_map"],
        mesh["elem_order"], n, analysis_type)
    recon.load_mode(eigvec, freq)

    def field_at(z, r):
        return recon.calculate_fields(z, r, theta_time=time_phase)
    return field_at, _HOM_FIELDS


# ---------------------------------------------------------------------------
# サンプリング
# ---------------------------------------------------------------------------
def _empty(shape_tuple, field, return_complex):
    dtype = (complex if (return_complex and field not in _ABS_FIELDS) else float)
    return np.zeros(shape_tuple, dtype=dtype)


def sample_area(field_at, fields, z_range, r_range, nz, nr, scale,
                return_complex):
    z_vec = np.linspace(z_range[0], z_range[1], nz)
    r_vec = np.linspace(r_range[0], r_range[1], nr)
    out = {f: _empty((nz, nr), f, return_complex) for f in fields}
    mask = np.zeros((nz, nr), dtype=bool)
    for i in range(nz):
        for j in range(nr):
            res = field_at(z_vec[i], r_vec[j])
            if res is None:
                continue
            for f in fields:
                out[f][i, j] = res[f] * scale
            mask[i, j] = True
    out.update(z_vec=z_vec, r_vec=r_vec, mask=mask)
    return out


def sample_line(field_at, fields, p1, p2, npts, scale, return_complex):
    s_vec = np.linspace(0.0, 1.0, npts)
    z_pts = p1[0] + (p2[0] - p1[0]) * s_vec
    r_pts = p1[1] + (p2[1] - p1[1]) * s_vec
    length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    out = {f: _empty(npts, f, return_complex) for f in fields}
    mask = np.zeros(npts, dtype=bool)
    for i in range(npts):
        res = field_at(z_pts[i], r_pts[i])
        if res is None:
            continue
        for f in fields:
            out[f][i] = res[f] * scale
        mask[i] = True
    out.update(s=s_vec, distance=s_vec * length, z=z_pts, r=r_pts,
               mask=mask, length=length)
    return out


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def _write_h5(out_path, data, meta, shape, fields):
    import h5py
    with h5py.File(out_path, "w") as f:
        f.attrs["shape"] = shape
        for k, v in meta.items():
            f.attrs[k] = v if isinstance(
                v, (str, int, float, bool, np.integer, np.floating)) else str(v)
        keys = (["z_vec", "r_vec", "mask"] if shape == "area"
                else ["s", "distance", "z", "r", "mask"]) + list(fields)
        for k in keys:
            f.create_dataset(k, data=data[k])


def _write_txt(out_path, data, meta, shape, fields, return_complex):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# AxiCavity-FEM field export (shape={shape})\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        if shape == "area":
            coords = ["z", "r"]
            n_i, n_j = len(data["z_vec"]), len(data["r_vec"])
            idx = [(i, j) for i in range(n_i) for j in range(n_j)]
            getcoord = lambda ij: (data["z_vec"][ij[0]], data["r_vec"][ij[1]])
        else:
            coords = ["s", "distance", "z", "r"]
            idx = [(i,) for i in range(len(data["s"]))]
            getcoord = lambda ij: (data["s"][ij[0]], data["distance"][ij[0]],
                                   data["z"][ij[0]], data["r"][ij[0]])
        cols = list(coords)
        for fld in fields:
            if return_complex and fld not in _ABS_FIELDS:
                cols += [f"Re({fld})", f"Im({fld})"]
            else:
                cols.append(fld)
        cols.append("mask")
        f.write("# Columns: " + " ".join(cols) + "\n")
        for ij in idx:
            row = [f"{c:.9e}" for c in getcoord(ij)]
            for fld in fields:
                v = data[fld][ij] if shape != "area" else data[fld][ij[0], ij[1]]
                if return_complex and fld not in _ABS_FIELDS:
                    row += [f"{v.real:.6e}", f"{v.imag:.6e}"]
                else:
                    row.append(f"{float(np.real(v)):.6e}")
            m = data["mask"][ij] if shape != "area" else data["mask"][ij[0], ij[1]]
            row.append(str(int(m)))
            f.write(" ".join(row) + "\n")


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------
def export_fields(input_path, output, *, solver_type=None, mode=0, n=None,
                  phase=None, shape="area", z_range=None, r_range=None,
                  nz=200, nr=100, p1=None, p2=None, npts=500,
                  time_phase=0.0, scale=1.0, scale_to_power=None,
                  instant=False, fmt="both") -> dict:
    """場マップを HDF5/TXT に出力し、サンプリングした data dict を返す.

    Args:
        input_path: 入力 HDF5（solve/post の出力）。
        output: 出力ベースパス（拡張子は自動で .h5 / .txt）。
        solver_type: "tm0"/"hom"。None で H5 の solver_type を使用。
        mode: モードインデックス。
        n: 方位角次数（HOM）。None で自動。
        phase: 進行波の位相 [度]。None で先頭。
        shape: "area"/"line"/"axis"。
        z_range/r_range: ``(min, max)``。None で領域全体。
        nz/nr: area のグリッド点数。
        p1/p2: line の端点 ``(z, r)``。
        npts: line/axis の点数。
        time_phase: 瞬時値の時間位相 [度]。
        scale: 場の倍率。
        scale_to_power: 目標壁損失 [W]。post_process の P_loss から倍率を算出。
        instant: 進行波で瞬時実数値を出力（TM0）。
        fmt: "h5"/"txt"/"both"。

    Returns:
        サンプリングした data dict（場配列・座標・mask）。
    """
    data_in = read_results(input_path) if is_v2(input_path) \
        else load_v1_legacy(input_path)
    if solver_type is None:
        solver_type = data_in["solver_type"]

    modeset, analysis_type, n, phase = _select_modeset(
        data_in, solver_type, n, phase)
    if mode >= len(modeset["frequencies"]):
        raise IndexError(
            f"mode {mode} は範囲外です（モード数 {len(modeset['frequencies'])}）")

    is_traveling = analysis_type == "traveling"
    return_complex = (solver_type == "tm0") and is_traveling and (not instant)

    # scale-to-power（post_process の P_loss から）
    if scale_to_power is not None:
        p_loss = _lookup_p_loss(data_in, n, analysis_type, phase, mode)
        if not p_loss or p_loss <= 0:
            raise ValueError("--scale-to-power には post 済み (P_loss>0) が必要です。")
        scale = float(np.sqrt(scale_to_power / p_loss))

    field_at, fields = _build_field_fn(
        data_in, solver_type, mode, n, analysis_type, modeset,
        time_phase, return_complex)

    nodes = data_in["mesh"]["vertices"]
    if shape == "area":
        zr = z_range or (float(nodes[:, 0].min()), float(nodes[:, 0].max()))
        rr = r_range or (float(nodes[:, 1].min()), float(nodes[:, 1].max()))
        data = sample_area(field_at, fields, zr, rr, nz, nr, scale,
                           return_complex)
    elif shape in ("line", "axis"):
        if shape == "axis":
            zr = z_range or (float(nodes[:, 0].min()), float(nodes[:, 0].max()))
            pp1, pp2 = (zr[0], 0.0), (zr[1], 0.0)
        else:
            if p1 is None or p2 is None:
                raise ValueError("shape=line には p1, p2 が必要です。")
            pp1, pp2 = p1, p2
        data = sample_line(field_at, fields, pp1, pp2, npts, scale,
                           return_complex)
    else:
        raise ValueError(f"未知の shape: {shape}")

    meta = {
        "solver_type": solver_type, "mode_index": int(mode),
        "frequency_GHz": float(modeset["frequencies"][mode]),
        "analysis_type": analysis_type, "n": int(n),
        "phase_shift_deg": float(phase) if phase is not None else 0.0,
        "time_phase_deg": float(time_phase), "scale_factor": float(scale),
        "is_complex": bool(return_complex), "shape": shape,
    }

    base, _ = os.path.splitext(output)
    if fmt in ("h5", "both"):
        _write_h5(base + ".h5", data, meta, shape, fields)
    if fmt in ("txt", "both"):
        _write_txt(base + ".txt", data, meta, shape, fields, return_complex)

    data["_meta"] = meta
    data["_fields"] = fields
    return data


def _lookup_p_loss(data_in, n, analysis_type, phase, mode):
    pp = data_in.get("post_process")
    if not pp or n not in pp:
        return None
    rec = pp[n]
    if analysis_type == "standing":
        modes = rec.get("standing")
    else:
        modes = rec.get("traveling", {}).get(phase)
    if not modes or mode >= len(modes):
        return None
    return float(modes[mode].get("P_loss", 0.0))
