"""export の --scale-to-power（GUI の「目標壁損失」）の回帰テスト.

バグ（3.0.0 まで）: TM0 の post の P_loss は軸上ピーク E_z = 1 V/m に規格化した値だが、export はそれで求めた倍率を
solve の固有ベクトル（規格化していない）に掛けていた。実際の壁損失は目標の max|E_z(r=0)|² 倍（1 W の指定で多くは 1e9〜1e10 W 台）。

照合は export が書いた場だけから行い、post の P_loss は使わない:
- 壁面積分 P = ½ Rs ∮|H_tan|² dS = π Rs ∫|H_tan|² r dl
- 解析解（H_φ = 複素振幅）:
    ピルボックス TM010: E_z = E0 J0(kr), H_φ = (E0/η) J1(kr), ω = x01 c/(R √ε_r), η は媒質の値
        P = Rs H_w² π R (R+L)（H_w = H_φ(R)）→ H_w = sqrt(P/(Rs π R (R+L))), E0 = η H_w / J1(x01)
        Q = ωμ₀RL/(2Rs(R+L))
    導波管 TM01 の周期セル: E_z = E0 J0(k_c r) e^{-jβz}, H_φ = (ωε₀/k_c) E0 J1(k_c r) e^{-jβz}
        P = π Rs R L H_w² → H_w = sqrt(P/(π Rs R L)), E0 = H_w k_c/(ωε₀ J1(x01)), Q = ωμ₀R/(2Rs)
- ωU/Q（U は書き出した E・H の体積積分、Q は解析解）
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import j1, jn_zeros, jnp_zeros

pytest.importorskip("gmsh")
h5py = pytest.importorskip("h5py")

from axicavity_fem.cli.main import cli_main  # noqa: E402
from axicavity_fem.gui.jobs.commands import export_argv  # noqa: E402
from axicavity_fem.gui.jobs.pipeline import write_materials_json  # noqa: E402
from axicavity_fem.reports.export_fields import export_fields  # noqa: E402
from axicavity_fem.shared.constants import C0, EPS0, MU0  # noqa: E402
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region  # noqa: E402
from axicavity_fem.shared.hdf5_io import read_results  # noqa: E402
from axicavity_fem.shared.multi_region_model import Loop, MultiRegionGeometry, Region, Segment  # noqa: E402

SIGMA = 5.8e7
TARGET = 1000.0
X01, X02 = jn_zeros(0, 2)
X11 = jn_zeros(1, 1)[0]
ETA0 = np.sqrt(MU0 / EPS0)
PILL_R_MM, PILL_L_MM = 50.0, 100.0
WG_R_MM, WG_L_MM, THETA = 40.0, 35.0, 120.0
OTHER_THETA = 60.0                 # 導波管の解では先頭の位相（THETA は 2 番目）
EPS = 1e-8  # HOM の点探索（trifinder）が壁の上を外さないよう、ごくわずか内側を通る


def _rs(omega):
    return np.sqrt(omega * MU0 / (2 * SIGMA))


# ---------------------------------------------------------------------------
# メッシュ・解析（test_tm0_traveling_phase_invariance.py と同じ作り方）
# ---------------------------------------------------------------------------
def _rect_geom(length_mm, radius_mm, end_bc, mesh_size, inner_mm=0.0, eps_r=1.0, material="vacuum"):
    pts = [(0.0, inner_mm), (length_mm, inner_mm), (length_mm, radius_mm), (0.0, radius_mm)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="PEC" if inner_mm > 0 else "None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name=end_bc),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name=end_bc)]
    region = Region(id=0, name="Cell", outer_loop_id=0, material_tag=material, eps_r=eps_r)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[region], unit="mm", mesh_size=mesh_size)


def _solve_post(folder, geom, order=2, solver="tm0", extra=()):
    folder.mkdir(parents=True, exist_ok=True)
    msh, raw, proc = folder / "m.msh", folder / "raw.h5", folder / "proc.h5"
    export_msh_multi_region(geom, msh, mesh_order=order, verbose=0)
    write_materials_json(geom, msh)
    assert cli_main(["solve", "--type", solver, "-m", str(msh), "--elem-order", str(order), "--num-modes", "4",
                     *extra, "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", solver, "-i", str(raw), "-o", str(proc), "--cond", str(SIGMA)]) == 0
    return proc


def _closest_mode(proc, omega, n=0, theta=None):
    """解析解の周波数に最も近いモード (index, ω)."""
    rec = read_results(proc)["results_by_n"][n]
    freqs = rec["standing"]["frequencies"] if theta is None else rec["traveling"][theta]["frequencies"]
    k = int(np.argmin(np.abs(np.asarray(freqs) * 2e9 * np.pi - omega)))
    return k, float(freqs[k]) * 2e9 * np.pi


def _line(proc, tmp_path, p1, p2, mode, npts=1001, **kw):
    kw.setdefault("scale_to_power", TARGET)
    return export_fields(str(proc), str(tmp_path / "line"), mode=mode, shape="line", p1=p1, p2=p2, npts=npts,
                         fmt="h5", **kw)


def _wall_power(data, omega, comps=("H_theta",)):
    assert data["mask"].all()
    h2 = sum(np.abs(data[c]) ** 2 for c in comps)
    return np.pi * _rs(omega) * trapezoid(h2 * data["r"], x=data["distance"])


def _energies(data):
    """書き出した面内の場から (U_E, U_H) = ½ε₀∫|E|²dV, ½μ₀∫|H|²dV."""
    assert data["mask"].all()
    r, z = data["r_vec"], data["z_vec"]
    w = 2 * np.pi * r
    u_e = 0.5 * EPS0 * trapezoid(trapezoid(np.abs(data["E_abs"]) ** 2 * w, x=r, axis=1), x=z)
    u_h = 0.5 * MU0 * trapezoid(trapezoid(np.abs(data["H_theta"]) ** 2 * w, x=r, axis=1), x=z)
    return u_e, u_h


# ---------------------------------------------------------------------------
# ピルボックス TM010（定在波）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pillbox(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pillbox")
    return _solve_post(folder, _rect_geom(PILL_L_MM, PILL_R_MM, "PEC", 4.0))


def _pill_walls(proc, tmp_path, k):
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    return (_line(proc, tmp_path, (0.0, R), (L, R), k), _line(proc, tmp_path, (0.0, 0.0), (0.0, R), k),
            _line(proc, tmp_path, (L, 0.0), (L, R), k))


@pytest.mark.parametrize("order, mesh_size, tol", [(1, 2.0, 1e-3), (2, 4.0, 1e-3)])
def test_pillbox_wall_integral_is_the_target(tmp_path, order, mesh_size, tol):
    proc = _solve_post(tmp_path / "m", _rect_geom(PILL_L_MM, PILL_R_MM, "PEC", mesh_size), order=order)
    R = PILL_R_MM * 1e-3
    k, omega = _closest_mode(proc, X01 * C0 / R)
    power = sum(_wall_power(d, omega) for d in _pill_walls(proc, tmp_path, k))
    assert power == pytest.approx(TARGET, rel=tol, abs=0)


def test_pillbox_fields_match_analytic(pillbox, tmp_path):
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    k, omega = _closest_mode(pillbox, X01 * C0 / R)
    h_wall = np.sqrt(TARGET / (_rs(omega) * np.pi * R * (R + L)))
    side = _line(pillbox, tmp_path, (0.0, R), (L, R), k)
    assert np.abs(side["H_theta"]) == pytest.approx(h_wall, rel=1e-3, abs=0)
    axis = export_fields(str(pillbox), str(tmp_path / "axis"), mode=k, shape="axis", npts=301,
                         scale_to_power=TARGET, fmt="h5")
    assert np.max(np.abs(axis["Ez"])) == pytest.approx(ETA0 * h_wall / j1(X01), rel=5e-3, abs=0)


def test_pillbox_omega_u_over_q_is_the_target(pillbox, tmp_path):
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    k, omega = _closest_mode(pillbox, X01 * C0 / R)
    area = export_fields(str(pillbox), str(tmp_path / "area"), mode=k, shape="area", z_range=(0.0, L), r_range=(0.0, R),
                         nz=201, nr=101, scale_to_power=TARGET, fmt="h5")
    q = omega * MU0 * R * L / (2 * _rs(omega) * (R + L))
    u_e, u_h = _energies(area)
    assert omega * u_e / q == pytest.approx(TARGET, rel=1e-3, abs=0)
    assert omega * u_h / q == pytest.approx(TARGET, rel=1e-3, abs=0)


def test_gui_argv_writes_the_scaled_field(pillbox, tmp_path):
    """GUI の「目標壁損失」は CLI の引数になる。書かれた h5 の値と scale_factor を確かめる."""
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    k, omega = _closest_mode(pillbox, X01 * C0 / R)
    out = tmp_path / "field"
    argv = export_argv("tm0", str(pillbox), str(out), "line", k,
                       params={"p1": (0.0, R), "p2": (L, R), "npts": 11, "scale_to_power": TARGET, "fmt": "h5"})
    assert cli_main(argv) == 0
    with h5py.File(str(out) + ".h5", "r") as f:
        h = f["H_theta"][()]
        scale = float(f.attrs["scale_factor"])
    h_wall = np.sqrt(TARGET / (_rs(omega) * np.pi * R * (R + L)))
    assert np.abs(h) == pytest.approx(h_wall, rel=1e-3, abs=0)
    raw = _line(pillbox, tmp_path, (0.0, R), (L, R), k, npts=11, scale_to_power=None)
    assert raw["_meta"]["scale_factor"] == 1.0
    np.testing.assert_allclose(h, raw["H_theta"] * scale, rtol=1e-12)


# ---------------------------------------------------------------------------
# 導波管 TM01（進行波）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def waveguide(tmp_path_factory):
    folder = tmp_path_factory.mktemp("waveguide")
    return _solve_post(folder, _rect_geom(WG_L_MM, WG_R_MM, "None", 2.0), extra=("-p", f"{OTHER_THETA:g},{THETA:g}"))


def _wg_reference(proc):
    R, L = WG_R_MM * 1e-3, WG_L_MM * 1e-3
    kc, bz = X01 / R, np.radians(THETA) / L
    k, omega = _closest_mode(proc, np.hypot(kc, bz) * C0, theta=THETA)
    return R, L, kc, k, omega


def test_traveling_wall_integral_and_fields(waveguide, tmp_path):
    R, L, kc, k, omega = _wg_reference(waveguide)
    side = _line(waveguide, tmp_path, (0.0, R), (L, R), k, phase=THETA)
    assert _wall_power(side, omega) == pytest.approx(TARGET, rel=1e-3, abs=0)
    h_wall = np.sqrt(TARGET / (np.pi * _rs(omega) * R * L))
    assert np.abs(side["H_theta"]) == pytest.approx(h_wall, rel=1e-3, abs=0)
    axis = export_fields(str(waveguide), str(tmp_path / "axis"), mode=k, phase=THETA, shape="axis", npts=301,
                         scale_to_power=TARGET, fmt="h5")
    e0 = h_wall * kc / (omega * EPS0 * j1(X01))
    assert np.abs(axis["Ez"]) == pytest.approx(e0, rel=5e-3, abs=0)
    area = export_fields(str(waveguide), str(tmp_path / "area"), mode=k, phase=THETA, shape="area", z_range=(0.0, L),
                         r_range=(0.0, R), nz=141, nr=161, scale_to_power=TARGET, fmt="h5")
    q = omega * MU0 * R / (2 * _rs(omega))
    u_e, u_h = _energies(area)
    assert omega * u_e / q == pytest.approx(TARGET, rel=1e-3, abs=0)
    assert omega * u_h / q == pytest.approx(TARGET, rel=1e-3, abs=0)


@pytest.mark.parametrize("theta", [OTHER_THETA, THETA])
@pytest.mark.parametrize("zero", [X01, X02], ids=["TM01", "TM02"])
def test_traveling_every_phase_and_mode(waveguide, tmp_path, theta, zero):
    """U_stored と P_loss は書き出す位相・モードの値を読む（先頭の位相や mode 0 の値を読むと目標からずれる）."""
    R, L = WG_R_MM * 1e-3, WG_L_MM * 1e-3
    k, omega = _closest_mode(waveguide, np.hypot(zero / R, np.radians(theta) / L) * C0, theta=theta)
    assert (k > 0) == (zero == X02)
    side = _line(waveguide, tmp_path, (0.0, R), (L, R), k, phase=theta)
    assert _wall_power(side, omega) == pytest.approx(TARGET, rel=1e-3, abs=0)


def test_traveling_instant_values_use_the_same_scale(waveguide, tmp_path):
    """瞬時値は Re[H̃ e^{jθ}]: θ=0° と 90° の 2 乗和が複素振幅の 2 乗（壁面積分が目標）."""
    R, L, _, k, omega = _wg_reference(waveguide)
    kw = dict(phase=THETA, instant=True)
    p0 = _wall_power(_line(waveguide, tmp_path, (0.0, R), (L, R), k, time_phase=0.0, **kw), omega)
    p90 = _wall_power(_line(waveguide, tmp_path, (0.0, R), (L, R), k, time_phase=90.0, **kw), omega)
    assert p0 + p90 == pytest.approx(TARGET, rel=1e-3, abs=0)


def _edit_h5(src, dst, edit):
    shutil.copy2(src, dst)
    with h5py.File(dst, "a") as f:
        edit(f)
    return dst


def test_independent_of_the_global_phase_of_the_eigenvector(waveguide, tmp_path):
    R, L, _, k, omega = _wg_reference(waveguide)

    def rotate(f):
        g = f[f"results/n0/traveling/phase_{int(round(THETA * 10)):05d}/mode_{k}"]
        u = (g["eigenvector_re"][()] + 1j * g["eigenvector_im"][()]) * np.exp(1j * np.radians(73.0))
        g["eigenvector_re"][...] = u.real
        g["eigenvector_im"][...] = u.imag

    rotated = _edit_h5(waveguide, tmp_path / "rot.h5", rotate)
    a = _line(waveguide, tmp_path, (0.0, R), (L, R), k, phase=THETA, npts=51)
    b = _line(rotated, tmp_path, (0.0, R), (L, R), k, phase=THETA, npts=51)
    assert a["_meta"]["scale_factor"] == pytest.approx(b["_meta"]["scale_factor"], rel=1e-9)
    np.testing.assert_allclose(np.abs(a["H_theta"]), np.abs(b["H_theta"]), rtol=1e-9)


def test_post_of_older_versions(waveguide, tmp_path):
    """3.0.0 以前の進行波の post は U・P_loss が別の規格化（E_z の実部）だが、両者の比は同じ。h5 のスキーマも同じ."""
    R, L, _, k, omega = _wg_reference(waveguide)

    def renormalize(f):
        g = f[f"post_process/n0/traveling/phase_{int(round(THETA * 10)):05d}/mode_{k}"]
        assert "norm_factor" not in g.attrs
        for name in ("U_stored", "P_loss"):
            g.attrs[name] = g.attrs[name] * 2.37

    old = _edit_h5(waveguide, tmp_path / "old.h5", renormalize)
    side = _line(old, tmp_path, (0.0, R), (L, R), k, phase=THETA)
    assert _wall_power(side, omega) == pytest.approx(TARGET, rel=1e-3, abs=0)


# ---------------------------------------------------------------------------
# 規格化が効かない / 効き方が違うメッシュ
# ---------------------------------------------------------------------------
def test_dielectric_on_the_axis(tmp_path):
    """一様 ε_r: 倍率は P_loss/U の比で決まるので、post の規格化によらず壁損失が目標になる."""
    eps_r = 4.0
    geom = _rect_geom(PILL_L_MM, PILL_R_MM, "PEC", 4.0, eps_r=eps_r, material="diel")
    proc = _solve_post(tmp_path / "m", geom, extra=("--target-freq", "1.1"))   # 自動のシフトは真空の TM010 付近を探す
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    k, omega = _closest_mode(proc, X01 * C0 / (R * np.sqrt(eps_r)))
    assert omega == pytest.approx(X01 * C0 / (R * np.sqrt(eps_r)), rel=1e-4)
    walls = _pill_walls(proc, tmp_path, k)
    assert sum(_wall_power(d, omega) for d in walls) == pytest.approx(TARGET, rel=1e-3, abs=0)
    h_wall = np.sqrt(TARGET / (_rs(omega) * np.pi * R * (R + L)))
    assert np.abs(walls[0]["H_theta"]) == pytest.approx(h_wall, rel=1e-3, abs=0)
    axis = export_fields(str(proc), str(tmp_path / "axis"), mode=k, shape="axis", npts=301,
                         scale_to_power=TARGET, fmt="h5")
    assert np.max(np.abs(axis["Ez"])) == pytest.approx(ETA0 / np.sqrt(eps_r) * h_wall / j1(X01), rel=5e-3, abs=0)


def test_no_axis_in_the_mesh(tmp_path):
    """軸が無い同軸は post の規格化係数が 1。全ての壁の積分で確かめる."""
    inner, outer, length = 5e-3, 40e-3, 35e-3
    proc = _solve_post(tmp_path / "m", _rect_geom(35.0, 40.0, "PEC", 3.0, inner_mm=5.0))
    f = read_results(proc)["results_by_n"][0]["standing"]["frequencies"][0]
    omega = float(f) * 2e9 * np.pi
    power = sum(_wall_power(_line(proc, tmp_path, p1, p2, 0), omega)
                for p1, p2 in (((0.0, inner), (length, inner)), ((0.0, outer), (length, outer)),
                               ((0.0, inner), (0.0, outer)), ((length, inner), (length, outer))))
    assert power == pytest.approx(TARGET, rel=1e-3, abs=0)


# ---------------------------------------------------------------------------
# 他の経路を変えていないこと
# ---------------------------------------------------------------------------
def test_scale_alone_stays_relative_to_the_stored_eigenvector(pillbox, tmp_path):
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    raw = _line(pillbox, tmp_path, (0.0, R), (L, R), 0, npts=11, scale_to_power=None)
    twice = _line(pillbox, tmp_path, (0.0, R), (L, R), 0, npts=11, scale_to_power=None, scale=2.0)
    np.testing.assert_allclose(twice["H_theta"], 2.0 * raw["H_theta"], rtol=1e-14)


def test_scale_to_power_takes_precedence_over_scale(pillbox, tmp_path):
    """GUI のダイアログは両方を argv に載せられる. 目標壁損失が優先で、--scale は掛からない."""
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    only = _line(pillbox, tmp_path, (0.0, R), (L, R), 0, npts=11)
    both = _line(pillbox, tmp_path, (0.0, R), (L, R), 0, npts=11, scale=2.0)
    assert both["_meta"]["scale_factor"] == only["_meta"]["scale_factor"]
    np.testing.assert_array_equal(both["H_theta"], only["H_theta"])


def test_needs_post_result(tmp_path, pillbox):
    without_post = _edit_h5(pillbox, tmp_path / "np.h5", lambda f: f.__delitem__("post_process"))
    with pytest.raises(ValueError, match="post"):
        export_fields(str(without_post), str(tmp_path / "x"), shape="axis", npts=3, scale_to_power=1.0, fmt="h5")


def test_hom_tm110_wall_integral_is_the_target(tmp_path):
    """HOM の P_loss は固有ベクトルそのものの値なので元から正しい経路（TM110 は壁で H_z = 0）."""
    R, L = PILL_R_MM * 1e-3, PILL_L_MM * 1e-3
    proc = _solve_post(tmp_path / "m", _rect_geom(PILL_L_MM, PILL_R_MM, "PEC", 4.0), solver="hom",
                       extra=("--az-order", "1"))
    k, omega = _closest_mode(proc, X11 * C0 / R, n=1)
    kw = dict(solver_type="hom", n=1, npts=2001)
    side = _line(proc, tmp_path, (EPS, R - EPS), (L - EPS, R - EPS), k, **kw)
    ends = [_line(proc, tmp_path, (z, 1e-5), (z, R - EPS), k, **kw) for z in (EPS, L - EPS)]
    power = (_wall_power(side, omega, ("Hz", "H_theta"))
             + sum(_wall_power(d, omega, ("Hr", "H_theta")) for d in ends))
    assert power == pytest.approx(TARGET, rel=1e-3, abs=0)


def test_hom_file_with_several_n_and_phases(tmp_path):
    """1 つの h5 に複数の n と位相があるとき、書き出す n・位相の P_loss を読む（導波管 TE21、瞬時値の θ=0° と 90° の 2 乗和）."""
    R, L = WG_R_MM * 1e-3, WG_L_MM * 1e-3
    proc = _solve_post(tmp_path / "m", _rect_geom(WG_L_MM, WG_R_MM, "None", 3.0), solver="hom",
                       extra=("--az-order", "1", "2", "-p", f"{OTHER_THETA:g},{THETA:g}"))
    k, omega = _closest_mode(proc, np.hypot(jnp_zeros(2, 1)[0] / R, np.radians(THETA) / L) * C0, n=2, theta=THETA)
    power = sum(_wall_power(_line(proc, tmp_path, (EPS, R - EPS), (L - EPS, R - EPS), k, solver_type="hom", n=2,
                                  phase=THETA, time_phase=time_phase), omega, ("Hz", "H_theta"))
                for time_phase in (0.0, 90.0))
    assert power == pytest.approx(TARGET, rel=2e-3, abs=0)
