"""HOM の post（P_flow, v_g, α, P_loss, Q, U, P_diel）を解析解と電力の積分によらない基準で照合する回帰テスト.

コアの E_θ・H_z・H_r は物理量の 1/j 倍（E_φ = jE_θ。要素行列 |nE_t − ∇(rE_θ)|²/r が実数になる取り方。
``field_recon`` と ``gui/core/revolve.py`` も同じ）。3.0.0 までの ``fem_hom/post_process`` には次の誤りがあった:

1. ``calc_p_flow_hom`` の第 2 項が −n·Re[E_θ E_z*]。S_z = ½Re[E_r H_φ* − E_φ H_r*] から出る正しい項は −n·Im[E_θ E_z*]。
   E_z を持つ n ≥ 1 のモードで P_flow, v_g, α が誤る（中空導波管の TM11 で v_g −25%、周期構造では符号が逆のモードもある）。
2. ``_wall_loss`` の curl に余分な j（``1j*(n/r)*E_z``、``1j*(n/r)*E_r``）。|A − B|² が |A|² + |B|² になり、TE では H_z が過小、
   TM では 0 のはずの H_z が現れる（導波管の Q: TE11 +38%、TE21 +63%、TM11 −2.9%、TM21 −4.6%）。n = 0 は影響なし。
3. 定在波の U の係数が 1.0（正しくは ½。ピーク振幅に対する全エネルギー U = ½ε₀∫|E|²dV は定在波でも進行波でも同じ）。
   定在波の U, Q, Q_wall, P_diel が n = 0 を含めてすべて 2 倍。

解析解（R, L は半径と長さ、x は J_n または J_n' の零点、k_c = x/R、η = √(μ₀/ε₀)。基準はこのファイルの式で、FEM の値を使わない）:

- 導波管の周期セル: β = θ/L, k² = k_c² + β², v_g/c = β/k
    TM_nm: α = R_s k/(R η β)、TE_nm: α = R_s/(R k η β)·(k_c² + k²n²/(x² − n²))、Q = ω/(2 α v_g)
- ピルボックス TM_nm0: Q = (R/δ)/(1 + R/L)、TM_nmp (p ≥ 1): Q = (R/δ)/(1 + 2R/L)、TE_nmp: 下の ``_pillbox_ref``

電力の積分によらない基準:

- v_g = L dω/dθ（固有周波数だけ）
- 球形空洞の同じ l のモードは回転で移り合うので Q は n によらない
- TM0 ソルバ（H_φ ベース）との一致、θ = 0 の周期セルと端面 E-short の定在波の一致
- ``field_recon`` の H を壁で直接積分した P_loss、書き出した場の壁面積分（目標壁損失）
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import j1, jn_zeros, jnp_zeros, jvp

pytest.importorskip("gmsh")

from axicavity_fem.cli.main import cli_main  # noqa: E402
from axicavity_fem.fem_hom.field_recon import HOMFieldReconstructor, split_hom_dofs  # noqa: E402
from axicavity_fem.gui.core.revolve import ModeFields, field_point_arrays, revolve_mesh, split_triangles  # noqa: E402
from axicavity_fem.fem_hom.post_process import compute_hom_parameters  # noqa: E402
from axicavity_fem.fem_hom.solver import solve_hom_standing, solve_hom_traveling  # noqa: E402
from axicavity_fem.fem_tm0.post_process import compute_tm0_parameters  # noqa: E402
from axicavity_fem.fem_tm0.solver import solve_tm0_standing  # noqa: E402
from axicavity_fem.reports.export_fields import export_fields  # noqa: E402
from axicavity_fem.shared.boundary_groups import classify_boundaries  # noqa: E402
from axicavity_fem.shared.constants import C0, EPS0, MU0  # noqa: E402
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region  # noqa: E402
from axicavity_fem.shared.material_resolver import (  # noqa: E402
    build_eps_r_per_element,
    build_material_table_from_geometry,
    build_tan_delta_per_element,
)
from axicavity_fem.shared.mesh_loader import load_mesh, load_mesh_hom  # noqa: E402
from axicavity_fem.shared.multi_region_model import Loop, MultiRegionGeometry, Region, Segment  # noqa: E402

LENGTH_MM, RADIUS_MM = 35.0, 40.0
LENGTH, RADIUS = LENGTH_MM * 1e-3, RADIUS_MM * 1e-3
THETA, DTHETA = 120.0, 1.0
SIGMA = 5.8e7
ETA = np.sqrt(MU0 / EPS0)
TARGET = 1000.0
SPHERE_RADIUS_MM = 50.0
EPS = 1e-8                # HOM の点探索（trifinder）が壁の上を外さないよう、ごくわずか内側を通る

WAVEGUIDE_MODES = [("TM", 0, 1), ("TM", 1, 1), ("TM", 2, 1), ("TM", 1, 2), ("TE", 1, 1), ("TE", 2, 1)]
WAVEGUIDE_IDS = [f"{kind}{n}{m}" for kind, n, m in WAVEGUIDE_MODES]
PILLBOX_MODES = [("TM", 0, 1, 0), ("TM", 1, 1, 0), ("TM", 2, 1, 0), ("TM", 1, 1, 1), ("TE", 1, 1, 1), ("TE", 2, 1, 1),
                 ("TE", 1, 1, 2)]
PILLBOX_IDS = [f"{kind}{n}{m}{p}" for kind, n, m, p in PILLBOX_MODES]


def _waveguide_tol(kind, n, m):
    """2 mm メッシュの Q・α の許容（実測の最大は低次で 0.36%、TM12 で 0.80%。誤差はメッシュの精度で決まる）."""
    return 0.012 if (kind, n, m) == ("TM", 1, 2) else 0.005


# ---------------------------------------------------------------------------
# メッシュ
# ---------------------------------------------------------------------------
def _rect_geom(end_bc, mesh_size, length_mm=LENGTH_MM, radius_mm=RADIUS_MM, material="vacuum", eps_r=1.0, tan_delta=0.0):
    pts = [(0.0, 0.0), (length_mm, 0.0), (length_mm, radius_mm), (0.0, radius_mm)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name=end_bc),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name=end_bc)]
    region = Region(id=0, name="Cell", outer_loop_id=0, material_tag=material, eps_r=eps_r, tan_delta=tan_delta)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[region], unit="mm", mesh_size=mesh_size)


def _sphere_geom(mesh_size):
    a = SPHERE_RADIUS_MM
    pts = [(-a, 0.0), (a, 0.0), (0.0, a)]                   # 180° の円弧は向きが決まらないので 2 本に分ける
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="arc", point_indices=[1, 2], bc_name="PEC", center=(0.0, 0.0), radius=a,
                    theta1=0.0, theta2=90.0),
            Segment(id=2, type="arc", point_indices=[2, 0], bc_name="PEC", center=(0.0, 0.0), radius=a,
                    theta1=90.0, theta2=180.0)]
    region = Region(id=0, name="Sphere", outer_loop_id=0, material_tag="vacuum")
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2])],
                               regions=[region], unit="mm", mesh_size=mesh_size)


def _write(geom, path, order=2):
    export_msh_multi_region(geom, path, mesh_order=order, verbose=0)
    return str(path)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hom_post")
    return {"waveguide": _write(_rect_geom("None", 2.0), folder / "wg.msh"),
            "pillbox": _write(_rect_geom("PEC", 2.0), folder / "pb.msh"),
            "sphere": _write(_sphere_geom(4.0), folder / "sphere.msh")}


# ---------------------------------------------------------------------------
# 解析解
# ---------------------------------------------------------------------------
def _zero(kind, n, m):
    return float(jn_zeros(n, m)[m - 1] if kind == "TM" else jnp_zeros(n, m)[m - 1])


def _rs(omega):
    return np.sqrt(omega * MU0 / (2.0 * SIGMA))


def _waveguide_ref(kind, n, m, theta=THETA):
    x = _zero(kind, n, m)
    kc, beta = x / RADIUS, np.radians(theta) / LENGTH
    k = np.hypot(kc, beta)
    omega = k * C0
    if kind == "TM":
        alpha = _rs(omega) * k / (RADIUS * ETA * beta)
    else:
        alpha = _rs(omega) / (RADIUS * k * ETA * beta) * (kc ** 2 + k ** 2 * n ** 2 / (x ** 2 - n ** 2))
    vg_c = beta / k
    return dict(f_ghz=omega / (2e9 * np.pi), omega=omega, vg_c=vg_c, alpha=alpha, q=omega / (2.0 * alpha * vg_c * C0),
                kc=kc, beta=beta)


def _pillbox_ref(kind, n, m, p):
    """ピルボックスの (f [GHz], ω, Q).

    TE_nmp（n ≥ 1）: H_z = J_n(k_c r) cos nφ sin βz。J_n(x)² は U と P で共通なので落とす。
      I = ∫₀ᴿ J_n² r dr / J_n(x)² = (R²/2)(1 − n²/x²)
      U = ½μ₀·π(L/2)·I·(1 + β²/k_c²)
      P = ½Rs·[R π (L/2)(1 + (nβ/(k_c² R))²) + 2π (β²/k_c²) I]    （側面 + 両端面）
    """
    x = _zero(kind, n, m)
    kc, beta = x / RADIUS, p * np.pi / LENGTH
    k = np.hypot(kc, beta)
    omega = k * C0
    rs = _rs(omega)
    if kind == "TM":
        delta = 2.0 * rs / (omega * MU0)
        q = (RADIUS / delta) / (1.0 + (RADIUS / LENGTH if p == 0 else 2.0 * RADIUS / LENGTH))
    else:
        integ = 0.5 * RADIUS ** 2 * (1.0 - n ** 2 / x ** 2)
        u = 0.5 * MU0 * np.pi * (0.5 * LENGTH) * integ * (1.0 + beta ** 2 / kc ** 2)
        side = RADIUS * np.pi * (0.5 * LENGTH) * (1.0 + (n * beta / (kc ** 2 * RADIUS)) ** 2)
        ends = 2.0 * np.pi * (beta ** 2 / kc ** 2) * integ
        q = omega * u / (0.5 * rs * (side + ends))
    return omega / (2e9 * np.pi), omega, q


# ---------------------------------------------------------------------------
# 共通処理
# ---------------------------------------------------------------------------
def _nearest(frequencies, f_ghz):
    return int(np.argmin(np.abs(np.asarray(frequencies) - f_ghz)))


def _solve_waveguide(path, kind, n, m, thetas=(THETA,), order=2):
    target = _waveguide_ref(kind, n, m)["f_ghz"] * 1.01
    return solve_hom_traveling(load_mesh_hom(path, n, order), num_modes=3, phase_shifts=thetas, target_freq_ghz=target)


def _traveling_mode(path, kind, n, m, order=2):
    ref = _waveguide_ref(kind, n, m)
    res = _solve_waveguide(path, kind, n, m, order=order)
    i = _nearest(res.periodic[THETA].frequencies, ref["f_ghz"])
    mode = compute_hom_parameters(res, conductivity=SIGMA).per_phase[THETA][i]
    assert mode.frequency_ghz == pytest.approx(ref["f_ghz"], rel=1e-3)
    return ref, mode


def _standing(path, n, f_ghz, **kw):
    res = solve_hom_standing(load_mesh_hom(path, n, 2), num_modes=4, target_freq_ghz=f_ghz * 1.01, **kw)
    i = _nearest(res.normal.frequencies, f_ghz)
    assert res.normal.frequencies[i] == pytest.approx(f_ghz, rel=1e-3)
    return res, i


def _standing_params(res, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_hom_parameters(res, conductivity=SIGMA, **kw).per_phase[0.0]


# ---------------------------------------------------------------------------
# 1) 進行波: v_g（P_flow の第 2 項）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, n, m", WAVEGUIDE_MODES, ids=WAVEGUIDE_IDS)
def test_group_velocity_matches_waveguide_analytic(meshes, kind, n, m):
    ref, mode = _traveling_mode(meshes["waveguide"], kind, n, m)
    assert mode.v_group_c == pytest.approx(ref["vg_c"], rel=0.005, abs=0)
    assert mode.p_flow * LENGTH / mode.stored_energy / C0 == pytest.approx(mode.v_group_c, rel=1e-12, abs=0)


@pytest.mark.parametrize("kind, n, m", WAVEGUIDE_MODES, ids=WAVEGUIDE_IDS)
def test_group_velocity_matches_d_omega_d_theta_in_waveguide(meshes, kind, n, m):
    """電力の積分を使わない基準 v_g = L dω/dθ."""
    thetas = (THETA - DTHETA, THETA, THETA + DTHETA)
    res = _solve_waveguide(meshes["waveguide"], kind, n, m, thetas)
    f = {th: res.periodic[th].frequencies[_nearest(res.periodic[th].frequencies, _waveguide_ref(kind, n, m, th)["f_ghz"])]
         for th in thetas}
    vg_fd = LENGTH * 2e9 * np.pi * (f[thetas[2]] - f[thetas[0]]) / np.radians(2 * DTHETA) / C0
    i = _nearest(res.periodic[THETA].frequencies, _waveguide_ref(kind, n, m)["f_ghz"])
    assert compute_hom_parameters(res).per_phase[THETA][i].v_group_c == pytest.approx(vg_fd, rel=0.01, abs=0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_group_velocity_matches_d_omega_d_theta_in_sband(sample_mesh, n):
    """周期構造（ハイブリッドモード）: モード 0〜5 で |Δv_g| < max(0.003c, 3.5%).

    このメッシュの実測の最大は 0.0048c、相対 2.4%。差は h² で減るがメッシュの作り方で揺れる（gmsh の版が変わっても
    落ちないよう、メッシュサイズ ×0.8〜×1.3 で許容の 2/3 以下になる幅にしてある）。修正前は最大 0.085c で、
    n=2 の mode 0 と mode 5 は符号が逆だった。
    """
    mesh = load_mesh_hom(str(sample_mesh("s-band_1cell", 2)), n, 2)
    thetas = (THETA - DTHETA, THETA, THETA + DTHETA)
    res = solve_hom_traveling(mesh, num_modes=6, phase_shifts=thetas)
    params = compute_hom_parameters(res).per_phase[THETA]
    assert len(params) == 6
    length = float(res.vertices[:, 0].max() - res.vertices[:, 0].min())
    M = res.M_global

    def follow(vec, modes):
        overlap = [abs(np.vdot(e, M @ vec)) for e in modes.eigenvectors]
        return modes.frequencies[int(np.argmax(overlap))]

    modes = res.periodic[THETA]
    for vec, mode in zip(modes.eigenvectors, params):
        fp, fm = follow(vec, res.periodic[thetas[2]]), follow(vec, res.periodic[thetas[0]])
        vg_fd = length * 2e9 * np.pi * (fp - fm) / np.radians(2 * DTHETA) / C0
        assert abs(mode.v_group_c - vg_fd) < max(3e-3, 0.035 * abs(vg_fd)), f"f={mode.frequency_ghz:.4f} GHz"


def test_core_e_theta_is_one_over_j_times_physical(meshes):
    """修正の根拠: TM では E_φ = (βn/(k_c² r)) E_z（同相）なのでコアの E_θ = E_φ/j は E_z より 90° 遅れる."""
    res = _solve_waveguide(meshes["waveguide"], "TM", 1, 1)
    ref = _waveguide_ref("TM", 1, 1)
    modes = res.periodic[THETA]
    i = _nearest(modes.frequencies, ref["f_ghz"])
    rec = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "traveling")
    rec.load_mode(modes.eigenvectors[i], modes.frequencies[i])
    z, r = 17.5e-3, 17.3e-3
    a, b = rec.calculate_fields(z, r, 0.0), rec.calculate_fields(z, r, 90.0)
    ratio = (a["E_theta"] - 1j * b["E_theta"]) / (a["Ez"] - 1j * b["Ez"])
    assert ratio == pytest.approx(-1j * ref["beta"] / (ref["kc"] ** 2 * r), rel=0.01, abs=0)


def test_p_flow_does_not_depend_on_global_phase(meshes):
    res = _solve_waveguide(meshes["waveguide"], "TM", 1, 1)
    modes = res.periodic[THETA]
    before = compute_hom_parameters(res).per_phase[THETA]
    modes.eigenvectors[:] = modes.eigenvectors * np.exp(1j * np.radians(37.0))
    for a, b in zip(before, compute_hom_parameters(res).per_phase[THETA]):
        assert b.p_flow == pytest.approx(a.p_flow, rel=1e-9, abs=0)


def test_bloch_mode_at_180_degrees_carries_no_power(meshes):
    """θ=180° の TM11 は前進波と後退波の重ね合わせで P_flow = 0（修正前は 0.1c 前後）. 縮退の混合比によらない."""
    res = _solve_waveguide(meshes["waveguide"], "TM", 1, 1, (180.0,))
    modes = res.periodic[180.0]
    i = _nearest(modes.frequencies, _waveguide_ref("TM", 1, 1, 180.0)["f_ghz"])
    assert abs(compute_hom_parameters(res).per_phase[180.0][i].v_group_c) < 1e-3


# ---------------------------------------------------------------------------
# 2) 進行波: Q（_wall_loss）と α = P_loss/(2 P_flow L)（両方の修正が要る）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, n, m", WAVEGUIDE_MODES, ids=WAVEGUIDE_IDS)
def test_traveling_q_matches_waveguide_analytic(meshes, kind, n, m):
    ref, mode = _traveling_mode(meshes["waveguide"], kind, n, m)
    assert mode.q_wall == pytest.approx(ref["q"], rel=_waveguide_tol(kind, n, m), abs=0)


@pytest.mark.parametrize("kind, n, m", WAVEGUIDE_MODES, ids=WAVEGUIDE_IDS)
def test_attenuation_matches_waveguide_analytic(meshes, kind, n, m):
    """修正前は TE11 −28%、TE21 −39%、TM11 +38%、TM21 +76%."""
    ref, mode = _traveling_mode(meshes["waveguide"], kind, n, m)
    assert mode.attenuation == pytest.approx(ref["alpha"], rel=_waveguide_tol(kind, n, m), abs=0)


@pytest.mark.parametrize("kind, n, m", [("TM", 1, 1), ("TE", 1, 1)], ids=["TM11", "TE11"])
def test_first_order_elements(tmp_path, kind, n, m):
    path = _write(_rect_geom("None", 1.5), tmp_path / "wg1.msh", order=1)
    ref, mode = _traveling_mode(path, kind, n, m, order=1)
    assert mode.v_group_c == pytest.approx(ref["vg_c"], rel=0.02, abs=0)
    assert mode.q_wall == pytest.approx(ref["q"], rel=0.04, abs=0)
    assert mode.attenuation == pytest.approx(ref["alpha"], rel=0.04, abs=0)


# ---------------------------------------------------------------------------
# 3) 定在波: Q（_wall_loss と U の係数）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, n, m, p", PILLBOX_MODES, ids=PILLBOX_IDS)
def test_standing_q_matches_pillbox_analytic(meshes, kind, n, m, p):
    """修正前は解析解の 2.0〜2.33 倍. 2 mm メッシュの誤差の実測の最大は TE112 の 1.0%."""
    f_ghz, _, q = _pillbox_ref(kind, n, m, p)
    res, i = _standing(meshes["pillbox"], n, f_ghz)
    mode = _standing_params(res)[i]
    assert mode.q_wall == pytest.approx(q, rel=0.015, abs=0)
    assert mode.q_factor == pytest.approx(q, rel=0.015, abs=0)


def test_straight_wall_q_converges_like_h_squared(tmp_path):
    """曲線の壁（既知の制約。h の 1 次）と違い、直線の壁の Q は h² で収束する（TE211 の誤差は実測で −0.69 / −0.18%）."""
    f_ghz, _, q = _pillbox_ref("TE", 2, 1, 1)
    errors = []
    for h in (4.0, 2.0):
        res, i = _standing(_write(_rect_geom("PEC", h), tmp_path / f"pb{h:g}.msh"), 2, f_ghz)
        errors.append(abs(_standing_params(res)[i].q_wall / q - 1.0))
    assert errors[1] < 0.003 and errors[0] / errors[1] > 3.0, errors


@pytest.mark.parametrize("n", [0, 1, 2])
def test_standing_u_is_half_eps0_integral(meshes, n):
    """同じ固有ベクトルで U = ½ε₀∫|E|²dV = πε₀ vᴴMv（M は ∫(...)r dA で 2π を含まない）."""
    res = solve_hom_standing(load_mesh_hom(meshes["pillbox"], n, 2), num_modes=4)
    params = _standing_params(res)
    assert len(params) >= 3
    for mode, v in zip(params, res.normal.eigenvectors):
        v_m_v = np.real(np.vdot(v, res.M_global @ v))
        assert mode.stored_energy == pytest.approx(np.pi * EPS0 * v_m_v, rel=1e-3, abs=0)


def test_tm010_q_agrees_with_tm0_solver(meshes):
    f_ghz, _, _ = _pillbox_ref("TM", 0, 1, 0)
    res, i = _standing(meshes["pillbox"], 0, f_ghz)
    mesh = load_mesh(meshes["pillbox"], 2)
    tm0 = compute_tm0_parameters(solve_tm0_standing(mesh, num_modes=4),
                                 classify_boundaries(mesh.physical_groups, mesh.nodes), SIGMA,
                                 n_scan=300).per_phase[0.0]
    q_tm0 = min(tm0, key=lambda m: abs(m.frequency_ghz - f_ghz)).q_factor
    assert _standing_params(res)[i].q_factor == pytest.approx(q_tm0, rel=0.01, abs=0)


@pytest.mark.parametrize("n", [0, 1])
def test_standing_matches_periodic_cell_at_zero_phase(tmp_path, n):
    """θ=0 の周期セル ≡ 端面 E-short の定在波（TM_n10）. Q も U も同じ（修正前は定在波だけ 2 倍）."""
    x = _zero("TM", n, 1)
    f_ghz = x * C0 / (RADIUS * 2e9 * np.pi)
    standing = _write(_rect_geom("E-short", 3.0), tmp_path / "sw.msh")
    periodic = _write(_rect_geom("None", 3.0), tmp_path / "tw.msh")
    res_sw, i = _standing(standing, n, f_ghz)
    sw = _standing_params(res_sw)[i]
    res_tw = solve_hom_traveling(load_mesh_hom(periodic, n, 2), num_modes=4, phase_shifts=(0.0,),
                                 target_freq_ghz=f_ghz * 1.01)
    tw = compute_hom_parameters(res_tw, conductivity=SIGMA).per_phase[0.0][_nearest(res_tw.periodic[0.0].frequencies, f_ghz)]
    assert sw.q_factor == pytest.approx(tw.q_factor, rel=0.01, abs=0)
    assert sw.stored_energy == pytest.approx(tw.stored_energy, rel=1e-3, abs=0)


def test_uniform_dielectric_q_wall_p_diel_and_total_q(tmp_path):
    """ε_r=4, tanδ=1e-4 一様充填の TM010. 修正前は Q_wall と P_diel が 2 倍で、合計 Q の配分もずれる."""
    eps_r, tand = 4.0, 1e-4
    geom = _rect_geom("PEC", 3.0, material="diel", eps_r=eps_r, tan_delta=tand)
    table = build_material_table_from_geometry(geom)
    mesh = load_mesh_hom(_write(geom, tmp_path / "diel.msh"), 0, 2)
    omega = _zero("TM", 0, 1) * C0 / (RADIUS * np.sqrt(eps_r))
    f_ghz = omega / (2e9 * np.pi)
    res = solve_hom_standing(mesh, num_modes=3, material_table=table, target_freq_ghz=f_ghz * 1.01)
    i = _nearest(res.normal.frequencies, f_ghz)
    mode = _standing_params(res, eps_r_per_element=build_eps_r_per_element(mesh, table),
                            tan_delta_per_element=build_tan_delta_per_element(mesh, table))[i]
    q_wall = omega * MU0 * RADIUS * LENGTH / (2.0 * _rs(omega) * (RADIUS + LENGTH))
    assert mode.q_diel == pytest.approx(1.0 / tand, rel=1e-3, abs=0)
    assert mode.q_wall == pytest.approx(q_wall, rel=0.01, abs=0)
    assert mode.q_factor == pytest.approx(1.0 / (1.0 / q_wall + tand), rel=0.01, abs=0)
    v = res.normal.eigenvectors[i]
    u_true = np.pi * EPS0 * np.real(np.vdot(v, res.M_global @ v))    # M は ε_r 込み
    assert mode.p_diel == pytest.approx(omega * tand * u_true, rel=1e-3, abs=0)


# ---------------------------------------------------------------------------
# 4) 解析式を使わない判定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ka, orders", [(2.7437, (0, 1)), (3.8702, (0, 1, 2))], ids=["TM_l1", "TM_l2"])
def test_sphere_q_does_not_depend_on_n(meshes, ka, orders):
    """同じ l のモードは回転で移り合う. n=0 は余分な j の経路を通らない（修正前は n=1, 2 が +25〜67%）."""
    f_ghz = (ka / (SPHERE_RADIUS_MM * 1e-3)) * C0 / (2e9 * np.pi)
    q = {}
    for n in orders:
        res, i = _standing(meshes["sphere"], n, f_ghz)
        q[n] = _standing_params(res)[i].q_wall
    for n in orders[1:]:
        assert q[n] == pytest.approx(q[0], rel=0.02, abs=0), q


def _direct_wall_loss(res, recon, omega, time_phases=(0.0,)):
    """``field_recon`` の H の接線成分を壁に沿って 5 点 Gauss で積分した P_loss（進行波は時間位相 0°, 90° の和が |H|²）."""
    vertices = np.asarray(res.vertices)
    inverse = {v: k for k, v in res.edge_index_map.items()}
    wall = {inverse[i] for i in res.pec_loss_edge_indices}
    gx, gw = np.polynomial.legendre.leggauss(5)
    total = 0.0
    for e, simplex in enumerate(np.asarray(res.simplices)):
        corner = simplex[:3]
        for i in range(3):
            n1, n2 = int(corner[i]), int(corner[(i + 1) % 3])
            if tuple(sorted((n1, n2))) not in wall:
                continue
            p1, p2 = vertices[n1], vertices[n2]
            length = np.linalg.norm(p2 - p1)
            t = (p2 - p1) / length
            for xi, w in zip(gx, gw):
                z, r = p1 + 0.5 * (xi + 1.0) * (p2 - p1)
                for phase in time_phases:
                    f = recon.calculate_fields(z, r, phase, elem_idx=e)
                    total += ((f["Hz"] * t[0] + f["Hr"] * t[1]) ** 2 + f["H_theta"] ** 2) * r * 0.5 * w * length
    return np.pi * _rs(omega) * total


@pytest.mark.parametrize("n", [1, 2])
def test_wall_loss_equals_direct_integral_of_recon_h(sample_mesh, n):
    """解析解の無い形状（曲線と斜めの壁）: ``field_recon`` の H の 5 点 Gauss 直接積分と一致する."""
    res = solve_hom_standing(load_mesh_hom(str(sample_mesh("s-band_1cell", 2)), n, 2), num_modes=3)
    params = _standing_params(res)
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, n, "standing")
    for m, mp in enumerate(params):
        recon.load_mode(res.normal.eigenvectors[m], res.normal.frequencies[m])
        # post は軸上の E_θ を置き換えない解を使う（置き換えると解析解との差はほぼ同じで Q がわずかに悪くなる）ので揃える
        recon.E_theta = split_hom_dofs(res.normal.eigenvectors[m], res.num_edges, len(res.simplices), len(res.vertices), 2, n,
                                       res.vertices, res.simplices)[3]
        direct = _direct_wall_loss(res, recon, 2.0 * np.pi * res.normal.frequencies[m] * 1e9)
        assert mp.p_loss == pytest.approx(direct, rel=2e-4, abs=0)   # 実測の最大は 2.2e-5


@pytest.mark.parametrize("n", [1, 2])
def test_traveling_wall_loss_equals_direct_integral_of_recon_h(tmp_path, n):
    """進行波（複素の固有ベクトル、周期面に達する壁）でも ``field_recon`` の H の直接積分と一致する."""
    geom = MultiRegionGeometry.load_json(Path(__file__).resolve().parent.parent / "samples" / "s-band_1cell.gmshproj")
    for end in (1, 7):
        geom.segments[end].bc_name = "E-short"
    res = solve_hom_traveling(load_mesh_hom(_write(geom, tmp_path / "cell.msh"), n, 2), num_modes=4, phase_shifts=(THETA,))
    modes = res.periodic[THETA]
    params = compute_hom_parameters(res, conductivity=SIGMA).per_phase[THETA]
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, n, "traveling")
    for m, mp in enumerate(params):
        recon.load_mode(modes.eigenvectors[m], modes.frequencies[m])
        direct = _direct_wall_loss(res, recon, 2.0 * np.pi * modes.frequencies[m] * 1e9, time_phases=(0.0, 90.0))
        assert mp.p_loss == pytest.approx(direct, rel=1e-6, abs=0)   # 実測の最大は 5e-9


# ---------------------------------------------------------------------------
# 4b) 軸上（r = 0）の H_r: n = 1 の軸上の H は一様な横向きの場なので |H_r| = |H_φ|（コアの変数で H_r = −H_θ）。n ≥ 2 は 0
#     解析解: E_z = E0 J1(k_c r) の TM は |H_r(0)| = ωε₀E0/(2k_c)、H_z = H0 J1(k_c r) の TE は |H_r(0)| = βH0/(2k_c)
# ---------------------------------------------------------------------------
AXIS_R = 1e-10              # GUI の節点評価（calculate_all_node_fields）と同じ。r = 0 ちょうどの点探索は領域外になることがある


def _amplitudes(recon, z, r):
    a = recon.calculate_fields(z, r, 0.0)
    if recon.is_standing:
        return a
    b = recon.calculate_fields(z, r, 90.0)
    return {k: a[k] - 1j * b[k] for k in a}


def _traveling_recon(path, kind, n, m):
    ref = _waveguide_ref(kind, n, m)
    res = _solve_waveguide(path, kind, n, m)
    modes = res.periodic[THETA]
    i = _nearest(modes.frequencies, ref["f_ghz"])
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, n, "traveling")
    return ref, recon.load_mode(modes.eigenvectors[i], modes.frequencies[i])


def _axis_zs():
    return np.linspace(0.0, LENGTH, 9)[1:-1]


def test_on_axis_hr_pillbox_tm110(meshes):
    f_ghz, omega, _ = _pillbox_ref("TM", 1, 1, 0)
    res, i = _standing(meshes["pillbox"], 1, f_ghz)
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "standing")
    recon.load_mode(res.normal.eigenvectors[i], res.normal.frequencies[i])
    kc = _zero("TM", 1, 1) / RADIUS
    r0 = 0.46 * RADIUS                                    # J1 の最大の近く
    e0 = abs(_amplitudes(recon, LENGTH / 2, r0)["Ez"]) / abs(j1(kc * r0))
    ref = omega * EPS0 / (2.0 * kc) * e0
    for z in _axis_zs():
        # 実測の誤差は h = 2 mm で 0.35% 以下（メッシュで揺れる）。3.0.0 はほぼ 0（E_z の項を落としていた）
        assert abs(_amplitudes(recon, z, AXIS_R)["Hr"]) == pytest.approx(ref, rel=0.01, abs=0)


@pytest.mark.parametrize("kind", ["TM", "TE"])
def test_on_axis_hr_waveguide_n1(meshes, kind):
    ref, recon = _traveling_recon(meshes["waveguide"], kind, 1, 1)
    r0 = 0.5 * RADIUS
    mid = _amplitudes(recon, LENGTH / 2, r0)
    if kind == "TM":
        expected = ref["omega"] * EPS0 / (2.0 * ref["kc"]) * abs(mid["Ez"]) / abs(j1(ref["kc"] * r0))
    else:
        expected = ref["beta"] / (2.0 * ref["kc"]) * abs(mid["Hz"]) / abs(j1(ref["kc"] * r0))
    for z in _axis_zs():
        # 実測の誤差は h = 2 mm で 0.35% 以下（メッシュで揺れる）。3.0.0 は TM11 で真値の 0.25〜0.29 倍、TE11 で最大 10% の誤差
        assert abs(_amplitudes(recon, z, AXIS_R)["Hr"]) == pytest.approx(expected, rel=0.01, abs=0)


@pytest.mark.parametrize("kind", ["TM", "TE"])
def test_on_axis_hr_has_the_sign_and_phase_of_the_off_axis_field(meshes, kind):
    """振幅だけでは符号を見られない: 軸上と軸から離れた点の複素比を解析解と比べる.

    軸から離れた点は curl の式で、軸上は正則性（H_r = −H_θ）で求めるので、符号が逆なら比が −1 倍になる。
    """
    ref, recon = _traveling_recon(meshes["waveguide"], kind, 1, 1)
    r0 = 0.25 * RADIUS
    x = ref["kc"] * r0
    profile = 2.0 * j1(x) / x if kind == "TM" else 2.0 * jvp(1, x)      # H_r(r0) / H_r(0)
    for z in _axis_zs():
        ratio = _amplitudes(recon, z, r0)["Hr"] / _amplitudes(recon, z, AXIS_R)["Hr"]
        assert abs(ratio - profile) < 0.03, (z, ratio, profile)


def test_on_axis_hr_of_a_standing_mode_has_the_sign_of_the_off_axis_field(meshes):
    f_ghz, _, _ = _pillbox_ref("TM", 1, 1, 0)
    res, i = _standing(meshes["pillbox"], 1, f_ghz)
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "standing")
    recon.load_mode(res.normal.eigenvectors[i], res.normal.frequencies[i])
    r0 = 0.25 * RADIUS
    x = _zero("TM", 1, 1) / RADIUS * r0
    for z in _axis_zs():
        ratio = _amplitudes(recon, z, r0)["Hr"] / _amplitudes(recon, z, AXIS_R)["Hr"]
        assert abs(ratio - 2.0 * j1(x) / x) < 0.03, (z, ratio)


def test_on_axis_hr_is_zero_for_n2(meshes):
    _, recon = _traveling_recon(meshes["waveguide"], "TM", 2, 1)
    for z in _axis_zs():
        assert _amplitudes(recon, z, AXIS_R)["Hr"] == 0.0


@pytest.mark.parametrize("kind", ["TM", "TE"])
def test_on_axis_e_theta_equals_e_r_for_n1(meshes, kind):
    """n = 1 の軸上は一様な横向きの場なので E_θ = E_r（3.0.0 は隣接節点の平均で 3〜16% 違った）."""
    _, recon = _traveling_recon(meshes["waveguide"], kind, 1, 1)
    values = [_amplitudes(recon, z, AXIS_R) for z in np.linspace(0.0, LENGTH, 41)[1:-1]]
    peak = max(abs(v["Er"]) for v in values)
    assert peak > 0.0
    for v in values:
        assert abs(v["E_theta"] - v["Er"]) < 0.01 * peak


@pytest.mark.parametrize("r_mm", [1e-6, 1e-3, 0.05, 0.1, 0.11, 0.14, 0.17, 0.2, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("kind", ["TM", "TE"])
def test_fields_very_close_to_the_axis_stay_regular(meshes, kind, r_mm):
    """軸のごく近くでも H_z, H_r は発散しない（3.0.0: TE11 の H_z は r = 1 µm で最大 H の 500 倍以上、0.1 mm で 4 倍、1 mm で 32%）."""
    ref, recon = _traveling_recon(meshes["waveguide"], kind, 1, 1)
    r = r_mm * 1e-3
    x = ref["kc"] * r
    r0 = 0.25 * RADIUS
    x0 = ref["kc"] * r0
    profile = (lambda y: 2.0 * j1(y) / y) if kind == "TM" else (lambda y: 2.0 * jvp(1, y))
    zs = np.linspace(0.001, LENGTH - 0.001, 61)
    h_axis = abs(_amplitudes(recon, LENGTH / 2, r0)["Hr"]) / abs(profile(x0))
    h_z_of = (lambda y: 0.0) if kind == "TM" else (lambda y: abs(_amplitudes(recon, LENGTH / 2, r0)["Hz"]) * abs(j1(y)) / abs(j1(x0)))
    for z in zs:
        fields = _amplitudes(recon, z, r)
        assert abs(fields["Hr"]) == pytest.approx(h_axis * abs(profile(x)), abs=0.02 * h_axis)       # 実測 0.7% 以下
        assert abs(abs(fields["Hz"]) - h_z_of(x)) < 0.08 * h_axis                                    # 実測 5.6% 以下


@pytest.mark.parametrize("r_mm", [0.11, 0.14, 0.2, 0.5])
@pytest.mark.parametrize("kind", ["TM", "TE"])
def test_fields_close_to_the_axis_as_complex_numbers(meshes, kind, r_mm):
    """帯のすぐ外の H_r は振幅が合っても位相がずれる（2 mm メッシュで複素の誤差は最大 11%）."""
    ref, recon = _traveling_recon(meshes["waveguide"], kind, 1, 1)
    r0 = 0.25 * RADIUS
    profile = (lambda y: 2.0 * j1(y) / y) if kind == "TM" else (lambda y: 2.0 * jvp(1, y))
    expected = profile(ref["kc"] * r_mm * 1e-3) / profile(ref["kc"] * r0)
    for z in np.linspace(0.001, LENGTH - 0.001, 61):
        ratio = _amplitudes(recon, z, r_mm * 1e-3)["Hr"] / _amplitudes(recon, z, r0)["Hr"]
        assert abs(ratio - expected) < 0.15, (z, ratio, expected)


def test_axis_zone_does_not_apply_to_a_mesh_without_the_axis(tmp_path):
    """細い内導体（a = 0.5 mm、h = 12 mm）の壁で H_r = −H_θ にしない（壁の法線成分は 0. 帯が効くと最大の H の 62%）."""
    pts = [(0.0, 0.5), (LENGTH_MM, 0.5), (LENGTH_MM, RADIUS_MM), (0.0, RADIUS_MM)]
    bcs = ["PEC", "None", "PEC", "None"]
    segs = [Segment(id=i, type="line", point_indices=[i, (i + 1) % 4], bc_name=bc) for i, bc in enumerate(bcs)]
    geom = MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[Region(id=0, name="Cell", outer_loop_id=0, material_tag="vacuum")], unit="mm", mesh_size=12.0)
    path = tmp_path / "m.msh"
    _write(geom, path)
    res = solve_hom_traveling(load_mesh_hom(str(path), 1, 2), num_modes=1, phase_shifts=(THETA,))
    modes = res.periodic[THETA]
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "traveling")
    recon.load_mode(modes.eigenvectors[0], modes.frequencies[0])
    field = _amplitudes(recon, LENGTH / 2, 0.5e-3)
    assert abs(field["Hr"]) < 1e-3 * abs(field["H_theta"])


def test_axis_regularity_does_not_change_the_engineering_values(meshes, monkeypatch):
    """工学量は軸上の E_θ を置き換えない解（split_hom_dofs）を使うので、この修正で値は変わらない."""
    ref = _waveguide_ref("TM", 1, 1)
    res = _solve_waveguide(meshes["waveguide"], "TM", 1, 1)
    modes = res.periodic[THETA]
    i = _nearest(modes.frequencies, ref["f_ghz"])
    recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "traveling")
    recon.load_mode(modes.eigenvectors[i], modes.frequencies[i])
    plain = split_hom_dofs(modes.eigenvectors[i], res.num_edges, len(res.simplices), len(res.vertices), 2, 1, res.vertices, res.simplices)[3]
    axis = np.abs(np.asarray(res.vertices)[:, 1]) < 1e-9
    assert recon.E_theta[~axis] == pytest.approx(plain[~axis], rel=1e-14, abs=0)
    assert np.max(np.abs(recon.E_theta[axis] - plain[axis])) > 0.0
    assert np.max(np.abs(plain[axis])) > 0.1 * np.max(np.abs(recon.E_theta[axis]))     # 工学量の側は隣接節点の平均のまま
    def must_not_run(self):
        raise AssertionError("post が軸上の置き換えを通った")

    monkeypatch.setattr(HOMFieldReconstructor, "_apply_axis_regularity", must_not_run)
    assert compute_hom_parameters(res, conductivity=SIGMA).per_phase[THETA][i].q_factor > 0.0
    monkeypatch.undo()
    # n ≥ 2 の軸上の E_θ は 0 のまま（E_r に置き換えない）
    two = _waveguide_ref("TE", 2, 1)
    res2 = _solve_waveguide(meshes["waveguide"], "TE", 2, 1)
    modes2 = res2.periodic[THETA]
    k = _nearest(modes2.frequencies, two["f_ghz"])
    recon2 = HOMFieldReconstructor(res2.simplices, res2.vertices, res2.edge_index_map, 2, 2, "traveling")
    recon2.load_mode(modes2.eigenvectors[k], modes2.frequencies[k])
    assert np.all(recon2.E_theta[axis] == 0.0)


def test_on_axis_hr_of_a_hybrid_mode_is_stable_with_the_mesh(sample_mesh, tmp_path):
    """解析解の無いハイブリッドモード（アイリス付きセル）: 軸上の値はメッシュを細かくしても変わらない（3.0.0 は真値の最大 2.5 倍）.

    軸から離れた点の値はメッシュで大きく揺れる（r = 2.5 mm, h = 4.5 mm で 0.11、h = 5 mm で 0.06）ので基準にしない。
    """
    fine = MultiRegionGeometry.load_json(Path(__file__).resolve().parents[1] / "samples" / "s-band_1cell.gmshproj")
    fine.mesh_size = 2.5
    fine_path = _write(fine, tmp_path / "fine.msh")
    peak = []
    for path in (str(sample_mesh("s-band_1cell", 2)), fine_path):
        res = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=2, phase_shifts=(THETA,))
        modes = res.periodic[THETA]
        recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "traveling")
        z_range = np.asarray(res.vertices)[:, 0]
        zs = np.linspace(z_range.min(), z_range.max(), 9)[1:-1]
        values = []
        for m in range(2):
            recon.load_mode(modes.eigenvectors[m], modes.frequencies[m])
            values.append(max(abs(_amplitudes(recon, z, AXIS_R)["Hr"]) for z in zs))
        peak.append(values)
    assert peak[0] == pytest.approx(peak[1], rel=0.05, abs=0)        # 実測 h = 5 と 2.5 mm の差は 2.4% 程度


def test_on_axis_export_and_nodes_use_the_same_hr(tmp_path):
    """``export --shape axis`` と節点評価（場の図・3D）の軸上の点も H_r = −H_θ."""
    f_ghz, _, _ = _pillbox_ref("TM", 1, 1, 0)
    proc = _post_cli(tmp_path / "m", "PEC", 1, f_ghz)
    from axicavity_fem.shared.hdf5_io import read_results
    rec = read_results(proc)["results_by_n"][1]["standing"]
    k = _nearest(rec["frequencies"], f_ghz)
    data = export_fields(str(proc), str(tmp_path / "axis"), solver_type="hom", n=1, mode=k, shape="axis", npts=41, fmt="h5")
    inside = data["mask"]
    assert inside.sum() >= 39
    assert data["Hr"][inside] == pytest.approx(-data["H_theta"][inside], rel=1e-12, abs=0)
    assert np.max(np.abs(data["Hr"][inside])) > 0.0

    mesh = read_results(proc)["mesh"]
    recon = HOMFieldReconstructor(mesh["simplices"], mesh["vertices"], mesh["edge_index_map"], 2, 1, "standing")
    recon.load_mode(rec["eigenvectors"][k], rec["frequencies"][k])
    nodes = recon.calculate_all_node_fields()
    axis = np.abs(np.asarray(mesh["vertices"])[:, 1]) < 1e-9
    assert nodes["Hr"][axis] == pytest.approx(-nodes["H_theta"][axis], rel=1e-12, abs=0)
    assert np.max(np.abs(nodes["Hr"][axis])) > 0.0


@pytest.mark.parametrize("traveling", [False, True], ids=["standing", "traveling"])
def test_3d_field_is_single_valued_on_the_axis(meshes, traveling):
    """回転体の軸上の点は 1 点なので、3D の H はどの φ のリングでも同じベクトルになる（3.0.0 は φ で変わった）."""
    if traveling:
        _, recon = _traveling_recon(meshes["waveguide"], "TM", 1, 1)
        f0, f90 = recon.calculate_all_node_fields(0.0), recon.calculate_all_node_fields(90.0)
        amp = {k: f0[k] - 1j * f90[k] for k in f0}
    else:
        f_ghz, _, _ = _pillbox_ref("TM", 1, 1, 0)
        res, i = _standing(meshes["pillbox"], 1, f_ghz)
        recon = HOMFieldReconstructor(res.simplices, res.vertices, res.edge_index_map, 2, 1, "standing")
        recon.load_mode(res.normal.eigenvectors[i], res.normal.frequencies[i])
        amp = recon.calculate_all_node_fields(0.0)
    fields = ModeFields(n=1, traveling=traveling, Ez=amp["Ez"], Er=amp["Er"], Ephi=amp["E_theta"], Hz=amp["Hz"], Hr=amp["Hr"],
                        Hphi=amp["H_theta"], is_hom=True, freq_ghz=recon.freq_GHz)
    vertices = np.asarray(recon.vertices)
    grid = revolve_mesh(vertices, split_triangles(recon.simplices, 2), n_phi=12)
    h = field_point_arrays(fields, grid, time_phase_deg=30.0)["H"]
    axis = np.nonzero(np.abs(vertices[:, 1]) < 1e-9)[0]
    rings = h.reshape(grid.ring_count, grid.node_count, 3)[:, axis, :]
    assert np.max(np.abs(rings)) > 0.0
    assert np.max(np.abs(rings - rings[0])) < 1e-9 * np.max(np.abs(h))


# ---------------------------------------------------------------------------
# 5) 目標壁損失（export）: HOM の倍率は post の P_loss をそのまま使うので、_wall_loss が正しければ書き出した場の壁損失が目標になる
# ---------------------------------------------------------------------------
def _post_cli(folder, end_bc, n, f_ghz, extra=()):
    folder.mkdir(parents=True, exist_ok=True)
    msh, raw, proc = folder / "m.msh", folder / "raw.h5", folder / "proc.h5"
    _write(_rect_geom(end_bc, 3.0), msh)
    assert cli_main(["solve", "--type", "hom", "-m", str(msh), "--elem-order", "2", "--num-modes", "3", "--az-order",
                     str(n), "--target-freq", f"{f_ghz * 1.01:.5f}", *extra, "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "hom", "-i", str(raw), "-o", str(proc), "--cond", str(SIGMA)]) == 0
    return proc


def _wall_power(data, omega, comps):
    assert data["mask"].all()
    h2 = sum(np.abs(data[c]) ** 2 for c in comps)
    return np.pi * _rs(omega) * trapezoid(h2 * data["r"], x=data["distance"])


def _line(proc, tmp_path, p1, p2, mode, n, **kw):
    return export_fields(str(proc), str(tmp_path / "line"), solver_type="hom", n=n, mode=mode, shape="line", p1=p1, p2=p2,
                         npts=1001, scale_to_power=TARGET, fmt="h5", **kw)


@pytest.mark.parametrize("kind, n, m, p", [("TE", 1, 1, 1), ("TM", 1, 1, 1)], ids=["TE111", "TM111"])
def test_scale_to_power_standing_wall_integral(tmp_path, kind, n, m, p):
    f_ghz, omega, _ = _pillbox_ref(kind, n, m, p)
    proc = _post_cli(tmp_path / "m", "PEC", n, f_ghz)
    from axicavity_fem.shared.hdf5_io import read_results
    k = _nearest(read_results(proc)["results_by_n"][n]["standing"]["frequencies"], f_ghz)
    side = _line(proc, tmp_path, (EPS, RADIUS - EPS), (LENGTH - EPS, RADIUS - EPS), k, n)
    ends = [_line(proc, tmp_path, (z, 1e-5), (z, RADIUS - EPS), k, n) for z in (EPS, LENGTH - EPS)]
    power = _wall_power(side, omega, ("Hz", "H_theta")) + sum(_wall_power(d, omega, ("Hr", "H_theta")) for d in ends)
    assert power == pytest.approx(TARGET, rel=2e-3, abs=0)


@pytest.mark.parametrize("kind, n, m", [("TE", 1, 1), ("TM", 1, 1)], ids=["TE11", "TM11"])
def test_scale_to_power_traveling_wall_integral(tmp_path, kind, n, m):
    """HOM の進行波の書き出しは瞬時値. θ=0° と 90° の 2 乗和が複素振幅の 2 乗."""
    ref = _waveguide_ref(kind, n, m)
    proc = _post_cli(tmp_path / "m", "None", n, ref["f_ghz"], extra=("-p", f"{THETA:g}"))
    from axicavity_fem.shared.hdf5_io import read_results
    k = _nearest(read_results(proc)["results_by_n"][n]["traveling"][THETA]["frequencies"], ref["f_ghz"])
    power = 0.0
    for time_phase in (0.0, 90.0):
        side = _line(proc, tmp_path, (EPS, RADIUS - EPS), (LENGTH - EPS, RADIUS - EPS), k, n, phase=THETA,
                     time_phase=time_phase)
        power += _wall_power(side, ref["omega"], ("Hz", "H_theta"))
    assert power == pytest.approx(TARGET, rel=2e-3, abs=0)
