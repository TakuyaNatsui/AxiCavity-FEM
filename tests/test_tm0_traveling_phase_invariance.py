"""進行波 TM0 の post が固有ベクトルの全体位相によらず、正しい値を出すことの回帰テスト.

バグ（3.0.0 まで）: 軸上 E_z の実部（瞬時値）で規格化と電圧積分をしていたため、ARPACK の初期ベクトルで
決まる全体位相 e^{jφ} によって U, P_loss, P_diel, P_flow, V_eff, V_acc, R/Q が変わっていた（Q, v_g は比なので不変）。
SciPy 1.16 では新しいプロセスの最初の解は毎回同じ位相なので、値が再現しても正しいとは限らなかった（SciPy 1.18 では実行のたびに変わる）。

照合は 3 種類:
- 解析解: 中空円筒導波管 TM01 の周期セル（半径 R、長さ L、位相 θ）。
    β_z = θ/L, k_c = x01/R, ω²/c² = k_c² + β_z²
    E_z = J0(k_c r) e^{-jβ_z z}, H_φ = (ωε₀/k_c) J1(k_c r) e^{-jβ_z z}（軸上 |E_z| = 1）
    U = (μ₀/2)(ωε₀/k_c)² · 2π L · (R²/2) J1(x01)²
- 定在波ソルバ（固有ベクトルが実数で、このバグと無関係）: θ=0/180° は端面 E-short / M-short の 1 セルと一致し、
  θ=120° の前進波は 3 セル E-short の 2π/3 モードと R/Q_SW = 1.5 × R/Q_TW（kL = θ の厳密な同期 β）で結びつく。
- ピルボックス TM010 の解析解: 定在波の経路が変わっていないことの確認。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j1, jn_zeros

pytest.importorskip("gmsh")

from axicavity_fem.cli.main import cli_main  # noqa: E402
from axicavity_fem.fem_tm0.field_recon import TM0FieldReconstructor  # noqa: E402
from axicavity_fem.fem_tm0.post_process import compute_tm0_parameters  # noqa: E402
from axicavity_fem.fem_tm0.solver import solve_tm0_standing, solve_tm0_traveling  # noqa: E402
from axicavity_fem.shared.boundary_groups import classify_boundaries  # noqa: E402
from axicavity_fem.shared.constants import C0, EPS0, MU0  # noqa: E402
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region  # noqa: E402
from axicavity_fem.shared.hdf5_io import read_results  # noqa: E402
from axicavity_fem.shared.material_resolver import (  # noqa: E402
    build_eps_r_per_element,
    build_material_table_from_geometry,
    build_tan_delta_per_element,
)
from axicavity_fem.shared.mesh_loader import MeshData, load_mesh  # noqa: E402
from axicavity_fem.shared.multi_region_model import (  # noqa: E402
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)

LENGTH_MM, RADIUS_MM = 35.0, 40.0
LENGTH, RADIUS = LENGTH_MM * 1e-3, RADIUS_MM * 1e-3
THETA = 120.0
N_SCAN = 500
ALL_KEYS = ("rq_eff", "rq_apparent", "v_eff", "v_acc", "stored_energy", "p_loss", "p_flow",
            "q_factor", "q_wall", "q_diel", "p_diel", "v_group_c", "attenuation", "r_shunt_m")


# ---------------------------------------------------------------------------
# メッシュ（test_dielectric_loss.py の _pillbox_geom と同じ作り方）
# ---------------------------------------------------------------------------
def _waveguide_geom(mesh_size=2.0, inner_mm=0.0, end_bc="None", eps_r=1.0, tan_delta=0.0, material="vacuum"):
    """長さ L、半径 R の周期セル。inner_mm > 0 なら軸を含まない同軸."""
    pts = [(0.0, inner_mm), (LENGTH_MM, inner_mm), (LENGTH_MM, RADIUS_MM), (0.0, RADIUS_MM)]
    inner_bc = "PEC" if inner_mm > 0 else "None"
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name=inner_bc),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name=end_bc),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name=end_bc)]
    region = Region(id=0, name="Cell", outer_loop_id=0, material_tag=material, eps_r=eps_r, tan_delta=tan_delta)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[region], unit="mm", mesh_size=mesh_size)


def _sleeve_geom(sleeve_r_mm=30.0, mesh_size=3.0):
    """軸側が真空、外側が誘電体（ε_r = 2.5、tanδ = 0.01）の 2 領域セル."""
    L, a, b = LENGTH_MM, sleeve_r_mm, RADIUS_MM
    pts = [(0.0, 0.0), (L, 0.0), (L, a), (0.0, a), (L, b), (0.0, b)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="None"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="None"),
            Segment(id=4, type="line", point_indices=[2, 4], bc_name="None"),
            Segment(id=5, type="line", point_indices=[4, 5], bc_name="PEC"),
            Segment(id=6, type="line", point_indices=[5, 3], bc_name="None")]
    loops = [Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 2])]
    regions = [Region(id=0, name="Vac", outer_loop_id=0, material_tag="vacuum"),
               Region(id=1, name="Sleeve", outer_loop_id=1, material_tag="diel", eps_r=2.5, tan_delta=0.01)]
    return MultiRegionGeometry(points=pts, segments=segs, loops=loops, regions=regions, unit="mm", mesh_size=mesh_size)


def _mesh_from(geom, path, order=2):
    export_msh_multi_region(geom, path, mesh_order=order, verbose=0)
    return load_mesh(str(path), order)


@pytest.fixture(scope="module")
def waveguide(tmp_path_factory):
    mesh = _mesh_from(_waveguide_geom(), tmp_path_factory.mktemp("waveguide") / "wg.msh")
    return mesh, solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(THETA,))


@pytest.fixture(scope="module")
def sband(sample_mesh):
    return load_mesh(str(sample_mesh("s-band_1cell", 2)), 2)


# ---------------------------------------------------------------------------
# 共通処理
# ---------------------------------------------------------------------------
def _post(mesh, result, theta=THETA, beta=1.0, **kw):
    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    return compute_tm0_parameters(result, classification, beta=beta, n_scan=N_SCAN, **kw).per_phase[theta]


def _rotated(result, theta, degrees):
    """固有ベクトルに全体位相を掛ける（degrees はモードごとに指定できる）."""
    rec = result.phase_results[theta]
    ev = np.array(rec["eigenvectors"]) * np.exp(1j * np.radians(np.atleast_1d(degrees)))[:, None]
    phases = dict(result.phase_results)
    phases[theta] = {"frequencies": rec["frequencies"], "eigenvectors": ev}
    return replace(result, phase_results=phases)


def _assert_same(a, b, rel=1e-8, keys=ALL_KEYS):
    for name in keys:
        assert getattr(b, name) == pytest.approx(getattr(a, name), rel=rel, abs=1e-30), name


def _analytic(theta=THETA, beta=1.0):
    x01 = jn_zeros(0, 1)[0]
    kc = x01 / RADIUS
    bz = np.radians(theta) / LENGTH
    omega = np.hypot(kc, bz) * C0
    u = (MU0 / 2) * (omega * EPS0 / kc) ** 2 * 2 * np.pi * LENGTH * RADIUS ** 2 / 2 * j1(x01) ** 2
    v_eff = LENGTH * abs(np.sinc((omega / (beta * C0) - bz) * LENGTH / (2 * np.pi)))   # np.sinc(x) = sin(πx)/(πx)
    v_acc = LENGTH * abs(np.sinc(bz * LENGTH / (2 * np.pi)))
    return omega, bz, u, v_eff, v_acc


# ---------------------------------------------------------------------------
# 全体位相への不変性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degrees", [37.0, 90.0, 133.0])
def test_post_does_not_depend_on_global_phase(waveguide, degrees):
    mesh, result = waveguide
    _assert_same(_post(mesh, result)[0], _post(mesh, _rotated(result, THETA, degrees))[0])


def test_several_modes_and_phases_each_with_its_own_phase(sband):
    thetas = (60.0, 120.0)
    result = solve_tm0_traveling(sband, num_modes=3, phase_shifts=thetas)
    for theta in thetas:
        n = len(result.phase_results[theta]["frequencies"])
        rotated = _rotated(result, theta, np.linspace(17.0, 150.0, n))
        for a, b in zip(_post(sband, result, theta), _post(sband, rotated, theta)):
            _assert_same(a, b)


def test_first_order_elements(tmp_path):
    mesh = _mesh_from(_waveguide_geom(mesh_size=1.5), tmp_path / "wg1.msh", order=1)
    result = solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(THETA,))
    omega, _, u_ana, v_ana, _ = _analytic()
    ref = _post(mesh, result)[0]
    assert ref.rq_eff == pytest.approx(v_ana ** 2 / (omega * u_ana), rel=0.01)
    _assert_same(ref, _post(mesh, _rotated(result, THETA, 73.0))[0])


def test_dielectric_and_tan_delta(tmp_path):
    geom = _sleeve_geom()
    mesh = _mesh_from(geom, tmp_path / "sleeve.msh")
    table = build_material_table_from_geometry(geom)
    result = solve_tm0_traveling(mesh, num_modes=2, phase_shifts=(THETA,), material_table=table)
    kw = dict(eps_r_per_element=build_eps_r_per_element(mesh, table),
              tan_delta_per_element=build_tan_delta_per_element(mesh, table))
    for a, b in zip(_post(mesh, result, **kw), _post(mesh, _rotated(result, THETA, [37.0, 133.0]), **kw)):
        assert a.p_diel > 0
        assert 1 / a.q_factor == pytest.approx(1 / a.q_wall + 1 / a.q_diel, rel=1e-12)
        _assert_same(a, b)


# ---------------------------------------------------------------------------
# 規格化（PHYSICS_AND_CONVENTIONS.md: max_z |E_z(r=0)| = 1 V/m）
# ---------------------------------------------------------------------------
def test_normalized_to_unit_peak_of_complex_ez(waveguide):
    """新しい引数に依存しない形にしてあるので、修正前は TypeError でなく値で失敗する."""
    mesh, result = waveguide
    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    params = compute_tm0_parameters(result, classification, n_scan=N_SCAN)
    u = params.normalized_eigenvectors[THETA][0]
    f = params.per_phase[THETA][0].frequency_ghz
    recon = TM0FieldReconstructor(mesh.nodes, mesh.elements, mesh.element_order, "traveling")
    zs = np.linspace(mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max(), N_SCAN)
    ez = np.array([recon.calculate_fields(u, f, z, 0.0, return_complex=True)["Ez"] for z in zs])
    assert np.max(np.abs(ez)) == pytest.approx(1.0, rel=1e-10)


def test_axial_scan_default_stays_real(waveguide):
    """axial_ez_scan の既定は従来どおり実部（定在波の経路を変えない）."""
    mesh, result = waveguide
    rec = result.phase_results[THETA]
    recon = TM0FieldReconstructor(mesh.nodes, mesh.elements, mesh.element_order, "traveling")
    _, ez = recon.axial_ez_scan(rec["eigenvectors"][0], rec["frequencies"][0], n_scan=20)
    assert np.isrealobj(ez)


def test_axis_outside_the_mesh_keeps_raw_scale(tmp_path):
    """軸がメッシュに無いとき（同軸）は全点領域外で、従来どおり規格化しない."""
    mesh = _mesh_from(_waveguide_geom(inner_mm=5.0), tmp_path / "coax.msh")
    result = solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(THETA,))
    mode = _post(mesh, result)[0]
    assert mode.norm_factor == 1.0 and mode.rq_eff == 0.0 and mode.v_eff == 0.0
    assert mode.q_factor > 0 and np.isfinite(mode.stored_energy)


# ---------------------------------------------------------------------------
# 値の照合: 解析解
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta_kind", ["light", "half", "synchronous"])
def test_r_over_q_matches_waveguide_analytic(waveguide, beta_kind):
    """導波管 TM01 の解析解に一致する（R/Q は 1% 以内。修正前は 50% 以上ずれる）."""
    mesh, result = waveguide
    omega, bz, _, _, _ = _analytic()
    beta = {"light": 1.0, "half": 0.5, "synchronous": omega / (C0 * bz)}[beta_kind]
    omega, bz, u_ana, v_ana, v_acc_ana = _analytic(beta=beta)
    mode = _post(mesh, result, beta=beta)[0]
    assert mode.frequency_ghz * 1e9 * 2 * np.pi == pytest.approx(omega, rel=1e-4)
    assert mode.rq_eff == pytest.approx(v_ana ** 2 / (omega * u_ana), rel=0.01)
    assert mode.stored_energy == pytest.approx(u_ana, rel=0.02, abs=0)
    assert mode.v_acc == pytest.approx(v_acc_ana, rel=0.01)
    assert mode.v_group_c * C0 == pytest.approx(C0 ** 2 * bz / omega, rel=0.01)
    sigma = 5.8e7
    rs = np.sqrt(omega * MU0 / (2 * sigma))
    q_ana = omega * MU0 * RADIUS / (2 * rs)
    v_g = C0 ** 2 * bz / omega
    assert mode.q_factor == pytest.approx(q_ana, rel=0.01)
    assert mode.p_loss == pytest.approx(omega * u_ana / q_ana, rel=0.03, abs=0)
    assert mode.p_flow == pytest.approx(v_g * u_ana / LENGTH, rel=0.03, abs=0)
    assert mode.attenuation == pytest.approx(omega / (2 * q_ana * v_g), rel=0.02)
    assert mode.r_shunt_m == pytest.approx(v_ana ** 2 / (omega * u_ana / q_ana * LENGTH), rel=0.02)


def test_v_acc_is_the_infinite_beta_limit_of_v_eff(waveguide):
    mesh, result = waveguide
    mode = _post(mesh, result, beta=1e9)[0]
    assert mode.v_eff == pytest.approx(mode.v_acc, rel=1e-6)


# ---------------------------------------------------------------------------
# 値の照合: 定在波ソルバ（空間高調波を含む周期構造）
# ---------------------------------------------------------------------------
def _with_end_bc(mesh, name):
    z = mesh.nodes[:, 0]
    ends = np.where((np.abs(z - z.min()) < 1e-9) | (np.abs(z - z.max()) < 1e-9))[0]
    return MeshData(nodes=mesh.nodes, elements=mesh.elements, element_order=mesh.element_order,
                    physical_groups={"PEC": mesh.physical_groups["PEC"], name: ends})


@pytest.mark.parametrize("theta, end_bc", [(0.0, "E-short"), (180.0, "M-short")])
def test_theta_0_and_180_match_the_standing_solver(sband, theta, end_bc):
    """Re(E_z) がほぼ消える全体位相でも、端面 BC の定在波と同じ値になる（修正前は U が 1/cos²φ 倍）."""
    tw = solve_tm0_traveling(sband, num_modes=1, phase_shifts=(theta,))
    rec = tw.phase_results[theta]
    recon = TM0FieldReconstructor(sband.nodes, sband.elements, 2, "traveling")
    zs = np.linspace(sband.nodes[:, 0].min(), sband.nodes[:, 0].max(), 50)
    ez = [recon.calculate_fields(rec["eigenvectors"][0], rec["frequencies"][0], z, 0.0, return_complex=True)["Ez"]
          for z in zs]
    worst = -np.degrees(np.angle(ez[int(np.argmax(np.abs(ez)))])) + 89.0   # cos φ = 0.017
    mode = _post(sband, _rotated(tw, theta, worst), theta)[0]

    mesh_sw = _with_end_bc(sband, end_bc)
    sw = _post(mesh_sw, solve_tm0_standing(mesh_sw, num_modes=4), 0.0)
    ref = min(sw, key=lambda m: abs(m.frequency_ghz - mode.frequency_ghz))
    assert ref.frequency_ghz == pytest.approx(mode.frequency_ghz, rel=1e-5)
    for name in ("rq_eff", "stored_energy", "v_eff", "q_factor"):
        assert getattr(mode, name) == pytest.approx(getattr(ref, name), rel=1e-3, abs=0), name


def _without_port_pec(mesh):
    """端面（r < iris の部分）を PEC から外す。周期セルの端面は壁ではない."""
    pec = mesh.physical_groups["PEC"]
    z, r = mesh.nodes[pec, 0], mesh.nodes[pec, 1]
    on_end = (np.abs(z - mesh.nodes[:, 0].min()) < 1e-9) | (np.abs(z - mesh.nodes[:, 0].max()) < 1e-9)
    tip = r[on_end].max()
    return MeshData(nodes=mesh.nodes, elements=mesh.elements, element_order=mesh.element_order,
                    physical_groups={"PEC": pec[~(on_end & (r < tip - 1e-9))]})


def _replicate(mesh, n):
    """セルを z 方向に n 個並べたメッシュ（節点は座標で共有）."""
    L = mesh.nodes[:, 0].max() - mesh.nodes[:, 0].min()
    index, nodes, maps = {}, [], []
    for c in range(n):
        m = np.zeros(len(mesh.nodes), dtype=int)
        for i, (z, r) in enumerate(mesh.nodes):
            key = (round(float(z + c * L), 9), round(float(r), 9))
            if key not in index:
                index[key] = len(nodes)
                nodes.append((z + c * L, r))
            m[i] = index[key]
        maps.append(m)
    pec = np.unique(np.concatenate([m[mesh.physical_groups["PEC"]] for m in maps]))
    return MeshData(nodes=np.array(nodes), elements=np.vstack([m[mesh.elements] for m in maps]),
                    element_order=mesh.element_order, physical_groups={"PEC": pec})


def test_three_cell_standing_wave_gives_1p5_times_the_traveling_r_over_q(sband):
    """2π/3 定在波 = 前進波 + 後退波: V は 3 倍、U は 6 倍。R/Q_SW(3 セル) = 1.5 R/Q_TW(1 セル)（kL = θ の厳密な同期 β）."""
    cell = _without_port_pec(sband)
    solved = solve_tm0_traveling(cell, num_modes=1, phase_shifts=(THETA,))
    f_tw = solved.phase_results[THETA]["frequencies"][0]
    L = cell.nodes[:, 0].max() - cell.nodes[:, 0].min()
    beta = 2 * np.pi * f_tw * 1e9 * L / (np.radians(THETA) * C0)    # kL = θ を厳密に
    tw = _post(cell, solved, beta=beta)[0]
    m3 = _replicate(cell, 3)
    sw = _post(m3, solve_tm0_standing(m3, num_modes=4), 0.0, beta=beta)
    ref = min(sw, key=lambda m: abs(m.frequency_ghz - tw.frequency_ghz))
    assert ref.frequency_ghz == pytest.approx(tw.frequency_ghz, rel=1e-5)
    assert ref.rq_eff == pytest.approx(1.5 * tw.rq_eff, rel=2e-4)
    assert ref.q_factor == pytest.approx(tw.q_factor, rel=2e-4)


# ---------------------------------------------------------------------------
# post のやり直しと、定在波の経路
# ---------------------------------------------------------------------------
def test_cli_post_is_idempotent_and_writes_real_scalars(sample_mesh, tmp_path):
    """既存結果は post のやり直しで直る: 固有ベクトルは変わらず、2 回目も同じ値（複素数は保存されない）."""
    msh = sample_mesh("s-band_1cell", 2)
    raw, one, two = tmp_path / "tw.h5", tmp_path / "one.h5", tmp_path / "two.h5"
    assert cli_main(["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "2",
                     "--num-modes", "2", "-p", "60,120", "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(one)]) == 0
    assert cli_main(["post", "--type", "tm0", "-i", str(one), "-o", str(two)]) == 0
    a, b = read_results(one), read_results(two)
    assert a["post_process"] == b["post_process"]
    raw_results = read_results(raw)["results_by_n"][0]["traveling"]
    for theta, modes in a["post_process"][0]["traveling"].items():
        np.testing.assert_array_equal(a["results_by_n"][0]["traveling"][theta]["eigenvectors"],
                                      raw_results[theta]["eigenvectors"])
        for m in modes:
            assert all(np.isrealobj(v) and np.isfinite(v) for v in m.values())
    mesh = load_mesh(str(msh), 2)
    direct = solve_tm0_traveling(mesh, num_modes=2, phase_shifts=(60.0, 120.0))
    for theta, modes in a["post_process"][0]["traveling"].items():
        rec = {"frequencies": raw_results[theta]["frequencies"], "eigenvectors": raw_results[theta]["eigenvectors"]}
        ref = compute_tm0_parameters(replace(direct, phase_results={theta: rec}),
                                     classify_boundaries(mesh.physical_groups, mesh.nodes)).per_phase[theta]
        for m, r in zip(modes, ref):
            assert m["R_over_Q"] == pytest.approx(r.rq_eff, rel=1e-10)
            assert m["U_stored"] == pytest.approx(r.stored_energy, rel=1e-10, abs=0)
            assert m["P_flow_zmin"] == pytest.approx(r.p_flow, rel=1e-10, abs=0)


def test_standing_pillbox_tm010_matches_analytic(tmp_path):
    """定在波の経路は変えない（samples/cylinder100mm: R = 50 mm, L = 100 mm）."""
    geom = MultiRegionGeometry.load_json(Path(__file__).resolve().parents[1] / "samples" / "cylinder100mm.gmshproj")
    geom.mesh_size = geom.mesh_size * 0.6
    mesh = _mesh_from(geom, tmp_path / "pillbox.msh")
    modes = _post(mesh, solve_tm0_standing(mesh, num_modes=4), 0.0)
    x01 = jn_zeros(0, 1)[0]
    R, L = 0.05, 0.1
    omega = x01 * C0 / R
    mode = min(modes, key=lambda m: abs(m.frequency_ghz * 2e9 * np.pi - omega))
    assert mode.frequency_ghz * 2e9 * np.pi == pytest.approx(omega, rel=1e-3)
    u = (EPS0 / 2) * np.pi * R ** 2 * L * j1(x01) ** 2
    v = L * abs(np.sinc(omega * L / C0 / (2 * np.pi)))
    assert mode.rq_eff == pytest.approx(v ** 2 / (omega * u), rel=0.01)
    assert mode.stored_energy == pytest.approx(u, rel=0.02, abs=0)
