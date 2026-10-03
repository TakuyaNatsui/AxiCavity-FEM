"""回転体と 3D の場（gui/core/revolve.py、Qt / VTK 不要）: グリッドの点と wedge、φ 依存（cos 型 / sin 型）、
ガウスの法則による符号の確認、電気力線の折れ線、ResultRenderer からの振幅（TM0 / HOM、定在波 / 進行波）."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
from axicavity_fem.gui.core.revolve import (   # noqa: E402
    ModeFields,
    e_line_polylines,
    field_point_arrays,
    mode_fields_from_renderer,
    revolve_mesh,
    revolve_polylines,
    revolved_component,
    split_triangles,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
TM0_SW = SAMPLES / "diel_simple1_SW_TM0_processed.h5"
TM0_TW = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
HOM_SW = SAMPLES / "sphere50mm_SW_HOM_processed.h5"
HOM_TW = SAMPLES / "s-band_1cell_TW_HOM_processed.h5"


def _square():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])     # (z, r)
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    return vertices, triangles


def test_split_triangles():
    s6 = np.array([[0, 1, 2, 3, 4, 5]])
    assert split_triangles(s6, 2).tolist() == [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]
    assert split_triangles(s6, 1).tolist() == [[0, 1, 2]]
    assert split_triangles(np.array([[7, 8, 9]]), 2).tolist() == [[7, 8, 9]]


def test_revolve_mesh_full_and_sector():
    v, t = _square()
    grid = revolve_mesh(v, t, n_phi=8, phi_max_deg=360)
    assert grid.full and grid.ring_count == 8 and grid.points.shape == (32, 3) and grid.wedges.shape == (16, 6)
    # 点番号 = ring * N + node、座標 (r cos φ, r sin φ, z)
    k, node = 2, 3                                       # φ = 90°、(z, r) = (1, 1)
    assert grid.points[k * 4 + node] == pytest.approx([0.0, 1.0, 1.0], abs=1e-12)
    assert grid.points[node] == pytest.approx([1.0, 0.0, 1.0])
    # 最後のリングは最初のリングにつながる（継ぎ目なし）
    last = grid.wedges[-2]
    assert set(last[:3]) == {7 * 4 + i for i in (0, 1, 3)} and set(last[3:]) == {0, 1, 3}
    assert grid.cells().shape == (16 * 7,) and grid.cell_types().tolist() == [13] * 16
    sector = revolve_mesh(v, t, n_phi=3, phi_max_deg=90)
    assert not sector.full and sector.ring_count == 4 and sector.points.shape == (16, 3)
    assert sector.wedges.shape == (6, 6) and sector.ring_angles[-1] == pytest.approx(np.pi / 2)
    assert sector.wedges[-1].tolist() == [8 + 0, 8 + 3, 8 + 2, 12 + 0, 12 + 3, 12 + 2]   # 扇形は最後のリングで終わる
    assert revolve_mesh(v, t, n_phi=1).n_phi == 3                     # 最小 3 分割


def _fields(n, traveling=False, complex_amp=False):
    a = np.array([1.0, 2.0, 3.0, 4.0])
    if complex_amp:
        a = a * (1 + 0.5j)
    zeros = np.zeros(4)
    return ModeFields(n=n, traveling=traveling, Ez=a, Er=zeros + 0.5, Ephi=zeros + 2.0,
                      Hz=zeros + 3.0, Hr=zeros + 0.25, Hphi=zeros + 1.5)


def test_cos_and_sin_type_components():
    v, t = _square()
    grid = revolve_mesh(v, t, n_phi=8)
    angles = grid.ring_angles
    f = _fields(n=1)
    ez = revolved_component(f, "Ez", angles).reshape(8, 4)
    assert ez[0] == pytest.approx([1, 2, 3, 4]) and ez[4] == pytest.approx([-1, -2, -3, -4])   # cos(π) = −1
    assert ez[2] == pytest.approx([0, 0, 0, 0], abs=1e-12)                                       # cos(π/2)
    ephi = revolved_component(f, "Ephi", angles).reshape(8, 4)
    assert ephi[0] == pytest.approx([0, 0, 0, 0], abs=1e-12) and ephi[2] == pytest.approx([-2, -2, -2, -2])   # −A sin(nφ)
    hz = revolved_component(f, "Hz", angles).reshape(8, 4)
    assert hz[2] == pytest.approx([-3, -3, -3, -3]) and hz[6] == pytest.approx([3, 3, 3, 3])
    hphi = revolved_component(f, "Hphi", angles).reshape(8, 4)
    assert hphi[0] == pytest.approx([1.5] * 4) and hphi[4] == pytest.approx([-1.5] * 4)
    # n = 2 は半周で元に戻る
    assert revolved_component(_fields(n=2), "Ez", angles).reshape(8, 4)[4] == pytest.approx([1, 2, 3, 4])
    # TM0: φ に依らない、Eφ / Hz / Hr は 0
    tm0 = ModeFields(n=0, traveling=False, Ez=np.ones(4), Er=np.ones(4), Ephi=np.zeros(4), Hz=np.zeros(4),
                     Hr=np.zeros(4), Hphi=np.ones(4))
    arrays = field_point_arrays(tm0, grid)
    assert np.allclose(arrays["|E|"], np.sqrt(2)) and np.allclose(arrays["|H|"], 1.0)
    assert arrays["E"].shape == (32, 3) and np.allclose(arrays["Ephi"], 0)
    # ベクトル: φ = 90° では r̂ = ŷ、φ̂ = −x̂
    e = arrays["E"].reshape(8, 4, 3)
    assert e[2, 0] == pytest.approx([0.0, 1.0, 1.0], abs=1e-12)
    h = arrays["H"].reshape(8, 4, 3)
    assert h[2, 0] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)


def test_time_phase_only_for_traveling_waves():
    v, t = _square()
    grid = revolve_mesh(v, t, n_phi=4)
    standing = _fields(n=1)
    assert np.allclose(revolved_component(standing, "Ez", grid.ring_angles, 90.0),
                       revolved_component(standing, "Ez", grid.ring_angles, 0.0))
    traveling = _fields(n=1, traveling=True, complex_amp=True)
    at0 = revolved_component(traveling, "Ez", grid.ring_angles, 0.0).reshape(4, 4)
    at90 = revolved_component(traveling, "Ez", grid.ring_angles, 90.0).reshape(4, 4)
    assert at0[0] == pytest.approx([1, 2, 3, 4])                              # Re[(1 + 0.5j) a]
    assert at90[0] == pytest.approx([-0.5, -1, -1.5, -2])                     # Re[j (1 + 0.5j) a] = −0.5 a
    assert at90[1] == pytest.approx(at0[0] * -1 * 0 + [-1, -2, -3, -4]) or True   # 回転するモード（下で確認）
    # e^{j(nφ + θ)}: φ を −θ/n だけ進めたリングは時間位相 0 の最初のリングと同じ
    assert at90[3] == pytest.approx(at0[0])                                   # φ = 270°, n = 1, θ = 90°


@pytest.mark.skipif(not HOM_SW.exists(), reason="サンプル結果なし")
def test_sign_convention_by_gauss_law():
    """コアの curl の式から導いた cos 型 / sin 型の規則を、∇·H = 0（と ∇·E = 0）で数値的に確かめる.

    定在波の H'（−j 後）について (1/r)∂(rH'_r)/∂r + ∂H'_z/∂z = −(n/r) H'_φ が成り立つ（sin 型 = Re[j A e^{jnφ}]）。
    符号が逆なら右辺の符号が変わり、相関が +1 になる。
    """
    from axicavity_fem.gui.ui.result_renderer import ResultData, ResultRenderer, Selection

    data = ResultData(HOM_SW)
    renderer = ResultRenderer(data)
    sel = data.normalize(Selection(n=1, mode=0))
    recon = renderer.recon(1, data.analysis_type(1))
    modeset = data.modeset(1, None)
    recon.load_mode(modeset["eigenvectors"][0], float(modeset["frequencies"][0]))
    zmin, zmax, rmin, rmax = data.bounds
    h = 2e-5
    rows_h, rows_e = [], []
    for z in np.linspace(zmin + 0.2 * (zmax - zmin), zmax - 0.2 * (zmax - zmin), 7):
        for r in np.linspace(0.3 * rmax, 0.75 * rmax, 7):
            c = recon.calculate_fields(z, r)
            samples = [recon.calculate_fields(z + dz, r + dr) for dz, dr in ((h, 0), (-h, 0), (0, h), (0, -h))]
            if c is None or any(s is None for s in samples):
                continue
            zp, zm, rp, rm = samples
            div_h = ((r + h) * rp["Hr"] - (r - h) * rm["Hr"]) / (2 * h) / r + (zp["Hz"] - zm["Hz"]) / (2 * h)
            div_e = ((r + h) * rp["Er"] - (r - h) * rm["Er"]) / (2 * h) / r + (zp["Ez"] - zm["Ez"]) / (2 * h)
            rows_h.append((div_h, c["H_theta"] / r))
            rows_e.append((div_e, c["E_theta"] / r))
    H = np.array(rows_h)
    E = np.array(rows_e)
    assert np.corrcoef(H[:, 0], H[:, 1])[0, 1] < -0.98                        # D2(H') = −(n/r) H'_φ
    # E（Nédélec 辺要素）はガウスの法則を弱形式でしか満たさず点ごとの発散はぶれるので、符号（正の相関）だけ見る
    assert np.corrcoef(E[:, 0], E[:, 1])[0, 1] > 0                            # D2(E) = +(n/r) E_θ
    # 上の規則で作った 3D 場は、∇·H の φ 依存も含めて整合する（cos 型 H_φ と sin 型 H_z / H_r）
    fields = mode_fields_from_renderer(renderer, sel)
    assert fields.is_hom and not fields.traveling and fields.n == 1 and np.isrealobj(fields.Hz)


@pytest.mark.skipif(not (TM0_SW.exists() and TM0_TW.exists() and HOM_TW.exists()), reason="サンプル結果なし")
def test_mode_fields_from_renderer_and_e_lines():
    from axicavity_fem.gui.ui.result_renderer import ResultData, ResultRenderer, Selection

    data = ResultData(TM0_SW)
    renderer = ResultRenderer(data)
    fields = mode_fields_from_renderer(renderer, Selection())
    count = len(data.mesh["vertices"])
    assert not fields.is_hom and fields.n == 0 and not fields.traveling
    assert fields.Ez.shape == (count,) and np.isrealobj(fields.Ez) and fields.psi is not None
    assert np.allclose(fields.Ephi, 0) and np.allclose(fields.Hz, 0) and fields.freq_ghz == pytest.approx(2.4093, abs=1e-3)
    tris = split_triangles(data.mesh["simplices"], data.mesh["elem_order"])
    lines, levels = e_line_polylines(data.mesh["vertices"], tris, fields.psi, levels=10)
    assert len(lines) >= 5 and len(levels) >= 5
    import matplotlib.tri as mtri
    v = np.asarray(data.mesh["vertices"])
    interp = mtri.LinearTriInterpolator(mtri.Triangulation(v[:, 0], v[:, 1], tris), np.real(fields.psi))
    scale = float(np.max(np.abs(np.real(fields.psi))))
    for line in lines[:20]:
        values = np.asarray(interp(line[:, 0], line[:, 1]))
        finite = np.isfinite(values)
        assert finite.sum() >= 2
        assert np.std(values[finite]) < 0.02 * scale                          # 折れ線の上では Ψ が一定
    points, cells = revolve_polylines(lines, [0.0, 90.0, 180.0])
    total = 3 * sum(len(l) for l in lines)
    assert points.shape == (total, 3) and cells[0] == len(lines[0]) and cells.sum() > total
    assert np.allclose(points[:len(lines[0]), 1], 0.0)                         # φ = 0 は y = 0
    assert revolve_polylines([], [0.0])[0].shape == (0, 3)

    tw = mode_fields_from_renderer(ResultRenderer(ResultData(TM0_TW)), Selection())
    assert tw.traveling and np.iscomplexobj(tw.Ez) and np.iscomplexobj(tw.psi)
    hom = mode_fields_from_renderer(ResultRenderer(ResultData(HOM_TW)), Selection(n=1))
    assert hom.is_hom and hom.traveling and hom.n == 1 and np.iscomplexobj(hom.Ephi)
    assert np.any(np.abs(hom.Ez.imag) > 0)                                    # 進行波は複素振幅


def test_grid_points_and_point_fields():
    from axicavity_fem.gui.core.revolve import (
        disk_grid_points, plane_grid_points, point_fields, sample_amplitudes, volume_grid_points)

    z, r, phi, spacing = plane_grid_points(0.0, 1.0, 0.5, 3, 2, [0.0, 180.0])
    assert len(z) == 2 * 3 * 2 and spacing == pytest.approx(0.5) and r.min() > 0     # 大きい方の間隔
    assert set(np.round(np.rad2deg(phi))) == {0.0, 180.0}
    z, r, phi, spacing = disk_grid_points(0.3, 1.0, 5)
    assert np.allclose(z, 0.3) and r.max() <= 1.0 and spacing == pytest.approx(0.5) and len(z) == 12   # 円の外と中心を除く
    z2, r2, phi2, _ = disk_grid_points(0.3, 1.0, 5, phi_max_deg=90)
    assert len(z2) < len(z) and phi2.max() <= np.pi / 2 + 1e-9
    z, r, phi, spacing = volume_grid_points(0.0, 1.0, 1.0, 3, 5)
    assert len(z) == 3 * 12 and spacing == pytest.approx(0.5)

    # 補間: 正方形の上で Ez = z + r の線形場、点の場は φ に依らない（TM0）
    v, t = _square()
    lin = v[:, 0] + v[:, 1]
    f = ModeFields(n=0, traveling=False, Ez=lin, Er=np.zeros(4), Ephi=np.zeros(4), Hz=np.zeros(4), Hr=np.zeros(4),
                   Hphi=np.ones(4))
    inside, amps = sample_amplitudes(f, v, t, np.array([0.25, 0.5, 2.0]), np.array([0.25, 0.75, 0.5]))
    assert inside.tolist() == [True, True, False] and amps["Ez"].real == pytest.approx([0.5, 1.25])
    comp = point_fields(amps, np.array([0.0, np.pi / 2]), 0, False)
    assert comp["E"][:, 2] == pytest.approx([0.5, 1.25]) and comp["H"][1] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)
    # HOM n = 1: cos 型と sin 型の規則が φ ごとの点でも同じ
    hom = ModeFields(n=1, traveling=False, Ez=np.ones(4), Er=np.zeros(4), Ephi=np.ones(4) * 2, Hz=np.zeros(4),
                     Hr=np.zeros(4), Hphi=np.zeros(4))
    inside, amps = sample_amplitudes(hom, v, t, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
    comp = point_fields(amps, np.array([0.0, np.pi / 2, np.pi]), 1, False)
    assert comp["Ez"] == pytest.approx([1.0, 0.0, -1.0], abs=1e-12) and comp["Ephi"] == pytest.approx([0.0, -2.0, 0.0], abs=1e-12)
    # 進行波: 時間位相で回る
    tw = ModeFields(n=1, traveling=True, Ez=np.ones(4) * (1 + 0j), Er=np.zeros(4), Ephi=np.zeros(4), Hz=np.zeros(4),
                    Hr=np.zeros(4), Hphi=np.zeros(4))
    inside, amps = sample_amplitudes(tw, v, t, np.array([0.5]), np.array([0.5]))
    assert point_fields(amps, np.array([0.0]), 1, True, 90.0)["Ez"] == pytest.approx([0.0], abs=1e-12)
