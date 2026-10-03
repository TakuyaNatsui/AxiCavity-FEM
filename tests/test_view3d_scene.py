"""3D 描画（gui/ui/view3d_scene.py）をオフスクリーンの pyvista で: アクターの有無がオプションに従う、場の差し替え、
扇形、横断面、矢印、電気力線、色の範囲、PNG と GIF。Qt は使わない（QtInteractor は offscreen で落ちる）."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pv = pytest.importorskip("pyvista")

from axicavity_fem.gui.core.revolve import ModeFields, mode_fields_from_renderer   # noqa: E402
from axicavity_fem.gui.ui.view3d_scene import (                                     # noqa: E402
    COLOR_FIELDS,
    Scene3D,
    View3DOptions,
    amplitude_range,
    boundary_edges,
    render_gif,
    sample_indices,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
TM0_TW = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
HOM_SW = SAMPLES / "s-band_1cell_SW_HOM_processed.h5"


@pytest.fixture
def plotter():
    p = pv.Plotter(off_screen=True, window_size=(320, 240))
    yield p
    p.close()


def _square_mesh():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    simplices = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    return vertices, simplices


def _fields(n=0, traveling=False):
    ones = np.ones(5)
    return ModeFields(n=n, traveling=traveling, Ez=ones, Er=0.5 * ones, Ephi=np.zeros(5), Hz=np.zeros(5),
                      Hr=np.zeros(5), Hphi=ones, psi=np.array([0.0, 0.0, 1.0, 1.0, 0.5]))


def test_helpers():
    v, s = _square_mesh()
    edges = boundary_edges(s, v)
    assert len(edges) == 3                                          # 軸上（r = 0）の辺は除く
    assert {tuple(sorted(e)) for e in edges} == {(1, 3), (2, 3), (0, 2)}
    assert sample_indices(10, 3).tolist() == [0, 4, 9] and sample_indices(2, 5).tolist() == [0, 1]
    f = _fields()
    assert amplitude_range(f, "|E|") == pytest.approx((0.0, np.sqrt(1.25)))
    assert amplitude_range(f, "Er") == (-0.5, 0.5) and amplitude_range(f, "Hz") == (-1.0, 1.0)
    assert amplitude_range(f, "|H|") == (0.0, 1.0)


def test_actors_follow_options(plotter):
    v, s = _square_mesh()
    scene = Scene3D(plotter)
    assert not scene.has_mesh
    scene.set_mesh(v, s, 1, View3DOptions(n_phi=8))
    assert scene.has_mesh and scene.actor_names == ["wall", "meridian"]        # 場が無くても形は出る
    scene.set_fields(_fields(), 0.0)
    assert set(scene.actor_names) == {"wall", "meridian"} and scene._scalar_bar_title == "E abs"
    grid = scene._pv_grid
    assert grid.n_points == 8 * 5 and grid.n_cells == 8 * 4 and "|E|" in grid.point_data
    assert np.allclose(grid.point_data["|E|"], np.sqrt(1.25))
    # 横断面・矢印・電気力線
    scene.set_options(View3DOptions(n_phi=8, show_slice=True, slice_z=0.5, show_arrows=True, arrow_nz=6, arrow_nxy=8,
                                    show_e_lines=True, e_line_levels=4, e_line_count=6))
    assert set(scene.actor_names) == {"wall", "meridian", "slice", "arrows", "elines"}
    # 矢印は直交格子の点（子午面の z × r と横断面の x × y）。節点ではない
    _key, az, ar, aphi, spacing, amps = scene._arrow_cache
    assert spacing == pytest.approx(2 / 7) and set(np.round(np.rad2deg(aphi[az != 0.5]))) == {0.0, 180.0}   # 横断面の格子の間隔
    assert len(amps["Ez"]) == len(az) and (ar > 0).all()
    volume = View3DOptions(n_phi=8, show_arrows=True, arrow_mode="volume", arrow_nz=4, arrow_nxy=6)
    scene.set_options(volume)
    _key, az, ar, aphi, spacing, amps = scene._arrow_cache
    assert len(set(np.round(az, 9))) == 4 and (ar <= 1.0 + 1e-9).all() and "arrows" in scene.actor_names
    scene.set_options(View3DOptions(n_phi=8, show_slice=True, slice_z=0.5, show_arrows=True, arrow_nz=6, arrow_nxy=8,
                                    show_e_lines=True, e_line_levels=4, e_line_count=6))
    section = scene._slice_z()
    assert section.n_points > 0 and np.allclose(section.points[:, 2], 0.5)
    assert scene.plotter.renderer.actors["elines"].GetMapper() is not None
    # 壁だけ、スカラーバー無し
    scene.set_options(View3DOptions(n_phi=8, show_meridian=False, show_scalar_bar=False, color_field="Hphi"))
    assert scene.actor_names == ["wall"] and scene._scalar_bar_title is None
    assert scene._clim == (-1.0, 1.0)
    scene.clear()
    assert scene.actor_names == [] and not scene.has_mesh


def test_sector_and_n_phi_rebuild_grid(plotter):
    v, s = _square_mesh()
    scene = Scene3D(plotter)
    scene.set_mesh(v, s, 1, View3DOptions(n_phi=8))
    scene.set_fields(_fields(n=1), 0.0)
    key = scene._grid_key
    scene.set_options(View3DOptions(n_phi=8, color_field="Ez"))            # 色だけ → グリッドはそのまま
    assert scene._grid_key == key
    scene.set_options(View3DOptions(n_phi=12, sector_deg=270))
    assert scene._grid_key != key and not scene._grid.full and scene._grid.ring_count == 13
    faces = scene._meridian_faces()
    assert faces.n_cells == 2 * 4                                            # φ = 0 と 270° の面
    last_ring = scene._grid.points[12 * 5:]
    off_axis = np.hypot(last_ring[:, 0], last_ring[:, 1]) > 1e-9
    assert np.allclose(np.arctan2(last_ring[off_axis, 1], last_ring[off_axis, 0]) % (2 * np.pi), np.deg2rad(270))
    assert "|E|" in scene._pv_grid.point_data and len(scene._pv_grid.point_data["|E|"]) == 13 * 5
    scene.set_options(View3DOptions(n_phi=7))                                # 奇数は偶数に（φ = 180° の面のため）
    assert scene._grid.n_phi == 8 and scene._grid.full


def test_time_phase_changes_traveling_fields_only(plotter):
    v, s = _square_mesh()
    scene = Scene3D(plotter)
    scene.set_mesh(v, s, 1, View3DOptions(n_phi=8))
    fields = ModeFields(n=0, traveling=True, Ez=np.ones(5) * (1 + 1j), Er=np.zeros(5), Ephi=np.zeros(5),
                        Hz=np.zeros(5), Hr=np.zeros(5), Hphi=np.ones(5), psi=np.ones(5) * 1j)
    scene.set_fields(fields, 0.0)
    ez0 = np.array(scene._pv_grid.point_data["Ez"])
    scene.set_time_phase(90.0)
    ez90 = np.array(scene._pv_grid.point_data["Ez"])
    assert np.allclose(ez0, 1.0) and np.allclose(ez90, -1.0)
    # アニメーション（矢印の作り直し）で視点が動かない: カメラを動かしてから時間位相を変えても同じ
    scene.set_options(View3DOptions(n_phi=8, show_arrows=True, arrow_nz=6, arrow_nxy=8))
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 20
    plotter.camera.zoom(1.7)
    before = plotter.camera_position
    scene.set_time_phase(180.0)
    scene.set_options(View3DOptions(n_phi=8, show_arrows=True, arrow_nz=10, arrow_nxy=12, show_slice=True))
    after = plotter.camera_position
    assert np.allclose(np.asarray(before.position), np.asarray(after.position))
    assert np.allclose(np.asarray(before.focal_point), np.asarray(after.focal_point))
    assert plotter.camera.parallel_scale == pytest.approx(plotter.camera.parallel_scale)
    assert scene._clim == (0.0, pytest.approx(np.sqrt(2)))                  # 範囲は振幅（|A|）で固定


@pytest.mark.skipif(not (TM0_TW.exists() and HOM_SW.exists()), reason="サンプル結果なし")
def test_real_results_png_and_gif(plotter, tmp_path):
    from axicavity_fem.gui.ui.result_renderer import ResultData, ResultRenderer, Selection

    data = ResultData(TM0_TW)
    renderer = ResultRenderer(data)
    fields = mode_fields_from_renderer(renderer, Selection())
    mesh = data.mesh
    scene = Scene3D(plotter)
    options = View3DOptions(n_phi=24, show_slice=True, show_arrows=True, arrow_nz=12, arrow_nxy=12, show_e_lines=True,
                            e_line_levels=8, e_line_count=8)
    scene.set_mesh(mesh["vertices"], mesh["simplices"], mesh["elem_order"], options)
    scene.set_fields(fields, 30.0)
    assert set(scene.actor_names) >= {"wall", "meridian", "slice", "arrows", "elines"}
    for name in COLOR_FIELDS:
        assert name in scene._pv_grid.point_data
    png = tmp_path / "view.png"
    scene.view("front")
    scene.render_png(png)
    assert png.exists() and png.stat().st_size > 1000
    gif = tmp_path / "anim.gif"
    seen = []
    ok = render_gif(mesh["vertices"], mesh["simplices"], mesh["elem_order"], fields, options, gif, n_frames=3, fps=4,
                    size_px=(200, 150), progress=lambda i, n: seen.append(i) or True)
    assert ok and gif.exists() and gif.stat().st_size > 500 and seen == [0, 1, 2]
    assert not render_gif(mesh["vertices"], mesh["simplices"], mesh["elem_order"], fields, options, tmp_path / "no.gif",
                          n_frames=3, size_px=(200, 150), progress=lambda i, n: i < 1)
    # HOM: n = 1 の φ 依存（子午面の φ = 0 と 180° で E_z の符号が逆）
    hom = ResultData(HOM_SW)
    fields = mode_fields_from_renderer(ResultRenderer(hom), Selection(n=1))
    scene.set_mesh(hom.mesh["vertices"], hom.mesh["simplices"], hom.mesh["elem_order"], View3DOptions(n_phi=16))
    scene.set_fields(fields, 0.0)
    n = scene._grid.node_count
    ez = np.array(scene._pv_grid.point_data["Ez"])
    assert np.allclose(ez[8 * n:9 * n], -ez[:n], atol=1e-9 * np.abs(ez).max())
