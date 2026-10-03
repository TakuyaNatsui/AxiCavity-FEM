"""結果の読み込みと描画（gui/ui/result_renderer.py、Qt 非依存）: ver2.3 のサンプル結果（samples/*.h5）で
TM0 定在波 / 進行波、HOM、post の値、モード表、場の値、GIF（ver2.3 test_result_viewer の移植）."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("matplotlib")
from matplotlib.backends.backend_agg import FigureCanvasAgg   # noqa: E402
from matplotlib.figure import Figure                         # noqa: E402

from axicavity_fem.gui.ui.result_renderer import (         # noqa: E402
    MODE_COLUMNS,
    ResultData,
    ResultRenderer,
    Selection,
    ViewOptions,
    mode_rows,
    render_gif,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
TM0_SW = SAMPLES / "diel_simple1_SW_TM0_processed.h5"
TM0_TW = SAMPLES / "s-band_1cell_TW_TM0_processed.h5"
HOM_SW = SAMPLES / "s-band_1cell_SW_HOM_processed.h5"
HOM_TW = SAMPLES / "s-band_1cell_TW_HOM_processed.h5"
RAW = SAMPLES / "s-band_1cell_SW_TM0.h5"

pytestmark = pytest.mark.skipif(not all(p.exists() for p in (TM0_SW, TM0_TW, HOM_SW, HOM_TW, RAW)),
                                reason="samples のサンプル結果なし")


def _figure():
    fig = Figure(figsize=(6, 4), dpi=60)
    FigureCanvasAgg(fig)
    return fig


def test_tm0_standing_with_dielectric():
    data = ResultData(TM0_SW)
    assert data.solver_type == "tm0" and not data.is_hom and not data.is_traveling
    assert data.n_orders == [0] and data.phases() == [] and data.num_modes() == 10 and data.has_post
    sel = data.normalize(Selection(n=5, phase=90.0, mode=99))
    assert sel == Selection(n=0, phase=None, mode=9, time_phase=0.0)
    params = data.post_params(Selection())
    assert params["Q"] > 1000 and params["R_over_Q"] > 0 and params["Q_wall"] == pytest.approx(params["Q"])
    zmin, zmax, rmin, rmax = data.bounds
    assert zmin == 0 and rmin == 0 and zmax > 0 and rmax > 0
    assert data.title(Selection()) == "Mode 0: f = 2.409300 GHz (standing)"

    renderer = ResultRenderer(data)
    fig = _figure()
    freq, sel = renderer.draw(fig, Selection(), ViewOptions(show_lines=True, show_vectors=True, show_e_wall=True,
                                                            show_mesh=True))
    assert freq == pytest.approx(2.4093, abs=1e-3) and len(fig.axes) == 2           # 図 + カラーバー
    ax = fig.axes[0]
    assert ax.get_title() == data.title(sel) and ax.get_xlabel() == "z [m]"
    assert len(ax.collections) > 5                                                  # 塗り + 等高線 + 境界 + quiver
    assert renderer.material_geometry()[0], "誘電体界面が無い"
    assert renderer.pec_geometry()[0]
    # 色も線も無し: カラーバーが無い
    fig = _figure()
    renderer.draw(fig, Selection(mode=1), ViewOptions(show_color=False, show_lines=False))
    assert len(fig.axes) == 1
    # 場の値（領域内 / 領域外）
    inside = renderer.field_at(Selection(), (zmin + zmax) / 2, (rmin + rmax) / 2)
    assert set(inside) >= {"H_theta", "Ez", "Er", "E_abs"} and inside["E_abs"] > 0
    assert renderer.field_at(Selection(), zmax * 3, rmax * 3) is None
    lines = renderer.describe_field(Selection(), (zmin + zmax) / 2, (rmin + rmax) / 2)
    assert lines[0].startswith("z = ") and any(line.startswith("H_theta = ") for line in lines)


def test_mode_rows_and_columns():
    rows = mode_rows(ResultData(TM0_SW))
    assert len(rows) == 10 and rows[0]["index"] == 0 and rows[3]["f_GHz"] > rows[0]["f_GHz"]
    assert {"Q", "Q_wall", "Q_diel", "R_over_Q", "V_eff", "U_stored", "P_loss", "P_diel"} <= set(rows[0])
    assert [c[0] for c in MODE_COLUMNS][:4] == ["Q", "Q_wall", "Q_diel", "R_over_Q"]
    raw = ResultData(RAW)
    assert not raw.has_post and raw.post_params(Selection()) is None
    rows = mode_rows(raw)
    assert set(rows[0]) == {"index", "f_GHz"}


def test_tm0_traveling_phase_and_time_phase():
    data = ResultData(TM0_TW)
    assert data.is_traveling and data.phases() == [120.0]
    sel = data.normalize(Selection())
    assert sel.phase == 120.0
    renderer = ResultRenderer(data)
    a = renderer.field_at(Selection(phase=120.0, time_phase=0.0), 0.017, 0.02)
    b = renderer.field_at(Selection(phase=120.0, time_phase=90.0), 0.017, 0.02)
    assert a is not None and b is not None and a["Ez"] != pytest.approx(b["Ez"])   # 時間位相で瞬時値が変わる
    fig = _figure()
    freq, sel = renderer.draw(fig, Selection(time_phase=45.0), ViewOptions(show_lines=True))
    assert sel.phase == 120.0 and "traveling" in fig.axes[0].get_title()
    assert data.post_params(sel)["group_velocity"] > 0


def test_hom_two_panels_and_orders():
    data = ResultData(HOM_SW)
    assert data.is_hom and data.n_orders == [0, 1, 2]
    renderer = ResultRenderer(data)
    fig = _figure()
    freq, sel = renderer.draw(fig, Selection(n=1, mode=0), ViewOptions(show_vectors=True, nz=6, nr=4))
    assert sel.n == 1 and len(fig.axes) == 4                                 # 2 パネル + カラーバー 2 本
    assert fig._suptitle.get_text().startswith("HOM n=1 mode 0")
    assert data.frequencies(1)[0] == pytest.approx(4.0204, abs=1e-3)
    res = renderer.field_at(Selection(n=1), 0.017, 0.02)
    assert set(res) >= {"Ez", "Er", "E_theta", "Hz", "Hr", "H_theta"}
    # 進行波の HOM: 位相と n
    data = ResultData(HOM_TW)
    assert data.is_traveling and data.phases(2) == [120.0]
    sel = data.normalize(Selection(n=2))
    assert sel.phase == 120.0 and data.post_params(sel)["v_phase_c"] > 0
    ResultRenderer(data).draw(_figure(), sel, ViewOptions(show_color=False))


def test_render_gif_and_cancel(tmp_path):
    data = ResultData(TM0_TW)
    renderer = ResultRenderer(data)
    out = tmp_path / "anim.gif"
    seen = []
    ok = render_gif(renderer, Selection(), ViewOptions(), out, n_frames=3, fps=4, size_px=(320, 240),
                    progress=lambda i, n: seen.append((i, n)) or True)
    assert ok and out.exists() and out.stat().st_size > 1000 and seen == [(0, 3), (1, 3), (2, 3)]
    from PIL import Image
    with Image.open(out) as image:
        assert image.n_frames == 3
    cancelled = tmp_path / "no.gif"
    assert not render_gif(renderer, Selection(), ViewOptions(), cancelled, n_frames=4, progress=lambda i, n: i < 2)
    assert not cancelled.exists()


def test_invalid_file_raises(tmp_path):
    import h5py
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f.attrs["schema_version"] = "2.3"
        f.create_group("mesh").create_dataset("vertices", data=np.zeros((3, 2)))
        f["mesh"].create_dataset("simplices", data=np.zeros((1, 3), dtype=int))
    with pytest.raises(ValueError):
        ResultData(bad)
