"""Superfish (.af) 入出力の GUI 配線テスト (ヘッドレス)。

MyFrame.mr_import_superfish / mr_export_superfish が Multi-Region Editor と
shared/superfish_io を正しく仲介することを確認する。表示の最終確認は手動。
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
wx = pytest.importorskip("wx")

from axicavity_fem.shared.multi_region_model import Loop, Region, Segment


# 5cm x 2cm の長方形 (閉ループ)
SQUARE_AF = """Gmsh2Fish
$reg kprob=1, dx=1.000000e-01, xdri=2.5, ydri=1.0, nbsup=1, nbslo=0, nbslf=0, nbsrt=0, freq=2856.0, kmethod=1, beta=1.0 $

$po x=0.0, y=0.0 $
$po x=5.0, y=0.0 $
$po x=5.0, y=2.0 $
$po x=0.0, y=2.0 $
$po x=0.0, y=0.0 $
"""

OPEN_AF = """$po x=0.0, y=0.0 $
$po x=5.0, y=0.0 $
$po x=5.0, y=2.0 $
"""


@pytest.fixture
def frame():
    from axicavity_fem.gui.main_frame import MyFrame
    try:
        app = wx.App(False)
        frm = MyFrame(None, wx.ID_ANY, "")
    except Exception as e:
        pytest.skip(f"wx GUI を構築できません: {e}")
    yield frm
    frm.Destroy()
    app.Destroy()


@pytest.fixture
def msgbox_calls(monkeypatch):
    """wx.MessageBox を抑止し、(message, caption) を記録する。"""
    calls = []

    def fake_messagebox(message, caption="", *args, **kwargs):
        calls.append((str(message), str(caption)))
        return wx.OK

    monkeypatch.setattr(wx, "MessageBox", fake_messagebox)
    return calls


def _set_square_geometry(frame, unit="mm"):
    """50mm x 20mm の長方形 1 領域 (Vacuum) を MR パネルにセットする。"""
    from axicavity_fem.shared.multi_region_model import MultiRegionGeometry
    geom = MultiRegionGeometry(
        points=[(0.0, 0.0), (50.0, 0.0), (50.0, 20.0), (0.0, 20.0)],
        segments=[Segment(id=i, type="line", point_indices=[a, b])
                  for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)])],
        loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
        regions=[Region(id=0, name="Vacuum", outer_loop_id=0)],
        unit=unit,
    )
    frame.mr_editor_panel.set_geometry(geom)
    frame.mr_unit_choice.SetStringSelection(unit)
    return geom


# ---------------------------------------------------------------------------
# インポート
# ---------------------------------------------------------------------------
def test_import_af_switches_to_mr_tab(frame, tmp_path, msgbox_calls):
    p = tmp_path / "square.af"
    p.write_text(SQUARE_AF, encoding="utf-8")
    frame.mr_unit_choice.SetStringSelection("mm")

    assert frame.mr_import_superfish(str(p)) is True
    assert frame.notebook_1.GetSelection() == frame._mr_tab_index()
    geom = frame.mr_editor_panel.get_geometry()
    # cm -> mm 換算 (x10)、閉ループ + Vacuum 領域
    assert len(geom.points) == 4
    assert geom.points[1] == pytest.approx((50.0, 0.0))
    assert len(geom.loops) == 1
    assert len(geom.regions) == 1
    assert geom.regions[0].material_tag == "vacuum"
    # Region リストにも反映される
    assert frame.mr_region_list.GetCount() == 1
    # 新規プロジェクト扱い (dirty, パス未設定)
    assert frame.mr_editor_panel.is_dirty is True
    assert frame.mr_current_project_path is None


def test_import_open_contour_shows_info(frame, tmp_path, msgbox_calls):
    p = tmp_path / "open.af"
    p.write_text(OPEN_AF, encoding="utf-8")
    frame.mr_unit_choice.SetStringSelection("mm")

    assert frame.mr_import_superfish(str(p)) is True
    geom = frame.mr_editor_panel.get_geometry()
    assert len(geom.points) == 3
    assert len(geom.segments) == 2
    assert geom.loops == []
    assert geom.regions == []
    # Close Loop を促す Info ダイアログが表示される
    assert any("Close Loop" in msg for msg, _ in msgbox_calls)


def test_import_invalid_file_shows_error(frame, tmp_path, msgbox_calls):
    p = tmp_path / "empty.af"
    p.write_text("Gmsh2Fish\n", encoding="utf-8")

    assert frame.mr_import_superfish(str(p)) is False
    assert any("失敗" in msg for msg, _ in msgbox_calls)


# ---------------------------------------------------------------------------
# エクスポート
# ---------------------------------------------------------------------------
def test_export_af_success(frame, tmp_path, msgbox_calls):
    _set_square_geometry(frame, unit="mm")
    frame.mr_mesh_size_ctrl.SetValue("2")
    out = tmp_path / "out.af"

    assert frame.mr_export_superfish(str(out)) is True
    text = out.read_text(encoding="utf-8")
    po_lines = [l for l in text.splitlines() if l.startswith("$po")]
    assert len(po_lines) == 5  # 開始点 + 4 segment
    assert "dx=2.000000e-01" in text  # 2mm = 0.2cm
    assert any("出力しました" in msg for msg, _ in msgbox_calls)


def test_export_rejects_multi_region(frame, tmp_path, msgbox_calls):
    _set_square_geometry(frame, unit="mm")
    geom = frame.mr_editor_panel.get_geometry()
    geom.regions.append(Region(id=1, name="Dielectric", outer_loop_id=0,
                               material_tag="dielectric_1", eps_r=9.0))
    frame.mr_mesh_size_ctrl.SetValue("2")
    out = tmp_path / "out.af"

    assert frame.mr_export_superfish(str(out)) is False
    assert not out.exists()
    assert any("単一領域" in msg for msg, _ in msgbox_calls)


def test_export_rejects_invalid_mesh_size(frame, tmp_path, msgbox_calls):
    _set_square_geometry(frame, unit="mm")
    frame.mr_mesh_size_ctrl.SetValue("not_a_number(")
    out = tmp_path / "out.af"

    assert frame.mr_export_superfish(str(out)) is False
    assert not out.exists()
    assert any("Mesh Size" in msg for msg, _ in msgbox_calls)
