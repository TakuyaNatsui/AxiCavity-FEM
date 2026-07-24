"""ver2.1 段階3: MultiRegionEditor GUI のスモークテスト.

ヘッドレスで MyFrame を構築し、Multi-Region タブ上の編集 API が
MultiRegionGeometry を正しく更新することを確認する。表示の最終確認は手動。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
wx = pytest.importorskip("wx")

from axicavity_fem.shared.multi_region_model import (
    MultiRegionGeometry,
    SCHEMA_VERSION,
)


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


def test_mr_panel_constructed(frame):
    """Multi-Region Editor と FEM Analysis タブが存在すること。

    旧 Mesh Geometry Edit タブは廃止済み (wxGlade 再生成後は 2 タブ)。
    再生成前は 3 タブのままなので、タブ数は固定せずラベルの有無で検証する。
    """
    titles = [frame.notebook_1.GetPageText(i)
              for i in range(frame.notebook_1.GetPageCount())]
    assert "Multi-Region Editor" in titles
    assert "FEM Analysis" in titles
    # 再生成後 (2 タブ) では旧タブが消えていることを確認
    if frame.notebook_1.GetPageCount() == 2:
        assert "Mesh Geometry Edit" not in titles
    assert frame.mr_editor_panel is not None
    geom = frame.mr_editor_panel.get_geometry()
    assert isinstance(geom, MultiRegionGeometry)
    assert geom.points == []
    assert geom.regions == []


def test_mr_panel_set_mode(frame):
    from axicavity_fem.gui.multi_region_editor import (
        MODE_ADD_POINTS, MODE_BUILD_LOOP, MODE_EDIT_BC,
    )
    p = frame.mr_editor_panel
    p.set_mode(MODE_BUILD_LOOP)
    assert p.mode == MODE_BUILD_LOOP
    p.set_mode(MODE_EDIT_BC)
    assert p.mode == MODE_EDIT_BC
    p.set_mode(MODE_ADD_POINTS)
    assert p.mode == MODE_ADD_POINTS


def test_mr_geometry_crud(frame, tmp_path):
    """API レベルで点・segment・loop・region を作り、JSON 往復で復元できる。"""
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    # 正方形の 4 点
    geom.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    # 4 segments
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b], bc_name="PEC"))
    p.set_geometry(geom)  # refresh
    # ループ確定
    p.pending_loop_segments = [0, 1, 2, 3]
    lid = p.finalize_pending_loop("CCW")
    assert lid == 0
    # 領域追加
    rid = p.add_region(name="Vacuum", outer_loop_id=lid, hole_loop_ids=[],
                       material_tag="vacuum", eps_r=1.0)
    assert rid == 0
    geom2 = p.get_geometry()
    assert len(geom2.regions) == 1
    # JSON 往復
    out = tmp_path / "mr.gmshproj"
    geom2.save_json(out)
    loaded = MultiRegionGeometry.load_json(out)
    assert loaded.to_dict()["schema_version"] == SCHEMA_VERSION
    assert len(loaded.regions) == 1
    assert loaded.regions[0].material_tag == "vacuum"


def test_mr_region_tan_delta_crud(frame, tmp_path):
    """ver2.3: add_region/update_region の tan_delta が JSON 往復まで通る。"""
    from axicavity_fem.shared.multi_region_model import Segment
    p = frame.mr_editor_panel
    geom = p.get_geometry()
    geom.points.extend([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b], bc_name="PEC"))
    p.set_geometry(geom)
    p.pending_loop_segments = [0, 1, 2, 3]
    lid = p.finalize_pending_loop("CCW")
    rid = p.add_region(name="Diel", outer_loop_id=lid, hole_loop_ids=[],
                       material_tag="diel", eps_r=9.0, tan_delta=1e-4)
    assert p.get_geometry().regions[0].tan_delta == 1e-4
    # update で変更
    p.update_region(rid, "Diel", lid, [], "diel", 9.0, None,
                    tan_delta=2e-4, tan_delta_expr=None)
    assert p.get_geometry().regions[0].tan_delta == 2e-4
    # JSON 往復
    out = tmp_path / "mr_tand.gmshproj"
    p.get_geometry().save_json(out)
    loaded = MultiRegionGeometry.load_json(out)
    assert loaded.regions[0].tan_delta == 2e-4


def test_mr_load_project_dispatch(frame, tmp_path):
    """schema_version='2.1' の .gmshproj を読むと MR タブに反映される。"""
    from axicavity_fem.shared.multi_region_model import Segment
    geom = MultiRegionGeometry()
    geom.points.extend([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    from axicavity_fem.shared.multi_region_model import Loop, Region
    geom.loops.append(Loop(id=0, segment_ids=[0, 1, 2, 3]))
    geom.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                               material_tag="vacuum", eps_r=1.0))
    path = tmp_path / "sample.gmshproj"
    geom.save_json(path)

    ok = frame.load_project(str(path))
    assert ok is True
    assert frame.notebook_1.GetSelection() == frame._mr_tab_index()  # MR タブに切替
    loaded_geom = frame.mr_editor_panel.get_geometry()
    assert len(loaded_geom.regions) == 1
    assert frame.mr_region_list.GetCount() == 1


def test_mr_save_project_writes_v21(frame, tmp_path):
    """MR タブをアクティブにして保存すると schema_version='2.1' で書かれる。"""
    from axicavity_fem.shared.multi_region_model import Segment, Loop, Region
    geom = frame.mr_editor_panel.get_geometry()
    geom.points.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
        geom.segments.append(Segment(id=i, type="line", point_indices=[a, b]))
    geom.loops.append(Loop(id=0, segment_ids=[0, 1, 2]))
    geom.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                               material_tag="vacuum", eps_r=1.0))
    frame.mr_editor_panel.set_geometry(geom)

    frame.notebook_1.SetSelection(frame._mr_tab_index())  # MR タブをアクティブに
    out = tmp_path / "saved.gmshproj"
    assert frame.save_project(str(out)) is True
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert len(data["regions"]) == 1


def test_mr_legacy_load_converts_to_mr(frame, tmp_path):
    """schema_version 無しの旧 .gmshproj は Multi-Region 形式に変換して読む。"""
    legacy = {
        "points": [[0.0, 0.0], [50.0, 0.0], [50.0, 25.0], [0.0, 25.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "PEC"},
            {"type": "line", "points": [1, 2], "physical_name": "PEC"},
            {"type": "line", "points": [2, 3], "physical_name": "PEC"},
            {"type": "line", "points": [3, 0], "physical_name": "PEC"},
        ],
        "loop_closed": True,
        "settings": {"unit": "mm", "xmin": "0", "xmax": "100",
                     "ymin": "0", "ymax": "100", "mesh_size": "5", "mesh_order": 0},
    }
    path = tmp_path / "legacy.gmshproj"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    ok = frame.load_project(str(path))
    assert ok is True
    # MR タブに切り替わり、Vacuum 1 領域として読み込まれる
    assert frame.notebook_1.GetSelection() == frame._mr_tab_index()
    geom = frame.mr_editor_panel.get_geometry()
    assert len(geom.points) == 4
    assert len(geom.loops) == 1
    assert len(geom.regions) == 1
    assert geom.regions[0].eps_r == 1.0
    assert all(s.bc_name == "PEC" for s in geom.segments)
    # 変換読込後に保存すると v2.2 形式になる
    out = tmp_path / "converted.gmshproj"
    assert frame.save_project(str(out)) is True
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


def test_mr_legacy_load_with_arc(frame, tmp_path):
    """旧形式の円弧 segment も Multi-Region に変換される。"""
    legacy = {
        "points": [[0.0, 0.0], [50.0, 0.0], [0.0, 25.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "PEC"},
            {"type": "arc", "points": [1, 2], "physical_name": "PEC",
             "center": [25.0, 12.5], "radius": 27.95, "theta1": 0.0, "theta2": 90.0},
            {"type": "line", "points": [2, 0], "physical_name": "PEC"},
        ],
        "loop_closed": True,
        "settings": {"unit": "mm"},
    }
    path = tmp_path / "legacy_arc.gmshproj"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert frame.load_project(str(path)) is True
    geom = frame.mr_editor_panel.get_geometry()
    arcs = [s for s in geom.segments if s.type == "arc"]
    assert len(arcs) == 1
    assert arcs[0].center is not None and arcs[0].radius is not None
