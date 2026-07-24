"""multi_region_model のラウンドトリップ・互換読込テスト。"""

from __future__ import annotations

import json
import math

import pytest

from axicavity_fem.shared.multi_region_model import (
    BC_NAMES,
    DEFAULT_BC,
    SCHEMA_VERSION,
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)


# ---------------------------------------------------------------------------
# Segment 単体
# ---------------------------------------------------------------------------
def test_segment_line_basic():
    s = Segment(id=0, type="line", point_indices=[0, 1])
    assert s.bc_name == DEFAULT_BC
    d = s.to_dict()
    s2 = Segment.from_dict(d)
    assert s2.type == "line"
    assert s2.point_indices == [0, 1]
    assert "center" not in d


def test_segment_arc_basic():
    s = Segment(
        id=3, type="arc", point_indices=[2, 5],
        bc_name="E-short",
        center=(1.0, 2.0), radius=3.0, theta1=0.0, theta2=90.0,
    )
    d = s.to_dict()
    s2 = Segment.from_dict(d)
    assert s2.type == "arc"
    assert s2.center == (1.0, 2.0)
    assert s2.radius == 3.0
    assert s2.bc_name == "E-short"


def test_segment_invalid_type():
    with pytest.raises(ValueError):
        Segment(id=0, type="bezier", point_indices=[0, 1])


def test_segment_invalid_bc():
    with pytest.raises(ValueError):
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="Dirichlet")


def test_segment_arc_requires_center():
    with pytest.raises(ValueError):
        Segment(id=0, type="arc", point_indices=[0, 1])


# ---------------------------------------------------------------------------
# Loop / Region
# ---------------------------------------------------------------------------
def test_loop_default_orientation_ccw():
    l = Loop(id=0, segment_ids=[0, 1, 2])
    assert l.orientation == "CCW"


def test_loop_invalid_orientation():
    with pytest.raises(ValueError):
        Loop(id=0, segment_ids=[0], orientation="UP")


def test_region_eps_r_must_be_positive():
    with pytest.raises(ValueError):
        Region(id=0, name="x", outer_loop_id=0, eps_r=0.0)
    with pytest.raises(ValueError):
        Region(id=0, name="x", outer_loop_id=0, eps_r=-1.0)


# ---------------------------------------------------------------------------
# MultiRegionGeometry: ID 採番・ルックアップ
# ---------------------------------------------------------------------------
def _make_square_geom(side: float = 10.0, eps_r: float = 1.0) -> MultiRegionGeometry:
    """1 領域・正方形の最小ジオメトリ。"""
    pts = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1]),
        Segment(id=1, type="line", point_indices=[1, 2]),
        Segment(id=2, type="line", point_indices=[2, 3]),
        Segment(id=3, type="line", point_indices=[3, 0]),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3], orientation="CCW")
    region = Region(
        id=0, name="Vacuum", outer_loop_id=0,
        material_tag=VACUUM_TAG, eps_r=eps_r,
    )
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=[loop], regions=[region],
    )


def test_next_ids():
    geom = _make_square_geom()
    assert geom.next_segment_id() == 4
    assert geom.next_loop_id() == 1
    assert geom.next_region_id() == 1


def test_lookup_by_id():
    geom = _make_square_geom()
    assert geom.segment_by_id(2).point_indices == [2, 3]
    assert geom.loop_by_id(0).orientation == "CCW"
    assert geom.region_by_id(0).material_tag == VACUUM_TAG
    with pytest.raises(KeyError):
        geom.segment_by_id(999)


# ---------------------------------------------------------------------------
# 妥当性検証
# ---------------------------------------------------------------------------
def test_validate_ok():
    geom = _make_square_geom()
    assert geom.validate() == []


def test_validate_out_of_range_point():
    geom = _make_square_geom()
    geom.segments[0].point_indices = [0, 99]
    errs = geom.validate()
    assert any("out of range" in e for e in errs)


def test_validate_duplicate_material_tag():
    geom = _make_square_geom()
    # 2 つ目の領域 (実体は無いがメタ的に追加) で同じタグを使う
    geom.regions.append(Region(
        id=1, name="Vacuum2", outer_loop_id=0,
        material_tag=VACUUM_TAG, eps_r=1.0,
    ))
    errs = geom.validate()
    assert any("material_tag" in e for e in errs)


def test_validate_unknown_loop_in_region():
    geom = _make_square_geom()
    geom.regions[0].hole_loop_ids = [42]
    errs = geom.validate()
    assert any("hole_loop_id" in e for e in errs)


# ---------------------------------------------------------------------------
# 符号付き面積 (向き判定)
# ---------------------------------------------------------------------------
def test_signed_area_ccw_positive():
    geom = _make_square_geom(side=10.0)
    area = geom.loop_signed_area(0)
    assert area == pytest.approx(100.0)  # 10x10 正方形, CCW


def test_signed_area_cw_negative():
    geom = _make_square_geom(side=10.0)
    # 反転して CW にする
    for s in geom.segments:
        s.point_indices = list(reversed(s.point_indices))
    geom.loops[0].segment_ids = list(reversed(geom.loops[0].segment_ids))
    area = geom.loop_signed_area(0)
    assert area == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# JSON ラウンドトリップ
# ---------------------------------------------------------------------------
def test_to_from_dict_roundtrip():
    geom = _make_square_geom(side=7.5, eps_r=2.25)
    d = geom.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    geom2 = MultiRegionGeometry.from_dict(d)
    assert geom2.validate() == []
    assert geom2.regions[0].eps_r == 2.25
    assert len(geom2.segments) == 4
    assert geom2.loop_signed_area(0) == pytest.approx(7.5 * 7.5)


def test_save_load_json_roundtrip(tmp_path):
    geom = _make_square_geom(side=12.0, eps_r=4.0)
    p = tmp_path / "geom.json"
    geom.save_json(p)
    geom2 = MultiRegionGeometry.load_json(p)
    assert geom2.regions[0].eps_r == 4.0
    assert geom2.unit == "mm"
    assert len(geom2.points) == 4


def test_unsupported_schema_version_raises():
    bad = {"schema_version": "9.9", "points": [], "segments": [],
           "loops": [], "regions": []}
    with pytest.raises(ValueError):
        MultiRegionGeometry.from_dict(bad)


# ---------------------------------------------------------------------------
# ver2.2: 変数 (variables) のラウンドトリップ・後方互換
# ---------------------------------------------------------------------------
def test_variables_roundtrip():
    geom = _make_square_geom(side=5.0, eps_r=1.0)
    geom.variables = [("a", "100"), ("b", "10"), ("c", "a + b")]
    d = geom.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION == "2.3"
    assert d["variables"] == [["a", "100"], ["b", "10"], ["c", "a + b"]]
    geom2 = MultiRegionGeometry.from_dict(d)
    assert geom2.variables == [("a", "100"), ("b", "10"), ("c", "a + b")]


def test_variables_save_load_json(tmp_path):
    geom = _make_square_geom(side=8.0, eps_r=1.0)
    geom.variables = [("w", "50"), ("h", "w/2")]
    p = tmp_path / "geom_vars.json"
    geom.save_json(p)
    geom2 = MultiRegionGeometry.load_json(p)
    assert geom2.variables == [("w", "50"), ("h", "w/2")]


def test_variables_default_empty_when_absent():
    # variables キーが無い dict でも空リストで読める (後方互換)
    geom = _make_square_geom(side=5.0, eps_r=1.0)
    d = geom.to_dict()
    del d["variables"]
    geom2 = MultiRegionGeometry.from_dict(d)
    assert geom2.variables == []


def test_schema_21_without_variables_still_loads():
    # 旧 schema_version=2.1 (variables 無し) も MR 形式として読め、変数は空
    geom = _make_square_geom(side=5.0, eps_r=1.0)
    d = geom.to_dict()
    d["schema_version"] = "2.1"
    d.pop("variables", None)
    geom2 = MultiRegionGeometry.from_dict(d)
    assert geom2.validate() == []
    assert geom2.variables == []


# ---------------------------------------------------------------------------
# 円弧つき (PEC + E-short ミックス) のラウンドトリップ
# ---------------------------------------------------------------------------
def test_arc_segment_roundtrip(tmp_path):
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="E-short"),
        Segment(id=1, type="arc", point_indices=[1, 2],
                bc_name="PEC",
                center=(10.0, 2.5), radius=2.5, theta1=-90.0, theta2=90.0),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="M-short"),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Vacuum", outer_loop_id=0)
    geom = MultiRegionGeometry(
        points=pts, segments=segs, loops=[loop], regions=[region],
    )
    p = tmp_path / "arc.json"
    geom.save_json(p)
    geom2 = MultiRegionGeometry.load_json(p)
    s_arc = geom2.segment_by_id(1)
    assert s_arc.type == "arc"
    assert s_arc.center == (10.0, 2.5)
    assert s_arc.radius == 2.5
    bcs = [s.bc_name for s in geom2.segments]
    assert bcs == ["E-short", "PEC", "PEC", "M-short"]


# ---------------------------------------------------------------------------
# ver2 (legacy .gmshproj) 互換読込
# ---------------------------------------------------------------------------
def test_from_legacy_single_loop_basic():
    legacy = {
        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "PEC"},
            {"type": "line", "points": [1, 2], "physical_name": "PEC"},
            {"type": "line", "points": [2, 3], "physical_name": "E-short"},
            {"type": "line", "points": [3, 0], "physical_name": "M-short"},
        ],
        "loop_closed": True,
        "settings": {
            "unit": "mm",
            "xmin": "0", "xmax": "100",
            "ymin": "0", "ymax": "100",
            "mesh_size": "5",
            "mesh_order": 1,
        },
    }
    geom = MultiRegionGeometry.from_legacy_single_loop(legacy)
    assert geom.validate() == []
    assert len(geom.regions) == 1
    assert geom.regions[0].material_tag == VACUUM_TAG
    assert geom.regions[0].eps_r == 1.0
    assert geom.unit == "mm"
    assert geom.mesh_size == 5.0
    assert geom.settings["mesh_order"] == 1
    bcs = [s.bc_name for s in geom.segments]
    assert bcs == ["PEC", "PEC", "E-short", "M-short"]
    # CCW 向き判定
    assert geom.loops[0].orientation == "CCW"


def test_from_legacy_single_loop_orientation_always_ccw():
    """ver2.1 仕様: OCC モードでは向きが自動補正されるため、
    legacy 読込時の Loop.orientation は常に CCW に固定される（CW 検出は廃止）。"""
    # 入力は CW (時計回り) でも CCW でも、orientation ラベルは CCW になる
    legacy = {
        "points": [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "PEC"},
            {"type": "line", "points": [1, 2], "physical_name": "PEC"},
            {"type": "line", "points": [2, 3], "physical_name": "PEC"},
            {"type": "line", "points": [3, 0], "physical_name": "PEC"},
        ],
        "loop_closed": True,
        "settings": {"unit": "mm", "mesh_size": "5"},
    }
    geom = MultiRegionGeometry.from_legacy_single_loop(legacy)
    assert geom.loops[0].orientation == "CCW"


def test_from_legacy_with_arc():
    legacy = {
        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "E-short"},
            {"type": "arc",  "points": [1, 2], "physical_name": "PEC",
             "center": [10.0, 2.5], "radius": 2.5, "theta1": -90.0, "theta2": 90.0},
            {"type": "line", "points": [2, 3], "physical_name": "PEC"},
            {"type": "line", "points": [3, 0], "physical_name": "M-short"},
        ],
        "loop_closed": True,
        "settings": {"unit": "mm", "mesh_size": "2"},
    }
    geom = MultiRegionGeometry.from_legacy_single_loop(legacy)
    arc = geom.segment_by_id(1)
    assert arc.type == "arc"
    assert arc.center == (10.0, 2.5)
    assert arc.radius == 2.5


def test_from_dict_auto_dispatches_legacy():
    """schema_version が無ければ legacy 形式として読まれる。"""
    legacy = {
        "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "PEC"},
            {"type": "line", "points": [1, 2], "physical_name": "PEC"},
            {"type": "line", "points": [2, 3], "physical_name": "PEC"},
            {"type": "line", "points": [3, 0], "physical_name": "PEC"},
        ],
        "loop_closed": True,
    }
    geom = MultiRegionGeometry.from_dict(legacy)
    assert len(geom.regions) == 1
    assert geom.regions[0].eps_r == 1.0


def test_legacy_physical_name_none_becomes_default():
    """ver2 の 'None' (string) や None (null) を既定 PEC に倒す。"""
    legacy = {
        "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "None"},
            {"type": "line", "points": [1, 2], "physical_name": None},
            {"type": "line", "points": [2, 0]},
        ],
        "loop_closed": True,
    }
    geom = MultiRegionGeometry.from_legacy_single_loop(legacy)
    for s in geom.segments:
        assert s.bc_name == DEFAULT_BC


# ---------------------------------------------------------------------------
# 多領域 + 穴 を表現できることのスモーク
# ---------------------------------------------------------------------------
def test_two_regions_with_hole():
    """外側矩形 (誘電体) の中に内側矩形 (vacuum, ドーナツの穴) を持つ構成。"""
    # 外側 [-20, 20] x [0, 20]
    # 内側 [-5,  5] x [5, 15]
    pts = [
        (-20.0, 0.0), (20.0, 0.0), (20.0, 20.0), (-20.0, 20.0),  # 0-3 外
        (-5.0, 5.0), (5.0, 5.0), (5.0, 15.0), (-5.0, 15.0),      # 4-7 内
    ]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),  # 底
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="PEC"),
        # 穴 (内側ループ): 誘電体から見ると CW で並ぶ向き
        # 内側矩形を時計回りに辿る: 4(LL) → 7(UL) → 6(UR) → 5(LR) → 4
        Segment(id=4, type="line", point_indices=[4, 7]),
        Segment(id=5, type="line", point_indices=[7, 6]),
        Segment(id=6, type="line", point_indices=[6, 5]),
        Segment(id=7, type="line", point_indices=[5, 4]),
        # 内側 vacuum 用 (内側ループを CCW に並べる)
        # 本当は穴側と外周側で segment を共有させたいが、データモデル上は別 segment
        # として持つ前提 (Gmsh 側で fragment が共有境界を整える)。
        Segment(id=8, type="line", point_indices=[4, 5]),
        Segment(id=9, type="line", point_indices=[5, 6]),
        Segment(id=10, type="line", point_indices=[6, 7]),
        Segment(id=11, type="line", point_indices=[7, 4]),
    ]
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3], orientation="CCW"),       # 外側外周
        Loop(id=1, segment_ids=[4, 5, 6, 7], orientation="CW"),        # 穴
        Loop(id=2, segment_ids=[8, 9, 10, 11], orientation="CCW"),     # vacuum 外周
    ]
    regions = [
        Region(id=0, name="Dielectric", outer_loop_id=0,
               hole_loop_ids=[1], material_tag="dielectric_1", eps_r=4.0),
        Region(id=1, name="Vacuum", outer_loop_id=2,
               material_tag="vacuum", eps_r=1.0),
    ]
    geom = MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=regions,
    )
    errs = geom.validate()
    assert errs == [], f"validation errors: {errs}"
    assert geom.regions[0].eps_r == 4.0
    assert geom.regions[1].eps_r == 1.0
    # 外側ループは CCW (面積 +)
    assert geom.loop_signed_area(0) > 0
    # 穴ループは CW (面積 -)
    assert geom.loop_signed_area(1) < 0
