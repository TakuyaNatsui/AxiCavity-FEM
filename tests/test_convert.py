"""convert: スケッチ + 設定 → MultiRegionGeometry（有効 BC・領域の解決・円と大きな弧・検証）."""

import math

import pytest

from axicavity_fem.gui.core.convert import (
    ConvertError,
    apply_point_expressions,
    auto_bc,
    check_geometry,
    curve_use_counts,
    effective_bc,
    match_regions,
    normalize_material_tag,
    resolve_regions,
    simplify_boundaries,
    sketch_profiles,
    to_multi_region,
)
from axicavity_fem.gui.core.document import (
    Param,
    RegionSetting,
    SketchArc,
    SketchLine,
    SketchPoint,
    create_empty_document,
    new_id,
)
from axicavity_fem.gui.core.sketch.edit_ops import insert_point_on_curve
from axicavity_fem.gui.core.sketch.model import (
    add_arc,
    add_circle,
    add_line,
    add_point,
    add_polyline,
    add_rectangle,
    get_entity,
)


def pillbox(z=100.0, r=50.0):
    """軸上の線 [0]、右 [1]、上 [2]、左 [3] の長方形."""
    doc = create_empty_document("t")
    line_ids, point_ids = add_rectangle(doc.sketch, (0, 0), (z, r))
    return doc, line_ids, point_ids


def two_boxes():
    """z=50 で接する 2 つの長方形。共有辺は 1 つ目の右辺（add_line が重複線を作らない）."""
    doc = create_empty_document("t")
    a, _ = add_rectangle(doc.sketch, (0, 0), (50, 50))
    b, _ = add_rectangle(doc.sketch, (50, 0), (100, 50), tolerance=1e-6)
    assert b[3] == a[1]
    return doc, a, b


# --- 有効 BC ---

def test_pillbox_default_bcs_and_region():
    doc, lines, _ = pillbox()
    assert effective_bc(doc, lines[0]) == ("None", "axis")
    for lid in lines[1:]:
        assert effective_bc(doc, lid) == ("PEC", "default")
    res = to_multi_region(doc)
    geom = res.geom
    assert geom.validate() == [] and res.warnings == []
    assert [s.bc_name for s in geom.segments] == ["None", "PEC", "PEC", "PEC"]
    assert [s.type for s in geom.segments] == ["line"] * 4
    assert geom.points == [(0, 0), (100, 0), (100, 50), (0, 50)]          # エンティティの順
    assert len(geom.loops) == 1 and geom.loops[0].segment_ids == [0, 1, 2, 3]
    assert len(geom.regions) == 1
    r = geom.regions[0]
    assert (r.name, r.material_tag, r.eps_r, r.mu_r, r.tan_delta) == ("Vacuum", "vacuum", 1.0, 1.0, 0.0)
    assert geom.unit == "mm" and geom.mesh_size == 5.0 and geom.settings == {"mesh_order": 1}
    assert res.segments_of_curve == {lid: [i] for i, lid in enumerate(lines)}
    assert res.region_of_profile == {sketch_profiles(doc)[0].id: 0}


def test_explicit_bc_overrides_and_axis_tolerance_follows_units():
    doc, lines, _ = pillbox()
    doc.boundaries[lines[2]] = "M-short"
    doc.boundaries[lines[0]] = "E-short"
    assert effective_bc(doc, lines[2]) == ("M-short", "explicit")
    assert effective_bc(doc, lines[0]) == ("E-short", "explicit")
    assert auto_bc(doc, lines[0]) == ("None", "axis")
    codes = [i.code for i in check_geometry(doc)]
    assert "axis_wall_bc" in codes
    # 軸のしきい値はコアと同じ 1e-6 m: r = 5e-4 の線は mm（= 5e-7 m）では軸、m では軸ではない
    doc_m = create_empty_document("m", units="m")
    ids, _ = add_rectangle(doc_m.sketch, (0, 5e-4), (1, 0.5))
    assert auto_bc(doc_m, ids[0]) == ("PEC", "default")
    doc_mm = create_empty_document("mm")
    ids, _ = add_rectangle(doc_mm.sketch, (0, 5e-4), (1000, 500))
    assert auto_bc(doc_mm, ids[0]) == ("None", "axis")


def test_shared_edge_is_interface_and_simplify_removes_redundant_explicit():
    doc, a, _b = two_boxes()
    shared = a[1]
    assert effective_bc(doc, shared) == ("None", "interface")
    assert curve_use_counts(sketch_profiles(doc))[shared] == 2
    doc.boundaries[shared] = "PEC"
    assert effective_bc(doc, shared) == ("PEC", "explicit")
    assert "interface_wall_bc" in [i.code for i in check_geometry(doc)]
    doc.boundaries[shared] = "None"                 # 自動判定と同じ → 整理で消える
    doc.boundaries[a[2]] = "PEC"                    # 既定と同じ → 消える
    doc.boundaries[a[3]] = "E-short"                # 違う → 残る
    doc.boundaries["ghost"] = "PEC"                 # 存在しない曲線 → 消える
    assert simplify_boundaries(doc) == 3
    assert doc.boundaries == {a[3]: "E-short"}
    geom = to_multi_region(doc).geom
    assert len(geom.regions) == 2 and len(geom.loops) == 2 and len(geom.segments) == 7
    assert [r.name for r in geom.regions] == ["Vacuum", "Vacuum2"]
    assert [r.material_tag for r in geom.regions] == ["vacuum", "vacuum2"]
    assert sum(s.bc_name == "None" for s in geom.segments) == 3            # 軸 2 本 + 界面 1 本


# --- 領域の解決 ---

def test_region_settings_match_by_key_then_anchor_then_similarity():
    doc, a, b = two_boxes()
    profiles = sketch_profiles(doc)
    right = next(p for p in profiles if p.anchor[0] > 50)
    setting = RegionSetting(key=right.id, anchor=right.anchor, name="Ceramic", materialTag="ceramic",
                            epsR=9.64, tanDelta=1e-4)
    doc.regions = [setting]
    resolved = {r.profile.id: r for r in resolve_regions(doc, profiles)}
    assert resolved[right.id].name == "Ceramic" and resolved[right.id].eps_r == 9.64
    other = next(p for p in profiles if p is not right)
    assert resolved[other.id].setting is None and resolved[other.id].name == "Vacuum"
    # key が古くなっても anchor で見つかる
    setting.key = "stale"
    assert match_regions(doc, profiles) == {right.id: setting}
    # anchor も外れたら外周エンティティ集合の類似度（1 辺を分割: 3/6 = 0.5）で見つかる
    setting.key, setting.anchor = right.id, (-100.0, -100.0)
    insert_point_on_curve(doc.sketch, b[1], (100, 25))
    profiles2 = sketch_profiles(doc)
    right2 = next(p for p in profiles2 if p.anchor[0] > 50)
    assert right2.id != right.id
    assert match_regions(doc, profiles2) == {right2.id: setting}
    # 似ていなければ対応しない
    setting.key = "x|y|z"
    assert match_regions(doc, profiles2) == {}
    geom = to_multi_region(doc).geom
    assert [r.name for r in geom.regions] == ["Vacuum", "Vacuum2"]


def test_region_expressions_and_material_tags():
    doc, lines, _ = pillbox()
    doc.params = [Param(new_id(), "er", "9.64"), Param(new_id(), "td", "5.7e-6")]
    prof = sketch_profiles(doc)[0]
    doc.regions = [RegionSetting(key=prof.id, anchor=prof.anchor, name="Ceramic Ring", materialTag="Ceramic Ring",
                                 epsR=1.0, epsRExpr="er", tanDelta=0.0, tanDeltaExpr="td")]
    res = to_multi_region(doc)
    r = res.geom.regions[0]
    assert (r.name, r.material_tag) == ("Ceramic Ring", "ceramic_ring")
    assert r.eps_r == pytest.approx(9.64) and r.eps_r_expr == "er"
    assert r.tan_delta == pytest.approx(5.7e-6) and r.tan_delta_expr == "td"
    assert any(i.code == "tag_renamed" for i in res.issues)
    assert normalize_material_tag("PEC") == "pec_mat" and normalize_material_tag("E-short") == "e_short_mat"
    assert normalize_material_tag("") == "region" and normalize_material_tag("日本語") == "region"
    assert normalize_material_tag("  Vacuum  ") == "vacuum"


def test_duplicate_names_and_tags_are_made_unique():
    doc, a, b = two_boxes()
    profiles = sketch_profiles(doc)
    doc.regions = [RegionSetting(key=p.id, anchor=p.anchor, name="X", materialTag="x") for p in profiles]
    geom = to_multi_region(doc).geom
    assert sorted(r.name for r in geom.regions) == ["X", "X2"]
    assert sorted(r.material_tag for r in geom.regions) == ["x", "x2"]
    assert geom.validate() == []


# --- 円・大きな円弧 ---

def test_circle_hole_becomes_four_arcs():
    doc, lines, _ = pillbox()
    circle_id, _ = add_circle(doc.sketch, (50, 25), 10)
    assert effective_bc(doc, circle_id) == ("None", "interface")
    res = to_multi_region(doc)
    geom = res.geom
    assert geom.validate() == []
    arcs = [s for s in geom.segments if s.type == "arc"]
    assert len(arcs) == 4 and len(geom.segments) == 8
    assert [(s.theta1, s.theta2) for s in arcs] == [(0, 90), (90, 180), (180, 270), (270, 360)]
    assert all(s.center == (50, 25) and s.radius == 10 and s.bc_name == "None" for s in arcs)
    assert len(geom.points) == 8                                   # 長方形 4 点 + 円周上 4 点
    assert len(geom.loops) == 2                                    # 円のループは穴と外周で共有
    outer, inner = geom.regions
    assert outer.hole_loop_ids == [inner.outer_loop_id] and inner.hole_loop_ids == []
    assert res.segments_of_curve[circle_id] == [s.id for s in arcs]


def test_semicircle_arc_is_split_in_two():
    doc = create_empty_document("t")
    s = doc.sketch
    add_polyline(s, [(0, 50), (0, 0), (100, 0), (100, 50)], False)
    arc_id = add_arc(s, (50, 50), (100, 50), (0, 50), tolerance=1e-6)     # 0° → 180°
    res = to_multi_region(doc)
    geom = res.geom
    assert geom.validate() == []
    arcs = [seg for seg in geom.segments if seg.type == "arc"]
    assert len(arcs) == 2 and res.segments_of_curve[arc_id] == [a.id for a in arcs]
    assert [(a.theta1, a.theta2) for a in arcs] == [(0, 90), (90, 180)]
    assert (50.0, 100.0) in [tuple(p) for p in geom.points]           # 分割点
    assert len(geom.points) == 5 and len(geom.loops) == 1
    assert set(geom.loops[0].segment_ids) == {0, 1, 2, 3, 4}
    # 弧の向きが逆（loop でも逆順にたどる）でも同じ
    doc2 = create_empty_document("t")
    s2 = doc2.sketch
    add_polyline(s2, [(0, 50), (0, 0), (100, 0), (100, 50)], False)
    add_arc(s2, (50, 50), (0, 50), (100, 50), tolerance=1e-6)            # 180° → 360°（下に膨らむ）
    assert len(to_multi_region(doc2).geom.segments) == 5


# --- 式 ---

def test_point_expressions_are_applied_and_carried_to_geometry():
    doc = create_empty_document("t")
    doc.params = [Param(new_id(), "L", "100"), Param(new_id(), "R", "50")]
    s = doc.sketch
    p0 = add_point(s, (0, 0))
    p1 = add_point(s, (1, 1), x_expr="L")
    p2 = add_point(s, (1, 1), x_expr="L", y_expr="R")
    p3 = add_point(s, (1, 1), y_expr="R")
    for a, b in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
        add_line(s, a, b)
    assert apply_point_expressions(doc) == []
    assert (get_entity(s, p1).x, get_entity(s, p2).y, get_entity(s, p3).x) == (100, 50, 1)
    doc.mesh.sizeExpr = "L/20"
    doc.view.zmin, doc.view.zmax, doc.view.rmin, doc.view.rmax = -5.0, 105.0, 0.0, 60.0
    res = to_multi_region(doc)
    geom = res.geom
    assert geom.point_exprs == [(None, None), ("L", None), ("L", "R"), (None, "R")]
    assert geom.variables == [("L", "100"), ("R", "50")]
    assert geom.mesh_size == 5.0 and geom.mesh_size_expr == "L/20"
    assert geom.settings == {"mesh_order": 1, "xmin": "-5.0", "xmax": "105.0", "ymin": "0.0", "ymax": "60.0"}
    # 評価できない式はエラー
    get_entity(s, p1).xExpr = "L +"
    assert apply_point_expressions(doc)[0].startswith("点の Z の式")
    doc.params.append(Param(new_id(), "bad", "1 +"))
    assert "param_error" in [i.code for i in check_geometry(doc)]


def test_arc_center_expression_is_carried():
    doc = create_empty_document("t")
    doc.params = [Param(new_id(), "a", "24.5")]
    s = doc.sketch
    add_polyline(s, [(0, 50), (0, 0), (100, 0), (100, 22)], False)
    arc_id = add_arc(s, (100, 24.5), (97.5, 24.5), (100, 22), tolerance=1e-6)   # 180° → 270°（劣弧）
    center = get_entity(s, get_entity(s, arc_id).center)
    center.yExpr = "a"
    add_line(s, (97.5, 24.5), (0, 50), tolerance=1e-6)
    geom = to_multi_region(doc).geom
    arc = next(seg for seg in geom.segments if seg.type == "arc")
    assert arc.center_expr == (None, "a") and arc.center == (100, 24.5)
    assert arc.theta1 == pytest.approx(180) and arc.theta2 == pytest.approx(270)


# --- 検証 ---

def test_check_geometry_errors_and_warnings():
    assert [i.code for i in check_geometry(create_empty_document("e"))] == ["no_profile"]

    doc, lines, _ = pillbox()
    add_line(doc.sketch, (10, 10), (20, 20))
    doc.mesh.size = 0
    issues = {i.code: i for i in check_geometry(doc)}
    assert issues["mesh_size"].level == "error"
    assert issues["unused_curve"].level == "warning" and len(issues["unused_curve"].entity_ids) == 1
    doc.mesh.size, doc.mesh.sizeExpr = 5.0, "L/0"
    assert "mesh_size" in [i.code for i in check_geometry(doc)]
    doc.mesh.sizeExpr = None

    neg = create_empty_document("n")
    add_rectangle(neg.sketch, (0, -10), (100, 50))
    issues = {i.code: i for i in check_geometry(neg)}
    assert issues["negative_r"].level == "error" and len(issues["negative_r"].entity_ids) == 2
    with pytest.raises(ConvertError):
        to_multi_region(neg)
    assert to_multi_region(neg, strict=False).geom.points[0] == (0, -10)


def test_traveling_wave_end_planes_and_axis_dielectric_warnings():
    doc, lines, _ = pillbox()
    doc.analysis.wave = "traveling"
    issues = {i.code: i for i in check_geometry(doc)}
    assert set(issues["tw_end_pec"].entity_ids) == {lines[1], lines[3]}
    doc.boundaries[lines[1]] = "E-short"
    doc.boundaries[lines[3]] = "E-short"
    assert "tw_end_pec" not in [i.code for i in check_geometry(doc)]
    prof = sketch_profiles(doc)[0]
    doc.regions = [RegionSetting(key=prof.id, anchor=prof.anchor, name="D", materialTag="d", epsR=4.0)]
    assert "axis_region_eps" in [i.code for i in check_geometry(doc)]


def test_arc_crossing_the_axis_is_an_error():
    doc = create_empty_document("t")
    s = doc.sketch
    add_arc(s, (50, 5), (40, 5), (60, 5))                 # 180° → 360°: 下に膨らみ r < 0 を通る
    add_line(s, (60, 5), (40, 5), tolerance=1e-6)
    codes = [i.code for i in check_geometry(doc)]
    assert "arc_crosses_axis" in codes and "negative_r" not in codes
    ok = create_empty_document("t")
    add_arc(ok.sketch, (50, 5), (60, 5), (40, 5))         # 0° → 180°: 上に膨らむ
    add_line(ok.sketch, (40, 5), (60, 5), tolerance=1e-6)
    assert "arc_crosses_axis" not in [i.code for i in check_geometry(ok)]
