"""shared/superfish_io.py の純粋テスト (wx 不要)。

- インポート: 直線/円弧のパース、単位換算、閉合判定、Vacuum 領域自動生成
- エクスポート: バリデーション (単一領域・穴なし・閉ループ)、ループ walk の
  向き解決、$reg 行の書式
- 往復 (export -> import) での幾何一致
"""

from __future__ import annotations

import math
import re

import pytest

from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)
from axicavity_fem.shared.superfish_io import (
    SuperfishExportError,
    SuperfishImportError,
    export_superfish,
    import_superfish,
    validate_superfish_exportable,
)


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
# 5cm x 2cm の長方形 (閉ループ)
SQUARE_AF = """Gmsh2Fish
$reg kprob=1, dx=1.000000e-01, xdri=2.5, ydri=1.0, nbsup=1, nbslo=0, nbslf=0, nbsrt=0, freq=2856.0, kmethod=1, beta=1.0 $

$po x=0.0, y=0.0 $
$po x=5.0, y=0.0 $
$po x=5.0, y=2.0 $
$po x=0.0, y=2.0 $
$po x=0.0, y=0.0 $
"""

# 円弧 1 本を含む閉形状 (cm):
# (5,0) --arc(center=(5,1), r=1, theta_end=0deg)--> (6,1) --> (6,3) --> (0,3)
# --> (0,0) --> (5,0)
ARC_AF = """$po x=5.0, y=0.0 $
$po nt=2, r=1.0, theta=0.0, x0=5.0, y0=1.0 $
$po x=6.0, y=3.0 $
$po x=0.0, y=3.0 $
$po x=0.0, y=0.0 $
$po x=5.0, y=0.0 $
"""


def _write_af(tmp_path, body, name="test.af"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _square_geom(unit="mm"):
    """50mm x 20mm の長方形 1 領域 (Vacuum) の MultiRegionGeometry。"""
    pts = [(0.0, 0.0), (50.0, 0.0), (50.0, 20.0), (0.0, 20.0)]
    segs = [Segment(id=i, type="line", point_indices=[a, b])
            for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)])]
    return MultiRegionGeometry(
        points=pts,
        segments=segs,
        loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
        regions=[Region(id=0, name="Vacuum", outer_loop_id=0)],
        unit=unit,
        mesh_size=2.0,
    )


# ---------------------------------------------------------------------------
# 1. インポート
# ---------------------------------------------------------------------------
def test_import_square_mm(tmp_path):
    p = _write_af(tmp_path, SQUARE_AF)
    geom = import_superfish(p, unit="mm")
    # cm -> mm は x10。閉合で末尾の重複点は除去され 4 点になる
    assert len(geom.points) == 4
    assert geom.points[0] == pytest.approx((0.0, 0.0))
    assert geom.points[1] == pytest.approx((50.0, 0.0))
    assert geom.points[2] == pytest.approx((50.0, 20.0))
    assert geom.points[3] == pytest.approx((0.0, 20.0))
    # 最終 segment の終点は先頭点に付け替えられる
    assert geom.segments[-1].point_indices[1] == 0
    # 1 Loop + Vacuum 1 Region が自動生成される
    assert len(geom.loops) == 1
    assert len(geom.regions) == 1
    assert geom.regions[0].material_tag == VACUUM_TAG
    assert geom.regions[0].eps_r == 1.0
    # BC は全て PEC
    assert all(s.bc_name == "PEC" for s in geom.segments)
    assert geom.unit == "mm"
    assert geom.validate() == []


@pytest.mark.parametrize("unit,expected_z", [
    ("m", 0.05), ("cm", 5.0), ("mm", 50.0), ("inch", 5.0 / 2.54),
])
def test_import_unit_scaling(tmp_path, unit, expected_z):
    p = _write_af(tmp_path, SQUARE_AF)
    geom = import_superfish(p, unit=unit)
    assert geom.points[1][0] == pytest.approx(expected_z)
    assert geom.unit == unit


def test_import_arc(tmp_path):
    p = _write_af(tmp_path, ARC_AF)
    geom = import_superfish(p, unit="cm")
    # 5 点・5 segment の閉ループ
    assert len(geom.points) == 5
    assert len(geom.segments) == 5
    assert len(geom.loops) == 1
    assert len(geom.regions) == 1
    # 先頭 segment が円弧
    arc = geom.segments[0]
    assert arc.type == "arc"
    assert arc.center == pytest.approx((5.0, 1.0))
    assert arc.radius == pytest.approx(1.0)
    # 始点 (5,0) は角度 -90 度、終点 (6,1) は角度 0 度 -> 劣弧 [-90, 0]
    assert arc.theta1 == pytest.approx(-90.0)
    assert arc.theta2 == pytest.approx(0.0)
    # 円弧の終点座標 (theta_end から再構成される)
    assert geom.points[1] == pytest.approx((6.0, 1.0))
    # 劣弧 (span <= 180)
    assert 0 < (arc.theta2 - arc.theta1) <= 180.0 + 1e-9


def test_import_not_closed(tmp_path):
    body = """$po x=0.0, y=0.0 $
$po x=5.0, y=0.0 $
$po x=5.0, y=2.0 $
"""
    p = _write_af(tmp_path, body)
    geom = import_superfish(p, unit="mm")
    assert len(geom.points) == 3
    assert len(geom.segments) == 2
    assert geom.loops == []
    assert geom.regions == []
    assert geom.validate() == []


def test_import_no_po_lines(tmp_path):
    p = _write_af(tmp_path, "Gmsh2Fish\n$reg kprob=1 $\n")
    with pytest.raises(SuperfishImportError):
        import_superfish(p, unit="mm")


def test_import_bad_unit(tmp_path):
    p = _write_af(tmp_path, SQUARE_AF)
    with pytest.raises(SuperfishImportError):
        import_superfish(p, unit="furlong")


def test_import_missing_file(tmp_path):
    with pytest.raises(SuperfishImportError):
        import_superfish(tmp_path / "nonexistent.af", unit="mm")


# ---------------------------------------------------------------------------
# 2. エクスポートのバリデーション
# ---------------------------------------------------------------------------
def test_validate_ok():
    assert validate_superfish_exportable(_square_geom()) == []


def test_validate_two_regions():
    geom = _square_geom()
    geom.regions.append(Region(id=1, name="Dielectric", outer_loop_id=0,
                               material_tag="dielectric_1", eps_r=9.0))
    errors = validate_superfish_exportable(geom)
    assert errors and "単一領域" in errors[0]


def test_validate_with_hole():
    geom = _square_geom()
    # 内側に穴ループ (10..40mm の正方形) を追加
    base = len(geom.points)
    geom.points.extend([(10.0, 5.0), (40.0, 5.0), (40.0, 15.0), (10.0, 15.0)])
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        geom.segments.append(Segment(id=4 + i, type="line",
                                     point_indices=[base + a, base + b]))
    geom.loops.append(Loop(id=1, segment_ids=[4, 5, 6, 7], orientation="CW"))
    geom.regions[0].hole_loop_ids = [1]
    errors = validate_superfish_exportable(geom)
    assert any("穴" in e for e in errors)


def test_validate_disconnected_loop():
    geom = _square_geom()
    # segment の順序を入れ替えて非連結にする (0->1 の次に 2->3 が来る)
    geom.loops[0].segment_ids = [0, 2, 1, 3]
    errors = validate_superfish_exportable(geom)
    assert any("連結" in e for e in errors)


def test_validate_open_loop():
    geom = _square_geom()
    # 最後の segment を除いた開いたループ
    geom.loops[0].segment_ids = [0, 1, 2]
    errors = validate_superfish_exportable(geom)
    assert any("閉じていません" in e for e in errors)


def test_export_raises_on_invalid(tmp_path):
    geom = _square_geom()
    geom.regions.append(Region(id=1, name="R2", outer_loop_id=0,
                               material_tag="dielectric_1"))
    out = tmp_path / "out.af"
    with pytest.raises(SuperfishExportError):
        export_superfish(geom, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# 3. エクスポートの書式と往復
# ---------------------------------------------------------------------------
def test_export_reg_line_format(tmp_path):
    geom = _square_geom(unit="mm")  # 50x20 mm = 5x2 cm, mesh_size=2mm
    out = tmp_path / "out.af"
    export_superfish(geom, out)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "Gmsh2Fish"
    m = re.match(
        r"\$reg kprob=1, dx=([\d.eE+\-]+), xdri=([\d.eE+\-]+), "
        r"ydri=([\d.eE+\-]+), nbsup=1, nbslo=0, nbslf=0, nbsrt=0, "
        r"freq=2856\.0, kmethod=1, beta=1\.0 \$",
        lines[1],
    )
    assert m, f"$reg 行の書式が不正: {lines[1]}"
    assert float(m.group(1)) == pytest.approx(0.2)   # dx = 2mm = 0.2cm
    assert float(m.group(2)) == pytest.approx(2.5)   # xdri = 5cm/2
    assert float(m.group(3)) == pytest.approx(1.0)   # ydri = 2cm/2
    # $po 行は開始点 + 4 segment 分 = 5 行
    po_lines = [l for l in lines if l.startswith("$po")]
    assert len(po_lines) == 5


def test_export_mesh_size_override(tmp_path):
    geom = _square_geom(unit="mm")
    out = tmp_path / "out.af"
    export_superfish(geom, out, mesh_size=5.0)
    text = out.read_text(encoding="utf-8")
    m = re.search(r"dx=([\d.eE+\-]+)", text)
    assert float(m.group(1)) == pytest.approx(0.5)  # 5mm = 0.5cm


def test_roundtrip_square(tmp_path):
    geom = _square_geom(unit="mm")
    out = tmp_path / "out.af"
    export_superfish(geom, out)
    geom2 = import_superfish(out, unit="mm")
    assert len(geom2.points) == 4
    orig = {(round(z, 6), round(r, 6)) for z, r in geom.points}
    back = {(round(z, 6), round(r, 6)) for z, r in geom2.points}
    assert orig == back
    assert len(geom2.regions) == 1
    assert geom2.validate() == []


def test_roundtrip_with_arc(tmp_path):
    geom = _square_geom(unit="mm")
    # segment 1 (点1->点2 の右辺) を劣弧に変換: 中心 (62.5, 10), r=hypot(12.5,10)
    center = (62.5, 10.0)
    radius = math.hypot(50.0 - center[0], 0.0 - center[1])
    a1 = math.degrees(math.atan2(0.0 - center[1], 50.0 - center[0]))
    a2 = math.degrees(math.atan2(20.0 - center[1], 50.0 - center[0]))
    da = a2 - a1
    while da <= -180.0:
        da += 360.0
    while da > 180.0:
        da -= 360.0
    t1, t2 = (a1, a1 + da) if da >= 0 else (a2, a2 + abs(da))
    geom.segments[1] = Segment(id=1, type="arc", point_indices=[1, 2],
                               center=center, radius=radius,
                               theta1=t1, theta2=t2)
    out = tmp_path / "out.af"
    export_superfish(geom, out)
    geom2 = import_superfish(out, unit="mm")
    assert len(geom2.points) == 4
    arcs = [s for s in geom2.segments if s.type == "arc"]
    assert len(arcs) == 1
    assert arcs[0].center == pytest.approx(center)
    assert arcs[0].radius == pytest.approx(radius)
    # 円弧終点は theta (.6e 書式 = 旧実装と同精度) から再構成されるため
    # ~1e-6 相対の丸め誤差を許容する
    for p, q in zip(geom.points, geom2.points):
        assert q == pytest.approx(p, abs=1e-3)


def test_roundtrip_reversed_segment(tmp_path):
    """逆向き接続 (point_indices が逆順) の segment を含む loop の walk。"""
    geom = _square_geom(unit="mm")
    # segment 2 を逆向き定義に置き換える (loop の順序は同じ)
    geom.segments[2] = Segment(id=2, type="line", point_indices=[3, 2])
    assert validate_superfish_exportable(geom) == []
    out = tmp_path / "out.af"
    export_superfish(geom, out)
    geom2 = import_superfish(out, unit="mm")
    orig = {(round(z, 6), round(r, 6)) for z, r in geom.points}
    back = {(round(z, 6), round(r, 6)) for z, r in geom2.points}
    assert orig == back
    assert len(geom2.regions) == 1


@pytest.mark.parametrize("unit", ["m", "cm", "mm", "inch"])
def test_roundtrip_units(tmp_path, unit):
    """export -> import (同一 unit) で座標が保存される (cm 換算が逆写像)。"""
    geom = _square_geom(unit=unit)
    out = tmp_path / "out.af"
    export_superfish(geom, out)
    geom2 = import_superfish(out, unit=unit)
    for p, q in zip(geom.points, geom2.points):
        assert q == pytest.approx(p)
