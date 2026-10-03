"""ver2.1: 円弧 (arc) を含む MultiRegionGeometry のコアロジックとエクスポートのテスト.

- arc 数学 (中心/半径/劣弧角度) の正しさ
- arc を含む geom が .msh / .geo / Python に正しくエクスポートされること

ver3 では wx エディタ（``gui.multi_region_editor``）を廃止したため、ver2.3 の
``arc_params_from_two_points`` / ``recompute_arc_params`` をこのファイルに複写して
コアの検証だけを残す（エディタ操作の検証は ver3 の tests/test_edit_ops.py）。
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys

import pytest

from axicavity_fem.shared.gmsh_export_occ import (
    export_geo_multi_region,
    export_msh_multi_region,
    export_python_script_multi_region,
)
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)


# ---------------------------------------------------------------------------
# ver2.3 gui/multi_region_editor.py からの複写（PointLineEditor 互換の劣弧の規約）
# ---------------------------------------------------------------------------
def _minor_arc_angles(p1, p2, center):
    a1 = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
    a2 = math.degrees(math.atan2(p2[1] - center[1], p2[0] - center[0]))
    da = a2 - a1
    while da <= -180.0:
        da += 360.0
    while da > 180.0:
        da -= 360.0
    if da >= 0:
        theta1, theta2 = a1, a1 + da
    else:
        theta1, theta2 = a2, a2 + abs(da)
    if theta2 <= theta1:
        theta2 += 360.0
    return theta1, theta2


def arc_params_from_two_points(p1, p2):
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-12:
        raise ValueError("2 端点が一致しているため円弧を作れません。")
    perp_x, perp_y = -dy / dist, dx / dist
    center = (mx + perp_x * (dist / 2.0), my + perp_y * (dist / 2.0))
    radius = math.hypot(p1[0] - center[0], p1[1] - center[1])
    theta1, theta2 = _minor_arc_angles(p1, p2, center)
    return center, radius, theta1, theta2


def recompute_arc_params(p1, p2, center):
    radius = math.hypot(p1[0] - center[0], p1[1] - center[1])
    theta1, theta2 = _minor_arc_angles(p1, p2, center)
    return radius, theta1, theta2


# ---------------------------------------------------------------------------
# 1. 純粋関数: arc 数学
# ---------------------------------------------------------------------------
def test_arc_params_radius_matches_endpoints():
    p1, p2 = (0.0, 0.0), (10.0, 0.0)
    center, radius, t1, t2 = arc_params_from_two_points(p1, p2)
    d1 = math.hypot(p1[0] - center[0], p1[1] - center[1])
    d2 = math.hypot(p2[0] - center[0], p2[1] - center[1])
    assert d1 == pytest.approx(radius, rel=1e-12)
    assert d2 == pytest.approx(radius, rel=1e-12)


def test_arc_params_endpoints_on_theta():
    p1, p2 = (0.0, 0.0), (10.0, 0.0)
    center, radius, t1, t2 = arc_params_from_two_points(p1, p2)
    # theta1, theta2 の角度が p1, p2 (順不同) に対応する
    pts = set()
    for th in (t1, t2):
        x = center[0] + radius * math.cos(math.radians(th))
        y = center[1] + radius * math.sin(math.radians(th))
        pts.add((round(x, 6), round(y, 6)))
    assert (round(p1[0], 6), round(p1[1], 6)) in pts
    assert (round(p2[0], 6), round(p2[1], 6)) in pts


def test_arc_params_minor_arc_span_le_180():
    # 劣弧なので span は 180 度以下
    for p2 in [(10.0, 0.0), (3.0, 4.0), (-5.0, 2.0)]:
        _, _, t1, t2 = arc_params_from_two_points((0.0, 0.0), p2)
        assert 0 < (t2 - t1) <= 180.0 + 1e-9


def test_recompute_arc_params_center_change():
    p1, p2 = (0.0, 0.0), (10.0, 0.0)
    # 中心を (5, 3) に変えると radius は中心-端点距離
    r, t1, t2 = recompute_arc_params(p1, p2, (5.0, 3.0))
    assert r == pytest.approx(math.hypot(5.0, 3.0), rel=1e-12)
    assert 0 < (t2 - t1) <= 180.0 + 1e-9


# ---------------------------------------------------------------------------
# 2. arc を含む geom のエクスポート
# ---------------------------------------------------------------------------
def _pillbox_with_arc() -> MultiRegionGeometry:
    """右辺 (r=a) を円弧にした pillbox。"""
    a, L = 50.0, 100.0
    pts = [(0.0, 0.0), (L, 0.0), (L, a), (0.0, a)]
    center, radius, t1, t2 = arc_params_from_two_points((L, 0.0), (L, a))
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="arc", point_indices=[1, 2], bc_name="PEC",
                center=center, radius=radius, theta1=t1, theta2=t2),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="E-short"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="PEC"),
    ]
    loops = [Loop(id=0, segment_ids=[0, 1, 2, 3])]
    regions = [Region(id=0, name="Vacuum", outer_loop_id=0,
                      material_tag="vacuum", eps_r=1.0)]
    return MultiRegionGeometry(points=pts, segments=segs, loops=loops,
                              regions=regions, unit="mm", mesh_size=10.0)


def test_arc_geom_exports_msh(tmp_path):
    geom = _pillbox_with_arc()
    out = tmp_path / "arc.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    assert out.exists() and out.stat().st_size > 0


def test_arc_geom_exports_geo_with_circle(tmp_path):
    geom = _pillbox_with_arc()
    out = tmp_path / "arc.geo"
    export_geo_multi_region(geom, out)
    text = out.read_text(encoding="utf-8")
    assert "Circle(" in text  # 円弧が GEO に出力される


def test_arc_geom_exports_python_with_circlearc(tmp_path):
    geom = _pillbox_with_arc()
    out = tmp_path / "arc_build.py"
    export_python_script_multi_region(geom, out, msh_output="arc.msh")
    text = out.read_text(encoding="utf-8")
    ast.parse(text)
    assert "addCircleArc" in text


def test_none_bc_excluded_from_physical_groups(tmp_path):
    """bc_name="None" の segment は 1D Physical Group に含まれない (内部境界/軸)。"""
    from axicavity_fem.shared.gmsh_export_occ import inspect_msh_physical_groups
    a, L = 50.0, 100.0
    geom = MultiRegionGeometry(
        points=[(0.0, 0.0), (L, 0.0), (L, a), (0.0, a)],
        segments=[
            Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),   # 軸 r=0
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        ],
        loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
        regions=[Region(id=0, name="Vacuum", outer_loop_id=0,
                        material_tag="vacuum", eps_r=1.0)],
        unit="mm", mesh_size=20.0,
    )
    assert geom.validate() == []
    out = tmp_path / "none_bc.msh"
    export_msh_multi_region(geom, out, mesh_order=1, verbose=0)
    pg = inspect_msh_physical_groups(out)
    names_1d = set(pg.get(1, {}).keys())
    assert "None" not in names_1d
    assert "PEC" in names_1d and "E-short" in names_1d


def test_none_bc_excluded_from_geo_and_python(tmp_path):
    geom = MultiRegionGeometry(
        points=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        segments=[
            Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="E-short"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="PEC"),
        ],
        loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
        regions=[Region(id=0, name="Vacuum", outer_loop_id=0,
                        material_tag="vacuum", eps_r=1.0)],
        unit="mm", mesh_size=5.0,
    )
    geo = tmp_path / "none.geo"
    export_geo_multi_region(geom, geo)
    geo_txt = geo.read_text(encoding="utf-8")
    assert 'Physical Curve("None"' not in geo_txt
    assert 'Physical Curve("PEC"' in geo_txt

    py = tmp_path / "none_build.py"
    export_python_script_multi_region(geom, py, msh_output="none.msh")
    py_txt = py.read_text(encoding="utf-8")
    ast.parse(py_txt)
    assert 'if bc == "None"' in py_txt  # 生成スクリプトで None をスキップ


def test_arc_python_script_runs(tmp_path):
    geom = _pillbox_with_arc()
    script = tmp_path / "arc_build.py"
    msh = tmp_path / "arc_out.msh"
    export_python_script_multi_region(geom, script, msh_output=str(msh))
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, cwd=str(tmp_path))
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert msh.exists() and msh.stat().st_size > 0
