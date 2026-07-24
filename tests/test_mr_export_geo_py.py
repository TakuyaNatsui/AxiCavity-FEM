"""ver2.1: Multi-Region の `.geo` / Python スクリプト出力のスモークテスト.

`.geo` (gmsh の unrolled) は gmsh で再読込できること、Python スクリプトは
正しい Python としてパースでき、内部に必要な API 呼び出しが含まれることを確認。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from axicavity_fem.shared.gmsh_export_occ import (
    export_geo_multi_region,
    export_python_script_multi_region,
)
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)


def _two_region_pillbox() -> MultiRegionGeometry:
    a = 50.0
    L = 100.0
    Lh = L / 2.0
    pts = [
        (0.0, 0.0), (Lh, 0.0), (Lh, a), (0.0, a),
        (L, 0.0), (L, a),
    ]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="E-short"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
    ]
    loops = [
        Loop(id=0, segment_ids=[0, 1, 2, 3]),
        Loop(id=1, segment_ids=[4, 5, 6, 1]),
    ]
    regions = [
        Region(id=0, name="Left", outer_loop_id=0,
               material_tag="vacuum", eps_r=1.0),
        Region(id=1, name="Right", outer_loop_id=1,
               material_tag="dielectric_1", eps_r=4.0),
    ]
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=loops, regions=regions,
        unit="mm", mesh_size=10.0,
    )


def test_export_geo_writes_readable_text(tmp_path):
    geom = _two_region_pillbox()
    out = tmp_path / "two_region.geo"
    export_geo_multi_region(geom, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # OCC モード宣言
    assert 'SetFactory("OpenCASCADE")' in text
    # 構築要素
    assert "Point(" in text
    assert "Line(" in text
    assert "Curve Loop(" in text
    assert "Plane Surface(" in text
    # 多領域なので fragment あり
    assert "BooleanFragments" in text
    # Physical Group
    assert 'Physical Surface("vacuum"' in text
    assert 'Physical Surface("dielectric_1"' in text
    assert 'Physical Curve("PEC"' in text


def test_export_geo_can_be_meshed_by_gmsh(tmp_path):
    """生成した .geo を gmsh.open + generate で読み込め、Physical Group が復元できる。"""
    import gmsh

    geom = _two_region_pillbox()
    out = tmp_path / "two_region.geo"
    export_geo_multi_region(geom, out)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0.0)
        gmsh.open(str(out))
        # .geo に `Mesh 2;` 命令を含めてあるので gmsh.open でメッシュ生成済み
        names_2d = {gmsh.model.getPhysicalName(2, t)
                    for d, t in gmsh.model.getPhysicalGroups(2)}
        names_1d = {gmsh.model.getPhysicalName(1, t)
                    for d, t in gmsh.model.getPhysicalGroups(1)}
        assert {"vacuum", "dielectric_1"} <= names_2d
        assert {"PEC", "E-short", "M-short"} <= names_1d
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def test_export_python_script_parses(tmp_path):
    geom = _two_region_pillbox()
    out = tmp_path / "build_mesh.py"
    export_python_script_multi_region(geom, out, mesh_order=1,
                                       msh_output="out.msh")
    text = out.read_text(encoding="utf-8")
    # 構文として有効
    ast.parse(text)
    # 主要な gmsh API 呼び出しが含まれる
    assert "gmsh.model.occ.addPoint" in text
    assert "gmsh.model.occ.addLine" in text
    assert "gmsh.model.occ.addCurveLoop" in text
    assert "gmsh.model.occ.addPlaneSurface" in text
    assert "gmsh.model.occ.fragment" in text  # 多領域なので必要
    assert "gmsh.model.addPhysicalGroup" in text
    assert "gmsh.model.mesh.generate(2)" in text
    # 領域情報の埋め込み
    assert "vacuum" in text and "dielectric_1" in text


def test_export_python_script_runs_and_produces_msh(tmp_path):
    """生成されたスクリプトを実行すると .msh が出力される。"""
    geom = _two_region_pillbox()
    script = tmp_path / "build_mesh.py"
    msh = tmp_path / "out.msh"
    export_python_script_multi_region(geom, script, mesh_order=1,
                                       msh_output=str(msh))
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, cwd=str(tmp_path))
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert msh.exists() and msh.stat().st_size > 0


def test_export_python_script_single_region_no_fragment(tmp_path):
    """単一領域・穴なしのスクリプトには fragment が含まれない（恒等マップ）。"""
    geom = MultiRegionGeometry(
        points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        segments=[
            Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name="PEC"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="E-short"),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name="PEC"),
        ],
        loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
        regions=[Region(id=0, name="V", outer_loop_id=0,
                        material_tag="vacuum", eps_r=1.0)],
        unit="mm", mesh_size=5.0,
    )
    out = tmp_path / "single.py"
    export_python_script_multi_region(geom, out, msh_output="single.msh")
    text = out.read_text(encoding="utf-8")
    assert "fragment 不要" in text
    assert "occ.fragment(" not in text
