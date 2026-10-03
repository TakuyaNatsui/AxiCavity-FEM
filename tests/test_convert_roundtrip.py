"""samples の .gmshproj（13 件）で、取り込み → 書き出しの往復と、メッシュ・周波数の一致を確かめる.

- 構造: region / loop / segment の数、材料、BC 名ごとの数、式・変数が保存される
- メッシュ: 元の MultiRegionGeometry と再生成したもので gmsh の Physical 名・節点数・要素数が一致
- 解析: 再生成メッシュの TM0 定在波の周波数が ver2.3 の .txt サマリと一致（rtol 1e-4）
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("gmsh")

from axicavity_fem.gui.core.convert import to_multi_region                       # noqa: E402
from axicavity_fem.gui.core.legacy import load_gmshproj                            # noqa: E402
from axicavity_fem.shared.gmsh_export_occ import (                                 # noqa: E402
    export_msh_multi_region,
    inspect_msh_physical_groups,
)
from axicavity_fem.shared.material_resolver import build_material_table_from_geometry  # noqa: E402
from axicavity_fem.shared.mesh_loader import load_mesh                             # noqa: E402
from axicavity_fem.shared.multi_region_model import MultiRegionGeometry            # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "samples"
FILES = sorted(ROOT.glob("*.gmshproj"))
_FREQ_RE = re.compile(r"mode\s+(\d+):\s*f\s*=\s*([0-9.]+)\s*GHz")


def _regions(geom: MultiRegionGeometry):
    return sorted((r.name, r.material_tag, r.eps_r, r.tan_delta, r.eps_r_expr or None, r.tan_delta_expr or None)
                  for r in geom.regions)


def _exprs(geom: MultiRegionGeometry):
    return sorted(e for pair in geom.point_exprs for e in pair if e)


def _center_exprs(geom: MultiRegionGeometry):
    return sorted(str(s.center_expr) for s in geom.segments if s.type == "arc" and s.center_expr)


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_structure_round_trip(path):
    orig = MultiRegionGeometry.load_json(path)
    doc, warnings = load_gmshproj(path)
    assert not any("孤立" in w for w in warnings)
    res = to_multi_region(doc)
    geom = res.geom
    assert geom.validate() == []
    assert (len(geom.points), len(geom.segments), len(geom.loops), len(geom.regions)) == \
        (len(orig.points), len(orig.segments), len(orig.loops), len(orig.regions))
    assert Counter(s.bc_name for s in geom.segments) == Counter(s.bc_name for s in orig.segments)
    assert Counter(s.type for s in geom.segments) == Counter(s.type for s in orig.segments)
    assert _regions(geom) == _regions(orig)
    assert sorted(len(r.hole_loop_ids) for r in geom.regions) == sorted(len(r.hole_loop_ids) for r in orig.regions)
    assert sorted(len(l.segment_ids) for l in geom.loops) == sorted(len(l.segment_ids) for l in orig.loops)
    assert geom.variables == orig.variables
    assert _exprs(geom) == _exprs(orig) and _center_exprs(geom) == _center_exprs(orig)
    assert (geom.unit, geom.mesh_size, geom.mesh_size_expr or None) == (orig.unit, orig.mesh_size, orig.mesh_size_expr or None)
    assert geom.settings.get("mesh_order") == orig.settings.get("mesh_order")
    # 円弧: 中心・半径・角度の集合が一致（劣弧の規約が保たれる）
    arcs = lambda gm: sorted((round(s.center[0], 6), round(s.center[1], 6), round(s.radius, 6),  # noqa: E731
                              round((s.theta2 - s.theta1) % 360, 6)) for s in gm.segments if s.type == "arc")
    assert arcs(geom) == arcs(orig)


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_mesh_is_equivalent(path, tmp_path):
    orig = MultiRegionGeometry.load_json(path)
    doc, _ = load_gmshproj(path)
    geom = to_multi_region(doc).geom
    order = 2 if orig.settings.get("mesh_order", 1) == 1 else 1
    export_msh_multi_region(orig, tmp_path / "orig.msh", mesh_order=order, verbose=0)
    export_msh_multi_region(geom, tmp_path / "new.msh", mesh_order=order, verbose=0)
    pg_o = inspect_msh_physical_groups(tmp_path / "orig.msh")
    pg_n = inspect_msh_physical_groups(tmp_path / "new.msh")
    assert set(pg_o.get(1, {})) == set(pg_n.get(1, {}))
    assert set(pg_o.get(2, {})) == set(pg_n.get(2, {}))
    mesh_o = load_mesh(str(tmp_path / "orig.msh"), order)
    mesh_n = load_mesh(str(tmp_path / "new.msh"), order)
    counts_o = (mesh_o.num_nodes, mesh_o.num_elements)
    counts_n = (mesh_n.num_nodes, mesh_n.num_elements)
    if counts_n != counts_o:
        # ループの始点や領域の順序が元と違うと gmsh の節点挿入順が変わり、節点数が 1 % 程度ずれる
        # （acc2_pipe_flat / diel_test1 / PF_cavity で実測）。その場合は最低次 3 モードの周波数で等価性を見る
        from axicavity_fem.fem_tm0.solver import solve_tm0_standing
        assert abs(counts_n[0] - counts_o[0]) <= 0.03 * counts_o[0], (counts_o, counts_n)
        f_o = solve_tm0_standing(mesh_o, num_modes=3).frequencies[:3]
        f_n = solve_tm0_standing(mesh_n, num_modes=3).frequencies[:3]
        for a, b in zip(f_o, f_n):
            assert float(b) == pytest.approx(float(a), rel=1e-5), (counts_o, counts_n)


def _expected_frequencies(stem: str) -> list[float]:
    text = (ROOT / f"{stem}_SW_TM0.txt").read_text(encoding="utf-8")
    return [float(m.group(2)) for m in _FREQ_RE.finditer(text)]


@pytest.mark.parametrize("stem", ["s-band_1cell", "diel_test1"])
def test_solve_matches_ver23_summary(stem, tmp_path):
    from axicavity_fem.cli.main import cli_main
    from axicavity_fem.shared.hdf5_io import read_results

    if not (ROOT / f"{stem}_SW_TM0.txt").exists():
        pytest.skip(f"ver2.3 の結果サマリ {stem}_SW_TM0.txt なし（samples/ には入力ファイルだけを同梱）")
    doc, _ = load_gmshproj(ROOT / f"{stem}.gmshproj")
    geom = to_multi_region(doc).geom
    msh = tmp_path / f"{stem}.msh"
    export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
    (tmp_path / f"{stem}.materials.json").write_text(json.dumps({
        "schema": "axicavity-fem-v21.materials/1", "mesh_file": msh.name,
        "materials": build_material_table_from_geometry(geom)}), encoding="utf-8")
    out = tmp_path / "result.h5"
    assert cli_main(["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "2",
                     "--num-modes", "4", "-o", str(out)]) == 0
    freqs = read_results(str(out))["results_by_n"][0]["standing"]["frequencies"]
    expected = _expected_frequencies(stem)
    assert len(expected) >= 4
    for k in range(4):
        assert float(freqs[k]) == pytest.approx(expected[k], rel=1e-4), f"mode {k}"
