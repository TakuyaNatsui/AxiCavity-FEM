"""旧形式の取り込みと書き出し: 旧 ver2 単一ループ・schema 2.1〜2.3・Superfish .af."""

import json
from pathlib import Path

import pytest

from axicavity_fem.gui.core.convert import check_geometry, effective_bc, sketch_profiles, to_multi_region
from axicavity_fem.gui.core.document import SketchArc, SketchLine
from axicavity_fem.gui.core.legacy import load_gmshproj, load_superfish, save_gmshproj, save_superfish
from axicavity_fem.gui.core.sketch.edit_ops import close_polyline
from axicavity_fem.shared.multi_region_model import Loop, MultiRegionGeometry, Region, Segment
from axicavity_fem.shared.superfish_io import SuperfishExportError

ROOT = Path(__file__).resolve().parents[1] / "samples"


def _curves(doc):
    return [e for e in doc.sketch.entities if isinstance(e, (SketchLine, SketchArc))]


def test_legacy_single_loop_format(tmp_path):
    legacy = {
        "points": [[0, 0], [100, 0], [100, 50], [0, 50]],
        "segments": [
            {"type": "line", "points": [0, 1], "physical_name": "None"},      # 軸
            {"type": "line", "points": [1, 2], "physical_name": "PEC"},
            {"type": "line", "points": [2, 3], "physical_name": "PEC"},
            {"type": "line", "points": [3, 0], "physical_name": "E-short"},
        ],
        "loop_closed": True,
        "settings": {"unit": "cm", "xmin": "0", "xmax": "120", "ymin": "0", "ymax": "60",
                     "mesh_size": "2.5", "mesh_order": 0},
    }
    path = tmp_path / "old.gmshproj"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    doc, warnings = load_gmshproj(path)
    assert warnings and "旧形式" in warnings[0]
    assert doc.meta.name == "old" and doc.meta.units == "cm"
    assert (doc.mesh.size, doc.mesh.order) == (2.5, 1)
    assert (doc.view.zmin, doc.view.zmax, doc.view.rmin, doc.view.rmax) == (0, 120, 0, 60)
    curves = _curves(doc)
    assert len(curves) == 4 and len(doc.regions) == 1
    assert effective_bc(doc, curves[0].id) == ("None", "axis")             # 旧 "None" は自動判定に任せる
    assert effective_bc(doc, curves[1].id) == ("PEC", "default")           # 既定と同じ指定は整理で消える
    assert effective_bc(doc, curves[3].id) == ("E-short", "explicit")
    assert doc.regions[0].name == "Vacuum" and doc.regions[0].key == sketch_profiles(doc)[0].id
    assert [i.code for i in check_geometry(doc)] == []


def test_schema_21_without_expression_fields(tmp_path):
    geom = MultiRegionGeometry(
        points=[(0, 0), (100, 0), (100, 50), (0, 50)],
        segments=[Segment(id=i, type="line", point_indices=[i, (i + 1) % 4], bc_name="PEC" if i else "None")
                  for i in range(4)],
        loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
        regions=[Region(id=0, name="Vacuum", outer_loop_id=0, material_tag="vacuum", eps_r=1.0)],
        unit="mm", mesh_size=4.0)
    data = geom.to_dict()
    data["schema_version"] = "2.1"
    for key in ("point_exprs", "mesh_size_expr", "variables"):
        data.pop(key, None)
    path = tmp_path / "v21.gmshproj"
    path.write_text(json.dumps(data), encoding="utf-8")
    doc, warnings = load_gmshproj(path)
    assert warnings == [] and doc.params == [] and doc.mesh.size == 4.0
    assert len(_curves(doc)) == 4 and len(doc.regions) == 1
    assert doc.boundaries == {}                                            # 全部自動判定と同じ


@pytest.mark.parametrize("stem", ["diel_test1", "s-band_1cell", "sphere50mm"])
def test_save_and_reload_gmshproj(stem, tmp_path):
    doc, _ = load_gmshproj(ROOT / f"{stem}.gmshproj")
    out = tmp_path / f"{stem}_v3.gmshproj"
    warnings = save_gmshproj(doc, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "2.3"
    doc2, _ = load_gmshproj(out)
    assert len(_curves(doc2)) == len(_curves(doc)) and len(doc2.regions) == len(doc.regions)
    assert sorted((r.name, r.materialTag, r.epsR, r.tanDelta) for r in doc2.regions) == sorted(
        (r.name, r.materialTag, r.epsR, r.tanDelta) for r in doc.regions)
    assert [(p.name, p.expression) for p in doc2.params] == [(p.name, p.expression) for p in doc.params]
    assert to_multi_region(doc2).geom.to_dict()["segments"] == to_multi_region(doc).geom.to_dict()["segments"]


def test_superfish_closed_sample():
    doc, warnings = load_superfish(ROOT / "acc2_pipe_flat.af", "mm")
    assert warnings == [] and doc.meta.name == "acc2_pipe_flat"
    curves = _curves(doc)
    assert sum(isinstance(e, SketchLine) for e in curves) == 31
    assert sum(isinstance(e, SketchArc) for e in curves) == 44
    assert len(doc.regions) == 1 and doc.boundaries == {}                 # BC は自動（軸は None、他は PEC）
    geom = to_multi_region(doc).geom
    assert geom.validate() == [] and len(geom.regions) == 1
    assert sum(s.bc_name == "None" for s in geom.segments) >= 1


def test_superfish_open_outline_then_close(tmp_path):
    path = tmp_path / "open.af"
    path.write_text("open\n\n$reg kprob=1 $\n$po x=0.0, y=0.0 $\n$po x=10.0, y=0.0 $\n$po x=10.0, y=5.0 $\n",
                    encoding="utf-8")
    doc, warnings = load_superfish(path, "cm")
    assert doc.regions == [] and any("閉じて" in w for w in warnings)
    assert len(_curves(doc)) == 2 and doc.meta.units == "cm"
    assert [i.code for i in check_geometry(doc)] == ["no_profile"]
    line_id, _ = close_polyline(doc.sketch)
    assert line_id is not None
    geom = to_multi_region(doc).geom
    assert len(geom.regions) == 1 and len(geom.segments) == 3


def test_save_superfish(tmp_path):
    doc, _ = load_gmshproj(ROOT / "s-band_1cell.gmshproj")
    out = tmp_path / "cell.af"
    assert save_superfish(doc, out) == []
    text = out.read_text(encoding="utf-8")
    assert text.count("$po") == 9                                          # 8 segment + 始点に戻る 1 行
    multi, _ = load_gmshproj(ROOT / "diel_test1.gmshproj")
    with pytest.raises(SuperfishExportError):
        save_superfish(multi, tmp_path / "multi.af")
