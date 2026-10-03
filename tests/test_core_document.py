"""ver3 ドキュメント（AxiDocument）の生成・JSON 往復・不正な入力の補正."""

import copy
import json

import pytest

from axicavity_fem.gui.core.document import (
    SCHEMA_VERSION,
    AxiDocument,
    Param,
    RegionSetting,
    SketchPoint,
    create_empty_document,
    new_id,
)
from axicavity_fem.gui.core.hashes import geometry_hash, mesh_hash, model_hash
from axicavity_fem.gui.core.serialize import (
    DocumentParseError,
    document_to_dict,
    parse_document,
    serialize_document,
)
from axicavity_fem.gui.core.sketch.model import add_rectangle


def test_creates_empty_document():
    doc = create_empty_document("test")
    assert doc.schemaVersion == SCHEMA_VERSION
    assert doc.meta.name == "test" and doc.meta.units == "mm"
    assert doc.sketch.entities == [] and doc.sketch.plane.plane == "XY"
    assert doc.regions == [] and doc.boundaries == {}
    assert (doc.mesh.size, doc.mesh.order) == (5.0, 2)
    assert (doc.analysis.type, doc.analysis.wave, doc.analysis.numModes) == ("tm0", "standing", 10)
    assert (doc.post.cond, doc.post.beta) == (5.8e7, 1.0)
    assert doc.feature(doc.sketch.id) is doc.sketch and doc.feature("nope") is None


def _sample() -> AxiDocument:
    doc = create_empty_document("rt")
    doc.params.append(Param(id=new_id(), name="L", expression="100"))
    line_ids, _ = add_rectangle(doc.sketch, (0, 0), (100, 50))
    doc.sketch.entities.append(SketchPoint(id=new_id(), x=1, y=2, xExpr="L/100"))
    doc.regions.append(RegionSetting(key="k", anchor=(50.0, 25.0), name="Ceramic",
                                     materialTag="ceramic", epsR=9.64, tanDelta=5.7e-6))
    doc.boundaries[line_ids[0]] = "None"
    doc.boundaries[line_ids[2]] = "E-short"
    doc.mesh.size, doc.mesh.sizeExpr = 2.5, "L/40"
    doc.analysis.type, doc.analysis.wave, doc.analysis.phases = "hom", "traveling", "0:180:20"
    doc.analysis.targetFreqGHz = 2.856
    doc.report.animate = True
    doc.view.zmin, doc.view.zmax = -10.0, 110.0
    return doc


def test_round_trips_through_json():
    doc = _sample()
    text = serialize_document(doc)
    restored = parse_document(text)
    assert restored == doc
    data = json.loads(text)
    assert list(data) == ["schemaVersion", "meta", "params", "sketch", "regions", "boundaries",
                          "mesh", "analysis", "post", "report", "view"]
    assert data["sketch"]["type"] == "sketch"
    point = next(e for e in data["sketch"]["entities"] if e.get("xExpr"))
    assert point == {"type": "point", "id": point["id"], "x": 1, "y": 2, "xExpr": "L/100"}   # None は省略
    assert data["regions"][0]["anchor"] == [50.0, 25.0]
    assert "tanDeltaExpr" not in data["regions"][0]
    assert data["view"] == {"zmin": -10.0, "zmax": 110.0}


def test_rejects_unknown_schema_versions_and_invalid_json():
    with pytest.raises(DocumentParseError):
        parse_document(json.dumps({"schemaVersion": 999}))
    with pytest.raises(DocumentParseError):
        parse_document(json.dumps({"schemaVersion": True}))
    with pytest.raises(DocumentParseError):
        parse_document("{not json")
    with pytest.raises(DocumentParseError):
        parse_document(json.dumps({"schemaVersion": 1, "sketch": {"entities": [{"type": "spline"}]}}))


def test_fills_missing_or_invalid_fields():
    partial = {
        "schemaVersion": SCHEMA_VERSION,
        "meta": {"name": "p", "units": "furlong"},
        "params": [{"name": "a", "expression": "1"}, {"expression": "no-name"}],
        "regions": [
            {"key": "k1", "anchor": [1, 2], "name": "Vacuum", "epsR": "bad", "tanDelta": -1},
            {"key": "k2", "anchor": [1]},
            {"anchor": [1, 2]},
        ],
        "boundaries": {"c1": "PEC", "c2": "Bogus", "c3": "None"},
        "mesh": {"size": -3, "order": 3},
        "analysis": {"type": "tm1", "wave": "sideways", "numModes": 3.6, "targetFreqGHz": -1,
                     "runPost": False},
        "post": {"cond": 0, "beta": "x"},
        "report": {"dpi": 0, "animate": "yes"},
    }
    doc = parse_document(json.dumps(partial))
    assert doc.meta.units == "mm"
    assert [(p.name, p.expression) for p in doc.params] == [("a", "1")]
    assert [(r.key, r.epsR, r.tanDelta) for r in doc.regions] == [("k1", 1.0, 0.0)]
    assert doc.boundaries == {"c1": "PEC", "c3": "None"}
    assert (doc.mesh.size, doc.mesh.order) == (5.0, 2)
    a = doc.analysis
    assert (a.type, a.wave, a.numModes, a.targetFreqGHz, a.runPost) == ("tm0", "standing", 4, None, False)
    assert (doc.post.cond, doc.post.beta) == (5.8e7, 1.0)
    assert (doc.report.dpi, doc.report.animate) == (120, False)
    assert doc.sketch.entities == []       # sketch が無くても空のスケッチ
    assert "namedSelections" not in document_to_dict(doc)


def test_hashes_track_the_right_changes():
    sample = _sample()
    doc = copy.deepcopy(sample)
    g0, m0, h0 = geometry_hash(doc), mesh_hash(doc), model_hash(doc)
    # 材料値は形状・メッシュのハッシュを変えない
    doc.regions[0].epsR = 4.0
    assert (geometry_hash(doc), mesh_hash(doc)) == (g0, m0) and model_hash(doc) != h0
    # 境界条件の指定はメッシュのハッシュだけを変える
    doc.boundaries["x"] = "PEC"
    assert geometry_hash(doc) == g0 and mesh_hash(doc) != m0
    # 形状の変更は全部を変える
    m1 = mesh_hash(doc)
    doc.sketch.entities[0].x += 1
    assert geometry_hash(doc) != g0 and mesh_hash(doc) != m1
    # meta（名前・日時）は model_hash に入らない
    doc2 = copy.deepcopy(sample)
    doc2.meta.name, doc2.meta.modifiedAt = "other", "2026-01-01"
    assert model_hash(doc2) == h0
