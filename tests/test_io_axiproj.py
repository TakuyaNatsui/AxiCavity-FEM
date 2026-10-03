"""プロジェクトファイル .axiproj（version 1）の形式（gui/io/axiproj.py、GUI 非依存）.

- マニフェストの書き込み/読み込みの往復、拡張子の補完、データフォルダ、activeResult だけの書き換え
- 別アプリ・壊れた JSON・未対応バージョン・形状無しの拒否
- 結果フォルダ: 番号の採番、走査（状態の判定・新しい順・名前・要約）、削除、データフォルダの複製
- ファイル名の整形、command.log
"""

import json

import pytest

from axicavity_fem.gui.core.document import Param, create_empty_document
from axicavity_fem.gui.core.hashes import model_hash
from axicavity_fem.gui.io import axiproj as p


def _doc(name="Cavity"):
    doc = create_empty_document(name)
    doc.params.append(Param(id="p1", name="R", expression="50"))
    return doc


def test_write_read_roundtrip(tmp_path):
    doc = _doc()
    doc.mesh.size = 7.0
    doc.analysis.type = "hom"
    written = p.write_project(tmp_path / "buncher", p.ProjectFile(
        document=doc, active_result="0002-tm0-sw", ui={"tab": "physics"}, generator="test"))
    assert written.name == "buncher.axiproj"
    assert p.data_dir_of(written).is_dir() and p.data_dir_of(written).name == "buncher.axiproj.data"
    assert p.project_name(written) == "buncher"
    assert p.results_dir_of(written) == p.data_dir_of(written) / "results"
    assert p.mesh_dir_of(written).name == "mesh" and p.exports_dir_of(written).name == "exports"

    manifest = json.loads(written.read_text(encoding="utf-8"))
    assert manifest["app"] == "AxiCavity-FEM" and manifest["version"] == 1
    assert manifest["document"]["mesh"]["size"] == 7.0 and manifest["document"]["analysis"]["type"] == "hom"

    loaded = p.read_project(written)
    assert model_hash(loaded.document) == model_hash(doc)
    assert loaded.document.params[0].name == "R"
    assert loaded.active_result == "0002-tm0-sw"
    assert loaded.ui == {"tab": "physics"} and loaded.generator == "test" and loaded.saved

    # activeResult だけの書き換えで形状は変わらない
    p.update_manifest(written, activeResult=None)
    again = p.read_project(written)
    assert again.active_result is None
    assert model_hash(again.document) == model_hash(doc)


def test_rejects_other_formats(tmp_path):
    other = tmp_path / "other.axiproj"
    other.write_text(json.dumps({"app": "Cavity3D-FEM", "version": 2, "document": {}}), encoding="utf-8")
    with pytest.raises(p.ProjectFormatError, match="ではありません"):
        p.read_project(other)
    broken = tmp_path / "broken.axiproj"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(p.ProjectFormatError):
        p.read_project(broken)
    future = tmp_path / "future.axiproj"
    future.write_text(json.dumps({"app": "AxiCavity-FEM", "version": 2, "document": {}}), encoding="utf-8")
    with pytest.raises(p.ProjectFormatError, match="未対応"):
        p.read_project(future)
    empty = tmp_path / "empty.axiproj"
    empty.write_text(json.dumps({"app": "AxiCavity-FEM", "version": 1}), encoding="utf-8")
    with pytest.raises(p.ProjectFormatError, match="document"):
        p.read_project(empty)
    with pytest.raises(p.ProjectFormatError):
        p.read_project(tmp_path / "missing.axiproj")


def _fake_result(data_dir, rid, kind="tm0-sw", done=True, meta=None):
    d = data_dir / "results" / rid
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"schemaVersion": 1, "kind": kind}), encoding="utf-8")
    if done:
        (d / p.raw_file_name(kind)).write_bytes(b"h5")
    if meta:
        p.write_result_meta(d, **meta)
    return d


def test_results_scan_and_numbering(tmp_path):
    data = tmp_path / "x.axiproj.data"
    assert p.scan_results(data) == [] and p.next_result_id(data, "tm0-sw") == "0001-tm0-sw"
    _fake_result(data, "0001-tm0-sw", meta={"label": "粗いメッシュ", "geometryHash": "g1", "meshHash": "h1",
                                             "modelHash": "m1", "status": "done", "number": 1,
                                             "summary": {"modes": [{"f_GHz": 2.856, "Q": 1.2e4}]},
                                             "files": {"raw": "model_SW_TM0.h5"}})
    _fake_result(data, "0002-hom-tw", kind="hom-tw", done=False, meta={"status": "running", "number": 2})
    _fake_result(data, "0003-tm0-sw", done=False, meta={"status": "error", "error": "boom"})
    _fake_result(data, "0004-tm0-tw", kind="tm0-tw", done=True, meta={"status": "running"})   # 完了後に落ちた
    _fake_result(data, "0005-tm0-sw", done=False)                                              # 何も書けずに落ちた
    (data / "results" / "notes").mkdir()                                                       # job.json の無いフォルダ
    entries = p.scan_results(data)
    assert [e.id for e in entries] == ["0005-tm0-sw", "0004-tm0-tw", "0003-tm0-sw", "0002-hom-tw", "0001-tm0-sw"]
    by_id = {e.id: e for e in entries}
    first = by_id["0001-tm0-sw"]
    assert first.status == "done" and first.label == "粗いメッシュ" and first.number == 1
    assert (first.geometry_hash, first.mesh_hash, first.model_hash) == ("g1", "h1", "m1")
    assert first.summary["modes"][0]["f_GHz"] == 2.856 and first.created_at
    assert first.file("raw") == first.dir / "model_SW_TM0.h5" and first.has_output
    assert by_id["0002-hom-tw"].status == "running" and by_id["0002-hom-tw"].kind == "hom-tw"
    assert by_id["0003-tm0-sw"].status == "error" and by_id["0003-tm0-sw"].error == "boom"
    assert by_id["0004-tm0-tw"].status == "done" and by_id["0004-tm0-tw"].has_output
    assert by_id["0005-tm0-sw"].status == "interrupted" and not by_id["0005-tm0-sw"].has_output
    assert p.next_result_id(data, "hom-sw") == "0006-hom-sw"

    p.delete_result(by_id["0003-tm0-sw"])
    assert [e.id for e in p.scan_results(data)] == ["0005-tm0-sw", "0004-tm0-tw", "0002-hom-tw", "0001-tm0-sw"]


def test_copy_data_dir_and_names(tmp_path):
    proj = tmp_path / "proj.axiproj"
    data = p.data_dir_of(proj)
    _fake_result(data, "0001-tm0-sw")
    p.append_command_log(data, "axicavity-fem solve --type tm0")
    p.copy_data_dir(proj, tmp_path / "copy.axiproj")
    copied = p.data_dir_of(tmp_path / "copy.axiproj")
    assert [e.id for e in p.scan_results(copied)] == ["0001-tm0-sw"]
    log = (copied / "command.log").read_text(encoding="utf-8")
    assert log.startswith("[") and log.rstrip().endswith("axicavity-fem solve --type tm0")
    p.copy_data_dir(proj, proj)                                       # 同じ場所は何もしない
    assert (data / "results" / "0001-tm0-sw").exists()

    assert p.sanitize_file_name('a/b:c*?"<>|') == "a_b_c_"
    assert p.sanitize_file_name("   ") == "cavity"
    assert p.suggested_file_name(_doc("S-band 1cell")) == "S-band 1cell.axiproj"
    assert p.normalize_project_path(tmp_path / "x").name == "x.axiproj"
    assert p.normalize_project_path(tmp_path / "x.axiproj").name == "x.axiproj"
    assert p.result_kind(_doc().analysis) == "tm0-sw"
    assert p.raw_file_name("hom-tw") == "model_TW_HOM.h5"
