"""ProjectController（.axiproj のプロジェクト管理）の検証（offscreen、gmsh 不要）.

新規 → 作図 → 名前を付けて保存 → 変更で未保存 → Undo で戻る → 開き直し → 結果フォルダの関係（同じ / 形状変更前 /
メッシュ変更前 / 設定変更前）→ 名前の変更・削除 → 名前を付けて保存でデータフォルダごと複製 → 旧 .gmshproj / .af の
取り込み（未保存の新規）→ 保存 → 再読込の等価性（サンプル 13 件）。
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
from PySide6 import QtWidgets                                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.core.convert import to_multi_region      # noqa: E402
from axicavity_fem.gui.core.hashes import model_hash            # noqa: E402
from axicavity_fem.gui.core.legacy import save_superfish        # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_rectangle, count_entities  # noqa: E402
from axicavity_fem.gui.io import axiproj                        # noqa: E402
from axicavity_fem.gui.project.controller import ProjectController  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _env():
    store = DocumentStore()
    project = ProjectController(store, generator="test")
    project.new()
    return store, project


def _draw_box(store, x0=0, y0=0, x1=100, y1=50):
    store.update_active_sketch(lambda s: add_rectangle(s, (x0, y0), (x1, y1)) is not None)


def _fake_result(project, rid, kind="tm0-sw", **meta):
    d = project.results_dir / rid
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
    (d / axiproj.raw_file_name(kind)).write_bytes(b"h5")
    geom, mesh, model = project.current_hashes()
    fields = dict(number=int(rid[:4]), kind=kind, status="done", geometryHash=geom, meshHash=mesh, modelHash=model)
    fields.update(meta)
    axiproj.write_result_meta(d, **fields)
    return d


def test_project_lifecycle(app, tmp_path):
    store, project = _env()
    changes = []
    project.changed.connect(lambda: changes.append(project.is_dirty()))
    assert project.path is None and not project.is_dirty() and project.name == "Untitled"
    _draw_box(store)
    assert project.is_dirty() and changes[-1] is True
    with pytest.raises(ValueError):
        project.save()
    with pytest.raises(ValueError):
        project.prepare_run()

    # 名前を付けて保存（拡張子は補完。データフォルダができる。名前は文書にも入る）
    path = project.save_as(tmp_path / "work" / "cavity")
    assert path.name == "cavity.axiproj" and not project.is_dirty()
    assert project.data_dir == axiproj.data_dir_of(path) and project.data_dir.is_dir()
    assert project.name == "cavity" and store.document.meta.name == "cavity"
    assert project.results == [] and project.mesh_dir == project.data_dir / "mesh"

    # 設定の変更で未保存、Undo で保存時の状態に戻れば未保存でなくなる
    store.set_mesh(size=3.0)
    assert project.is_dirty()
    store.undo()
    assert not project.is_dirty()
    store.set_boundary([e.id for e in store.active_sketch().entities][4:5], "E-short")
    assert project.is_dirty()
    assert project.save() == path and not project.is_dirty()

    # 結果フォルダと現在のモデルの関係（ハッシュで判定）
    _fake_result(project, "0001-tm0-sw", label="最初")
    project.refresh_results()
    assert [e.id for e in project.results] == ["0001-tm0-sw"]
    entry = project.results[0]
    assert entry.status == "done" and entry.label == "最初"
    assert project.result_relation(entry) == ""
    store.set_analysis(numModes=3)                       # ソルバー設定だけ → 設定変更前
    assert project.result_relation(entry) == "settings"
    store.set_mesh(order=1)                              # メッシュ設定 → メッシュ変更前
    assert project.result_relation(entry) == "mesh"
    store.undo()
    store.undo()
    assert project.result_relation(entry) == ""
    _draw_box(store, 200, 0, 250, 20)                    # 形状 → 形状変更前
    assert project.result_relation(entry) == "geometry"
    store.undo()
    assert project.result_relation(entry) == "" and not project.is_dirty()
    assert project.prepare_run() == "0002-tm0-sw"        # 自動保存 + 次の番号
    store.set_analysis(type="hom", wave="traveling")
    assert project.prepare_run() == "0002-hom-tw" and not project.is_dirty()

    # 表示中の結果（マニフェストに記録）、名前の変更
    assert project.select_result("0001-tm0-sw")
    assert axiproj.read_project(path).active_result == "0001-tm0-sw"
    project.rename_result("0001-tm0-sw", "  細かい  ")
    assert project.result("0001-tm0-sw").label == "細かい"

    # 開き直す（別のウィンドウ相当）: 結果・表示中の結果・未保存でない
    store2, project2 = _env()
    assert project2.open(path) == []
    assert project2.path == path and not project2.is_dirty() and project2.name == "cavity"
    assert [e.id for e in project2.results] == ["0001-tm0-sw"] and project2.active_result_id == "0001-tm0-sw"
    assert count_entities(store2.active_sketch()) == count_entities(store.active_sketch())
    assert store2.document.analysis.type == "hom"
    assert project2.result_relation(project2.results[0]) == "settings"   # hom-tw に変えた後に保存したので

    # 名前を付けて保存: データフォルダ（結果・command.log）ごと複製し、元はそのまま
    project.log_command("axicavity-fem solve --type tm0 -m model.msh")
    copy = project.save_as(tmp_path / "work" / "cavity_v2.axiproj")
    assert project.path == copy and project.name == "cavity_v2"
    assert [e.id for e in project.results] == ["0001-tm0-sw"]
    assert project.results[0].dir.parent == axiproj.data_dir_of(copy) / "results"
    assert (axiproj.data_dir_of(copy) / "command.log").exists()
    assert (axiproj.data_dir_of(path) / "results" / "0001-tm0-sw").exists()

    # 削除（実行中は消せない）
    project.set_running(project.results[0].dir)
    with pytest.raises(RuntimeError):
        project.delete_result("0001-tm0-sw")
    project.set_running(None)
    project.delete_result("0001-tm0-sw")
    assert project.results == [] and project.active_result_id is None
    assert not (axiproj.data_dir_of(copy) / "results" / "0001-tm0-sw").exists()

    # 新規で全部初期化
    project.new()
    assert project.path is None and not project.is_dirty() and store.document.params == []
    assert count_entities(store.active_sketch())["line"] == 0


def test_register_and_finish_result(app, tmp_path):
    store, project = _env()
    _draw_box(store)
    project.save_as(tmp_path / "run.axiproj")
    rid = project.prepare_run()
    result_dir = project.results_dir / rid
    result_dir.mkdir(parents=True)
    (result_dir / "job.json").write_text(json.dumps({"kind": "tm0-sw"}), encoding="utf-8")
    project.register_submitted(result_dir, "tm0-sw")
    assert project.running_dir == result_dir and project.active_result_id == rid
    assert project.results[0].status == "running"
    with pytest.raises(RuntimeError):
        project.open(tmp_path / "run.axiproj")           # 実行中は切り替えない
    (result_dir / "model_SW_TM0.h5").write_bytes(b"h5")
    project.finish_result(result_dir, "done", summary={"modes": [{"f_GHz": 1.0}]},
                          files={"raw": "model_SW_TM0.h5", "processed": "model_SW_TM0_processed.h5"})
    assert project.running_dir is None
    entry = project.results[0]
    assert entry.status == "done" and entry.summary["modes"][0]["f_GHz"] == 1.0
    assert entry.file("processed") == result_dir / "model_SW_TM0_processed.h5"
    assert entry.geometry_hash == project.current_hashes()[0] and entry.finished_at

    # 実行中のまま開き直すと「中断」
    store2, project2 = _env()
    rid2 = project2.open(tmp_path / "run.axiproj") == [] and project2.prepare_run()
    d2 = project2.results_dir / rid2
    d2.mkdir()
    (d2 / "job.json").write_text(json.dumps({"kind": "tm0-sw"}), encoding="utf-8")
    project2.register_submitted(d2, "tm0-sw")
    store3, project3 = _env()
    project3.open(tmp_path / "run.axiproj")
    assert {e.id: e.status for e in project3.results} == {rid: "done", rid2: "interrupted"}


def test_open_legacy_and_wrong_formats(app, tmp_path):
    store, project = _env()
    sample = SAMPLES / "diel_test1.gmshproj"
    warnings = project.open(sample)
    assert isinstance(warnings, list)
    assert project.path is None and project.is_dirty() and project.name == "diel_test1"
    assert len(store.profiles) >= 2
    before = model_hash(store.document)
    path = project.save_as(tmp_path / "diel.axiproj")
    assert not project.is_dirty() and store.document.meta.name == "diel"

    store2, project2 = _env()
    project2.open(path)
    assert model_hash(store2.document) == before
    geom1, geom2 = to_multi_region(store.document).geom, to_multi_region(store2.document).geom
    assert [s.bc_name for s in geom1.segments] == [s.bc_name for s in geom2.segments]

    # Superfish（現在の単位に換算して取り込み）
    af = tmp_path / "shape.af"
    _draw_box(store2, 0, 0, 100, 50)
    store2.new_document()
    _draw_box(store2, 0, 0, 100, 50)
    save_superfish(store2.document, af)
    project2.open(af)
    assert project2.path is None and project2.is_dirty() and project2.name == "shape"
    assert len(store2.profiles) == 1

    # 開けない形式・壊れたプロジェクトでは今の状態のまま
    with pytest.raises(axiproj.ProjectFormatError):
        project.open(tmp_path / "x.txt")
    bad = tmp_path / "bad.axiproj"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(axiproj.ProjectFormatError):
        project.open(bad)
    assert project.path == path and not project.is_dirty()


@pytest.mark.parametrize("sample", sorted(p.name for p in SAMPLES.glob("*.gmshproj")))
def test_samples_survive_project_roundtrip(app, tmp_path, sample):
    """ver2.3 のサンプルを取り込み → .axiproj に保存 → 開き直しても変換結果（MultiRegionGeometry）が同じ."""
    store, project = _env()
    project.open(SAMPLES / sample)
    original = to_multi_region(store.document).geom
    path = project.save_as(tmp_path / sample.replace(".gmshproj", ""))
    store2, project2 = _env()
    project2.open(path)
    assert model_hash(store2.document) == model_hash(store.document)
    reloaded = to_multi_region(store2.document).geom
    assert reloaded.to_dict() == original.to_dict()
