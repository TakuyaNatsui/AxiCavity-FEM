"""AnalysisController（gui/jobs/analysis_controller.py）: 検証、投入（自動保存 → 結果フォルダ → 子プロセス）、
結果の登録（result.json の summary）、メッシュの再利用、post のやり直し、レポート、command.log、キャンセル."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("gmsh")
pytest.importorskip("pytestqt")
from PySide6 import QtWidgets                                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.jobs.analysis_controller import AnalysisController  # noqa: E402
from axicavity_fem.gui.jobs.mesh_controller import MeshController  # noqa: E402
from axicavity_fem.gui.jobs.session import JobSession           # noqa: E402
from axicavity_fem.gui.project.controller import ProjectController  # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402
from test_mesh_controller import draw_box                      # noqa: E402


@pytest.fixture
def env(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)
    project = ProjectController(store, generator="test")
    project.new()
    session = JobSession()
    mesh = MeshController(store, session)
    analysis = AnalysisController(store, session, project, mesh)
    yield store, controller, project, session, mesh, analysis
    session.shutdown()


def test_run_post_report_and_cancel(env, tmp_path, qtbot):
    store, controller, project, session, mesh, analysis = env
    assert analysis.check()[0]                                     # 閉領域が無い
    draw_box(store, controller, 0, 0, 100, 50)
    store.set_mesh(size=10.0)
    store.set_analysis(numModes=2)
    assert analysis.check() == ([], [])
    with pytest.raises(ValueError):
        analysis.run()                                             # 未保存
    project.save_as(tmp_path / "run.axiproj")

    # メッシュタブのメッシュを再利用して実行
    assert mesh.generate(project.prepare_mesh())
    qtbot.waitUntil(lambda: not mesh.running, timeout=180_000)
    assert mesh.state() == "current"
    changes = []
    analysis.changed.connect(lambda: changes.append(analysis.running))
    assert analysis.run(), analysis.last_error
    assert analysis.running and project.running_dir is not None
    rid = analysis.job.id
    assert rid == "0001-tm0-sw" and project.results[0].status == "running"
    assert not analysis.run()                                      # 実行中は二重投入しない
    qtbot.waitUntil(lambda: not analysis.running, timeout=300_000)
    assert analysis.job.status == "done", (analysis.job.error, analysis.job.traceback)
    qtbot.waitUntil(lambda: project.results[0].status == "done", timeout=10_000)
    entry = project.results[0]
    assert entry.id == rid and project.running_dir is None and project.active_result_id == rid
    assert entry.summary["modes"][0]["Q"] > 0 and len(entry.summary["modes"]) == 2
    assert entry.file("raw").exists() and entry.file("processed").exists()
    job = json.loads((entry.dir / "job.json").read_text(encoding="utf-8"))
    assert job["meshReused"] and (entry.dir / "model.msh").exists()
    assert any("メッシュを再利用" in line for line in analysis.job.log)
    log = (project.data_dir / "command.log").read_text(encoding="utf-8")
    assert f"[{rid}] axicavity-fem solve --type tm0 -m model.msh" in log and "axicavity-fem post" in log
    assert project.result_relation(entry) == "" and not project.is_dirty()
    assert analysis.report_path(entry) is None

    # post のやり直し（導電率を変える）: 結果の要約が更新される
    q_before = entry.summary["modes"][0]["Q"]
    store.set_post(cond=2.0e7)
    assert analysis.run_post(entry) and project.running_dir == entry.dir
    qtbot.waitUntil(lambda: not analysis.running, timeout=120_000)
    assert analysis.job.status == "done", analysis.job.error
    qtbot.waitUntil(lambda: project.results[0].summary["modes"][0]["Q"] < q_before, timeout=10_000)
    assert project.running_dir is None and project.results[0].status == "done"

    # レポート
    entry = project.results[0]
    assert analysis.run_report(entry)
    qtbot.waitUntil(lambda: not analysis.running, timeout=300_000)
    assert analysis.job.status == "done", analysis.job.error
    qtbot.waitUntil(lambda: analysis.report_path(project.results[0]) is not None, timeout=10_000)
    assert analysis.report_path(project.results[0]).name == "index.html"

    # 強制停止: 実行中のジョブを kill → cancelled として残る
    store.set_analysis(numModes=4)
    assert analysis.run()
    qtbot.waitUntil(lambda: session._process is not None, timeout=30_000)
    assert analysis.cancel(force=True)
    qtbot.waitUntil(lambda: not analysis.running, timeout=60_000)
    assert analysis.job.status == "cancelled"
    qtbot.waitUntil(lambda: project.results[0].status == "cancelled", timeout=10_000)
    assert project.results[0].id == "0002-tm0-sw" and project.running_dir is None
    session.wait()
