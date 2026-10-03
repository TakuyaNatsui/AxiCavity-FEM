"""JobSession（子プロセス、QProcess）: 実メッシュジョブの完了、同時投入の拒否、強制停止、前の子の遅れた終了.

移植元: EM-CAD-py tests/test_em_session.py。メッシュジョブは gmsh の子プロセス（数秒）。
"""

import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("gmsh")
pytest.importorskip("pytestqt")
from PySide6 import QtCore, QtWidgets                          # noqa: E402

from axicavity_fem.gui.core.convert import to_multi_region      # noqa: E402
from axicavity_fem.gui.core.document import create_empty_document  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_rectangle   # noqa: E402
from axicavity_fem.gui.jobs import runner                       # noqa: E402
from axicavity_fem.gui.jobs.session import JobSession, JobState  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mesh_job_dir(tmp_path, name="mesh"):
    doc = create_empty_document("box")
    add_rectangle(doc.sketch, (0, 0), (100, 50))
    doc.mesh.size = 10.0
    folder = tmp_path / name
    folder.mkdir()
    to_multi_region(doc).geom.save_json(folder / "geometry.json")
    (folder / "job.json").write_text(json.dumps({"kind": "mesh", "meshOrder": 2}), encoding="utf-8")
    return folder


def _fake_job(tmp_path, name: str, code: str) -> JobState:
    folder = tmp_path / name
    folder.mkdir()
    return JobState(id=name, dir=folder, kind="mesh", command=[sys.executable, "-c", code])


def test_session_runs_mesh_job(qtbot, app, tmp_path):
    session = JobSession()
    changes: list[str] = []
    logs: list[str] = []
    finished: list[object] = []
    session.job_changed.connect(lambda j: changes.append(j.status))
    session.log_appended.connect(lambda _j, line: logs.append(line))
    session.job_finished.connect(finished.append)
    folder = _mesh_job_dir(tmp_path)
    job = session.submit("mesh", folder)
    assert session.running and job.status == "running" and job.id == "mesh"
    with pytest.raises(RuntimeError):
        session.submit("mesh", folder)                       # 同時に 1 本だけ
    qtbot.waitUntil(lambda: job.finished, timeout=180_000)
    assert job.status == "done", (job.error, job.traceback)
    assert finished == [job] and not session.running
    assert job.result["kind"] == "mesh" and job.result["stats"]["elements"] > 0
    assert "meshing" in changes and changes[-1] == "done"
    assert any("メッシュ生成完了" in line for line in logs)
    assert (folder / "model.msh").exists() and (folder / "mesh_preview.npz").exists()
    assert (folder / "log.txt").exists()
    session.wait()
    with pytest.raises(ValueError):
        session.submit("mesh", tmp_path / "missing")
    with pytest.raises(ValueError):
        session.submit("nope", folder)


def test_force_stop_and_late_exit_of_previous_child(qtbot, app, tmp_path):
    session = JobSession()
    done = json.dumps({"event": "done", "result": {"ok": True}})

    # 強制停止: 眠っている子を kill → cancelled（結果なし）
    sleeper = _fake_job(tmp_path, "sleeper", "import time; time.sleep(30)")
    session._start(sleeper)
    qtbot.waitUntil(lambda: session._process is not None
                    and session._process.state() == QtCore.QProcess.Running, timeout=30_000)
    assert session.cancel(force=True)
    qtbot.waitUntil(lambda: sleeper.finished, timeout=30_000)
    assert sleeper.status == "cancelled" and not session.running
    assert not session.cancel()

    # 終端イベント（done）の後、子がまだ終わらないうちに次のジョブを始めても取り違えない
    first = _fake_job(tmp_path, "first", f"import time; print({done!r}, flush=True); time.sleep(3)")
    session._start(first)
    qtbot.waitUntil(lambda: first.finished, timeout=30_000)
    assert first.status == "done" and not session.running          # 子はまだ 3 s 生きている
    second = _fake_job(tmp_path, "second", f"import time; time.sleep(4); print({done!r}, flush=True)")
    session._start(second)
    qtbot.waitUntil(lambda: second.finished, timeout=30_000)
    assert second.status == "done", (second.status, second.error)
    assert second.result == {"ok": True}
    assert session.wait(10_000)

    # 結果を返さずに終わる子 → error（stderr の末尾が詳細）
    crash = _fake_job(tmp_path, "crash", "import sys; print('boom', file=sys.stderr); sys.exit(3)")
    session._start(crash)
    qtbot.waitUntil(lambda: crash.finished, timeout=30_000)
    assert crash.status == "error" and "3" in crash.error and "boom" in (crash.traceback or "")
    # cancel.flag で自分から止まる子 → cancelled
    cancelled = json.dumps({"event": "cancelled"})
    quitter = _fake_job(tmp_path, "quitter", f"print({cancelled!r}, flush=True)")
    session._start(quitter)
    qtbot.waitUntil(lambda: quitter.finished, timeout=30_000)
    assert quitter.status == "cancelled"
    session.shutdown()
    assert runner.parse_event("not json") is None and runner.parse_event(done)["event"] == "done"
