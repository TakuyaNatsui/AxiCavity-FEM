"""メッシュジョブ（gui/jobs/pipeline.py と runner.py）: 生成物・統計・プレビュー配列、ユーザー指定パスへの書き出し、
子プロセスのイベント（done / cancelled / error）と fd 1 の付け替え（gmsh の出力がイベントに混ざらない）.
"""

import json
import subprocess

import numpy as np
import pytest

pytest.importorskip("gmsh")

from axicavity_fem.gui.core.convert import sketch_profiles, to_multi_region   # noqa: E402
from axicavity_fem.gui.core.document import RegionSetting, create_empty_document  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_rectangle             # noqa: E402
from axicavity_fem.gui.jobs import pipeline, runner                       # noqa: E402


def _document(with_dielectric=False, order=2):
    doc = create_empty_document("box")
    add_rectangle(doc.sketch, (0, 0), (100, 50))
    if with_dielectric:
        add_rectangle(doc.sketch, (100, 0), (150, 50), tolerance=1e-6)
        right = max(sketch_profiles(doc), key=lambda p: p.anchor[0])
        doc.regions.append(RegionSetting(key=right.id, anchor=right.anchor, name="Ceramic",
                                         materialTag="ceramic", epsR=9.0, tanDelta=1e-4))
    doc.mesh.size = 10.0
    doc.mesh.order = order
    return doc


def _job_dir(tmp_path, name="job", order=2, out_path=None, with_dielectric=False, geom=None):
    folder = tmp_path / name
    folder.mkdir()
    geom = geom or to_multi_region(_document(with_dielectric, order)).geom
    geom.save_json(folder / pipeline.GEOMETRY_FILE)
    job = {"kind": "mesh", "meshOrder": order}
    if out_path is not None:
        job["outPath"] = str(out_path)
    (folder / pipeline.JOB_FILE).write_text(json.dumps(job), encoding="utf-8")
    return folder


def _run(cmd, cwd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=runner.child_environment(), cwd=str(cwd), timeout=300)


def _events(result):
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    events = [runner.parse_event(line) for line in lines]
    assert all(e is not None for e in events), result.stdout      # gmsh の出力が混ざっていない
    return events


def test_run_mesh_job_in_process(tmp_path):
    folder = _job_dir(tmp_path, with_dielectric=True)
    stages, logs = [], []
    summary = pipeline.run_mesh_job(folder, progress=lambda s, _m: stages.append(s), log=logs.append)
    assert stages[0] == "meshing" and stages[-1] == "done"
    stats = summary["stats"]
    assert summary["kind"] == "mesh" and summary["meshOrder"] == 2 and summary["units"] == "mm"
    assert stats["elements"] > 20 and stats["nodes"] > stats["elements"]      # 2 次: 節点 > 要素
    assert stats["order"] == 2 and 0 < stats["meanEdge"] < 20
    assert stats["bc"]["PEC"] > 0 and stats["bc"]["None"] > 0 and stats["bc"]["E-short"] == 0
    assert stats["interfaces"] > 0                                            # 真空 / 誘電体の界面
    assert {r["name"] for r in stats["regions"]} == {"Vacuum", "Ceramic"}
    assert sum(r["elements"] for r in stats["regions"]) == stats["elements"]
    assert stats["bounds"][0] == pytest.approx([0, 0], abs=1e-6) and stats["bounds"][1] == pytest.approx([150, 50], abs=1e-6)
    assert (folder / "model.msh").exists() and (folder / pipeline.MESH_SUMMARY_FILE).exists()
    materials = json.loads((folder / pipeline.MATERIALS_FILE).read_text(encoding="utf-8"))
    assert materials["schema"] == "axicavity-fem-v21.materials/1" and materials["mesh_file"] == "model.msh"
    assert materials["materials"]["ceramic"] == {"eps_r": 9.0, "mu_r": 1.0, "tan_delta": 1e-4}
    assert materials["materials"]["vacuum"]["eps_r"] == 1.0
    with np.load(folder / pipeline.MESH_PREVIEW_FILE) as npz:
        assert set(npz.files) >= {"nodes", "simplices", "elem_order", "region_ids", "bc_PEC", "bc_None",
                                  "bc_E-short", "bc_M-short", "interfaces"}
        assert npz["simplices"].shape == (stats["elements"], 3) and npz["nodes"].shape[1] == 2
        assert npz["bc_PEC"].shape[1:] == (3, 2) and npz["bc_E-short"].shape == (0, 3, 2)   # 2 次は中点経由
        assert npz["nodes"].max(axis=0) == pytest.approx([0.15, 0.05], abs=1e-9)             # m 単位
        assert int(npz["elem_order"]) == 2 and set(npz["region_ids"].tolist()) >= {1, 2}
    saved = json.loads((folder / pipeline.MESH_SUMMARY_FILE).read_text(encoding="utf-8"))
    assert saved["stats"]["elements"] == stats["elements"]
    assert any("メッシュ生成完了" in line for line in logs)


def test_export_msh_to_user_path(tmp_path):
    out = tmp_path / "out" / "cavity.msh"
    folder = _job_dir(tmp_path, "exp", order=1, out_path=out)
    summary = pipeline.run_mesh_job(folder)
    assert out.exists() and (tmp_path / "out" / "cavity.materials.json").exists()
    assert summary["files"]["mesh"] == str(out) and summary["stats"]["order"] == 1
    assert not (folder / "model.msh").exists()
    with np.load(folder / pipeline.MESH_PREVIEW_FILE) as npz:
        assert npz["bc_PEC"].shape[1:] == (2, 2)                              # 1 次は両端だけ


def test_runner_cli_events_cancel_and_error(tmp_path):
    folder = _job_dir(tmp_path, "cli")
    result = _run(runner.job_command("mesh", folder), folder)
    assert result.returncode == 0, result.stderr
    events = _events(result)
    assert events[-1]["event"] == "done" and events[-1]["result"]["stats"]["elements"] > 0
    assert [e["stage"] for e in events if e["event"] == "progress"][0] == "meshing"
    assert any(e["event"] == "log" for e in events)
    assert (folder / "log.txt").exists()                                      # fd 1 の付け替え先

    cancel = _job_dir(tmp_path, "cancel")
    runner.request_cancel(cancel)
    result = _run(runner.job_command("mesh", cancel), cancel)
    assert result.returncode == 0 and _events(result)[-1]["event"] == "cancelled"
    assert not (cancel / "model.msh").exists()

    geom = to_multi_region(_document()).geom
    geom.regions = []
    bad = _job_dir(tmp_path, "bad", geom=geom)
    result = _run(runner.job_command("mesh", bad), bad)
    events = _events(result)
    assert result.returncode == 1 and events[-1]["event"] == "error"
    assert "Region" in events[-1]["error"] and "Traceback" in events[-1]["traceback"]

    result = _run(runner.job_command("plot", folder), folder)               # 未実装の種類（PNG は GUI が保存）
    assert result.returncode == 1 and "NotImplementedError" in _events(result)[-1]["error"]
    assert runner.parse_event("Info    : Meshing 1D...") is None
