"""解析ジョブ（gui/jobs/analysis_pipeline.py）: メッシュ生成 → solve → post → 要約、post のみ、レポート、
ver2.3 のサマリ（samples/*.txt）との周波数の一致、進行波・HOM."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("gmsh")

from axicavity_fem.gui.core.convert import to_multi_region   # noqa: E402
from axicavity_fem.gui.core.legacy import load_gmshproj      # noqa: E402
from axicavity_fem.gui.jobs import analysis_pipeline, pipeline  # noqa: E402
from axicavity_fem.gui.jobs.commands import build_commands, post_argv, report_argv  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _job_dir(tmp_path, name, doc):
    folder = tmp_path / name
    folder.mkdir()
    to_multi_region(doc).geom.save_json(folder / pipeline.GEOMETRY_FILE)
    cmds = build_commands(doc)
    job = {"kind": cmds["kind"], "meshOrder": int(doc.mesh.order), "argv": {"solve": cmds["solve"], "post": cmds["post"]},
           "files": {"mesh": "model.msh", "raw": cmds["raw"], "processed": cmds["processed"]}}
    (folder / pipeline.JOB_FILE).write_text(json.dumps(job), encoding="utf-8")
    return folder


def _ver23_frequencies(txt_path: Path) -> list[float]:
    text = txt_path.read_text(encoding="utf-8")
    return [float(m) for m in re.findall(r"f\s*=\s*([0-9.]+) GHz", text)]


def test_tm0_standing_with_post_matches_ver23(tmp_path):
    doc, _ = load_gmshproj(SAMPLES / "cylinder100mm.gmshproj")
    doc.analysis.numModes = 3
    folder = _job_dir(tmp_path, "tm0", doc)
    stages, logs = [], []
    result = analysis_pipeline.run_job("analysis", folder, progress=lambda s, _m: stages.append(s), log=logs.append)
    assert stages == ["meshing", "solve", "post", "summary", "done"]
    assert (folder / "model.msh").exists() and (folder / "model.materials.json").exists()
    files = result["files"]
    assert files["raw"] == "model_SW_TM0.h5" and files["processed"] == "model_SW_TM0_processed.h5"
    assert (folder / files["processed"]).exists() and (folder / "model_SW_TM0_processed.txt").exists()
    assert files["processedTxt"] == "model_SW_TM0_processed.txt"
    summary = result["summary"]
    assert summary["solverType"] == "tm0" and summary["wave"] == "standing" and summary["hasPost"]
    modes = summary["modes"]
    assert len(modes) == 3 and [m["index"] for m in modes] == [0, 1, 2]
    assert all(m["n"] == 0 and m["phase"] is None for m in modes)
    assert modes[0]["Q"] > 1000 and modes[0]["R_over_Q"] > 0 and modes[0]["U_stored"] > 0
    assert any("$ axicavity-fem solve" in line for line in logs)
    reference = SAMPLES / "cylinder100mm_SW_TM0_processed.txt"
    if reference.exists():
        expected = _ver23_frequencies(reference)
        assert modes[0]["f_GHz"] == pytest.approx(expected[0], rel=1e-4)

    # post だけやり直し（導電率を変える）と、レポート
    doc.post.cond = 2.0e7
    job = json.loads((folder / pipeline.JOB_FILE).read_text(encoding="utf-8"))
    job["argv"]["post"] = post_argv(doc, files["raw"])
    job["argv"]["report"] = report_argv(doc, files["processed"])
    (folder / pipeline.JOB_FILE).write_text(json.dumps(job), encoding="utf-8")
    again = analysis_pipeline.run_job("post", folder)
    q_lower = again["summary"]["modes"][0]["Q"]
    assert q_lower < modes[0]["Q"]                                         # 導電率が下がると Q も下がる
    report = analysis_pipeline.run_job("report", folder)
    assert (folder / report["files"]["report"]).exists()                    # model_SW_TM0_processed_report/index.html
    assert report["summary"]["modes"][0]["Q"] == pytest.approx(q_lower)


def test_traveling_and_hom_without_post(tmp_path):
    doc, _ = load_gmshproj(SAMPLES / "cylinder100mm.gmshproj")
    doc.analysis.numModes = 2
    doc.analysis.wave = "traveling"
    doc.analysis.phases = "120"
    doc.analysis.runPost = False
    folder = _job_dir(tmp_path, "tw", doc)
    result = analysis_pipeline.run_job("analysis", folder)
    summary = result["summary"]
    assert summary["wave"] == "traveling" and summary["phases"] == [120.0] and not summary["hasPost"]
    assert [m["phase"] for m in summary["modes"]] == [120.0, 120.0] and "Q" not in summary["modes"][0]
    assert result["files"]["processed"] is None and (folder / "model_TW_TM0.h5").exists()

    doc.analysis.type = "hom"
    doc.analysis.azOrders = "1"
    doc.analysis.wave = "standing"
    doc.analysis.runPost = True
    doc.mesh.order = 1
    folder = _job_dir(tmp_path, "hom", doc)
    result = analysis_pipeline.run_job("analysis", folder)
    summary = result["summary"]
    assert summary["solverType"] == "hom" and summary["nOrders"] == [1]
    assert len(summary["modes"]) == 2 and summary["modes"][0]["n"] == 1 and summary["modes"][0]["Q"] > 0
    assert (folder / "model_SW_HOM_processed.h5").exists()


def test_cancel_and_errors(tmp_path):
    doc, _ = load_gmshproj(SAMPLES / "cylinder100mm.gmshproj")
    folder = _job_dir(tmp_path, "cancel", doc)
    with pytest.raises(pipeline.JobCancelled):
        analysis_pipeline.run_job("analysis", folder, should_cancel=lambda: True)
    assert not (folder / "model.msh").exists()
    bad = _job_dir(tmp_path, "bad", doc)
    job = json.loads((bad / pipeline.JOB_FILE).read_text(encoding="utf-8"))
    job["argv"]["solve"][job["argv"]["solve"].index("-m") + 1] = "missing.msh"
    (bad / pipeline.JOB_FILE).write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(RuntimeError, match="solve"):
        analysis_pipeline.run_job("analysis", bad)
    with pytest.raises(ValueError):
        analysis_pipeline.run_job("nope", bad)


def test_export_field_job(tmp_path):
    """場の書き出し（kind = export）: ver2.3 のサンプル結果を入力に area / axis を h5 + txt で書く."""
    from axicavity_fem.gui.jobs.commands import export_argv

    source = SAMPLES / "diel_simple1_SW_TM0_processed.h5"
    if not source.exists():
        pytest.skip("サンプル結果なし")
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "model_SW_TM0_processed.h5").write_bytes(source.read_bytes())
    argv = export_argv("tm0", "model_SW_TM0_processed.h5", "exports/field_area_m0", "area", 0,
                       params={"z_range": (0.0, 0.03), "r_range": (0.0, 0.02), "nz": 12, "nr": 6, "fmt": "both",
                               "scale_to_power": 1000.0})
    (folder / pipeline.JOB_FILE).write_text(json.dumps({"kind": "export", "argv": {"export": argv}}), encoding="utf-8")
    (folder / "exports").mkdir()
    stages = []
    result = analysis_pipeline.run_job("export", folder, progress=lambda s, _m: stages.append(s))
    assert stages == ["export", "done"] and result["files"]["export"] == "exports/field_area_m0"
    assert (folder / "exports" / "field_area_m0.h5").exists() and (folder / "exports" / "field_area_m0.txt").exists()
    assert "summary" in result and result["summary"] == {}
    # 絶対パス（外部の h5）でも動く
    out = tmp_path / "axis_out"
    argv = export_argv("tm0", str(source), str(out), "axis", 1, params={"npts": 20, "fmt": "txt"})
    (folder / pipeline.JOB_FILE).write_text(json.dumps({"kind": "export", "argv": {"export": argv}}), encoding="utf-8")
    analysis_pipeline.run_job("export", folder)
    assert out.with_suffix(".txt").exists() and not out.with_suffix(".h5").exists()
