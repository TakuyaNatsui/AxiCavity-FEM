"""バッチ実行（gui/batch.py）: .axiproj / .gmshproj を開く、パラメータを変えると点と拘束が追従する、解析（メッシュ →
solve → post）、結果の読み取り、履歴への登録、別名保存、コマンドライン."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("planegcs")
pytest.importorskip("gmsh")

from axicavity_fem.gui.batch import Project, RunResult, build_parser, main, run_project   # noqa: E402
from axicavity_fem.gui.core.document import Param, create_empty_document                  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_rectangle, point_pos                 # noqa: E402
from axicavity_fem.gui.io.axiproj import scan_results                                     # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
C_LIGHT = 299792458.0
J01 = 2.404825557695773


def pillbox_project(tmp_path: Path, name: str = "pill") -> Project:
    """パラメータ L, a の長方形（ピルボックス）を式付きの点で作ってプロジェクトに保存する."""
    doc = create_empty_document("pill")
    doc.params = [Param(id="p_L", name="L", expression="100"), Param(id="p_a", name="a", expression="50")]
    _lines, points = add_rectangle(doc.sketch, (0, 0), (100, 50))
    by_pos = {tuple(round(v, 6) for v in point_pos(doc.sketch, pid)): pid for pid in points}
    for (x, y), pid in by_pos.items():
        point = next(e for e in doc.sketch.entities if e.id == pid)
        if x > 0:
            point.xExpr = "L"
        if y > 0:
            point.yExpr = "a"
    doc.mesh.size = 8.0
    doc.analysis.numModes = 2
    project = Project.from_document(doc)
    project.save_as(tmp_path / f"{name}.axiproj")
    return project


def test_open_set_params_and_save(tmp_path):
    project = pillbox_project(tmp_path)
    assert project.path == tmp_path / "pill.axiproj" and project.params == {"L": "100", "a": "50"}
    assert project.param_values == {"L": 100.0, "a": 50.0}
    project.set_params(L=110, a="L*0.4")                                  # 上の行（L）を下の行（a）が参照できる
    assert project.params == {"L": "110.0", "a": "L*0.4"} and project.param_values["a"] == 44.0
    ys = sorted({round(point_pos(project.document.sketch, e.id)[1], 6)
                 for e in project.document.sketch.entities if e.type == "point" and not e.projection})
    assert ys == [0.0, 44.0]                                              # 式付きの点が追従
    with pytest.raises(ValueError):
        project.set_params(L="a*2")                                       # 前方参照は不可（上の行から順に評価）
    assert project.params == {"L": "110.0", "a": "L*0.4"}                 # 失敗したら元に戻る
    with pytest.raises(ValueError):
        project.set_params(L=120, a="sqrt(")                              # 2 つ目で失敗 → 1 つ目も戻る
    assert project.params == {"L": "110.0", "a": "L*0.4"} and project.store.expression_errors == []
    with pytest.raises(KeyError):
        project.set_params(L=130, zzz=1)                                  # 名前の誤りは何も変えない
    assert project.params["L"] == "110.0"
    with pytest.raises(ValueError):
        project.add_param("L", 1)                                         # 既にある名前
    with pytest.raises(ValueError):
        project.add_param("u", "nope + 1")                                # 評価できない → 足さない
    assert "u" not in project.params
    with pytest.raises(KeyError):
        project.set_params(zzz=1)
    with pytest.raises(ValueError):
        project.set_params(a="sqrt(")                                     # 式が評価できない
    project.set_params(a=45)
    project.add_param("t", 5)
    assert project.params["t"] == "5.0"
    project.analysis.numModes = 3
    project.set_analysis(type="hom", azOrders="0 1")
    saved = project.save_as(tmp_path / "pill_a45.axiproj")
    again = Project.open(saved)
    assert again.params == {"L": "110.0", "a": "45.0", "t": "5.0"} and again.analysis.type == "hom"
    assert again.analysis.numModes == 3 and again.check() == ([], [])
    original = Project.open(tmp_path / "pill.axiproj")
    assert original.params == {"L": "100", "a": "50"}                     # 元は書き換えない


def test_run_and_scan(tmp_path):
    project = pillbox_project(tmp_path)
    logs = []
    result = project.run(log=logs.append)
    assert isinstance(result, RunResult) and result.kind == "tm0-sw" and not result.registered
    assert result.dir.parent == tmp_path / "pill.axiproj.data" / "batch" and result.dir.name.endswith("-tm0-sw")
    assert (result.dir / "model.msh").exists() and result.file("raw").exists() and result.file("processed").exists()
    f = result.frequencies()
    assert f.shape == (2,) and f[1] > f[0]
    analytic = J01 * C_LIGHT / (2 * np.pi * 0.05) / 1e9                  # TM010 = 2.2949 GHz
    assert f[0] == pytest.approx(analytic, rel=2e-3)
    assert result.values("Q")[0] > 1000 and np.isfinite(result.values("R_over_Q")[0])
    assert result.modes[0]["n"] == 0 and result.modes[0]["phase"] is None and result.n_orders == [0]
    assert any("solve" in line for line in logs)
    log = (tmp_path / "pill.axiproj.data" / "command.log").read_text(encoding="utf-8")
    assert "[batch/" in log and "axicavity-fem solve --type tm0" in log
    job = json.loads((result.dir / "job.json").read_text(encoding="utf-8"))
    assert job["params"] == {"L": "100", "a": "50"}
    # スキャン: 半径を変えると TM010 が 1/a で動く（メッシュは既存の .msh を再利用しない: 形が変わるので作り直す）
    freqs = []
    for a in (40.0, 60.0):
        project.set_params(a=a)
        freqs.append(project.run(post=False).frequencies()[0])
    assert project.analysis.runPost                                       # post=False はその回だけ
    assert freqs[0] > freqs[1] and freqs[0] == pytest.approx(J01 * C_LIGHT / (2 * np.pi * 0.04) / 1e9, rel=3e-3)
    assert len(list((tmp_path / "pill.axiproj.data" / "batch").iterdir())) == 3
    # register: GUI の結果履歴に入る
    registered = project.run(register=True, out_dir=None)
    assert registered.registered and registered.dir.parent.name == "results" and registered.dir.name == "0001-tm0-sw"
    entries = scan_results(tmp_path / "pill.axiproj.data")
    assert len(entries) == 1 and entries[0].status == "done" and entries[0].summary["modes"][0]["f_GHz"] > 0
    # 任意の出力先
    custom = project.run(out_dir=tmp_path / "custom", post=False)
    assert custom.dir == tmp_path / "custom" and custom.file("processed") is None


def test_open_geometry_files_and_run_project(tmp_path):
    project = Project.open(SAMPLES / "cylinder100mm.gmshproj")
    assert project.path is None and project.source.name == "cylinder100mm.gmshproj"
    project.set_analysis(numModes=1)
    result = project.run(post=False)
    assert result.dir.parent == SAMPLES / "cylinder100mm_batch"
    assert result.frequencies().shape == (1,)
    import shutil
    shutil.rmtree(SAMPLES / "cylinder100mm_batch", ignore_errors=True)
    with pytest.raises(ValueError):
        project.run(register=True)
    with pytest.raises(ValueError):
        Project.open(tmp_path / "x.txt")
    # 1 行版
    pillbox_project(tmp_path, "one")
    result = run_project(tmp_path / "one.axiproj", {"a": 45}, numModes=1, runPost=False, out_dir=tmp_path / "one_out")
    assert result.frequencies()[0] == pytest.approx(J01 * C_LIGHT / (2 * np.pi * 0.045) / 1e9, rel=3e-3)


def test_command_line(tmp_path, capsys):
    pillbox_project(tmp_path, "cli")
    project = str(tmp_path / "cli.axiproj")
    assert main([project, "--info"]) == 0
    out = capsys.readouterr().out
    assert "L" in out and "= 100" in out and "tm0 standing" in out
    json_path = tmp_path / "out" / "r.json"
    rc = main([project, "--set", "a=45", "--modes", "1", "--no-post", "--json", str(json_path), "--quiet",
               "--out", str(tmp_path / "cli_out"), "--save-as", str(tmp_path / "cli_a45.axiproj")])
    out = capsys.readouterr().out
    assert rc == 0 and "mode  0: f =" in out and "saved" in out
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["kind"] == "tm0-sw" and data["dir"] == str(tmp_path / "cli_out")
    assert data["summary"]["modes"][0]["f_GHz"] == pytest.approx(J01 * C_LIGHT / (2 * np.pi * 0.045) / 1e9, rel=3e-3)
    assert Project.open(tmp_path / "cli_a45.axiproj").params["a"] == "45"
    assert main([project, "--set", "nope=1"]) == 1 and "error" in capsys.readouterr().err
    assert main([project, "--set", "a=45", "--no-run"]) == 0
    assert main([project, "--type", "hom", "--az-order", "0", "1", "--wave", "traveling", "--phases", "120",
                 "--info"]) == 0
    assert "hom traveling phases 120 n = 0 1" in capsys.readouterr().out
    args = build_parser().parse_args([project, "--lc", "5", "--order", "1", "--cond", "3e7", "--beta", "0.9"])
    assert args.lc == 5.0 and args.order == 1 and args.cond == 3e7


def test_command_line_prints_progress(tmp_path, capfd):
    """-q なしでも動く（解析中は標準出力がログへ付け替えられるので、進捗の print が再帰しないこと）."""
    rc = main([str(SAMPLES / "cylinder100mm.gmshproj"), "--modes", "2", "--no-post", "--out", str(tmp_path / "o")])
    out = capfd.readouterr().out
    assert rc == 0, out
    assert "[done]" in out and "mode  0: f = 2.29485" in out
