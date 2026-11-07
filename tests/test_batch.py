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


def test_reused_out_folder_builds_a_new_mesh(tmp_path):
    """同じ --out で形状を変えると別の周波数になる（3.0.0 は古い model.msh を再利用して同じ値を返した）."""
    project = pillbox_project(tmp_path)
    folder = tmp_path / "same"
    first = project.run(out_dir=folder)
    assert (folder / first.files["processed"]).exists()
    project.set_params(a=40)
    second = project.run(out_dir=folder, post=False)
    assert second.summary["modes"][0]["f_GHz"] > first.summary["modes"][0]["f_GHz"] * 1.2      # TM010 ∝ 1/a
    assert not (folder / first.files["processed"]).exists()                                    # 前の post の結果も残らない


def test_given_mesh_is_still_used(tmp_path):
    project = pillbox_project(tmp_path)
    first = project.run(out_dir=tmp_path / "first", post=False)
    mesh = tmp_path / "first" / "model.msh"
    project.set_params(a=40)
    again = project.run(out_dir=tmp_path / "again", post=False, mesh_file=mesh)
    assert again.summary["modes"][0]["f_GHz"] == pytest.approx(first.summary["modes"][0]["f_GHz"], rel=1e-9)


def test_invalid_input_does_not_remove_the_previous_results(tmp_path):
    """入力の誤りは消す前に止まる."""
    project = pillbox_project(tmp_path)
    folder = tmp_path / "same"
    first = project.run(out_dir=folder)
    project.document.post.cond = -1.0
    with pytest.raises(ValueError, match="導電率"):
        project.run(out_dir=folder)
    assert (folder / first.files["processed"]).exists() and (folder / "model.msh").exists()


def test_text_summaries_of_the_previous_run_are_removed(tmp_path):
    project = pillbox_project(tmp_path)
    folder = tmp_path / "same"
    first = project.run(out_dir=folder)
    processed_txt = (folder / first.files["processed"]).with_suffix(".txt")
    assert processed_txt.exists()
    project.run(out_dir=folder, post=False)
    assert not processed_txt.exists()


def test_given_mesh_inside_the_out_folder_is_used_in_place(tmp_path):
    project = pillbox_project(tmp_path)
    folder = tmp_path / "same"
    first = project.run(out_dir=folder, post=False)
    project.set_params(a=40)
    again = project.run(out_dir=folder, post=False, mesh_file=folder / "model.msh")
    assert again.summary["modes"][0]["f_GHz"] == pytest.approx(first.summary["modes"][0]["f_GHz"], rel=1e-9)


def _dielectric_project(tmp_path):
    from axicavity_fem.gui.core.convert import sketch_profiles
    from axicavity_fem.gui.core.document import RegionSetting

    project = pillbox_project(tmp_path, "diel")
    prof = sketch_profiles(project.document)[0]
    project.document.regions = [RegionSetting(key=prof.id, anchor=prof.anchor, name="D", materialTag="d", epsR=2.0)]
    return project, project.document.regions[0]


def test_given_mesh_uses_the_current_materials(tmp_path):
    """ε_r を変えて同じメッシュで解くと周波数が変わる（3.0.0 はメッシュの隣の古い材料で解いた）."""
    project, region = _dielectric_project(tmp_path)
    first = project.run(out_dir=tmp_path / "first", post=False)
    region.epsR = 2.2                                              # 自動のシフト（1.50 GHz）より下にならない範囲
    again = project.run(out_dir=tmp_path / "again", post=False, mesh_file=tmp_path / "first" / "model.msh")
    assert again.summary["modes"][0]["f_GHz"] == pytest.approx(first.summary["modes"][0]["f_GHz"] * (2.0 / 2.2) ** 0.5, rel=1e-6)


def test_negative_phase_runs_and_a_bad_argument_is_recorded_as_an_error(tmp_path):
    """先頭が負の位相（-60,90）を argparse に通す. 引数の誤りで止まっても結果が running のまま残らない."""
    project = pillbox_project(tmp_path)
    project.document.analysis.wave = "traveling"
    project.document.analysis.phases = "-60,90"
    result = project.run(register=True, post=False)
    assert result.summary["modes"]
    from axicavity_fem.gui.jobs import analysis_pipeline
    with pytest.raises(RuntimeError, match="引数"):
        analysis_pipeline._run_cli(["solve", "--no-such-option"], lambda _m: None)


def test_unwritable_json_gives_an_error_line(tmp_path, capfd):
    rc = main([str(SAMPLES / "cylinder100mm.gmshproj"), "--modes", "1", "--no-post", "--out", str(tmp_path / "o"), "--lc", "12",
               "--json", str(tmp_path)])
    assert rc == 1
    assert "--json" in capfd.readouterr().err


def test_solver_warning_is_kept_when_post_fails_and_with_register(tmp_path, monkeypatch):
    """警告は解いた直後に残す: post が失敗しても、register の結果を GUI で開いても見える."""
    from axicavity_fem.cli import cmd_post
    from axicavity_fem.gui.core.convert import sketch_profiles
    from axicavity_fem.gui.core.document import RegionSetting

    project = pillbox_project(tmp_path, "warn")
    prof = sketch_profiles(project.document)[0]
    project.document.regions = [RegionSetting(key=prof.id, anchor=prof.anchor, name="D", materialTag="d", epsR=4.0)]

    def fail(args):
        raise RuntimeError("post failed")

    monkeypatch.setattr(cmd_post, "run", fail)
    with pytest.raises(RuntimeError):
        project.run(register=True)
    assert any("見逃して" in w for w in scan_results(project.data_dir)[0].warnings)
    monkeypatch.undo()
    project.run(register=True, post=False)
    assert any("見逃して" in w for w in scan_results(project.data_dir)[0].warnings)


def test_mistyped_mesh_file_does_not_touch_the_folder(tmp_path):
    project = pillbox_project(tmp_path)
    folder = tmp_path / "same"
    first = project.run(out_dir=folder)
    before = sorted(p.name for p in folder.iterdir())
    with pytest.raises(ValueError, match="mesh_file"):
        project.run(out_dir=folder, mesh_file=tmp_path / "nope.msh")
    broken = tmp_path / "broken.msh"
    broken.write_text("not a mesh", encoding="utf-8")
    with pytest.raises(ValueError, match="mesh_file"):
        project.run(out_dir=folder, mesh_file=broken)
    assert sorted(p.name for p in folder.iterdir()) == before
    assert (folder / first.files["processed"]).exists()


def test_user_files_in_a_folder_that_was_not_a_run_are_kept(tmp_path):
    project = pillbox_project(tmp_path)
    folder = tmp_path / "mine"
    folder.mkdir()
    (folder / "model.msh").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="model.msh"):
        project.run(out_dir=folder)
    assert (folder / "model.msh").read_text(encoding="utf-8") == "mine"


def _foreign_mesh(tmp_path):
    """材料タグ e（ε_r = 2）の外部メッシュと隣の材料ファイル（samples/diel_simple1 と同じ作り）."""
    from axicavity_fem.gui.jobs.pipeline import write_materials_json
    from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
    from axicavity_fem.shared.multi_region_model import MultiRegionGeometry

    geom = MultiRegionGeometry.load_json(SAMPLES / "diel_simple1.gmshproj")
    folder = tmp_path / "ext"
    folder.mkdir()
    msh = folder / "cav.msh"
    export_msh_multi_region(geom, msh, mesh_order=2, verbose=0)
    return msh, write_materials_json(geom, msh)


def test_mesh_with_foreign_tags_uses_its_own_materials(tmp_path):
    """文書に無いタグのメッシュは隣の材料で解く（真空で解いて +19% になった）."""
    msh, sidecar = _foreign_mesh(tmp_path)
    project = pillbox_project(tmp_path)
    with_tags = project.run(out_dir=tmp_path / "out", post=False, mesh_file=msh)
    direct = Project.from_document(project.document)
    assert with_tags.summary["modes"][0]["f_GHz"] != pytest.approx(direct.run(out_dir=tmp_path / "doc", post=False).summary["modes"][0]["f_GHz"], rel=1e-3)
    assert with_tags.summary["modes"][0]["f_GHz"] == pytest.approx(2.409, rel=5e-3)
    sidecar.unlink()
    with pytest.raises(ValueError, match="材料タグ"):
        project.run(out_dir=tmp_path / "out2", post=False, mesh_file=msh)


def test_check_includes_the_analysis_settings(tmp_path):
    project = pillbox_project(tmp_path)
    project.document.post.cond = -1.0
    errors, _ = project.check()
    assert any("導電率" in e for e in errors)


def test_mesh_of_another_order_is_refused_before_touching_the_folder(tmp_path):
    project = pillbox_project(tmp_path)
    folder = tmp_path / "same"
    first = project.run(out_dir=folder)
    project.document.mesh.order = 1
    with pytest.raises(ValueError, match="三角形要素"):
        project.run(out_dir=folder, mesh_file=folder / "model.msh")
    assert (folder / first.files["processed"]).exists()


def test_user_outputs_are_kept_also_with_a_mesh_file(tmp_path):
    project = pillbox_project(tmp_path)
    source = project.run(out_dir=tmp_path / "source", post=False)
    folder = tmp_path / "mine"
    folder.mkdir()
    (folder / "model_SW_TM0_processed.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="前の実行のものではない"):
        project.run(out_dir=folder, post=False, mesh_file=source.dir / "model.msh")
    assert (folder / "model_SW_TM0_processed.txt").read_text(encoding="utf-8") == "mine"


@pytest.mark.parametrize("field, value", [("cond", float("nan")), ("cond", float("inf"))])
def test_not_a_number_is_refused(tmp_path, field, value):
    project = pillbox_project(tmp_path)
    setattr(project.document.post, field, value)
    assert any("導電率" in e for e in project.check()[0])


def test_command_line_prints_warnings_also_when_quiet(tmp_path, capfd):
    rc = main([str(SAMPLES / "diel_simple2.gmshproj"), "--modes", "1", "--no-post", "--out", str(tmp_path / "o"), "-q"])
    assert rc == 0
    assert "見逃して" in capfd.readouterr().err


def test_mesh_with_a_region_named_after_a_boundary_is_accepted(tmp_path):
    """2D の Physical が PEC のメッシュは金属の領域で、文書に無い材料タグではない."""
    from test_hom_periodic_wall import _metal_region_mesh

    msh = _metal_region_mesh(tmp_path / "metal.msh", 2)
    project = pillbox_project(tmp_path)
    assert project.run(out_dir=tmp_path / "out", post=False, mesh_file=msh).summary["modes"]


def test_mesh_without_regions_is_refused_for_a_dielectric_document(tmp_path):
    """2D の Physical が無いメッシュでは材料を当てはめられない（黙って真空で解いて 2 倍の周波数になった）."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("m")
        geo = gmsh.model.geo
        p = [geo.addPoint(z, r, 0.0, 5e-3) for z, r in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.05), (0.0, 0.05))]
        lines = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, lines[1:]), "PEC")
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        msh = tmp_path / "plain.msh"
        gmsh.write(str(msh))
    finally:
        gmsh.finalize()
    project, _region = _dielectric_project(tmp_path)
    with pytest.raises(ValueError, match="領域の情報"):
        project.run(out_dir=tmp_path / "out", post=False, mesh_file=msh)


def test_command_line_prints_only_solver_warnings(tmp_path, capfd, monkeypatch):
    """ライブラリの警告（ResourceWarning など）は解析の結果とは関係がないので標準エラーに出さない."""
    def fake_run(self, **kwargs):
        return RunResult(dir=tmp_path, kind="tm0-sw", summary={}, files={}, elapsed={},
                         warnings=["unclosed file <x>", "[TM0] 低いモード", "[TM0] 低いモード"], registered=False)

    monkeypatch.setattr(Project, "run", fake_run)
    assert main([str(SAMPLES / "cylinder100mm.gmshproj"), "--no-post", "--out", str(tmp_path / "o"), "-q"]) == 0
    err = capfd.readouterr().err
    assert err.count("[TM0] 低いモード") == 1 and "unclosed" not in err
