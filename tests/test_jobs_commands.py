"""CLI 相当の argv（gui/jobs/commands.py）: ver2.3 の wx GUI と同じコマンド文字列、post / report、入力の検証."""

from axicavity_fem.gui.core.document import create_empty_document
from axicavity_fem.gui.jobs.commands import (
    build_commands,
    command_line,
    post_argv,
    report_argv,
    solve_argv,
    validate_analysis,
)


def test_solve_argv_matches_wx_gui():
    doc = create_empty_document("c")
    assert solve_argv(doc) == ["solve", "--type", "tm0", "-m", "model.msh", "--elem-order", "2",
                               "--num-modes", "10", "-o", "model_SW_TM0.h5"]
    doc.analysis.targetFreqGHz = 2.856
    doc.analysis.numModes = 3
    doc.mesh.order = 1
    assert solve_argv(doc)[6:] == ["1", "--num-modes", "3", "-o", "model_SW_TM0.h5", "--target-freq", "2.856"]
    doc.analysis.type = "hom"
    doc.analysis.azOrders = "0 1 2"
    doc.analysis.wave = "traveling"
    doc.analysis.phases = "0:180:20"
    argv = solve_argv(doc)
    assert argv[:4] == ["solve", "--type", "hom", "-m"] and argv[-8:] == [
        "--target-freq", "2.856", "--az-order", "0", "1", "2", "-p", "0:180:20"]
    assert argv[argv.index("-o") + 1] == "model_TW_HOM.h5"


def test_post_report_and_build_commands():
    doc = create_empty_document("c")
    assert post_argv(doc, "model_SW_TM0.h5") == ["post", "--type", "tm0", "-i", "model_SW_TM0.h5", "-o",
                                                 "model_SW_TM0_processed.h5", "--cond", "5.8e+07", "--beta", "1"]
    doc.analysis.type = "hom"
    doc.post.cond = 3.5e7
    assert post_argv(doc, "x.h5") == ["post", "--type", "hom", "-i", "x.h5", "-o", "x_processed.h5", "--cond", "3.5e+07"]
    assert report_argv(doc, "x_processed.h5") == ["report", "--type", "hom", "-i", "x_processed.h5", "-o",
                                                  "x_processed_report"]
    doc.report.animate = True
    doc.report.timePhase = 45
    doc.report.showMesh = False
    doc.report.dpi = 96
    assert report_argv(doc, "x.h5")[7:] == ["--animate", "--time-phase", "45", "--no-mesh", "--dpi", "96"]

    doc = create_empty_document("c")
    cmds = build_commands(doc)
    assert cmds["kind"] == "tm0-sw" and cmds["raw"] == "model_SW_TM0.h5"
    assert cmds["processed"] == "model_SW_TM0_processed.h5" and cmds["post"][0] == "post"
    doc.analysis.runPost = False
    cmds = build_commands(doc)
    assert cmds["post"] is None and cmds["processed"] is None
    assert command_line(cmds["solve"]).startswith("axicavity-fem solve --type tm0 -m model.msh")
    assert command_line(["report", "-o", "my report"]) == "axicavity-fem report -o 'my report'"


def test_validate_analysis():
    doc = create_empty_document("c")
    assert validate_analysis(doc) == []
    doc.analysis.numModes = 0
    doc.analysis.targetFreqGHz = -1
    doc.analysis.type = "hom"
    doc.analysis.azOrders = "0 x"
    doc.analysis.wave = "traveling"
    doc.analysis.phases = ""
    doc.post.cond = 0
    errors = validate_analysis(doc)
    assert len(errors) == 5
    doc.analysis.type = "tm0"
    doc.analysis.phases = "0"
    doc.post.beta = 2
    errors = validate_analysis(doc)
    assert any("β" in e for e in errors) and any("位相が 0" in e for e in errors)


def test_export_argv():
    from axicavity_fem.gui.jobs.commands import export_argv

    argv = export_argv("tm0", "model_SW_TM0_processed.h5", "exports/f_area_m0", "area", 0,
                       params={"z_range": (0.0, 0.1), "r_range": (0.0, 0.05), "nz": 50, "nr": 30,
                               "scale_to_power": 1000.0, "fmt": "both"})
    assert argv == ["export", "--type", "tm0", "-i", "model_SW_TM0_processed.h5", "-o", "exports/f_area_m0",
                    "--shape", "area", "-m", "0", "--z-range", "0,0.1", "--r-range", "0,0.05", "--nz", "50",
                    "--nr", "30", "--scale-to-power", "1000", "--format", "both"]
    argv = export_argv("hom", "x.h5", "out", "line", 2, n=1, phase=120.0, time_phase=45.0,
                       params={"p1": (0.0, 0.01), "p2": (0.1, 0.04), "npts": 123, "fmt": "txt", "scale": 2.0})
    assert argv[:11] == ["export", "--type", "hom", "-i", "x.h5", "-o", "out", "--shape", "line", "-m", "2"]
    assert argv[11:] == ["--n", "1", "--phase", "120", "--time-phase", "45", "--p1", "0,0.01", "--p2", "0.1,0.04",
                         "--npts", "123", "--scale", "2", "--format", "txt"]
    argv = export_argv("tm0", "x.h5", "out", "axis", 0, phase=120.0, params={"instant": True, "z_range": (0, 0.2)})
    assert argv[10:] == ["0", "--phase", "120", "--z-range", "0,0.2", "--npts", "500", "--instant", "--format", "both"]
    # CLI の parser を通る
    from axicavity_fem.cli.main import build_parser
    args = build_parser().parse_args(argv)
    assert args.shape == "axis" and args.instant and args.z_range == "0,0.2"
