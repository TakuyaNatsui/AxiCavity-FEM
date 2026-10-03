"""USER_MANUAL.md（v3 版）が GUI と食い違っていないこと: リボンの全ボタン名が載っている、画像がある、
CLI の例が今の parser で通る、古いコマンド名が残っていない."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "USER_MANUAL.md"
JA = ROOT / "src" / "axicavity_fem" / "gui" / "i18n" / "ja.json"

# リボンのボタン（ja.json のキー）。マニュアルの本文にこの表示名が出ていること
RIBBON_KEYS = [
    "ribbon.new", "ribbon.open", "ribbon.recent", "ribbon.save", "ribbon.saveAs", "ribbon.undo", "ribbon.redo",
    "ribbon.import", "ribbon.export", "ribbon.importGmshproj", "ribbon.importSuperfish", "ribbon.exportGmshproj",
    "ribbon.exportSuperfish", "ribbon.exportMsh", "ribbon.exportGeo", "ribbon.exportPython", "ribbon.units",
    "ribbon.language", "ribbon.about", "ribbon.quit",
    "ribbon.select", "ribbon.line", "ribbon.rectangle", "ribbon.circle", "ribbon.arc", "ribbon.polygon", "ribbon.point",
    "ribbon.fillet", "ribbon.construction", "ribbon.insertPoint", "ribbon.deleteReconnect", "ribbon.lineToArc",
    "ribbon.arcToLine", "ribbon.closePolyline", "ribbon.splitIntersections", "ribbon.fixAxis", "ribbon.fixZ0",
    "ribbon.fit", "ribbon.dimension", "ribbon.params",
    "physics.bc.PEC", "physics.bc.E-short", "physics.bc.M-short", "physics.bc.None", "physics.bc.auto",
    "physics.editRegion", "physics.validate",
    "mesh.ribbon.generate", "mesh.ribbon.stop", "mesh.ribbon.show", "mesh.ribbon.openFolder",
    "analysis.ribbon.run", "analysis.ribbon.cancel", "analysis.ribbon.forceStop", "analysis.ribbon.post",
    "analysis.ribbon.report", "analysis.ribbon.openReport",
    "results.ribbon.show", "results.ribbon.hColor", "results.ribbon.eLines", "results.ribbon.vectors",
    "results.ribbon.mesh", "results.ribbon.eWall", "results.ribbon.gif", "results.ribbon.exportField",
    "results.ribbon.png", "results.ribbon.openResult", "results.ribbon.openFolder", "results.ribbon.view3d",
    "ribbon.axisRange", "ribbon.grid", "ribbon.pointLabels", "ribbon.bcColors", "ribbon.resetLayout",
]


def _label(tree: dict, key: str) -> str:
    node = tree
    for part in key.split("."):
        node = node[part]
    return node


@pytest.fixture(scope="module")
def manual() -> str:
    return MANUAL.read_text(encoding="utf-8")


def test_manual_names_every_ribbon_button(manual):
    tree = json.loads(JA.read_text(encoding="utf-8"))
    missing = []
    for key in RIBBON_KEYS:
        label = _label(tree, key).rstrip("…").strip()
        if label not in manual:
            missing.append(f"{key} = {label!r}")
    assert missing == [], f"マニュアルに無いボタン名: {missing}"


def test_manual_images_exist(manual):
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", manual)
    assert len(images) >= 5
    for rel in images:
        assert (ROOT / rel).exists(), f"画像が無い: {rel}（tools/manual_screenshots.py で撮る）"


def test_manual_cli_examples_parse(manual):
    from axicavity_fem.cli.main import build_parser

    parser = build_parser()
    lines = [line.strip() for block in re.findall(r"```bash\n(.*?)```", manual, re.S) for line in block.splitlines()]
    commands = [line for line in lines if line.startswith("axicavity-fem ")]
    assert len(commands) >= 8
    for line in commands:
        argv = line.split("#")[0].split()[1:]
        args = parser.parse_args(argv)                  # 引数の名前が今の CLI と合っている
        assert args.command in ("solve", "post", "report", "export", "plot", "info"), line
    from axicavity_fem.gui.batch import build_parser as build_batch_parser
    batch = [line for line in lines if line.startswith("axicavity-fem-run ")]
    assert len(batch) >= 4
    for line in batch:
        build_batch_parser().parse_args(line.split("#")[0].split()[1:])
    assert "axicavity-fem-gui" in manual


def test_manual_has_no_stale_version_names(manual):
    """古いコマンド名でコマンド例を書いていない（pip uninstall の説明で名前を挙げるのは可）."""
    lines = [line.strip() for block in re.findall(r"```bash\n(.*?)```", manual, re.S) for line in block.splitlines()]
    stale = [line for line in lines if re.match(r"(python\s+)?axicavity-fem-v(2\d|3)\b", line)]
    assert stale == [], stale
    for old in ("axicavity-fem-v22-gui", "axicavity-fem-v23-gui", "axicavity-fem-v3", "wxGlade"):
        assert old not in manual, old
    assert "tests/test_known_bugs_v23.py" in manual


def test_manual_python_examples_compile_and_match_the_api(manual):
    """§11（batch）: Python のコード例が構文として正しく、本文に書いた p.… / r.… の名前が実装にある."""
    from axicavity_fem.gui.batch import Project, RunResult

    blocks = re.findall(r"```python\n(.*?)```", manual, re.S)
    assert len(blocks) >= 10
    for i, code in enumerate(blocks):
        compile(code, f"USER_MANUAL.md python block {i}", "exec")
    start = manual.index("## 11. Python からプロジェクトを解析する")
    section = manual[start:manual.index("## 12. v2.3 からの移行")]
    project_attrs = ("path", "source", "warnings", "store", "project_file")
    for name in set(re.findall(r"(?<![A-Za-z_])p\.([A-Za-z_]+)", section)):
        assert hasattr(Project, name) or name in project_attrs, f"p.{name}"
    for name in set(re.findall(r"(?<![A-Za-z_])r\.([A-Za-z_]+)", section)):
        assert hasattr(RunResult, name) or name in RunResult.__dataclass_fields__, f"r.{name}"
    for key in re.findall(r'r\.file\("([a-zA-Z]+)"\)', section):
        assert key in ("raw", "processed", "mesh", "rawTxt", "processedTxt"), key
