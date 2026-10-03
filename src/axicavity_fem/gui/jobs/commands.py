"""解析設定 → CLI 相当の argv（純 Python）.

ver2.3 の wx GUI（``main_frame.OnRunSolver`` / ``OnRunPost`` / ``OnBtnCreateReport``）が組み立てていたコマンドと
**同じ文字列**を作る。子プロセスはこの argv を ``cli.main.build_parser().parse_args`` に通して ``cmd_solve.run`` などを
呼ぶので、CLI と同じ経路・引数検証になり、``command.log`` に「そのまま再実行できる」コマンドを残せる。
パスはジョブフォルダ（結果フォルダ）からの相対（子はそこを cwd にする）。
"""

from __future__ import annotations

import shlex
from typing import Optional

from .. import frozen
from ..core.document import AxiDocument
from ..io.axiproj import raw_file_name, result_kind

# command.log などに書く「そのまま実行できる」コマンドの名前（Windows 版では EXE が同じサブコマンドを持つ）
PROGRAM = frozen.EXE_NAME if frozen.is_compiled() else "axicavity-fem"
MESH_FILE = "model.msh"


def processed_file_name(raw: str) -> str:
    return raw[:-3] + "_processed.h5" if raw.endswith(".h5") else raw + "_processed.h5"


def report_dir_name(h5: str) -> str:
    return (h5[:-3] if h5.endswith(".h5") else h5) + "_report"


def solve_argv(doc: AxiDocument, mesh_file: str = MESH_FILE, output: Optional[str] = None) -> list[str]:
    a = doc.analysis
    output = output or raw_file_name(result_kind(a))
    argv = ["solve", "--type", a.type, "-m", mesh_file, "--elem-order", str(int(doc.mesh.order)),
            "--num-modes", str(int(a.numModes)), "-o", output]
    if a.targetFreqGHz is not None and a.targetFreqGHz > 0:
        argv += ["--target-freq", repr(float(a.targetFreqGHz))]
    if a.type == "hom":
        argv += ["--az-order"] + (a.azOrders.split() or ["0"])
    if a.wave == "traveling":
        argv += ["-p", a.phases.strip()]
    return argv


def post_argv(doc: AxiDocument, raw: str, output: Optional[str] = None) -> list[str]:
    a = doc.analysis
    argv = ["post", "--type", a.type, "-i", raw, "-o", output or processed_file_name(raw),
            "--cond", f"{doc.post.cond:g}"]
    if a.type == "tm0":
        argv += ["--beta", f"{doc.post.beta:g}"]
    return argv


def report_argv(doc: AxiDocument, h5: str, output_dir: Optional[str] = None) -> list[str]:
    r = doc.report
    argv = ["report", "--type", doc.analysis.type, "-i", h5, "-o", output_dir or report_dir_name(h5)]
    if r.animate:
        argv.append("--animate")
    if r.timePhase:
        argv += ["--time-phase", f"{r.timePhase:g}"]
    if not r.showMesh:
        argv.append("--no-mesh")
    if int(r.dpi) != 120:
        argv += ["--dpi", str(int(r.dpi))]
    return argv


def export_argv(solver_type: str, input_file: str, output: str, shape: str, mode: int, n: Optional[int] = None,
                phase: Optional[float] = None, time_phase: float = 0.0, params: Optional[dict] = None) -> list[str]:
    """場の書き出し（``export`` サブコマンド）の argv. ``params`` は書き出しダイアログの値
    （z_range / r_range / nz / nr / p1 / p2 / npts / scale_to_power / scale / instant / fmt）."""
    p = params or {}
    argv = ["export", "--type", solver_type, "-i", input_file, "-o", output, "--shape", shape, "-m", str(int(mode))]
    if solver_type == "hom" and n is not None:
        argv += ["--n", str(int(n))]
    if phase is not None:
        argv += ["--phase", f"{float(phase):g}"]
    if time_phase:
        argv += ["--time-phase", f"{float(time_phase):g}"]
    pair = lambda v: f"{v[0]:g},{v[1]:g}"  # noqa: E731
    if shape == "area":
        if p.get("z_range") is not None:
            argv += ["--z-range", pair(p["z_range"])]
        if p.get("r_range") is not None:
            argv += ["--r-range", pair(p["r_range"])]
        argv += ["--nz", str(int(p.get("nz", 200))), "--nr", str(int(p.get("nr", 100)))]
    elif shape == "line":
        if p.get("p1") is not None and p.get("p2") is not None:
            argv += ["--p1", pair(p["p1"]), "--p2", pair(p["p2"])]
        argv += ["--npts", str(int(p.get("npts", 500)))]
    else:
        if p.get("z_range") is not None:
            argv += ["--z-range", pair(p["z_range"])]
        argv += ["--npts", str(int(p.get("npts", 500)))]
    if p.get("scale_to_power") is not None:
        argv += ["--scale-to-power", f"{float(p['scale_to_power']):g}"]
    if p.get("scale") not in (None, 1, 1.0):
        argv += ["--scale", f"{float(p['scale']):g}"]
    if p.get("instant"):
        argv.append("--instant")
    argv += ["--format", str(p.get("fmt", "both"))]
    return argv


def build_commands(doc: AxiDocument, mesh_file: str = MESH_FILE) -> dict:
    """解析ジョブの argv 一式: ``{"solve": [...], "post": [...] | None, "raw": name, "processed": name, "kind": …}``."""
    kind = result_kind(doc.analysis)
    raw = raw_file_name(kind)
    processed = processed_file_name(raw)
    return {
        "kind": kind, "raw": raw, "processed": processed if doc.analysis.runPost else None,
        "solve": solve_argv(doc, mesh_file, raw),
        "post": post_argv(doc, raw, processed) if doc.analysis.runPost else None,
    }


def command_line(argv: list[str], program: str = PROGRAM) -> str:
    """表示・command.log 用の 1 行（空白を含む引数は引用）."""
    return " ".join([program] + [shlex.quote(a) if (" " in a or not a) else a for a in argv])


def validate_analysis(doc: AxiDocument) -> list[str]:
    """解析設定の入力ミス（エラー文の一覧。空なら OK）."""
    from ...shared.cli_common import parse_phase_list

    a = doc.analysis
    errors: list[str] = []
    if int(a.numModes) < 1:
        errors.append("モード数は 1 以上にしてください")
    if a.targetFreqGHz is not None and a.targetFreqGHz <= 0:
        errors.append("予測周波数は正の値にしてください（GHz）")
    if a.type == "hom":
        orders = a.azOrders.split()
        if not orders or not all(o.isdigit() for o in orders):
            errors.append("方位角次数は空白区切りの 0 以上の整数にしてください（例: 0 1 2）")
    if a.wave == "traveling":
        try:
            phases = parse_phase_list(a.phases.strip())
        except Exception:  # noqa: BLE001
            phases = []
        if not phases or not a.phases.strip():
            errors.append("進行波の位相を入力してください（例: 120 / 60,90 / 0:180:20）")
        elif all(abs(p) < 1e-12 for p in phases):
            errors.append("進行波の位相が 0 です（定在波になります）")
    if doc.post.cond <= 0:
        errors.append("導電率は正の値にしてください [S/m]")
    if a.type == "tm0" and not (0 < doc.post.beta <= 1.0):
        errors.append("β = v/c は 0 < β ≤ 1 にしてください")
    return errors
