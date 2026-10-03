"""AxiCavity-FEM の入口を 1 つにまとめたもの（Windows 版 EXE の main。開発環境では ``python -m axicavity_fem.gui.launcher``）.

使い方（EXE の場合。``AxiCavity-FEM.exe`` の代わりに ``python -m axicavity_fem.gui.launcher`` でも同じ）::

    AxiCavity-FEM.exe                          GUI
    AxiCavity-FEM.exe <file>                   GUI でファイルを開く（.axiproj / .gmshproj / .af / .h5）
    AxiCavity-FEM.exe gui [file]
    AxiCavity-FEM.exe solve|post|info|export|plot|report ...
                                               コマンドライン（pip 版の axicavity-fem と同じ）
    AxiCavity-FEM.exe run <project> [...]      パラメータを変えて解析（pip 版の axicavity-fem-run と同じ）
    AxiCavity-FEM.exe version                  版と同梱物（PARDISO・メッシャ・3D 表示）
    AxiCavity-FEM.exe selftest                 GUI と同じ子プロセス経路で円筒空洞を解いて確かめる
    AxiCavity-FEM.exe job <kind> <dir> --events
                                               GUI が使う子プロセス（内部用）

EXE では gmsh（GPL）を本体に入れないので、起動時に ``.msh`` 読込専用の互換モジュール（:mod:`.mshlite`）を
``gmsh`` として登録し、メッシュ生成は同梱のメッシャ（別プロセス）で行う（:mod:`.jobs.meshing`）。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import frozen

CORE_COMMANDS = ("solve", "post", "info", "export", "plot", "report")
OPEN_SUFFIXES = (".axiproj", ".gmshproj", ".af", ".h5", ".hdf5")
PILLBOX_F010_GHZ = 2.404825557695773 * 299792458.0 / (2 * 3.141592653589793 * 0.05) / 1e9   # 半径 50 mm


def prepare_runtime() -> None:
    """EXE のとき（または ``AXICAVITY_GMSH=lite``）: ``import gmsh`` が読込専用の互換モジュールを返すようにする."""
    if frozen.is_compiled() or os.environ.get("AXICAVITY_GMSH", "").lower() == "lite":
        from .mshlite import install_as_gmsh

        install_as_gmsh()


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):           # 子プロセスの出力は親が UTF-8 で読む。リダイレクト先も UTF-8
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _is_gui_request(argv: list[str]) -> bool:
    if not argv or argv[0] == "gui":
        return True
    return len(argv) == 1 and Path(argv[0]).suffix.lower() in OPEN_SUFFIXES


def _run_gui(files: list[str]) -> int:
    os.environ.setdefault("QT_API", "pyside6")        # pyvistaqt（qtpy）のバインディング選択
    if frozen.is_compiled():
        _leave_install_folder()
    from .app import main as gui_main

    return int(gui_main(files) or 0)


def _leave_install_folder() -> None:
    """EXE をダブルクリックすると作業フォルダが EXE のフォルダになる。保存ダイアログの既定がそこにならないよう
    ドキュメントへ移す（コマンドラインから起動したときは移さない）."""
    try:
        if Path.cwd().resolve() == frozen.app_dir():
            documents = Path.home() / "Documents"
            os.chdir(documents if documents.is_dir() else Path.home())
    except OSError:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    prepare_runtime()
    if _is_gui_request(argv):
        return _run_gui(argv[1:] if argv and argv[0] == "gui" else argv)
    _utf8_streams()
    command, rest = argv[0], argv[1:]
    if command in CORE_COMMANDS:
        from ..cli.main import cli_main

        return int(cli_main(argv) or 0)
    if command == "run":
        from .batch import main as batch_main

        return int(batch_main(rest) or 0)
    if command == "job":
        from .jobs import runner

        return int(runner.main(rest) or 0)
    if command == "version":
        from .app_info import info_text

        print(info_text())
        return 0
    if command == "selftest":
        return selftest(rest)
    if command in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if command in ("-V", "--version"):
        from . import GUI_VERSION

        print(f"AxiCavity-FEM {GUI_VERSION}")
        return 0
    print(f"不明なコマンド: {command}\n\n{__doc__}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest(argv: list[str]) -> int:
    """同梱物の確認と、GUI と同じ子プロセス経路での解析（メッシュ生成 → solve → post）."""
    parser = argparse.ArgumentParser(prog="AxiCavity-FEM selftest")
    parser.add_argument("--work-dir", help="作業フォルダ（省略で一時フォルダ）")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    from .app_info import info_text, pardiso_status

    print(info_text())
    print()
    problems: list[str] = []
    problems += _check_resources()
    pardiso_ok, pardiso = pardiso_status()
    print(f"PARDISO: {pardiso}")
    if frozen.is_compiled() and not pardiso_ok:
        problems.append("PARDISO（Intel MKL）が使えない")
    problems += _check_analysis_job(args.work_dir, args.timeout)
    if problems:
        print("selftest → NG: " + " / ".join(problems))
        return 1
    print("selftest → OK")
    return 0


def _check_resources() -> list[str]:
    """アイコン・翻訳・3D 表示（同梱漏れは黙って空になるので明示的に確かめる）."""
    problems = []
    from .i18n.tr import _load as load_language
    from .ui.icon_set import ACTION_ICONS, ICON_DIR

    svg = {p.stem for p in ICON_DIR.glob("*.svg")}
    missing = sorted(set(ACTION_ICONS.values()) - svg)
    print(f"アイコン: {len(svg)} 個" + (f"（不足 {missing[:5]}）" if missing else ""))
    if missing:
        problems.append("アイコンの同梱漏れ")
    for lang in ("ja", "en"):
        ok = bool(load_language(lang))
        print(f"翻訳 {lang}: {'OK' if ok else 'なし'}")
        if not ok:
            problems.append(f"翻訳 {lang} の同梱漏れ")
    from .ui.view3d_window import is_available

    available = is_available()
    print(f"3D 表示: {'OK' if available else 'なし'}")
    if frozen.is_compiled() and not available:
        problems.append("3D 表示（pyvista）の同梱漏れ")
    return problems


def _pillbox_job(folder: Path) -> None:
    """半径 50 mm・長さ 100 mm の円筒空洞（TM010 の解析解 = PILLBOX_F010_GHZ）の解析ジョブを作る."""
    import json

    from .core.convert import to_multi_region
    from .core.document import create_empty_document
    from .core.sketch.model import add_rectangle
    from .jobs.commands import MESH_FILE, build_commands
    from .jobs.pipeline import GEOMETRY_FILE, JOB_FILE

    doc = create_empty_document("selftest")
    add_rectangle(doc.sketch, (0.0, 0.0), (100.0, 50.0))
    doc.mesh.size = 6.0
    doc.analysis.numModes = 3
    converted = to_multi_region(doc, strict=True)
    folder.mkdir(parents=True, exist_ok=True)
    converted.geom.save_json(folder / GEOMETRY_FILE)
    cmds = build_commands(doc)
    job = {"kind": cmds["kind"], "solverType": "tm0", "wave": "standing", "meshOrder": int(doc.mesh.order),
           "units": doc.meta.units, "argv": {"solve": cmds["solve"], "post": cmds["post"]},
           "files": {"mesh": MESH_FILE, "raw": cmds["raw"], "processed": cmds["processed"]}}
    (folder / JOB_FILE).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_analysis_job(work_dir: Optional[str], timeout: float) -> list[str]:
    from PySide6 import QtCore

    from .jobs import runner
    from .jobs.session import JobSession

    folder = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="axicavity_selftest_"))
    job_dir = folder / "pillbox"
    _pillbox_job(job_dir)
    print("子プロセスのコマンド: " + " ".join(runner.job_command("analysis", job_dir)))
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv[:1])
    session = JobSession()
    finished: list = []
    session.job_finished.connect(finished.append)
    session.log_appended.connect(lambda _job, line: print("  | " + line))
    t0 = time.perf_counter()
    job = session.submit("analysis", job_dir)
    timer = QtCore.QTimer()
    timer.setInterval(100)
    timer.timeout.connect(lambda: (finished or time.perf_counter() - t0 > timeout) and app.quit())
    timer.start()
    app.exec()
    timer.stop()
    if not finished:
        session.shutdown()
        return [f"解析ジョブが {timeout:.0f} 秒で終わらない"]
    if job.status != "done":
        return [f"解析ジョブ: {job.status} {job.error or ''}".strip()]
    modes = ((job.result or {}).get("summary") or {}).get("modes") or []
    if not modes:
        return ["解析ジョブ: モードが無い"]
    f0 = float(modes[0]["f_GHz"])
    err = abs(f0 - PILLBOX_F010_GHZ) / PILLBOX_F010_GHZ
    print(f"円筒空洞 TM010: f = {f0:.6f} GHz（解析解 {PILLBOX_F010_GHZ:.6f} GHz、誤差 {err:.1e}）、"
          f"Q = {modes[0].get('Q', float('nan')):.4g}、{time.perf_counter() - t0:.1f} s")
    return [] if err < 1e-4 else [f"TM010 の周波数がずれている（{f0:.6f} GHz）"]


if __name__ == "__main__":
    raise SystemExit(main())
