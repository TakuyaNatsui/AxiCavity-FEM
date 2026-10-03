"""ジョブを子プロセスで実行するための入口とイベント形式（EM-CAD-py ``em/runner.py`` の転用）.

GUI（:class:`~axicavity_fem.gui.jobs.session.JobSession`、QProcess）はメッシュ生成・解析を **子プロセス**
で実行する::

    python -m axicavity_fem.gui.jobs.runner <kind> <job_dir> --events

``kind`` は "mesh"（メッシュ生成）/ "export_msh"（ユーザー指定のパスへ .msh 書き出し）/ "analysis"（solve → post、M7）/
"post" / "report" / "export"（場の書き出し）/ "plot"（未実装。PNG は GUI が保存する）。``job_dir`` に入力（``geometry.json``、``job.json`` …）を置き、
生成物もそこに書く。

親 → 子: キャンセルは ``<job_dir>/cancel.flag`` を作る（:func:`request_cancel`）。子は段階の切れ目でこのファイルを
見て止まる（固有値計算の途中では止まらない）。強制停止は親がプロセスを kill する。

子 → 親: 起動時の標準出力（fd 1 の複製）に JSON を 1 行ずつ::

    {"event": "progress", "stage": "meshing", "message": "..."}
    {"event": "log", "line": "..."}
    {"event": "done", "result": {...}}
    {"event": "error", "error": "TypeName: message", "traceback": "..."}
    {"event": "cancelled"}

子は ``--events`` のとき **fd 1 を ``<job_dir>/log.txt`` に付け替える**（gmsh が C レベルで書く出力をイベントの列に
混ぜないため）。Python の ``print`` は ``pipeline.capture_output`` で log イベントになる。標準エラー出力は診断用
（親はエラー時にログへ流す）。終了コードは done 0 / cancelled 0 / error 1。子の環境は UTF-8（:func:`child_environment`）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional, TextIO

from .. import frozen

CANCEL_FLAG = "cancel.flag"
LOG_FILE = "log.txt"
JOB_KINDS = ("mesh", "export_msh", "analysis", "post", "report", "export", "plot")


def program_command() -> list[str]:
    """子プロセスの入口を起動するコマンド.

    開発・pip: ``python -m axicavity_fem.gui.jobs.runner``。Windows 版（EXE）: ``AxiCavity-FEM.exe job``
    （EXE 自身。runner の種類名 post / report / export はコア CLI のサブコマンドと同じ名前なので ``job`` の下に置く）。
    """
    if frozen.is_compiled():
        return [frozen.executable_path(), "job"]
    return [sys.executable, "-m", "axicavity_fem.gui.jobs.runner"]


def job_command(kind: str, job_dir: str | Path) -> list[str]:
    return program_command() + [kind, str(job_dir), "--events"]


def child_environment() -> dict[str, str]:
    """子プロセスの環境変数（UTF-8 出力を強制）."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---- キャンセル（フラグファイル） -------------------------------------------

def cancel_flag_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / CANCEL_FLAG


def request_cancel(job_dir: str | Path) -> None:
    cancel_flag_path(job_dir).write_text("cancel", encoding="utf-8")


def clear_cancel(job_dir: str | Path) -> None:
    try:
        cancel_flag_path(job_dir).unlink()
    except OSError:
        pass


def cancel_requested(job_dir: str | Path) -> bool:
    return cancel_flag_path(job_dir).exists()


# ---- イベント -------------------------------------------------------------

def emit(stream: TextIO, event: dict) -> None:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def parse_event(line: str) -> Optional[dict]:
    """子の標準出力 1 行をイベントに変換する（JSON でなければ None）."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and "event" in event else None


# ---- 子プロセス側 -------------------------------------------------------------

def _callbacks(job_dir: Path, events: bool, out: TextIO
               ) -> tuple[Callable[[str, str], None], Callable[[str], None], Callable[[], bool]]:
    if events:
        def progress(stage: str, message: str) -> None:
            emit(out, {"event": "progress", "stage": stage, "message": message})

        def log(line: str) -> None:
            emit(out, {"event": "log", "line": line})
    else:
        def progress(stage: str, message: str) -> None:
            out.write(f"[{stage}] {message}\n")
            out.flush()

        def log(line: str) -> None:
            out.write(line + "\n")
            out.flush()

    return progress, log, lambda: cancel_requested(job_dir)


def _redirect_fd1_to_log(job_dir: Path) -> TextIO:
    """fd 1 を ``log.txt``（追記）に付け替え、元の fd 1 を複製したテキストストリーム（イベント用）を返す."""
    events_fd = os.dup(1)
    log_fd = os.open(str(job_dir / LOG_FILE), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log_fd, 1)
    os.close(log_fd)
    return os.fdopen(events_fd, "w", encoding="utf-8", errors="replace", buffering=1)


def run_child(kind: str, job_dir: str | Path, events: bool, out: TextIO | None = None) -> int:
    """子プロセス本体。終了コードを返す.

    ``out`` を省略し ``events`` が真なら fd 1 を ``log.txt`` に付け替えて、イベントは元の標準出力へ書く。
    ``out`` を渡すとそこへ書く（テスト用。fd は触らない）。
    """
    from . import pipeline

    job_dir = Path(job_dir)
    if out is None:
        out = _redirect_fd1_to_log(job_dir) if events else sys.stdout
    progress, log, should_cancel = _callbacks(job_dir, events, out)
    try:
        if kind in ("mesh", "export_msh"):
            result = pipeline.run_mesh_job(job_dir, progress=progress, log=log, should_cancel=should_cancel)
        elif kind in ("analysis", "post", "report", "export"):
            from . import analysis_pipeline

            result = analysis_pipeline.run_job(kind, job_dir, progress=progress, log=log,
                                               should_cancel=should_cancel)
        elif kind in JOB_KINDS:
            raise NotImplementedError(f"ジョブ '{kind}' は未実装です（後の段階）")
        else:
            raise ValueError(f"不明なジョブの種類: {kind}")
    except pipeline.JobCancelled:      # runner は -m で __main__ として動くので、クラスは pipeline 側に置く
        if events:
            emit(out, {"event": "cancelled"})
        else:
            out.write("キャンセルされました\n")
        return 0
    except Exception as exc:  # noqa: BLE001 — 何であれ親へ報告する
        tb = traceback.format_exc()
        if events:
            emit(out, {"event": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": tb})
        else:
            out.write(tb)
            out.write(f"エラー: {type(exc).__name__}: {exc}\n")
        return 1
    if events:
        emit(out, {"event": "done", "result": result})
    else:
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m axicavity_fem.gui.jobs.runner",
                                     description="AxiCavity-FEM GUI のジョブを子プロセスで実行する")
    parser.add_argument("kind", choices=JOB_KINDS)
    parser.add_argument("job_dir")
    parser.add_argument("--events", action="store_true", help="進捗を JSON 行で標準出力へ（GUI の子プロセス用）")
    args = parser.parse_args(argv)
    if frozen.is_compiled() or os.environ.get("AXICAVITY_GMSH", "").lower() == "lite":
        from ..mshlite import install_as_gmsh     # EXE と同じ構成（.msh は gmsh 無しで読む）を開発環境でも試せる

        install_as_gmsh()
    for stream in (sys.stdout, sys.stderr):      # 親は UTF-8 で読む
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    return run_child(args.kind, args.job_dir, events=args.events)


if __name__ == "__main__":
    raise SystemExit(main())
