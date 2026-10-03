"""ジョブの実行（子プロセス、QProcess）と状態（EM-CAD-py ``em/session.py`` の転用）.

子は :mod:`.runner`（``python -m axicavity_fem.gui.jobs.runner <kind> <job_dir> --events``）で、進捗・ログ・結果を
標準出力の JSON 行で返す。同時に 1 本。

キャンセルは ``<job_dir>/cancel.flag``（pipeline の ``should_cancel`` がこれを見る。段階の切れ目で止まる）。
``cancel(force=True)`` は子プロセスを kill する（その時点で結果は無い）。

ジョブフォルダ（``job_dir``）の入力（geometry.json, job.json …）は呼び出し側（MeshController など）が用意する。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from . import runner

TERMINAL_STATUSES = ("done", "error", "cancelled", "interrupted")
DONE_MESSAGES = {"mesh": "メッシュ生成完了", "export_msh": "メッシュ書き出し完了", "analysis": "解析完了",
                 "post": "post 完了", "report": "レポート作成完了", "export": "書き出し完了", "plot": "図の保存完了"}
MAX_LOG_LINES = 5000


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class JobState:
    """1 ジョブの状態."""

    id: str
    dir: Path
    kind: str                       # runner.JOB_KINDS
    command: list[str] = field(default_factory=list)
    status: str = "queued"          # queued / running / <stage> / done / error / cancelled / interrupted
    stage: str = "queued"
    message: str = ""
    log: list[str] = field(default_factory=list)
    result: Optional[dict] = None   # done イベントの result
    error: Optional[str] = None
    traceback: Optional[str] = None
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES


class JobSession(QtCore.QObject):
    """ジョブの投入・実行・キャンセル（同時に 1 本。子プロセスで実行）.

    Signals:
        job_changed(JobState): 状態・段階・メッセージが変わった。
        log_appended(JobState, str): ログが 1 行増えた。
        job_finished(JobState): 終了した（done / error / cancelled）。
    """

    job_changed = QtCore.Signal(object)
    log_appended = QtCore.Signal(object, str)
    job_finished = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current: Optional[JobState] = None
        self._process: Optional[QtCore.QProcess] = None
        self._stdout_buf = b""          # 行末までのバイト列（UTF-8 の途中でチャンクが切れても壊さない）
        self._stderr_buf = ""
        self._got_terminal_event = False
        self._force_stopped = False

    @property
    def running(self) -> bool:
        return self.current is not None and not self.current.finished

    # ---- 投入 -----------------------------------------------------------

    def submit(self, kind: str, job_dir: str | Path, job_id: Optional[str] = None) -> JobState:
        """``job_dir``（入力を置いた既存のフォルダ）でジョブを始める."""
        if self.running:
            raise RuntimeError("別のジョブを実行中です。")
        if kind not in runner.JOB_KINDS:
            raise ValueError(f"不明なジョブの種類: {kind}")
        job_dir = Path(job_dir)
        if not job_dir.is_dir():
            raise ValueError(f"ジョブフォルダがありません: {job_dir}")
        job = JobState(id=job_id or job_dir.name, dir=job_dir, kind=kind,
                       command=runner.job_command(kind, job_dir))
        self._start(job)
        return job

    def cancel(self, force: bool = False) -> bool:
        """実行中のジョブにキャンセルを要求する（段階の切れ目で止まる）.

        ``force=True`` は子プロセスを直ちに kill する（固有値計算の途中でも止まる。結果は残らない）。
        """
        job = self.current
        if job is None or job.finished:
            return False
        if force:
            proc = self._process
            self._force_stopped = True
            self._append_log(job, "ユーザーが強制停止しました（子プロセスを終了します）")
            if proc is not None and proc.state() != QtCore.QProcess.NotRunning:
                proc.kill()
            else:
                self._finish("cancelled", "強制停止しました")
            return True
        runner.request_cancel(job.dir)
        job.message = "キャンセル中..."
        self._append_log(job, "ユーザーがキャンセルしました（次の段階で停止します）")
        self.job_changed.emit(job)
        return True

    # ---- 実行 -----------------------------------------------------------

    def _append_log(self, job: JobState, line: str) -> None:
        job.log.append(line)
        if len(job.log) > MAX_LOG_LINES:
            del job.log[: len(job.log) - MAX_LOG_LINES]
        try:
            with open(job.dir / runner.LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        self.log_appended.emit(job, line)

    def _start(self, job: JobState) -> None:
        self.current = job
        runner.clear_cancel(job.dir)
        job.status = job.stage = "running"
        job.message = "開始..."
        job.started_at = _now()
        self._append_log(job, f"ジョブ開始: {job.kind} ({job.id})")

        command = list(job.command)
        proc = QtCore.QProcess(self)
        proc.setProgram(command[0])
        proc.setArguments(command[1:])
        proc.setWorkingDirectory(str(job.dir))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        for key, value in runner.child_environment().items():
            env.insert(key, value)
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        # どの子プロセスのシグナルかは受け側で sender() で見分ける（_emitter）: 終端イベント（done など）の後、
        # 子が実際に終わる前に次のジョブを始めることがある。前の子の finished を次のジョブのものと取り違えない。
        # self を捕まえるラムダで結線すると循環参照になり、子の deleteLater の途中でセッションが解放されて落ちる
        # （EM-CAD-py で再現）ので、メソッドのまま結線する
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        self._process = proc
        self._stdout_buf, self._stderr_buf = b"", ""
        self._got_terminal_event = False
        self._force_stopped = False
        self.job_changed.emit(job)
        self._append_log(job, "子プロセス: " + " ".join(command))
        proc.start()

    # ---- 子プロセスからのイベント ------------------------------------------

    def _emitter(self) -> Optional[QtCore.QProcess]:
        """シグナルを出した子プロセス（シグナル以外から呼ばれたときは今の子プロセス）."""
        sender = self.sender()
        return sender if isinstance(sender, QtCore.QProcess) else self._process

    def _on_stdout(self) -> None:
        proc = self._emitter()
        if proc is None:
            return
        if proc is not self._process:
            proc.readAllStandardOutput()              # 終わったジョブの子の遅れた出力は捨てる
            return
        self._stdout_buf += bytes(proc.readAllStandardOutput().data())
        while b"\n" in self._stdout_buf:
            raw, self._stdout_buf = self._stdout_buf.split(b"\n", 1)
            self._handle_line(raw.decode("utf-8", errors="replace").rstrip("\r"))

    def _on_stderr(self) -> None:
        proc = self._emitter()
        if proc is None:
            return
        if proc is not self._process:
            proc.readAllStandardError()
            return
        self._stderr_buf += bytes(proc.readAllStandardError().data()).decode("utf-8", errors="replace")

    def _handle_line(self, line: str) -> None:
        if not line.strip():
            return
        event = runner.parse_event(line)
        if event is None:
            self._on_log(line)                       # JSON でない出力もログへ
            return
        kind = event.get("event")
        if kind == "progress":
            self._on_progress(str(event.get("stage", "")), str(event.get("message", "")))
        elif kind == "log":
            self._on_log(str(event.get("line", "")))
        elif kind == "done":
            self._got_terminal_event = True
            self._on_done(event.get("result"))
        elif kind == "error":
            self._got_terminal_event = True
            self._on_failed(str(event.get("error", "")), str(event.get("traceback", "")))
        elif kind == "cancelled":
            self._got_terminal_event = True
            self._on_cancelled()

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        proc = self._emitter()
        if proc is not None and proc is not self._process:
            proc.deleteLater()                        # 終わったジョブの子が、次のジョブを始めた後に終わった
            return
        # 残りの出力を処理してから、終端イベントが無ければ異常終了として扱う
        self._on_stdout()
        if self._stdout_buf.strip():
            self._handle_line(self._stdout_buf.decode("utf-8", errors="replace"))
            self._stdout_buf = b""
        self._on_stderr()
        job = self.current
        if job is not None and not job.finished and not self._got_terminal_event:
            if self._force_stopped:
                self._on_cancelled()
            else:
                crashed = exit_status == QtCore.QProcess.CrashExit
                what = ("子プロセスが異常終了しました（gmsh / ソルバのクラッシュ）" if crashed
                        else f"子プロセスが結果を返さずに終了しました（終了コード {exit_code}）")
                detail = "\n".join(self._stderr_buf.strip().splitlines()[-20:])
                self._on_failed(what, detail)
        self._cleanup_process()

    def _on_process_error(self, error) -> None:
        if self._emitter() is not self._process:
            return
        job = self.current
        if job is None or job.finished:
            return
        if error == QtCore.QProcess.FailedToStart:
            self._on_failed("子プロセスを起動できません", " ".join(job.command))
            self._cleanup_process()

    def _on_progress(self, stage: str, message: str) -> None:
        job = self.current
        if job is None or job.finished:
            return
        job.stage = stage
        if stage not in TERMINAL_STATUSES:
            job.status = stage
        job.message = message
        self.job_changed.emit(job)

    def _on_log(self, line: str) -> None:
        if self.current is not None:
            self._append_log(self.current, line)

    def _finish(self, status: str, message: str) -> None:
        job = self.current
        if job is None or job.finished:
            return
        job.status = job.stage = status
        job.message = message
        job.finished_at = _now()
        self._append_log(job, message)
        self.job_changed.emit(job)
        self.job_finished.emit(job)

    def _on_done(self, result) -> None:
        job = self.current
        if job is not None:
            job.result = result if isinstance(result, dict) else None
        kind = job.kind if job is not None else ""
        self._finish("done", DONE_MESSAGES.get(kind, "完了"))

    def _on_failed(self, error: str, tb: str) -> None:
        if self.current is not None:
            self.current.error = error
            self.current.traceback = tb
        self._finish("error", f"エラー: {error}")

    def _on_cancelled(self) -> None:
        job = self.current
        if job is not None:
            runner.clear_cancel(job.dir)
        self._finish("cancelled", "キャンセルしました")

    def _cleanup_process(self) -> None:
        proc = self._process
        if proc is not None:
            proc.deleteLater()
        self._process = None

    def wait(self, timeout_ms: int = 600_000) -> bool:
        """（テスト用）子プロセスの終了を待つ。イベントループは回さない."""
        proc = self._process
        if proc is None or proc.state() == QtCore.QProcess.NotRunning:
            return True
        return proc.waitForFinished(timeout_ms)

    def shutdown(self) -> None:
        """GUI 終了時: キャンセルを要求し、少し待って残っていれば kill する."""
        job = self.current
        proc = self._process
        if job is not None and not job.finished:
            runner.request_cancel(job.dir)
        if proc is not None and proc.state() != QtCore.QProcess.NotRunning:
            if not proc.waitForFinished(3_000):
                proc.kill()
                proc.waitForFinished(3_000)
