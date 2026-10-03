"""解析タブの状態: 解析の投入（自動保存 → 結果フォルダ → 子プロセス）、post の再実行、レポート作成、結果への記録.

ジョブは :class:`~axicavity_fem.gui.jobs.session.JobSession`（同時に 1 本）。結果フォルダは
``<名前>.axiproj.data/results/NNNN-<kind>/``（:class:`~axicavity_fem.gui.project.controller.ProjectController`）。
メッシュタブで生成した同じ設定のメッシュがあればコピーして再利用し、無ければ子プロセスが ``geometry.json`` から作る。
CLI 相当のコマンドは ``command.log`` に残す（wx 版と同じ書式）。
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from ..app.store import DocumentStore
from ..core.convert import ConvertError, check_geometry, to_multi_region
from ..io.axiproj import ResultEntry, raw_file_name, result_kind
from ..project.controller import ProjectController
from .commands import build_commands, command_line, post_argv, processed_file_name, report_argv, validate_analysis
from .mesh_controller import MeshController
from .pipeline import GEOMETRY_FILE, JOB_FILE, MATERIALS_FILE, MESH_FILE
from .session import JobSession, JobState


class AnalysisController(QtCore.QObject):
    """Signals:
        changed(): 実行中 / 完了 / エラーが変わった。
    """

    changed = QtCore.Signal()

    def __init__(self, store: DocumentStore, session: JobSession, project: ProjectController,
                 mesh: MeshController, parent=None):
        super().__init__(parent)
        self.store = store
        self.session = session
        self.project = project
        self.mesh = mesh
        self.job: Optional[JobState] = None
        self.last_error: Optional[str] = None
        session.job_changed.connect(self._on_job_changed)
        session.job_finished.connect(self._on_job_finished)

    @property
    def running(self) -> bool:
        return self.job is not None and not self.job.finished

    # ------------------------------------------------------------------
    # 検証・確認
    # ------------------------------------------------------------------

    def check(self) -> tuple[list[str], list[str]]:
        """(エラー, 警告)。エラーがあれば実行しない."""
        issues = check_geometry(self.store.document)
        errors = [i.message for i in issues if i.level == "error"] + validate_analysis(self.store.document)
        warnings = [i.message for i in issues if i.level != "error"]
        return errors, warnings

    def commands(self) -> dict:
        return build_commands(self.store.document)

    def preview_lines(self) -> list[str]:
        cmds = self.commands()
        lines = [command_line(cmds["solve"])]
        if cmds["post"]:
            lines.append(command_line(cmds["post"]))
        return lines

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """解析を投入する（プロジェクトの自動保存 → 結果フォルダ → 子プロセス）. 投入できたら True.

        未保存のプロジェクトでは ValueError（UI は先に名前を付けて保存させる）。
        """
        if self.session.running:
            self._fail("別のジョブを実行中です。")
            return False
        errors, _ = self.check()
        if errors:
            self._fail("\n".join(errors))
            return False
        doc = self.store.document
        try:
            converted = to_multi_region(doc, strict=True)
        except ConvertError as exc:
            self._fail(str(exc))
            return False
        rid = self.project.prepare_run()                       # ValueError なら未保存
        result_dir = self.project.results_dir / rid
        result_dir.mkdir(parents=True, exist_ok=True)
        converted.geom.save_json(result_dir / GEOMETRY_FILE)
        cmds = build_commands(doc)
        job = {"kind": cmds["kind"], "solverType": doc.analysis.type, "wave": doc.analysis.wave,
               "meshOrder": int(doc.mesh.order), "units": doc.meta.units,
               "argv": {"solve": cmds["solve"], "post": cmds["post"]},
               "files": {"mesh": MESH_FILE, "raw": cmds["raw"], "processed": cmds["processed"]},
               "createdAt": _dt.datetime.now().isoformat(timespec="seconds"),
               "generator": self.project.generator}
        reuse = self.mesh.reusable_mesh()
        if reuse is not None:
            shutil.copy2(reuse, result_dir / MESH_FILE)
            materials = reuse.with_name(MATERIALS_FILE)
            if materials.exists():
                shutil.copy2(materials, result_dir / MATERIALS_FILE)
            job["meshReused"] = str(reuse)
        (result_dir / JOB_FILE).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        self.project.register_submitted(result_dir, cmds["kind"])
        self._log_commands(rid, [cmds["solve"]] + ([cmds["post"]] if cmds["post"] else []))
        self.last_error = None
        self.job = self.session.submit("analysis", result_dir, job_id=rid)
        self.changed.emit()
        return True

    def run_post(self, entry: ResultEntry) -> bool:
        """既存の結果に post をやり直す（導電率・β を変えたときなど）."""
        raw = entry.file("raw") or entry.dir / raw_file_name(entry.kind)
        if self.session.running or not raw.exists():
            self._fail("生の結果（solve の出力）がありません。")
            return False
        argv = post_argv(self.store.document, raw.name, processed_file_name(raw.name))
        self._update_job(entry.dir, kind="post", argv_key="post", argv=argv)
        self.project.set_running(entry.dir)
        self._log_commands(entry.id, [argv])
        self.last_error = None
        self.job = self.session.submit("post", entry.dir, job_id=f"{entry.id}-post")
        self.changed.emit()
        return True

    def run_report(self, entry: ResultEntry) -> bool:
        """HTML レポートを作る（post 済みがあればそれ、無ければ生の結果から）."""
        processed = entry.file("processed")
        source = processed if processed is not None and processed.exists() else \
            (entry.file("raw") or entry.dir / raw_file_name(entry.kind))
        if self.session.running or not source.exists():
            self._fail("結果ファイルがありません。先に解析を実行してください。")
            return False
        argv = report_argv(self.store.document, source.name)
        self._update_job(entry.dir, kind="report", argv_key="report", argv=argv)
        self.project.set_running(entry.dir)
        self._log_commands(entry.id, [argv])
        self.last_error = None
        self.job = self.session.submit("report", entry.dir, job_id=f"{entry.id}-report")
        self.changed.emit()
        return True

    def run_export(self, argv: list[str], entry: Optional[ResultEntry] = None,
                   job_dir: Optional[Path] = None) -> bool:
        """場の書き出し（``export``）を子プロセスで実行する.

        プロジェクトの結果なら ``entry.dir`` を作業フォルダにして argv は相対パス、外部の h5 なら ``job_dir``
        （一時フォルダ）で argv は絶対パス。結果の状態（result.json）は変えない。
        """
        if self.session.running:
            self._fail("別のジョブを実行中です。")
            return False
        folder = entry.dir if entry is not None else job_dir
        if folder is None:
            raise ValueError("job_dir が必要です")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self._update_job(folder, kind="export", argv_key="export", argv=argv)
        label = entry.id if entry is not None else "export"
        self._log_commands(label, [argv])
        self.last_error = None
        self.job = self.session.submit("export", folder, job_id=f"{label}-export")
        self.changed.emit()
        return True

    def cancel(self, force: bool = False) -> bool:
        return self.running and self.session.cancel(force=force)

    @staticmethod
    def report_path(entry: Optional[ResultEntry]) -> Optional[Path]:
        if entry is None:
            return None
        path = entry.file("report")
        if path is not None and path.exists():
            return path
        source = entry.file("processed") or entry.file("raw") or entry.dir / raw_file_name(entry.kind)
        candidate = source.with_name(source.stem + "_report") / "index.html"
        return candidate if candidate.exists() else None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _fail(self, message: str) -> None:
        self.last_error = message
        self.changed.emit()

    def _update_job(self, result_dir: Path, kind: str, argv_key: str, argv: list[str]) -> None:
        path = result_dir / JOB_FILE
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            job = {}
        job.setdefault("argv", {})[argv_key] = argv
        job["lastKind"] = kind
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_commands(self, label: str, argvs: list[list[str]]) -> None:
        for argv in argvs:
            self.project.log_command(f"[{label}] {command_line(argv)}")

    def _on_job_changed(self, job: JobState) -> None:
        if job is self.job:
            self.changed.emit()

    def _on_job_finished(self, job: JobState) -> None:
        if job is not self.job:
            return
        result = job.result or {}
        entry = self.project.result(job.id.split("-post")[0].split("-report")[0])
        files = dict(entry.files) if entry is not None else {}
        files.update({k: v for k, v in (result.get("files") or {}).items() if v})
        summary = result.get("summary") or None
        if job.kind == "export":
            if job.status == "error":
                self.last_error = job.error
            self.changed.emit()
            return
        if job.kind == "analysis":
            self.project.finish_result(job.dir, job.status, error=job.error or "",
                                       summary=summary, files=files if files else None)
        else:
            self.project.set_running(None)
            if job.status == "done":
                self.project.finish_result(job.dir, "done", summary=summary, files=files if files else None)
            else:
                self.project.refresh_results()
        if job.status == "error":
            self.last_error = job.error
        self.changed.emit()
