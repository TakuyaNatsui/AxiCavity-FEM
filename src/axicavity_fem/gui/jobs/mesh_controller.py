"""メッシュタブの状態: メッシュだけの生成（子プロセス）、統計、スケッチ画面への重ね表示、解析での再利用
（EM-CAD-py ``em/mesh_controller.py`` の転用。3D シーンの代わりに ``preview_geometry()`` を UI に渡す）.

- 生成は :meth:`JobSession.submit`（kind "mesh"）→ 子プロセス :func:`pipeline.run_mesh_job`。出力先はプロジェクトの
  ``<名前>.axiproj.data/mesh/`` の下の**生成ごとの新しいサブフォルダ**（:mod:`.mesh_folder`）。失敗・中止したら
  直前のメッシュに戻る
- 完了すると GUI が ``mesh_key.json`` に形状ハッシュとメッシュハッシュ（``core.hashes``）を書く。現在のモデルの
  メッシュハッシュと一致すれば「最新」で、解析実行時に :meth:`reusable_mesh` がその ``model.msh`` を返す
- プレビューは ``mesh_preview.npz``（子が書く。m 単位）を文書の単位に換算して :meth:`preview_geometry` で返す
  （``ui.sketch_view.MeshPreview`` の引数）。形状が変わったら隠す
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore

from ..app.store import DocumentStore
from ..core.convert import UNIT_TO_M, ConvertError, to_multi_region
from ..core.hashes import geometry_hash, mesh_hash
from .mesh_folder import MESH_KEY_FILE, current_generation, new_generation_dir, remove_stale
from .pipeline import BC_NAMES, GEOMETRY_FILE, JOB_FILE, MESH_FILE, MESH_PREVIEW_FILE, MESH_SUMMARY_FILE
from .session import JobSession, JobState

__all__ = ["MESH_KEY_FILE", "MeshController", "MeshInfo"]


@dataclass
class MeshInfo:
    """生成済みのメッシュ 1 つ（mesh/ の下の生成フォルダ）."""

    dir: Path
    summary: dict
    mesh_hash: Optional[str]
    geometry_hash: Optional[str]
    created_at: str = ""

    @property
    def stats(self) -> dict:
        return self.summary.get("stats") or {}

    @property
    def msh_path(self) -> Path:
        return self.dir / MESH_FILE

    @property
    def materials_path(self) -> Path:
        return self.dir / "model.materials.json"


class MeshController(QtCore.QObject):
    """メッシュタブの状態と操作.

    Signals:
        changed(): 実行中/完了/エラー、生成済みメッシュ、表示のいずれかが変わった。
    """

    changed = QtCore.Signal()

    def __init__(self, store: DocumentStore, session: JobSession, parent=None):
        super().__init__(parent)
        self.store = store
        self.session = session
        self.job: Optional[JobState] = None
        self.info: Optional[MeshInfo] = None
        self.visible = False
        self.last_error: Optional[str] = None
        self._pending: Optional[tuple[str, str]] = None      # 投入時の (形状, メッシュ) ハッシュ
        self._root: Optional[Path] = None                     # プロジェクトの mesh/（生成フォルダの親）
        self._preview: Optional[dict] = None
        self._hash_key = None
        self._hashes = ("", "")
        session.job_changed.connect(self._on_job_changed)
        session.job_finished.connect(self._on_job_finished)
        store.document_changed.connect(self._on_document_changed)

    # ------------------------------------------------------------------
    # 問い合わせ
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.job is not None and not self.job.finished

    def current_hashes(self) -> tuple[str, str]:
        """現在のモデルの (形状ハッシュ, メッシュハッシュ). 文書の版ごとにキャッシュする."""
        store = self.store
        key = (id(store.document), store.sketch_version)
        if key != self._hash_key:
            self._hashes = (geometry_hash(store.document), mesh_hash(store.document))
            self._hash_key = key
        return self._hashes

    def state(self) -> str:
        """"none"（未生成）/ "current"（今の設定と同じ）/ "settings"（設定が変わった）/ "geometry"（形状が変わった）."""
        if self.info is None:
            return "none"
        geom, mesh = self.current_hashes()
        if self.info.geometry_hash != geom:
            return "geometry"
        if self.info.mesh_hash != mesh:
            return "settings"
        return "current"

    def reusable_mesh(self) -> Optional[Path]:
        """今の設定と同じメッシュがあればその model.msh（解析で再利用する）."""
        if self.info is None or self.state() != "current":
            return None
        path = self.info.msh_path
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # 生成・読み込み
    # ------------------------------------------------------------------

    def set_error(self, message: Optional[str]) -> None:
        self.last_error = message
        self.changed.emit()

    def generate(self, mesh_root: str | Path) -> bool:
        """メッシュだけを子プロセスで作る. 投入できたら True（できなければ ``last_error``）.

        ``mesh_root``（プロジェクトの ``mesh/``）の下に新しいサブフォルダを作ってそこに出力する
        （前のメッシュは成功するまで残す）。
        """
        if self.session.running:
            self.set_error("別のジョブを実行中です。")
            return False
        doc = self.store.document
        try:
            converted = to_multi_region(doc, strict=True)
        except ConvertError as exc:
            self.set_error(str(exc))
            return False
        self._root = Path(mesh_root)
        folder = None
        try:
            pending = self.current_hashes()
            folder = new_generation_dir(self._root)
            converted.geom.save_json(folder / GEOMETRY_FILE)
            (folder / JOB_FILE).write_text(json.dumps(
                {"kind": "mesh", "meshOrder": int(doc.mesh.order), "units": doc.meta.units,
                 "warnings": converted.warnings}, ensure_ascii=False, indent=2), encoding="utf-8")
            self.job = self.session.submit("mesh", folder)
        except Exception as exc:  # noqa: BLE001 — 失敗理由はパネルに出す
            if folder is not None:
                try:
                    for child in folder.iterdir():
                        child.unlink()
                    folder.rmdir()
                except OSError:
                    pass
            self.set_error(f"{type(exc).__name__}: {exc}")
            return False
        self._pending = pending
        self.set_visible(False)
        self.info = None
        self._preview = None
        self.last_error = None
        self.changed.emit()
        return True

    def cancel(self, force: bool = True) -> bool:
        return self.running and self.session.cancel(force=force)

    def load(self, mesh_root: str | Path) -> bool:
        """プロジェクトの ``mesh/`` から今のメッシュ（完了した一番新しい生成）を読む. 無ければ False."""
        self._root = Path(mesh_root)
        folder = current_generation(self._root)
        if folder is None:
            if self.info is not None:
                self.info = None
                self._preview = None
                self.visible = False
                self.changed.emit()
            return False
        return self._load_folder(folder)

    def _load_folder(self, folder: Path) -> bool:
        """生成フォルダ 1 つ（mesh_summary.json と mesh_key.json）を読む. 読めなければ False."""
        try:
            summary = json.loads((folder / MESH_SUMMARY_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        try:
            key = json.loads((folder / MESH_KEY_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            key = {}
        self.info = MeshInfo(dir=folder, summary=summary, mesh_hash=key.get("meshHash"),
                             geometry_hash=key.get("geometryHash"), created_at=str(key.get("createdAt") or ""))
        self._preview = None
        self.changed.emit()
        return True

    def reset(self) -> None:
        """プロジェクトを新規作成 / 開き直すとき."""
        self.visible = False
        self.info = None
        self._preview = None
        self._root = None
        self.job = None
        self.last_error = None
        self.changed.emit()

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        visible = bool(visible) and self.info is not None
        if visible == self.visible:
            return
        self.visible = visible
        self.changed.emit()

    def preview_geometry(self) -> Optional[dict]:
        """スケッチ画面に重ねる幾何（文書の単位）: ``{"triangles": [...], "bc_segments": {...}, "interfaces": [...]}``.

        読めなければ None。``ui.sketch_view.MeshPreview(**...)`` にそのまま渡せる。
        """
        if self.info is None:
            return None
        if self._preview is None:
            try:
                with np.load(self.info.dir / MESH_PREVIEW_FILE) as npz:
                    arrays = {k: npz[k] for k in npz.files}
            except (OSError, KeyError, ValueError):
                return None
            units = str(self.info.summary.get("units") or self.store.document.meta.units)
            scale = 1.0 / UNIT_TO_M.get(units, 1.0)
            nodes = np.asarray(arrays["nodes"], dtype=float) * scale
            tri = nodes[np.asarray(arrays["simplices"], dtype=int)]
            triangles = [[tuple(p) for p in t] for t in tri.tolist()]
            bc_segments = {}
            for name in BC_NAMES:
                segs = arrays.get(f"bc_{name}")
                if segs is not None and len(segs):
                    bc_segments[name] = [[tuple(p) for p in s] for s in (np.asarray(segs) * scale).tolist()]
            interfaces = arrays.get("interfaces")
            interface_lines = ([[tuple(p) for p in s] for s in (np.asarray(interfaces) * scale).tolist()]
                               if interfaces is not None and len(interfaces) else [])
            self._preview = {"triangles": triangles, "bc_segments": bc_segments, "interfaces": interface_lines}
        return self._preview

    # ------------------------------------------------------------------
    # イベント
    # ------------------------------------------------------------------

    def _on_job_changed(self, job: JobState) -> None:
        if job is self.job:
            self.changed.emit()

    def _on_job_finished(self, job: JobState) -> None:
        if job is not self.job:
            return
        loaded = False
        if job.status == "done":
            geom, mesh = self._pending or self.current_hashes()
            key = {"geometryHash": geom, "meshHash": mesh,
                   "createdAt": _dt.datetime.now().isoformat(timespec="seconds")}
            try:
                (job.dir / MESH_KEY_FILE).write_text(json.dumps(key, indent=2), encoding="utf-8")
            except OSError as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            else:
                loaded = self._load_folder(job.dir)
            if loaded:
                # 古い生成は消す。使用中で消せなければ残し、次に成功したときにまた消す
                remove_stale(self._root if self._root is not None else job.dir.parent, keep=job.dir)
                self.set_visible(True)                # 生成したらメッシュを見せる
        elif job.status == "error":
            self.last_error = job.error
        if not loaded and self._root is not None:
            self.load(self._root)                     # 失敗・中止: 直前のメッシュ（あれば）に戻す
        self._pending = None
        self.changed.emit()

    def _on_document_changed(self) -> None:
        if self.info is None:
            return
        if self.visible and self.state() == "geometry":
            self.set_visible(False)                  # 形状が変わったら重ねない
        else:
            self.changed.emit()
