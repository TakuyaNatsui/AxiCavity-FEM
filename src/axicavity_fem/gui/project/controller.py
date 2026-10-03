"""プロジェクト（.axiproj version 1）の状態と操作（Qt 依存、描画なし。EM-CAD-py ``project/controller.py`` の転用）.

- 保存 / 開く / 名前を付けて保存。未保存の変更は**モデルハッシュ**（``core.hashes.model_hash``）で判定する
  （Undo で保存時の状態に戻れば未保存でなくなる）
- 旧 ``.gmshproj`` と Superfish ``.af`` は「取り込み」= 未保存の新規プロジェクトとして開く（保存すると .axiproj）
- 解析の結果は ``results/<NNNN-kind>/`` に**履歴として全部残す**。メッシュ生成・解析実行の直前にプロジェクトを
  **自動保存**する（結果とそれを作ったモデルを必ず対にするため。未保存の新規プロジェクトでは実行前に保存先を
  決めてもらう — UI 側）
- 結果の一覧は ``results/`` の走査で作り（フォルダが正本）、名前・状態・ハッシュ・要約は各 ``result.json``

UI（MainWindow）は :attr:`changed` シグナルでタイトル・ツリーを更新し、ファイルダイアログや確認は UI 側で出す。
ここは例外（:class:`~axicavity_fem.gui.io.axiproj.ProjectFormatError` / OSError / ValueError）で失敗を返す。
メッシュ（M6）・解析ジョブ（M7）・結果の表示（M8）は、それぞれのコントローラがここの ``prepare_*`` /
``register_submitted`` / ``finish_result`` / ``set_running`` を呼ぶ。
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from ..app.store import DocumentStore
from ..core.hashes import geometry_hash, mesh_hash, model_hash
from ..core.legacy import GMSHPROJ_SUFFIX, SUPERFISH_SUFFIX, load_gmshproj, load_superfish
from ..io.axiproj import (
    EXPORTS_DIR,
    MESH_DIR,
    PROJECT_SUFFIX,
    RESULTS_DIR,
    ProjectFile,
    ProjectFormatError,
    ResultEntry,
    append_command_log,
    copy_data_dir,
    data_dir_of,
    delete_result,
    next_result_id,
    normalize_project_path,
    project_name,
    read_project,
    result_kind,
    scan_results,
    update_manifest,
    write_project,
    write_result_meta,
)

IMPORT_SUFFIXES = (GMSHPROJ_SUFFIX, SUPERFISH_SUFFIX)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class ProjectController(QtCore.QObject):
    """開いているプロジェクト 1 つ分の状態.

    Signals:
        changed(): パス・未保存の状態・結果の一覧・表示中の結果が変わった。
    """

    changed = QtCore.Signal()

    def __init__(self, store: DocumentStore, generator: str = "", parent=None):
        super().__init__(parent)
        self.store = store
        self.generator = generator
        self.path: Optional[Path] = None
        self.results: list[ResultEntry] = []
        self.active_result_id: Optional[str] = None
        self.ui_state: dict = {}
        self.running_dir: Optional[Path] = None      # 実行中の結果フォルダ（M7 が set_running で知らせる）
        self._hash_key = None
        self._hashes = ("", "", "")
        self._saved_hash: Optional[str] = None
        self._dirty = False
        self._mark_saved()
        store.document_changed.connect(self._on_document_changed)
        store.settings_changed.connect(lambda _s: self._on_document_changed())

    # ------------------------------------------------------------------
    # 問い合わせ
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> Optional[Path]:
        return data_dir_of(self.path) if self.path is not None else None

    @property
    def mesh_dir(self) -> Optional[Path]:
        return self.data_dir / MESH_DIR if self.data_dir is not None else None

    @property
    def results_dir(self) -> Optional[Path]:
        return self.data_dir / RESULTS_DIR if self.data_dir is not None else None

    @property
    def exports_dir(self) -> Optional[Path]:
        return self.data_dir / EXPORTS_DIR if self.data_dir is not None else None

    @property
    def name(self) -> str:
        return project_name(self.path) if self.path is not None else self.store.document.meta.name

    def current_hashes(self) -> tuple[str, str, str]:
        """現在のモデルの (形状ハッシュ, メッシュハッシュ, モデルハッシュ). 文書の版ごとにキャッシュする."""
        store = self.store
        key = (id(store.document), store.sketch_version)
        if key != self._hash_key:
            doc = store.document
            self._hashes = (geometry_hash(doc), mesh_hash(doc), model_hash(doc))
            self._hash_key = key
        return self._hashes

    def is_dirty(self) -> bool:
        return self._dirty

    def result(self, result_id: Optional[str]) -> Optional[ResultEntry]:
        return next((e for e in self.results if e.id == result_id), None)

    def result_relation(self, entry: ResultEntry) -> str:
        """結果と現在のモデルの関係: "" = 同じ / "geometry" = 形状変更前 / "mesh" = メッシュ変更前 /
        "settings" = 設定変更前 / "unknown" = 不明（ハッシュ無し）."""
        if entry.geometry_hash is None:
            return "unknown"
        geom, mesh, model = self.current_hashes()
        if entry.geometry_hash != geom:
            return "geometry"
        if entry.mesh_hash is not None and entry.mesh_hash != mesh:
            return "mesh"
        if entry.model_hash is not None and entry.model_hash != model:
            return "settings"
        return ""

    # ------------------------------------------------------------------
    # 新規 / 開く / 保存
    # ------------------------------------------------------------------

    def new(self, name: str = "Untitled", units: str = "mm") -> None:
        """新しい（未保存の）プロジェクトにする."""
        self.path = None
        self.results = []
        self.active_result_id = None
        self.ui_state = {}
        self.running_dir = None
        self.store.new_document(name, units)
        self._mark_saved()
        self.changed.emit()

    def open(self, path: str | Path) -> list[str]:
        """プロジェクト（.axiproj）か形状ファイル（.gmshproj / .af）を開く。取り込みの注意（文字列）を返す.

        .gmshproj / .af は未保存の新規プロジェクトとして開く。失敗は ProjectFormatError / DocumentParseError /
        OSError / ValueError（今のプロジェクトはそのまま）。
        """
        path = Path(path)
        if self.running_dir is not None:
            raise RuntimeError("解析の実行中はプロジェクトを切り替えられません。")
        name = path.name.lower()
        if name.endswith(PROJECT_SUFFIX):
            project = read_project(path)
            self.path = path
            self.ui_state = project.ui
            self.active_result_id = None
            self.store.load_document(project.document)
            self._mark_saved()
            self.refresh_results()
            active = self.result(project.active_result)
            if active is not None and active.status == "done":
                self.active_result_id = active.id
            warnings: list[str] = []
        elif name.endswith(GMSHPROJ_SUFFIX) or name.endswith(SUPERFISH_SUFFIX):
            if name.endswith(GMSHPROJ_SUFFIX):
                document, warnings = load_gmshproj(path)
            else:
                document, warnings = load_superfish(path, self.store.document.meta.units)
            document.meta.name = path.stem
            self.path = None
            self.results = []
            self.active_result_id = None
            self.ui_state = {}
            self.store.load_document(document)
            self._saved_hash = None                  # まだプロジェクトとして保存していない
            self._dirty = True
        else:
            raise ProjectFormatError(f"開けない形式です: {path.name}（.axiproj / .gmshproj / .af）")
        self.changed.emit()
        return list(warnings)

    def save(self) -> Path:
        """上書き保存（パスが無ければ ValueError → UI は save_as を使う）."""
        if self.path is None:
            raise ValueError("保存先が決まっていません（名前を付けて保存）。")
        return self._write(self.path)

    def save_as(self, path: str | Path) -> Path:
        """名前を付けて保存。既存のプロジェクトならデータフォルダ（メッシュ・結果）ごと複製する."""
        if self.running_dir is not None:
            raise RuntimeError("解析の実行中は名前を付けて保存できません。")
        path = normalize_project_path(path)
        old = self.path
        if old is not None and path.resolve() != old.resolve():
            copy_data_dir(old, path)
        self.path = path
        written = self._write(path)
        self.refresh_results()
        return written

    def _write(self, path: Path) -> Path:
        self.store.document.meta.name = project_name(path)
        written = write_project(path, ProjectFile(
            document=self.store.document, active_result=self.active_result_id,
            ui=self.ui_state, generator=self.generator))
        self._mark_saved()
        self.changed.emit()
        return written

    def _mark_saved(self) -> None:
        self._saved_hash = self.current_hashes()[2]
        self._dirty = False

    def _on_document_changed(self) -> None:
        dirty = self._saved_hash is None or self.current_hashes()[2] != self._saved_hash
        if dirty != self._dirty:
            self._dirty = dirty
            self.changed.emit()

    # ------------------------------------------------------------------
    # 解析と結果（M6〜M8 のコントローラが使う）
    # ------------------------------------------------------------------

    def prepare_mesh(self) -> Path:
        """メッシュ生成の直前: プロジェクトを自動保存し、メッシュの置き場所（mesh/）を返す.

        未保存（パス無し）なら ValueError（UI は先に名前を付けて保存させる）。
        """
        self.save()
        return self.mesh_dir

    def prepare_run(self) -> str:
        """解析実行の直前: プロジェクトを自動保存し、次の結果フォルダ名（``0003-tm0-sw``）を返す.

        未保存（パス無し）なら ValueError（UI は先に名前を付けて保存させる）。
        """
        self.save()
        return next_result_id(self.data_dir, result_kind(self.store.document.analysis))

    def register_submitted(self, result_dir: Path, kind: str) -> None:
        """投入したジョブを結果として登録する（result.json: 番号・種類・日時・ハッシュ・状態 running）."""
        if self.results_dir is None or Path(result_dir).parent.resolve() != self.results_dir.resolve():
            return
        geom, mesh, model = self.current_hashes()
        name = Path(result_dir).name
        number = int(name.split("-", 1)[0]) if name[:1].isdigit() else 0
        write_result_meta(result_dir, number=number, kind=kind, createdAt=_now(), status="running",
                          geometryHash=geom, meshHash=mesh, modelHash=model, generator=self.generator)
        self.running_dir = Path(result_dir)
        self._set_active(name)
        self.refresh_results()

    def finish_result(self, result_dir: Path, status: str, error: str = "",
                      summary: Optional[dict] = None, files: Optional[dict] = None) -> None:
        """ジョブの終了を結果に記録する（状態・日時・要約・ファイル名）."""
        fields = dict(status=status, finishedAt=_now(), error=error or "")
        if summary is not None:
            fields["summary"] = summary
        if files is not None:
            fields["files"] = dict(files)
        write_result_meta(result_dir, **fields)
        if self.running_dir is not None and Path(result_dir).resolve() == self.running_dir.resolve():
            self.running_dir = None
        self.refresh_results()

    def set_running(self, result_dir: Optional[Path]) -> None:
        self.running_dir = Path(result_dir) if result_dir is not None else None

    def log_command(self, text: str) -> Optional[Path]:
        """データフォルダの command.log に追記する（未保存なら何もしない）."""
        if self.data_dir is None:
            return None
        return append_command_log(self.data_dir, text)

    def refresh_results(self) -> None:
        """results/ を走査し直す（実行中でない "running" は中断扱い）."""
        self.results = scan_results(self.data_dir) if self.data_dir is not None else []
        for entry in self.results:
            if entry.status == "running" and (self.running_dir is None
                                              or self.running_dir.resolve() != entry.dir.resolve()):
                entry.status = "interrupted"
        if self.active_result_id is not None and self.result(self.active_result_id) is None:
            self.active_result_id = None
        self.changed.emit()

    def select_result(self, result_id: Optional[str]) -> bool:
        """表示する結果を選ぶ（表示そのものは ui/result_model.py の ResultModel）. 実行中の結果は選べない."""
        if result_id is None:
            self._set_active(None)
            self.changed.emit()
            return True
        entry = self.result(result_id)
        if entry is None or entry.status == "running":
            return False
        self._set_active(result_id)
        self.changed.emit()
        return True

    def _set_active(self, result_id: Optional[str]) -> None:
        self.active_result_id = result_id
        if self.path is not None:
            try:
                update_manifest(self.path, activeResult=result_id)
            except (OSError, ProjectFormatError):
                pass                                    # 表示中の記録だけなので失敗しても続ける

    def rename_result(self, result_id: str, label: str) -> None:
        entry = self.result(result_id)
        if entry is None:
            return
        write_result_meta(entry.dir, label=label.strip())
        self.refresh_results()

    def delete_result(self, result_id: str) -> None:
        """結果フォルダを削除する（元に戻せない）。実行中の結果は消せない."""
        entry = self.result(result_id)
        if entry is None:
            return
        if self.running_dir is not None and self.running_dir.resolve() == entry.dir.resolve():
            raise RuntimeError("実行中の結果は削除できません。")
        if self.active_result_id == result_id:
            self._set_active(None)
        delete_result(entry)
        self.refresh_results()
