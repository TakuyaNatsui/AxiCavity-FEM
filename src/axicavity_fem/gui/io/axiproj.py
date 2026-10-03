"""プロジェクトファイル ``.axiproj``（version 1）とデータフォルダの操作（純 Python。EM-CAD-py ``io/cv3dproj.py`` の転用）.

「マニフェスト + 同名のデータフォルダ」方式で、形状（スケッチ・材料・境界条件・設定）と解析結果の履歴を
1 つのプロジェクトで管理する（計画 §7.3、2026-09-25 決定）::

    <名前>.axiproj                 マニフェスト（JSON）。ユーザーが開くのはこれ
    <名前>.axiproj.data/
        command.log                CLI 相当のコマンドの記録（ver2.3 と同じ書式。M7）
        mesh/<日時>/               メッシュタブで生成したメッシュ（model.msh, model.materials.json,
                                   mesh_preview.npz, mesh_summary.json, mesh_key.json, log.txt。M6）
        results/<NNNN-kind>/       解析 1 回分（job.json, geometry.json, model.msh, model.materials.json,
                                   model_{SW|TW}_{TM0|HOM}.h5/.txt, *_processed.h5/.txt, *_report/, exports/,
                                   log.txt）+ result.json（番号・名前・日時・ハッシュ・状態・要約。M7）
        exports/                   書き出しの既定の出力先（M8）

マニフェスト（version 1）::

    {"app": "AxiCavity-FEM", "version": 1, "generator": "AxiCavity-FEM v3 GUI 3.0.0", "saved": "…",
     "document": { … core.serialize の JSON（schemaVersion 1）… }, "activeResult": "0002-tm0-sw", "ui": { … }}

- 結果の一覧はマニフェストに持たず、``results/`` を走査して作る（フォルダが正本。クラッシュしても結果を失わない）。
  名前・状態・ハッシュ・要約は各結果の ``result.json`` に書く。
- 結果と現在のモデルの対応は形状ハッシュ / メッシュハッシュ / モデルハッシュ（``core.hashes``）で判定する
  （開き直した後も「形状変更前の結果」が分かる）。
- 旧 ``.gmshproj``（ver2.3 まで）はプロジェクトではなく形状の取り込み / 書き出し（``core.legacy``）。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.document import AnalysisSettings, AxiDocument
from ..core.serialize import document_from_dict, document_to_dict

PROJECT_SUFFIX = ".axiproj"
PROJECT_APP = "AxiCavity-FEM"
PROJECT_VERSION = 1
DATA_SUFFIX = ".data"
RESULTS_DIR = "results"
MESH_DIR = "mesh"
EXPORTS_DIR = "exports"
COMMAND_LOG = "command.log"
RESULT_META_FILE = "result.json"
RESULT_KINDS = ("tm0-sw", "tm0-tw", "hom-sw", "hom-tw")
# 生の結果ファイル（ver2.3 の CLI / wx GUI と同じ名前。post は *_processed.h5、レポートは *_processed_report/）
RAW_FILES = {"tm0-sw": "model_SW_TM0.h5", "tm0-tw": "model_TW_TM0.h5",
             "hom-sw": "model_SW_HOM.h5", "hom-tw": "model_TW_HOM.h5"}
FILE_FILTER = "AxiCavity-FEM project (*.axiproj)"

_RESULT_DIR_RE = re.compile(r"^(\d{4,})-([a-z0-9-]+)$")


class ProjectFormatError(ValueError):
    """プロジェクトファイルとして読めない（壊れている・別アプリ・未対応バージョン）."""


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def result_kind(analysis: AnalysisSettings) -> str:
    """解析設定 → 結果の種類（``tm0-sw`` など）."""
    return f"{analysis.type}-{'sw' if analysis.wave == 'standing' else 'tw'}"


def raw_file_name(kind: str) -> str:
    return RAW_FILES.get(kind, "model_SW_TM0.h5")


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------

def normalize_project_path(path: str | Path) -> Path:
    """拡張子 .axiproj が無ければ付ける."""
    path = Path(path)
    return path if path.name.endswith(PROJECT_SUFFIX) else path.with_name(path.name + PROJECT_SUFFIX)


def data_dir_of(path: str | Path) -> Path:
    """プロジェクトファイルに対応するデータフォルダ（``<名前>.axiproj.data``）."""
    return Path(str(normalize_project_path(path)) + DATA_SUFFIX)


def results_dir_of(path: str | Path) -> Path:
    return data_dir_of(path) / RESULTS_DIR


def mesh_dir_of(path: str | Path) -> Path:
    return data_dir_of(path) / MESH_DIR


def exports_dir_of(path: str | Path) -> Path:
    return data_dir_of(path) / EXPORTS_DIR


def project_name(path: str | Path) -> str:
    """表示用の名前（拡張子を除いたファイル名）."""
    name = Path(path).name
    return name[: -len(PROJECT_SUFFIX)] if name.endswith(PROJECT_SUFFIX) else Path(path).stem


def sanitize_file_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return cleaned or "cavity"


def suggested_file_name(doc: AxiDocument) -> str:
    return sanitize_file_name(doc.meta.name) + PROJECT_SUFFIX


# ---------------------------------------------------------------------------
# マニフェスト
# ---------------------------------------------------------------------------

@dataclass
class ProjectFile:
    """マニフェストの中身（結果の一覧は含まない → :func:`scan_results`）."""

    document: AxiDocument
    active_result: Optional[str] = None
    ui: dict = field(default_factory=dict)
    saved: str = ""
    generator: str = ""


def _atomic_write_text(path: Path, text: str) -> None:
    """一時ファイルに書いてから置き換える（書き込み中に落ちても元のファイルを壊さない）.

    OneDrive などが一瞬ロックしていると ``os.replace`` が PermissionError になるので 1 回だけ待って再試行する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except PermissionError:
        time.sleep(0.3)
        os.replace(tmp, path)


def write_project(path: str | Path, project: ProjectFile) -> Path:
    """マニフェストを書き、データフォルダを用意する。書いたパス（拡張子付き）を返す."""
    path = normalize_project_path(path)
    data_dir_of(path).mkdir(parents=True, exist_ok=True)
    project.saved = _now()
    manifest = {
        "app": PROJECT_APP,
        "version": PROJECT_VERSION,
        "generator": project.generator,
        "saved": project.saved,
        "document": document_to_dict(project.document),
        "activeResult": project.active_result,
        "ui": dict(project.ui),
    }
    _atomic_write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"プロジェクトファイルを読めません: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("app") != PROJECT_APP:
        raise ProjectFormatError("AxiCavity-FEM のプロジェクトファイル（.axiproj）ではありません。")
    version = manifest.get("version")
    if version != PROJECT_VERSION:
        raise ProjectFormatError(f"未対応のプロジェクトバージョン {version}（対応: {PROJECT_VERSION}）。")
    if not isinstance(manifest.get("document"), dict):
        raise ProjectFormatError("プロジェクトに形状（document）がありません。")
    return manifest


def read_project(path: str | Path) -> ProjectFile:
    """マニフェストを読む。形式が違えば :class:`ProjectFormatError`."""
    manifest = _read_manifest(Path(path))
    try:
        document = document_from_dict(manifest["document"])
    except ValueError as exc:
        raise ProjectFormatError(f"形状を読めません: {exc}") from exc
    return ProjectFile(
        document=document,
        active_result=manifest.get("activeResult"),
        ui=dict(manifest.get("ui") or {}),
        saved=str(manifest.get("saved") or ""),
        generator=str(manifest.get("generator") or ""),
    )


def update_manifest(path: str | Path, **patch) -> None:
    """マニフェストの一部（activeResult / ui など）だけを書き換える（形状は触らない）."""
    path = normalize_project_path(path)
    manifest = _read_manifest(path)
    manifest.update(patch)
    _atomic_write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 結果（results/<NNNN-kind>/）
# ---------------------------------------------------------------------------

@dataclass
class ResultEntry:
    """プロジェクト内の解析結果 1 件（ジョブフォルダ）."""

    id: str                        # フォルダ名（"0003-tm0-sw"）
    dir: Path
    number: int
    kind: str                      # tm0-sw / tm0-tw / hom-sw / hom-tw
    label: str = ""                # ユーザーが付けた名前（空なら既定の名前で表示）
    created_at: str = ""
    finished_at: str = ""
    status: str = "interrupted"    # running / done / error / cancelled / interrupted
    error: str = ""
    geometry_hash: Optional[str] = None
    mesh_hash: Optional[str] = None
    model_hash: Optional[str] = None
    summary: Optional[dict] = None  # result.json の summary（modes の一覧。完了時のみ）
    files: dict = field(default_factory=dict)   # {"raw": "model_SW_TM0.h5", "processed": …, "report": …}

    def file(self, key: str) -> Optional[Path]:
        """result.json の files にある相対パスを絶対パスに（無ければ None）."""
        name = self.files.get(key)
        return self.dir / name if name else None

    @property
    def has_output(self) -> bool:
        """生の結果ファイル（h5）があるか."""
        raw = self.file("raw") or (self.dir / raw_file_name(self.kind))
        return raw.exists()


def read_result_meta(result_dir: str | Path) -> dict:
    path = Path(result_dir) / RESULT_META_FILE
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def write_result_meta(result_dir: str | Path, **fields) -> dict:
    """result.json に項目を書き足す（既存の項目は残す）. 書いた後の辞書を返す."""
    result_dir = Path(result_dir)
    meta = read_result_meta(result_dir)
    meta.update(fields)
    _atomic_write_text(result_dir / RESULT_META_FILE, json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def _load_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_result(result_dir: str | Path) -> Optional[ResultEntry]:
    """結果フォルダ 1 つを読む。job.json が無いフォルダは None."""
    result_dir = Path(result_dir)
    job = _load_json(result_dir / "job.json")
    if job is None:
        return None
    meta = read_result_meta(result_dir)
    m = _RESULT_DIR_RE.match(result_dir.name)
    number = int(meta.get("number") or (m.group(1) if m else 0))
    kind = str(meta.get("kind") or job.get("kind") or (m.group(2) if m else "tm0-sw"))
    files = meta.get("files") if isinstance(meta.get("files"), dict) else {}
    entry = ResultEntry(
        id=result_dir.name, dir=result_dir, number=number, kind=kind,
        label=str(meta.get("label") or ""),
        created_at=str(meta.get("createdAt") or ""),
        finished_at=str(meta.get("finishedAt") or ""),
        error=str(meta.get("error") or ""),
        geometry_hash=meta.get("geometryHash"),
        mesh_hash=meta.get("meshHash"),
        model_hash=meta.get("modelHash"),
        files=dict(files),
    )
    if not entry.created_at:
        entry.created_at = _dt.datetime.fromtimestamp(
            (result_dir / "job.json").stat().st_mtime).isoformat(timespec="seconds")
    status = str(meta.get("status") or "")
    if status in ("done", "error", "cancelled"):
        entry.status = status
    elif entry.has_output:
        # 完了後に GUI が落ちて result.json が "running" のまま残っても、結果があれば完了とみなす
        entry.status = "done"
    elif status == "running":
        entry.status = "running"
    else:
        entry.status = "interrupted"
    if entry.status == "done" or entry.has_output:
        summary = meta.get("summary")
        entry.summary = summary if isinstance(summary, dict) else None
    return entry


def scan_results(data_dir: str | Path) -> list[ResultEntry]:
    """``results/`` の全結果を番号の新しい順に返す."""
    root = Path(data_dir) / RESULTS_DIR
    if not root.is_dir():
        return []
    entries = [e for d in root.iterdir() if d.is_dir() and (e := read_result(d)) is not None]
    return sorted(entries, key=lambda e: (e.number, e.created_at, e.id), reverse=True)


def next_result_id(data_dir: str | Path, kind: str) -> str:
    """次の結果フォルダ名（``0004-tm0-sw``）. 番号は既存の最大 + 1."""
    root = Path(data_dir) / RESULTS_DIR
    numbers = [0]
    if root.is_dir():
        for d in root.iterdir():
            m = _RESULT_DIR_RE.match(d.name)
            if m:
                numbers.append(int(m.group(1)))
    return f"{max(numbers) + 1:04d}-{kind}"


def delete_result(entry: ResultEntry) -> None:
    """結果フォルダを削除する（元に戻せない。確認は呼び出し側で）."""
    shutil.rmtree(entry.dir)


def copy_data_dir(src_project: str | Path, dst_project: str | Path) -> None:
    """「名前を付けて保存」: データフォルダ（メッシュ・結果・書き出し）を新しい名前の場所へ複製する."""
    src, dst = data_dir_of(src_project), data_dir_of(dst_project)
    if not src.is_dir() or src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def append_command_log(data_dir: str | Path, text: str) -> Path:
    """``command.log`` に 1 行追記する（ver2.3 と同じく日時付き。UTF-8）."""
    path = Path(data_dir) / COMMAND_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
    return path
