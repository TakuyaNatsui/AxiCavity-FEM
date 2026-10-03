"""メッシュタブの生成物の置き場所（プロジェクトの ``<名前>.axiproj.data/mesh/``。EM-CAD-py ``em/mesh_folder.py`` の転用）.

生成のたびに**新しいサブフォルダ**（``mesh/20260925-161530/``）に作り、成功したら古いサブフォルダを消す
（消せなければ残して、次に成功したときにまた消す）。同じ名前のフォルダを消して作り直す方式だと、Windows では
他のプロセス（OneDrive の同期、エクスプローラー、ウイルス対策、終了直前の子プロセス …）がファイルを開いているだけで
PermissionError になる。

- 今のメッシュ = 完了の印（:data:`MESH_KEY_FILE`、GUI が成功時に書く）と要約がある一番新しいサブフォルダ。
  失敗・中止した生成（印が無い）は無視されるので、直前のメッシュがそのまま今のメッシュになる
- 消すのはこのモジュールが作った名前のサブフォルダだけ（利用者が置いた物には触らない）

純 Python（Qt 非依存）。
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
from pathlib import Path
from typing import Optional

from .pipeline import MESH_SUMMARY_FILE

MESH_KEY_FILE = "mesh_key.json"
_GENERATION_RE = re.compile(r"^\d{8}-\d{6}(-\d+)?$")


def _sort_key(folder: Path) -> tuple[str, int]:
    """日時（名前の先頭 15 文字）、同じ秒の中は -2, -3 … の番号の順（-10 が -9 より後になるよう数で比べる）."""
    suffix = folder.name[16:]
    return folder.name[:15], int(suffix) if suffix else 1


def new_generation_dir(root: str | Path, now: Optional[_dt.datetime] = None) -> Path:
    """``root`` の下に新しい空のサブフォルダ（日時の名前。同じ秒なら ``-2`` …）を作って返す."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    for i in range(1, 1000):
        folder = root / (stamp if i == 1 else f"{stamp}-{i}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise OSError(f"メッシュの出力先を作れません: {root}")


def generations(root: str | Path) -> list[Path]:
    """生成のサブフォルダ（古い順）."""
    root = Path(root)
    if not root.is_dir():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and _GENERATION_RE.match(p.name)]
    return sorted(folders, key=_sort_key)


def is_complete(folder: str | Path) -> bool:
    """成功した生成か（完了の印と要約がある）."""
    folder = Path(folder)
    return (folder / MESH_KEY_FILE).is_file() and (folder / MESH_SUMMARY_FILE).is_file()


def current_generation(root: str | Path) -> Optional[Path]:
    """今のメッシュのフォルダ: 完了した一番新しいサブフォルダ。無ければ None."""
    for folder in reversed(generations(root)):
        if is_complete(folder):
            return folder
    return None


def remove_stale(root: str | Path, keep: str | Path) -> list[Path]:
    """``keep`` 以外の生成フォルダを消す。消せなかったものを返す（例外は出さない。次に成功したときにまた消す）."""
    keep = Path(keep)
    failed: list[Path] = []
    for folder in generations(root):
        if folder == keep:
            continue
        try:
            shutil.rmtree(folder)
        except OSError:
            failed.append(folder)
    return failed
