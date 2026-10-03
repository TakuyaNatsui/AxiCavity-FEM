"""アプリの設定（言語・最近使ったファイル・ウィンドウとドックの配置）の永続化（QSettings）.

既定は OS 標準の場所（Windows はレジストリ HKCU\\Software\\AxiCavity-FEM\\v3。pip 版と Windows 版で共通）。
環境変数 ``AXICAVITY_SETTINGS_DIR`` があればそのフォルダの ``axicavity.ini`` を使う（テストで
ユーザーの設定を汚さないため）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

MAX_RECENT = 8
# ドックの構成を変えたら上げる（古い配置は読まずに初期配置にする。QMainWindow.saveState の版）
LAYOUT_VERSION = 1


def settings() -> QtCore.QSettings:
    folder = os.environ.get("AXICAVITY_SETTINGS_DIR")
    if folder:
        return QtCore.QSettings(str(Path(folder) / "axicavity.ini"), QtCore.QSettings.IniFormat)
    return QtCore.QSettings("AxiCavity-FEM", "v3")


def saved_language() -> Optional[str]:
    value = settings().value("language")
    return str(value) if value in ("ja", "en") else None


def save_language(lang: str) -> None:
    settings().setValue("language", lang)


def recent_projects() -> list[str]:
    value = settings().value("recentProjects") or []
    if isinstance(value, str):                  # 1 件だけのとき QSettings は文字列で返す
        value = [value]
    return [str(v) for v in value if v]


def add_recent_project(path: str | Path) -> list[str]:
    path = str(Path(path).resolve())
    items = [p for p in recent_projects() if os.path.normcase(p) != os.path.normcase(path)]
    items.insert(0, path)
    del items[MAX_RECENT:]
    settings().setValue("recentProjects", items)
    return items


def clear_recent_projects() -> None:
    settings().setValue("recentProjects", [])


def _bytes(value) -> Optional[QtCore.QByteArray]:
    return value if isinstance(value, QtCore.QByteArray) and not value.isEmpty() else None


def load_window_layout() -> tuple[Optional[QtCore.QByteArray], Optional[QtCore.QByteArray]]:
    """前回のメインウィンドウの (saveGeometry, saveState)。版が違う・無ければ None."""
    s = settings()
    try:
        version = int(s.value("layout/version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version != LAYOUT_VERSION:
        return None, None
    return _bytes(s.value("layout/geometry")), _bytes(s.value("layout/state"))


def save_window_layout(geometry: QtCore.QByteArray, state: QtCore.QByteArray) -> None:
    s = settings()
    s.setValue("layout/version", LAYOUT_VERSION)
    s.setValue("layout/geometry", geometry)
    s.setValue("layout/state", state)


def load_window_geometry(name: str) -> Optional[QtCore.QByteArray]:
    """別ウィンドウ（3D 表示など）の前回の saveGeometry（無ければ None）."""
    return _bytes(settings().value(f"windows/{name}"))


def save_window_geometry(name: str, geometry: QtCore.QByteArray) -> None:
    settings().setValue(f"windows/{name}", geometry)


def clear_window_geometry(name: str) -> None:
    settings().remove(f"windows/{name}")
