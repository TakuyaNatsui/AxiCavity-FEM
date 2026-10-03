""".msh の純 Python リーダと、gmsh の読込 API の互換モジュール（Windows 版 EXE で gmsh の代わりに使う）.

- :mod:`.reader`: MSH 4.1 / 2.2（ASCII / binary）のリーダ
- :mod:`.gmsh_api`: ``gmsh`` モジュールのうちコアの ``.msh`` 読込が使う関数だけ
- :func:`install_as_gmsh`: ``import gmsh`` がこの互換モジュールを返すようにする（EXE の起動時。開発環境では使わない）
"""

from __future__ import annotations

import sys

from . import gmsh_api
from .reader import MshData, read_msh


def install_as_gmsh() -> None:
    """``sys.modules["gmsh"]`` に読込専用の互換モジュールを置く（本物の gmsh が読み込み済みなら何もしない）."""
    current = sys.modules.get("gmsh")
    if current is None or current is gmsh_api:
        sys.modules["gmsh"] = gmsh_api


def is_installed() -> bool:
    return sys.modules.get("gmsh") is gmsh_api


__all__ = ["MshData", "gmsh_api", "install_as_gmsh", "is_installed", "read_msh"]
