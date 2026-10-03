"""Windows 版（Nuitka でコンパイルした EXE）として動いているかと、EXE 自身の場所.

- Nuitka は ``sys.frozen`` を設定しない。コンパイルされたモジュールには ``__compiled__`` が入る。
- standalone の ``sys.executable`` は ``<dist>/python.exe`` という**存在しないパス**（ビルド時の Python の名前を
  dist フォルダに付け替えたもの）なので、EXE のパスは OS（``GetModuleFileNameW``）に聞く（EM-CAD-py と同じ）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXE_NAME = "AxiCavity-FEM.exe"


def is_compiled() -> bool:
    """Nuitka でコンパイルされた EXE として動いているか."""
    return "__compiled__" in globals()


def executable_path() -> str:
    """実行中のプログラム（開発なら python、EXE なら AxiCavity-FEM.exe 自身）のパス."""
    if not is_compiled():
        return sys.executable
    if os.name == "nt":
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetModuleFileNameW(None, buf, len(buf)):
            return buf.value
    else:
        try:
            return os.readlink("/proc/self/exe")
        except OSError:
            pass
    return os.path.abspath(sys.argv[0])


def app_dir() -> Path:
    """EXE のフォルダ（開発環境ではリポジトリのルート）."""
    if is_compiled():
        return Path(executable_path()).resolve().parent
    return Path(__file__).resolve().parents[3]
