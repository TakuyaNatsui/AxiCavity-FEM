"""Windows 版のビルド設定: Nuitka に渡すオプションのうち、抜けると EXE でだけ壊れるもの（Nuitka は実行しない）."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_exe(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_exe_under_test", ROOT / "packaging" / "build_exe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "mkl_dll_patterns", lambda: [])     # MKL の無い環境でもコマンドは組み立てられる
    return module


def _command(module, tmp_path):
    args = argparse.Namespace(version="0.0.0", jobs=1, msvc=None, zig=False, extra=[])
    return module.nuitka_command(args, tmp_path, None)


def test_excluded_modules_behave_as_missing_modules(build_exe, tmp_path):
    """除外したモジュールを ``find_spec`` で問い合わせると Nuitka は既定で例外を出す. pyvista は読み込み時に
    ``find_spec('mypy')`` を呼ぶので、ビルド環境に mypy があると EXE の 3D 表示が丸ごと使えなくなった."""
    assert "--no-deployment-flag=excluded-module-usage" in _command(build_exe, tmp_path)


def test_mypy_stays_out_of_the_build(build_exe):
    """pyvista.typing.mypy_plugin が辿るが、実行には使わない."""
    assert "mypy" in build_exe.NOFOLLOW


def test_core_cli_arguments_are_not_taken_for_self_execution(build_exe, tmp_path):
    assert "--no-deployment-flag=self-execution" in _command(build_exe, tmp_path)


def test_ico_holds_every_size_from_the_app_icon(qapp, tmp_path):
    """EXE のアイコン: 16〜256 px の PNG エントリ（32 px 以下は線を減らした版）."""
    import struct

    from PySide6 import QtGui

    from axicavity_fem.gui.ui.icon_set import render_app_icon
    spec = importlib.util.spec_from_file_location("make_ico_under_test", ROOT / "packaging" / "make_ico.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.write_ico(tmp_path / "app.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind, count) == (0, 1, len(module.SIZES))
    for i, size in enumerate(module.SIZES):
        w, h, _, _, _, _, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        assert (w, h) == ((0, 0) if size == 256 else (size, size))
        image = QtGui.QImage.fromData(data[offset:offset + length], "PNG")
        assert (image.width(), image.height()) == (size, size)
        want = render_app_icon(size)
        assert image.convertToFormat(want.format()) == want
