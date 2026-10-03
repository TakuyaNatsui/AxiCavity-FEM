"""アプリのアイコン（src/axicavity_fem/gui/ui/app_icon.svg）から Windows の .ico を作る.

    python packaging/make_ico.py <out.ico>

各サイズ（16〜256 px）を Qt の SVG レンダラで PNG にし、PNG 圧縮のエントリを並べた ICO を書く（Windows Vista 以降で読める）。
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def png_bytes(size: int) -> bytes:
    from PySide6 import QtCore

    from axicavity_fem.gui.ui.icon_set import APP_ICON_FILE, render_svg

    image = render_svg(APP_ICON_FILE, size)
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def write_ico(out: Path) -> Path:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    _app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(sys.argv[:1])  # noqa: F841 — 描画に必要
    images = [(size, png_bytes(size)) for size in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, png in images:
        dim = 0 if size >= 256 else size                  # 256 は 0 と書く
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + entries + data)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "app.ico")
    print(f"書き出し: {write_ico(target)}")
