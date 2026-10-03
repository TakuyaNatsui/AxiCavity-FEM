"""アプリの状態（DocumentStore）と GUI の入口.

    axicavity-fem-gui [project]          （pyproject の gui-scripts → :func:`main`）
    python -m axicavity_fem.gui [project]   （``gui/__main__.py``）

:func:`main` は PySide6 とメインウィンドウを関数の中で import する（``app.store`` を使う UI モジュールとの
循環 import を避けるため。子プロセスの runner（M6）も GUI を import しない）。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """GUI を起動する（引数に形状 / プロジェクトのパスがあれば開く）."""
    argv = list(sys.argv[1:] if argv is None else argv)
    from PySide6 import QtWidgets

    from ..ui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    _set_identity(app)
    win = MainWindow()
    win.show()
    if argv:
        win.open_path(Path(argv[0]))
    return app.exec()


def _set_identity(app) -> None:
    """アプリ名とアイコン（タスクバーも python ではなくこのアイコンにするため、Windows では AppUserModelID を設定）."""
    from .. import GUI_VERSION
    from ..ui.icon_set import app_icon

    app.setApplicationName("AxiCavity-FEM")
    app.setApplicationVersion(GUI_VERSION)
    app.setWindowIcon(app_icon())
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AxiCavityFEM.GUI")
        except (AttributeError, OSError):
            pass
