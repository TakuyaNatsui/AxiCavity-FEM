"""3D 表示（pyvista / pyvistaqt、任意の依存 ``[viz3d]``）が無くても GUI が起動すること.

``pip install -e ".[gui]"`` だけの環境を、別プロセスで pyvista 系の import を止めて再現する
（同じプロセスでやると他のテストの import に影響するため）。3D 表示のボタンは無効になる。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("planegcs")

SCRIPT = r"""
import importlib.abc, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyvista", "pyvistaqt", "vtk", "vtkmodules", "qtpy"):
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, Block())
from PySide6 import QtWidgets
app = QtWidgets.QApplication([])
from axicavity_fem.gui.ui import view3d_window
from axicavity_fem.gui.ui.main_window import MainWindow
win = MainWindow()
print("available", view3d_window.is_available(), "enabled", win.view3d_action.isEnabled())
print("loaded", sorted(m for m in sys.modules if m.split(".")[0] in ("pyvista", "vtk", "vtkmodules")))
win.project._mark_saved()
win.close()
"""


def test_gui_starts_without_pyvista(tmp_path):
    env = dict(os.environ, AXICAVITY_SETTINGS_DIR=str(tmp_path), PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "available False enabled False" in result.stdout
    assert "loaded []" in result.stdout
