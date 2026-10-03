"""版と同梱物の情報（``AxiCavity-FEM.exe version`` と、ファイルタブの「バージョン情報」で使う）.

PARDISO（Intel MKL）が使えるか、メッシュ生成がどこで動くか（プロセス内の gmsh / 別プロセスのメッシャ）、
3D 表示（pyvista）があるかを調べる。Qt に依存しない（CLI からも使う）。
"""

from __future__ import annotations

import platform
from importlib import metadata

from . import GUI_VERSION, frozen

GITHUB_URL = "https://github.com/TakuyaNatsui/AxiCavity-FEM"


_MODULE_OF = {"PySide6": "PySide6", "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib", "h5py": "h5py",
              "planegcs": "planegcs", "pypardiso": "pypardiso", "vtk": "vtkmodules", "pyvista": "pyvista"}


def _version(dist: str) -> str:
    """配布物の版（EXE では配布物のメタデータが無いことがあるので、モジュールの ``__version__`` も見る）."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        pass
    module = _MODULE_OF.get(dist)
    if module:
        try:
            import importlib

            mod = importlib.import_module(module)
            return str(getattr(mod, "__version__", "-"))
        except Exception:  # noqa: BLE001
            pass
    return "-"


def pardiso_status() -> tuple[bool, str]:
    """(使えるか, 説明). 小さな連立方程式を実際に解いて MKL が読み込めることを確かめる."""
    try:
        import numpy as np
        import pypardiso
        import scipy.sparse as sp

        x = pypardiso.spsolve(sp.identity(3, format="csr") * 2.0, np.ones(3))
        if not np.allclose(x, 0.5):
            return False, "pypardiso の結果が不正"
        return True, f"pypardiso {_version('pypardiso')}（Intel MKL {_mkl_version(pypardiso)}）"
    except Exception as exc:  # noqa: BLE001 — 未導入・MKL が読めないなど
        return False, f"なし（SuperLU を使用。{type(exc).__name__}）"


def _mkl_version(pypardiso) -> str:
    """読み込まれた MKL の版（EXE には pip の mkl のメタデータが無いので MKL 自身に聞く）."""
    try:
        import ctypes
        import re

        buf = ctypes.create_string_buffer(256)
        pypardiso.ps.libmkl.MKL_Get_Version_String(buf, len(buf))
        match = re.search(r"Version\s+([0-9][0-9.]*)", buf.value.decode("ascii", "replace"))
        if match:
            return match.group(1)
    except Exception:  # noqa: BLE001 — pypardiso の内部が変わったときなど
        pass
    return _version("mkl")


def mesh_status(run_mesher: bool = True) -> str:
    """メッシュ生成の方式（EXE: 別プロセスのメッシャ、開発・pip: プロセス内の gmsh）."""
    from .jobs import meshing

    if meshing.use_mesher_process():
        if not run_mesher:
            try:
                return "メッシャ（別プロセス）: " + " ".join(meshing.mesher_command())
            except meshing.MesherError as exc:
                return f"メッシャ: {exc}"
        return "メッシャ（別プロセス）: " + meshing.mesher_version()
    try:
        import gmsh

        return f"プロセス内の gmsh {gmsh.__version__}"
    except Exception as exc:  # noqa: BLE001
        return f"gmsh なし（{type(exc).__name__}）"


def viz3d_status() -> str:
    try:
        import pyvista
        import pyvistaqt  # noqa: F401

        return f"pyvista {pyvista.__version__} / VTK {_version('vtk')}"
    except Exception as exc:  # noqa: BLE001
        return f"なし（{type(exc).__name__}。pip install \"axicavity-fem[viz3d]\"）"


def collect(run_mesher: bool = True) -> dict[str, str]:
    import axicavity_fem

    try:
        from PySide6 import QtCore

        qt = f"PySide6 {_version('PySide6')} (Qt {QtCore.qVersion()})"
    except Exception:  # noqa: BLE001
        qt = "-"
    ok, pardiso = pardiso_status()
    return {
        "AxiCavity-FEM": f"{GUI_VERSION}" + ("（Windows 版 EXE）" if frozen.is_compiled() else ""),
        "solver core": axicavity_fem.__version__,
        "Python": platform.python_version(),
        "numpy / scipy": f"{_version('numpy')} / {_version('scipy')}",
        "matplotlib / h5py": f"{_version('matplotlib')} / {_version('h5py')}",
        "Qt": qt,
        "planegcs": _version("planegcs"),
        "PARDISO": pardiso,
        "mesh": mesh_status(run_mesher),
        "3D view": viz3d_status(),
        "executable": frozen.executable_path(),
    }


def info_text(run_mesher: bool = True) -> str:
    info = collect(run_mesher)
    width = max(len(k) for k in info)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in info.items())
