"""Windows 版 AxiCavity-FEM.exe（Nuitka standalone）の main スクリプト.

EXE のときだけ同梱物の場所を環境変数で教えてから、1 つの入口 :func:`axicavity_fem.gui.launcher.main` を呼ぶ
（GUI・コマンドライン・GUI の子プロセスがすべてこの EXE）。開発環境では ``python -m axicavity_fem.gui.launcher`` と同じ。
"""

from __future__ import annotations

import os
import sys


def _prepare_environment() -> None:
    if "__compiled__" not in globals():
        return
    # Nuitka standalone では sys.prefix が dist フォルダ（sys.executable は存在しない python.exe を指す）
    here = sys.prefix
    # Intel MKL のスレッド層は TBB（同梱は mkl_tbb_thread + tbb12。intel-openmp は別のライセンスなので入れない）
    os.environ.setdefault("MKL_THREADING_LAYER", "TBB")
    # pypardiso: mkl_rt の既定の探し方は sys.prefix/Lib*/**（= <dist>/Library/bin）だが、直接指定しておく
    if "PYPARDISO_MKL_RT" not in os.environ:
        mkl_dir = os.path.join(here, "Library", "bin")
        if os.path.isdir(mkl_dir):
            for name in sorted(os.listdir(mkl_dir)):
                if name.startswith("mkl_rt") and name.endswith(".dll"):
                    os.environ["PYPARDISO_MKL_RT"] = os.path.join(mkl_dir, name)
                    break
    # メッシャ（gmsh を含む GPL の別プログラム）: <dist>/mesher/
    os.environ.setdefault("AXICAVITY_MESHER_DIR", os.path.join(here, "mesher"))
    os.environ.setdefault("QT_API", "pyside6")


if __name__ == "__main__":
    _prepare_environment()
    from axicavity_fem.gui.launcher import main

    sys.exit(main())
