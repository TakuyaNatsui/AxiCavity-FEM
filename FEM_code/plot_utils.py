"""FEM_code/plot_utils.py

後方互換用ラッパ。
実体はプロジェクト直下の ``plot_common.py`` に移設済み。
既存の ``from FEM_code.plot_utils import FEMPlotter`` / ``import plot_utils``
経由の呼び出しはそのまま動作する。
"""

import os
import sys

# プロジェクトルートを import path に追加 (plot_common.py を解決するため)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from plot_common import (  # noqa: E402, F401
    STYLE,
    setup_axes,
    draw_mesh_overlay,
    draw_pec_boundary,
    plot_bipolar_contour,
    plot_quiver_styled,
    BaseFEMPlotter,
    TM0Plotter,
    HOMPlotter,
)

# 後方互換: 旧名 FEMPlotter は TM0Plotter と同一
FEMPlotter = TM0Plotter

__all__ = [
    'FEMPlotter',
    'TM0Plotter',
    'HOMPlotter',
    'BaseFEMPlotter',
    'STYLE',
    'setup_axes',
    'draw_mesh_overlay',
    'draw_pec_boundary',
    'plot_bipolar_contour',
    'plot_quiver_styled',
]
