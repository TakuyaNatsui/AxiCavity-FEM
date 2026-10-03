"""pytest 共通 fixture.

``samples/`` には入力ファイル（``.gmshproj`` など）だけを置く（メッシュや結果は同梱しない）。メッシュが要るテストは
:func:`sample_mesh` でテストの最初に一度だけ作る（gmsh が要る）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session")
def sample_mesh(tmp_path_factory):
    """``sample_mesh(name, order=2)`` → ``samples/<name>.gmshproj`` から作った ``.msh`` のパス（材料 JSON も隣に書く）."""
    pytest.importorskip("gmsh")
    from axicavity_fem.gui.jobs.pipeline import write_materials_json
    from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
    from axicavity_fem.shared.multi_region_model import MultiRegionGeometry

    folder = tmp_path_factory.mktemp("sample_meshes")
    made: dict[tuple[str, int], Path] = {}

    def make(name: str, order: int = 2) -> Path:
        key = (name, int(order))
        if key not in made:
            geom = MultiRegionGeometry.load_json(SAMPLES / f"{name}.gmshproj")
            out = folder / f"{name}_order{order}.msh"
            export_msh_multi_region(geom, out, mesh_order=int(order), verbose=0)
            write_materials_json(geom, out)          # solve が自動で読む材料のサイドカー
            made[key] = out
        return made[key]

    return make
