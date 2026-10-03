"""gmsh の読込 API の互換モジュール（mshlite、Windows 版 EXE で gmsh の代わりに使う）が gmsh と同じ結果を返すこと.

計算コアの ``shared/mesh_loader.load_mesh`` を、本物の gmsh と mshlite の両方で読ませて比べる:
要素（= 節点の並びと全体行列の番号）・Physical グループ・要素の領域 ID が完全一致し、節点座標が最下位ビットの
違いを除いて一致すること（gmsh の Windows 版は数値の読み取りで最近接に丸めないことがある）、固有周波数が一致すること。MSH 4.1（ASCII / binary）と 2.2（ASCII / binary）を扱う。
"""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from axicavity_fem.gui.mshlite import gmsh_api, read_msh  # noqa: E402
from axicavity_fem.shared import gmsh_export_occ, mesh_loader  # noqa: E402

from conftest import SAMPLES  # noqa: E402

ALL_SAMPLES = sorted(p.stem for p in SAMPLES.glob("*.gmshproj"))


def _load(path, order, lite: bool, monkeypatch):
    if lite:
        monkeypatch.setattr(mesh_loader, "gmsh", gmsh_api)
    try:
        return mesh_loader.load_mesh(path, element_order=order)
    finally:
        monkeypatch.setattr(mesh_loader, "gmsh", gmsh)


def _assert_same(a, b):
    # 座標: gmsh（Windows 版の C ランタイム）の数値の読み取りは最近接に丸めないことがあり、数か所で最下位 1 ビット違う。
    # 並び・要素・Physical は完全一致
    assert a.nodes.shape == b.nodes.shape
    assert np.allclose(a.nodes, b.nodes, rtol=4 * np.finfo(float).eps, atol=0)
    assert np.array_equal(a.elements, b.elements)
    assert a.element_order == b.element_order
    assert a.physical_groups.keys() == b.physical_groups.keys()
    for name in a.physical_groups:
        assert np.array_equal(a.physical_groups[name], b.physical_groups[name]), name
    if a.element_region_ids is None:
        assert b.element_region_ids is None and b.region_names is None
    else:
        assert np.array_equal(a.element_region_ids, b.element_region_ids)
        assert a.region_names == b.region_names


def _convert(src, dst, version: float, binary: bool):
    """gmsh で読み直して別の形式で書く."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(src))
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
        gmsh.write(str(dst))
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name", ALL_SAMPLES)
@pytest.mark.parametrize("order", [1, 2])
def test_load_mesh_identical_to_gmsh(name, order, sample_mesh, monkeypatch):
    path = sample_mesh(name, order)
    _assert_same(_load(path, order, False, monkeypatch), _load(path, order, True, monkeypatch))


@pytest.mark.parametrize("name", ["s-band_1cell", "diel_test1", "acc2_pipe_flat"])
def test_binary_msh41_identical_to_gmsh(name, sample_mesh, monkeypatch, tmp_path):
    src = sample_mesh(name, 2)
    path = tmp_path / f"{name}_bin.msh"
    _convert(src, path, 4.1, binary=True)
    assert read_msh(path).version == "4.1"
    _assert_same(_load(path, 2, False, monkeypatch), _load(path, 2, True, monkeypatch))


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("name", ["cylinder100mm", "diel_test1"])
def test_msh22_identical_to_gmsh(name, binary, sample_mesh, monkeypatch, tmp_path):
    """MSH 2.2（旧形式）も gmsh と同じ（節点の所属は gmsh と同じ規則で決める）."""
    src = sample_mesh(name, 2)
    path = tmp_path / f"{name}_v22.msh"
    _convert(src, path, 2.2, binary=binary)
    assert read_msh(path).version == "2.2"
    _assert_same(_load(path, 2, False, monkeypatch), _load(path, 2, True, monkeypatch))


@pytest.mark.parametrize("name", ["s-band_1cell", "diel_test1"])
def test_tm0_and_hom_frequencies_identical(name, sample_mesh, monkeypatch):
    from axicavity_fem.fem_hom.solver import solve_hom
    from axicavity_fem.fem_tm0.solver import solve_tm0_standing

    path = sample_mesh(name, 2)
    ref = solve_tm0_standing(_load(path, 2, False, monkeypatch), num_modes=4).frequencies
    lite = solve_tm0_standing(_load(path, 2, True, monkeypatch), num_modes=4).frequencies
    assert np.allclose(ref, lite, rtol=1e-12, atol=0)
    f_ref = solve_hom(str(path), 1, element_order=2, num_modes=3).normal.frequencies
    monkeypatch.setattr(mesh_loader, "gmsh", gmsh_api)
    f_lite = solve_hom(str(path), 1, element_order=2, num_modes=3).normal.frequencies
    assert np.allclose(np.sort(f_ref), np.sort(f_lite), rtol=1e-12, atol=0)


def test_inspect_physical_groups_identical(sample_mesh, monkeypatch):
    path = sample_mesh("diel_test1", 2)
    ref = gmsh_export_occ.inspect_msh_physical_groups(path)
    monkeypatch.setattr(gmsh_export_occ, "gmsh", gmsh_api)
    assert gmsh_export_occ.inspect_msh_physical_groups(path) == ref


def test_generation_api_is_not_available():
    with pytest.raises(NotImplementedError, match="メッシャ"):
        gmsh_api.model.occ.addPoint(0, 0, 0)
    with pytest.raises(NotImplementedError, match="メッシャ"):
        gmsh_api.write("x.msh")
    assert not hasattr(gmsh_api, "__path__")             # import の問い合わせは普通の AttributeError
