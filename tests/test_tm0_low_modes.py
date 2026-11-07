"""TM0 の自動シフト c/(4 r_max) より低いモード: 周波数の計算は変えず、見逃しを警告する.

基準: 本数は密行列の固有値（K, M の全スペクトル）で数え、モードは予測周波数を指定した解、または解析解（同軸）と比べる。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

pytest.importorskip("gmsh")

from axicavity_fem.fem_tm0.solver import solve_tm0_standing, solve_tm0_traveling  # noqa: E402
from axicavity_fem.shared.constants import C0  # noqa: E402
from axicavity_fem.shared.eigensolver import count_eigenvalues_below  # noqa: E402
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region  # noqa: E402
from axicavity_fem.shared.material_resolver import build_material_table_from_geometry  # noqa: E402
from axicavity_fem.shared.mesh_loader import load_mesh  # noqa: E402
from axicavity_fem.shared.multi_region_model import Loop, MultiRegionGeometry, Region, Segment  # noqa: E402


def _polygon(points, bcs, h, eps_r=1.0):
    n = len(points)
    segs = [Segment(id=i, type="line", point_indices=[i, (i + 1) % n], bc_name=bc) for i, bc in enumerate(bcs)]
    tag = "vacuum" if eps_r == 1.0 else "diel"
    return MultiRegionGeometry(points=list(points), segments=segs, loops=[Loop(id=0, segment_ids=list(range(n)))],
                               regions=[Region(id=0, name="Cell", outer_loop_id=0, material_tag=tag, eps_r=eps_r)], unit="mm", mesh_size=h)


def _mesh(geom, path, order=2):
    export_msh_multi_region(geom, path, mesh_order=order, verbose=0)
    return load_mesh(path, element_order=order)


REENTRANT = ([(0.0, 15.0), (33.0, 15.0), (33.0, 0.0), (35.0, 0.0), (35.0, 40.0), (0.0, 40.0)],
             ["PEC", "PEC", "None", "PEC", "PEC", "PEC"])
PILLBOX = ([(0.0, 0.0), (35.0, 0.0), (35.0, 40.0), (0.0, 40.0)], ["None", "PEC", "PEC", "PEC"])
COAX = ([(0.0, 10.0), (200.0, 10.0), (200.0, 40.0), (0.0, 40.0)], ["PEC", "PEC", "PEC", "PEC"])


def _solve(mesh, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_tm0_standing(mesh, **kw)
    return res, [str(w.message) for w in caught if "[TM0]" in str(w.message)]


def test_count_matches_the_dense_spectrum():
    """対称な順序の LU の負の対角の数 = shift より小さい固有値の数."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 40))
    K = A @ A.T
    B = rng.standard_normal((40, 40))
    M = B @ B.T + 40.0 * np.eye(40)
    values = scipy.linalg.eigh(K, M, eigvals_only=True)
    Ks, Ms = sp.csr_matrix(K), sp.csr_matrix(M)
    for shift in (0.5 * values[3] + 0.5 * values[4], 0.5 * values[19] + 0.5 * values[20], values[-1] * 1.01):
        assert count_eigenvalues_below(Ks, Ms, shift) == int(np.sum(values < shift))


def test_count_for_a_complex_hermitian_problem():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 30)) + 1j * rng.standard_normal((30, 30))
    K = A @ A.conj().T
    M = np.eye(30, dtype=complex)
    values = np.linalg.eigvalsh(K)
    shift = 0.5 * (values[9] + values[10])
    assert count_eigenvalues_below(sp.csr_matrix(K), sp.csr_matrix(M), shift) == 10


def test_reentrant_cavity_warns_and_the_mode_is_really_there(tmp_path):
    """リエントラント空洞: 自動シフト 1.87 GHz より低い 0.86 GHz を返さない. 警告の案内どおり指定すると求まる."""
    mesh = _mesh(_polygon(*REENTRANT, 2.5), tmp_path / "m.msh")
    res, messages = _solve(mesh, num_modes=2)
    assert res.frequencies[0] > 4.0
    assert len(messages) == 1 and "1 本" in messages[0] and "--target-freq" in messages[0]
    found, again = _solve(mesh, num_modes=2, target_freq_ghz=0.5)
    assert found.frequencies[0] == pytest.approx(0.8636, rel=2e-3)
    assert again == []


def test_warning_does_not_change_the_result(tmp_path):
    """警告の計算は固有値の計算に影響しない（固有値は 2 回目の解と同じ）."""
    mesh = _mesh(_polygon(*REENTRANT, 5.0), tmp_path / "m.msh")
    first, _ = _solve(mesh, num_modes=3)
    second, _ = _solve(mesh, num_modes=3)
    assert first.frequencies == pytest.approx(second.frequencies, rel=1e-9, abs=0)


@pytest.mark.parametrize("order", [1, 2])
def test_shorted_coaxial_line_counts_real_modes_but_not_the_static_one(tmp_path, order):
    """同軸の TEM: 解析解 0.7495 / 1.499 / 2.2484 GHz. 2 本（0.75, 1.50）が抜ける. 静的な解 H ∝ 1/r は数えない.

    静的な解の離散の周波数は 0 にならない（1 次要素で 0.46 GHz）ので、周波数の閾値では判定できない。
    """
    mesh = _mesh(_polygon(*COAX, 5.0), tmp_path / "m.msh", order=order)
    res, messages = _solve(mesh, num_modes=3)
    assert res.frequencies[0] == pytest.approx(C0 * 3 / (2 * 0.2) / 1e9, rel=3e-3 if order == 2 else 0.05)
    assert len([m for m in messages if "見逃して" in m]) == 1 and "2 本" in messages[0]


def test_many_missed_modes_are_counted_exactly(tmp_path):
    """L = 2000 mm の同軸: 求めた最低より下の解析解 n·c/2L の本数と一致する（周波数の閾値だと 22 本になった）."""
    points = [(0.0, 10.0), (2000.0, 10.0), (2000.0, 40.0), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["PEC"] * 4, 4.0), tmp_path / "m.msh")
    res, messages = _solve(mesh, num_modes=1)
    expected = int(round(res.frequencies[0] * 1e9 / (C0 / (2 * 2.0)))) - 1       # 求めた最低が n 番目なら、下に n − 1 本
    assert f"{expected} 本" in messages[0]


def test_real_mode_near_zero_is_not_taken_for_the_static_one(tmp_path):
    """片端 M-short の長い同軸には静的な解が無い: 最低の (2n−1)c/4L も数える（周波数の閾値だと 2 本少なかった）."""
    points = [(0.0, 10.0), (2000.0, 10.0), (2000.0, 40.0), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["PEC", "M-short", "PEC", "PEC"], 4.0), tmp_path / "m.msh")
    res, messages = _solve(mesh, num_modes=1)
    expected = int(res.frequencies[0] * 1e9 / (C0 / (2 * 2.0)) + 0.5) - 1          # (2n − 1)c/4L
    assert f"{expected} 本" in messages[0]


@pytest.mark.parametrize("order, h", [(1, 5.0), (1, 2.5), (2, 5.0)])
def test_coaxial_cavity_without_lower_modes_does_not_warn(tmp_path, order, h):
    """短い同軸 L = 30 mm: 解析解の最低は 4.89 GHz. 静的な解だけが下にある（1 次要素で 0.46 GHz）ので見逃しではない."""
    points = [(0.0, 10.0), (30.0, 10.0), (30.0, 40.0), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["PEC"] * 4, h), tmp_path / "m.msh", order=order)
    _, messages = _solve(mesh, num_modes=1)
    assert not [m for m in messages if "見逃して" in m]


def test_count_matches_the_dense_spectrum_of_a_real_tm0_problem(tmp_path):
    """本物の TM0 の行列（M は特異に近い）でも、数えた本数は全固有値と一致する."""
    from axicavity_fem.fem_tm0.assembly import assemble_global_matrices
    from axicavity_fem.fem_tm0.boundary import build_dirichlet_nodes, create_transformation_matrix
    from axicavity_fem.shared.boundary_groups import classify_boundaries

    mesh = _mesh(_polygon(*PILLBOX, 8.0), tmp_path / "m.msh")
    K, M = assemble_global_matrices(mesh)
    dirichlet = build_dirichlet_nodes(classify_boundaries(mesh.physical_groups, mesh.nodes), mesh.nodes)
    T, _ = create_transformation_matrix(mesh.num_nodes, dirichlet)
    T = T.real
    Kr, Mr = (T.T @ K @ T).tocsc(), (T.T @ M @ T).tocsc()
    values = np.real(scipy.linalg.eigvals(Kr.toarray(), Mr.toarray()))
    values = np.sort(values[np.isfinite(values) & (values > 0.0)])
    for k in (0, 2, 7):
        shift = 0.5 * (values[k] + values[k + 1])
        assert count_eigenvalues_below(Kr, Mr, shift) == k + 1


def test_counting_does_not_change_the_result_or_lose_it(tmp_path, monkeypatch):
    from axicavity_fem.fem_tm0 import solver

    mesh = _mesh(_polygon(*REENTRANT, 5.0), tmp_path / "m.msh")
    first, messages = _solve(mesh, num_modes=2)
    assert messages

    def memory_error(*args):
        raise MemoryError("no memory")

    monkeypatch.setattr(solver, "count_eigenvalues_below", memory_error)
    second, messages = _solve(mesh, num_modes=2)
    assert messages == []                                                # 数えられなければ黙って結果を返す
    assert second.frequencies == pytest.approx(first.frequencies, rel=1e-9, abs=0)


def test_very_large_problems_are_not_counted(tmp_path, monkeypatch, capsys):
    from axicavity_fem.fem_tm0 import solver

    monkeypatch.setattr(solver, "_COUNT_MAX_DOF", 10)
    _, messages = _solve(_mesh(_polygon(*REENTRANT, 5.0), tmp_path / "m.msh"), num_modes=1)
    assert messages == [] and "数えません" in capsys.readouterr().out


def test_static_solution_returned_as_a_mode_is_pointed_out(tmp_path):
    """細い内導体と粗いメッシュでは静的な解がシフトより上に来る（r = 1〜40 mm、h = 5 mm で 2.78 GHz、実モードは 3.48 GHz から）."""
    points = [(0.0, 1.0), (30.0, 1.0), (30.0, 40.0), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["PEC"] * 4, 5.0), tmp_path / "m.msh")
    res, messages = _solve(mesh, num_modes=2)
    assert res.frequencies[0] < 3.2
    assert any("静的な解" in m and "モード 0" in m for m in messages)


def test_ordinary_modes_of_a_coaxial_cavity_are_not_called_static(tmp_path):
    points = [(0.0, 10.0), (30.0, 10.0), (30.0, 40.0), (0.0, 40.0)]
    _, messages = _solve(_mesh(_polygon(points, ["PEC"] * 4, 2.5), tmp_path / "m.msh"), num_modes=3)
    assert not [m for m in messages if "静的な解" in m]


def test_uniform_dielectric_warns_only_when_a_mode_is_missed(tmp_path):
    """ε_r = 4: TM010 は 1.43 GHz でシフト 1.87 GHz より低いので警告. ε_r = 1.5 は低くならないので警告しない."""
    for eps_r, expected in ((4.0, 1), (1.5, 0)):
        geom = _polygon(*PILLBOX, 5.0, eps_r=eps_r)
        mesh = _mesh(geom, tmp_path / f"m{eps_r}.msh")
        _, messages = _solve(mesh, num_modes=1, material_table=build_material_table_from_geometry(geom))
        assert len(messages) == expected, eps_r


def test_ordinary_cavities_do_not_warn(tmp_path, sample_mesh):
    _, messages = _solve(_mesh(_polygon(*PILLBOX, 5.0), tmp_path / "m.msh"), num_modes=3)
    assert messages == []
    _, messages = _solve(load_mesh(sample_mesh("s-band_1cell", 2), element_order=2), num_modes=5)
    assert messages == []


def _traveling_messages(mesh, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_tm0_traveling(mesh, **kw)
    return res, [str(w.message) for w in caught if "[TM0]" in str(w.message)]


def _periodic_coax(tmp_path, order=2, h=5.0):
    points = [(0.0, 10.0), (200.0, 10.0), (200.0, 40.0), (0.0, 40.0)]
    return _mesh(_polygon(points, ["PEC", "None", "PEC", "None"], h), tmp_path / "m.msh", order=order)


def test_traveling_warns_once_and_names_the_phase(tmp_path):
    """進行波はシフトに近いモードを返す: 長い同軸の周期セルの TEM（θ = 60°）は 0.25, 1.25, 1.75 GHz で、1 本なら 1.75 GHz だけが返る."""
    res, messages = _traveling_messages(_periodic_coax(tmp_path), num_modes=1, phase_shifts=(60.0, 120.0))
    assert res.phase_results[60.0]["frequencies"][0] == pytest.approx(C0 * 7.0 / 6.0 / (0.2 * 1e9), rel=2e-3)     # 420°
    missed = [m for m in messages if "見逃して" in m]
    assert len(missed) == 1 and "2 本" in missed[0]


def test_traveling_does_not_count_the_static_solution_at_zero_phase(tmp_path):
    """θ = 0 の静的な解は 1 次要素で 0.24 GHz. それに引きずられて θ = 60° の本物の本数を落とさない."""
    _, messages = _traveling_messages(_periodic_coax(tmp_path, order=1, h=2.5), num_modes=1, phase_shifts=(0.0, 60.0))
    missed = [m for m in messages if "見逃して" in m]
    assert len(missed) == 1 and "θ = 60°" in missed[0] and "2 本" in missed[0]


def test_end_planes_that_do_not_pair_are_reported(tmp_path):
    """端面の r が 0.1 µm 違うと節点の多くが対にならず、4.05 GHz が 2.87 GHz になった."""
    points = [(0.0, 0.0), (35.0, 0.0), (35.0, 40.0001), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["None", "None", "PEC", "None"], 2.5), tmp_path / "m.msh")
    _, messages = _traveling_messages(mesh, num_modes=1, phase_shifts=(120.0,), target_freq_ghz=4.0)
    assert any("対になりません" in m for m in messages)


def test_cell_without_end_planes_is_reported(tmp_path):
    points = [(0.0, 0.0), (30.0, 0.0), (15.0, 40.0)]
    mesh = _mesh(_polygon(points, ["None", "PEC", "PEC"], 5.0), tmp_path / "m.msh")
    _, messages = _traveling_messages(mesh, num_modes=1, phase_shifts=(120.0,), target_freq_ghz=4.0)
    assert any("1 つも対になりません" in m for m in messages)


def test_wall_on_one_end_plane_is_not_reported(tmp_path):
    """対のない節点が壁だけなら正しいので警告しない."""
    points = [(0.0, 0.0), (35.0, 0.0), (35.0, 20.0), (35.0, 40.0), (1.0, 40.0), (1.0, 20.0), (0.0, 20.0)]
    mesh = _mesh(_polygon(points, ["None", "E-short", "PEC", "PEC", "PEC", "PEC", "E-short"], 2.5), tmp_path / "m.msh")
    _, messages = _traveling_messages(mesh, num_modes=1, phase_shifts=(120.0,), target_freq_ghz=3.0)
    assert not [m for m in messages if "対になりません" in m]


def test_traveling_with_a_prediction_does_not_warn(tmp_path):
    mesh = _mesh(_polygon(*REENTRANT, 5.0), tmp_path / "m.msh")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(0.0,), target_freq_ghz=0.5)
    assert not [w for w in caught if "[TM0]" in str(w.message)]


def test_static_solution_alone_does_not_warn(tmp_path):
    """短い同軸（TEM 4.28 GHz）: 最低次より下にあるのは周波数 0 の静的な解だけなので警告しない."""
    points = [(0.0, 10.0), (35.0, 10.0), (35.0, 40.0), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["PEC"] * 4, 5.0), tmp_path / "m.msh")
    res, messages = _solve(mesh, num_modes=1)
    assert res.frequencies[0] == pytest.approx(C0 / (2 * 0.035) / 1e9, rel=2e-3)
    assert messages == []


def test_prediction_above_the_lowest_mode_does_not_warn(tmp_path):
    """予測周波数を指定したら、それより低いモードが残っていても警告しない（その周波数の近くを頼んでいる）."""
    mesh = _mesh(_polygon(*PILLBOX, 5.0), tmp_path / "m.msh")
    res, messages = _solve(mesh, num_modes=1, target_freq_ghz=7.0)
    assert res.frequencies[0] > 3.0
    assert messages == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve_tm0_traveling(mesh, num_modes=1, phase_shifts=(0.0,), target_freq_ghz=7.0)
    assert not [w for w in caught if "[TM0]" in str(w.message)]


def test_end_plane_tilted_by_a_rounding_is_reported(tmp_path):
    """端面がわずかに傾くと周期条件がほぼ掛からない（dz = 0.1 µm で 4.05 GHz が 3.50 GHz）."""
    points = [(0.0, 0.0), (35.0, 0.0), (35.0001, 40.0), (0.0, 40.0)]
    mesh = _mesh(_polygon(points, ["None", "None", "PEC", "None"], 2.5), tmp_path / "m.msh")
    _, messages = _traveling_messages(mesh, num_modes=1, phase_shifts=(120.0,), target_freq_ghz=4.0)
    assert any("対になりません" in m for m in messages)


def test_scan_counts_every_phase(tmp_path):
    """0:360 の走査: 両端（θ = 0, 360）では下に静的な解しか無いが、途中の位相で見逃す（両端だけ数えると気づかない）."""
    _, messages = _traveling_messages(_periodic_coax(tmp_path), num_modes=1, phase_shifts=(0.0, 60.0, 360.0))
    assert any("見逃して" in m for m in messages)


def test_static_solution_alone_below_is_not_counted_even_if_its_estimate_is_high(tmp_path):
    """先細りの内導体: レイリー商は最低の固有値の上限にすぎない. 比べずに静的な解の本数を引く."""
    points = [(0.0, 0.2), (15.0, 5.0), (30.0, 5.0), (30.0, 40.0), (0.0, 40.0)]
    _, messages = _solve(_mesh(_polygon(points, ["PEC"] * 5, 2.5), tmp_path / "m.msh"), num_modes=1)
    assert not [m for m in messages if "見逃して" in m]


def test_disconnected_domains_count_their_own_static_solutions(tmp_path):
    """軸のあるピルボックスと離れた同軸: 静的な解は同軸の側にだけある."""
    points = [(0.0, 0.0), (35.0, 0.0), (35.0, 40.0), (0.0, 40.0), (50.0, 10.0), (85.0, 10.0), (85.0, 40.0), (50.0, 40.0)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None")] + [
        Segment(id=i, type="line", point_indices=[i, (i + 1) % 4], bc_name="PEC") for i in (1, 2, 3)] + [
        Segment(id=4 + i, type="line", point_indices=[4 + i, 4 + (i + 1) % 4], bc_name="PEC") for i in range(4)]
    geom = MultiRegionGeometry(points=points, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 7])],
                               regions=[Region(id=0, name="A", outer_loop_id=0, material_tag="a"), Region(id=1, name="B", outer_loop_id=1, material_tag="b")],
                               unit="mm", mesh_size=5.0)
    _, messages = _solve(_mesh(geom, tmp_path / "m.msh"), num_modes=2)
    assert not [m for m in messages if "見逃して" in m]


def test_two_coaxial_domains_have_two_static_solutions(tmp_path):
    """離れた同軸 2 つ: 静的な解は 2 本で、どちらも見逃しではない."""
    points = [(0.0, 10.0), (35.0, 10.0), (35.0, 40.0), (0.0, 40.0), (50.0, 10.0), (85.0, 10.0), (85.0, 40.0), (50.0, 40.0)]
    segs = [Segment(id=i, type="line", point_indices=[i, (i + 1) % 4], bc_name="PEC") for i in range(4)] + [
        Segment(id=4 + i, type="line", point_indices=[4 + i, 4 + (i + 1) % 4], bc_name="PEC") for i in range(4)]
    geom = MultiRegionGeometry(points=points, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 7])],
                               regions=[Region(id=0, name="A", outer_loop_id=0, material_tag="a"), Region(id=1, name="B", outer_loop_id=1, material_tag="b")],
                               unit="mm", mesh_size=5.0)
    _, messages = _solve(_mesh(geom, tmp_path / "m.msh", order=1), num_modes=1)
    assert not [m for m in messages if "見逃して" in m]


def test_nodes_outside_any_element_are_not_a_static_solution():
    from axicavity_fem.fem_tm0.solver import _static_components

    K = sp.csr_matrix(np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    components = _static_components(K, [])
    assert [list(c) for c in components] == [[0, 1]]
