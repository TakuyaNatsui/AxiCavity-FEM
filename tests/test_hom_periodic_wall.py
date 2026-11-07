"""HOM の壁と周期面の回帰テスト: 周期面にある壁の点、周期面のエッジの符号、内部の辺、周期面の許容.

3.0.0 までは周期面の節点のうち r = r_max だけを周期条件から外して PEC に委ねていた。そのため:

- r < r_max にある PEC 壁の点（アイリスの先端、同軸の内導体）に E_θ = 0 が課されず、n ≥ 1 の周波数が低く出ていた
  （S バンドのセル、h = 2.5 mm で最大 −0.35%、Q は最大 −3.1%）
- 外壁が磁気壁（M-short）だと r = r_max の角の周期条件が外れていた

基準はこのソルバの周期セルの解を使わない:

- 同軸セル: TE_nm は J_n'(k a)Y_n'(k b) = J_n'(k b)Y_n'(k a)、TM_nm は J_n(k a)Y_n(k b) = J_n(k b)Y_n(k a) の根 k_c
- 外壁が磁気壁の円筒（双対）: E_z ∝ J_n(k_c r) で J_n'(k_c R) = 0、H_z ∝ J_n(k_c r) で J_n(k_c R) = 0
- S バンドのセル（z 対称）: θ = 60°, 120° は 3 セルの定在波（端面 E-short）に、θ = 0°, 180° は 1 セルの定在波
  （端面 E-short または M-short）に同じ周波数のモードがある

後半（3.0.1 のレビューで見つかった誤り。どれも 3.0.0 以前から）:

- 周期面のエッジの符号が三角形の向きの一致を仮定していて、z 方向に領域を分けたセルで周波数が全く違っていた
  （GUI の領域は向きが逆になる）。基準は単一領域のメッシュ、解析解、独立な TM0
- 両端が壁の節点の内部の辺まで壁として固定していた（鋭角の角、1 要素幅の隙間）。基準は独立な TM0（n = 0）と解析解
- 周期面の許容が絶対値 10 µm で、30 µm より細かいメッシュで止まるか誤った値を返した。基準は解析解
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jn_zeros, jnp_zeros, jv, jvp, yv, yvp

pytest.importorskip("gmsh")

from axicavity_fem.fem_hom.boundary import find_periodic_boundary_pairs, periodic_tolerance  # noqa: E402
from axicavity_fem.fem_hom.field_recon import split_hom_dofs  # noqa: E402
from axicavity_fem.fem_hom.post_process import compute_hom_parameters  # noqa: E402
from axicavity_fem.fem_hom.solver import solve_hom_standing, solve_hom_traveling  # noqa: E402
from axicavity_fem.fem_tm0.post_process import compute_tm0_parameters  # noqa: E402
from axicavity_fem.fem_tm0.solver import solve_tm0_standing, solve_tm0_traveling  # noqa: E402
from axicavity_fem.gui.core.convert import to_multi_region  # noqa: E402
from axicavity_fem.gui.core.document import create_empty_document  # noqa: E402
from axicavity_fem.gui.core.sketch.model import add_rectangle  # noqa: E402
from axicavity_fem.shared.boundary_groups import classify_boundaries  # noqa: E402
from axicavity_fem.shared.constants import C0  # noqa: E402
from axicavity_fem.shared.edge_index import create_edge_index_map  # noqa: E402
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region  # noqa: E402
from axicavity_fem.shared.material_resolver import build_material_table_from_geometry  # noqa: E402
from axicavity_fem.shared.mesh_loader import compute_hom_boundary, load_mesh, load_mesh_hom  # noqa: E402
from axicavity_fem.shared.multi_region_model import Loop, MultiRegionGeometry, Region, Segment  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
LENGTH_MM, A_MM, B_MM = 35.0, 10.0, 40.0
LENGTH, A, B = LENGTH_MM * 1e-3, A_MM * 1e-3, B_MM * 1e-3
BETA = np.radians(120.0) / LENGTH


def _rect(r0, r1, walls, end_bc, h):
    """z ∈ [0, L], r ∈ [r0, r1] の長方形（mm）. ``walls`` は下の辺と上の辺の境界条件."""
    pts = [(0.0, r0), (LENGTH_MM, r0), (LENGTH_MM, r1), (0.0, r1)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name=walls[0]),
            Segment(id=1, type="line", point_indices=[1, 2], bc_name=end_bc),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name=walls[1]),
            Segment(id=3, type="line", point_indices=[3, 0], bc_name=end_bc)]
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[Region(id=0, name="Cell", outer_loop_id=0, material_tag="vacuum")],
                               unit="mm", mesh_size=h)


def _write(geom, path, order=2):
    export_msh_multi_region(geom, path, mesh_order=order, verbose=0)
    return str(path)


def _sample(folder, name, ends, label, h=2.5):
    geom = MultiRegionGeometry.load_json(SAMPLES / f"{name}.gmshproj")
    geom.mesh_size = h
    found = 0
    for seg in geom.segments:
        if seg.id in ends:
            seg.bc_name = label
            found += 1
    assert found == len(ends)
    return _write(geom, folder / f"{name}_{label}.msh")


def _coax_kc(kind, n):
    """同軸の最低次の TE_n1 / TM_n1 の k_c（最初の根）."""
    if kind == "TM":
        f = lambda k: jv(n, k * A) * yv(n, k * B) - jv(n, k * B) * yv(n, k * A)
    else:
        f = lambda k: jvp(n, k * A) * yvp(n, k * B) - jvp(n, k * B) * yvp(n, k * A)
    ks = np.linspace(1.0, 500.0, 5000)
    values = f(ks)
    i = np.nonzero(values[:-1] * values[1:] < 0)[0][0]
    return brentq(f, ks[i], ks[i + 1])


def _f_ghz(kc, beta=BETA):
    return np.hypot(kc, beta) * C0 / (2e9 * np.pi)


def _nearest_error(frequencies, ref):
    frequencies = np.asarray(frequencies)
    return frequencies[np.argmin(np.abs(frequencies - ref))] / ref - 1.0


def _e_theta(res, modeset, m, n):
    return split_hom_dofs(modeset.eigenvectors[m], res.num_edges, len(res.simplices), len(res.vertices), res.element_order,
                          n, res.vertices, res.simplices)[3]


def _plane_nodes(vertices):
    z = vertices[:, 0]
    return [np.nonzero(np.abs(z - z0) < 1e-9)[0] for z0 in (z.min(), z.max())]


# ---------------------------------------------------------------------------
# 周期面の節点ペア
# ---------------------------------------------------------------------------
def _small_cell():
    vertices = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])     # (z, r)
    simplices = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2]])
    edges, _ = create_edge_index_map(simplices)
    return vertices, simplices, edges


def test_node_pairs_skip_the_points_of_wall_edges():
    vertices, _, edges = _small_cell()
    _, pairs, _, _, images = find_periodic_boundary_pairs(vertices, edges, 1, [edges[(0, 3)], edges[(2, 5)]])
    assert pairs == [(1, 4)] and images == []         # 軸（0, 3）と外壁（2, 5）の点は外れ、r = 1 は周期条件のまま
    _, pairs, _, _, images = find_periodic_boundary_pairs(vertices, edges, 1, ())
    assert sorted(pairs) == [(0, 3), (1, 4), (2, 5)] and images == []  # 壁が無ければ全部つなぐ（r_max の特別扱いはしない）
    assert find_periodic_boundary_pairs(vertices, edges, 0, [edges[(0, 3)]])[1] == []     # n = 0 は節点 DOF なし


def test_a_wall_on_one_plane_also_fixes_its_image_on_the_other():
    vertices, _, edges = _small_cell()
    _, pairs, _, _, images = find_periodic_boundary_pairs(vertices, edges, 1, [edges[(4, 5)]])     # z_max 側の周期面の辺は壁に数えない
    assert sorted(pairs) == [(0, 3), (1, 4), (2, 5)] and images == []
    _, pairs, _, _, images = find_periodic_boundary_pairs(vertices, edges, 1, [edges[(1, 5)]])     # z_min の点 1 と z_max の点 5
    assert sorted(pairs) == [(0, 3)]
    assert sorted(images) == [2, 4]                   # 点 1 の像は 4、点 5 の像は 2。どちらも壁の像なので固定する


def test_edge_sign_compares_the_direction_along_r():
    """基底の向きは番号の小さい節点 → 大きい節点. 片方の面だけ番号が逆なら符号は −1（三角形の向きによらない）."""
    vertices = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])                  # z_max 側は r の大きい節点の番号が小さい
    edges = {(0, 1): 0, (2, 3): 1, (0, 3): 2, (1, 2): 3, (0, 2): 4}
    pairs = find_periodic_boundary_pairs(vertices, edges, 0, ())[0]
    assert pairs == [(0, 1, -1.0)]
    vertices = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    edges = {(0, 1): 0, (2, 3): 1, (0, 2): 2, (1, 3): 3, (0, 3): 4}
    assert find_periodic_boundary_pairs(vertices, edges, 0, ())[0] == [(0, 1, 1.0)]


def test_periodic_tolerance_follows_the_mesh():
    vertices, _, edges = _small_cell()
    assert periodic_tolerance(vertices, edges) == 1e-5                                     # 普通のメッシュは上限
    assert periodic_tolerance(vertices * 1e-4, edges) == pytest.approx(1e-7)               # 辺 0.1 mm のセルはセル長の 1e-3 倍
    assert periodic_tolerance(vertices * [1.0, 1e-6], edges) == pytest.approx(1e-8)         # 面の辺 1 µm の 1e-2 倍


def test_periodic_tolerance_ignores_small_edges_away_from_the_planes():
    """周期面と関係のない細かい辺に引きずられない（3.0.0 より許容を狭くすると座標の丸めで止まる）."""
    vertices = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0], [1e-3, 1e-3], [5e-4, 5e-4], [5e-4 + 2e-5, 5e-4]])
    edges = {(0, 1): 0, (2, 3): 1, (0, 4): 2, (1, 4): 3, (2, 5): 4, (3, 5): 5, (4, 5): 6}
    assert periodic_tolerance(vertices, edges) == pytest.approx(1e-6)    # セル長 1 mm の 1e-3 倍. 20 µm の辺なら 2e-7 になる


# ---------------------------------------------------------------------------
# 同軸セル: 内導体の壁が周期面に達する（r = a < r_max）解析解
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coax(tmp_path_factory):
    folder = tmp_path_factory.mktemp("coax")
    return {end: _write(_rect(A_MM, B_MM, ("PEC", "PEC"), end, 2.0), folder / f"{end}.msh") for end in ("None", "E-short")}


@pytest.mark.parametrize("end", ["None", "E-short"])
@pytest.mark.parametrize("kind, n", [("TE", 1), ("TE", 2), ("TM", 1)], ids=["TE11", "TE21", "TM11"])
def test_coax_cell_matches_analytic(coax, end, kind, n):
    ref = _f_ghz(_coax_kc(kind, n))
    res = solve_hom_traveling(load_mesh_hom(coax[end], n, 2), num_modes=3, phase_shifts=(120.0,))
    # 実測: TE11 −6.8e-8、TE21 +1.1e-7、TM11 +3.9e-7（3.0.0 は TE11 −1.2e-5、TE21 −5.2e-6）
    assert abs(_nearest_error(res.periodic[120.0].frequencies, ref)) < 1e-6


def test_coax_cell_first_order_elements(tmp_path):
    """1 次要素の変換行列でも内導体の角が固定される. 周波数の誤差はメッシュで決まる（実測 +7.5e-5, +4.2e-4）."""
    path = _write(_rect(A_MM, B_MM, ("PEC", "PEC"), "None", 2.0), tmp_path / "m.msh", order=1)
    for kind, n in (("TE", 1), ("TE", 2)):
        res = solve_hom_traveling(load_mesh_hom(path, n, 1), num_modes=3, phase_shifts=(120.0,))
        modes = res.periodic[120.0]
        assert abs(_nearest_error(modes.frequencies, _f_ghz(_coax_kc(kind, n)))) < 1e-3
        e_theta = _e_theta(res, modes, 0, n)
        for nodes in _plane_nodes(res.vertices):
            inner = nodes[np.abs(res.vertices[nodes, 1] - A) < 1e-9]
            assert len(inner) == 1 and e_theta[inner[0]] == 0.0


# ---------------------------------------------------------------------------
# 外壁が磁気壁: r_max の角も周期条件でつなぐ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("end", ["None", "E-short"])
@pytest.mark.parametrize("n", [1, 2])
def test_magnetic_outer_wall_matches_analytic(tmp_path, end, n):
    path = _write(_rect(0.0, B_MM, ("None", "M-short"), end, 2.0), tmp_path / "m.msh")
    res = solve_hom_traveling(load_mesh_hom(path, n, 2), num_modes=4, phase_shifts=(120.0,))
    freqs = res.periodic[120.0].frequencies
    refs = sorted(_f_ghz(x / B) for x in (jnp_zeros(n, 1)[0], jn_zeros(n, 1)[0]))
    # 実測 2.4e-7〜1.7e-6（3.0.0 は端面 None で 1.8e-4 まで、端面 E-short では 1.30 GHz の偽モードも出た）
    assert abs(_nearest_error(freqs, refs[0])) < 6e-6 and abs(_nearest_error(freqs, refs[1])) < 6e-6
    assert freqs[0] == pytest.approx(refs[0], rel=6e-6)


# ---------------------------------------------------------------------------
# S バンドのセル: アイリスの先端が周期面にある。定在波が基準
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sband(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sband")
    meshes = {label: _sample(folder, "s-band_1cell", (1, 7), label) for label in ("M-short", "E-short", "None")}
    meshes["3cell"] = _sample(folder, "s-band_3cell", (7, 26), "E-short")
    return meshes


@pytest.mark.parametrize("n", [1, 2])
def test_sband_cell_matches_standing_waves_at_every_phase(sband, n):
    three = solve_hom_standing(load_mesh_hom(sband["3cell"], n, 2), num_modes=24).normal.frequencies
    one = np.concatenate([solve_hom_standing(load_mesh_hom(sband[label], n, 2), num_modes=8).normal.frequencies
                          for label in ("E-short", "M-short")])
    res = solve_hom_traveling(load_mesh_hom(sband["M-short"], n, 2), num_modes=4, phase_shifts=(0.0, 60.0, 120.0, 180.0))
    # 3.0.0 は最低次のモードで θ = 0°: −3.5e-3、60°: −2.9e-3、120°: −1.8e-3、180°: −1.4e-3（n = 1）
    for theta, ref, tol in ((60.0, three, 5e-5), (120.0, three, 5e-5),     # 実測 h = 2.4〜2.75 mm で 2.6e-6〜1.6e-5（h に単調でない）
                            (0.0, one, 5e-7), (180.0, one, 5e-7)):        # 実測 7.6e-8 以下
        for f in res.periodic[theta].frequencies:
            assert abs(_nearest_error(ref, f)) < tol, (theta, f)


def test_end_label_does_not_change_traveling_modes(sband):
    """周期面のラベルは周期条件で上書きされるので、周波数は変わらない（実測の差 1.1e-7 以下）.

    内部の固定辺が無いセルに限る（このセルは 0 本）。鋭角の角では差が出る（既知の制約）。
    """
    freqs = {label: solve_hom_traveling(load_mesh_hom(sband[label], 1, 2), num_modes=4, phase_shifts=(120.0,)).periodic[120.0].frequencies
             for label in ("M-short", "E-short", "None")}
    assert freqs["E-short"] == pytest.approx(freqs["None"], rel=2e-7, abs=0)
    assert freqs["M-short"] == pytest.approx(freqs["None"], rel=2e-7, abs=0)


@pytest.mark.parametrize("label", ["M-short", "E-short", "None"])
def test_e_theta_is_zero_at_the_iris_tip(sband, label):
    res = solve_hom_traveling(load_mesh_hom(sband[label], 1, 2), num_modes=2, phase_shifts=(120.0,))
    modes = res.periodic[120.0]
    for m in range(2):
        e_theta = _e_theta(res, modes, m, 1)
        for nodes in _plane_nodes(res.vertices):
            r = res.vertices[nodes, 1]
            order = np.argsort(r)
            assert e_theta[nodes[order[-1]]] == 0.0                     # 周期面の一番外側の点がアイリスの先端
            assert abs(e_theta[nodes[order[-2]]]) > 1e-3 * np.max(np.abs(e_theta))  # その内側は周期条件で値を持つ


# ---------------------------------------------------------------------------
# z 方向に領域を分けたセル（領域ごとに三角形の向きが逆になる）
# ---------------------------------------------------------------------------
def _z_split_cell(h):
    doc = create_empty_document("t")
    left, _ = add_rectangle(doc.sketch, (0, 0), (15, 40))
    right, _ = add_rectangle(doc.sketch, (15, 0), (30, 40), tolerance=1e-6)
    doc.mesh.size = h
    for line in (left[3], right[1]):
        doc.boundaries[line] = "None"
    geom = to_multi_region(doc).geom
    geom.mesh_size = h
    return geom


def _single_cell(h):
    pts = [(0.0, 0.0), (30.0, 0.0), (30.0, 40.0), (0.0, 40.0)]
    segs = [Segment(id=i, type="line", point_indices=[i, (i + 1) % 4], bc_name=bc) for i, bc in enumerate(("None", "None", "PEC", "None"))]
    return MultiRegionGeometry(points=pts, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3])],
                               regions=[Region(id=0, name="Cell", outer_loop_id=0, material_tag="vacuum")], unit="mm", mesh_size=h)


@pytest.fixture(scope="module")
def z_split(tmp_path_factory):
    folder = tmp_path_factory.mktemp("z_split")
    return {"split": _write(_z_split_cell(2.5), folder / "split.msh"), "single": _write(_single_cell(2.5), folder / "single.msh")}


def test_gui_cell_really_has_both_triangle_orientations(z_split):
    """このテストの前提: GUI の経路で作ると領域ごとに三角形の向きが逆になる."""
    mesh = load_mesh(z_split["split"], element_order=2)
    corners = mesh.nodes[mesh.elements[:, :3]]
    area = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
            - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1]))
    assert np.any(area > 0.0) and np.any(area < 0.0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_z_split_cell_matches_a_single_region(z_split, n):
    """3.0.0 は n = 1 で 3.990 GHz が 1.791 GHz、n = 0 で 4.396 GHz が 3.381 GHz、v_g も全く違った."""
    params, freqs = {}, {}
    for label in ("split", "single"):
        res = solve_hom_traveling(load_mesh_hom(z_split[label], n, 2), num_modes=3, phase_shifts=(120.0,))
        freqs[label] = res.periodic[120.0].frequencies
        params[label] = compute_hom_parameters(res, conductivity=5.8e7).per_phase[120.0]
    assert freqs["split"] == pytest.approx(freqs["single"], rel=2e-5, abs=0)
    for a, b in zip(params["split"], params["single"]):
        assert a.q_factor == pytest.approx(b.q_factor, rel=2e-3)
        assert a.v_group_c == pytest.approx(b.v_group_c, abs=1e-3)
    beta = np.radians(120.0) / 30e-3
    kc = float(jn_zeros(n, 1)[0] if n == 0 else jnp_zeros(n, 1)[0]) / 40e-3
    assert freqs["split"][0] == pytest.approx(np.hypot(kc, beta) * C0 / (2e9 * np.pi), rel=1e-5)


def test_z_split_first_order_elements(tmp_path):
    path = _write(_z_split_cell(2.5), tmp_path / "split.msh", order=1)
    res = solve_hom_traveling(load_mesh_hom(path, 1, 1), num_modes=1, phase_shifts=(120.0,))
    kc, beta = float(jnp_zeros(1, 1)[0]) / 40e-3, np.radians(120.0) / 30e-3
    assert res.periodic[120.0].frequencies[0] == pytest.approx(np.hypot(kc, beta) * C0 / (2e9 * np.pi), rel=2e-3)


@pytest.mark.parametrize("theta", [120.0, 180.0])
def test_sample_with_a_dielectric_window_matches_tm0(tmp_path, theta):
    """samples/diel_simple1（z の右半分が ε_r = 2）: n = 0 の HOM と独立な TM0 は同じ周波数."""
    geom = MultiRegionGeometry.load_json(SAMPLES / "diel_simple1.gmshproj")
    path = _write(geom, tmp_path / "m.msh")
    table = build_material_table_from_geometry(geom)
    tm0 = solve_tm0_traveling(load_mesh(path, element_order=2), num_modes=4, phase_shifts=(theta,), material_table=table)
    hom = solve_hom_traveling(load_mesh_hom(path, 0, 2), num_modes=4, phase_shifts=(theta,), material_table=table)
    assert hom.periodic[theta].frequencies == pytest.approx(tm0.phase_results[theta]["frequencies"], rel=1e-5, abs=0)


# ---------------------------------------------------------------------------
# 内部の辺は壁ではない
# ---------------------------------------------------------------------------
def _polygon(points, bcs, h):
    n = len(points)
    segs = [Segment(id=i, type="line", point_indices=[i, (i + 1) % n], bc_name=bc) for i, bc in enumerate(bcs)]
    return MultiRegionGeometry(points=list(points), segments=segs, loops=[Loop(id=0, segment_ids=list(range(n)))],
                               regions=[Region(id=0, name="Cell", outer_loop_id=0, material_tag="vacuum")], unit="mm", mesh_size=h)


ACUTE_ONE = [(0.0, 0.0), (35.0, 0.0), (35.0, 40.0), (0.0, 27.26)]           # 角 70°、内部の固定辺が h によらず 1 本あった
ACUTE_TWO = [(0.0, 0.0), (35.0, 0.0), (35.0, 40.0), (17.5, 33.63), (0.0, 40.0)]   # 角 70° が 2 つ


def _interior_wall_edges(mesh):
    uses = {}
    for simplex in mesh.simplices[:, :3]:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(simplex[a]), int(simplex[b]))))
            uses[key] = uses.get(key, 0) + 1
    inverse = {index: key for key, index in mesh.edge_index_map.items()}
    return ([e for e in mesh.boundary_edge_indices if uses[inverse[e]] == 2],
            [e for e in mesh.pec_loss_edge_indices if uses[inverse[e]] == 2])


@pytest.mark.parametrize("points", [ACUTE_ONE, ACUTE_TWO], ids=["one", "two"])
@pytest.mark.parametrize("h", [5.0, 2.5])
def test_acute_corner_has_no_interior_wall_edge(tmp_path, points, h):
    path = _write(_polygon(points, ["None"] + ["PEC"] * (len(points) - 1), h), tmp_path / "m.msh")
    for n in (0, 1):
        dirichlet, loss = _interior_wall_edges(load_mesh_hom(path, n, 2))
        assert dirichlet == [] and loss == []


@pytest.mark.parametrize("points", [ACUTE_ONE, ACUTE_TWO], ids=["one", "two"])
def test_acute_corner_frequency_and_q_match_tm0(tmp_path, points):
    """3.0.0 は h = 2.5 mm で f が +0.36%、Q が +4.5%（第 1 モード、角 1 つ）. 周波数の判定だけ直すと Q は −10% になる."""
    path = _write(_polygon(points, ["None"] + ["PEC"] * (len(points) - 1), 2.5), tmp_path / "m.msh")
    mesh = load_mesh(path, element_order=2)
    tm0 = solve_tm0_standing(mesh, num_modes=3)
    tm0_q = compute_tm0_parameters(tm0, classify_boundaries(mesh.physical_groups, mesh.nodes), conductivity=5.8e7).per_phase[0.0]
    hom = solve_hom_standing(load_mesh_hom(path, 0, 2), num_modes=3)
    hom_q = compute_hom_parameters(hom, conductivity=5.8e7).per_phase[0.0]
    assert hom.normal.frequencies == pytest.approx(tm0.frequencies, rel=2e-4, abs=0)         # 実測 1.5e-4 以下
    for a, b in zip(hom_q, tm0_q):
        assert a.q_factor == pytest.approx(b.q_factor, rel=0.01)                              # 実測 0.1〜0.6%（曲線壁と同じ 1 次の収束）


def test_traveling_modes_do_not_depend_on_the_end_label_at_an_acute_corner(tmp_path):
    """3.0.0 は端面 E-short / PEC（マニュアルと GUI が勧める設定）で 0.38〜0.78% 違った."""
    freqs = {}
    for label in ("None", "M-short", "E-short", "PEC"):
        path = _write(_polygon(ACUTE_TWO, ["None", label, "PEC", "PEC", label], 2.5), tmp_path / f"{label}.msh")
        freqs[label] = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=3, phase_shifts=(180.0,)).periodic[180.0].frequencies
    for label in ("M-short", "E-short", "PEC"):
        assert freqs[label] == pytest.approx(freqs["None"], rel=1e-6, abs=0)


@pytest.mark.parametrize("label", ["None", "E-short", "PEC"])
def test_thin_cell_has_no_spurious_mode_with_a_wall_label_on_the_end(tmp_path, label):
    """セルの長さがメッシュと同じくらい: 3.0.0 は端面 E-short / PEC で解析解から 9% 離れた偽モードが出た."""
    length = 5.0
    path = _write(_polygon([(0.0, 0.0), (length, 0.0), (length, 40.0), (0.0, 40.0)], ["None", label, "PEC", label], 5.0), tmp_path / "m.msh")
    freqs = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=3, phase_shifts=(30.0,)).periodic[30.0].frequencies
    beta = np.radians(30.0) / (length * 1e-3)
    refs = sorted(np.hypot(float(x) / 40e-3, beta) * C0 / (2e9 * np.pi) for x in list(jn_zeros(1, 3)) + list(jnp_zeros(1, 3)))
    assert freqs == pytest.approx(refs[:3], rel=5e-4, abs=0)


def _reentrant(h):
    """半径 15 mm のポスト、隙間 2 mm. 隙間に要素が 1 層でも内部の辺を固定しない."""
    points = [(0.0, 15.0), (33.0, 15.0), (33.0, 0.0), (35.0, 0.0), (35.0, 40.0), (0.0, 40.0)]
    return _polygon(points, ["PEC", "PEC", "None", "PEC", "PEC", "PEC"], h)


def test_reentrant_gap_keeps_the_fundamental_mode(tmp_path):
    """h = 2.5 mm: 3.0.0 は n = 0 で基本モードが消えた（4.283 GHz から）. TM0 と 0.6%、第 2 モードは 0.05% 以内."""
    path = _write(_reentrant(2.5), tmp_path / "m.msh")
    tm0 = solve_tm0_standing(load_mesh(path, element_order=2), num_modes=2, target_freq_ghz=0.5)
    hom = solve_hom_standing(load_mesh_hom(path, 0, 2), num_modes=2)
    assert hom.normal.frequencies[0] == pytest.approx(tm0.frequencies[0], rel=0.006)
    assert hom.normal.frequencies[1] == pytest.approx(tm0.frequencies[1], rel=5e-4)


def _post_on_the_axis(h):
    """軸の途中に載った導体の円柱（r < 10 mm、z = 15〜20 mm）: 壁が軸の両端でない点で軸に接する."""
    points = [(0.0, 0.0), (15.0, 0.0), (15.0, 10.0), (20.0, 10.0), (20.0, 0.0), (35.0, 0.0), (35.0, 40.0), (0.0, 40.0)]
    return _polygon(points, ["None", "PEC", "PEC", "PEC", "None", "PEC", "PEC", "PEC"], h)


def test_wall_touching_the_axis_is_fixed_also_at_n0(tmp_path):
    """軸の途中に接する壁の辺は n = 0 でも固定する（軸の辺は固定しない）. 3.0.0 は軸上の節点を除くときに壁の辺まで外していた."""
    mesh = load_mesh_hom(_write(_post_on_the_axis(2.5), tmp_path / "m.msh"), 0, 2)
    fixed = set(mesh.boundary_edge_indices)
    z, r = mesh.vertices[:, 0], mesh.vertices[:, 1]
    tip, axis = [], []
    for (a, b), index in mesh.edge_index_map.items():
        if abs(z[a] - z[b]) < 1e-9 and min(abs(z[a] - 15e-3), abs(z[a] - 20e-3)) < 1e-9 and min(r[a], r[b]) < 1e-9:
            tip.append(index)
        if r[a] < 1e-9 and r[b] < 1e-9:
            axis.append(index)
    assert len(tip) == 2 and axis
    assert all(index in fixed for index in tip)
    assert not any(index in fixed for index in axis)


def test_wall_touching_the_axis_matches_tm0(tmp_path):
    path = _write(_post_on_the_axis(2.5), tmp_path / "m.msh")
    tm0 = solve_tm0_standing(load_mesh(path, element_order=2), num_modes=3)
    hom = solve_hom_standing(load_mesh_hom(path, 0, 2), num_modes=3)
    assert hom.normal.frequencies == pytest.approx(tm0.frequencies, rel=2e-3, abs=0)


@pytest.mark.parametrize("order, tolerance", [(2, 3e-5), (1, 5e-3)])
def test_interior_pec_sheet_reaching_the_periodic_plane(tmp_path, order, tolerance):
    """領域の境界の PEC（r = 15 mm）は壁のまま. シートの端点も周期条件でつながない. 解析解は半径 15 mm の円管の TE11."""
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 15.0), (0.0, 15.0), (30.0, 40.0), (0.0, 40.0)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"), Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
            Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"), Segment(id=3, type="line", point_indices=[3, 0], bc_name="None"),
            Segment(id=4, type="line", point_indices=[2, 4], bc_name="None"), Segment(id=5, type="line", point_indices=[4, 5], bc_name="PEC"),
            Segment(id=6, type="line", point_indices=[5, 3], bc_name="None")]
    geom = MultiRegionGeometry(points=points, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 2])],
                               regions=[Region(id=0, name="A", outer_loop_id=0, material_tag="a"), Region(id=1, name="B", outer_loop_id=1, material_tag="b")],
                               unit="mm", mesh_size=2.0)
    mesh = load_mesh_hom(_write(geom, tmp_path / "m.msh", order=order), 1, order)
    res = solve_hom_traveling(mesh, num_modes=4, phase_shifts=(0.0,))
    assert abs(_nearest_error(res.periodic[0.0].frequencies, _f_ghz(float(jnp_zeros(1, 1)[0]) / 15e-3, beta=0.0))) < tolerance
    e_theta = _e_theta(res, res.periodic[0.0], 0, 1)
    z, r = res.vertices[:, 0], res.vertices[:, 1]
    ends = np.nonzero((np.abs(r - 15e-3) < 1e-9) & ((np.abs(z) < 1e-9) | (np.abs(z - 30e-3) < 1e-9)))[0]
    assert len(ends) == 2 and np.all(e_theta[ends] == 0.0)


def test_first_order_wall_loss_excludes_interior_edges(tmp_path):
    """1 次要素の Q: 3.0.0 の TM0 は内部の辺を壁に数えて h = 2.5 mm で −5.6%（修正後 −0.18%）."""
    points = ACUTE_ONE
    bcs = ["None", "PEC", "PEC", "PEC"]
    second = _write(_polygon(points, bcs, 2.5), tmp_path / "second.msh")
    mesh2 = load_mesh(second, element_order=2)
    reference = compute_tm0_parameters(solve_tm0_standing(mesh2, num_modes=1), classify_boundaries(mesh2.physical_groups, mesh2.nodes),
                                       conductivity=5.8e7).per_phase[0.0][0].q_factor
    first = _write(_polygon(points, bcs, 2.5), tmp_path / "first.msh", order=1)
    mesh1 = load_mesh(first, element_order=1)
    q_tm0 = compute_tm0_parameters(solve_tm0_standing(mesh1, num_modes=1), classify_boundaries(mesh1.physical_groups, mesh1.nodes),
                                   conductivity=5.8e7).per_phase[0.0][0].q_factor
    assert q_tm0 == pytest.approx(reference, rel=0.005)
    hom = load_mesh_hom(first, 0, 1)
    assert _interior_wall_edges(hom) == ([], [])


def test_stored_mesh_gives_the_same_wall_edges_without_region_ids(tmp_path):
    """post は h5 のメッシュから壁の辺を求め直す. 2 次要素では領域の情報がなくても同じ結果になる."""
    mesh = load_mesh_hom(_write(_z_split_cell(5.0), tmp_path / "m.msh"), 1, 2)
    again = compute_hom_boundary(mesh.vertices, mesh.simplices, mesh.edge_index_map, mesh.physical_groups, 1)
    assert sorted(again[0]) == sorted(mesh.boundary_edge_indices)
    assert sorted(again[2]) == sorted(mesh.pec_loss_edge_indices)


# ---------------------------------------------------------------------------
# 周期面の許容（細かいメッシュ）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h", [0.03, 0.025, 0.02])
def test_periodic_planes_with_a_very_fine_mesh(tmp_path, h):
    """R = 0.4 mm: 3.0.0 は h = 0.03, 0.02 mm で止まり、h = 0.025 mm で f +1.1e-3、v_g −6.4% を返した."""
    length, radius = 0.35, 0.40
    path = _write(_polygon([(0.0, 0.0), (length, 0.0), (length, radius), (0.0, radius)], ["None", "None", "PEC", "None"], h), tmp_path / "m.msh")
    res = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=1, phase_shifts=(120.0,))
    kc, beta = float(jnp_zeros(1, 1)[0]) / (radius * 1e-3), np.radians(120.0) / (length * 1e-3)
    assert res.periodic[120.0].frequencies[0] == pytest.approx(np.hypot(kc, beta) * C0 / (2e9 * np.pi), rel=5e-6)
    assert compute_hom_parameters(res, conductivity=5.8e7).per_phase[120.0][0].v_group_c == pytest.approx(beta / np.hypot(kc, beta), abs=2e-3)


def test_end_planes_that_do_not_match_give_a_clear_error(tmp_path):
    """片方の端面だけに壁があるセルは HOM の進行波では解けない. エラー文が切り方の変更を案内する."""
    points = [(0.0, 0.0), (35.0, 0.0), (35.0, 20.0), (35.0, 40.0), (5.0, 40.0), (5.0, 20.0), (0.0, 20.0)]
    path = _write(_polygon(points, ["None", "E-short", "PEC", "PEC", "PEC", "PEC", "E-short"], 2.5), tmp_path / "m.msh")
    with pytest.raises(RuntimeError, match="セルの切り方"):
        solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=1, phase_shifts=(120.0,))


def test_tm0_solves_a_wall_on_one_end_plane_correctly(tmp_path):
    """同じ周期構造の切り方を変えたセルと一致（対のない面内の壁は自然境界 = PEC になる）. HOM と違って検証で止めない."""
    middle = _polygon([(0.0, 0.0), (35.0, 0.0), (35.0, 40.0), (18.0, 40.0), (18.0, 20.0), (17.0, 20.0), (17.0, 40.0), (0.0, 40.0)],
                      ["None", "None", "PEC", "PEC", "PEC", "PEC", "PEC", "None"], 1.5)
    one_side = _polygon([(0.0, 0.0), (35.0, 0.0), (35.0, 20.0), (35.0, 40.0), (1.0, 40.0), (1.0, 20.0), (0.0, 20.0)],
                        ["None", "E-short", "PEC", "PEC", "PEC", "PEC", "E-short"], 1.5)
    freqs = [solve_tm0_traveling(load_mesh(_write(g, tmp_path / f"{i}.msh"), element_order=2), num_modes=2, phase_shifts=(120.0,),
                                 target_freq_ghz=3.0).phase_results[120.0]["frequencies"] for i, g in enumerate((middle, one_side))]
    assert freqs[1] == pytest.approx(freqs[0], rel=2e-4, abs=0)


# ---------------------------------------------------------------------------
# 周期面の片方だけに触れる壁: 相手側の節点も壁の像
# ---------------------------------------------------------------------------
def _tube_cell(z0, z1, h, order=2):
    """r = 15 mm の筒（厚さ 0、z0〜z1 が PEC）のセル. 真空の 2 領域の境界に置く."""
    breaks = sorted({0.0, z0, z1, 30.0})
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 40.0), (0.0, 40.0)] + [(z, 15.0) for z in breaks]
    index = {z: 4 + i for i, z in enumerate(breaks)}
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="None"),
            Segment(id=1, type="line", point_indices=[1, index[30.0]], bc_name="None"),
            Segment(id=2, type="line", point_indices=[index[30.0], 2], bc_name="None"),
            Segment(id=3, type="line", point_indices=[2, 3], bc_name="PEC"),
            Segment(id=4, type="line", point_indices=[3, index[0.0]], bc_name="None"),
            Segment(id=5, type="line", point_indices=[index[0.0], 0], bc_name="None")]
    interface = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        interface.append(len(segs))
        segs.append(Segment(id=len(segs), type="line", point_indices=[index[a], index[b]],
                            bc_name="PEC" if z0 <= a and b <= z1 else "None"))
    loops = [Loop(id=0, segment_ids=[0, 1] + interface[::-1] + [5]), Loop(id=1, segment_ids=[2, 3, 4] + interface)]
    regions = [Region(id=0, name="A", outer_loop_id=0, material_tag="a"), Region(id=1, name="B", outer_loop_id=1, material_tag="b")]
    return MultiRegionGeometry(points=points, segments=segs, loops=loops, regions=regions, unit="mm", mesh_size=h)


def test_tube_touching_one_periodic_plane_matches_another_cut(tmp_path):
    """同じ周期構造の別の切り方（筒が周期面に触れない）が基準. 相手側の節点が自由だと h = 1.25 mm で −0.8% だった."""
    refs = {}
    for label, (z0, z1) in (("inside", (7.5, 22.5)), ("touching", (0.0, 15.0))):
        path = _write(_tube_cell(z0, z1, 1.25), tmp_path / f"{label}.msh")
        refs[label] = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=2, phase_shifts=(120.0,)).periodic[120.0].frequencies
    assert refs["touching"] == pytest.approx(refs["inside"], rel=5e-4, abs=0)


# ---------------------------------------------------------------------------
# 1 次要素の壁はメッシュの線要素で決まる
# ---------------------------------------------------------------------------
def _embedded_line_mesh(path, order):
    """1 つの領域の中に PEC の線を埋め込んだ外部メッシュ（r = 15 mm）. 線の両端は端面の点を共有する."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("embedded")
        h = 2.5e-3
        geo = gmsh.model.geo
        p0, p1, p2, p3 = (geo.addPoint(z, r, 0.0, h) for z, r in ((0.0, 0.0), (0.03, 0.0), (0.03, 0.04), (0.0, 0.04)))
        a, b = geo.addPoint(0.0, 0.015, 0.0, h), geo.addPoint(0.03, 0.015, 0.0, h)
        axis, outer = geo.addLine(p0, p1), geo.addLine(p2, p3)
        ends = [geo.addLine(p1, b), geo.addLine(b, p2), geo.addLine(p3, a), geo.addLine(a, p0)]
        inner = geo.addLine(a, b)
        surface = geo.addPlaneSurface([geo.addCurveLoop([axis, ends[0], ends[1], outer, ends[2], ends[3]])])
        geo.synchronize()
        gmsh.model.mesh.embed(1, [inner], 2, surface)
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [outer, inner] + ends), "PEC")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [surface]), "vacuum")
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


@pytest.mark.parametrize("order", [1, 2])
def test_pec_line_embedded_in_one_region(tmp_path, order):
    """節点だけ固定して辺を固定しないと 2.87 GHz になった（解析 4.99654 GHz、15 < r < 40 の TEM p = 1）."""
    path = _embedded_line_mesh(tmp_path / "m.msh", order)
    freqs = solve_hom_standing(load_mesh_hom(path, 0, order), num_modes=2).normal.frequencies
    assert freqs[0] == pytest.approx(C0 / (2.0 * 0.03) / 1e9, rel=2e-3)
    assert compute_hom_boundary(*_boundary_args(load_mesh_hom(path, 0, order)))[2]       # 壁損失にも線の辺が入る


def _boundary_args(mesh):
    return mesh.vertices, mesh.simplices, mesh.edge_index_map, mesh.physical_groups, 0, mesh.physical_edges


def _coax_two_regions(end_bc="PEC", split=True):
    """同軸 a = 10, b = 12, L = 100 mm（r 方向に要素 1 層）. z = 50 mm で真空の 2 領域に分ける."""
    if not split:
        return _polygon([(0.0, 10.0), (100.0, 10.0), (100.0, 12.0), (0.0, 12.0)], ["PEC", end_bc, "PEC", "PEC"], 2.5)
    points = [(0.0, 10.0), (50.0, 10.0), (100.0, 10.0), (100.0, 12.0), (50.0, 12.0), (0.0, 12.0)]
    segs = [Segment(id=0, type="line", point_indices=[0, 1], bc_name="PEC"), Segment(id=1, type="line", point_indices=[1, 4], bc_name="None"),
            Segment(id=2, type="line", point_indices=[4, 5], bc_name="PEC"), Segment(id=3, type="line", point_indices=[5, 0], bc_name="PEC"),
            Segment(id=4, type="line", point_indices=[1, 2], bc_name="PEC"), Segment(id=5, type="line", point_indices=[2, 3], bc_name=end_bc),
            Segment(id=6, type="line", point_indices=[3, 4], bc_name="PEC")]
    return MultiRegionGeometry(points=points, segments=segs, loops=[Loop(id=0, segment_ids=[0, 1, 2, 3]), Loop(id=1, segment_ids=[4, 5, 6, 1])],
                               regions=[Region(id=0, name="A", outer_loop_id=0, material_tag="a"), Region(id=1, name="B", outer_loop_id=1, material_tag="b")],
                               unit="mm", mesh_size=2.5)


@pytest.mark.parametrize("order", [1, 2])
def test_unlabeled_interface_across_a_one_element_gap_is_not_a_wall(tmp_path, order):
    """解析 TEM p = 1 は 1.49896 GHz. 界面を壁にすると基本モードが消えて 2.997 GHz から始まった."""
    path = _write(_coax_two_regions(), tmp_path / "m.msh", order=order)
    ref = C0 / (2.0 * 0.1) / 1e9
    freqs = solve_hom_standing(load_mesh_hom(path, 0, order), num_modes=2, target_freq_ghz=ref).normal.frequencies
    assert abs(_nearest_error(freqs, ref)) < 3e-3


@pytest.mark.parametrize("order", [1, 2])
def test_magnetic_end_across_a_one_element_gap_is_not_a_wall(tmp_path, order):
    """z = L が M-short: (2p−1)c/(4L). 0.749 GHz は HOM の閾値で落ちるので 2.24844 GHz を見る. 壁にすると 1.498 GHz の列になった."""
    path = _write(_coax_two_regions("M-short", split=False), tmp_path / "m.msh", order=order)
    ref = 3.0 * C0 / (4.0 * 0.1) / 1e9
    freqs = solve_hom_standing(load_mesh_hom(path, 0, order), num_modes=2, target_freq_ghz=ref).normal.frequencies
    assert abs(_nearest_error(freqs, ref)) < 3e-3


def test_first_order_post_counts_the_pec_interface_like_the_direct_call(tmp_path):
    """post は h5 から壁の辺を求め直す. 領域の番号に頼ると +41% になった（2 次要素と同じくシートの損失を数える）."""
    from axicavity_fem.cli.main import cli_main
    from axicavity_fem.shared.hdf5_io import read_results

    msh, raw, proc = tmp_path / "m.msh", tmp_path / "raw.h5", tmp_path / "proc.h5"
    _write(_tube_cell(0.0, 30.0, 2.5), msh, order=1)
    assert cli_main(["solve", "--type", "hom", "-m", str(msh), "--elem-order", "1", "--num-modes", "2", "--az-order", "0", "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "hom", "-i", str(raw), "-o", str(proc)]) == 0
    stored = read_results(proc)
    assert stored["mesh"]["physical_edges"]["PEC"].shape[1] == 2
    res = solve_hom_standing(load_mesh_hom(str(msh), 0, 1), num_modes=2)
    direct = compute_hom_parameters(res, conductivity=5.8e7).per_phase[0.0]
    assert stored["post_process"][0]["standing"][0]["Q"] == pytest.approx(direct[0].q_factor, rel=1e-6)


def test_old_first_order_file_warns_that_interior_lines_are_not_counted(tmp_path):
    """線要素を持たない古い h5 は外周の辺だけを壁に数える（2 次要素の古い h5 は中点で判定できるので警告しない）."""
    h5py = pytest.importorskip("h5py")
    from axicavity_fem.cli.main import cli_main

    for order, warns in ((1, True), (2, False)):
        msh, raw, proc = tmp_path / f"m{order}.msh", tmp_path / f"raw{order}.h5", tmp_path / f"proc{order}.h5"
        _write(_polygon(ACUTE_ONE, ["None", "PEC", "PEC", "PEC"], 5.0), msh, order=order)
        assert cli_main(["solve", "--type", "hom", "-m", str(msh), "--elem-order", str(order), "--num-modes", "1", "--az-order", "0", "-o", str(raw)]) == 0
        with h5py.File(raw, "a") as f:
            del f["mesh/physical_edges"]
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            assert cli_main(["post", "--type", "hom", "-i", str(raw), "-o", str(proc)]) == 0
        assert any("線要素" in str(w.message) for w in caught) == warns


def test_tm0_first_order_wall_loss_follows_the_lines(tmp_path):
    """M-short の端面が PEC の間を 1 要素で渡るセル: 線要素を見ないと端面を壁損失に数える."""
    from axicavity_fem.fem_tm0.post_process import _collect_pec_edges

    path = _write(_coax_two_regions("M-short", split=False), tmp_path / "m.msh", order=1)
    mesh = load_mesh(path, element_order=1)
    pec = classify_boundaries(mesh.physical_groups, mesh.nodes).pec_nodes
    edges = _collect_pec_edges(mesh.elements, pec, 1, mesh.physical_edges)
    z = mesh.nodes[:, 0]
    assert not [e for e in edges if abs(z[e[0]] - 0.1) < 1e-9 and abs(z[e[1]] - 0.1) < 1e-9]      # z = L の辺は無い
    assert [e for e in edges if abs(z[e[0]]) < 1e-9 and abs(z[e[1]]) < 1e-9]                       # z = 0 の PEC の辺はある


# ---------------------------------------------------------------------------
# 周期面の節点の座標のずれ
# ---------------------------------------------------------------------------
def _waveguide_mesh(tmp_path, n=1):
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 40.0), (0.0, 40.0)]
    return load_mesh_hom(_write(_polygon(points, ["None", "None", "PEC", "None"], 2.5), tmp_path / "m.msh"), n, 2)


def _plane_node_at_mid(mesh, z_end):
    z, r = mesh.vertices[:, 0], mesh.vertices[:, 1]
    plane = np.nonzero(np.abs(z - z_end) < 1e-9)[0]
    return plane[np.argmin(np.abs(r[plane] - 0.02))]


def test_small_rounding_of_plane_coordinates_is_absorbed(tmp_path):
    """外部メッシュの 1 µm のずれ: 3.0.0 と同じく動く. 許容を狭くすると n = 1 が −46% になった."""
    mesh = _waveguide_mesh(tmp_path)
    mesh.vertices = mesh.vertices.copy()
    mesh.vertices[_plane_node_at_mid(mesh, 0.0), 0] += 1e-6
    mesh.vertices[_plane_node_at_mid(mesh, 0.03), 0] -= 1e-6
    res = solve_hom_traveling(mesh, num_modes=1, phase_shifts=(120.0,))
    kc, beta = float(jnp_zeros(1, 1)[0]) / 0.04, np.radians(120.0) / 0.03
    assert res.periodic[120.0].frequencies[0] == pytest.approx(np.hypot(kc, beta) * C0 / (2e9 * np.pi), rel=5e-5)
    # post も同じ許容で面の要素を数える（狭いと面の要素を落として v_g が −17% になった）
    v_g = compute_hom_parameters(res, conductivity=5.8e7).per_phase[120.0][0].v_group_c
    assert v_g == pytest.approx(beta / np.hypot(kc, beta), abs=5e-3)


def test_larger_offset_of_a_plane_node_stops_instead_of_giving_a_wrong_value(tmp_path):
    mesh = _waveguide_mesh(tmp_path)
    mesh.vertices = mesh.vertices.copy()
    mesh.vertices[_plane_node_at_mid(mesh, 0.0), 0] += 2e-5
    mesh.vertices[_plane_node_at_mid(mesh, 0.03), 0] -= 2e-5
    with pytest.raises(RuntimeError, match="そろっていません|すぐそば"):
        solve_hom_traveling(mesh, num_modes=1, phase_shifts=(120.0,))


def test_first_order_e_short_end_is_fixed_but_has_no_wall_loss(tmp_path):
    """E-short の端面が PEC の間を 1 要素で渡る: 辺は固定するが、壁損失には数えない."""
    path = _write(_coax_two_regions("E-short", split=False), tmp_path / "m.msh", order=1)
    mesh = load_mesh_hom(path, 0, 1)
    z = mesh.vertices[:, 0]
    end = [index for (a, b), index in mesh.edge_index_map.items() if abs(z[a] - 0.1) < 1e-9 and abs(z[b] - 0.1) < 1e-9]
    assert end
    assert all(index in set(mesh.boundary_edge_indices) for index in end)
    assert not any(index in set(mesh.pec_loss_edge_indices) for index in end)


def test_first_order_axis_edges_are_fixed_for_n1_without_a_label(tmp_path):
    """軸の辺に線要素のラベルは無いが、n ≥ 1 では固定する（両端の辺は端面のラベルに従う）. n = 0 は固定しない."""
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 40.0), (0.0, 40.0)]
    path = _write(_polygon(points, ["None", "PEC", "PEC", "PEC"], 2.5), tmp_path / "m.msh", order=1)
    for n, expected in ((1, True), (0, False)):
        mesh = load_mesh_hom(path, n, 1)
        r = mesh.vertices[:, 1]
        axis = [index for (a, b), index in mesh.edge_index_map.items() if r[a] < 1e-9 and r[b] < 1e-9]
        fixed = set(mesh.boundary_edge_indices)
        assert axis and all((index in fixed) == expected for index in axis)


# ---------------------------------------------------------------------------
# 周期面のそばの壁: 正しいメッシュは止めず、周期面の辺を取り違えるものだけ止める
# ---------------------------------------------------------------------------
def _knife_cell(alpha_deg, h):
    """刃型のアイリス（L = 30, a = 15, b = 40 mm）. 壁が端面から α 傾く. 先端が周期面の上にある."""
    dz = 25.0 * np.tan(np.radians(alpha_deg))
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 15.0), (30.0 - dz, 40.0), (dz, 40.0), (0.0, 15.0)]
    return _polygon(points, ["None", "None", "PEC", "PEC", "PEC", "None"], h)


def _knife_reference(alpha_deg, h):
    """同じ周期構造をセルの中央で切ったもの（壁は端面に触れない）."""
    dz = 25.0 * np.tan(np.radians(alpha_deg))
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 40.0), (15.0 + dz, 40.0), (15.0, 15.0), (15.0 - dz, 40.0), (0.0, 40.0)]
    return _polygon(points, ["None", "None", "PEC", "PEC", "PEC", "PEC", "None"], h)


@pytest.mark.parametrize("order", [1, 2])
def test_wall_close_to_the_periodic_plane_is_not_stopped(tmp_path, order):
    """端面に斜めに達する壁: 先端の近くの節点は周期面のすぐそばにあるが、正しいメッシュ（検査で止めると解けなくなった）."""
    freqs = {}
    for label, geom in (("knife", _knife_cell(5.0, 1.0)), ("reference", _knife_reference(5.0, 1.0))):
        path = _write(geom, tmp_path / f"{label}.msh", order=order)
        freqs[label] = solve_hom_traveling(load_mesh_hom(path, 1, order), num_modes=1, phase_shifts=(120.0,)).periodic[120.0].frequencies
    assert freqs["knife"] == pytest.approx(freqs["reference"], rel=1e-3 if order == 1 else 2e-4, abs=0)


def test_shallow_wall_next_to_the_plane_is_not_stopped(tmp_path):
    """α = 2°, h = 1.25 mm: 壁の次の節点が面から 44 µm（許容の 5 倍以内）. 壁の点は面に近くてもよい."""
    freqs = {}
    for label, geom in (("knife", _knife_cell(2.0, 1.25)), ("reference", _knife_reference(2.0, 1.25))):
        path = _write(geom, tmp_path / f"{label}.msh")
        freqs[label] = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=1, phase_shifts=(120.0,)).periodic[120.0].frequencies
    assert freqs["knife"] == pytest.approx(freqs["reference"], rel=5e-4, abs=0)


def test_outer_node_just_outside_the_plane_stops(tmp_path):
    """面の節点が 40 µm 外れる（辺の中点は許容の外なので周期面の辺とは取り違えない）: 対にならない節点を残さない."""
    mesh = _waveguide_mesh(tmp_path)
    mesh.vertices = mesh.vertices.copy()
    mesh.vertices[_plane_node_at_mid(mesh, 0.0), 0] += 4e-5
    mesh.vertices[_plane_node_at_mid(mesh, 0.03), 0] -= 4e-5
    with pytest.raises(RuntimeError, match="そろっていません"):
        solve_hom_traveling(mesh, num_modes=1, phase_shifts=(120.0,))


def test_edge_almost_on_the_periodic_plane_stops(tmp_path):
    """α = 1°: 壁の辺の中点が許容の内側に入り、周期面の辺と取り違える（止めないと −1.1%）."""
    path = _write(_knife_cell(1.0, 1.0), tmp_path / "m.msh")
    with pytest.raises(RuntimeError, match="すぐそば"):
        solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=1, phase_shifts=(120.0,))


def test_thin_cell_with_a_small_length_still_uses_the_periodic_condition(tmp_path):
    """セル長が 10 µm より短くても周期条件が掛かる（位相によらない結果を黙って返していた）."""
    length = 0.005
    freqs = {}
    for theta in (30.0, 120.0):
        path = _write(_polygon([(0.0, 0.0), (length, 0.0), (length, 40.0), (0.0, 40.0)], ["None", "None", "PEC", "None"], 5.0), tmp_path / f"{theta}.msh")
        freqs[theta] = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=1, phase_shifts=(theta,)).periodic[theta].frequencies
    assert abs(freqs[30.0][0] - freqs[120.0][0]) > 1.0


def test_coarse_traveling_mesh_does_not_fail_on_the_number_of_modes(tmp_path):
    """複素の eigsh は k ≤ N − 2（k = N − 1 で生の TypeError になっていた）."""
    path = _write(_polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 40.0), (0.0, 40.0)], ["None", "None", "PEC", "None"], 50.0), tmp_path / "m.msh")
    res = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=4, phase_shifts=(120.0,))
    assert len(res.periodic[120.0].frequencies) >= 1


# ---------------------------------------------------------------------------
# メッシュの読込
# ---------------------------------------------------------------------------
def _convert_msh(src, dst, version):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(src))
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.write(str(dst))
    finally:
        gmsh.finalize()


def _mixed_labels_geo(path, order):
    """曲線の番号: 軸（ラベルなし）, z = L（E-short）, 外周（PEC）, z = 0（PEC）. MSH 2.2 は共有の節点が番号の小さい曲線にしか入らない."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("m")
        geo = gmsh.model.geo
        p = [geo.addPoint(z, r, 0.0, 3e-3) for z, r in ((0.0, 0.0), (0.035, 0.0), (0.035, 0.04), (0.0, 0.04))]
        lines = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [lines[1]]), "E-short")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, lines[2:]), "PEC")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [surface]), "vacuum")
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("order", [1, 2])
def test_msh22_with_mixed_labels_gives_the_same_walls_as_msh41(tmp_path, order):
    """MSH 2.2 では曲線の端の節点が抜けて角の壁の辺が壁でなくなり、Q が +5〜8% になった."""
    new, old = tmp_path / "m41.msh", tmp_path / "m22.msh"
    _mixed_labels_geo(new, order)
    _convert_msh(new, old, 2.2)
    a, b = load_mesh(new, element_order=order), load_mesh(old, element_order=order)
    for name in ("PEC", "E-short"):
        pa = {tuple(np.round(a.nodes[i], 12)) for i in a.physical_groups[name]}
        pb = {tuple(np.round(b.nodes[i], 12)) for i in b.physical_groups[name]}
        assert pa == pb, name


def _metal_region_mesh(path, order):
    """z 20〜35 mm の領域の 2D Physical の名前が "PEC" の外部メッシュ（領域を金属とみなす）."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("m")
        geo = gmsh.model.geo
        p = [geo.addPoint(z, r, 0.0, 2e-3) for z, r in ((0.0, 0.0), (0.02, 0.0), (0.035, 0.0), (0.035, 0.04), (0.02, 0.04), (0.0, 0.04))]
        axis_a, axis_b = geo.addLine(p[0], p[1]), geo.addLine(p[1], p[2])
        end, outer_b, outer_a, start = geo.addLine(p[2], p[3]), geo.addLine(p[3], p[4]), geo.addLine(p[4], p[5]), geo.addLine(p[5], p[0])
        middle = geo.addLine(p[4], p[1])
        left = geo.addPlaneSurface([geo.addCurveLoop([axis_a, -middle, outer_a, start])])
        right = geo.addPlaneSurface([geo.addCurveLoop([axis_b, end, outer_b, middle])])
        geo.synchronize()
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [end, outer_a, outer_b, start]), "PEC")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [left]), "vacuum")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [right]), "PEC")
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


def test_region_named_pec_is_metal_for_both_element_orders(tmp_path):
    """1 次要素だけ節点を固定して辺を固定せず、1.23 GHz の偽モードが出た. 2 次要素と同じく金属の領域になる."""
    freqs = {}
    for order in (1, 2):
        path = _metal_region_mesh(tmp_path / f"m{order}.msh", order)
        freqs[order] = solve_hom_standing(load_mesh_hom(path, 1, order), num_modes=1).normal.frequencies[0]
    assert freqs[1] == pytest.approx(freqs[2], rel=5e-3)


def test_element_order_that_does_not_match_the_mesh_gives_a_clear_error(tmp_path, capsys):
    from axicavity_fem.cli.main import cli_main

    msh = tmp_path / "m.msh"
    _write(_polygon(ACUTE_ONE, ["None", "PEC", "PEC", "PEC"], 10.0), msh, order=1)
    assert cli_main(["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "2", "--num-modes", "1", "-o", str(tmp_path / "r.h5")]) == 1
    assert "--elem-order 1" in capsys.readouterr().out
    assert cli_main(["solve", "--type", "hom", "-m", str(msh), "--elem-order", "2", "--num-modes", "1", "--az-order", "1", "-o", str(tmp_path / "r.h5")]) == 1


def test_old_first_order_file_without_interior_edges_does_not_warn(tmp_path):
    """内部の辺が無ければ外周の規則と線要素の結果は同じなので、古い h5 でも警告しない."""
    h5py = pytest.importorskip("h5py")
    import warnings as _warnings

    from axicavity_fem.cli.main import cli_main

    for solver, extra in (("hom", ["--az-order", "1"]), ("tm0", [])):
        msh, raw, proc = tmp_path / f"{solver}.msh", tmp_path / f"{solver}_raw.h5", tmp_path / f"{solver}_proc.h5"
        _write(_polygon([(0.0, 0.0), (30.0, 0.0), (30.0, 40.0), (0.0, 40.0)], ["None", "PEC", "PEC", "PEC"], 5.0), msh, order=1)
        assert cli_main(["solve", "--type", solver, "-m", str(msh), "--elem-order", "1", "--num-modes", "1", *extra, "-o", str(raw)]) == 0
        with h5py.File(raw, "a") as f:
            del f["mesh/physical_edges"]
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            assert cli_main(["post", "--type", solver, "-i", str(raw), "-o", str(proc)]) == 0
        assert not any("線要素" in str(w.message) for w in caught), solver


def test_old_first_order_tm0_file_warns_when_an_interior_edge_is_dropped(tmp_path):
    h5py = pytest.importorskip("h5py")
    import warnings as _warnings

    from axicavity_fem.cli.main import cli_main

    msh, raw, proc = tmp_path / "m.msh", tmp_path / "raw.h5", tmp_path / "proc.h5"
    _write(_polygon(ACUTE_ONE, ["None", "PEC", "PEC", "PEC"], 5.0), msh, order=1)
    assert cli_main(["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "1", "--num-modes", "1", "-o", str(raw)]) == 0
    with h5py.File(raw, "a") as f:
        del f["mesh/physical_edges"]
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(proc)]) == 0
    assert any("[TM0]" in str(w.message) and "線要素" in str(w.message) for w in caught)


def test_magnetic_wall_next_to_the_plane_is_not_stopped(tmp_path):
    """面取りした磁気壁の節点が面から 30 µm: PEC と同じく壁なので止めない."""
    dz = 0.03
    points = [(0.0, 0.0), (30.0, 0.0), (30.0, 35.0), (30.0 - dz, 40.0), (dz, 40.0), (0.0, 35.0)]
    path = _write(_polygon(points, ["None", "None", "M-short", "M-short", "M-short", "None"], 8.0), tmp_path / "m.msh")
    res = solve_hom_traveling(load_mesh_hom(path, 1, 2), num_modes=1, phase_shifts=(120.0,))
    assert res.periodic[120.0].frequencies[0] > 0.0


def test_hom_cell_without_end_planes_warns(tmp_path):
    """端面が 1 点のセルは周期条件が掛からない（TM0 と同じく知らせる）."""
    path = _write(_polygon([(0.0, 0.0), (30.0, 0.0), (15.0, 30.0)], ["None", "PEC", "PEC"], 5.0), tmp_path / "m.msh")
    with pytest.warns(UserWarning, match="位相によりません"):
        solve_hom_traveling(load_mesh_hom(path, 0, 2), num_modes=1, phase_shifts=(60.0, 120.0))


def test_msh22_region_named_pec_has_the_same_nodes_as_msh41(tmp_path):
    """2D のグループでも MSH 2.2 は境界の節点が抜けていた（n = 1 の最低が 1.33 GHz、正しくは 4.57 GHz）."""
    new, old = tmp_path / "m41.msh", tmp_path / "m22.msh"
    _metal_region_mesh(new, 2)
    _convert_msh(new, old, 2.2)
    a, b = load_mesh(new, element_order=2), load_mesh(old, element_order=2)
    pa = {tuple(np.round(a.nodes[i], 12)) for i in a.physical_groups["PEC"]}
    pb = {tuple(np.round(b.nodes[i], 12)) for i in b.physical_groups["PEC"]}
    assert pa == pb


def test_region_named_pec_is_not_reported_as_an_unknown_material(tmp_path):
    from axicavity_fem.shared.material_resolver import build_eps_r_per_element
    import warnings as _warnings

    mesh = load_mesh(_metal_region_mesh(tmp_path / "m.msh", 2), element_order=2)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        build_eps_r_per_element(mesh, {"vacuum": {"eps_r": 1.0}, "other": {"eps_r": 2.0}})
    assert not [w for w in caught if "[Material]" in str(w.message)]
