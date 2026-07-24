"""PARDISO shift-invert バックエンドの検証（ver2.3、pypardiso 未導入ならスキップ）.

OPinv は (K-σM)^{-1} の分解器を差し替えるだけで固有値問題は不変なので、
PARDISO 経路と SuperLU 経路の固有値は一致するはずである。

ただし TM0 定在波の縮約 pencil は M が特異（質量行列の r 重みが軸 r=0 で
消えるため半正定値）で、ARPACK shift-invert はランダム初期ベクトル次第で
モード集合全体を取りこぼすことがある（ver2.2 からの既知の性質。
test_fem_tm0_vs_ver1.py の注記参照。バックエンドによらず SuperLU でも起きる）。
そのため本テストは集合全体ではなく、production と同じ方針で
**TM010（解析解）に最も近いモードを対応付けて比較**し、
取りこぼした試行はリトライする。

合格基準:
    - 両経路とも TM010 を検出（離散化誤差 < 2%、3 回以内のリトライ許容）
    - 両経路の TM010 固有値が相対誤差 1e-8 以内で一致
    - 複素行列（進行波/HOM エルミート縮約）では PARDISO を使わず
      従来経路で解けること（pypardiso は実数専用のためのガード）
"""

from __future__ import annotations

import numpy as np
import pytest

pypardiso = pytest.importorskip("pypardiso")
pytest.importorskip("gmsh")

from axicavity_fem.fem_tm0.assembly import assemble_global_matrices  # noqa: E402
from axicavity_fem.fem_tm0.boundary import (                         # noqa: E402
    build_dirichlet_nodes,
    create_transformation_matrix,
)
from axicavity_fem.shared.boundary_groups import classify_boundaries  # noqa: E402
from axicavity_fem.shared.eigensolver import (                       # noqa: E402
    HAS_PARDISO,
    solve_eigenmodes_eigsh,
)
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region  # noqa: E402
from axicavity_fem.shared.mesh_loader import load_mesh               # noqa: E402
from axicavity_fem.shared.multi_region_model import (                # noqa: E402
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
    VACUUM_TAG,
)

_C0 = 299792458.0
_J01 = 2.4048255576957728   # ベッセル J0 の第1零点（TM010）
_MAX_TRIALS = 3             # ARPACK のモード取りこぼしに対するリトライ回数


def _pillbox_geom() -> MultiRegionGeometry:
    """半径 a=50mm, 長さ L=100mm の pillbox（test_fem_tm0_dielectric と同形状）."""
    a = 50.0
    L = 100.0
    pts = [(0.0, 0.0), (L, 0.0), (L, a), (0.0, a)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Region", outer_loop_id=0,
                    material_tag=VACUUM_TAG, eps_r=1.0)
    return MultiRegionGeometry(
        points=pts, segments=segs, loops=[loop], regions=[region],
        unit="mm", mesh_size=10.0,
    )


@pytest.fixture(scope="module")
def reduced_matrices(tmp_path_factory):
    """pillbox の縮約行列 K_red, M_red とシフト σ（定在波 TM0 相当）."""
    out = tmp_path_factory.mktemp("pardiso") / "pill.msh"
    export_msh_multi_region(_pillbox_geom(), out, mesh_order=2, verbose=0)
    mesh = load_mesh(out, element_order=2)

    K, M = assemble_global_matrices(mesh)
    classification = classify_boundaries(mesh.physical_groups, mesh.nodes)
    dirichlet_nodes = build_dirichlet_nodes(classification, mesh.nodes)
    T, _ = create_transformation_matrix(mesh.num_nodes, dirichlet_nodes)
    T = T.real
    K_red = (T.T @ K @ T).tocsr()
    M_red = (T.T @ M @ T).tocsr()

    r_max = float(np.max(mesh.nodes[:, 1]))
    sigma = (2 * np.pi * (_C0 / (4 * r_max)) / _C0) ** 2
    lam010 = (_J01 / r_max) ** 2    # TM010 の解析固有値 k² [1/m²]
    return K_red, M_red, sigma, lam010


def _solve_tm010(K_red, M_red, sigma, lam010, **kwargs):
    """production と同じ設定で解き、TM010 に最も近い固有値を返す.

    ARPACK がモード集合を取りこぼした試行（TM010 が 2% 以内に無い）は
    リトライし、_MAX_TRIALS 回失敗したらテスト失敗にする。
    """
    for _ in range(_MAX_TRIALS):
        # pardiso_min_dof=0: テストメッシュは小さいので閾値を外して
        # PARDISO 経路を強制的に通す
        vals, _ = solve_eigenmodes_eigsh(K_red, M_red, num_eigenmodes=8,
                                         sigma=sigma, which="LA",
                                         pardiso_min_dof=0, **kwargs)
        vals = np.real(vals)
        lam = vals[np.argmin(np.abs(vals - lam010))]
        if abs(lam - lam010) / lam010 < 2e-2:
            return lam
    pytest.fail(f"TM010 を {_MAX_TRIALS} 回の試行で検出できず "
                f"(kwargs={kwargs})")


def test_pardiso_matches_superlu(reduced_matrices):
    """PARDISO 経路と SuperLU 経路の TM010 固有値が一致すること."""
    assert HAS_PARDISO
    K_red, M_red, sigma, lam010 = reduced_matrices

    lam_pardiso = _solve_tm010(K_red, M_red, sigma, lam010,
                               use_pardiso=True)
    lam_superlu = _solve_tm010(K_red, M_red, sigma, lam010,
                               use_pardiso=False)

    # 同一行列の同一固有値なので、検出できていれば厳密に一致する
    assert lam_pardiso == pytest.approx(lam_superlu, rel=1e-8)


def test_complex_matrix_falls_back_to_superlu(reduced_matrices):
    """複素行列では PARDISO を使わず従来経路で解けること（実数専用ガード）.

    fem_hom のエルミート縮約や進行波では行列が複素になる。
    実数行列を複素型にキャストしたもの（値は同じ）で TM010 が変わらない
    ことを確認する。
    """
    K_red, M_red, sigma, lam010 = reduced_matrices

    lam_real = _solve_tm010(K_red, M_red, sigma, lam010,
                            use_pardiso=False)
    lam_cplx = _solve_tm010(K_red.astype(np.complex128),
                            M_red.astype(np.complex128),
                            sigma, lam010, use_pardiso=True)

    assert lam_cplx == pytest.approx(lam_real, rel=1e-8)
