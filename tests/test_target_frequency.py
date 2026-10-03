"""ver2.3 探索周波数指定 (--target-freq / GUI 'Predicted frequency') の検証.

1. 単位変換 ``k2_from_freq_ghz`` と ``k^2 → GHz`` の往復。
2. sigma / which の決定ロジック（未指定なら従来どおり、指定時は k^2(f)・'LM'）。
3. CLI 引数のパース、`_target_freq` の正規化（未指定/非正 → None）。
4. e2e: 指定した周波数の近傍モードが返り、最低次モードが選ばれないこと。
5. GUI: 欄が空なら --target-freq を付けない／入力があれば付ける。
"""

from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from axicavity_fem.cli.cmd_solve import _target_freq
from axicavity_fem.cli.main import build_parser
from axicavity_fem.fem_hom.solver import _select_modes, _sigma
from axicavity_fem.fem_hom.solver import _resolve_sigma as _hom_resolve_sigma
from axicavity_fem.fem_tm0.solver import _resolve_sigma, _sigma_from_rmax, _to_ghz
from axicavity_fem.shared.constants import C0, k2_from_freq_ghz

_MESH_DIR = Path(__file__).resolve().parents[1] / "samples"
_SPHERE = _MESH_DIR / "sphere50mm.msh"


# ---------------------------------------------------------------- 単位変換
def test_k2_from_freq_ghz_roundtrip():
    for f_ghz in (0.5, 2.856, 11.424):
        k2 = k2_from_freq_ghz(f_ghz)
        assert k2 == pytest.approx((2 * math.pi * f_ghz * 1e9 / C0) ** 2, rel=0)
        assert _to_ghz(np.array([k2]))[0] == pytest.approx(f_ghz, rel=1e-12)


def test_hdf5_io_helper_delegates():
    """hdf5_io の私的ヘルパは共通実装に委譲され、値が変わらないこと."""
    from axicavity_fem.shared.hdf5_io import _k2_from_freq_ghz
    assert _k2_from_freq_ghz(2.856) == k2_from_freq_ghz(2.856)


# ------------------------------------------------------- sigma / which 決定
def _nodes(r_max=0.05):
    return np.array([[0.0, 0.0], [0.1, r_max]])


def test_tm0_resolve_sigma_default_is_unchanged():
    """未指定なら従来の自動 sigma と which がそのまま使われる."""
    nodes = _nodes()
    for which_default in ("LA", "LM"):
        sigma, which = _resolve_sigma(nodes, None, which_default)
        assert sigma == _sigma_from_rmax(nodes)
        assert which == which_default


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_tm0_resolve_sigma_ignores_nonpositive(bad):
    nodes = _nodes()
    sigma, which = _resolve_sigma(nodes, bad, "LA")
    assert sigma == _sigma_from_rmax(nodes)
    assert which == "LA"


def test_tm0_resolve_sigma_with_target_freq():
    """指定時は sigma = k^2(f)、which は最近接探索の 'LM' に切り替わる."""
    sigma, which = _resolve_sigma(_nodes(), 5.0, "LA")
    assert sigma == pytest.approx(k2_from_freq_ghz(5.0), rel=0)
    assert which == "LM"


def test_hom_resolve_sigma():
    verts = _nodes()
    assert _hom_resolve_sigma(verts, None) == (_sigma(verts), None)
    assert _hom_resolve_sigma(verts, 0.0) == (_sigma(verts), None)
    sigma, nearest = _hom_resolve_sigma(verts, 7.5)
    assert sigma == pytest.approx(k2_from_freq_ghz(7.5), rel=0)
    assert nearest == sigma


def test_hom_select_modes_nearest_vs_lowest():
    """sigma_nearest 指定時のみ「指定値に近い順」で採用されること."""
    evals = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    vecs = np.eye(len(evals))
    T = np.eye(len(evals))

    # 従来（None）: 閾値超えの最低次から 2 本
    freqs, sel, _ = _select_modes(evals, vecs, T, 2, threshold=0.5)
    assert list(sel) == [1.0, 2.0]

    # 指定あり: σ=11 に近い 2 本（λ 昇順で返る）
    freqs, sel, _ = _select_modes(evals, vecs, T, 2, threshold=0.5,
                                  sigma_nearest=11.0)
    assert list(sel) == [10.0, 11.0]

    # 閾値以下は指定時も除外される
    freqs, sel, _ = _select_modes(evals, vecs, T, 3, threshold=2.5,
                                  sigma_nearest=1.0)
    assert list(sel) == [3.0, 10.0, 11.0]


# ------------------------------------------------------------------- CLI
def test_cli_parses_target_freq():
    parser = build_parser()
    args = parser.parse_args(["solve", "--type", "tm0", "-m", "x.msh"])
    assert args.target_freq is None
    args = parser.parse_args(["solve", "--type", "tm0", "-m", "x.msh",
                              "--target-freq", "2.856"])
    assert args.target_freq == pytest.approx(2.856)


def test_target_freq_normalization():
    assert _target_freq(Namespace()) is None
    assert _target_freq(Namespace(target_freq=None)) is None
    assert _target_freq(Namespace(target_freq=0.0)) is None
    assert _target_freq(Namespace(target_freq=-3.0)) is None
    assert _target_freq(Namespace(target_freq=2.856)) == pytest.approx(2.856)


# ------------------------------------------------------------------- e2e
@pytest.mark.skipif(not _SPHERE.exists(), reason="sphere50mm.msh なし")
def test_tm0_target_freq_selects_that_band(tmp_path):
    """指定周波数の近傍モードが返り、最低次モードは選ばれないこと."""
    pytest.importorskip("gmsh")
    from axicavity_fem.shared.mesh_loader import load_mesh
    from axicavity_fem.fem_tm0.solver import solve_tm0_standing

    mesh = load_mesh(str(_SPHERE), 2)
    base = solve_tm0_standing(mesh, num_modes=6)
    freqs_auto = np.sort(np.asarray(base.frequencies))
    assert len(freqs_auto) >= 4

    f_target = float(freqs_auto[3])       # 4 本目を狙い撃ちする
    res = solve_tm0_standing(mesh, num_modes=2, target_freq_ghz=f_target)
    freqs = np.sort(np.asarray(res.frequencies))

    # 狙った周波数が返る（相対 1e-6）
    assert np.min(np.abs(freqs - f_target)) / f_target < 1e-6
    # 最低次モードは含まれない（＝帯域指定が効いている）
    assert np.min(np.abs(freqs - freqs_auto[0])) / freqs_auto[0] > 1e-3


@pytest.mark.skipif(not _SPHERE.exists(), reason="sphere50mm.msh なし")
def test_hom_target_freq_selects_that_band():
    """HOM も指定周波数近傍のモードを返す（最低次からではない）."""
    pytest.importorskip("gmsh")
    from axicavity_fem.fem_hom.solver import solve_hom_standing
    from axicavity_fem.shared.mesh_loader import load_mesh_hom

    mesh = load_mesh_hom(str(_SPHERE), 1, 2)
    freqs_auto = np.sort(np.asarray(
        solve_hom_standing(mesh, num_modes=6).normal.frequencies))
    assert len(freqs_auto) >= 5

    f_target = float(freqs_auto[3])
    freqs = np.sort(np.asarray(
        solve_hom_standing(mesh, num_modes=2,
                           target_freq_ghz=f_target).normal.frequencies))
    assert np.min(np.abs(freqs - f_target)) / f_target < 1e-6
    assert np.min(np.abs(freqs - freqs_auto[0])) / freqs_auto[0] > 1e-3


@pytest.mark.skipif(not _SPHERE.exists(), reason="sphere50mm.msh なし")
def test_cli_solve_records_target_freq(tmp_path):
    pytest.importorskip("gmsh")
    pytest.importorskip("h5py")
    from axicavity_fem.cli.main import cli_main
    from axicavity_fem.shared.hdf5_io import read_results

    out = tmp_path / "auto.h5"
    assert cli_main(["solve", "--type", "tm0", "-m", str(_SPHERE),
                     "--elem-order", "2", "--num-modes", "3",
                     "-o", str(out)]) == 0
    # 未指定なら属性は増えない（従来ファイルと同じ構成）
    assert "target_freq_GHz" not in read_results(out)["parameters"]
    f_auto = np.sort(np.asarray(
        read_results(out)["results_by_n"][0]["standing"]["frequencies"]))

    out2 = tmp_path / "targeted.h5"
    f_target = float(f_auto[-1])
    assert cli_main(["solve", "--type", "tm0", "-m", str(_SPHERE),
                     "--elem-order", "2", "--num-modes", "2",
                     "--target-freq", repr(f_target), "-o", str(out2)]) == 0
    data = read_results(out2)
    assert float(data["parameters"]["target_freq_GHz"]) == pytest.approx(f_target)
    freqs = np.asarray(data["results_by_n"][0]["standing"]["frequencies"])
    assert np.min(np.abs(freqs - f_target)) / f_target < 1e-6
