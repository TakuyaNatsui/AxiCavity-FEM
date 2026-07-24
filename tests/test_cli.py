"""段階5 検証: 統一 CLI (solve / post / info) の往復.

CLI ``solve`` が段階3/4 のソルバと同じ周波数を持つ v2 HDF5 を生成し、``post`` が
工学パラメータを追記、``info`` が動作することを確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("h5py")

from axicavity_fem.cli.main import cli_main  # noqa: E402
from axicavity_fem.shared.hdf5_io import is_v2, read_results  # noqa: E402

_SAMPLES = Path(__file__).resolve().parents[1] / "samples"
_CYL_2ND = _SAMPLES / "cylinder100mm.msh"      # 円筒空洞（2次要素, r=50mm, L=100mm）
_TRAVELING = _SAMPLES / "s-band_1cell.msh"     # 周期境界を持つ S バンド 1 セル

_J01 = 2.4048255576957728


@pytest.mark.skipif(not _CYL_2ND.exists(), reason="cylinder100mm.msh なし")
def test_cli_tm0_solve_post_info(tmp_path):
    out = tmp_path / "tm0.h5"
    rc = cli_main(["solve", "--type", "tm0", "-m", str(_CYL_2ND),
                   "--elem-order", "2", "--num-modes", "8", "-o", str(out)])
    assert rc == 0 and out.exists() and is_v2(out)

    data = read_results(out)
    assert data["solver_type"] == "tm0"
    freqs = np.asarray(data["results_by_n"][0]["standing"]["frequencies"])
    # 基本モード TM010 が含まれること（eigsh('LA') はモード集合が不安定なため
    # 厳密な集合一致ではなく解析 TM010 の検出で健全性を確認する）
    r_max = float(data["mesh"]["vertices"][:, 1].max())
    f010 = _J01 * 299792458.0 / (2 * np.pi * r_max) / 1e9
    assert np.min(np.abs(freqs - f010)) / f010 < 2e-2, f"freqs={np.sort(freqs)}"

    rc = cli_main(["post", "--type", "tm0", "-i", str(out)])
    assert rc == 0
    post = read_results(out)["post_process"]
    assert post[0]["standing"][0]["Q"] > 0
    assert post[0]["standing"][0]["P_loss"] > 0

    assert cli_main(["info", "-i", str(out)]) == 0


@pytest.mark.skipif(not _CYL_2ND.exists(), reason="cylinder100mm.msh なし")
def test_cli_hom_solve_post(tmp_path):
    from axicavity_fem.fem_hom.solver import solve_hom

    out = tmp_path / "hom.h5"
    rc = cli_main(["solve", "--type", "hom", "-m", str(_CYL_2ND),
                   "--elem-order", "2", "--az-order", "0", "1", "2",
                   "--num-modes", "3", "-o", str(out)])
    assert rc == 0 and out.exists()

    data = read_results(out)
    assert data["solver_type"] == "hom"
    for n in (0, 1, 2):
        freqs_cli = np.sort(data["results_by_n"][n]["standing"]["frequencies"])
        freqs_dir = np.sort(solve_hom(str(_CYL_2ND), n, element_order=2,
                                      num_modes=3).normal.frequencies)
        assert np.allclose(freqs_cli, freqs_dir, rtol=1e-6), f"n={n}"

    rc = cli_main(["post", "--type", "hom", "-i", str(out)])
    assert rc == 0
    post = read_results(out)["post_process"]
    assert post[1]["standing"][0]["Q"] > 0


@pytest.mark.skipif(not _TRAVELING.exists(), reason="s-band_1cell.msh なし")
def test_cli_tm0_traveling_roundtrip(tmp_path):
    out = tmp_path / "tw.h5"
    rc = cli_main(["solve", "--type", "tm0", "-m", str(_TRAVELING),
                   "--elem-order", "2", "--num-modes", "3", "-p", "120",
                   "-o", str(out)])
    assert rc == 0 and out.exists()

    data = read_results(out)
    trav = data["results_by_n"][0]["traveling"]
    assert 120.0 in trav
    assert np.iscomplexobj(trav[120.0]["eigenvectors"])
    assert len(trav[120.0]["frequencies"]) == 3

    rc = cli_main(["post", "--type", "tm0", "-i", str(out)])
    assert rc == 0
    post = read_results(out)["post_process"]
    assert "traveling" in post[0]
    assert post[0]["traveling"][120.0][0]["P_flow_zmin"] != 0.0
