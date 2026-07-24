"""cli/summary_txt.py の純粋テスト（メッシュ・gmsh 不要）。"""

from __future__ import annotations

import numpy as np

from axicavity_fem.cli.summary_txt import (
    render_post_summary,
    render_solve_summary,
    txt_path_for,
    write_post_summary,
    write_solve_summary,
)


_PARAMS = {
    "solver_type": "tm0",
    "mesh_file": "cavity.msh",
    "elem_order": 2,
    "num_modes": 3,
    "phase": "0",
}


def test_txt_path_for():
    assert txt_path_for("a/b/result.h5") == "a/b/result.txt"
    assert txt_path_for("x_processed.h5") == "x_processed.txt"


def test_render_solve_summary_standing():
    results_by_n = {0: {"standing": {"frequencies": np.array([1.23, 2.34, 3.45])}}}
    text = render_solve_summary(_PARAMS, results_by_n)
    assert "solve" in text
    assert "cavity.msh" in text
    assert "TM0" in text
    # 英語ヘッダ・日本語なし
    assert "analysis type" in text
    assert not any(ord(c) > 0x3000 for c in text)   # 全角文字を含まない
    assert "mode  0: f = 1.23 GHz" in text
    assert "mode  2: f = 3.45 GHz" in text


def test_render_solve_summary_high_precision():
    """周波数は 15 有効桁程度で出力される (10 桁以上の精度)。"""
    results_by_n = {0: {"standing": {"frequencies": np.array([2.8685631234567890])}}}
    text = render_solve_summary(_PARAMS, results_by_n)
    assert "2.86856312345679" in text          # 15 有効桁


def test_render_solve_summary_traveling_and_hom():
    results_by_n = {
        0: {"traveling": {120.0: {"frequencies": np.array([2.8])}}},
        1: {"standing": {"frequencies": np.array([3.1, 3.2])}},
    }
    text = render_solve_summary(_PARAMS, results_by_n)
    assert "traveling phase=120deg" in text
    assert "[n=0]" in text and "[n=1]" in text


def test_render_post_summary_has_q():
    post_by_n = {0: {"standing": [
        {"frequency_GHz": 1.23, "Q": 1.2e4, "R_over_Q": 55.0,
         "P_loss": 3.4e-2, "U_stored": 1e-9, "P_flow_zmin": 0.0,
         "V_eff": 1e5, "group_velocity": 1e8, "attenuation": 0.1},
    ]}}
    text = render_post_summary(_PARAMS, post_by_n)
    assert "f       = 1.23 GHz" in text
    assert "Q       = 1.200000e+04" in text
    assert "R/Q" in text and "P_loss" in text
    assert not any(ord(c) > 0x3000 for c in text)   # 日本語（全角）を含まない


def test_write_solve_summary_creates_file(tmp_path):
    h5 = tmp_path / "res.h5"
    results_by_n = {0: {"standing": {"frequencies": np.array([1.0])}}}
    path = write_solve_summary(str(h5), _PARAMS, results_by_n)
    assert path == str(tmp_path / "res.txt")
    assert (tmp_path / "res.txt").read_text(encoding="utf-8").count("mode") >= 1


def test_write_post_summary_creates_file(tmp_path):
    h5 = tmp_path / "res_processed.h5"
    post_by_n = {0: {"standing": [{"frequency_GHz": 1.0, "Q": 1e4}]}}
    path = write_post_summary(str(h5), _PARAMS, post_by_n)
    assert path == str(tmp_path / "res_processed.txt")
    assert "Q" in (tmp_path / "res_processed.txt").read_text(encoding="utf-8")
