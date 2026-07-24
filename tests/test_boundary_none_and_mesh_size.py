"""ver2.3: メッシュ概要図まわりの純粋テスト。

- `bc_boundary_segments` が BC 未指定の外周辺を "None" として返す
- `mean_edge_length` が平均辺長を返す
"""

from __future__ import annotations

import numpy as np
import pytest

from axicavity_fem.shared.boundary_groups import (
    bc_boundary_segments,
    mean_edge_length,
)


def _unit_square_two_tris():
    """(0,0)-(1,0)-(1,1)-(0,1) を 2 三角形に分割した 1 次要素メッシュ。

    外周辺は 4 本: (0,1)下, (1,2)右, (2,3)上, (3,0)左。対角 (0,2) は内部辺。
    """
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    simplices = np.array([[0, 1, 2], [0, 2, 3]])
    return verts, simplices


def test_mean_edge_length_unit_square():
    verts, simplices = _unit_square_two_tris()
    # 一意な辺: 4 本の長さ1 + 対角 sqrt(2) = 5 本
    expected = (4 * 1.0 + np.sqrt(2.0)) / 5.0
    assert mean_edge_length(simplices, verts) == pytest.approx(expected)


def test_mean_edge_length_empty():
    assert mean_edge_length(np.empty((0, 3), dtype=int),
                            np.empty((0, 2))) == 0.0


def test_mean_edge_length_ignores_midside_nodes():
    """2 次要素 (6 節点) でも角節点の辺長のみで平均する。"""
    verts, simplices = _unit_square_two_tris()
    # 中点ノードを追加して (E, 6) にする（値は平均に影響しないはず）
    mids = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 1.0], [1.0, 0.5]])
    verts6 = np.vstack([verts, mids])
    simplices6 = np.array([[0, 1, 2, 4, 8, 5], [0, 2, 3, 5, 7, 6]])
    expected = (4 * 1.0 + np.sqrt(2.0)) / 5.0
    assert mean_edge_length(simplices6, verts6) == pytest.approx(expected)


def test_bc_segments_none_when_no_physical_groups():
    """physical_groups が無ければ外周辺はすべて "None"（黒線）になる。"""
    verts, simplices = _unit_square_two_tris()
    out = bc_boundary_segments(simplices, verts, {}, 1)
    assert out["PEC"] == [] and out["E-short"] == [] and out["M-short"] == []
    assert len(out["None"]) == 4          # 外周辺 4 本（対角は含まない）


def test_bc_segments_none_is_complement_of_assigned():
    """PEC を割り当てた辺は "None" に含まれない。"""
    verts, simplices = _unit_square_two_tris()
    # 上辺 (2,3) と 右辺 (1,2) の節点を PEC にする → 辺 (1,2),(2,3) が PEC
    pg = {"PEC": np.array([1, 2, 3])}
    out = bc_boundary_segments(simplices, verts, pg, 1)
    assert len(out["PEC"]) == 2
    # 残りの外周辺 (0,1) 下辺 と (3,0) 左辺 が None
    assert len(out["None"]) == 2
    none_mids = sorted(tuple(np.round(s.mean(axis=0), 6)) for s in out["None"])
    assert none_mids == [(0.0, 0.5), (0.5, 0.0)]


def test_bc_segments_axis_is_none():
    """軸 r=0 の辺に BC が無ければ "None" として返る（黒線対象）。"""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    simplices = np.array([[0, 1, 2], [0, 2, 3]])
    pg = {"PEC": np.array([1, 2, 3])}   # 軸 (0,1) 以外を PEC 相当に
    out = bc_boundary_segments(simplices, verts, pg, 1)
    # r=0 上の辺 (0,1) は None 側に入る
    ys = [float(s[:, 1].mean()) for s in out["None"]]
    assert any(abs(y) < 1e-12 for y in ys)
