"""残差チェック＋偽モード除外・補充（v2.3 安全網）の検証.

軸対称 TM0/HOM の M は特異に近く、ARPACK の返すモード集合には収束して
いない偽固有値が混入する（shared/eigensolver.py の _RESIDUAL_TOL の
コメント参照）。solve_eigenmodes_eigsh は全モードの相対残差をチェックし、
閾値超の偽モードを出力から除外、不足分は要求数を増やして解き直して補充する。

合格基準:
    - 正常に解けた場合は 1 回の求解で要求数のモードを返す
    - 偽モードが混入した場合、出力から除外され補充される
    - 1 回目が全滅でもリトライして正しい固有対を返す
    - 全試行が全滅でも例外にせず、警告を出して解を返す
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from axicavity_fem.shared import eigensolver
from axicavity_fem.shared.eigensolver import (
    _relative_residuals,
    solve_eigenmodes_eigsh,
)


def _spd_problem(n=40):
    """1 次元ラプラシアン K と単位質量 M（良条件の SPD ペア）."""
    main = 2.0 * np.ones(n)
    off = -np.ones(n - 1)
    K = sp.diags([off, main, off], [-1, 0, 1]).tocsr()
    M = sp.identity(n, format="csr")
    return K, M


def _garbage(K, k, seed):
    """偽の固有対（適当な値と乱数ベクトル）を返す."""
    rng = np.random.default_rng(seed)
    vals = np.linspace(10.0, 20.0, k)
    vecs = rng.standard_normal((K.shape[0], k))
    return vals, vecs


def test_good_solve_returns_without_retry(monkeypatch):
    """正常に解けた場合は 1 回の求解で要求数のモードを返すこと."""
    K, M = _spd_problem()
    calls = {"n": 0}
    orig = eigensolver._eigsh_once

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(eigensolver, "_eigsh_once", counting)
    vals, vecs = solve_eigenmodes_eigsh(K, M, num_eigenmodes=3, sigma=0.01)

    assert calls["n"] == 1
    assert len(vals) == 3
    assert _relative_residuals(K, M, vals, vecs).max() < 1e-8


def test_spurious_mode_excluded_and_refilled(monkeypatch, capsys):
    """偽モードが混入したとき、出力から除外され補充されること."""
    K, M = _spd_problem()
    calls = {"n": 0}
    orig = eigensolver._eigsh_once
    poison = 12.345   # 偽固有値（真のスペクトル [0, 4] の外）

    def with_spurious(K_, M_, k, *args, **kwargs):
        calls["n"] += 1
        vals, vecs = orig(K_, M_, k, *args, **kwargs)
        if calls["n"] == 1:
            # 最後のモードを偽固有対に差し替える（残差が巨大になる）
            vals = vals.copy()
            vecs = vecs.copy()
            vals[-1] = poison
            vecs[:, -1] = np.random.default_rng(0).standard_normal(K_.shape[0])
        return vals, vecs

    monkeypatch.setattr(eigensolver, "_eigsh_once", with_spurious)
    vals, vecs = solve_eigenmodes_eigsh(K, M, num_eigenmodes=3, sigma=0.01)

    assert calls["n"] == 2                      # 不足 → 解き直しで補充
    assert len(vals) == 3
    assert poison not in vals                   # 偽モードは出力に残らない
    assert _relative_residuals(K, M, vals, vecs).max() < 1e-8
    assert "解き直し" in capsys.readouterr().out


def test_garbage_first_attempt_triggers_retry(monkeypatch):
    """1 回目が全滅でもリトライして正しい固有対を返すこと."""
    K, M = _spd_problem()
    calls = {"n": 0}
    orig = eigensolver._eigsh_once

    def flaky(K_, M_, k, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _garbage(K_, k, seed=1)
        return orig(K_, M_, k, *args, **kwargs)

    monkeypatch.setattr(eigensolver, "_eigsh_once", flaky)
    vals, vecs = solve_eigenmodes_eigsh(K, M, num_eigenmodes=3, sigma=0.01)

    assert calls["n"] == 2
    assert len(vals) == 3
    assert _relative_residuals(K, M, vals, vecs).max() < 1e-8


def test_all_attempts_bad_returns_fallback_with_warning(monkeypatch, capsys):
    """全試行が全滅でも例外にせず、警告を出して解を返すこと."""
    K, M = _spd_problem()
    calls = {"n": 0}

    def always_garbage(K_, M_, k, *args, **kwargs):
        calls["n"] += 1
        return _garbage(K_, k, seed=calls["n"])

    monkeypatch.setattr(eigensolver, "_eigsh_once", always_garbage)
    vals, vecs = solve_eigenmodes_eigsh(K, M, num_eigenmodes=3, sigma=0.01)

    assert calls["n"] == eigensolver._SOLVE_ATTEMPTS
    assert vals is not None and len(vals) > 0
    assert "収束モードなし" in capsys.readouterr().out


def test_eigs_traveling_path_also_solves():
    """進行波経路（eigs、複素）も安全網込みで正しく解けること."""
    from axicavity_fem.shared.eigensolver import solve_eigenmodes_eigs

    K, M = _spd_problem()
    Kc = K.astype(np.complex128)
    Mc = M.astype(np.complex128)
    vals, vecs = solve_eigenmodes_eigs(Kc, Mc, num_eigenmodes=3, sigma=0.01)

    assert len(vals) == 3
    res = _relative_residuals(Kc, Mc, vals, vecs)
    assert res.max() < 1e-8
    # |λ| 昇順ソートの既存仕様が維持されていること
    assert np.all(np.diff(np.abs(vals)) >= 0)
