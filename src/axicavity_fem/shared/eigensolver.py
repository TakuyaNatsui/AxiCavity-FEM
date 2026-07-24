"""一般化固有値問題ソルバ (eig / LOBPCG / eigsh / eigs).

ver1 ``FEM_HOM_code/eigensolver.py`` を移植し、進行波（複素・非エルミート）解析用に
``scipy.sparse.linalg.eigs`` ベースの :func:`solve_eigenmodes_eigs` を追加した版。

一般化固有値問題 ``K v = λ M v`` を扱う:
  - :func:`solve_eigenmodes`        密行列 ``scipy.linalg.eig`` (小規模)
  - :func:`solve_eigenmodes_lobpcg` ``scipy.sparse.linalg.lobpcg`` (大規模・最小固有値)
  - :func:`solve_eigenmodes_eigsh`  ``eigsh`` (実対称, shift-invert) ← 定在波 TM0
  - :func:`solve_eigenmodes_eigs`   ``eigs`` (複素・非エルミート) ← 進行波 TM0
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, lobpcg

# オプション依存: Intel MKL PARDISO（pip install pypardiso）。
# あれば実対称 shift-invert（eigsh）の (K-σM) 分解をマルチコアの
# PARDISO で行う。無ければ従来どおり eigsh 内部の SuperLU を使う。
# pypardiso は実数 float64 専用のため、複素の eigs 経路（進行波）は対象外。
try:
    import pypardiso
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False

# PARDISO は呼び出しごとに約 1 秒の固定コスト（解析フェーズ・スレッド起動）
# があり、2D 問題では小規模だと SuperLU より遅い。実測（pillbox TM0 定在波）
# ではクロスオーバーが約 3 万 DOF（N=23k で 0.6 倍、N=93k で 3.1 倍）
# のため、それ未満は自動的に SuperLU を使う。
_PARDISO_MIN_DOF = 20_000

# 残差チェック＋偽モード除外（v2.3 安全網）:
# 軸対称 TM0/HOM の質量行列 M は特異に近い（r 重みが軸 r=0 で消えることに
# 加え、2次要素では 4 点則（3次精度）が 5 次の被積分関数 r·G_i·G_j を
# 積分しきれず M が数値的に不定になる）。このため ARPACK が返すモード集合
# には、収束していない偽固有値が混入する（実測: 粗い pillbox・which='LA'
# ではどの実行でも数本混ざり、まれに集合全体が偽になる）。
# 正常モードの相対残差 ‖Kx−λMx‖/‖λMx‖ ≲ 1e-4、偽モード ≳ 1 と明確に
# 分かれるため、閾値超過のモードを出力から除外し、要求数に足りなければ
# 要求モード数を増やして解き直して補充する。補充しきれない場合は警告を
# 出して収束分のみ（それも無ければ従来どおりの解）を返す。
# 収束した固有対の数値は一切変わらない。
_RESIDUAL_TOL = 1e-2
_SOLVE_ATTEMPTS = 3
_K_EXTRA = 4          # リトライごとに追加要求するモード数


def _relative_residuals(K, M, eigenvalues, eigenvectors):
    """モードごとの相対残差 ‖Kx−λMx‖/‖λMx‖ (n_modes,) を返す."""
    res = np.empty(len(eigenvalues))
    for i in range(len(eigenvalues)):
        x = eigenvectors[:, i]
        mx = M @ x
        lam = eigenvalues[i]
        res[i] = (np.linalg.norm(K @ x - lam * mx)
                  / max(np.linalg.norm(lam * mx), 1e-300))
    return res


def _select_clean(vals, vecs, res, k, sigma, which):
    """残差が閾値未満のモードから選択基準に合う k 本以内を選ぶ.

    which='LM'（σ 最近接）は |λ−σ| の小さい順、それ以外（'LA' 等、
    σ の上側）は λ の小さい順に k 本まで選び、λ 昇順で返す。
    """
    ok = res < _RESIDUAL_TOL
    vals_c, vecs_c = vals[ok], vecs[:, ok]
    if len(vals_c) > k:
        if which == "LM" and sigma is not None:
            pick = np.argsort(np.abs(vals_c - sigma))[:k]
        else:
            pick = np.argsort(np.real(vals_c))[:k]
        vals_c, vecs_c = vals_c[pick], vecs_c[:, pick]
    order = np.argsort(np.real(vals_c))
    return vals_c[order], vecs_c[:, order]


def solve_eigenmodes(K, M, num_eigenmodes=5):
    """密行列による一般化固有値問題 ``K v = λ M v`` を解く (小規模問題向け).

    Returns:
        (eigenvalues, eigenvectors)。固有値は実部昇順、先頭 ``num_eigenmodes`` 個。
    """
    K_dense = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
    M_dense = M.toarray() if hasattr(M, "toarray") else np.asarray(M)

    eigenvalues, eigenvectors = eig(K_dense, M_dense)
    order = np.argsort(eigenvalues.real)[:num_eigenmodes]
    return eigenvalues[order], eigenvectors[:, order]


def solve_eigenmodes_lobpcg(K, M, num_eigenmodes=5,
                            tol=None, maxiter=1000, seed=0):
    """LOBPCG による最小固有値問題 (大規模疎行列, M は対称正定値).

    Returns:
        (eigenvalues, eigenvectors)。失敗時は ``(None, None)``。
    """
    N = K.shape[0]
    if N == 0:
        return None, None
    if K.shape != (N, N) or M.shape != (N, N):
        return None, None
    if num_eigenmodes <= 0:
        return None, None
    if num_eigenmodes >= N:
        if N <= 1:
            return None, None
        num_eigenmodes = N - 1

    if seed is not None:
        np.random.seed(seed)
    X_initial = np.random.rand(N, num_eigenmodes)

    try:
        eigenvalues, eigenvectors = lobpcg(
            K, X_initial, B=M, tol=tol, maxiter=maxiter,
            largest=False, verbosityLevel=0)
    except Exception:
        return None, None
    return eigenvalues, eigenvectors


def solve_eigenmodes_eigsh(K, M, num_eigenmodes=5, sigma=None,
                           which="LM", tol=1e-9, use_pardiso=True,
                           pardiso_min_dof=_PARDISO_MIN_DOF):
    """eigsh (shift-invert) による実対称一般化固有値問題.

    定在波 TM0 では ver1 と同じく ``which='LA'`` を指定する。

    shift-invert の (K-σM)^{-1} は、pypardiso が利用可能かつ問題が十分
    大きい（N >= pardiso_min_dof）なら PARDISO（MKL・マルチコア）で分解し
    ``OPinv`` として渡す。OPinv は分解器の差し替えにすぎず固有値問題
    そのものは不変なので、結果は SuperLU 経路と ARPACK の収束誤差の範囲で
    一致する（tests/test_pardiso.py で検証）。

    v2.3 安全網: 返された固有対の相対残差をチェックし、閾値
    ``_RESIDUAL_TOL`` 超の偽モードを出力から除外する。収束モードが
    要求数に満たない場合は要求モード数を増やして最大 ``_SOLVE_ATTEMPTS``
    回解き直して補充する（M が特異に近いことに起因する ARPACK の
    偽モード対策。tests/test_residual_retry.py で検証）。このため
    返るモード数が要求より少ないことがある（その場合は警告を表示）。

    Args:
        K, M: 実対称疎行列 (CSR)。
        num_eigenmodes: 計算する固有モード数。
        sigma: シフト値 (k^2 の推定値)。
        which: ARPACK の選択基準 ('LM', 'LA' など)。
        tol: 収束許容誤差。
        use_pardiso: False で PARDISO を使わず常に SuperLU を使う
            （バックエンド間の照合テスト用）。
        pardiso_min_dof: PARDISO を使う最小 DOF 数。小規模では
            PARDISO の固定コスト（約 1 秒）が支配的で SuperLU の方が
            速いため（既定 20,000 ≈ 実測クロスオーバー）。

    Returns:
        (eigenvalues, eigenvectors)。
    """
    N = K.shape[0]
    if num_eigenmodes >= N:
        num_eigenmodes = max(1, N - 1)

    # 残差チェック＋偽モード除外（_RESIDUAL_TOL の説明コメント参照）
    best_clean = None
    fallback = None
    fallback_res = np.inf
    for attempt in range(_SOLVE_ATTEMPTS):
        k_try = min(num_eigenmodes + _K_EXTRA * attempt, N - 1)
        vals, vecs = _eigsh_once(K, M, k_try, sigma, which, tol,
                                 use_pardiso, pardiso_min_dof)
        res = _relative_residuals(K, M, vals, vecs)
        res_max = float(res.max())
        if res_max < fallback_res:
            fallback, fallback_res = (vals, vecs), res_max
        vals_c, vecs_c = _select_clean(vals, vecs, res, num_eigenmodes,
                                       sigma, which)
        if len(vals_c) >= num_eigenmodes:
            return vals_c, vecs_c
        if best_clean is None or len(vals_c) > len(best_clean[0]):
            best_clean = (vals_c, vecs_c)
        if attempt < _SOLVE_ATTEMPTS - 1:
            print(f"情報: 収束モード {len(vals_c)}/{num_eigenmodes} 本 "
                  f"(残差 >= {_RESIDUAL_TOL:.0e} の偽モードを除外)"
                  f"要求数を増やして解き直し ({attempt + 2}/{_SOLVE_ATTEMPTS})")
    if len(best_clean[0]) > 0:
        print(f"警告: 収束モードが要求数に届かず "
              f"{len(best_clean[0])}/{num_eigenmodes} 本のみ返します")
        return best_clean
    print(f"警告: 収束モードなし（最小の最大相対残差 {fallback_res:.1e}）。"
          "従来どおり残差最小の解を返します（偽モードの可能性大）")
    return fallback


def _eigsh_once(K, M, k, sigma, which, tol, use_pardiso, pardiso_min_dof):
    """eigsh を 1 回実行する（バックエンド選択のみ、リトライなし）."""
    if (sigma is not None and use_pardiso and HAS_PARDISO
            and K.shape[0] >= pardiso_min_dof
            and not np.iscomplexobj(K.data) and not np.iscomplexobj(M.data)):
        return _eigsh_shift_invert_pardiso(K, M, k, sigma, which, tol)
    return eigsh(K, k=k, M=M, sigma=sigma, which=which, tol=tol)


def _eigsh_shift_invert_pardiso(K, M, k, sigma, which, tol):
    """(K-σM) を PARDISO で分解し OPinv として eigsh に渡す."""
    # PARDISO の要求形式: float64 の CSR、int32 インデックス
    A_shift = (K - sigma * M).tocsr()
    A_shift = sp.csr_matrix(
        (A_shift.data.astype(np.float64),
         A_shift.indices.astype(np.int32),
         A_shift.indptr.astype(np.int32)),
        shape=A_shift.shape)
    A_shift.sort_indices()

    solver = pypardiso.PyPardisoSolver()
    try:
        solver.factorize(A_shift)
    except Exception as exc:
        # PARDISO はオプション加速なので、分解に失敗したら SuperLU に戻す
        solver.free_memory(everything=True)
        print(f"PARDISO 分解に失敗、SuperLU に切り替え: {exc}")
        return eigsh(K, k=k, M=M, sigma=sigma, which=which, tol=tol)

    op_inv = LinearOperator(
        A_shift.shape, dtype=np.float64,
        matvec=lambda x: solver.solve(A_shift, x))
    print(f"shift-invert 分解: PARDISO (MKL), N = {A_shift.shape[0]}")
    try:
        return eigsh(K, k=k, M=M, sigma=sigma, OPinv=op_inv,
                     which=which, tol=tol)
    finally:
        solver.free_memory(everything=True)


def solve_eigenmodes_eigs(K, M, num_eigenmodes=5, sigma=None,
                          which="LM", tol=1e-9):
    """eigs (shift-invert) による複素・非エルミート一般化固有値問題.

    進行波 TM0（周期境界の位相因子により行列が複素・非エルミートになる）で使う。
    固有値は ``|λ|`` 昇順にソートして返す。

    Args:
        K, M: 複素疎行列 (CSR)。
        num_eigenmodes: 計算する固有モード数。
        sigma: シフト値 (k^2 の推定値)。
        which: ARPACK の選択基準。
        tol: 収束許容誤差。

    Returns:
        (eigenvalues, eigenvectors)。``|λ|`` 昇順。
    """
    N = K.shape[0]
    if num_eigenmodes >= N:
        num_eigenmodes = max(1, N - 1)

    # eigsh と同じ残差チェック＋偽モード除外（M 特異由来の偽モード対策）
    best_clean = None
    fallback = None
    fallback_res = np.inf
    for attempt in range(_SOLVE_ATTEMPTS):
        k_try = min(num_eigenmodes + _K_EXTRA * attempt, N - 2)
        eigenvalues, eigenvectors = eigs(
            K, k=k_try, M=M, sigma=sigma, which=which, tol=tol)
        res = _relative_residuals(K, M, eigenvalues, eigenvectors)
        res_max = float(res.max())
        if res_max < fallback_res:
            fallback, fallback_res = (eigenvalues, eigenvectors), res_max
        vals_c, vecs_c = _select_clean(eigenvalues, eigenvectors, res,
                                       num_eigenmodes, sigma, which)
        if len(vals_c) >= num_eigenmodes:
            eigenvalues, eigenvectors = vals_c, vecs_c
            break
        if best_clean is None or len(vals_c) > len(best_clean[0]):
            best_clean = (vals_c, vecs_c)
        if attempt < _SOLVE_ATTEMPTS - 1:
            print(f"情報: 収束モード {len(vals_c)}/{num_eigenmodes} 本 "
                  f"(残差 >= {_RESIDUAL_TOL:.0e} の偽モードを除外)"
                  f"要求数を増やして解き直し ({attempt + 2}/{_SOLVE_ATTEMPTS})")

        if len(best_clean[0]) > 0:
            print(f"警告: 収束モードが要求数に届かず "
                  f"{len(best_clean[0])}/{num_eigenmodes} 本のみ返します")
            eigenvalues, eigenvectors = best_clean
        else:
            print(f"警告: 収束モードなし（最小の最大相対残差 "
                  f"{fallback_res:.1e}）。従来どおり残差最小の解を返します"
                  "（偽モードの可能性大）")
            eigenvalues, eigenvectors = fallback

    order = np.argsort(np.abs(eigenvalues))
    return eigenvalues[order], eigenvectors[:, order]
