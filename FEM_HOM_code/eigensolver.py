"""
eigensolver.py
固有値ソルバー: 密行列 eig, LOBPCG, eigsh

元の FEM_helmholtz_calclation.py から分離:
  - solve_eigenmodes       (密行列 scipy.linalg.eig)
  - solve_eigenmodes_lobpcg (scipy.sparse.linalg.lobpcg)
  - solve_eigenmodes_eigsh  (scipy.sparse.linalg.eigsh, shift-invert)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy.sparse.linalg import lobpcg, eigsh


def solve_eigenmodes(K, M, num_eigenmodes=5):
    """密行列による一般化固有値問題を解く (小規模問題向け)

    K @ v = lambda * M @ v

    Args:
        K: 剛性行列 (疎行列 or 密行列)
        M: 質量行列 (疎行列 or 密行列)
        num_eigenmodes: 出力する固有モード数

    Returns:
        eigenvalues: 固有値 (昇順, 実部でソート, num_eigenmodes 個)
        eigenvectors: 対応する固有ベクトル
    """
    K_dense = K.toarray() if hasattr(K, 'toarray') else np.asarray(K)
    M_dense = M.toarray() if hasattr(M, 'toarray') else np.asarray(M)

    eigenvalues, eigenvectors = eig(K_dense, M_dense)

    # 実部でソートし、小さい順に取り出す
    sorted_indices = np.argsort(eigenvalues.real)
    eigenvalues_out = eigenvalues[sorted_indices[:num_eigenmodes]]
    eigenvectors_out = eigenvectors[:, sorted_indices[:num_eigenmodes]]

    return eigenvalues_out, eigenvectors_out


def solve_eigenmodes_lobpcg(K, M, num_eigenmodes=5,
                             tol=None, maxiter=1000, seed=0):
    """LOBPCG による最小固有値問題 (大規模疎行列向け)

    Args:
        K: 剛性行列 (CSR)
        M: 質量行列 (CSR, 対称正定値)
        num_eigenmodes: 計算する最小固有モード数
        tol: 収束許容誤差 (None でデフォルト)
        maxiter: 最大反復回数
        seed: 乱数シード (再現性用, None で無効)

    Returns:
        eigenvalues: 最小固有値 (昇順)
        eigenvectors: 対応する固有ベクトル
        (失敗時は (None, None))
    """
    N = K.shape[0]

    if N == 0:
        print("Error: Matrices are empty.")
        return None, None
    if K.shape != (N, N) or M.shape != (N, N):
        print(f"Error: Matrix shapes K={K.shape}, M={M.shape} must match.")
        return None, None
    if num_eigenmodes <= 0:
        print("Error: num_eigenmodes must be positive.")
        return None, None
    if num_eigenmodes >= N:
        print(f"Warning: num_eigenmodes ({num_eigenmodes}) >= N ({N}). "
              f"Adjusting to {N - 1}.")
        if N > 1:
            num_eigenmodes = N - 1
        else:
            print("Error: Matrix dimension too small for LOBPCG.")
            return None, None

    if seed is not None:
        np.random.seed(seed)
    X_initial = np.random.rand(N, num_eigenmodes)

    try:
        eigenvalues, eigenvectors = lobpcg(
            K, X_initial, B=M,
            tol=tol, maxiter=maxiter,
            largest=False, verbosityLevel=0)

        if (eigenvalues.shape != (num_eigenmodes,) or
                eigenvectors.shape != (N, num_eigenmodes)):
            print("Warning: LOBPCG returned unexpected shapes.")

    except Exception as e:
        print(f"Error during LOBPCG execution: {e}")
        return None, None

    return eigenvalues, eigenvectors


def solve_eigenmodes_eigsh(K, M, num_eigenmodes=5, sigma=None, tol=1e-9):
    """eigsh (shift-invert) による一般化固有値問題 (推奨ソルバー)

    shift-invert モードで sigma 付近の固有値を効率的に計算する。

    Args:
        K: 剛性行列 (CSR)
        M: 質量行列 (CSR)
        num_eigenmodes: 計算する固有モード数
        sigma: シフト値 (k^2 の推定値)。None の場合は自動推定しない。
        tol: 収束許容誤差

    Returns:
        eigenvalues: 固有値 (sigma 付近、昇順)
        eigenvectors: 対応する固有ベクトル
    """
    N = K.shape[0]
    if num_eigenmodes >= N:
        num_eigenmodes = max(1, N - 1)

    eigenvalues, eigenvectors = eigsh(
        K, k=num_eigenmodes, M=M,
        sigma=sigma, which='LM', tol=tol)

    return eigenvalues, eigenvectors
