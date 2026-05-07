"""
plot_hom_field.py
HOM FEM 計算結果 (HDF5) から電場ベクトルを可視化するスクリプト

定在波 (Normal) と進行波 (Periodic) の両方に対応する。
  - E_z, E_r: エッジ DOF → Nédélec 形状関数で要素中心に補間 → quiver 表示
  - E_theta  : 節点 DOF → tricontourf 表示 (n > 0 のみ)
  - |E_zr|   : E_z, E_r の大きさ → tricontourf 表示

使用例:
  # 定在波 n=0 のモード 0
  python plot_hom_field.py result.h5 --n 0 --mode 0

  # 定在波 n=1 のモード 2
  python plot_hom_field.py result.h5 --n 1 --mode 2

  # 進行波 n=0, 位相 60°, モード 0
  python plot_hom_field.py result.h5 --n 0 --mode 0 --periodic --phase 60

  # 進行波を位相 phi=45° での瞬時場として表示
  python plot_hom_field.py result.h5 --n 0 --mode 0 --periodic --phase 60 --snapshot 45
"""

import argparse
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from FEM_element_function import (calculate_edge_shape_functions,
                                  calculate_edge_shape_functions_2nd)

# プロジェクトルートを sys.path に追加して plot_common を解決
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from plot_common import STYLE  # noqa: E402


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------
def get_edge_orientation_sign(global1, global2):
    return 1 if global1 < global2 else -1


def reconstruct_edge_index_map(hf):
    """HDF5 から edge_index_map を復元する"""
    keys_array = hf["mesh/edge_map_keys"][:]    # (num_edges, 2)
    values_array = hf["mesh/edge_map_values"][:] # (num_edges,)
    edge_index_map = {
        tuple(keys_array[i]): int(values_array[i])
        for i in range(len(values_array))
    }
    return edge_index_map


def compute_Ez_Er_at_centers(edge_vectors, simplices, vertices, edge_index_map):
    """Nédélec 形状関数で要素中心の (E_z, E_r) を計算する

    Args:
        edge_vectors: エッジ DOF ベクトル (num_edges,), 実数または複素数
        simplices: 要素コーナーノード (num_elements x 3)
        vertices: 節点座標 (num_nodes x 2) [z, r]
        edge_index_map: {(n_i, n_j): edge_idx}

    Returns:
        centers: 要素中心座標 (num_elements x 2)
        Ez: 各要素中心の E_z (num_elements,)
        Er: 各要素中心の E_r (num_elements,)
    """
    is_complex = np.iscomplexobj(edge_vectors)
    dtype = complex if is_complex else float

    centers = np.zeros((len(simplices), 2))
    Ez = np.zeros(len(simplices), dtype=dtype)
    Er = np.zeros(len(simplices), dtype=dtype)

    for elem_idx, simplex in enumerate(simplices):
        verts = vertices[simplex]  # (3, 2)
        center = np.mean(verts, axis=0)
        centers[elem_idx] = center

        # 3辺のエッジ DOF と向き符号
        local_edges = [
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[2], simplex[0]),
        ]
        E_dof = np.zeros(3, dtype=dtype)
        for i, (n1, n2) in enumerate(local_edges):
            key = tuple(sorted((n1, n2)))
            if key in edge_index_map:
                edge_idx = edge_index_map[key]
                sign = get_edge_orientation_sign(simplex[i], simplex[(i+1) % 3])
                E_dof[i] = sign * edge_vectors[edge_idx]

        # Nédélec 形状関数で補間
        N1, N2, N3 = calculate_edge_shape_functions(center, verts)
        vec = E_dof[0] * N1 + E_dof[1] * N2 + E_dof[2] * N3
        Ez[elem_idx] = vec[0]
        Er[elem_idx] = vec[1]

    return centers, Ez, Er


def compute_Ez_Er_at_centers_2nd(edge_vectors, edge_vectors_lt, face_vectors,
                                  simplices, vertices, edge_index_map):
    """2次要素の形状関数 (N1-N8) で要素中心の (E_z, E_r) を計算する

    Args:
        edge_vectors:    CT/LN エッジ DOF (num_edges,)
        edge_vectors_lt: LT/LN エッジ DOF (num_edges,)
        face_vectors:    面内 DOF (2*num_elements,)
        simplices:       要素コーナーノード (num_elements x 3) — 6節点の場合は[:3]を渡す
        vertices:        節点座標 (num_nodes x 2) [z, r]
        edge_index_map:  {(n_i, n_j): edge_idx}

    Returns:
        centers: 要素中心座標 (num_elements x 2)
        Ez: 各要素中心の E_z (num_elements,)
        Er: 各要素中心の E_r (num_elements,)
    """
    is_complex = np.iscomplexobj(edge_vectors)
    dtype = complex if is_complex else float

    centers = np.zeros((len(simplices), 2))
    Ez = np.zeros(len(simplices), dtype=dtype)
    Er = np.zeros(len(simplices), dtype=dtype)

    for elem_idx, simplex in enumerate(simplices):
        verts = vertices[simplex]   # (3, 2)
        center = np.mean(verts, axis=0)
        centers[elem_idx] = center

        # 3辺のエッジ対応 (element_assembly と同じ順序)
        local_edges = [
            (simplex[0], simplex[1]),   # 辺0 → N1/N4
            (simplex[1], simplex[2]),   # 辺1 → N2/N5
            (simplex[2], simplex[0]),   # 辺2 → N3/N6
        ]

        ct_dofs = np.zeros(3, dtype=dtype)
        lt_dofs = np.zeros(3, dtype=dtype)
        for k, (n1, n2) in enumerate(local_edges):
            key = tuple(sorted((n1, n2)))
            if key in edge_index_map:
                edge_idx = edge_index_map[key]
                # CT/LN: エッジ向き符号を適用
                sign = 1 if n1 < n2 else -1
                ct_dofs[k] = sign * edge_vectors[edge_idx]
                # LT/LN: 符号なし
                lt_dofs[k] = edge_vectors_lt[edge_idx]

        # 面内 DOF (各要素に2つ)
        f0 = face_vectors[2 * elem_idx]
        f1 = face_vectors[2 * elem_idx + 1]

        # 2次形状関数 [N1, ..., N8] を要素中心で評価
        N = calculate_edge_shape_functions_2nd(center, verts)

        # 電場ベクトル = Σ dof_k * N_k
        vec = (ct_dofs[0] * N[0] + ct_dofs[1] * N[1] + ct_dofs[2] * N[2] +
               lt_dofs[0] * N[3] + lt_dofs[1] * N[4] + lt_dofs[2] * N[5] +
               f0          * N[6] + f1          * N[7])
        Ez[elem_idx] = vec[0]
        Er[elem_idx] = vec[1]

    return centers, Ez, Er


def get_pec_boundary_segments(hf, bc_path, edge_index_map_inv, vertices):
    """PEC 境界辺の線分座標リストを返す"""
    segments = []
    if bc_path in hf and "boundary_edges_pec" in hf[bc_path]:
        pec_indices = set(hf[bc_path]["boundary_edges_pec"][:].tolist())
        for edge_idx, (n1, n2) in edge_index_map_inv.items():
            if edge_idx in pec_indices:
                p1 = vertices[n1]
                p2 = vertices[n2]
                segments.append([(p1[0], p1[1]), (p2[0], p2[1])])
    return segments


# ---------------------------------------------------------------------------
# 描画関数
# ---------------------------------------------------------------------------
def plot_field(ax, vertices, simplices, centers, Ez, Er, E_theta,
               pec_segments, title, n_mode, cmap=None, vec_density=1.0):
    """電場を1つの axes に描画する

    n_mode=0: E_z, E_r quiver + |E_zr| コンター
    n_mode>0: E_z, E_r quiver + E_theta コンター (別 axes から呼ばれる想定)

    スタイルは TM0 側と統一 (plot_common.STYLE を参照)。
    """
    if cmap is None:
        cmap = STYLE.DEFAULT_CMAP
    tri = Triangulation(vertices[:, 0], vertices[:, 1], simplices)

    # --- スカラー場コンター (E_theta がある場合のみ) ---
    if E_theta is not None:
        scalar = E_theta.real
        clabel = r'$E_\theta$ [a.u.]'
        vmax = np.max(np.abs(scalar))
        if vmax < 1e-30:
            vmax = 1.0
        cf = ax.tricontourf(tri, scalar / vmax, levels=51, cmap=cmap,
                            vmin=-1.0, vmax=1.0)
        plt.colorbar(cf, ax=ax, label=clabel + ' (normalized)')

    # --- メッシュ (TM0 と同じスタイル) ---
    ax.triplot(tri, color=STYLE.MESH_COLOR,
               linewidth=STYLE.MESH_LW, alpha=STYLE.MESH_ALPHA)

    # --- PEC 境界 (黒線で TM0 と統一) ---
    if pec_segments:
        lc = LineCollection(pec_segments,
                            colors=STYLE.BORDER_COLOR,
                            linewidths=STYLE.BORDER_LW,
                            zorder=10, label='PEC')
        ax.add_collection(lc)

    # --- E_z, E_r ベクトル quiver ---
    n_elem = len(centers)
    step = 1
    idx = np.arange(0, n_elem, step)

    Ez_plot = Ez.real[idx]
    Er_plot = Er.real[idx]
    mag = np.sqrt(Ez_plot**2 + Er_plot**2)
    max_mag = np.max(mag) if np.max(mag) > 1e-30 else 1.0
    # 長さゼロの矢印（ドット）を除外
    nonzero = mag > max_mag * 1e-4
    idx = idx[nonzero]
    Ez_plot = Ez_plot[nonzero]
    Er_plot = Er_plot[nonzero]
    mag = mag[nonzero]
    z_range = vertices[:, 0].max() - vertices[:, 0].min()
    r_range = vertices[:, 1].max() - vertices[:, 1].min()
    arrow_scale = max(z_range, r_range) * 2.0 / np.sqrt(n_elem / step)

    q = ax.quiver(
        centers[idx, 0], centers[idx, 1],
        Ez_plot / max_mag, Er_plot / max_mag,
        angles='xy', scale_units='xy',
        scale=1.0 / arrow_scale,
        width=0.002, pivot=STYLE.QUIVER_PIVOT,
        color=STYLE.QUIVER_COLOR, alpha=STYLE.QUIVER_ALPHA, zorder=6,
        label=r'$(E_z, E_r)$'
    )

    ax.set_aspect('equal')
    ax.set_xlabel(STYLE.LABEL_Z)
    ax.set_ylabel(STYLE.LABEL_R)
    ax.grid(True, linestyle=STYLE.GRID_LS, alpha=STYLE.GRID_ALPHA)
    ax.set_title(title, fontsize=STYLE.TITLE_FONTSIZE)
    ax.legend(loc='upper right', fontsize=STYLE.LEGEND_FONTSIZE)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def load_mode_data(hf, n, mode_index, is_periodic, phase_deg):
    """HDF5 からモードデータを読み込む

    Returns:
        edge_vectors:    CT/LN エッジ DOF (num_edges,), 実数または複素数
        edge_vectors_lt: LT/LN エッジ DOF (2次要素のみ, num_edges,) or None
        face_vectors:    面内 DOF (2次要素のみ, 2*num_elements,) or None
        E_theta:         節点 E_theta (num_nodes,) or None
        freq_GHz:        共振周波数 [GHz]
        k2:              固有値 k^2
        elem_order:      要素次数 (1 or 2)
    """
    base = f"/results/n{n}"

    if not is_periodic:
        mode_path = f"{base}/Normal/mode_{mode_index}"
    else:
        group_name = f"PB_Phase_{phase_deg:05.1f}_deg".replace('.', '_')
        mode_path = f"{base}/Periodic/{group_name}/mode_{mode_index}"

    if mode_path not in hf:
        # モードデータがない場合の詳細エラーメッセージ
        if not is_periodic:
            check_path = f"{base}/Normal"
        else:
            group_name = f"PB_Phase_{phase_deg:05.1f}_deg".replace('.', '_')
            check_path = f"{base}/Periodic/{group_name}"

        if check_path in hf:
            available = list(hf[check_path].keys())
            mode_keys = [k for k in available if k.startswith('mode_')]
            if not mode_keys:
                raise KeyError(
                    f"HDF5 にモードデータがありません: {check_path}\n"
                    f"  → このファイルは計算が失敗または中断されたものです。\n"
                    f"  → HOM Solver を再実行して新しいファイルを作成してください。\n"
                    f"  (存在するグループ: {available})")
        available = list(hf[base].keys()) if base in hf else "(なし)"
        raise KeyError(f"HDF5 にパスが見つかりません: {mode_path}\n"
                       f"利用可能なグループ: {available}")

    grp = hf[mode_path]
    freq_GHz = float(grp.attrs.get("frequency_GHz", 0.0))
    k2 = float(grp.attrs.get("eigenvalue_k2", 0.0))
    elem_order = int(grp.attrs.get("elem_order", 1))

    if not is_periodic:
        # 定在波: 実数ベクトル
        edge_vectors = grp["edge_vectors"][:]
        edge_vectors_lt = grp["edge_vectors_lt"][:] if "edge_vectors_lt" in grp else None
        face_vectors    = grp["face_vectors"][:] if "face_vectors" in grp else None
        E_theta = grp["E_theta"][:] if "E_theta" in grp else None
    else:
        # 進行波: _re / _im データセット
        ev_re = grp["edge_vectors_re"][:]
        ev_im = grp["edge_vectors_im"][:]
        edge_vectors = ev_re + 1j * ev_im
        if "edge_vectors_lt_re" in grp:
            edge_vectors_lt = (grp["edge_vectors_lt_re"][:]
                               + 1j * grp["edge_vectors_lt_im"][:])
        else:
            edge_vectors_lt = None
        if "face_vectors_re" in grp:
            face_vectors = (grp["face_vectors_re"][:]
                            + 1j * grp["face_vectors_im"][:])
        else:
            face_vectors = None
        if "E_theta_re" in grp:
            E_theta = grp["E_theta_re"][:] + 1j * grp["E_theta_im"][:]
        else:
            E_theta = None

    return edge_vectors, edge_vectors_lt, face_vectors, E_theta, freq_GHz, k2, elem_order


def main():
    parser = argparse.ArgumentParser(
        description="HOM FEM 結果 HDF5 から電場ベクトルを可視化する")
    parser.add_argument("hdf5_file", help="入力 HDF5 ファイルパス")
    parser.add_argument("--n", type=int, default=0,
                        help="方位角モード次数 (デフォルト: 0)")
    parser.add_argument("--mode", type=int, default=0,
                        help="表示するモード番号 (デフォルト: 0)")
    parser.add_argument("--periodic", action="store_true",
                        help="進行波（周期境界）モードを表示")
    parser.add_argument("--phase", type=float, default=0.0,
                        help="進行波の位相 [deg] (--periodic 時に必要)")
    parser.add_argument("--snapshot", type=float, default=None,
                        help="進行波の瞬時場表示位相 phi [deg]。"
                             "Re(E * e^{j*phi}) を表示。指定なしで実部表示。")
    parser.add_argument("--density", type=float, default=0.3,
                        help="quiver 矢印の密度 (0~1, デフォルト: 0.3)")
    parser.add_argument("--save", type=str, default=None,
                        help="図を保存するファイルパス (.png 等)")
    args = parser.parse_args()

    with h5py.File(args.hdf5_file, 'r') as hf:
        # --- メッシュの読み込み ---
        vertices = hf["mesh/vertices"][:]      # (num_nodes, 2) [z, r]
        simplices_full = hf["mesh/simplices"][:]   # (num_elements, 3 or 6)
        edge_index_map = reconstruct_edge_index_map(hf)
        # 逆引き用: edge_idx → (n1, n2)
        edge_index_map_inv = {v: k for k, v in edge_index_map.items()}

        # --- モードデータの読み込み ---
        try:
            (edge_vectors, edge_vectors_lt, face_vectors,
             E_theta_raw, freq_GHz, k2, elem_order) = load_mode_data(
                hf, args.n, args.mode, args.periodic, args.phase)
        except KeyError as e:
            print(f"エラー: {e}", file=sys.stderr)
            sys.exit(1)

        # --- 進行波の位相回転（瞬時場） ---
        if args.periodic and args.snapshot is not None:
            phi = np.deg2rad(args.snapshot)
            edge_vectors = (edge_vectors * np.exp(1j * phi)).real
            if edge_vectors_lt is not None:
                edge_vectors_lt = (edge_vectors_lt * np.exp(1j * phi)).real
            if face_vectors is not None:
                face_vectors = (face_vectors * np.exp(1j * phi)).real
            if E_theta_raw is not None:
                E_theta_raw = (E_theta_raw * np.exp(1j * phi)).real
            snap_label = f", φ={args.snapshot:.1f}°"
        else:
            snap_label = ""

        # --- PEC 境界情報 ---
        if not args.periodic:
            bc_path = f"/results/n{args.n}/Normal/parameters"
        else:
            group_name = f"PB_Phase_{args.phase:05.1f}_deg".replace('.', '_')
            bc_path = f"/results/n{args.n}/Periodic/{group_name}/parameters"
        pec_segments = get_pec_boundary_segments(
            hf, bc_path, edge_index_map_inv, vertices)

    # 2次要素の場合はコーナーノードのみ使用 (要素ごとの辺計算に必要)
    simplices = simplices_full[:, :3]

    # --- E_z, E_r の計算 ---
    if elem_order == 2 and edge_vectors_lt is not None and face_vectors is not None:
        centers, Ez, Er = compute_Ez_Er_at_centers_2nd(
            edge_vectors, edge_vectors_lt, face_vectors,
            simplices, vertices, edge_index_map)
    else:
        centers, Ez, Er = compute_Ez_Er_at_centers(
            edge_vectors, simplices, vertices, edge_index_map)

    # --- タイトル ---
    wave_type = "Traveling" if args.periodic else "Standing"
    if args.periodic:
        wave_type += f" (θ={args.phase:.1f}°)"
    title_base = (f"{wave_type}, n={args.n}, Mode {args.mode}"
                  f"{snap_label}\n"
                  f"f = {freq_GHz:.4f} GHz  (k² = {k2:.4e} m⁻²)")

    # --- 描画レイアウト ---
    if args.n > 0 and E_theta_raw is not None:
        # n>0: (E_z, E_r) + E_theta を横並び
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title_base, fontsize=11)

        plot_field(axes[0], vertices, simplices, centers, Ez, Er,
                   None, pec_segments,
                   r'$(E_z, E_r)$ ベクトル場',
                   args.n, cmap='jet', vec_density=args.density)

        plot_field(axes[1], vertices, simplices, centers, Ez, Er,
                   E_theta_raw, pec_segments,
                   r'$E_\theta$ コンター + $(E_z, E_r)$ ベクトル',
                   args.n, cmap=STYLE.DEFAULT_CMAP, vec_density=args.density)
    else:
        # n=0: |E_zr| コンター + quiver 1枚
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        fig.suptitle(title_base, fontsize=11)
        plot_field(ax, vertices, simplices, centers, Ez, Er,
                   None, pec_segments,
                   r'$|E_{zr}|$ コンター + $(E_z, E_r)$ ベクトル',
                   args.n, cmap='jet', vec_density=args.density)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"図を保存しました: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
