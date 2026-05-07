import argparse
import numpy as np
import h5py
import os
import sys
import shutil
import scipy.constants as const
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri

# プロジェクトルートと自身のディレクトリを sys.path に追加。
# プロジェクトルートを入れることで FEM_HOM_code をパッケージとして解決でき、
# field_calculator_hom 内の `from FEM_HOM_code.FEM_element_function import ...`
# が機能する (TM0 の FEM_element_function と名前衝突するため必要)。
_HOM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HOM_DIR, os.pardir))
for p in (_PROJECT_ROOT, _HOM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from FEM_HOM_code.FEM_element_function import (
    calculate_edge_shape_functions,
    calculate_edge_shape_functions_2nd,
    calculate_curl_edge_shape_functions_2nd,
    calculate_area_coordinates,
    calculate_quadratic_nodal_shape_functions,
    grad_area_coordinates,
    grad_quadratic_nodal_shape_functions,
    calculate_triangle_area_double
)
from FEM_HOM_code.gaussian_quadrature_triangle import integration_points_triangle

def calculate_triangle_area(verts):
    return abs(calculate_triangle_area_double(verts)) / 2.0

def reconstruct_edge_index_map(hf):
    keys_array = hf["mesh/edge_map_keys"][:]
    values_array = hf["mesh/edge_map_values"][:]
    edge_index_map = {tuple(keys_array[i]): int(values_array[i]) for i in range(len(values_array))}
    return edge_index_map

def get_pec_edges(hf, bc_path, simplices, edge_index_map):
    """P_loss 積分用 PEC 辺のリストを取得する。

    boundary_edges_pec_loss (物理 PEC 壁のみ) が存在すればそれを使用し、
    なければ boundary_edges_pec にフォールバックする。
    フォールバック時は z=0 軸上の辺 (E-short 対称境界) を除外する。
    """
    pec_edges = []
    if bc_path not in hf:
        return pec_edges

    vertices = hf["mesh/vertices"][:]

    # 優先: boundary_edges_pec_loss (物理 PEC 壁のみ)
    if "boundary_edges_pec_loss" in hf[bc_path]:
        pec_indices = set(hf[bc_path]["boundary_edges_pec_loss"][:].tolist())
        print(f"    P_loss 用 PEC 辺: boundary_edges_pec_loss を使用 ({len(pec_indices)} 辺)")
    elif "boundary_edges_pec" in hf[bc_path]:
        # 旧ファイル: E-short 辺 (両端が z≈0 の辺) を除外
        all_pec = set(hf[bc_path]["boundary_edges_pec"][:].tolist())
        keys_array = hf["mesh/edge_map_keys"][:]
        values_array = hf["mesh/edge_map_values"][:]
        inv_map = {int(values_array[i]): tuple(keys_array[i]) for i in range(len(values_array))}
        z_min_domain = np.min(vertices[:, 0])
        z_max_domain = np.max(vertices[:, 0])
        z_range = z_max_domain - z_min_domain
        pec_indices = set()
        excluded = 0
        for idx in all_pec:
            if idx in inv_map:
                n1, n2 = inv_map[idx]
                z1, z2 = vertices[n1, 0], vertices[n2, 0]
                # 両端が z_min (E-short 面) または z_max に極めて近い辺は対称境界 → 除外
                is_flat_zmin = (abs(z1 - z_min_domain) < z_range * 1e-4 and
                                abs(z2 - z_min_domain) < z_range * 1e-4)
                is_flat_zmax = (abs(z1 - z_max_domain) < z_range * 1e-4 and
                                abs(z2 - z_max_domain) < z_range * 1e-4)
                if is_flat_zmin or is_flat_zmax:
                    excluded += 1
                else:
                    pec_indices.add(idx)
        print(f"    P_loss 用 PEC 辺: boundary_edges_pec を使用、{excluded} 辺をフラット境界として除外 ({len(pec_indices)} 辺残)")
    else:
        return pec_edges

    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        local_edges = [
            (corner[0], corner[1]),
            (corner[1], corner[2]),
            (corner[2], corner[0]),
        ]
        for i, (n1, n2) in enumerate(local_edges):
            key = tuple(sorted((n1, n2)))
            if key in edge_index_map:
                edge_idx = edge_index_map[key]
                if edge_idx in pec_indices:
                    pec_edges.append((elem_idx, i))
    return pec_edges

def calc_p_flow_hom(simplices, vertices, edge_vectors, edge_vectors_lt,
                    face_vectors, E_theta_nodal, edge_index_map,
                    mesh_order, omega, n_mode):
    """z=z_min断面のポインティングベクトルz成分の積分から電力流を計算する。

    P = (pi/(omega*mu0)) * integral { Im[Er * curl_Ezr*]*r
                                     - n*Re[Ephi*Ez*]
                                     + Im[Ephi*(dEphi/dz)*]*r } dr

    H = j/(omega*mu0) * curl E より導出。FEM は e^{-jnφ} 規約を使うため
    H_r の n/r 項は n/(r*omega*mu0)*Ez となり、term2 は負号になる。
    n=0 では第2・第3項はゼロ。
    """
    mu0 = 4.0 * np.pi * 1e-7

    z_coords = vertices[:, 0]
    z_min = np.min(z_coords)
    z_tol = (np.max(z_coords) - z_min) * 1e-6
    z_min_set = set(np.where(np.abs(z_coords - z_min) < z_tol)[0])

    if mesh_order == 2:
        edge_defs = [(0, [0, 1, 3]), (1, [1, 2, 4]), (2, [2, 0, 5])]
    else:
        edge_defs = [(0, [0, 1]), (1, [1, 2]), (2, [2, 0])]

    def bary(eid, t):
        if eid == 0: return np.array([1-t, t,   0.0])
        if eid == 1: return np.array([0.0, 1-t, t  ])
        return             np.array([t,   0.0, 1-t])

    seen_edges = set()
    boundary = []
    for elem_idx, elem in enumerate(simplices):
        for eid, idxs in edge_defs:
            gnodes = tuple(int(elem[i]) for i in idxs)
            if all(nd in z_min_set for nd in gnodes):
                key = tuple(sorted(gnodes))
                if key not in seen_edges:
                    seen_edges.add(key)
                    # コーナーノード2点のみ使用（idxs[-1]は2次要素の中点なので不可）
                    boundary.append((eid, (int(elem[idxs[0]]), int(elem[idxs[1]])), elem, elem_idx))

    xi_pts = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    gweights = np.array([5.0/9, 8.0/9, 5.0/9])

    p_flow_total = 0.0

    for eid, (n1_idx, n2_idx), elem, elem_idx in boundary:
        corner = elem[:3].astype(int)
        verts = vertices[corner]
        area = calculate_triangle_area(verts)
        grad_L = grad_area_coordinates(verts)

        local_edge_pairs = [(corner[0], corner[1]), (corner[1], corner[2]), (corner[2], corner[0])]
        ct_dofs = np.zeros(3, dtype=complex)
        lt_dofs = np.zeros(3, dtype=complex)
        for k, (e1, e2) in enumerate(local_edge_pairs):
            key = tuple(sorted((e1, e2)))
            idx = edge_index_map[key]
            s = 1 if e1 < e2 else -1
            ct_dofs[k] = s * edge_vectors[idx]
            if mesh_order == 2 and edge_vectors_lt is not None:
                lt_dofs[k] = edge_vectors_lt[idx]
        f_dofs = face_vectors[2*elem_idx : 2*elem_idx+2] \
                 if mesh_order == 2 and face_vectors is not None else [0.0, 0.0]
        nod_dofs = E_theta_nodal[elem] if (n_mode != 0 and E_theta_nodal is not None) \
                   else np.zeros(len(elem))

        L_edge = abs(vertices[n2_idx, 1] - vertices[n1_idx, 1])
        if L_edge < 1e-15:
            continue

        edge_sum = 0.0
        for xi, w in zip(xi_pts, gweights):
            t = (xi + 1.0) / 2.0
            L = bary(eid, t)
            r = float(np.dot(L, verts[:, 1]))
            pt = np.array([float(np.dot(L, verts[:, 0])), r])

            if mesh_order == 2:
                N = calculate_edge_shape_functions_2nd(pt, verts)
                curls = calculate_curl_edge_shape_functions_2nd(pt, verts)
                G = calculate_quadratic_nodal_shape_functions(L)
                gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
                vec_zr = (ct_dofs[0]*N[0] + ct_dofs[1]*N[1] + ct_dofs[2]*N[2] +
                          lt_dofs[0]*N[3] + lt_dofs[1]*N[4] + lt_dofs[2]*N[5] +
                          f_dofs[0]*N[6] + f_dofs[1]*N[7])
                curl_Ezr = (ct_dofs[0]*curls[0] + ct_dofs[1]*curls[1] + ct_dofs[2]*curls[2] +
                            lt_dofs[0]*curls[3] + lt_dofs[1]*curls[4] + lt_dofs[2]*curls[5] +
                            f_dofs[0]*curls[6] + f_dofs[1]*curls[7])
            else:
                N = calculate_edge_shape_functions(pt, verts)
                A2 = 2.0 * area
                G = L
                gradG = np.array(grad_L)
                vec_zr = ct_dofs[0]*N[0] + ct_dofs[1]*N[1] + ct_dofs[2]*N[2]
                curl_Ezr = (ct_dofs[0] + ct_dofs[1] + ct_dofs[2]) * (2.0 / A2)

            E_z = vec_zr[0]
            E_r = vec_zr[1]
            if n_mode != 0 and E_theta_nodal is not None:
                E_phi = np.dot(nod_dofs, G)
                dEphi_dz = np.dot(nod_dofs, gradG[:, 0])
            else:
                E_phi = 0.0
                dEphi_dz = 0.0

            term1 = np.imag(E_r * np.conj(curl_Ezr)) * r
            # FEM は e^{-jnφ} 規約 → H_r の n/r 項の符号が反転 → term2 は負号
            term2 = -n_mode * np.real(E_phi * np.conj(E_z))
            term3 = np.imag(E_phi * np.conj(dEphi_dz)) * r

            edge_sum += w * (term1 + term2 + term3)

        p_flow_total += edge_sum * L_edge / 2.0

    return (np.pi / (omega * mu0)) * p_flow_total


def run_hom_post_process(input_file, conductivity=5.8e7):
    """
    HOM FEM 解析結果から工学パラメータを計算する。
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    output_file = os.path.splitext(input_file)[0] + "_processed.h5"
    if input_file != output_file:
        shutil.copy2(input_file, output_file)

    with h5py.File(output_file, 'a') as hf:
        # メッシュ基本情報
        vertices = hf["mesh/vertices"][:]
        simplices = hf["mesh/simplices"][:]
        edge_index_map = reconstruct_edge_index_map(hf)

        # 物理定数
        eps0 = const.epsilon_0
        mu0 = const.mu_0
        c_light = const.c
        cell_length = float(np.max(vertices[:, 0]) - np.min(vertices[:, 0]))

        # 利用可能な n モードを検索
        ns = []
        if "results" in hf:
            for key in hf["results"].keys():
                if key.startswith("n"):
                    try:
                        ns.append(int(key[1:]))
                    except: pass
        ns.sort()

        results_summary = {}

        for n in ns:
            base = f"/results/n{n}"

            # 処理対象グループを収集: [(label, mode_path), ...]
            groups_to_process = []
            if "Normal" in hf[base]:
                groups_to_process.append(("Normal", f"{base}/Normal"))
            if "Periodic" in hf[base]:
                for phase_key in sorted(hf[f"{base}/Periodic"].keys()):
                    groups_to_process.append(
                        (f"Periodic/{phase_key}", f"{base}/Periodic/{phase_key}")
                    )
            if not groups_to_process:
                continue

            # PECエッジは位相によらないので最初のグループから一度だけ取得
            first_bc_path = f"{groups_to_process[0][1]}/parameters"
            pec_edges = get_pec_edges(hf, first_bc_path, simplices, edge_index_map)

            # (group_label, m_key, freq_GHz, U, P_loss, q_factor, v_group_c, atten)
            n_results = []

            for group_label, mode_path in groups_to_process:
                modes = [k for k in hf[mode_path].keys() if k.startswith("mode_")]
                modes.sort(key=lambda x: int(x.split("_")[1]))

                for m_key in modes:
                    grp = hf[f"{mode_path}/{m_key}"]
                    freq_GHz = float(grp.attrs.get("frequency_GHz", 0.0))
                    omega = 2.0 * np.pi * freq_GHz * 1e9
                    elem_order = int(grp.attrs.get("elem_order", 1))

                    # DOF 取得
                    # Normal (定在波): edge_vectors (実数)
                    # Periodic (進行波): edge_vectors_re + 1j*edge_vectors_im
                    if "edge_vectors" in grp:
                        edge_vectors = grp["edge_vectors"][:]
                        edge_vectors_lt = grp["edge_vectors_lt"][:] if "edge_vectors_lt" in grp else None
                        face_vectors = grp["face_vectors"][:] if "face_vectors" in grp else None
                        E_theta_nodal = grp["E_theta"][:] if "E_theta" in grp else None
                        is_traveling = False
                    else:
                        edge_vectors = grp["edge_vectors_re"][:] + 1j * grp["edge_vectors_im"][:]
                        if "edge_vectors_lt_re" in grp:
                            edge_vectors_lt = grp["edge_vectors_lt_re"][:] + 1j * grp["edge_vectors_lt_im"][:]
                        else:
                            edge_vectors_lt = None
                        if "face_vectors_re" in grp:
                            face_vectors = grp["face_vectors_re"][:] + 1j * grp["face_vectors_im"][:]
                        else:
                            face_vectors = None
                        if "E_theta_re" in grp:
                            E_theta_nodal = grp["E_theta_re"][:] + 1j * grp["E_theta_im"][:]
                        else:
                            E_theta_nodal = None
                        is_traveling = True

                    # --- 蓄積エネルギー U の計算 ---
                    # U_total = eps0 * 2pi * integral(|E|^2 * r) dz dr
                    # 進行波は 1/2 倍 (時間平均)
                    # 注意: HOM行列は係数2を省略して構築されているため (element_assembly.py参照)、
                    # TM0コードと整合させるために係数1.0 (定在波) / 0.5 (進行波) を使う。
                    # TM0: M_TM0 = 2*M_phys → U = pi*mu0*(u^T M u)/2 = pi*mu0*∫|H|²r dA
                    # HOM: M_HOM = M_phys (係数2省略) → U = 2*pi*eps0*∫|E|²r dA で整合
                    U = 0.0

                    pts_data = integration_points_triangle[7]
                    coords_L = pts_data['coords']
                    weights = pts_data['weights']

                    for elem_idx in range(len(simplices)):
                        simplex = simplices[elem_idx]
                        corner = simplex[:3]
                        verts = vertices[corner]
                        area = calculate_triangle_area(verts)

                        local_edge_pairs = [(corner[0], corner[1]), (corner[1], corner[2]), (corner[2], corner[0])]
                        ct_dofs = np.zeros(3, dtype=complex)
                        lt_dofs = np.zeros(3, dtype=complex)
                        for k, (n1, n2) in enumerate(local_edge_pairs):
                            key = tuple(sorted((n1, n2)))
                            edge_idx = edge_index_map[key]
                            sign = 1 if n1 < n2 else -1
                            ct_dofs[k] = sign * edge_vectors[edge_idx]
                            if elem_order == 2 and edge_vectors_lt is not None:
                                lt_dofs[k] = edge_vectors_lt[edge_idx]

                        f_dofs = face_vectors[2*elem_idx : 2*elem_idx+2] if elem_order == 2 and face_vectors is not None else [0, 0]
                        nod_dofs = E_theta_nodal[simplex] if E_theta_nodal is not None else np.zeros(len(simplex))

                        for p in range(7):
                            L = coords_L[p]
                            w = weights[p]
                            z = np.dot(L, verts[:, 0])
                            r = np.dot(L, verts[:, 1])

                            if elem_order == 2:
                                N = calculate_edge_shape_functions_2nd(np.array([z, r]), verts)
                                G = calculate_quadratic_nodal_shape_functions(L)
                                # Ez, Er
                                vec_zr = (ct_dofs[0]*N[0] + ct_dofs[1]*N[1] + ct_dofs[2]*N[2] +
                                          lt_dofs[0]*N[3] + lt_dofs[1]*N[4] + lt_dofs[2]*N[5] +
                                          f_dofs[0]*N[6] + f_dofs[1]*N[7])
                            else:
                                N = calculate_edge_shape_functions(np.array([z, r]), verts)
                                G = L
                                vec_zr = (ct_dofs[0]*N[0] + ct_dofs[1]*N[1] + ct_dofs[2]*N[2])

                            E2 = np.abs(vec_zr[0])**2 + np.abs(vec_zr[1])**2 + np.abs(np.dot(nod_dofs, G))**2
                            # 定在波: u_coeff=1.0 (係数2省略の行列に対する補正込み)
                            # 進行波: u_coeff=0.5 (時間平均による1/2)
                            u_coeff = 1.0 if not is_traveling else 0.5
                            U += (u_coeff * eps0 * E2) * (2.0 * np.pi * r) * (w * area)

                    # --- 壁面損失 P_loss の計算 ---
                    # P_loss = 1/2 * Rs * integral(|H_tan|^2) dS
                    P_loss = 0.0
                    rs_ohm = np.sqrt(omega * mu0 / (2.0 * conductivity)) if conductivity > 0 else 0.0

                    q1d_pts = [-np.sqrt(0.6), 0, np.sqrt(0.6)]
                    q1d_w = [5/9, 8/9, 5/9]

                    for elem_idx, edge_local_idx in pec_edges:
                        simplex = simplices[elem_idx]
                        corner = simplex[:3]
                        verts = vertices[corner]
                        grad_L = grad_area_coordinates(verts)

                        n1_idx = corner[edge_local_idx]
                        n2_idx = corner[(edge_local_idx + 1) % 3]
                        p1 = vertices[n1_idx]
                        p2 = vertices[n2_idx]
                        edge_len = np.linalg.norm(p2 - p1)
                        tangent = (p2 - p1) / edge_len
                        normal = np.array([-tangent[1], tangent[0]])

                        local_edge_pairs = [(corner[0], corner[1]), (corner[1], corner[2]), (corner[2], corner[0])]
                        ct_dofs = np.zeros(3, dtype=complex)
                        lt_dofs = np.zeros(3, dtype=complex)
                        for k, (e1, e2) in enumerate(local_edge_pairs):
                            key = tuple(sorted((e1, e2)))
                            idx = edge_index_map[key]
                            s = 1 if e1 < e2 else -1
                            ct_dofs[k] = s * edge_vectors[idx]
                            if elem_order == 2 and edge_vectors_lt is not None:
                                lt_dofs[k] = edge_vectors_lt[idx]
                        f_dofs = face_vectors[2*elem_idx : 2*elem_idx+2] if elem_order == 2 and face_vectors is not None else [0, 0]
                        nod_dofs = E_theta_nodal[simplex] if E_theta_nodal is not None else np.zeros(len(simplex))

                        for xi, wi in zip(q1d_pts, q1d_w):
                            v_s = (xi + 1.0) / 2.0
                            pt = p1 + v_s * (p2 - p1)
                            L = calculate_area_coordinates(pt, verts)
                            r_e = pt[1]

                            if elem_order == 2:
                                N = calculate_edge_shape_functions_2nd(pt, verts)
                                curls = calculate_curl_edge_shape_functions_2nd(pt, verts)
                                G = calculate_quadratic_nodal_shape_functions(L)
                                gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
                                curl_Ezr = (ct_dofs[0]*curls[0] + ct_dofs[1]*curls[1] + ct_dofs[2]*curls[2] +
                                            lt_dofs[0]*curls[3] + lt_dofs[1]*curls[4] + lt_dofs[2]*curls[5] +
                                            f_dofs[0]*curls[6] + f_dofs[1]*curls[7])
                                Ez_p = (ct_dofs[0]*N[0][0] + ct_dofs[1]*N[1][0] + ct_dofs[2]*N[2][0] +
                                        lt_dofs[0]*N[3][0] + lt_dofs[1]*N[4][0] + lt_dofs[2]*N[5][0] +
                                        f_dofs[0]*N[6][0] + f_dofs[1]*N[7][0])
                                Er_p = (ct_dofs[0]*N[0][1] + ct_dofs[1]*N[1][1] + ct_dofs[2]*N[2][1] +
                                        lt_dofs[0]*N[3][1] + lt_dofs[1]*N[4][1] + lt_dofs[2]*N[5][1] +
                                        f_dofs[0]*N[6][1] + f_dofs[1]*N[7][1])
                            else:
                                N = calculate_edge_shape_functions(pt, verts)
                                A2 = 2.0 * calculate_triangle_area(verts)
                                curl_Ezr = (ct_dofs[0] + ct_dofs[1] + ct_dofs[2]) * (2.0 / A2)
                                G = L
                                gradG = np.array(grad_L)
                                Ez_p = (ct_dofs[0]*N[0][0] + ct_dofs[1]*N[1][0] + ct_dofs[2]*N[2][0])
                                Er_p = (ct_dofs[0]*N[0][1] + ct_dofs[1]*N[1][1] + ct_dofs[2]*N[2][1])

                            Et_p = np.dot(nod_dofs, G)
                            dEt_dz = np.dot(nod_dofs, gradG[:, 0])
                            dEt_dr = np.dot(nod_dofs, gradG[:, 1])

                            r_s = max(r_e, 1e-9)
                            drEt_dr = Et_p + r_s * dEt_dr

                            # ∇×E (cylindrical coords, e^(jnφ), ∂/∂φ → jn)
                            curlE_r = 1j * (n / r_s) * Ez_p - dEt_dz
                            curlE_t = curl_Ezr
                            # 軸上 (r→0) では 1/r 項が数値的に発散するため Hz=Hr=0 を課す
                            if r_e > 1e-9:
                                curlE_z = (1.0 / r_s) * drEt_dr - 1j * (n / r_s) * Er_p
                            else:
                                curlE_z = 0.0

                            coef = 1j / (omega * mu0)

                            # 磁場の強さの2乗 |H|^2
                            if n == 0:
                                # n=0 (TM0モード) では H_theta (curlE_t) のみが寄与
                                H2 = np.abs(coef * curlE_t)**2
                            else:
                                # n > 0 では全成分を考慮
                                H_tan_vec_sq = np.abs(coef * curlE_t)**2 + \
                                               np.abs(coef * (-curlE_z * normal[1] + curlE_r * normal[0]))**2
                                H2 = H_tan_vec_sq

                            # P_loss_avg = 1/2 * Rs * integral(|H_peak|^2) dV
                            # dS = 2 * pi * r * dl
                            # => P_loss_avg = pi * Rs * integral(H2 * r) dl
                            # wi の合計は 2.0 なので、0.5 * wi で積分区間 L に正規化
                            P_loss += (np.pi * rs_ohm * H2 * r_s) * (0.5 * wi * edge_len)

                    q_factor = (omega * U / P_loss) if P_loss > 0 else 0.0

                    # --- 進行波: 電力流・群速度・位相速度・減衰定数 ---
                    if is_traveling:
                        p_flow = calc_p_flow_hom(
                            simplices, vertices, edge_vectors, edge_vectors_lt,
                            face_vectors, E_theta_nodal, edge_index_map,
                            elem_order, omega, n)
                        v_group_c = (p_flow * cell_length / U) / c_light if U > 0 else 0.0
                        # 位相速度: Vp = omega / beta, beta = theta_rad / cell_length
                        # group_label 例: "Periodic/PB_Phase_120_0_deg"
                        phase_key = group_label.split("/")[-1]  # "PB_Phase_120_0_deg"
                        theta_deg = float(phase_key.replace("PB_Phase_", "").replace("_deg", "").replace("_", "."))
                        theta_rad = np.deg2rad(theta_deg)
                        v_phase_c = (omega * cell_length) / (theta_rad * c_light) if theta_rad > 1e-10 else 0.0
                        atten = P_loss / (2.0 * p_flow * cell_length) if p_flow != 0 else 0.0
                    else:
                        p_flow, v_group_c, v_phase_c, atten = 0.0, 0.0, 0.0, 0.0

                    eng_grp = grp.require_group("engineering_parameters")
                    eng_grp.attrs["stored_energy"] = U
                    eng_grp.attrs["p_loss"] = P_loss
                    eng_grp.attrs["q_factor"] = q_factor
                    if is_traveling:
                        eng_grp.attrs["p_flow"] = p_flow
                        eng_grp.attrs["v_group_c"] = v_group_c
                        eng_grp.attrs["v_phase_c"] = v_phase_c
                        eng_grp.attrs["attenuation"] = atten

                    if is_traveling:
                        print(f"  n={n}, [{group_label}] {m_key}: f={freq_GHz:.6f} GHz, "
                              f"vg={v_group_c:.4f}c, vp={v_phase_c:.4f}c, atten={atten:.3e} Np/m, "
                              f"U={U:.3e} J, P_loss={P_loss:.3e} W, Q={q_factor:.2e}")
                    else:
                        print(f"  n={n}, [{group_label}] {m_key}: f={freq_GHz:.6f} GHz, "
                              f"U={U:.3e} J, P_loss={P_loss:.3e} W, Q={q_factor:.2e}")
                    n_results.append((group_label, m_key, freq_GHz, U, P_loss, q_factor,
                                      v_group_c, v_phase_c, atten))

            results_summary[n] = n_results

        summary_file = os.path.splitext(output_file)[0] + "_parameters.txt"
        with open(summary_file, 'w') as f:
            f.write("HOM Post-Process Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Conductivity: {conductivity:.2e} S/m\n\n")
            for n, res in results_summary.items():
                f.write(f"Azimuthal Order n = {n}\n")
                current_label = None
                for label, m, freq, u, pl, q, vg, vp, at in res:
                    if label != current_label:
                        current_label = label
                        f.write(f"\n  [{label}]\n")
                        is_tw = label.startswith("Periodic")
                        if is_tw:
                            f.write(f"  {'Mode':<10} {'Freq [GHz]':<14} {'Vg [c]':<12} {'Vp [c]':<12} "
                                    f"{'Atten [Np/m]':<14} {'U [J]':<12} {'P_loss [W]':<12} {'Q Factor':<12}\n")
                            f.write("  " + "-" * 100 + "\n")
                        else:
                            f.write(f"  {'Mode':<10} {'Freq [GHz]':<14} {'U [J]':<12} "
                                    f"{'P_loss [W]':<12} {'Q Factor':<12}\n")
                            f.write("  " + "-" * 62 + "\n")
                    is_tw = label.startswith("Periodic")
                    if is_tw:
                        f.write(f"  {m:<10} {freq:<14.6f} {vg:<12.4f} {vp:<12.4f} "
                                f"{at:<14.4e} {u:<12.3e} {pl:<12.3e} {q:<12.2e}\n")
                    else:
                        f.write(f"  {m:<10} {freq:<14.6f} {u:<12.3e} {pl:<12.3e} {q:<12.2e}\n")
                f.write("\n")
        print(f"Saved summary to: {summary_file}")

# ============================================================================
# レポート生成 (可視化 + HTML)
# ============================================================================
def run_hom_report_generation(processed_file, skip_anim=True):
    """
    HOM の processed.h5 からレポート（PNG/GIF + HTML）を生成する。

    skip_anim=True (デフォルト) は GIF アニメーションを生成しない。
    GUI 側のチェックボックスで明示的に True 指定された場合のみアニメを作る。
    """
    import datetime
    from field_calculator_hom import HOMFieldCalculator
    from plot_utils_hom import FEMPlotterHOM

    if not os.path.exists(processed_file):
        print(f"Error: {processed_file} not found."); return

    report_dir = os.path.splitext(processed_file)[0] + "_report"
    os.makedirs(report_dir, exist_ok=True)

    calc = HOMFieldCalculator(processed_file)
    if not calc.available_ns:
        print("No HOM results found in the file.")
        return

    plotter = FEMPlotterHOM(calc)

    # メッシュ概要図
    fig_m, ax_m = plt.subplots(figsize=(10, 6))
    tri = matplotlib.tri.Triangulation(calc.vertices[:, 0], calc.vertices[:, 1],
                                        calc.simplices[:, :3])
    ax_m.triplot(tri, color='gray', linewidth=0.2, alpha=0.5)
    ax_m.set_aspect('equal')
    ax_m.set_xlabel('z [m]')
    ax_m.set_ylabel('r [m]')
    ax_m.set_title(f"Mesh Overview: {len(calc.vertices)} vertices, "
                   f"{len(calc.simplices)} elements")
    fig_m.savefig(os.path.join(report_dir, "mesh_overview.png"), dpi=120)
    plt.close(fig_m)

    # n / phase / mode を網羅して画像生成
    # results_all = { n: { phase_label: [ {mode_idx, freq, U, P_loss, Q, vg, vp, atten,
    #                                      images: {...}}, ...] } }
    results_all = {}

    for n in calc.available_ns:
        calc.set_n_and_phase(n, 0.0)
        analysis_type = calc.analysis_type
        phases = calc.phase_shifts if analysis_type == 'traveling' else [0.0]
        results_all[n] = {}

        for ph in phases:
            calc.set_n_and_phase(n, ph)
            ph_key = f"{ph:.1f}"
            mode_results = []

            for plot_idx, (real_mode_idx, freq) in enumerate(calc.modes):
                print(f"  [n={n}, phase={ph}°] Visualizing mode {real_mode_idx}: "
                      f"f={freq:.4f} GHz")
                params = calc.get_mode_parameters(n, real_mode_idx) or {}

                prefix = f"n{n}_ph{ph_key.replace('.', '_')}"
                imgs = {}

                # フィールド分布 (E + H)
                title = (f"n={n}, Mode {real_mode_idx}, "
                         f"f={freq:.4f} GHz")
                if analysis_type == 'traveling':
                    title += f", phase={ph}°"
                fname = f"{prefix}_mode{real_mode_idx:02d}_field.png"
                plotter.plot_mode_to_file(plot_idx, report_dir, fname,
                                          theta=0.0, title_prefix=title)
                imgs['field'] = fname

                # 軸上 Ez (n=0 のみ。n>=1 では物理的に Ez=0 なので省略)
                if n == 0:
                    fname_ax = f"{prefix}_mode{real_mode_idx:02d}_axial.png"
                    plotter.plot_axial_field(plot_idx, report_dir, fname_ax,
                                             analysis_type=analysis_type)
                    imgs['axial'] = fname_ax

                # アニメ (進行波で skip_anim=False のときのみ)
                if analysis_type == 'traveling' and not skip_anim:
                    try:
                        anim_name = f"{prefix}_mode{real_mode_idx:02d}_anim.gif"
                        anim_path = os.path.join(report_dir, anim_name)
                        plotter.create_animation(plot_idx, anim_path,
                                                 fps=10, n_frames=24,
                                                 title_prefix=title)
                        imgs['anim'] = anim_name
                    except Exception as e:
                        print(f"  [Warning] Animation failed: {e}")

                mode_results.append({
                    'mode_idx': real_mode_idx,
                    'frequency_GHz': freq,
                    'stored_energy': params.get('stored_energy', 0.0),
                    'p_loss': params.get('p_loss', 0.0),
                    'q_factor': params.get('q_factor', 0.0),
                    'images': imgs,
                })

            results_all[n][ph_key] = mode_results

    # HTML レポート生成
    info = {
        'input_file': processed_file,
        'num_vertices': len(calc.vertices),
        'num_elements': len(calc.simplices),
    }
    generate_hom_html_report(report_dir, results_all, info)
    print(f"\nReport generation completed. Report: "
          f"{os.path.join(report_dir, 'index.html')}")


def generate_hom_html_report(output_dir, results_all, info):
    """HOM レポート用の index.html を生成する。"""
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = [f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HOM FEM Analysis Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f4f7f6; color: #333; }}
.header {{ text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 10px; margin-bottom: 30px; }}
.card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 25px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
th {{ background-color: #2c3e50; color: white; text-align: center; }}
tr:nth-child(even) {{ background-color: #f9f9f9; }}
.mode-card {{ margin-top: 30px; border-top: 2px solid #3498db; padding-top: 15px; }}
.image-grid {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; }}
.img-box {{ flex: 1; min-width: 600px; text-align: center; }}
.img-box img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
.anim-box {{ max-width: 900px; margin: 20px auto; }}
.tag {{ padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
.tag-hom {{ background: #8e44ad; color: white; }}
</style></head><body>
<div class="header">
<h1>HOM FEM Analysis Report</h1>
<p>Generated on: {now} &nbsp;|&nbsp; <span class="tag tag-hom">HOM</span></p>
</div>
<div class="card">
<h2>File &amp; Mesh Info</h2>
<p><strong>Input File:</strong> {info['input_file']}</p>
<p><strong>Vertices:</strong> {info['num_vertices']} &nbsp;|&nbsp; <strong>Elements:</strong> {info['num_elements']}</p>
<div style="text-align:center;"><img src="mesh_overview.png" style="max-width:80%; border:1px solid #ccc;"></div>
</div>
"""]

    # n ごとに章立て
    for n, ph_dict in results_all.items():
        html.append(f"<h2>Azimuthal Order n = {n}</h2>")

        for ph_key, modes in ph_dict.items():
            ph_float = float(ph_key)
            is_tw = (len(ph_dict) > 1) or (ph_float != 0.0)
            wave_type = "Traveling Wave" if is_tw else "Standing Wave"
            html.append(f"<h3>{wave_type}"
                        f"{' (phase = ' + ph_key + '°)' if is_tw else ''}</h3>")

            # サマリー表
            html.append('<div class="card"><table><thead><tr>'
                        '<th>Mode</th><th>Freq [GHz]</th>'
                        '<th>U [J]</th><th>P_loss [W]</th><th>Q</th>'
                        '</tr></thead><tbody>')
            for m in modes:
                html.append(
                    f"<tr><td>{m['mode_idx']}</td>"
                    f"<td>{m['frequency_GHz']:.6f}</td>"
                    f"<td>{m['stored_energy']:.4e}</td>"
                    f"<td>{m['p_loss']:.4e}</td>"
                    f"<td>{m['q_factor']:.4e}</td></tr>")
            html.append('</tbody></table></div>')

            # 各モードの分布
            for m in modes:
                imgs = m['images']
                html.append('<div class="card mode-card">')
                html.append(f"<h4>Mode {m['mode_idx']}: "
                            f"{m['frequency_GHz']:.6f} GHz</h4>")
                html.append('<div class="image-grid">')
                if 'field' in imgs:
                    html.append(f'<div class="img-box">'
                                f'<img src="{imgs["field"]}"></div>')
                if 'axial' in imgs:
                    html.append(f'<div class="img-box">'
                                f'<p>Axial Ez</p>'
                                f'<img src="{imgs["axial"]}"></div>')
                html.append('</div>')
                if 'anim' in imgs:
                    html.append('<div class="anim-box">'
                                '<p style="text-align:center"><b>Time Evolution Animation</b></p>'
                                f'<img src="{imgs["anim"]}" style="width:100%"></div>')
                html.append('<div style="display:grid; '
                            'grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); '
                            'gap:10px; margin-top:15px;">')
                html.append(f'<div class="card" style="box-shadow:none; border:1px solid #eee;">'
                            f'<b>U:</b> {m["stored_energy"]:.4e} J</div>')
                html.append(f'<div class="card" style="box-shadow:none; border:1px solid #eee;">'
                            f'<b>P_loss:</b> {m["p_loss"]:.4e} W</div>')
                html.append(f'<div class="card" style="box-shadow:none; border:1px solid #eee;">'
                            f'<b>Q:</b> {m["q_factor"]:.4e}</div>')
                html.append('</div></div>')

    html.append("</body></html>")

    with open(os.path.join(output_dir, "index.html"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(html))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOM Post-Process")
    parser.add_argument("h5_file", help="Input HOM result HDF5")
    parser.add_argument("--cond", type=float, default=5.8e7,
                        help="Wall conductivity [S/m]")
    parser.add_argument("--mode", choices=['all', 'calc', 'report'],
                        default='all',
                        help="all=both stages, calc=parameters only, "
                             "report=report only")
    parser.add_argument("--no-anim", action='store_true',
                        help="Skip GIF animation generation in report")
    args = parser.parse_args()

    if args.mode == 'calc':
        run_hom_post_process(args.h5_file, args.cond)
    elif args.mode == 'report':
        run_hom_report_generation(args.h5_file, skip_anim=args.no_anim)
    else:
        run_hom_post_process(args.h5_file, args.cond)
        processed = os.path.splitext(args.h5_file)[0] + "_processed.h5"
        # _processed.h5 を入力にしている場合はそのまま、生 h5 を入力にした
        # 場合は新しく作られた _processed.h5 をレポートに使う
        report_input = processed if os.path.exists(processed) else args.h5_file
        run_hom_report_generation(report_input, skip_anim=args.no_anim)
