"""HOM Q値・R/Q・群速度・減衰定数・電力流の計算.

ver1 ``FEM_HOM_code/post_process_hom.py`` の物理計算部
（``calc_p_flow_hom`` と ``run_hom_post_process`` の工学パラメータ計算）を移植した版。
HDF5 I/O・HTML レポートは含まない（shared.hdf5_io / reports が担当）。

P_flow のラインインテグラルは規約セーフティ層
:mod:`axicavity_fem.physics_core.power_flow` の ``edge_line_integral_3point`` を用いる
（係数は HOM 固有の ``+π/(ωμ₀)``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics_core.power_flow import edge_line_integral_3point
from ..shared.constants import C0, EPS0, MU0
from ..shared.element_functions import (
    calculate_area_coordinates,
    calculate_curl_edge_shape_functions_2nd,
    calculate_edge_shape_functions,
    calculate_edge_shape_functions_2nd,
    calculate_quadratic_nodal_shape_functions,
    calculate_triangle_area_double,
    grad_area_coordinates,
    grad_quadratic_nodal_shape_functions,
)
from ..shared.quadrature import integration_points_triangle
from .field_recon import split_hom_dofs


def _triangle_area(verts):
    return abs(calculate_triangle_area_double(verts)) / 2.0


def _bary_on_edge(eid, t):
    if eid == 0:
        return np.array([1 - t, t, 0.0])
    if eid == 1:
        return np.array([0.0, 1 - t, t])
    return np.array([t, 0.0, 1 - t])


def _element_ct_lt(corner, edge_vectors, edge_vectors_lt, edge_index_map,
                   element_order):
    """要素のローカル CT/LN・LT/LN DOF（向き符号適用済み）を返す."""
    ct = np.zeros(3, dtype=complex)
    lt = np.zeros(3, dtype=complex)
    pairs = [(corner[0], corner[1]), (corner[1], corner[2]),
             (corner[2], corner[0])]
    for k, (e1, e2) in enumerate(pairs):
        idx = edge_index_map[tuple(sorted((e1, e2)))]
        s = 1 if e1 < e2 else -1
        ct[k] = s * edge_vectors[idx]
        if element_order == 2 and edge_vectors_lt is not None:
            lt[k] = edge_vectors_lt[idx]
    return ct, lt


def get_pec_edges(simplices, edge_index_map, pec_loss_edge_indices):
    """P_loss 用 PEC 辺を ``[(elem_idx, local_edge_idx)]`` で返す."""
    pec_set = set(int(i) for i in pec_loss_edge_indices)
    out = []
    for elem_idx, simplex in enumerate(simplices):
        corner = simplex[:3]
        for i, (n1, n2) in enumerate(
                [(corner[0], corner[1]), (corner[1], corner[2]),
                 (corner[2], corner[0])]):
            key = tuple(sorted((n1, n2)))
            if key in edge_index_map and edge_index_map[key] in pec_set:
                out.append((elem_idx, i))
    return out


def calc_p_flow_hom(simplices, vertices, edge_vectors, edge_vectors_lt,
                    face_vectors, E_theta_nodal, edge_index_map,
                    element_order, omega, n_mode, mu0=MU0):
    """z=z_min 断面のポインティング z 成分積分から電力流 P_flow を計算する.

    ``P = (π/(ωμ₀)) ∫ {Im[E_r·curl_Ezr*]·r − n·Re[E_φ·E_z*] + Im[E_φ·(∂E_φ/∂z)*]·r} dr``

    Args:
        simplices, vertices, edge_index_map: メッシュ。
        edge_vectors, edge_vectors_lt, face_vectors, E_theta_nodal: 分割済み DOF
            （E_theta_nodal は物理 E_theta）。
        element_order, omega, n_mode: 要素次数・角周波数・方位角次数。

    Returns:
        P_flow [W]。
    """
    z_coords = vertices[:, 0]
    z_min = np.min(z_coords)
    z_tol = (np.max(z_coords) - z_min) * 1e-6
    z_min_set = set(np.where(np.abs(z_coords - z_min) < z_tol)[0])

    if element_order == 2:
        edge_defs = [(0, [0, 1, 3]), (1, [1, 2, 4]), (2, [2, 0, 5])]
    else:
        edge_defs = [(0, [0, 1]), (1, [1, 2]), (2, [2, 0])]

    seen = set()
    boundary = []
    for elem_idx, elem in enumerate(simplices):
        for eid, idxs in edge_defs:
            gnodes = tuple(int(elem[i]) for i in idxs)
            if all(nd in z_min_set for nd in gnodes):
                key = tuple(sorted(gnodes))
                if key not in seen:
                    seen.add(key)
                    boundary.append(
                        (eid, (int(elem[idxs[0]]), int(elem[idxs[1]])),
                         elem, elem_idx))

    p_flow_total = 0.0
    for eid, (n1_idx, n2_idx), elem, elem_idx in boundary:
        corner = elem[:3].astype(int)
        verts = vertices[corner]
        grad_L = grad_area_coordinates(verts)
        area = _triangle_area(verts)

        ct, lt = _element_ct_lt(corner, edge_vectors, edge_vectors_lt,
                                edge_index_map, element_order)
        f_dofs = (face_vectors[2 * elem_idx:2 * elem_idx + 2]
                  if element_order == 2 and face_vectors is not None
                  else [0.0, 0.0])
        nod = (E_theta_nodal[elem] if (n_mode != 0 and E_theta_nodal is not None)
               else np.zeros(len(elem)))

        L_edge = abs(vertices[n2_idx, 1] - vertices[n1_idx, 1])
        if L_edge < 1e-15:
            continue

        def integrand(t, eid=eid, verts=verts, grad_L=grad_L, area=area,
                      ct=ct, lt=lt, f_dofs=f_dofs, nod=nod):
            L = _bary_on_edge(eid, t)
            r = float(np.dot(L, verts[:, 1]))
            pt = np.array([float(np.dot(L, verts[:, 0])), r])
            if element_order == 2:
                N = calculate_edge_shape_functions_2nd(pt, verts)
                curls = calculate_curl_edge_shape_functions_2nd(pt, verts)
                G = calculate_quadratic_nodal_shape_functions(L)
                gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
                vec_zr = (ct[0] * N[0] + ct[1] * N[1] + ct[2] * N[2]
                          + lt[0] * N[3] + lt[1] * N[4] + lt[2] * N[5]
                          + f_dofs[0] * N[6] + f_dofs[1] * N[7])
                curl_Ezr = (ct[0] * curls[0] + ct[1] * curls[1]
                            + ct[2] * curls[2] + lt[0] * curls[3]
                            + lt[1] * curls[4] + lt[2] * curls[5]
                            + f_dofs[0] * curls[6] + f_dofs[1] * curls[7])
            else:
                N = calculate_edge_shape_functions(pt, verts)
                A2 = 2.0 * area
                G = L
                gradG = np.array(grad_L)
                vec_zr = ct[0] * N[0] + ct[1] * N[1] + ct[2] * N[2]
                curl_Ezr = (ct[0] + ct[1] + ct[2]) * (2.0 / A2)

            E_z, E_r = vec_zr[0], vec_zr[1]
            if n_mode != 0 and E_theta_nodal is not None:
                E_phi = np.dot(nod, G)
                dEphi_dz = np.dot(nod, gradG[:, 0])
            else:
                E_phi = dEphi_dz = 0.0

            term1 = np.imag(E_r * np.conj(curl_Ezr)) * r
            term2 = -n_mode * np.real(E_phi * np.conj(E_z))
            term3 = np.imag(E_phi * np.conj(dEphi_dz)) * r
            return term1 + term2 + term3

        p_flow_total += edge_line_integral_3point(integrand, L_edge)

    return (np.pi / (omega * mu0)) * p_flow_total


@dataclass
class HOMModeParameters:
    """1 モード分の HOM 工学パラメータ."""

    frequency_ghz: float
    stored_energy: float       # U [J]
    p_loss: float              # 壁損失 [W]
    q_factor: float
    p_flow: float = 0.0        # 進行波
    v_group_c: float = 0.0
    v_phase_c: float = 0.0
    attenuation: float = 0.0
    # ver2.3: 誘電体損失（摂動法）。q_factor は全損失込みの合計 Q。
    q_wall: float = 0.0        # 壁損失のみの Q = ωU/P_loss（従来の q_factor）
    q_diel: float = 0.0        # 誘電体損失のみの Q（0 = 無損失）
    p_diel: float = 0.0        # 誘電体損失 [W]


@dataclass
class HOMParameters:
    """全モード・全位相の HOM 工学パラメータ.

    Attributes:
        n, analysis_type, conductivity, cell_length: 条件。
        per_phase: ``{phase_deg: list[HOMModeParameters]}``。
    """

    n: int
    analysis_type: str
    conductivity: float
    cell_length: float
    per_phase: dict = field(default_factory=dict)


def _stored_energy(simplices, vertices, edge_index_map, element_order,
                   ev, ev_lt, face, E_theta, u_coeff, eps0,
                   eps_r_per_element=None):
    """U = u_coeff·2π·∫ε₀·ε_r·|E|²·r dA を全要素 7 点求積で計算する.

    誘電体では電気エネルギー密度 ½ε₀ε_r|E|² に従い、各要素の寄与に ε_r を
    乗じる（E 場ベース）。``eps_r_per_element=None`` なら全域 ε_r=1（真空）。
    """
    pts = integration_points_triangle[7]
    coords_L = pts["coords"]
    weights = pts["weights"]

    U = 0.0
    for elem_idx, simplex in enumerate(simplices):
        eps_r = (1.0 if eps_r_per_element is None
                 else float(eps_r_per_element[elem_idx]))
        corner = simplex[:3]
        verts = vertices[corner]
        area = _triangle_area(verts)
        ct, lt = _element_ct_lt(corner, ev, ev_lt, edge_index_map,
                                element_order)
        f_dofs = (face[2 * elem_idx:2 * elem_idx + 2]
                  if element_order == 2 and face is not None else [0, 0])
        nod = E_theta[simplex] if E_theta is not None else np.zeros(len(simplex))

        for p in range(len(weights)):
            L = coords_L[p]
            w = weights[p]
            z = np.dot(L, verts[:, 0])
            r = np.dot(L, verts[:, 1])
            if element_order == 2:
                N = calculate_edge_shape_functions_2nd(np.array([z, r]), verts)
                G = calculate_quadratic_nodal_shape_functions(L)
                vec_zr = (ct[0] * N[0] + ct[1] * N[1] + ct[2] * N[2]
                          + lt[0] * N[3] + lt[1] * N[4] + lt[2] * N[5]
                          + f_dofs[0] * N[6] + f_dofs[1] * N[7])
            else:
                N = calculate_edge_shape_functions(np.array([z, r]), verts)
                G = L
                vec_zr = ct[0] * N[0] + ct[1] * N[1] + ct[2] * N[2]
            E2 = (np.abs(vec_zr[0]) ** 2 + np.abs(vec_zr[1]) ** 2
                  + np.abs(np.dot(nod, G)) ** 2)
            U += (u_coeff * eps0 * eps_r * E2) * (2.0 * np.pi * r) * (w * area)
    return U


def _wall_loss(simplices, vertices, edge_index_map, element_order, n,
               pec_edges, ev, ev_lt, face, E_theta, omega, conductivity, mu0):
    """P_loss = π·Rs·∫|H_tan|²·r dl を PEC 辺で 3 点求積する."""
    rs_ohm = (np.sqrt(omega * mu0 / (2.0 * conductivity))
              if conductivity > 0 else 0.0)
    q1d_pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
    q1d_w = [5 / 9, 8 / 9, 5 / 9]
    coef = 1j / (omega * mu0)

    P_loss = 0.0
    for elem_idx, edge_local_idx in pec_edges:
        corner = simplices[elem_idx][:3]
        verts = vertices[corner]
        grad_L = grad_area_coordinates(verts)

        n1_idx = corner[edge_local_idx]
        n2_idx = corner[(edge_local_idx + 1) % 3]
        p1, p2 = vertices[n1_idx], vertices[n2_idx]
        edge_len = np.linalg.norm(p2 - p1)
        tangent = (p2 - p1) / edge_len
        normal = np.array([-tangent[1], tangent[0]])

        ct, lt = _element_ct_lt(corner, ev, ev_lt, edge_index_map,
                                element_order)
        f_dofs = (face[2 * elem_idx:2 * elem_idx + 2]
                  if element_order == 2 and face is not None else [0, 0])
        nod = (E_theta[simplices[elem_idx]] if E_theta is not None
               else np.zeros(len(simplices[elem_idx])))

        for xi, wi in zip(q1d_pts, q1d_w):
            v_s = (xi + 1.0) / 2.0
            pt = p1 + v_s * (p2 - p1)
            L = calculate_area_coordinates(pt, verts)
            r_e = pt[1]
            if element_order == 2:
                N = calculate_edge_shape_functions_2nd(pt, verts)
                curls = calculate_curl_edge_shape_functions_2nd(pt, verts)
                G = calculate_quadratic_nodal_shape_functions(L)
                gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
                curl_Ezr = (ct[0] * curls[0] + ct[1] * curls[1]
                            + ct[2] * curls[2] + lt[0] * curls[3]
                            + lt[1] * curls[4] + lt[2] * curls[5]
                            + f_dofs[0] * curls[6] + f_dofs[1] * curls[7])
                Ez_p = (ct[0] * N[0][0] + ct[1] * N[1][0] + ct[2] * N[2][0]
                        + lt[0] * N[3][0] + lt[1] * N[4][0] + lt[2] * N[5][0]
                        + f_dofs[0] * N[6][0] + f_dofs[1] * N[7][0])
                Er_p = (ct[0] * N[0][1] + ct[1] * N[1][1] + ct[2] * N[2][1]
                        + lt[0] * N[3][1] + lt[1] * N[4][1] + lt[2] * N[5][1]
                        + f_dofs[0] * N[6][1] + f_dofs[1] * N[7][1])
            else:
                N = calculate_edge_shape_functions(pt, verts)
                A2 = 2.0 * _triangle_area(verts)
                G = L
                gradG = np.array(grad_L)
                curl_Ezr = (ct[0] + ct[1] + ct[2]) * (2.0 / A2)
                Ez_p = ct[0] * N[0][0] + ct[1] * N[1][0] + ct[2] * N[2][0]
                Er_p = ct[0] * N[0][1] + ct[1] * N[1][1] + ct[2] * N[2][1]

            Et_p = np.dot(nod, G)
            dEt_dz = np.dot(nod, gradG[:, 0])
            dEt_dr = np.dot(nod, gradG[:, 1])
            r_s = max(r_e, 1e-9)
            drEt_dr = Et_p + r_s * dEt_dr

            curlE_r = 1j * (n / r_s) * Ez_p - dEt_dz
            curlE_t = curl_Ezr
            curlE_z = ((1.0 / r_s) * drEt_dr - 1j * (n / r_s) * Er_p
                       if r_e > 1e-9 else 0.0)

            if n == 0:
                H2 = np.abs(coef * curlE_t) ** 2
            else:
                H2 = (np.abs(coef * curlE_t) ** 2
                      + np.abs(coef * (-curlE_z * normal[1]
                                       + curlE_r * normal[0])) ** 2)
            P_loss += (np.pi * rs_ohm * H2 * r_s) * (0.5 * wi * edge_len)
    return P_loss


def compute_hom_parameters(result, conductivity: float = 5.8e7,
                           eps_r_per_element=None,
                           tan_delta_per_element=None) -> HOMParameters:
    """:class:`HOMResult` から HOM 工学パラメータ（U, P_loss, Q, 群速度等）を計算する.

    Args:
        result: :class:`~axicavity_fem.fem_hom.solver.HOMResult`。
            ``pec_loss_edge_indices`` を保持していること。
        conductivity: 壁の導電率 [S/m]。
        eps_r_per_element: (Ne,) 要素ごと比誘電率。蓄積エネルギー U の電気
            エネルギー積分に ε_r を乗じる（誘電体対応）。``None`` で真空。
            P_loss（壁の磁場ベース）は ε_r 非依存なので影響なし。
        tan_delta_per_element: (Ne,) 要素ごと誘電正接 tanδ。指定すると摂動法で
            W_tand = Σ_e ε_r·tanδ·∫|E|² を ``_stored_energy`` と同一機構で積分し
            Q_diel = U/W_tand、P_diel = ω·W_tand を計算、``q_factor`` は
            合計 Q = ωU/(P_loss+P_diel) になる（``q_wall`` が従来値を保持）。
            ``None``/全ゼロで従来どおり。一様充填では Q_diel = 1/tanδ。

    Returns:
        :class:`HOMParameters`。
    """
    simplices = np.asarray(result.simplices)
    vertices = np.asarray(result.vertices)
    edge_index_map = result.edge_index_map
    order = result.element_order
    n = result.n
    num_edges = result.num_edges
    num_elements = len(simplices)
    num_nodes = len(vertices)

    cell_length = float(np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    pec_edges = get_pec_edges(simplices, edge_index_map,
                              result.pec_loss_edge_indices)

    out = HOMParameters(n=n, analysis_type=result.analysis_type,
                        conductivity=conductivity, cell_length=cell_length)

    if result.analysis_type == "standing":
        phases = {0.0: result.normal}
    else:
        phases = dict(result.periodic)

    for phase_deg, modeset in phases.items():
        is_traveling = result.analysis_type != "standing"
        u_coeff = 0.5 if is_traveling else 1.0
        mode_params = []
        for m in range(len(modeset.frequencies)):
            freq_ghz = float(modeset.frequencies[m])
            omega = 2.0 * np.pi * freq_ghz * 1e9
            ev, ev_lt, face, E_theta = split_hom_dofs(
                modeset.eigenvectors[m], num_edges, num_elements, num_nodes,
                order, n, vertices, simplices)

            U = _stored_energy(simplices, vertices, edge_index_map, order,
                               ev, ev_lt, face, E_theta, u_coeff, EPS0,
                               eps_r_per_element=eps_r_per_element)
            P_loss = _wall_loss(simplices, vertices, edge_index_map, order, n,
                                pec_edges, ev, ev_lt, face, E_theta, omega,
                                conductivity, MU0)
            q_wall = (omega * U / P_loss) if P_loss > 0 else 0.0

            # ver2.3: 誘電体損失（摂動法）。tanδ 無指定/全ゼロなら完全スキップ
            # （p_diel=0.0 で q_factor は従来値とビット同一）。
            q_diel = 0.0
            p_diel = 0.0
            if (tan_delta_per_element is not None
                    and np.any(np.asarray(tan_delta_per_element) > 0.0)):
                tand = np.asarray(tan_delta_per_element, dtype=float)
                eff = (tand * np.asarray(eps_r_per_element, dtype=float)
                       if eps_r_per_element is not None else tand)
                # U と同一機構・同一係数で ε_r·tanδ 重みの |E|² 積分を評価
                W_tand = _stored_energy(simplices, vertices, edge_index_map,
                                        order, ev, ev_lt, face, E_theta,
                                        u_coeff, EPS0, eps_r_per_element=eff)
                if W_tand > 0.0:
                    q_diel = U / W_tand
                    p_diel = omega * W_tand

            p_total = P_loss + p_diel
            q_factor = (omega * U / p_total) if p_total > 0 else 0.0

            mp = HOMModeParameters(frequency_ghz=freq_ghz, stored_energy=U,
                                   p_loss=P_loss, q_factor=q_factor,
                                   q_wall=q_wall, q_diel=q_diel,
                                   p_diel=p_diel)
            if is_traveling:
                p_flow = calc_p_flow_hom(
                    simplices, vertices, ev, ev_lt, face, E_theta,
                    edge_index_map, order, omega, n)
                mp.p_flow = p_flow
                mp.v_group_c = ((p_flow * cell_length / U) / C0
                                if U > 0 else 0.0)
                theta_rad = np.deg2rad(phase_deg)
                mp.v_phase_c = ((omega * cell_length) / (theta_rad * C0)
                                if theta_rad > 1e-10 else 0.0)
                mp.attenuation = (P_loss / (2.0 * p_flow * cell_length)
                                  if p_flow != 0 else 0.0)
            mode_params.append(mp)
        out.per_phase[phase_deg] = mode_params

    return out
