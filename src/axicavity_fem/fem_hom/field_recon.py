"""HOM エッジ DOF から E, H 場を再構成（r=0 で L'Hôpital, モード別分岐）.

ver1 ``FEM_HOM_code/field_calculator_hom.py`` の場再構成ロジックを HDF5 非依存に
移植したもの。フラットな固有ベクトルを DOF 種別（CT/LN・LT/LN・face・節点 rE_theta）に
分割し、E（エッジ + 節点）と H = (j/ωμ₀)∇×E を計算する。

節点 DOF は ``rE_theta``（r を掛けた量）なので、物理 ``E_theta`` へは
:func:`calculate_E_theta_from_rE_theta` で変換する（n=1 は軸上を近傍平均で補間）。
"""

from __future__ import annotations

import numpy as np
from matplotlib.tri import Triangulation

from ..shared.constants import MU0
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


def calculate_E_theta_from_rE_theta(rE_theta_values, vertices, n, simplices):
    """節点 DOF ``rE_theta`` から物理 ``E_theta`` を返す（n=1 は軸上を近傍平均で補間）.

    Args:
        rE_theta_values: 節点ごとの rE_theta 値 (num_nodes,)。None なら None を返す。
        vertices: 節点座標 (N, 2) = [z, r]。
        n: 方位角モード次数。
        simplices: 要素節点インデックス。

    Returns:
        物理 E_theta 配列 (num_nodes,)、または None。
    """
    num_nodes = len(vertices)
    if rE_theta_values is None or len(rE_theta_values) != num_nodes:
        if n > 0:
            raise ValueError("rE_theta_values size mismatch or None for n > 0.")
        return None

    dtype = float if np.isrealobj(rE_theta_values) else complex
    E_theta = np.zeros(num_nodes, dtype=dtype)
    r_coords = vertices[:, 1]
    nz = np.where(r_coords > 1e-9)[0]
    E_theta[nz] = rE_theta_values[nz] / r_coords[nz]

    if n == 1:
        axis_nodes = np.where(np.abs(r_coords) < 1e-9)[0]
        if len(axis_nodes) > 0:
            node_to_elems = [[] for _ in range(num_nodes)]
            for elem_idx, simplex in enumerate(simplices):
                for node_idx in simplex:
                    node_to_elems[node_idx].append(elem_idx)
            for i in axis_nodes:
                neighbor_vals = []
                seen = set()
                for elem_idx in node_to_elems[i]:
                    for k in simplices[elem_idx]:
                        if k != i and abs(vertices[k, 1]) > 1e-9 and k not in seen:
                            seen.add(k)
                            neighbor_vals.append(E_theta[k])
                if neighbor_vals:
                    E_theta[i] = np.mean(np.array(neighbor_vals), axis=0)
    return E_theta


def split_hom_dofs(eigenvector, num_edges, num_elements, num_nodes,
                   element_order, n, vertices, simplices):
    """フラット固有ベクトルを (CT/LN, LT/LN, face, E_theta_phys) に分解する.

    節点成分は ``rE_theta`` → 物理 ``E_theta`` へ変換して返す。

    Returns:
        (edge_vectors, edge_vectors_lt, face_vectors, E_theta)。
        n=0 や 1 次要素では該当しない成分は None。
    """
    vec = np.asarray(eigenvector)
    if element_order == 2:
        edge_vectors = vec[0:2 * num_edges:2]
        edge_vectors_lt = vec[1:2 * num_edges:2]
        face_offset = 2 * num_edges
        node_offset = 2 * num_edges + 2 * num_elements
        face_vectors = vec[face_offset:node_offset]
        rE = vec[node_offset:node_offset + num_nodes] if n > 0 else None
    else:
        edge_vectors = vec[:num_edges]
        edge_vectors_lt = None
        face_vectors = None
        rE = vec[num_edges:num_edges + num_nodes] if n > 0 else None

    E_theta = (calculate_E_theta_from_rE_theta(rE, vertices, n, simplices)
               if (n > 0 and rE is not None) else None)
    return edge_vectors, edge_vectors_lt, face_vectors, E_theta


class HOMFieldReconstructor:
    """HOM の固有ベクトルから (z, r) 点の E, H 場を再構成する.

    使い方::

        recon = HOMFieldReconstructor(simplices, vertices, edge_index_map,
                                      element_order, n, analysis_type)
        recon.load_mode(eigenvector, frequency_ghz)
        f = recon.calculate_fields(z, r)
    """

    def __init__(self, simplices, vertices, edge_index_map, element_order, n,
                 analysis_type, mu0=MU0):
        self.simplices = np.asarray(simplices)
        self.vertices = np.asarray(vertices)
        self.edge_index_map = edge_index_map
        self.element_order = element_order
        self.n = n
        self.analysis_type = analysis_type
        self.mu0 = mu0

        self._tri = Triangulation(self.vertices[:, 0], self.vertices[:, 1],
                                  self.simplices[:, :3])
        self._trifinder = self._tri.get_trifinder()

        self.edge_vectors = None
        self.edge_vectors_lt = None
        self.face_vectors = None
        self.E_theta = None
        self.freq_GHz = 0.0
        self.omega = 0.0

    @property
    def is_standing(self) -> bool:
        return self.analysis_type == "standing"

    def load_mode(self, eigenvector, frequency_ghz):
        """フラット固有ベクトルを分割して内部状態にセットする."""
        num_edges = len(self.edge_index_map)
        self.edge_vectors, self.edge_vectors_lt, self.face_vectors, self.E_theta = \
            split_hom_dofs(eigenvector, num_edges, len(self.simplices),
                           len(self.vertices), self.element_order, self.n,
                           self.vertices, self.simplices)
        self.freq_GHz = float(frequency_ghz)
        self.omega = 2.0 * np.pi * self.freq_GHz * 1e9
        return self

    def calculate_fields(self, z, r, theta_time=0.0, elem_idx=None):
        """点 (z, r) における E, H 各成分を返す（領域外は None）.

        定在波は磁場に -j を掛けて電場と同時に最大振幅を表示する（ver1 互換）。
        進行波は指定位相 ``theta_time`` [度] の瞬時値（実部）を返す。

        ``elem_idx`` を与えると点探索（trifinder）を省略してその要素で評価する。
        曲線境界の中点ノードは直線三角形からはみ出して trifinder が領域外と
        判定するため、節点評価では含有要素を直接指定して 0 落ちを防ぐ。
        """
        if elem_idx is None:
            elem_idx = self._trifinder(z, r)
            if elem_idx == -1:
                return None

        simplex = self.simplices[elem_idx]
        simplex_cnr = simplex[:3]
        verts = self.vertices[simplex_cnr]
        pt = np.array([z, r])
        n = self.n

        L = calculate_area_coordinates(pt, verts)
        gL = grad_area_coordinates(verts)

        local_edges = [(simplex_cnr[0], simplex_cnr[1]),
                       (simplex_cnr[1], simplex_cnr[2]),
                       (simplex_cnr[2], simplex_cnr[0])]
        dtype = complex if np.iscomplexobj(self.edge_vectors) else float
        ct_dofs = np.zeros(3, dtype=dtype)
        lt_dofs = np.zeros(3, dtype=dtype)
        for k, (n1, n2) in enumerate(local_edges):
            edge_idx = self.edge_index_map[tuple(sorted((n1, n2)))]
            sign = 1 if n1 < n2 else -1
            ct_dofs[k] = sign * self.edge_vectors[edge_idx]
            if self.element_order == 2 and self.edge_vectors_lt is not None:
                lt_dofs[k] = self.edge_vectors_lt[edge_idx]

        if self.element_order == 2 and self.face_vectors is not None:
            f0 = self.face_vectors[2 * elem_idx]
            f1 = self.face_vectors[2 * elem_idx + 1]
        else:
            f0 = f1 = 0.0

        if self.E_theta is not None:
            if self.element_order == 2:
                nodal_dofs = self.E_theta[simplex]
                G = calculate_quadratic_nodal_shape_functions(L)
                gradG = grad_quadratic_nodal_shape_functions(L, gL)
            else:
                nodal_dofs = self.E_theta[simplex_cnr]
                G = L
                gradG = np.array(gL)
        else:
            n_nodes = 6 if self.element_order == 2 else 3
            nodal_dofs = np.zeros(n_nodes, dtype=dtype)
            G = np.zeros(n_nodes)
            gradG = np.zeros((n_nodes, 2))

        if self.element_order == 2:
            N = calculate_edge_shape_functions_2nd(pt, verts)
            vec = (ct_dofs[0] * N[0] + ct_dofs[1] * N[1] + ct_dofs[2] * N[2]
                   + lt_dofs[0] * N[3] + lt_dofs[1] * N[4] + lt_dofs[2] * N[5]
                   + f0 * N[6] + f1 * N[7])
        else:
            N = calculate_edge_shape_functions(pt, verts)
            vec = ct_dofs[0] * N[0] + ct_dofs[1] * N[1] + ct_dofs[2] * N[2]

        Ez_comp, Er_comp = vec[0], vec[1]
        Etheta_comp = np.dot(nodal_dofs, G)
        dEtheta_dz = np.dot(nodal_dofs, gradG[:, 0])
        dEtheta_dr = np.dot(nodal_dofs, gradG[:, 1])

        if self.element_order == 2:
            curls = calculate_curl_edge_shape_functions_2nd(pt, verts)
            curl_Ezr = (ct_dofs[0] * curls[0] + ct_dofs[1] * curls[1]
                        + ct_dofs[2] * curls[2] + lt_dofs[0] * curls[3]
                        + lt_dofs[1] * curls[4] + lt_dofs[2] * curls[5]
                        + f0 * curls[6] + f1 * curls[7])
        else:
            A2 = calculate_triangle_area_double(verts)
            curl_Ezr = (ct_dofs[0] + ct_dofs[1] + ct_dofs[2]) * (2.0 / A2)

        safe_r = r if abs(r) > 1e-9 else 1e-9
        drEtheta_dr = Etheta_comp + safe_r * dEtheta_dr

        curl_E_r = (n / safe_r) * Ez_comp - dEtheta_dz
        curl_E_theta = curl_Ezr
        curl_E_z = (1.0 / safe_r) * drEtheta_dr - (n / safe_r) * Er_comp

        coef = 1j / (self.omega * self.mu0)
        Htheta_comp = coef * curl_E_theta
        if abs(r) > 1e-9:
            Hz_comp = coef * curl_E_z
            Hr_comp = coef * curl_E_r
        else:
            if n == 0:
                Hz_comp = Hr_comp = 0
            else:
                Hz_comp = 0
                Hr_comp = coef * (-dEtheta_dz)

        comp_phase = np.exp(1j * np.deg2rad(theta_time))
        if self.is_standing:
            Ez_plot = Ez_comp.real
            Er_plot = Er_comp.real
            Etheta_plot = Etheta_comp.real
            H_mod = -1j * np.array([Hz_comp, Hr_comp, Htheta_comp])
            Hz_plot, Hr_plot, Htheta_plot = (H_mod[0].real, H_mod[1].real,
                                             H_mod[2].real)
        else:
            Ez_plot = (Ez_comp * comp_phase).real
            Er_plot = (Er_comp * comp_phase).real
            Etheta_plot = (Etheta_comp * comp_phase).real
            Hz_plot = (Hz_comp * comp_phase).real
            Hr_plot = (Hr_comp * comp_phase).real
            Htheta_plot = (Htheta_comp * comp_phase).real

        return {
            "Ez": Ez_plot, "Er": Er_plot, "E_theta": Etheta_plot,
            "E_abs": np.sqrt(Ez_plot ** 2 + Er_plot ** 2 + Etheta_plot ** 2),
            "Hz": Hz_plot, "Hr": Hr_plot, "H_theta": Htheta_plot,
            "H_abs": np.sqrt(Hz_plot ** 2 + Hr_plot ** 2 + Htheta_plot ** 2),
        }

    def calculate_all_node_fields(self, theta_time=0.0):
        """全節点上の E, H 場（各節点を含む 1 要素を使った擬似補間）を返す.

        曲線境界の中点ノードは直線三角形からはみ出して trifinder が領域外と
        判定するため、点探索が失敗した節点はその節点を含む要素で評価し直す
        （0 落ち防止）。
        """
        num_nodes = len(self.vertices)
        out = {k: np.zeros(num_nodes) for k in
               ("Ez", "Er", "E_theta", "Hz", "Hr", "H_theta")}
        node_to_elem = {}
        for elem_idx, simplex in enumerate(self.simplices):
            for node_idx in simplex:
                node_to_elem.setdefault(int(node_idx), elem_idx)
        for i in range(num_nodes):
            z, r = self.vertices[i]
            res = self.calculate_fields(z, max(r, 1e-10), theta_time)
            if res is None:
                res = self.calculate_fields(z, max(r, 1e-10), theta_time,
                                            elem_idx=node_to_elem.get(i))
            if res:
                for k in out:
                    out[k][i] = res[k]
        return out
