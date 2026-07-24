"""H_phi から E_z, E_r を再構成（軸上 r=0 は L'Hôpital 適用）.

ver1 ``FEM_code/field_calculator.py:FieldCalculator`` の場再構成ロジックを移植し、
HDF5 依存を切り離した版。メッシュ・固有ベクトル・周波数を直接受け取って計算する。

物理（TM0, DOF: ψ = H_phi）::

    ∇×H = jωε₀ε_r E より E = (1/(jωε₀ε_r)) ∇×H
    E_z = (1/(jωε₀ε_r)) (∂H_phi/∂r + H_phi/r)
    E_r = -(1/(jωε₀ε_r)) (∂H_phi/∂z)
    軸上 r→0: E_z = (1/(jωε₀ε_r)) · 2 (∂H_phi/∂r),  E_r = 0   (L'Hôpital)

ver2.1: 誘電体領域 (ε_r≠1) では H→E 換算で **ε_r で割る**。境界に垂直な
電場 E_n は電束 D_n=ε₀ε_r E_n の連続性により誘電体側で 1/ε_r に小さくなる。
``eps_r_per_element=None`` のときは全域 ε_r=1（真空）で ver2/ver1 と完全一致。

定在波と進行波で係数の取り方が ver1 と一致するように分岐している
（ver1 互換: 節点場の定在波は係数を掛けず生値、点場は両者とも 1/(ωε₀) 系を掛ける）。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from ..shared.constants import EPS0
from ..shared.element_functions import (
    calculate_area_coordinates,
    calculate_quadratic_nodal_shape_functions,
    grad_area_coordinates,
    grad_quadratic_nodal_shape_functions,
)

# 2 次要素の節点における面積座標（頂点 3 + 辺中点 3）
_L_NODES_2ND = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
])


class TM0FieldReconstructor:
    """TM0 の固有ベクトル（H_phi 節点値）から電磁場を再構成する.

    Attributes:
        nodes, elements, element_order, analysis_type: メッシュ・解析種別。
        eps0: 真空誘電率。
        eps_r_per_element: (Ne,) 要素ごと比誘電率。H→E 換算で E を ε_r で割る。
            ``None`` なら全域真空 (ε_r=1)。
    """

    def __init__(self, nodes, elements, element_order, analysis_type,
                 eps0=EPS0, eps_r_per_element=None):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements)
        self.element_order = element_order
        self.analysis_type = analysis_type
        self.eps0 = eps0
        if eps_r_per_element is None:
            self.eps_r_per_element = None
        else:
            self.eps_r_per_element = np.asarray(eps_r_per_element, dtype=float)
            if self.eps_r_per_element.shape != (len(self.elements),):
                raise ValueError(
                    f"eps_r_per_element shape {self.eps_r_per_element.shape} != "
                    f"(num_elements={len(self.elements)},)")
        self._tree = KDTree(np.mean(self.nodes[self.elements[:, :3]], axis=1))

    def _eps_r(self, elem_idx) -> float:
        """要素 elem_idx の比誘電率（eps_r_per_element 未設定なら 1.0）."""
        if self.eps_r_per_element is None:
            return 1.0
        return float(self.eps_r_per_element[elem_idx])

    @property
    def is_standing(self) -> bool:
        return self.analysis_type == "standing"

    # ------------------------------------------------------------------
    def find_element(self, z, r):
        """点 (z, r) を含む要素インデックスと面積座標 L を返す（領域外は None）."""
        p = np.array([z, r])
        _, idxs = self._tree.query(p, k=min(10, len(self.elements)))
        for idx in np.atleast_1d(idxs):
            vertices = self.nodes[self.elements[idx][:3]]
            L = calculate_area_coordinates(p, vertices)
            if np.all(L >= -1e-6):
                return idx, L
        return None, None

    # ------------------------------------------------------------------
    def calculate_fields(self, eigenvector, frequency_ghz, z, r,
                         theta=0.0, return_complex=False):
        """点 (z, r) における H_theta, E_z, E_r を計算する.

        Args:
            eigenvector: H_phi 節点値の固有ベクトル (N,)。
            frequency_ghz: モード周波数 [GHz]。
            z, r: 評価点座標。
            theta: 瞬時位相 [rad]（exp(jθ)）。
            return_complex: True で複素振幅、False で瞬時実数値を返す。

        Returns:
            dict（H_theta, Ez, Er, E_abs）。領域外は None。
        """
        idx, L = self.find_element(z, r)
        if idx is None:
            return None

        elem = self.elements[idx]
        vertices = self.nodes[elem[:3]]
        u_global = np.asarray(eigenvector) * np.exp(1j * theta)
        u_elem = u_global[elem]

        grad_L = grad_area_coordinates(vertices)
        if self.element_order == 1:
            G = L
            gradG = np.array(grad_L)
        else:
            G = calculate_quadratic_nodal_shape_functions(L)
            gradG = grad_quadratic_nodal_shape_functions(L, grad_L)

        h_theta = np.dot(u_elem, G)
        grad_h = np.dot(u_elem, gradG)   # [dH/dz, dH/dr]
        dh_dz, dh_dr = grad_h[0], grad_h[1]

        omega = 2 * np.pi * frequency_ghz * 1e9
        # 誘電体: E = (1/(jωε₀ε_r))∇×H → 係数を ε_r で割る
        eps_r = self._eps_r(idx)
        coeff = (1.0 / (omega * self.eps0 * eps_r) if self.is_standing
                 else -1j / (omega * self.eps0 * eps_r))

        if abs(r) < 1e-9:
            ez, er = 2.0 * dh_dr, 0.0
        else:
            ez, er = (h_theta / r) + dh_dr, -dh_dz

        Ez_c, Er_c = ez * coeff, er * coeff

        if return_complex:
            return {"H_theta": h_theta, "Ez": Ez_c, "Er": Er_c,
                    "E_abs": np.sqrt(np.abs(Ez_c) ** 2 + np.abs(Er_c) ** 2)}
        return {"H_theta": np.real(h_theta),
                "Ez": np.real(Ez_c), "Er": np.real(Er_c),
                "E_abs": np.sqrt(np.real(Ez_c) ** 2 + np.real(Er_c) ** 2)}

    # ------------------------------------------------------------------
    def axial_ez_scan(self, eigenvector, frequency_ghz, n_scan=2000):
        """軸上 (r=0) の E_z を z_min→z_max でスキャンする.

        Returns:
            ``(z_scan, ez_axial)``。``ez_axial`` は ver1 互換で各点の実部（瞬時値）。
        """
        z_min, z_max = np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0])
        z_scan = np.linspace(z_min, z_max, n_scan)
        ez_axial = np.array([
            (self.calculate_fields(eigenvector, frequency_ghz, zp, 0.0) or
             {"Ez": 0.0})["Ez"]
            for zp in z_scan
        ])
        return z_scan, ez_axial

    # ------------------------------------------------------------------
    def calculate_all_node_fields(self, eigenvector, frequency_ghz,
                                  theta=0.0, return_complex=False):
        """全節点での H_theta, Psi, E_z, E_r を要素寄与の平均で計算する.

        ver1 互換: 定在波は E に係数を掛けず生値（実数）を返し、進行波は
        係数 ``-j/(ωε₀)`` を掛ける。

        誘電体: 各要素の寄与を **その要素の ε_r で割って**から節点平均する。
        界面節点は両側で E_n が不連続なため、平均値は両領域の中間になる
        （ベクトル表示は点単位の :meth:`calculate_fields` の方が不連続を正確に示す）。
        """
        n_nodes = len(self.nodes)
        ez_total = np.zeros(n_nodes, dtype=complex)
        er_total = np.zeros(n_nodes, dtype=complex)
        node_count = np.zeros(n_nodes)

        omega = 2 * np.pi * frequency_ghz * 1e9
        coeff = -1j / (omega * self.eps0)
        u_global = np.asarray(eigenvector) * np.exp(1j * theta)

        for e_idx, elem in enumerate(self.elements):
            vertices = self.nodes[elem[:3]]
            u_elem = u_global[elem]
            grad_L = grad_area_coordinates(vertices)
            inv_eps_r = 1.0 / self._eps_r(e_idx)

            if self.element_order == 1:
                grad_h = np.dot(u_elem, grad_L)
                for node_idx in elem:
                    r = self.nodes[node_idx, 1]
                    h_val = u_global[node_idx]
                    if abs(r) < 1e-9:
                        ez, er = 2.0 * grad_h[1], 0.0
                    else:
                        ez, er = (h_val / r) + grad_h[1], -grad_h[0]
                    ez_total[node_idx] += ez * inv_eps_r
                    er_total[node_idx] += er * inv_eps_r
                    node_count[node_idx] += 1
            else:
                for i_local in range(6):
                    node_idx = elem[i_local]
                    gradG = grad_quadratic_nodal_shape_functions(
                        _L_NODES_2ND[i_local], grad_L)
                    grad_h = np.dot(u_elem, gradG)
                    r = self.nodes[node_idx, 1]
                    h_val = u_global[node_idx]
                    if abs(r) < 1e-9:
                        ez, er = 2.0 * grad_h[1], 0.0
                    else:
                        ez, er = (h_val / r) + grad_h[1], -grad_h[0]
                    ez_total[node_idx] += ez * inv_eps_r
                    er_total[node_idx] += er * inv_eps_r
                    node_count[node_idx] += 1

        mask = node_count > 0
        if self.is_standing:
            ez_node = np.zeros(n_nodes)
            er_node = np.zeros(n_nodes)
            ez_node[mask] = (ez_total[mask] / node_count[mask]).real
            er_node[mask] = (er_total[mask] / node_count[mask]).real
            psi = self.nodes[:, 1] * u_global
            return {"H_theta": u_global, "Psi": psi,
                    "Ez": ez_node, "Er": er_node,
                    "E_abs": np.sqrt(ez_node ** 2 + er_node ** 2)}

        ez_node = np.zeros(n_nodes, dtype=complex)
        er_node = np.zeros(n_nodes, dtype=complex)
        ez_node[mask] = (ez_total[mask] / node_count[mask]) * coeff
        er_node[mask] = (er_total[mask] / node_count[mask]) * coeff
        psi = -1j * self.nodes[:, 1] * u_global   # E と位相を合わせる -j

        e_abs = np.sqrt(np.abs(ez_node) ** 2 + np.abs(er_node) ** 2)
        if return_complex:
            return {"H_theta": u_global, "Psi": psi,
                    "Ez": ez_node, "Er": er_node, "E_abs": e_abs}
        return {"H_theta": np.real(u_global), "Psi": np.real(psi),
                "Ez": np.real(ez_node), "Er": np.real(er_node),
                "E_abs": np.sqrt(np.real(ez_node) ** 2 + np.real(er_node) ** 2)}
