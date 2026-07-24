"""TM0 Q値・R/Q・V_eff・群速度・減衰定数の計算.

ver1 ``FEM_code/post_process_unified.py`` の物理計算部
（``calc_p_flow`` と ``run_parameter_calculation`` の工学パラメータ計算）を移植した版。
HDF5 I/O・HTML レポート生成は含まない（それぞれ別レイヤ shared.hdf5_io / reports が担当）。

P_flow は規約セーフティ層 :mod:`axicavity_fem.physics_core.power_flow` のプリミティブ
（``edge_line_integral_3point`` / ``tm0_p_flow_coefficient``）を用いて
**z=z_min 断面のラインインテグラル**として計算する。

ver2.1 制約（誘電体導入時）
---------------------------
要素行列の組立 (``fem_tm0.assembly``) は要素ごとの ε_r をサポート済み。一方、
ここの工学パラメータ計算は以下を真空 (ε_r=1, μ_r=1) 一様前提としている:

- 蓄積エネルギー: ``U = π μ₀ Re(u^H M u) / 2``（H ベース）
- 軸上 ``E_z`` 計算（``field_recon`` 経由）: ``E_z = -j/(ω ε₀) · (H_φ/r + ∂H_φ/∂r)``
- 壁損失 ``P_loss``: 真空の表面インピーダンスを使用
- ``P_flow`` 公式: 前置係数 ``-π/(ω ε₀)``（``power_flow.tm0_p_flow_coefficient``）

誘電体を部分的に置いた場合、共振周波数 (= 固有値の平方根) の計算は正しいが、
上記の派生量は **軸上または ports に誘電体が無い** ことを前提とする。ソルバ
本体 (``solver.solve_tm0_*``) で ``material_table`` に ε_r ≠ 1 が含まれる
場合、ポート断面 (z=z_min) と軸 (r=0) 上の要素が vacuum であるかを sanity
check し、そうでなければ警告を出す。

ver2.3: 誘電体損失 Q_diel（摂動法）
-----------------------------------
``tan_delta_per_element`` を渡すと、実固有値解はそのままに（摂動法）、
再構成 E 場の体積積分から誘電体損失を計算する::

    Q_diel = [Σ_e ε_r ∫|E|² r dA] / [Σ_e ε_r tanδ ∫|E|² r dA]
    P_diel = ω U / Q_diel
    Q_wall = ω U / P_loss          （従来の q_factor）
    Q      = ω U / (P_loss + P_diel)   （合計 Q。tanδ=0 なら従来値と同一）

一様充填では Q_diel = 1/tanδ が恒等的に成立する。モード形状の変化を無視する
摂動近似のため tanδ ≪ 1（目安 tanδ ≲ 0.05）で有効。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics_core.power_flow import (
    edge_line_integral_3point,
    tm0_p_flow_coefficient,
)
from ..shared.boundary_groups import BoundaryClassification
from ..shared.constants import C0, EPS0, MU0
from ..shared.element_functions import (
    calculate_quadratic_nodal_shape_functions,
    grad_area_coordinates,
    grad_quadratic_nodal_shape_functions,
)
from ..shared.quadrature import integration_points_triangle
from .field_recon import TM0FieldReconstructor


# ---------------------------------------------------------------------------
# 境界線積分（P_loss 用、ver1 TM0 FEM_element_function より移植）
# ---------------------------------------------------------------------------
def calculate_boundary_integral_quadratic(edge_nodes_coords, edge_values):
    """2 次エッジ（3 節点）上で ``∫ H^2 r dl`` を計算する（3 点ガウス求積）.

    Args:
        edge_nodes_coords: 3 節点の座標 (3, 2) = [始点, 終点, 中点]。
        edge_values: 各節点の物理量 |H_theta| 値 (3,)。

    Returns:
        積分値 ``∫ H^2 r dl``。
    """
    p1, p2, p3 = edge_nodes_coords
    v1, v2, v3 = edge_values
    L_total = np.linalg.norm(p2 - p1)

    xi = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    wi = np.array([5 / 9, 8 / 9, 5 / 9])

    integral = 0.0
    for x, w in zip(xi, wi):
        n1 = 0.5 * x * (x - 1)
        n2 = 0.5 * x * (x + 1)
        n3 = 1 - x ** 2
        h_interp = v1 * n1 + v2 * n2 + v3 * n3
        r_interp = p1[1] * n1 + p2[1] * n2 + p3[1] * n3
        integral += w * (h_interp ** 2 * r_interp)
    return integral * (L_total / 2.0)


# ---------------------------------------------------------------------------
# P_flow（進行波, z=z_min 断面のラインインテグラル）
# ---------------------------------------------------------------------------
def _bary_on_edge(eid: int, t: float) -> np.ndarray:
    """辺 ID と t∈[0,1] から面積座標 (L1, L2, L3) を返す."""
    if eid == 0:
        return np.array([1 - t, t, 0.0])
    if eid == 1:
        return np.array([0.0, 1 - t, t])
    return np.array([t, 0.0, 1 - t])


def tm0_p_flow_zmin(eigenvector, nodes, elements, element_order, omega,
                    eps0: float = EPS0, eps_r_per_element=None) -> float:
    """TM0 進行波の P_flow を z=z_min 断面のラインインテグラルで計算する.

    公式（physics_core/power_flow.py 参照）::

        P_flow = -π/(ω ε₀ ε_r) · ∫₀^{r_max} Im[(∂ψ/∂z) ψ*] r dr   (ψ = H_phi)

    軸方向ポインティング S_z = ½Re(E_r·H_θ*) で E_r = (j/(ωε₀ε_r))∂ψ/∂z
    なので、誘電体では断面上の各要素の寄与に **1/ε_r** が掛かる。断面で ε_r が
    r 方向に変わりうるため、係数 -π/(ωε₀) は外に出したまま、各エッジ寄与を
    その要素の ε_r で割る。``eps_r_per_element=None`` なら全域真空。

    Args:
        eigenvector: H_phi 節点値の複素配列 (N,)。
        nodes: 節点座標 (N, 2) = [z, r]。
        elements: 要素節点インデックス (E, 3 or 6)。
        element_order: 要素次数 (1 or 2)。
        omega: 角周波数 [rad/s]。
        eps0: 真空誘電率 [F/m]。
        eps_r_per_element: (Ne,) 要素ごと比誘電率。``None`` で真空。

    Returns:
        P_flow [W]。
    """
    nodes = np.asarray(nodes)
    psi = np.asarray(eigenvector)

    z_min = np.min(nodes[:, 0])
    z_tol = (np.max(nodes[:, 0]) - z_min) * 1e-6
    z_min_set = set(np.where(np.abs(nodes[:, 0] - z_min) < z_tol)[0])

    if element_order == 2:
        edge_defs = [(0, [0, 1, 3]), (1, [1, 2, 4]), (2, [2, 0, 5])]
    else:
        edge_defs = [(0, [0, 1]), (1, [1, 2]), (2, [2, 0])]

    seen_edges = set()
    boundary = []
    for elem_idx, elem in enumerate(elements):
        for eid, idxs in edge_defs:
            global_nodes = tuple(elem[idxs])
            if all(n in z_min_set for n in global_nodes):
                key = tuple(sorted(global_nodes))
                if key not in seen_edges:
                    seen_edges.add(key)
                    boundary.append((eid, global_nodes, elem, elem_idx))

    p_flow_total = 0.0
    for eid, edge_nodes, elem, elem_idx in boundary:
        eps_r = (1.0 if eps_r_per_element is None
                 else float(eps_r_per_element[elem_idx]))
        vtx_coords = nodes[list(elem[:3])]
        psi_e = psi[list(elem)]
        grad_L = grad_area_coordinates(vtx_coords)

        L_edge = abs(nodes[edge_nodes[1], 1] - nodes[edge_nodes[0], 1])
        if L_edge < 1e-15:
            continue

        def integrand(t, eid=eid, psi_e=psi_e, grad_L=grad_L,
                      vtx_coords=vtx_coords):
            L_coords = _bary_on_edge(eid, t)
            if element_order == 1:
                G = L_coords
                grad_G = np.array(grad_L)
            else:
                G = calculate_quadratic_nodal_shape_functions(L_coords)
                grad_G = grad_quadratic_nodal_shape_functions(L_coords, grad_L)
            psi_pt = np.dot(G, psi_e)
            dpsi_dz = np.dot(grad_G[:, 0], psi_e)
            r = np.dot(L_coords, vtx_coords[:, 1])
            return np.imag(dpsi_dz * np.conj(psi_pt)) * r

        # 誘電体: この断面エッジが属する要素の ε_r で割る（E_r ∝ 1/ε_r）
        p_flow_total += edge_line_integral_3point(integrand, L_edge) / eps_r

    return tm0_p_flow_coefficient(omega, eps0) * p_flow_total


# ---------------------------------------------------------------------------
# PEC 境界エッジの抽出
# ---------------------------------------------------------------------------
def _collect_pec_edges(elements, pec_nodes, element_order):
    """PEC 節点集合に両端（+ 中点）が含まれる要素エッジを重複なく返す."""
    pec_set = set(int(n) for n in pec_nodes)
    edge_indices = ([[0, 1, 3], [1, 2, 4], [2, 0, 5]] if element_order == 2
                    else [[0, 1], [1, 2], [2, 0]])
    edges = []
    for elem in elements:
        for idxs in edge_indices:
            if all(n in pec_set for n in elem[idxs]):
                edges.append(elem[idxs])
    return sorted(set(tuple(sorted(e)) for e in edges))


def _wall_loss_integral(pec_edges, nodes, u_abs, element_order) -> float:
    """PEC 壁での ``∫ H^2 r dl`` を全エッジ合計する（ver1 互換）."""
    total = 0.0
    for edge in pec_edges:
        coords = nodes[list(edge)]
        vals_abs = u_abs[list(edge)]
        if element_order == 2:
            dist = [np.linalg.norm(coords[i] - coords[j])
                    for i, j in [(0, 1), (0, 2), (1, 2)]]
            max_idx = int(np.argmax(dist))
            p_idxs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)][max_idx]
            total += calculate_boundary_integral_quadratic(
                coords[list(p_idxs)], vals_abs[list(p_idxs)])
        else:
            p1, p2 = coords
            v1, v2 = vals_abs
            total += 0.5 * (v1 ** 2 * p1[1] + v2 ** 2 * p2[1]) \
                * np.linalg.norm(p2 - p1)
    return total


# ---------------------------------------------------------------------------
# 誘電体損失（ver2.3 摂動法）
# ---------------------------------------------------------------------------
def _dielectric_weight_integrals(nodes, elements, element_order, u,
                                 eps_r_per_element, tan_delta_per_element):
    """Q_diel 用の |E|² 重み付き体積積分ペア (W_total, W_tand) を計算する.

    field_recon と同一の式で H_φ から生の curl 成分
    ``ez_raw = H/r + ∂H/∂r``, ``er_raw = -∂H/∂z`` を求め（E = curl/(jωε₀ε_r)、
    共通定数 1/(ω²ε₀²) は比で相殺されるため掛けない）、7 点求積で

        I_e = ∫_e (|ez_raw|² + |er_raw|²) r dA
        W_total = Σ_e I_e / ε_r(e)          （∝ Σ_e ε_r ∫|E|² r dA）
        W_tand  = Σ_e tanδ(e) · I_e / ε_r(e)

    を返す。``Q_diel = W_total / W_tand``（一様充填で恒等的に 1/tanδ）。

    注意: uᴴKu の剛性行列恒等式は使わない。K の弱形式はクロス項により要素間の
    エネルギー配分が再構成 E と異なり、領域分割の重みとして不正確なため。
    """
    pts = integration_points_triangle[7]
    coords_L = pts["coords"]
    weights = pts["weights"]

    u = np.asarray(u)
    w_total = 0.0
    w_tand = 0.0
    for idx, elem in enumerate(elements):
        verts = nodes[elem[:3]]
        grad_L = grad_area_coordinates(verts)
        v1 = verts[1] - verts[0]
        v2 = verts[2] - verts[0]
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        u_elem = u[elem]
        eps_r = (1.0 if eps_r_per_element is None
                 else float(eps_r_per_element[idx]))
        tand = float(tan_delta_per_element[idx])

        I_e = 0.0
        for p in range(len(weights)):
            L = coords_L[p]
            if element_order == 2:
                G = calculate_quadratic_nodal_shape_functions(L)
                gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
            else:
                G = L
                gradG = np.array(grad_L)
            h = np.dot(u_elem, G)
            grad_h = np.dot(u_elem, gradG)
            dh_dz, dh_dr = grad_h[0], grad_h[1]
            r = float(np.dot(L, verts[:, 1]))
            if abs(r) < 1e-9:
                # 軸上の極限（field_recon と同じロピタル処理。7 点則は全点
                # 内部なので通常は到達しない）
                ez_raw, er_raw = 2.0 * dh_dr, 0.0
            else:
                ez_raw, er_raw = h / r + dh_dr, -dh_dz
            I_e += weights[p] * (abs(ez_raw) ** 2 + abs(er_raw) ** 2) * r
        I_e *= area
        w_total += I_e / eps_r
        w_tand += tand * I_e / eps_r
    return w_total, w_tand


# ---------------------------------------------------------------------------
# 工学パラメータ計算
# ---------------------------------------------------------------------------
@dataclass
class TM0ModeParameters:
    """1 モード分の工学パラメータ."""

    frequency_ghz: float
    stored_energy: float          # U [J]
    p_loss: float                 # 壁損失 [W]
    q_factor: float
    v_eff: float                  # 実効加速電圧（走行時間係数あり）[V]
    v_acc: float                  # 加速電圧（走行時間係数なし）[V]
    rq_eff: float                 # R/Q effective [Ω]
    rq_apparent: float            # R/Q apparent [Ω]
    norm_factor: float
    # 定在波
    r_shunt_eff: float = 0.0
    r_shunt_apparent: float = 0.0
    r_shunt_eff_m: float = 0.0
    r_shunt_apparent_m: float = 0.0
    # 進行波
    p_flow: float = 0.0
    v_group_c: float = 0.0
    v_phase_c: float = 0.0
    attenuation: float = 0.0
    r_shunt_m: float = 0.0
    # ver2.3: 誘電体損失（摂動法）。q_factor は全損失込みの合計 Q。
    q_wall: float = 0.0           # 壁損失のみの Q = ωU/P_loss（従来の q_factor）
    q_diel: float = 0.0           # 誘電体損失のみの Q（0 = 無損失）
    p_diel: float = 0.0           # 誘電体損失 [W]


@dataclass
class TM0Parameters:
    """全モード・全位相の工学パラメータ.

    Attributes:
        analysis_type: "standing" / "traveling"。
        conductivity, beta, n_scan, cell_length: 計算条件。
        per_phase: ``{phase_deg: list[TM0ModeParameters]}``。
        normalized_eigenvectors: ``{phase_deg: np.ndarray(num_modes, N)}``。
    """

    analysis_type: str
    conductivity: float
    beta: float
    n_scan: int
    cell_length: float
    per_phase: dict = field(default_factory=dict)
    normalized_eigenvectors: dict = field(default_factory=dict)


def _process_one_phase(recon, nodes, elements, element_order, frequencies,
                       eigenvectors, M_global, pec_edges, *, analysis_type,
                       conductivity, beta, phase_deg, cell_length, z_scan,
                       eps_r_per_element=None, tan_delta_per_element=None):
    dz = z_scan[1] - z_scan[0]
    is_traveling = analysis_type == "traveling"
    n_modes = len(frequencies)

    params = []
    normalized = np.zeros((n_modes, len(nodes)), dtype=complex)

    for m in range(n_modes):
        freq_ghz = frequencies[m]
        omega = 2 * np.pi * freq_ghz * 1e9

        # 軸上 Ez スキャンと規格化係数（生の固有ベクトルで計算）
        _, ez_axial = recon.axial_ez_scan(eigenvectors[m], freq_ghz,
                                           n_scan=len(z_scan))
        max_abs = np.max(np.abs(ez_axial))
        norm_factor = 1.0 / max_abs if max_abs > 0 else 1.0

        u_norm = (np.asarray(eigenvectors[m]) * norm_factor).astype(complex)
        normalized[m] = u_norm

        # 蓄積エネルギー U = π μ0 Re(u^H M u) / 2
        uMu = np.vdot(u_norm, M_global @ u_norm)
        U = np.pi * MU0 * np.real(uMu) / 2.0

        # 加速電圧
        ph_tt = np.exp(1j * omega * z_scan / (beta * C0))
        v_eff = np.abs(np.trapz(ez_axial * norm_factor * ph_tt, dx=dz))
        v_acc = np.abs(np.trapz(ez_axial * norm_factor, dx=dz))
        rq_eff = (v_eff ** 2) / (omega * U) if U > 0 else 0.0
        rq_apparent = (v_acc ** 2) / (omega * U) if U > 0 else 0.0

        # 壁損失 P_loss
        rs_ohm = np.sqrt(omega * MU0 / (2.0 * conductivity))
        integral_h2r = _wall_loss_integral(
            pec_edges, nodes, np.abs(u_norm), element_order)
        p_loss = np.pi * rs_ohm * integral_h2r
        q_wall = (omega * U / p_loss) if p_loss > 0 else 0.0

        # ver2.3: 誘電体損失（摂動法）。tanδ 無指定/全ゼロなら完全スキップ
        # （p_diel=0.0 で q_factor は従来値とビット同一）。
        q_diel = 0.0
        p_diel = 0.0
        if (tan_delta_per_element is not None
                and np.any(np.asarray(tan_delta_per_element) > 0.0)):
            w_total, w_tand = _dielectric_weight_integrals(
                nodes, elements, element_order, u_norm,
                eps_r_per_element, tan_delta_per_element)
            if w_tand > 0.0:
                q_diel = w_total / w_tand
                p_diel = omega * U / q_diel

        p_total = p_loss + p_diel
        q_factor = (omega * U / p_total) if p_total > 0 else 0.0

        mode = TM0ModeParameters(
            frequency_ghz=freq_ghz, stored_energy=U, p_loss=p_loss,
            q_factor=q_factor, v_eff=v_eff, v_acc=v_acc,
            rq_eff=rq_eff, rq_apparent=rq_apparent, norm_factor=norm_factor,
            q_wall=q_wall, q_diel=q_diel, p_diel=p_diel)

        if is_traveling:
            p_flow = tm0_p_flow_zmin(u_norm, nodes, elements,
                                     element_order, omega,
                                     eps_r_per_element=eps_r_per_element)
            mode.p_flow = p_flow
            mode.v_group_c = ((p_flow * cell_length / U) / C0
                              if U > 0 else 0.0)
            theta_rad = np.deg2rad(phase_deg)
            mode.v_phase_c = ((omega * cell_length) / (theta_rad * C0)
                              if theta_rad > 1e-10 else 0.0)
            mode.attenuation = (p_loss / (2 * p_flow * cell_length)
                                if p_flow != 0 else 0.0)
            mode.r_shunt_m = ((v_eff ** 2 / p_loss) / cell_length
                              if p_loss > 0 else 0.0)
        else:
            mode.r_shunt_eff = (v_eff ** 2 / p_loss) if p_loss > 0 else 0.0
            mode.r_shunt_apparent = (v_acc ** 2 / p_loss) if p_loss > 0 else 0.0
            mode.r_shunt_eff_m = (mode.r_shunt_eff / cell_length
                                  if cell_length > 0 else 0.0)
            mode.r_shunt_apparent_m = (mode.r_shunt_apparent / cell_length
                                       if cell_length > 0 else 0.0)

        params.append(mode)

    return params, normalized


def compute_tm0_parameters(result, classification: BoundaryClassification,
                           conductivity: float = 5.8e7, beta: float = 1.0,
                           n_scan: int = 2000,
                           eps_r_per_element=None,
                           tan_delta_per_element=None) -> TM0Parameters:
    """:class:`TM0Result` から工学パラメータを計算する.

    Args:
        result: :class:`~axicavity_fem.fem_tm0.solver.TM0Result`。
        classification: PEC 壁の節点を含む境界分類（P_loss 用）。
        conductivity: 壁の導電率 [S/m]。
        beta: 粒子の β = v/c（走行時間係数）。
        n_scan: 軸上スキャン点数。
        eps_r_per_element: (Ne,) 要素ごと比誘電率。P_flow の E_r 換算で
            断面エッジを ε_r で割る（誘電体ポート対応）。``None`` で真空。
            U は磁場 M ベースで ε_r 非依存、V_eff は軸上（真空）前提のため不変。
        tan_delta_per_element: (Ne,) 要素ごと誘電正接 tanδ。指定すると摂動法で
            Q_diel / P_diel を計算し、``q_factor`` は合計 Q = ωU/(P_loss+P_diel)
            になる（``q_wall`` が従来の壁損失 Q を保持）。``None``/全ゼロで従来
            どおり。R_shunt・attenuation は従来どおり壁損失ベースのまま。

    Returns:
        :class:`TM0Parameters`。
    """
    nodes = np.asarray(result.nodes)
    elements = np.asarray(result.elements)
    order = result.element_order
    M_global = result.M_global

    z_min, z_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    cell_length = z_max - z_min
    z_scan = np.linspace(z_min, z_max, n_scan)

    pec_edges = _collect_pec_edges(elements, classification.pec_nodes, order)

    recon = TM0FieldReconstructor(nodes, elements, order, result.analysis_type)

    out = TM0Parameters(
        analysis_type=result.analysis_type, conductivity=conductivity,
        beta=beta, n_scan=n_scan, cell_length=cell_length)

    if result.analysis_type == "standing":
        phases = {0.0: (result.frequencies, result.eigenvectors)}
    else:
        phases = {ph: (d["frequencies"], d["eigenvectors"])
                  for ph, d in result.phase_results.items()}

    for phase_deg, (freqs, eigvecs) in phases.items():
        params, normalized = _process_one_phase(
            recon, nodes, elements, order, freqs, eigvecs, M_global,
            pec_edges, analysis_type=result.analysis_type,
            conductivity=conductivity, beta=beta, phase_deg=phase_deg,
            cell_length=cell_length, z_scan=z_scan,
            eps_r_per_element=eps_r_per_element,
            tan_delta_per_element=tan_delta_per_element)
        out.per_phase[phase_deg] = params
        out.normalized_eigenvectors[phase_deg] = normalized

    return out
