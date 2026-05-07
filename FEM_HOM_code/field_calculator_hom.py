import numpy as np
import h5py
from matplotlib.tri import Triangulation
import scipy.constants as const

import os
import sys

# プロジェクトルートを sys.path に追加して FEM_HOM_code をパッケージとして
# 解決可能にする (TM0 の FEM_element_function と名前衝突するため、パッケージ
# 修飾名でインポートする必要がある)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

def reconstruct_edge_index_map(hf):
    keys_array = hf["mesh/edge_map_keys"][:]
    values_array = hf["mesh/edge_map_values"][:]
    edge_index_map = {tuple(keys_array[i]): int(values_array[i]) for i in range(len(values_array))}
    return edge_index_map

class HOMFieldCalculator:
    """
    HOM (High Order Mode) 結果用の HDF5 を読み込み、
    指定された (z, r) 座標における電場Eと磁場Hを計算するクラス。
    """
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.hf = h5py.File(h5_file, 'r')
        
        # メッシュのロード
        self.vertices = self.hf["mesh/vertices"][:]       # (num_nodes, 2) [z, r]
        self.simplices = self.hf["mesh/simplices"][:]     # (num_elements, 3 or 6)
        self.edge_index_map = reconstruct_edge_index_map(self.hf)
        
        # 三角形要素の検索用
        self.tri = Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.simplices[:, :3])
        self.trifinder = self.tri.get_trifinder()
        
        self.available_ns = []
        self._find_available_ns()
        
        self.current_n = None
        self.analysis_type = 'standing'
        self.modes = []
        self.phase_shifts = []
        self.phase_shift = 0.0
        
        # キャッシュ
        self.edge_vectors = None
        self.edge_vectors_lt = None
        self.face_vectors = None
        self.E_theta = None
        self.freq_GHz = 0.0
        self.omega = 0.0
        self.k2 = 0.0
        self.elem_order = 1
        
        self.mu_0 = const.mu_0

    def __del__(self):
        if hasattr(self, 'hf') and self.hf:
            self.hf.close()

    def _find_available_ns(self):
        if "results" in self.hf:
            for key in self.hf["results"].keys():
                if key.startswith("n"):
                    try:
                        n_val = int(key[1:])
                        self.available_ns.append(n_val)
                    except:
                        pass
            self.available_ns.sort()

    def set_n_and_phase(self, n, phase=0.0):
        base = f"/results/n{n}"
        if base not in self.hf:
            return False
            
        self.current_n = n
        if "Periodic" in self.hf[base]:
            self.analysis_type = 'traveling'
            self.phase_shifts = []
            for k in self.hf[base]["Periodic"].keys():
                if k.startswith("PB_Phase_"):
                    v = k.replace("PB_Phase_", "").replace("_deg", "").replace("_", ".")
                    self.phase_shifts.append(float(v))
            self.phase_shifts.sort()
            
            if phase in self.phase_shifts:
                self.phase_shift = phase
            elif len(self.phase_shifts) > 0:
                self.phase_shift = self.phase_shifts[0]
                
            group_name = f"PB_Phase_{self.phase_shift:05.1f}_deg".replace('.', '_')
            mode_path = f"{base}/Periodic/{group_name}"
        else:
            self.analysis_type = 'standing'
            self.phase_shifts = []
            self.phase_shift = 0.0
            mode_path = f"{base}/Normal"
            
        self.modes = []
        if mode_path in self.hf:
            for k in self.hf[mode_path].keys():
                if k.startswith("mode_"):
                    mode_idx = int(k.split("_")[1])
                    freq = self.hf[f"{mode_path}/{k}"].attrs.get("frequency_GHz", 0.0)
                    self.modes.append((mode_idx, freq))
            self.modes.sort(key=lambda x: x[0])
            
        self.frequencies = [m[1] for m in self.modes]
        return True

    def get_mode_parameters(self, n, mode_idx):
        """
        指定された n とモードインデックスの工学パラメータを HDF5 から取得する。
        """
        base = f"/results/n{n}"
        if self.analysis_type == 'standing':
            param_path = f"{base}/Normal/mode_{mode_idx}/engineering_parameters"
        else:
            # 進行波（Periodic）の場合は現在の位相に応じたパス
            group_name = f"PB_Phase_{self.phase_shift:05.1f}_deg".replace('.', '_')
            param_path = f"{base}/Periodic/{group_name}/mode_{mode_idx}/engineering_parameters"

        if param_path in self.hf:
            grp = self.hf[param_path]
            return {
                'stored_energy': grp.attrs.get('stored_energy', 0.0),
                'p_loss': grp.attrs.get('p_loss', 0.0),
                'q_factor': grp.attrs.get('q_factor', 0.0)
            }
        return None

    def load_mode_data(self, mode_idx):
        base = f"/results/n{self.current_n}"
        if self.analysis_type == 'standing':
            mode_path = f"{base}/Normal/mode_{mode_idx}"
            is_periodic = False
        else:
            group_name = f"PB_Phase_{self.phase_shift:05.1f}_deg".replace('.', '_')
            mode_path = f"{base}/Periodic/{group_name}/mode_{mode_idx}"
            is_periodic = True
            
        grp = self.hf[mode_path]
        self.freq_GHz = float(grp.attrs.get("frequency_GHz", 0.0))
        self.omega = 2.0 * np.pi * self.freq_GHz * 1e9
        self.k2 = float(grp.attrs.get("eigenvalue_k2", 0.0))
        self.elem_order = int(grp.attrs.get("elem_order", 1))

        if not is_periodic:
            self.edge_vectors = grp["edge_vectors"][:]
            self.edge_vectors_lt = grp["edge_vectors_lt"][:] if "edge_vectors_lt" in grp else None
            self.face_vectors = grp["face_vectors"][:] if "face_vectors" in grp else None
            self.E_theta = grp["E_theta"][:] if "E_theta" in grp else None
        else:
            ev_re = grp["edge_vectors_re"][:]
            ev_im = grp["edge_vectors_im"][:]
            self.edge_vectors = ev_re + 1j * ev_im
            if "edge_vectors_lt_re" in grp:
                self.edge_vectors_lt = grp["edge_vectors_lt_re"][:] + 1j * grp["edge_vectors_lt_im"][:]
            else:
                self.edge_vectors_lt = None
            if "face_vectors_re" in grp:
                self.face_vectors = grp["face_vectors_re"][:] + 1j * grp["face_vectors_im"][:]
            else:
                self.face_vectors = None
            if "E_theta_re" in grp:
                self.E_theta = grp["E_theta_re"][:] + 1j * grp["E_theta_im"][:]
            else:
                self.E_theta = None

    def calculate_fields(self, z, r, mode_idx, theta_time=0.0):
        """
        指定された r, z における各種電場・磁場成分を返す。
        ユーザー指示に従い、定在波の場合は磁場に -j を掛けて出力する。
        """
        elem_idx = self.trifinder(z, r)
        if elem_idx == -1:
            return None
            
        simplex = self.simplices[elem_idx]
        if self.elem_order == 2:
            simplex_cnr = simplex[:3]
        else:
            simplex_cnr = simplex
            
        verts = self.vertices[simplex_cnr]
        pt = np.array([z, r])
        n = self.current_n
        
        # 三角形各点の座標
        z_coords = verts[:, 0]
        r_coords = verts[:, 1]
        
        # Area coords
        L = calculate_area_coordinates(pt, verts)
        gL = grad_area_coordinates(verts)
        L1, L2, L3 = L
        gL1, gL2, gL3 = gL
        
        # --- 電場 DOF 収集 ---
        local_edges = [
            (simplex_cnr[0], simplex_cnr[1]),
            (simplex_cnr[1], simplex_cnr[2]),
            (simplex_cnr[2], simplex_cnr[0]),
        ]
        
        dtype = complex if np.iscomplexobj(self.edge_vectors) else float
        ct_dofs = np.zeros(3, dtype=dtype)
        lt_dofs = np.zeros(3, dtype=dtype)
        for k, (n1, n2) in enumerate(local_edges):
            key = tuple(sorted((n1, n2)))
            edge_idx = self.edge_index_map[key]
            sign = 1 if n1 < n2 else -1
            ct_dofs[k] = sign * self.edge_vectors[edge_idx]
            if self.elem_order == 2 and self.edge_vectors_lt is not None:
                lt_dofs[k] = self.edge_vectors_lt[edge_idx]
                
        if self.elem_order == 2 and self.face_vectors is not None:
            f0 = self.face_vectors[2 * elem_idx]
            f1 = self.face_vectors[2 * elem_idx + 1]
        else:
            f0 = 0.0
            f1 = 0.0
            
        # Nodal DOFs for E_theta
        if self.E_theta is not None:
            if self.elem_order == 2:
                nodal_dofs = self.E_theta[simplex]
                G = calculate_quadratic_nodal_shape_functions(L)
                gradG = grad_quadratic_nodal_shape_functions(L, gL)
            else:
                nodal_dofs = self.E_theta[simplex_cnr]
                G = L
                gradG = np.array(gL)
            # 注: n=1 ダイポールモードでは E_theta(r=0) ≠ 0 が正則な状態。
            # 軸上では E_r + i*E_theta = 0 の regularity 条件により
            # (1/r)*d(r*E_theta)/dr + j*n*E_r/r の個別項は発散するが和は有限となる。
            # 1次 Nédélec では数値的に cancellation を保証できないため、
            # 可視化側でパーセンタイルクリップによりアウトライアを抑制する。
        else:
            n_nodes = 6 if self.elem_order == 2 else 3
            nodal_dofs = np.zeros(n_nodes, dtype=dtype)
            G = np.zeros(n_nodes)
            gradG = np.zeros((n_nodes, 2))

        # --- 電場 (E) の計算 ---
        if self.elem_order == 2:
            N = calculate_edge_shape_functions_2nd(pt, verts)
            vec = (ct_dofs[0]*N[0] + ct_dofs[1]*N[1] + ct_dofs[2]*N[2] +
                   lt_dofs[0]*N[3] + lt_dofs[1]*N[4] + lt_dofs[2]*N[5] +
                   f0*N[6] + f1*N[7])
        else:
            N = calculate_edge_shape_functions(pt, verts)
            vec = ct_dofs[0]*N[0] + ct_dofs[1]*N[1] + ct_dofs[2]*N[2]
            
        Ez_comp = vec[0]
        Er_comp = vec[1]
        Etheta_comp = np.dot(nodal_dofs, G)
        
        # E_theta の偏微分
        # G_z = gradG[:, 0], G_r = gradG[:, 1]
        dEtheta_dz = np.dot(nodal_dofs, gradG[:, 0])
        dEtheta_dr = np.dot(nodal_dofs, gradG[:, 1])
        
        # --- カールの計算 (Faraday Law) ---
        if self.elem_order == 2:
            curls = calculate_curl_edge_shape_functions_2nd(pt, verts)
            curl_Ezr = (ct_dofs[0]*curls[0] + ct_dofs[1]*curls[1] + ct_dofs[2]*curls[2] +
                        lt_dofs[0]*curls[3] + lt_dofs[1]*curls[4] + lt_dofs[2]*curls[5] +
                        f0*curls[6] + f1*curls[7])
        else:
            A2 = calculate_triangle_area_double(verts)
            curl_1st = 2.0 / A2
            curl_Ezr = (ct_dofs[0] + ct_dofs[1] + ct_dofs[2]) * curl_1st

        # H = (j / \omega \mu_0) \nabla \times E
        safe_r = r if abs(r) > 1e-9 else 1e-9

        # d/dr (r E_theta) = E_theta + r * dE_theta/dr
        drEtheta_dr = Etheta_comp + safe_r * dEtheta_dr

        # ∇×E (円筒座標, FEM の実数 φ 分解: E_z/E_r ~ sin(nφ), E_θ ~ cos(nφ))
        #
        # FEM 行列は実数 (cos/sin 規約) のため ∂/∂φ は純実数係数になる:
        #   ∂[sin(nφ)]/∂φ = +n cos(nφ)  →  係数 +n（虚数単位なし）
        #   ∂[cos(nφ)]/∂φ = -n sin(nφ)  →  係数 -n（虚数単位なし）
        #
        # (∇×E)_r amp [cos(nφ) 成分] = (n/r)E_z - ∂E_θ/∂z
        # (∇×E)_θ amp [sin(nφ) 成分] = ∂E_r/∂z - ∂E_z/∂r   ← curl_Ezr から取得
        # (∇×E)_z amp [cos(nφ) 成分] = (1/r)∂(rE_θ)/∂r - (n/r)E_r
        curl_E_r     = (n / safe_r) * Ez_comp - dEtheta_dz
        curl_E_theta = curl_Ezr
        curl_E_z     = (1.0 / safe_r) * drEtheta_dr - (n / safe_r) * Er_comp
        
        coef = 1j / (self.omega * self.mu_0)

        Htheta_comp = coef * curl_E_theta
        if abs(r) > 1e-9:
            Hz_comp = coef * curl_E_z
            Hr_comp = coef * curl_E_r
        else:
            # r=0 における L'Hôpital 解析的極限
            # FEM の実数 φ 分解 (E_z/E_r ~ sin(nφ), E_θ ~ cos(nφ)) を前提とする。
            #
            # curl_E_r = (n/r)E_z - ∂E_θ/∂z  は r→0 で不定形 (E_z(r=0)=0 なので 0/0)
            #   L'Hôpital: n·∂E_z/∂r|_0 - ∂E_θ/∂z|_0
            #   TE モード(E_z=0): n·0 - ∂E_θ/∂z|_0 = -∂E_θ/∂z|_0  (軸上補間済の E_θ を使用)
            #   TM モード(E_z≠0): +n·∂E_z/∂r|_0 の項も本来は必要
            #     (FEM の sin(nφ) 規約では TM n=1 の H_r も r=0 で非ゼロ)
            #     しかし TM モードでは E_θ ≈ 0 かつ form 上 H_z が支配的なため，
            #     r=0 の H_r 誤差は実用上軽微。精度が必要な場合は ∂E_z/∂r を追加のこと。
            #
            # curl_E_z = (1/r)∂(rE_θ)/∂r - (n/r)E_r  は r→0 で不定形
            #   n=1: H_z ~ J_1(0)=0 → 物理的に 0
            #   n>=2: J_n(0)=0, J_n'(0)=0 → 物理的に 0
            #   → 数値的キャンセレーション不安定なので 0 に固定
            if n == 0:
                Hz_comp = 0
                Hr_comp = 0
            else:
                # H_z(0) = 0 (analytic, J_n(0)=0 for n>=1)
                Hz_comp = 0
                # H_r(0) = -coef * ∂E_θ/∂z|_0   (L'Hôpital 極限)
                Hr_comp = coef * (-dEtheta_dz)

        # --- 時間位相の適用 ---
        theta_rad = np.deg2rad(theta_time)
        comp_phase = np.exp(1j * theta_rad)
        
        if self.analysis_type == 'standing':
            # 定在波: 磁場には -j を掛けて、最大振幅を電場と同時に表示する
            Ez_plot = Ez_comp.real
            Er_plot = Er_comp.real
            Etheta_plot = Etheta_comp.real
            
            H_mod = -1j * np.array([Hz_comp, Hr_comp, Htheta_comp])
            Hz_plot = H_mod[0].real
            Hr_plot = H_mod[1].real
            Htheta_plot = H_mod[2].real
            
        else:
            # 進行波: 指定された位相における瞬時場 (Real part)
            Ez_plot = (Ez_comp * comp_phase).real
            Er_plot = (Er_comp * comp_phase).real
            Etheta_plot = (Etheta_comp * comp_phase).real
            
            Hz_plot = (Hz_comp * comp_phase).real
            Hr_plot = (Hr_comp * comp_phase).real
            Htheta_plot = (Htheta_comp * comp_phase).real

        return {
            'Ez': Ez_plot,
            'Er': Er_plot,
            'E_theta': Etheta_plot,
            'E_abs': np.sqrt(Ez_plot**2 + Er_plot**2 + Etheta_plot**2), # rz平面上の表示だけでなく、絶対値の参考として
            
            'Hz': Hz_plot,
            'Hr': Hr_plot,
            'H_theta': Htheta_plot,
            'H_abs': np.sqrt(Hz_plot**2 + Hr_plot**2 + Htheta_plot**2),
        }

    def calculate_all_node_fields(self, mode_idx, theta_time=0.0):
        """
        補間（コンター描画等）用に節点上での値を計算（擬似的に）する。
        各節点について、そこを含む1つの要素を使って calculate_fields を呼ぶ。
        """
        num_nodes = len(self.vertices)
        node_to_elem = {}
        for elem_idx, simplex in enumerate(self.simplices[:, :3]):
            for node_idx in simplex:
                if node_idx not in node_to_elem:
                    node_to_elem[node_idx] = elem_idx
                    
        Ez = np.zeros(num_nodes)
        Er = np.zeros(num_nodes)
        E_theta = np.zeros(num_nodes)
        Hz = np.zeros(num_nodes)
        Hr = np.zeros(num_nodes)
        H_theta = np.zeros(num_nodes)
        
        for i in range(num_nodes):
            z, r = self.vertices[i]
            # r=0上の特異点を避けるためわずかにずらす (計算上の都合)
            calc_r = max(r, 1e-10)
            res = self.calculate_fields(z, calc_r, mode_idx, theta_time)
            if res:
                Ez[i] = res['Ez']
                Er[i] = res['Er']
                E_theta[i] = res['E_theta']
                Hz[i] = res['Hz']
                Hr[i] = res['Hr']
                H_theta[i] = res['H_theta']
                
        return {
            'Ez': Ez, 'Er': Er, 'E_theta': E_theta,
            'Hz': Hz, 'Hr': Hr, 'H_theta': H_theta
        }
