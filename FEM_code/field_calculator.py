import numpy as np
import h5py
import os
import sys

# 自分自身のディレクトリをパスに追加して、同じ階層のモジュールをインポート可能にする
sys.path.append(os.path.dirname(__file__))

from scipy.spatial import KDTree
from FEM_element_function import (
    calculate_area_coordinates, 
    grad_area_coordinates,
    calculate_quadratic_nodal_shape_functions,
    grad_quadratic_nodal_shape_functions
)

class FieldCalculator:
    def __init__(self, h5_file):
        """
        HDF5ファイルから解析結果を読み込み、計算の準備を行う。
        """
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Result file not found: {h5_file}")
            
        with h5py.File(h5_file, 'r') as f:
            self.nodes = f['mesh/nodes'][:]
            self.elements = f['mesh/elements'][:]
            self.mesh_order = f['mesh'].attrs['order']
            self.analysis_type = f.attrs['analysis_type']
            
            # 位相データのリストを取得
            self.phase_shifts = f.attrs.get('phase_shifts', [])
            if len(self.phase_shifts) == 0:
                # 定在波または単一位相（旧形式）
                self.phase_shift = f.attrs.get('phase_shift', 0.0)
                if 'results/frequencies' in f:
                    self.frequencies = f['results/frequencies'][:]
                    # 常に複素数として読み込む (dtype=complex)
                    #self.eigenvectors = f['results/eigenvectors'][:].astype(complex)
                    #やはり定在波は実数として読み込む
                    if self.analysis_type == 'standing' :
                        self.eigenvectors = f['results/eigenvectors'][:]
                    else :
                        self.eigenvectors = f['results/eigenvectors'][:].astype(complex)
                else:
                    self.frequencies = np.array([])
                    self.eigenvectors = np.array([], dtype=complex)
            else:
                # 複数位相（新形式）: デフォルトで最初の位相を読み込む
                self.phase_shift = self.phase_shifts[0]
                phase_path = f"results/phase_{self.phase_shift}"
                self.frequencies = f[f"{phase_path}/frequencies"][:]
                self.eigenvectors = f[f"{phase_path}/eigenvectors"][:].astype(complex)

            # 平均メッシュサイズの概算 (sqrt(全面積/要素数))
            total_area = 0.0
            for elem in self.elements:
                v = self.nodes[elem[:3]]
                total_area += 0.5 * np.abs((v[1,0]-v[0,0])*(v[2,1]-v[0,1]) - (v[2,0]-v[0,0])*(v[1,1]-v[0,1]))
            self.average_mesh_size = np.sqrt(2.0 * total_area / len(self.elements)) if len(self.elements) > 0 else 0
            
            # 追加の結果データ (壁面損失など) - 複数のパスをチェック
            self.wall_loss = None
            if 'results/wall_loss' in f:
                self.wall_loss = f['results/wall_loss'][:]
            elif 'engineering_parameters/p_loss' in f:
                self.wall_loss = f['engineering_parameters/p_loss'][:]
            # post_processed パスからも読む (processed HDF5 対応)
            if self.wall_loss is None and 'post_processed' in f:
                ph_key = f'post_processed/phase_{self.phase_shift}/engineering_parameters/p_loss'
                if ph_key in f:
                    self.wall_loss = f[ph_key][:]
                
            # PEC境界ノード - 複数のパスをチェック
            self.pec_nodes = []
            if 'mesh/boundary_nodes/PEC' in f:
                self.pec_nodes = f['mesh/boundary_nodes/PEC'][:]
            elif 'mesh/physical_groups/PEC' in f:
                self.pec_nodes = f['mesh/physical_groups/PEC'][:]
            elif 'mesh/physical_groups' in f:
                # PECやWallなどの名前を含むグループを探す
                for name in f['mesh/physical_groups'].keys():
                    if 'PEC' in name.upper() or 'WALL' in name.upper():
                        self.pec_nodes = f[f'mesh/physical_groups/{name}'][:]
                        break
            
        # 要素検索を高速化するための準備
        # 各要素の重心を計算し、KDTreeを構築する
        element_centers = np.mean(self.nodes[self.elements[:, :3]], axis=1)
        self.tree = KDTree(element_centers)
        
        # 物理定数
        self.eps0 = 8.8541878128e-12
        self.h5_file = h5_file # 再読み込み用にファイルパスを保持
        
        # 解析タイプの自動判定 (Standing or Traveling)
        test_eig = self.eigenvectors[0] if len(self.eigenvectors) > 0 else np.array([])
        self.is_traveling = np.iscomplexobj(test_eig) and np.max(np.abs(np.imag(test_eig))) > 1e-12

    def calculate_peak_fields(self, mode_index=0):
        """
        全節点における磁場 H_theta, Psi, 電場 Ez, Er の最大振幅 (Peak) を計算する。
        """
        # すべての成分を複素数として計算
        res_complex = self.calculate_all_node_fields(mode_index, theta=0.0, return_complex=True)
        
        return {
            'H_theta': np.abs(res_complex['H_theta']),
            'Psi': np.abs(res_complex['Psi']),
            'Ez': np.abs(res_complex['Ez']),
            'Er': np.abs(res_complex['Er']),
            'E_abs': res_complex['E_abs'] # これはすでに絶対値
        }

    def set_phase(self, phase):
        """
        別の位相の結果を読み込む。
        """
        if phase not in self.phase_shifts:
            if phase == self.phase_shift: return
            raise ValueError(f"Phase {phase} not found in result file.")
            
        with h5py.File(self.h5_file, 'r') as f:
            phase_path = f"results/phase_{phase}"
            self.frequencies = f[f"{phase_path}/frequencies"][:]
            self.eigenvectors = f[f"{phase_path}/eigenvectors"][:]
            self.phase_shift = phase
            
            # 壁面損失なども位相ごとのグループにある場合は更新する
            eng_path = f"{phase_path}/engineering_parameters"
            if eng_path in f:
                if 'p_loss' in f[eng_path]:
                    self.wall_loss = f[f"{eng_path}/p_loss"][:]
            # post_processed パスからも読む (processed HDF5 対応)
            if self.wall_loss is None and 'post_processed' in f:
                pp_eng_path = f'post_processed/phase_{phase}/engineering_parameters/p_loss'
                if pp_eng_path in f:
                    self.wall_loss = f[pp_eng_path][:]

    def find_element(self, z, r):
        """
        地点 (z, r) を含む要素のインデックスと重心座標を返す。
        """
        p = np.array([z, r])
        # 重心が近い要素から順に判定
        dist, idxs = self.tree.query(p, k=10) # 上位10個の候補
        
        for idx in idxs:
            elem = self.elements[idx]
            vertices = self.nodes[elem[:3]]
            L = calculate_area_coordinates(p, vertices)
            
            # 全ての重心座標が 0 以上（わずかな誤差を許容）であれば要素内
            if np.all(L >= -1e-6):
                return idx, L
                
        return None, None

    def calculate_all_node_fields(self, mode_index=0, theta=0.0, return_complex=False):
        """
        全節点における磁場 H_theta, 電気力線成分 Psi, 電場 Ez, Er を一括計算する。
        theta [rad] は瞬時位相 (exp(j*theta))。
        """
        if mode_index >= len(self.frequencies):
            raise IndexError("Mode index out of range")
            
        n_nodes = len(self.nodes)
        ez_total = np.zeros(n_nodes, dtype=complex)
        er_total = np.zeros(n_nodes, dtype=complex)
        node_count = np.zeros(n_nodes)
        
        freq = self.frequencies[mode_index] * 1e9
        omega = 2 * np.pi * freq
        coeff = -1j / (omega * self.eps0) # マクスウェル方程式 ∇×H = jωεE より E = (1/jωε)∇×H = (-j/ωε)∇×H
        
        # 位相因子を掛ける
        phase_factor = np.exp(1j * theta)
        u_global = self.eigenvectors[mode_index] * phase_factor
        
        # 1. 電場の計算（要素ごとの寄与を節点で平均化）
        for i_elem, elem in enumerate(self.elements):
            vertices = self.nodes[elem[:3]]
            u_elem = u_global[elem]
            grad_L = grad_area_coordinates(vertices)
            
            # 各節点での値を計算
            if self.mesh_order == 1:
                # 1次要素: 勾配は要素内で一定
                grad_h = np.dot(u_elem, grad_L) # [dH/dz, dH/dr]
                for node_idx in elem:
                    r = self.nodes[node_idx, 1]
                    h_val = u_global[node_idx]
                    if abs(r) < 1e-9:
                        ez = 2.0 * grad_h[1]
                        er = 0.0
                    else:
                        ez = (h_val / r) + grad_h[1]
                        er = -grad_h[0]
                    ez_total[node_idx] += ez
                    er_total[node_idx] += er
                    node_count[node_idx] += 1
            else:
                # 2次要素: 各節点での勾配を計算
                # L_nodes = [[1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0], [0,0.5,0.5], [0.5,0,0.5]]
                L_nodes = np.array([[1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0], [0,0.5,0.5], [0.5,0,0.5]])
                for i_local in range(6):
                    node_idx = elem[i_local]
                    L = L_nodes[i_local]
                    gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
                    grad_h = np.dot(u_elem, gradG)
                    
                    r = self.nodes[node_idx, 1]
                    h_val = u_global[node_idx]
                    if abs(r) < 1e-9:
                        ez = 2.0 * grad_h[1]
                        er = 0.0
                    else:
                        ez = (h_val / r) + grad_h[1]
                        er = -grad_h[0]
                    ez_total[node_idx] += ez
                    er_total[node_idx] += er
                    node_count[node_idx] += 1
        
        # 平均化
        if self.analysis_type == 'standing' :
            ez_node = np.zeros(n_nodes)
            er_node = np.zeros(n_nodes)
            mask = node_count > 0
            ez_node[mask] = (ez_total[mask] / node_count[mask]) 
            er_node[mask] = (er_total[mask] / node_count[mask]) 

        else :
            ez_node = np.zeros(n_nodes, dtype=complex)
            er_node = np.zeros(n_nodes, dtype=complex)
            mask = node_count > 0
            ez_node[mask] = (ez_total[mask] / node_count[mask]) * coeff
            er_node[mask] = (er_total[mask] / node_count[mask]) * coeff
        
        h_theta = u_global
        if self.analysis_type == 'standing' :
            psi = self.nodes[:, 1] * h_theta
        else :
            psi = -1j * self.nodes[:, 1] * h_theta # E field と位相を合わせるための -j

        if self.analysis_type == 'standing' :
            return {
                'H_theta': h_theta,
                'Psi': psi,
                'Ez': ez_node,
                'Er': er_node,
                'E_abs': np.sqrt(np.abs(ez_node)**2 + np.abs(er_node)**2) # Peak magnitude
            }
        else :
            if return_complex:
                return {
                    'H_theta': h_theta,
                    'Psi': psi,
                    'Ez': ez_node,
                    'Er': er_node,
                    'E_abs': np.sqrt(np.abs(ez_node)**2 + np.abs(er_node)**2) # Peak magnitude
                }

            # 瞬時値（実部）を返す
            return {
                'H_theta': np.real(h_theta),
                'Psi': np.real(psi),
                'Ez': np.real(ez_node),
                'Er': np.real(er_node),
                'E_abs': np.sqrt(np.real(ez_node)**2 + np.real(er_node)**2) # Instant magnitude
            }

    def calculate_grid_fields(self, mode_index=0, z_steps=50, r_steps=50, theta=0.0):
        """
        領域内を格子状にサンプリングして電磁場を計算する（quiver表示用）。
        theta [rad] は瞬時位相。
        """
        z_min, r_min = np.min(self.nodes, axis=0)
        z_max, r_max = np.max(self.nodes, axis=0)
        
        z_grid = np.linspace(z_min, z_max, z_steps)
        r_grid = np.linspace(r_min, r_max, r_steps)
        zz, rr = np.meshgrid(z_grid, r_grid)
        
        ez_grid = np.zeros_like(zz)
        er_grid = np.zeros_like(zz)
        mask = np.zeros_like(zz, dtype=bool)
        
        for i in range(r_steps):
            for j in range(z_steps):
                res = self.calculate_fields(zz[i,j], rr[i,j], mode_index, theta=theta)
                if res:
                    ez_grid[i,j] = res['Ez']
                    er_grid[i,j] = res['Er']
                    mask[i,j] = True
                    
        return {
            'zz': zz,
            'rr': rr,
            'Ez': ez_grid,
            'Er': er_grid,
            'mask': mask
        }

    def get_boundary_edges(self):
        """
        全要素から、一度しか現れないエッジ（外部境界）を抽出する。
        """
        from collections import Counter
        edges = []
        # 各要素の頂点(0,1,2)のみを使用
        for elem in self.elements[:, :3]:
            e1 = tuple(sorted((elem[0], elem[1])))
            e2 = tuple(sorted((elem[1], elem[2])))
            e3 = tuple(sorted((elem[2], elem[0])))
            edges.extend([e1, e2, e3])
        
        counts = Counter(edges)
        boundary_edges = [edge for edge, count in counts.items() if count == 1]
        return boundary_edges

    def get_pec_fields(self, mode_index=0):
        """
        PEC境界ノード位置における電場ベクトルを返す。
        """
        if len(self.pec_nodes) == 0:
            return None
            
        all_fields = self.calculate_all_node_fields(mode_index)
        idx = self.pec_nodes
        return {
            'z': self.nodes[idx, 0],
            'r': self.nodes[idx, 1],
            'Ez': all_fields['Ez'][idx],
            'Er': all_fields['Er'][idx]
        }

    def calculate_fields(self, z, r, mode_index=0, theta=0.0, return_complex=False):
        """
        地点 (z, r) における H_theta, Ez, Er を計算する。
        """
        if mode_index >= len(self.frequencies):
            raise IndexError("Mode index out of range")
        
        # 位相因子
        phase_factor = np.exp(1j * theta)
        
        idx, L = self.find_element(z, r)
        if idx is None:
            return None # 領域外
            
        elem = self.elements[idx]
        vertices = self.nodes[elem[:3]]
        u_global = self.eigenvectors[mode_index] * phase_factor
        u_elem = u_global[elem]
        
        # 形状関数と勾配の計算
        if self.mesh_order == 1:
            # 1次要素 (u = sum u_i L_i)
            G = L
            grad_L = grad_area_coordinates(vertices)
            gradG = np.array(grad_L)
        else:
            # 2次要素 (u = sum u_i G_i)
            G = calculate_quadratic_nodal_shape_functions(L)
            grad_L = grad_area_coordinates(vertices)
            gradG = grad_quadratic_nodal_shape_functions(L, grad_L)
            
        # 磁場とその勾配の合成
        h_theta = np.dot(u_elem, G)
        grad_h = np.dot(u_elem, gradG) # [dH/dz, dH/dr]
        
        dh_dz = grad_h[0]
        dh_dr = grad_h[1]
        
        # 電場の計算
        freq = self.frequencies[mode_index] * 1e9
        omega = 2 * np.pi * freq
        if self.analysis_type == 'standing' :
            coeff = 1 / (omega * self.eps0)
        else :
            coeff = -1j / (omega * self.eps0)
        
        # 軸上判定
        if abs(r) < 1e-9:
            ez = 2.0 * dh_dr
            er = 0.0
        else:
            ez = (h_theta / r) + dh_dr
            er = -dh_dz
            
        # 電界の各成分（複素数）
        Ez_c = ez * coeff
        Er_c = er * coeff

        if return_complex:
            return {
                'H_theta': h_theta,
                'Ez': Ez_c,
                'Er': Er_c,
                'E_abs': np.sqrt(np.abs(Ez_c)**2 + np.abs(Er_c)**2)
            }

        # 瞬時値（実部）を返す
        return {
            'H_theta': np.real(h_theta),
            'Ez': np.real(Ez_c),
            'Er': np.real(Er_c),
            'E_abs': np.sqrt(np.real(Ez_c)**2 + np.real(Er_c)**2)
        }

# --- テスト用 ---
if __name__ == "__main__":
    import sys
    test_h5 = "analysis_result.h5"
    if len(sys.argv) > 1:
        test_h5 = sys.argv[1]
        
    if os.path.exists(test_h5):
        calc = FieldCalculator(test_h5)
        # 1. 一括計算のテスト
        print("Testing bulk calculation...")
        res_bulk = calc.calculate_all_node_fields(mode_index=0)
        print(f"Bulk calc finished. Nodes: {len(res_bulk['H_theta'])}")
        
        # 2. 地点計算のテスト
        print("\nTesting point calculation...")
        # 軸上の適当な点でテスト
        res = calc.calculate_fields(0.0, 0.0, mode_index=0)
        if res:
            print(f"On axis: H={res['H_theta']:.4e}, Ez={res['Ez']:.4e}, Er={res['Er']:.4e}")
        
        # 軸から少し離れた点
        res = calc.calculate_fields(0.0, 0.01, mode_index=0)
        if res:
            print(f"Off axis: H={res['H_theta']:.4e}, Ez={res['Ez']:.4e}, Er={res['Er']:.4e}")
    else:
        print(f"Test file {test_h5} not found. Skip tests.")
