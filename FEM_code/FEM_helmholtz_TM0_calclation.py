import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys # エラー終了のため
#import h5py
import os
import pickle

from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, block_diag, find
from scipy.linalg import eig
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg import eigsh 
from scipy.spatial import KDTree

import time # Timing for debugging complex parts

import gmsh


from FEM_element_function import calculate_triangle_area_double, calculate_area_coordinates, grad_area_coordinates
from gaussian_quadrature_triangle import gaussian_quadrature_triangle, integration_points_triangle


import gmsh
import numpy as np

def load_gmsh_mesh(filename, element_order=1):
    """
    Gmshのメッシュファイルを読み込み、FEM計算用のNumPy配列を出力する関数
    
    Parameters:
    -----------
    filename : str
        読み込むメッシュファイルのパス
    element_order : int
        要素の次数。1 なら 1次要素(3節点)、2 なら 2次要素(6節点)として読み込む。
    """
    
    # 次数に応じて要素タイプと節点数を設定
    if element_order == 1:
        elem_type = 2       # 3節点三角形
        nodes_per_elem = 3
    elif element_order == 2:
        elem_type = 9       # 6節点三角形
        nodes_per_elem = 6
    else:
        raise ValueError("element_order は 1 か 2 を指定してください。")

    if not gmsh.isInitialized():
        gmsh.initialize()
        
    try:
        gmsh.open(filename)
        
        # 1. 節点 (Nodes) の取得 (2次要素の中間節点もすべて自動で取得されます)
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(nodeCoords).reshape(-1, 3)[:, :2]
        
        tag2idx = {tag: i for i, tag in enumerate(nodeTags)}
        
        # 2. 要素 (Elements) の取得
        _, elemNodeTags = gmsh.model.mesh.getElementsByType(elem_type)
        
        # 次数に合わせた nodes_per_elem（3 または 6）で reshape
        elements = np.array([tag2idx[tag] for tag in elemNodeTags], dtype=int).reshape(-1, nodes_per_elem)
        
        # 3. 物理グループ (Physical Groups) の取得 (全く同じ処理でOKです)
        physical_groups = {}
        p_groups = gmsh.model.getPhysicalGroups()
        
        for dim, p_tag in p_groups:
            name = gmsh.model.getPhysicalName(dim, p_tag) or f"PhysicalGroup_{dim}D_{p_tag}"
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, p_tag)
            
            group_nodes_tags =[]
            for entity_tag in entities:
                nTags, _, _ = gmsh.model.mesh.getNodes(dim, entity_tag, includeBoundary=True)
                group_nodes_tags.extend(nTags)
            
            group_nodes_tags = np.unique(group_nodes_tags)
            group_nodes_idx = np.array([tag2idx[tag] for tag in group_nodes_tags if tag in tag2idx], dtype=int)
            
            if name in physical_groups:
                physical_groups[name] = np.unique(np.concatenate([physical_groups[name], group_nodes_idx]))
            else:
                physical_groups[name] = group_nodes_idx
            
    finally:
        gmsh.finalize()
        
    return nodes, elements, physical_groups



def assemble_stiffness_matrix_element(vertices, mesh_order = 1): # k_ij
    n_points=4
    if mesh_order == 1:
        r1, r2, r3 = vertices
        r1_r = r1[1]
        r2_r = r2[1]
        r3_r = r3[1]
        rc = (r1_r + r2_r + r3_r)/3
        A2 = calculate_triangle_area_double(vertices)
        A = A2 / 2
        grad_L = grad_area_coordinates(vertices)

        def func_rover_LiLj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            #rv = np.array([z, r])
            L = [L1, L2, L3]
            return L[i] * L[j] /r
        

        grad_L = grad_area_coordinates(vertices)

        K_e = np.zeros((3, 3))

        for i in range(3) :
            for j in range(3) :
                term1 = A*rc *(grad_L[i][0]*grad_L[j][0] + grad_L[i][1]*grad_L[j][1])
                term2 = gaussian_quadrature_triangle(func_rover_LiLj, vertices, n_points, i, j )
                term3 = A/3 *grad_L[j][1] +A/3*grad_L[i][1]
                K_e [i, j] = term1 + term2 + term3

        return K_e
    

    elif mesh_order == 2 :
        #vertices = vertices[:3]
        def func_r_gradGi_garadGj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            L = [L1, L2, L3]
            grad_L = grad_area_coordinates(vertices)
            gradG = [0] *6
            for k in range(0,6) :
                if k <3 : gradG[k] = ( 4*L[k] -1 ) *grad_L[k]
                else :    gradG[k] = 4*( L[(k-2)%3]*grad_L[k-3] + L[k-3]*grad_L[(k-2)%3] )
            return r*np.dot(gradG[i], gradG[j])

        def func_rover_GiGj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            L = [L1, L2, L3]
            G = [0] *6
            for k in range(0,6) :
                if k <3 : G[k] = (2*L[k]-1)*L[k]
                else :    G[k] = 4*L[k-3]*L[(k-2)%3]
            return G[i] * G[j] /r
        
        def func_Gi_gradGj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            L = [L1, L2, L3]
            grad_L = grad_area_coordinates(vertices)
            gradG = [0] *6
            for k in range(0,6) :
                if k <3 : gradG[k] = ( 4*L[k] -1 ) *grad_L[k]
                else :    gradG[k] = 4*( L[(k-2)%3]*grad_L[k-3] + L[k-3]*grad_L[(k-2)%3] )
            G = [0] *6
            for k in range(0,6) :
                if k <3 : G[k] = (2*L[k]-1)*L[k]
                else :    G[k] = 4*L[k-3]*L[(k-2)%3]
            # G[j] = (G[j][0], G[j][1])
            return G[i] *gradG[j][1]
        
        def func_k_all(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            L = [L1, L2, L3]
            grad_L = grad_area_coordinates(vertices)
            gradG = [0] *6
            for k in range(0,6) :
                if k <3 : gradG[k] = ( 4*L[k] -1 ) *grad_L[k]
                else :    gradG[k] = 4*( L[(k-2)%3]*grad_L[k-3] + L[k-3]*grad_L[(k-2)%3] )
            G = [0] *6
            for k in range(0,6) :
                if k <3 : G[k] = (2*L[k]-1)*L[k]
                else :    G[k] = 4*L[k-3]*L[(k-2)%3]

            return r*np.dot(gradG[i], gradG[j]) +  G[i] * G[j] /r +G[i] *gradG[j][1] +G[j] *gradG[i][1]


        K_e = np.zeros((6, 6))

        curve_index = check_curve_element( vertices )
        if curve_index != 0 :
            #print(f'curve@{vertices[curve_index]} {np.sqrt(vertices[curve_index][0]**2 +vertices[curve_index][1]**2)} ' )

            def func_k_all_Jacobian(L1, L2, L3, vertices, i, j, curve_index):
                rv1, rv2, rv3 = vertices[:3]
                z1, z2, z3 = rv1[0], rv2[0], rv3[0]
                r1, r2, r3 = rv1[1], rv2[1], rv3[1]
                r = L1*r1+L2*r2+L3*r3
                z = L1*z1+L2*z2+L3*z3
                L = [L1, L2, L3]
                grad_L = grad_area_coordinates(vertices[:3])
                G = [0] *6
                for k in range(0,6) :
                    if k <3 : G[k] = (2*L[k]-1)*L[k]
                    else :    G[k] = 4*L[k-3]*L[(k-2)%3]
                gradG = [0] *6
                for k in range(0,6) :
                    if k <3 : gradG[k] = ( 4*L[k] -1 ) *grad_L[k]
                    else :    gradG[k] = 4*( L[(k-2)%3]*grad_L[k-3] + L[k-3]*grad_L[(k-2)%3] )

                rv_curve_orign = vertices[curve_index]
                rv_curve_convert = 0.5*(vertices[(curve_index-2)%3]+ vertices[curve_index-3])
                J = 1+np.dot(gradG[curve_index], ( rv_curve_orign - rv_curve_convert ) )

                return (r*np.dot(gradG[i], gradG[j]) +  G[i] * G[j] /r +G[i] *gradG[j][1] +G[j] *gradG[i][1])*J
            for i in range(6) :
                for j in range(6) :
                    K_e [i, j] = gaussian_quadrature_triangle( func_k_all_Jacobian, vertices, n_points, i, j, curve_index )

        else :
            for i in range(6) :
                for j in range(6) :
                    #term1 = gaussian_quadrature_triangle( func_r_gradGi_garadGj, vertices, n_points, i, j )
                    #term2 = gaussian_quadrature_triangle( func_rover_GiGj, vertices, n_points, i, j )
                    #term3 = gaussian_quadrature_triangle(func_Gi_gradGj, vertices, n_points, i, j ) + gaussian_quadrature_triangle(func_Gi_gradGj, vertices, n_points, j, i )

                    #K_e [i, j] = term1 + term2 + term3
                    K_e [i, j] = gaussian_quadrature_triangle( func_k_all, vertices[:3], n_points, i, j )

        return K_e



def assemble_mass_matrix_element(vertices, mesh_order = 1): 
    n_points=4
    if mesh_order == 1:
        r1, r2, r3 = vertices
        r1_r = r1[1]
        r2_r = r2[1]
        r3_r = r3[1]
        rc = (r1_r + r2_r + r3_r)/3
        A2 = calculate_triangle_area_double(vertices)
        A = A2 / 2
        grad_L = grad_area_coordinates(vertices)

        def func_r_LiLj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            #rv = np.array([z, r])
            L = [L1, L2, L3]
            return L[i] * L[j] *r
        
        M_e = np.zeros((3, 3))
        for i in range(3) :
            for j in range(3) :
                M_e[i, j] = gaussian_quadrature_triangle(func_r_LiLj, vertices, n_points, i, j )

        return M_e
    
    elif mesh_order == 2 :
        #vertices = vertices[:3]
        #r1, r2, r3 = vertices
        #r1_r = r1[1]
        #r2_r = r2[1]
        #r3_r = r3[1]
        #rc = (r1_r + r2_r + r3_r)/3
        #A2 = calculate_triangle_area_double(vertices)
        #A = A2 / 2
        #grad_L = grad_area_coordinates(vertices)

        def func_r_GiGj(L1, L2, L3, vertices, i, j):
            rv1, rv2, rv3 = vertices
            z1, z2, z3 = rv1[0], rv2[0], rv3[0]
            r1, r2, r3 = rv1[1], rv2[1], rv3[1]
            r = L1*r1+L2*r2+L3*r3
            z = L1*z1+L2*z2+L3*z3
            L = [L1, L2, L3]
            G = [0] *6
            for k in range(0,6) :
                if k <3 : G[k] = (2*L[k]-1)*L[k]
                else :    G[k] = 4*L[k-3]*L[(k-2)%3]
            return r* G[i] * G[j]
        
        M_e = np.zeros((6, 6))

        curve_index = check_curve_element( vertices )
        if curve_index != 0 :
            #print(f'curve@{vertices[curve_index]} {np.sqrt(vertices[curve_index][0]**2 +vertices[curve_index][1]**2)} ' )

            def func_r_GiGj_Jacobian(L1, L2, L3, vertices, i, j, curve_index):
                rv1, rv2, rv3 = vertices[:3]
                z1, z2, z3 = rv1[0], rv2[0], rv3[0]
                r1, r2, r3 = rv1[1], rv2[1], rv3[1]
                r = L1*r1+L2*r2+L3*r3
                z = L1*z1+L2*z2+L3*z3
                L = [L1, L2, L3]
                grad_L = grad_area_coordinates(vertices[:3])
                G = [0] *6
                for k in range(0,6) :
                    if k <3 : G[k] = (2*L[k]-1)*L[k]
                    else :    G[k] = 4*L[k-3]*L[(k-2)%3]
                gradG = [0] *6
                for k in range(0,6) :
                    if k <3 : gradG[k] = ( 4*L[k] -1 ) *grad_L[k]
                    else :    gradG[k] = 4*( L[(k-2)%3]*grad_L[k-3] + L[k-3]*grad_L[(k-2)%3] )

                rv_curve_orign = vertices[curve_index]
                rv_curve_convert = 0.5*(vertices[(curve_index-2)%3]+ vertices[curve_index-3])
                J = 1+np.dot(gradG[curve_index], ( rv_curve_orign - rv_curve_convert ) )

                return r* G[i] * G[j] *J
            for i in range(6) :
                for j in range(6) :
                    M_e[i, j] = gaussian_quadrature_triangle(func_r_GiGj_Jacobian, vertices, n_points, i, j, curve_index )

        else :
            for i in range(6) :
                for j in range(6) :
                    M_e[i, j] = gaussian_quadrature_triangle(func_r_GiGj, vertices[:3], n_points, i, j )

        return M_e

# -------------------------------------------------------------
# ２次要素のエッジ中点にあるnodeが曲線上に乗っているかどうか？
#-----------------------------------------------------------
def check_curve_element( vertices ) :
    if len(vertices) != 6 : return 0
    else :
        curve_index = 0 
        for i in range(3, 6) :
            r_edge = vertices[(i-2)%3] -vertices[(i-3)%3]
            r_curve_pos = vertices[i] -vertices[(i-3)%3]
            s = r_edge[0]*r_curve_pos[1] -r_edge[1]*r_curve_pos[0] #外積の成分
            l2 = r_edge[0]*r_edge[0] +r_edge[1]*r_edge[1] # r_edgeの長さの２乗
            if np.abs(s/l2) > 1e-6 : #曲線要素
                #print('curve ', r_edge, 2*r_curve_pos)
                curve_index = i

        return curve_index




# --- 境界条件関連関数 (追加) ---
def find_nodes_on_r0_boundary(nodes, r_threshold=1e-6):
    """r=0 境界上のノードを検出する関数"""
    r0_boundary_nodes_index = []
    for i, point in enumerate(nodes):
        if abs(point[1]) < r_threshold: # r座標 (y座標) が threshold より小さい場合
            r0_boundary_nodes_index.append(i)
    return r0_boundary_nodes_index


def apply_dirichlet_boundary_condition(K_matrix, M_matrix, boundary_nodes_index):
    """ディリクレ境界条件を適用する関数 (u=0)"""
    modified_K_matrix = K_matrix.copy()
    for node_index in boundary_nodes_index:
        modified_K_matrix[node_index, :] = 0 # 行をゼロクリア
        modified_K_matrix[:, node_index] = 0 # 列をゼロクリア (対称性を保つため)
        modified_K_matrix[node_index, node_index] = 0 # 対角成分を1に設定

    modified_M_matrix = M_matrix.copy()
    for node_index in boundary_nodes_index:
        modified_M_matrix[node_index, :] = 0 # 行をゼロクリア
        modified_M_matrix[:, node_index] = 0 # 列をゼロクリア (対称性を保つため)
        modified_M_matrix[node_index, node_index] = 1 # 対角成分を1に設定

    return modified_K_matrix, modified_M_matrix


# --- グローバルマトリクス組み立て関数 ---
def assemble_global_matrix(nodes, elements, assemble_element_matrix_func, mesh_order=1):
    N_nodes = len(nodes)
    global_matrix = np.zeros((N_nodes, N_nodes))

    for element_nodes_index in elements:
        # 要素頂点座標を取得
        vertices_e = nodes[element_nodes_index]
        # 要素マトリクスを計算
        element_matrix = assemble_element_matrix_func(vertices_e, mesh_order)
        # グローバルマトリクスに要素マトリクスを足し合わせる
        if mesh_order == 1 :
            for i in range(3):
                for j in range(3):
                    global_matrix[element_nodes_index[i], element_nodes_index[j]] += element_matrix[i, j]
        elif mesh_order == 2 :
            for i in range(6):
                for j in range(6):
                    global_matrix[element_nodes_index[i], element_nodes_index[j]] += element_matrix[i, j]

    return global_matrix


# --- グローバルマトリクス組み立て関数 ---
def assemble_global_matrix_sparse(nodes, elements, assemble_element_matrix_func, mesh_order=1):
    matrix_size = len(nodes)
    # COOフォーマット用リストを初期化
    rows = []
    cols = []
    data = []
    for element_index, simplex in enumerate(elements):
        vertices_element = nodes[simplex]
        element_matrix = assemble_element_matrix_func(vertices_element, mesh_order)
        if mesh_order == 1 : nodeNum = 3
        elif mesh_order == 2 : nodeNum = 6
        for i in range(nodeNum):
            for j in range(nodeNum):
                global_row_index = simplex[i]
                global_col_index = simplex[j]
                val = element_matrix[i, j]
                if val != 0: # ゼロ要素は追加しない
                    rows.append(global_row_index)
                    cols.append(global_col_index)
                    data.append(val)

    # COOリストからCOO形式の疎行列を作成
    # coo_matrix は同じ (row, col) に複数の値があると、それらを自動的に合計してくれる
    global_coo = coo_matrix((data, (rows, cols)), shape=(matrix_size, matrix_size), dtype=np.float64) # dtypeを指定

    # COOからCSR形式に変換（計算にはCSRが効率的なことが多い）
    global_sparse = global_coo.tocsr()
    return global_sparse


def assemble_global_matrix_vectorized_1st(nodes, elements):
    """
    1次三角形要素の全体行列組み立てをNumPyでベクトル化して行う関数。
    """
    N_nodes = len(nodes)
    N_elem = len(elements)
    
    # 頂点座標の取得 (E, 3, 2)
    P = nodes[elements]
    z = P[:, :, 0]
    r = P[:, :, 1]
    
    # 面積の計算 (E,)
    # 2A = |z0(r1-r2) + z1(r2-r0) + z2(r0-r1)|
    twoA = (z[:, 0]*(r[:, 1] - r[:, 2]) + 
            z[:, 1]*(r[:, 2] - r[:, 0]) + 
            z[:, 2]*(r[:, 0] - r[:, 1]))
    A = np.abs(twoA) / 2.0
    
    # 形状関数の勾配 (E, 3, 2)
    # gradLi = 1/(2A) * [zi_k - zi_j, ri_j - ri_k] (i, j, k は巡回)
    # b_i = r_j - r_k, c_i = z_k - z_j
    b = np.zeros((N_elem, 3))
    c = np.zeros((N_elem, 3))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        b[:, i] = r[:, j] - r[:, k]
        c[:, i] = z[:, k] - z[:, j]
    
    # 勾配ベクトル (E, 3, 2) -> [dz, dr]
    # grad_L[e, i, 0] = c[e, i] / (2A[e] * sign)
    # grad_L[e, i, 1] = b[e, i] / (2A[e] * sign)
    sign = np.sign(twoA)[:, np.newaxis]
    grad_L = np.stack([c, b], axis=2) / (2.0 * A[:, np.newaxis, np.newaxis] * sign[:, :, np.newaxis])

    # 重心の r 座標
    rc = np.mean(r, axis=1) # (E,)
    
    # 1次要素用の積分点 (n_points=4)
    points_data = integration_points_triangle[4]
    coords_L = points_data['coords']   # (4, 3)
    weights = points_data['weights']     # (4,)
    
    # 各積分点における r 座標の計算 (E, 4)
    # r_p = sum(L_k * r_k)
    r_p = r @ coords_L.T 
    
    # --- Stiffness Matrix K_e (E, 3, 3) ---
    K_e = np.zeros((N_elem, 3, 3))
    
    # grad_L[:, i, 0] は r方向の勾配、grad_L[:, i, 1] は z方向の勾配に対応
    # term1: A * rc * (gradLi . gradLj)
    # term3: A/3 * (gradLi_r + gradLj_r)
    for i in range(3):
        for j in range(3):
            # 勾配の内積
            grad_dot = grad_L[:, i, 0] * grad_L[:, j, 0] + grad_L[:, i, 1] * grad_L[:, j, 1]
            K_e[:, i, j] += 2* A * rc * grad_dot #係数２を省略しない（後のエネルギー計算のため）
            
            # term3 (radial terms): (A/3) * (gradLi_r + gradLj_r)
            K_e[:, i, j] += 2* (A / 3.0) * (grad_L[:, i, 0] + grad_L[:, j, 0])
            
            # term2: integral(Li * Lj / r) dA
            term2 = np.zeros(N_elem)
            for p in range(len(weights)):
                L_i_p = coords_L[p, i]
                L_j_p = coords_L[p, j]
                term2 += weights[p] * (L_i_p * L_j_p / r_p[:, p])
            K_e[:, i, j] += 2* A * term2 #係数２を省略しない（後のエネルギー計算のため）

    # --- Mass Matrix M_e (E, 3, 3) ---
    M_e = np.zeros((N_elem, 3, 3))
    for i in range(3):
        for j in range(3):
            # integral(r * Li * Lj) dA
            term_m = np.zeros(N_elem)
            for p in range(len(weights)):
                L_i_p = coords_L[p, i]
                L_j_p = coords_L[p, j]
                term_m += weights[p] * (r_p[:, p] * L_i_p * L_j_p)
            M_e[:, i, j] += 2* A * term_m #係数２を省略しない（後のエネルギー計算のため）

    # --- 疎行列の組み立て ---
    # 行列要素のインデックスを生成
    # row = elements[:, [0,0,0,1,1,1,2,2,2]].flatten()
    # col = elements[:, [0,1,2,0,1,2,0,1,2]].flatten()
    # より一般的に:
    rows = np.repeat(elements, 3, axis=1).flatten()
    cols = np.tile(elements, (1, 3)).flatten()
    
    K_global = coo_matrix((K_e.flatten(), (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
    M_global = coo_matrix((M_e.flatten(), (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
    
    return K_global, M_global


def assemble_global_matrix_vectorized_2nd(nodes, elements):
    """
    2次三角形要素（曲線要素対応）の全体行列組み立てをNumPyでベクトル化して行う関数。
    アイソパラメトリック写像を用いて全要素を一括処理する。
    """
    N_nodes = len(nodes)
    N_elem = len(elements)
    
    # 節点座標の取得 (E, 6, 2)
    P = nodes[elements]
    # P[:, i, 0] は z, P[:, i, 1] は r
    
    # 積分点の設定 (n_points=7 推奨だが、元コードに合わせて4点でも可。ここでは4点を使用)
    n_points = 4
    points_data = integration_points_triangle[n_points]
    L_points = points_data['coords']   # (P, 3) -> (L1, L2, L3)
    weights = points_data['weights']     # (P,)
    
    # 静的な形状関数およびその勾配の計算 (積分点において共通)
    num_pts = len(weights)
    G = np.zeros((num_pts, 6))
    # 基準座標系(L1, L2)における勾配 dG/dL1, dG/dL2
    gradG_ref = np.zeros((num_pts, 6, 2))
    
    for p in range(num_pts):
        L1, L2, L3 = L_points[p]
        # G1-G3: 頂点, G4-G6: 辺中点 (1-2, 2-3, 3-1)
        G[p, 0] = (2*L1 - 1) * L1
        G[p, 1] = (2*L2 - 1) * L2
        G[p, 2] = (2*L3 - 1) * L3
        G[p, 3] = 4 * L1 * L2
        G[p, 4] = 4 * L2 * L3
        G[p, 5] = 4 * L3 * L1
        
        # dG/dL1 (L3 = 1 - L1 - L2 を考慮)
        # dG/dL1_total = dG/dL1 - dG/dL3
        dG_dL = np.zeros((6, 3))
        dG_dL[0, 0] = 4*L1 - 1
        dG_dL[1, 1] = 4*L2 - 1
        dG_dL[2, 2] = 4*L3 - 1
        dG_dL[3, 0], dG_dL[3, 1] = 4*L2, 4*L1
        dG_dL[4, 1], dG_dL[4, 2] = 4*L3, 4*L2
        dG_dL[5, 0], dG_dL[5, 2] = 4*L3, 4*L1
        
        gradG_ref[p, :, 0] = dG_dL[:, 0] - dG_dL[:, 2] # dG/dL1
        gradG_ref[p, :, 1] = dG_dL[:, 1] - dG_dL[:, 2] # dG/dL2

    # --- 要素行列の計算 ---
    K_e = np.zeros((N_elem, 6, 6))
    M_e = np.zeros((N_elem, 6, 6))
    
    for p in range(num_pts):
        # 1. ヤコビ行列の計算 J = [dz/dL1, dz/dL2; dr/dL1, dr/dL2]
        # gradG_ref[p] (6, 2)
        # P (E, 6, 2)
        # J = P^T @ gradG_ref (E, 2, 2)
        # J[:, 0, 0] = dz/dL1, J[:, 1, 0] = dz/dL2, J[:, 0, 1] = dr/dL1, J[:, 1, 1] = dr/dL2
        # (E, 2, 6) @ (6, 2) -> (E, 2, 2)
        Jac = np.einsum('eij,jk->eik', P.transpose(0, 2, 1), gradG_ref[p]) 
        
        # 行列式 detJ
        detJ = Jac[:, 0, 0] * Jac[:, 1, 1] - Jac[:, 0, 1] * Jac[:, 1, 0]
        abs_detJ = np.abs(detJ)
        
        # 逆行列 J^-1
        # invJ = [[dr/dL2, -dz/dL2], [-dr/dL1, dz/dL1]] / detJ
        invJ = np.zeros_like(Jac)
        invJ[:, 0, 0] = Jac[:, 1, 1] / detJ
        invJ[:, 0, 1] = -Jac[:, 0, 1] / detJ
        invJ[:, 1, 0] = -Jac[:, 1, 0] / detJ
        invJ[:, 1, 1] = Jac[:, 0, 0] / detJ
        
        # 2. グローバル勾配 gradG_glob = gradG_ref @ J^-1
        # (6, 2) @ (E, 2, 2) -> (E, 6, 2)
        gradG_glob = np.einsum('jk,ekl->ejl', gradG_ref[p], invJ)
        
        # 3. 物理座標 (r)
        r_p = P[:, :, 1] @ G[p] # (E,)
        
        # 4. 積分項の計算
        # Stiffness K: (r * gradGi.gradGj + Gi*Gj/r + Gi*gradGj_r + Gj*gradGi_r) * detJ
        # Mass M: r * Gi * Gj * detJ
        
        # Outer products for nodes (i, j)
        GiGj = np.outer(G[p], G[p]) # (6, 6)
        
        # gradGi . gradGj (E, 6, 6)
        grad_dot = np.einsum('eik,ejk->eij', gradG_glob, gradG_glob)
        
        # gradG_r terms (index 1 of gradG_glob is r-derivative)
        # Gi * gradGj_r + Gj * gradGi_r
        # Gi is (6,), gradG_glob[:, :, 1] is (E, 6)
        grad_r = gradG_glob[:, :, 1]
        term_ij = G[p][np.newaxis, :, np.newaxis] * grad_r[:, np.newaxis, :] # (E, 6, 6)
        term_ji = G[p][np.newaxis, np.newaxis, :] * grad_r[:, :, np.newaxis] # (E, 6, 6)
        grad_r_sum = term_ij + term_ji
        
        # 体積要素 dV = abs_detJ * weights[p]
        dV = abs_detJ * weights[p] # (E,)
        
        # 行列への加算
        K_e += (r_p[:, np.newaxis, np.newaxis] * grad_dot + 
                GiGj[np.newaxis, :, :] / r_p[:, np.newaxis, np.newaxis] + 
                grad_r_sum) * dV[:, np.newaxis, np.newaxis]
                
        M_e += (r_p[:, np.newaxis, np.newaxis] * GiGj[np.newaxis, :, :]) * dV[:, np.newaxis, np.newaxis]

    # --- 疎行列の組み立て ---
    rows = np.repeat(elements, 6, axis=1).flatten()
    cols = np.tile(elements, (1, 6)).flatten()
    
    K_global = coo_matrix((K_e.flatten(), (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
    M_global = coo_matrix((M_e.flatten(), (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
    
    return K_global, M_global
def create_transformation_matrix(N, dirichlet_nodes, periodic_pairs=None, phase_shift_deg=0.0):
    """
    独立自由度への変換行列 T を生成する。周期境界条件とディリクレ境界条件を統合。
    
    x = T @ x_i
    
    Args:
        N (int): 元の自由度数
        dirichlet_nodes (list): Dirichlet条件 (u=0) を適用するノードインデックス
        periodic_pairs (list of tuples): (idx_min, idx_max) のペア。 idx_max は従属変数とする。
        phase_shift_deg (float): 位相差 (度)。 x_max = exp(i * phase) * x_min
        
    Returns:
        tuple: (T, internal_indices)
               T (csr_matrix, complex): 変換行列 (N x M)
               internal_indices (np.ndarray): 独立自由度の元のノードインデックス
    """
    if periodic_pairs is None: periodic_pairs = []
    
    dirichlet_set = set(dirichlet_nodes)
    periodic_max_to_min = {p_max: p_min for p_min, p_max in periodic_pairs}
    periodic_max_set = set(periodic_max_to_min.keys())
    
    # 独立自由度 (Internal) の抽出:
    # Dirichlet でも Periodic_Max (従属側) でもないノード
    internal_indices = []
    node_to_internal_idx = {}
    
    curr_idx = 0
    for i in range(N):
        if i not in dirichlet_set and i not in periodic_max_set:
            internal_indices.append(i)
            node_to_internal_idx[i] = curr_idx
            curr_idx += 1
            
    internal_indices = np.array(internal_indices)
    M = len(internal_indices)
    
    # 変換行列 T (N x M) の構築
    # T の要素は複素数になる可能性がある
    # 物理規約 (e^{+jωt}) に合わせた PBC: x_max = e^{-jθ} · x_min
    # +z 方向進行波の空間依存は e^{-jkz} → z_max での位相は e^{-jkL} = e^{-jθ}
    phase_rad = np.deg2rad(phase_shift_deg)
    phase_factor = np.exp(-1j * phase_rad)
    
    rows = []
    cols = []
    data = []
    
    for i in range(N):
        if i in dirichlet_set:
            # Dirichlet: 0なので T の行はすべて0 (何もしない)
            continue
        elif i in periodic_max_set:
            # 従属ノード: x_i = exp(i*phi) * x_min
            idx_min = periodic_max_to_min[i]
            if idx_min in node_to_internal_idx:
                rows.append(i)
                cols.append(node_to_internal_idx[idx_min])
                data.append(phase_factor)
        else:
            # 独立ノード (または周期境界の独立側): x_i = 1 * x_i'
            if i in node_to_internal_idx:
                rows.append(i)
                cols.append(node_to_internal_idx[i])
                data.append(1.0 + 0.0j)
                
    T = coo_matrix((data, (rows, cols)), shape=(N, M), dtype=np.complex128).tocsr()
    
    print(f"Matrix reduction: N={N} -> M={M} (Dirichlet: {len(dirichlet_nodes)}, Periodic Pairs: {len(periodic_pairs)})")
    return T, internal_indices


def apply_bc_transformation(K_global, M_global, T):
    """
    変換行列 T を用いて縮小された行列を計算する (Transformation Method)。

    K_reduced = T.T @ K_global @ T
    M_reduced = T.T @ M_global @ T

    Args:
        K_global (csr_matrix): 全体剛性行列
        M_global (csr_matrix): 全体質量行列
        T (csr_matrix): 変換行列 (N x M)

    Returns:
        tuple: (K_reduced, M_reduced)
               K_reduced (csr_matrix): 縮小された剛性行列 (M x M)
               M_reduced (csr_matrix): 縮小された質量行列 (M x M)
    """
    # Ensure matrices are CSR for efficient multiplication
    if not isinstance(K_global, csr_matrix): K_global = K_global.tocsr()
    if not isinstance(M_global, csr_matrix): M_global = M_global.tocsr()
    if not isinstance(T, csr_matrix): T = T.tocsr()

    # Perform the transformation: T.T @ K @ T and T.T @ M @ T
    # T.T is M x N, K_global is N x N, T is N x M -> result is M x M
    K_reduced = T.transpose() @ K_global @ T
    M_reduced = T.transpose() @ M_global @ T

    print(f"Reduced matrix K shape: {K_reduced.shape}")
    print(f"Reduced matrix M shape: {M_reduced.shape}")
    return K_reduced, M_reduced

def reconstruct_eigenvector_transformation(eigenvector_reduced, T):
    """
    変換行列 T を用いて、縮小された固有ベクトルを元のサイズに復元する。
    x = T @ x_i

    Args:
        eigenvector_reduced (np.ndarray): 縮小された固有ベクトル (サイズ M)
        T (csr_matrix): 変換行列 (N x M)

    Returns:
        np.ndarray: 元のサイズの固有ベクトル (サイズ N)
    """
    # T is N x M, eigenvector_reduced is M x 1 -> result is N x 1
    if T.shape[1] != len(eigenvector_reduced):
        raise ValueError(f"Shape mismatch: T columns {T.shape[1]} != eigenvector length {len(eigenvector_reduced)}")

    eigenvector_full = T @ eigenvector_reduced
    return eigenvector_full

# --- ここまで変換法関連の関数 ---


def identify_periodic_boundaries(nodes, tol=1e-8):
    """
    z軸に垂直な直線端を自動判別し、同一r座標を持つ(z_min, z_max)ペアを抽出する。
    
    Args:
        nodes (np.ndarray): 節点座標 (N, 2)
        tol (float): 座標比較用の許容誤差
        
    Returns:
        list of tuples: (node_idx_zmin, node_idx_zmax) のリスト
    """
    z_coords = nodes[:, 0]
    r_coords = nodes[:, 1]
    
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    if (z_max - z_min) < tol:
        return []
        
    z_min_indices = np.where(np.abs(z_coords - z_min) < tol)[0]
    z_max_indices = np.where(np.abs(z_coords - z_max) < tol)[0]
    
    periodic_pairs = []
    
    if len(z_min_indices) > 0 and len(z_max_indices) > 0:
        r_max_coords = r_coords[z_max_indices].reshape(-1, 1)
        tree = KDTree(r_max_coords)
        
        for idx_min in z_min_indices:
            r_val = r_coords[idx_min]
            # r=0 (中心軸上) のノードはDirichlet条件を優先するため、周期境界のペアから除外する
            if r_val < tol:
                continue
                
            dist, idx_in_max = tree.query([[r_val]])
            if dist[0] < tol:
                idx_max = z_max_indices[idx_in_max[0]]
                periodic_pairs.append((idx_min, idx_max))
                
    periodic_pairs.sort(key=lambda x: r_coords[x[0]])
    return periodic_pairs


def run_fem_analysis_standingTM0(mesh_file, mesh_order=2, num_modes=10):
    """
    定在波 (Standing Wave) 解析を実行する関数 (実数ベース)
    """
    nodes, elements, physical_groups = load_gmsh_mesh(mesh_file, mesh_order)

    print(f"節点数: {len(nodes)}")
    print(f"要素数: {len(elements)}")

    # --- 全体行列の組み立て ---
    print('Assembling matrices...')
    start_time = time.time()
    if mesh_order == 1:
        K_global, M_global = assemble_global_matrix_vectorized_1st(nodes, elements)
    elif mesh_order == 2:
        K_global, M_global = assemble_global_matrix_vectorized_2nd(nodes, elements)
    else:
        K_global = assemble_global_matrix_sparse(nodes, elements, assemble_stiffness_matrix_element, mesh_order)
        M_global = assemble_global_matrix_sparse(nodes, elements, assemble_mass_matrix_element, mesh_order)
    matrix_assembly_time = time.time() - start_time
    print(f'Finish assemble matrix. Time: {matrix_assembly_time:.4f} [s]')

    # --- 境界条件 (Dirichlet: r=0 および指定されたノード) ---
    dirichlet_nodes = []
    if "Dirichlet" in physical_groups:
        dirichlet_nodes = list(physical_groups["Dirichlet"])
    Mshort_nodes = []
    if "M-short" in physical_groups: #M-shortもDirichlet条件
        Mshort_nodes = list(physical_groups["M-short"])
    r0_boundary_nodes_index = find_nodes_on_r0_boundary(nodes)
    dirichlet_nodes = sorted(list(set(dirichlet_nodes) | set(Mshort_nodes) | set(r0_boundary_nodes_index)))
    #dirichlet_nodes = sorted(list(set(dirichlet_nodes) | set(r0_boundary_nodes_index)))

    # 変換行列 T (実数)
    T, _ = create_transformation_matrix(len(nodes), dirichlet_nodes)
    T = T.real # 強制的に実数として扱う

    # 行列の縮小
    K_reduced = T.T @ K_global @ T
    M_reduced = T.T @ M_global @ T

    # --- 固有値解析 (実対称行列用 eigsh) ---
    r_max = np.max(nodes[:, 1])
    sigma = (2 * np.pi * (299792458 / (4 * r_max)) / 299792458)**2
    
    solve_start = time.time()
    eigenvalues, eigenvectors_reduced = eigsh(K_reduced, k=num_modes, M=M_reduced,
                sigma=sigma, which='LA', tol=1e-9)
    solve_time = time.time() - solve_start
    print(f'Finish solving eigenvalue problem. Time: {solve_time:.4f} [s]')
    
    frequency_values = (299792458.0 / (2 * np.pi)) * np.sqrt(np.abs(eigenvalues)) / 1e9 # GHz

    # 固有ベクトルの復元
    eigenvectors_full = []
    for i in range(len(eigenvalues)):
        ev_full = T @ eigenvectors_reduced[:, i]
        eigenvectors_full.append(np.array(ev_full).flatten())
    
    return {
        "nodes": nodes,
        "elements": elements,
        "frequencies": frequency_values,
        "eigenvectors": np.array(eigenvectors_full),
        "mesh_order": mesh_order,
        "r0_nodes": r0_boundary_nodes_index,
        "M_global": M_global,
        "physical_groups": physical_groups,
        "matrix_assembly_time": matrix_assembly_time,
        "eigenvalue_solve_time": solve_time,
        "analysis_type": "standing"
    }


def run_fem_analysis_travelingTM0(mesh_file, mesh_order=2, num_modes=10, phase_shifts=[120.0]):
    """
    進行波 (Traveling Wave) 解析を実行する関数 (複素数ベース, 周期境界条件)
    複数の位相シフト(phase_shifts)をリストで受け取り、全体行列の組み立てを1回で済ませて連続計算する。
    """
    nodes, elements, physical_groups = load_gmsh_mesh(mesh_file, mesh_order)

    print(f"節点数: {len(nodes)}")
    print(f"要素数: {len(elements)}")

    # --- 全体行列の組み立て ---
    print('Assembling matrices...')
    start_time = time.time()
    if mesh_order == 1:
        K_global, M_global = assemble_global_matrix_vectorized_1st(nodes, elements)
    elif mesh_order == 2:
        K_global, M_global = assemble_global_matrix_vectorized_2nd(nodes, elements)
    else:
        K_global = assemble_global_matrix_sparse(nodes, elements, assemble_stiffness_matrix_element, mesh_order)
        M_global = assemble_global_matrix_sparse(nodes, elements, assemble_mass_matrix_element, mesh_order)
    matrix_assembly_time = time.time() - start_time

    # --- 境界条件 (Periodic + Dirichlet) ---
    dirichlet_nodes = []
    if "Dirichlet" in physical_groups:
        dirichlet_nodes = list(physical_groups["Dirichlet"])
    Mshort_nodes = []
    if "M-short" in physical_groups: #M-shortもDirichlet条件
        Mshort_nodes = list(physical_groups["M-short"])
    r0_boundary_nodes_index = find_nodes_on_r0_boundary(nodes)
    dirichlet_nodes = sorted(list(set(dirichlet_nodes) | set(Mshort_nodes) | set(r0_boundary_nodes_index )))

    periodic_pairs = identify_periodic_boundaries(nodes)
    if len(periodic_pairs) > 0:
        r0_set = set(r0_boundary_nodes_index)
        dirichlet_set = set(dirichlet_nodes)
        filtered_pairs = []
        for p_min, p_max in periodic_pairs:
            if p_min in r0_set or p_max in r0_set: continue
            if p_min in dirichlet_set: dirichlet_nodes.remove(p_min)
            if p_max in dirichlet_set: dirichlet_nodes.remove(p_max)
            filtered_pairs.append((p_min, p_max))
        periodic_pairs = filtered_pairs

    r_max = np.max(nodes[:, 1])
    sigma = (2 * np.pi * (299792458 / (4 * r_max)) / 299792458)**2
    from scipy.sparse.linalg import eigs

    # 各位相ごとの結果を格納する辞書
    phase_results = {}

    for phase in phase_shifts:
        print(f"\\n--- Solving for Phase Shift: {phase} deg ---")
        # 変換行列 T (複素数)
        T, _ = create_transformation_matrix(len(nodes), dirichlet_nodes, periodic_pairs, phase)

        # 行列の縮小 (Hermitian transpose)
        K_reduced = T.conjugate().transpose() @ K_global @ T
        M_reduced = T.conjugate().transpose() @ M_global @ T

        # --- 固有値解析 (複素行列用 eigs) ---
        solve_start = time.time()
        eigenvalues, eigenvectors_reduced = eigs(K_reduced, k=num_modes, M=M_reduced,
                    sigma=sigma, which='LM', tol=1e-9)
        solve_time = time.time() - solve_start
        print(f'Finish solving eigenvalue problem. Time: {solve_time:.4f} [s]')
        
        # ソート
        idx = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors_reduced = eigenvectors_reduced[:, idx]
        
        frequency_values = (299792458.0 / (2 * np.pi)) * np.sqrt(np.abs(eigenvalues)) / 1e9 # GHz

        # 固有ベクトルの復元
        eigenvectors_full = []
        for i in range(len(eigenvalues)):
            ev_full = T @ eigenvectors_reduced[:, i]
            eigenvectors_full.append(np.array(ev_full).flatten())
            
        phase_results[phase] = {
            "frequencies": frequency_values,
            "eigenvectors": np.array(eigenvectors_full),
            "eigenvalue_solve_time": solve_time
        }

    return {
        "nodes": nodes,
        "elements": elements,
        "mesh_order": mesh_order,
        "r0_nodes": r0_boundary_nodes_index,
        "M_global": M_global,
        "physical_groups": physical_groups,
        "matrix_assembly_time": matrix_assembly_time,
        "phase_results": phase_results,
        "analysis_type": "traveling_multi"
    }

# --- メイン処理 ---
if __name__ == "__main__":
    # テスト用実行
    mesh_file = "./mesh_data/sphere_2nd_order_005.msh"
    results = run_fem_analysis(mesh_file, mesh_order=2, num_modes=5)
    
    print("Frequencies [GHz]:")
    for i, freq in enumerate(results["frequencies"]):
        print(f" Mode {i}: {freq:.10f} GHz")
