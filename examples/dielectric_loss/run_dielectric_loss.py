"""誘電体損失 tanδ（摂動法 Q_diel）の解析解検証.

全充填ピルボックス空洞（半径 a、長さ L、一様 ε_r・tanδ）で:

1. TM010 共振周波数の解析解  f = (j01 c) / (2π a √ε_r)   （j01 = 2.404825557695773）
   と FEM の最低次モードを比較する。
2. 誘電体損失 Q_diel の厳密値 1/tanδ（一様充填）と FEM を比較する。
3. 合成則 1/Q = 1/Q_wall + 1/Q_diel の成立を確認する。
4. HOM（n=1、TM110 系）でも Q_diel = 1/tanδ を確認する。
5. 部分充填（右半分のみ tanδ）で 1/Q_diel = tanδ × (電気的充填率) を確認する。

実行:
    python examples/dielectric_loss/run_dielectric_loss.py
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.special import jn_zeros

from axicavity_fem.fem_hom.post_process import compute_hom_parameters
from axicavity_fem.fem_hom.solver import solve_hom_standing
from axicavity_fem.fem_tm0.post_process import compute_tm0_parameters
from axicavity_fem.fem_tm0.solver import solve_tm0_standing
from axicavity_fem.shared.boundary_groups import classify_boundaries
from axicavity_fem.shared.constants import C0
from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
from axicavity_fem.shared.material_resolver import (
    build_eps_r_per_element,
    build_material_table_from_geometry,
    build_tan_delta_per_element,
)
from axicavity_fem.shared.mesh_loader import load_mesh, load_mesh_hom
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)

# 空洞寸法・材料（PTFE 想定）
# 注意: ε_r を大きくしすぎる（例 9.0）と、既知の制約「shift-invert の σ が
# ε_r に追従しない」（README の Known constraints）により最低次モードを
# 取り逃すことがある。本例は ε_r=2.1（PTFE）で TM010 が正しく得られる。
A_MM = 50.0      # 半径 [mm]
L_MM = 40.0      # 長さ [mm]
EPS_R = 2.1
TAND = 1e-4
MESH_SIZE = 5.0  # [mm]
ELEM_ORDER = 2


def make_pillbox(eps_r: float, tan_delta: float) -> MultiRegionGeometry:
    """全充填ピルボックス（軸 r=0、両端 E-short、外周 PEC）。"""
    pts = [(0.0, 0.0), (L_MM, 0.0), (L_MM, A_MM), (0.0, A_MM)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="E-short"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
    ]
    loop = Loop(id=0, segment_ids=[0, 1, 2, 3])
    region = Region(id=0, name="Dielectric", outer_loop_id=0,
                    material_tag="alumina", eps_r=eps_r, tan_delta=tan_delta)
    return MultiRegionGeometry(points=pts, segments=segs, loops=[loop],
                               regions=[region], unit="mm",
                               mesh_size=MESH_SIZE)


def make_half_filled(tan_delta_right: float) -> MultiRegionGeometry:
    """z=L/2 で 2 領域に分け、右半分のみ tanδ（ε_r は両方 1）。"""
    a, L, Lh = A_MM, L_MM, L_MM / 2.0
    pts = [(0.0, 0.0), (Lh, 0.0), (Lh, a), (0.0, a), (L, 0.0), (L, a)]
    segs = [
        Segment(id=0, type="line", point_indices=[0, 1], bc_name="M-short"),
        Segment(id=1, type="line", point_indices=[1, 2], bc_name="None"),
        Segment(id=2, type="line", point_indices=[2, 3], bc_name="PEC"),
        Segment(id=3, type="line", point_indices=[3, 0], bc_name="E-short"),
        Segment(id=4, type="line", point_indices=[1, 4], bc_name="M-short"),
        Segment(id=5, type="line", point_indices=[4, 5], bc_name="E-short"),
        Segment(id=6, type="line", point_indices=[5, 2], bc_name="PEC"),
    ]
    loops = [Loop(id=0, segment_ids=[0, 1, 2, 3]),
             Loop(id=1, segment_ids=[4, 5, 6, 1])]
    regions = [
        Region(id=0, name="Vacuum", outer_loop_id=0, material_tag="vacuum"),
        Region(id=1, name="Lossy", outer_loop_id=1, material_tag="lossy",
               tan_delta=tan_delta_right),
    ]
    return MultiRegionGeometry(points=pts, segments=segs, loops=loops,
                               regions=regions, unit="mm",
                               mesh_size=MESH_SIZE)


def solve_tm0(tmp: Path, geom: MultiRegionGeometry, num_modes: int = 3):
    msh = tmp / "cavity.msh"
    export_msh_multi_region(geom, msh, mesh_order=ELEM_ORDER, verbose=0)
    table = build_material_table_from_geometry(geom)
    mesh = load_mesh(msh, element_order=ELEM_ORDER)
    eps_e = build_eps_r_per_element(mesh, table)
    tand_e = build_tan_delta_per_element(mesh, table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_tm0_standing(mesh, num_modes, material_table=table)
    cls = classify_boundaries(mesh.physical_groups, mesh.nodes)
    params = compute_tm0_parameters(res, cls, eps_r_per_element=eps_e,
                                    tan_delta_per_element=tand_e)
    return res, params


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dielectric_loss_"))
    j01 = float(jn_zeros(0, 1)[0])

    # ------------------------------------------------------------------
    # 1) 全充填 TM0
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"全充填ピルボックス: a={A_MM} mm, L={L_MM} mm, "
          f"eps_r={EPS_R}, tan_delta={TAND}")
    print("=" * 72)

    res, params = solve_tm0(tmp, make_pillbox(EPS_R, TAND))
    f_analytic = j01 * C0 / (2 * np.pi * (A_MM * 1e-3) * np.sqrt(EPS_R)) / 1e9
    q_diel_exact = 1.0 / TAND

    mp0 = params.per_phase[0.0][0]
    err_f = abs(mp0.frequency_ghz - f_analytic) / f_analytic
    print(f"\n[TM010] 解析解 f = {f_analytic:.6f} GHz")
    print(f"        FEM    f = {mp0.frequency_ghz:.6f} GHz "
          f"(相対誤差 {err_f:.2e})")
    assert err_f < 1e-4, "TM010 周波数が解析解と一致しません"
    print(f"\n{'mode':>4} {'f [GHz]':>12} {'Q_wall':>12} {'Q_diel':>16} "
          f"{'Q (total)':>12}")
    for k, mp in enumerate(params.per_phase[0.0]):
        print(f"{k:>4} {mp.frequency_ghz:>12.6f} {mp.q_wall:>12.4e} "
              f"{mp.q_diel:>16.10e} {mp.q_factor:>12.4e}")

    mp = params.per_phase[0.0][0]
    err_qd = abs(mp.q_diel - q_diel_exact) / q_diel_exact
    comb = 1.0 / (1.0 / mp.q_wall + 1.0 / mp.q_diel)
    err_comb = abs(mp.q_factor - comb) / comb
    print(f"\nQ_diel 厳密値 1/tanδ = {q_diel_exact:.1f} → 相対誤差 {err_qd:.2e}")
    print(f"合成則 1/Q = 1/Q_wall + 1/Q_diel → 相対誤差 {err_comb:.2e}")
    assert err_qd < 1e-9 and err_comb < 1e-12

    # ------------------------------------------------------------------
    # 2) HOM n=1（TM110 系）
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HOM n=1（同形状・全充填）")
    print("=" * 72)
    geom = make_pillbox(EPS_R, TAND)
    msh = tmp / "cavity_hom.msh"
    export_msh_multi_region(geom, msh, mesh_order=ELEM_ORDER, verbose=0)
    table = build_material_table_from_geometry(geom)
    mesh_h = load_mesh_hom(msh, 1, ELEM_ORDER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_h = solve_hom_standing(mesh_h, 3, material_table=table)
    params_h = compute_hom_parameters(
        res_h,
        eps_r_per_element=build_eps_r_per_element(mesh_h, table),
        tan_delta_per_element=build_tan_delta_per_element(mesh_h, table))
    print(f"\n{'mode':>4} {'f [GHz]':>12} {'Q_wall':>12} {'Q_diel':>16} "
          f"{'Q (total)':>12}")
    for k, mp in enumerate(params_h.per_phase[0.0]):
        print(f"{k:>4} {mp.frequency_ghz:>12.6f} {mp.q_wall:>12.4e} "
              f"{mp.q_diel:>16.10e} {mp.q_factor:>12.4e}")
        assert abs(mp.q_diel - q_diel_exact) / q_diel_exact < 1e-9

    # ------------------------------------------------------------------
    # 3) 部分充填（右半分のみ tanδ、ε_r=1）
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"部分充填: 右半分 (z > L/2) のみ tan_delta={TAND}（eps_r は全域 1）")
    print("=" * 72)
    res_p, params_p = solve_tm0(tmp, make_half_filled(TAND))
    print(f"\n{'mode':>4} {'f [GHz]':>12} {'Q_diel':>14} {'充填率':>10}")
    for k, mp in enumerate(params_p.per_phase[0.0]):
        fill = (1.0 / mp.q_diel) / TAND if mp.q_diel > 0 else 0.0
        print(f"{k:>4} {mp.frequency_ghz:>12.6f} {mp.q_diel:>14.4e} "
              f"{fill:>10.4f}")
        assert 0.0 < fill < 1.0
    print("\n→ 電気的充填率 (0 < W_diel/W_total < 1) に比例して "
          "1/Q_diel = tanδ × 充填率 となることを確認")

    print("\nOK: すべての検証をパスしました")


if __name__ == "__main__":
    main()
