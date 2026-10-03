"""自動生成: MultiRegionGeometry を gmsh API で再構築するスクリプト.

axicavity-fem-v23 GUI の Multi-Region Editor から書き出されたもの。
単位: mm  →  m 換算係数 scale = 0.001
メッシュサイズ lc = mesh_size * scale = 5.0 * 0.001 = 0.005
領域数 = 1  segment 数 = 8  loop 数 = 1

build_model(**user_vars) を外部から呼ぶと、変数表の値を上書きして
メッシュを再生成できます (周波数自動調整などに利用)。
例: build_model(a=120, gap=3, out_msh='tuned.msh')
"""

import gmsh
from math import *  # 式で sin/cos/sqrt/pi 等を使えるように (ver2.2)

_SCALE = 0.001
_MESH_ORDER = 2
_OUT_MSH = 's-band_1cell.msh'


def build_model(out_msh=_OUT_MSH, mesh_order=_MESH_ORDER, **user_vars):
    """geom を構築し .msh を書き出す。user_vars で変数表の値を上書き可能。"""
    # === user variables (変数表) — user_vars で上書き可、依存変数は再計算 ===
    f = user_vars.get('f', 2856e6)
    c = user_vars.get('c', 299792458)
    L = user_vars.get('L', (c/f/3)*1e3)
    a = user_vars.get('a', 22)
    b = user_vars.get('b', 45.37)
    t = user_vars.get('t', 5)

    _LC = 0.005

    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 2.0)
    gmsh.model.add('s-band_1cell')

    # === 1. Points (z, r) in source-unit, converted to m by *_SCALE ===
    point_tags = []
    point_tags.append(gmsh.model.occ.addPoint(0.0 * _SCALE, 0.0 * _SCALE, 0.0, _LC))  # idx 0
    point_tags.append(gmsh.model.occ.addPoint((L) * _SCALE, 0.0 * _SCALE, 0.0, _LC))  # idx 1
    point_tags.append(gmsh.model.occ.addPoint((L) * _SCALE, (a) * _SCALE, 0.0, _LC))  # idx 2
    point_tags.append(gmsh.model.occ.addPoint((L-t/2) * _SCALE, (a+t/2) * _SCALE, 0.0, _LC))  # idx 3
    point_tags.append(gmsh.model.occ.addPoint((L-t/2) * _SCALE, (b) * _SCALE, 0.0, _LC))  # idx 4
    point_tags.append(gmsh.model.occ.addPoint((t/2) * _SCALE, (b) * _SCALE, 0.0, _LC))  # idx 5
    point_tags.append(gmsh.model.occ.addPoint((t/2) * _SCALE, (a+t/2) * _SCALE, 0.0, _LC))  # idx 6
    point_tags.append(gmsh.model.occ.addPoint(0.0 * _SCALE, (a) * _SCALE, 0.0, _LC))  # idx 7

    # === 2. Segments (line / circle arc) ===
    seg_curve_tag = {}
    seg_curve_tag[0] = gmsh.model.occ.addLine(point_tags[0], point_tags[1])  # bc='None'
    seg_curve_tag[1] = gmsh.model.occ.addLine(point_tags[1], point_tags[2])  # bc='PEC'
    _center_tag_2 = gmsh.model.occ.addPoint((L) * _SCALE, (a+t/2) * _SCALE, 0.0, _LC)
    seg_curve_tag[2] = gmsh.model.occ.addCircleArc(point_tags[2], _center_tag_2, point_tags[3])  # arc, bc='PEC'
    seg_curve_tag[3] = gmsh.model.occ.addLine(point_tags[3], point_tags[4])  # bc='PEC'
    seg_curve_tag[4] = gmsh.model.occ.addLine(point_tags[4], point_tags[5])  # bc='PEC'
    seg_curve_tag[5] = gmsh.model.occ.addLine(point_tags[5], point_tags[6])  # bc='PEC'
    _center_tag_6 = gmsh.model.occ.addPoint(0.0 * _SCALE, (a+t/2) * _SCALE, 0.0, _LC)
    seg_curve_tag[6] = gmsh.model.occ.addCircleArc(point_tags[6], _center_tag_6, point_tags[7])  # arc, bc='PEC'
    seg_curve_tag[7] = gmsh.model.occ.addLine(point_tags[7], point_tags[0])  # bc='PEC'

    # === 3. Curve loops ===
    loop_tag = {}
    loop_tag[0] = gmsh.model.occ.addCurveLoop([seg_curve_tag[0], seg_curve_tag[1], seg_curve_tag[2], seg_curve_tag[3], seg_curve_tag[4], seg_curve_tag[5], seg_curve_tag[6], seg_curve_tag[7]])  # orientation='CCW'

    # === 4. Plane surfaces (region = outer loop + hole loops) ===
    region_surface_tag = {}
    region_surface_tag[0] = gmsh.model.occ.addPlaneSurface([loop_tag[0]])  # name='Vacuum' material_tag='vacuum' eps_r=1.0

    gmsh.model.occ.synchronize()

    # === 5. Boolean fragment to share boundary nodes between regions ===
    surface_dimtags_in = [(2, t) for t in region_surface_tag.values()]
    curve_dimtags_in = [(1, t) for t in seg_curve_tag.values()]
    objects_in = surface_dimtags_in + curve_dimtags_in
    # 単一領域・穴なしのため fragment 不要 (恒等マップ)
    out_map = [[dt] for dt in objects_in]
    n_surf = len(surface_dimtags_in)
    surface_out_map = out_map[:n_surf]
    curve_out_map = out_map[n_surf:]

    # region.id → fragment 後 Surface tag リスト
    regions = [(0, 'vacuum')]
    surface_tags_by_region_id = {}
    for (rid, _mat), mapped in zip(regions, surface_out_map):
        surface_tags_by_region_id[rid] = [t for (d, t) in mapped if d == 2]

    # segment.id → fragment 後 Curve tag リスト
    segments = [(0, 'None'), (1, 'PEC'), (2, 'PEC'), (3, 'PEC'), (4, 'PEC'), (5, 'PEC'), (6, 'PEC'), (7, 'PEC')]
    curve_tags_by_segment_id = {}
    for (sid, _bc), mapped in zip(segments, curve_out_map):
        curve_tags_by_segment_id[sid] = [t for (d, t) in mapped if d == 1]

    # === 6. Physical Groups ===
    # 6a. 2D: 領域ごとに material_tag を割り当てる
    for rid, mat in regions:
        tags = surface_tags_by_region_id[rid]
        if not tags:
            continue
        ptag = gmsh.model.addPhysicalGroup(2, tags, tag=rid + 1)
        gmsh.model.setPhysicalName(2, ptag, mat)

    # 6b. 1D: BC 名ごとに curve をまとめる ("None" は内部境界/軸のため除外)
    bc_to_curve_tags = {}
    for sid, bc in segments:
        if bc == "None":
            continue
        ctags = curve_tags_by_segment_id.get(sid, [])
        if not ctags:
            continue
        bc_to_curve_tags.setdefault(bc, set()).update(ctags)
    for bc, ctags in bc_to_curve_tags.items():
        ptag = gmsh.model.addPhysicalGroup(1, sorted(ctags))
        gmsh.model.setPhysicalName(1, ptag, bc)

    # === 7. Mesh generation ===
    gmsh.model.mesh.generate(2)
    if mesh_order == 2:
        gmsh.model.mesh.setOrder(2)
    gmsh.write(out_msh)
    gmsh.finalize()
    print('wrote', out_msh)
    return out_msh


if __name__ == '__main__':
    build_model()
