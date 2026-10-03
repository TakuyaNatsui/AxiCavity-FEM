"""自動生成: MultiRegionGeometry を gmsh API で再構築するスクリプト.

axicavity-fem-v22 GUI の Multi-Region Editor から書き出されたもの。
単位: mm  →  m 換算係数 scale = 0.001
メッシュサイズ lc = mesh_size * scale = 5.0 * 0.001 = 0.005
領域数 = 1  segment 数 = 18  loop 数 = 1
"""

import gmsh
from math import *  # 式で sin/cos/sqrt/pi 等を使えるように (ver2.2)

_SCALE = 0.001

# === user variables (Multi-Region Editor の変数表) ===
b = 234.69
a = 50
r1 = 10
a2 = 91.375
phi = pi*24.969/180
a1 = a+r1+r1*cos(phi)

_LC = 0.005
_MESH_ORDER = 2
_OUT_MSH = 'acc2_pipe_flat.msh'

gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 2.0)
gmsh.model.add('acc2_pipe_flat')

# === 1. Points (z, r) in source-unit, converted to m by *_SCALE ===
point_tags = []
point_tags.append(gmsh.model.occ.addPoint(-250.0 * _SCALE, 0.0 * _SCALE, 0.0, _LC))  # idx 0
point_tags.append(gmsh.model.occ.addPoint(250.0 * _SCALE, 0.0 * _SCALE, 0.0, _LC))  # idx 1
point_tags.append(gmsh.model.occ.addPoint(250.0 * _SCALE, (a) * _SCALE, 0.0, _LC))  # idx 2
point_tags.append(gmsh.model.occ.addPoint((110+r1) * _SCALE, (a) * _SCALE, 0.0, _LC))  # idx 3
point_tags.append(gmsh.model.occ.addPoint(110.0 * _SCALE, (a+r1) * _SCALE, 0.0, _LC))  # idx 4
point_tags.append(gmsh.model.occ.addPoint((110+r1-r1*sin(pi*24.969/180)) * _SCALE, (a1) * _SCALE, 0.0, _LC))  # idx 5
point_tags.append(gmsh.model.occ.addPoint((150-r1+r1*sin(phi)) * _SCALE, (a2-r1*cos(phi)) * _SCALE, 0.0, _LC))  # idx 6
point_tags.append(gmsh.model.occ.addPoint(150.0 * _SCALE, (a2) * _SCALE, 0.0, _LC))  # idx 7
point_tags.append(gmsh.model.occ.addPoint((150-130) * _SCALE, (b) * _SCALE, 0.0, _LC))  # idx 8
point_tags.append(gmsh.model.occ.addPoint((-(150-130)) * _SCALE, (b) * _SCALE, 0.0, _LC))  # idx 9
point_tags.append(gmsh.model.occ.addPoint(-150.0 * _SCALE, (a2) * _SCALE, 0.0, _LC))  # idx 10
point_tags.append(gmsh.model.occ.addPoint((-(150-r1+r1*sin(phi))) * _SCALE, (a2-r1*cos(phi)) * _SCALE, 0.0, _LC))  # idx 11
point_tags.append(gmsh.model.occ.addPoint((-(110+r1-r1*sin(pi*24.969/180))) * _SCALE, (a+r1+r1*cos(pi*24.969/180)) * _SCALE, 0.0, _LC))  # idx 12
point_tags.append(gmsh.model.occ.addPoint(-110.0 * _SCALE, (a+r1) * _SCALE, 0.0, _LC))  # idx 13
point_tags.append(gmsh.model.occ.addPoint((-(110+r1)) * _SCALE, (a) * _SCALE, 0.0, _LC))  # idx 14
point_tags.append(gmsh.model.occ.addPoint(-250.0 * _SCALE, (a) * _SCALE, 0.0, _LC))  # idx 15
point_tags.append(gmsh.model.occ.addPoint(150.0 * _SCALE, (b-130) * _SCALE, 0.0, _LC))  # idx 16
point_tags.append(gmsh.model.occ.addPoint(-150.0 * _SCALE, (b-130) * _SCALE, 0.0, _LC))  # idx 17

# === 2. Segments (line / circle arc) ===
seg_curve_tag = {}
seg_curve_tag[0] = gmsh.model.occ.addLine(point_tags[0], point_tags[1])  # bc='None'
seg_curve_tag[1] = gmsh.model.occ.addLine(point_tags[1], point_tags[2])  # bc='M-short'
seg_curve_tag[2] = gmsh.model.occ.addLine(point_tags[2], point_tags[3])  # bc='PEC'
_center_tag_3 = gmsh.model.occ.addPoint((110.+r1) * _SCALE, (a+r1) * _SCALE, 0.0, _LC)
seg_curve_tag[3] = gmsh.model.occ.addCircleArc(point_tags[3], _center_tag_3, point_tags[4])  # arc, bc='PEC'
_center_tag_4 = gmsh.model.occ.addPoint((110+r1) * _SCALE, (a+r1) * _SCALE, 0.0, _LC)
seg_curve_tag[4] = gmsh.model.occ.addCircleArc(point_tags[4], _center_tag_4, point_tags[5])  # arc, bc='PEC'
seg_curve_tag[5] = gmsh.model.occ.addLine(point_tags[5], point_tags[6])  # bc='PEC'
_center_tag_6 = gmsh.model.occ.addPoint((150-r1) * _SCALE, (a2) * _SCALE, 0.0, _LC)
seg_curve_tag[6] = gmsh.model.occ.addCircleArc(point_tags[6], _center_tag_6, point_tags[7])  # arc, bc='PEC'
seg_curve_tag[8] = gmsh.model.occ.addLine(point_tags[8], point_tags[9])  # bc='PEC'
_center_tag_12 = gmsh.model.occ.addPoint((-(150-r1)) * _SCALE, (a2) * _SCALE, 0.0, _LC)
seg_curve_tag[12] = gmsh.model.occ.addCircleArc(point_tags[10], _center_tag_12, point_tags[11])  # arc, bc='PEC'
seg_curve_tag[13] = gmsh.model.occ.addLine(point_tags[11], point_tags[12])  # bc='PEC'
_center_tag_14 = gmsh.model.occ.addPoint((-(110+r1)) * _SCALE, (a+r1) * _SCALE, 0.0, _LC)
seg_curve_tag[14] = gmsh.model.occ.addCircleArc(point_tags[12], _center_tag_14, point_tags[13])  # arc, bc='PEC'
_center_tag_15 = gmsh.model.occ.addPoint((-110.-r1) * _SCALE, (a+r1) * _SCALE, 0.0, _LC)
seg_curve_tag[15] = gmsh.model.occ.addCircleArc(point_tags[13], _center_tag_15, point_tags[14])  # arc, bc='PEC'
seg_curve_tag[16] = gmsh.model.occ.addLine(point_tags[14], point_tags[15])  # bc='PEC'
seg_curve_tag[17] = gmsh.model.occ.addLine(point_tags[15], point_tags[0])  # bc='M-short'
seg_curve_tag[19] = gmsh.model.occ.addLine(point_tags[7], point_tags[16])  # bc='PEC'
_center_tag_20 = gmsh.model.occ.addPoint(20.0 * _SCALE, 104.69 * _SCALE, 0.0, _LC)
seg_curve_tag[20] = gmsh.model.occ.addCircleArc(point_tags[16], _center_tag_20, point_tags[8])  # arc, bc='PEC'
_center_tag_21 = gmsh.model.occ.addPoint(-20.0 * _SCALE, 104.69 * _SCALE, 0.0, _LC)
seg_curve_tag[21] = gmsh.model.occ.addCircleArc(point_tags[9], _center_tag_21, point_tags[17])  # arc, bc='PEC'
seg_curve_tag[22] = gmsh.model.occ.addLine(point_tags[17], point_tags[10])  # bc='PEC'

# === 3. Curve loops ===
loop_tag = {}
loop_tag[0] = gmsh.model.occ.addCurveLoop([seg_curve_tag[0], seg_curve_tag[1], seg_curve_tag[2], seg_curve_tag[3], seg_curve_tag[4], seg_curve_tag[5], seg_curve_tag[6], seg_curve_tag[19], seg_curve_tag[20], seg_curve_tag[8], seg_curve_tag[21], seg_curve_tag[22], seg_curve_tag[12], seg_curve_tag[13], seg_curve_tag[14], seg_curve_tag[15], seg_curve_tag[16], seg_curve_tag[17]])  # orientation='CCW'

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
segments = [(0, 'None'), (1, 'M-short'), (2, 'PEC'), (3, 'PEC'), (4, 'PEC'), (5, 'PEC'), (6, 'PEC'), (8, 'PEC'), (12, 'PEC'), (13, 'PEC'), (14, 'PEC'), (15, 'PEC'), (16, 'PEC'), (17, 'M-short'), (19, 'PEC'), (20, 'PEC'), (21, 'PEC'), (22, 'PEC')]
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
if _MESH_ORDER == 2:
    gmsh.model.mesh.setOrder(2)
gmsh.write(_OUT_MSH)
gmsh.finalize()
print('wrote', _OUT_MSH)
