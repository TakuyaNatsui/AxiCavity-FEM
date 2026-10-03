"""ver2.1: 多領域 MultiRegionGeometry を Gmsh (.msh) に出力する。

OpenCASCADE (`gmsh.model.occ`) + `fragment()` で共有境界の節点一致を保証し、
領域ごとに 2D Physical Surface、BC ごとに 1D Physical Curve を付与する。

Physical Group 命名規約:
    - 2D Physical Surface 名 = Region.material_tag (例 "vacuum", "dielectric_1")
    - 1D Physical Curve 名 = Segment.bc_name ("PEC" / "E-short" / "M-short")

戻り値の `ExportResult` には、領域・BC・元 segment ID → fragment 後の Gmsh
タグのマッピングを格納している（テスト・デバッグ用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import gmsh

from .multi_region_model import BC_NONE, MultiRegionGeometry, Region, Segment


# 単位文字列 → m へのスケーリング
_UNIT_SCALE = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "inch": 0.0254,
}


@dataclass
class ExportResult:
    """エクスポート結果のメタ情報（テスト・デバッグ用）。

    Attributes:
        surface_tags_by_region_id: region.id → fragment 後 Surface tag のリスト。
            fragment で 1 領域が複数 surface に分割される可能性があるためリスト。
        curve_tags_by_segment_id: segment.id → fragment 後 Curve tag のリスト。
            共有境界化により 1 セグメントが複数 curve に分割されることがある。
        physical_groups: {dim: {name: gmsh physical group tag}} のマップ。
    """

    surface_tags_by_region_id: dict[int, list[int]] = field(default_factory=dict)
    curve_tags_by_segment_id: dict[int, list[int]] = field(default_factory=dict)
    physical_groups: dict[int, dict[str, int]] = field(default_factory=dict)


def _build_model_in_gmsh(
    geom: MultiRegionGeometry,
    *,
    model_name: str,
) -> tuple[float, ExportResult]:
    """既に gmsh.initialize() 済みの状態で、geom から OCC モデルを構築する。

    Returns:
        (lc, ExportResult) — lc は m 単位のメッシュサイズ、ExportResult は fragment
        後の Surface/Curve/Physical Group のマッピング。
    """
    scale = _UNIT_SCALE.get(geom.unit, 1.0)
    lc = float(geom.mesh_size) * scale
    gmsh.model.add(model_name)

    # --- 1. point ---
    point_tags: list[int] = [
        gmsh.model.occ.addPoint(p[0] * scale, p[1] * scale, 0.0, lc)
        for p in geom.points
    ]

    # --- 2. segment → curve ---
    seg_curve_tag: dict[int, int] = {}
    for seg in geom.segments:
        p1 = point_tags[seg.point_indices[0]]
        p2 = point_tags[seg.point_indices[1]]
        if seg.type == "line":
            ct = gmsh.model.occ.addLine(p1, p2)
        else:  # arc
            cp = gmsh.model.occ.addPoint(
                seg.center[0] * scale, seg.center[1] * scale, 0.0, lc,
            )
            ct = gmsh.model.occ.addCircleArc(p1, cp, p2)
        seg_curve_tag[seg.id] = ct

    # --- 3. loop → curveLoop ---
    loop_tag: dict[int, int] = {}
    for loop in geom.loops:
        curve_tags = [seg_curve_tag[sid] for sid in loop.segment_ids]
        loop_tag[loop.id] = gmsh.model.occ.addCurveLoop(curve_tags)

    # --- 4. region → planeSurface ---
    region_surface_tag: dict[int, int] = {}
    for region in geom.regions:
        loop_tags = [loop_tag[region.outer_loop_id]] + [
            loop_tag[h] for h in region.hole_loop_ids
        ]
        region_surface_tag[region.id] = gmsh.model.occ.addPlaneSurface(loop_tags)

    gmsh.model.occ.synchronize()

    # --- 5. fragment で共有境界 ---
    surface_dimtags_in = [(2, t) for t in region_surface_tag.values()]
    curve_dimtags_in = [(1, t) for t in seg_curve_tag.values()]
    objects_in = surface_dimtags_in + curve_dimtags_in

    if len(geom.regions) > 1 or any(r.hole_loop_ids for r in geom.regions):
        _, out_map = gmsh.model.occ.fragment(objects_in, [])
        gmsh.model.occ.synchronize()
    else:
        out_map = [[dt] for dt in objects_in]

    n_surf = len(surface_dimtags_in)
    surface_out_map = out_map[:n_surf]
    curve_out_map = out_map[n_surf:]

    surface_tags_by_region_id: dict[int, list[int]] = {}
    for region, mapped in zip(geom.regions, surface_out_map):
        surface_tags_by_region_id[region.id] = [
            tag for (dim, tag) in mapped if dim == 2
        ]

    curve_tags_by_segment_id: dict[int, list[int]] = {}
    for seg, mapped in zip(geom.segments, curve_out_map):
        curve_tags_by_segment_id[seg.id] = [
            tag for (dim, tag) in mapped if dim == 1
        ]

    # --- 6. Physical Groups ---
    physical_groups: dict[int, dict[str, int]] = {1: {}, 2: {}}
    for region in geom.regions:
        tags = surface_tags_by_region_id[region.id]
        if not tags:
            continue
        ptag = gmsh.model.addPhysicalGroup(2, tags, tag=region.id + 1)
        gmsh.model.setPhysicalName(2, ptag, region.material_tag)
        physical_groups[2][region.material_tag] = ptag

    bc_to_curve_tags: dict[str, set[int]] = {}
    for seg in geom.segments:
        if seg.bc_name == BC_NONE:
            continue  # "None" は Physical Curve を付けない (内部境界/軸)
        ctags = curve_tags_by_segment_id.get(seg.id, [])
        if not ctags:
            continue
        bc_to_curve_tags.setdefault(seg.bc_name, set()).update(ctags)

    for bc_name, ctags in bc_to_curve_tags.items():
        ptag = gmsh.model.addPhysicalGroup(1, sorted(ctags))
        gmsh.model.setPhysicalName(1, ptag, bc_name)
        physical_groups[1][bc_name] = ptag

    result = ExportResult(
        surface_tags_by_region_id=surface_tags_by_region_id,
        curve_tags_by_segment_id=curve_tags_by_segment_id,
        physical_groups=physical_groups,
    )
    return lc, result


def export_msh_multi_region(
    geom: MultiRegionGeometry,
    out_path: str | Path,
    *,
    mesh_order: int = 1,
    model_name: str | None = None,
    verbose: int = 1,
) -> ExportResult:
    """`geom` を Gmsh で 2D メッシュ化して `out_path` (.msh) に書き出す。

    Args:
        geom: 多領域幾何モデル。`geom.validate()` が空でない場合は ValueError。
        out_path: 出力ファイル (.msh)。
        mesh_order: 1 または 2。
        model_name: Gmsh モデル名。省略時はファイル名（拡張子除く）。
        verbose: Gmsh の verbosity (0..5)。

    Returns:
        ExportResult: タグマッピング等のメタ情報。
    """
    errs = geom.validate()
    if errs:
        raise ValueError("MultiRegionGeometry validation failed:\n  - "
                         + "\n  - ".join(errs))
    if mesh_order not in (1, 2):
        raise ValueError(f"mesh_order must be 1 or 2, got {mesh_order}")
    if not geom.regions:
        raise ValueError("MultiRegionGeometry has no regions")

    out_path = Path(out_path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", float(verbose))
        _, result = _build_model_in_gmsh(
            geom, model_name=model_name or out_path.stem,
        )

        gmsh.model.mesh.generate(2)
        if mesh_order == 2:
            gmsh.model.mesh.setOrder(2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out_path))
        return result
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def export_geo_multi_region(
    geom: MultiRegionGeometry,
    out_path: str | Path,
    *,
    mesh_order: int = 1,
) -> Path:
    """`geom` を `.geo` (OpenCASCADE モード) のテキストとして書き出す。

    `gmsh.write("...geo_unrolled")` は OCC モデルでは XAO へのラッパーしか
    出さず人間可読でないため、ここでは geom のデータから直接 GEO スクリプトを
    生成する。Gmsh GUI でこのファイルを開けば、OCC + BooleanFragments を
    使った構築過程を逐行で確認できる。

    生成される GEO は以下の構造:
        SetFactory("OpenCASCADE");
        Point(1) = {x, y, 0, lc}; ...
        Line(1) = {p1, p2}; / Circle(N) = {p1, center, p2};
        Curve Loop(L) = {seg_ids};
        Plane Surface(S) = {outer_loop, hole_loop, ...};
        BooleanFragments{ Surface{...}; Delete; }{ } (多領域時のみ)
        Physical Surface("material_tag") = {pre_fragment_surface_ids};
        Physical Curve("bc_name") = {pre_fragment_curve_ids};
        Mesh.ElementOrder = N;
        Mesh 2;

    注: Physical Group は **fragment 前のタグ**で指定する。Gmsh は OCC モデルで
    Boolean 操作後も Physical Group の指定対象 (元の Surface/Curve) を追跡
    するため、これで動作する。
    """
    errs = geom.validate()
    if errs:
        raise ValueError("MultiRegionGeometry validation failed:\n  - "
                         + "\n  - ".join(errs))
    if mesh_order not in (1, 2):
        raise ValueError(f"mesh_order must be 1 or 2, got {mesh_order}")
    if not geom.regions:
        raise ValueError("MultiRegionGeometry has no regions")

    out_path = Path(out_path)
    scale = _UNIT_SCALE.get(geom.unit, 1.0)
    lc = float(geom.mesh_size) * scale

    lines: list[str] = []
    w = lines.append

    w(f"// Auto-generated GEO from MultiRegionGeometry "
      f"(axicavity-fem GUI / Multi-Region Editor)")
    w(f"// Unit: {geom.unit}  (scale to m = {scale})")
    w(f"// mesh_size in source unit = {geom.mesh_size}  →  lc [m] = {lc}")
    w(f"// Regions: {len(geom.regions)},  Segments: {len(geom.segments)},  "
      f"Loops: {len(geom.loops)}")
    w("")
    w('SetFactory("OpenCASCADE");')
    w(f"lc = {lc};")
    w("")

    # 1-based tag マップ
    point_tag: dict[int, int] = {}
    next_point_tag = 1
    w("// === Points ===  (Gmsh tag = idx+1, coords in meters)")
    for idx, (z, r) in enumerate(geom.points):
        tag = next_point_tag
        next_point_tag += 1
        point_tag[idx] = tag
        w(f"Point({tag}) = {{{float(z) * scale}, {float(r) * scale}, 0, lc}};")
    w("")

    # 円弧の center 点は追加 Point として割り当て
    w("// === Curves: Line / Circle arc ===")
    seg_curve_tag: dict[int, int] = {}
    next_curve_tag = 1
    for seg in geom.segments:
        p1 = point_tag[seg.point_indices[0]]
        p2 = point_tag[seg.point_indices[1]]
        ct = next_curve_tag
        next_curve_tag += 1
        seg_curve_tag[seg.id] = ct
        if seg.type == "line":
            w(f"Line({ct}) = {{{p1}, {p2}}};"
              f"  // seg.id={seg.id}, bc={seg.bc_name!r}")
        else:
            cz, cr = seg.center  # type: ignore[misc]
            cp_tag = next_point_tag
            next_point_tag += 1
            w(f"Point({cp_tag}) = {{{float(cz) * scale}, "
              f"{float(cr) * scale}, 0, lc}};  // arc center for seg {seg.id}")
            w(f"Circle({ct}) = {{{p1}, {cp_tag}, {p2}}};"
              f"  // seg.id={seg.id}, bc={seg.bc_name!r}")
    w("")

    w("// === Curve Loops ===  (orientation per Loop record)")
    loop_tag_map: dict[int, int] = {}
    next_loop_tag = 1
    for loop in geom.loops:
        lt = next_loop_tag
        next_loop_tag += 1
        loop_tag_map[loop.id] = lt
        seg_ref = ", ".join(str(seg_curve_tag[sid]) for sid in loop.segment_ids)
        w(f"Curve Loop({lt}) = {{{seg_ref}}};"
          f"  // loop.id={loop.id}, orientation={loop.orientation!r}")
    w("")

    w("// === Plane Surfaces ===  (1 outer loop + 0..N holes)")
    region_surface_tag: dict[int, int] = {}
    next_surf_tag = 1
    for region in geom.regions:
        st = next_surf_tag
        next_surf_tag += 1
        region_surface_tag[region.id] = st
        loop_refs = [str(loop_tag_map[region.outer_loop_id])] + [
            str(loop_tag_map[h]) for h in region.hole_loop_ids
        ]
        w(f"Plane Surface({st}) = {{{', '.join(loop_refs)}}};"
          f"  // region.id={region.id} name={region.name!r} "
          f"material_tag={region.material_tag!r} eps_r={region.eps_r}")
    w("")

    need_fragment = (len(geom.regions) > 1
                     or any(r.hole_loop_ids for r in geom.regions))
    if need_fragment:
        w("// === Boolean Fragments ===")
        w("// 共有境界の節点を一致させるため、全 Surface + 全 Curve を融合する。")
        w("// Gmsh は OCC の Boolean 操作後も Physical Group の対象を追跡するので、")
        w("// 下の Physical Surface/Curve は fragment 前のタグで指定して OK。")
        surf_list = ", ".join(str(t) for t in region_surface_tag.values())
        curve_list = ", ".join(str(t) for t in seg_curve_tag.values())
        w(f"BooleanFragments{{ Surface{{{surf_list}}}; Delete; }}"
          f"{{ Curve{{{curve_list}}}; Delete; }}")
        w("")

    w("// === Physical Groups ===")
    w("// 2D: 領域ごとに material_tag (1 region = 1 group)")
    for region in geom.regions:
        st = region_surface_tag[region.id]
        w(f'Physical Surface("{region.material_tag}", {region.id + 1}) = {{{st}}};'
          f"  // {region.name!r} eps_r={region.eps_r}")
    w("")
    w('// 1D: BC 名ごとに segment をまとめる ("None" は内部境界/軸のため除外)')
    bc_to_seg_ids: dict[str, list[int]] = {}
    for seg in geom.segments:
        if seg.bc_name == BC_NONE:
            continue
        bc_to_seg_ids.setdefault(seg.bc_name, []).append(seg_curve_tag[seg.id])
    next_phys_curve_tag = 100  # 任意の開始タグ
    for bc_name, ctags in bc_to_seg_ids.items():
        w(f'Physical Curve("{bc_name}", {next_phys_curve_tag}) = '
          f'{{{", ".join(str(t) for t in ctags)}}};')
        next_phys_curve_tag += 1
    w("")

    w("// === Mesh settings ===")
    w(f"Mesh.ElementOrder = {int(mesh_order)};")
    w("Mesh.MeshSizeExtendFromBoundary = 1;")
    w("Mesh.MeshSizeFromPoints = 1;")
    w("Mesh 2;")
    w("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def export_python_script_multi_region(
    geom: MultiRegionGeometry,
    out_path: str | Path,
    *,
    mesh_order: int = 1,
    msh_output: str | Path | None = None,
) -> Path:
    """`geom` を構築する**自己完結 Python スクリプト**を生成する。

    生成されたスクリプトは `gmsh` API を直接使ってジオメトリを構築し、
    .msh ファイルを書き出す。`shared/gmsh_export_occ.py:_build_model_in_gmsh`
    と同じ呼び出し列を生成する（教育用に注釈付き）。

    Args:
        geom: 多領域幾何モデル。
        out_path: 出力 Python ファイルパス。
        mesh_order: スクリプト内のメッシュ次数 (1 or 2)。
        msh_output: スクリプト実行時の .msh 出力パス。None なら
            ``<out_path stem>.msh``。

    Returns:
        書き出した Python ファイルのパス。
    """
    errs = geom.validate()
    if errs:
        raise ValueError("MultiRegionGeometry validation failed:\n  - "
                         + "\n  - ".join(errs))
    if mesh_order not in (1, 2):
        raise ValueError(f"mesh_order must be 1 or 2, got {mesh_order}")

    out_path = Path(out_path)
    if msh_output is None:
        msh_output = out_path.with_suffix(".msh").name

    scale = _UNIT_SCALE.get(geom.unit, 1.0)
    lc = float(geom.mesh_size) * scale

    lines: list[str] = []
    w = lines.append

    def wb(s: str = "") -> None:
        """build_model 関数の本体行を 4 スペースインデントで追加する。"""
        lines.append(("    " + s) if s else "")

    w('"""自動生成: MultiRegionGeometry を gmsh API で再構築するスクリプト.')
    w("")
    w("axicavity-fem GUI の Multi-Region Editor から書き出されたもの。")
    w(f"単位: {geom.unit}  →  m 換算係数 scale = {scale}")
    w(f"メッシュサイズ lc = mesh_size * scale = {geom.mesh_size} * {scale} = {lc}")
    w(f"領域数 = {len(geom.regions)}  segment 数 = {len(geom.segments)}  "
      f"loop 数 = {len(geom.loops)}")
    w("")
    w("build_model(**user_vars) を外部から呼ぶと、変数表の値を上書きして")
    w("メッシュを再生成できます (周波数自動調整などに利用)。")
    w("例: build_model(a=120, gap=3, out_msh='tuned.msh')")
    w('"""')
    w("")
    w("import gmsh")
    w("from math import *  # 式で sin/cos/sqrt/pi 等を使えるように (ver2.2)")
    w("")
    w(f"_SCALE = {scale!r}")
    w(f"_MESH_ORDER = {int(mesh_order)}")
    w(f"_OUT_MSH = {str(msh_output)!r}")
    w("")
    w("")
    w("def build_model(out_msh=_OUT_MSH, mesh_order=_MESH_ORDER, **user_vars):")
    w('    """geom を構築し .msh を書き出す。user_vars で変数表の値を上書き可能。"""')
    if geom.variables:
        wb("# === user variables (変数表) — user_vars で上書き可、依存変数は再計算 ===")
        for _vn, _ve in geom.variables:
            wb(f"{_vn} = user_vars.get({_vn!r}, {_ve})")
        wb()
    # mesh size は変数を参照しうるので関数内 (変数定義の後) で計算する
    if geom.mesh_size_expr:
        wb(f"_LC = ({geom.mesh_size_expr}) * _SCALE  # mesh_size を式で指定")
    else:
        wb(f"_LC = {lc!r}")
    wb()
    wb("gmsh.initialize()")
    wb("gmsh.option.setNumber('General.Verbosity', 2.0)")
    wb(f"gmsh.model.add({out_path.stem!r})")
    wb()
    wb("# === 1. Points (z, r) in source-unit, converted to m by *_SCALE ===")
    wb("point_tags = []")
    for i, (z, r) in enumerate(geom.points):
        ze, re_ = (geom.point_exprs[i] if i < len(geom.point_exprs) else (None, None))
        zt = f"({ze})" if ze else f"{float(z)!r}"
        rt = f"({re_})" if re_ else f"{float(r)!r}"
        wb(f"point_tags.append(gmsh.model.occ.addPoint("
           f"{zt} * _SCALE, {rt} * _SCALE, 0.0, _LC))  # idx {i}")
    wb()
    wb("# === 2. Segments (line / circle arc) ===")
    wb("seg_curve_tag = {}")
    for seg in geom.segments:
        p1, p2 = seg.point_indices
        if seg.type == "line":
            wb(f"seg_curve_tag[{seg.id}] = gmsh.model.occ.addLine("
               f"point_tags[{p1}], point_tags[{p2}])  "
               f"# bc={seg.bc_name!r}")
        else:
            cz, cr = seg.center  # type: ignore[misc]
            cze, cre = seg.center_expr if seg.center_expr else (None, None)
            czt = f"({cze})" if cze else f"{float(cz)!r}"
            crt = f"({cre})" if cre else f"{float(cr)!r}"
            wb(f"_center_tag_{seg.id} = gmsh.model.occ.addPoint("
               f"{czt} * _SCALE, {crt} * _SCALE, 0.0, _LC)")
            wb(f"seg_curve_tag[{seg.id}] = gmsh.model.occ.addCircleArc("
               f"point_tags[{p1}], _center_tag_{seg.id}, point_tags[{p2}])  "
               f"# arc, bc={seg.bc_name!r}")
    wb()
    wb("# === 3. Curve loops ===")
    wb("loop_tag = {}")
    for loop in geom.loops:
        seg_list = ", ".join(f"seg_curve_tag[{sid}]" for sid in loop.segment_ids)
        wb(f"loop_tag[{loop.id}] = gmsh.model.occ.addCurveLoop([{seg_list}])  "
           f"# orientation={loop.orientation!r}")
    wb()
    wb("# === 4. Plane surfaces (region = outer loop + hole loops) ===")
    wb("region_surface_tag = {}")
    for region in geom.regions:
        loop_args = [f"loop_tag[{region.outer_loop_id}]"] + [
            f"loop_tag[{h}]" for h in region.hole_loop_ids
        ]
        wb(f"region_surface_tag[{region.id}] = gmsh.model.occ.addPlaneSurface("
           f"[{', '.join(loop_args)}])  "
           f"# name={region.name!r} material_tag={region.material_tag!r} "
           f"eps_r={region.eps_r}")
    wb()
    wb("gmsh.model.occ.synchronize()")
    wb()
    wb("# === 5. Boolean fragment to share boundary nodes between regions ===")
    wb("surface_dimtags_in = [(2, t) for t in region_surface_tag.values()]")
    wb("curve_dimtags_in = [(1, t) for t in seg_curve_tag.values()]")
    wb("objects_in = surface_dimtags_in + curve_dimtags_in")
    need_fragment = (len(geom.regions) > 1
                     or any(r.hole_loop_ids for r in geom.regions))
    if need_fragment:
        wb("_, out_map = gmsh.model.occ.fragment(objects_in, [])")
        wb("gmsh.model.occ.synchronize()")
    else:
        wb("# 単一領域・穴なしのため fragment 不要 (恒等マップ)")
        wb("out_map = [[dt] for dt in objects_in]")
    wb("n_surf = len(surface_dimtags_in)")
    wb("surface_out_map = out_map[:n_surf]")
    wb("curve_out_map = out_map[n_surf:]")
    wb()
    wb("# region.id → fragment 後 Surface tag リスト")
    wb(f"regions = {[(r.id, r.material_tag) for r in geom.regions]!r}")
    wb("surface_tags_by_region_id = {}")
    wb("for (rid, _mat), mapped in zip(regions, surface_out_map):")
    wb("    surface_tags_by_region_id[rid] = [t for (d, t) in mapped if d == 2]")
    wb()
    wb("# segment.id → fragment 後 Curve tag リスト")
    wb(f"segments = {[(s.id, s.bc_name) for s in geom.segments]!r}")
    wb("curve_tags_by_segment_id = {}")
    wb("for (sid, _bc), mapped in zip(segments, curve_out_map):")
    wb("    curve_tags_by_segment_id[sid] = [t for (d, t) in mapped if d == 1]")
    wb()
    wb("# === 6. Physical Groups ===")
    wb("# 6a. 2D: 領域ごとに material_tag を割り当てる")
    wb("for rid, mat in regions:")
    wb("    tags = surface_tags_by_region_id[rid]")
    wb("    if not tags:")
    wb("        continue")
    wb("    ptag = gmsh.model.addPhysicalGroup(2, tags, tag=rid + 1)")
    wb("    gmsh.model.setPhysicalName(2, ptag, mat)")
    wb()
    wb('# 6b. 1D: BC 名ごとに curve をまとめる ("None" は内部境界/軸のため除外)')
    wb("bc_to_curve_tags = {}")
    wb("for sid, bc in segments:")
    wb('    if bc == "None":')
    wb("        continue")
    wb("    ctags = curve_tags_by_segment_id.get(sid, [])")
    wb("    if not ctags:")
    wb("        continue")
    wb("    bc_to_curve_tags.setdefault(bc, set()).update(ctags)")
    wb("for bc, ctags in bc_to_curve_tags.items():")
    wb("    ptag = gmsh.model.addPhysicalGroup(1, sorted(ctags))")
    wb("    gmsh.model.setPhysicalName(1, ptag, bc)")
    wb()
    wb("# === 7. Mesh generation ===")
    wb("gmsh.model.mesh.generate(2)")
    wb("if mesh_order == 2:")
    wb("    gmsh.model.mesh.setOrder(2)")
    wb("gmsh.write(out_msh)")
    wb("gmsh.finalize()")
    wb("print('wrote', out_msh)")
    wb("return out_msh")
    w("")
    w("")
    w("if __name__ == '__main__':")
    w("    build_model()")
    w("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# 補助: メッシュファイルから領域別・BC 別の要素・節点を読み戻す
# ---------------------------------------------------------------------------
def inspect_msh_physical_groups(msh_path: str | Path) -> dict[int, dict[str, int]]:
    """`.msh` から (dim, name) → Physical Group tag のマップを読み出す。

    テスト用。
    """
    msh_path = Path(msh_path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0.0)
        gmsh.open(str(msh_path))
        out: dict[int, dict[str, int]] = {0: {}, 1: {}, 2: {}, 3: {}}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            out.setdefault(dim, {})[name] = tag
        return out
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
