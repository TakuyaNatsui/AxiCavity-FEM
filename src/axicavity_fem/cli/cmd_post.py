"""post サブコマンド: 固有解から工学パラメータを計算し v2 HDF5 へ追記する."""

from __future__ import annotations

import os
import shutil
import warnings

import h5py

from ..shared.constants import C0
from ..shared.hdf5_io import append_post_process, is_v2, read_results


def _tm0_params_to_attrs(mp):
    # ver2.3: "Q" は全損失込みの合計 Q（tanδ=0 なら従来値と同一）。
    return {
        "Q": mp.q_factor, "Q_wall": mp.q_wall, "Q_diel": mp.q_diel,
        "R_over_Q": mp.rq_eff, "V_eff": mp.v_eff,
        "V_acc": mp.v_acc, "U_stored": mp.stored_energy, "P_loss": mp.p_loss,
        "P_diel": mp.p_diel, "P_flow_zmin": mp.p_flow,
        "group_velocity": mp.v_group_c * C0, "attenuation": mp.attenuation,
        "frequency_GHz": mp.frequency_ghz,
    }


def _hom_params_to_attrs(mp):
    # ver2.3: "Q" は全損失込みの合計 Q（tanδ=0 なら従来値と同一）。
    return {
        "Q": mp.q_factor, "Q_wall": mp.q_wall, "Q_diel": mp.q_diel,
        "U_stored": mp.stored_energy, "P_loss": mp.p_loss,
        "P_diel": mp.p_diel,
        "P_flow_zmin": mp.p_flow, "group_velocity": mp.v_group_c * C0,
        "v_phase_c": mp.v_phase_c, "attenuation": mp.attenuation,
        "frequency_GHz": mp.frequency_ghz,
    }


def _post_tm0(data, cond, beta):
    from ..fem_tm0.assembly import assemble_global_matrices
    from ..fem_tm0.boundary import find_nodes_on_r0_boundary
    from ..fem_tm0.post_process import compute_tm0_parameters
    from ..fem_tm0.solver import TM0Result
    from ..shared.boundary_groups import classify_boundaries
    from ..shared.mesh_loader import MeshData

    mesh = data["mesh"]
    md = MeshData(nodes=mesh["vertices"], elements=mesh["simplices"],
                  element_order=mesh["elem_order"],
                  physical_groups=mesh["physical_groups"])
    # M は TM0 では ε_r 非依存（K に 1/ε_r が乗る）なので真空 M でよい。
    # ε_r は電場の再構成（軸上の E_z と E_r。規格化、V、R/Q、P_flow に効く）に要る。
    _, M_global = assemble_global_matrices(md)
    eps_r_per_element = mesh.get("eps_r_per_element")  # 誘電体（None で真空）
    tan_delta_per_element = mesh.get("tan_delta_per_element")  # None で無損失
    classification = classify_boundaries(mesh["physical_groups"], mesh["vertices"])
    r0 = find_nodes_on_r0_boundary(mesh["vertices"])

    rec = data["results_by_n"][0]
    post = {}
    if "standing" in rec:
        ms = rec["standing"]
        result = TM0Result(
            nodes=mesh["vertices"], elements=mesh["simplices"],
            element_order=mesh["elem_order"], analysis_type="standing",
            physical_groups=mesh["physical_groups"], r0_nodes=r0,
            M_global=M_global, frequencies=ms["frequencies"],
            eigenvectors=ms["eigenvectors"], physical_edges=mesh.get("physical_edges"))
        params = compute_tm0_parameters(
            result, classification, cond, beta,
            eps_r_per_element=eps_r_per_element,
            tan_delta_per_element=tan_delta_per_element)
        post["standing"] = [_tm0_params_to_attrs(mp)
                            for mp in params.per_phase[0.0]]
    if "traveling" in rec:
        phase_results = {ph: {"frequencies": d["frequencies"],
                              "eigenvectors": d["eigenvectors"]}
                         for ph, d in rec["traveling"].items()}
        result = TM0Result(
            nodes=mesh["vertices"], elements=mesh["simplices"],
            element_order=mesh["elem_order"], analysis_type="traveling",
            physical_groups=mesh["physical_groups"], r0_nodes=r0,
            M_global=M_global, phase_results=phase_results, physical_edges=mesh.get("physical_edges"))
        params = compute_tm0_parameters(
            result, classification, cond, beta,
            eps_r_per_element=eps_r_per_element,
            tan_delta_per_element=tan_delta_per_element)
        post["traveling"] = {ph: [_tm0_params_to_attrs(mp) for mp in plist]
                             for ph, plist in params.per_phase.items()}
    return {0: post}


def _post_hom(data, cond):
    from ..fem_hom.post_process import compute_hom_parameters
    from ..fem_hom.solver import HOMModeSet, HOMResult
    from ..shared.mesh_loader import compute_hom_boundary

    mesh = data["mesh"]
    edge_index_map = mesh["edge_index_map"]
    num_edges = mesh["num_edges"]
    eps_r_per_element = mesh.get("eps_r_per_element")  # 誘電体（None で真空）
    tan_delta_per_element = mesh.get("tan_delta_per_element")  # None で無損失
    post_by_n = {}

    for n, rec in data["results_by_n"].items():
        _, _, pec_loss = compute_hom_boundary(
            mesh["vertices"], mesh["simplices"], edge_index_map,
            mesh["physical_groups"], n, mesh.get("physical_edges"))

        def _make_result(analysis_type, normal=None, periodic=None):
            return HOMResult(
                n=n, element_order=mesh["elem_order"],
                simplices=mesh["simplices"], vertices=mesh["vertices"],
                num_edges=num_edges, edge_index_map=edge_index_map,
                physical_groups=mesh["physical_groups"],
                analysis_type=analysis_type, normal=normal,
                periodic=periodic or {}, pec_loss_edge_indices=pec_loss)

        post = {}
        if "standing" in rec:
            ms = rec["standing"]
            result = _make_result("standing", normal=HOMModeSet(
                ms["frequencies"], ms["eigenvalues"], ms["eigenvectors"]))
            params = compute_hom_parameters(
                result, cond, eps_r_per_element=eps_r_per_element,
                tan_delta_per_element=tan_delta_per_element)
            post["standing"] = [_hom_params_to_attrs(mp)
                                for mp in params.per_phase[0.0]]
        if "traveling" in rec:
            periodic = {ph: HOMModeSet(d["frequencies"], d["eigenvalues"],
                                       d["eigenvectors"])
                        for ph, d in rec["traveling"].items()}
            result = _make_result("traveling", periodic=periodic)
            params = compute_hom_parameters(
                result, cond, eps_r_per_element=eps_r_per_element,
                tan_delta_per_element=tan_delta_per_element)
            post["traveling"] = {ph: [_hom_params_to_attrs(mp) for mp in plist]
                                 for ph, plist in params.per_phase.items()}
        post_by_n[n] = post
    return post_by_n


def _output_problem(target: str) -> str | None:
    """出力先に書けない理由を返す（計算の前に確かめる）. 問題なければ None."""
    if os.path.isdir(target):
        return f"出力先がフォルダです: {target}"
    if os.path.splitext(target)[1].lower() == ".txt":       # 同じ名前にサマリの .txt を書くので拒否する
        return f"出力先が .txt です。.h5 を指定してください: {target}"
    if not os.path.isdir(os.path.dirname(os.path.abspath(target))):
        return f"出力先のフォルダがありません: {target}"
    if os.path.exists(target) and not os.access(target, os.W_OK):
        return f"出力先が読み取り専用です: {target}"
    return None


def _write_output(input_file: str, target: str, post_by_n: dict) -> None:
    """入力の写しに post を追記して、できあがってから出力と置き換える.

    途中で止まっても出力はそのまま残る（入力と出力が同じファイルでも）。入力の属性は写さない。
    """
    temporary = target + ".tmp"
    try:
        shutil.copyfile(input_file, temporary)
        append_post_process(temporary, post_by_n)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def run(args) -> int:
    """post サブコマンドの本体."""
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        return 1
    if not h5py.is_hdf5(args.input_file):
        print(f"エラー: 入力が HDF5 ファイルではありません: {args.input_file}")
        return 1
    if not is_v2(args.input_file):
        print(f"エラー: ver1 形式のファイルは post できません。solve し直してください: {args.input_file}")
        return 1

    target = args.output_file or args.input_file
    problem = _output_problem(target)
    if problem:
        print(f"エラー: {problem}")
        return 1

    data = read_results(args.input_file)
    solver_type = data["solver_type"] or args.type
    print(f"[post] type={solver_type} input={args.input_file} "
          f"cond={args.cond} beta={args.beta}")

    if solver_type == "tm0":
        post_by_n = _post_tm0(data, args.cond, args.beta)
    else:
        post_by_n = _post_hom(data, args.cond)

    try:
        _write_output(args.input_file, target, post_by_n)
    except OSError as exc:                  # 他のプログラムが出力を開いている、など（出力はそのまま残る）
        print(f"エラー: 出力に書き込めませんでした（前の内容は残っています）: {target}: {exc}")
        return 1

    # 計算結果の簡易テキスト出力 (.h5 → .txt)。周波数に加え Q 値なども書き出す。
    from .summary_txt import write_post_summary
    params = dict(data.get("parameters") or {})
    params.setdefault("solver_type", solver_type)
    try:
        txt = write_post_summary(target, params, post_by_n)
        print(f"[post] テキストサマリを保存: {txt}")
    except OSError as exc:                  # h5 はもう新しい値なので失敗にしない（表計算ソフトが開いている、など）
        warnings.warn(f"テキストサマリを書けませんでした（.h5 は保存済み）: {exc}", stacklevel=2)

    for n, rec in post_by_n.items():
        if "standing" in rec:
            for k, p in enumerate(rec["standing"]):
                extra = (f" Q_diel={p['Q_diel']:.4e}"
                         if p.get("Q_diel", 0.0) > 0 else "")
                print(f"  n={n} mode{k}: f={p['frequency_GHz']:.6f} GHz "
                      f"Q={p['Q']:.4e} P_loss={p['P_loss']:.4e} W" + extra)
        if "traveling" in rec:
            for ph, plist in rec["traveling"].items():
                for k, p in enumerate(plist):
                    print(f"  n={n} ph={ph} mode{k}: f={p['frequency_GHz']:.6f} "
                          f"GHz Q={p['Q']:.4e} Vg={p['group_velocity']/C0:.4f}c")
    print(f"[post] パラメータを追記: {target}")
    return 0
