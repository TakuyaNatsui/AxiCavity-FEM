"""solve サブコマンド: FEM 固有値解析を実行し v2 HDF5 へ保存する."""

from __future__ import annotations

import json
import os

import numpy as np

from ..shared.cli_common import parse_phase_list
from ..shared.hdf5_io import write_results


def _load_material_table(mesh_file: str, materials_arg: str | None
                         ) -> dict | None:
    """material_table を読み込む。

    明示指定 (``--materials``) があればそれを使う。無ければ
    ``<mesh>.materials.json`` を自動探索する。どちらも無ければ ``None``。

    JSON 形式:
        {"schema": "axicavity-fem-v21.materials/1",
         "materials": {"<tag>": {"eps_r": float, "mu_r": float,
                                 "tan_delta": float}, ...}}
    """
    path = materials_arg
    if path is None:
        guess = os.path.splitext(mesh_file)[0] + ".materials.json"
        if os.path.exists(guess):
            path = guess
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"materials JSON が見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 旧形式 (フラット) と新形式 ({"materials": {...}}) の両方を受ける
    table = data.get("materials", data)
    # サニティ: 値が dict であること
    cleaned: dict[str, dict[str, float]] = {}
    for tag, entry in table.items():
        if isinstance(entry, dict):
            cleaned[str(tag)] = {
                "eps_r": float(entry.get("eps_r", 1.0)),
                "mu_r": float(entry.get("mu_r", 1.0)),
                "tan_delta": float(entry.get("tan_delta", 0.0)),
            }
    print(f"[solve] material_table 読込: {path} ({len(cleaned)} 材料)")
    for tag, ent in cleaned.items():
        print(f"  - {tag}: eps_r={ent['eps_r']}, mu_r={ent['mu_r']}, "
              f"tan_delta={ent['tan_delta']}")
    return cleaned


def _target_freq(args) -> float | None:
    """``--target-freq`` [GHz] を取り出す（未指定・非正なら None, ver2.3）."""
    val = getattr(args, "target_freq", None)
    if val is None:
        return None
    val = float(val)
    if val <= 0.0:
        return None
    return val


def _default_output(mesh_file: str, solver_type: str) -> str:
    base = os.path.splitext(os.path.basename(mesh_file))[0]
    return f"{base}_{solver_type}.h5"


def _edge_map_arrays(edge_index_map):
    keys = np.array(list(edge_index_map.keys()), dtype=int)
    values = np.array(list(edge_index_map.values()), dtype=int)
    return keys, values


def _modeset_dict(frequencies, eigenvalues, eigenvectors):
    return {"frequencies": np.asarray(frequencies),
            "eigenvalues": (np.asarray(eigenvalues)
                            if eigenvalues is not None else None),
            "eigenvectors": np.asarray(eigenvectors)}


def _solve_tm0(args, phase_list):
    from ..fem_tm0.solver import solve_tm0_standing, solve_tm0_traveling
    from ..shared.mesh_loader import load_mesh
    from ..shared.material_resolver import (build_eps_r_per_element,
                                            build_tan_delta_per_element)

    mesh = load_mesh(args.mesh_file, args.elem_order)
    is_traveling = not (len(phase_list) == 1 and phase_list[0] == 0.0)

    material_table = _load_material_table(
        args.mesh_file, getattr(args, "materials_file", None))
    target_freq = _target_freq(args)

    if is_traveling:
        res = solve_tm0_traveling(mesh, args.num_modes, phase_list,
                                  material_table=material_table,
                                  target_freq_ghz=target_freq)
        rec = {"traveling": {
            ph: _modeset_dict(d["frequencies"], None, d["eigenvectors"])
            for ph, d in res.phase_results.items()}}
    else:
        res = solve_tm0_standing(mesh, args.num_modes,
                                 material_table=material_table,
                                 target_freq_ghz=target_freq)
        rec = {"standing": _modeset_dict(res.frequencies, None,
                                         res.eigenvectors)}

    mesh_dict = {
        "vertices": mesh.nodes, "simplices": mesh.elements,
        "edge_map_keys": None, "edge_map_values": None,
        "physical_groups": mesh.physical_groups,
        "elem_order": mesh.element_order, "num_edges": 0,
        # ver2.1: H→E 換算で ÷ε_r するため要素ごと ε_r を保存（誘電体のみ）
        "eps_r_per_element": build_eps_r_per_element(mesh, material_table),
        # ver2.3: 誘電体損失 Q_diel 用の要素ごと tanδ（誘電体のみ）
        "tan_delta_per_element": build_tan_delta_per_element(
            mesh, material_table),
    }
    return mesh_dict, {0: rec}, material_table


def _solve_hom(args, phase_list):
    from ..fem_hom.solver import solve_hom_standing, solve_hom_traveling
    from ..shared.mesh_loader import load_mesh_hom

    material_table = _load_material_table(
        args.mesh_file, getattr(args, "materials_file", None))
    target_freq = _target_freq(args)

    is_traveling = not (len(phase_list) == 1 and phase_list[0] == 0.0)
    results_by_n = {}
    mesh_dict = None

    from ..shared.material_resolver import (build_eps_r_per_element,
                                            build_tan_delta_per_element)
    for n in args.az_order:
        mesh = load_mesh_hom(args.mesh_file, n, args.elem_order)
        if mesh_dict is None:
            keys, values = _edge_map_arrays(mesh.edge_index_map)
            mesh_dict = {
                "vertices": mesh.vertices, "simplices": mesh.simplices,
                "edge_map_keys": keys, "edge_map_values": values,
                "physical_groups": mesh.physical_groups,
                "elem_order": mesh.element_order, "num_edges": mesh.num_edges,
                # ver2.1: 情報として保存（HOM 場再構成での利用は後続）
                "eps_r_per_element": build_eps_r_per_element(
                    mesh, material_table),
                # ver2.3: 誘電体損失 Q_diel 用の要素ごと tanδ（誘電体のみ）
                "tan_delta_per_element": build_tan_delta_per_element(
                    mesh, material_table),
            }
        if is_traveling:
            res = solve_hom_traveling(mesh, args.num_modes, phase_list,
                                      material_table=material_table,
                                      target_freq_ghz=target_freq)
            rec = {"traveling": {
                ph: _modeset_dict(ms.frequencies, ms.eigenvalues, ms.eigenvectors)
                for ph, ms in res.periodic.items()}}
        else:
            res = solve_hom_standing(mesh, args.num_modes,
                                     material_table=material_table,
                                     target_freq_ghz=target_freq)
            rec = {"standing": _modeset_dict(
                res.normal.frequencies, res.normal.eigenvalues,
                res.normal.eigenvectors)}
        results_by_n[n] = rec

    return mesh_dict, results_by_n, material_table


def run(args) -> int:
    """solve サブコマンドの本体."""
    if not os.path.exists(args.mesh_file):
        print(f"エラー: メッシュファイルが見つかりません: {args.mesh_file}")
        return 1

    phase_list = parse_phase_list(args.phase)
    solver_type = args.type
    output = args.output_file or _default_output(args.mesh_file, solver_type)

    target_freq = _target_freq(args)
    print(f"[solve] type={solver_type} mesh={args.mesh_file} "
          f"elem-order={args.elem_order} num-modes={args.num_modes} "
          f"phase={phase_list} "
          f"target-freq={'auto' if target_freq is None else f'{target_freq} GHz'}")

    if solver_type == "tm0":
        mesh_dict, results_by_n, material_table = _solve_tm0(args, phase_list)
    else:
        mesh_dict, results_by_n, material_table = _solve_hom(args, phase_list)

    parameters = {
        "solver_type": solver_type, "elem_order": args.elem_order,
        "num_modes": args.num_modes, "phase": args.phase,
        "mesh_file": os.path.basename(args.mesh_file),
    }
    # ver2.3: 指定したときのみ記録（未指定の H5 は従来と同じ属性構成のまま）
    if target_freq is not None:
        parameters["target_freq_GHz"] = target_freq
    write_results(output, solver_type=solver_type, mesh=mesh_dict,
                  results_by_n=results_by_n, parameters=parameters,
                  materials=material_table)

    # 計算結果の簡易テキスト出力 (.h5 → .txt)
    from .summary_txt import write_solve_summary
    txt = write_solve_summary(output, parameters, results_by_n)
    print(f"[solve] テキストサマリを保存: {txt}")

    # 周波数サマリー
    for n, rec in results_by_n.items():
        if "standing" in rec:
            freqs = rec["standing"]["frequencies"]
            print(f"  n={n} standing: " +
                  ", ".join(f"{x:.6f}" for x in freqs) + " GHz")
        if "traveling" in rec:
            for ph, ms in rec["traveling"].items():
                print(f"  n={n} phase={ph}: " +
                      ", ".join(f"{x:.6f}" for x in ms["frequencies"]) + " GHz")
    print(f"[solve] 結果を保存: {output}")
    return 0
