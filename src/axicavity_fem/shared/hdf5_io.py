"""統一 HDF5 スキーマ (schema_version=2.0) の Read/Write + v1 legacy reader.

TM0 と HOM の出力を同一階層構造で扱う（詳細は docs/HDF5_SCHEMA.md）。
固有ベクトルは **フラットな 1 配列** で格納し、TM0（節点 H_phi）と HOM（エッジ+節点）を
統一する。HOM の場再構成・ポストプロセスは読み戻したフラット配列を
``fem_hom.field_recon.split_hom_dofs`` で分割する。

```
/                      attrs: schema_version, solver_type, axicavity_fem_version
├── mesh/              vertices, simplices, [edge_map_keys/values], physical_groups/<name>
│                      [eps_r_per_element]  (Ne,)  ver2.1: 誘電体メッシュのみ
│                      [tan_delta_per_element] (Ne,) ver2.3: 損失誘電体のみ
├── materials/<tag>/   attrs: eps_r, mu_r, tan_delta   ver2.1/2.3: 多領域/誘電体のみ
├── parameters/        attrs: CLI 引数 + 物理規約
└── results/n{N}/
    ├── standing/mode_{K}/      eigenvector            attrs: eigenvalue_k2, frequency_GHz, ...
    └── traveling/phase_{xxxxx}/mode_{K}/  eigenvector_re, eigenvector_im
/post_process/n{N}/...          post コマンドで追記（mode_{K} の attrs にパラメータ）
```

ver2.1 拡張（schema_version=2.1）: `/mesh/eps_r_per_element`（要素ごと比誘電率）と
`/materials/<tag>`（領域別 eps_r/mu_r）を追加。誘電体メッシュ（material_table 指定）
のときのみ書き出し、無い場合は省略する。読込時に無ければ ``None``（真空扱い）で
v2.0 ファイルとも後方互換。

ver2.3 拡張（schema_version=2.2）: `/mesh/tan_delta_per_element`（要素ごと誘電正接
tanδ）と `/materials/<tag>` の ``tan_delta`` 属性を追加。損失誘電体のときのみ
書き出し、読込時に無ければ ``None``/0.0（無損失扱い）で旧ファイルと後方互換。
"""

from __future__ import annotations

import h5py
import numpy as np

from .. import __version__
from .constants import C0, k2_from_freq_ghz

SCHEMA_VERSION = "2.2"


def _phase_key(theta_deg: float) -> str:
    """θ[度] → ``phase_{θ×10 を 5 桁ゼロ埋め}``（120.0° → 'phase_01200'）."""
    return f"phase_{int(round(theta_deg * 10)):05d}"


def _k2_from_freq_ghz(freq_ghz: float) -> float:
    return k2_from_freq_ghz(freq_ghz)


# ===========================================================================
# 書き出し
# ===========================================================================
def write_results(path, *, solver_type, mesh, results_by_n, parameters=None,
                  materials=None):
    """solve 結果を v2 スキーマで書き出す.

    Args:
        path: 出力 .h5 パス。
        solver_type: "tm0" または "hom"。
        mesh: dict（vertices, simplices, edge_map_keys, edge_map_values,
            physical_groups, elem_order, num_edges）。edge_map は None 可。
            ver2.1: ``eps_r_per_element`` (Ne,) を含めると `/mesh` に書き出す。
        results_by_n: ``{n: rec}``。rec は
            ``{"standing": modeset}`` または ``{"traveling": {theta_deg: modeset}}``。
            modeset は ``{"frequencies", "eigenvalues"(None可), "eigenvectors"}``。
        parameters: 任意のスカラ/文字列 dict（CLI 引数等）。
        materials: ``{material_tag: {"eps_r": float, "mu_r": float}}``（任意）。
            指定時 `/materials/<tag>` に attrs として書き出す。
    """
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["solver_type"] = solver_type
        f.attrs["axicavity_fem_version"] = __version__

        mg = f.create_group("mesh")
        mg.create_dataset("vertices", data=np.asarray(mesh["vertices"]))
        mg.create_dataset("simplices", data=np.asarray(mesh["simplices"]))
        if mesh.get("edge_map_keys") is not None:
            mg.create_dataset("edge_map_keys",
                              data=np.asarray(mesh["edge_map_keys"]))
            mg.create_dataset("edge_map_values",
                              data=np.asarray(mesh["edge_map_values"]))
        pgrp = mg.create_group("physical_groups")
        for name, idx in (mesh.get("physical_groups") or {}).items():
            pgrp.create_dataset(name, data=np.asarray(idx))
        # ver2.1: 要素ごと比誘電率（誘電体メッシュのみ）
        if mesh.get("eps_r_per_element") is not None:
            mg.create_dataset("eps_r_per_element",
                              data=np.asarray(mesh["eps_r_per_element"],
                                              dtype=float))
        # ver2.3: 要素ごと誘電正接 tanδ（損失誘電体のみ）
        if mesh.get("tan_delta_per_element") is not None:
            mg.create_dataset("tan_delta_per_element",
                              data=np.asarray(mesh["tan_delta_per_element"],
                                              dtype=float))
        mg.attrs["num_nodes"] = len(mesh["vertices"])
        mg.attrs["num_edges"] = int(mesh.get("num_edges", 0))
        mg.attrs["elem_order"] = int(mesh["elem_order"])

        # ver2.1: 領域別材料テーブル（任意）
        if materials:
            mtg = f.create_group("materials")
            for tag, props in materials.items():
                g = mtg.create_group(str(tag))
                g.attrs["eps_r"] = float(props.get("eps_r", 1.0))
                g.attrs["mu_r"] = float(props.get("mu_r", 1.0))
                g.attrs["tan_delta"] = float(props.get("tan_delta", 0.0))

        pg = f.create_group("parameters")
        for key, val in (parameters or {}).items():
            pg.attrs[key] = val

        rg = f.create_group("results")
        for n, rec in results_by_n.items():
            ng = rg.create_group(f"n{n}")
            if "standing" in rec:
                sg = ng.create_group("standing")
                _write_modeset(sg, rec["standing"], traveling=False)
            if "traveling" in rec:
                tg = ng.create_group("traveling")
                for theta_deg, modeset in rec["traveling"].items():
                    pgg = tg.create_group(_phase_key(theta_deg))
                    pgg.attrs["theta_deg"] = float(theta_deg)
                    pgg.attrs["theta_rad"] = float(np.deg2rad(theta_deg))
                    _write_modeset(pgg, modeset, traveling=True,
                                   theta_deg=theta_deg)


def _write_modeset(parent, modeset, *, traveling, theta_deg=0.0):
    freqs = np.asarray(modeset["frequencies"])
    evals = modeset.get("eigenvalues")
    evecs = np.asarray(modeset["eigenvectors"])
    for k in range(len(freqs)):
        mg = parent.create_group(f"mode_{k}")
        if traveling:
            mg.create_dataset("eigenvector_re", data=np.real(evecs[k]))
            mg.create_dataset("eigenvector_im", data=np.imag(evecs[k]))
            mg.attrs["mode_type"] = "traveling"
            mg.attrs["theta_deg"] = float(theta_deg)
            mg.attrs["theta_rad"] = float(np.deg2rad(theta_deg))
        else:
            mg.create_dataset("eigenvector", data=np.real(evecs[k]))
            mg.attrs["mode_type"] = "standing"
        k2 = (float(evals[k]) if evals is not None
              else _k2_from_freq_ghz(float(freqs[k])))
        mg.attrs["eigenvalue_k2"] = k2
        mg.attrs["frequency_GHz"] = float(freqs[k])
        mg.attrs["frequency_Hz"] = float(freqs[k]) * 1e9


# ===========================================================================
# 読み込み
# ===========================================================================
def read_results(path) -> dict:
    """v2 スキーマの .h5 を in-memory dict に読み込む.

    Returns:
        ``{schema_version, solver_type, mesh, parameters, results_by_n,
        post_process}``。``results_by_n`` の modeset は
        ``{frequencies, eigenvalues, eigenvectors}`` で、進行波は複素 eigenvectors。
    """
    with h5py.File(path, "r") as f:
        out = {
            "schema_version": f.attrs.get("schema_version"),
            "solver_type": f.attrs.get("solver_type"),
            "mesh": _read_mesh(f["mesh"]),
            "parameters": dict(f["parameters"].attrs) if "parameters" in f else {},
            "materials": _read_materials(f),
            "results_by_n": {},
            "post_process": None,
        }
        if "results" in f:
            for nkey in f["results"]:
                n = int(nkey[1:])
                out["results_by_n"][n] = _read_n_group(f["results"][nkey])
        if "post_process" in f:
            out["post_process"] = _read_post(f["post_process"])
    return out


def _read_materials(f) -> dict:
    """`/materials/<tag>` を ``{tag: {eps_r, mu_r, tan_delta}}`` で読む（無ければ空 dict）."""
    out = {}
    if "materials" in f:
        for tag in f["materials"]:
            a = f["materials"][tag].attrs
            out[tag] = {"eps_r": float(a.get("eps_r", 1.0)),
                        "mu_r": float(a.get("mu_r", 1.0)),
                        "tan_delta": float(a.get("tan_delta", 0.0))}
    return out


def _read_mesh(mg) -> dict:
    mesh = {
        "vertices": mg["vertices"][:],
        "simplices": mg["simplices"][:],
        "elem_order": int(mg.attrs.get("elem_order", 1)),
        "num_edges": int(mg.attrs.get("num_edges", 0)),
        "edge_map_keys": mg["edge_map_keys"][:] if "edge_map_keys" in mg else None,
        "edge_map_values": (mg["edge_map_values"][:]
                            if "edge_map_values" in mg else None),
        # ver2.1: 要素ごと比誘電率（無ければ None=真空）
        "eps_r_per_element": (mg["eps_r_per_element"][:]
                              if "eps_r_per_element" in mg else None),
        # ver2.3: 要素ごと誘電正接 tanδ（無ければ None=無損失）
        "tan_delta_per_element": (mg["tan_delta_per_element"][:]
                                  if "tan_delta_per_element" in mg else None),
        "physical_groups": {},
    }
    if "physical_groups" in mg:
        for name in mg["physical_groups"]:
            mesh["physical_groups"][name] = mg["physical_groups"][name][:]
    if mesh["edge_map_keys"] is not None:
        mesh["edge_index_map"] = {
            tuple(int(x) for x in mesh["edge_map_keys"][i]):
                int(mesh["edge_map_values"][i])
            for i in range(len(mesh["edge_map_values"]))
        }
    else:
        mesh["edge_index_map"] = None
    return mesh


def _read_modeset(group, traveling):
    mode_keys = sorted((k for k in group if k.startswith("mode_")),
                       key=lambda s: int(s.split("_")[1]))
    freqs, evals, evecs = [], [], []
    for mk in mode_keys:
        mg = group[mk]
        freqs.append(float(mg.attrs["frequency_GHz"]))
        evals.append(float(mg.attrs["eigenvalue_k2"]))
        if traveling:
            evecs.append(mg["eigenvector_re"][:] + 1j * mg["eigenvector_im"][:])
        else:
            evecs.append(mg["eigenvector"][:])
    return {"frequencies": np.array(freqs), "eigenvalues": np.array(evals),
            "eigenvectors": np.array(evecs) if evecs else np.empty((0,))}


def _read_n_group(ng) -> dict:
    rec = {}
    if "standing" in ng:
        rec["standing"] = _read_modeset(ng["standing"], traveling=False)
    if "traveling" in ng:
        trav = {}
        for pkey in ng["traveling"]:
            theta_deg = float(ng["traveling"][pkey].attrs["theta_deg"])
            trav[theta_deg] = _read_modeset(ng["traveling"][pkey],
                                            traveling=True)
        rec["traveling"] = trav
    return rec


def _read_post(pp) -> dict:
    out = {}
    for nkey in pp:
        n = int(nkey[1:])
        out[n] = {}
        ng = pp[nkey]
        if "standing" in ng:
            out[n]["standing"] = _read_post_modes(ng["standing"])
        if "traveling" in ng:
            out[n]["traveling"] = {
                float(ng["traveling"][pk].attrs["theta_deg"]):
                    _read_post_modes(ng["traveling"][pk])
                for pk in ng["traveling"]}
    return out


def _read_post_modes(group):
    mode_keys = sorted((k for k in group if k.startswith("mode_")),
                       key=lambda s: int(s.split("_")[1]))
    return [dict(group[mk].attrs) for mk in mode_keys]


# ===========================================================================
# post_process 追記
# ===========================================================================
def append_post_process(path, post_by_n):
    """post 結果を ``/post_process`` グループへ追記する.

    Args:
        path: 既存の v2 .h5 パス。
        post_by_n: ``{n: rec}``。rec は ``{"standing": [params, ...]}`` または
            ``{"traveling": {theta_deg: [params, ...]}}``。params は
            attrs にする dict（Q, R_over_Q, ...）。
    """
    with h5py.File(path, "a") as f:
        if "post_process" in f:
            del f["post_process"]
        pp = f.create_group("post_process")
        for n, rec in post_by_n.items():
            ng = pp.create_group(f"n{n}")
            if "standing" in rec:
                _write_post_modes(ng.create_group("standing"), rec["standing"])
            if "traveling" in rec:
                tg = ng.create_group("traveling")
                for theta_deg, params_list in rec["traveling"].items():
                    g = tg.create_group(_phase_key(theta_deg))
                    g.attrs["theta_deg"] = float(theta_deg)
                    _write_post_modes(g, params_list)


def _write_post_modes(parent, params_list):
    for k, params in enumerate(params_list):
        mg = parent.create_group(f"mode_{k}")
        for key, val in params.items():
            mg.attrs[key] = val


# ===========================================================================
# info ダンプ
# ===========================================================================
def dump_info(path) -> str:
    """.h5 の構造（グループ・データセット形状・主要 attrs）を文字列で返す."""
    lines = [f"File: {path}"]

    def visit(name, obj):
        indent = "  " * name.count("/")
        base = name.split("/")[-1]
        if isinstance(obj, h5py.Dataset):
            lines.append(f"{indent}{base}  dataset {obj.shape} {obj.dtype}")
        else:
            lines.append(f"{indent}{base}/")
        for ak, av in obj.attrs.items():
            lines.append(f"{indent}  @{ak} = {av}")

    with h5py.File(path, "r") as f:
        for ak, av in f.attrs.items():
            lines.append(f"@{ak} = {av}")
        f.visititems(visit)
    return "\n".join(lines)


# ===========================================================================
# v1 (ver1) レガシー読み込み
# ===========================================================================
def is_v2(path) -> bool:
    """ファイルが v2 スキーマ（schema_version 属性あり）かを返す."""
    with h5py.File(path, "r") as f:
        return "schema_version" in f.attrs


def load_v1_legacy(path) -> dict:
    """ver1 形式（TM0 フラット / HOM ネスト）を v2 in-memory 構造へ変換する.

    ``schema_version`` 属性の有無で v2 と区別する。TM0 は ``mesh/nodes``、
    HOM は ``mesh/vertices`` の有無で判別する。

    Returns:
        :func:`read_results` と同形式の dict。

    Raises:
        ValueError: 形式を判別できないとき。
    """
    with h5py.File(path, "r") as f:
        if "schema_version" in f.attrs:
            raise ValueError("これは v2 ファイルです。read_results を使ってください。")
        if "mesh" in f and "nodes" in f["mesh"]:
            return _load_v1_tm0(f)
        if "mesh" in f and "vertices" in f["mesh"]:
            return _load_v1_hom(f)
    raise ValueError("ver1 形式を判別できませんでした。")


def _load_v1_tm0(f) -> dict:
    nodes = f["mesh/nodes"][:]
    elements = f["mesh/elements"][:]
    order = int(f["mesh"].attrs.get("order", 1))
    pgroups = {}
    if "mesh/physical_groups" in f:
        for name in f["mesh/physical_groups"]:
            pgroups[name] = f[f"mesh/physical_groups/{name}"][:]

    results_by_n = {0: {}}
    if "results/frequencies" in f:   # 定在波
        results_by_n[0]["standing"] = {
            "frequencies": f["results/frequencies"][:],
            "eigenvalues": None,
            "eigenvectors": f["results/eigenvectors"][:],
        }
    elif "phase_shifts" in f.attrs:   # 進行波
        trav = {}
        for ph in f.attrs["phase_shifts"]:
            grp = f[f"results/phase_{ph}"]
            trav[float(ph)] = {
                "frequencies": grp["frequencies"][:],
                "eigenvalues": None,
                "eigenvectors": grp["eigenvectors"][:],
            }
        results_by_n[0]["traveling"] = trav

    return {
        "schema_version": None, "solver_type": "tm0",
        "mesh": {"vertices": nodes, "simplices": elements,
                 "elem_order": order, "num_edges": 0,
                 "edge_map_keys": None, "edge_map_values": None,
                 "edge_index_map": None, "physical_groups": pgroups},
        "parameters": dict(f.attrs), "results_by_n": results_by_n,
        "post_process": None,
    }


def _load_v1_hom(f) -> dict:
    """ver1 HOM ネスト型を v2 構造へ変換する（フラット固有ベクトルを再合成）."""
    vertices = f["mesh/vertices"][:]
    simplices = f["mesh/simplices"][:]
    keys = f["mesh/edge_map_keys"][:]
    vals = f["mesh/edge_map_values"][:]
    num_edges = int(f["mesh"].attrs.get("num_edges", len(vals)))
    num_elements = len(simplices)
    num_nodes = len(vertices)

    def recombine(grp, n, complex_mode):
        order = int(grp.attrs.get("elem_order", 1))

        def get(name):
            if complex_mode:
                return grp[f"{name}_re"][:] + 1j * grp[f"{name}_im"][:]
            return grp[name][:]

        if order == 2:
            N = 2 * num_edges + 2 * num_elements + (num_nodes if n > 0 else 0)
            vec = np.zeros(N, dtype=complex if complex_mode else float)
            ct = get("edge_vectors")
            lt = get("edge_vectors_lt")
            face = get("face_vectors")
            vec[0:2 * num_edges:2] = ct
            vec[1:2 * num_edges:2] = lt
            vec[2 * num_edges:2 * num_edges + 2 * num_elements] = face
            if n > 0:
                node_off = 2 * num_edges + 2 * num_elements
                E_theta = get("E_theta")
                vec[node_off:node_off + num_nodes] = E_theta * vertices[:, 1]
        else:
            N = num_edges + (num_nodes if n > 0 else 0)
            vec = np.zeros(N, dtype=complex if complex_mode else float)
            vec[:num_edges] = get("edge_vectors")
            if n > 0:
                E_theta = get("E_theta")
                vec[num_edges:num_edges + num_nodes] = E_theta * vertices[:, 1]
        return vec

    results_by_n = {}
    for nkey in f["results"]:
        n = int(nkey[1:])
        base = f[f"results/{nkey}"]
        rec = {}
        if "Normal" in base:
            modeset = _v1_hom_modeset(base["Normal"], n, False, recombine)
            rec["standing"] = modeset
        if "Periodic" in base:
            trav = {}
            for pkey in base["Periodic"]:
                grp = base["Periodic"][pkey]
                theta = float(grp.attrs.get("theta_deg", 0.0)) \
                    if "theta_deg" in grp.attrs else _phase_from_key(pkey)
                trav[theta] = _v1_hom_modeset(grp, n, True, recombine)
            rec["traveling"] = trav
        results_by_n[n] = rec

    return {
        "schema_version": None, "solver_type": "hom",
        "mesh": {"vertices": vertices, "simplices": simplices,
                 "elem_order": int(f["mesh"].attrs.get("order", 1))
                 if "order" in f["mesh"].attrs else _infer_order(simplices),
                 "num_edges": num_edges,
                 "edge_map_keys": keys, "edge_map_values": vals,
                 "edge_index_map": {tuple(int(x) for x in keys[i]): int(vals[i])
                                    for i in range(len(vals))},
                 "physical_groups": {}},
        "parameters": dict(f.attrs), "results_by_n": results_by_n,
        "post_process": None,
    }


def _v1_hom_modeset(group, n, complex_mode, recombine):
    mode_keys = sorted((k for k in group if k.startswith("mode_")),
                       key=lambda s: int(s.split("_")[1]))
    freqs, evals, evecs = [], [], []
    for mk in mode_keys:
        grp = group[mk]
        freqs.append(float(grp.attrs.get("frequency_GHz", 0.0)))
        evals.append(float(grp.attrs.get("eigenvalue_k2", 0.0)))
        evecs.append(recombine(grp, n, complex_mode))
    return {"frequencies": np.array(freqs), "eigenvalues": np.array(evals),
            "eigenvectors": np.array(evecs) if evecs else np.empty((0,))}


def _phase_from_key(pkey: str) -> float:
    return float(pkey.replace("PB_Phase_", "").replace("_deg", "")
                 .replace("_", "."))


def _infer_order(simplices) -> int:
    return 2 if simplices.shape[1] >= 6 else 1
