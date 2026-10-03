"""ジョブの本体（子プロセスで動く。純 Python + ver2.3 のコア。Qt 非依存）.

メッシュジョブ（``kind = mesh`` / ``export_msh``）::

    入力  job_dir/geometry.json   MultiRegionGeometry.to_dict()（GUI が core.convert で作る）
          job_dir/job.json        {"kind": "mesh", "meshOrder": 1|2, "outPath": "<任意: ユーザー指定の .msh>"}
    出力  model.msh + model.materials.json（ver2.3 の wx GUI と同じサイドカー）、mesh_preview.npz（スケッチ画面に
          重ねる三角形・境界条件ごとの辺・誘電体界面）、mesh_summary.json（統計）、log.txt（gmsh の出力）

ver2.3 の関数をそのまま呼ぶ: ``export_msh_multi_region``（:mod:`.meshing` 経由。EXE ではメッシャの別プロセス）→ ``load_mesh`` → ``bc_boundary_segments`` /
``material_interface_geometry`` / ``mean_edge_length``。解析ジョブ（solve → post）は M7 で足す。
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ...shared.multi_region_model import MultiRegionGeometry
from ..core.convert import UNIT_TO_M

GEOMETRY_FILE = "geometry.json"
JOB_FILE = "job.json"
MESH_FILE = "model.msh"
MATERIALS_FILE = "model.materials.json"
MESH_SUMMARY_FILE = "mesh_summary.json"
MESH_PREVIEW_FILE = "mesh_preview.npz"
MATERIALS_SCHEMA = "axicavity-fem-v21.materials/1"
BC_NAMES = ("PEC", "E-short", "M-short", "None")

class JobCancelled(Exception):
    """キャンセルされた（段階の切れ目で ``should_cancel`` が真）.

    runner ではなくここに置く: runner は ``python -m`` で ``__main__`` として動くため、runner 側に置くと
    ``__main__.JobCancelled`` と ``…jobs.runner.JobCancelled`` の 2 つになって except が効かない。
    """


LogFn = Callable[[str], None]
ProgressFn = Callable[[str, str], None]
CancelFn = Callable[[], bool]


class _LogStream(io.TextIOBase):
    """print 出力を行単位でログコールバックへ流す."""

    def __init__(self, log: LogFn):
        super().__init__()
        self._log = log
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log(line.rstrip())
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._log(self._buf.rstrip())
        self._buf = ""


@contextlib.contextmanager
def capture_output(log: LogFn, warn_list: list[str]):
    """コアの print と warnings をログへ流す（他スレッドの警告は元の出力へ）."""
    owner = threading.get_ident()
    original = warnings.showwarning

    def on_warning(message, category, filename, lineno, file=None, line=None):
        if threading.get_ident() != owner:
            original(message, category, filename, lineno, file, line)
            return
        text = str(message)
        warn_list.append(text)
        log(f"警告: {text}")

    with contextlib.redirect_stdout(_LogStream(log)), warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = on_warning
        yield


def load_job(job_dir: str | Path) -> dict:
    path = Path(job_dir) / JOB_FILE
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def materials_sidecar_path(msh_path: str | Path) -> Path:
    """``model.msh`` → ``model.materials.json``（ver2.3 の wx GUI と同じ）."""
    msh_path = Path(msh_path)
    return msh_path.with_name(msh_path.stem + ".materials.json")


def write_materials_json(geom: MultiRegionGeometry, msh_path: str | Path) -> Path:
    """材料のサイドカー JSON を書く（``axicavity-fem solve`` が自動で読む）."""
    from ...shared.material_resolver import build_material_table_from_geometry

    path = materials_sidecar_path(msh_path)
    payload = {"schema": MATERIALS_SCHEMA, "mesh_file": Path(msh_path).name,
               "materials": build_material_table_from_geometry(geom)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _stack_segments(segments: list, order: int) -> np.ndarray:
    """(P, 2) 配列のリスト → (K, P, 2)（空なら (0, P, 2)）."""
    points = 3 if order == 2 else 2
    if not segments:
        return np.zeros((0, points, 2), dtype=np.float64)
    return np.stack([np.asarray(s, dtype=np.float64) for s in segments])


def mesh_preview_data(mesh, geom: MultiRegionGeometry) -> tuple[dict, dict]:
    """スケッチ画面に重ねる配列（m 単位）と統計を作る.

    配列: ``nodes`` (N, 2)、``simplices`` (E, 3)、``elem_order``、``region_ids`` (E,)、
    ``bc_<名前>`` (K, P, 2)（P = 2 または 3。2 次要素は辺中点経由）、``interfaces`` (K, P, 2)。
    """
    from ...shared.boundary_groups import bc_boundary_segments, material_interface_geometry, mean_edge_length
    from ...shared.material_resolver import (
        build_eps_r_per_element,
        build_material_table_from_geometry,
        build_tan_delta_per_element,
    )

    order = int(mesh.element_order)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    simplices = np.asarray(mesh.simplices, dtype=np.int32)
    arrays: dict = {"nodes": nodes, "simplices": simplices, "elem_order": np.int32(order)}
    region_ids = mesh.element_region_ids
    arrays["region_ids"] = (np.asarray(region_ids, dtype=np.int32) if region_ids is not None
                            else np.full(len(simplices), -1, dtype=np.int32))
    bc_segments = bc_boundary_segments(mesh.elements, nodes, mesh.physical_groups, order)
    bc_counts: dict[str, int] = {}
    for name in BC_NAMES:
        segs = bc_segments.get(name, [])
        arrays[f"bc_{name}"] = _stack_segments(segs, order)
        bc_counts[name] = len(segs)
    table = build_material_table_from_geometry(geom)
    eps = build_eps_r_per_element(mesh, table)
    tan = build_tan_delta_per_element(mesh, table)
    interfaces, _ = material_interface_geometry(mesh.elements, nodes, order, eps, tan)
    arrays["interfaces"] = _stack_segments(interfaces, order)

    scale = UNIT_TO_M.get(geom.unit, 1.0)
    names = {r.material_tag: r.name for r in geom.regions}
    regions = []
    if region_ids is not None and mesh.region_names:
        counts = {int(tag): int((np.asarray(region_ids) == tag).sum()) for tag in mesh.region_names}
        for tag, material in mesh.region_names.items():
            regions.append({"tag": material, "name": names.get(material, material), "elements": counts[int(tag)]})
    lo, hi = nodes.min(axis=0) / scale, nodes.max(axis=0) / scale
    stats = {
        "nodes": int(mesh.num_nodes), "elements": int(mesh.num_elements), "order": order,
        "meanEdge": float(mean_edge_length(mesh.simplices, nodes) / scale), "units": geom.unit,
        "bc": bc_counts, "interfaces": int(len(interfaces)), "regions": regions,
        "bounds": [[float(lo[0]), float(lo[1])], [float(hi[0]), float(hi[1])]],
    }
    return arrays, stats


def run_mesh_job(job_dir: str | Path, progress: ProgressFn | None = None, log: LogFn | None = None,
                 should_cancel: CancelFn | None = None) -> dict:
    """メッシュを作り、材料 JSON・プレビュー・統計を書く。要約（camelCase）を返す.

    ``job.json`` に ``outPath`` があれば .msh とサイドカーをそのパスに書き、プレビューと統計はジョブフォルダに書く
    （ファイルタブの「メッシュ書き出し」）。
    """
    from ...shared.mesh_loader import load_mesh
    from .meshing import generate_msh

    progress = progress or (lambda stage, msg: None)
    log = log or (lambda msg: None)
    job_dir = Path(job_dir)
    job = load_job(job_dir)
    geom = MultiRegionGeometry.load_json(job_dir / GEOMETRY_FILE)
    order = int(job.get("meshOrder", 2))
    if order not in (1, 2):
        raise ValueError(f"meshOrder は 1 か 2（{order}）")
    out_path = job.get("outPath")
    msh_path = Path(out_path) if out_path else job_dir / MESH_FILE
    warn_list: list[str] = []

    def stage(name: str, msg: str) -> None:
        if should_cancel is not None and should_cancel():
            raise JobCancelled("キャンセルされました")
        log(msg)
        progress(name, msg)

    errs = geom.validate()
    if errs:
        raise ValueError("形状が不正です:\n  - " + "\n  - ".join(errs))
    if not geom.regions:
        raise ValueError("閉領域（Region）がありません。")

    t0 = time.perf_counter()
    with capture_output(log, warn_list):
        stage("meshing", f"メッシュ生成中... (lc = {geom.mesh_size:g} {geom.unit}, {order} 次, "
                         f"領域 {len(geom.regions)})")
        msh_path.parent.mkdir(parents=True, exist_ok=True)
        generate_msh(geom, msh_path, mesh_order=order, verbose=1, log=log)   # EXE ではメッシャ（別プロセス）
        materials_path = write_materials_json(geom, msh_path)
        stage("loading", "メッシュを読み込み中...")
        mesh = load_mesh(msh_path, element_order=order)
        arrays, stats = mesh_preview_data(mesh, geom)
    elapsed = time.perf_counter() - t0
    np.savez_compressed(job_dir / MESH_PREVIEW_FILE, **arrays)
    summary = {
        "kind": "mesh", "meshOrder": order, "units": geom.unit, "meshSize": float(geom.mesh_size),
        "stats": stats, "warnings": warn_list, "elapsedS": {"meshing": round(elapsed, 3)},
        "files": {"mesh": str(msh_path) if out_path else msh_path.name,
                  "materials": str(materials_path) if out_path else materials_path.name,
                  "preview": MESH_PREVIEW_FILE},
    }
    (job_dir / MESH_SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stage("done", f"メッシュ生成完了: 節点 {stats['nodes']}, 要素 {stats['elements']}（{order} 次）, "
                  f"平均辺長 {stats['meanEdge']:.3g} {geom.unit}（{elapsed:.1f} s）")
    return summary

