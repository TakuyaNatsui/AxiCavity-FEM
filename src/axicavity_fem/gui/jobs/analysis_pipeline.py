"""解析ジョブの本体（子プロセスで動く。ver2.3 の CLI をそのまま呼ぶ）.

``kind = analysis``: （メッシュが無ければ生成 →）solve → post（``runPost`` のとき）→ 結果の要約
``kind = post``:     既存の生の結果に post
``kind = report``:   HTML レポート（``cmd_report``）
``kind = export``:   場の書き出し（``cmd_export``。argv のパスは結果フォルダからの相対か絶対）

``job.json``::

    {"kind": "tm0-sw", "meshOrder": 2, "argv": {"solve": [...], "post": [...] | null, "report": [...]},
     "files": {"mesh": "model.msh", "raw": "model_SW_TM0.h5", "processed": "model_SW_TM0_processed.h5"}}

argv は ``cli.main.build_parser().parse_args`` に通して ``cmd_solve.run`` / ``cmd_post.run`` / ``cmd_report.run`` を呼ぶ
（CLI と同じ経路・引数検証。パスはジョブフォルダからの相対で、子はそこを cwd にする）。段階の切れ目で cancel.flag を見る
（固有値計算の途中では止まらない）。
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ...shared.multi_region_model import MultiRegionGeometry
from .commands import PROGRAM
from .pipeline import (
    GEOMETRY_FILE,
    CancelFn,
    JobCancelled,
    LogFn,
    ProgressFn,
    capture_output,
    load_job,
    write_materials_json,
)

POST_KEYS = ("Q", "Q_wall", "Q_diel", "R_over_Q", "V_eff", "V_acc", "U_stored", "P_loss", "P_diel", "P_flow_zmin",
             "group_velocity", "v_phase_c", "attenuation")


def _clean(value):
    """JSON に書ける値に（numpy → Python、非有限は None）."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    return value


def summarize_results(h5_path: str | Path) -> dict:
    """結果 HDF5（solve または post 済み）の要約: モードごとの f と post のパラメータ（result.json の summary）."""
    from ...shared.hdf5_io import read_results

    data = read_results(str(h5_path))
    post = data.get("post_process") or {}
    modes: list[dict] = []
    phases: set[float] = set()
    traveling = False

    def post_modes(n: int, phase: Optional[float]) -> list[dict]:
        rec = post.get(n) or {}
        if phase is None:
            return rec.get("standing") or []
        trav = rec.get("traveling") or {}
        for key, value in trav.items():
            if abs(float(key) - phase) < 1e-9:
                return value
        return []

    for n in sorted(data["results_by_n"]):
        rec = data["results_by_n"][n]
        if "standing" in rec:
            pm = post_modes(n, None)
            for k, f in enumerate(rec["standing"]["frequencies"]):
                m = {"n": int(n), "phase": None, "index": k, "f_GHz": float(f)}
                if k < len(pm):
                    m.update({key: _clean(pm[k][key]) for key in POST_KEYS if key in pm[k]})
                modes.append(m)
        if "traveling" in rec:
            traveling = True
            for phase in sorted(rec["traveling"]):
                phases.add(float(phase))
                pm = post_modes(n, float(phase))
                for k, f in enumerate(rec["traveling"][phase]["frequencies"]):
                    m = {"n": int(n), "phase": float(phase), "index": k, "f_GHz": float(f)}
                    if k < len(pm):
                        m.update({key: _clean(pm[k][key]) for key in POST_KEYS if key in pm[k]})
                    modes.append(m)
    params = {k: _clean(v) for k, v in (data.get("parameters") or {}).items()}
    return {"solverType": str(data.get("solver_type") or params.get("solver_type") or ""),
            "wave": "traveling" if traveling else "standing",
            "nOrders": sorted(int(n) for n in data["results_by_n"]), "phases": sorted(phases),
            "hasPost": bool(post), "numModes": len({(m["n"], m["phase"]) for m in modes}) and
            max(m["index"] for m in modes) + 1 if modes else 0,
            "elemOrder": params.get("elem_order"), "meshFile": params.get("mesh_file"),
            "materials": {k: {kk: _clean(vv) for kk, vv in v.items()} for k, v in (data.get("materials") or {}).items()},
            "modes": modes}


def _run_cli(argv: list[str], log: LogFn) -> None:
    """ver2.3 の CLI をこのプロセスで実行する（終了コードが 0 でなければ例外）."""
    from ...cli.main import build_parser

    args = build_parser().parse_args(argv)
    if args.command == "solve":
        from ...cli import cmd_solve as cmd
    elif args.command == "post":
        from ...cli import cmd_post as cmd
    elif args.command == "report":
        from ...cli import cmd_report as cmd
    elif args.command == "export":
        from ...cli import cmd_export as cmd
    else:
        raise ValueError(f"対応していないコマンド: {args.command}")
    log(f"$ {PROGRAM} " + " ".join(argv))
    rc = cmd.run(args)
    if rc:
        raise RuntimeError(f"{args.command} が失敗しました（終了コード {rc}）")


def _ensure_mesh(job: dict, log: LogFn) -> str:
    """model.msh が無ければ geometry.json から作る（材料 JSON も）. メッシュファイル名を返す."""
    from .meshing import generate_msh

    mesh_file = (job.get("files") or {}).get("mesh") or "model.msh"
    if Path(mesh_file).exists():
        source = job.get("meshReused")
        log(f"メッシュを再利用: {source or mesh_file}（メッシュタブで生成した同じ設定のメッシュ）"
            if source else f"メッシュ: {mesh_file}")
        return mesh_file
    geom = MultiRegionGeometry.load_json(GEOMETRY_FILE)
    order = int(job.get("meshOrder", 2))
    log(f"メッシュ生成中... (lc = {geom.mesh_size:g} {geom.unit}, {order} 次, 領域 {len(geom.regions)})")
    generate_msh(geom, mesh_file, mesh_order=order, verbose=1, log=log)   # EXE ではメッシャ（別プロセス）
    write_materials_json(geom, mesh_file)
    return mesh_file


def run_job(kind: str, job_dir: str | Path, progress: ProgressFn | None = None, log: LogFn | None = None,
            should_cancel: CancelFn | None = None) -> dict:
    """analysis / post / report を実行し、結果（files, summary, elapsedS）を返す."""
    progress = progress or (lambda stage, msg: None)
    log = log or (lambda msg: None)
    job_dir = Path(job_dir)
    job = load_job(job_dir)
    argv = job.get("argv") or {}
    files = dict(job.get("files") or {})
    warn_list: list[str] = []
    elapsed: dict[str, float] = {}

    def stage(name: str, msg: str) -> None:
        if should_cancel is not None and should_cancel():
            raise JobCancelled("キャンセルされました")
        log(msg)
        progress(name, msg)

    def timed(name: str, fn) -> None:
        t0 = time.perf_counter()
        fn()
        elapsed[name] = round(time.perf_counter() - t0, 3)

    previous_cwd = os.getcwd()
    os.chdir(job_dir)
    try:
        with capture_output(log, warn_list):
            if kind == "analysis":
                stage("meshing", "メッシュを準備しています...")
                timed("meshing", lambda: _ensure_mesh(job, log))
                stage("solve", "固有値計算中...（この段階はキャンセルできません）")
                timed("solve", lambda: _run_cli(list(argv["solve"]), log))
                if argv.get("post"):
                    stage("post", "パラメータ計算中...")
                    timed("post", lambda: _run_cli(list(argv["post"]), log))
                else:
                    files["processed"] = None
            elif kind == "post":
                stage("post", "パラメータ計算中...")
                timed("post", lambda: _run_cli(list(argv["post"]), log))
                files["processed"] = argv["post"][argv["post"].index("-o") + 1]
            elif kind == "report":
                stage("report", "レポート作成中...")
                timed("report", lambda: _run_cli(list(argv["report"]), log))
                out_dir = argv["report"][argv["report"].index("-o") + 1]
                files["report"] = str(Path(out_dir) / "index.html")
            elif kind == "export":
                stage("export", "場を書き出し中...")
                timed("export", lambda: _run_cli(list(argv["export"]), log))
                files["export"] = argv["export"][argv["export"].index("-o") + 1]
            else:
                raise ValueError(f"不明なジョブの種類: {kind}")
            summary: dict = {}
            if kind != "export":
                stage("summary", "結果を読み込み中...")
                source = files.get("processed") if files.get("processed") and Path(files["processed"]).exists() \
                    else files.get("raw")
                summary = summarize_results(source) if source and Path(source).exists() else {}
            for key in ("raw", "processed"):
                name = files.get(key)
                if name and Path(name).exists():
                    txt = str(Path(name).with_suffix(".txt"))
                    files[f"{key}Txt"] = txt if Path(txt).exists() else None
    finally:
        os.chdir(previous_cwd)
    result = {"kind": kind, "files": files, "summary": summary, "warnings": warn_list, "elapsedS": elapsed}
    modes = summary.get("modes") or []
    if kind == "export":
        stage("done", f"場の書き出し完了: {files['export']}")
    elif modes:
        first = modes[0]
        head = f"f = {first['f_GHz']:.6f} GHz" + (f", Q = {first['Q']:.4g}" if first.get("Q") else "")
        stage("done", f"{'解析' if kind == 'analysis' else kind} 完了: {len(modes)} モード（{head}）")
    else:
        stage("done", f"{kind} 完了")
    return result
