"""メッシュ生成（形状 → ``.msh``）の入口.

- 開発環境・pip: 今までどおり同じプロセスで gmsh（``shared.gmsh_export_occ.export_msh_multi_region``）。
- Windows 版（EXE）: 本体に gmsh（GPL）を入れないので、同梱の **メッシャ**（``mesher/``: 埋め込み Python + gmsh +
  ``axicavity_mesh.py``）を別プロセスで動かす。本体とは JSON（形状）と ``.msh``（結果）のファイルだけでやり取りする。

切り替え: 環境変数 ``AXICAVITY_MESHER``（``process`` / ``inprocess``）、無ければ EXE なら process。メッシャの場所:
``AXICAVITY_MESHER_DIR``（``python.exe`` と ``axicavity_mesh.py`` のあるフォルダ）→ EXE の隣の ``mesher/`` →
開発環境ではこの Python とリポジトリの ``mesher/axicavity_mesh.py``。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from .. import frozen

MESHER_SCRIPT = "axicavity_mesh.py"
LogFn = Callable[[str], None]


class MesherError(RuntimeError):
    """メッシャ（別プロセス）が失敗した."""


def use_mesher_process() -> bool:
    mode = os.environ.get("AXICAVITY_MESHER", "").strip().lower()
    if mode in ("process", "subprocess"):
        return True
    if mode in ("inprocess", "in-process", "inline"):
        return False
    return frozen.is_compiled()


def mesher_command() -> list[str]:
    """メッシャを起動するコマンド（見つからなければ :class:`MesherError`）."""
    folder = os.environ.get("AXICAVITY_MESHER_DIR")
    if folder or frozen.is_compiled():
        base = Path(folder) if folder else frozen.app_dir() / "mesher"
        exe = base / ("python.exe" if os.name == "nt" else "bin/python3")
        script = base / MESHER_SCRIPT
        if not exe.exists() or not script.exists():
            raise MesherError(f"メッシャが見つかりません: {base}（EXE と同じフォルダの mesher/ を消していないか確認してください）")
        return [str(exe), "-B", str(script)]
    script = frozen.app_dir() / "mesher" / MESHER_SCRIPT
    if not script.exists():
        raise MesherError(f"メッシャのスクリプトが見つかりません: {script}")
    return [sys.executable, "-B", str(script)]


def generate_msh(geom, msh_path: str | Path, mesh_order: int = 2, verbose: int = 1,
                 log: Optional[LogFn] = None) -> Path:
    """``geom``（MultiRegionGeometry）から ``msh_path`` を作る（EXE ではメッシャの別プロセスで）."""
    msh_path = Path(msh_path)
    if not use_mesher_process():
        from ...shared.gmsh_export_occ import export_msh_multi_region

        export_msh_multi_region(geom, msh_path, mesh_order=int(mesh_order), verbose=int(verbose))
        return msh_path
    return run_mesher(geom, msh_path, mesh_order, verbose, log)


def run_mesher(geom, msh_path: Path, mesh_order: int, verbose: int, log: Optional[LogFn]) -> Path:
    log = log or (lambda _m: None)
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_file = msh_path.with_name(msh_path.name + ".mesher-geometry.json")
    job_file = msh_path.with_name(msh_path.name + ".mesher-job.json")
    result_file = msh_path.with_name(msh_path.name + ".mesher.json")
    geom.save_json(geometry_file)
    job_file.write_text(json.dumps({"geometry": str(geometry_file), "out": str(msh_path),
                                    "meshOrder": int(mesh_order), "verbose": int(verbose)},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    result_file.unlink(missing_ok=True)
    cmd = mesher_command() + ["mesh", str(job_file)]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # GUI から起動してもコンソールを出さない
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env, creationflags=creation)
    except OSError as exc:
        raise MesherError(f"メッシャを起動できません: {exc}") from exc
    tie = tie_to_this_process(proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
        rc = proc.wait()
    finally:
        release_tie(tie)
    result = {}
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = {}
    for f in (geometry_file, job_file, result_file):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    if rc != 0 or not result.get("ok") or not msh_path.exists():
        err = result.get("error") or {}
        message = err.get("message") or f"メッシャが失敗しました（終了コード {rc}）"
        if err.get("type") == "ValueError":
            raise ValueError(message)
        raise MesherError(message + (f"\n{err['traceback']}" if err.get("traceback") else ""))
    log(f"メッシャ: {msh_path.name}（{time.perf_counter() - t0:.2f} s）")
    return msh_path


# ---- 親が終わったらメッシャも終わらせる（Windows の Job Object） --------------------------

def tie_to_this_process(proc: subprocess.Popen):
    """子プロセスを Job Object（JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE）に入れる.

    GUI の「強制停止」はジョブの子プロセス（このプロセス）を kill するので、孫のメッシャが残らないように、
    このプロセスが終わると OS がメッシャも終わらせるようにする。Windows 以外・失敗時は何もしない。
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):                       # noqa: N801
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC_LIMIT(ctypes.Structure):                       # noqa: N801
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED_LIMIT(ctypes.Structure):                    # noqa: N801
            _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000              # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return None
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:  # noqa: BLE001 — 失敗してもメッシュ生成は続ける
        return None


def release_tie(job) -> None:
    if job is None or os.name != "nt":
        return
    try:
        import ctypes

        ctypes.WinDLL("kernel32").CloseHandle(job)
    except Exception:  # noqa: BLE001
        pass


def mesher_version(timeout: float = 60.0) -> str:
    """メッシャの版（``axicavity_mesh.py version`` の出力）。使えなければ理由."""
    try:
        cmd = mesher_command() + ["version"]
    except MesherError as exc:
        return f"なし（{exc}）"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                             env=dict(os.environ, PYTHONUTF8="1"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"起動できません（{exc}）"
    text = (out.stdout or out.stderr).strip()
    return text if out.returncode == 0 else f"エラー（{text}）"
