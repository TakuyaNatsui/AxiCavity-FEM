"""出来上がった Windows 版（dist フォルダ）を動かして確かめる.

    python packaging/smoke_test_exe.py <dist> [--gui-seconds 20] [--work <dir>]

確かめること（EXE だけで動く経路。Python 環境には依存しない）:
    1. AxiCavity-FEM.exe version        起動、PARDISO（MKL）、メッシャ（gmsh）、3D 表示（pyvista）の検出
    2. AxiCavity-FEM.exe selftest       GUI と同じ JobSession → QProcess → 自分自身の job（メッシャ → solve → post）
    3. AxiCavity-FEM.exe run <sample>   バッチ（メッシャ → solve → post）。円筒空洞 TM010 = 2.2949 GHz
    4. AxiCavity-FEM.exe solve / post / report / info   コア CLI（.msh を gmsh 無しで読む）
    5. AxiCavity-FEM.exe job mesh <dir> --events        GUI のメッシュジョブ（JSON イベント、プレビュー）
    6. 2 万自由度を超えるメッシュで PARDISO（MKL）の分解が使われる
    7. （--gui-seconds > 0）GUI を起動して数秒間落ちないこと
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

F010_GHZ = 2.2948506
EXE_NAME = "AxiCavity-FEM.exe"


class Checker:
    def __init__(self, exe: Path, work: Path):
        self.exe = exe
        self.work = work
        self.failures: list[str] = []

    def run(self, label: str, args: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
        t0 = time.perf_counter()
        result = subprocess.run([str(self.exe), *args], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout, cwd=str(self.work))
        print(f"--- {label}: 終了コード {result.returncode}（{time.perf_counter() - t0:.1f} s）")
        return result

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        print(f"    {'OK' if ok else 'NG'}  {label}" + (f" — {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dist")
    parser.add_argument("--gui-seconds", type=float, default=0.0)
    parser.add_argument("--work", default=None)
    args = parser.parse_args(argv)
    dist = Path(args.dist).resolve()
    exe = dist / EXE_NAME
    if not exe.exists():
        print(f"{exe} がありません", file=sys.stderr)
        return 1
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="axicavity_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    c = Checker(exe, work)

    r = c.run("version", ["version"])
    print(r.stdout)
    c.check("version", r.returncode == 0, r.stderr[-500:])
    c.check("PARDISO（MKL）あり", "pypardiso" in r.stdout and "MKL" in r.stdout)
    c.check("メッシャ（gmsh）あり", "メッシャ（別プロセス）" in r.stdout and "gmsh 4." in r.stdout)
    c.check("3D 表示（pyvista）あり", "pyvista" in r.stdout and "なし" not in r.stdout.split("3D view", 1)[-1])

    r = c.run("selftest", ["selftest", "--work-dir", str(work / "selftest")])
    print("\n".join(r.stdout.splitlines()[-6:]))
    c.check("selftest", r.returncode == 0 and "selftest → OK" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:])

    sample = dist / "samples" / "cylinder100mm.gmshproj"
    run_dir = work / "run"
    r = c.run("run（バッチ）", ["run", str(sample), "--modes", "3", "--out", str(run_dir), "--json",
                               str(work / "run.json"), "-q"])
    c.check("run の終了コード", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
    try:
        payload = json.loads((work / "run.json").read_text(encoding="utf-8"))
        f0 = float(payload["summary"]["modes"][0]["f_GHz"])
        c.check(f"run: TM010 = {f0:.6f} GHz", abs(f0 - F010_GHZ) / F010_GHZ < 1e-4)
    except Exception as exc:  # noqa: BLE001
        c.check("run の JSON", False, repr(exc))

    msh = run_dir / "model.msh"
    out = work / "cli.h5"
    r = c.run("solve（コア CLI）", ["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "2", "--num-modes", "3",
                                    "-o", str(out)])
    c.check("solve", r.returncode == 0 and out.exists(), r.stdout[-1500:] + r.stderr[-1500:])
    processed = work / "cli_processed.h5"
    r = c.run("post", ["post", "--type", "tm0", "-i", str(out), "-o", str(processed), "--cond", "5.8e7"])
    c.check("post", r.returncode == 0 and processed.exists(), r.stdout[-1500:] + r.stderr[-1500:])
    r = c.run("report", ["report", "--type", "tm0", "-i", str(processed), "-o", str(work / "report")])
    c.check("report（matplotlib Agg）", r.returncode == 0 and (work / "report" / "index.html").exists(),
            r.stdout[-1500:] + r.stderr[-1500:])
    r = c.run("info", ["info", "-i", str(processed)])
    c.check("info", r.returncode == 0)

    job = work / "meshjob"
    job.mkdir()
    shutil.copy2(run_dir / "geometry.json", job / "geometry.json")
    (job / "job.json").write_text(json.dumps({"kind": "mesh", "meshOrder": 2}), encoding="utf-8")
    r = c.run("job mesh（GUI の子プロセス）", ["job", "mesh", str(job), "--events"])
    events = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    c.check("job mesh の done イベント", r.returncode == 0 and bool(events) and events[-1].get("event") == "done",
            r.stdout[-1500:] + r.stderr[-1500:])
    c.check("メッシュのプレビュー", (job / "mesh_preview.npz").exists() and (job / "model.msh").exists())

    big = work / "big"
    r = c.run("run（細かいメッシュ）", ["run", str(sample), "--lc", "0.8", "--modes", "2", "--no-post", "--out", str(big),
                                       "-q"])
    c.check("細かいメッシュの run", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
    r = c.run("solve（PARDISO）", ["solve", "--type", "tm0", "-m", str(big / "model.msh"), "--elem-order", "2",
                                    "--num-modes", "2", "-o", str(work / "big.h5")])
    c.check("PARDISO（MKL）で分解", r.returncode == 0 and "PARDISO (MKL)" in r.stdout, r.stdout[-1500:] + r.stderr[-800:])

    if args.gui_seconds > 0:
        proc = subprocess.Popen([str(exe)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", cwd=str(work))
        time.sleep(args.gui_seconds)
        alive = proc.poll() is None
        if alive:
            proc.kill()
        _out, errs = proc.communicate(timeout=30)
        c.check(f"GUI が {args.gui_seconds:.0f} 秒間動いている", alive and "Traceback" not in (errs or ""), errs[-1500:])

    print("\n結果: " + ("すべて OK" if not c.failures else "NG: " + " / ".join(c.failures)))
    print(f"作業フォルダ: {work}")
    return 0 if not c.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
