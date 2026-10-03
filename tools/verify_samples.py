"""samples の全サンプルを ver3 の経路で解析し、ver2.3 のサマリ（*.txt）と周波数を比べる統合スクリプト.

    python tools/verify_samples.py [--out <作業フォルダ>] [--only s-band_1cell,...] [--rtol 1e-4]

各 ``<name>.gmshproj`` について、ver2.3 の結果サマリ ``<name>_{SW|TW}_{TM0|HOM}.txt``（solve の出力。無ければ
``*_processed.txt``）から解析条件（種類・要素次数・モード数・位相・方位角次数。予測周波数は同名の .h5 の
``target_freq_GHz``）を読み取り、2 通りで解析して (n, 位相, モード) ごとの周波数を比べる:

- **core**: 同梱の ``<name>.msh``（+ ``.materials.json``）をそのまま使う → ver3 の計算コアが ver2.3 と同じことの確認
  （許容 rtol 1e-9。参照の h5 の ``axicavity_fem_version`` が最終の 2.3.0 より古いときは、TM0 の求積則（4 点 → 7 点）や
  ARPACK の収束の違いで 1e-3 程度ずれるので rtol 1e-3）
- **regen**: GUI と同じ経路（``core.legacy.load_gmshproj`` → ``.axiproj`` 保存 → ``to_multi_region`` →
  ``analysis_pipeline.run_job("analysis")``: gmsh でメッシュを作り直し → ``cmd_solve``）→ 同梱の .msh とは節点が
  変わり得る（gmsh の版・曲線境界の節点配置）ので許容は ``--rtol``（既定 1e-4。計画 §6.3）

状態: ``ok``（両方一致）/ ``mesh-diff``（core は一致、regen が rtol 超）/ ``mismatch``（core が不一致）/ ``error``。
終了コードは mismatch / error があれば 1。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
sys.path.insert(0, str(ROOT / "src"))

from axicavity_fem.gui.core.convert import check_geometry, to_multi_region   # noqa: E402
from axicavity_fem.gui.core.legacy import load_gmshproj                     # noqa: E402
from axicavity_fem.gui.io.axiproj import ProjectFile, read_project, write_project   # noqa: E402
from axicavity_fem.gui.jobs import analysis_pipeline, pipeline              # noqa: E402
from axicavity_fem.gui.jobs.commands import build_commands                  # noqa: E402

CORE_RTOL = 1e-9
FINAL_CORE_VERSION = "2.3.0"         # h5 の axicavity_fem_version。これより古い参照は求積則が違う（1e-3 まで許容）
OLD_REFERENCE_RTOL = 1e-3
_HEAD =re.compile(r"^\[n=(\d+)\]\s+(standing|traveling)(?:\s+phase=([0-9.]+)deg)?", re.M)
_FREQ = re.compile(r"f\s*=\s*([0-9.]+)\s*GHz")
_FIELDS = {
    "type": re.compile(r"^(?:解析タイプ|analysis type)\s*:\s*(\S+)", re.M),
    "order": re.compile(r"^(?:要素次数|element order)\s*:\s*(\d+)", re.M),
    "modes": re.compile(r"^(?:要求モード数|num modes)\s*:\s*(\d+)", re.M),
}


def parse_summary(path: Path) -> dict:
    """ver2.3 の .txt → {type, order, modes, target, freqs: {(n, phase|None): [f...]}}."""
    text = path.read_text(encoding="utf-8")
    info = {key: pat.search(text).group(1).strip() for key, pat in _FIELDS.items()}
    freqs: dict = {}
    heads = list(_HEAD.finditer(text))
    for i, m in enumerate(heads):
        block = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        phase = float(m.group(3)) if m.group(2) == "traveling" else None
        freqs[(int(m.group(1)), phase)] = [float(f) for f in _FREQ.findall(block)]
    target, version = None, None
    h5 = path.with_suffix(".h5")
    if h5.exists():
        try:
            import h5py
            with h5py.File(h5, "r") as f:
                value = f["parameters"].attrs.get("target_freq_GHz") if "parameters" in f else None
                version = f.attrs.get("axicavity_fem_version")
            target = float(value) if value is not None else None
            version = version.decode() if isinstance(version, bytes) else (str(version) if version is not None else None)
        except Exception:  # noqa: BLE001
            target = None
    return {"type": info["type"].lower(), "order": int(info["order"]), "modes": int(info["modes"]),
            "target": target, "version": version, "freqs": freqs}


def reference_files(name: str) -> list[Path]:
    out = []
    for wave in ("SW", "TW"):
        for kind in ("TM0", "HOM"):
            raw = SAMPLES / f"{name}_{wave}_{kind}.txt"
            processed = SAMPLES / f"{name}_{wave}_{kind}_processed.txt"
            if raw.exists():
                out.append(raw)
            elif processed.exists():
                out.append(processed)
    return out


def configure(doc, ref: dict) -> None:
    a = doc.analysis
    a.type = ref["type"]
    a.numModes = ref["modes"]
    a.runPost = False
    a.targetFreqGHz = ref["target"]
    keys = list(ref["freqs"])
    phases = sorted({p for _n, p in keys if p is not None})
    a.wave = "traveling" if phases else "standing"
    a.phases = ",".join(f"{p:g}" for p in phases) if phases else "120"
    a.azOrders = " ".join(str(n) for n in sorted({n for n, _p in keys}))
    doc.mesh.order = ref["order"]


def solve(doc, folder: Path, shipped_mesh: Path | None) -> tuple[dict, float]:
    """ジョブフォルダを作って解析（shipped_mesh があればそれをコピーして使う）. (freqs, 秒)."""
    folder.mkdir(parents=True, exist_ok=True)
    to_multi_region(doc, strict=True).geom.save_json(folder / pipeline.GEOMETRY_FILE)
    cmds = build_commands(doc)
    job = {"kind": cmds["kind"], "meshOrder": int(doc.mesh.order), "argv": {"solve": cmds["solve"], "post": None},
           "files": {"mesh": "model.msh", "raw": cmds["raw"], "processed": None}}
    (folder / pipeline.JOB_FILE).write_text(json.dumps(job), encoding="utf-8")
    if shipped_mesh is not None:
        shutil.copy2(shipped_mesh, folder / "model.msh")
        materials = shipped_mesh.with_suffix(".materials.json")
        if materials.exists():
            shutil.copy2(materials, folder / "model.materials.json")
    t0 = time.perf_counter()
    result = analysis_pipeline.run_job("analysis", folder, log=lambda _m: None)
    got: dict = {}
    for m in result["summary"]["modes"]:
        got.setdefault((m["n"], m["phase"]), []).append(m["f_GHz"])
    return got, time.perf_counter() - t0


def compare(expected: dict, got: dict) -> tuple[int, float, list[str]]:
    compared, worst, missing = 0, 0.0, []
    for key, freqs in expected.items():
        actual = got.get(key)
        if actual is None:
            missing.append(f"n={key[0]} phase={key[1]}")
            continue
        for fe, fa in zip(freqs, actual):
            compared += 1
            worst = max(worst, abs(fa - fe) / fe)
    return compared, worst, missing


def run_case(name: str, ref_path: Path, work: Path, rtol: float) -> dict:
    doc, warnings = load_gmshproj(SAMPLES / f"{name}.gmshproj")
    ref = parse_summary(ref_path)
    configure(doc, ref)
    issues = [i for i in check_geometry(doc) if i.level == "error"]
    if issues:
        return {"name": name, "ref": ref_path.name, "status": "error", "detail": "; ".join(i.message for i in issues)}
    project = work / f"{name}.axiproj"
    write_project(project, ProjectFile(document=doc, generator="verify_samples"))
    doc = read_project(project).document          # 保存 → 読み直し（GUI と同じ）
    row: dict = {"name": name, "ref": ref_path.name, "target": ref["target"], "refVersion": ref["version"],
                 "warnings": warnings}
    # 参照が最終の ver2.3 コア（2.3.0）より前に作られたものなら、求積則や ARPACK の収束の違いで 1e-3 程度ずれる
    old_reference = ref["version"] != FINAL_CORE_VERSION
    core_rtol = OLD_REFERENCE_RTOL if old_reference else CORE_RTOL
    shipped = SAMPLES / f"{name}.msh"
    if shipped.exists():
        got, secs = solve(doc, work / f"{ref_path.stem}_core", shipped)
        compared, worst, missing = compare(ref["freqs"], got)
        row.update(coreCompared=compared, coreRel=worst, coreMissing=missing, coreS=round(secs, 1),
                   coreRtol=core_rtol)
        core_ok = compared > 0 and not missing and worst <= core_rtol
    else:
        core_ok = None
    if old_reference:
        rtol = max(rtol, OLD_REFERENCE_RTOL)
    got, secs = solve(doc, work / f"{ref_path.stem}_regen", None)
    compared, worst, missing = compare(ref["freqs"], got)
    row.update(regenCompared=compared, regenRel=worst, regenMissing=missing, regenS=round(secs, 1))
    regen_ok = compared > 0 and not missing and worst <= rtol
    if core_ok is False:
        row["status"] = "mismatch"
    elif regen_ok:
        row["status"] = "ok"
    else:
        row["status"] = "mesh-diff" if core_ok else "mismatch"
    return row


def _describe(row: dict) -> str:
    if row.get("detail"):
        return row["detail"]
    parts = []
    if "coreRel" in row:
        parts.append(f"core {row['coreCompared']} modes rel {row['coreRel']:.1e}"
                     + (f" missing {row['coreMissing']}" if row.get("coreMissing") else ""))
    parts.append(f"regen {row['regenCompared']} modes rel {row['regenRel']:.1e} ({row['regenS']} s)"
                 + (f" missing {row['regenMissing']}" if row.get("regenMissing") else ""))
    if row.get("target"):
        parts.append(f"target {row['target']:g} GHz")
    if row.get("refVersion") and row["refVersion"] != FINAL_CORE_VERSION:
        parts.append(f"ref core {row['refVersion']} (older than {FINAL_CORE_VERSION}: rtol {OLD_REFERENCE_RTOL:g})")
    return "; ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "build" / "verify_samples"))
    ap.add_argument("--only", default="", help="カンマ区切りのサンプル名")
    ap.add_argument("--rtol", type=float, default=1e-4, help="regen（メッシュ作り直し）の許容相対誤差")
    args = ap.parse_args(argv)
    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    rows = []
    for proj in sorted(SAMPLES.glob("*.gmshproj")):
        name = proj.stem
        if only and name not in only:
            continue
        refs = reference_files(name)
        if not refs:
            rows.append({"name": name, "ref": "-", "status": "no-reference"})
            print(f"{name:20s} (ver2.3 のサマリなし)")
            continue
        for ref_path in refs:
            try:
                row = run_case(name, ref_path, work, args.rtol)
            except Exception as exc:  # noqa: BLE001
                row = {"name": name, "ref": ref_path.name, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(f"{name:20s} {ref_path.name:30s} {row['status']:10s} {_describe(row)}")
            sys.stdout.flush()
    (work / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r for r in rows if r["status"] in ("mismatch", "error")]
    diff = [r for r in rows if r["status"] == "mesh-diff"]
    print(f"\n{len(rows) - len(bad) - len(diff)} ok / {len(diff)} mesh-diff / {len(bad)} bad → {work / 'summary.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
