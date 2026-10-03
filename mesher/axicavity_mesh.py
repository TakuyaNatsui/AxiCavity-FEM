#!/usr/bin/env python3
# axicavity_mesh.py — AxiCavity-FEM のメッシュ生成ヘルパ（gmsh を使う別プログラム）
#
# Copyright (c) 2026 Takuya Natsui. MIT License (see LICENSE in the repository).
# Windows 版では同梱の gmsh（GPL-2.0-or-later）と一緒に「メッシャ」として配布する。メッシャ全体の配布条件は
# 同じフォルダの README.txt / LICENSE を参照。
"""形状（AxiCavity-FEM の MultiRegionGeometry の JSON）から gmsh で ``.msh`` を作る.

Windows 版（EXE）は本体に gmsh を入れない。本体はこのスクリプトを同梱の埋め込み Python + gmsh で別プロセスとして
起動し、ファイル（形状の JSON、``.msh``）だけでやり取りする（``axicavity_fem/gui/jobs/meshing.py``）。
メッシュは計算コアの ``shared/gmsh_export_occ.export_msh_multi_region`` で作る（pip 版の GUI と同じ関数・同じ結果）。
numpy には依存しない（コアの 2 モジュールは標準ライブラリと gmsh だけを使う）。

使い方::

    python axicavity_mesh.py mesh <job.json>
    python axicavity_mesh.py version

job.json::

    {"geometry": "geometry.json", "out": "model.msh", "meshOrder": 2, "verbose": 1}

結果 ``<out>.mesher.json``::

    {"ok": true, "msh": "…/model.msh", "elapsedS": 0.42}
    {"ok": false, "error": {"type": "ValueError", "message": "…"}}   （ValueError 以外は "traceback" も）

進捗は gmsh の出力として標準出力に出る（呼び出し側がジョブのログへ流す）。
終了コード: 成功 0、入力の問題（ValueError）1、その他の例外 2。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

VERSION = "3.0.0"
HERE = Path(__file__).resolve().parent

# 配布物: このフォルダの lib/（コアの 2 モジュールのコピー）。開発環境: リポジトリの src/
for _candidate in (HERE / "lib", HERE.parent / "src"):
    if (_candidate / "axicavity_fem" / "shared" / "gmsh_export_occ.py").exists():
        sys.path.insert(0, str(_candidate))
        break


def run_mesh_job(job: dict) -> dict:
    from axicavity_fem.shared.gmsh_export_occ import export_msh_multi_region
    from axicavity_fem.shared.multi_region_model import MultiRegionGeometry

    geom = MultiRegionGeometry.load_json(job["geometry"])
    out = Path(job["out"])
    order = int(job.get("meshOrder", 2))
    if order not in (1, 2):
        raise ValueError(f"meshOrder は 1 か 2（{order}）")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    export_msh_multi_region(geom, out, mesh_order=order, verbose=int(job.get("verbose", 1)))
    return {"ok": True, "msh": str(out), "elapsedS": round(time.perf_counter() - t0, 3)}


def _result_path(job: dict) -> Path:
    return Path(str(job.get("out", "model.msh")) + ".mesher.json")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "version":
        import gmsh

        print(f"axicavity_mesh {VERSION}; gmsh {gmsh.__version__}; python {sys.version.split()[0]}")
        return 0
    if argv[0] != "mesh" or len(argv) != 2:
        print("使い方: axicavity_mesh.py mesh <job.json> | version", file=sys.stderr)
        return 2
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    result_path = _result_path(job)
    try:
        result = run_mesh_job(job)
        code = 0
    except ValueError as exc:
        result = {"ok": False, "error": {"type": "ValueError", "message": str(exc)}}
        code = 1
    except Exception as exc:  # noqa: BLE001 — 呼び出し側に理由を返す
        result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc()}}
        code = 2
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if code:
        print(result["error"]["message"], file=sys.stderr)
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
