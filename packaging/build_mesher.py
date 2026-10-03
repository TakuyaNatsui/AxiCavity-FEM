"""Windows 版に同梱するメッシャ（mesher/。gmsh を含む GPL の別プログラム、ソース同梱）を組み立てる.

    python packaging/build_mesher.py --out <dir> [--download]

組み立てる内容::

    mesher/
      python.exe, python313.dll, python313.zip, python313._pth …   python.org の embeddable package（PSF License）
      gmsh.py, gmsh-4.15.dll        pip の gmsh パッケージ（GPL-2.0-or-later）をそのままコピー
      axicavity_mesh.py             メッシュ生成ヘルパ（このリポジトリの mesher/axicavity_mesh.py、MIT）
      lib/axicavity_fem/…           ヘルパが使う計算コアの 2 モジュール（gmsh_export_occ / multi_region_model、MIT。そのまま）
      LICENSE                       GPL v2 の全文（gmsh のライセンス）
      LICENSE-Python.txt            埋め込み Python のライセンス
      README.txt                    構成とライセンスの説明
      src/gmsh-<ver>-source.tgz     gmsh のソース（DLL を再配布するので同梱）
      src/SOURCES.txt               同梱物とソースの所在

本体（AxiCavity-FEM.exe）はこのメッシャを別プロセスとして起動し、形状の JSON と .msh だけでやり取りする
（``axicavity_fem/gui/jobs/meshing.py``）。メッシャは numpy に依存しない。

ダウンロード（不足分のみ。``--download`` のときだけ。``<out>/../downloads/`` にキャッシュ）:
    python-<ver>-embed-amd64.zip  https://www.python.org/ftp/python/<ver>/
    gmsh-<ver>-source.tgz         https://gmsh.info/src/
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "mesher"
CORE_FILES = ("axicavity_fem/__init__.py", "axicavity_fem/shared/__init__.py",
              "axicavity_fem/shared/gmsh_export_occ.py", "axicavity_fem/shared/multi_region_model.py")


def python_embed_info() -> tuple[str, str]:
    """(バージョン, embeddable zip の URL)。実行中の Python と同じバージョンを使う."""
    ver = platform.python_version()
    arch = "amd64" if platform.machine().lower() in ("amd64", "x86_64") else "arm64"
    return ver, f"https://www.python.org/ftp/python/{ver}/python-{ver}-embed-{arch}.zip"


def gmsh_info() -> tuple[str, Path, Path]:
    """(gmsh のバージョン, gmsh.py のパス, gmsh DLL のパス)."""
    code = "import gmsh, os; print(gmsh.__version__); print(os.path.abspath(gmsh.__file__)); print(gmsh.libpath)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).split("\n")
    return out[0].strip(), Path(out[1].strip()), Path(out[2].strip())


def fetch(url: str, dest: Path, download: bool, retries: int = 3) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not download:
        print(f"未取得: {dest.name}（{url}）— --download で取得するか {dest.parent} に置く")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        print(f"取得: {url}" + (f"（{attempt} 回目）" if attempt > 1 else ""))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "axicavity-fem-build/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            if tmp.stat().st_size == 0:
                raise OSError("0 バイトでした")
            tmp.replace(dest)
            return dest
        except (OSError, urllib.error.URLError) as exc:
            print(f"  失敗: {exc}")
            tmp.unlink(missing_ok=True)
    return None


def _rmtree(path: Path, attempts: int = 5) -> None:
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(1.0)


README = """AxiCavity-FEM mesher — メッシュ生成（gmsh を含む GPL の別プログラム）
=====================================================================

このフォルダは AxiCavity-FEM の Windows 版に同梱しているメッシュ生成用の **別プログラム** です。
本体（AxiCavity-FEM.exe）から子プロセスとして起動され、形状の JSON と .msh ファイルだけでやり取りします。

内容
----
  axicavity_mesh.py   メッシュ生成ヘルパ（MIT License。ソースそのもの）
  lib/axicavity_fem/  ヘルパが使う AxiCavity-FEM の 2 モジュール（MIT License。ソースそのもの）
  gmsh.py, gmsh-*.dll Gmsh（https://gmsh.info、GNU GPL v2 以降）の Python API と本体
  python.exe ほか     Python の埋め込み用配布物（PSF License）
  LICENSE             GNU GPL v2 の全文
  LICENSE-Python.txt  埋め込み Python のライセンス
  src/                再配布している gmsh のソース（同じバージョン）と、ソースの所在

ライセンス
----------
  Gmsh は GNU General Public License（v2 以降）で配布されています。このフォルダ全体（メッシャ）は GPL の
  条件で配布します。ソースは src/ と、このフォルダの .py ファイル、および
  https://github.com/TakuyaNatsui/AxiCavity-FEM にあります。
  AxiCavity-FEM の本体はこのプログラムを別プロセスとして利用するだけで、gmsh のコードを含みません。

使い方（本体が自動で行います）
------------------------------
  python.exe axicavity_mesh.py mesh <job.json>
  python.exe axicavity_mesh.py version
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="組み立て先（例: <build>/mesher）")
    parser.add_argument("--downloads", default=None, help="取得したファイルの置き場（既定 <out>/../downloads）")
    parser.add_argument("--download", action="store_true", help="不足ファイルをダウンロードする")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    downloads = Path(args.downloads).resolve() if args.downloads else out.parent / "downloads"

    py_ver, py_url = python_embed_info()
    gmsh_ver, gmsh_py, gmsh_dll = gmsh_info()
    gmsh_src_url = f"https://gmsh.info/src/gmsh-{gmsh_ver}-source.tgz"
    embed_zip = fetch(py_url, downloads / Path(py_url).name, args.download)
    gmsh_src = fetch(gmsh_src_url, downloads / Path(gmsh_src_url).name, args.download)
    if embed_zip is None or gmsh_src is None:
        print("必要なファイルが無いので組み立てを中止します", file=sys.stderr)
        return 1

    if out.exists():
        _rmtree(out)
    out.mkdir(parents=True)

    # 1. 埋め込み Python（自分のフォルダと lib/ と標準ライブラリだけを見る。site は読まない）
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(out)
    if (out / "LICENSE.txt").exists():
        (out / "LICENSE.txt").rename(out / "LICENSE-Python.txt")
    pth = next(out.glob("python*._pth"))
    pth.write_text(f"{pth.name.replace('._pth', '.zip')}\n.\nlib\n", encoding="utf-8")

    # 2. gmsh（GPL-2.0-or-later）とヘルパ、ヘルパが使うコアのモジュール
    shutil.copy2(gmsh_py, out / "gmsh.py")
    shutil.copy2(gmsh_dll, out / gmsh_dll.name)
    shutil.copy2(SRC_DIR / "axicavity_mesh.py", out / "axicavity_mesh.py")
    for rel in CORE_FILES:
        target = out / "lib" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / rel, target)

    # 3. ライセンスとソース
    shutil.copy2(ROOT / "packaging" / "licenses" / "GPL-2.0.txt", out / "LICENSE")
    (out / "README.txt").write_text(README, encoding="utf-8")
    src = out / "src"
    src.mkdir()
    shutil.copy2(gmsh_src, src / gmsh_src.name)
    (src / "SOURCES.txt").write_text(
        "メッシャ（GPL）に含まれるプログラムと、その対応するソース:\n\n"
        "  axicavity_mesh.py, lib/   このフォルダにソースのまま（MIT License）。"
        "https://github.com/TakuyaNatsui/AxiCavity-FEM\n"
        f"  gmsh {gmsh_ver}               {gmsh_src.name}（{gmsh_src_url}）。GPL-2.0-or-later\n"
        f"  Python {py_ver}             埋め込み用配布物（PSF License）。ソース: https://www.python.org/downloads/source/\n",
        encoding="utf-8")

    # 4. 動作確認
    exe = out / ("python.exe" if sys.platform == "win32" else "bin/python3")
    result = subprocess.run([str(exe), "-B", str(out / "axicavity_mesh.py"), "version"],
                            capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        print("メッシャの起動に失敗しました", file=sys.stderr)
        return result.returncode
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"メッシャ完成: {out}（{size:.0f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
