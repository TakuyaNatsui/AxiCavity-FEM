"""Windows 版に同梱する第三者ライブラリのライセンス表記（THIRD_PARTY_NOTICES.txt）を作る.

    python packaging/third_party_notices.py --report <build>/nuitka-report.xml --out <dist>/THIRD_PARTY_NOTICES.txt

Nuitka のレポート（``--report``）に載った同梱モジュールの配布物（distribution）と、データファイルとして入れた
Intel MKL / TBB について、名前・版・ライセンス（メタデータ）と dist-info のライセンス文書の全文を連結する。
ビルドに使った Python 環境で実行すること（配布物のメタデータを読む）。

EM-CAD-py（同じ作者）の packaging/third_party_notices.py を AxiCavity-FEM 用にしたもの。
注意: 同じ Python 環境に入っていても EXE に入らない配布物は載せない（固定の一覧はデータファイルの分だけ）。
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSE_DIR = ROOT / "packaging" / "licenses"

# モジュールではなくデータファイル（DLL）として入れるもの（Nuitka のレポートに出ない）
DATA_FILE_DISTRIBUTIONS = ["mkl", "tbb", "onemkl-license"]
# 載せない配布物（ビルド時のツール・EXE に入らないもの）
EXCLUDE = {"nuitka", "ordered-set", "zstandard", "setuptools", "pip", "wheel", "gmsh", "intel-openmp",
           "intel-cmplr-lib-ur", "axicavity-fem"}

# dist-info に全文が無いライセンスの補足（packaging/licenses/ のファイルを連結する）
EXTRA_TEXTS = {
    "PySide6": [("LGPL-3.0.txt", "https://www.gnu.org/licenses/lgpl-3.0.txt"),
                ("GPL-3.0.txt", "https://www.gnu.org/licenses/gpl-3.0.txt")],
    "PySide6_Essentials": [("LGPL-3.0.txt", "https://www.gnu.org/licenses/lgpl-3.0.txt")],
    "PySide6_Addons": [("LGPL-3.0.txt", "https://www.gnu.org/licenses/lgpl-3.0.txt")],
    "shiboken6": [("LGPL-3.0.txt", "https://www.gnu.org/licenses/lgpl-3.0.txt")],
    "planegcs": [("LGPL-2.1.txt", "https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt")],
}

HEADER = """THIRD-PARTY NOTICES — AxiCavity-FEM（Windows 版）に同梱している第三者ソフトウェアとそのライセンス
==============================================================================

AxiCavity-FEM 本体（このフォルダの AxiCavity-FEM.exe）は MIT License です（LICENSE）。ソースコードは
https://github.com/TakuyaNatsui/AxiCavity-FEM にあります。

本体は以下のソフトウェアを含みます。各ライセンスの条件に従って著作権表示とライセンス文書を再掲します。
Qt（PySide6 / shiboken6）と planegcs は LGPL のもとで動的リンクしており、同梱の DLL / .pyd を
同等のライブラリに差し替えることができます（DLL を改ざん検知・暗号化していません）。Intel MKL / TBB は
Intel Simplified Software License のもとで再配布しています（改変していません）。

メッシュ生成は mesher フォルダの別プログラム（Gmsh を含み GNU GPL v2 以降で配布。ソースを同梱）で行います。
本体は gmsh のコードを含まず、mesher を子プロセスとして起動してファイルだけでやり取りします。
mesher の構成とライセンスは mesher/README.txt と mesher/LICENSE を参照してください。
"""


def distributions_from_report(report: Path) -> list[str]:
    """Nuitka の XML レポートから同梱モジュールの配布物名を集める."""
    tree = ET.parse(report)
    modules = {m.get("name", "") for m in tree.iter("module")}
    tops = {name.split(".")[0] for name in modules if name}
    # __main__ は本体の main スクリプト。packages_distributions() は __main__ を、それを RECORD に持つ配布物
    # （PyMuPDF など）に対応させてしまい、同梱していない AGPL のライブラリが表記に載る
    tops.discard("__main__")
    mapping = metadata.packages_distributions()
    found = set()
    for top in tops:
        for dist in mapping.get(top, []):
            found.add(dist)
    return sorted(found, key=str.lower)


def license_summary(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    lic = meta.get("License")
    classifiers = [c for c in meta.get_all("Classifier", []) if c.startswith("License ::")]
    parts = [p for p in (expr, lic if lic and len(lic) < 200 else None) if p]
    parts += [c.replace("License :: ", "") for c in classifiers]
    return "; ".join(dict.fromkeys(parts)) or "(メタデータにライセンス表記なし)"


def license_files(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """dist-info 内のライセンス文書（License-File と LICENSE*/COPYING*/NOTICE*）."""
    names: list[str] = list(dist.metadata.get_all("License-File", []))
    for f in dist.files or []:
        rel = str(f).replace("\\", "/")
        if ".dist-info/" in rel:
            inside = rel.split(".dist-info/", 1)[1]
            base = Path(inside).name.lower()
            if base.startswith(("license", "copying", "notice", "authors")) and inside not in names:
                names.append(inside)
    out = []
    seen = set()
    for name in names:
        text = None
        for candidate in (name, str(Path("licenses") / name)):
            try:
                text = dist.read_text(candidate)
            except (OSError, UnicodeDecodeError):
                text = None
            if text:
                break
        if text and text not in seen:
            seen.add(text)
            out.append((Path(name).name, text))
    return out


def build_notices(dist_names: list[str]) -> str:
    lines = [HEADER, ""]
    seen = set()
    for name in dist_names:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        key = dist.metadata["Name"].lower()
        if key in seen or key in EXCLUDE:
            continue
        seen.add(key)
        lines += ["-" * 78, f"{dist.metadata['Name']} {dist.version}", f"  License: {license_summary(dist)}"]
        home = dist.metadata.get("Home-page") or ""
        for url in dist.metadata.get_all("Project-URL", []):
            if not home and ("homepage" in url.lower() or "source" in url.lower()):
                home = url.split(",", 1)[-1].strip()
        if home:
            lines.append(f"  URL: {home}")
        lines.append("")
        for fname, text in license_files(dist):
            lines += [f"  --- {fname} ---", text.rstrip(), ""]
        for fname, url in EXTRA_TEXTS.get(dist.metadata["Name"], []):
            path = LICENSE_DIR / fname
            if path.exists():
                lines += [f"  --- {fname} ---", path.read_text(encoding="utf-8").rstrip(), ""]
            else:
                lines += [f"  ライセンス全文: {url}", ""]
    lines += ["-" * 78, "Python", "  PSF License — https://docs.python.org/3/license.html", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="THIRD_PARTY_NOTICES.txt")
    parser.add_argument("--report", required=True, help="Nuitka の --report XML")
    args = parser.parse_args(argv)
    try:
        names = distributions_from_report(Path(args.report))
    except (OSError, ET.ParseError) as exc:
        print(f"Nuitka のレポートを読めません: {exc}", file=sys.stderr)
        return 1
    print(f"Nuitka レポートから {len(names)} 配布物を検出")
    names = sorted(set(names) | set(DATA_FILE_DISTRIBUTIONS), key=str.lower)
    text = build_notices(names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"書き出し: {out}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
