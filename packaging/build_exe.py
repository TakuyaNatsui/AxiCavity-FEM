"""Windows 版 AxiCavity-FEM（AxiCavity-FEM.exe = GUI + コマンドライン、Nuitka standalone）を作る.

    python packaging/build_exe.py --build-dir C:\\build\\axicavity                # ビルド → 後処理 → zip
    python packaging/build_exe.py --build-dir … --dry-run                           # Nuitka のコマンドだけ表示
    python packaging/build_exe.py --build-dir … --skip-nuitka                       # 後処理と zip だけやり直す

前提: ``pip install nuitka ordered-set zstandard``、gmsh・pypardiso（pip 版 MKL）・pyvista が入った Python。
C コンパイラは MSVC が無ければ Nuitka が zig を自動で取得する（Python 3.13 以降。``--assume-yes-for-downloads``）。

構成（docs/DEVELOPER_GUIDE.md の「Windows 版（EXE）」）:
    - ``--standalone``（onefile ではない）、``--windows-console-mode=attach``（端末から呼べばコマンドライン、
      ダブルクリックならコンソール無し）。GUI の子プロセスもこの EXE 自身（``AxiCavity-FEM.exe job …``）。
    - gmsh（GPL）は本体に入れない（``--nofollow-import-to=gmsh``）。``.msh`` の読込は純 Python の互換モジュール
      （``axicavity_fem.gui.mshlite``）、生成は ``mesher/``（埋め込み Python + gmsh の別プログラム。build_mesher.py）。
    - pip 版の Intel MKL（pypardiso 用）は Nuitka が自動で入れないので ``Library/bin/`` にデータファイルとして入れる
      （スレッド層は TBB。intel-openmp は別のライセンスなので入れない）。
    - ビルド先は OneDrive の外に（数万ファイルの同期とロックで遅く・失敗しやすい）。Claude デスクトップアプリから
      実行するときは ``%LOCALAPPDATA%`` も避ける（MSIX の仮想化で別の場所に書かれる）。
    - zig のキャッシュ（``ZIG_LOCAL_CACHE_DIR``）はビルド先の ``zig-cache/`` に分ける。Nuitka の既定の共有キャッシュだと、
      別のプロジェクト（EM-CAD-py など）のビルドで作った定数のオブジェクトが返ることがある（定数のソース
      ``__constants_data___constant.c`` は名前もフラグも同じで、``#embed`` するファイルは前回の絶対パスで照合されるため）。
      そうなると EXE が "Frozen object named 'encodings' is invalid" で起動しない。ビルド後に定数を照合して検出する。

EM-CAD-py（同じ作者）の packaging/build_exe.py を AxiCavity-FEM 用にしたもの。
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT / "packaging" / "AxiCavity-FEM.py"
EXE_NAME = "AxiCavity-FEM.exe"

# 本体に入れないもの: GPL の gmsh、旧 GUI、テスト・開発用、pyvista などの任意依存のうち使わない重いもの
NOFOLLOW = [
    "gmsh", "wx", "tkinter",
    "IPython", "jupyter", "ipywidgets", "pytest", "pytest_qt",
    "pandas", "pyarrow", "numba", "llvmlite",
    "trame", "trame_vtk", "trame_vuetify", "trame_client", "trame_server",
    "sympy", "mpmath", "requests", "cryptography", "urllib3", "fsspec", "jinja2", "lxml",
    # Qt Data Visualization はオープンソース版が GPL-3.0 のみ（LGPL ではない）。qtpy が import 時に読みにいく
    # （ImportError は qtpy が無視する）ので、入れると 3D 表示のたびに GPL の DLL が本体のプロセスに載る
    "PySide6.QtDataVisualization", "qtpy.QtDataVisualization",
]
# GPL-3.0 のみで提供される Qt のアドオン（LGPL ではない）。dist にあればビルドを止める
GPL_ONLY_QT_DLLS = ("qt6charts", "qt6datavisualization", "qt6graphs", "qt6virtualkeyboard", "qt6quick3d",
                    "qt6lottie", "qt6quicktimeline", "qt6shadertools")
INCLUDE_PACKAGES = ["axicavity_fem"]
INCLUDE_PACKAGE_DATA = ["axicavity_fem.gui"]       # ui/icons/*.svg・ui/app_icon.svg・i18n/*.json
# 版を ``importlib.metadata`` でしか知る方法が無いもの（バージョン情報の表示用。planegcs には __version__ が無い）。
# pip の mkl は Python のパッケージが無いので Nuitka が受け付けない（版は app_info が MKL 自身に聞く）
INCLUDE_METADATA = ["planegcs"]

# pip 版 MKL（Intel Simplified Software License）のランタイム DLL。スレッド層は TBB（同ライセンス）
MKL_DLL_PATTERNS = (
    "mkl_rt.*.dll", "mkl_core.*.dll", "mkl_tbb_thread.*.dll", "mkl_sequential.*.dll",
    "mkl_def.*.dll", "mkl_mc3.*.dll", "mkl_avx2.*.dll", "mkl_avx512.*.dll", "mkl_avx10.*.dll",
    "mkl_vml_*.dll", "tbb12.dll",
)

# 同梱する文書とサンプル（samples/ は入力ファイルだけ）
DOC_FILES = ["USER_MANUAL.md", "PHYSICS_AND_CONVENTIONS.md", "LICENSE"]
SAMPLE_PATTERNS = ["*.gmshproj", "*.af", "*.py", "README.md"]


def app_version() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from axicavity_fem.gui import GUI_VERSION

    return GUI_VERSION


def installed(dist: str) -> bool:
    try:
        importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def mkl_dll_patterns() -> list[tuple[str, str]]:
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return []
    return [(str(library_bin / pat), "Library/bin/") for pat in MKL_DLL_PATTERNS
            if glob.glob(str(library_bin / pat))]


def nuitka_command(args: argparse.Namespace, build_dir: Path, icon: Path | None) -> list[str]:
    version = args.version
    numeric = ".".join((version.split("+")[0].split(".") + ["0", "0", "0", "0"])[:4])
    cmd = [sys.executable, "-m", "nuitka",
           "--standalone",
           f"--output-dir={build_dir}",
           f"--report={build_dir / 'nuitka-report.xml'}",
           f"--output-filename={EXE_NAME}",
           "--windows-console-mode=attach",
           "--enable-plugin=pyside6",
           "--noinclude-numba-mode=nofollow",
           "--noinclude-pytest-mode=nofollow",
           "--noinclude-IPython-mode=nofollow",
           "--noinclude-setuptools-mode=nofollow",
           "--noinclude-unittest-mode=nofollow",
           "--python-flag=isolated",                 # 外部の site-packages や PYTHONPATH を読まない
           "--no-deployment-flag=self-execution",    # コア CLI の `-m <mesh>` を自己呼び出しと誤認させない
           "--company-name=AxiCavity-FEM", "--product-name=AxiCavity-FEM",
           f"--file-version={numeric}", f"--product-version={numeric}",
           "--file-description=AxiCavity-FEM: 2-D FEM solver for axisymmetric RF cavity modes",
           "--copyright=(c) 2026 Takuya Natsui. MIT License",
           f"--jobs={args.jobs}",
           "--lto=no",
           "--assume-yes-for-downloads"]
    for name in INCLUDE_PACKAGES:
        cmd.append(f"--include-package={name}")
    for name in INCLUDE_PACKAGE_DATA:
        cmd.append(f"--include-package-data={name}")
    for name in INCLUDE_METADATA:
        if installed(name):
            cmd.append(f"--include-distribution-metadata={name}")
    cmd.append("--nofollow-import-to=" + ",".join(NOFOLLOW))
    for src, dst in mkl_dll_patterns():
        cmd.append(f"--include-data-files={src}={dst}")
    if icon is not None:
        cmd.append(f"--windows-icon-from-ico={icon}")
    if args.msvc:
        cmd.append(f"--msvc={args.msvc}")
    if args.zig:
        cmd.append("--zig")
    cmd.extend(args.extra or [])
    cmd.append(str(MAIN_SCRIPT))
    return cmd


def constants_embedded(build_dir: Path, exe: Path) -> bool:
    """EXE に今回のビルドの定数（``<name>.build/blobs/__constant.bin``）がそのまま入っているか.

    zig のキャッシュが別のビルドのオブジェクトを返したときに False になる。blob が無ければ確かめようがないので True。
    """
    blob = build_dir / (MAIN_SCRIPT.stem + ".build") / "blobs" / "__constant.bin"
    if not blob.is_file():
        return True
    expected, data = blob.read_bytes(), exe.read_bytes()
    pos = data.find(expected[:4096])
    return pos >= 0 and data[pos:pos + len(expected)] == expected


def gpl_only_qt_dlls(dist: Path) -> list[str]:
    """dist に入ってしまった GPL のみの Qt モジュールの DLL（MKL と同じプロセスに載せない）."""
    return sorted(p.name for p in dist.rglob("*.dll") if p.name.lower().startswith(GPL_ONLY_QT_DLLS))


def remove_tree(path: Path) -> None:
    """フォルダを消す。読み取り専用の属性（OneDrive のフォルダから copytree で写る）が付いていても消す."""
    def retry(func, name, _exc):
        os.chmod(name, stat.S_IWRITE)
        func(name)

    shutil.rmtree(path, onexc=retry)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        remove_tree(dst)
    shutil.copytree(src, dst)


def assemble(dist: Path, mesher: Path, build_dir: Path, version: str) -> None:
    """dist にメッシャ・文書・サンプル・ライセンス表記を入れる."""
    copy_tree(mesher, dist / "mesher")
    samples = dist / "samples"
    if samples.exists():
        remove_tree(samples)
    samples.mkdir()
    for pattern in SAMPLE_PATTERNS:
        for path in sorted((ROOT / "samples").glob(pattern)):
            shutil.copy2(path, samples / path.name)
    for name in DOC_FILES:
        shutil.copy2(ROOT / name, dist / name)
    copy_tree(ROOT / "docs" / "images", dist / "docs" / "images")
    readme = (ROOT / "packaging" / "README_dist.txt").read_text(encoding="utf-8").replace("{version}", version)
    (dist / "README.txt").write_text(readme, encoding="utf-8")
    subprocess.check_call([sys.executable, str(ROOT / "packaging" / "third_party_notices.py"),
                           "--report", str(build_dir / "nuitka-report.xml"),
                           "--out", str(dist / "THIRD_PARTY_NOTICES.txt")], cwd=str(ROOT))


def make_zip(dist: Path, zip_path: Path, root_name: str) -> Path:
    """dist を <root_name>/ というフォルダ名で zip にする."""
    tmp = zip_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(dist.rglob("*")):
            if path.is_file():
                zf.write(path, f"{root_name}/{path.relative_to(dist).as_posix()}")
    tmp.replace(zip_path)
    return zip_path


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-dir", default=str(ROOT / "build" / "exe"),
                        help="ビルド先（OneDrive の外を推奨。例: C:\\build\\axicavity）")
    parser.add_argument("--version", default=None, help="版（既定: axicavity_fem.gui.GUI_VERSION）")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--msvc", default=None, help="MSVC を使う（例: latest）")
    parser.add_argument("--zig", action="store_true", help="zig を C コンパイラにする")
    parser.add_argument("--skip-nuitka", action="store_true", help="コンパイルを飛ばして後処理と zip だけ")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs="*", help="Nuitka へそのまま渡す追加オプション")
    args = parser.parse_args(argv)
    args.version = args.version or app_version()
    build_dir = Path(args.build_dir).resolve()
    dist = build_dir / (MAIN_SCRIPT.stem + ".dist")
    if "onedrive" in str(build_dir).lower():
        print("注意: ビルド先が OneDrive 配下です。--build-dir で外を指定することを勧めます")

    icon = build_dir / "app.ico"
    cmd = nuitka_command(args, build_dir, icon)
    print("Nuitka コマンド:\n  " + "\n  ".join(cmd))
    if args.dry_run:
        return 0
    build_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    subprocess.check_call([sys.executable, str(ROOT / "packaging" / "make_ico.py"), str(icon)], cwd=str(ROOT))
    mesher = build_dir / "mesher"
    code = subprocess.call([sys.executable, str(ROOT / "packaging" / "build_mesher.py"), "--out", str(mesher),
                            "--download"], cwd=str(ROOT))
    if code != 0:
        print("メッシャを組み立てられませんでした", file=sys.stderr)
        return code

    zig_cache = build_dir / "zig-cache"
    if not args.skip_nuitka:
        env = dict(os.environ, ZIG_LOCAL_CACHE_DIR=str(zig_cache))   # 他のプロジェクトのビルドと共有しない
        code = subprocess.call(cmd, cwd=str(ROOT), env=env)
        if code != 0:
            print(f"Nuitka が失敗しました（終了コード {code}）", file=sys.stderr)
            return code
        print(f"Nuitka 完了（{(time.perf_counter() - t0) / 60:.1f} 分）")
    if not (dist / EXE_NAME).exists():
        print(f"{dist / EXE_NAME} がありません", file=sys.stderr)
        return 1
    if not constants_embedded(build_dir, dist / EXE_NAME):
        print(f"{EXE_NAME} に今回のビルドの定数が入っていません（zig のキャッシュが別のビルドのものを返した）。"
              f"{zig_cache} を消して作り直してください", file=sys.stderr)
        return 1
    gpl_qt = gpl_only_qt_dlls(dist)
    if gpl_qt:
        print("GPL のみの Qt モジュールが入っています（NOFOLLOW に足して作り直す）: " + ", ".join(gpl_qt),
              file=sys.stderr)
        return 1

    assemble(dist, mesher, build_dir, args.version)
    print(f"dist: {dist}（{size_mb(dist):.0f} MB）")
    if not args.no_zip:
        root_name = f"AxiCavity-FEM-{args.version}-win64"
        zip_path = make_zip(dist, build_dir / f"{root_name}.zip", root_name)
        print(f"zip: {zip_path}（{size_mb(zip_path):.0f} MB）")
    print(f"完了（{(time.perf_counter() - t0) / 60:.1f} 分）。確認: python packaging/smoke_test_exe.py {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
