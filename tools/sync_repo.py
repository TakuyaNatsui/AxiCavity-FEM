"""作業フォルダの公開対象のファイルを、GitHub のリポジトリの clone（別フォルダ）へ反映する（保守用）.

    python tools/sync_repo.py C:\\Users\\<you>\\repos\\AxiCavity-FEM --dry-run   # 何が変わるかだけ表示
    python tools/sync_repo.py C:\\Users\\<you>\\repos\\AxiCavity-FEM             # 反映（git の操作はしない）

作業フォルダ（OneDrive など）は git 管理の外に置き、公開は clone で行う運用のためのもの。公開対象は
``.gitignore`` に従う（clone の git に判定させる）。作業フォルダに無くなった追跡中のファイルは clone から消す。
反映のあと clone で ``git status`` / ``git add -A`` / ``git commit`` / ``git push`` を行う。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".pytest_cache", ".venv", "venv", ".idea", ".vscode"}
SKIP_DIR_SUFFIXES = (".egg-info", ".axiproj.data", "_report", "_batch")


def git(clone: Path, *args: str, stdin: bytes | None = None) -> str:
    # バイト列でやり取りする（Windows のテキストモードだと標準入力の改行が CRLF になり、パスに CR が付いて照合できない）
    result = subprocess.run(["git", "-C", str(clone), *args], input=stdin, capture_output=True)
    if result.returncode not in (0, 1):                  # check-ignore は「無視されない」を 1 で返す
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout.decode("utf-8")


def candidate_files() -> list[str]:
    out = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(SKIP_DIR_SUFFIXES) for part in rel.parts[:-1]):
            continue
        if path.is_file() and not path.name.endswith((".pyc", ".pyo")):
            out.append(rel.as_posix())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clone", help="GitHub のリポジトリの clone")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    clone = Path(args.clone).resolve()
    if not (clone / ".git").is_dir():
        print(f"git の clone ではありません: {clone}", file=sys.stderr)
        return 1
    if not args.dry_run:                                  # 判定は作業フォルダの .gitignore で行う
        for name in (".gitignore", ".gitattributes"):
            if (ROOT / name).exists():
                shutil.copy2(ROOT / name, clone / name)
    files = candidate_files()
    # 判定には作業フォルダの .gitignore を使う（--dry-run で clone の .gitignore がまだ古くても正しく判定する）
    out = git(clone, "-c", f"core.excludesFile={(ROOT / '.gitignore').as_posix()}", "check-ignore", "--no-index",
              "--stdin", "-z", stdin=("\0".join(files) + "\0").encode("utf-8"))
    ignored = {f for f in out.split("\0") if f}
    publish = [f for f in files if f not in ignored]
    tracked = {f for f in git(clone, "ls-files", "-z").split("\0") if f}

    added = changed = 0
    for rel in publish:
        src, dst = ROOT / rel, clone / rel
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        if dst.exists():
            changed += 1
        else:
            added += 1
        print(("更新 " if dst.exists() else "追加 ") + rel)
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    removed = sorted(tracked - set(publish))
    for rel in removed:
        print("削除 " + rel)
        if not args.dry_run:
            (clone / rel).unlink(missing_ok=True)
    print(f"\n公開対象 {len(publish)} ファイル（無視 {len(ignored)}）: 追加 {added}、更新 {changed}、削除 {len(removed)}"
          + ("（--dry-run: 何も変えていません）" if args.dry_run else ""))
    if not args.dry_run:
        print("次: clone で git status → git add -A → git commit → git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
