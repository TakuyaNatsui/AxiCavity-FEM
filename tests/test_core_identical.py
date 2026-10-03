"""計算コアが変わっていないことを確かめる（v3 は GUI だけを作り直し、計算コアは v2.3.0 と同じコードを使う）.

対象: ``src/axicavity_fem/{shared,physics_core,fem_tm0,fem_hom,reports,cli}/**/*.py`` と ``src/axicavity_fem/__init__.py``。
``core_files.sha256`` は 3.0.0 公開時のコアの SHA-256（改行は LF に正規化。Windows の clone で CRLF になっても同じ値）。
GitHub のタグ v2.3.0 との違いは ``__init__.py`` の ``__version__``（と docstring 1 行）だけ。

コアを意図的に直したときは ``python tests/test_core_identical.py --update`` で一覧を作り直し、
CHANGELOG に理由を書くこと（黙って変えない）。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "axicavity_fem"
MANIFEST = Path(__file__).resolve().parent / "core_files.sha256"
CORE_PACKAGES = ("shared", "physics_core", "fem_tm0", "fem_hom", "reports", "cli")


def core_digests(root: Path = CORE_ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    files = [root / "__init__.py"]
    for pkg in CORE_PACKAGES:
        files += [p for p in sorted((root / pkg).rglob("*.py")) if "__pycache__" not in p.parts]
    for p in files:
        data = p.read_bytes().replace(b"\r\n", b"\n")
        out[p.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    return out


def read_manifest(path: Path = MANIFEST) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def write_manifest(path: Path = MANIFEST) -> None:
    lines = ["# 計算コアの SHA-256（改行は LF に正規化）。tests/test_core_identical.py --update で作り直す"]
    lines += [f"{digest}  {name}" for name, digest in sorted(core_digests().items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_core_files_unchanged():
    expected, actual = read_manifest(), core_digests()
    assert set(actual) == set(expected), (
        f"増えたファイル: {sorted(set(actual) - set(expected))} / 無くなったファイル: {sorted(set(expected) - set(actual))}")
    changed = sorted(name for name in actual if actual[name] != expected[name])
    assert changed == [], f"内容が変わったコアファイル: {changed}（意図した変更ならモジュール docstring の手順で一覧を更新）"


if __name__ == "__main__":
    if "--update" in sys.argv[1:]:
        write_manifest()
        print(f"更新しました: {MANIFEST}（{len(core_digests())} ファイル）")
    else:
        print(__doc__)
