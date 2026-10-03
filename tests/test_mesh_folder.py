"""メッシュタブの生成物の置き場所（gui/jobs/mesh_folder.py）: 生成ごとのサブフォルダ・今のメッシュの判定・古いものの削除.

移植元: EM-CAD-py tests/test_em_mesh_folder.py（同じフォルダを消して作り直すと PermissionError になる問題への対応）。
"""

import datetime as dt
import json
import os

import pytest

from axicavity_fem.gui.jobs.mesh_folder import (
    MESH_KEY_FILE,
    current_generation,
    generations,
    is_complete,
    new_generation_dir,
    remove_stale,
)
from axicavity_fem.gui.jobs.pipeline import MESH_SUMMARY_FILE

T0 = dt.datetime(2026, 9, 25, 16, 15, 30)


def _complete(folder):
    """成功した生成に見せる（要約 + 完了の印）."""
    (folder / MESH_SUMMARY_FILE).write_text(json.dumps({"stats": {}}), encoding="utf-8")
    (folder / MESH_KEY_FILE).write_text(json.dumps({"meshHash": "h"}), encoding="utf-8")
    (folder / "model.msh").write_text("mesh", encoding="utf-8")
    return folder


def test_new_generation_dir_is_unique_and_ordered(tmp_path):
    root = tmp_path / "p.axiproj.data" / "mesh"
    a = new_generation_dir(root, now=T0)
    b = new_generation_dir(root, now=T0)                    # 同じ秒 → -2
    c = new_generation_dir(root, now=T0)
    later = new_generation_dir(root, now=T0 + dt.timedelta(seconds=1))
    assert (a.name, b.name, c.name, later.name) == (
        "20260925-161530", "20260925-161530-2", "20260925-161530-3", "20260925-161531")
    assert all(p.is_dir() and not any(p.iterdir()) for p in (a, b, c, later))
    for _ in range(4, 11):                                  # -10 は -9 より後（文字列でなく番号で並べる）
        new_generation_dir(root, now=T0)
    names = [p.name for p in generations(root)]
    assert names[-2:] == ["20260925-161530-10", "20260925-161531"]
    (root / "notes").mkdir()                                # 利用者が置いたフォルダは生成とみなさない
    assert all(p.name != "notes" for p in generations(root))


def test_current_generation_is_newest_complete(tmp_path):
    root = tmp_path / "mesh"
    assert current_generation(root) is None                  # mesh/ が無い
    root.mkdir()
    assert current_generation(root) is None
    old = _complete(new_generation_dir(root, now=T0))
    assert current_generation(root) == old
    failed = new_generation_dir(root, now=T0 + dt.timedelta(minutes=1))
    (failed / "geometry.json").write_text("{}", encoding="utf-8")
    assert not is_complete(failed)
    assert current_generation(root) == old                   # 失敗・中止した生成は無視（直前のメッシュのまま）
    new = _complete(new_generation_dir(root, now=T0 + dt.timedelta(minutes=2)))
    assert current_generation(root) == new


def test_remove_stale_keeps_current_and_user_files(tmp_path):
    root = tmp_path / "mesh"
    root.mkdir()
    (root / "readme.txt").write_text("利用者のメモ", encoding="utf-8")
    old = _complete(new_generation_dir(root, now=T0))
    new = _complete(new_generation_dir(root, now=T0 + dt.timedelta(minutes=1)))
    assert remove_stale(root, keep=new) == []
    assert generations(root) == [new] and is_complete(new)
    assert not old.exists()
    assert (root / "readme.txt").exists()                   # 知らないファイルには触らない


@pytest.mark.skipif(os.name != "nt", reason="Windows のファイルロック（開いているファイルは消せない）")
def test_remove_stale_tolerates_files_in_use(tmp_path):
    root = tmp_path / "mesh"
    old = _complete(new_generation_dir(root, now=T0))
    new = _complete(new_generation_dir(root, now=T0 + dt.timedelta(minutes=1)))
    with open(old / "model.msh", "rb"):                     # Python の open は削除を共有しない = 消せない
        failed = remove_stale(root, keep=new)               # 例外にしない
        assert failed == [old]
        assert current_generation(root) == new
        newer = new_generation_dir(root, now=T0 + dt.timedelta(minutes=2))
        assert newer.is_dir()                                # 次の生成も問題なく始められる
    assert remove_stale(root, keep=new) == []                # 閉じた後の次の機会に消える
    assert generations(root) == [new]
