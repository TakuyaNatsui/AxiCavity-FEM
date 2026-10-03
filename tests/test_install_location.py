"""editable インストールの実体がこのフォルダを指していることを確かめる（開発用の確認）.

import 名 ``axicavity_fem`` は v2.x の版と同じなので、別フォルダの版が入っていると
テストは黙ってそちらを検証してしまう。直し方: このフォルダで ``pip install -e ".[gui,viz3d,dev]"``
（同じパッケージ名 ``axicavity-fem`` の別フォルダの版は上書きされる）。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import axicavity_fem


def test_editable_install_points_to_this_tree():
    expected = (Path(__file__).resolve().parents[1] / "src" / "axicavity_fem").resolve()
    installed = Path(inspect.getfile(axicavity_fem)).resolve().parent
    assert installed == expected, (
        f"axicavity_fem の実体が {installed} を指している（期待: {expected}）。"
        "このフォルダで editable install をやり直すこと（モジュール docstring 参照）")
