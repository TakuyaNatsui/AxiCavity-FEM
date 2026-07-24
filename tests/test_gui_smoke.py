"""段階6 検証: GUI のインポート・構築・CLI コマンド配線.

ウィンドウ表示や MainLoop は行わず、フレーム構築と OnRunSolver/OnRunPost が
新統一 CLI (`axicavity-fem ...`) のコマンドを正しく組み立てることを確認する。
ディスプレイが無い環境では skip する。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
wx = pytest.importorskip("wx")

_SAMPLES = Path(__file__).resolve().parents[1] / "samples"
_MESH = _SAMPLES / "cylinder100mm.msh"
_MESH_SPHERE = _SAMPLES / "sphere50mm.msh"


@pytest.fixture
def frame():
    """ヘッドレスで MyFrame を構築する（表示しない）。失敗時は skip。"""
    from axicavity_fem.gui.main_frame import MyFrame
    try:
        app = wx.App(False)
        frm = MyFrame(None, wx.ID_ANY, "")
    except Exception as e:  # ディスプレイ無し等
        pytest.skip(f"wx GUI を構築できません: {e}")
    yield frm
    frm.Destroy()
    app.Destroy()


def test_gui_imports():
    from axicavity_fem.gui import (  # noqa: F401
        app, main_frame, main_frame_ui, multi_region_editor, result_viewer_ui)


def test_build_base_cmd(frame):
    base = frame._build_base_cmd()
    assert isinstance(base, list) and len(base) >= 1
    # axicavity-fem 実行ファイル、または python -m フォールバック
    assert base[0].lower().endswith(("axicavity-fem", "axicavity-fem.exe")) \
        or base[-1] == "axicavity_fem.cli.main"


def test_wave_type_change_clears_result_fields(frame):
    """Wave Type 切替でも Raw / Processed の結果パス欄がクリアされる (v2.3)。"""
    frame.fem_raw_result_path_ctrl.SetValue("old_raw.h5")
    frame.fem_processed_result_path_ctrl.SetValue("old_proc.h5")
    frame.fem_wave_type_radio.SetSelection(1)   # Traveling
    frame.on_fem_wave_type_radio(None)
    assert frame.fem_raw_result_path_ctrl.GetValue() == ""
    assert frame.fem_processed_result_path_ctrl.GetValue() == ""


def test_export_default_filename_uses_project_name(frame):
    """Export ダイアログの既定ファイル名は project 名に一致する (v2.3)。"""
    # 未保存なら空
    frame.mr_current_project_path = None
    assert frame._export_default(".msh") == ("", "")
    # 保存済みなら <stem>.msh
    frame.mr_current_project_path = os.path.join("some", "dir", "cavity.gmshproj")
    dd, df = frame._export_default(".msh")
    assert df == "cavity.msh"
    assert dd.endswith("dir")


@pytest.mark.skipif(not _MESH.exists(), reason="cylinder100mm.msh なし")
def test_onrunsolver_builds_cli_cmd(frame, monkeypatch):
    captured = {}
    monkeypatch.setattr(frame, "run_command_async",
                        lambda cmd, label: captured.update(cmd=cmd, label=label))

    frame.fem_mesh_path_ctrl.SetValue(str(_MESH))
    frame.radio_box_analysis_mode.SetSelection(0)   # TM0
    frame.radio_mesh_order.SetSelection(1)          # 2nd
    frame.fem_wave_type_radio.SetSelection(0)       # Standing
    frame.num_modes_ctrl.SetValue(6)
    frame.fem_raw_result_path_ctrl.SetValue(str(_MESH.with_suffix(".tm0.h5")))

    frame.OnRunSolver(None)
    cmd = captured["cmd"]
    assert "solve" in cmd and "--type" in cmd
    assert cmd[cmd.index("--type") + 1] == "tm0"
    assert "-m" in cmd and str(_MESH) in cmd
    assert "--elem-order" in cmd and cmd[cmd.index("--elem-order") + 1] == "2"


@pytest.mark.skipif(not _MESH.exists(), reason="cylinder100mm.msh なし")
def test_onrunsolver_hom_traveling_cmd(frame, monkeypatch):
    captured = {}
    monkeypatch.setattr(frame, "run_command_async",
                        lambda cmd, label: captured.update(cmd=cmd))

    frame.fem_mesh_path_ctrl.SetValue(str(_MESH))
    frame.radio_box_analysis_mode.SetSelection(1)   # HOM
    frame.hom_order_ctrl.SetValue("0 1 2")
    frame.fem_wave_type_radio.SetSelection(1)        # Traveling
    frame.fem_phase_shift_ctrl.SetValue("120")
    frame.fem_raw_result_path_ctrl.SetValue(str(_MESH.with_suffix(".hom.h5")))

    frame.OnRunSolver(None)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--type") + 1] == "hom"
    assert "--az-order" in cmd
    az_i = cmd.index("--az-order")
    assert cmd[az_i + 1:az_i + 4] == ["0", "1", "2"]
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "120"


@pytest.mark.skipif(not _MESH_SPHERE.exists(), reason="sphere50mm.msh なし")
def test_onrunsolver_predicted_frequency(frame, monkeypatch, tmp_path):
    """Predicted frequency 欄 → --target-freq の配線 (v2.3)。

    空欄なら引数を付けない（従来挙動）、入力があれば付ける、
    数値でなければコマンドを起動せずエラーにする。
    """
    captured = {}
    monkeypatch.setattr(frame, "run_command_async",
                        lambda cmd, label: captured.update(cmd=cmd))
    frame.fem_mesh_path_ctrl.SetValue(str(_MESH_SPHERE))
    frame.radio_box_analysis_mode.SetSelection(0)   # TM0
    frame.fem_wave_type_radio.SetSelection(0)       # Standing
    frame.fem_raw_result_path_ctrl.SetValue(str(tmp_path / "out.h5"))

    # 1) 空欄 → 引数なし
    frame.predicted_frequency_ctrl.SetValue("")
    frame.OnRunSolver(None)
    assert "--target-freq" not in captured["cmd"]

    # 2) 入力あり → 引数あり
    captured.clear()
    frame.predicted_frequency_ctrl.SetValue(" 2.856 ")
    frame.OnRunSolver(None)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--target-freq") + 1] == repr(2.856)

    # 3) 不正入力 → 起動しない
    captured.clear()
    msgs = []
    monkeypatch.setattr("axicavity_fem.gui.main_frame.wx.MessageBox",
                        lambda *a, **k: msgs.append(a))
    frame.predicted_frequency_ctrl.SetValue("abc")
    frame.OnRunSolver(None)
    assert captured == {} and msgs


@pytest.mark.skipif(not _MESH.exists(), reason="cylinder100mm.msh なし")
def test_onrunpost_builds_cli_cmd(frame, monkeypatch, tmp_path):
    raw = tmp_path / "raw.h5"
    raw.write_bytes(b"dummy")   # 存在チェック用
    captured = {}
    monkeypatch.setattr(frame, "run_command_async",
                        lambda cmd, label: captured.update(cmd=cmd))

    frame.radio_box_analysis_mode.SetSelection(0)   # TM0
    frame.fem_raw_result_path_ctrl.SetValue(str(raw))
    frame.fem_conductivity_ctrl.SetValue("5.8e7")
    frame.fem_beta_ctrl.SetValue("1.0")

    frame.OnRunPost(None)
    cmd = captured["cmd"]
    assert "post" in cmd and cmd[cmd.index("--type") + 1] == "tm0"
    assert "-i" in cmd and str(raw) in cmd
    assert "--cond" in cmd and "--beta" in cmd
