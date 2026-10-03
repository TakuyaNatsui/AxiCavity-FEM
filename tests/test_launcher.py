"""1 つの入口（gui/launcher.py。Windows 版 EXE の main）と、EXE 用の切り替え（frozen / runner / meshing）.

EXE そのもののテストは ``packaging/smoke_test_exe.py``。ここでは開発環境で同じ経路を通す:
``AXICAVITY_GMSH=lite``（.msh を gmsh 無しで読む）と ``AXICAVITY_MESHER=process``（メッシュ生成を別プロセスのメッシャで）
にすると、EXE と同じ構成（本体に gmsh を読み込まない）で GUI の子プロセス経路が動く。
"""

from __future__ import annotations

import os
import subprocess
import sys
import pytest

from axicavity_fem.gui import frozen, launcher
from axicavity_fem.gui.jobs import meshing, runner

from conftest import SAMPLES


def _run(args, env_extra=None, timeout=600):
    env = dict(os.environ, PYTHONUTF8="1", **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "axicavity_fem.gui.launcher", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=timeout)


def test_gui_request_detection():
    assert launcher._is_gui_request([])
    assert launcher._is_gui_request(["gui"])
    assert launcher._is_gui_request(["gui", "x.axiproj"])
    assert launcher._is_gui_request(["cavity.axiproj"])
    assert launcher._is_gui_request(["result.h5"])
    assert not launcher._is_gui_request(["solve", "--type", "tm0"])
    assert not launcher._is_gui_request(["run", "x.axiproj"])
    assert not launcher._is_gui_request(["x.axiproj", "--set", "a=1"])


def test_version_and_help(capsys):
    assert launcher.main(["--version"]) == 0
    assert "AxiCavity-FEM 3.0.0" in capsys.readouterr().out
    assert launcher.main(["help"]) == 0
    assert "selftest" in capsys.readouterr().out
    assert launcher.main(["no-such-command"]) == 2
    assert "不明なコマンド" in capsys.readouterr().err


def test_version_subcommand_lists_components():
    result = _run(["version"])
    assert result.returncode == 0, result.stderr
    for key in ("AxiCavity-FEM", "solver core", "PARDISO", "mesh", "3D view"):
        assert key in result.stdout


def test_core_cli_and_batch_pass_through(sample_mesh, tmp_path, capsys):
    msh = sample_mesh("cylinder100mm", 2)
    out = tmp_path / "pillbox.h5"
    assert launcher.main(["solve", "--type", "tm0", "-m", str(msh), "--elem-order", "2", "--num-modes", "2",
                          "-o", str(out)]) == 0
    assert out.exists()
    assert launcher.main(["info", "-i", str(out)]) == 0
    with pytest.raises(SystemExit):                          # argparse のエラーは CLI と同じ
        launcher.main(["solve", "--no-such-option"])
    assert launcher.main(["run", str(SAMPLES / "cylinder100mm.gmshproj"), "--modes", "2", "--no-post",
                          "--out", str(tmp_path / "batch"), "-q"]) == 0
    assert list((tmp_path / "batch").glob("*.h5"))


def test_compiled_mode_commands(monkeypatch, tmp_path):
    exe = tmp_path / "AxiCavity-FEM.exe"
    monkeypatch.setattr(frozen, "is_compiled", lambda: True)
    monkeypatch.setattr(frozen, "executable_path", lambda: str(exe))
    assert runner.program_command() == [str(exe), "job"]
    assert runner.job_command("analysis", "d") == [str(exe), "job", "analysis", "d", "--events"]
    assert frozen.app_dir() == tmp_path
    monkeypatch.delenv("AXICAVITY_MESHER", raising=False)
    monkeypatch.delenv("AXICAVITY_MESHER_DIR", raising=False)
    assert meshing.use_mesher_process()
    with pytest.raises(meshing.MesherError, match="mesher"):
        meshing.mesher_command()
    (tmp_path / "mesher").mkdir()
    (tmp_path / "mesher" / "python.exe").write_bytes(b"")
    (tmp_path / "mesher" / "axicavity_mesh.py").write_text("", encoding="utf-8")
    assert meshing.mesher_command() == [str(tmp_path / "mesher" / "python.exe"), "-B",
                                        str(tmp_path / "mesher" / "axicavity_mesh.py")]


def test_dev_app_dir_is_the_repository():
    assert (frozen.app_dir() / "pyproject.toml").exists()
    assert (frozen.app_dir() / "mesher" / "axicavity_mesh.py").exists()


@pytest.mark.parametrize("exe_like", [False, True], ids=["dev", "exe-like"])
def test_selftest(exe_like, tmp_path):
    """GUI と同じ子プロセス経路（JobSession → QProcess → 子の runner）で円筒空洞を解く.

    exe-like: EXE と同じく本体・子プロセスとも gmsh を使わず（mshlite）、メッシュは別プロセスのメッシャで作る。
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("gmsh")
    env = {"AXICAVITY_GMSH": "lite", "AXICAVITY_MESHER": "process"} if exe_like else {}
    result = _run(["selftest", "--work-dir", str(tmp_path)], env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "selftest → OK" in result.stdout
    assert "子プロセスのコマンド:" in result.stdout
    if exe_like:
        assert "メッシャ:" in result.stdout and "メッシャ（別プロセス）" in result.stdout
    assert (tmp_path / "pillbox" / "model_SW_TM0_processed.h5").exists()


def test_mshlite_is_used_only_when_requested(monkeypatch):
    import axicavity_fem.gui.mshlite as mshlite

    monkeypatch.delenv("AXICAVITY_GMSH", raising=False)
    launcher.prepare_runtime()
    assert not mshlite.is_installed()                     # 開発環境・pip では本物の gmsh を使う
