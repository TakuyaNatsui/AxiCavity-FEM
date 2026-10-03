"""メッシュ生成の入口（gui/jobs/meshing.py）と、別プロセスのメッシャ（mesher/axicavity_mesh.py）.

Windows 版（EXE）は gmsh を本体に入れず、メッシャ（埋め込み Python + gmsh）を別プロセスで動かす。開発環境では
``AXICAVITY_MESHER=process`` で同じ経路（この Python + リポジトリの ``mesher/axicavity_mesh.py``）を通せるので、
プロセス内で作ったメッシュと同じになること、エラーの伝わり方、GUI のメッシュジョブがこの経路で動くことを確かめる。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("gmsh")

from axicavity_fem.gui.jobs import meshing, runner  # noqa: E402
from axicavity_fem.shared.mesh_loader import load_mesh  # noqa: E402
from axicavity_fem.shared.multi_region_model import MultiRegionGeometry  # noqa: E402

from conftest import SAMPLES  # noqa: E402


@pytest.mark.parametrize("name", ["cylinder100mm", "diel_test1"])
def test_mesher_process_makes_the_same_mesh(name, tmp_path, monkeypatch):
    geom = MultiRegionGeometry.load_json(SAMPLES / f"{name}.gmshproj")
    monkeypatch.setenv("AXICAVITY_MESHER", "inprocess")
    inproc = meshing.generate_msh(geom, tmp_path / "in.msh", mesh_order=2, verbose=0)
    monkeypatch.setenv("AXICAVITY_MESHER", "process")
    lines: list[str] = []
    child = meshing.generate_msh(geom, tmp_path / "out" / "child.msh", mesh_order=2, verbose=1, log=lines.append)
    assert child.exists() and any("メッシャ:" in line for line in lines)
    assert not list((tmp_path / "out").glob("*.mesher*"))          # 受け渡しの一時ファイルは消える
    a, b = load_mesh(inproc, 2), load_mesh(child, 2)
    assert (a.nodes == b.nodes).all() and (a.elements == b.elements).all()
    assert a.physical_groups.keys() == b.physical_groups.keys()


def test_mesher_value_error_comes_back_as_value_error(tmp_path, monkeypatch):
    geom = MultiRegionGeometry.load_json(SAMPLES / "cylinder100mm.gmshproj")
    monkeypatch.setenv("AXICAVITY_MESHER", "process")
    with pytest.raises(ValueError, match="meshOrder"):
        meshing.generate_msh(geom, tmp_path / "x.msh", mesh_order=3)


def test_missing_mesher_folder_is_reported(tmp_path, monkeypatch):
    geom = MultiRegionGeometry.load_json(SAMPLES / "cylinder100mm.gmshproj")
    monkeypatch.setenv("AXICAVITY_MESHER", "process")
    monkeypatch.setenv("AXICAVITY_MESHER_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(meshing.MesherError, match="メッシャが見つかりません"):
        meshing.generate_msh(geom, tmp_path / "x.msh")
    assert meshing.mesher_version().startswith("なし")


def test_mesher_version():
    text = meshing.mesher_version()
    assert "axicavity_mesh" in text and "gmsh 4." in text


def test_gui_mesh_job_through_the_mesher(tmp_path):
    """GUI のメッシュジョブ（子プロセス）がメッシャ（孫プロセス）でメッシュを作る."""
    from axicavity_fem.gui.jobs.pipeline import GEOMETRY_FILE, JOB_FILE

    geom = MultiRegionGeometry.load_json(SAMPLES / "diel_test1.gmshproj")
    geom.save_json(tmp_path / GEOMETRY_FILE)
    (tmp_path / JOB_FILE).write_text(json.dumps({"kind": "mesh", "meshOrder": 2}), encoding="utf-8")
    env = runner.child_environment()
    env["AXICAVITY_MESHER"] = "process"
    result = subprocess.run(runner.job_command("mesh", tmp_path), capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, cwd=str(tmp_path), timeout=300)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert events[-1]["event"] == "done"
    assert (tmp_path / "model.msh").exists() and (tmp_path / "model.materials.json").exists()
    assert any(e.get("event") == "log" and "メッシャ:" in e.get("line", "") for e in events)
    assert runner.program_command()[0] == sys.executable           # 開発環境では python -m …runner
