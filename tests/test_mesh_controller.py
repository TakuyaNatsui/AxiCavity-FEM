"""MeshController（gui/jobs/mesh_controller.py）: 生成済みフォルダの読込と状態（最新 / 設定変更 / 形状変更）、再利用、
プレビューの単位換算、表示の自動オフ、実際の生成（子プロセス）と古い世代の削除.
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtWidgets                                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.core.document import SketchLine          # noqa: E402
from axicavity_fem.gui.core.sketch.model import point_pos       # noqa: E402
from axicavity_fem.gui.jobs.mesh_controller import MESH_KEY_FILE, MeshController  # noqa: E402
from axicavity_fem.gui.jobs.mesh_folder import generations, new_generation_dir  # noqa: E402
from axicavity_fem.gui.jobs.pipeline import MESH_PREVIEW_FILE, MESH_SUMMARY_FILE  # noqa: E402
from axicavity_fem.gui.jobs.session import JobSession           # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402


@pytest.fixture
def env(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)
    store.new_document()
    session = JobSession()
    mesh = MeshController(store, session)
    yield store, controller, session, mesh
    session.shutdown()


def click(controller, x, y):
    controller.pointer_move((x, y))
    controller.pointer_down((x, y))
    controller.pointer_up((x, y))


def draw_box(store, controller, x0=0, y0=0, x1=100, y1=50):
    store.set_tool("rectangle")
    click(controller, x0, y0)
    click(controller, x1, y1)
    store.set_tool("select")


def fake_generation(mesh: MeshController, root, units="mm"):
    """子プロセスが書く mesh_summary.json / mesh_preview.npz（m 単位）と、GUI が書く mesh_key.json を
    現在のモデルのハッシュで作る."""
    folder = new_generation_dir(root)
    stats = {"nodes": 6, "elements": 4, "order": 1, "meanEdge": 25.0, "units": units,
             "bc": {"PEC": 3, "E-short": 0, "M-short": 0, "None": 2}, "interfaces": 0,
             "regions": [{"tag": "vacuum", "name": "Vacuum", "elements": 4}],
             "bounds": [[0, 0], [100, 50]]}
    (folder / MESH_SUMMARY_FILE).write_text(json.dumps(
        {"kind": "mesh", "units": units, "stats": stats, "warnings": [], "elapsedS": {"meshing": 0.5}}),
        encoding="utf-8")
    nodes = np.array([[0, 0], [0.05, 0], [0.1, 0], [0, 0.05], [0.05, 0.05], [0.1, 0.05]])
    np.savez(folder / MESH_PREVIEW_FILE, nodes=nodes, simplices=np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]),
             elem_order=np.int32(1), region_ids=np.zeros(4, np.int32),
             bc_PEC=np.array([[[0, 0.05], [0.05, 0.05]], [[0.05, 0.05], [0.1, 0.05]], [[0.1, 0], [0.1, 0.05]]]),
             **{"bc_E-short": np.zeros((0, 2, 2)), "bc_M-short": np.zeros((0, 2, 2))},
             bc_None=np.array([[[0, 0], [0.05, 0]], [[0.05, 0], [0.1, 0]]]), interfaces=np.zeros((0, 2, 2)))
    geom, mesh_h = mesh.current_hashes()
    (folder / MESH_KEY_FILE).write_text(json.dumps(
        {"geometryHash": geom, "meshHash": mesh_h, "createdAt": "2026-09-25T12:00:00"}), encoding="utf-8")
    (folder / "model.msh").write_text("dummy", encoding="utf-8")
    return folder


def test_load_state_reuse_and_preview(env, tmp_path):
    store, controller, session, mesh = env
    changes = []
    mesh.changed.connect(lambda: changes.append(mesh.state()))
    assert mesh.state() == "none" and mesh.reusable_mesh() is None and mesh.preview_geometry() is None
    draw_box(store, controller)
    root = tmp_path / "mesh"
    assert not mesh.load(root)                                    # まだ無い
    folder = fake_generation(mesh, root)
    assert mesh.load(root) and mesh.info is not None and mesh.info.dir == folder
    assert mesh.state() == "current" and mesh.reusable_mesh() == folder / "model.msh"
    assert mesh.info.stats["elements"] == 4 and mesh.info.created_at.startswith("2026")

    # プレビューは文書の単位（mm）に換算される
    geometry = mesh.preview_geometry()
    assert len(geometry["triangles"]) == 4 and geometry["triangles"][0][1] == pytest.approx((50, 0))
    assert len(geometry["bc_segments"]["PEC"]) == 3 and geometry["bc_segments"]["None"][1][1] == pytest.approx((100, 0))
    assert "E-short" not in geometry["bc_segments"] and geometry["interfaces"] == []
    assert mesh.preview_geometry() is geometry                    # キャッシュ

    # 表示
    assert not mesh.visible
    mesh.set_visible(True)
    assert mesh.visible

    # メッシュの設定・境界条件を変えると「設定変更」（再利用しない）。材料は関係しない。Undo で戻る
    store.set_mesh(size=3.0)
    assert mesh.state() == "settings" and mesh.reusable_mesh() is None and mesh.visible
    store.undo()
    assert mesh.state() == "current"
    sketch = store.active_sketch()
    top = next(e for e in sketch.entities if isinstance(e, SketchLine)
               and point_pos(sketch, e.p1)[1] == 50 and point_pos(sketch, e.p2)[1] == 50)
    store.set_boundary([top.id], "E-short")
    assert mesh.state() == "settings"
    store.undo()
    store.set_region(store.profiles[0].id, epsR="4")
    assert mesh.state() == "current"

    # 形状を変えると「形状変更」で表示が消える
    store.set_tool("select")
    controller.pointer_move((100, 50))
    controller.pointer_down((100, 50))
    controller.pointer_move((110, 60))
    controller.pointer_up((110, 60))
    assert mesh.state() == "geometry" and not mesh.visible
    store.undo()
    assert mesh.state() == "current"
    mesh.reset()
    assert mesh.info is None and mesh.state() == "none" and not mesh.visible


def test_generate_rejects_invalid_geometry(env, tmp_path):
    store, controller, session, mesh = env
    assert not mesh.generate(tmp_path / "mesh")                   # 閉領域が無い
    assert mesh.last_error and not session.running
    assert not (tmp_path / "mesh").exists() or generations(tmp_path / "mesh") == []


@pytest.mark.skipif(pytest.importorskip("gmsh") is None, reason="gmsh")
def test_generate_real_mesh_and_cleanup(env, tmp_path, qtbot):
    store, controller, session, mesh = env
    draw_box(store, controller)
    store.set_mesh(size=10.0)
    root = tmp_path / "p.axiproj.data" / "mesh"
    old = fake_generation(mesh, root)                             # 古い世代（成功後に消える）
    assert mesh.load(root) and mesh.info.dir == old
    mesh.set_visible(True)
    assert mesh.generate(root), mesh.last_error
    assert mesh.running and mesh.info is None and not mesh.visible
    qtbot.waitUntil(lambda: not mesh.running, timeout=180_000)
    assert mesh.job.status == "done", (mesh.job.error, mesh.job.traceback)
    assert mesh.info is not None and mesh.info.dir != old and mesh.state() == "current"
    assert mesh.visible and (mesh.info.dir / MESH_KEY_FILE).exists()
    assert mesh.reusable_mesh() == mesh.info.dir / "model.msh" and mesh.info.materials_path.exists()
    assert generations(root) == [mesh.info.dir]                   # 古い世代は消えた
    geometry = mesh.preview_geometry()
    assert len(geometry["triangles"]) == mesh.info.stats["elements"] > 20
    xs = [p[0] for tri in geometry["triangles"] for p in tri]
    assert max(xs) == pytest.approx(100, abs=1e-6)                # mm に換算
    assert len(geometry["bc_segments"]["PEC"]) == mesh.info.stats["bc"]["PEC"] > 0
    session.wait()
