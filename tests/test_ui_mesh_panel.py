"""メッシュタブの右パネル（ui/mesh_panel.py）: 設定（lc は式可・次数）、状態、統計、表示（offscreen、pytest-qt）.

メッシャは走らせない（生成済みのフォルダを合成して読ませる。実際の生成は test_mesh_controller.py）。
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtGui, QtWidgets                            # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language, tr          # noqa: E402
from axicavity_fem.gui.jobs.mesh_controller import MeshController  # noqa: E402
from axicavity_fem.gui.jobs.session import JobSession           # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402
from axicavity_fem.gui.ui.mesh_panel import MeshPanel           # noqa: E402
from test_mesh_controller import draw_box, fake_generation     # noqa: E402


@pytest.fixture
def env(qtbot, tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    set_language("ja")
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)
    store.new_document()
    session = JobSession()
    mesh = MeshController(store, session)
    panel = MeshPanel(store, mesh)
    qtbot.addWidget(panel)
    panel.resize(400, 800)
    panel.show()
    yield store, controller, mesh, panel
    session.shutdown()


def test_mesh_panel_settings_state_and_stats(env, tmp_path):
    store, controller, mesh, panel = env
    assert not panel.generate_button.isEnabled()                  # 閉領域が無い
    assert panel.state_label.text() == tr("mesh.state.none")
    assert not panel.stats_box.isVisibleTo(panel) and not panel.show_check.isEnabled()
    assert "mm" in panel.size_label.text()

    draw_box(store, controller)
    assert panel.generate_button.isEnabled()
    requested = []
    panel.generate_requested.connect(lambda: requested.append(1))
    panel.generate_button.click()
    assert requested == [1]

    # メッシュサイズ: 数値、式（プレビュー）、不正な値は戻す
    panel.size_edit.setText("4")
    panel.size_edit.editingFinished.emit()
    assert (store.document.mesh.size, store.document.mesh.sizeExpr) == (4.0, None)
    pid = store.add_param()
    store.update_param(pid, name="R", expression="50")
    panel.size_edit.setText("R/10")
    panel.size_edit.textEdited.emit("R/10")
    assert panel.size_preview.text() == "= 5.000000"
    panel.size_edit.editingFinished.emit()
    assert (store.document.mesh.size, store.document.mesh.sizeExpr) == (5.0, "R/10")
    assert panel.size_edit.text() == "R/10"
    store.update_param(pid, expression="100")
    assert store.document.mesh.size == 10.0                       # パラメータに追従
    panel.size_edit.setText("-1")
    panel.size_edit.editingFinished.emit()
    assert store.document.mesh.size == 10.0 and panel.size_edit.text() == "R/10"
    panel.size_edit.setText("foo+")
    panel.size_edit.textEdited.emit("foo+")
    assert panel.size_preview.text() == "= ?"
    panel.size_edit.editingFinished.emit()
    assert store.document.mesh.sizeExpr == "R/10"
    panel.order_combo.setCurrentIndex(panel.order_combo.findData(1))
    assert store.document.mesh.order == 1
    panel.order_combo.setCurrentIndex(panel.order_combo.findData(2))
    assert store.document.mesh.order == 2

    # 生成済みメッシュを読む → 最新・統計・境界条件の内訳
    folder = fake_generation(mesh, tmp_path / "mesh")
    assert mesh.load(tmp_path / "mesh")
    assert panel.state_label.text() == tr("mesh.state.current")
    assert panel.stats_box.isVisibleTo(panel) and panel.folder_button.isEnabled()
    doc = QtGui.QTextDocument()
    doc.setHtml(panel.stats_label.text())
    text = doc.toPlainText()
    assert "4" in text and "25" in text and "Vacuum" in text
    assert panel.bc_list.count() == 4 and "PEC" in panel.bc_list.item(0).text()

    # 表示のチェックとコントローラの表示は連動
    panel.show_check.setChecked(True)
    assert mesh.visible
    mesh.set_visible(False)
    assert not panel.show_check.isChecked()

    # メッシュの設定を変えると「要再生成」（再利用しない）
    panel.size_edit.setText("3")
    panel.size_edit.editingFinished.emit()
    assert mesh.state() == "settings" and panel.state_label.text() == tr("mesh.state.settings")
    store.undo()
    assert mesh.state() == "current"
    assert folder.exists()

    # 形状を変えると「形状が変わっています」
    draw_box(store, controller, 200, 0, 240, 20)
    assert panel.state_label.text() == tr("mesh.state.geometry")
    mesh.reset()
    assert panel.state_label.text() == tr("mesh.state.none") and not panel.stats_box.isVisibleTo(panel)
    panel.retranslate()
