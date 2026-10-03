"""物理タブの右パネル（ui/physics_panel.py）: 領域表の編集（名前・材料タグ・εr 式・tanδ）、境界条件表と選択の連動、
選択曲線への適用、検証（offscreen、pytest-qt）."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("planegcs")
pytest.importorskip("pytestqt")
from PySide6 import QtWidgets                                   # noqa: E402

from axicavity_fem.gui.app.store import DocumentStore           # noqa: E402
from axicavity_fem.gui.core.document import SketchLine          # noqa: E402
from axicavity_fem.gui.core.sketch.model import point_pos       # noqa: E402
from axicavity_fem.gui.i18n.tr import set_language, tr          # noqa: E402
from axicavity_fem.gui.sketch.controller import SketchController  # noqa: E402
from axicavity_fem.gui.ui.physics_panel import PhysicsPanel     # noqa: E402
from test_mesh_controller import click, draw_box               # noqa: E402


@pytest.fixture
def env(qtbot):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    set_language("ja")
    store = DocumentStore()
    controller = SketchController(store)
    controller.set_scale(0.1)
    store.new_document()
    panel = PhysicsPanel(store)
    qtbot.addWidget(panel)
    panel.resize(420, 900)
    panel.show()
    return store, controller, panel


def test_regions_boundaries_and_validation(env):
    store, controller, panel = env
    assert panel.regions_empty.isVisibleTo(panel) and panel.boundaries_empty.isVisibleTo(panel)
    draw_box(store, controller, 0, 0, 50, 50)
    draw_box(store, controller, 50, 0, 100, 50)
    assert len(store.profiles) == 2
    assert panel.regions_table.rowCount() == 2 and panel.boundaries_table.rowCount() == 7
    assert panel.regions_table.item(0, 0).text().startswith("Vacuum")
    assert tr("physics.validation.ok") == panel.validation.text()

    # 領域の編集: 名前・材料タグ（正規化）・εr（式）・tanδ・μr
    right = next(p for p in store.profiles if p.anchor[0] > 50)
    row = panel._region_ids.index(right.id)
    panel.regions_table.item(row, 0).setText("Ceramic")
    panel.regions_table.item(row, 1).setText("Ceramic 1")
    pid = store.add_param()
    store.update_param(pid, name="er", expression="9.64")
    panel.regions_table.item(row, 2).setText("er")
    panel.regions_table.item(row, 4).setText("1e-4")
    setting = store.region_setting(right.id)
    assert (setting.name, setting.materialTag, setting.epsR, setting.epsRExpr, setting.tanDelta) == \
        ("Ceramic", "ceramic_1", 9.64, "er", 1e-4)
    assert panel.regions_table.item(row, 2).text() == "er"
    panel.regions_table.item(row, 3).setText("0")                   # 不正 → 戻す
    assert store.region_setting(right.id).muR == 1.0 and panel.regions_table.item(row, 3).text() == "1"
    panel.regions_table.item(row, 2).setText("er +")                # 評価できない → 戻す
    assert store.region_setting(right.id).epsRExpr == "er" and panel.regions_table.item(row, 2).text() == "er"
    panel._on_region_clicked(row, 0)
    assert store.selected_region_id == right.id
    panel._on_region_clicked(row, 0)
    assert store.selected_region_id is None

    # 境界条件表: 内部界面は None（自動）、行の選択で曲線を選択、適用で変わる
    sketch = store.active_sketch()
    shared = next(e for e in sketch.entities if isinstance(e, SketchLine)
                  and point_pos(sketch, e.p1)[0] == 50 and point_pos(sketch, e.p2)[0] == 50)
    row = panel._curve_ids.index(shared.id)
    assert panel.boundaries_table.item(row, 1).text() == "None"
    assert panel.boundaries_table.item(row, 2).text() == tr("physics.source.interface")
    top = next(e for e in sketch.entities if isinstance(e, SketchLine)
               and point_pos(sketch, e.p1)[1] == 50 and point_pos(sketch, e.p2)[1] == 50)
    top_row = panel._curve_ids.index(top.id)
    panel.boundaries_table.selectRow(top_row)
    assert store.selection == [top.id]
    assert panel.selected_label.text() == tr("physics.boundaries.selected", count=1)
    panel.bc_combo.setCurrentIndex(panel.bc_combo.findData("E-short"))
    panel.apply_button.click()
    assert store.effective_bc(top.id) == ("E-short", "explicit")
    assert panel.boundaries_table.item(top_row, 1).text() == "E-short"
    assert panel.boundaries_table.item(top_row, 2).text() == tr("physics.source.explicit")
    panel.bc_combo.setCurrentIndex(panel.bc_combo.findData(None))
    panel.apply_button.click()
    assert store.effective_bc(top.id) == ("PEC", "default")
    store.set_selection([])
    assert not panel.apply_button.isEnabled() and panel.selected_label.text() == ""
    # 選択の変更が表に反映される
    store.set_selection([top.id, shared.id])
    assert {i.row() for i in panel.boundaries_table.selectedIndexes()} == {top_row, row}

    # 検証: 開いた線を足すと警告
    store.set_tool("line")
    click(controller, 150, 5)
    click(controller, 170, 25)
    controller.finish()
    assert panel.validation.text() != tr("physics.validation.ok") and "#b45309" in panel.validation.text()
    panel.retranslate()
