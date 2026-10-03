"""MainWindow を実ウィンドウ（既定のプラットフォーム）で短時間開いて結線を確かめる.

    python tests/_main_window_check.py <作業フォルダ>

offscreen では見えない配置の問題（リボンの幅・ドックの初期配置・ツールバーの 2 段の高さ）を確かめる。
test_ui_main_window_real.py が別プロセスで実行する。失敗は AssertionError → 終了コード 1。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtTest, QtWidgets


def main() -> int:
    work = Path(sys.argv[1])
    os.environ["AXICAVITY_SETTINGS_DIR"] = str(work / "settings")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    from axicavity_fem.gui.i18n.tr import set_language
    from axicavity_fem.gui.ui.main_window import RIBBON_TABS, MainWindow

    set_language("ja")
    QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Discard)
    win = MainWindow()
    win.resize(1400, 850)
    win.show()
    win.activateWindow()
    QtTest.QTest.qWaitForWindowExposed(win)
    QtTest.QTest.qWait(200)

    # 1. リボンはメニューバーの位置（左右のドックに挟まれず、ウィンドウ幅いっぱい）
    assert win.menuWidget() is win.ribbon, "リボンが menuWidget ではない"
    assert win.ribbon.width() >= win.width() - 4, (win.ribbon.width(), win.width())
    assert win.ribbon.mapTo(win, QtCore.QPoint(0, 0)).x() <= 2
    names = [win.ribbon.tabText(i) for i in range(win.ribbon.count())]
    assert names == ["ファイル", "モデリング", "物理", "メッシュ", "解析", "結果", "表示"], names
    # 2 段のモデリングタブが切れずに見える（隠れたタブは配置されないので、開いてから測る）
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("modeling"))
    app.processEvents()
    QtTest.QTest.qWait(100)
    bar1, bar2 = win._toolbars["modeling"], win._toolbars["modeling2"]
    assert bar2.isVisible() and 20 <= bar2.height() <= 60, bar2.height()
    assert bar2.mapTo(win, QtCore.QPoint(0, 0)).y() > bar1.mapTo(win, QtCore.QPoint(0, 0)).y()
    for act in bar1.actions():
        if act.data():
            widget = bar1.widgetForAction(act)
            assert widget is not None and widget.isVisible(), act.text()
            assert widget.mapTo(win, QtCore.QPoint(0, 0)).x() + widget.width() <= win.width(), act.text()

    # 2. 初期配置: 左ブラウザ / 右プロパティ / 下ログ
    assert not win._dock_tree.isFloating() and not win._dock_props.isFloating()
    assert win._dock_tree.mapTo(win, QtCore.QPoint(0, 0)).x() < win._dock_props.mapTo(win, QtCore.QPoint(0, 0)).x()
    assert win._dock_log.isVisible()

    # 3. 作図（矩形）→ ツリーと Undo
    view = win.sketch_view
    view.fit_sketch()
    win.tool_actions["rectangle"].trigger()
    for x, y in ((0, 0), (40, 20)):
        p = view.mapFromScene(QtCore.QPointF(x, y))
        QtTest.QTest.mouseMove(view.viewport(), p)
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, p)
    assert len(win.store.profiles) == 1, "閉領域ができていない"
    assert "閉領域 1" in win.tree_item(("section", "shape")).text(0)
    QtTest.QTest.keyClick(win, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    app.processEvents()
    assert len(win.store.profiles) == 0, "Ctrl+Z が効かない"

    # 4. 物理タブで overlay が切り替わる
    win.ribbon.setCurrentIndex(RIBBON_TABS.index("physics"))
    app.processEvents()
    assert win.store.overlay == "physics"
    # 5. 結果タブ: サンプルの結果を開くと中央が結果ビュー（matplotlib）に切り替わり、モード表が出る
    sample = Path(__file__).resolve().parents[1] / "samples" / "s-band_1cell_TW_TM0_processed.h5"
    if sample.exists():
        assert win.open_result_file(sample)
        app.processEvents()
        QtTest.QTest.qWait(300)
        assert RIBBON_TABS[win.ribbon.currentIndex()] == "results"
        assert win.stack.currentWidget() is win.result_view and win.result_view.canvas.width() > 200
        assert win.results_panel.table.rowCount() > 0 and win.props_stack.currentWidget() is win.results_panel
        for _ in range(20):
            if win.result_view.last_title:
                break
            QtTest.QTest.qWait(200)
        assert win.result_view.last_title.startswith("Mode 0"), win.result_view.last_error
        win.ribbon.setCurrentIndex(RIBBON_TABS.index("modeling"))
        app.processEvents()
        assert win.stack.currentWidget() is win.sketch_view
        # 6. 3D 表示（pyvista があれば）: 別ウィンドウが開いてアクターが出る → 閉じるとトグルが戻る
        from axicavity_fem.gui.ui import view3d_window
        if view3d_window.is_available():
            win.ribbon.setCurrentIndex(RIBBON_TABS.index("results"))
            win.view3d_action.trigger()
            for _ in range(50):
                QtTest.QTest.qWait(100)
                if win.view3d is not None and win.view3d.scene.actor_names:
                    break
            assert win.view3d is not None and win.view3d.isVisible(), "3D ウィンドウが開かない"
            assert set(win.view3d.scene.actor_names) >= {"wall", "meridian"}, win.view3d_controller.error
            assert win.view3d.interactor.width() > 200
            win.view3d.close()
            QtTest.QTest.qWait(100)
            assert not win.view3d_action.isChecked()
    win.close()
    app.processEvents()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
