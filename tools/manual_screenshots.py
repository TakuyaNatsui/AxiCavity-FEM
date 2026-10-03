"""USER_MANUAL.md 用のスクリーンショットを撮る（実ウィンドウ。既定のプラットフォームで開く）.

    python tools/manual_screenshots.py [--out docs/images] [--sample samples/s-band_1cell.gmshproj]

撮るもの（docs/images/v3_*.png）:
    v3_modeling.png   モデリングタブ（サンプルの形状、右パネルにパラメータ）
    v3_physics.png    物理タブ（境界条件の色分け、領域表）
    v3_mesh.png       メッシュタブ（一時プロジェクトでメッシュを生成して重ね表示）
    v3_analysis.png   解析タブ
    v3_run_confirm.png 実行確認ダイアログ
    v3_results.png    結果タブ（同名の *_TW_TM0_processed.h5 があればそれ、無ければ生成した結果）
    v3_file.png       ファイルタブ
    v3_view3d.png     3D 表示のウィンドウ（pyvista があるとき）

一時プロジェクトは作業フォルダ（既定 build/manual_shots）に作る。設定（配置・最近使ったファイル）は
AXICAVITY_SETTINGS_DIR を一時フォルダにして本物を汚さない。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    ap.add_argument("--sample", default=str(ROOT / "samples" / "s-band_1cell.gmshproj"))
    ap.add_argument("--work", default=str(ROOT / "build" / "manual_shots"))
    ap.add_argument("--size", default="1400x880")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    os.environ["AXICAVITY_SETTINGS_DIR"] = tempfile.mkdtemp(prefix="axicavity_shots_")
    w, h = (int(v) for v in args.size.lower().split("x"))

    from PySide6 import QtCore, QtTest, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    from axicavity_fem.gui.i18n.tr import set_language
    from axicavity_fem.gui.ui.dialogs.run_confirm import RunConfirmDialog, run_summary
    from axicavity_fem.gui.ui.main_window import RIBBON_TABS, MainWindow
    from axicavity_fem.gui.core.document import SketchArc, SketchLine

    set_language("ja")
    win = MainWindow()
    win.resize(w, h)
    win.show()
    win.activateWindow()
    QtTest.QTest.qWaitForWindowExposed(win)
    QtTest.QTest.qWait(300)

    import re
    absolute_dir = re.compile(r"[A-Za-z]:[\\/][^\s\"'（）]*[\\/]")

    def hide_local_paths() -> None:
        """公開する画像に手元のフォルダ名を出さない（ログのパスはファイル名だけにする）."""
        text = win.log.toPlainText()
        clean = absolute_dir.sub("", text)
        if clean != text:
            win.log.setPlainText(clean)

    def shot(name: str, widget=None) -> None:
        app.processEvents()
        hide_local_paths()
        QtTest.QTest.qWait(250)
        (widget or win).grab().save(str(out / f"{name}.png"))
        print("saved", out / f"{name}.png")

    def tab(name: str) -> None:
        win.ribbon.setCurrentIndex(RIBBON_TABS.index(name))
        app.processEvents()
        QtTest.QTest.qWait(150)

    sample = Path(args.sample)
    assert win.open_path(sample), sample
    win.sketch_view.fit_sketch()
    tab("file")
    shot("v3_file")
    tab("modeling")
    win.store.set_params_edit(True)
    shot("v3_modeling")
    tab("physics")
    shot("v3_physics")

    # メッシュ: 一時プロジェクトに保存してから生成
    project = work / f"{sample.stem}.axiproj"
    win.project.save_as(project)
    mesh_root = win.project.prepare_mesh()
    tab("mesh")
    assert win.mesh.generate(mesh_root), win.mesh.last_error
    for _ in range(600):
        if not win.mesh.running:
            break
        QtTest.QTest.qWait(200)
    assert win.mesh.job is not None and win.mesh.job.status == "done", win.mesh.job.error
    win.mesh.set_visible(True)
    shot("v3_mesh")

    tab("analysis")
    win.store.set_analysis(numModes=4)
    shot("v3_analysis")
    store = win.store
    curves = [e for e in store.active_sketch().entities if isinstance(e, (SketchLine, SketchArc)) and not e.construction]
    counts: dict[str, int] = {}
    for c in curves:
        counts[store.effective_bc(c.id)[0]] = counts.get(store.effective_bc(c.id)[0], 0) + 1
    sections = run_summary(store.document, store.resolve_regions(), counts, True,
                           win.mesh.info.stats.get("elements"),
                           str(Path(f"{project.name}.data") / "results" / "0001-tm0-sw"),   # 手元のパスを出さない
                           win.analysis.preview_lines(), [w for w in win.analysis.check()[1]])
    dialog = RunConfirmDialog(sections, win)
    dialog.show()
    QtTest.QTest.qWait(300)
    shot("v3_run_confirm", dialog)
    dialog.close()

    # 結果: 同名の進行波 TM0 の結果があればそれを開く。無ければ解析して表示
    result = sample.with_name(f"{sample.stem}_TW_TM0_processed.h5")
    if result.exists():
        assert win.open_result_file(result)
    else:
        assert win.analysis.run(), win.analysis.last_error
        for _ in range(3000):
            if not win.analysis.running:
                break
            QtTest.QTest.qWait(200)
    tab("results")
    win.result_model.set_options(show_vectors=True)
    for _ in range(50):
        if win.result_view.last_title:
            break
        QtTest.QTest.qWait(200)
    shot("v3_results")

    # 3D 表示（pyvista があれば）: 切り欠き・横断面・電気力線・矢印で撮る
    from axicavity_fem.gui.ui import view3d_window
    if view3d_window.is_available():
        win.view3d_action.trigger()
        for _ in range(50):
            QtTest.QTest.qWait(100)
            if win.view3d is not None and win.view3d.scene.actor_names:
                break
        win.view3d_controller.set_options(sector_deg=270.0, show_slice=True, show_e_lines=True, show_arrows=True)
        QtTest.QTest.qWait(500)
        win.view3d.resize(1000, 640)
        win.view3d.scene.view("iso")
        win.view3d.interactor.camera.azimuth = 200
        win.view3d.interactor.camera.elevation = 15
        win.view3d.interactor.render()
        QtTest.QTest.qWait(500)
        shot("v3_view3d", win.view3d)
        win.view3d.close()
    win.project._mark_saved()
    win.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
