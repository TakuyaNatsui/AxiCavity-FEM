"""リボンのアイコン（``ui/icons/*.svg``。EM-CAD-py ``emcad/ui/icon_set.py`` からの転用）.

24×24 の線画で、線は濃いグレー、強調は青、境界条件は 2D 画面と同じ ver2.3 の色（PEC 橙・E-short 青・
M-short 緑・None 灰）。SVG は ``tools/make_icons.py`` が書き出す（手で直さず、そちらを直して実行し直す。
出荷する SVG は ``ACTION_ICONS`` に現れる名前だけ）。1 文字の切り替えは文字だけ。

QtSvg の :class:`QSvgRenderer` で複数の大きさのピクスマップにして QIcon にする（Qt の SVG アイコン
プラグインに頼らない。EXE でも同じ。無効状態の灰色は Qt が作る）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6 import QtCore, QtGui, QtSvg

ICON_DIR = Path(__file__).parent / "icons"
# 描いておく大きさ（物理ピクセル）。Qt は必要な大きさに一番近いものを縮小して使う
PIXMAP_SIZES = (16, 18, 20, 24, 32, 36, 40, 48, 64)

# 操作（i18n のキー）→ アイコン名
ACTION_ICONS: dict[str, str] = {
    # ファイル
    "ribbon.new": "new", "ribbon.open": "open", "ribbon.recent": "recent", "ribbon.save": "save",
    "ribbon.saveAs": "save_as", "ribbon.undo": "undo", "ribbon.redo": "redo",
    "ribbon.import": "import", "ribbon.export": "export_model",
    "ribbon.importGmshproj": "import", "ribbon.importSuperfish": "import_superfish",
    "ribbon.exportGmshproj": "export_model", "ribbon.exportSuperfish": "export_superfish",
    "ribbon.exportMsh": "export_msh", "ribbon.exportGeo": "export_geo", "ribbon.exportPython": "export_python",
    "ribbon.units": "units", "ribbon.language": "language", "ribbon.about": "about", "ribbon.quit": "quit",
    # モデリング: 作図
    "ribbon.select": "select", "ribbon.line": "line", "ribbon.rectangle": "rectangle",
    "ribbon.circle": "circle", "ribbon.arc": "arc", "ribbon.polygon": "polygon", "ribbon.point": "point",
    "ribbon.fillet": "fillet", "ribbon.construction": "construction",
    "ribbon.insertPoint": "insert_point", "ribbon.deleteReconnect": "delete_reconnect",
    "ribbon.lineToArc": "line_to_arc", "ribbon.arcToLine": "arc_to_line",
    "ribbon.closePolyline": "close_polyline", "ribbon.splitIntersections": "split_intersections", "ribbon.fixAxis": "fix_axis", "ribbon.fixZ0": "fix_z0",
    "ribbon.fit": "fit",
    # モデリング: 拘束・寸法・パラメータ
    "ribbon.coincident": "coincident", "ribbon.horizontal": "horizontal", "ribbon.vertical": "vertical",
    "ribbon.parallel": "parallel", "ribbon.perpendicular": "perpendicular", "ribbon.tangent": "tangent",
    "ribbon.equal": "equal", "ribbon.concentric": "concentric", "ribbon.fix": "fix",
    "ribbon.symmetric": "symmetric", "ribbon.pointOnCurve": "point_on_curve", "ribbon.dimension": "dimension",
    "ribbon.params": "params",
    # 物理
    "physics.bc.PEC": "bc_pec", "physics.bc.E-short": "bc_eshort", "physics.bc.M-short": "bc_mshort",
    "physics.bc.None": "bc_none", "physics.bc.auto": "bc_auto", "physics.editRegion": "region",
    "physics.validate": "validate",
    # メッシュ
    "mesh.ribbon.generate": "mesh", "mesh.ribbon.stop": "stop", "mesh.ribbon.show": "mesh_show",
    "mesh.ribbon.openFolder": "open_folder",
    # 解析
    "analysis.ribbon.run": "run", "analysis.ribbon.cancel": "cancel", "analysis.ribbon.forceStop": "stop",
    "analysis.ribbon.post": "post", "analysis.ribbon.report": "report", "analysis.ribbon.openReport": "open_report",
    # 結果
    "results.ribbon.show": "show_results", "results.ribbon.hColor": "heatmap", "results.ribbon.eLines": "e_lines",
    "results.ribbon.vectors": "arrows", "results.ribbon.mesh": "mesh_show", "results.ribbon.eWall": "e_wall",
    "results.ribbon.gif": "gif", "results.ribbon.exportField": "field_map", "results.ribbon.png": "png",
    "results.ribbon.openResult": "open_result", "results.ribbon.openFolder": "open_folder",
    "results.ribbon.view3d": "view3d",
    "results.ribbon.exportArea": "field_map", "results.ribbon.exportLine": "field_map", "results.ribbon.exportAxis": "field_map",
    # 表示
    "ribbon.grid": "grid", "ribbon.pointLabels": "point_labels", "ribbon.bcColors": "bc_colors",
    "ribbon.axisRange": "axis_range", "ribbon.resetLayout": "reset_layout",
}


def has_icon(name: str) -> bool:
    return (ICON_DIR / f"{name}.svg").is_file()


APP_ICON_FILE = Path(__file__).resolve().parent / "app_icon.svg"
APP_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def render_svg(path: Path, size: int) -> QtGui.QImage:
    """SVG を size × size の画像に描く（Qt の SVG 画像プラグインに頼らない）."""
    image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.transparent)
    renderer = QtSvg.QSvgRenderer(str(path))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    renderer.render(painter, QtCore.QRectF(0, 0, size, size))
    painter.end()
    return image


def app_icon() -> QtGui.QIcon:
    """アプリのアイコン（ウィンドウ・タスクバー。EXE のアイコンは packaging/make_ico.py が同じ SVG から作る）."""
    result = QtGui.QIcon()
    if APP_ICON_FILE.is_file():
        for size in APP_ICON_SIZES:
            result.addPixmap(QtGui.QPixmap.fromImage(render_svg(APP_ICON_FILE, size)))
    return result


@lru_cache(maxsize=None)
def icon(name: str) -> QtGui.QIcon:
    """アイコン名（``icons/<name>.svg``）の QIcon。ファイルが無ければ空の QIcon."""
    path = ICON_DIR / f"{name}.svg"
    result = QtGui.QIcon()
    if not path.is_file():
        return result
    renderer = QtSvg.QSvgRenderer(str(path))
    for size in PIXMAP_SIZES:
        image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        renderer.render(painter, QtCore.QRectF(0, 0, size, size))
        painter.end()
        result.addPixmap(QtGui.QPixmap.fromImage(image))
    return result


def action_icon(key: str) -> QtGui.QIcon | None:
    """操作のキーに対応するアイコン（無ければ None）."""
    name = ACTION_ICONS.get(key)
    return icon(name) if name and has_icon(name) else None


def set_action_icon(target, key: str) -> None:
    """QAction / QToolButton に操作のキーのアイコンを付ける（対応するアイコンが無ければ何もしない）."""
    found = action_icon(key)
    if found is not None:
        target.setIcon(found)
