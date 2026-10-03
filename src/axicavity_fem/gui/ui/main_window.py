"""メインウィンドウ（EM-CAD-py ``emcad/ui/main_window.py`` の骨格を転用）.

レイアウト:
    上: リボン（メニューバーの位置 = ウィンドウ幅いっぱい。タブ: ファイル / モデリング / 物理 / メッシュ /
        解析 / 結果 / 表示。モデリングは 2 段: 作図・編集、拘束・寸法・パラメータ）
    左: ブラウザドック（プロジェクトの構成: パラメータ / 形状 / 物理（領域・境界条件）/ メッシュ / 解析 / 結果。
        節をクリックすると対応するタブに切り替わる）
    中央: 2D スケッチ画面（リボンのタブに応じて overlay = sketch / physics / mesh）。結果タブでは結果ビュー
        （matplotlib の場の図）に切り替わる
    右: プロパティドック（リボンのタブに連動: モデリング / 物理 / メッシュ / 解析 / 結果）
    下: ログドック、ステータスバー（ヒント、カーソル座標 z / r、選択曲線の境界条件）

状態はすべて :class:`~axicavity_fem.gui.app.store.DocumentStore` が持ち、UI はシグナルで追従する。
ショートカット付きの操作はメインウィンドウにも登録する（リボンの非表示タブにあると Qt が無効にするため）。
"""

from __future__ import annotations

import html
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import GUI_VERSION
from ..app.store import CONSTRAINT_TOOLS, SKETCH_TOOLS, DocumentStore
from ..core.convert import ConvertError, check_geometry, resolve_regions, to_multi_region
from ..core.document import BC_NAMES, UNITS, SketchArc, SketchLine, SketchPoint, is_reference
from ..core.serialize import DocumentParseError
from ..core.sketch.model import count_entities, get_entity
from ..i18n.tr import language, set_language, tr
from ..io.axiproj import PROJECT_SUFFIX, ProjectFormatError, ResultEntry, sanitize_file_name
from ..io.gmshproj import GMSHPROJ_SUFFIX, save_gmshproj
from ..io.superfish import SUPERFISH_SUFFIX, save_superfish
from ..jobs.analysis_controller import AnalysisController
from ..jobs.commands import export_argv
from ..jobs.mesh_controller import MeshController
from ..jobs.session import JobSession
from ..project.controller import ProjectController
from ..sketch.controller import SketchController
from .app_settings import (
    LAYOUT_VERSION,
    add_recent_project,
    clear_recent_projects,
    load_window_layout,
    recent_projects,
    save_language,
    save_window_layout,
    saved_language,
)
from .focus import text_entry_has_focus
from .icon_set import set_action_icon
from .analysis_panel import AnalysisPanel
from .dialogs.export_field import SHAPES, ExportFieldDialog
from .dialogs.gif import GifDialog, save_gif_with_progress
from .dialogs.run_confirm import RunConfirmDialog, run_summary
from .mesh_panel import MeshPanel
from .physics_panel import PhysicsPanel
from .result_model import ResultModel
from . import view3d_window
from .result_view import ResultView
from .results_panel import ResultsPanel
from .sketch_panels import SketchPropertyPanel
from .sketch_view import BC_COLORS, MeshPreview, SketchView

RIBBON_TABS = ("file", "modeling", "physics", "mesh", "analysis", "results", "view")
# リボンのタブ → スケッチ画面の overlay（他のタブでは直前のまま）
TAB_OVERLAYS = {"modeling": "sketch", "physics": "physics", "mesh": "mesh"}
# ブラウザの節 → クリックで開くリボンのタブ（None は切り替えない）
SECTION_TABS = {
    "project": None, "params": "modeling", "shape": "modeling", "physics": "physics", "regions": "physics",
    "boundaries": "physics", "mesh": "mesh", "analysis": "analysis", "results": "results",
}
# スケッチ画面が処理するツールのキー（ツールチップに出すだけ。QAction のショートカットにはしない）
RESULT_SUFFIXES = (".h5", ".hdf5")
# 結果タブのオプション（リボンのキー → ViewOptions のフィールド）
RESULT_OPTIONS = (("hColor", "show_color"), ("eLines", "show_lines"), ("vectors", "show_vectors"),
                  ("mesh", "show_mesh"), ("eWall", "show_e_wall"))
SKETCH_TOOL_KEYS = {"select": "S", "line": "L", "rectangle": "R", "circle": "C", "arc": "A", "polygon": "P"}
# ドックの初期の大きさ [px]（ブラウザ・プロパティの幅、ログの高さ）
DOCK_WIDTHS = (260, 380)
LOG_HEIGHT = 150


def ribbon_tooltip(title: str, body: str = "", key: str | None = None) -> str:
    """リボンのツールチップ（太字の名前とキー、その下に説明）."""
    head = f"<b>{html.escape(title)}</b>"
    if key and key not in title:
        head += f"&nbsp;&nbsp;<span style='color:#6b7280'>{html.escape(key)}</span>"
    return head + (f"<br>{html.escape(body)}" if body else "")


def chip_icon(color: str, size: int = 12) -> QtGui.QIcon:
    """ツリーの色見本（境界条件の色）."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setBrush(QtGui.QColor(color))
    painter.setPen(QtGui.QPen(QtGui.QColor("#374151"), 1))
    painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, size - 1, size - 1), 2, 2)
    painter.end()
    return QtGui.QIcon(pixmap)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        lang = saved_language()
        if lang:
            set_language(lang)
        self.resize(1400, 900)
        self.store = DocumentStore(parent=self)
        self.controller = SketchController(self.store, parent=self)
        self.project = ProjectController(self.store, generator=f"AxiCavity-FEM v3 GUI {GUI_VERSION}",
                                         parent=self)
        self.session = JobSession(parent=self)
        self.mesh = MeshController(self.store, self.session, parent=self)
        self.analysis = AnalysisController(self.store, self.session, self.project, self.mesh, parent=self)
        self._mesh_root: Optional[Path] = None
        self._mesh_preview_source = None
        self._tree_mesh_label = ""
        self._export_job = None
        self._export_field_tmp: Optional[Path] = None
        self.view3d: Optional[view3d_window.View3DWindow] = None

        # --- 中央: スケッチ画面 / 結果ビュー ---
        self.sketch_view = SketchView(self.store, self.controller, self)
        self.result_model = ResultModel(parent=self)
        self.result_view = ResultView(self.result_model)
        self.result_view.point_picked.connect(self._on_field_pick)
        self.view3d_controller = view3d_window.View3DController(self.result_model, parent=self)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.sketch_view)
        self.stack.addWidget(self.result_view)

        # --- リボン ---
        self.ribbon = QtWidgets.QTabWidget()
        self.ribbon.setDocumentMode(True)
        self.ribbon.setMaximumHeight(96)
        self._toolbars: dict[str, QtWidgets.QToolBar] = {}
        self._menu_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._tooltip_specs: dict[QtGui.QAction, tuple] = {}   # ツールチップ（言語の切替で作り直す）
        for tab in RIBBON_TABS:
            self.ribbon.addTab(self._make_tab(tab), tr(f"ribbon.{tab}"))
        # リボンはメニューバーの位置に置く（左右のドックに挟まれず、ウィンドウの端から端まで）
        self.setMenuWidget(self.ribbon)
        self.setCentralWidget(self.stack)

        # --- ドック ---
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self._tree_collapsed: set[tuple] = set()      # 利用者が畳んだ節（ツリーを作り直しても保つ）
        self.tree.itemCollapsed.connect(lambda item: self._on_tree_expanded(item, False))
        self.tree.itemExpanded.connect(lambda item: self._on_tree_expanded(item, True))
        self.tree.itemClicked.connect(self._on_tree_clicked)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._dock_tree = self._add_dock(tr("tree.title"), self.tree, QtCore.Qt.LeftDockWidgetArea, "dock_tree")
        self.props_stack = QtWidgets.QStackedWidget()
        self.sketch_panel = SketchPropertyPanel(self.store)
        self.sketch_panel.numeric_entered.connect(self.controller.numeric)
        self.physics_panel = PhysicsPanel(self.store)
        self.mesh_panel = MeshPanel(self.store, self.mesh)
        self.mesh_panel.generate_requested.connect(self._on_generate_mesh)
        self.analysis_panel = AnalysisPanel(self.store, self.analysis)
        self.analysis_panel.run_requested.connect(self._on_run_analysis)
        self.results_panel = ResultsPanel(self.store, self.result_model, self.analysis, self.project)
        self.panels: dict[str, QtWidgets.QWidget] = {"modeling": self.sketch_panel,
                                                     "physics": self.physics_panel, "mesh": self.mesh_panel,
                                                     "analysis": self.analysis_panel, "results": self.results_panel}
        for panel in (self.sketch_panel, self.physics_panel, self.mesh_panel, self.analysis_panel,
                      self.results_panel):
            self.props_stack.addWidget(panel)
        self._dock_props = self._add_dock(tr("props.title"), self.props_stack,
                                          QtCore.Qt.RightDockWidgetArea, "dock_props")
        self._dock_props.setMinimumWidth(320)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self._dock_log = self._add_dock(tr("dock.log"), self.log, QtCore.Qt.BottomDockWidgetArea, "dock_log")
        self.cursor_label = QtWidgets.QLabel("")
        self.bc_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.bc_label)
        self.statusBar().addPermanentWidget(self.cursor_label)
        self.statusBar().showMessage("")

        self._build_actions()
        self.store.tool_changed.connect(self._sync_tool_actions)
        self.store.document_changed.connect(self._on_document_changed)
        self.store.settings_changed.connect(lambda _s: self._on_document_changed())
        self.store.hint_changed.connect(lambda key: self.statusBar().showMessage(tr(key) if key else ""))
        self.store.selection_changed.connect(self._on_selection_changed)
        self.store.params_edit_changed.connect(self._sync_params_actions)
        self.store.cursor_changed.connect(self._on_cursor)
        self.ribbon.currentChanged.connect(self._on_ribbon_tab)
        self.project.changed.connect(self._on_project_changed)
        self.mesh.changed.connect(self._on_mesh_changed)
        self.analysis.changed.connect(self._sync_analysis_actions)
        self.result_model.loaded.connect(self._sync_results_actions)
        self.result_model.selection_changed.connect(self._sync_results_actions)
        self.result_model.options_changed.connect(self._sync_results_actions)
        self.session.log_appended.connect(lambda _j, line: self.log_message(line))
        self.session.job_finished.connect(self._on_job_finished)
        self.project.new()
        self._sync_edit_actions()
        self._on_mesh_changed()
        self._sync_analysis_actions()
        # 配置: 前回のウィンドウの位置・大きさとドックの配置を再現する（無ければ最初の表示で初期配置。
        # 表示タブの「レイアウトを初期状態に戻す」で戻せる）
        geometry, state = load_window_layout()
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._layout_restored = state is not None and self.restoreState(state, LAYOUT_VERSION)
        self._layout_initialized = False
        self.log_message(f"AxiCavity-FEM v3 GUI {GUI_VERSION} 起動")

    # ------------------------------------------------------------------
    # 構築
    # ------------------------------------------------------------------

    def _new_toolbar(self, name: str) -> QtWidgets.QToolBar:
        bar = QtWidgets.QToolBar()
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        bar.setIconSize(QtCore.QSize(24, 24))
        self._toolbars[name] = bar
        return bar

    def _two_rows(self, top: str, bottom: str) -> QtWidgets.QWidget:
        """2 段のリボン（モデリング）。高さが足りないので、アイコンは小さく文字の横に置く."""
        widget = QtWidgets.QWidget()
        rows = QtWidgets.QVBoxLayout(widget)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        for name in (top, bottom):
            bar = self._new_toolbar(name)
            bar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            bar.setIconSize(QtCore.QSize(18, 18))
            rows.addWidget(bar)
        return widget

    def _make_tab(self, tab: str) -> QtWidgets.QWidget:
        """タブ 1 つ分（モデリングは 2 段: 上 = 作図と編集、下 = 拘束・寸法・パラメータ）."""
        if tab == "modeling":
            return self._two_rows("modeling", "modeling2")
        return self._new_toolbar(tab)

    def _add_dock(self, title: str, widget: QtWidgets.QWidget, area, name: str) -> QtWidgets.QDockWidget:
        """ドック. ``name`` は配置の保存・復元（saveState）用の固定名（題名は言語で変わるので使わない）."""
        dock = QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(name)
        self.addDockWidget(area, dock)
        return dock

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._layout_initialized:
            self._layout_initialized = True
            if not self._layout_restored:
                # 表示されてから大きさを決める（表示前の resizeDocks は効かないことがある）
                QtCore.QTimer.singleShot(0, self._apply_default_layout)

    def _apply_default_layout(self) -> None:
        """初期配置: 左にブラウザ、右にプロパティ、下にログ（浮かせたドックも戻す）."""
        for dock, area in ((self._dock_tree, QtCore.Qt.LeftDockWidgetArea),
                           (self._dock_props, QtCore.Qt.RightDockWidgetArea),
                           (self._dock_log, QtCore.Qt.BottomDockWidgetArea)):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()
        self.resizeDocks([self._dock_tree, self._dock_props], list(DOCK_WIDTHS), QtCore.Qt.Horizontal)
        self.resizeDocks([self._dock_log], [LOG_HEIGHT], QtCore.Qt.Vertical)

    def _on_reset_layout(self) -> None:
        self._apply_default_layout()
        self.log_message(tr("ribbon.resetLayoutDone"))

    def _action(self, tab: str, key: str, slot=None, shortcut: str | None = None,
                checkable: bool = False, tooltip_key: str | None = None,
                key_hint: str | None = None, stage: str | None = None,
                menu: QtWidgets.QMenu | None = None) -> QtGui.QAction:
        """リボンの操作. ツールチップは名前（太字）・ショートカット・説明（``tooltip_key``、無ければ ``tip.<key>``）.

        ``key_hint`` はツールチップに出すだけのキー（スケッチ画面が処理する L / C などは QAction の
        ショートカットにしない。入力欄での文字入力を奪うため）。``slot`` が無ければ「未実装（``stage``）」をログに出す。
        """
        act = QtGui.QAction(tr(key), self)
        act.setData(key)
        act.setCheckable(checkable)
        set_action_icon(act, key)
        self._tooltip_specs[act] = (key, tooltip_key or f"tip.{key}", shortcut or key_hint)
        self._apply_tooltip(act)
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
            # リボンの別タブを開いていても効くように、常に表示されているメインウィンドウにも関連付ける
            # （QAction のショートカットは、関連付いたウィジェットが見えているときだけ有効）
            self.addAction(act)
        if slot is not None:
            act.triggered.connect(slot)
        else:
            act.triggered.connect(lambda: self.log_message(
                f"{tr(key)}: {tr('file.notYet', stage=stage or '-')}"))
        if menu is not None:
            menu.addAction(act)
        else:
            self._toolbars[tab].addAction(act)
        return act

    def _menu_button(self, tab: str, key: str) -> QtWidgets.QMenu:
        """リボンのドロップダウン（「取り込み ▾」など）。返ったメニューに ``_action(..., menu=...)`` で項目を足す."""
        bar = self._toolbars[tab]
        button = QtWidgets.QToolButton()
        button.setText(tr(key))
        button.setToolTip(ribbon_tooltip(tr(key), tr(f"tip.{key}")))
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setToolButtonStyle(bar.toolButtonStyle())
        button.setIconSize(bar.iconSize())
        set_action_icon(button, key)
        menu = QtWidgets.QMenu(self)
        button.setMenu(menu)
        bar.addWidget(button)
        self._menu_buttons[key] = button
        return menu

    def _group_label(self, tab: str, key: str) -> QtWidgets.QLabel:
        """リボンの中のグループ見出し（「拘束:」など。薄い色の文字。言語の切替で作り直す）."""
        label = QtWidgets.QLabel(tr(key))
        label.setStyleSheet("color: #6b7280; padding: 0 4px 0 6px;")
        self._toolbars[tab].addWidget(label)
        self._group_labels[label] = key
        return label

    def _apply_tooltip(self, act: QtGui.QAction) -> None:
        key, body_key, hint = self._tooltip_specs[act]
        body = tr(body_key)
        act.setToolTip(ribbon_tooltip(tr(key), "" if body == body_key else body, hint))

    def _build_actions(self) -> None:
        # ファイル: プロジェクト | 元に戻す | 取り込み | 書き出し | 単位・言語 | 終了
        bar = self._toolbars["file"]
        self._action("file", "ribbon.new", self._on_new, "Ctrl+N")
        self.open_action = self._action("file", "ribbon.open", self._on_open, "Ctrl+O")
        self.recent_button = QtWidgets.QToolButton()
        self.recent_button.setText(tr("ribbon.recent"))
        self.recent_button.setToolTip(ribbon_tooltip(tr("ribbon.recent"), tr("tip.ribbon.recent")))
        self.recent_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # addWidget したボタンはツールバーの表示形式に追従しないので、同じ形式を指定する
        self.recent_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.recent_button.setIconSize(bar.iconSize())
        set_action_icon(self.recent_button, "ribbon.recent")
        self.recent_menu = QtWidgets.QMenu(self)
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
        self.recent_button.setMenu(self.recent_menu)
        bar.addWidget(self.recent_button)
        self.save_action = self._action("file", "ribbon.save", self._on_save, "Ctrl+S")
        self.save_as_action = self._action("file", "ribbon.saveAs", self._on_save_as, "Ctrl+Shift+S")
        bar.addSeparator()
        self.undo_action = self._action("file", "ribbon.undo", self._on_undo, "Ctrl+Z")
        self.redo_action = self._action("file", "ribbon.redo", self._on_redo, "Ctrl+Y")
        bar.addSeparator()
        self.import_menu = self._menu_button("file", "ribbon.import")
        self._action("file", "ribbon.importGmshproj", self._on_import_gmshproj, menu=self.import_menu)
        self._action("file", "ribbon.importSuperfish", self._on_import_superfish, menu=self.import_menu)
        self.export_menu = self._menu_button("file", "ribbon.export")
        self._action("file", "ribbon.exportGmshproj", self._on_export_gmshproj, menu=self.export_menu)
        self._action("file", "ribbon.exportSuperfish", self._on_export_superfish, menu=self.export_menu)
        self.export_menu.addSeparator()
        self.export_msh_action = self._action("file", "ribbon.exportMsh", self._on_export_msh, menu=self.export_menu)
        self._action("file", "ribbon.exportGeo", self._on_export_geo, menu=self.export_menu)
        self._action("file", "ribbon.exportPython", self._on_export_python, menu=self.export_menu)
        bar.addSeparator()
        self._action("file", "ribbon.units", self._on_units)
        self.language_action = QtGui.QAction(tr("ribbon.language") + " ja/en", self)
        self.language_action.setToolTip(ribbon_tooltip(tr("ribbon.language"), tr("tip.ribbon.language")))
        set_action_icon(self.language_action, "ribbon.language")
        self.language_action.triggered.connect(self._on_toggle_language)
        bar.addAction(self.language_action)
        bar.addSeparator()
        self._action("file", "ribbon.about", self._on_about)
        self._action("file", "ribbon.quit", self.close, "Ctrl+Q")

        # モデリング（上段）: 作図ツール | 構築線 | 点・曲線の編集 | r=0 | 全体表示
        self.tool_actions: dict[str, QtGui.QAction] = {}
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for tool in SKETCH_TOOLS:
            act = self._action("modeling", f"ribbon.{tool}",
                               lambda _=False, t=tool: self.store.set_tool(t), checkable=True,
                               tooltip_key=f"tip.sketchTool.{tool}", key_hint=SKETCH_TOOL_KEYS.get(tool))
            group.addAction(act)
            self.tool_actions[tool] = act
        bar = self._toolbars["modeling"]
        bar.addSeparator()
        self.construction_action = self._action(
            "modeling", "ribbon.construction",
            lambda: self.store.toggle_construction(list(self.store.selection)),
            tooltip_key="ribbon.constructionHint")
        bar.addSeparator()
        self.insert_point_action = self._action("modeling", "ribbon.insertPoint", self._on_insert_point)
        self.delete_reconnect_action = self._action("modeling", "ribbon.deleteReconnect",
                                                    self._on_delete_reconnect)
        # 線→円弧 / 円弧→線: 線（円弧）を選んでいればすぐ変換、選んでいなければクリックで変換するツールになる
        self.line_to_arc_action = self._action("modeling", "ribbon.lineToArc", self._on_line_to_arc, checkable=True)
        self.arc_to_line_action = self._action("modeling", "ribbon.arcToLine", self._on_arc_to_line, checkable=True)
        for tool_id, act in (("lineToArc", self.line_to_arc_action), ("arcToLine", self.arc_to_line_action)):
            group.addAction(act)
            self.tool_actions[tool_id] = act
        self.close_polyline_action = self._action("modeling", "ribbon.closePolyline", self._on_close_polyline)
        self.split_action = self._action("modeling", "ribbon.splitIntersections", self._on_split_intersections)
        bar.addSeparator()
        self._action("modeling", "ribbon.fit", self.sketch_view.fit_sketch, key_hint="F")

        # モデリング（下段）: 「拘束:」拘束 11 種 + r=0 / z=0 に固定 | 「寸法:」寸法 | パラメータ
        self._group_labels: dict[QtWidgets.QLabel, str] = {}
        self._group_label("modeling2", "ribbon.groupConstraints")
        for tool in CONSTRAINT_TOOLS:
            act = self._action("modeling2", f"ribbon.{tool}",
                               lambda _=False, t=tool: self.store.set_tool(t), checkable=True,
                               tooltip_key=f"constraintTool.{tool}")
            group.addAction(act)
            self.tool_actions[tool] = act
        self.fix_axis_action = self._action("modeling2", "ribbon.fixAxis", lambda: self._fix_selected_points("r"))
        self.fix_z0_action = self._action("modeling2", "ribbon.fixZ0", lambda: self._fix_selected_points("z"))
        self._toolbars["modeling2"].addSeparator()
        self._group_label("modeling2", "ribbon.groupDimension")
        act = self._action("modeling2", "ribbon.dimension",
                           lambda: self.store.set_tool("dimension"), checkable=True,
                           tooltip_key="constraintTool.dimension")
        group.addAction(act)
        self.tool_actions["dimension"] = act
        self._toolbars["modeling2"].addSeparator()
        self.params_action = self._action(
            "modeling2", "ribbon.params",
            lambda: self.store.set_params_edit(not self.store.params_edit), checkable=True)

        # 物理: 境界条件 | 領域 | 検証
        self.bc_actions: dict[str, QtGui.QAction] = {}
        for bc in BC_NAMES:
            self.bc_actions[bc] = self._action(
                "physics", f"physics.bc.{bc}", lambda _=False, b=bc: self._on_assign_bc(b))
        self.bc_actions["auto"] = self._action("physics", "physics.bc.auto",
                                               lambda: self._on_assign_bc(None))
        self._toolbars["physics"].addSeparator()
        self.edit_region_action = self._action("physics", "physics.editRegion", self._on_edit_region)
        self._toolbars["physics"].addSeparator()
        self._action("physics", "physics.validate", self._on_validate)

        # メッシュ / 解析
        self.mesh_generate_action = self._action("mesh", "mesh.ribbon.generate", self._on_generate_mesh)
        self.mesh_stop_action = self._action("mesh", "mesh.ribbon.stop", lambda: self.mesh.cancel(force=True))
        self._toolbars["mesh"].addSeparator()
        self.mesh_show_action = self._action("mesh", "mesh.ribbon.show", self._on_toggle_mesh_visible,
                                             checkable=True)
        self.mesh_folder_action = self._action("mesh", "mesh.ribbon.openFolder", self._open_mesh_folder)
        self.run_action = self._action("analysis", "analysis.ribbon.run", self._on_run_analysis)
        self.cancel_action = self._action("analysis", "analysis.ribbon.cancel", lambda: self.analysis.cancel())
        self.force_stop_action = self._action("analysis", "analysis.ribbon.forceStop", self._on_force_stop)
        self._toolbars["analysis"].addSeparator()
        self.post_action = self._action("analysis", "analysis.ribbon.post", self._on_run_post)
        self._toolbars["analysis"].addSeparator()
        self.report_action = self._action("analysis", "analysis.ribbon.report", self._on_create_report)
        self.open_report_action = self._action("analysis", "analysis.ribbon.openReport", self._on_open_report)
        # 結果: 表示 | 表示オプション | GIF・場の書き出し・PNG | 結果ファイル・フォルダ
        self.show_results_action = self._action("results", "results.ribbon.show", self._on_toggle_results_view,
                                                checkable=True)
        self.view3d_action = self._action("results", "results.ribbon.view3d", self._on_toggle_view3d, checkable=True)
        self._toolbars["results"].addSeparator()
        self.result_option_actions: dict[str, QtGui.QAction] = {}
        for key, option in RESULT_OPTIONS:
            self.result_option_actions[option] = self._action(
                "results", f"results.ribbon.{key}",
                lambda on=False, o=option: self.result_model.set_options(**{o: bool(on)}), checkable=True)
        self._toolbars["results"].addSeparator()
        self.gif_action = self._action("results", "results.ribbon.gif", self._on_save_gif)
        self.export_field_menu = self._menu_button("results", "results.ribbon.exportField")
        self.export_field_actions: dict[str, QtGui.QAction] = {}
        for shape in SHAPES:
            self.export_field_actions[shape] = self._action(
                "results", f"results.ribbon.export{shape.capitalize()}",
                lambda _=False, sh=shape: self._on_export_field(sh), menu=self.export_field_menu)
        self.png_action = self._action("results", "results.ribbon.png", self._on_save_png)
        self._toolbars["results"].addSeparator()
        self.open_result_action = self._action("results", "results.ribbon.openResult", self._on_open_result)
        self.result_folder_action = self._action("results", "results.ribbon.openFolder", self._on_open_result_folder)

        # 表示
        self._action("view", "ribbon.fit", self.sketch_view.fit_sketch, key_hint="F")
        self._action("view", "ribbon.axisRange", self._on_axis_range)
        self._toolbars["view"].addSeparator()
        self.grid_action = self._action("view", "ribbon.grid", self.sketch_view.set_show_grid, checkable=True)
        self.grid_action.setChecked(True)
        self.point_labels_action = self._action("view", "ribbon.pointLabels",
                                                self.sketch_view.set_show_point_labels, checkable=True)
        self.bc_colors_action = self._action("view", "ribbon.bcColors",
                                             self.sketch_view.set_show_bc_colors, checkable=True)
        self._toolbars["view"].addSeparator()
        self.reset_layout_action = self._action("view", "ribbon.resetLayout", self._on_reset_layout)

        # 全選択（wx 版では Superfish 書き出しと衝突していた）。入力欄にフォーカスがあるときは奪わない
        self.select_all_action = QtGui.QAction(self)
        self.select_all_action.setShortcut(QtGui.QKeySequence("Ctrl+A"))
        self.select_all_action.triggered.connect(self._on_select_all)
        self.addAction(self.select_all_action)

        self._sync_tool_actions(self.store.active_tool)

    # ------------------------------------------------------------------
    # リボンのタブ・パネル
    # ------------------------------------------------------------------

    def _on_ribbon_tab(self, _index: int) -> None:
        tab = RIBBON_TABS[self.ribbon.currentIndex()]
        self._set_results_view(tab == "results")
        overlay = TAB_OVERLAYS.get(tab)
        if overlay is not None:
            self.store.set_overlay(overlay)
            if overlay != "sketch" and self.store.active_tool != "select":
                self.store.set_tool("select")           # 物理・メッシュでは選択だけ
        self._update_props_stack()

    def _update_props_stack(self) -> None:
        """プロパティドックをリボンのタブに合わせる（パネルの無いタブでは直前のパネルのまま）."""
        panel = self.panels.get(RIBBON_TABS[self.ribbon.currentIndex()])
        if panel is not None:
            self.props_stack.setCurrentWidget(panel)

    def _show_tab(self, tab: str) -> None:
        self.ribbon.setCurrentIndex(RIBBON_TABS.index(tab))

    def _set_results_view(self, show: bool) -> None:
        """中央を結果ビュー（show）/ スケッチ画面に切り替える（結果タブで自動。「結果表示」ボタンでも）."""
        self.stack.setCurrentWidget(self.result_view if show else self.sketch_view)
        self.show_results_action.blockSignals(True)
        self.show_results_action.setChecked(show)
        self.show_results_action.blockSignals(False)

    def _on_toggle_results_view(self, checked: bool) -> None:
        if checked and RIBBON_TABS[self.ribbon.currentIndex()] != "results":
            self._show_tab("results")                # タブを開くと _on_ribbon_tab が切り替える
        else:
            self._set_results_view(bool(checked))

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def _selected(self, kinds) -> list:
        sketch = self.store.active_sketch()
        found = []
        for sid in self.store.selection:
            e = get_entity(sketch, sid)
            if isinstance(e, kinds) and not is_reference(e):    # 参照の原点・軸は編集の対象にしない
                found.append(e)
        return found

    def _on_insert_point(self) -> None:
        for curve in self._selected((SketchLine, SketchArc)):
            self.store.insert_point_on_curve(curve.id, self._curve_midpoint(curve))
            break

    def _curve_midpoint(self, curve):
        from ..core.sketch.model import arc_params, point_pos
        from ..core.sketch import geometry as g
        sketch = self.store.active_sketch()
        if isinstance(curve, SketchLine):
            a, b = point_pos(sketch, curve.p1), point_pos(sketch, curve.p2)
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        p = arc_params(sketch, curve)
        return g.point_at_angle(p.center, p.radius, p.startAngle + p.sweep / 2)

    def _on_delete_reconnect(self) -> None:
        ids = [p.id for p in self._selected(SketchPoint)]
        if ids:
            self.store.delete_points_reconnect(ids)

    def _on_line_to_arc(self) -> None:
        self._convert_or_pick(SketchLine, self.store.convert_line_to_arc, "lineToArc")

    def _on_arc_to_line(self) -> None:
        self._convert_or_pick(SketchArc, self.store.convert_arc_to_line, "arcToLine")

    def _convert_or_pick(self, kind, convert, tool_id: str) -> None:
        """選んでいる線（円弧）があればすべて変換する（変換後の曲線を選択）。無ければクリックで変換するツールにする."""
        targets = self._selected(kind)
        if targets:
            converted = [new for new in (convert(e.id) for e in targets) if new is not None]
            if converted:
                self.store.set_selection(converted)
        else:
            self.store.set_tool(tool_id)
        self._sync_tool_actions(self.store.active_tool)    # 押したボタンのチェックを今のツールに合わせ直す

    def _on_split_intersections(self) -> None:
        count = self.store.planarize()
        self.statusBar().showMessage(tr("hint.splitDone", count=count) if count else tr("hint.splitNone"), 5000)
        if count:
            self.log_message(tr("hint.splitDone", count=count))

    def _on_close_polyline(self) -> None:
        if self.store.close_polyline() is None:
            self.statusBar().showMessage(tr("hint.closePolylineNone"), 4000)

    def _fix_selected_points(self, coord: str) -> None:
        """選択した点（線なら両端）を z 軸 r = 0（coord="r"）/ r 軸 z = 0（coord="z"）に拘束する."""
        ids = [e.id for e in self._selected((SketchPoint, SketchLine))]
        if ids and not self.store.fix_points_to_zero(ids, coord):
            info = self.store.solve_info
            if info is not None and info.status in ("conflict", "failed", "invalid"):
                self.statusBar().showMessage(tr("solve.conflict"), 5000)

    def _on_select_all(self) -> None:
        if text_entry_has_focus():
            return
        sketch = self.store.active_sketch()
        if sketch is not None:
            self.store.set_tool("select")
            self.store.set_selection([e.id for e in sketch.entities if not is_reference(e)])

    def _on_assign_bc(self, bc: Optional[str]) -> None:
        ids = [c.id for c in self._selected((SketchLine, SketchArc))]
        if not ids:
            return
        self.store.set_boundary(ids, bc)
        self.log_message(tr("physics.bcApplied", bc=bc, count=len(ids)) if bc
                         else tr("physics.bcAuto", count=len(ids)))
        self._on_selection_changed()

    def _on_validate(self) -> None:
        issues = check_geometry(self.store.document)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level != "error"]
        for issue in issues:
            self.log_message(f"[{issue.level}] {issue.code}: {issue.message}")
        if not issues:
            text = tr("physics.validateOk", count=len(self.store.profiles))
        else:
            text = tr("physics.validateResult", errors=len(errors), warnings=len(warnings))
            text += "\n\n" + "\n".join(f"- {i.message}" for i in issues[:12])
        (QtWidgets.QMessageBox.warning if errors else QtWidgets.QMessageBox.information)(
            self, tr("physics.validateTitle"), text)
        if issues and issues[0].entity_ids:
            self.store.set_selection(list(issues[0].entity_ids))

    def _on_about(self) -> None:
        from .dialogs.about import AboutDialog

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)      # 同梱物の確認に少しかかる
        try:
            dialog = AboutDialog(self)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        dialog.exec()

    def _on_units(self) -> None:
        current = self.store.document.meta.units
        units = self._ask_units(current)
        if units is None or units == current:
            return
        convert = self._ask_unit_conversion(current, units)
        if convert is None:
            return
        self.store.set_units(units, convert_values=convert)
        self.log_message(tr("units.changed", units=units))
        self.sketch_view.fit_sketch()

    def _ask_units(self, current: str) -> Optional[str]:
        units, ok = QtWidgets.QInputDialog.getItem(self, tr("units.title"), tr("units.choose"), list(UNITS),
                                                   UNITS.index(current), False)
        return units if ok else None

    def _ask_unit_conversion(self, old: str, new: str) -> Optional[bool]:
        """数値を換算する (True) / ラベルだけ (False) / キャンセル (None)."""
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, tr("units.title"),
                                    tr("units.prompt", old=old, new=new), parent=self)
        convert = box.addButton(tr("units.convert"), QtWidgets.QMessageBox.AcceptRole)
        box.addButton(tr("units.relabel"), QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or box.buttonRole(clicked) != QtWidgets.QMessageBox.AcceptRole:
            return None
        return clicked is convert

    def _on_axis_range(self) -> None:
        v = self.store.document.view
        current = "" if None in (v.zmin, v.zmax, v.rmin, v.rmax) else \
            ", ".join(f"{x:g}" for x in (v.zmin, v.zmax, v.rmin, v.rmax))
        text, ok = QtWidgets.QInputDialog.getText(self, tr("view.axisRange"), tr("view.axisRangePrompt"),
                                                  text=current)
        if not ok:
            return
        if not text.strip():
            self.store.set_view(zmin=None, zmax=None, rmin=None, rmax=None)
            self.sketch_view.fit_sketch()
            return
        try:
            values = [float(t) for t in text.replace("，", ",").split(",")]
            if len(values) != 4 or values[1] <= values[0] or values[3] <= values[2]:
                raise ValueError(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, tr("view.axisRange"), tr("view.axisRangeError"))
            return
        self.store.set_view(zmin=values[0], zmax=values[1], rmin=values[2], rmax=values[3])
        self.sketch_view.fit_view_settings()

    # ------------------------------------------------------------------
    # ブラウザツリー
    # ------------------------------------------------------------------

    def _refresh_tree(self) -> None:
        """ブラウザ: プロジェクトの構成（パラメータ / 形状 / 物理 / メッシュ / 解析 / 結果）.

        節をクリックすると対応するリボンのタブに切り替わる（SECTION_TABS）。作り直しても利用者が畳んだ節と
        スクロール位置は保つ。
        """
        tree = self.tree
        scroll = tree.verticalScrollBar().value()
        tree.blockSignals(True)
        tree.clear()
        store = self.store
        doc = store.document
        root = self._tree_section(None, "project", self.project.name)
        tree.addTopLevelItem(root)

        params = self._tree_section(root, "params", tr("tree.params", count=len(doc.params)))
        for p in doc.params:
            item = QtWidgets.QTreeWidgetItem([f"{p.name} = {p.expression}"])
            item.setData(0, QtCore.Qt.UserRole, ("param", p.id))
            params.addChild(item)

        counts = count_entities(doc.sketch)
        self._tree_section(root, "shape", tr("tree.shapeInfo", point=counts["point"], line=counts["line"],
                                             arc=counts["arc"], profiles=len(store.profiles)))

        physics = self._tree_section(root, "physics", tr("tree.physics"))
        regions = resolve_regions(doc, store.profiles)
        region_node = self._tree_section(physics, "regions", tr("tree.regions", count=len(regions)))
        for r in regions:
            text = (tr("tree.regionItemTan", name=r.name, eps=f"{r.eps_r:g}", tan=f"{r.tan_delta:g}")
                    if r.tan_delta > 0 else tr("tree.regionItemEps", name=r.name, eps=f"{r.eps_r:g}"))
            item = QtWidgets.QTreeWidgetItem([text])
            item.setData(0, QtCore.Qt.UserRole, ("region", r.profile.id))
            region_node.addChild(item)
        curves = [e for e in doc.sketch.entities if isinstance(e, (SketchLine, SketchArc)) and not e.construction]
        by_bc: dict[str, list[str]] = {bc: [] for bc in BC_NAMES}
        for c in curves:
            by_bc.setdefault(store.effective_bc(c.id)[0], []).append(c.id)
        boundary_node = self._tree_section(physics, "boundaries", tr("tree.boundaries", count=len(curves)))
        for bc, ids in by_bc.items():
            item = QtWidgets.QTreeWidgetItem([tr("tree.boundaryCount", name=bc, count=len(ids))])
            item.setIcon(0, chip_icon(BC_COLORS.get(bc, "#888888")))
            item.setData(0, QtCore.Qt.UserRole, ("bc", bc))
            if not ids:
                item.setForeground(0, QtGui.QColor("#9ca3af"))
            boundary_node.addChild(item)

        self._tree_mesh_label = self._mesh_tree_label()
        mesh_node = self._tree_section(root, "mesh", self._tree_mesh_label)
        mesh_state = "running" if self.mesh.running else self.mesh.state()
        mesh_node.setToolTip(0, tr(f"mesh.state.{mesh_state}"))
        if mesh_state in ("settings", "geometry"):
            mesh_node.setForeground(0, QtGui.QColor("#b45309"))
        self._tree_section(root, "analysis", self._analysis_tree_label())
        root.addChild(self._results_tree_item())

        # 畳んだ節だけ覚えている（新しい節は開いて出す）
        for item in self._tree_items(root):
            if item.childCount():
                item.setExpanded(self._tree_key(item) not in self._tree_collapsed)
        tree.doItemsLayout()                          # スクロール範囲を確定させてから位置を戻す
        tree.verticalScrollBar().setValue(scroll)
        tree.blockSignals(False)

    def _tree_section(self, parent: Optional[QtWidgets.QTreeWidgetItem], name: str,
                      text: str) -> QtWidgets.QTreeWidgetItem:
        """ブラウザの節（クリックで SECTION_TABS のタブを開く）."""
        item = QtWidgets.QTreeWidgetItem([text])
        item.setData(0, QtCore.Qt.UserRole, ("section", name))
        if SECTION_TABS.get(name):
            item.setToolTip(0, tr("tree.sectionHint"))
        if parent is not None:
            parent.addChild(item)
        return item

    @classmethod
    def _tree_items(cls, item: QtWidgets.QTreeWidgetItem):
        """item とその子孫（深さ優先）."""
        yield item
        for i in range(item.childCount()):
            yield from cls._tree_items(item.child(i))

    @staticmethod
    def _tree_key(item: QtWidgets.QTreeWidgetItem) -> tuple:
        data = item.data(0, QtCore.Qt.UserRole)
        return tuple(data) if data else ("text", item.text(0))

    def tree_item(self, data: tuple) -> Optional[QtWidgets.QTreeWidgetItem]:
        """UserRole のデータが ``data`` の項目（テストと結線の確認用）."""
        for i in range(self.tree.topLevelItemCount()):
            for item in self._tree_items(self.tree.topLevelItem(i)):
                value = item.data(0, QtCore.Qt.UserRole)
                if value and tuple(value) == tuple(data):
                    return item
        return None

    # ---- 結果（プロジェクトの履歴） ----

    def _result_headline(self, entry: ResultEntry) -> str:
        modes = (entry.summary or {}).get("modes") or []
        freqs = [m.get("f_GHz") for m in modes if isinstance(m, dict) and m.get("f_GHz") is not None]
        if not freqs:
            return ""
        if len(freqs) == 1:
            return tr("project.headline.mode1", f=f"{freqs[0]:.4f}")
        return tr("project.headline.modes", count=len(freqs), fMin=f"{min(freqs):.4f}", fMax=f"{max(freqs):.4f}")

    def _result_label(self, entry: ResultEntry) -> str:
        name = entry.label or tr(f"project.kind.{entry.kind}")
        text = f"#{entry.number} {name}"
        headline = self._result_headline(entry) if entry.status == "done" else ""
        if headline:
            text += f" — {headline}"
        notes = []
        if entry.status != "done":
            notes.append(tr(f"project.status.{entry.status}"))
        relation = self.project.result_relation(entry)
        if relation in ("geometry", "mesh", "settings"):
            notes.append(tr(f"project.relation.{relation}"))
        if notes:
            text += f"（{', '.join(notes)}）"
        return text

    def _results_tree_item(self) -> QtWidgets.QTreeWidgetItem:
        project = self.project
        node = QtWidgets.QTreeWidgetItem([tr("tree.results", count=len(project.results))])
        node.setData(0, QtCore.Qt.UserRole, ("section", "results"))
        node.setToolTip(0, tr("tree.sectionHint"))
        if project.path is None:
            node.setToolTip(0, tr("tree.resultsUnsaved"))
            node.setForeground(0, QtGui.QColor("#6b7280"))
        for entry in project.results:
            item = QtWidgets.QTreeWidgetItem([self._result_label(entry)])
            item.setData(0, QtCore.Qt.UserRole, ("result", entry.id))
            item.setToolTip(0, tr("tree.resultTooltip", created=entry.created_at.replace("T", " "),
                                  status=tr(f"project.status.{entry.status}"), dir=str(entry.dir)))
            if entry.status == "running":
                item.setForeground(0, QtGui.QColor("#1d4ed8"))
            elif entry.status == "error":
                item.setForeground(0, QtGui.QColor("#b91c1c"))
            elif entry.status != "done" or project.result_relation(entry) == "geometry":
                item.setForeground(0, QtGui.QColor("#9ca3af"))
            if entry.id == project.active_result_id:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            node.addChild(item)
        return node

    def _select_result(self, result_id: str) -> None:
        """結果を表示中にする（結果タブを開き、結果ビューに場を描く）."""
        entry = self.project.result(result_id)
        if entry is None or entry.status == "running":
            return
        self._show_tab("results")
        if self.project.select_result(result_id):
            self.log_message(tr("project.selected", label=self._result_label(entry)))
        entry = self.project.result(result_id) or entry
        if not self.result_model.load_entry(entry):
            self.log_message(tr("results.openError", message=self.result_model.error))

    def _on_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.tree.itemAt(pos)
        data = item.data(0, QtCore.Qt.UserRole) if item is not None else None
        if not data or data[0] != "result":
            return
        entry = self.project.result(data[1])
        if entry is None:
            return
        menu = QtWidgets.QMenu(self)
        rename = menu.addAction(tr("project.rename"))
        open_folder = menu.addAction(tr("project.openFolder"))
        menu.addSeparator()
        delete = menu.addAction(tr("project.delete"))
        delete.setEnabled(entry.status != "running")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is rename:
            label, ok = QtWidgets.QInputDialog.getText(self, tr("project.rename"), tr("project.renamePrompt"),
                                                       text=entry.label)
            if ok:
                self.project.rename_result(entry.id, label)
        elif chosen is open_folder:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(entry.dir)))
        elif chosen is delete:
            label = self._result_label(entry)
            if QtWidgets.QMessageBox.question(
                    self, tr("project.delete"), tr("project.deleteConfirm", label=label)) \
                    != QtWidgets.QMessageBox.Yes:
                return
            try:
                self.project.delete_result(entry.id)
            except (OSError, RuntimeError) as exc:
                QtWidgets.QMessageBox.critical(self, tr("project.delete"), str(exc))
                return
            self.log_message(tr("project.deleted", label=label))

    def _on_tree_expanded(self, item: QtWidgets.QTreeWidgetItem, expanded: bool) -> None:
        key = self._tree_key(item)
        if expanded:
            self._tree_collapsed.discard(key)
        else:
            self._tree_collapsed.add(key)

    def _mesh_tree_label(self) -> str:
        m = self.store.document.mesh
        size = m.sizeExpr if m.sizeExpr else f"{m.size:g}"
        mesh = self.mesh
        extra = ""
        if mesh.running:
            extra = tr("tree.meshRunning")
        elif mesh.info is not None:
            elements = mesh.info.stats.get("elements")
            extra = tr("tree.meshElements", count=f"{elements:,}" if isinstance(elements, int) else "-")
            if mesh.state() != "current":
                extra += tr("tree.meshStale")
        return tr("tree.meshNode3", size=size, units=self.store.document.meta.units, order=m.order, extra=extra)

    def _analysis_tree_label(self) -> str:
        a = self.store.document.analysis
        wave = tr("tree.waveStanding") if a.wave == "standing" else tr("tree.waveTraveling", phases=a.phases)
        if a.type == "hom":
            return tr("tree.analysisHom", orders=a.azOrders, wave=wave, modes=a.numModes)
        return tr("tree.analysisTm0", wave=wave, modes=a.numModes)

    def _on_tree_clicked(self, item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
        """クリック: 節は対応するタブへ。項目は選択（領域・境界条件の曲線をハイライト）."""
        data = item.data(0, QtCore.Qt.UserRole)
        if not data:
            return
        kind, value = data
        store = self.store
        if kind == "section":
            tab = SECTION_TABS.get(value)
            if tab is not None:
                self._show_tab(tab)
            if value == "params":
                store.set_params_edit(True)
        elif kind == "param":
            self._show_tab("modeling")
            store.set_params_edit(True)
        elif kind == "result":
            self._select_result(value)
        elif kind == "region":
            self._show_tab("physics")
            store.set_selected_region(None if store.selected_region_id == value else value)
        elif kind == "bc":
            self._show_tab("physics")
            sketch = store.active_sketch()
            store.set_selection([e.id for e in sketch.entities
                                 if isinstance(e, (SketchLine, SketchArc)) and not e.construction
                                 and store.effective_bc(e.id)[0] == value])

    # ------------------------------------------------------------------
    # ストアへの追従
    # ------------------------------------------------------------------

    def _sync_tool_actions(self, tool: str) -> None:
        for tool_id, act in self.tool_actions.items():
            act.blockSignals(True)
            act.setChecked(tool_id == tool)
            act.blockSignals(False)

    def _sync_params_actions(self, editing: bool) -> None:
        self.params_action.blockSignals(True)
        self.params_action.setChecked(editing)
        self.params_action.blockSignals(False)

    def _sync_edit_actions(self) -> None:
        points = self._selected(SketchPoint)
        lines = self._selected(SketchLine)
        arcs = self._selected(SketchArc)
        curves = lines + arcs
        self.construction_action.setEnabled(bool(self.store.selection))
        self.insert_point_action.setEnabled(len(curves) == 1)
        self.delete_reconnect_action.setEnabled(bool(points))
        self.fix_axis_action.setEnabled(bool(points or lines))
        self.fix_z0_action.setEnabled(bool(points or lines))
        for act in self.bc_actions.values():
            act.setEnabled(bool(curves))

    def _on_selection_changed(self) -> None:
        self._sync_edit_actions()
        curves = self._selected((SketchLine, SketchArc))
        if len(curves) == 1:
            bc, source = self.store.effective_bc(curves[0].id)
            self.bc_label.setText(tr("status.bc", bc=bc, source=tr(f"physics.source.{source}")))
        elif curves:
            self.bc_label.setText(tr("status.selected", count=len(self.store.selection)))
        else:
            self.bc_label.setText("")

    def _on_cursor(self, cursor) -> None:
        if cursor is None:
            self.cursor_label.setText("")
        else:
            self.cursor_label.setText(tr("status.cursor", x=f"{cursor[0]:.3f}", y=f"{cursor[1]:.3f}",
                                         units=self.store.document.meta.units))

    def _on_document_changed(self) -> None:
        self.undo_action.setEnabled(self.store.can_undo())
        self.redo_action.setEnabled(self.store.can_redo())
        self._sync_edit_actions()
        self._refresh_tree()
        self._update_title()
        self._sync_analysis_actions()                # 閉領域の有無でメッシュ生成・解析の可否が変わる
        for line in self.store.expression_errors:
            self.log_message(f"{tr('solve.invalid')}: {line}")

    def log_message(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ------------------------------------------------------------------
    # プロジェクト（.axiproj。保存・開く・結果の履歴は ProjectController）
    # ------------------------------------------------------------------

    def _update_title(self) -> None:
        project = self.project
        name = project.path.name if project.path is not None else project.name
        self.setWindowTitle(f"{tr('app.title')} — {name}[*]")
        self.setWindowModified(project.is_dirty())

    def _on_project_changed(self) -> None:
        root = self.project.mesh_dir
        if root != self._mesh_root:
            self._mesh_root = root
            self.mesh.reset()
            if root is not None:
                self.mesh.load(root)
        self._update_title()
        self._refresh_tree()
        self._sync_analysis_actions()
        model = self.result_model
        if model.entry is not None:
            model.refresh_entry(self.project.result(model.entry.id))   # post 後は読み直し、削除されたら消す

    def _confirm_discard(self, title: str) -> bool:
        """未保存の変更があれば 保存 / 破棄 / キャンセル を聞く。続けてよければ True."""
        if not self.project.is_dirty():
            return True
        buttons = (QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
                   | QtWidgets.QMessageBox.Cancel)
        answer = QtWidgets.QMessageBox.question(
            self, title, tr("file.unsavedChanges", name=self.project.name), buttons,
            QtWidgets.QMessageBox.Save)
        if answer == QtWidgets.QMessageBox.Save:
            return self._on_save()
        return answer == QtWidgets.QMessageBox.Discard

    def _blocked_by_run(self) -> bool:
        if self.project.running_dir is not None or self.session.running:
            QtWidgets.QMessageBox.information(self, tr("app.title"), tr("file.runningBlock"))
            return True
        return False

    def _on_new(self) -> None:
        if self._blocked_by_run() or not self._confirm_discard(tr("ribbon.new")):
            return
        self.project.new()
        self.sketch_view.fit_sketch()
        self.log_message(tr("file.created"))

    def _on_open(self) -> None:
        if self._blocked_by_run() or not self._confirm_discard(tr("ribbon.open")):
            return
        start = str(self.project.path.parent) if self.project.path is not None else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("ribbon.open"), start, tr("file.openFilter"))
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        """プロジェクト（.axiproj）か形状ファイル（.gmshproj / .af）を開く（起動引数からも使う）."""
        path = Path(path)
        if path.suffix.lower() in RESULT_SUFFIXES:
            return self.open_result_file(path)
        try:
            warnings = self.project.open(path)
        except (ProjectFormatError, DocumentParseError, OSError, RuntimeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("ribbon.open"), tr("file.openError", message=str(exc)))
            return False
        self.sketch_view.fit_sketch()
        is_project = path.name.lower().endswith(PROJECT_SUFFIX)
        self.log_message(tr("file.opened" if is_project else "file.imported", path=str(path)))
        for line in warnings:
            self.log_message(tr("file.importWarning", message=line))
        add_recent_project(path)
        return True

    def _on_import(self, title_key: str, filter_key: str) -> None:
        if self._blocked_by_run() or not self._confirm_discard(tr(title_key)):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr(title_key), "", tr(filter_key))
        if path:
            self.open_path(path)

    def _on_import_gmshproj(self) -> None:
        self._on_import("ribbon.importGmshproj", "file.gmshprojFilter")

    def _on_import_superfish(self) -> None:
        self._on_import("ribbon.importSuperfish", "file.superfishFilter")

    def _on_save(self) -> bool:
        if self.project.path is None:
            return self._on_save_as()
        try:
            path = self.project.save()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("ribbon.save"), tr("file.saveError", message=str(exc)))
            return False
        self.log_message(tr("file.saved", path=str(path)))
        return True

    def _on_save_as(self) -> bool:
        if self._blocked_by_run():
            return False
        if self.project.path is not None:
            default = str(self.project.path)
        else:
            default = sanitize_file_name(self.project.name) + PROJECT_SUFFIX
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("ribbon.saveAs"), default,
                                                        tr("file.projectFilter"))
        if not path:
            return False
        try:
            written = self.project.save_as(path)
        except (OSError, RuntimeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("ribbon.saveAs"), tr("file.saveError", message=str(exc)))
            return False
        add_recent_project(written)
        self.log_message(tr("file.saved", path=str(written)))
        return True

    # ------------------------------------------------------------------
    # メッシュ（M6）・書き出し
    # ------------------------------------------------------------------

    def _on_edit_region(self) -> None:
        self._show_tab("physics")
        self.physics_panel.focus_regions()

    def _on_generate_mesh(self) -> None:
        if self.session.running:
            return
        if self.project.path is None:
            QtWidgets.QMessageBox.information(self, tr("mesh.generateTitle"), tr("mesh.saveBeforeMesh"))
            if not self._on_save_as():
                return
        try:
            mesh_root = self.project.prepare_mesh()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("mesh.generateTitle"), tr("file.saveError", message=str(exc)))
            return
        self.log_message(tr("mesh.autoSaved", path=str(self.project.path)))
        self._show_tab("mesh")
        if not self.mesh.generate(mesh_root):
            QtWidgets.QMessageBox.warning(self, tr("mesh.generateTitle"), self.mesh.last_error or "")

    def _on_toggle_mesh_visible(self, checked: bool) -> None:
        self.mesh.set_visible(bool(checked))
        if checked and self.mesh.visible:
            self._show_tab("mesh")

    def _open_mesh_folder(self) -> None:
        info = self.mesh.info
        if info is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(info.dir)))

    def _on_mesh_changed(self) -> None:
        mesh = self.mesh
        running = mesh.running
        self.mesh_generate_action.setEnabled(not self.session.running and bool(self.store.profiles))
        self.mesh_generate_action.setVisible(not running)
        self.mesh_stop_action.setVisible(running)
        self.mesh_show_action.setEnabled(mesh.info is not None)
        self.mesh_show_action.blockSignals(True)
        self.mesh_show_action.setChecked(mesh.visible)
        self.mesh_show_action.blockSignals(False)
        self.mesh_folder_action.setEnabled(mesh.info is not None)
        self.export_msh_action.setEnabled(not self.session.running)
        geometry = mesh.preview_geometry() if mesh.visible else None
        if geometry is not self._mesh_preview_source:
            self._mesh_preview_source = geometry
            self.sketch_view.set_mesh_preview(MeshPreview(**geometry) if geometry else None)
        if self._mesh_tree_label() != self._tree_mesh_label:
            self._refresh_tree()                     # 進捗のたびには作り直さない

    def _on_job_finished(self, job) -> None:
        if job.kind == "analysis" and job.status == "done":
            entry = self.project.result(job.id)
            if entry is not None and entry.status == "done":
                self._select_result(job.id)          # 完了した結果を表示する
        elif job.kind == "export":
            self._on_export_field_finished(job)
        if job is self._export_job:
            self._export_job = None
            if job.status == "done":
                path = (job.result or {}).get("files", {}).get("mesh", "")
                self.log_message(tr("mesh.exported", path=path))
                self.statusBar().showMessage(tr("mesh.exported", path=path), 8000)
            elif job.status == "error":
                QtWidgets.QMessageBox.critical(self, tr("mesh.exportTitle"),
                                               tr("file.exportError", message=job.error or ""))
            shutil.rmtree(job.dir, ignore_errors=True)
        self._on_mesh_changed()

    def _geometry_for_export(self, title: str):
        try:
            return to_multi_region(self.store.document, strict=True)
        except ConvertError as exc:
            QtWidgets.QMessageBox.warning(self, title, str(exc))
            return None

    def _on_export_msh(self) -> None:
        title = tr("mesh.exportTitle")
        if self.session.running:
            QtWidgets.QMessageBox.information(self, title, tr("file.runningBlock"))
            return
        converted = self._geometry_for_export(title)
        if converted is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, f"{self.project.name}.msh", tr("file.mshFilter"))
        if not path:
            return
        job_dir = Path(tempfile.mkdtemp(prefix="axicavity_msh_"))
        converted.geom.save_json(job_dir / "geometry.json")
        (job_dir / "job.json").write_text(json.dumps(
            {"kind": "export_msh", "meshOrder": int(self.store.document.mesh.order), "outPath": str(path)},
            ensure_ascii=False), encoding="utf-8")
        self._export_job = self.session.submit("export_msh", job_dir, job_id="export_msh")
        self.log_message(tr("mesh.exporting", path=path))
        self._on_mesh_changed()

    def _on_export_geo(self) -> None:
        title = tr("ribbon.exportGeo")
        converted = self._geometry_for_export(title)
        if converted is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, f"{self.project.name}.geo", tr("file.geoFilter"))
        if not path:
            return
        from ...shared.gmsh_export_occ import export_geo_multi_region

        try:
            export_geo_multi_region(converted.geom, path, mesh_order=int(self.store.document.mesh.order))
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, title, tr("file.exportError", message=str(exc)))
            return
        self.log_message(tr("file.exported", path=path))

    def _on_export_python(self) -> None:
        title = tr("ribbon.exportPython")
        converted = self._geometry_for_export(title)
        if converted is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, f"{self.project.name}.py",
                                                        tr("file.pythonFilter"))
        if not path:
            return
        from ...shared.gmsh_export_occ import export_python_script_multi_region

        try:
            export_python_script_multi_region(converted.geom, path, mesh_order=int(self.store.document.mesh.order))
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, title, tr("file.exportError", message=str(exc)))
            return
        self.log_message(tr("file.exported", path=path))

    # ------------------------------------------------------------------
    # 解析（M7）
    # ------------------------------------------------------------------

    def _target_result(self):
        """post / レポートの対象: 表示中の結果、無ければ一番新しい完了した結果."""
        project = self.project
        shown = self.result_model.entry
        entry = project.result(shown.id if shown is not None else project.active_result_id)
        if entry is not None and entry.status == "done":
            return entry
        return next((e for e in project.results if e.status == "done"), None)

    def _sync_analysis_actions(self) -> None:
        analysis, session = self.analysis, self.session
        running = analysis.running
        self.run_action.setVisible(not running)
        self.run_action.setEnabled(not session.running and bool(self.store.profiles))
        self.cancel_action.setVisible(running)
        self.force_stop_action.setVisible(running)
        target = self._target_result()
        self.post_action.setEnabled(not session.running and target is not None)
        self.report_action.setEnabled(not session.running and target is not None)
        self.open_report_action.setEnabled(analysis.report_path(target) is not None)
        self._on_mesh_changed()
        self._sync_results_actions()

    def _on_run_analysis(self) -> None:
        if self.session.running:
            QtWidgets.QMessageBox.information(self, tr("analysis.ribbon.run"), tr("file.runningBlock"))
            return
        errors, warnings = self.analysis.check()
        if errors:
            QtWidgets.QMessageBox.warning(self, tr("analysis.ribbon.run"),
                                          tr("analysis.cannotRun", message="\n".join(errors)))
            self._show_tab("analysis")
            return
        if self.project.path is None:
            QtWidgets.QMessageBox.information(self, tr("analysis.ribbon.run"), tr("analysis.saveBeforeRun"))
            if not self._on_save_as():
                return
        store = self.store
        curves = [e for e in store.active_sketch().entities
                  if isinstance(e, (SketchLine, SketchArc)) and not e.construction]
        bc_counts: dict[str, int] = {}
        for c in curves:
            bc_counts[store.effective_bc(c.id)[0]] = bc_counts.get(store.effective_bc(c.id)[0], 0) + 1
        reuse = self.mesh.reusable_mesh()
        elements = self.mesh.info.stats.get("elements") if (reuse is not None and self.mesh.info) else None
        kind = self.analysis.commands()["kind"]
        output = str(self.project.results_dir / f"{self._next_result_number():04d}-{kind}")
        sections = run_summary(store.document, store.resolve_regions(), bc_counts, reuse is not None, elements,
                               output, self.analysis.preview_lines(), warnings)
        dialog = RunConfirmDialog(sections, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            ok = self.analysis.run()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("analysis.ribbon.run"), tr("file.saveError", message=str(exc)))
            return
        if not ok:
            QtWidgets.QMessageBox.warning(self, tr("analysis.ribbon.run"),
                                          tr("analysis.cannotRun", message=self.analysis.last_error or ""))
            return
        self.log_message(tr("analysis.autoSaved", path=str(self.project.path)))
        self.log_message(tr("analysis.submitted", id=self.analysis.job.id))
        self._show_tab("analysis")

    def _next_result_number(self) -> int:
        from ..io.axiproj import next_result_id
        if self.project.data_dir is None:
            return 1
        return int(next_result_id(self.project.data_dir, "x").split("-", 1)[0])

    def _on_force_stop(self) -> None:
        if not self.analysis.running:
            return
        if QtWidgets.QMessageBox.question(self, tr("analysis.ribbon.forceStop"),
                                          tr("analysis.forceStopConfirm")) != QtWidgets.QMessageBox.Yes:
            return
        self.analysis.cancel(force=True)

    def _on_run_post(self) -> None:
        entry = self._target_result()
        if entry is None:
            QtWidgets.QMessageBox.information(self, tr("analysis.ribbon.post"), tr("analysis.noResult"))
            return
        if self.analysis.run_post(entry):
            self.log_message(tr("analysis.postSubmitted", id=entry.id))
        elif self.analysis.last_error:
            QtWidgets.QMessageBox.warning(self, tr("analysis.ribbon.post"), self.analysis.last_error)

    def _on_create_report(self) -> None:
        entry = self._target_result()
        if entry is None:
            QtWidgets.QMessageBox.information(self, tr("analysis.ribbon.report"), tr("analysis.noResult"))
            return
        if self.analysis.run_report(entry):
            self.log_message(tr("analysis.reportSubmitted", id=entry.id))
        elif self.analysis.last_error:
            QtWidgets.QMessageBox.warning(self, tr("analysis.ribbon.report"), self.analysis.last_error)

    def _on_open_report(self) -> None:
        path = self.analysis.report_path(self._target_result())
        if path is None:
            QtWidgets.QMessageBox.information(self, tr("analysis.ribbon.openReport"), tr("analysis.noReport"))
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------
    # 結果（M8）: 結果ビュー・表示オプション・GIF・場の書き出し・PNG・外部の h5
    # ------------------------------------------------------------------

    def _sync_results_actions(self) -> None:
        model = self.result_model
        has = model.has_data
        for option, act in self.result_option_actions.items():
            act.blockSignals(True)
            act.setChecked(bool(getattr(model.options, option)))
            act.blockSignals(False)
            act.setEnabled(has)
        self.result_option_actions["show_lines"].setVisible(not model.is_hom)
        self.result_option_actions["show_e_wall"].setVisible(not model.is_hom)
        self.gif_action.setEnabled(has and model.is_traveling)
        self._menu_buttons["results.ribbon.exportField"].setEnabled(has and not self.session.running)
        self.png_action.setEnabled(has)
        self.result_folder_action.setEnabled(model.path is not None)
        available = view3d_window.is_available()
        self.view3d_action.setEnabled(has and available)
        if not available:
            self.view3d_action.setToolTip(ribbon_tooltip(tr("results.ribbon.view3d"), tr("view3d.unavailable")))

    def _on_toggle_view3d(self, checked: bool) -> None:
        """3D 表示の別ウィンドウ（初めて押したときに作る。pyvista が無ければ案内して戻す）."""
        if not checked:
            if self.view3d is not None:
                self.view3d.hide()
            return
        if self.view3d is None:
            if not view3d_window.is_available():
                QtWidgets.QMessageBox.information(self, tr("results.ribbon.view3d"), tr("view3d.unavailable"))
                self._set_view3d_checked(False)
                return
            try:
                self.view3d = view3d_window.View3DWindow(self.view3d_controller, self)
            except Exception as exc:  # noqa: BLE001 — VTK が作れない環境
                QtWidgets.QMessageBox.critical(self, tr("results.ribbon.view3d"),
                                               tr("view3d.openError", message=f"{type(exc).__name__}: {exc}"))
                self._set_view3d_checked(False)
                return
            self.view3d.closed.connect(lambda: self._set_view3d_checked(False))
        self.view3d.open_window()

    def _set_view3d_checked(self, checked: bool) -> None:
        self.view3d_action.blockSignals(True)
        self.view3d_action.setChecked(checked)
        self.view3d_action.blockSignals(False)

    def open_result_file(self, path: str | Path) -> bool:
        """プロジェクト外の結果 h5 を結果ビューで開く（CLI や ver2.3 の出力。起動引数からも）."""
        if not self.result_model.load_file(path):
            QtWidgets.QMessageBox.critical(self, tr("results.openTitle"),
                                           tr("results.openError", message=self.result_model.error))
            return False
        self.project.select_result(None)
        self._show_tab("results")
        self.log_message(tr("results.opened", path=str(path)))
        return True

    def _on_open_result(self) -> None:
        start = str(self.project.path.parent) if self.project.path is not None else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("results.openTitle"), start,
                                                        tr("results.openFilter"))
        if path:
            self.open_result_file(path)

    def _on_open_result_folder(self) -> None:
        path = self.result_model.path
        if path is None:
            QtWidgets.QMessageBox.information(self, tr("results.ribbon.openFolder"), tr("results.noFolder"))
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.parent)))

    def _on_field_pick(self, z: float, r: float) -> None:
        model = self.result_model
        lines = model.renderer.describe_field(model.selection, z, r) if model.renderer is not None else None
        if lines is None:
            self.statusBar().showMessage(tr("results.outside", z=f"{z:.4f}", r=f"{r:.4f}"), 4000)
            return
        self.results_panel.show_field_values(lines)
        QtWidgets.QMessageBox.information(self, tr("results.fieldTitleDialog"), "\n".join(lines))

    def _on_save_png(self) -> None:
        model = self.result_model
        if not model.has_data:
            return
        default = str(model.output_base("")) + ".png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("results.pngTitle"), default, tr("results.pngFilter"))
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.result_view.save_png(path, dpi=int(self.store.document.report.dpi))
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("results.pngTitle"), tr("file.exportError", message=str(exc)))
            return
        self.log_message(tr("results.pngSaved", path=path))

    def _on_save_gif(self) -> None:
        model = self.result_model
        if not model.has_data or model.renderer is None:
            return
        if not model.is_traveling:
            QtWidgets.QMessageBox.information(self, tr("results.ribbon.gif"), tr("results.gifNeedsTraveling"))
            return
        dialog = GifDialog(str(model.output_base("_anim")) + ".gif", parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        params = dialog.params()
        try:
            out = save_gif_with_progress(model.renderer, model.selection, model.options, params,
                                         self.result_view.canvas_size(), self)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, tr("gif.title"), tr("results.gifError", message=str(exc)))
            return
        if out is None:
            self.log_message(tr("results.gifCancelled"))
        else:
            self.log_message(tr("results.gifSaved", path=str(out), frames=params["n_frames"], fps=params["fps"]))
            self.statusBar().showMessage(tr("results.gifSaved", path=str(out), frames=params["n_frames"],
                                            fps=params["fps"]), 8000)

    def _on_export_field(self, shape: str) -> None:
        """場の書き出し（area / line / axis）: ダイアログ → 子プロセスで ``export``（command.log に残す）."""
        model = self.result_model
        data, sel = model.data, model.selection
        if data is None or model.path is None:
            return
        if self.session.running:
            QtWidgets.QMessageBox.information(self, tr("results.exportTitle"), tr("file.runningBlock"))
            return
        zmin, zmax, rmin, rmax = data.bounds
        dialog = ExportFieldDialog(shape, str(model.output_base(f"_field_{shape}")), (zmin, zmax), (rmin, rmax),
                                   traveling_tm0=(not data.is_hom and model.is_traveling),
                                   has_post=data.post_params(sel) is not None, parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        params = dialog.params()
        output = params.pop("output")
        entry = model.entry
        job_dir: Optional[Path] = None
        if entry is not None:
            base = entry.dir.resolve()

            def rel(p) -> str:
                try:
                    return Path(p).resolve().relative_to(base).as_posix()
                except ValueError:
                    return str(p)
            input_arg, output_arg = rel(model.path), rel(output)
        else:
            input_arg, output_arg = str(model.path), str(output)
            job_dir = Path(tempfile.mkdtemp(prefix="axicavity_export_"))
            self._export_field_tmp = job_dir
        argv = export_argv(data.solver_type, input_arg, output_arg, shape, sel.mode,
                           n=sel.n if data.is_hom else None, phase=sel.phase, time_phase=sel.time_phase,
                           params=params)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        if not self.analysis.run_export(argv, entry=entry, job_dir=job_dir):
            QtWidgets.QMessageBox.warning(self, tr("results.exportTitle"), self.analysis.last_error or "")
            return
        self.log_message(tr("results.exportSubmitted", output=output))

    def _on_export_field_finished(self, job) -> None:
        if job.status == "done":
            output = (job.result or {}).get("files", {}).get("export", "")
            self.log_message(tr("results.exportDone", output=output))
            self.statusBar().showMessage(tr("results.exportDone", output=output), 8000)
        elif job.status == "error":
            QtWidgets.QMessageBox.critical(self, tr("results.exportTitle"),
                                           tr("results.exportError", message=job.error or ""))
        if self._export_field_tmp is not None:
            shutil.rmtree(self._export_field_tmp, ignore_errors=True)
            self._export_field_tmp = None

    def _export(self, title: str, filter_key: str, suffix: str, writer) -> bool:
        default = f"{self.project.name}{suffix}"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, default, tr(filter_key))
        if not path:
            return False
        try:
            warnings = writer(self.store.document, path)
        except Exception as exc:  # noqa: BLE001 — 変換できない形状など
            QtWidgets.QMessageBox.critical(self, title, tr("file.exportError", message=str(exc)))
            return False
        self.log_message(tr("file.exported", path=path))
        for line in warnings or []:
            self.log_message(tr("file.importWarning", message=line))
        return True

    def _on_export_gmshproj(self) -> bool:
        return self._export(tr("ribbon.exportGmshproj"), "file.gmshprojFilter", GMSHPROJ_SUFFIX, save_gmshproj)

    def _on_export_superfish(self) -> bool:
        return self._export(tr("ribbon.exportSuperfish"), "file.superfishFilter", SUPERFISH_SUFFIX, save_superfish)

    def _rebuild_recent_menu(self) -> None:
        menu = self.recent_menu
        menu.clear()
        paths = [p for p in recent_projects() if Path(p).exists()]
        for p in paths:
            act = menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.setStatusTip(p)
            act.triggered.connect(lambda _=False, path=p: self._open_recent(path))
        if not paths:
            empty = menu.addAction(tr("ribbon.noRecent"))
            empty.setEnabled(False)
        menu.addSeparator()
        menu.addAction(tr("ribbon.clearRecent"), clear_recent_projects)

    def _open_recent(self, path: str) -> None:
        if not self._confirm_discard(tr("ribbon.open")):
            return
        self.open_path(path)

    def _on_undo(self) -> None:
        if self.store.undo():
            self.log_message(tr("ribbon.undo"))

    def _on_redo(self) -> None:
        if self.store.redo():
            self.log_message(tr("ribbon.redo"))

    def _on_toggle_language(self) -> None:
        set_language("en" if language() == "ja" else "ja")
        save_language(language())
        self._retranslate()

    def _retranslate(self) -> None:
        self._update_title()
        self.recent_button.setText(tr("ribbon.recent"))
        self.recent_button.setToolTip(ribbon_tooltip(tr("ribbon.recent"), tr("tip.ribbon.recent")))
        self.language_action.setText(tr("ribbon.language") + " ja/en")
        self.language_action.setToolTip(ribbon_tooltip(tr("ribbon.language"), tr("tip.ribbon.language")))
        for i, tab in enumerate(RIBBON_TABS):
            self.ribbon.setTabText(i, tr(f"ribbon.{tab}"))
        for act, (key, _body, _hint) in self._tooltip_specs.items():
            act.setText(tr(key))
        for key, button in self._menu_buttons.items():
            button.setText(tr(key))
            button.setToolTip(ribbon_tooltip(tr(key), tr(f"tip.{key}")))
        for label, key in self._group_labels.items():
            label.setText(tr(key))
        for act in self._tooltip_specs:
            self._apply_tooltip(act)
        for dock, key in ((self._dock_tree, "tree.title"), (self._dock_props, "props.title"),
                          (self._dock_log, "dock.log")):
            dock.setWindowTitle(tr(key))
        self.sketch_view.retranslate()
        self.sketch_panel.retranslate()
        self.physics_panel.retranslate()
        self.mesh_panel.retranslate()
        self.analysis_panel.retranslate()
        self.results_panel.retranslate()
        self.result_view.invalidate()
        if self.view3d is not None:
            self.view3d.retranslate()
        self._refresh_tree()
        self._on_selection_changed()

    def closeEvent(self, event) -> None:
        if self.session.running and QtWidgets.QMessageBox.question(
                self, tr("app.title"), tr("file.runningClose")) != QtWidgets.QMessageBox.Yes:
            event.ignore()
            return
        if not self._confirm_discard(tr("app.title")):
            event.ignore()
            return
        save_window_layout(self.saveGeometry(), self.saveState(LAYOUT_VERSION))
        if self.view3d is not None:
            self.view3d.hide()
        self.session.shutdown()
        super().closeEvent(event)
