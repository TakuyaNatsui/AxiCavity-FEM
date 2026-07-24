import wx
import os
import sys
import json
import shutil
import threading
import subprocess
import webbrowser
from .main_frame_ui import MyFrameUI
from .multi_region_editor import (
    MultiRegionEditorPanel,
    MODE_AUTO_POINT_SEGMENT,
    MODE_ADD_POINTS,
    MODE_EDIT_POINTS,
    MODE_ADD_SEGMENTS,
    MODE_BUILD_LOOP,
    MODE_EDIT_BC,
)
from ..shared.multi_region_model import (
    MultiRegionGeometry, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, BC_NAMES,
)
from ..shared.expression_eval import (
    evaluate_variables, eval_scalar, expr_or_none, is_plain_number, EvalError,
)
from ..shared.superfish_io import (
    import_superfish, export_superfish, validate_superfish_exportable,
    SuperfishImportError, SuperfishExportError,
)
# ResultViewer は可視化層(reports)依存のため OnViewResults 内で遅延 import する

# Multi-Region Editor タブのラベル (インデックスは _mr_tab_index() で動的に取得)。
_MR_TAB_LABEL = "Multi-Region Editor"
_MR_MODES = [MODE_ADD_POINTS, MODE_EDIT_POINTS, MODE_ADD_SEGMENTS,
             MODE_BUILD_LOOP, MODE_EDIT_BC]

class MyFrame(MyFrameUI):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        
        # プロジェクト管理用
        self.last_mesh_path = None
        self.last_analysis_result_h5 = None
        self.last_processed_h5 = None
        self.last_hom_result_h5 = None
        tip = "単値: 120  /  スキャン: 開始:終了:ステップ  例) 0:180:20"
        self.fem_phase_shift_ctrl.SetToolTip(tip)
        self.fem_phase_shift_ctrl.Enable(False)  # Standing Wave がデフォルトのため無効化
        self.hom_order_ctrl.Enable(False)         # TM0 がデフォルトのため無効化
        self.predicted_frequency_ctrl.SetToolTip(
            "探索したい周波数 [GHz]（例: 2.856）。\n"
            "入力するとこの周波数に近いモードから順に計算する。\n"
            "空欄なら従来どおり自動推定（最低次モードから）。")

        # ステータスバーの作成
        self.statusbar = self.CreateStatusBar(1)
        self.statusbar.SetStatusText("Welcome!")

        # Multi-Region Editor (ver2.1 段階3) の初期化
        self.mr_editor_panel = MultiRegionEditorPanel(
            self.notebook_1_pane_3, parent_frame=self, statusbar=self.statusbar)
        self.sizer_mr_canvas.Add(self.mr_editor_panel, 1, wx.EXPAND, 0)
        self.sizer_mr_canvas.Layout()
        # 変数定義テーブル (wxGrid) を右ペインに手書きで追加 (ver2.2)
        self._build_mr_variables_grid()
        self.mr_current_project_path = None
        self.mr_selected_region_id = None
        # 既定: Simple モード → MODE_AUTO_POINT_SEGMENT
        self.mr_editor_panel.set_mode(MODE_AUTO_POINT_SEGMENT)
        self._apply_view_mode()
        self._on_mr_unit_change_apply()
        self._mr_apply_axes_limits()
        self._refresh_mr_lists()
        # Point/Center 座標欄の数値プレビュー (式入力時に評価値を隣に表示, ver2.3)
        self._ensure_mr_value_preview_labels()
        self._bind_mr_value_preview_events()
        self._refresh_mr_value_previews()

    # --- プロジェクト保存・読み込みロジック ---
    def _mr_tab_index(self) -> int:
        """Multi-Region Editor タブの現在のインデックスを返す。

        タブ構成が変わっても追随できるよう、ページラベルで動的に探す。
        """
        nb = self.notebook_1
        for i in range(nb.GetPageCount()):
            if nb.GetPageText(i) == _MR_TAB_LABEL:
                return i
        return 0

    def _active_tab_is_mr(self) -> bool:
        try:
            return self.notebook_1.GetSelection() == self._mr_tab_index()
        except Exception:
            return False

    def save_project(self, path):
        """現在の Multi-Region Editor の状態を .gmshproj (v2.2 schema) で保存する。"""
        try:
            geom = self.mr_editor_panel.get_geometry()
            geom.unit = self.mr_unit_choice.GetStringSelection()
            try:
                geom.mesh_size = self._mr_eval(self.mr_mesh_size_ctrl)
                geom.mesh_size_expr = expr_or_none(self.mr_mesh_size_ctrl.GetValue())
            except ValueError:
                pass
            geom.settings.update({
                "xmin": self.mr_xmin_ctrl.GetValue(),
                "xmax": self.mr_xmax_ctrl.GetValue(),
                "ymin": self.mr_ymin_ctrl.GetValue(),
                "ymax": self.mr_ymax_ctrl.GetValue(),
                "mesh_order": self.mr_mesh_order_radiobox.GetSelection(),
            })
            geom.save_json(path)
            self.mr_current_project_path = path
            self.mr_editor_panel.is_dirty = False
            self.on_mr_editor_changed()
            return True
        except Exception as e:
            wx.MessageBox(f"Failed to save Multi-Region project: {e}",
                          "Error", wx.OK | wx.ICON_ERROR)
            return False

    # ------------------------------------------------------------------
    # 変数定義テーブル (wxGrid, ver2.2)
    # ------------------------------------------------------------------
    def _build_mr_variables_grid(self):
        """変数定義テーブル (wx.grid.Grid) をモードレス別ウィンドウに作る。

        ver2.3: 右ペインが手狭になったため、表は専用のモードレスウィンドウ
        (`mr_variables_frame`) に移した。'Open Parametor Window...' ボタン
        (`OnBtnOpenParametor`) で表示し、閉じても Hide されるだけで
        グリッドの内容と geom.variables は保持される（メインフレーム終了時に
        子フレームとして一緒に破棄される）。
        列: Name | Expression | Value(評価結果, 読み取り専用)。
        """
        import wx.grid as gridlib
        frame = wx.Frame(self, wx.ID_ANY, "Parameters (Variables)",
                         size=(420, 480),
                         style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT)
        panel = wx.Panel(frame, wx.ID_ANY)
        grid = gridlib.Grid(panel, wx.ID_ANY)
        grid.CreateGrid(6, 3)
        grid.SetColLabelValue(0, "Name")
        grid.SetColLabelValue(1, "Expression")
        grid.SetColLabelValue(2, "Value")
        grid.SetColSize(0, 90)
        grid.SetColSize(1, 180)
        grid.SetColSize(2, 100)
        grid.SetRowLabelSize(0)          # 行番号ラベルを非表示
        grid.SetColLabelSize(20)
        grid.Bind(gridlib.EVT_GRID_CELL_CHANGED, self.OnMrVariableCellChanged)
        szr = wx.BoxSizer(wx.VERTICAL)
        szr.Add(wx.StaticText(panel, wx.ID_ANY,
                              "Variables  (例: a=100, b=10 → 座標欄に a+b)"),
                0, wx.ALL, 4)
        szr.Add(grid, 1, wx.EXPAND | wx.ALL, 2)
        panel.SetSizer(szr)
        frame.Bind(wx.EVT_CLOSE, self._on_mr_variables_frame_close)
        self.mr_variables_frame = frame
        self.mr_variables_grid = grid
        self._mr_readonly_value_col()    # Value 列 (col 2) を読み取り専用に
        # 旧・右ペイン埋め込み用プレースホルダは隠してスペースを空ける
        placeholder = getattr(self, "panel_mr_variables", None)
        if placeholder is not None:
            placeholder.Hide()
            self.sizer_mr_controls.Layout()

    def _on_mr_variables_frame_close(self, event):
        """パラメータウィンドウの×ボタン: 破棄せず隠すだけ（内容は保持）。"""
        self.mr_variables_frame.Hide()

    def OnBtnOpenParametor(self, event):
        """'Open Parametor Window...' ボタン: パラメータ表をモードレス表示する。"""
        frame = getattr(self, "mr_variables_frame", None)
        if frame is not None:
            frame.Show()
            frame.Raise()
        if event:
            event.Skip()

    def _mr_readonly_value_col(self):
        """変数グリッドの Value 列 (col 2) を全行読み取り専用にする。"""
        g = self.mr_variables_grid
        for r in range(g.GetNumberRows()):
            g.SetReadOnly(r, 2, True)

    def _mr_grid_rows(self):
        g = self.mr_variables_grid
        return [(g.GetCellValue(r, 0), g.GetCellValue(r, 1))
                for r in range(g.GetNumberRows())]

    def _sync_mr_variables_from_grid(self):
        """grid の内容を評価し、Value 列と geom.variables を更新する。"""
        g = self.mr_variables_grid
        rows = self._mr_grid_rows()
        _, results = evaluate_variables(rows)
        for r, res in enumerate(results):
            if g.GetCellValue(r, 2) != res:
                g.SetCellValue(r, 2, res)
        # 非空行のみ (name, expr) を永続化
        kept = [(n.strip(), e.strip()) for n, e in rows
                if n.strip() and e.strip()]
        self.mr_editor_panel.geom.variables = kept
        # 末尾行が埋まっていれば空行を追加 (自動拡張)
        last = g.GetNumberRows() - 1
        if last < 0 or g.GetCellValue(last, 0).strip() or g.GetCellValue(last, 1).strip():
            g.AppendRows(1)
            self._mr_readonly_value_col()

    def OnMrVariableCellChanged(self, event):
        self._sync_mr_variables_from_grid()
        # 変数が変わったら式を持つ座標・寸法・材料・arc を再計算して再描画 (ver2.2)
        errors = self.mr_editor_panel.recompute_from_variables(self.get_mr_variables())
        if errors:
            self.statusbar.SetStatusText(
                f"変数再計算: {len(errors)} 件のエラー ({errors[0]})")
        else:
            self.statusbar.SetStatusText("変数を更新し座標を再計算しました。")
        if hasattr(self.mr_editor_panel, "mark_dirty"):
            self.mr_editor_panel.mark_dirty()
        # 変数変更で式の評価値が変わるためプレビューも更新
        if hasattr(self, "_refresh_mr_value_previews"):
            self._refresh_mr_value_previews()
        if event:
            event.Skip()

    def get_mr_variables(self):
        """現在の評価済み変数辞書を返す (座標欄などの数式評価で使う)。"""
        try:
            variables, _ = evaluate_variables(self.mr_editor_panel.geom.variables)
            return variables
        except Exception:
            return {}

    def set_mr_variables_rows(self, rows):
        """geom.variables を grid に反映する (プロジェクト読込時)。"""
        g = self.mr_variables_grid
        if g.GetNumberRows() > 0:
            g.DeleteRows(0, g.GetNumberRows())
        n = max(len(rows) + 2, 6)
        g.AppendRows(n)
        for r, (name, expr) in enumerate(rows):
            g.SetCellValue(r, 0, str(name))
            g.SetCellValue(r, 1, str(expr))
        self._sync_mr_variables_from_grid()

    def _mr_eval(self, ctrl):
        """MR 入力欄の文字列を現在の変数で数式評価して float を返す (ver2.2)。

        数値でも数式 (例: "a+b", "sqrt(100)") でも可。失敗時は EvalError
        (=ValueError) を送出し、呼び出し側の except ValueError ハンドラが
        エラーダイアログを表示する。
        """
        return eval_scalar(ctrl.GetValue(), self.get_mr_variables())

    def _apply_mr_geometry_to_ui(self, geom, path):
        """MultiRegionGeometry を MR エディタと右ペイン UI に反映する。

        schema 2.1/2.2 の読込と旧形式 (単一領域) の変換読込で共用する。
        """
        self.mr_editor_panel.set_geometry(geom)
        self.set_mr_variables_rows(geom.variables)
        self.mr_current_project_path = path
        # settings から軸範囲・mesh_size 等を復元
        self.mr_unit_choice.SetStringSelection(geom.unit)
        self.mr_mesh_size_ctrl.SetValue(geom.mesh_size_expr or str(geom.mesh_size))
        s = geom.settings or {}
        self.mr_xmin_ctrl.SetValue(str(s.get("xmin", "0")))
        self.mr_xmax_ctrl.SetValue(str(s.get("xmax", "100")))
        self.mr_ymin_ctrl.SetValue(str(s.get("ymin", "0")))
        self.mr_ymax_ctrl.SetValue(str(s.get("ymax", "100")))
        if "mesh_order" in s:
            try:
                self.mr_mesh_order_radiobox.SetSelection(int(s["mesh_order"]))
            except (TypeError, ValueError):
                pass
        self._on_mr_unit_change_apply()
        self._mr_apply_axes_limits()
        self._refresh_mr_lists()
        self.notebook_1.SetSelection(self._mr_tab_index())
        self.on_mr_editor_changed()

    def load_project(self, path):
        """JSON形式の.gmshprojファイルから状態を復元する (すべて Multi-Region へ)。

        schema_version="2.1"/"2.2" は MR 形式としてそのまま読み込む。
        schema_version が無い旧 ver2 形式 (単一領域) は
        MultiRegionGeometry.from_legacy_single_loop で 1 Vacuum 領域に
        変換して読み込む。
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get("schema_version") in SUPPORTED_SCHEMA_VERSIONS:
                geom = MultiRegionGeometry.from_dict(data)
                self._apply_mr_geometry_to_ui(geom, path)
                return True

            # 旧形式 (schema_version なし) を Multi-Region 形式に変換して読込
            geom = MultiRegionGeometry.from_legacy_single_loop(data)
            self._apply_mr_geometry_to_ui(geom, None)  # 保存で v2.2 形式に上書きさせる
            self.statusbar.SetStatusText(
                "旧形式 (単一領域) を Multi-Region 形式に変換して読み込みました。"
                "保存すると v2.2 形式になります。")
            return True
        except Exception as e:
            wx.MessageBox(f"Failed to load project: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return False

    # --- メニューハンドラ ---
    def OnMenuSave(self, event):
        path = self.mr_current_project_path
        if path:
            self.save_project(path)
        else:
            self.OnMenuSaveAs(event)

    def OnMenuSaveAs(self, event):
        wildcard = "GMSH Project files (*.gmshproj)|*.gmshproj"
        with wx.FileDialog(self, "Save Project As", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.save_project(dlg.GetPath())

    def OnMenuLoad(self, event):
        if getattr(self.mr_editor_panel, "is_dirty", False):
            if wx.MessageBox("Current changes will be lost. Continue?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.NO:
                return
        wildcard = "GMSH Project files (*.gmshproj)|*.gmshproj"
        with wx.FileDialog(self, "Load Project", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.load_project(dlg.GetPath())

    def OnMenuImportSuperfish(self, event):
        """Superfish (.af) を Multi-Region Editor にインポートする。"""
        if getattr(self.mr_editor_panel, "is_dirty", False):
            if wx.MessageBox("Current changes will be lost. Continue?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.NO:
                return
        wildcard = "Superfish files (*.af)|*.af|All files (*.*)|*.*"
        with wx.FileDialog(self, "Import Superfish File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.mr_import_superfish(dlg.GetPath())

    # --- イベントハンドラ ---
    def on_mode_change(self, event):
        """Multi-Region Simple モードの Edit Mode radiobox 切替。"""
        obj = event.GetEventObject() if event else None
        # mr_modeSimple_radiobox または プログラム呼び出し (obj=None) のときのみ処理
        if obj is None or obj is getattr(self, "mr_modeSimple_radiobox", None):
            self._apply_mr_simple_mode()
        if event:
            event.Skip()

    def _apply_mr_simple_mode(self):
        """mr_modeSimple_radiobox の選択を Multi-Region エディタのモードに反映。

        index 0 "Edit Points" → MODE_AUTO_POINT_SEGMENT (描画/移動/ダブルクリック挿入)
        index 1 "Edit Lines"  → MODE_EDIT_BC (segment 選択 → BC/円弧編集)
        """
        idx = self.mr_modeSimple_radiobox.GetSelection()
        if idx == 1:
            self.mr_editor_panel.set_mode(MODE_EDIT_BC)
        else:
            self.mr_editor_panel.set_mode(MODE_AUTO_POINT_SEGMENT)

    def on_mr_mode_autoswitched(self, mode):
        """エディタがクリック対象に応じて自動でモードを切替えたとき、
        mr_modeSimple_radiobox の選択表示を同期する (SetSelection は
        EVT_RADIOBOX を発火しないため再帰しない)。"""
        rb = getattr(self, "mr_modeSimple_radiobox", None)
        if rb is None:
            return
        if mode == MODE_EDIT_BC:
            rb.SetSelection(1)
        elif mode == MODE_AUTO_POINT_SEGMENT:
            rb.SetSelection(0)

    def OnBtnResetAll(self, event):
        """New Project (Ctrl+N) / Reset All: Multi-Region データをリセットする。"""
        if wx.MessageBox("Reset all Multi-Region data?", "Confirm",
                         wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            self.mr_editor_panel.reset()
            self.mr_current_project_path = None
            self.mr_editor_panel.set_mode(MODE_AUTO_POINT_SEGMENT)
            if hasattr(self, "mr_modeSimple_radiobox"):
                self.mr_modeSimple_radiobox.SetSelection(0)
            self._update_mr_simple_ui_state()
            self._refresh_mr_lists()
            self.on_mr_selection_changed()
        if event:
            event.Skip()

    def OnBtnExportSuperfish(self, event):
        """Superfish (.af) エクスポート (Multi-Region Editor のデータを出力)。"""
        wildcard = "Superfish files (*.af)|*.af"
        with wx.FileDialog(self, "Export Superfish", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.mr_export_superfish(dlg.GetPath())
        if event: event.Skip()

    # --- FEM Analysis Logic ---
    def OnBrowseMesh(self, event):
        wildcard = "MSH files (*.msh)|*.msh|All files (*.*)|*.*"
        with wx.FileDialog(self, "Select Mesh File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.fem_mesh_path_ctrl.SetValue(dlg.GetPath())
        if event: event.Skip()

    def OnBrowseRawResult(self, event):
        wildcard = "HDF5 files (*.h5)|*.h5|All files (*.*)|*.*"
        with wx.FileDialog(self, "Select Raw Analysis Result", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.fem_raw_result_path_ctrl.SetValue(path)
                self.last_analysis_result_h5 = path
        if event: event.Skip()

    def OnBrowseProcessedResult(self, event):
        wildcard = "HDF5 files (*.h5)|*.h5|All files (*.*)|*.*"
        with wx.FileDialog(self, "Select Processed Result", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.fem_processed_result_path_ctrl.SetValue(path)
                self.last_processed_h5 = path
        if event: event.Skip()

    def on_radio_box_analysis_mode(self, event):
        """Analysis Mode 切替時：HOM固有コントロールの有効化とパスフィールドのクリア"""
        is_hom = self.radio_box_analysis_mode.GetSelection() == 1
        self.hom_order_ctrl.Enable(is_hom)
        self.fem_raw_result_path_ctrl.SetValue("")
        self.fem_processed_result_path_ctrl.SetValue("")
        if event: event.Skip()

    def on_fem_wave_type_radio(self, event):
        """Traveling Wave 選択時のみ Phase Shift 入力を有効化。

        Analysis Mode 切替時と同様、結果は無効化されるためパスフィールドをクリアする。
        """
        is_traveling = self.fem_wave_type_radio.GetSelection() == 1
        self.fem_phase_shift_ctrl.Enable(is_traveling)
        self.fem_raw_result_path_ctrl.SetValue("")
        self.fem_processed_result_path_ctrl.SetValue("")
        if event: event.Skip()

    def OnRunSolver(self, event):
        mesh_path = self.fem_mesh_path_ctrl.GetValue()
        if not mesh_path or not os.path.exists(mesh_path):
            wx.MessageBox(f"Mesh file not found:\n{mesh_path}", "Error", wx.OK | wx.ICON_ERROR)
            return

        is_hom       = self.radio_box_analysis_mode.GetSelection() == 1
        n_modes      = self.num_modes_ctrl.GetValue()
        order        = 2 if self.radio_mesh_order.GetSelection() == 1 else 1
        is_traveling = self.fem_wave_type_radio.GetSelection() == 1
        phase_str    = self.fem_phase_shift_ctrl.GetValue()
        solver_type  = "hom" if is_hom else "tm0"

        # Predicted frequency [GHz]: 空欄なら従来どおり sigma を自動推定
        target_freq_str = self.predicted_frequency_ctrl.GetValue().strip()
        if target_freq_str:
            try:
                target_freq = float(target_freq_str)
            except ValueError:
                wx.MessageBox(
                    f"Predicted frequency に数値を入力してください（GHz）:\n"
                    f"{target_freq_str}", "Error", wx.OK | wx.ICON_ERROR)
                return
            if target_freq <= 0.0:
                wx.MessageBox("Predicted frequency は正の値にしてください（GHz）。",
                              "Error", wx.OK | wx.ICON_ERROR)
                return
        else:
            target_freq = None

        output_path = self.fem_raw_result_path_ctrl.GetValue().strip()
        if not output_path:
            suffix = f"_{'TW' if is_traveling else 'SW'}_{solver_type.upper()}.h5"
            output_path = os.path.splitext(mesh_path)[0] + suffix
            self.fem_raw_result_path_ctrl.SetValue(output_path)

        # 統一 CLI: axicavity-fem solve --type {tm0,hom} ...
        cmd = self._build_base_cmd() + [
            "solve", "--type", solver_type,
            "-m", mesh_path,
            "--elem-order", str(order),
            "--num-modes", str(n_modes),
            "-o", output_path,
        ]
        if target_freq is not None:
            cmd += ["--target-freq", repr(target_freq)]
        if is_hom:
            order_str = self.hom_order_ctrl.GetValue().strip() or "0"
            cmd += ["--az-order"] + order_str.split()
        if is_traveling:
            if not phase_str:
                wx.MessageBox("Phase Shift を入力してください。", "Error", wx.OK | wx.ICON_ERROR)
                return
            cmd += ["-p", phase_str]

        self.last_analysis_result_h5 = output_path
        if is_hom:
            self.last_hom_result_h5 = output_path
        label = f"{solver_type.upper()} Solver" + (" (Traveling)" if is_traveling else "")
        self.run_command_async(cmd, label)

    def OnRunPost(self, event):
        is_hom  = self.radio_box_analysis_mode.GetSelection() == 1
        raw_path = self.fem_raw_result_path_ctrl.GetValue()
        if not raw_path:
            raw_path = self.last_hom_result_h5 if is_hom else self.last_analysis_result_h5
        if not raw_path or not os.path.exists(raw_path):
            wx.MessageBox("Raw analysis result not found. Please run Solver first or select a file.", "Error", wx.OK | wx.ICON_ERROR)
            return

        cond = self.fem_conductivity_ctrl.GetValue()
        solver_type = "hom" if is_hom else "tm0"
        output_h5 = os.path.splitext(raw_path)[0] + "_processed.h5"

        # 統一 CLI: axicavity-fem post --type {tm0,hom} -i RAW -o OUT --cond C [--beta B]
        cmd = self._build_base_cmd() + [
            "post", "--type", solver_type,
            "-i", raw_path, "-o", output_h5,
            "--cond", cond,
        ]
        if not is_hom:
            cmd += ["--beta", self.fem_beta_ctrl.GetValue()]
        label = f"{solver_type.upper()} Parameter Calculation"

        self.last_processed_h5 = output_h5
        self.fem_processed_result_path_ctrl.SetValue(output_h5)
        self.run_command_async(cmd, label)

    def OnBtnCreateReport(self, event):
        is_hom = self.radio_box_analysis_mode.GetSelection() == 1
        # processed があれば優先、なければ raw を使う
        path = self.fem_processed_result_path_ctrl.GetValue() or \
            self.last_processed_h5
        if not path or not os.path.exists(path):
            path = self.fem_raw_result_path_ctrl.GetValue() or \
                self.last_analysis_result_h5
        if not path or not os.path.exists(path):
            wx.MessageBox("結果ファイルが見つかりません。Solver/Post を先に実行してください。",
                          "Error", wx.OK | wx.ICON_ERROR)
            return

        solver_type = "hom" if is_hom else "tm0"
        out_dir = os.path.splitext(path)[0] + "_report"
        cmd = self._build_base_cmd() + [
            "report", "--type", solver_type, "-i", path, "-o", out_dir]
        # "Create animation" にチェックがあれば進行波に GIF を追加（--animate）
        if self.checkbox_report_anim.GetValue():
            cmd.append("--animate")
        self.run_command_async(cmd, f"{solver_type.upper()} HTML Report")

    def OnOpenReport(self, event):
        processed_path = self.fem_processed_result_path_ctrl.GetValue()
        if not processed_path:
            processed_path = self.last_processed_h5
        if not processed_path:
            wx.MessageBox("No processed result file specified.", "Info")
            return

        report_path = os.path.splitext(processed_path)[0] + "_report/index.html"
        if os.path.exists(report_path):
            webbrowser.open(f"file:///{os.path.abspath(report_path)}")
        else:
            wx.MessageBox(f"Report file not found:\n{report_path}", "Error")

    def OnViewResults(self, event):
        is_hom = self.radio_box_analysis_mode.GetSelection() == 1

        # processed があればそちらを優先、なければ raw を使う
        result_h5 = self.fem_processed_result_path_ctrl.GetValue()
        if not result_h5 or not os.path.exists(result_h5):
            result_h5 = self.fem_raw_result_path_ctrl.GetValue()
        if not result_h5 or not os.path.exists(result_h5):
            result_h5 = self.last_hom_result_h5 if is_hom else self.last_analysis_result_h5

        if not result_h5 or not os.path.exists(result_h5):
            wildcard = "HDF5 files (*.h5)|*.h5|All files (*.*)|*.*"
            with wx.FileDialog(self, "Select Analysis Result", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    result_h5 = dlg.GetPath()
                else:
                    return

        # ResultViewer は可視化層 (reports) 依存のため遅延 import する（段階7で実装）
        try:
            from .result_viewer import ResultViewer
        except Exception:
            wx.MessageBox(
                "結果ビューア (Result Viewer) は可視化層 (reports) の実装後に"
                "有効化されます（段階7）。",
                "Info", wx.OK | wx.ICON_INFORMATION)
            return

        dlg = ResultViewer(self, result_h5)
        dlg.ShowModal()
        dlg.Destroy()
        if event: event.Skip()

    def _build_base_cmd(self):
        """統一 CLI の起動コマンド（先頭部分）を返す.

        ``axicavity-fem`` が PATH にあればそれを、無ければ
        ``python -m axicavity_fem.cli.main`` をフォールバックとして使う。
        （shutil.which フォールバックをここに集約）
        """
        exe = shutil.which("axicavity-fem")
        if exe:
            return [exe]
        return [sys.executable, "-m", "axicavity_fem.cli.main"]

    def log_command(self, cmd, label):
        """コマンドをログファイルに記録する"""
        import datetime
        cmd_log = "command.log"
        with open(cmd_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {label}\n")
            f.write(f"{'='*70}\n")
            f.write(f"{' '.join(cmd)}\n\n")

    def run_command_async(self, cmd, label):
        self.fem_log_ctrl.AppendText(f"\n--- Starting {label} ---\nCommand: {' '.join(cmd)}\n")
        self.log_command(cmd, label)

        def target():
            try:
                # subprocess.Popen で実行。Windows の日本語環境 (CP932) に対応
                # cmd はリスト形式（_build_base_cmd 由来）なので shell=False。
                # Windows 日本語環境 (CP932) の出力デコードに対応。
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='cp932', errors='replace', shell=False
                )

                if process.stdout:
                    for line in process.stdout:
                        wx.CallAfter(self.fem_log_ctrl.AppendText, line)

                process.wait()
                wx.CallAfter(self.fem_log_ctrl.AppendText, f"--- {label} Finished (Exit Code: {process.returncode}) ---\n")

                if process.returncode == 0:
                    wx.CallAfter(wx.MessageBox, f"{label} completed successfully.", "Success")
                else:
                    wx.CallAfter(wx.MessageBox, f"{label} failed. See log for details.", "Error", wx.OK | wx.ICON_ERROR)

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                wx.CallAfter(self.fem_log_ctrl.AppendText, f"Error executing command: {e}\n{error_msg}\n")

        thread = threading.Thread(target=target)
        thread.start()

    # ==================================================================
    # Multi-Region Editor (ver2.1 段階3) のハンドラ
    # ==================================================================
    def _on_mr_unit_change_apply(self):
        unit = self.mr_unit_choice.GetStringSelection()
        self.mr_editor_panel.set_axes_labels(unit)

    def _mr_apply_axes_limits(self):
        try:
            xmin = self._mr_eval(self.mr_xmin_ctrl)
            xmax = self._mr_eval(self.mr_xmax_ctrl)
            ymin = self._mr_eval(self.mr_ymin_ctrl)
            ymax = self._mr_eval(self.mr_ymax_ctrl)
            self.mr_editor_panel.set_axes_limits(xmin, xmax, ymin, ymax)
        except ValueError:
            pass

    def _refresh_mr_lists(self):
        geom = self.mr_editor_panel.get_geometry()
        # Loops (OCC では向き自動補正のため orientation 表記なし)
        loop_labels = [self._loop_label(lp) for lp in geom.loops]
        self.mr_loop_list.Set(loop_labels)
        # Regions
        region_labels = [
            f"{r.id}: {r.name} (mat={r.material_tag}, eps_r={r.eps_r}"
            + (f", tan_d={r.tan_delta}" if r.tan_delta > 0 else "") + ")"
            for r in geom.regions]
        self.mr_region_list.Set(region_labels)
        # Region 編集フォームの選択肢 (Outer Loop / Holes)
        loop_choices = [f"loop {lp.id}" for lp in geom.loops]
        self.mr_region_outer_choice.Set(loop_choices)
        if loop_choices:
            self.mr_region_outer_choice.SetSelection(0)
        self.mr_region_holes_list.Set(loop_choices)

    def _loop_label(self, loop) -> str:
        return f"{loop.id}: segs={loop.segment_ids}"

    def _is_mr_simple_mode(self) -> bool:
        # mr_view_mode_radiobox は MyFrameUI に存在しないテスト経路もありうるので
        # 防御的に hasattr チェック
        rb = getattr(self, "mr_view_mode_radiobox", None)
        return rb is None or rb.GetSelection() == 0

    def _apply_view_mode(self):
        """Simple/Advanced 切替時に右ペインの該当 UI を Show/Hide する。

        Loop/Region/Edit Mode RadioBox は Advanced のみ表示。
        Simple では canvas クリック = 点追加+自動 segment 生成のみ。
        """
        simple = self._is_mr_simple_mode()
        '''
        # 非表示候補 (Advanced のみで使う)
        adv_widgets = []
        for name in ("mr_mode_radiobox", "mr_loop_list",
                     "mr_close_loop_button", "mr_cancel_loop_button",
                     "mr_delete_loop_button",
                     "mr_region_list", "mr_region_name_ctrl",
                     "mr_region_outer_choice", "mr_region_holes_list",
                     "mr_region_mat_ctrl", "mr_region_eps_ctrl",
                     "mr_add_region_button", "mr_update_region_button",
                     "mr_delete_region_button"):
            w = getattr(self, name, None)
            if w is not None:
                adv_widgets.append(w)
        # Loops / Regions の StaticBox 自体も隠したいが、wxGlade 生成では
        # StaticBoxSizer がローカル変数で `self.` 化されていない。代替として
        # 中身（ListBox/Button/TextCtrl）を全部 Hide すれば見た目上の効果は得られる。
        for w in adv_widgets:
            w.Show(not simple)
        '''
        self.panel_advanced_mode.Show(not simple)
        self.panel_simple_mode.Show(simple)

        # Simple のみ Point/Line Edit の自動切替を有効化
        self.mr_editor_panel.auto_mode_switch = simple

        # Simple ではエディタモードを Simple radiobox の選択に合わせる
        if simple:
            self._apply_mr_simple_mode()
            self._update_mr_simple_ui_state()

        # Simple モードでは Delete Segment を禁止 (ループが壊れるため)。
        # Advanced では有効。
        if hasattr(self, "mr_delete_segment_button"):
            self.mr_delete_segment_button.Enable(not simple)

        # レイアウト再計算
        try:
            self.panel_advanced_mode.Layout()
            self.panel_simple_mode.Layout()
            self.notebook_1_pane_3.Layout() ##親パネルをLayout()しなくては見た目が反映されない
            #self.sizer_mr_controls.Layout()
        except Exception:
            pass

        # ヒント文言
        if hasattr(self, "mr_hint_ctrl"):
            if simple:
                self.mr_hint_ctrl.SetValue(
                    "Simple モード: キャンバスをクリックして点を打つと、直前の点との"
                    "直線 segment が自動生成されます。3 点以上で [Close Loop] を押すと"
                    "Vacuum 領域 (eps_r=1) が自動作成されます。\n"
                    "[Edit Points]: 点をドラッグ移動、ライン上をダブルクリックで点を挿入。\n"
                    "[Edit Lines]: segment を選択して BC 変更や Convert to Arc。"
                )
            else:
                self.mr_hint_ctrl.SetValue(
                    "Advanced モード: Edit Mode で動作を切り替え、"
                    "複数領域や穴あき形状、誘電体 (eps_r ≠ 1) を扱えます。"
                )

    def _update_mr_simple_ui_state(self):
        """Simple モードの mr_modeSimple_radiobox の有効/無効を更新する。

        点が無いうちは編集モード切替を無効化 (Mesh Geometry Edit と同様)。
        """
        rb = getattr(self, "mr_modeSimple_radiobox", None)
        if rb is None:
            return
        n_pts = len(self.mr_editor_panel.get_geometry().points)
        rb.Enable(n_pts > 0)

    def on_mr_editor_changed(self):
        title = "AxiCavity-FEM v2.3 - Multi-Region"
        if self.mr_editor_panel.is_dirty:
            title += " *"
        # 既存タイトル更新と衝突しないように MR タブ時のみセット
        if self._active_tab_is_mr():
            self.SetTitle(title)
        self._refresh_mr_lists()
        self._update_mr_simple_ui_state()

    def on_mr_selection_changed(self):
        geom = self.mr_editor_panel.get_geometry()
        # Point info
        pidx = self.mr_editor_panel.selected_point_index
        if pidx is not None and 0 <= pidx < len(geom.points):
            z, r = geom.points[pidx]
            ze, re_ = (geom.point_exprs[pidx]
                       if pidx < len(geom.point_exprs) else (None, None))
            self.mr_selected_point_label.SetLabel(f"Point: {pidx}")
            # 式があれば式を、無ければ数値を表示 (ver2.2)
            self.mr_point_z_ctrl.SetValue(ze if ze else f"{z:.6f}")
            self.mr_point_r_ctrl.SetValue(re_ if re_ else f"{r:.6f}")
        else:
            self.mr_selected_point_label.SetLabel("Point: -")
            self.mr_point_z_ctrl.SetValue("")
            self.mr_point_r_ctrl.SetValue("")
        # Segment info
        sid = self.mr_editor_panel.selected_segment_id
        seg = None
        if sid is not None:
            try:
                seg = geom.segment_by_id(sid)
                self.mr_selected_segment_label.SetLabel(
                    f"Segment: {sid}  type={seg.type}  pts={seg.point_indices}  "
                    f"bc={seg.bc_name}")
                # bc_choice の選択を seg.bc_name に同期 ("None" 含む)
                self.mr_bc_choice.SetStringSelection(seg.bc_name)
            except KeyError:
                self.mr_selected_segment_label.SetLabel(f"Segment: {sid}")
                seg = None
        else:
            self.mr_selected_segment_label.SetLabel("Segment: -")
        self._sync_arc_controls(seg)
        self._refresh_mr_value_previews()

    def _sync_arc_controls(self, seg):
        """選択 segment の type に応じて円弧編集 UI を同期する。

        - arc: 中心 Z/R フィールドへ値表示、Convert to Line / Update Center を有効、
          Convert to Arc を無効。
        - line: 中心欄をクリア・無効、Convert to Arc を有効、Convert to Line /
          Update Center を無効。
        - 未選択: すべて無効・クリア。
        ウィジェットが未生成 (Generate Source 前) でも落ちないよう hasattr で防御。
        """
        has_arc_ui = all(hasattr(self, n) for n in (
            "mr_convert_arc_button", "mr_convert_line_button",
            "mr_update_center_button", "mr_arc_center_z_ctrl",
            "mr_arc_center_r_ctrl"))
        if not has_arc_ui:
            return
        if seg is None:
            self.mr_convert_arc_button.Enable(False)
            self.mr_convert_line_button.Enable(False)
            self.mr_update_center_button.Enable(False)
            self.mr_arc_center_z_ctrl.SetValue("")
            self.mr_arc_center_r_ctrl.SetValue("")
            self.mr_arc_center_z_ctrl.Enable(False)
            self.mr_arc_center_r_ctrl.Enable(False)
            return
        is_arc = (seg.type == "arc")
        self.mr_convert_arc_button.Enable(not is_arc)
        self.mr_convert_line_button.Enable(is_arc)
        self.mr_update_center_button.Enable(is_arc)
        self.mr_arc_center_z_ctrl.Enable(is_arc)
        self.mr_arc_center_r_ctrl.Enable(is_arc)
        if is_arc and seg.center is not None:
            cze, cre = seg.center_expr if seg.center_expr else (None, None)
            self.mr_arc_center_z_ctrl.SetValue(cze if cze else f"{seg.center[0]:.6f}")
            self.mr_arc_center_r_ctrl.SetValue(cre if cre else f"{seg.center[1]:.6f}")
        else:
            self.mr_arc_center_z_ctrl.SetValue("")
            self.mr_arc_center_r_ctrl.SetValue("")

    def OnMrUnitChange(self, event):
        self._on_mr_unit_change_apply()
        if event: event.Skip()

    def OnMrUpdateLimits(self, event):
        self._mr_apply_axes_limits()
        if event: event.Skip()

    def OnMrModeChange(self, event):
        idx = self.mr_mode_radiobox.GetSelection()
        if 0 <= idx < len(_MR_MODES):
            self.mr_editor_panel.set_mode(_MR_MODES[idx])
        if event: event.Skip()

    def OnMrUpdatePoint(self, event):
        idx = self.mr_editor_panel.selected_point_index
        if idx is None:
            return
        zt = self.mr_point_z_ctrl.GetValue()
        rt = self.mr_point_r_ctrl.GetValue()
        vars_ = self.get_mr_variables()
        try:
            z = eval_scalar(zt, vars_)
            r = eval_scalar(rt, vars_)
        except ValueError:
            wx.MessageBox("座標の値が不正です。", "Error", wx.OK | wx.ICON_ERROR)
            return
        # 生テキストが式なら式として保存 (素の数値は None=定数)
        self.mr_editor_panel.update_point_coords(
            idx, z, r, expr_or_none(zt), expr_or_none(rt))
        self._refresh_mr_value_previews()
        if event: event.Skip()

    def OnMrDeletePoint(self, event):
        self.mr_editor_panel.delete_selected_point()
        if event: event.Skip()

    def OnMrBcChoice(self, event):
        bc = self.mr_bc_choice.GetStringSelection()
        if bc:
            self.mr_editor_panel.set_selected_segment_bc(bc)
        if event: event.Skip()

    def OnMrDeleteSegment(self, event):
        self.mr_editor_panel.delete_selected_segment()
        if event: event.Skip()

    def OnMrConvertToArc(self, event):
        if not self.mr_editor_panel.convert_selected_to_arc():
            wx.MessageBox("円弧化する直線 segment を先に選択してください。",
                          "Info", wx.OK | wx.ICON_INFORMATION)
        if event: event.Skip()

    def OnMrConvertToLine(self, event):
        if not self.mr_editor_panel.convert_selected_to_line():
            wx.MessageBox("直線化する円弧 segment を先に選択してください。",
                          "Info", wx.OK | wx.ICON_INFORMATION)
        if event: event.Skip()

    def OnMrUpdateCenter(self, event):
        seg = self.mr_editor_panel._selected_segment()
        if seg is None or seg.type != "arc":
            wx.MessageBox("円弧 segment を選択してから中心を更新してください。",
                          "Info", wx.OK | wx.ICON_INFORMATION)
            return
        czt = self.mr_arc_center_z_ctrl.GetValue()
        crt = self.mr_arc_center_r_ctrl.GetValue()
        vars_ = self.get_mr_variables()
        try:
            cz = eval_scalar(czt, vars_)
            cr = eval_scalar(crt, vars_)
        except ValueError:
            wx.MessageBox("中心座標の値が不正です。", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.mr_editor_panel.update_arc_center(
            cz, cr, expr_or_none(czt), expr_or_none(crt))
        self._refresh_mr_value_previews()
        if event: event.Skip()

    def OnMrCloseLoop(self, event):
        """Advanced モード用 Close Loop (Loops 枠内のボタン)。
        Simple モードでも誤って呼ばれた場合はフォールバックで Simple 動作。"""
        if self._is_mr_simple_mode():
            self._do_simple_close_loop()
        else:
            # Advanced モード: 既存の pending_loop_segments を Loop 化 (常に CCW ラベル)
            lid = self.mr_editor_panel.finalize_pending_loop("CCW")
            if lid is None:
                wx.MessageBox(
                    "Build Loop モードで segment を選択してから 'Close Loop' を押してください。",
                    "Info", wx.OK | wx.ICON_INFORMATION)
        if event: event.Skip()

    def OnBtnMrCloseLoop(self, event):
        """Simple モード専用の Close Loop ボタン (mr_simple_close_loop_button)。

        点と自動 segment 列から、閉じ用 segment を追加して 1 Loop にまとめ、
        Vacuum 領域 (eps_r=1.0, material_tag="vacuum") を自動作成する。
        既に Loop / Region が存在する場合は警告のみ (Reset でやり直し)。
        """
        self._do_simple_close_loop()
        if event: event.Skip()

    def _do_simple_close_loop(self):
        """Simple モード Close Loop の本体 (Advanced からのフォールバックでも共用)。"""
        rid = self.mr_editor_panel.finalize_simple_loop_with_vacuum()
        if rid is not None:
            # 閉じた後は "Edit Points" に戻す (点ドラッグ/ライン上ダブルクリック挿入が可能)
            if hasattr(self, "mr_modeSimple_radiobox"):
                self.mr_modeSimple_radiobox.SetSelection(0)
                self._update_mr_simple_ui_state()
            self.mr_editor_panel.set_mode(MODE_AUTO_POINT_SEGMENT)
            wx.MessageBox(
                "閉ループを作成し、Vacuum 領域 (eps_r=1) を自動生成しました。\n"
                "Edit Points で点をドラッグ/ライン上ダブルクリックで点追加、"
                "Edit Lines で segment を選んで BC/円弧に変更できます。\n"
                "Mesh Output から .msh をエクスポートして FEM 解析へ進めます。",
                "Info", wx.OK | wx.ICON_INFORMATION)

    def OnMrCancelLoop(self, event):
        self.mr_editor_panel.cancel_pending_loop()
        if event: event.Skip()

    def OnMrDeleteLoop(self, event):
        sel = self.mr_loop_list.GetSelection()
        geom = self.mr_editor_panel.get_geometry()
        if sel < 0 or sel >= len(geom.loops):
            return
        lid = geom.loops[sel].id
        self.mr_editor_panel.delete_loop(lid)
        if event: event.Skip()

    def OnMrLoopSelected(self, event):
        # OCC モードで向きは不要のため、選択 → 表示反映の必要なし
        if event: event.Skip()

    def OnMrViewModeChange(self, event):
        """Simple/Advanced 切替。Advanced→Simple へ戻すとき領域 > 1 または
        非 vacuum 材料があれば警告 (UI を隠すだけで内部状態は失わない)。"""
        geom = self.mr_editor_panel.get_geometry()
        going_simple = self._is_mr_simple_mode()
        if going_simple:
            n_reg = len(geom.regions)
            non_vac = [r for r in geom.regions
                       if r.material_tag != "vacuum" or abs(r.eps_r - 1.0) > 1e-12]
            if n_reg > 1 or non_vac:
                wx.MessageBox(
                    f"現在 {n_reg} 個の Region があり、Simple モードの想定 "
                    "(単一 Vacuum 領域) と一致しません。\n"
                    "UI は Simple 表示に切り替わりますが、内部のデータは保持されます。"
                    "Advanced に戻すと再編集できます。",
                    "Info", wx.OK | wx.ICON_INFORMATION)
        self._apply_view_mode()
        if event: event.Skip()

    def OnMrAddRegion(self, event):
        geom = self.mr_editor_panel.get_geometry()
        if not geom.loops:
            wx.MessageBox("Loop を先に作成してください。", "Error", wx.OK | wx.ICON_ERROR)
            return
        try:
            name = self.mr_region_name_ctrl.GetValue().strip() or "Region"
            outer_sel = self.mr_region_outer_choice.GetSelection()
            if outer_sel < 0:
                wx.MessageBox("Outer Loop を選択してください。", "Error", wx.OK | wx.ICON_ERROR)
                return
            outer_lid = geom.loops[outer_sel].id
            hole_lids = []
            for i in range(self.mr_region_holes_list.GetCount()):
                if self.mr_region_holes_list.IsChecked(i):
                    if i == outer_sel:
                        continue
                    hole_lids.append(geom.loops[i].id)
            mat = self.mr_region_mat_ctrl.GetValue().strip() or "vacuum"
            eps_r = self._mr_eval(self.mr_region_eps_ctrl)
            eps_r_expr = expr_or_none(self.mr_region_eps_ctrl.GetValue())
            # ver2.3: tan_d 欄（wxGlade 再生成前は無いので getattr 防御）
            tand_ctrl = getattr(self, "mr_region_tand_ctrl", None)
            tan_delta = self._mr_eval(tand_ctrl) if tand_ctrl is not None else 0.0
            tan_delta_expr = (expr_or_none(tand_ctrl.GetValue())
                              if tand_ctrl is not None else None)
            self.mr_editor_panel.add_region(
                name, outer_lid, hole_lids, mat, eps_r, eps_r_expr,
                tan_delta=tan_delta, tan_delta_expr=tan_delta_expr)
        except ValueError as e:
            wx.MessageBox(f"値が不正です: {e}", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnMrUpdateRegion(self, event):
        sel = self.mr_region_list.GetSelection()
        geom = self.mr_editor_panel.get_geometry()
        if sel < 0 or sel >= len(geom.regions):
            return
        rid = geom.regions[sel].id
        try:
            name = self.mr_region_name_ctrl.GetValue().strip() or "Region"
            outer_sel = self.mr_region_outer_choice.GetSelection()
            if outer_sel < 0:
                return
            outer_lid = geom.loops[outer_sel].id
            hole_lids = []
            for i in range(self.mr_region_holes_list.GetCount()):
                if self.mr_region_holes_list.IsChecked(i):
                    if i == outer_sel:
                        continue
                    hole_lids.append(geom.loops[i].id)
            mat = self.mr_region_mat_ctrl.GetValue().strip() or "vacuum"
            eps_r = self._mr_eval(self.mr_region_eps_ctrl)
            eps_r_expr = expr_or_none(self.mr_region_eps_ctrl.GetValue())
            # ver2.3: tan_d 欄（wxGlade 再生成前は無いので getattr 防御）
            tand_ctrl = getattr(self, "mr_region_tand_ctrl", None)
            tan_delta = self._mr_eval(tand_ctrl) if tand_ctrl is not None else 0.0
            tan_delta_expr = (expr_or_none(tand_ctrl.GetValue())
                              if tand_ctrl is not None else None)
            self.mr_editor_panel.update_region(
                rid, name, outer_lid, hole_lids, mat, eps_r, eps_r_expr,
                tan_delta=tan_delta, tan_delta_expr=tan_delta_expr)
        except ValueError as e:
            wx.MessageBox(f"値が不正です: {e}", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnMrDeleteRegion(self, event):
        sel = self.mr_region_list.GetSelection()
        geom = self.mr_editor_panel.get_geometry()
        if sel < 0 or sel >= len(geom.regions):
            return
        rid = geom.regions[sel].id
        self.mr_editor_panel.delete_region(rid)
        if event: event.Skip()

    def OnMrRegionSelected(self, event):
        sel = self.mr_region_list.GetSelection()
        geom = self.mr_editor_panel.get_geometry()
        if 0 <= sel < len(geom.regions):
            r = geom.regions[sel]
            self.mr_region_name_ctrl.SetValue(r.name)
            self.mr_region_mat_ctrl.SetValue(r.material_tag)
            self.mr_region_eps_ctrl.SetValue(r.eps_r_expr or str(r.eps_r))
            tand_ctrl = getattr(self, "mr_region_tand_ctrl", None)
            if tand_ctrl is not None:
                tand_ctrl.SetValue(r.tan_delta_expr or str(r.tan_delta))
            # Outer Loop の選択
            for i, lp in enumerate(geom.loops):
                if lp.id == r.outer_loop_id:
                    self.mr_region_outer_choice.SetSelection(i)
                    break
            # Holes のチェック
            for i, lp in enumerate(geom.loops):
                self.mr_region_holes_list.Check(i, lp.id in r.hole_loop_ids)
        if event: event.Skip()

    def _export_default(self, ext):
        """Export ダイアログの (defaultDir, defaultFile) を現在の project 名から作る。

        project 未保存なら ("", "")。ext は先頭ドット付き（例: ".msh"）。
        """
        path = self.mr_current_project_path
        if path:
            return (os.path.dirname(path),
                    os.path.splitext(os.path.basename(path))[0] + ext)
        return "", ""

    def OnMrExportGeo(self, event):
        geom = self.mr_editor_panel.get_geometry()
        if not geom.regions:
            wx.MessageBox("少なくとも 1 つの Region を作成してから出力してください。",
                          "Error", wx.OK | wx.ICON_ERROR)
            return
        try:
            mesh_size = self._mr_eval(self.mr_mesh_size_ctrl)
        except ValueError:
            wx.MessageBox("Mesh Size の値が不正です。", "Error", wx.OK | wx.ICON_ERROR)
            return
        geom.mesh_size = mesh_size
        geom.mesh_size_expr = expr_or_none(self.mr_mesh_size_ctrl.GetValue())
        geom.unit = self.mr_unit_choice.GetStringSelection()
        wildcard = "GEO files (*.geo)|*.geo|GEO unrolled (*.geo_unrolled)|*.geo_unrolled"
        dd, df = self._export_default(".geo")
        with wx.FileDialog(self, "Export Multi-Region GEO", wildcard=wildcard,
                           defaultDir=dd, defaultFile=df,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        try:
            self.mr_editor_panel.export_geo(path)
            wx.MessageBox(
                f"GEO を出力しました: {path}\n\n"
                "Gmsh GUI で開くと OCC 内部の point/curve/surface/Physical Group が確認できます。",
                "Success")
        except Exception as e:
            import traceback
            wx.MessageBox(f"GEO 出力に失敗しました:\n{e}\n\n{traceback.format_exc()}",
                          "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnMrExportPythonScript(self, event):
        geom = self.mr_editor_panel.get_geometry()
        if not geom.regions:
            wx.MessageBox("少なくとも 1 つの Region を作成してから出力してください。",
                          "Error", wx.OK | wx.ICON_ERROR)
            return
        try:
            mesh_size = self._mr_eval(self.mr_mesh_size_ctrl)
        except ValueError:
            wx.MessageBox("Mesh Size の値が不正です。", "Error", wx.OK | wx.ICON_ERROR)
            return
        geom.mesh_size = mesh_size
        geom.mesh_size_expr = expr_or_none(self.mr_mesh_size_ctrl.GetValue())
        geom.unit = self.mr_unit_choice.GetStringSelection()
        mesh_order = 2 if self.mr_mesh_order_radiobox.GetSelection() == 1 else 1
        wildcard = "Python files (*.py)|*.py"
        dd, df = self._export_default(".py")
        with wx.FileDialog(self, "Export Multi-Region Python Script",
                           wildcard=wildcard, defaultDir=dd, defaultFile=df,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        try:
            self.mr_editor_panel.export_python_script(path, mesh_order=mesh_order)
            wx.MessageBox(
                f"Python スクリプトを出力しました: {path}\n\n"
                "`python <この .py>` で実行すると、同等の .msh が生成されます。\n"
                "ファイルを読むと gmsh API での構築過程が確認できます。",
                "Success")
        except Exception as e:
            import traceback
            wx.MessageBox(f"Python スクリプト出力に失敗しました:\n{e}\n\n{traceback.format_exc()}",
                          "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnMrExportMsh(self, event):
        geom = self.mr_editor_panel.get_geometry()
        if not geom.regions:
            wx.MessageBox("少なくとも 1 つの Region を作成してから出力してください。",
                          "Error", wx.OK | wx.ICON_ERROR)
            return
        try:
            mesh_size = self._mr_eval(self.mr_mesh_size_ctrl)
        except ValueError:
            wx.MessageBox("Mesh Size の値が不正です。", "Error", wx.OK | wx.ICON_ERROR)
            return
        geom.mesh_size = mesh_size
        geom.mesh_size_expr = expr_or_none(self.mr_mesh_size_ctrl.GetValue())
        geom.unit = self.mr_unit_choice.GetStringSelection()
        mesh_order = 2 if self.mr_mesh_order_radiobox.GetSelection() == 1 else 1
        wildcard = "MSH files (*.msh)|*.msh"
        dd, df = self._export_default(".msh")
        with wx.FileDialog(self, "Export Multi-Region MSH", wildcard=wildcard,
                           defaultDir=dd, defaultFile=df,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        try:
            self.mr_editor_panel.export_msh(path, mesh_order=mesh_order)
            self.last_mesh_path = path
            self.fem_mesh_path_ctrl.SetValue(path)
            # 新しいメッシュを出力したので古い解析結果パスは無効。クリアする。
            self.fem_raw_result_path_ctrl.SetValue("")
            self.fem_processed_result_path_ctrl.SetValue("")
            wx.MessageBox(f"Mesh を出力しました: {path}", "Success")
        except Exception as e:
            import traceback
            wx.MessageBox(f"メッシュ出力に失敗しました:\n{e}\n\n{traceback.format_exc()}",
                          "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    # ------------------------------------------------------------------
    # Superfish (.af) 入出力 (shared/superfish_io 経由)
    # ------------------------------------------------------------------
    def mr_import_superfish(self, path) -> bool:
        """Superfish (.af) を Multi-Region Editor に読み込む (ダイアログなし)。

        単位は Draw Area の Unit 選択に従う。閉ループを検出した場合は
        Vacuum 領域が自動生成される。閉じていない場合は点・segment のみ
        読み込み、Close Loop を促す。
        """
        unit = self.mr_unit_choice.GetStringSelection()
        try:
            geom = import_superfish(path, unit)
        except SuperfishImportError as e:
            wx.MessageBox(f"Superfish ファイルの読み込みに失敗しました:\n{e}",
                          "Error", wx.OK | wx.ICON_ERROR)
            return False
        self.mr_editor_panel.set_geometry(geom)
        self.set_mr_variables_rows(geom.variables)  # 空リスト = 変数表クリア
        self.mr_current_project_path = None
        self.notebook_1.SetSelection(self._mr_tab_index())
        # mark_dirty -> on_mr_editor_changed 経由でリスト・タイトル・
        # Simple UI 状態が更新される
        self.mr_editor_panel.mark_dirty()
        if geom.loops:
            self.statusbar.SetStatusText(
                f"Superfish をインポートしました (単位: {unit}): {path}")
        else:
            wx.MessageBox(
                "輪郭が閉じていないためループ/領域は作成していません。\n"
                "[Close Loop] で閉じると Vacuum 領域が自動生成されます。",
                "Info", wx.OK | wx.ICON_INFORMATION)
        return True

    def mr_export_superfish(self, path) -> bool:
        """Multi-Region Editor のデータを Superfish (.af) に出力する (ダイアログなし)。

        Superfish は単一領域境界のみ表現できるため、Region が 1 つ・穴なし・
        閉ループの場合のみ出力できる。BC 情報は保存されない。
        """
        geom = self.mr_editor_panel.get_geometry()
        geom.unit = self.mr_unit_choice.GetStringSelection()
        try:
            mesh_size = self._mr_eval(self.mr_mesh_size_ctrl)
        except ValueError:
            wx.MessageBox("Mesh Size の値が不正です。", "Error", wx.OK | wx.ICON_ERROR)
            return False
        errors = validate_superfish_exportable(geom)
        if errors:
            wx.MessageBox(
                "Superfish (.af) に出力できません:\n- " + "\n- ".join(errors),
                "Error", wx.OK | wx.ICON_ERROR)
            return False
        try:
            export_superfish(geom, path, mesh_size)
        except SuperfishExportError as e:
            wx.MessageBox(f"Superfish 出力に失敗しました:\n{e}",
                          "Error", wx.OK | wx.ICON_ERROR)
            return False
        wx.MessageBox(
            f"Superfish ファイルを出力しました: {path}\n\n"
            "※ BC 情報は .af 形式では保存されません。", "Success")
        return True

    # ------------------------------------------------------------------
    # Point/Center 座標欄の数値プレビュー (ver2.3)
    # ------------------------------------------------------------------
    # プレビューラベルと対応する入力欄の対 (順序は表示順)
    _MR_PREVIEW_PAIRS = (
        ("mr_point_z_value_label", "mr_point_z_ctrl"),
        ("mr_point_r_value_label", "mr_point_r_ctrl"),
        ("mr_arc_center_z_value_label", "mr_arc_center_z_ctrl"),
        ("mr_arc_center_r_value_label", "mr_arc_center_r_ctrl"),
    )

    def _ensure_mr_value_preview_labels(self):
        """座標欄の数値プレビュー用 StaticText を用意する。

        wxGlade で main_frame_ui.py に生成済みならそれを使う (灰色に整える)。
        未生成 (Generate Source 前) の場合は各入力欄の直後に手書きで挿入する
        フォールバック (_build_mr_variables_grid のプレースホルダ方式と同様)。
        """
        grey = wx.Colour(0x80, 0x80, 0x80)
        for label_name, ctrl_name in self._MR_PREVIEW_PAIRS:
            existing = getattr(self, label_name, None)
            if existing is not None:
                existing.SetForegroundColour(grey)
                continue
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is None:
                continue
            lbl = wx.StaticText(ctrl.GetParent(), wx.ID_ANY, "")
            lbl.SetForegroundColour(grey)
            sizer = ctrl.GetContainingSizer()
            if sizer is not None:
                index = None
                for i, item in enumerate(sizer.GetChildren()):
                    if item.GetWindow() is ctrl:
                        index = i + 1
                        break
                if index is not None:
                    sizer.Insert(index, lbl, 0,
                                 wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
                else:
                    sizer.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
            setattr(self, label_name, lbl)

    def _bind_mr_value_preview_events(self):
        """座標欄のタイプ中にプレビューを追従させる (EVT_TEXT)。"""
        for _, ctrl_name in self._MR_PREVIEW_PAIRS:
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is not None:
                ctrl.Bind(wx.EVT_TEXT, self.OnMrCoordText)

    def OnMrCoordText(self, event):
        self._refresh_mr_value_previews()
        if event:
            event.Skip()

    def _format_value_preview(self, text) -> str:
        """入力欄テキストの数値プレビュー文字列を返す。

        式のときのみ ``= <値>`` を返す。空文字・素の数値は空 (併記しない)。
        評価不能 (未定義変数・構文エラー) は ``= ?``。
        """
        t = (text or "").strip()
        if not t or is_plain_number(t):
            return ""
        try:
            v = eval_scalar(t, self.get_mr_variables())
        except ValueError:
            return "= ?"
        return f"= {v:.6f}"

    def _refresh_mr_value_previews(self):
        """4 つの座標欄から対応するプレビューラベルを更新する。

        無効化された欄 (line 選択時の Center 欄など) は空にする。
        """
        for label_name, ctrl_name in self._MR_PREVIEW_PAIRS:
            lbl = getattr(self, label_name, None)
            ctrl = getattr(self, ctrl_name, None)
            if lbl is None or ctrl is None:
                continue
            if not ctrl.IsEnabled():
                lbl.SetLabel("")
                continue
            lbl.SetLabel(self._format_value_preview(ctrl.GetValue()))
