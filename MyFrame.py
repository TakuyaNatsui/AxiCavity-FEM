import wx
import os
import json
import threading
import subprocess
import webbrowser
from MyFrameUI import MyFrameUI
from PointLineEditorPanel import PointLineEditorPanel
from ResultViewer import ResultViewer

class MyFrame(MyFrameUI):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        
        # プロジェクト管理用
        self.current_project_path = None
        self.last_mesh_path = None
        self.last_analysis_result_h5 = None
        self.last_processed_h5 = None
        self.last_hom_result_h5 = None
        tip = "単値: 120  /  スキャン: 開始:終了:ステップ  例) 0:180:20"
        self.fem_phase_shift_ctrl.SetToolTip(tip)
        self.fem_phase_shift_ctrl.Enable(False)  # Standing Wave がデフォルトのため無効化
        self.hom_order_ctrl.Enable(False)         # TM0 がデフォルトのため無効化
        
        # ステータスバーの作成
        self.statusbar = self.CreateStatusBar(1)
        self.statusbar.SetStatusText("Welcome!")
        
        # エディタパネルの初期化
        self.editor_panel = PointLineEditorPanel(self.notebook_1_pane_1, self.statusbar)
        
        # レイアウトへの挿入
        self.sizer_editor.Insert(0, self.editor_panel, 1, wx.EXPAND, 0)
        
        # 初期状態の設定
        self.set_panel_colors()
        self.update_ui_state()
        self.sizer_editor.Layout()
        
        # 軸ラベルの初期表示
        self.on_unit_change(None)

    # --- プロジェクト保存・読み込みロジック ---
    def save_project(self, path):
        """現在の状態をJSON形式で.gmshprojファイルに保存する"""
        try:
            data = {
                "points": self.editor_panel.points,
                "segments": [],
                "loop_closed": self.editor_panel.loop_closed,
                "settings": {
                    "unit": self.unit_choice.GetStringSelection(),
                    "xmin": self.xmin_ctrl.GetValue(),
                    "xmax": self.xmax_ctrl.GetValue(),
                    "ymin": self.ymin_ctrl.GetValue(),
                    "ymax": self.ymax_ctrl.GetValue(),
                    "mesh_size": self.mesh_size_ctrl.GetValue(),
                    "mesh_order": self.radio_box_mesh_order.GetSelection()
                }
            }
            
            for seg in self.editor_panel.segments:
                s_data = seg.copy()
                s_data.pop('artist', None)
                data["segments"].append(s_data)
                
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            self.current_project_path = path
            self.editor_panel.is_dirty = False
            self.on_editor_changed()
            return True
        except Exception as e:
            wx.MessageBox(f"Failed to save project: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return False

    def load_project(self, path):
        """JSON形式の.gmshprojファイルから状態を復元する"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.editor_panel.reset()
            
            s = data.get("settings", {})
            self.unit_choice.SetStringSelection(s.get("unit", "mm"))
            self.xmin_ctrl.SetValue(s.get("xmin", "0"))
            self.xmax_ctrl.SetValue(s.get("xmax", "100"))
            self.ymin_ctrl.SetValue(s.get("ymin", "0"))
            self.ymax_ctrl.SetValue(s.get("ymax", "100"))
            self.mesh_size_ctrl.SetValue(s.get("mesh_size", "5"))
            if "mesh_order" in s:
                self.radio_box_mesh_order.SetSelection(s["mesh_order"])
            
            for p in data.get("points", []):
                self.editor_panel.points.append(tuple(p))
                m = self.editor_panel.axes.plot(p[0], p[1], 'ro', markersize=5, picker=5)[0]
                self.editor_panel.markers.append(m)
            
            for s_data in data.get("segments", []):
                p1, p2 = s_data['points']
                pt1, pt2 = self.editor_panel.points[p1], self.editor_panel.points[p2]
                
                if s_data['type'] == 'line':
                    line = self.editor_panel.axes.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-')[0]
                    s_data['artist'] = line
                else:
                    import matplotlib.patches as patches
                    c = s_data['center']
                    r = s_data['radius']
                    arc = patches.Arc(tuple(c), width=2*r, height=2*r, angle=0,
                                      theta1=s_data['theta1'], theta2=s_data['theta2'], 
                                      color='green', linewidth=1.0)
                    self.editor_panel.axes.add_patch(arc)
                    s_data['artist'] = arc
                self.editor_panel.segments.append(s_data)
            
            self.editor_panel.loop_closed = data.get("loop_closed", False)
            self.current_project_path = path
            self.editor_panel.is_dirty = False
            self.editor_panel.canvas.draw()
            self.on_editor_changed()
            self.on_unit_change(None) # 軸ラベル更新
            
            # 向きの自動判定と提案
            self.check_loop_orientation()
            
            return True
        except Exception as e:
            wx.MessageBox(f"Failed to load project: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return False

    def check_loop_orientation(self):
        """ループの向きを確認し、時計回り(CW)なら反転を提案する"""
        if self.editor_panel.loop_closed:
            orientation = self.editor_panel.get_loop_orientation()
            if orientation == "CW":
                msg = "現在のループは時計回り(CW)です。電磁場解析コードの多くは反時計回り(CCW)を推奨しています。\n\nループの向きを反転しますか？"
                if wx.MessageBox(msg, "向きの確認", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
                    self.editor_panel.reverse_loop()
                    self.update_ui_state()

    # --- メニューハンドラ ---
    def OnMenuSave(self, event):
        if self.current_project_path:
            self.save_project(self.current_project_path)
        else:
            self.OnMenuSaveAs(event)

    def OnMenuSaveAs(self, event):
        wildcard = "GMSH Project files (*.gmshproj)|*.gmshproj"
        with wx.FileDialog(self, "Save Project As", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.save_project(dlg.GetPath())

    def OnMenuLoad(self, event):
        if self.editor_panel.is_dirty:
            if wx.MessageBox("Current changes will be lost. Continue?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.NO:
                return
        wildcard = "GMSH Project files (*.gmshproj)|*.gmshproj"
        with wx.FileDialog(self, "Load Project", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.load_project(dlg.GetPath())

    def OnMenuImportSuperfish(self, event):
        if self.editor_panel.is_dirty:
            if wx.MessageBox("Current changes will be lost. Continue?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.NO:
                return
        wildcard = "Superfish files (*.af)|*.af|All files (*.*)|*.*"
        with wx.FileDialog(self, "Import Superfish File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                unit = self.unit_choice.GetStringSelection()
                self.editor_panel.import_superfish(dlg.GetPath(), unit)
                # インポート後も向きを確認
                self.check_loop_orientation()

    # --- 通知の受け取り ---
    def on_selection_changed(self):
        idx = self.editor_panel.selected_point_index
        if idx is not None:
            p = self.editor_panel.points[idx]
            self.selected_point_label.SetLabel(f"Selected Point {idx}")
            self.coord_x_ctrl.SetValue(f"{p[0]:.6f}")
            self.coord_y_ctrl.SetValue(f"{p[1]:.6f}")
        else:
            self.selected_point_label.SetLabel("Selected Point - ")
            self.coord_x_ctrl.SetValue(""); self.coord_y_ctrl.SetValue("")

        s_idx = self.editor_panel.selected_segment_index
        if s_idx is not None:
            seg = self.editor_panel.segments[s_idx]
            p1, p2 = seg['points']
            self.selected_line_label.SetLabel(f"Selected Segment: {s_idx} ({seg['type']}) [Pt {p1} -> Pt {p2}]")
            self.physical_name_ctrl.SetValue(seg.get('physical_name', ''))
            if seg['type'] == 'arc':
                c = seg['center']
                self.center_x_ctrl.SetValue(f"{c[0]:.6f}")
                self.center_y_ctrl.SetValue(f"{c[1]:.6f}")
            else:
                self.center_x_ctrl.SetValue(""); self.center_y_ctrl.SetValue("")
        else:
            self.selected_line_label.SetLabel("Selected Segment: -")
            self.physical_name_ctrl.SetValue("")
            self.center_x_ctrl.SetValue(""); self.center_y_ctrl.SetValue("")
        self.update_ui_state()

    def on_editor_changed(self):
        title = "Point/Line Editor"
        if self.editor_panel.is_dirty: title += " *"
        self.SetTitle(title)
        self.update_ui_state()

    def update_ui_state(self):
        num_points = len(self.editor_panel.points)
        is_loop_closed = self.editor_panel.loop_closed
        mode = self.editor_panel.mode
        
        self.close_loop_button.Enable(num_points >= 3 and not is_loop_closed)
        self.reverse_loop_button.Enable(is_loop_closed)
        self.mode_radiobox.Enable(num_points > 0)
        
        # ヒントメッセージの更新
        hint = ""
        if not is_loop_closed:
            if num_points == 0:
                hint = "グラフエリアをクリックして、最初の点を追加してください。"
            elif num_points < 3:
                hint = f"】続けて点をクリックしてください（現在 {num_points} 点）。3点以上で形状を閉じることができます。"
            else:
                hint = f"現在 {num_points} 点打たれています。打ち終わったら 'Close Loop' を押して形状を確定させてください。"
        else:
            orientation = self.editor_panel.get_loop_orientation()
            ori_text = ""
            if orientation == "CW":
                ori_text = "【警告: 時計回り(CW)】反時計回り(CCW)への反転を推奨します。 "
            
            if mode == "edit_points":
                hint = f"{ori_text}点をドラッグして移動、または数値を入力して更新できます。線の上をダブルクリックすると新しい点を挿入できます。"
            elif mode == "edit_lines":
                hint = f"{ori_text}線を選択して円弧への変換や、Physical Name（境界条件名）の設定を行ってください。"
            else:
                hint = f"{ori_text}設定が完了しました。Mesh Sizeと次数を確認し、下部のボタンからメッシュを書き出してください。"
        
        self.text_ctrl_hint_message.SetValue(hint)
        self.set_panel_colors()

    def set_panel_colors(self):
        mode = self.editor_panel.mode
        active_color = wx.Colour(191, 242, 255); inactive_color = wx.Colour(226, 226, 226)
        if mode == "edit_points":
            self.panel_point_edit.SetBackgroundColour(active_color)
            self.panel_line_edit.SetBackgroundColour(inactive_color)
        elif mode == "edit_lines":
            self.panel_point_edit.SetBackgroundColour(inactive_color)
            self.panel_line_edit.SetBackgroundColour(active_color)
        else:
            self.panel_point_edit.SetBackgroundColour(inactive_color)
            self.panel_line_edit.SetBackgroundColour(inactive_color)
        self.panel_point_edit.Refresh(); self.panel_line_edit.Refresh()

    # --- イベントハンドラ ---
    def on_mode_change(self, event):
        selection = self.mode_radiobox.GetSelection()
        if selection == 0: self.editor_panel.set_mode("edit_points")
        else: self.editor_panel.set_mode("edit_lines")
        if event: event.Skip()

    def OnBtnCloseLoop(self, event):
        if self.editor_panel.close_loop():
            self.mode_radiobox.SetSelection(0)
            self.editor_panel.set_mode("edit_points")
            # ループを閉じた直後に向きを確認
            self.check_loop_orientation()
        if event: event.Skip()

    def OnBtnResetAll(self, event):
        if wx.MessageBox("Reset all data?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            self.editor_panel.reset()
            self.mode_radiobox.SetSelection(0)
            self.editor_panel.set_mode("add_init_points")
            self.current_project_path = None
        if event: event.Skip()

    def OnBtnUpdateCoord(self, event):
        idx = self.editor_panel.selected_point_index
        if idx is not None:
            try:
                nx, ny = float(self.coord_x_ctrl.GetValue()), float(self.coord_y_ctrl.GetValue())
                self.editor_panel.update_point_coords(idx, nx, ny)
            except ValueError:
                wx.MessageBox("Invalid coordinate values.", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnBtnDeletePoint(self, event):
        idx = self.editor_panel.selected_point_index
        if idx is not None:
            if wx.MessageBox(f"Delete point {idx}?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
                self.editor_panel.delete_point(idx)
                self.on_selection_changed()
        if event: event.Skip()

    def OnBtnConvertToArc(self, event):
        self.editor_panel.convert_selected_to_arc()
        self.on_selection_changed()
        if event: event.Skip()

    def OnBtnConvetToLine(self, event):
        self.editor_panel.convert_selected_to_line()
        self.on_selection_changed()
        if event: event.Skip()

    def OnBtnUpdateLimits(self, event):
        try:
            xmin, xmax = float(self.xmin_ctrl.GetValue()), float(self.xmax_ctrl.GetValue())
            ymin, ymax = float(self.ymin_ctrl.GetValue()), float(self.ymax_ctrl.GetValue())
            self.editor_panel.set_axes_limits(xmin, xmax, ymin, ymax)
        except ValueError:
            wx.MessageBox("Invalid limit values.", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def on_unit_change(self, event):
        unit = self.unit_choice.GetStringSelection()
        self.editor_panel.axes.set_xlabel(f"X [{unit}]")
        self.editor_panel.axes.set_ylabel(f"Y [{unit}]")
        self.editor_panel.canvas.draw_idle()
        if event: event.Skip()

    def on_physical_name_change(self, event):
        idx = self.editor_panel.selected_segment_index
        if idx is not None:
            self.editor_panel.segments[idx]['physical_name'] = self.physical_name_ctrl.GetValue()
            self.editor_panel.mark_dirty()
        if event: event.Skip()

    def OnBtnExportGeo(self, event):
        try:
            mesh_size = float(self.mesh_size_ctrl.GetValue())
            unit = self.unit_choice.GetStringSelection()
            mesh_order = 2 if self.radio_box_mesh_order.GetSelection() == 1 else 1
            wildcard = "GEO files (*.geo)|*.geo"
            with wx.FileDialog(self, "Export GEO", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    path = dlg.GetPath()
                    self.editor_panel.export_geo(path, mesh_size, unit, mesh_order)
                    msh_path = os.path.splitext(path)[0] + ".msh"
                    self.last_mesh_path = msh_path
                    self.fem_mesh_path_ctrl.SetValue(msh_path)
        except ValueError:
            wx.MessageBox("Invalid mesh size.", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnBtnReverseLoop(self, event):
        self.editor_panel.reverse_loop()
        if event: event.Skip()

    def OnBtnUpdateCenter(self, event):
        idx = self.editor_panel.selected_segment_index
        if idx is not None and self.editor_panel.segments[idx]['type'] == 'arc':
            try:
                cx, cy = float(self.center_x_ctrl.GetValue()), float(self.center_y_ctrl.GetValue())
                self.editor_panel.update_arc_center(idx, cx, cy)
            except ValueError:
                wx.MessageBox("Invalid center values.", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnBtnExportPythonScript(self, event):
        try:
            mesh_size = float(self.mesh_size_ctrl.GetValue())
            unit = self.unit_choice.GetStringSelection()
            mesh_order = 2 if self.radio_box_mesh_order.GetSelection() == 1 else 1
            wildcard = "Python files (*.py)|*.py"
            with wx.FileDialog(self, "Export Python Script", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    self.editor_panel.export_python_script(dlg.GetPath(), mesh_size, unit, mesh_order)
        except ValueError:
            wx.MessageBox("Invalid mesh size.", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnBtnExportSuperfish(self, event):
        try:
            mesh_size = float(self.mesh_size_ctrl.GetValue())
            unit = self.unit_choice.GetStringSelection()
            wildcard = "Superfish files (*.af)|*.af"
            with wx.FileDialog(self, "Export Superfish", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    self.editor_panel.export_superfish(dlg.GetPath(), mesh_size, unit)
        except ValueError:
            wx.MessageBox("Invalid mesh size.", "Error", wx.OK | wx.ICON_ERROR)
        if event: event.Skip()

    def OnBtnExportMshDirect(self, event):
        try:
            mesh_size = float(self.mesh_size_ctrl.GetValue())
            unit = self.unit_choice.GetStringSelection()
            mesh_order = 2 if self.radio_box_mesh_order.GetSelection() == 1 else 1
            wildcard = "MSH files (*.msh)|*.msh"
            with wx.FileDialog(self, "Export MSH", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    path = dlg.GetPath()
                    self.editor_panel.export_msh_direct(path, mesh_size, unit, mesh_order)
                    self.last_mesh_path = path
                    self.fem_mesh_path_ctrl.SetValue(path)
        except ValueError:
            wx.MessageBox("Invalid mesh size.", "Error", wx.OK | wx.ICON_ERROR)
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
        """Traveling Wave 選択時のみ Phase Shift 入力を有効化"""
        is_traveling = self.fem_wave_type_radio.GetSelection() == 1
        self.fem_phase_shift_ctrl.Enable(is_traveling)
        if event: event.Skip()

    def OnRunSolver(self, event):
        mesh_path = self.fem_mesh_path_ctrl.GetValue()
        if not mesh_path or not os.path.exists(mesh_path):
            wx.MessageBox(f"Mesh file not found:\n{mesh_path}", "Error", wx.OK | wx.ICON_ERROR)
            return

        is_hom      = self.radio_box_analysis_mode.GetSelection() == 1
        n_modes     = self.num_modes_ctrl.GetValue()
        order       = 2 if self.radio_mesh_order.GetSelection() == 1 else 1
        is_traveling = self.fem_wave_type_radio.GetSelection() == 1
        phase_str   = self.fem_phase_shift_ctrl.GetValue()

        if is_hom:
            order_str   = self.hom_order_ctrl.GetValue().strip() or "0"
            output_path = self.fem_raw_result_path_ctrl.GetValue().strip()
            if not output_path:
                suffix = "_TW_HOM.h5" if is_traveling else "_SW_HOM.h5"
                output_path = os.path.splitext(mesh_path)[0] + suffix
                self.fem_raw_result_path_ctrl.SetValue(output_path)

            cmd = (
                ["python", "FEM_HOM_code/run_analysis.py",
                 "-m", mesh_path,
                 "--az-order"] + order_str.split() +
                ["--num-modes", str(n_modes),
                 "--elem-order", str(order),
                 "-o", output_path]
            )
            if is_traveling:
                if not phase_str:
                    wx.MessageBox("Phase Shift を入力してください。", "Error", wx.OK | wx.ICON_ERROR)
                    return
                cmd += ["-p", phase_str]

            self.last_hom_result_h5 = output_path
            label = "HOM Solver (Traveling)" if is_traveling else "HOM Solver"
            self.run_command_async(cmd, label)
        else:
            suffix   = "_TW_TM0.h5" if is_traveling else "_SW_TM0.h5"
            output_h5 = os.path.splitext(mesh_path)[0] + suffix

            cmd = [
                "python", "FEM_code/run_analysis.py",
                "-m", mesh_path,
                "--num-modes", str(n_modes),
                "--elem-order", str(order),
                "-o", output_h5
            ]
            if is_traveling:
                cmd += ["-p", phase_str]

            self.last_analysis_result_h5 = output_h5
            self.fem_raw_result_path_ctrl.SetValue(output_h5)
            label = "TM0 Solver (Traveling)" if is_traveling else "TM0 Solver"
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
        output_h5 = os.path.splitext(raw_path)[0] + "_processed.h5"

        if is_hom:
            cmd = [
                "python", "FEM_HOM_code/post_process_hom.py", raw_path,
                "--cond", cond,
                "--mode", "calc"
            ]
            label = "HOM Parameter Calculation"
        else:
            beta = self.fem_beta_ctrl.GetValue()
            cmd = [
                "python", "FEM_code/post_process_unified.py",
                "-i", raw_path,
                "-o", output_h5,
                "-c", cond,
                "-b", beta,
                "--mode", "calc"
            ]
            label = "Parameter Calculation"

        self.last_processed_h5 = output_h5
        self.fem_processed_result_path_ctrl.SetValue(output_h5)
        self.run_command_async(cmd, label)

    def OnBtnCreateReport(self, event):
        is_hom = self.radio_box_analysis_mode.GetSelection() == 1
        processed_path = self.fem_processed_result_path_ctrl.GetValue()
        if not processed_path:
            processed_path = self.last_processed_h5
        if not processed_path or not os.path.exists(processed_path):
            wx.MessageBox("Processed result file not found. Run Post-Process first.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # チェックボックス状態を読み取って --no-anim を組み立てる
        create_anim = self.checkbox_report_anim.GetValue()

        if is_hom:
            cmd = [
                "python", "FEM_HOM_code/post_process_hom.py", processed_path,
                "--mode", "report"
            ]
            label = "HOM Report Generation"
        else:
            cmd = [
                "python", "FEM_code/post_process_unified.py",
                "-i", processed_path,
                "--mode", "report"
            ]
            label = "Report Generation"

        if not create_anim:
            cmd += ["--no-anim"]

        self.run_command_async(cmd, label)

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

        dlg = ResultViewer(self, result_h5)
        dlg.ShowModal()
        dlg.Destroy()
        if event: event.Skip()

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
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='cp932', errors='replace', shell=True
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
