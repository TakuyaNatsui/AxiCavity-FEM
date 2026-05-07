import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import numpy as np
import os
import sys
import datetime
import subprocess
import threading
import h5py

from FEM_code.field_calculator import FieldCalculator
from FEM_code.plot_utils import FEMPlotter
from FEM_HOM_code.field_calculator_hom import HOMFieldCalculator
from FEM_HOM_code.plot_utils_hom import FEMPlotterHOM
from ResultViewerUI import ResultViewerUI

class ResultViewer(ResultViewerUI):
    def __init__(self, parent, h5_file=None):
        super().__init__(parent)
        
        self.calc = None
        self.plotter = None
        self.current_mode_idx = 0
        self.grid_params = (40, 40)
        self.is_hom = False
        
        # --- Matplotlib Setup in panel_graph ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.panel_graph, -1, self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_aspect('equal')
        self.toolbar = NavigationToolbar(self.canvas)
        
        graph_sizer = wx.BoxSizer(wx.VERTICAL)
        graph_sizer.Add(self.canvas, 1, wx.EXPAND)
        graph_sizer.Add(self.toolbar, 0, wx.EXPAND)
        self.panel_graph.SetSizer(graph_sizer)
        
        # Bind events
        self.Bind(wx.EVT_CHOICE, self.on_HOM_n_change, self.choice_HOM_n)
        self.Bind(wx.EVT_CHOICE, self.on_mode_change, self.choice_mode)
        self.Bind(wx.EVT_CHOICE, self.on_sim_phase_change, self.choice_sim_phase)
        self.Bind(wx.EVT_SPINCTRL, self.on_option_change, self.spin_ctrl_time_phase)
        self.Bind(wx.EVT_CHECKBOX, self.on_option_change, self.checkbox_H_theta)
        self.Bind(wx.EVT_CHECKBOX, self.on_option_change, self.checkbox_E_lines)
        self.Bind(wx.EVT_SPINCTRL, self.on_option_change, self.spin_ctrl_E_line_levels)
        self.Bind(wx.EVT_CHECKBOX, self.on_option_change, self.checkbox_vectors)
        self.Bind(wx.EVT_SPINCTRL, self.on_option_change, self.spin_ctrl_vectors_z)
        self.Bind(wx.EVT_SPINCTRL, self.on_option_change, self.spin_ctrl_vectors_r)
        self.Bind(wx.EVT_CHECKBOX, self.on_option_change, self.checkbox_show_mesh)
        self.Bind(wx.EVT_CHECKBOX, self.on_option_change, self.checkbox_e_wall)
        
        self.Bind(wx.EVT_BUTTON, self.on_close, self.button_CLOSE)
        # Export ボタンは wxGlade 側 (.wxg の <events>) で自動バインドされるため
        # ここで再度 self.Bind を呼ぶと 2 重バインドになり、ダイアログが 2 回開く。
        # OnBtnExportArea / OnBtnExportLine / OnBtnExportAxis をオーバーライドだけする。
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.set_status("Ready. Load an HDF5 result file to begin.")
        self.choice_HOM_n.Disable()
        
        if h5_file and os.path.exists(h5_file):
            self.load_result_file(h5_file)

    def set_status(self, text):
        self.text_ctrl_status.SetValue(text)

    def on_close(self, event):
        self.EndModal(wx.ID_OK)

    def load_result_file(self, h5_file):
        try:
            self.is_hom = False
            with h5py.File(h5_file, 'r') as f:
                if 'results' in f:
                    for k in f['results'].keys():
                        if k.startswith('n'):
                            self.is_hom = True
                            break

            if self.is_hom:
                self.calc = HOMFieldCalculator(h5_file)
                self.plotter = FEMPlotterHOM(self.calc)
                self.SetTitle(f"HOM Result Viewer - {os.path.basename(h5_file)}")
                self.choice_HOM_n.Enable(True)
                
                self.choice_HOM_n.Clear()
                self.choice_HOM_n.AppendItems([str(n) for n in self.calc.available_ns])
                if self.calc.available_ns:
                    self.choice_HOM_n.SetSelection(0)
                    self.calc.set_n_and_phase(self.calc.available_ns[0])
                
                analysis_type = self.calc.analysis_type
            else:
                self.calc = FieldCalculator(h5_file)
                self.plotter = FEMPlotter(self.calc)
                self.choice_HOM_n.Disable()
                self.SetTitle(f"TM0 Result Viewer - {os.path.basename(h5_file)}")
                analysis_type = getattr(self.calc, 'analysis_type', 'standing')

            if analysis_type == 'standing':
                self.spin_ctrl_time_phase.Enable(False)
                self.choice_sim_phase.Enable(False)
                wave_str = "Standing Wave Mode"
            else:
                self.spin_ctrl_time_phase.Enable(True)
                self.choice_sim_phase.Enable(True)
                wave_str = "Traveling Wave Mode"
                
            self.set_status(f"Loaded {os.path.basename(h5_file)} ({wave_str})")

            # Update combo boxes
            self._update_ui_choices()
            
            self.update_plots()
        except Exception as e:
            import traceback
            traceback.print_exc()
            wx.MessageBox(f"Failed to load result file: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _update_ui_choices(self):
        # Mode lists
        if self.is_hom:
            mode_choices = [f"Mode {m[0]}: {m[1]:.6f} GHz" for m in self.calc.modes]
        else:
            mode_choices = [f"Mode {i}: {f:.6f} GHz" for i, f in enumerate(self.calc.frequencies)]
            
        self.choice_mode.Clear()
        self.choice_mode.AppendItems(mode_choices)
        if mode_choices:
            self.choice_mode.SetSelection(0)
            self.current_mode_idx = 0
        
        # Phase lists
        self.choice_sim_phase.Clear()
        if len(self.calc.phase_shifts) > 0:
            # Convert to list to ensure .index() works even if it's a numpy array
            phase_list = list(self.calc.phase_shifts)
            self.choice_sim_phase.AppendItems([f"{ph}°" for ph in phase_list])
            if hasattr(self.calc, 'phase_shift') and self.calc.phase_shift in phase_list:
                idx = phase_list.index(self.calc.phase_shift)
                self.choice_sim_phase.SetSelection(idx)
        elif hasattr(self.calc, 'phase_shift'):
            self.choice_sim_phase.AppendItems([f"{self.calc.phase_shift}°"])
            self.choice_sim_phase.SetSelection(0)

    def on_HOM_n_change(self, event):
        if not self.is_hom: return
        n_str = self.choice_HOM_n.GetStringSelection()
        if not n_str: return
        n = int(n_str)
        self.calc.set_n_and_phase(n, self.calc.phase_shift)
        self._update_ui_choices()
        self.update_plots()

    def on_mode_change(self, event):
        self.current_mode_idx = self.choice_mode.GetSelection()
        self.update_plots()

    def on_sim_phase_change(self, event):
        if not self.calc or len(self.calc.phase_shifts) == 0:
            return
        idx = self.choice_sim_phase.GetSelection()
        phase = self.calc.phase_shifts[idx]
        
        if self.is_hom:
            n_str = self.choice_HOM_n.GetStringSelection()
            if n_str:
                self.calc.set_n_and_phase(int(n_str), phase)
        else:
            self.calc.set_phase(phase)
            
        self._update_ui_choices()
        self.update_plots()

    def on_click(self, event):
        if not event.dblclick:
            return
            
        z, r = event.xdata, event.ydata
        if self.calc is None or z is None or r is None:
            return
            
        try:
            theta_rad = np.deg2rad(self.spin_ctrl_time_phase.GetValue())
            if self.is_hom:
                real_mode = self.calc.modes[self.current_mode_idx][0]
                res = self.calc.calculate_fields(z, r, real_mode, theta_time=self.spin_ctrl_time_phase.GetValue())
                
                if res is None:
                    wx.MessageBox(f"Point ({z:.4f}, {r:.4f}) is outside the cavity domain.", "Info")
                    return
                
                full_msg = (f"Coordinates:\n  z = {z:.6f} m\n  r = {r:.6f} m\n\n"
                            f"Field values:\n"
                            f"  E_z = {res['Ez']:.6e} V/m\n"
                            f"  E_r = {res['Er']:.6e} V/m\n"
                            f"  E_theta = {res['E_theta']:.6e} V/m\n"
                            f"  |E| = {res['E_abs']:.6e} V/m\n\n"
                            f"  H_z = {res['Hz']:.6e} A/m\n"
                            f"  H_r = {res['Hr']:.6e} A/m\n"
                            f"  H_theta = {res['H_theta']:.6e} A/m\n"
                            f"  |H| = {res['H_abs']:.6e} A/m")
            else:
                analysis_type = getattr(self.calc, 'analysis_type', 'standing')
                if analysis_type == 'standing':
                    res = self.calc.calculate_fields(z, r, self.current_mode_idx, theta=0.0)
                    msg_header = "Peak values (Real component at t=0):"
                else:
                    res = self.calc.calculate_fields(z, r, self.current_mode_idx, theta=theta_rad)
                    msg_header = "Instantaneous values:"

                if res is None:
                    wx.MessageBox(f"Point ({z:.4f}, {r:.4f}) is outside the cavity domain.", "Info")
                    return
                
                psi = r * res['H_theta']
                full_msg = (f"Coordinates:\n  z = {z:.6f} m\n  r = {r:.6f} m\n\n"
                            f"{msg_header}\n"
                            f"  H_theta   = {res['H_theta']:.6e} A/m\n"
                            f"  Psi (r*H) = {psi:.6e} A\n"
                            f"  Ez        = {res['Ez']:.6e} V/m\n"
                            f"  Er        = {res['Er']:.6e} V/m\n"
                            f"  |E|       = {res['E_abs']:.6e} V/m")
            
            wx.MessageBox(full_msg, f"Field Values", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Calculation failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_option_change(self, event):
        self.update_plots()

    # ------------------------------------------------------------
    # Export 機能 (Area / Line / Axis)
    # ------------------------------------------------------------
    def _get_current_mode_info(self):
        """現在表示中のモードに対応する CLI 引数情報を返す。"""
        if self.calc is None:
            return None
        info = {
            'h5_file': self.calc.h5_file,
            'time_phase': int(self.spin_ctrl_time_phase.GetValue()),
            'phase_shift': float(self.calc.phase_shift),
            'analysis_type': getattr(self.calc, 'analysis_type', 'standing'),
        }
        if self.is_hom:
            real_mode_idx = self.calc.modes[self.current_mode_idx][0]
            info['mode_index'] = int(real_mode_idx)
            info['n'] = int(self.calc.current_n)
            info['z_min'] = float(np.min(self.calc.vertices[:, 0]))
            info['z_max'] = float(np.max(self.calc.vertices[:, 0]))
            info['r_min'] = float(np.min(self.calc.vertices[:, 1]))
            info['r_max'] = float(np.max(self.calc.vertices[:, 1]))
            info['code_type'] = 'HOM'
        else:
            info['mode_index'] = int(self.current_mode_idx)
            info['n'] = None
            info['z_min'] = float(np.min(self.calc.nodes[:, 0]))
            info['z_max'] = float(np.max(self.calc.nodes[:, 0]))
            info['r_min'] = float(np.min(self.calc.nodes[:, 1]))
            info['r_max'] = float(np.max(self.calc.nodes[:, 1]))
            info['code_type'] = 'TM0'
        return info

    def _open_export_dialog(self, shape):
        info = self._get_current_mode_info()
        if info is None:
            wx.MessageBox("結果ファイルが読み込まれていません。", "Error",
                          wx.OK | wx.ICON_ERROR)
            return
        dlg = ExportDialog(self, shape, info)
        if dlg.ShowModal() == wx.ID_OK:
            cmd = dlg.build_command()
            self._run_export_command(cmd, dlg.label)
        dlg.Destroy()

    def OnBtnExportArea(self, event):
        self._open_export_dialog('area')
        if event:
            event.Skip()

    def OnBtnExportLine(self, event):
        self._open_export_dialog('line')
        if event:
            event.Skip()

    def OnBtnExportAxis(self, event):
        self._open_export_dialog('axis')
        if event:
            event.Skip()

    def OnBtnSaveGIF(self, event):
        info = self._get_current_mode_info()
        if info is None:
            wx.MessageBox("結果ファイルが読み込まれていません。", "Error",
                          wx.OK | wx.ICON_ERROR)
            if event:
                event.Skip()
            return
        if info.get('analysis_type', 'standing') == 'standing':
            wx.MessageBox("GIFアニメーションは進行波モードのみ対応しています。",
                          "Info", wx.OK | wx.ICON_INFORMATION)
            if event:
                event.Skip()
            return

        dlg = GifAnimationDialog(self, info)
        if dlg.ShowModal() == wx.ID_OK:
            params = dlg.get_params()
            dlg.Destroy()
            self._generate_gif(params)
        else:
            dlg.Destroy()
        if event:
            event.Skip()

    def _generate_gif(self, params):
        """オフスクリーンレンダリングでGIFアニメーションを生成する。"""
        try:
            from PIL import Image
        except ImportError:
            wx.MessageBox(
                "Pillowライブラリが見つかりません。\n"
                "pip install Pillow  でインストールしてください。",
                "Error", wx.OK | wx.ICON_ERROR)
            return

        n_frames = params['n_frames']
        fps = params['fps']
        output_path = params['output_path']
        theta_list = np.linspace(0.0, 360.0, n_frames, endpoint=False)

        # 現在の表示設定を読み取る
        show_h_theta = self.checkbox_H_theta.GetValue()
        show_psi = self.checkbox_E_lines.GetValue()
        show_vectors = self.checkbox_vectors.GetValue()
        show_mesh = self.checkbox_show_mesh.GetValue()
        n_levels = self.spin_ctrl_E_line_levels.GetValue()
        z_steps = self.spin_ctrl_vectors_z.GetValue()
        r_steps = self.spin_ctrl_vectors_r.GetValue()
        mode_idx = self.current_mode_idx

        # オフスクリーン Figure (現在のキャンバスと同サイズ)
        from matplotlib.figure import Figure as MplFigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        w_px, h_px = self.canvas.GetSize()
        dpi = 100
        fig_w = max(w_px, 400) / dpi
        fig_h = max(h_px, 300) / dpi
        fig_off = MplFigure(figsize=(fig_w, fig_h), dpi=dpi)
        canvas_agg = FigureCanvasAgg(fig_off)

        frames = []
        progress = wx.ProgressDialog(
            "GIF生成中", "フレームをレンダリングしています...",
            maximum=n_frames, parent=self,
            style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)

        try:
            for i, theta_deg in enumerate(theta_list):
                progress.Update(i, f"フレーム {i+1}/{n_frames}  (θ={theta_deg:.1f}°)")
                fig_off.clf()

                if self.is_hom:
                    ax_left = fig_off.add_subplot(121)
                    ax_right = fig_off.add_subplot(122)
                    self.plotter.plot_mode(
                        mode_idx,
                        ax_left=ax_left,
                        ax_right=ax_right,
                        theta=theta_deg,
                        show_colormaps=show_h_theta,
                        show_vectors=show_vectors,
                        show_mesh=show_mesh,
                        E_lines=n_levels,
                        v_steps=(z_steps, r_steps)
                    )
                    freq = self.calc.frequencies[mode_idx]
                    real_mode_idx = self.calc.modes[mode_idx][0]
                    n = self.calc.current_n
                    phase_shift = getattr(self.calc, 'phase_shift', 0.0)
                    fig_off.suptitle(
                        f"HOM n={n}, Mode {real_mode_idx}: f={freq:.4f} GHz"
                        f", phase={phase_shift:.0f}°  θ={theta_deg:.1f}°")
                else:
                    ax = fig_off.add_subplot(111)
                    self.plotter.plot_mode_snapshot(
                        mode_idx,
                        theta=theta_deg,
                        ax=ax,
                        show_h_theta=show_h_theta,
                        show_psi=show_psi,
                        show_vectors=show_vectors,
                        show_mesh=show_mesh,
                        E_lines=n_levels,
                        v_steps=(z_steps, r_steps)
                    )
                    freq = self.calc.frequencies[mode_idx]
                    phase_shift = getattr(self.calc, 'phase_shift', 0.0)
                    ax.set_title(
                        f"Mode {mode_idx}: f={freq:.4f} GHz"
                        f", phase={phase_shift:.0f}°  θ={theta_deg:.1f}°")

                fig_off.tight_layout()
                canvas_agg.draw()
                cw, ch = canvas_agg.get_width_height()
                buf = np.frombuffer(canvas_agg.buffer_rgba(), dtype=np.uint8)
                img = Image.fromarray(buf.reshape(ch, cw, 4)).convert('RGB')
                frames.append(img)

        finally:
            progress.Destroy()

        if not frames:
            return

        duration_ms = int(1000 / fps)
        try:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=duration_ms,
                optimize=False
            )
            wx.MessageBox(
                f"GIFアニメーションを保存しました。\n{output_path}\n\n"
                f"{n_frames}フレーム, {fps}fps",
                "保存完了", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"GIF保存に失敗しました: {e}", "Error",
                          wx.OK | wx.ICON_ERROR)

    def _log_command(self, cmd, label):
        """command.log にコマンドを追記する (MyFrame と同じ書式)。"""
        try:
            with open("command.log", 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 70}\n")
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"{label}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"{' '.join(cmd)}\n\n")
        except Exception:
            pass

    def _run_export_command(self, cmd, label):
        """サブプロセスでエクスポートを実行する (非同期)。"""
        self._log_command(cmd, label)
        self.set_status(f"Running: {' '.join(cmd)}")

        def target():
            log_lines = []
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='cp932', errors='replace',
                    shell=True
                )
                if proc.stdout:
                    for line in proc.stdout:
                        log_lines.append(line.rstrip())
                proc.wait()
                rc = proc.returncode
                summary = "\n".join(log_lines[-20:])
                if rc == 0:
                    wx.CallAfter(wx.MessageBox,
                                 f"{label} 完了。\n\n{summary}",
                                 "Export Complete", wx.OK | wx.ICON_INFORMATION)
                else:
                    wx.CallAfter(wx.MessageBox,
                                 f"{label} 失敗 (exit={rc})。\n\n{summary}",
                                 "Export Error", wx.OK | wx.ICON_ERROR)
                wx.CallAfter(self.set_status,
                             f"{label} finished (exit={rc}).")
            except Exception as e:
                wx.CallAfter(wx.MessageBox,
                             f"Export failed: {e}",
                             "Error", wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def update_plots(self):
        if self.calc is None or self.plotter is None:
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.set_aspect('equal')
            ax.text(0.5, 0.5, "Please load a result file (.h5)", transform=ax.transAxes, ha='center')
            self.canvas.draw()
            return

        wx.BeginBusyCursor()
        try:
            self.figure.clf()

            z_steps = self.spin_ctrl_vectors_z.GetValue()
            r_steps = self.spin_ctrl_vectors_r.GetValue()
            n_levels = self.spin_ctrl_E_line_levels.GetValue()
            theta_deg = self.spin_ctrl_time_phase.GetValue()

            if self.is_hom:
                self.axes_left = self.figure.add_subplot(121)
                self.axes_right = self.figure.add_subplot(122)
                
                self.plotter.plot_mode(
                    self.current_mode_idx,
                    ax_left=self.axes_left,
                    ax_right=self.axes_right,
                    theta=theta_deg,
                    show_colormaps=self.checkbox_H_theta.GetValue(),
                    show_vectors=self.checkbox_vectors.GetValue(),
                    show_mesh=self.checkbox_show_mesh.GetValue(),
                    E_lines=n_levels,
                    v_steps=(z_steps, r_steps)
                )
                
                freq = self.calc.frequencies[self.current_mode_idx]
                real_mode_idx = self.calc.modes[self.current_mode_idx][0]
                n = self.calc.current_n
                phase_str = f", phase={self.calc.phase_shift:.0f}°" if getattr(self.calc, 'analysis_type', 'standing') != 'standing' else ""
                title_str = f"HOM n={n}, Mode {real_mode_idx}: f = {freq:.6f} GHz{phase_str}"
                self.figure.suptitle(title_str)

                # Display engineering parameters in status text
                status_msg = title_str + "\n" + "="*len(title_str) + "\n"
                params = self.calc.get_mode_parameters(n, real_mode_idx)
                if params:
                    status_msg += f"Stored Energy (U): {params.get('stored_energy', 0):.6e} J\n"
                    status_msg += f"Wall Loss (P_loss): {params.get('p_loss', 0):.6e} W\n"
                    status_msg += f"Q Factor: {params.get('q_factor', 0):.2e}\n"
                else:
                    status_msg += "No engineering parameters calculated yet. Run Post-Process."
                
                self.text_ctrl_status.SetValue(status_msg)
                
            else:
                self.axes = self.figure.add_subplot(111)
                analysis_type = getattr(self.calc, 'analysis_type', 'standing')
                
                if analysis_type == 'standing':
                    self.plotter.plot_mode_standing(
                        self.current_mode_idx,
                        ax=self.axes,
                        show_h_theta=self.checkbox_H_theta.GetValue(),
                        show_psi=self.checkbox_E_lines.GetValue(),
                        show_vectors=self.checkbox_vectors.GetValue(),
                        show_mesh=self.checkbox_show_mesh.GetValue(),
                        E_lines=n_levels,
                        v_steps=(z_steps, r_steps)
                    )
                else:
                    self.plotter.plot_mode_snapshot(
                        self.current_mode_idx,
                        theta=theta_deg,
                        ax=self.axes,
                        show_h_theta=self.checkbox_H_theta.GetValue(),
                        show_psi=self.checkbox_E_lines.GetValue(),
                        show_vectors=self.checkbox_vectors.GetValue(),
                        show_mesh=self.checkbox_show_mesh.GetValue(),
                        E_lines=n_levels,
                        v_steps=(z_steps, r_steps)
                    )

                freq = self.calc.frequencies[self.current_mode_idx]
                title_str = f"Mode {self.current_mode_idx}: f = {freq:.6f} GHz"
                if analysis_type != 'standing':
                    title_str += f", phase={self.calc.phase_shift:.0f}°"
                if hasattr(self.calc, 'wall_loss') and self.calc.wall_loss is not None and self.current_mode_idx < len(self.calc.wall_loss):
                    P_wall = self.calc.wall_loss[self.current_mode_idx]
                    title_str += f", P_loss = {P_wall:.3e} W"
                self.axes.set_title(title_str)

            self.figure.tight_layout()
            self.canvas.draw()
        finally:
            wx.EndBusyCursor()

class GifAnimationDialog(wx.Dialog):
    """GIFアニメーション保存用パラメータ入力ダイアログ。"""

    def __init__(self, parent, info):
        super().__init__(parent, title="Save GIF Animation",
                         size=(480, 300),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.info = info

        sizer = wx.BoxSizer(wx.VERTICAL)

        # 情報表示
        code = info['code_type']
        mode = info['mode_index']
        n_str = f"  n={info['n']}" if info['n'] is not None else ""
        head = f"Code: {code}    Mode: {mode}{n_str}    (Traveling Wave)"
        sizer.Add(wx.StaticText(self, label=head), 0, wx.ALL, 8)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 4)

        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1, 1)

        grid.Add(wx.StaticText(self, label="1ループのフレーム数:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_frames = wx.SpinCtrl(self, min=4, max=360, initial=36)
        grid.Add(self.spin_frames, 0)

        grid.Add(wx.StaticText(self, label="FPS:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_fps = wx.SpinCtrl(self, min=1, max=60, initial=12)
        grid.Add(self.spin_fps, 0)

        grid.Add(wx.StaticText(self, label="出力ファイル名:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        out_sizer = wx.BoxSizer(wx.HORIZONTAL)
        default_gif = self._default_output_path()
        self.txt_output = wx.TextCtrl(self, value=default_gif)
        btn_browse = wx.Button(self, label="...")
        btn_browse.Bind(wx.EVT_BUTTON, self.on_browse)
        out_sizer.Add(self.txt_output, 1, wx.EXPAND | wx.RIGHT, 4)
        out_sizer.Add(btn_browse, 0)
        grid.Add(out_sizer, 1, wx.EXPAND)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 8)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 8)

        self.SetSizer(sizer)
        self.Layout()

    def _default_output_path(self):
        base, _ = os.path.splitext(self.info['h5_file'])
        suffix = f"_mode{self.info['mode_index']}"
        if self.info['n'] is not None:
            suffix += f"_n{self.info['n']}"
        suffix += "_anim.gif"
        return base + suffix

    def on_browse(self, event):
        wildcard = "GIF files (*.gif)|*.gif|All files (*.*)|*.*"
        with wx.FileDialog(self, "出力GIFファイル", wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            dlg.SetPath(self.txt_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                if not path.lower().endswith('.gif'):
                    path += '.gif'
                self.txt_output.SetValue(path)

    def get_params(self):
        return {
            'n_frames': self.spin_frames.GetValue(),
            'fps': self.spin_fps.GetValue(),
            'output_path': self.txt_output.GetValue(),
        }


class ExportDialog(wx.Dialog):
    """電磁場データエクスポート用パラメータ入力ダイアログ。

    呼び出し側で shape ('area' / 'line' / 'axis') と info dict を渡す。
    OK で ShowModal が wx.ID_OK を返したら build_command() で CLI コマンドを取得する。
    """

    def __init__(self, parent, shape, info):
        super().__init__(parent, title=f"Export Field Data ({shape.title()})",
                         size=(620, 560),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.shape = shape
        self.info = info
        self.is_traveling = (info.get('analysis_type', 'standing') != 'standing')
        self.label = (f"Export {shape.title()} "
                      f"({'HOM' if info['code_type'] == 'HOM' else 'TM0'} "
                      f"mode {info['mode_index']})")

        sizer = wx.BoxSizer(wx.VERTICAL)

        # 情報表示
        wave_str = ('Traveling' if self.is_traveling else 'Standing')
        head = (f"Code: {info['code_type']}    "
                f"Mode: {info['mode_index']}    "
                + (f"n: {info['n']}    " if info['n'] is not None else "")
                + f"Wave: {wave_str}    "
                f"Phase: {info['phase_shift']}°    "
                f"Time: {info['time_phase']}°")
        sizer.Add(wx.StaticText(self, label=head),
                  0, wx.ALL, 8)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 4)

        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=4, hgap=8)
        grid.AddGrowableCol(1, 1)

        # Z range (area / axis でも使用可能)
        z_min_def = f"{info['z_min']:.6g}"
        z_max_def = f"{info['z_max']:.6g}"
        grid.Add(wx.StaticText(self, label="Z range [m] (min,max):"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        z_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_zmin = wx.TextCtrl(self, value=z_min_def, size=(120, -1))
        self.txt_zmax = wx.TextCtrl(self, value=z_max_def, size=(120, -1))
        z_sizer.Add(self.txt_zmin, 0, wx.RIGHT, 4)
        z_sizer.Add(self.txt_zmax, 0)
        grid.Add(z_sizer, 1, wx.EXPAND)

        if shape == 'area':
            grid.Add(wx.StaticText(self, label="R range [m] (min,max):"),
                     0, wx.ALIGN_CENTER_VERTICAL)
            r_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self.txt_rmin = wx.TextCtrl(self, value=f"{info['r_min']:.6g}",
                                        size=(120, -1))
            self.txt_rmax = wx.TextCtrl(self, value=f"{info['r_max']:.6g}",
                                        size=(120, -1))
            r_sizer.Add(self.txt_rmin, 0, wx.RIGHT, 4)
            r_sizer.Add(self.txt_rmax, 0)
            grid.Add(r_sizer, 1, wx.EXPAND)

            grid.Add(wx.StaticText(self, label="Grid (nz, nr):"),
                     0, wx.ALIGN_CENTER_VERTICAL)
            g_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self.spin_nz = wx.SpinCtrl(self, min=2, max=10000, initial=200)
            self.spin_nr = wx.SpinCtrl(self, min=2, max=10000, initial=100)
            g_sizer.Add(self.spin_nz, 0, wx.RIGHT, 4)
            g_sizer.Add(self.spin_nr, 0)
            grid.Add(g_sizer, 1, wx.EXPAND)

        elif shape == 'line':
            grid.Add(wx.StaticText(self, label="P1 (z,r) [m]:"),
                     0, wx.ALIGN_CENTER_VERTICAL)
            self.txt_p1 = wx.TextCtrl(self,
                                      value=f"{info['z_min']:.6g},0.0")
            grid.Add(self.txt_p1, 1, wx.EXPAND)

            grid.Add(wx.StaticText(self, label="P2 (z,r) [m]:"),
                     0, wx.ALIGN_CENTER_VERTICAL)
            self.txt_p2 = wx.TextCtrl(self,
                                      value=f"{info['z_max']:.6g},0.0")
            grid.Add(self.txt_p2, 1, wx.EXPAND)

            grid.Add(wx.StaticText(self, label="Number of points:"),
                     0, wx.ALIGN_CENTER_VERTICAL)
            self.spin_npts = wx.SpinCtrl(self, min=2, max=100000, initial=500)
            grid.Add(self.spin_npts, 0)

        else:  # axis
            grid.Add(wx.StaticText(self, label="Number of points:"),
                     0, wx.ALIGN_CENTER_VERTICAL)
            self.spin_npts = wx.SpinCtrl(self, min=2, max=100000, initial=500)
            grid.Add(self.spin_npts, 0)

        # スケーリング
        grid.Add(wx.StaticText(self, label="Scale to power [W]:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        scale_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_scale = wx.CheckBox(self, label="enable")
        self.txt_power = wx.TextCtrl(self, value="1.0", size=(120, -1))
        self.txt_power.Enable(False)
        self.chk_scale.Bind(wx.EVT_CHECKBOX,
                            lambda e: self.txt_power.Enable(self.chk_scale.GetValue()))
        scale_sizer.Add(self.chk_scale, 0, wx.RIGHT, 8)
        scale_sizer.Add(self.txt_power, 0)
        grid.Add(scale_sizer, 1, wx.EXPAND)

        # 出力モード選択 (進行波のみ意味がある)
        grid.Add(wx.StaticText(self, label="Output mode:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        self.radio_outmode = wx.RadioBox(
            self, label="",
            choices=["Complex (Re/Im)", "Instant real"],
            majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        if self.is_traveling:
            self.radio_outmode.SetSelection(0)  # default: complex
        else:
            self.radio_outmode.SetSelection(1)
            self.radio_outmode.Disable()  # 定在波は実数のみ
        grid.Add(self.radio_outmode, 0)

        # 出力フォーマット
        grid.Add(wx.StaticText(self, label="Output format:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        self.radio_format = wx.RadioBox(
            self, label="",
            choices=["Both (HDF5 + Text)", "HDF5 only", "Text only"],
            majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        self.radio_format.SetSelection(0)
        grid.Add(self.radio_format, 0)

        # 出力ファイル
        grid.Add(wx.StaticText(self, label="Output base path:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        out_sizer = wx.BoxSizer(wx.HORIZONTAL)
        default_base = self._default_output_path()
        self.txt_output = wx.TextCtrl(self, value=default_base)
        btn_browse = wx.Button(self, label="...")
        btn_browse.Bind(wx.EVT_BUTTON, self.on_browse)
        out_sizer.Add(self.txt_output, 1, wx.EXPAND | wx.RIGHT, 4)
        out_sizer.Add(btn_browse, 0)
        grid.Add(out_sizer, 1, wx.EXPAND)

        sizer.Add(grid, 1, wx.EXPAND | wx.ALL, 8)

        # 注意書き
        note = ("出力は同名ベースで .h5 と .txt の 2 ファイル。\n"
                "Scale to power は processed.h5 (P_loss あり) が必要。")
        sizer.Add(wx.StaticText(self, label=note), 0, wx.ALL, 8)

        # OK / Cancel
        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 8)

        self.SetSizer(sizer)
        self.Layout()

    def _default_output_path(self):
        h5 = self.info['h5_file']
        base, _ = os.path.splitext(h5)
        suffix = {
            'area': '_field_area',
            'line': '_field_line',
            'axis': '_field_axis',
        }[self.shape]
        suffix += f"_m{self.info['mode_index']}"
        if self.info['n'] is not None:
            suffix += f"_n{self.info['n']}"
        return base + suffix

    def on_browse(self, event):
        wildcard = "All files (*.*)|*.*"
        with wx.FileDialog(self, "Output base path", wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            dlg.SetPath(self.txt_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                base, _ = os.path.splitext(dlg.GetPath())
                self.txt_output.SetValue(base)

    def build_command(self):
        info = self.info
        if info['code_type'] == 'HOM':
            script = "FEM_HOM_code/export_field_data.py"
        else:
            script = "FEM_code/export_field_data.py"

        cmd = [sys.executable, script,
               "-i", info['h5_file'],
               "-m", str(info['mode_index']),
               "-o", self.txt_output.GetValue(),
               "--shape", self.shape,
               "--time-phase", str(info['time_phase'])]

        if info['code_type'] == 'HOM':
            cmd += ["--n", str(info['n'])]
            cmd += ["--phase", str(info['phase_shift'])]
        else:
            if info['phase_shift']:
                cmd += ["--phase", str(info['phase_shift'])]

        zmin = self.txt_zmin.GetValue().strip()
        zmax = self.txt_zmax.GetValue().strip()
        if zmin and zmax:
            cmd += ["--z-range", f"{zmin},{zmax}"]

        if self.shape == 'area':
            rmin = self.txt_rmin.GetValue().strip()
            rmax = self.txt_rmax.GetValue().strip()
            if rmin and rmax:
                cmd += ["--r-range", f"{rmin},{rmax}"]
            cmd += ["--nz", str(self.spin_nz.GetValue()),
                    "--nr", str(self.spin_nr.GetValue())]
        elif self.shape == 'line':
            cmd += ["--p1", self.txt_p1.GetValue().strip(),
                    "--p2", self.txt_p2.GetValue().strip(),
                    "--npts", str(self.spin_npts.GetValue())]
        else:  # axis
            cmd += ["--npts", str(self.spin_npts.GetValue())]

        if self.chk_scale.GetValue():
            cmd += ["--scale-to-power", self.txt_power.GetValue().strip()]

        # 出力モード: 進行波で "Instant real" を選択した場合のみ --instant 付与
        # (default は複素振幅出力。定在波は --instant が無視される)
        if self.is_traveling and self.radio_outmode.GetSelection() == 1:
            cmd += ["--instant"]

        # 出力フォーマット: 0=Both, 1=HDF5 only, 2=Text only
        fmt_map = {0: 'both', 1: 'h5', 2: 'txt'}
        cmd += ["--format", fmt_map[self.radio_format.GetSelection()]]

        return cmd


if __name__ == "__main__":
    app = wx.App()
    dlg = ResultViewer(None)
    dlg.ShowModal()
    app.MainLoop()
