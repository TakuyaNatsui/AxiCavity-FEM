"""インタラクティブ結果ビューア（ver2 field_recon / reports へ再配線）.

ver1 ``ResultViewer.py`` を移植し、データ取得を ver2 の
:func:`axicavity_fem.shared.hdf5_io.read_results` + 場再構成
（:mod:`axicavity_fem.fem_tm0.field_recon` / :mod:`axicavity_fem.fem_hom.field_recon`）に、
場データ出力を :func:`axicavity_fem.reports.export_fields.export_fields` に置き換えた版。

描画は埋め込み matplotlib (WXAgg) キャンバスに field_recon の節点場を tricontourf で表示する。
"""

from __future__ import annotations

import os
import threading

import wx
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.tri as mtri
import numpy as np

from ..shared.boundary_groups import (material_interface_geometry,
                                      pec_boundary_geometry)
from ..shared.hdf5_io import is_v2, load_v1_legacy, read_results
from .result_viewer_ui import ResultViewerUI

_CMAP_BIPOLAR = "jet"
_CMAP_MAG = "hot_r"


def _triang(nodes, simplices):
    s = np.asarray(simplices)[:, :3]
    return mtri.Triangulation(np.asarray(nodes)[:, 0], np.asarray(nodes)[:, 1], s)


def _triang_vis(nodes, simplices, elem_order):
    """塗り/等高線用 triangulation。2次要素は中点ノードで4分割する（ver1 互換）."""
    nodes = np.asarray(nodes)
    s = np.asarray(simplices)
    if elem_order == 2 and s.shape[1] >= 6:
        tris = np.vstack([
            s[:, [0, 3, 5]],
            s[:, [3, 1, 4]],
            s[:, [5, 4, 2]],
            s[:, [3, 4, 5]],
        ])
    else:
        tris = s[:, :3]
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)


class ResultViewer(ResultViewerUI):
    """v2 HDF5 結果を読み込み、場マップを表示・エクスポートするダイアログ."""

    def __init__(self, parent, h5_file=None):
        super().__init__(parent)

        self.data = None
        self.solver_type = None
        self.h5_file = h5_file
        self._recon = None
        self._recon_key = None
        self._pec_cache = None
        self._mat_cache = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.panel_graph, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        graph_sizer = wx.BoxSizer(wx.VERTICAL)
        graph_sizer.Add(self.canvas, 1, wx.EXPAND)
        graph_sizer.Add(self.toolbar, 0, wx.EXPAND)
        self.panel_graph.SetSizer(graph_sizer)

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
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.set_status("Ready. Load an HDF5 result file to begin.")
        self.choice_HOM_n.Disable()
        if h5_file and os.path.exists(h5_file):
            self.load_result_file(h5_file)

    # ------------------------------------------------------------------
    def set_status(self, text):
        self.text_ctrl_status.SetValue(text)

    def on_close(self, event):
        self.EndModal(wx.ID_OK)

    @property
    def is_hom(self) -> bool:
        return self.solver_type == "hom"

    def load_result_file(self, h5_file):
        try:
            self.h5_file = h5_file
            self.data = (read_results(h5_file) if is_v2(h5_file)
                         else load_v1_legacy(h5_file))
            self.solver_type = self.data["solver_type"] or "tm0"
            self._recon = None
            self._recon_key = None
            self._pec_cache = None
            self._mat_cache = None

            ns = sorted(self.data["results_by_n"].keys())
            self.choice_HOM_n.Enable(self.is_hom)
            self.choice_HOM_n.Clear()
            self.choice_HOM_n.AppendItems([str(n) for n in ns])
            self.choice_HOM_n.SetSelection(0)

            title = "HOM" if self.is_hom else "TM0"
            self.SetTitle(f"{title} Result Viewer - {os.path.basename(h5_file)}")

            self._update_ui_choices()
            self.update_plots()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            wx.MessageBox(f"Failed to load result file: {e}", "Error",
                          wx.OK | wx.ICON_ERROR)

    # ------------------------------------------------------------------
    def _current_n(self):
        if self.is_hom:
            s = self.choice_HOM_n.GetStringSelection()
            return int(s) if s else sorted(self.data["results_by_n"])[0]
        return 0

    def _current_rec(self):
        return self.data["results_by_n"][self._current_n()]

    def _analysis_type(self):
        return "standing" if "standing" in self._current_rec() else "traveling"

    def _phase_list(self):
        rec = self._current_rec()
        if "traveling" in rec:
            return sorted(rec["traveling"].keys())
        return []

    def _current_modeset(self):
        rec = self._current_rec()
        if "standing" in rec:
            return rec["standing"], None
        phases = self._phase_list()
        idx = max(0, self.choice_sim_phase.GetSelection())
        phase = phases[idx] if idx < len(phases) else phases[0]
        return rec["traveling"][phase], phase

    def _update_ui_choices(self):
        modeset, _ = self._current_modeset()
        freqs = modeset["frequencies"]
        self.choice_mode.Clear()
        self.choice_mode.AppendItems(
            [f"Mode {i}: {f:.6f} GHz" for i, f in enumerate(freqs)])
        if len(freqs):
            self.choice_mode.SetSelection(0)

        self.choice_sim_phase.Clear()
        phases = self._phase_list()
        if phases:
            self.choice_sim_phase.AppendItems([f"{p}deg" for p in phases])
            self.choice_sim_phase.SetSelection(0)
        is_traveling = self._analysis_type() == "traveling"
        self.spin_ctrl_time_phase.Enable(is_traveling)
        self.choice_sim_phase.Enable(is_traveling)

        # E Lines は TM0 専用。HOM では E Lines / Levels を非表示にし、
        # 代わりに Vectors を既定で ON にする。
        show_elines = not self.is_hom
        self.checkbox_E_lines.Show(show_elines)
        self.label_E_levels.Show(show_elines)
        self.spin_ctrl_E_line_levels.Show(show_elines)
        if self.is_hom:
            self.checkbox_E_lines.SetValue(False)
            self.checkbox_vectors.SetValue(True)
        # ラベル等の表示/非表示を反映（コントロールバーを再レイアウト）
        try:
            self.checkbox_vectors.GetContainingSizer().Layout()
            self.Layout()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _get_recon(self, n, analysis_type):
        key = (self.solver_type, n, analysis_type)
        if self._recon_key == key:
            return self._recon
        mesh = self.data["mesh"]
        if self.is_hom:
            from ..fem_hom.field_recon import HOMFieldReconstructor
            recon = HOMFieldReconstructor(
                mesh["simplices"], mesh["vertices"], mesh["edge_index_map"],
                mesh["elem_order"], n, analysis_type)
        else:
            from ..fem_tm0.field_recon import TM0FieldReconstructor
            recon = TM0FieldReconstructor(
                mesh["vertices"], mesh["simplices"], mesh["elem_order"],
                analysis_type,
                eps_r_per_element=mesh.get("eps_r_per_element"))
        self._recon, self._recon_key = recon, key
        return recon

    def _node_fields(self, time_phase_deg):
        n = self._current_n()
        analysis_type = self._analysis_type()
        modeset, _ = self._current_modeset()
        mode = max(0, self.choice_mode.GetSelection())
        eigvec = modeset["eigenvectors"][mode]
        freq = float(modeset["frequencies"][mode])
        recon = self._get_recon(n, analysis_type)
        if self.is_hom:
            recon.load_mode(eigvec, freq)
            node = recon.calculate_all_node_fields(theta_time=time_phase_deg)
        else:
            node = recon.calculate_all_node_fields(
                eigvec, freq, theta=np.deg2rad(time_phase_deg),
                return_complex=(analysis_type == "traveling"))
        return node, freq, recon, mode

    # ------------------------------------------------------------------
    def update_plots(self):
        if self.data is None:
            return
        wx.BeginBusyCursor()
        try:
            tp = self.spin_ctrl_time_phase.GetValue()
            freq, mode = self._draw_on(self.figure, tp)
            self._show_params(freq, mode)
            self.canvas.draw()
        finally:
            wx.EndBusyCursor()

    def _draw_on(self, fig, time_phase_deg):
        """指定 figure に現在の表示設定で 1 フレーム描画し ``(freq, mode)`` を返す.

        画面更新（update_plots）と GIF 生成（オフスクリーン Agg figure）で共用する。
        """
        fig.clf()
        tp = time_phase_deg
        node, freq, recon, mode = self._node_fields(tp)
        mesh = self.data["mesh"]
        triang_vis = _triang_vis(mesh["vertices"], mesh["simplices"],
                                 mesh["elem_order"])
        triang_raw = _triang(mesh["vertices"], mesh["simplices"])
        show_mesh = self.checkbox_show_mesh.GetValue()
        show_color = self.checkbox_H_theta.GetValue()
        show_lines = self.checkbox_E_lines.GetValue()
        levels = self.spin_ctrl_E_line_levels.GetValue()

        if self.is_hom:
            axl = fig.add_subplot(121)
            axr = fig.add_subplot(122)
            self._panel(axl, triang_vis, triang_raw,
                        np.real(node["E_theta"]),
                        r"E-field (color $E_\theta$, vec $E_z,E_r$)",
                        show_mesh, show_color)
            self._panel(axr, triang_vis, triang_raw,
                        np.real(node["H_theta"]),
                        r"H-field (color $H_\theta$, vec $H_z,H_r$)",
                        show_mesh, show_color)
            if self.checkbox_vectors.GetValue():
                self._quiver_hom(axl, axr, recon, tp)
            self._draw_pec_boundary(axl)   # PEC 境界線は常時表示
            self._draw_pec_boundary(axr)
            n = self._current_n()
            fig.suptitle(
                f"HOM n={n} mode {mode}: f = {freq:.6f} GHz "
                f"({self._analysis_type()})")
        else:
            ax = fig.add_subplot(111)
            modeset, _ = self._current_modeset()
            eigvec = np.asarray(modeset["eigenvectors"][mode])
            hmax = float(np.max(np.abs(eigvec))) or 1.0
            self._panel(ax, triang_vis, triang_raw,
                        np.real(node["H_theta"]), "H_phi",
                        show_mesh, show_color, vmax=hmax)
            if show_lines:
                psi = np.real(node["Psi"])
                if self._analysis_type() == "traveling":
                    psi_max = float(np.max(np.abs(
                        mesh["vertices"][:, 1] * eigvec)))
                    lv = (np.linspace(-psi_max, psi_max, levels)
                          if psi_max > 1e-20 and levels >= 2 else levels)
                else:
                    lv = levels
                # E-Lines: 青紫の実線（負レベルが破線になる既定を linestyles で抑制）
                ax.tricontour(triang_vis, psi, levels=lv,
                              colors="blueviolet", linewidths=1.5,
                              linestyles="solid", alpha=0.9)
            if self.checkbox_vectors.GetValue():
                self._quiver_tm0(ax, recon, node, tp)
            self._draw_pec_boundary(ax)    # PEC 境界線は常時表示
            if self.checkbox_e_wall.GetValue():
                self._draw_pec_e_vectors(ax, node)
            ax.set_title(f"Mode {mode}: f = {freq:.6f} GHz "
                         f"({self._analysis_type()})")

        fig.tight_layout()
        return freq, mode

    def _panel(self, ax, triang_fill, triang_mesh, values, title, show_mesh,
               show_color, vmax=None):
        ax.set_aspect("equal")
        ax.set_xlabel("z [m]")
        ax.set_ylabel("r [m]")
        ax.set_title(title)
        if show_color:
            amax = vmax if vmax else (float(np.max(np.abs(values))) or 1.0)
            tcf = ax.tricontourf(triang_fill, values, levels=40,
                                 cmap=_CMAP_BIPOLAR, vmin=-amax, vmax=amax)
            ax.get_figure().colorbar(tcf, ax=ax, shrink=0.8)
        if show_mesh:
            ax.triplot(triang_mesh, color="gray", lw=0.3, alpha=0.4)

    def _pec_geometry(self):
        """PEC 境界の幾何 ``(segments, edges)`` を返す（メッシュ単位でキャッシュ）.

        計算は :func:`axicavity_fem.shared.boundary_groups.pec_boundary_geometry`
        に委譲（レポート側 plot_hom と共通）。``segments`` は黒太線用、``edges`` は
        PEC エッジ中点の電場ベクトル用（TM0）。
        """
        if self._pec_cache is not None:
            return self._pec_cache
        mesh = self.data["mesh"]
        self._pec_cache = pec_boundary_geometry(
            mesh["simplices"], mesh["vertices"],
            mesh.get("physical_groups"), mesh["elem_order"])
        return self._pec_cache

    def _draw_pec_boundary(self, ax):
        """PEC 境界（電気壁）を黒の太線で重ね描く（ver1 準拠）."""
        segments, _ = self._pec_geometry()
        if segments:
            ax.add_collection(LineCollection(
                segments, colors="black", linewidths=1.2, zorder=10))
        self._draw_material_interfaces(ax)

    def _draw_material_interfaces(self, ax):
        """誘電体界面（隣接要素で ε_r / tanδ が変わる内部辺）をマゼンタ破線で描く.

        計算は :func:`axicavity_fem.shared.boundary_groups.material_interface_geometry`
        に委譲（レポート側 plot_tm0/plot_hom と共通）。真空のみのメッシュでは
        何も描かない。
        """
        if self._mat_cache is None:
            mesh = self.data["mesh"]
            self._mat_cache = material_interface_geometry(
                mesh["simplices"], mesh["vertices"], mesh["elem_order"],
                eps_r_per_element=mesh.get("eps_r_per_element"),
                tan_delta_per_element=mesh.get("tan_delta_per_element"))
        segments, _ = self._mat_cache
        if segments:
            ax.add_collection(LineCollection(
                segments, colors="magenta", linewidths=1.5,
                linestyles="--", zorder=9))

    def _draw_pec_e_vectors(self, ax, node):
        """PEC エッジ中点に電場ベクトル (Ez, Er) を赤で描く（TM0 専用）.

        曲線境界でも 0 落ちしない節点場 ``node``（calculate_all_node_fields の結果、
        辺中点ノードを含む）を再利用する。2 次要素は中点ノード値、1 次要素は両端平均。
        """
        _, edges = self._pec_geometry()
        if not edges:
            return
        ez = np.real(node["Ez"])
        er = np.real(node["Er"])
        Z, R, U, V = [], [], [], []
        for e in edges:
            if e["mid"] is not None:
                u, v = ez[e["mid"]], er[e["mid"]]
            else:
                u = 0.5 * (ez[e["a"]] + ez[e["b"]])
                v = 0.5 * (er[e["a"]] + er[e["b"]])
            Z.append(e["coord"][0]); R.append(e["coord"][1])
            U.append(u); V.append(v)
        ax.quiver(Z, R, U, V, color="red", alpha=0.9, pivot="tail", zorder=11)

    def _quiver_tm0(self, ax, recon, node, time_phase_deg):
        modeset, _ = self._current_modeset()
        mode = max(0, self.choice_mode.GetSelection())
        eigvec = modeset["eigenvectors"][mode]
        freq = float(modeset["frequencies"][mode])
        nodes = self.data["mesh"]["vertices"]
        nz = max(2, self.spin_ctrl_vectors_z.GetValue())
        nr = max(2, self.spin_ctrl_vectors_r.GetValue())
        zg = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), nz)
        rg = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), nr)
        Z, R, U, V = [], [], [], []
        for z in zg:
            for r in rg:
                res = recon.calculate_fields(eigvec, freq, z, r,
                                             theta=np.deg2rad(time_phase_deg))
                if res is None:
                    continue
                Z.append(z); R.append(r)
                U.append(np.real(res["Ez"])); V.append(np.real(res["Er"]))
        if Z:
            ax.quiver(Z, R, U, V, color="darkmagenta", alpha=0.8,
                      pivot="middle")

    def _quiver_hom(self, ax_e, ax_h, recon, time_phase_deg):
        """HOM の面内ベクトルを E 場パネルに (Ez,Er)、H 場パネルに (Hz,Hr) で描く."""
        nodes = self.data["mesh"]["vertices"]
        nz = max(2, self.spin_ctrl_vectors_z.GetValue())
        nr = max(2, self.spin_ctrl_vectors_r.GetValue())
        zg = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), nz)
        rg = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), nr)
        Z, R, Ez, Er, Hz, Hr = [], [], [], [], [], []
        for z in zg:
            for r in rg:
                res = recon.calculate_fields(z, max(r, 1e-10),
                                             theta_time=time_phase_deg)
                if res is None:
                    continue
                Z.append(z); R.append(r)
                Ez.append(res["Ez"]); Er.append(res["Er"])
                Hz.append(res["Hz"]); Hr.append(res["Hr"])
        if Z:
            ax_e.quiver(Z, R, Ez, Er, color="darkmagenta", alpha=0.8,
                        pivot="middle")
            ax_h.quiver(Z, R, Hz, Hr, color="darkmagenta", alpha=0.8,
                        pivot="middle")

    def _show_params(self, freq, mode):
        n = self._current_n()
        pp = self.data.get("post_process")
        msg = f"Mode {mode}: f = {freq:.6f} GHz\n"
        if pp and n in pp:
            rec = pp[n]
            plist = (rec.get("standing") if self._analysis_type() == "standing"
                     else next(iter(rec.get("traveling", {}).values()), None))
            if plist and mode < len(plist):
                p = plist[mode]
                for key in ("Q", "Q_wall", "Q_diel", "U_stored", "P_loss",
                            "P_diel", "P_flow_zmin", "group_velocity"):
                    if key in p:
                        msg += f"{key}: {p[key]:.6e}\n"
        else:
            msg += "（工学パラメータ未計算。post を実行してください）"
        self.set_status(msg)

    # ------------------------------------------------------------------
    def on_HOM_n_change(self, event):
        self._update_ui_choices()
        self.update_plots()

    def on_mode_change(self, event):
        self.update_plots()

    def on_sim_phase_change(self, event):
        self._update_ui_choices()
        self.update_plots()

    def on_option_change(self, event):
        self.update_plots()

    def on_click(self, event):
        if not event.dblclick or self.data is None:
            return
        z, r = event.xdata, event.ydata
        if z is None or r is None:
            return
        tp = self.spin_ctrl_time_phase.GetValue()
        modeset, _ = self._current_modeset()
        mode = max(0, self.choice_mode.GetSelection())
        recon = self._get_recon(self._current_n(), self._analysis_type())
        if self.is_hom:
            recon.load_mode(modeset["eigenvectors"][mode],
                            float(modeset["frequencies"][mode]))
            res = recon.calculate_fields(z, r, theta_time=tp)
            if res is None:
                wx.MessageBox(f"({z:.4f}, {r:.4f}) は領域外です。", "Info")
                return
            msg = (f"z={z:.6f}  r={r:.6f}\n"
                   f"E_z={res['Ez']:.4e}  E_r={res['Er']:.4e}  "
                   f"E_theta={res['E_theta']:.4e}\n"
                   f"H_z={res['Hz']:.4e}  H_r={res['Hr']:.4e}  "
                   f"H_theta={res['H_theta']:.4e}")
        else:
            res = recon.calculate_fields(
                modeset["eigenvectors"][mode],
                float(modeset["frequencies"][mode]), z, r,
                theta=np.deg2rad(tp))
            if res is None:
                wx.MessageBox(f"({z:.4f}, {r:.4f}) は領域外です。", "Info")
                return
            msg = (f"z={z:.6f}  r={r:.6f}\n"
                   f"H_theta={res['H_theta']:.4e}\n"
                   f"E_z={res['Ez']:.4e}  E_r={res['Er']:.4e}  "
                   f"|E|={res['E_abs']:.4e}")
        wx.MessageBox(msg, "Field Values", wx.OK | wx.ICON_INFORMATION)

    # ------------------------------------------------------------------
    # Export（reports.export_fields を直接呼び出す）
    # ------------------------------------------------------------------
    def _export(self, shape):
        if self.data is None:
            wx.MessageBox("結果ファイルが読み込まれていません。", "Error",
                          wx.OK | wx.ICON_ERROR)
            return
        modeset, phase = self._current_modeset()
        mode = max(0, self.choice_mode.GetSelection())
        base = os.path.splitext(self.h5_file)[0]
        out = f"{base}_field_{shape}_m{mode}"
        if self.is_hom:
            out += f"_n{self._current_n()}"

        time_phase = float(self.spin_ctrl_time_phase.GetValue())
        kwargs = dict(
            solver_type=self.solver_type, mode=mode,
            n=self._current_n() if self.is_hom else None,
            phase=phase, shape=shape, time_phase=time_phase,
            npts=500, nz=200, nr=100, fmt="both",
        )

        # area / line / axis はダイアログでパラメータを決める。
        if shape in ("area", "line", "axis"):
            verts = self.data["mesh"]["vertices"]
            z_bounds = (float(verts[:, 0].min()), float(verts[:, 0].max()))
            r_bounds = (float(verts[:, 1].min()), float(verts[:, 1].max()))
            dlg = ExportFieldDialog(self, shape=shape, default_output=out,
                                    z_bounds=z_bounds, r_bounds=r_bounds)
            try:
                if dlg.ShowModal() != wx.ID_OK:
                    return
                try:
                    params = dlg.get_params()
                except ValueError:
                    wx.MessageBox("数値の入力が不正です。", "Error",
                                  wx.OK | wx.ICON_ERROR)
                    return
            finally:
                dlg.Destroy()
            out = params.pop("output") or out
            kwargs.update(params)

        # 同等の CLI コマンドをログに残す
        self._log_export_command(shape, out, kwargs)
        self.set_status(f"Exporting {shape} ...")

        def target():
            from ..reports.export_fields import export_fields
            try:
                export_fields(self.h5_file, out, **kwargs)
                wx.CallAfter(wx.MessageBox,
                             f"Export 完了:\n{out}.h5 / {out}.txt",
                             "Export Complete", wx.OK | wx.ICON_INFORMATION)
                wx.CallAfter(self.set_status, f"Export finished: {out}")
            except Exception as e:  # noqa: BLE001
                wx.CallAfter(wx.MessageBox, f"Export failed: {e}", "Error",
                             wx.OK | wx.ICON_ERROR)

        threading.Thread(target=target, daemon=True).start()

    def _log_export_command(self, shape, out, kwargs):
        """場出力に相当する CLI コマンドを組み立て、親フレームのログに残す.

        ResultViewer は in-process で export_fields を呼ぶが、再現用に
        ``axicavity-fem export ...`` 相当のコマンドを command.log と
        FEM ログに記録する。
        """
        parent = self.GetParent()
        try:
            base_cmd = (parent._build_base_cmd()
                        if hasattr(parent, "_build_base_cmd")
                        else ["axicavity-fem"])
        except Exception:
            base_cmd = ["axicavity-fem"]

        cmd = base_cmd + ["export", "--type", self.solver_type,
                          "-i", self.h5_file, "-o", out,
                          "--shape", shape, "-m", str(kwargs["mode"])]
        if self.is_hom and kwargs.get("n") is not None:
            cmd += ["--n", str(kwargs["n"])]
        if kwargs.get("phase") is not None:
            cmd += ["--phase", str(kwargs["phase"])]
        if kwargs.get("time_phase"):
            cmd += ["--time-phase", str(kwargs["time_phase"])]
        if shape == "area":
            zr, rr = kwargs.get("z_range"), kwargs.get("r_range")
            if zr is not None:
                cmd += ["--z-range", f"{zr[0]},{zr[1]}"]
            if rr is not None:
                cmd += ["--r-range", f"{rr[0]},{rr[1]}"]
            cmd += ["--nz", str(kwargs.get("nz", 200)),
                    "--nr", str(kwargs.get("nr", 100))]
        elif shape == "line":
            p1, p2 = kwargs.get("p1"), kwargs.get("p2")
            if p1 is not None and p2 is not None:
                cmd += ["--p1", f"{p1[0]},{p1[1]}", "--p2", f"{p2[0]},{p2[1]}"]
            cmd += ["--npts", str(kwargs.get("npts", 500))]
        else:  # axis
            zr = kwargs.get("z_range")
            if zr is not None:
                cmd += ["--z-range", f"{zr[0]},{zr[1]}"]
            cmd += ["--npts", str(kwargs.get("npts", 500))]
        if kwargs.get("scale_to_power") is not None:
            cmd += ["--scale-to-power", str(kwargs["scale_to_power"])]
        cmd += ["--format", kwargs.get("fmt", "both")]

        label = f"Export {shape} field"
        if hasattr(parent, "log_command"):
            try:
                parent.log_command(cmd, label)
            except Exception:
                pass
        if hasattr(parent, "fem_log_ctrl"):
            try:
                parent.fem_log_ctrl.AppendText(
                    f"\n--- {label} (Result Viewer) ---\n"
                    f"{' '.join(cmd)}\n")
            except Exception:
                pass

    def OnBtnExportArea(self, event):
        self._export("area")

    def OnBtnExportLine(self, event):
        self._export("line")

    def OnBtnExportAxis(self, event):
        self._export("axis")

    # ------------------------------------------------------------------
    # GIF アニメーション保存（進行波: 時間位相 0→360° を周回）
    # ------------------------------------------------------------------
    def _default_gif_path(self):
        base = os.path.splitext(self.h5_file)[0]
        mode = max(0, self.choice_mode.GetSelection())
        suffix = f"_mode{mode}"
        if self.is_hom:
            suffix += f"_n{self._current_n()}"
        return base + suffix + "_anim.gif"

    def OnBtnSaveGIF(self, event):
        if self.data is None:
            wx.MessageBox("結果ファイルが読み込まれていません。", "Error",
                          wx.OK | wx.ICON_ERROR)
            return
        if self._analysis_type() != "traveling":
            wx.MessageBox("GIF アニメーションは進行波モードのみ対応しています。",
                          "Info", wx.OK | wx.ICON_INFORMATION)
            return
        dlg = GifAnimationDialog(self, self._default_gif_path())
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            params = dlg.get_params()
        finally:
            dlg.Destroy()
        self._generate_gif(params)

    def _generate_gif(self, params):
        """現在の表示設定のまま、時間位相 0→360° を周回する GIF をオフスクリーン生成する.

        各フレームは :meth:`_draw_on` をオフスクリーン Agg figure に対して呼び出すだけ
        なので、画面表示と同一の見た目になる。
        """
        from PIL import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as MplFigure

        n_frames = params["n_frames"]
        fps = params["fps"]
        out = params["output_path"]
        theta_list = np.linspace(0.0, 360.0, n_frames, endpoint=False)

        w_px, h_px = self.canvas.GetSize()
        dpi = 100
        fig_off = MplFigure(figsize=(max(w_px, 400) / dpi,
                                     max(h_px, 300) / dpi), dpi=dpi)
        canvas_agg = FigureCanvasAgg(fig_off)

        frames = []
        progress = wx.ProgressDialog(
            "GIF 生成中", "フレームをレンダリング中...", maximum=n_frames,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
        try:
            for i, theta in enumerate(theta_list):
                progress.Update(i, f"フレーム {i + 1}/{n_frames}  (θ={theta:.1f}°)")
                self._draw_on(fig_off, float(theta))
                canvas_agg.draw()
                cw, ch = canvas_agg.get_width_height()
                buf = np.frombuffer(canvas_agg.buffer_rgba(), dtype=np.uint8)
                frames.append(
                    Image.fromarray(buf.reshape(ch, cw, 4)).convert("RGB"))
        finally:
            progress.Destroy()

        if not frames:
            return
        try:
            frames[0].save(out, save_all=True, append_images=frames[1:],
                           loop=0, duration=int(1000 / fps), optimize=False)
            wx.MessageBox(f"GIF を保存しました:\n{out}\n\n"
                          f"{n_frames} frames, {fps} fps",
                          "保存完了", wx.OK | wx.ICON_INFORMATION)
            self.set_status(f"GIF saved: {out}")
        except Exception as e:  # noqa: BLE001
            wx.MessageBox(f"GIF 保存に失敗しました: {e}", "Error",
                          wx.OK | wx.ICON_ERROR)


class ExportFieldDialog(wx.Dialog):
    """場データ出力（area / line / axis）のパラメータ入力ダイアログ.

    area: Z/R 範囲とグリッド点数。line: 2 端点 P1/P2 と点数。
    axis: 軸 r=0 上の Z 範囲と点数。
    共通: Scale to power（空欄で無効）・出力形式・出力先。
    """

    def __init__(self, parent, *, shape, default_output, z_bounds, r_bounds,
                 nz=200, nr=100, npts=500):
        title = f"Export {shape.capitalize()} Field"
        super().__init__(parent, title=title, size=(520, 360),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.shape = shape
        sizer = wx.BoxSizer(wx.VERTICAL)
        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1, 1)

        def row(label, ctrl):
            grid.Add(wx.StaticText(self, label=label), 0,
                     wx.ALIGN_CENTER_VERTICAL)
            grid.Add(ctrl, 1, wx.EXPAND)

        zmin, zmax = z_bounds
        rmin, rmax = r_bounds
        if shape == "area":
            self.txt_zmin = wx.TextCtrl(self, value=f"{zmin:.6g}")
            self.txt_zmax = wx.TextCtrl(self, value=f"{zmax:.6g}")
            self.txt_rmin = wx.TextCtrl(self, value=f"{rmin:.6g}")
            self.txt_rmax = wx.TextCtrl(self, value=f"{rmax:.6g}")
            self.spin_nz = wx.SpinCtrl(self, min=2, max=5000, initial=nz)
            self.spin_nr = wx.SpinCtrl(self, min=2, max=5000, initial=nr)
            row("Z min [m]:", self.txt_zmin)
            row("Z max [m]:", self.txt_zmax)
            row("R min [m]:", self.txt_rmin)
            row("R max [m]:", self.txt_rmax)
            row("nz (Z 点数):", self.spin_nz)
            row("nr (R 点数):", self.spin_nr)
        elif shape == "axis":
            self.txt_zmin = wx.TextCtrl(self, value=f"{zmin:.6g}")
            self.txt_zmax = wx.TextCtrl(self, value=f"{zmax:.6g}")
            self.spin_npts = wx.SpinCtrl(self, min=2, max=20000, initial=npts)
            row("Z min [m]:", self.txt_zmin)
            row("Z max [m]:", self.txt_zmax)
            row("npts (点数):", self.spin_npts)
        else:  # line
            self.txt_p1z = wx.TextCtrl(self, value=f"{zmin:.6g}")
            self.txt_p1r = wx.TextCtrl(self, value=f"{rmin:.6g}")
            self.txt_p2z = wx.TextCtrl(self, value=f"{zmax:.6g}")
            self.txt_p2r = wx.TextCtrl(self, value=f"{rmin:.6g}")
            self.spin_npts = wx.SpinCtrl(self, min=2, max=20000, initial=npts)
            row("P1 Z [m]:", self.txt_p1z)
            row("P1 R [m]:", self.txt_p1r)
            row("P2 Z [m]:", self.txt_p2z)
            row("P2 R [m]:", self.txt_p2r)
            row("npts (点数):", self.spin_npts)

        # 共通: scale-to-power（空欄で無効）
        self.txt_power = wx.TextCtrl(self, value="")
        self.txt_power.SetHint("空欄=スケールなし。例: 1000 (=1kW 壁損失基準)")
        row("Scale to power [W]:", self.txt_power)

        self.choice_fmt = wx.Choice(self, choices=["both", "h5", "txt"])
        self.choice_fmt.SetSelection(0)
        row("Format:", self.choice_fmt)

        out_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_output = wx.TextCtrl(self, value=default_output)
        btn_browse = wx.Button(self, label="...")
        btn_browse.Bind(wx.EVT_BUTTON, self.on_browse)
        out_sizer.Add(self.txt_output, 1, wx.EXPAND | wx.RIGHT, 4)
        out_sizer.Add(btn_browse, 0)
        row("出力ベース名:", out_sizer)

        sizer.Add(grid, 1, wx.EXPAND | wx.ALL, 10)
        sizer.Add(wx.StaticText(
            self, label="出力は <ベース名>.h5 / .txt（Format に応じて）"),
            0, wx.LEFT | wx.RIGHT, 10)
        btns = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        self.SetSizer(sizer)
        self.Layout()

    def on_browse(self, event):
        with wx.FileDialog(self, "出力ベース名（拡張子なし）",
                           style=wx.FD_SAVE) as dlg:
            dlg.SetPath(self.txt_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                # 拡張子を除いてベース名にする
                self.txt_output.SetValue(os.path.splitext(dlg.GetPath())[0])

    def _power(self):
        s = self.txt_power.GetValue().strip()
        if not s:
            return None
        return float(s)

    def get_params(self) -> dict:
        """export_fields に渡す kwargs サブセット + 出力先を返す（検証込み）."""
        p = {
            "output": self.txt_output.GetValue().strip(),
            "fmt": self.choice_fmt.GetStringSelection(),
            "scale_to_power": self._power(),
        }
        if self.shape == "area":
            p["z_range"] = (float(self.txt_zmin.GetValue()),
                            float(self.txt_zmax.GetValue()))
            p["r_range"] = (float(self.txt_rmin.GetValue()),
                            float(self.txt_rmax.GetValue()))
            p["nz"] = int(self.spin_nz.GetValue())
            p["nr"] = int(self.spin_nr.GetValue())
        elif self.shape == "axis":
            p["z_range"] = (float(self.txt_zmin.GetValue()),
                            float(self.txt_zmax.GetValue()))
            p["npts"] = int(self.spin_npts.GetValue())
        else:
            p["p1"] = (float(self.txt_p1z.GetValue()),
                       float(self.txt_p1r.GetValue()))
            p["p2"] = (float(self.txt_p2z.GetValue()),
                       float(self.txt_p2r.GetValue()))
            p["npts"] = int(self.spin_npts.GetValue())
        return p


class GifAnimationDialog(wx.Dialog):
    """GIF アニメーション保存パラメータ入力ダイアログ（フレーム数 / FPS / 出力先）."""

    def __init__(self, parent, default_output_path):
        super().__init__(parent, title="Save GIF Animation",
                         size=(480, 220),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1, 1)

        grid.Add(wx.StaticText(self, label="1 ループのフレーム数:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_frames = wx.SpinCtrl(self, min=4, max=360, initial=36)
        grid.Add(self.spin_frames, 0)

        grid.Add(wx.StaticText(self, label="FPS:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_fps = wx.SpinCtrl(self, min=1, max=60, initial=12)
        grid.Add(self.spin_fps, 0)

        grid.Add(wx.StaticText(self, label="出力ファイル:"),
                 0, wx.ALIGN_CENTER_VERTICAL)
        out_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_output = wx.TextCtrl(self, value=default_output_path)
        btn_browse = wx.Button(self, label="...")
        btn_browse.Bind(wx.EVT_BUTTON, self.on_browse)
        out_sizer.Add(self.txt_output, 1, wx.EXPAND | wx.RIGHT, 4)
        out_sizer.Add(btn_browse, 0)
        grid.Add(out_sizer, 1, wx.EXPAND)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 10)
        btns = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        self.SetSizer(sizer)
        self.Layout()

    def on_browse(self, event):
        wildcard = "GIF files (*.gif)|*.gif|All files (*.*)|*.*"
        with wx.FileDialog(self, "出力 GIF ファイル", wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            dlg.SetPath(self.txt_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                if not path.lower().endswith(".gif"):
                    path += ".gif"
                self.txt_output.SetValue(path)

    def get_params(self):
        return {"n_frames": self.spin_frames.GetValue(),
                "fps": self.spin_fps.GetValue(),
                "output_path": self.txt_output.GetValue()}
