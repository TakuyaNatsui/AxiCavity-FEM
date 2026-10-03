"""結果（HDF5）の読み込みと場の描画（Qt 非依存。ver2.3 ``gui/result_viewer.py`` の描画部分の移植）.

- :class:`ResultData`: ``read_results`` の dict を包み、n / 位相 / モード / post のパラメータを問い合わせる
- :class:`ResultRenderer`: 指定の :class:`Figure` に 1 フレーム描く（画面と GIF のオフスクリーン描画で共用）。
  TM0: H_φ の色（jet、±max）、Ψ = r·H_φ の等高線（E 線）、格子 quiver（E_z, E_r）、PEC 境界の黒線、誘電体界面の
  マゼンタ破線、PEC 辺中点の E ベクトル（E-wall、赤）、メッシュ。HOM: E_θ / H_θ の 2 パネルと面内 quiver
- :func:`render_gif`: 進行波の時間位相 0→360° を周回する GIF（Agg キャンバス）
- :func:`mode_rows`: モード表の行（f と post の値）

matplotlib は ``Figure`` と ``Agg`` だけを使い ``pyplot`` は import しない（Qt キャンバスと混ぜない）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

from ...shared.boundary_groups import material_interface_geometry, pec_boundary_geometry
from ...shared.hdf5_io import is_v2, load_v1_legacy, read_results

CMAP_BIPOLAR = "jet"
# モード表の列: (post のキー, 見出し, 単位)。f は別扱い
MODE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("Q", "Q", ""), ("Q_wall", "Q_wall", ""), ("Q_diel", "Q_diel", ""), ("R_over_Q", "R/Q", "Ω"),
    ("V_eff", "V_eff", "V"), ("V_acc", "V_acc", "V"), ("U_stored", "U", "J"), ("P_loss", "P_loss", "W"),
    ("P_diel", "P_diel", "W"), ("P_flow_zmin", "P_flow", "W"), ("group_velocity", "v_g", "m/s"),
    ("attenuation", "α", "1/m"), ("v_phase_c", "v_ph/c", ""),
)
TM0_POPUP_KEYS = ("H_theta", "Ez", "Er", "E_abs")
HOM_POPUP_KEYS = ("Ez", "Er", "E_theta", "Hz", "Hr", "H_theta")


@dataclass(frozen=True)
class Selection:
    """表示するモード: 方位角次数 n、進行波の位相（定在波は None）、モード番号、時間位相 [deg]."""

    n: int = 0
    phase: Optional[float] = None
    mode: int = 0
    time_phase: float = 0.0


@dataclass(frozen=True)
class ViewOptions:
    show_color: bool = True         # H_θ（HOM は E_θ / H_θ）の色
    show_lines: bool = True         # E 線（TM0 だけ）
    levels: int = 20
    show_vectors: bool = False      # 格子の quiver
    nz: int = 30
    nr: int = 15
    show_mesh: bool = False
    show_e_wall: bool = False       # PEC 辺の E ベクトル（TM0 だけ）


def load_result(path: str | Path) -> dict:
    """結果 HDF5 を読む（v2 形式、無ければ ver1 の旧形式）."""
    path = str(path)
    return read_results(path) if is_v2(path) else load_v1_legacy(path)


def _triang(nodes, simplices):
    s = np.asarray(simplices)[:, :3]
    return mtri.Triangulation(np.asarray(nodes)[:, 0], np.asarray(nodes)[:, 1], s)


def _triang_vis(nodes, simplices, elem_order):
    """塗り / 等高線用。2 次要素は中点ノードで 4 分割する（ver1 互換）."""
    nodes = np.asarray(nodes)
    s = np.asarray(simplices)
    if elem_order == 2 and s.shape[1] >= 6:
        tris = np.vstack([s[:, [0, 3, 5]], s[:, [3, 1, 4]], s[:, [5, 4, 2]], s[:, [3, 4, 5]]])
    else:
        tris = s[:, :3]
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)


def _clean_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


class ResultData:
    """結果ファイル 1 つ（solve または post 済み）."""

    def __init__(self, path: str | Path, raw: Optional[dict] = None):
        self.path = Path(path)
        self.raw = raw if raw is not None else load_result(path)
        self.solver_type = str(self.raw.get("solver_type") or "tm0")
        if isinstance(self.solver_type, bytes):
            self.solver_type = self.solver_type.decode()
        self.mesh = self.raw["mesh"]
        self.results_by_n: dict = self.raw["results_by_n"]
        self.post: dict = self.raw.get("post_process") or {}
        if not self.results_by_n:
            raise ValueError("結果（results）がありません")

    # ---- 問い合わせ -----------------------------------------------------

    @property
    def is_hom(self) -> bool:
        return self.solver_type == "hom"

    @property
    def n_orders(self) -> list[int]:
        return sorted(int(n) for n in self.results_by_n)

    def analysis_type(self, n: Optional[int] = None) -> str:
        rec = self.results_by_n[self._n(n)]
        return "standing" if "standing" in rec else "traveling"

    @property
    def is_traveling(self) -> bool:
        return self.analysis_type() == "traveling"

    def phases(self, n: Optional[int] = None) -> list[float]:
        rec = self.results_by_n[self._n(n)]
        return sorted(float(p) for p in rec["traveling"]) if "traveling" in rec else []

    def _n(self, n: Optional[int]) -> int:
        orders = self.n_orders
        if n is None or int(n) not in self.results_by_n:
            return orders[0]
        return int(n)

    def _phase(self, n: int, phase: Optional[float]) -> Optional[float]:
        phases = self.phases(n)
        if not phases:
            return None
        if phase is not None:
            for p in phases:
                if abs(p - float(phase)) < 1e-9:
                    return p
        return phases[0]

    def normalize(self, sel: Selection) -> Selection:
        """存在する n / 位相 / モード番号に丸めた Selection."""
        n = self._n(sel.n)
        phase = self._phase(n, sel.phase)
        count = self.num_modes(n, phase)
        mode = min(max(0, int(sel.mode)), max(0, count - 1))
        return replace(sel, n=n, phase=phase, mode=mode)

    def modeset(self, n: Optional[int] = None, phase: Optional[float] = None) -> dict:
        n = self._n(n)
        rec = self.results_by_n[n]
        if "standing" in rec:
            return rec["standing"]
        trav = rec["traveling"]
        key = self._phase(n, phase)
        for p in trav:
            if abs(float(p) - key) < 1e-9:
                return trav[p]
        return trav[sorted(trav)[0]]

    def frequencies(self, n: Optional[int] = None, phase: Optional[float] = None) -> np.ndarray:
        return np.asarray(self.modeset(n, phase)["frequencies"], dtype=float)

    def num_modes(self, n: Optional[int] = None, phase: Optional[float] = None) -> int:
        return int(len(self.modeset(n, phase)["frequencies"]))

    @property
    def has_post(self) -> bool:
        return bool(self.post)

    def post_list(self, n: Optional[int] = None, phase: Optional[float] = None) -> list[dict]:
        n = self._n(n)
        rec = self.post.get(n) or {}
        if "standing" in rec:
            return rec["standing"] or []
        trav = rec.get("traveling") or {}
        key = self._phase(n, phase)
        for p, value in trav.items():
            if key is not None and abs(float(p) - key) < 1e-9:
                return value or []
        return []

    def post_params(self, sel: Selection) -> Optional[dict]:
        """モードの post パラメータ（無ければ None）. 値は float."""
        plist = self.post_list(sel.n, sel.phase)
        if sel.mode < len(plist):
            return {k: _clean_float(v) for k, v in plist[sel.mode].items()}
        return None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(zmin, zmax, rmin, rmax) [m]."""
        v = np.asarray(self.mesh["vertices"])
        return float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max())

    def title(self, sel: Selection) -> str:
        freq = float(self.frequencies(sel.n, sel.phase)[sel.mode])
        kind = self.analysis_type(sel.n)
        if self.is_hom:
            return f"HOM n={sel.n} mode {sel.mode}: f = {freq:.6f} GHz ({kind})"
        return f"Mode {sel.mode}: f = {freq:.6f} GHz ({kind})"


def mode_rows(data: ResultData, n: Optional[int] = None, phase: Optional[float] = None) -> list[dict]:
    """モード表の行: ``{"index", "f_GHz", <post のキー>...}``（post が無ければ f だけ）."""
    freqs = data.frequencies(n, phase)
    plist = data.post_list(n, phase)
    rows = []
    for k, f in enumerate(freqs):
        row = {"index": k, "f_GHz": float(f)}
        if k < len(plist):
            for key, _label, _unit in MODE_COLUMNS:
                if key in plist[k]:
                    row[key] = _clean_float(plist[k][key])
        rows.append(row)
    return rows


class ResultRenderer:
    """場の描画（ver2.3 ``ResultViewer._draw_on`` の移植）. 再構成器・境界の幾何・三角形分割はキャッシュする."""

    def __init__(self, data: ResultData):
        self.data = data
        self._recon = None
        self._recon_key = None
        self._pec = None
        self._mat = None
        self._tri = None

    # ---- 再構成 ---------------------------------------------------------

    def recon(self, n: int, analysis_type: str):
        key = (self.data.solver_type, int(n), analysis_type)
        if self._recon_key == key:
            return self._recon
        mesh = self.data.mesh
        if self.data.is_hom:
            from ...fem_hom.field_recon import HOMFieldReconstructor
            recon = HOMFieldReconstructor(mesh["simplices"], mesh["vertices"], mesh["edge_index_map"],
                                          mesh["elem_order"], int(n), analysis_type)
        else:
            from ...fem_tm0.field_recon import TM0FieldReconstructor
            recon = TM0FieldReconstructor(mesh["vertices"], mesh["simplices"], mesh["elem_order"], analysis_type,
                                          eps_r_per_element=mesh.get("eps_r_per_element"))
        self._recon, self._recon_key = recon, key
        return recon

    def _mode(self, sel: Selection):
        sel = self.data.normalize(sel)
        modeset = self.data.modeset(sel.n, sel.phase)
        eigvec = np.asarray(modeset["eigenvectors"][sel.mode])
        freq = float(modeset["frequencies"][sel.mode])
        return sel, eigvec, freq, self.data.analysis_type(sel.n)

    def node_fields(self, sel: Selection):
        """全節点の場と (freq, recon, sel)."""
        sel, eigvec, freq, analysis_type = self._mode(sel)
        recon = self.recon(sel.n, analysis_type)
        if self.data.is_hom:
            recon.load_mode(eigvec, freq)
            node = recon.calculate_all_node_fields(theta_time=sel.time_phase)
        else:
            node = recon.calculate_all_node_fields(eigvec, freq, theta=np.deg2rad(sel.time_phase),
                                                   return_complex=(analysis_type == "traveling"))
        return node, freq, recon, sel

    def field_at(self, sel: Selection, z: float, r: float) -> Optional[dict]:
        """点 (z, r) [m] の場（領域外は None）. 値は実数."""
        sel, eigvec, freq, analysis_type = self._mode(sel)
        recon = self.recon(sel.n, analysis_type)
        if self.data.is_hom:
            recon.load_mode(eigvec, freq)
            res = recon.calculate_fields(z, r, theta_time=sel.time_phase)
        else:
            res = recon.calculate_fields(eigvec, freq, z, r, theta=np.deg2rad(sel.time_phase))
        if res is None:
            return None
        return {k: float(np.real(v)) for k, v in res.items()}

    def describe_field(self, sel: Selection, z: float, r: float) -> Optional[list[str]]:
        """ダブルクリックのポップアップ用の行."""
        res = self.field_at(sel, z, r)
        if res is None:
            return None
        keys = HOM_POPUP_KEYS if self.data.is_hom else TM0_POPUP_KEYS
        lines = [f"z = {z:.6f} m, r = {r:.6f} m"]
        lines += [f"{k} = {res[k]:.4e}" for k in keys if k in res]
        return lines

    # ---- 描画 -----------------------------------------------------------

    def _triangulations(self):
        if self._tri is None:
            mesh = self.data.mesh
            self._tri = (_triang_vis(mesh["vertices"], mesh["simplices"], mesh["elem_order"]),
                         _triang(mesh["vertices"], mesh["simplices"]))
        return self._tri

    def draw(self, fig, sel: Selection, opts: ViewOptions) -> tuple[float, Selection]:
        """``fig`` に 1 フレーム描き ``(freq, normalized selection)`` を返す."""
        fig.clf()
        node, freq, recon, sel = self.node_fields(sel)
        triang_vis, triang_raw = self._triangulations()
        data = self.data
        if data.is_hom:
            axl = fig.add_subplot(121)
            axr = fig.add_subplot(122)
            self._panel(axl, triang_vis, triang_raw, np.real(node["E_theta"]),
                        r"E-field (color $E_\theta$, vec $E_z,E_r$)", opts.show_mesh, opts.show_color)
            self._panel(axr, triang_vis, triang_raw, np.real(node["H_theta"]),
                        r"H-field (color $H_\theta$, vec $H_z,H_r$)", opts.show_mesh, opts.show_color)
            if opts.show_vectors:
                self._quiver_hom(axl, axr, recon, sel.time_phase, opts)
            self._draw_boundaries(axl)
            self._draw_boundaries(axr)
            fig.suptitle(data.title(sel))
        else:
            ax = fig.add_subplot(111)
            modeset = data.modeset(sel.n, sel.phase)
            eigvec = np.asarray(modeset["eigenvectors"][sel.mode])
            hmax = float(np.max(np.abs(eigvec))) or 1.0
            self._panel(ax, triang_vis, triang_raw, np.real(node["H_theta"]), "H_phi",
                        opts.show_mesh, opts.show_color, vmax=hmax)
            if opts.show_lines:
                psi = np.real(node["Psi"])
                levels = max(2, int(opts.levels))
                if data.analysis_type(sel.n) == "traveling":
                    psi_max = float(np.max(np.abs(np.asarray(data.mesh["vertices"])[:, 1] * eigvec)))
                    lv = np.linspace(-psi_max, psi_max, levels) if psi_max > 1e-20 else levels
                else:
                    lv = levels
                ax.tricontour(triang_vis, psi, levels=lv, colors="blueviolet", linewidths=1.5,
                              linestyles="solid", alpha=0.9)
            if opts.show_vectors:
                self._quiver_tm0(ax, recon, eigvec, freq, sel.time_phase, opts)
            self._draw_boundaries(ax)
            if opts.show_e_wall:
                self._draw_pec_e_vectors(ax, node)
            ax.set_title(data.title(sel))
        fig.tight_layout()
        return freq, sel

    def _panel(self, ax, triang_fill, triang_mesh, values, title, show_mesh, show_color, vmax=None):
        ax.set_aspect("equal")
        ax.set_xlabel("z [m]")
        ax.set_ylabel("r [m]")
        ax.set_title(title)
        if show_color:
            amax = vmax if vmax else (float(np.max(np.abs(values))) or 1.0)
            tcf = ax.tricontourf(triang_fill, values, levels=40, cmap=CMAP_BIPOLAR, vmin=-amax, vmax=amax)
            ax.get_figure().colorbar(tcf, ax=ax, shrink=0.8)
        if show_mesh:
            ax.triplot(triang_mesh, color="gray", lw=0.3, alpha=0.4)

    def pec_geometry(self):
        if self._pec is None:
            mesh = self.data.mesh
            self._pec = pec_boundary_geometry(mesh["simplices"], mesh["vertices"], mesh.get("physical_groups"),
                                              mesh["elem_order"])
        return self._pec

    def material_geometry(self):
        if self._mat is None:
            mesh = self.data.mesh
            self._mat = material_interface_geometry(mesh["simplices"], mesh["vertices"], mesh["elem_order"],
                                                    eps_r_per_element=mesh.get("eps_r_per_element"),
                                                    tan_delta_per_element=mesh.get("tan_delta_per_element"))
        return self._mat

    def _draw_boundaries(self, ax) -> None:
        """PEC 境界（黒の太線）と誘電体界面（マゼンタ破線）は常に描く."""
        segments, _ = self.pec_geometry()
        if segments:
            ax.add_collection(LineCollection(segments, colors="black", linewidths=1.2, zorder=10))
        segments, _ = self.material_geometry()
        if segments:
            ax.add_collection(LineCollection(segments, colors="magenta", linewidths=1.5, linestyles="--",
                                             zorder=9))

    def _draw_pec_e_vectors(self, ax, node) -> None:
        _, edges = self.pec_geometry()
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
            Z.append(e["coord"][0])
            R.append(e["coord"][1])
            U.append(u)
            V.append(v)
        ax.quiver(Z, R, U, V, color="red", alpha=0.9, pivot="tail", zorder=11)

    def _grid(self, opts: ViewOptions):
        nodes = np.asarray(self.data.mesh["vertices"])
        zg = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), max(2, int(opts.nz)))
        rg = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), max(2, int(opts.nr)))
        return zg, rg

    def _quiver_tm0(self, ax, recon, eigvec, freq, time_phase, opts) -> None:
        zg, rg = self._grid(opts)
        Z, R, U, V = [], [], [], []
        for z in zg:
            for r in rg:
                res = recon.calculate_fields(eigvec, freq, z, r, theta=np.deg2rad(time_phase))
                if res is None:
                    continue
                Z.append(z)
                R.append(r)
                U.append(np.real(res["Ez"]))
                V.append(np.real(res["Er"]))
        if Z:
            ax.quiver(Z, R, U, V, color="darkmagenta", alpha=0.8, pivot="middle")

    def _quiver_hom(self, ax_e, ax_h, recon, time_phase, opts) -> None:
        zg, rg = self._grid(opts)
        Z, R, Ez, Er, Hz, Hr = [], [], [], [], [], []
        for z in zg:
            for r in rg:
                res = recon.calculate_fields(z, max(r, 1e-10), theta_time=time_phase)
                if res is None:
                    continue
                Z.append(z)
                R.append(r)
                Ez.append(res["Ez"])
                Er.append(res["Er"])
                Hz.append(res["Hz"])
                Hr.append(res["Hr"])
        if Z:
            ax_e.quiver(Z, R, Ez, Er, color="darkmagenta", alpha=0.8, pivot="middle")
            ax_h.quiver(Z, R, Hz, Hr, color="darkmagenta", alpha=0.8, pivot="middle")


ProgressFn = Callable[[int, int], bool]   # (i, n) → False で中止


def render_gif(renderer: ResultRenderer, sel: Selection, opts: ViewOptions, path: str | Path,
               n_frames: int = 36, fps: int = 12, size_px: tuple[int, int] = (800, 600), dpi: int = 100,
               progress: Optional[ProgressFn] = None) -> bool:
    """時間位相 0→360° を周回する GIF をオフスクリーン（Agg）で作る. 中止されたら False（ファイルは作らない）."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    n_frames = max(2, int(n_frames))
    w, h = size_px
    fig = Figure(figsize=(max(w, 400) / dpi, max(h, 300) / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    frames = []
    for i, theta in enumerate(np.linspace(0.0, 360.0, n_frames, endpoint=False)):
        if progress is not None and not progress(i, n_frames):
            return False
        renderer.draw(fig, replace(sel, time_phase=float(theta)), opts)
        canvas.draw()
        cw, ch = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(Image.fromarray(buf.reshape(ch, cw, 4)).convert("RGB"))
    frames[0].save(str(path), save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / max(1, int(fps))), optimize=False)
    return True
