"""3D 表示の描画（pyvista。Qt 非依存。``pv.BasePlotter`` を受け取るので ``pv.Plotter(off_screen=True)`` でテストできる）.

回転体（:mod:`~axicavity_fem.gui.core.revolve`）を pyvista の UnstructuredGrid にし、次のアクターを出す:

- ``wall``: 空洞の壁（2D の境界辺を回転した四角形。半透明。場の色）
- ``meridian``: 子午面（φ = 0 と 180°、扇形なら φ = 0 と切り欠きの角の面。2D の図を 3D に置いたもの。不透明）
- ``slice``: 横断面 z = 一定（グリッドを VTK で切る。HOM の cos nφ の模様）
- ``arrows``: E または H の矢印。直交格子（子午面の z × r 格子・横断面の x × y 格子、または体積の x × y × z 格子）の
  点で 2D 振幅を補間して φ 依存を掛ける（メッシュの節点には置かない）。長さは大きさに比例
- ``elines``: TM0 の電気力線（Ψ の等高線の折れ線を φ 方向に並べる）
- スカラーバー 1 本（色にする量。範囲は振幅の最大で固定するのでアニメーションで揺れない）

アニメーションの 1 フレームは :meth:`Scene3D.set_fields` だけ（point_data の差し替えと断面・矢印・電気力線の作り直し）。
背景・照明・座標軸は EM-CAD-py の SceneManager と同じ。GIF は :func:`render_gif`（オフスクリーンの plotter に同じ Scene3D）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class _LazyPyvista:
    """``pyvista`` を最初に使うときに import する.

    pyvista は任意の依存（``[viz3d]``）なので、無くてもこのモジュールの定数と :class:`View3DOptions` は使えるようにする
    （メインウィンドウは起動時にこのモジュールを読み、3D 表示は ``view3d_window.is_available()`` で確かめてから開く）。
    import 文は静的に書く（Nuitka が pyvista を同梱するように）。
    """

    def __getattr__(self, name: str):
        import pyvista

        return getattr(pyvista, name)


pv = _LazyPyvista()

from ..core.revolve import (  # noqa: E402
    ModeFields,
    RevolvedGrid,
    disk_grid_points,
    e_line_polylines,
    field_point_arrays,
    plane_grid_points,
    point_fields,
    revolve_mesh,
    revolve_polylines,
    sample_amplitudes,
    split_triangles,
    volume_grid_points,
)

COLOR_FIELDS = ("|E|", "Ez", "Er", "Ephi", "|H|", "Hz", "Hr", "Hphi")
FIELD_CMAP = "jet"
SCALAR_BAR_ARGS = dict(vertical=True, position_x=0.86, position_y=0.1, width=0.06, height=0.55, n_labels=5,
                       fmt="%.2e", color="#1f2937", title_font_size=14, label_font_size=12)
ARROW_COLOR = "#4a1a6b"
# スカラーバーの題名（VTK の文字は "|" や "φ" を描けないので ASCII で）
SCALAR_BAR_TITLES = {"|E|": "E abs", "Ez": "E z", "Er": "E r", "Ephi": "E phi", "|H|": "H abs", "Hz": "H z",
                     "Hr": "H r", "Hphi": "H phi"}
E_LINE_COLOR = "#7b3fe4"
VIEWS = ("iso", "front", "top", "right")


@dataclass(frozen=True)
class View3DOptions:
    show_wall: bool = True
    wall_opacity: float = 0.35
    show_meridian: bool = True
    show_slice: bool = False
    slice_z: Optional[float] = None          # None = 中央
    sector_deg: float = 360.0                # 表示する回転角（< 360 で切り欠き）
    show_arrows: bool = False
    arrow_field: str = "E"                   # "E" / "H"
    arrow_mode: str = "planes"               # "planes"（表示している断面の格子）/ "volume"（体積の x × y × z 格子）
    arrow_nz: int = 30                       # z 方向の分割数（2D の nz と同じ既定）
    arrow_nxy: int = 30                      # x, y 方向の分割数（子午面では r 方向に半分）
    show_e_lines: bool = False               # TM0 だけ
    e_line_levels: int = 20
    e_line_count: int = 12                   # φ 方向の本数
    color_field: str = "|E|"                 # COLOR_FIELDS
    n_phi: int = 48
    show_scalar_bar: bool = True


def _even(n: int) -> int:
    n = max(4, int(n))
    return n if n % 2 == 0 else n + 1


def boundary_edges(triangles: np.ndarray, vertices: np.ndarray, axis_tol: float = 1e-9) -> np.ndarray:
    """三角形分割の境界辺 (B, 2)（1 つの三角形にしか属さない辺）。軸上（両端 r ≈ 0）の辺は除く."""
    t = np.asarray(triangles)
    edges = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    border = uniq[counts == 1]
    r = np.asarray(vertices)[:, 1]
    rmax = float(r.max()) or 1.0
    on_axis = (np.abs(r[border[:, 0]]) <= axis_tol * rmax) & (np.abs(r[border[:, 1]]) <= axis_tol * rmax)
    return border[~on_axis]


def amplitude_range(fields: ModeFields, name: str) -> tuple[float, float]:
    """色の範囲（アニメーションで変わらないように振幅から）: |E| / |H| は 0..max、成分は ±max."""
    if name in ("|E|", "|H|"):
        kind = name[1]
        total = np.sqrt(sum(np.abs(np.asarray(fields.component(f"{kind}{c}"))) ** 2 for c in ("z", "r", "phi")))
        top = float(total.max()) if len(total) else 1.0
        return 0.0, top or 1.0
    top = float(np.max(np.abs(np.asarray(fields.component(name))))) if fields.node_count else 1.0
    top = top or 1.0
    return -top, top


def sample_indices(count: int, wanted: int, seed: int = 0) -> np.ndarray:
    """count 個から wanted 個を等間隔に選ぶ."""
    if wanted <= 0 or count == 0:
        return np.zeros(0, dtype=np.int64)
    if wanted >= count:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, wanted).astype(np.int64))


class Scene3D:
    """回転体と場を plotter に描く. アクター名は固定（作り直しは remove → add）."""

    def __init__(self, plotter: pv.BasePlotter):
        self.plotter = plotter
        self.options = View3DOptions()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None
        self._boundary: Optional[np.ndarray] = None
        self._grid: Optional[RevolvedGrid] = None
        self._pv_grid: Optional[pv.UnstructuredGrid] = None
        self._grid_key: Optional[tuple] = None
        self._fields: Optional[ModeFields] = None
        self._time_phase = 0.0
        self._arrays: dict = {}
        self._clim: tuple[float, float] = (0.0, 1.0)
        self._scalar_bar_title: Optional[str] = None
        self._arrow_cache: Optional[tuple] = None        # (key, z, r, phi, spacing, amplitudes)
        self.actor_names: list[str] = []
        self._setup()

    # ---- 場面 -------------------------------------------------------------

    def _setup(self) -> None:
        p = self.plotter
        try:
            import vtk
            vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()      # 面の上の線を隠さない
        except Exception:  # noqa: BLE001
            pass
        p.set_background("#f4f6f8", top="#dfe6ee")
        p.remove_all_lights()
        p.add_light(pv.Light(light_type="headlight", intensity=0.55))
        p.add_light(pv.Light(position=(-1.0, 1.5, 2.0), light_type="cameralight", intensity=0.6))
        p.add_light(pv.Light(position=(2.0, -0.5, 1.0), light_type="cameralight", intensity=0.25))
        p.add_axes()
        p.view_isometric()

    def view(self, name: str) -> None:
        p = self.plotter
        if name == "front":
            p.view_xz(negative=True)
        elif name == "top":
            p.view_xy()
        elif name == "right":
            p.view_yz()
        else:
            p.view_isometric()
        p.reset_camera()

    def render_png(self, path: str | Path, transparent: bool = False) -> None:
        self.plotter.screenshot(str(path), transparent_background=transparent)

    # ---- 入力 -------------------------------------------------------------

    def set_mesh(self, vertices, simplices, elem_order: int, options: Optional[View3DOptions] = None) -> None:
        """2D メッシュを差し替える（場は set_fields で）."""
        self._vertices = np.asarray(vertices, dtype=float)
        self._triangles = split_triangles(simplices, elem_order)
        self._boundary = boundary_edges(self._triangles, self._vertices)
        self._grid_key = None
        self._fields = None
        self._arrays = {}
        if options is not None:
            self.options = options
        self._ensure_grid()
        self._rebuild()
        self.plotter.reset_camera()                     # 新しい形状は全体が入るように（向きはそのまま）

    def set_fields(self, fields: ModeFields, time_phase_deg: float = 0.0,
                   options: Optional[View3DOptions] = None) -> None:
        """場（と時間位相）を差し替えて描き直す. アニメーションの 1 フレーム."""
        if options is not None:
            self.options = options
        self._fields = fields
        self._time_phase = float(time_phase_deg)
        self._ensure_grid()
        self._update_arrays()
        self._rebuild()

    def set_options(self, options: View3DOptions) -> None:
        self.options = options
        self._ensure_grid()
        if self._fields is not None and self._pv_grid is not None:
            if self._arrays_valid():
                self._clim = amplitude_range(self._fields, options.color_field)
            else:
                self._update_arrays()
        self._rebuild()

    def set_time_phase(self, time_phase_deg: float) -> None:
        if self._fields is None:
            return
        self._time_phase = float(time_phase_deg)
        self._update_arrays()
        self._rebuild()

    def clear(self) -> None:
        self._remove_actors()
        self._vertices = self._triangles = self._boundary = None
        self._grid = self._pv_grid = None
        self._grid_key = None
        self._fields = None
        self._arrays = {}
        self._arrow_cache = None
        self.plotter.render()

    @property
    def has_mesh(self) -> bool:
        return self._pv_grid is not None

    @property
    def bounds_z(self) -> tuple[float, float]:
        if self._vertices is None:
            return 0.0, 0.0
        return float(self._vertices[:, 0].min()), float(self._vertices[:, 0].max())

    # ---- 内部 -------------------------------------------------------------

    def _ensure_grid(self) -> None:
        if self._vertices is None:
            return
        o = self.options
        key = (_even(o.n_phi), float(o.sector_deg), len(self._vertices), len(self._triangles))
        if key == self._grid_key:
            return
        self._grid = revolve_mesh(self._vertices, self._triangles, n_phi=_even(o.n_phi), phi_max_deg=o.sector_deg)
        self._pv_grid = pv.UnstructuredGrid(self._grid.cells(), self._grid.cell_types(), self._grid.points)
        self._grid_key = key
        self._arrays = {}
        if self._fields is not None:
            self._update_arrays()

    def _arrays_valid(self) -> bool:
        return bool(self._arrays) and self._pv_grid is not None and \
            len(self._arrays.get("|E|", ())) == self._pv_grid.n_points

    def _update_arrays(self) -> None:
        if self._fields is None or self._grid is None or self._pv_grid is None:
            return
        self._arrays = field_point_arrays(self._fields, self._grid, self._time_phase)
        for name, values in self._arrays.items():
            self._pv_grid.point_data[name] = values
        self._clim = amplitude_range(self._fields, self.options.color_field)

    def _remove_actors(self) -> None:
        for name in self.actor_names:
            try:
                self.plotter.remove_actor(name, render=False)
            except Exception:  # noqa: BLE001
                pass
        self.actor_names = []
        if self._scalar_bar_title is not None:
            try:
                self.plotter.remove_scalar_bar(self._scalar_bar_title, render=False)
            except Exception:  # noqa: BLE001
                pass
            self._scalar_bar_title = None

    def _add(self, name: str, mesh, **kwargs) -> None:
        # reset_camera=False: アクターを作り直すたび（アニメーションの矢印など）に視点が動かないように。
        # 視点のリセットは set_mesh（新しいメッシュ）と view() だけ
        self.plotter.add_mesh(mesh, name=name, render=False, reset_camera=False, **kwargs)
        self.actor_names.append(name)

    def _color_kwargs(self, show_bar: bool) -> dict:
        o = self.options
        if not self._arrays:
            return {"color": "#c7d2fe", "show_scalar_bar": False}
        return {"scalars": o.color_field, "cmap": FIELD_CMAP, "clim": list(self._clim), "show_scalar_bar": False}

    def _rebuild(self) -> None:
        self._remove_actors()
        if self._pv_grid is None or self._grid is None:
            self.plotter.render()
            return
        o = self.options
        colored = None
        if o.show_wall:
            wall = self._wall_surface()
            if wall is not None:
                self._add("wall", wall, opacity=o.wall_opacity, smooth_shading=False, lighting=True,
                          **self._color_kwargs(False))
                colored = colored or "wall"
        if o.show_meridian:
            faces = self._meridian_faces()
            if faces is not None:
                self._add("meridian", faces, lighting=False, **self._color_kwargs(False))
                colored = colored or "meridian"
        if o.show_slice:
            section = self._slice_z()
            if section is not None and section.n_points:
                self._add("slice", section, lighting=False, **self._color_kwargs(False))
                colored = colored or "slice"
        if o.show_arrows and self._arrays:
            arrows = self._arrows()
            if arrows is not None and arrows.n_points:
                self._add("arrows", arrows, color=ARROW_COLOR, lighting=True, show_scalar_bar=False)
        if o.show_e_lines and self._fields is not None and self._fields.psi is not None:
            lines = self._e_lines()
            if lines is not None and lines.n_points:
                self._add("elines", lines, color=E_LINE_COLOR, line_width=1.5, lighting=False,
                          show_scalar_bar=False)
        if o.show_scalar_bar and colored is not None and self._arrays:
            actor = self.plotter.renderer.actors.get(colored)
            if actor is not None:
                title = SCALAR_BAR_TITLES.get(o.color_field, o.color_field)
                try:
                    self.plotter.add_scalar_bar(title=title, mapper=actor.mapper, render=False, **SCALAR_BAR_ARGS)
                    self._scalar_bar_title = title
                except Exception:  # noqa: BLE001
                    self._scalar_bar_title = None
        self.plotter.render()

    def _with_data(self, poly: pv.PolyData, indices: np.ndarray) -> pv.PolyData:
        for name, values in self._arrays.items():
            poly.point_data[name] = values[indices]
        return poly

    def _wall_surface(self) -> Optional[pv.PolyData]:
        grid = self._grid
        if grid is None or self._boundary is None or not len(self._boundary):
            return None
        n = grid.node_count
        rings = grid.ring_count
        pairs = [(k, (k + 1) % rings) for k in range(rings)] if grid.full else [(k, k + 1) for k in range(rings - 1)]
        b = self._boundary
        quads = []
        for k0, k1 in pairs:
            q = np.column_stack([b[:, 0] + k0 * n, b[:, 1] + k0 * n, b[:, 1] + k1 * n, b[:, 0] + k1 * n])
            quads.append(q)
        faces = np.hstack([np.full((sum(len(q) for q in quads), 1), 4, dtype=np.int64), np.vstack(quads)]).ravel()
        poly = pv.PolyData(grid.points, faces)
        return self._with_data(poly, np.arange(grid.points.shape[0]))

    def _meridian_faces(self) -> Optional[pv.PolyData]:
        grid = self._grid
        if grid is None:
            return None
        n = grid.node_count
        rings = [0, grid.ring_count // 2] if grid.full else [0, grid.ring_count - 1]
        tris = self._triangles
        faces = []
        for k in rings:
            faces.append(tris + k * n)
        all_faces = np.vstack(faces)
        cells = np.hstack([np.full((len(all_faces), 1), 3, dtype=np.int64), all_faces]).ravel()
        poly = pv.PolyData(grid.points, cells)
        return self._with_data(poly, np.arange(grid.points.shape[0]))

    def _slice_z(self) -> Optional[pv.PolyData]:
        if self._pv_grid is None:
            return None
        zmin, zmax = self.bounds_z
        z = self._slice_z_value()
        z = min(max(z, zmin + 1e-9 * (zmax - zmin or 1.0)), zmax - 1e-9 * (zmax - zmin or 1.0))
        try:
            return self._pv_grid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, z))
        except Exception:  # noqa: BLE001
            return None

    def _slice_z_value(self) -> float:
        zmin, zmax = self.bounds_z
        z = self.options.slice_z if self.options.slice_z is not None else 0.5 * (zmin + zmax)
        return min(max(float(z), zmin), zmax)

    def _arrow_points(self) -> tuple:
        """矢印を置く直交格子の点 (z, r, phi, 間隔). 断面上: 表示している子午面（z × r）と横断面（x × y）、体積: x × y × z."""
        o, grid = self.options, self._grid
        zmin, zmax = self.bounds_z
        rmax = float(self._vertices[:, 1].max()) if self._vertices is not None else 1.0
        sector = 360.0 if grid is None or grid.full else grid.phi_max_deg
        if o.arrow_mode == "volume":
            return volume_grid_points(zmin, zmax, rmax, o.arrow_nz, o.arrow_nxy, sector)
        parts = []
        if o.show_meridian or not o.show_slice:
            angles = [0.0, 180.0] if sector >= 360.0 else [0.0, sector]
            parts.append(plane_grid_points(zmin, zmax, rmax, o.arrow_nz, max(2, o.arrow_nxy // 2), angles))
        if o.show_slice:
            parts.append(disk_grid_points(self._slice_z_value(), rmax, o.arrow_nxy, sector))
        z = np.concatenate([p[0] for p in parts])
        r = np.concatenate([p[1] for p in parts])
        phi = np.concatenate([p[2] for p in parts])
        return z, r, phi, max(p[3] for p in parts)

    def _arrows(self) -> Optional[pv.PolyData]:
        o, fields = self.options, self._fields
        if fields is None or self._vertices is None or self._grid is None:
            return None
        key = (self._grid_key, id(fields), o.arrow_mode, o.arrow_nz, o.arrow_nxy, o.show_meridian, o.show_slice,
               round(self._slice_z_value(), 12))
        if self._arrow_cache is None or self._arrow_cache[0] != key:
            z, r, phi, spacing = self._arrow_points()
            inside, amps = sample_amplitudes(fields, self._vertices, self._triangles, z, r)
            self._arrow_cache = (key, z[inside], r[inside], phi[inside], spacing, amps)
        _key, z, r, phi, spacing, amps = self._arrow_cache
        if not len(z):
            return None
        comp = point_fields(amps, phi, fields.n, fields.traveling, self._time_phase)
        kind = o.arrow_field if o.arrow_field in ("E", "H") else "E"
        vectors = comp[kind]
        mags = comp[f"|{kind}|"]
        vmax = float(mags.max()) if len(mags) else 0.0
        if vmax <= 0:
            return None
        # 長さ: matplotlib の quiver と同じく「典型的な大きさ」を格子の間隔に合わせる（最大値基準だと局所的に強い
        # 場のところ以外が見えなくなる）。90 パーセンタイルを基準に、最長は間隔の 1.6 倍まで
        reference = float(np.percentile(mags, 90)) or vmax
        idx = np.flatnonzero(mags > 0.02 * reference)
        if not len(idx):
            return None
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])[idx]
        cloud = pv.PolyData(pts)
        cloud.point_data["vec"] = vectors[idx] / mags[idx][:, None]
        cloud.point_data["len"] = 0.9 * spacing * np.minimum(mags[idx] / reference, 1.6)
        return cloud.glyph(orient="vec", scale="len", factor=1.0,
                           geom=pv.Arrow(start=(-0.5, 0, 0), direction=(1, 0, 0), tip_length=0.3, tip_radius=0.1,
                                         shaft_radius=0.035))

    def _e_lines(self) -> Optional[pv.PolyData]:
        o = self.options
        fields, grid = self._fields, self._grid
        if fields is None or fields.psi is None or grid is None or self._vertices is None:
            return None
        psi = np.asarray(fields.psi)
        if fields.traveling:
            psi = np.real(psi * np.exp(1j * np.deg2rad(self._time_phase)))
            top = float(np.max(np.abs(np.asarray(fields.psi))))
            levels = np.linspace(-top, top, max(2, int(o.e_line_levels))) if top > 1e-20 else int(o.e_line_levels)
        else:
            psi = np.real(psi)
            levels = max(2, int(o.e_line_levels))
        lines, _ = e_line_polylines(self._vertices, self._triangles, psi, levels=levels)
        if not lines:
            return None
        count = max(1, int(o.e_line_count))
        if grid.full:
            angles = np.linspace(0.0, 360.0, count, endpoint=False)
        else:
            angles = np.linspace(0.0, grid.phi_max_deg, count + 1)
        points, cells = revolve_polylines(lines, angles)
        if not len(points):
            return None
        return pv.PolyData(points, lines=cells)


ProgressFn = Callable[[int, int], bool]


def render_gif(vertices, simplices, elem_order: int, fields: ModeFields, options: View3DOptions, path: str | Path,
               n_frames: int = 36, fps: int = 12, size_px: tuple[int, int] = (800, 600), camera=None,
               progress: Optional[ProgressFn] = None) -> bool:
    """時間位相 0→360° を周回する GIF をオフスクリーンの plotter で作る（中止で False）."""
    from PIL import Image

    n_frames = max(2, int(n_frames))
    plotter = pv.Plotter(off_screen=True, window_size=list(size_px))
    try:
        scene = Scene3D(plotter)
        scene.set_mesh(vertices, simplices, elem_order, options)
        scene.set_fields(fields, 0.0, options)
        if camera is not None:
            plotter.camera_position = camera
        else:
            scene.view("iso")
        frames = []
        for i, theta in enumerate(np.linspace(0.0, 360.0, n_frames, endpoint=False)):
            if progress is not None and not progress(i, n_frames):
                return False
            scene.set_time_phase(float(theta))
            image = plotter.screenshot(return_img=True)
            frames.append(Image.fromarray(np.asarray(image)[:, :, :3]))
    finally:
        plotter.close()
    frames[0].save(str(path), save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / max(1, int(fps))), optimize=False)
    return True
