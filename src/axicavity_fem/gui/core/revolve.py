"""2D 軸対称の場を z 軸まわりに回転して 3D にする（numpy と matplotlib.tri だけ。Qt / VTK 非依存）.

- :func:`split_triangles`: 2 次要素（6 節点）を 4 つの線形三角形に（2D の図と同じ分割）
- :func:`revolve_mesh`: 三角形分割を φ ∈ [0, phi_max] に n_phi 分割で回転した点と wedge（VTK_WEDGE = 13）の接続
- :class:`ModeFields` / :func:`field_point_arrays`: 2D 節点振幅から回転体の全点の E / H（ベクトルと成分）
- :func:`e_line_polylines` / :func:`revolve_polylines`: TM0 の電気力線（Ψ = r·H_φ の等高線）を折れ線にして回転
- :func:`mode_fields_from_renderer`: :class:`~axicavity_fem.gui.ui.result_renderer.ResultRenderer` から ModeFields

φ 依存（規約 e^{+jnφ}、時間は e^{+jθ}。PHYSICS_AND_CONVENTIONS.md §1 と fem_hom/field_recon.py の curl の式から）:
コアの 2D 振幅 A について、cos 型の成分（E_z, E_r, H_φ）は Re[A e^{jnφ} e^{jθ}]、sin 型の成分（E_φ, H_z, H_r）は
Re[j A e^{jnφ} e^{jθ}]。コアの E_θ・H_z・H_r は物理量の 1/j 倍で持たれている（実数の要素行列のため）。
定在波は A が実数で θ を無視（2D の図と同じ: E は t = 0、H は −j を掛けた最大振幅）。進行波の A は
f(θ=0°) − j·f(θ=90°) で復元する。符号は tests/test_revolve.py がガウスの法則（∇·H = 0）で確かめる。
座標は VTK の (x, y, z) = (r cos φ, r sin φ, z)（空洞の軸 = VTK の z）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.tri as mtri
import numpy as np
from matplotlib.figure import Figure

VTK_WEDGE = 13
COS_TYPE = ("Ez", "Er", "Hphi")
SIN_TYPE = ("Ephi", "Hz", "Hr")


# ---------------------------------------------------------------------------
# 三角形分割と回転体
# ---------------------------------------------------------------------------

def split_triangles(simplices, elem_order: int) -> np.ndarray:
    """描画用の線形三角形 (M, 3)。2 次要素は中点で 4 分割（result_renderer._triang_vis と同じ）."""
    s = np.asarray(simplices)
    if elem_order == 2 and s.shape[1] >= 6:
        return np.vstack([s[:, [0, 3, 5]], s[:, [3, 1, 4]], s[:, [5, 4, 2]], s[:, [3, 4, 5]]]).astype(np.int64)
    return s[:, :3].astype(np.int64)


@dataclass
class RevolvedGrid:
    """回転体: ``points`` (P, 3)、``wedges`` (W, 6)（VTK_WEDGE の点番号）、リング（φ の刻み）の角度."""

    points: np.ndarray
    wedges: np.ndarray
    ring_angles: np.ndarray      # [rad]、リングごと
    node_count: int              # 2D の節点数（1 リングの点数）
    full: bool                   # 全周（始端と終端のリングを共有）
    n_phi: int
    phi_max_deg: float

    @property
    def ring_count(self) -> int:
        return len(self.ring_angles)

    def cells(self) -> np.ndarray:
        """VTK の cells 配列（[6, i0..i5, 6, ...]）."""
        w = self.wedges
        return np.hstack([np.full((len(w), 1), 6, dtype=np.int64), w]).ravel()

    def cell_types(self) -> np.ndarray:
        return np.full(len(self.wedges), VTK_WEDGE, dtype=np.uint8)


def ring_angles(n_phi: int, phi_max_deg: float) -> tuple[np.ndarray, bool]:
    """φ の刻み [rad] と全周かどうか. 全周は終端 = 始端なので n_phi 個、扇形は n_phi + 1 個."""
    n_phi = max(3, int(n_phi))
    phi_max = float(phi_max_deg)
    full = phi_max >= 360.0 - 1e-9
    if full:
        return np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False), True
    return np.linspace(0.0, np.deg2rad(phi_max), n_phi + 1), False


def revolve_mesh(vertices, triangles, n_phi: int = 48, phi_max_deg: float = 360.0) -> RevolvedGrid:
    """2D 節点 (z, r) (N, 2) と線形三角形 (M, 3) を回転させた wedge グリッド.

    点番号は ``ring * N + node``。軸上の点（r = 0）は縮退したまま（VTK は縮退 wedge を描ける）。
    """
    v = np.asarray(vertices, dtype=float)
    tris = np.asarray(triangles, dtype=np.int64)
    angles, full = ring_angles(n_phi, phi_max_deg)
    n = len(v)
    z, r = v[:, 0], v[:, 1]
    cos, sin = np.cos(angles), np.sin(angles)
    points = np.empty((len(angles) * n, 3))
    for k in range(len(angles)):
        points[k * n:(k + 1) * n, 0] = r * cos[k]
        points[k * n:(k + 1) * n, 1] = r * sin[k]
        points[k * n:(k + 1) * n, 2] = z
    rings = len(angles)
    pairs = [(k, (k + 1) % rings) for k in range(rings)] if full else [(k, k + 1) for k in range(rings - 1)]
    wedges = np.empty((len(pairs) * len(tris), 6), dtype=np.int64)
    for i, (k0, k1) in enumerate(pairs):
        block = wedges[i * len(tris):(i + 1) * len(tris)]
        block[:, :3] = tris + k0 * n
        block[:, 3:] = tris + k1 * n
    return RevolvedGrid(points=points, wedges=wedges, ring_angles=angles, node_count=n, full=full,
                        n_phi=max(3, int(n_phi)), phi_max_deg=float(phi_max_deg))


# ---------------------------------------------------------------------------
# 場
# ---------------------------------------------------------------------------

@dataclass
class ModeFields:
    """2D 節点の振幅（コアの表現。複素または実数）. TM0 は Ephi = Hz = Hr = 0."""

    n: int
    traveling: bool
    Ez: np.ndarray
    Er: np.ndarray
    Ephi: np.ndarray
    Hz: np.ndarray
    Hr: np.ndarray
    Hphi: np.ndarray
    psi: Optional[np.ndarray] = None           # TM0 の Ψ = r·H_φ（電気力線用）
    is_hom: bool = False
    freq_ghz: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.Ez)

    def component(self, name: str) -> np.ndarray:
        return getattr(self, name)


def _phase_factors(fields: ModeFields, angles: np.ndarray, time_phase_deg: float) -> np.ndarray:
    theta = np.deg2rad(time_phase_deg) if fields.traveling else 0.0
    return np.exp(1j * (fields.n * angles + theta))          # リングごとの e^{j(nφ + θ)}


def revolved_component(fields: ModeFields, name: str, angles: np.ndarray, time_phase_deg: float = 0.0) -> np.ndarray:
    """成分 ``name`` の回転体の全点の値 (rings * N,)（点番号 = ring * N + node）."""
    a = np.asarray(fields.component(name))
    factors = _phase_factors(fields, angles, time_phase_deg)
    if name in SIN_TYPE:
        factors = 1j * factors
    return np.real(np.outer(factors, a)).ravel()


def field_point_arrays(fields: ModeFields, grid: RevolvedGrid, time_phase_deg: float = 0.0) -> dict:
    """回転体の全点の場: ``E`` / ``H`` (P, 3)、``|E|`` / ``|H|`` と成分 ``Ez Er Ephi Hz Hr Hphi`` (P,)."""
    angles = grid.ring_angles
    out = {name: revolved_component(fields, name, angles, time_phase_deg)
           for name in ("Ez", "Er", "Ephi", "Hz", "Hr", "Hphi")}
    n = grid.node_count
    cos = np.repeat(np.cos(angles), n)
    sin = np.repeat(np.sin(angles), n)
    for kind in ("E", "H"):
        z, r, phi = out[f"{kind}z"], out[f"{kind}r"], out[f"{kind}phi"]
        vec = np.column_stack([r * cos - phi * sin, r * sin + phi * cos, z])
        out[kind] = vec
        out[f"|{kind}|"] = np.linalg.norm(vec, axis=1)
    return out


SCALAR_FIELDS = ("|E|", "Ez", "Er", "Ephi", "|H|", "Hz", "Hr", "Hphi")


# ---------------------------------------------------------------------------
# 電気力線（TM0）
# ---------------------------------------------------------------------------

def e_line_polylines(vertices, triangles, psi, levels=20) -> tuple[list[np.ndarray], np.ndarray]:
    """Ψ の等高線の折れ線 [(K, 2) の (z, r)] と使ったレベル（2D の図と同じ matplotlib の等高線）."""
    v = np.asarray(vertices, dtype=float)
    tri = mtri.Triangulation(v[:, 0], v[:, 1], np.asarray(triangles, dtype=np.int64))
    values = np.real(np.asarray(psi))
    fig = Figure()
    ax = fig.add_subplot(111)
    cs = ax.tricontour(tri, values, levels=levels)
    lines: list[np.ndarray] = []
    for path in cs.get_paths():
        verts = np.asarray(path.vertices)
        codes = path.codes
        if codes is None:
            if len(verts) >= 2:
                lines.append(verts)
            continue
        start = 0
        for i in range(1, len(codes) + 1):
            if i == len(codes) or codes[i] == 1:          # MOVETO で折れ線が切れる
                if i - start >= 2:
                    lines.append(verts[start:i])
                start = i
    return lines, np.asarray(cs.levels, dtype=float)


def revolve_polylines(lines: list[np.ndarray], angles_deg) -> tuple[np.ndarray, np.ndarray]:
    """折れ線 (K, 2) の列を各角度に置く. 戻り値は VTK の (points (M, 3), lines [K, i0, i1, ...])."""
    points: list[np.ndarray] = []
    cells: list[int] = []
    offset = 0
    for phi in np.deg2rad(np.asarray(angles_deg, dtype=float)):
        c, s = np.cos(phi), np.sin(phi)
        for line in lines:
            k = len(line)
            if k < 2:
                continue
            z, r = line[:, 0], line[:, 1]
            points.append(np.column_stack([r * c, r * s, z]))
            cells.append(k)
            cells.extend(range(offset, offset + k))
            offset += k
    if not points:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.vstack(points), np.asarray(cells, dtype=np.int64)


# ---------------------------------------------------------------------------
# ResultRenderer から
# ---------------------------------------------------------------------------

def mode_fields_from_renderer(renderer, sel) -> ModeFields:
    """表示中のモードの 2D 振幅（``ResultRenderer`` のキャッシュ済み再構成器を使う）."""
    data = renderer.data
    sel = data.normalize(sel)
    modeset = data.modeset(sel.n, sel.phase)
    eigvec = np.asarray(modeset["eigenvectors"][sel.mode])
    freq = float(modeset["frequencies"][sel.mode])
    analysis_type = data.analysis_type(sel.n)
    traveling = analysis_type == "traveling"
    recon = renderer.recon(sel.n, analysis_type)
    count = len(np.asarray(data.mesh["vertices"]))
    zeros = np.zeros(count)
    if data.is_hom:
        recon.load_mode(eigvec, freq)
        f0 = recon.calculate_all_node_fields(theta_time=0.0)
        if traveling:
            f90 = recon.calculate_all_node_fields(theta_time=90.0)
            amp = {k: f0[k] - 1j * f90[k] for k in f0}
        else:
            amp = {k: np.asarray(f0[k], dtype=float) for k in f0}
        return ModeFields(n=sel.n, traveling=traveling, Ez=amp["Ez"], Er=amp["Er"], Ephi=amp["E_theta"],
                          Hz=amp["Hz"], Hr=amp["Hr"], Hphi=amp["H_theta"], is_hom=True, freq_ghz=freq)
    node = recon.calculate_all_node_fields(eigvec, freq, theta=0.0, return_complex=traveling)
    if traveling:
        ez, er, hphi, psi = (np.asarray(node[k], dtype=complex) for k in ("Ez", "Er", "H_theta", "Psi"))
    else:
        ez, er, hphi, psi = (np.real(np.asarray(node[k])).astype(float) for k in ("Ez", "Er", "H_theta", "Psi"))
    return ModeFields(n=0, traveling=traveling, Ez=ez, Er=er, Ephi=zeros, Hz=zeros, Hr=zeros, Hphi=hphi, psi=psi,
                      is_hom=False, freq_ghz=freq)


# ---------------------------------------------------------------------------
# 任意の点での場（矢印を直交格子に置くため）
# ---------------------------------------------------------------------------

def sample_amplitudes(fields: ModeFields, vertices, triangles, z, r) -> tuple[np.ndarray, dict]:
    """点 (z, r) の 2D 振幅を三角形上で線形補間する. 戻り値は (領域内のマスク, {成分: 振幅（領域内の点だけ）})."""
    v = np.asarray(vertices, dtype=float)
    tri = mtri.Triangulation(v[:, 0], v[:, 1], np.asarray(triangles, dtype=np.int64))
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    out: dict = {}
    for name in ("Ez", "Er", "Ephi", "Hz", "Hr", "Hphi"):
        a = np.asarray(fields.component(name))
        value = np.ma.filled(mtri.LinearTriInterpolator(tri, np.real(a))(z, r), np.nan).astype(complex)
        if np.iscomplexobj(a):
            value = value + 1j * np.ma.filled(mtri.LinearTriInterpolator(tri, np.imag(a))(z, r), np.nan)
        out[name] = value
    inside = np.isfinite(out["Ez"].real)
    return inside, {k: v[inside] for k, v in out.items()}


def point_fields(amplitudes: dict, phi, n: int, traveling: bool, time_phase_deg: float = 0.0) -> dict:
    """振幅（sample_amplitudes の出力）と各点の φ から E / H ベクトル (N, 3) と成分."""
    phi = np.asarray(phi, dtype=float)
    theta = np.deg2rad(time_phase_deg) if traveling else 0.0
    factor = np.exp(1j * (n * phi + theta))
    comp = {name: np.real(amplitudes[name] * factor) for name in COS_TYPE}
    comp.update({name: np.real(1j * amplitudes[name] * factor) for name in SIN_TYPE})
    cos, sin = np.cos(phi), np.sin(phi)
    for kind in ("E", "H"):
        zc, rc, pc = comp[f"{kind}z"], comp[f"{kind}r"], comp[f"{kind}phi"]
        comp[kind] = np.column_stack([rc * cos - pc * sin, rc * sin + pc * cos, zc])
        comp[f"|{kind}|"] = np.linalg.norm(comp[kind], axis=1)
    return comp


def plane_grid_points(zmin: float, zmax: float, rmax: float, n_z: int, n_r: int, angles_deg) -> tuple:
    """子午面（角度 angles_deg）の上の z × r 格子. 戻り値 (z, r, phi, 間隔 = 矢印の長さの基準（大きい方））."""
    zs = np.linspace(zmin, zmax, max(2, int(n_z)))
    rs = np.linspace(0.0, rmax, max(2, int(n_r)) + 1)[1:]          # 軸上は除く（φ が決まらない）
    zz, rr = np.meshgrid(zs, rs, indexing="ij")
    parts = [(zz.ravel(), rr.ravel(), np.full(zz.size, np.deg2rad(a))) for a in np.atleast_1d(angles_deg)]
    z = np.concatenate([p[0] for p in parts])
    r = np.concatenate([p[1] for p in parts])
    phi = np.concatenate([p[2] for p in parts])
    spacing = max(zs[1] - zs[0] if len(zs) > 1 else rmax, rs[1] - rs[0] if len(rs) > 1 else rmax)   # 矢印の長さの基準
    return z, r, phi, float(spacing)


def disk_grid_points(z0: float, rmax: float, n_xy: int, phi_max_deg: float = 360.0) -> tuple:
    """横断面 z = z0 の上の x × y 格子（半径 rmax の円の中、回転角の範囲）. 戻り値 (z, r, phi, 間隔)."""
    xs = np.linspace(-rmax, rmax, max(2, int(n_xy)))
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    r = np.hypot(xx.ravel(), yy.ravel())
    phi = np.arctan2(yy.ravel(), xx.ravel()) % (2.0 * np.pi)
    keep = (r <= rmax) & (r > 1e-12 * max(rmax, 1e-30)) & (phi <= np.deg2rad(phi_max_deg) + 1e-9)
    spacing = xs[1] - xs[0] if len(xs) > 1 else rmax
    return np.full(int(keep.sum()), float(z0)), r[keep], phi[keep], float(spacing)


def volume_grid_points(zmin: float, zmax: float, rmax: float, n_z: int, n_xy: int,
                       phi_max_deg: float = 360.0) -> tuple:
    """体積の x × y × z 格子（半径 rmax の円筒の中、回転角の範囲）. 戻り値 (z, r, phi, 間隔)."""
    zs = np.linspace(zmin, zmax, max(2, int(n_z)))
    xs = np.linspace(-rmax, rmax, max(2, int(n_xy)))
    xx, yy, zz = np.meshgrid(xs, xs, zs, indexing="ij")
    r = np.hypot(xx.ravel(), yy.ravel())
    phi = np.arctan2(yy.ravel(), xx.ravel()) % (2.0 * np.pi)
    keep = (r <= rmax) & (r > 1e-12 * max(rmax, 1e-30)) & (phi <= np.deg2rad(phi_max_deg) + 1e-9)
    spacing = max(zs[1] - zs[0] if len(zs) > 1 else rmax, xs[1] - xs[0] if len(xs) > 1 else rmax)   # 矢印の長さの基準
    return zz.ravel()[keep], r[keep], phi[keep], float(spacing)
