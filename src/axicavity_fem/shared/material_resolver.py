"""メッシュ要素 → eps_r (μ_r) の解決。

`MeshData.element_region_ids` + `region_names` (material_tag) と、外部から渡される
材料テーブル (material_tag → eps_r / mu_r) を結合し、要素ごとの物性配列を返す。

材料テーブルの source は 2 種類:
    1. `MultiRegionGeometry` (プロジェクトファイル経由) → `build_material_table_from_geometry`
    2. HDF5 /materials/ グループ → 段階8 で実装

要素が region_id を持たない場合 (ver2 形式の単一領域メッシュ) は、デフォルト
eps_r=1.0 (真空) を返す。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .mesh_loader import MeshData
from .multi_region_model import MultiRegionGeometry


def build_material_table_from_geometry(
    geom: MultiRegionGeometry,
) -> dict[str, dict[str, float]]:
    """`MultiRegionGeometry` から material_tag → {eps_r, mu_r, tan_delta} のテーブルを作る。"""
    table: dict[str, dict[str, float]] = {}
    for region in geom.regions:
        table[region.material_tag] = {
            "eps_r": float(region.eps_r),
            "mu_r": float(region.mu_r),
            "tan_delta": float(region.tan_delta),
        }
    return table


def build_eps_r_per_element(
    mesh: MeshData,
    material_table: Mapping[str, Mapping[str, float]] | None = None,
    *,
    default_eps_r: float = 1.0,
) -> np.ndarray | None:
    """要素ごとの eps_r 配列 (Ne,) を返す。

    Args:
        mesh: メッシュ。
        material_table: ``{material_tag: {"eps_r": float, ...}}``。``None`` なら
            全要素 ``default_eps_r``。
        default_eps_r: テーブルに登録のない領域 (または region_id=-1) の eps_r。

    Returns:
        (Ne,) float 配列。``mesh.element_region_ids is None`` の場合は
        ``None`` を返す (= 一様 ``default_eps_r``、ソルバ側で None を見て
        既存パスに分岐できるようにするため)。
    """
    if mesh.element_region_ids is None or mesh.region_names is None:
        return None

    eps_r_per_region = _resolve_per_region(
        mesh.region_names, material_table, "eps_r", default_eps_r,
    )

    return _broadcast_to_elements(mesh.element_region_ids, eps_r_per_region,
                                  default_eps_r)


def build_mu_r_per_element(
    mesh: MeshData,
    material_table: Mapping[str, Mapping[str, float]] | None = None,
    *,
    default_mu_r: float = 1.0,
) -> np.ndarray | None:
    """要素ごとの mu_r 配列 (Ne,) を返す。HOM 拡張時に使う。"""
    if mesh.element_region_ids is None or mesh.region_names is None:
        return None

    mu_r_per_region = _resolve_per_region(
        mesh.region_names, material_table, "mu_r", default_mu_r,
    )
    return _broadcast_to_elements(mesh.element_region_ids, mu_r_per_region,
                                  default_mu_r)


def build_tan_delta_per_element(
    mesh: MeshData,
    material_table: Mapping[str, Mapping[str, float]] | None = None,
    *,
    default_tan_delta: float = 0.0,
) -> np.ndarray | None:
    """要素ごとの誘電正接 tanδ 配列 (Ne,) を返す。

    ``mesh.element_region_ids is None`` (ver2 形式の単一領域メッシュ) の場合は
    ``None`` を返す (= 無損失として扱う)。
    """
    if mesh.element_region_ids is None or mesh.region_names is None:
        return None

    tand_per_region = _resolve_per_region(
        mesh.region_names, material_table, "tan_delta", default_tan_delta,
    )
    return _broadcast_to_elements(mesh.element_region_ids, tand_per_region,
                                  default_tan_delta)


def _resolve_per_region(
    region_names: dict[int, str],
    material_table: Mapping[str, Mapping[str, float]] | None,
    key: str,
    default: float,
) -> dict[int, float]:
    """region_id → 物性値 のマップを作る。"""
    table = material_table or {}
    out: dict[int, float] = {}
    for rid, name in region_names.items():
        entry = table.get(name)
        if entry is not None and key in entry:
            out[int(rid)] = float(entry[key])
        else:
            out[int(rid)] = float(default)
    return out


def _broadcast_to_elements(
    element_region_ids: np.ndarray,
    value_per_region: dict[int, float],
    default: float,
) -> np.ndarray:
    out = np.full(len(element_region_ids), default, dtype=float)
    for i, rid in enumerate(element_region_ids):
        rid_i = int(rid)
        if rid_i in value_per_region:
            out[i] = value_per_region[rid_i]
    return out


# ---------------------------------------------------------------------------
# ver2.1 制約用 sanity check
# ---------------------------------------------------------------------------
def check_port_and_axis_are_vacuum(
    mesh: MeshData,
    eps_r_per_element: np.ndarray | None,
    *,
    z_port: float | None = None,
    tol: float = 1e-12,
) -> list[str]:
    """軸 r=0 上、および任意のポート z=z_port 上の要素が eps_r=1 であるかを確認する。

    後処理 (蓄積エネルギー、P_flow、軸上 Ez) は真空前提のため、ここに ε_r ≠ 1
    の要素があると派生量が不正確になる。リストで違反メッセージを返す
    (空リストなら OK)。呼び出し側は warning か例外として扱う。

    Args:
        mesh: メッシュ。
        eps_r_per_element: 要素ごと ε_r 配列 (``build_eps_r_per_element`` の戻り値)。
            ``None`` なら全要素 vacuum とみなして即座に空リストを返す。
        z_port: 確認したいポート z 位置 [m]。``None`` なら z_min を使う。
        tol: ε_r が 1.0 から外れる許容差。

    Returns:
        違反説明文のリスト。
    """
    if eps_r_per_element is None:
        return []
    messages: list[str] = []
    nodes = mesh.nodes
    elements = mesh.elements
    # 要素の重心 (z, r)
    tri = nodes[elements[:, :3]]
    centroids = tri.mean(axis=1)

    # 軸 r ≈ 0 を「コーナーいずれかが r=0 近傍」の要素として定義
    is_on_axis = np.any(tri[:, :, 1] < 1e-10, axis=1)
    bad_axis = np.where(is_on_axis & (np.abs(eps_r_per_element - 1.0) > tol))[0]
    if len(bad_axis) > 0:
        messages.append(
            f"axis r=0 上の {len(bad_axis)} 要素が eps_r != 1 (例: 要素 {bad_axis[:3].tolist()}). "
            "軸上 E_z 計算は真空前提のため、軸上加速電圧 V/V_eff/R/Q は不正確。"
        )

    # ポート断面 z ≈ z_port
    if z_port is None:
        z_port = float(np.min(nodes[:, 0]))
    is_on_port = np.any(np.abs(tri[:, :, 0] - z_port) < 1e-10, axis=1)
    bad_port = np.where(is_on_port & (np.abs(eps_r_per_element - 1.0) > tol))[0]
    if len(bad_port) > 0:
        messages.append(
            f"ポート断面 z={z_port} 上の {len(bad_port)} 要素が eps_r != 1 "
            f"(例: 要素 {bad_port[:3].tolist()}). P_flow / 群速度は不正確。"
        )

    return messages
