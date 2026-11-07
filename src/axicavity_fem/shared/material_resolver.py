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

import warnings
from typing import Mapping

import numpy as np

from .boundary_groups import CANONICAL_NAMES, canonical_bc_name
from .mesh_loader import MeshData
from .multi_region_model import VACUUM_TAG, MultiRegionGeometry


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
        if (entry is None and table and key == "eps_r" and name not in (VACUUM_TAG, "Domain")
                and not name.startswith("PhysicalGroup_") and canonical_bc_name(name) not in CANONICAL_NAMES):
            warnings.warn(f"[Material] 材料タグ '{name}' が材料の表にありません。真空として計算します。", UserWarning, stacklevel=2)
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
