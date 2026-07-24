"""境界条件名 (PEC / E-short / M-short) の分類とエイリアス処理.

ver1 の ``Dirichlet`` は ``PEC`` のエイリアスとして警告付きで受け入れる。
FEM 行列上の数学的扱い (Dirichlet/Neumann) は呼び出し側 (fem_tm0/fem_hom) が
:class:`BoundaryClassification` から組み立てる。同じ物理境界でも DOF の種類により
扱いが反転する（詳細は docs/BC_NAMING.md）::

    BC 名     TM0 (DOF: H_phi)        HOM (DOF: E エッジ+節点)
    PEC       Neumann (自然境界)       Dirichlet (E_tan = 0)
    E-short   Neumann (自然境界)       Dirichlet (E_tan = 0)
    M-short   Dirichlet (H_phi = 0)    Neumann (自然境界)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# 正規の BC 名
PEC = "PEC"
E_SHORT = "E-short"
M_SHORT = "M-short"
CANONICAL_NAMES = (PEC, E_SHORT, M_SHORT)

# BC を付けない境界（対称軸 r=0・領域間の内部境界）。Physical Curve を持たない。
BC_NONE = "None"

# 後方互換エイリアス: {旧名: 正規名}
_ALIASES = {"Dirichlet": PEC}


@dataclass
class BoundaryClassification:
    """物理グループを正規 BC 名ごとの節点集合に整理した結果.

    FEM 行列上の Dirichlet/Neumann 振り分けは行わない（呼び出し側が DOF の種類に
    応じて :attr:`pec_nodes` 等から組み立てる）。

    Attributes:
        pec_nodes:     物理 PEC 壁の節点（``Dirichlet`` エイリアス含む）。
        eshort_nodes:  E-short（対称電気壁）の節点。
        mshort_nodes:  M-short（対称磁気壁）の節点。
        axis_nodes:    対称軸 r=0 上の節点。
        physical_groups: 入力の PhysicalGroup 辞書（参照用にそのまま保持）。
    """

    pec_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    eshort_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    mshort_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    axis_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    physical_groups: dict[str, np.ndarray] = field(default_factory=dict)


def find_axis_nodes(nodes: np.ndarray, r_threshold: float = 1e-6) -> np.ndarray:
    """対称軸 r=0 上の節点インデックスを返す.

    Args:
        nodes: 節点座標 (N, 2). 各行は ``[z, r]``。
        r_threshold: r 座標がこの値未満なら軸上とみなす。

    Returns:
        軸上節点インデックスの ``np.ndarray``。
    """
    return np.where(np.abs(nodes[:, 1]) < r_threshold)[0]


def classify_boundaries(
    physical_groups: dict[str, np.ndarray],
    nodes: np.ndarray,
    r_threshold: float = 1e-6,
    strict: bool = False,
) -> BoundaryClassification:
    """PhysicalGroup を正規 BC 名ごとに分類する.

    Args:
        physical_groups: ``{name: np.ndarray(node_indices)}``。
        nodes: 節点座標 (N, 2)。軸検出に使う。
        r_threshold: 軸 r=0 判定の閾値。
        strict: True のとき未知の境界グループ名でエラーを送出する。
            False のとき警告ログのみ（``Domain`` 等の体積グループは無視）。

    Returns:
        :class:`BoundaryClassification`。

    Raises:
        ValueError: ``strict=True`` かつ未知の境界グループ名があるとき。
    """
    buckets: dict[str, set[int]] = {name: set() for name in CANONICAL_NAMES}

    for raw_name, node_idx in physical_groups.items():
        canonical = _ALIASES.get(raw_name, raw_name)
        if raw_name in _ALIASES:
            logger.warning(
                "境界グループ名 %r は非推奨です。%r として扱います。",
                raw_name, canonical)
        if canonical in CANONICAL_NAMES:
            buckets[canonical].update(int(i) for i in node_idx)
        # Domain などの体積グループや 1D の補助線 (None 等) は境界 BC ではない
        # → strict 指定時のみ未知名として扱う

    if strict:
        known = set(CANONICAL_NAMES) | set(_ALIASES) | {"Domain"}
        unknown = [n for n in physical_groups if n not in known
                   and not n.startswith("PhysicalGroup_") and n != "None"]
        if unknown:
            raise ValueError(f"未知の境界グループ名: {unknown}")

    return BoundaryClassification(
        pec_nodes=np.array(sorted(buckets[PEC]), dtype=int),
        eshort_nodes=np.array(sorted(buckets[E_SHORT]), dtype=int),
        mshort_nodes=np.array(sorted(buckets[M_SHORT]), dtype=int),
        axis_nodes=find_axis_nodes(nodes, r_threshold),
        physical_groups=physical_groups,
    )


def _boundary_edges_with_mid(simplices, elem_order):
    """メッシュ外周辺（1 要素のみが共有する辺）と辺→中点ノード対応を返す.

    Returns:
        ``(boundary_keys, edge_mid)``。``boundary_keys`` は ``(a, b)``（a<b）のリスト、
        ``edge_mid`` は ``{(a, b): 中点ノード番号}``（2 次要素のみ）。
    """
    s = np.asarray(simplices)
    has_mid = elem_order == 2 and s.shape[1] >= 6
    edge_count: dict = {}
    edge_mid: dict = {}
    for t in s:
        c = (int(t[0]), int(t[1]), int(t[2]))
        pairs = ((c[0], c[1]), (c[1], c[2]), (c[2], c[0]))
        mids = ((int(t[3]), int(t[4]), int(t[5])) if has_mid
                else (None, None, None))
        for (a, b), m in zip(pairs, mids):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            if m is not None:
                edge_mid[key] = m
    boundary = [k for k, v in edge_count.items() if v == 1]
    return boundary, edge_mid


def _segments_from_edges(verts, edge_keys, edge_mid):
    """辺キー列から描画用 ``(segments, edges)`` を作る（2 次は中点経由のポリライン）."""
    segments = []
    edges = []
    for (a, b) in edge_keys:
        m = edge_mid.get((a, b))
        if m is not None:
            segments.append(np.array([verts[a], verts[m], verts[b]]))
            coord = (float(verts[m, 0]), float(verts[m, 1]))
        else:
            segments.append(np.array([verts[a], verts[b]]))
            coord = (float((verts[a, 0] + verts[b, 0]) / 2.0),
                     float((verts[a, 1] + verts[b, 1]) / 2.0))
        edges.append({"mid": m, "a": a, "b": b, "coord": coord})
    return segments, edges


def pec_boundary_geometry(simplices, vertices, physical_groups, elem_order):
    """PEC 境界の描画用幾何 ``(segments, edges)`` を返す（matplotlib 非依存）.

    GUI（Result Viewer）とレポート（plot_hom/plot_tm0）で共用するための純粋関数。

    Args:
        simplices: 要素節点インデックス (E, 3) または (E, 6)。
        vertices: 節点座標 (N, 2) = [z, r]。
        physical_groups: ``{name: node_indices}``（None/空可）。
        elem_order: 要素次数（2 のとき辺中点ノードを使う）。

    Returns:
        ``(segments, edges)``。
        - ``segments``: 黒太線用ポリラインのリスト（各要素は (P, 2) 座標配列。
          2 次要素は辺中点ノード経由で曲線境界に沿う）。
        - ``edges``: 電場ベクトル用記述 ``{"mid": 中点番号 or None, "a", "b": 端点,
          "coord": (z, r) 中点座標}`` のリスト。

    対象はメッシュ外周辺（1 要素のみが共有する辺）のうち両端が PEC ノードのもの。
    E-short（対称電気壁）は物理的 PEC 壁ではないため含めない。physical_groups が
    無い／PEC が無い場合は外周辺すべてにフォールバックする。
    """
    verts = np.asarray(vertices)
    boundary, edge_mid = _boundary_edges_with_mid(simplices, elem_order)

    wall: set = set()
    try:
        cls = classify_boundaries(physical_groups or {}, verts)
        wall = set(int(i) for i in cls.pec_nodes)
    except Exception:  # noqa: BLE001
        wall = set()
    if wall:
        edge_keys = [k for k in boundary if k[0] in wall and k[1] in wall]
        if not edge_keys:          # 分類が外周辺に一致しない場合は輪郭全体
            edge_keys = boundary
    else:
        edge_keys = boundary

    return _segments_from_edges(verts, edge_keys, edge_mid)


def material_interface_geometry(simplices, vertices, elem_order,
                                eps_r_per_element=None,
                                tan_delta_per_element=None,
                                rtol=1e-9):
    """材質（ε_r / tanδ）が変わる内部辺の描画用幾何 ``(segments, edges)`` を返す.

    隣接する 2 要素で eps_r または tan_delta が異なるコーナー辺を
    「誘電体界面」として抽出する（BC "None" の領域間共有境界のうち、
    材質が変わる線に一致する）。GUI（Result Viewer）とレポート
    （plot_tm0/plot_hom/mesh overview）で共用する純粋関数。

    Args:
        simplices: 要素節点インデックス (E, 3) または (E, 6)。
        vertices: 節点座標 (N, 2) = [z, r]。
        elem_order: 要素次数（2 のとき辺中点ノード経由の曲線ポリライン）。
        eps_r_per_element: (Ne,) 要素ごと比誘電率（None 可）。
        tan_delta_per_element: (Ne,) 要素ごと誘電正接（None 可）。
        rtol: 材質値の同一判定の相対許容誤差。

    Returns:
        ``(segments, edges)``（:func:`pec_boundary_geometry` と同形式）。
        材質情報が両方 None の場合は ``([], [])``。
    """
    vals = [np.asarray(a, dtype=float)
            for a in (eps_r_per_element, tan_delta_per_element)
            if a is not None]
    if not vals:
        return [], []

    s = np.asarray(simplices)
    has_mid = elem_order == 2 and s.shape[1] >= 6
    edge_elems: dict = {}
    edge_mid: dict = {}
    for idx, t in enumerate(s):
        c = (int(t[0]), int(t[1]), int(t[2]))
        pairs = ((c[0], c[1]), (c[1], c[2]), (c[2], c[0]))
        mids = ((int(t[3]), int(t[4]), int(t[5])) if has_mid
                else (None, None, None))
        for (a, b), m in zip(pairs, mids):
            key = (a, b) if a < b else (b, a)
            edge_elems.setdefault(key, []).append(idx)
            if m is not None:
                edge_mid[key] = m

    iface = []
    for key, elems in edge_elems.items():
        if len(elems) != 2:
            continue                    # 外周辺は界面ではない
        i, j = elems
        if any(not np.isclose(v[i], v[j], rtol=rtol, atol=0.0)
               for v in vals):
            iface.append(key)
    return _segments_from_edges(np.asarray(vertices), iface, edge_mid)


def mean_edge_length(simplices, vertices) -> float:
    """メッシュの平均辺長（= 平均メッシュサイズの目安）を返す.

    要素の 3 本の角節点辺を重複なく集め、その長さの平均を取る。2 次要素の
    中点ノードは無視する（幾何的な辺長は角節点で決まるため）。

    Args:
        simplices: 要素節点インデックス (E, 3) または (E, 6)。
        vertices: 節点座標 (N, 2) = [z, r]。

    Returns:
        平均辺長（座標と同じ単位）。辺が無ければ 0.0。
    """
    verts = np.asarray(vertices, dtype=float)
    s = np.asarray(simplices)
    keys: set = set()
    for t in s:
        c = (int(t[0]), int(t[1]), int(t[2]))
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            keys.add((a, b) if a < b else (b, a))
    if not keys:
        return 0.0
    idx = np.asarray(sorted(keys), dtype=int)
    d = verts[idx[:, 0]] - verts[idx[:, 1]]
    return float(np.hypot(d[:, 0], d[:, 1]).mean())


def bc_boundary_segments(simplices, vertices, physical_groups, elem_order):
    """境界条件ごとの境界辺セグメントを返す（メッシュ概要図の色分け用、matplotlib 非依存）.

    Args:
        simplices, vertices, physical_groups, elem_order: :func:`pec_boundary_geometry` と同じ。

    Returns:
        ``{"PEC": [...], "E-short": [...], "M-short": [...], "None": [...]}``。
        各値は (P, 2) 座標配列のリスト（2 次要素は辺中点ノード経由）。
        両端が当該 BC の節点である外周辺を採用する。``"None"`` は外周辺のうち
        いずれの BC にも属さないもの（対称軸 r=0 や BC 未指定の輪郭）。
        physical_groups が無い場合、外周辺はすべて ``"None"`` になる。
    """
    verts = np.asarray(vertices)
    boundary, edge_mid = _boundary_edges_with_mid(simplices, elem_order)
    try:
        cls = classify_boundaries(physical_groups or {}, verts)
    except Exception:  # noqa: BLE001
        segs, _ = _segments_from_edges(verts, boundary, edge_mid)
        return {PEC: [], E_SHORT: [], M_SHORT: [], BC_NONE: segs}

    node_sets = {
        PEC: set(int(i) for i in cls.pec_nodes),
        E_SHORT: set(int(i) for i in cls.eshort_nodes),
        M_SHORT: set(int(i) for i in cls.mshort_nodes),
    }
    out: dict = {}
    assigned: set = set()
    for name, nodes in node_sets.items():
        if nodes:
            keys = [k for k in boundary if k[0] in nodes and k[1] in nodes]
            assigned.update(keys)
            segs, _ = _segments_from_edges(verts, keys, edge_mid)
        else:
            segs = []
        out[name] = segs
    # どの BC にも属さない外周辺 = "None"（対称軸 r=0 など）
    none_keys = [k for k in boundary if k not in assigned]
    out[BC_NONE], _ = _segments_from_edges(verts, none_keys, edge_mid)
    return out
