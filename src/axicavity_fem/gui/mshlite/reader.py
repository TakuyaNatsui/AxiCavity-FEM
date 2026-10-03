"""Gmsh MSH ファイルの純 Python リーダ（gmsh ライブラリ非依存。numpy だけを使う）.

Windows 版（EXE）は gmsh（GPL）を本体に入れない（メッシュ生成は別プロセスのメッシャ）。計算コアの
``shared/mesh_loader.py`` は gmsh API で ``.msh`` を読むので、EXE ではこのリーダの上に作った読込専用の
gmsh 互換モジュール（:mod:`.gmsh_api`）を ``gmsh`` として差し込む。対応形式:

    MSH 4.1  ASCII / binary（gmsh 4.x の既定。``$Entities`` で要素ブロック → Physical を解決）
    MSH 2.2  ASCII / binary（要素ごとに physical / elementary タグを持つ旧形式）

形式の仕様: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
（4.1 の binary では ``size_t`` が data-size（通常 8）バイト、``int`` は 4 バイト、``double`` は 8 バイト。
``$PhysicalNames`` は binary でも ASCII で書かれる）。EM-CAD-py（同じ作者）の ``cavity3d/msh_reader.py`` を移植し、
節点の所属エンティティと、エンティティの境界（``$Entities``）も読むようにした。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 要素タイプ → 節点数
NODES_PER_TYPE = {
    1: 2,    # 2 節点線分
    2: 3,    # 3 節点三角形
    3: 4,    # 4 節点四辺形
    4: 4,    # 4 節点四面体
    5: 8,    # 8 節点六面体
    6: 6,    # 6 節点プリズム
    7: 5,    # 5 節点ピラミッド
    8: 3,    # 3 節点線分（2 次）
    9: 6,    # 6 節点三角形（2 次）
    10: 9,   # 9 節点四辺形（2 次）
    11: 10,  # 10 節点四面体（2 次）
    12: 27,  # 27 節点六面体（2 次）
    13: 18,  # 18 節点プリズム（2 次）
    14: 14,  # 14 節点ピラミッド（2 次）
    15: 1,   # 1 節点点要素
    16: 8,   # 8 節点四辺形（2 次セレンディピティ）
    17: 20,  # 20 節点六面体（2 次セレンディピティ）
    18: 15,  # 15 節点プリズム（2 次セレンディピティ）
    19: 13,  # 13 節点ピラミッド（2 次セレンディピティ）
}

# 要素タイプ → 次元
TYPE_DIM = {1: 1, 8: 1, 2: 2, 3: 2, 9: 2, 10: 2, 16: 2,
            4: 3, 5: 3, 6: 3, 7: 3, 11: 3, 12: 3, 13: 3, 14: 3, 17: 3, 18: 3, 19: 3,
            15: 0}


@dataclass
class ElementBlock:
    """同じ (次元, エンティティ, 要素タイプ) の要素のまとまり.

    Attributes:
        dim: エンティティの次元（0〜3）。
        entity_tag: エンティティ（elementary）タグ。
        elem_type: gmsh 要素タイプ（2: tri3, 9: tri6, 1: line2, 8: line3 …）。
        elem_tags: 要素タグ (n,)。
        node_tags: 要素の節点タグ (n, k)。gmsh の節点順そのまま。
        physical_tags: このブロックが属する Physical グループのタグ。
    """

    dim: int
    entity_tag: int
    elem_type: int
    elem_tags: np.ndarray
    node_tags: np.ndarray
    physical_tags: list[int] = field(default_factory=list)


@dataclass
class NodeBlock:
    """``$Nodes`` の 1 ブロック（同じエンティティに属する節点）. ``start:stop`` は全節点配列での範囲."""

    dim: int
    entity_tag: int
    start: int
    stop: int


@dataclass
class MshData:
    """MSH ファイルの内容（gmsh の内部表現に近い形）.

    Attributes:
        version: "4.1" / "2.2" など。
        node_tags: 節点タグ (N,)。ファイル順（gmsh の ``getNodes()`` の順と同じ）。
        node_coords: 節点座標 (N, 3)。ファイルの数値そのまま（単位換算しない）。
        blocks: 要素ブロック（ファイル順）。
        physical_names: ``{(dim, physical_tag): 名前}``。
        entity_physicals: ``{(dim, entity_tag): [physical_tag, ...]}``
            （4.1 は ``$Entities``、2.2 は要素のタグから構成）。
        node_blocks: 節点の所属エンティティ（4.1 のみ。2.2 は空）。
        entity_bounds: ``{(dim, entity_tag): [境界エンティティ（dim-1）のタグ, ...]}``（4.1 の ``$Entities``）。
    """

    version: str
    node_tags: np.ndarray
    node_coords: np.ndarray
    blocks: list[ElementBlock]
    physical_names: dict[tuple[int, int], str]
    entity_physicals: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    node_blocks: list[NodeBlock] = field(default_factory=list)
    entity_bounds: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def physical_groups(self, dim: int) -> list[int]:
        """次元 dim の Physical タグをタグ順で返す（``gmsh.model.getPhysicalGroups`` 相当）.

        gmsh と同じくエンティティの所属から集める（名前だけあってエンティティの無い Physical グループは含めない）。
        """
        tags = set()
        for (d, _etag), phys in self.entity_physicals.items():
            if d == dim:
                tags.update(phys)
        return sorted(tags)

    def physical_name(self, dim: int, tag: int) -> str:
        """Physical 名（無ければ空文字。``gmsh.model.getPhysicalName`` 相当）."""
        return self.physical_names.get((dim, tag), "")

    def entities_in_physical(self, dim: int, physical_tag: int) -> list[int]:
        """Physical グループ (dim, tag) に属するエンティティのタグ（タグ順）."""
        return sorted(etag for (d, etag), phys in self.entity_physicals.items()
                      if d == dim and physical_tag in phys)

    def blocks_in_physical(self, dim: int, physical_tag: int) -> list[ElementBlock]:
        """Physical グループ (dim, tag) に属する要素ブロックをエンティティタグ順で返す."""
        found = [b for b in self.blocks if b.dim == dim and physical_tag in b.physical_tags]
        return sorted(found, key=lambda b: b.entity_tag)


def read_msh(filename: str | Path) -> MshData:
    """MSH ファイルを読む（4.1 ASCII/binary、2.2 ASCII/binary）."""
    data = Path(filename).read_bytes()
    header = _section(data, b"$MeshFormat", b"$EndMeshFormat")
    if header is None:
        raise ValueError(f"MSH ファイルではありません（$MeshFormat がない）: {filename}")
    first_line = header.split(b"\n", 1)[0].split()
    if len(first_line) < 3:
        raise ValueError(f"$MeshFormat が不正です: {first_line!r}")
    version = first_line[0].decode()
    binary = int(first_line[1]) == 1
    data_size = int(first_line[2])

    if version.startswith("4."):
        if version != "4.1":
            raise ValueError(f"MSH {version} は非対応です（4.1 / 2.2 に対応）。")
        return _read_v41(data, binary, data_size)
    if version in ("2.2", "2.1", "2.0"):
        return _read_v22(data, binary, data_size)
    raise ValueError(f"MSH {version} は非対応です（4.1 / 2.2 に対応）。")


# ---------------------------------------------------------------------------
# 共通: セクション切り出しと読み取りカーソル
# ---------------------------------------------------------------------------

def _section(data: bytes, start: bytes, end: bytes) -> bytes | None:
    """``$Name`` 行の次の行から ``$EndName`` 行の直前までを返す（無ければ None）."""
    i = data.find(start + b"\n")
    if i < 0:
        i = data.find(start + b"\r\n")
        if i < 0:
            return None
    i = data.index(b"\n", i) + 1
    j = data.find(b"\n" + end, i)
    if j < 0:
        raise ValueError(f"{end.decode()} が見つかりません。")
    return data[i:j]


class _AsciiCursor:
    """空白区切りトークン列を順に数値へ変換する."""

    def __init__(self, section: bytes):
        self._tok = section.split()
        self._pos = 0

    def ints(self, n: int) -> np.ndarray:
        out = np.array(self._tok[self._pos:self._pos + n], dtype=np.int64)
        if len(out) != n:
            raise ValueError("MSH ファイルが途中で終わっています。")
        self._pos += n
        return out

    size_ts = ints

    def doubles(self, n: int) -> np.ndarray:
        out = np.array(self._tok[self._pos:self._pos + n], dtype=np.float64)
        if len(out) != n:
            raise ValueError("MSH ファイルが途中で終わっています。")
        self._pos += n
        return out

    def int(self) -> int:
        return int(self.ints(1)[0])

    size_t = int


class _BinaryCursor:
    """バイナリ列を順に数値へ変換する（int: 4 バイト, size_t: data_size バイト, double: 8）."""

    def __init__(self, section: bytes, data_size: int):
        self._buf = section
        self._pos = 0
        self._size_t = np.dtype(f"<u{data_size}")

    def _take(self, dtype: np.dtype, n: int) -> np.ndarray:
        nbytes = dtype.itemsize * n
        chunk = self._buf[self._pos:self._pos + nbytes]
        if len(chunk) != nbytes:
            raise ValueError("MSH ファイルが途中で終わっています。")
        self._pos += nbytes
        return np.frombuffer(chunk, dtype=dtype)

    def ints(self, n: int) -> np.ndarray:
        return self._take(np.dtype("<i4"), n).astype(np.int64)

    def size_ts(self, n: int) -> np.ndarray:
        return self._take(self._size_t, n).astype(np.int64)

    def doubles(self, n: int) -> np.ndarray:
        return self._take(np.dtype("<f8"), n)

    def int(self) -> int:
        return int(self.ints(1)[0])

    def size_t(self) -> int:
        return int(self.size_ts(1)[0])


def _read_physical_names(data: bytes) -> dict[tuple[int, int], str]:
    """``$PhysicalNames``（binary でも ASCII）を読む."""
    sec = _section(data, b"$PhysicalNames", b"$EndPhysicalNames")
    names: dict[tuple[int, int], str] = {}
    if sec is None:
        return names
    lines = sec.decode("utf-8", errors="replace").splitlines()
    n = int(lines[0].split()[0])
    for line in lines[1:1 + n]:
        dim_s, tag_s, name = line.split(None, 2)
        names[(int(dim_s), int(tag_s))] = name.strip().strip('"')
    return names


# ---------------------------------------------------------------------------
# MSH 4.1
# ---------------------------------------------------------------------------

def _read_v41(data: bytes, binary: bool, data_size: int) -> MshData:
    def cursor(sec: bytes):
        return _BinaryCursor(sec, data_size) if binary else _AsciiCursor(sec)

    physical_names = _read_physical_names(data)
    entity_physicals, entity_bounds = _read_v41_entities(data, cursor)
    node_tags, node_coords, node_blocks = _read_v41_nodes(data, cursor)
    blocks = _read_v41_elements(data, cursor, entity_physicals)
    return MshData(version="4.1", node_tags=node_tags, node_coords=node_coords,
                   blocks=blocks, physical_names=physical_names,
                   entity_physicals=entity_physicals, node_blocks=node_blocks,
                   entity_bounds=entity_bounds)


def _read_v41_entities(data: bytes, cursor
                       ) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    """``$Entities`` → (``{(dim, tag): [physical, ...]}``, ``{(dim, tag): [境界のタグ, ...]}``)."""
    sec = _section(data, b"$Entities", b"$EndEntities")
    physicals: dict[tuple[int, int], list[int]] = {}
    bounds: dict[tuple[int, int], list[int]] = {}
    if sec is None:
        return physicals, bounds
    c = cursor(sec)
    counts = c.size_ts(4)
    for dim, count in enumerate(counts):
        for _ in range(int(count)):
            tag = c.int()
            c.doubles(3 if dim == 0 else 6)          # 座標 / バウンディングボックス
            n_phys = c.size_t()
            physicals[(dim, tag)] = [int(p) for p in c.ints(n_phys)]
            if dim > 0:
                n_bound = c.size_t()
                bounds[(dim, tag)] = [abs(int(b)) for b in c.ints(n_bound)]   # 符号は向き
    return physicals, bounds


def _read_v41_nodes(data: bytes, cursor) -> tuple[np.ndarray, np.ndarray, list[NodeBlock]]:
    sec = _section(data, b"$Nodes", b"$EndNodes")
    if sec is None:
        raise ValueError("$Nodes がありません。")
    c = cursor(sec)
    n_blocks, n_nodes, _min_tag, _max_tag = (int(v) for v in c.size_ts(4))
    tags = np.empty(n_nodes, dtype=np.int64)
    coords = np.empty((n_nodes, 3), dtype=np.float64)
    blocks: list[NodeBlock] = []
    pos = 0
    for _ in range(n_blocks):
        dim, etag, parametric = (int(v) for v in c.ints(3))
        n = c.size_t()
        tags[pos:pos + n] = c.size_ts(n)
        ncols = 3 + (dim if parametric else 0)       # parametric なら u(,v(,w)) が続く
        block = c.doubles(n * ncols).reshape(n, ncols)
        coords[pos:pos + n] = block[:, :3]
        blocks.append(NodeBlock(dim, etag, pos, pos + n))
        pos += n
    if pos != n_nodes:
        raise ValueError(f"$Nodes の節点数が一致しません（{pos} != {n_nodes}）。")
    return tags, coords, blocks


def _read_v41_elements(data: bytes, cursor,
                       entity_physicals: dict[tuple[int, int], list[int]]) -> list[ElementBlock]:
    sec = _section(data, b"$Elements", b"$EndElements")
    if sec is None:
        raise ValueError("$Elements がありません。")
    c = cursor(sec)
    n_blocks, _n_elems, _min_tag, _max_tag = (int(v) for v in c.size_ts(4))
    blocks: list[ElementBlock] = []
    for _ in range(n_blocks):
        dim, etag, etype = (int(v) for v in c.ints(3))
        n = c.size_t()
        if etype not in NODES_PER_TYPE:
            raise ValueError(f"非対応の要素タイプ {etype} が含まれています。")
        k = NODES_PER_TYPE[etype]
        table = c.size_ts(n * (1 + k)).reshape(n, 1 + k)
        blocks.append(ElementBlock(
            dim=dim, entity_tag=etag, elem_type=etype,
            elem_tags=table[:, 0].copy(), node_tags=table[:, 1:].copy(),
            physical_tags=list(entity_physicals.get((dim, etag), []))))
    return blocks


# ---------------------------------------------------------------------------
# MSH 2.2
# ---------------------------------------------------------------------------

def _read_v22(data: bytes, binary: bool, data_size: int) -> MshData:
    physical_names = _read_physical_names(data)
    nodes_sec = _section(data, b"$Nodes", b"$EndNodes")
    elems_sec = _section(data, b"$Elements", b"$EndElements")
    if nodes_sec is None or elems_sec is None:
        raise ValueError("$Nodes / $Elements がありません。")

    # 要素 → (type, physical, elementary) ごとに集めてブロック化する
    groups: dict[tuple[int, int, int], tuple[list, list]] = {}
    if binary:
        node_tags, node_coords = _read_v22_nodes_binary(nodes_sec)
        _read_v22_elements_binary(elems_sec, groups)
    else:
        c = _AsciiCursor(nodes_sec)
        n_nodes = c.size_t()
        table = c.doubles(n_nodes * 4).reshape(n_nodes, 4)
        node_tags = table[:, 0].astype(np.int64)
        node_coords = table[:, 1:].copy()
        _read_v22_elements_ascii(elems_sec, groups)

    blocks = [
        ElementBlock(dim=TYPE_DIM.get(etype, -1), entity_tag=elementary, elem_type=etype,
                     elem_tags=np.array(tags, dtype=np.int64),
                     node_tags=np.array(conn, dtype=np.int64).reshape(len(tags), -1),
                     physical_tags=[physical] if physical != 0 else [])
        for (etype, physical, elementary), (tags, conn) in groups.items()]
    entity_physicals: dict[tuple[int, int], list[int]] = {}
    for blk in blocks:
        phys = entity_physicals.setdefault((blk.dim, blk.entity_tag), [])
        for p in blk.physical_tags:
            if p not in phys:
                phys.append(p)
    node_tags, node_coords, node_blocks = _classify_v22_nodes(node_tags, node_coords, blocks)
    return MshData(version="2.2", node_tags=node_tags, node_coords=node_coords,
                   blocks=blocks, physical_names=physical_names,
                   entity_physicals=entity_physicals, node_blocks=node_blocks)


def _classify_v22_nodes(node_tags: np.ndarray, node_coords: np.ndarray, blocks: list[ElementBlock]
                        ) -> tuple[np.ndarray, np.ndarray, list[NodeBlock]]:
    """2.2 には節点の所属が無いので gmsh と同じ規則で決める（gmsh 4.15 で実測）.

    エンティティを (次元, タグ) の順に見て、要素が使う節点のうちまだ所属の無いものをそのエンティティに入れる
    （低い次元が優先）。エンティティ内はタグの昇順。全節点はこのエンティティ順に並べ直す（どの要素も使わない
    節点は最後にタグ順）。境界の情報は無いので ``includeBoundary`` は効かない（これも gmsh と同じ）。
    """
    owner: dict[int, tuple[int, int]] = {}
    per_entity: dict[tuple[int, int], list[int]] = {}
    for blk in sorted(blocks, key=lambda b: (b.dim, b.entity_tag)):
        key = (blk.dim, blk.entity_tag)
        claimed = per_entity.setdefault(key, [])
        for tag in blk.node_tags.reshape(-1):
            tag = int(tag)
            if tag not in owner:
                owner[tag] = key
                claimed.append(tag)
    index = {int(t): i for i, t in enumerate(node_tags)}
    order: list[int] = []
    node_blocks: list[NodeBlock] = []
    for key in sorted(per_entity):
        tags = sorted(per_entity[key])
        node_blocks.append(NodeBlock(key[0], key[1], len(order), len(order) + len(tags)))
        order += [index[t] for t in tags]
    rest = sorted((int(t) for t in node_tags if int(t) not in owner))
    order += [index[t] for t in rest]
    order_arr = np.asarray(order, dtype=np.int64)
    return node_tags[order_arr], node_coords[order_arr], node_blocks


def _read_v22_elements_ascii(sec: bytes, groups: dict) -> None:
    lines = sec.split(b"\n")
    n_elems = int(lines[0])
    for line in lines[1:1 + n_elems]:
        vals = line.split()
        if not vals:
            continue
        etag, etype, n_tags = int(vals[0]), int(vals[1]), int(vals[2])
        tags = [int(v) for v in vals[3:3 + n_tags]]
        physical = tags[0] if n_tags >= 1 else 0
        elementary = tags[1] if n_tags >= 2 else 0
        k = NODES_PER_TYPE.get(etype)
        if k is None:
            raise ValueError(f"非対応の要素タイプ {etype} が含まれています。")
        conn = [int(v) for v in vals[3 + n_tags:3 + n_tags + k]]
        if len(conn) != k:
            raise ValueError(f"要素 {etag} の節点数が不正です。")
        entry = groups.setdefault((etype, physical, elementary), ([], []))
        entry[0].append(etag)
        entry[1].extend(conn)


def _read_v22_nodes_binary(sec: bytes) -> tuple[np.ndarray, np.ndarray]:
    nl = sec.index(b"\n")
    n_nodes = int(sec[:nl])
    dtype = np.dtype([("tag", "<i4"), ("xyz", "<f8", (3,))])
    table = np.frombuffer(sec[nl + 1:nl + 1 + dtype.itemsize * n_nodes], dtype=dtype)
    if len(table) != n_nodes:
        raise ValueError("$Nodes（binary）が途中で終わっています。")
    return table["tag"].astype(np.int64), table["xyz"].astype(np.float64)


def _read_v22_elements_binary(sec: bytes, groups: dict) -> None:
    nl = sec.index(b"\n")
    n_elems = int(sec[:nl])
    pos = nl + 1
    remaining = n_elems
    while remaining > 0:
        etype, n_follow, n_tags = struct.unpack_from("<3i", sec, pos)
        pos += 12
        k = NODES_PER_TYPE.get(etype)
        if k is None:
            raise ValueError(f"非対応の要素タイプ {etype} が含まれています。")
        width = 1 + n_tags + k
        table = np.frombuffer(sec, dtype="<i4", count=n_follow * width,
                              offset=pos).reshape(n_follow, width).astype(np.int64)
        pos += 4 * n_follow * width
        for row in table:
            physical = int(row[1]) if n_tags >= 1 else 0
            elementary = int(row[2]) if n_tags >= 2 else 0
            entry = groups.setdefault((etype, physical, elementary), ([], []))
            entry[0].append(int(row[0]))
            entry[1].extend(int(v) for v in row[1 + n_tags:])
        remaining -= n_follow
