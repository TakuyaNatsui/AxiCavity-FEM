"""gmsh の Python API のうち ``.msh`` を読む部分だけを、:mod:`.reader` の上に作った互換モジュール（読込専用）.

Windows 版（EXE）では本体に gmsh（GPL）を入れないので、:func:`axicavity_fem.gui.mshlite.install_as_gmsh` が
このモジュールを ``sys.modules["gmsh"]`` に置く。計算コアの ``shared/mesh_loader.py``（``import gmsh`` で
``getNodes`` / ``getElementsByType`` / ``getPhysicalGroups`` などを呼ぶ）はそのまま動き、節点の並び（= 全体行列の
番号）も gmsh で読んだときと同じになる（``tests/test_mshlite.py`` で gmsh と突き合わせて確認）。

対応する関数（コアが使うもの）::

    initialize / finalize / isInitialized / open
    option.setNumber / option.getNumber
    model.getPhysicalGroups / model.getPhysicalName / model.getEntitiesForPhysicalGroup
    model.mesh.getNodes / model.mesh.getElementsByType

メッシュ生成（``model.occ`` / ``model.mesh.generate`` / ``write`` など）は別プロセスのメッシャで行う
（:mod:`axicavity_fem.gui.jobs.meshing`）。ここで呼ぶと :class:`NotImplementedError`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .reader import TYPE_DIM, MshData, read_msh

__version__ = "mshlite"
GMSH_API_VERSION = "mshlite"
_UNSUPPORTED = "gmsh.{name} はこの版では使えません（.msh の読込専用。メッシュ生成は同梱のメッシャで行います）"


class _State:
    initialized = False
    data: MshData | None = None
    options: dict[str, float] = {}
    _sorted_tags: np.ndarray | None = None
    _sort_order: np.ndarray | None = None


_state = _State()


def _unsupported(name: str):
    if name.rsplit(".", 1)[-1].startswith("__"):        # __path__ / __spec__ などの問い合わせは普通の AttributeError
        raise AttributeError(name)
    raise NotImplementedError(_UNSUPPORTED.format(name=name))


def __getattr__(name: str):                       # write / merge / clear / logger など
    _unsupported(name)


def initialize(argv=None, readConfigFiles=True, run=False, interruptible=True) -> None:   # noqa: N803 — gmsh の引数名
    _state.initialized = True
    _state.data = None
    _state.options = {}


def finalize() -> None:
    _state.initialized = False
    _state.data = None
    _state._sorted_tags = _state._sort_order = None


def isInitialized() -> int:                         # noqa: N802 — gmsh の関数名
    return 1 if _state.initialized else 0


def open(fileName) -> None:                         # noqa: A001, N803
    if not _state.initialized:
        raise RuntimeError("gmsh が初期化されていません（initialize を先に）")
    path = Path(fileName)
    if path.suffix.lower() != ".msh":
        raise NotImplementedError(_UNSUPPORTED.format(name=f"open({path.suffix})"))
    _state.data = read_msh(path)
    order = np.argsort(_state.data.node_tags, kind="stable")
    _state._sort_order = order
    _state._sorted_tags = _state.data.node_tags[order]


def _data() -> MshData:
    if _state.data is None:
        raise RuntimeError("モデルがありません（open でメッシュを読んでから）")
    return _state.data


def _coords_of(tags: np.ndarray) -> np.ndarray:
    """節点タグ → 座標 (n, 3)."""
    data = _data()
    if len(tags) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pos = np.searchsorted(_state._sorted_tags, tags)
    pos = np.clip(pos, 0, len(_state._sorted_tags) - 1)
    if not np.array_equal(_state._sorted_tags[pos], tags):
        raise ValueError("要素が存在しない節点を参照しています")
    return data.node_coords[_state._sort_order[pos]]


class _Option:
    @staticmethod
    def setNumber(name: str, value: float) -> None:     # noqa: N802
        _state.options[name] = float(value)              # 表示の冗長度など。読込には影響しない

    @staticmethod
    def getNumber(name: str) -> float:                   # noqa: N802
        return _state.options.get(name, 0.0)

    def __getattr__(self, name: str):
        _unsupported(f"option.{name}")


class _Mesh:
    @staticmethod
    def getNodes(dim: int = -1, tag: int = -1, includeBoundary: bool = False,   # noqa: N802, N803
                 returnParametricCoord: bool = True):                            # noqa: N803
        """(節点タグ uint64, 座標 float64 (3n,), パラメトリック座標 float64 (空)).

        ``dim < 0`` なら全節点（4.1 はファイル順、2.2 は所属エンティティ順。どちらも gmsh と同じ）。エンティティを
        指定したらそのエンティティに属する節点、``includeBoundary`` ならその境界（下位の次元のエンティティを再帰的に）の節点も。
        """
        data = _data()
        if dim < 0:
            tags = data.node_tags
            coords = data.node_coords
        else:
            entities = ([tag] if tag >= 0 else
                        sorted({etag for (d, etag) in data.entity_physicals if d == dim}
                               | {b.entity_tag for b in data.blocks if b.dim == dim}
                               | {nb.entity_tag for nb in data.node_blocks if nb.dim == dim}))
            chunks: list[np.ndarray] = []
            for etag in entities:
                chunks.extend(_entity_node_tags(data, dim, etag, includeBoundary))
            tags = (np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.int64))
            coords = _coords_of(tags)
        return (np.asarray(tags, dtype=np.uint64), np.asarray(coords, dtype=np.float64).reshape(-1),
                np.zeros(0, dtype=np.float64))

    @staticmethod
    def getElementsByType(elementType: int, tag: int = -1, task: int = 0, numTasks: int = 1):  # noqa: N802, N803
        """(要素タグ uint64, 節点タグ uint64 (平坦)). エンティティのタグ順に、各エンティティ内はファイル順."""
        data = _data()
        dim = TYPE_DIM.get(int(elementType), -1)
        blocks = [b for b in data.blocks
                  if b.elem_type == int(elementType) and b.dim == dim and (tag < 0 or b.entity_tag == tag)]
        blocks.sort(key=lambda b: b.entity_tag)
        if not blocks:
            return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint64)
        elem_tags = np.concatenate([b.elem_tags for b in blocks])
        node_tags = np.concatenate([b.node_tags.reshape(-1) for b in blocks])
        return elem_tags.astype(np.uint64), node_tags.astype(np.uint64)

    def __getattr__(self, name: str):
        _unsupported(f"model.mesh.{name}")


def _entity_node_tags(data: MshData, dim: int, etag: int, include_boundary: bool) -> list[np.ndarray]:
    out = [data.node_tags[nb.start:nb.stop] for nb in data.node_blocks if nb.dim == dim and nb.entity_tag == etag]
    if include_boundary:                                 # 境界は $Entities のトポロジから（2.2 には無いので効かない）
        seen: set[tuple[int, int]] = set()
        stack = [(dim, etag)]
        while stack:                                     # 境界をたどる（面 → 曲線 → 点）
            d, t = stack.pop()
            for bt in data.entity_bounds.get((d, t), []):
                key = (d - 1, bt)
                if key in seen:
                    continue
                seen.add(key)
                out += [data.node_tags[nb.start:nb.stop] for nb in data.node_blocks
                        if nb.dim == key[0] and nb.entity_tag == key[1]]
                stack.append(key)
    return out


class _Model:
    mesh = _Mesh()

    @staticmethod
    def getPhysicalGroups(dim: int = -1) -> list[tuple[int, int]]:   # noqa: N802
        data = _data()
        dims = [dim] if dim >= 0 else [0, 1, 2, 3]
        return [(d, t) for d in dims for t in data.physical_groups(d)]

    @staticmethod
    def getPhysicalName(dim: int, tag: int) -> str:                  # noqa: N802
        return _data().physical_name(int(dim), int(tag))

    @staticmethod
    def getEntitiesForPhysicalGroup(dim: int, tag: int) -> np.ndarray:   # noqa: N802
        return np.asarray(_data().entities_in_physical(int(dim), int(tag)), dtype=np.int32)

    def __getattr__(self, name: str):
        _unsupported(f"model.{name}")


option = _Option()
model = _Model()
