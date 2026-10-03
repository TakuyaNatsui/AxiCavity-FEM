"""形状・メッシュ・モデルのハッシュ（EM-CAD-py ``io/cv3dproj.py`` の geometry_hash / mesh_hash /
model_hash の ver3 版）.

- ``geometry_hash``: 単位・パラメータ・スケッチ。境界条件や材料を変えても変わらない
  （結果の場をスケッチに重ねてよいかの判定）。
- ``mesh_hash``: 形状 ＋ 領域名・材料タグ ＋ 境界条件の指定 ＋ メッシュ設定。材料値（ε_r・tanδ）や
  ソルバー設定では変わらない（メッシュタブのメッシュを解析で再利用してよいかの判定）。
- ``model_hash``: meta（名前・日時）以外の全部（未保存の変更の判定）。
"""

from __future__ import annotations

import hashlib
import json

from .document import AxiDocument, is_reference
from .serialize import document_to_dict, to_jsonable


def _digest(obj) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _sketch_without_reference(sketch_dict: dict) -> dict:
    """参照ジオメトリ（原点・軸。文書を開くたびに足される）を除いたスケッチの辞書."""
    sketch_dict = dict(sketch_dict)
    sketch_dict["entities"] = [e for e in sketch_dict.get("entities", [])
                               if e.get("projection") != "reference"]
    return sketch_dict


def geometry_hash(doc: AxiDocument) -> str:
    return _digest({"units": doc.meta.units, "params": to_jsonable(doc.params),
                    "sketch": _sketch_without_reference(to_jsonable(doc.sketch))})


def mesh_hash(doc: AxiDocument) -> str:
    """メッシュを決める設定のハッシュ.

    領域は**解決済み**（閉領域ごとの名前と材料タグ。設定の無い閉領域は既定）で比べる。設定を既定のまま新しく
    作っただけ（例: εr を入力して RegionSetting ができた）ではメッシュは変わらないので、ハッシュも変わらない。
    """
    from .convert import resolve_regions      # 循環 import を避けるため関数内で

    regions = sorted((r.profile.id, r.name, r.material_tag) for r in resolve_regions(doc))
    return _digest({"geometry": geometry_hash(doc), "regions": regions,
                    "boundaries": dict(doc.boundaries), "mesh": to_jsonable(doc.mesh)})


def model_hash(doc: AxiDocument) -> str:
    raw = document_to_dict(doc)
    raw.pop("meta", None)
    if isinstance(raw.get("sketch"), dict):
        raw["sketch"] = _sketch_without_reference(raw["sketch"])
    raw["units"] = doc.meta.units
    return _digest(raw)
