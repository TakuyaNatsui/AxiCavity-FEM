"""旧形式（ver2.x ``.gmshproj``、Superfish ``.af``）の取り込みと書き出し（純 Python）.

読み込みは ver2.3 のコア（``MultiRegionGeometry.from_dict``: schema 2.1〜2.3 と、schema_version の無い
単一ループの旧形式）と ``superfish_io`` に任せ、:func:`convert.from_multi_region` で ver3 の文書にする。
書き出しは :func:`convert.to_multi_region` の結果をそのままコアに渡す。
"""

from __future__ import annotations

import json
from pathlib import Path

from ...shared.multi_region_model import MultiRegionGeometry
from ...shared.superfish_io import (
    SuperfishExportError,
    export_superfish,
    import_superfish,
    validate_superfish_exportable,
)
from .convert import from_multi_region, to_multi_region
from .document import AxiDocument, SketchArc, SketchLine

GMSHPROJ_SUFFIX = ".gmshproj"
SUPERFISH_SUFFIX = ".af"


def load_gmshproj(path: str | Path) -> tuple[AxiDocument, list[str]]:
    """``.gmshproj``（schema 2.1〜2.3、または旧 ver2 の単一ループ形式）を読む。戻り値は (文書, 警告)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geom = MultiRegionGeometry.from_dict(data)
    doc, warnings = from_multi_region(geom)
    doc.meta.name = Path(path).stem
    if data.get("schema_version") is None:
        # 旧形式の "None" / 未指定はコアの変換で PEC になるが、ver3 では自動判定（軸 → None）に任せる
        curves = [e for e in doc.sketch.entities if isinstance(e, (SketchLine, SketchArc))]
        for raw, curve in zip(data.get("segments", []), curves):
            if raw.get("physical_name") in (None, "", "None"):
                doc.boundaries.pop(curve.id, None)
        warnings.insert(0, "旧形式（単一領域）を Multi-Region 形式に変換して読み込みました")
    return doc, warnings


def save_gmshproj(doc: AxiDocument, path: str | Path) -> list[str]:
    """ver2.3 と同じ ``.gmshproj``（schema 2.3）に書き出す。戻り値は警告."""
    result = to_multi_region(doc)
    result.geom.save_json(path)
    return result.warnings


def load_superfish(path: str | Path, units: str = "mm") -> tuple[AxiDocument, list[str]]:
    """Superfish ``.af`` を読む（座標は cm から ``units`` へ換算）。閉じていなければ曲線だけ取り込む."""
    geom = import_superfish(path, units)
    doc, warnings = from_multi_region(geom)
    doc.boundaries.clear()          # .af に境界条件は無い（コアは全て PEC にする）→ 自動判定に任せる
    doc.meta.name = Path(path).stem
    return doc, warnings


def save_superfish(doc: AxiDocument, path: str | Path) -> list[str]:
    """Superfish ``.af`` に書き出す（単一領域・穴なし・閉ループのときだけ。BC は保存されない）."""
    result = to_multi_region(doc)
    errors = validate_superfish_exportable(result.geom)
    if errors:
        raise SuperfishExportError("Superfish 形式に書き出せません:\n  - " + "\n  - ".join(errors))
    export_superfish(result.geom, path)
    return result.warnings
