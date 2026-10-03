"""UI 文字列の翻訳（EM-CAD-py `emcad/i18n/tr.py` からの転用。ja.json / en.json）.

    from axicavity_fem.gui.i18n.tr import tr, set_language
    tr("ribbon.sketch")          # → "スケッチ"
    tr("feature.sketchName", n=3)  # 文字列中の {{n}}（i18next 形式）を置換

キーは "セクション.名前" の 2 階層（JSON のネスト）。見つからなければキーを返す。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

_DIR = Path(__file__).resolve().parent
_LANGS: dict[str, dict] = {}
_current = "ja"


def _load(lang: str) -> dict:
    if lang not in _LANGS:
        path = _DIR / f"{lang}.json"
        _LANGS[lang] = json.loads(path.read_text(encoding="utf-8")) \
            if path.exists() else {}
    return _LANGS[lang]


def set_language(lang: str) -> None:
    """表示言語を切り替える（"ja" / "en"）."""
    global _current
    _current = lang


def language() -> str:
    return _current


def tr(key: str, **kwargs) -> str:
    """翻訳文字列を返す。無ければ英語 → キーの順にフォールバック."""
    for lang in (_current, "en"):
        node = _load(lang)
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, str):
            if kwargs:
                return _PLACEHOLDER.sub(
                    lambda m: str(kwargs.get(m.group(1), m.group(0))), node)
            return node
    return key
