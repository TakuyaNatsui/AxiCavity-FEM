"""式評価（ver2.3 ``shared/expression_eval.py`` へのアダプタ）.

EM-CAD-py から転用したモジュール（``sketch/solver.py`` / ``ui/sketch_panels.py`` など）は
``evaluate_expression(source, params)`` / ``resolve_params(params)`` / ``ExpressionError`` を
使うので、その名前で ver2.3 の安全な評価器（``ast`` ベース、``eval`` 不使用、``sin`` / ``sqrt`` /
``pi`` … が使える。べき乗は ``**``）を提供する。評価器そのものはコアにあり、ここでは変えない。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...shared.expression_eval import (  # noqa: F401 — 再公開
    EvalError,
    eval_expr,
    evaluate_variables,
    expr_or_none,
    is_plain_number,
)

ExpressionError = EvalError


def evaluate_expression(source: str, params: Mapping[str, float] | None = None) -> float:
    """式（または数値）を評価する。失敗は :class:`ExpressionError`（= ver2.3 ``EvalError``）."""
    return eval_expr(source, dict(params or {}))


def resolve_params(params: Sequence, strict: bool = True) -> dict[str, float]:
    """パラメータ（``Param`` の列）を上から順に評価して名前 → 値の辞書にする.

    ``strict=True`` なら評価できない行があれば :class:`ExpressionError`（行の名前と理由を列挙）。
    ``strict=False`` なら評価できない行を飛ばす（ver2.3 ``evaluate_variables`` と同じ）。
    """
    rows = [(p.name, p.expression) for p in params]
    values, results = evaluate_variables(rows)
    if strict:
        errors = [f"{name or '?'}: {result}" for (name, _), result in zip(rows, results)
                  if result.startswith("エラー")]
        if errors:
            raise ExpressionError("; ".join(errors))
    return values


def param_results(params: Sequence) -> list[str]:
    """パラメータ表の「値」列（各行の評価結果の表示文字列。エラーなら ``エラー: …``）."""
    return evaluate_variables([(p.name, p.expression) for p in params])[1]


def format_value(value: float) -> str:
    """数値プレビュー用（ver2.3 の ``= 110.000000`` と同じ 6 桁固定）."""
    return f"{float(value):.6f}"


def format_compact(value: float) -> str:
    """表や寸法ラベル用の短い表記（小数 3 桁で丸め、整数なら小数点なし）."""
    r = round(float(value) * 1000) / 1000
    return str(int(r)) if r == int(r) else str(r)
