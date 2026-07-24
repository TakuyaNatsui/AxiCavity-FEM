"""数式・変数の安全な評価エンジン (ver2.2).

Multi-Region Editor の座標・寸法・材料などの入力欄で、数値だけでなく
``a + b`` のような数式や、ユーザ定義変数を使えるようにするための評価器。

評価は Python の ``ast`` を用いた安全な評価器で、``eval`` は使わない
（``smart-calc-note-main/smart_calc.py`` から移植）。サポートする演算は
四則演算・べき乗・剰余・整数除算・単項マイナス・数学関数 (math モジュール)。

主な公開 API:
- ``eval_expr(expr, variables)``   : 数式文字列 → float
- ``eval_scalar(text, variables)`` : 入力欄用。数値/式のどちらでも float に
- ``evaluate_variables(rows)``     : (name, expr) 行を順次評価して変数辞書を構築
- ``EvalError``                    : 評価エラー例外
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

# ---------------------------------------------------------------------------
# 評価エンジン (ast ベース、eval 不使用)
# ---------------------------------------------------------------------------

# 許可する二項演算子
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 許可する単項演算子
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 許可する数学関数・定数 (math モジュールから抽出)
_ALLOWED_NAMES: dict[str, Any] = {
    name: getattr(math, name)
    for name in [
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "sqrt", "exp", "log", "log10", "log2",
        "pow", "hypot", "degrees", "radians", "floor", "ceil", "fabs",
        "pi", "e", "tau",
    ]
}
_ALLOWED_NAMES["abs"] = abs
_ALLOWED_NAMES["min"] = min
_ALLOWED_NAMES["max"] = max


class EvalError(ValueError):
    """数式評価時のエラー.

    ``ValueError`` を継承しているため、既存の ``except ValueError`` ハンドラ
    (座標欄の不正入力処理など) でそのまま捕捉できる。
    """


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float:
    """AST ノードを再帰的に評価する."""
    if isinstance(node, ast.Constant):  # 数値リテラル
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise EvalError(f"数値以外の定数は使えません: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):  # 変数参照 / 定数
        if node.id in variables:
            return float(variables[node.id])
        if node.id in _ALLOWED_NAMES:
            val = _ALLOWED_NAMES[node.id]
            if isinstance(val, (int, float)):
                return float(val)
            raise EvalError(f"'{node.id}' は関数であり値ではありません")
        raise EvalError(f"未定義の変数: {node.id}")
    if isinstance(node, ast.BinOp):  # 二項演算
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise EvalError(f"未対応の演算子: {op_type.__name__}")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        try:
            return _BIN_OPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise EvalError("0 で割ることはできません") from exc
    if isinstance(node, ast.UnaryOp):  # 単項演算
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise EvalError(f"未対応の単項演算子: {op_type.__name__}")
        operand = _eval_node(node.operand, variables)
        return _UNARY_OPS[op_type](operand)
    if isinstance(node, ast.Call):  # 関数呼び出し
        if not isinstance(node.func, ast.Name):
            raise EvalError("複雑な関数呼び出しは使えません")
        fname = node.func.id
        if fname not in _ALLOWED_NAMES:
            raise EvalError(f"未定義の関数: {fname}")
        func = _ALLOWED_NAMES[fname]
        if not callable(func):
            raise EvalError(f"'{fname}' は関数ではありません")
        if node.keywords:
            raise EvalError("キーワード引数は使えません")
        args = [_eval_node(arg, variables) for arg in node.args]
        try:
            return float(func(*args))
        except (ValueError, TypeError) as exc:
            raise EvalError(f"関数 {fname} の評価に失敗: {exc}") from exc
    raise EvalError(f"未対応の構文: {type(node).__name__}")


def eval_expr(expr: str, variables: dict[str, float] | None = None) -> float:
    """数式文字列を評価して float を返す.

    Parameters
    ----------
    expr:
        評価する数式文字列 (例: ``"a + b"``, ``"100 * sin(pi/4)"``)。
    variables:
        変数名 → 数値の辞書 (省略時は空)。
    """
    variables = variables or {}
    if not isinstance(expr, str):
        raise EvalError("式は文字列である必要があります")
    expr = expr.strip()
    if not expr:
        raise EvalError("空の式です")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise EvalError(f"構文エラー: {exc.msg}") from exc
    return _eval_node(tree.body, variables)


def eval_scalar(text: str, variables: dict[str, float] | None = None) -> float:
    """入力欄の文字列を float に評価する (数値でも数式でも可).

    座標・寸法・材料値など、GUI の各入力欄の共通評価入口。空文字は EvalError。
    """
    return eval_expr(text, variables)


def is_plain_number(text: str) -> bool:
    """text が素の数値 (float() で変換可能) かどうか。"""
    try:
        float(text.strip())
        return True
    except (ValueError, AttributeError):
        return False


def expr_or_none(text: str) -> str | None:
    """入力欄テキストから「保存すべき式」を返す。

    素の数値 (例 "12.5") や空文字は式として保存する必要がないため None。
    数式 (例 "a+b", "sqrt(a)") は trim した文字列を返す。座標・寸法・材料の
    Update ハンドラで「この入力は式か定数か」を判定するのに使う (ver2.2)。
    """
    if text is None:
        return None
    t = text.strip()
    if not t or is_plain_number(t):
        return None
    return t


# ---------------------------------------------------------------------------
# 変数テーブルの評価
# ---------------------------------------------------------------------------


def evaluate_variables(
    rows: list[tuple[str, str]]
) -> tuple[dict[str, float], list[str]]:
    """(name, expr) の行を上から順に評価し、変数辞書と各行の結果を返す.

    - 上の行で定義した変数を下の行で参照できる (前方参照は不可 = 未定義エラー)。
    - name か expr が空の行はスキップ (結果は空文字列)。
    - 評価に失敗した行はエラーメッセージを結果に入れ、その変数は未定義のまま。

    Returns
    -------
    (variables, results):
        variables … 最終的な変数名 → 値の辞書。
        results   … 各行に対応する表示用文字列 (値または ``"エラー: ..."``)。
                    rows と同じ長さ。
    """
    variables: dict[str, float] = {}
    results: list[str] = []
    for name, expr in rows:
        name = (name or "").strip()
        expr = (expr or "").strip()
        if not name or not expr:
            results.append("")
            continue
        try:
            value = eval_expr(expr, variables)
            variables[name] = value
            results.append(f"{value:g}")
        except EvalError as exc:
            results.append(f"エラー: {exc}")
    return variables, results


def evaluate_expression(expr: str, variables: dict[str, float] | None = None) -> float:
    """単一式を評価する公開ヘルパ (後方互換の別名)."""
    return eval_expr(expr, variables)
