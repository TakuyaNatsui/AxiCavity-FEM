"""shared/expression_eval.py の数式・変数評価テスト (ver2.2)."""
import math

import pytest

from axicavity_fem.shared.expression_eval import (
    EvalError,
    eval_expr,
    eval_scalar,
    evaluate_variables,
)


# --- 基本演算 ---------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("1 + 2", 3.0),
    ("10 - 4", 6.0),
    ("3 * 4", 12.0),
    ("7 / 2", 3.5),
    ("7 // 2", 3.0),
    ("7 % 3", 1.0),
    ("2 ** 10", 1024.0),
    ("-5", -5.0),
    ("+5", 5.0),
    ("2 + 3 * 4", 14.0),
    ("(2 + 3) * 4", 20.0),
])
def test_basic_arithmetic(expr, expected):
    assert eval_expr(expr) == pytest.approx(expected)


def test_functions_and_constants():
    assert eval_expr("sqrt(16)") == pytest.approx(4.0)
    assert eval_expr("sin(pi/2)") == pytest.approx(1.0)
    assert eval_expr("cos(0)") == pytest.approx(1.0)
    assert eval_expr("max(1, 2, 3)") == pytest.approx(3.0)
    assert eval_expr("abs(-7)") == pytest.approx(7.0)
    assert eval_expr("2 * pi") == pytest.approx(2 * math.pi)


# --- 変数参照 ---------------------------------------------------------------

def test_variable_reference():
    assert eval_expr("a + b", {"a": 100.0, "b": 10.0}) == pytest.approx(110.0)
    assert eval_expr("2*a", {"a": 5.0}) == pytest.approx(10.0)


def test_undefined_variable_raises():
    with pytest.raises(EvalError):
        eval_expr("a + b", {"a": 1.0})


def test_syntax_error_raises():
    with pytest.raises(EvalError):
        eval_expr("1 +")
    with pytest.raises(EvalError):
        eval_expr("")


def test_division_by_zero():
    with pytest.raises(EvalError):
        eval_expr("1/0")


# --- セキュリティ: 危険な入力は EvalError -----------------------------------

@pytest.mark.parametrize("expr", [
    "__import__('os')",
    "().__class__",
    "a.b",
    "[1, 2, 3]",
    "{'x': 1}",
    "lambda: 1",
    "open('x')",
])
def test_dangerous_input_rejected(expr):
    with pytest.raises(EvalError):
        eval_expr(expr, {"a": 1.0})


# --- eval_scalar ------------------------------------------------------------

def test_eval_scalar_numeric_and_expr():
    assert eval_scalar("42") == pytest.approx(42.0)
    assert eval_scalar("3.14") == pytest.approx(3.14)
    assert eval_scalar("a*b", {"a": 3.0, "b": 4.0}) == pytest.approx(12.0)
    assert eval_scalar("  -5  ") == pytest.approx(-5.0)


def test_eval_scalar_empty_raises():
    with pytest.raises(EvalError):
        eval_scalar("")
    with pytest.raises(EvalError):
        eval_scalar("   ")


# --- evaluate_variables -----------------------------------------------------

def test_evaluate_variables_sequential():
    rows = [("a", "100"), ("b", "10"), ("c", "a + b")]
    variables, results = evaluate_variables(rows)
    assert variables == {"a": 100.0, "b": 10.0, "c": 110.0}
    assert results == ["100", "10", "110"]


def test_evaluate_variables_skips_empty_rows():
    rows = [("a", "5"), ("", ""), ("", "3"), ("b", "")]
    variables, results = evaluate_variables(rows)
    assert variables == {"a": 5.0}
    assert results == ["5", "", "", ""]


def test_evaluate_variables_error_row_isolated():
    # b は未定義 x を参照 → エラー行だが a と c は評価される
    rows = [("a", "2"), ("b", "x + 1"), ("c", "a * 3")]
    variables, results = evaluate_variables(rows)
    assert variables["a"] == pytest.approx(2.0)
    assert variables["c"] == pytest.approx(6.0)
    assert "b" not in variables
    assert results[0] == "2"
    assert results[1].startswith("エラー")
    assert results[2] == "6"


def test_evaluate_variables_forward_reference_is_error():
    # c は後で定義される d を参照 → 前方参照は未定義エラー
    rows = [("c", "d + 1"), ("d", "5")]
    variables, results = evaluate_variables(rows)
    assert "c" not in variables
    assert variables["d"] == pytest.approx(5.0)
    assert results[0].startswith("エラー")
