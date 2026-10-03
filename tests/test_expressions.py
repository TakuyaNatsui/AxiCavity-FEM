"""式評価アダプタ（ver2.3 の評価器を EM-CAD-py 互換の名前で使う）."""

import math

import pytest

from axicavity_fem.gui.core.document import Param
from axicavity_fem.gui.core.expressions import (
    ExpressionError,
    evaluate_expression,
    format_compact,
    format_value,
    param_results,
    resolve_params,
)


def test_evaluates_arithmetic_and_math():
    assert evaluate_expression("1 + 2 * 3") == 7
    assert evaluate_expression("(1 + 2) * 3") == 9
    assert evaluate_expression("2 ** 3") == 8
    assert evaluate_expression("-4 / 2 + 10") == 8
    assert evaluate_expression("1.5e1 + .5") == 15.5
    assert evaluate_expression("sin(pi / 2)") == pytest.approx(1.0)
    assert evaluate_expression("sqrt(16) + abs(-1)") == 5


def test_resolves_parameters():
    assert evaluate_expression("2 * R + 5", {"R": 10}) == 25
    with pytest.raises(ExpressionError):
        evaluate_expression("2 * Q")


@pytest.mark.parametrize("src", ["", "1 +", "(1 + 2", "1 / 0", "3 $ 4", "2 ^ 3", "__import__('os')"])
def test_rejects_malformed_or_unsupported_input(src):
    with pytest.raises(ExpressionError):
        evaluate_expression(src)


def test_resolves_document_parameters_in_order():
    values = resolve_params([Param("a", "R", "10"), Param("b", "L", "2 * R + 5")])
    assert values == {"R": 10, "L": 25}
    # 前方参照は不可（上の行だけ参照できる）
    with pytest.raises(ExpressionError):
        resolve_params([Param("a", "L", "2 * R"), Param("b", "R", "10")])
    assert resolve_params([Param("a", "L", "2 * R"), Param("b", "R", "10")], strict=False) == {"R": 10}


def test_param_results_show_values_and_errors():
    rows = [Param("a", "R", "10"), Param("b", "L", "2 * R"), Param("c", "bad", "R +"), Param("d", "", "")]
    results = param_results(rows)
    assert results[0] == "10" and results[1] == "20"
    assert results[2].startswith("エラー") and results[3] == ""


def test_formatting():
    assert format_value(110) == "110.000000"
    assert format_value(math.pi) == "3.141593"
    assert format_compact(2.5) == "2.5" and format_compact(3.0) == "3" and format_compact(1.23456) == "1.235"
