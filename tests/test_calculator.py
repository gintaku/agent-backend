"""Tests for tools/calculator.py."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc(expr: str) -> str:
    from tools.calculator import calculator
    return calculator.invoke({"expression": expr})


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

def test_addition():
    assert _calc("2 + 2") == "4"


def test_subtraction():
    assert _calc("10 - 3") == "7"


def test_multiplication():
    assert _calc("6 * 7") == "42"


def test_float_division():
    result = float(_calc("10 / 4"))
    assert result == pytest.approx(2.5)


def test_integer_division_result():
    # numexpr returns float for '/' — result should still be parseable
    result = float(_calc("8 / 2"))
    assert result == pytest.approx(4.0)


def test_parentheses_respected():
    assert _calc("(2 + 3) * 4") == "20"


def test_exponentiation():
    assert _calc("2 ** 10") == "1024"


def test_nested_expression():
    # 10 * (3 + 4) = 70
    assert _calc("10 * (3 + 4)") == "70"


# ---------------------------------------------------------------------------
# Math functions (numexpr built-ins)
# ---------------------------------------------------------------------------

def test_sqrt():
    result = float(_calc("sqrt(144)"))
    assert result == pytest.approx(12.0)


def test_sin_zero():
    result = float(_calc("sin(0)"))
    assert result == pytest.approx(0.0)


def test_abs_negative():
    result = float(_calc("abs(-42)"))
    assert result == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_expression_returns_error():
    result = _calc("not_a_valid_expression$$")
    assert result.startswith("Error")


def test_division_by_zero_returns_error_or_inf():
    # numexpr returns inf for float division by zero; that is acceptable
    result = _calc("1 / 0")
    # Accept either an "Error" message or a valid numeric string (inf / nan)
    assert "Error" in result or result.lower() in {"inf", "nan", "-inf"}


def test_empty_expression_returns_error():
    result = _calc("")
    assert result.startswith("Error")
