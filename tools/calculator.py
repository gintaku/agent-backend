"""Safe math evaluator tool using numexpr."""
from __future__ import annotations

import numexpr
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Supports arithmetic, exponentiation, and common math functions
    (sin, cos, tan, sqrt, log, exp, abs, etc.).
    Examples: '2 + 2', '10 * (3 + 4)', 'sqrt(144)', 'sin(3.14159 / 2)'
    """
    try:
        result = numexpr.evaluate(expression.strip())
        # numexpr returns numpy scalars; convert to Python native type
        scalar = result.item() if hasattr(result, "item") else result
        return str(scalar)
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating '{expression}': {exc}"
