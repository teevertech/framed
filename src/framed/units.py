"""Canonical internal length unit and conversion helpers.

All lengths, positions, and dimensions in this codebase are floats in a
single canonical unit. Right now that unit is inches; to refactor the
project to millimeters, edit the bodies of the helpers in this file and
re-run the test suite. No other module should construct length values
without going through these helpers, and no other module should hardcode
a unit assumption.

Why a function-based interface (rather than a `Length` type or `pint`):
constructing typed scalars everywhere is heavy for the value here, and a
typed `Length` leaks through every function signature. A small, well-
documented set of constructors at the data boundary plus a single
canonical float internally is simpler and tests just as cleanly.
"""
from __future__ import annotations

CANONICAL_UNIT_NAME: str = "in"
"""Human-readable name of the canonical unit. Update alongside the
helpers if the canonical unit ever changes."""


def inches(value: float) -> float:
    """Express `value` inches in canonical units."""
    return value


def feet(value: float) -> float:
    """Express `value` feet in canonical units."""
    return value * 12.0


def mm(value: float) -> float:
    """Express `value` millimeters in canonical units."""
    return value / 25.4


def to_inches(value: float) -> float:
    """Convert a canonical value to inches (for display / external I/O)."""
    return value


def to_feet(value: float) -> float:
    """Convert a canonical value to feet (for display / external I/O)."""
    return value / 12.0


def to_mm(value: float) -> float:
    """Convert a canonical value to millimeters (for display / external I/O)."""
    return value * 25.4
