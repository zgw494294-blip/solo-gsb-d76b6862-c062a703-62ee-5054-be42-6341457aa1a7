"""Exact decimal arithmetic for the program-time chain.

Every timestamp and duration in the reconcile / locate / intersect /
switch-plan pipeline is a :class:`decimal.Decimal` so that no binary floating
point is ever involved. Decimal *construction* from a string or integer is
always exact, but its arithmetic operators obey the current context, whose
default precision is only 28 significant digits. That is too small for the
data this service keeps exactly: a modern UTC POSIX timestamp already spends
ten digits on the integer seconds (e.g. ``1704067200``), so adding an
18-digit fractional second needs 28 digits and an 19-digit one needs 29 — the
default context silently rounds the fraction away, making two distinct
anchors compare equal and corrupting in-segment offsets.

The helpers in this module perform each addition/subtraction inside a local
context whose precision is derived from the operands themselves, so the
result is represented exactly regardless of how many fractional digits the
input carried. Decimal precision is never reduced and no floating point is
introduced.
"""

from __future__ import annotations

from decimal import Decimal, localcontext


def _significant_digits(value: Decimal) -> int:
    """Number of significant digits of ``value`` (zero has no significant)."""

    if not value:
        return 0
    return len(value.as_tuple().digits)


def _add_precision(left: Decimal, right: Decimal) -> int:
    """Precision that represents ``left + right`` exactly.

    Aligned addition/subtraction needs one significant digit per coefficient
    position from the most-significant digit (``adjusted()``) down to the most
    negative exponent; one extra digit covers a carry into a new position.
    """

    if not left or not right:
        # Adding/removing zero never changes the other operand's digits.
        return max(_significant_digits(left), _significant_digits(right), 1)

    top = max(left.adjusted(), right.adjusted())
    bottom_exponent = min(left.as_tuple().exponent, right.as_tuple().exponent)
    return top - bottom_exponent + 1 + 1


def exact_add(left: Decimal, right: Decimal) -> Decimal:
    """Exact ``left + right``, free of context-precision rounding."""

    with localcontext() as context:
        context.prec = _add_precision(left, right)
        return left + right


def exact_sub(left: Decimal, right: Decimal) -> Decimal:
    """Exact ``left - right``, free of context-precision rounding."""

    with localcontext() as context:
        context.prec = _add_precision(left, right)
        return left - right
