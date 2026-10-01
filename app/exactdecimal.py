"""Exact decimal arithmetic for the program-time / segment-timeline chains.

Every duration, POSIX timestamp and in-segment offset in the service is a
:class:`decimal.Decimal`, but the default decimal context caps every operation
at 28 significant digits. A UTC instant like
``2024-01-01T00:00:00.0000000000000000001Z`` is a 10-digit whole-second count
(``1704067200``) plus an 18-digit fraction — 29 significant digits — so plain
addition under the default context rounds the fraction away, silently making
two distinct instants equal and distorting derived in-segment offsets.

The helpers here perform each operation under a local context whose precision
is derived from the operands themselves, guaranteeing an exact, rounding-free
result for finite decimal inputs. Floating point is never used and no limit is
imposed on the number of supported fractional digits.
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, localcontext
from typing import Iterable


def _integer_digits(value: Decimal) -> int:
    """Number of digits in front of the decimal point (``0`` for ``|v| < 1``)."""

    sign, digits, exponent = value.as_tuple()
    if exponent >= 0:
        return len(digits) + exponent
    return max(len(digits) + exponent, 0)


def _fraction_digits(value: Decimal) -> int:
    """Number of digits after the decimal point in the value's coefficient."""

    return max(-value.as_tuple().exponent, 0)


@contextmanager
def exact_context(values: Iterable[Decimal]):
    """Local decimal context wide enough to add/subtract ``values`` exactly.

    Adding/subtracting finite decimals never introduces new fractional digits,
    and at most one extra digit can appear in front of the decimal point. The
    result may combine one operand's whole-second magnitude with another
    operand's fraction depth, so the exact result needs
    ``max(integer digits) + max(fraction digits) + 1`` significant digits.
    """

    operands = list(values)
    precision = (
        max((_integer_digits(value) for value in operands), default=0)
        + max((_fraction_digits(value) for value in operands), default=0)
        + 1
    )
    with localcontext() as context:
        context.prec = precision
        yield


def exact_add(*values: Decimal) -> Decimal:
    """Exact decimal sum (no rounding, regardless of operand digit count)."""

    result = Decimal(0)
    if not values:
        return result
    with exact_context([result, *values]):
        for value in values:
            result = result + value
    return result


def exact_sub(left: Decimal, right: Decimal) -> Decimal:
    """Exact decimal difference (no rounding, regardless of operand digits)."""

    with exact_context((left, right)):
        return left - right


def decimal_fraction(fraction_digits: str) -> Decimal:
    """Build the exact value ``0.fraction_digits`` from a string of digits.

    Construction from a coefficient tuple never consults the active decimal
    context, so no rounding happens no matter how many sub-second digits the
    input carries.
    """

    if not fraction_digits:
        return Decimal(0)
    return Decimal(
        (0, tuple(int(ch) for ch in fraction_digits), -len(fraction_digits))
    )
