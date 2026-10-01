"""Primary/backup playback switch plan over a UTC program-time window.

Both inputs go through exactly the same pipeline as ``/v1/intersect`` (and
``/v1/locate``): snapshots are merged by media sequence,
``#EXT-X-PROGRAM-DATE-TIME`` anchors are validated and normalized to UTC, and
program time is extrapolated with exact decimal ``EXTINF`` durations inside one
contiguous discontinuity run only. Unanchored runs have no known time.

Selection inside the requested half-open window ``[start_at, end_at)``:

* whenever the primary stream ``a`` has known coverage at an instant, the
  primary segment is used;
* the backup stream ``b`` is used *only* inside primary gaps, and each backup
  usage is clipped to that gap, so the primary resumes exactly when its
  coverage returns — even in the middle of a backup segment;
* only strictly positive half-open intersections count: zero-duration segments
  provide no coverage and merely touching endpoints do not cover each other.

The concatenated usage intervals must cover the whole requested window. If the
two streams together still leave an uncovered sub-range, the request fails
with 422 naming the *first* uncovered interval; no partial plan is returned.
Self-overlap of known-time ranges across discontinuity runs within one stream,
and every merge / anchor conflict, are rejected exactly as in
``/v1/intersect``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Optional

from .exactdecimal import exact_sub
from .intersect import (
    IntersectError,
    _assert_self_disjoint,
    _timed_segments,
)
from .locate import format_program_time
from .reconcile import ReconcileError

PRIMARY = "a"
BACKUP = "b"


class SwitchError(ReconcileError):
    """A failure while building a switch plan; always mapped to HTTP 422.

    ``stream`` ("a"/"b") identifies the offending input when the failure is
    attributable to one side; ``sequences`` lists ambiguous sequences;
    ``gap_start``/``gap_end`` carry the first UTC sub-range neither stream
    covers.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stream: Optional[str] = None,
        snapshot_index: Optional[int] = None,
        sequence: Optional[int] = None,
        sequences: Optional[list[int]] = None,
        gap_start: Optional[Decimal] = None,
        gap_end: Optional[Decimal] = None,
    ) -> None:
        super().__init__(
            code,
            message,
            snapshot_index=snapshot_index,
            sequence=sequence,
        )
        self.stream = stream
        self.sequences = sequences
        self.gap_start = gap_start
        self.gap_end = gap_end


@dataclass(frozen=True)
class PlanInterval:
    """One continuous usage interval served by exactly one source segment."""

    source: str
    sequence: int
    uri: str
    start_at: Decimal
    end_at: Decimal
    # Exact seconds from the segment's own start to the interval boundaries.
    offset_start: Decimal
    offset_end: Decimal


def _clipped_pieces(
    segments, lo: Decimal, hi: Decimal
) -> list[tuple[object, Decimal, Decimal]]:
    """Intersect every segment range with half-open ``[lo, hi)``, UTC-sorted.

    Zero- and negative-length intersections are dropped, so zero-duration
    segments (``end_at == start_at``) never contribute coverage.
    """

    pieces: list[tuple[object, Decimal, Decimal]] = []
    ordered = sorted(
        segments, key=lambda seg: (seg.start_at, seg.end_at, seg.sequence)
    )
    for seg in ordered:
        start = max(seg.start_at, lo)
        end = min(seg.end_at, hi)
        if start < end:
            pieces.append((seg, start, end))
    return pieces


def _cover_gap(
    segments, lo: Decimal, hi: Decimal
) -> list[PlanInterval]:
    """Cover primary gap ``[lo, hi)`` with backup segments or report the gap.

    Raises :class:`SwitchError` on the first uncovered sub-range inside the
    gap. Backup ranges are clipped to the gap, so an interval never spills
    into primary-covered time.
    """

    intervals: list[PlanInterval] = []
    cursor = lo
    for seg, start, end in _clipped_pieces(segments, lo, hi):
        if start > cursor:
            raise SwitchError(
                "uncovered_time",
                (
                    "requested window is not covered by either stream from "
                    f"{format_program_time(cursor)} to "
                    f"{format_program_time(start)}: primary stream 'a' has no "
                    "known segment there and backup stream 'b' does not cover it"
                ),
                gap_start=cursor,
                gap_end=start,
            )
        intervals.append(
            PlanInterval(
                source=BACKUP,
                sequence=seg.sequence,
                uri=seg.uri,
                start_at=start,
                end_at=end,
                offset_start=exact_sub(start, seg.start_at),
                offset_end=exact_sub(end, seg.start_at),
            )
        )
        cursor = end

    if cursor < hi:
        raise SwitchError(
            "uncovered_time",
            (
                "requested window is not covered by either stream from "
                f"{format_program_time(cursor)} to "
                f"{format_program_time(hi)}: primary stream 'a' has no known "
                "segment there and backup stream 'b' does not cover it"
            ),
            gap_start=cursor,
            gap_end=hi,
        )
    return intervals


def build_switch_plan(
    playlist_url_a: str,
    snapshots_a: Iterable[str],
    playlist_url_b: str,
    snapshots_b: Iterable[str],
    start_at: Decimal,
    end_at: Decimal,
) -> list[PlanInterval]:
    """Build the primary/backup switch plan covering ``[start_at, end_at)``.

    Both streams are validated independently (primary first, then backup)
    with the merge / anchor / self-overlap rules of ``/v1/intersect``. The
    returned intervals are in UTC order, continuous and non-overlapping over
    the whole window: primary wherever it is known, backup clipped to the
    primary gaps.
    """

    def _resolve(stream: str, playlist_url: str, snapshots: Iterable[str]):
        try:
            segments = _timed_segments(stream, playlist_url, snapshots)
            _assert_self_disjoint(stream, segments)
        except IntersectError as exc:
            raise SwitchError(
                exc.code,
                exc.message,
                stream=exc.stream,
                snapshot_index=exc.snapshot_index,
                sequence=exc.sequence,
                sequences=exc.sequences,
            ) from exc
        return segments

    segments_a = _resolve(PRIMARY, playlist_url_a, snapshots_a)
    segments_b = _resolve(BACKUP, playlist_url_b, snapshots_b)

    intervals: list[PlanInterval] = []
    cursor = start_at
    for seg, start, end in _clipped_pieces(segments_a, start_at, end_at):
        if cursor < start:
            # Primary has no known range [cursor, start); backup may fill it.
            intervals.extend(_cover_gap(segments_b, cursor, start))
        intervals.append(
            PlanInterval(
                source=PRIMARY,
                sequence=seg.sequence,
                uri=seg.uri,
                start_at=start,
                end_at=end,
                offset_start=exact_sub(start, seg.start_at),
                offset_end=exact_sub(end, seg.start_at),
            )
        )
        # Pieces are positive-length, sorted and (self-overlap having been
        # rejected) disjoint, so each end strictly passes the previous cursor.
        cursor = end

    if cursor < end_at:
        intervals.extend(_cover_gap(segments_b, cursor, end_at))

    return intervals
