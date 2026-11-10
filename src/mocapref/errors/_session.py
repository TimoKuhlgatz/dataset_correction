"""Helpers for checks that pool all trials of one session (one mounting of the sensors)."""
from __future__ import annotations

from ..recording import Recording, Segment


def as_session(recordings: Recording | list[Recording]) -> list[Recording]:
    """Accept one recording or a list of recordings of the same session."""
    return [recordings] if isinstance(recordings, Recording) else list(recordings)


def segments_by_name(recordings: list[Recording]) -> dict[str, list[Segment]]:
    """For every segment name, the segments of that name across all recordings, in recording order."""
    out: dict[str, list[Segment]] = {}
    for rec in recordings:
        for name, seg in rec.segments.items():
            out.setdefault(name, []).append(seg)
    return out
