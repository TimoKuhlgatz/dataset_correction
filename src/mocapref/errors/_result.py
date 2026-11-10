"""The result type every check function returns, and the standard per-segment entry it contains."""
from __future__ import annotations

from dataclasses import dataclass, field


def entry(detected: bool | None, estimate: float, unit: str, quality: float | None = None, **evidence) -> dict:
    """Standard per-segment result: the same four keys for every error, so results can be tabulated.

    detected: this segment's decision (True / False / None = undecided).
    estimate: the size of the error in `unit` (offset in ms, number of samples, angle in degrees, ...).
    quality: how much to trust the estimate, where the check has such a measure (e.g. peak correlation).
    evidence: anything else the check measured, kept for inspection.
    """
    return dict(detected=detected, estimate=estimate, unit=unit, quality=quality, **evidence)


@dataclass
class CheckResult:
    """Outcome of one error check on one recording (or one session of recordings).

    detected: True if the error is present, False if the check ran and found nothing above its threshold, None if
        the check could not decide (for example too little motion or no rest phase). Not detected is not absent:
        below the threshold is not zero.
    summary: one plain-language sentence.
    per_segment: one `entry(...)` per segment.
    thresholds: the decision thresholds used. All defaults are exploratory, not fixed rules.
    estimate, unit: the error's size for the whole recording, where the check computes one (time offset, drift);
        `unit` is empty for checks without a recording-level estimate; estimate is NaN when undecided.
    """

    error: str
    detected: bool | None
    summary: str
    per_segment: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    estimate: float = float('nan')
    unit: str = ''

    def __str__(self):
        state = {True: 'DETECTED', False: 'not detected', None: 'undecided'}[self.detected]
        return f'{self.error}: {state}. {self.summary}'
