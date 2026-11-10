"""Reference-side measures, one module per effect.

Every module here reads a :class:`~mocapref.datasets.readers.characterisation.schema.Recording` and returns a list of
:class:`Result`, one per pair, per segment or per sensor as its definition dictates. Nothing in this package
estimates orientation: the only fits performed are **constant** rotations and
**constant** offsets fitted by least squares over a whole trial, and the only filtering is **zero-phase**,
which a measurement may use and a causal method may not. Both are named in the docstring of every module that
uses them.

Every constant a measure uses is declared as a :class:`Constant` with its provenance: *physical* (datasheet or first principles), *data-free* (a rule that needs no data),
*read-from-dataset* (taken from the deposit itself, e.g. a stated full-scale range). Where a definition would
need a threshold, the measure reports a **curve over a grid of thresholds** in ``extras`` and any
single number quoted beside it names the level it was read at.

Layout: ``_common.py`` (shared measurement primitives), one module per effect, and the registry
:func:`all_measures`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from mocapref.datasets.readers.characterisation.schema import Recording, ReferenceKind, Side

Provenance = Literal["physical", "data-free", "read-from-dataset"]
"""Where a constant comes from. ``physical``: a datasheet value or
first principles. ``data-free``: a rule that needs no data (a grid, a window set by a stated rule, a
conversion). ``read-from-dataset``: taken from the deposit (a stated full-scale range, the declared rate).
No characterisation constant is *tuned*; a measure that would need a tuned threshold reports a curve instead."""


@dataclass(frozen=True)
class Constant:
    """A named constant with its value, its units and where it comes from."""

    name: str
    value: float | int | str | tuple
    units: str
    provenance: Provenance
    note: str = ""


@dataclass
class Result:
    """One measured number (or small dict of numbers) for one effect, on one unit of one recording.

    ``unit`` is the thing the number describes, written so that it reads without the file: a pair
    (``"thigh-shank"``), a pair seen from one of its segments (``"thigh-shank/thigh"``), a segment
    (``"thigh"``), a sensor (``"thigh#0"``) or a same-segment pair (``"thigh#0-#1"``). ``route`` is the name of
    the route (specification §3); where two routes exist the measure returns a second ``Result`` with the same
    ``unit`` and the other route's name — the two are never merged. ``side`` follows the segment's reference
    kind (specification rule 5) and is never pooled across sides. ``measurable`` is ``False`` with a ``reason``
    whenever the dataset cannot supply what the definition needs — never a zero standing in for a missing
    number. ``extras`` holds curves over threshold grids (rule 6), diagnostics and the measurement's own limit.
    """

    effect: str
    dataset: str
    subject: str
    trial: str
    unit: str
    value: float | dict | None
    units: str
    route: str
    side: Side
    measurable: bool = True
    reason: str | None = None
    n_samples: int = 0
    extras: dict = field(default_factory=dict)
    constants: tuple[Constant, ...] = ()

    def as_dict(self) -> dict:
        """A JSON-serialisable form."""
        d = asdict(self)
        d["constants"] = [asdict(c) for c in self.constants]
        return _jsonable(d)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


SENSOR_SIDE_KINDS: tuple[ReferenceKind, ...] = ("housing", "mechanical", "encoder", "none")
"""Reference kinds whose reference measures the sensor itself (or is not used): a number computed against them
is **sensor-side**. Markers on skin, on bony landmarks or a model fit make a number **reference-side**, because
it contains the motion of the mount and of the skin as well (specification rule 5)."""


def side_of(recording: Recording, segment: str) -> Side:
    """The side rule of specification rule 5, read off the segment's reference kind."""
    seg = recording.segments.get(segment)
    kind: ReferenceKind = "none"
    if seg is not None and seg.reference is not None:
        kind = seg.reference.kind
    return "sensor-side" if kind in SENSOR_SIDE_KINDS else "reference-side"


def not_measurable(
    effect: str, recording: Recording, unit: str, route: str, side: Side, reason: str, units: str = ""
) -> Result:
    """A result that says the dataset cannot supply what the definition needs, and why (rule 7)."""
    return Result(
        effect=effect,
        dataset=recording.dataset,
        subject=recording.subject,
        trial=recording.trial,
        unit=unit,
        value=None,
        units=units,
        route=route,
        side=side,
        measurable=False,
        reason=reason,
    )


def all_measures() -> dict[str, Callable[..., list[Result]]]:
    """The registry ``scripts/characterise.py`` iterates: effect name (plain words) → ``measure``."""
    from . import (
        alignment,
        clock_offset,
        envelope,
        excitation,
        gyro_bias_without_rest,
        impacts,
        initial_rest_period,
        joint_centre_decomposition,
        joint_centre_drift,
        moving_joint_centre,
        reference_accuracy_limit,
        soft_mounting,
    )

    return {
        "initial rest period": initial_rest_period.measure,
        "moving joint centre": moving_joint_centre.measure,
        "joint-centre drift": joint_centre_drift.measure,
        "joint-centre decomposition": joint_centre_decomposition.measure,
        "excitation": excitation.measure,
        "impacts": impacts.measure,
        "soft mounting": soft_mounting.measure,
        "sensor-to-segment alignment": alignment.measure,
        "clock offset": clock_offset.measure,
        "reference accuracy limit": reference_accuracy_limit.measure,
        "gyro bias without rest": gyro_bias_without_rest.measure,
        "sensor envelope": envelope.measure,
    }
