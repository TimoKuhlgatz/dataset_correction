"""The in-memory representation every adapter produces and every measure consumes.

A *dataset* holds *sequences* (one trial of one joint pair is the unit
that is scored; here a ``Recording`` is one trial with all its segments, and the pairs are listed on it).
Units are SI throughout: seconds, metres, radians per second, metres per second squared. Time vectors are
absolute within a recording and shared between streams only through their values, never through indices —
streams may have different rates, and a measure resamples what it needs (zero-phase, which a measurement may
use and a causal method may not).

Every optional field is ``None`` when the dataset cannot supply it, and a measure that needs it reports
"not measurable" for that recording rather than substituting anything.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ReferenceKind = Literal["housing", "skin", "mechanical", "anatomical", "skeletal-fit", "encoder", "none"]
"""Where the reference for a segment comes from. ``housing``: a marker cluster on the sensor housing (the
reference measures the sensor frame). ``skin``: markers on skin or bony landmarks (reference-side numbers).
``mechanical``: a rigid chain, rig or robot with optical markers on the segment. ``anatomical``: a frame built
from anatomical landmarks (used as the second frame when a housing cluster exists too). ``skeletal-fit``: an
inverse-kinematics model fit. ``encoder``: a joint encoder (no segment pose). ``none``: no reference."""

Side = Literal["sensor-side", "reference-side"]


@dataclass
class SensorStream:
    """One six-axis IMU as deposited, in the sensor frame."""

    t: np.ndarray  # (T,) seconds
    gyr: np.ndarray  # (T, 3) rad/s
    acc: np.ndarray  # (T, 3) m/s² (specific force, gravity included; if a dataset removed gravity, say so in meta)
    rate_hz: float  # the deposited rate
    family: str  # sensor family, e.g. "Xsens MTw", "Movella DOT", "Delsys Trigno Avanti", "Physilog6S", "IM948"
    range_gyr: float | None = None  # rad/s full scale, if documented
    range_acc: float | None = None  # m/s² full scale, if documented
    native_rate_hz: float | None = None  # the hardware rate, if documented and different from ``rate_hz``
    meta: dict = field(default_factory=dict)  # anything the adapter knows (units as deposited, flags)


@dataclass
class Pose:
    """A rigid frame over time from the reference: rotation frame→world and position of the frame's zero point."""

    t: np.ndarray  # (T,) seconds
    R: np.ndarray  # (T, 3, 3) rotation matrices, frame → world; NaN rows where the reference is missing
    p: np.ndarray  # (T, 3) metres, the frame's zero point in the world frame; NaN where missing
    rate_hz: float
    kind: ReferenceKind
    markers: np.ndarray | None = None  # (T, K, 3) metres, the raw markers the frame was built from, if any
    marker_names: list[str] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Segment:
    """One rigid body of the chain with what is attached to it."""

    name: str
    sensor: SensorStream | None  # the segment's IMU (None if the segment carries markers only)
    reference: Pose | None  # the reference frame of the segment (housing cluster where available)
    anatomical: Pose | None = None  # a second frame from anatomical markers, where both exist (specification §2.6, §2.7)
    extra_sensors: list[SensorStream] = field(default_factory=list)  # further IMUs on the same segment (same-segment pairs)
    mount: str | None = None  # how the sensor is attached, as documented ("strap", "tape", "printed holder", "bolted", None)


@dataclass
class Pair:
    """Two connected segments and the joint between them."""

    parent: str
    child: str
    joint: str  # "hinge", "2-dof", "ball", "mechanical", "none" (e.g. scapulothoracic), "unknown"
    joint_dof: int | None = None
    joint_centre_world: np.ndarray | None = None  # (T, 3) metres, a deposited optical joint centre, on the reference's time base
    joint_centre_t: np.ndarray | None = None  # (T,) seconds for ``joint_centre_world``
    meta: dict = field(default_factory=dict)


@dataclass
class Recording:
    """One trial of one subject: all segments, the pairs, and the trial-level facts a measure may need."""

    dataset: str  # kebab-case dataset name as in docs/concept.md §2 (e.g. "fullbody-17imu")
    subject: str
    trial: str
    segments: dict[str, Segment]
    pairs: list[Pair]
    same_segment_pairs: list[tuple[str, int, int]] = field(default_factory=list)  # (segment, index a, index b) into [sensor] + extra_sensors
    contacts_t: np.ndarray | None = None  # (C,) seconds of ground contacts from force plates or annotations, if deposited
    rest_annotation_s: float | None = None  # a documented initial rest period, if the dataset states one
    trial_type: str | None = None  # "walking", "running", "stairs", "static", "functional", ... as the dataset labels it
    meta: dict = field(default_factory=dict)  # licence, host, sensor family, notes the adapter wants to carry

    @property
    def duration_s(self) -> float:
        ends = [s.sensor.t[-1] - s.sensor.t[0] for s in self.segments.values() if s.sensor is not None and len(s.sensor.t)]
        return float(max(ends)) if ends else float("nan")

    def sensors_of(self, segment: str) -> list[SensorStream]:
        seg = self.segments[segment]
        return ([seg.sensor] if seg.sensor is not None else []) + list(seg.extra_sensors)


class Adapter:
    """What every dataset module exposes. ``recordings()`` is lazy: one trial in memory at a time."""

    name: str  # dataset name, kebab-case
    reference_kind: ReferenceKind  # the dataset's dominant reference kind (per-segment detail sits on the Segment)
    sensor_family: str
    licence: str  # as the host states it (licence collisions are sorted out later, not here)
    source: str  # the dataset's SOURCE.md path or DOI

    def trials(self) -> list[tuple[str, str]]:
        """All (subject, trial) identifiers, without loading anything."""
        raise NotImplementedError

    def load(self, subject: str, trial: str) -> Recording:
        raise NotImplementedError

    def recordings(self, limit: int | None = None) -> Iterator[Recording]:
        for i, (s, tr) in enumerate(self.trials()):
            if limit is not None and i >= limit:
                return
            yield self.load(s, tr)


def common_grid(t_a: np.ndarray, t_b: np.ndarray, rate_hz: float) -> np.ndarray:
    """A uniform time grid over the overlap of two streams (the measures resample onto it)."""
    t0 = max(float(t_a[0]), float(t_b[0]))
    t1 = min(float(t_a[-1]), float(t_b[-1]))
    n = int(np.floor((t1 - t0) * rate_hz)) + 1
    return t0 + np.arange(max(n, 0)) / rate_hz
