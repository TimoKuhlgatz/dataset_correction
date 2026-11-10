"""The common data structure every loader produces and every error check consumes.

One `Recording` is one trial of one dataset. It holds one `Segment` per instrumented body segment. A segment pairs
the IMU on it (`ImuStream`) with the optical reference orientation of the same segment (`ReferenceStream`).

Units everywhere: seconds, rad/s for the gyroscope, m/s^2 for the accelerometer.

Time model: every stream is sampled at a constant rate and starts at `start_s`, so sample k was taken at
`start_s + k / rate_hz`. Loaders set both streams to start at 0 s, exactly as the deposit pairs them. Timing errors
(time offset, clock drift) are therefore corrected by changing `start_s` and `rate_hz`, never by cutting or
resampling the data.

Validity: each stream carries a boolean mask `valid`. A correction that cannot repair a sample (for example a
clipped value) marks it invalid instead of inventing a value.

Corrections never modify a recording in place. They return a copy with one more entry in `corrections`, so the
deposited and the corrected version can always be compared.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ImuStream:
    """Gyroscope and accelerometer of one IMU, expressed in the sensor's own frame."""

    gyr: np.ndarray  # (N, 3) rad/s
    acc: np.ndarray  # (N, 3) m/s^2
    rate_hz: float
    start_s: float = 0.0
    valid: np.ndarray | None = None  # (N,) bool; None means all valid

    def __post_init__(self):
        self.gyr = np.asarray(self.gyr, float)
        self.acc = np.asarray(self.acc, float)
        if self.valid is None:
            self.valid = np.ones(len(self.gyr), bool)

    def __len__(self):
        return len(self.gyr)

    @property
    def times(self) -> np.ndarray:
        """Time stamp of every sample in seconds."""
        return self.start_s + np.arange(len(self)) / self.rate_hz

    @property
    def duration_s(self) -> float:
        return len(self) / self.rate_hz


@dataclass
class ReferenceStream:
    """Optical reference orientation of one segment, one rotation per frame.

    `orientation` is stored as the deposit gives it. Whether a rotation maps the segment frame to the world frame
    or the other way round is recorded in `convention` when the documentation states it, otherwise 'unknown'.
    Frames without a valid reference (occluded markers, NaN in the deposit) have `valid` False.
    """

    orientation: Rotation  # M rotations
    rate_hz: float
    start_s: float = 0.0
    valid: np.ndarray | None = None  # (M,) bool
    convention: str = 'unknown'  # 'segment_to_world', 'world_to_segment' or 'unknown'
    markers: dict[str, np.ndarray] | None = None  # optional raw marker trajectories, (M, 3) each, in mm

    def __post_init__(self):
        if self.valid is None:
            self.valid = np.ones(len(self.orientation), bool)

    def __len__(self):
        return len(self.orientation)

    @property
    def times(self) -> np.ndarray:
        return self.start_s + np.arange(len(self)) / self.rate_hz

    @property
    def duration_s(self) -> float:
        return len(self) / self.rate_hz


@dataclass
class Segment:
    """One body segment: its IMU and its optical reference."""

    name: str
    imu: ImuStream
    reference: ReferenceStream


@dataclass
class Correction:
    """One entry of a recording's correction log."""

    error: str  # e.g. 'synchronisation offset'
    description: str  # plain-language sentence of what was changed
    parameters: dict = field(default_factory=dict)


@dataclass
class Recording:
    """One trial of one dataset.

    `participant` and `session` identify where the trial belongs, so data can be split by participant and
    errors that stay constant over a session (mounting, gyroscope offset) can be checked over all its trials."""

    dataset: str
    trial: str
    segments: dict[str, Segment]
    joints: list[tuple[str, str]] = field(default_factory=list)  # pairs of segments connected by a joint
    metadata: dict = field(default_factory=dict)
    corrections: list[Correction] = field(default_factory=list)
    participant: str = ''  # who (or which device) was recorded; the unit for a participant-wise data split
    session: str = ''  # one mounting of the sensors; trials of one session share mounting and sensor offsets

    def __getitem__(self, segment_name: str) -> Segment:
        return self.segments[segment_name]

    @property
    def segment_names(self) -> list[str]:
        return list(self.segments)

    def copy(self) -> Recording:
        """A deep copy, so a correction can change it without touching the source recording."""
        return copy.deepcopy(self)

    def log_correction(self, error: str, description: str, **parameters) -> None:
        """Add one entry to the correction log. Correction functions call this on the copy they return."""
        self.corrections.append(Correction(error, description, parameters))

    def __repr__(self):
        parts = ', '.join(f'{s.name} ({len(s.imu)} IMU samples at {s.imu.rate_hz:g} Hz, '
                          f'{len(s.reference)} reference frames at {s.reference.rate_hz:g} Hz)'
                          for s in self.segments.values())
        return f'Recording({self.dataset} / {self.trial}: {parts}; {len(self.corrections)} corrections)'
