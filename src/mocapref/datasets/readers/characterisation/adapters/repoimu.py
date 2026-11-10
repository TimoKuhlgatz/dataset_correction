"""RepoIMU (Szczesna et al. 2016): a three-segment mechanical pendulum and a hand-held T-shaped wand, Vicon reference.
Deposit: the GitHub repository agnieszkaszczesna/RepoIMU, cloned to ``paths.dataset_dir("repoimu")``. No licence is
declared (all rights reserved by default): the files are read locally only.

Two readers of the one deposit: ``repoimu-pendulum`` (ADAPTER; ``Pendulum/``, 22 trials x 3 segment files, two
connected pairs) and ``repoimu-tstick`` (EXTRA_ADAPTERS[0]; ``TStick/``, 29 single-segment trials, no pair).

Files read: semicolon-separated CSV with two header rows and fourteen columns: ``Time``, Vicon quaternion
``W X Y Z``, accelerometer ``X Y Z``, gyroscope ``X Y Z``, magnetometer ``X Y Z``. The pendulum's three segment
files are joined by segment index. A comma decimal separator is normalised to a dot (one such field exists).

Units: none converted (time s, accelerometer m/s^2 with gravity, gyroscope rad/s). The magnetometer is normalised
and dimensionless; it is carried in ``SensorStream.meta["magnetometer_normalised"]`` and not used.

Reference (``kind="mechanical"``): ``Pose.R`` is the deposited Vicon quaternion (body to world). ``Pose.p`` is NaN:
no positions ship.

Deposit defects left in the data (none repaired): impossible jumps in the pendulum segment-3 reference (indices in
``Pose.meta["glitch_idx"]``); held reference runs of up to 2.2 s (``Pose.meta["frozen_runs"]``); the static trials
carry all-zero quaternions (NaN, ``Pose.meta["invalid_idx"]``); duplicated inertial samples in the pendulum set
(effective about 26 Hz accelerometer, 50 Hz gyroscope; ``held_mask`` and ``effective_rate_hz`` per stream);
gyroscope saturation near 10.5 rad/s on pendulum segment 3; repeated time stamps in the ``Test02a``/``Test02b``
files (``SensorStream.meta["duplicate_timestamp_idx"]``); an
undocumented, non-identity frame relation between unit and reference. Mounting, joint type and full-scale ranges
are not documented and stay None.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from .. import paths
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

N_COLS = 14
"""Units: columns.
Provenance: read-from-dataset — the deposited row layout (Time, quaternion W X Y Z, acceleration X Y Z, gyroscope X Y Z, magnetometer X Y Z)."""
HEADER_ROWS = 2  # both header variants are two rows; the columns are taken by position, never by name
QUAT_TOL = 0.01
"""Units: 1.
Provenance: data-free — a deposited quaternion further than one per cent from unit norm is not a rotation (data-free: a deposited quaternion further than 1 % from unit norm is not a rotation)."""
GLITCH_RAD_S = 15.0
"""Units: rad/s.
Provenance: data-free — data-free."""

#: The two trials RING's published RepoIMU number (3.44 +- 0.06 deg over "390 s, 90 Hz") was measured on.
RING_TRIALS = ("Test02a_Trial1", "Test02b_Trial1")


def _root() -> Path:
    return paths.dataset_dir("repoimu")


@lru_cache(maxsize=1)
def _glitch_census() -> dict[str, dict[str, str]]:
    """Per-file glitch counts stored with the deposit, keyed by file name.

    Small (88 rows) and read at most once per process; it carries counts only, so the sample locations are
    recomputed per trial in :func:`_pose`.
    """
    path = _root() / "_probe" / "glitch_census.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def _read(path: Path) -> tuple[np.ndarray, int]:
    """The fourteen deposited columns as floats, plus how many fields needed the comma→dot fix (defect 6)."""
    rows: list[list[float]] = []
    commas = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle):
            if line_no < HEADER_ROWS:
                continue
            fields = line.rstrip("\n").split(";")
            if len(fields) < N_COLS:
                continue  # a short final line; the deposit has none, but a truncated read must not crash
            values = []
            for field in fields[:N_COLS]:
                text = field.strip()
                if "," in text:
                    commas += 1
                    text = text.replace(",", ".")
                values.append(float(text) if text else np.nan)
            rows.append(values)
    return np.asarray(rows, dtype=float), commas


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True in a boolean mask."""
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return [(int(a), int(b - a)) for a, b in zip(idx[0::2], idx[1::2], strict=True)]


def _held_mask(a: np.ndarray) -> np.ndarray:
    """True where a row repeats its predecessor exactly — the zero-order hold of defect 4."""
    mask = np.zeros(len(a), dtype=bool)
    if len(a) > 1:
        mask[1:] = ~np.any(np.diff(a, axis=0) != 0, axis=1)
    return mask


def _rate(t: np.ndarray) -> float:
    span = float(t[-1] - t[0])
    return (len(t) - 1) / span if span > 0 else float("nan")


def _stream(raw: np.ndarray, commas: int, family: str, name: str) -> SensorStream:
    t = raw[:, 0]
    acc = raw[:, 5:8]  # m/s² as deposited, gravity included
    gyr = raw[:, 8:11]  # rad/s as deposited
    mag = raw[:, 11:14]  # normalised, dimensionless — not an SI channel and used by nothing
    held_acc, held_gyr = _held_mask(acc), _held_mask(gyr)
    dt = np.diff(t)
    return SensorStream(
        t=t,
        gyr=gyr,
        acc=acc,
        rate_hz=_rate(t),
        family=family,
        range_gyr=None,  # undocumented; the measured plateau is in meta below
        range_acc=None,  # undocumented
        native_rate_hz=None,  # undocumented
        meta={
            "file": name,
            "units_as_deposited": {"t": "s", "acc": "m/s^2 (gravity included)", "gyr": "rad/s"},
            "gravity_removed": False,
            "comma_fields": commas,
            "duplicate_timestamp_idx": np.flatnonzero(dt <= 0),
            "duplicate_timestamp_note": (
                "the deposited time column is quoted to 0.01 s on Test02a/Test02b, so a few consecutive rows "
                "carry an identical timestamp; the rows differ and are left in place"
            ),
            "held_mask_acc": held_acc,
            "held_mask_gyr": held_gyr,
            "effective_rate_acc_hz": _rate(t) * float(1.0 - held_acc.mean()),
            "effective_rate_gyr_hz": _rate(t) * float(1.0 - held_gyr.mean()),
            "held_note": (
                "a zero-order hold in the pendulum set only; the wand set has no held samples. Exposed as a "
                "mask, never removed"
            ),
            "gyr_abs_max_rad_s": float(np.nanmax(np.abs(gyr))) if len(gyr) else float("nan"),
            "gyr_norm_max_rad_s": float(np.nanmax(np.linalg.norm(gyr, axis=1))) if len(gyr) else float("nan"),
            "saturation_note": (
                "pendulum segment 3 pins at about 10.5 rad/s and some wand trials at about 7.36 rad/s; the "
                "full scale is undocumented, so no clipping flag is asserted, only the measured maximum"
            ),
            "magnetometer_normalised": mag,
            "magnetometer_note": "normalised and dimensionless, not SI; carried for completeness and used by nothing",
        },
    )


def _pose(raw: np.ndarray, name: str) -> Pose:
    t = raw[:, 0]
    q_wxyz = raw[:, 1:5]
    norm = np.linalg.norm(q_wxyz, axis=1)
    invalid = ~np.isfinite(norm) | (np.abs(norm - 1.0) > QUAT_TOL)

    q_xyzw = np.where(invalid[:, None], np.array([0.0, 0.0, 0.0, 1.0]), q_wxyz[:, [1, 2, 3, 0]])
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    R = Rotation.from_quat(q_xyzw).as_matrix()
    R[invalid] = np.nan

    frozen = np.zeros(len(raw), dtype=bool)
    if len(raw) > 1:
        frozen[1:] = ~np.any(np.diff(q_wxyz, axis=0) != 0, axis=1)

    # the: the frame-to-frame rotation divided by the file's
    # MEDIAN sample interval, not by the per-step interval, so that the repeated timestamps of defect 7 do
    # not manufacture an infinite rate where the deposit merely quoted the time column too coarsely.
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    rate = np.full(len(raw), np.nan)
    rate[0] = 0.0
    step_ok = (~invalid[:-1]) & (~invalid[1:])
    if step_ok.any() and dt_med > 0:
        a = Rotation.from_quat(q_xyzw[:-1][step_ok])
        b = Rotation.from_quat(q_xyzw[1:][step_ok])
        rate[1:][step_ok] = (a.inv() * b).magnitude() / dt_med
    rate[1:][~step_ok] = 0.0  # a step touching a missing reference has no defined rate; it is filed as invalid
    beside_invalid = np.zeros(len(raw), dtype=bool)
    beside_invalid[1:] = invalid[:-1] | invalid[1:]

    return Pose(
        t=t,
        R=R,
        p=np.full((len(raw), 3), np.nan),  # the deposit ships no positions at all
        rate_hz=_rate(t),
        kind="mechanical",
        markers=None,
        marker_names=None,
        meta={
            "file": name,
            "system": "Vicon Nexus, orientation quaternion only",
            "positions_available": False,
            "positions_note": (
                "no marker positions and no segment positions ship anywhere in the deposit, so Pose.p is "
                "all-NaN and no joint-centre quantity is observable on this dataset"
            ),
            "marker_placement": "not stated in the deposit and unverified (the data paper is closed access)",
            "quaternion_order_as_deposited": "W X Y Z",
            "duplicate_timestamp_idx": np.flatnonzero(np.diff(t) <= 0),
            "duplicate_timestamp_note": (
                "the reference shares the file's single time column, so it inherits the Test02a/Test02b tie: "
                "time is quoted to 0.01 s and a few consecutive rows carry an identical timestamp, which makes "
                "Pose.t non-decreasing rather than strictly increasing. The rows differ and are left in place"
            ),
            "invalid_idx": np.flatnonzero(invalid),
            "invalid_note": (
                "quaternion norm further than 1 % from unit, including the three static trials whose "
                "quaternion is identically zero; written as NaN rows, never filled"
            ),
            "frozen_runs": _runs_of_true(frozen),
            "frozen_note": (
                "runs of consecutive identical quaternions — occlusion gaps filled by hold, flagged by value "
                "equality because nothing in the file marks them. Never de-duplicated"
            ),
            "glitch_idx": np.flatnonzero(rate > GLITCH_RAD_S),
            "glitch_idx_beside_invalid": np.flatnonzero(beside_invalid),
            "glitch_rate_max_rad_s": float(np.nanmax(rate)) if len(rate) else float("nan"),
            "glitch_note": (
                f"samples whose frame-to-frame reference rotation rate exceeds {GLITCH_RAD_S:g} rad/s, the "
                "rotation taken over the file's median sample interval as an earlier count did; impulsive "
                "Vicon glitches, concentrated on pendulum segment 3. Exposed, never removed. A step touching a "
                "sample listed in invalid_idx has no defined rate and is filed there instead, which is why "
                "this count is 221 over the twenty segment-3 files where an earlier count reports 274: the "
                "53 remaining steps all touch an off-norm quaternion, verified sample by sample. The threshold "
                "was calibrated on the pendulum; on the wand it flags 758 steps that sit in only 4 of the 28 "
                "referenced files, and whether those are Vicon glitches or genuinely fast hand swings is not "
                "settled here — the flag is an exposure, not a verdict"
            ),
            "probe_glitch_census": _glitch_census().get(name),
            "reference_noise_floor_deg": (
                "0.04-0.27 deg r.m.s. relative to a still stretch's mean, measured on the deposit; the deposit "
                "states no reference accuracy anywhere"
            ),
        },
    )


class _RepoImuAdapter(Adapter):
    """Shared parsing for the two recording groups of the one deposit."""

    reference_kind = "mechanical"
    sensor_family = "not stated in the deposit (unverified; the data paper is closed access)"
    licence = "none declared (no LICENSE file; the GitHub API record gives license: null, so all rights reserved)"
    source = (
        "https://github.com/agnieszkaszczesna/RepoIMU (the deposit ships no SOURCE.md); "
        "data paper doi:10.1007/978-3-319-46418-3_45"
    )

    _common_meta: ClassVar[dict] = {
        "host": "GitHub, single upload, no releases and no versioning",
        "research_line": "Silesian University of Technology, Gliwice, with the Polish-Japanese Academy, Warsaw",
        "reference_side": False,
        "sensor_to_reference_frame": (
            "undocumented and non-identity — about 176 deg on pendulum segments 1-2, poorly identifiable "
            "because the motion is near-planar"
        ),
    }


class RepoImuPendulumAdapter(_RepoImuAdapter):
    name = "repoimu-pendulum"

    def trials(self) -> list[tuple[str, str]]:
        """The 22 pendulum trials, read from the file names of ``Pendulum/`` alone — no file is opened."""
        suffix = "_Segment_1.csv"
        stems = sorted(
            p.name[len("Pendulum_") : -len(suffix)] for p in (_root() / "Pendulum").glob(f"Pendulum_*{suffix}")
        )
        return [("pendulum", stem) for stem in stems]

    def load(self, subject: str, trial: str) -> Recording:
        if subject != "pendulum":
            raise KeyError(f"repoimu-pendulum has one apparatus, 'pendulum'; got {subject!r}")
        folder = _root() / "Pendulum"
        segments: dict[str, Segment] = {}
        overlaps: list[np.ndarray] = []
        for index in (1, 2, 3):
            name = f"Pendulum_{trial}_Segment_{index}.csv"
            raw, commas = _read(folder / name)
            segments[f"segment_{index}"] = Segment(
                name=f"segment_{index}",
                sensor=_stream(raw, commas, self.sensor_family, name),
                reference=_pose(raw, name),
                anatomical=None,  # a machine has no anatomy
                extra_sensors=[],
                mount=None,  # the mounting is nowhere described in the deposit
            )
            overlaps.append(np.round(raw[:, 0], 3))

        shared = len(set(overlaps[0]) & set(overlaps[1]) & set(overlaps[2]))
        static = "Static" in trial
        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=[
                Pair(
                    parent=f"segment_{a}",
                    child=f"segment_{b}",
                    joint="unknown",
                    joint_dof=None,
                    meta={
                        "inferred_joint": (
                            "revolute (hinge) is consistent with the measured axis dispersion of the relative "
                            "rotation (median 0.9-5.6 deg), but this is an inference, not a statement of the "
                            "deposit, and the near-planar motion makes it ill-conditioned"
                        ),
                        "joint_type_documented": False,
                    },
                )
                for a, b in ((1, 2), (2, 3))
            ],
            same_segment_pairs=[],  # one unit per segment
            contacts_t=None,  # no ground contact and no force plate
            rest_annotation_s=None,  # the deposit annotates no rest period
            trial_type="static" if static else "pendulum-swing",
            meta={
                "licence": self.licence,
                "source": self.source,
                "sensor_family": self.sensor_family,
                **self._common_meta,
                "segments_share_a_timestamp_grid": True,
                "shared_timestamps": shared,
                "shared_timestamps_note": (
                    "the three segment files carry identical timestamps on their overlap and differ only in "
                    "where they start and end (up to 0.4 s), so no inter-sensor clock offset exists or is "
                    "observable here; each segment is returned on its own deposited time vector, untrimmed"
                ),
                "reference_present": not static,
                "published_learned_estimator_number": (
                    "RING (bachhuber2024ringtmlr) reports 3.44 +- 0.06 deg on 'RepoIMU, 390 s, 90 Hz', which is "
                    "exactly Test02a_Trial1 plus Test02b_Trial1; RING's training is simulation-only, so this is "
                    "an evaluation-side entanglement, not training contamination"
                )
                if trial in RING_TRIALS
                else None,
                "defects": [
                    "impulsive reference glitches, concentrated on segment 3 (worst about 29 700 deg/s)",
                    "frozen (held) reference runs up to 200 samples = 2.2 s",
                    "the static trials publish an identically zero quaternion, i.e. no reference",
                    "a zero-order hold on the inertial channels: about 25.6 Hz accelerometer, 49.8 Hz gyroscope",
                    "gyroscope saturation at about 10.5 rad/s on segment 3",
                    "one comma decimal separator in Test02a_Trial1_Segment_3, normalised textually",
                    "repeated timestamps on Test02a/Test02b, so time is non-decreasing rather than increasing",
                    "no licence; no positions; undocumented mounting, joint type and sensor-to-reference frame",
                ],
            },
        )


class RepoImuTStickAdapter(_RepoImuAdapter):
    name = "repoimu-tstick"

    def trials(self) -> list[tuple[str, str]]:
        """The 29 wand trials, read from the file names of ``TStick/`` alone — no file is opened."""
        stems = sorted(p.name[len("TStick_") : -len(".csv")] for p in (_root() / "TStick").glob("TStick_*.csv"))
        return [("tstick", stem) for stem in stems]

    def load(self, subject: str, trial: str) -> Recording:
        if subject != "tstick":
            raise KeyError(f"repoimu-tstick has one apparatus, 'tstick'; got {subject!r}")
        name = f"TStick_{trial}.csv"
        raw, commas = _read(_root() / "TStick" / name)
        static = "Static" in trial
        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments={
                "tstick": Segment(
                    name="tstick",
                    sensor=_stream(raw, commas, self.sensor_family, name),
                    reference=_pose(raw, name),
                    anatomical=None,
                    extra_sensors=[],
                    mount=None,  # the mounting is nowhere described in the deposit
                )
            },
            pairs=[],  # one segment: this group cannot express a relative orientation at all
            same_segment_pairs=[],
            contacts_t=None,
            rest_annotation_s=None,  # the deposit annotates no rest period
            trial_type="static" if static else "wand-swing",
            meta={
                "licence": self.licence,
                "source": self.source,
                "sensor_family": self.sensor_family,
                **self._common_meta,
                "single_segment": True,
                "single_segment_note": (
                    "one sensor on a hand-held wand, so there is no connected pair and no relative orientation; "
                    "useful as a single-segment attitude case and as the same laboratory's undegraded control"
                ),
                "reference_present": not static,
                "sampling_note": (
                    "a genuine unduplicated 100 Hz on all channels, unlike the pendulum group, and the values "
                    "are effectively continuous rather than coarsely quantised — the two groups are not the "
                    "same sensor pipeline"
                ),
                "defects": [
                    "the static trial (Test01) publishes an identically zero quaternion, i.e. no reference",
                    "frozen reference runs up to 99 samples = 0.99 s (Test03_Trial1)",
                    "an apparent gyroscope clip at about 7.36 rad/s on some trials, at a non-standard full scale",
                    "no licence; no positions; undocumented mounting and sensor-to-reference frame",
                ],
            },
        )


ADAPTER = RepoImuPendulumAdapter()
EXTRA_ADAPTERS = [RepoImuTStickAdapter()]
