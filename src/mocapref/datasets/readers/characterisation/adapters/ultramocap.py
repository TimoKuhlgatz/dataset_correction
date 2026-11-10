"""ULTRA-MoCap upper limb (Central Florida; Fritsche et al., Scientific Data 2026, doi:10.1038/s41597-026-06687-5):
six Delsys Trigno Avanti IMUs on the right arm, OpenSim skeletal fit. Deposit figshare
doi:10.6084/m9.figshare.28751156.v1, CC BY 4.0. Thirteen participants, sixteen dynamic trials each, five exercises.

File read: ``_extracted/ULTra-MoCap-processed/All_subjects_data.h5`` (the processed release), groups
``subject_<1..13>/<AS|CB|EF|ER|OR>/<speed>``, each a 2-D dataset with ``column_names``. Values are stored as byte
strings and read as ``np.where(raw == b"", b"nan", raw).astype(float)``. Only inertial channels 1-6 are read; the
extra channel groups of the processed file are listed in ``meta["unmapped_channel_groups"]``.

Units: accelerometer mm/s^2 -> m/s^2 (specific force); gyroscope deg/s -> rad/s; reference joint angles deg ->
rad; time in s, kept as deposited. Rate 100 Hz (the Vicon grid); native inertial rate 148.148 Hz.

Reference (``kind="skeletal-fit"``): the authors' OpenSim inverse-kinematics joint angles, composed into segment
rotations along the arm chain:
    R(upper arm) = Rz(arm_flex_r) Rx(arm_add_r) Ry(arm_rot_r)
    R(forearm)   = R(upper arm) Rz(elbow_flex_r) Ry(pro_sup_r)
    R(hand)      = R(forearm) Rz(wrist_flex_r) Rx(wrist_dev_r)
The world is the model thorax (``Pose.meta["frame"]``), so relative orientations between arm segments are valid
but gravity direction is not available. The axis convention is the standard OpenSim arm convention, assumed and not
read from the model file (``Pose.meta["axis_convention_assumed"]``); the deposited angles are also carried on
``Pair.meta["reference_angles_rad"]``. ``Pose.p`` is NaN.

Sensors: channel 1 hand, 2 wrist (forearm, extra), 3 forearm, 4 biceps (upper arm), 5 triceps (upper arm, extra),
6 deltoid (upper arm, extra). ``same_segment_pairs``: forearm (3, 2) and humerus (4, 5); the deltoid does not move
rigidly with the upper arm and is not declared a pair.

Deposit defects left in the data: the processed file drops the first 20 s of every trial longer than 40 s (the
time column keeps its start; ``meta["dropped_prefix_s"]``); 2000 -> 100 Hz reduction by index picking without
anti-alias filter; one empty inertial row at the end of every trial (kept as NaN, ``meta["trailing_nan_samples"]``;
the loader drops it); an uncorrected inertial-to-reference lag of about 50 ms; the reference expresses only two
degrees of freedom at elbow and wrist; no rest period and no static inertial recording; the sensor-to-segment
rotation is undocumented. The shoulder has a reference but no proximal sensor (``meta["shoulder_reference_only"]``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

RATE_HZ = 100.0
"""The processed file's grid: ``dt`` is exactly 0.01 s on all 208 trials.

Units: Hz.
Provenance: read-from-dataset — the processed file's grid."""

NATIVE_RATE_HZ = 148.148
"""The true Delsys Trigno inertial update rate, recovered from the knot spectrum.

Units: Hz.
Provenance: read-from-dataset — read from the deposit."""

MM_S2_TO_M_S2 = 1e-3
"""Units: (m/s²) per (mm/s²).
Provenance: physical — an exact unit conversion."""
DEG = np.pi / 180.0
"""Units: rad per degree.
Provenance: physical — an exact unit conversion."""

CLOCK_OFFSET_S_MEASURED = 0.050
"""Review §2: the inertial stream lags the optical reference by a median +5 frames over 39 elbow trials.

Units: s.
Provenance: read-from-dataset — read from the deposit."""

REFERENCE_ACCURACY_LIMIT_DEG = (1.0, 5.0)
"""Review §3, from marker non-rigidity alone, before the OpenSim model's own residual.

Units: degrees.
Provenance: read-from-dataset — read from the deposit."""

N_SENSOR_CHANNELS = 6
"""Channels 1–6 are the six released sensors; they exist in all three column sets of the processed file.

Units: channels.
Provenance: read-from-dataset — how many sensors the deposit releases."""

#: The deposit's group keys → the exercise in plain words, and the speed keys → the level in plain words.
EXERCISES = {
    "AS": ("arm-swing", {"S": "slow", "N": "normal", "F": "fast", "VF": "very-fast"}),
    "CB": ("crossbody-reach", {"S": "slow", "N": "normal", "F": "fast"}),
    "EF": ("elbow-flexion", {"S": "slow", "N": "normal", "F": "fast"}),
    "ER": ("shoulder-rotation", {"S": "slow", "N": "normal", "F": "fast"}),
    "OR": ("overhead-reach", {"90": "90", "180": "180", "M": "max"}),
}

#: segment → (primary channel, further channels on the same segment), channels as in descriptor Table 3.
SEGMENT_CHANNELS = {
    "hand": (1, ()),
    "forearm": (3, (2,)),
    "upper_arm": (4, (5, 6)),
}
"""Units: channel index.
Provenance: read-from-dataset — which released channel sits on which segment."""

CHANNEL_PLACEMENT = {1: "hand", 2: "wrist", 3: "forearm", 4: "biceps", 5: "triceps", 6: "deltoid"}
"""Units: 1.
Provenance: read-from-dataset — the placement the deposit states for each channel."""

#: (segment, index a, index b) into ``[sensor] + extra_sensors``. The deltoid (index 2 on the upper arm) is
#: deliberately absent: a check of the deposit measured it does not move rigidly with the humerus (R² 0.778–0.811).
SAME_SEGMENT_PAIRS = [("forearm", 0, 1), ("upper_arm", 0, 1)]

#: (parent, child, joint, dof, the reference angles the model expresses for it).
PAIRS = [
    ("upper_arm", "forearm", "2-dof", 2, ("elbow_flex_r", "pro_sup_r")),
    ("forearm", "hand", "2-dof", 2, ("wrist_flex_r", "wrist_dev_r")),
]

SHOULDER_ANGLES = ("arm_flex_r", "arm_add_r", "arm_rot_r")
ELBOW_ANGLES = ("elbow_flex_r", "pro_sup_r")
WRIST_ANGLES = ("wrist_flex_r", "wrist_dev_r")


def _h5_path() -> Path:
    return dataset_dir("ultramocap") / "_extracted" / "ULTra-MoCap-processed" / "All_subjects_data.h5"


def _subject_id(group: str) -> str:
    """``subject_8`` → ``P08``, the identifier the descriptor and the raw release use."""
    return f"P{int(group.split('_')[1]):02d}"


def _subject_group(subject: str) -> str:
    return f"subject_{int(subject[1:])}"


def _trial_name(exercise_key: str, speed_key: str) -> str:
    exercise, speeds = EXERCISES[exercise_key]
    return f"{exercise}-{speeds[speed_key]}"


def _rot_x(angle: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
    return np.stack([o, z, z, z, c, -s, z, s, c], axis=-1).reshape(*angle.shape, 3, 3)


def _rot_y(angle: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(*angle.shape, 3, 3)


def _rot_z(angle: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
    return np.stack([c, -s, z, s, c, z, z, z, o], axis=-1).reshape(*angle.shape, 3, 3)


def _chain(*rotations: np.ndarray) -> np.ndarray:
    out = rotations[0]
    for r in rotations[1:]:
        out = np.einsum("nij,njk->nik", out, r)
    return out


class UltraMocapAdapter(Adapter):
    """The ULTRA-MoCap processed release as the characterisation schema sees it."""

    name = "ultramocap-upper-limb"
    reference_kind = "skeletal-fit"
    sensor_family = "Delsys Trigno Avanti"
    licence = "CC BY 4.0"
    source = "eval_datasets/ultramocap/SOURCE.md — doi:10.6084/m9.figshare.28751156.v1"

    def __init__(self) -> None:
        self.path = _h5_path()

    # ------------------------------------------------------------------ index

    @staticmethod
    @lru_cache(maxsize=1)
    def _index() -> dict[str, dict[str, tuple[str, str]]]:
        """Per subject, ``trial name → (exercise key, speed key)``, from the HDF5 group tree.

        Only the group tree is walked — no dataset is read, so no signal data is touched. The tree is small
        and the result is cached in memory for the life of the process.
        """
        import h5py

        path = _h5_path()
        out: dict[str, dict[str, tuple[str, str]]] = {}
        if not path.exists():
            return out
        with h5py.File(path, "r") as handle:
            for group in handle:
                subject = _subject_id(group)
                trials: dict[str, tuple[str, str]] = {}
                for exercise_key in handle[group]:
                    if exercise_key not in EXERCISES:
                        continue
                    for speed_key in handle[group][exercise_key]:
                        if speed_key in EXERCISES[exercise_key][1]:
                            trials[_trial_name(exercise_key, speed_key)] = (exercise_key, speed_key)
                out[subject] = trials
        return out

    def trials(self) -> list[tuple[str, str]]:
        """All 208 (subject, trial) identifiers; only the HDF5 group tree is walked."""
        index = self._index()
        return [
            (subject, trial) for subject in sorted(index, key=lambda s: int(s[1:])) for trial in sorted(index[subject])
        ]

    # ------------------------------------------------------------------ load

    def load(self, subject: str, trial: str) -> Recording:
        import h5py

        index = self._index().get(subject, {})
        if trial not in index:
            raise KeyError(f"{self.name}: no such trial {subject}/{trial}")
        exercise_key, speed_key = index[trial]
        exercise, speeds = EXERCISES[exercise_key]

        with h5py.File(self.path, "r") as handle:
            dataset = handle[f"{_subject_group(subject)}/{exercise_key}/{speed_key}"]
            columns = [c.decode() if isinstance(c, bytes) else str(c) for c in dataset.attrs["column_names"]]
            raw = dataset[:]
        values = np.where(raw == b"", b"nan", raw).astype(float)  # variable-length byte strings, deposit-side
        del raw
        column = {name: i for i, name in enumerate(columns)}
        t = values[:, column["time"]]

        streams: dict[int, SensorStream] = {}
        trailing_nan = 0
        for channel in range(1, N_SENSOR_CHANNELS + 1):
            acc = values[:, [column[f"ACC{a}{channel}"] for a in "XYZ"]] * MM_S2_TO_M_S2
            gyr = values[:, [column[f"GYRO{a}{channel}"] for a in "XYZ"]] * DEG
            trailing_nan = max(trailing_nan, int(np.isnan(acc).any(axis=1).sum()))
            streams[channel] = SensorStream(
                t=t,
                gyr=gyr,
                acc=acc,
                rate_hz=RATE_HZ,
                family=self.sensor_family,
                range_gyr=None,  # full scale not documented anywhere in the deposit or the descriptor
                range_acc=None,
                native_rate_hz=NATIVE_RATE_HZ,
                meta={
                    "channel": channel,
                    "placement": CHANNEL_PLACEMENT[channel],
                    "deposited_accelerometer": "mm/s² specific force, gravity included; factor 1e-3",
                    "deposited_gyroscope": "°/s; factor π/180",
                    "magnetometer_shipped": False,
                    "sensor_frame_convention": None,  # documented nowhere
                    "antialias_filtered": False,
                    "resampled_from_native_rate": True,
                },
            )

        angles = {
            name: values[:, column[name]] * DEG
            for name in (*SHOULDER_ANGLES, *ELBOW_ANGLES, *WRIST_ANGLES)
            if name in column
        }
        poses = self._poses(t, angles)

        segments: dict[str, Segment] = {}
        for segment, (primary, extra) in SEGMENT_CHANNELS.items():
            segments[segment] = Segment(
                name=segment,
                sensor=streams[primary],
                reference=poses.get(segment),
                anatomical=None,  # the processed file ships no marker positions at all
                mount=None,  # no strap, tape, holder or bolt is named anywhere in the deposit
                extra_sensors=[streams[c] for c in extra],
            )

        pairs = [
            Pair(
                parent=parent,
                child=child,
                joint=joint,
                joint_dof=dof,
                joint_centre_world=None,  # no joint centre and no position of any kind ships
                joint_centre_t=None,
                meta={
                    "reference_dof_expressed": 2,
                    "third_dof": "constrained to zero by the model, not measured",
                    "reference_angle_names": list(names),
                    "reference_angles_rad": {n: angles[n] for n in names if n in angles},
                    "reference_angles_t": t,
                },
            )
            for parent, child, joint, dof, names in PAIRS
        ]

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=list(SAME_SEGMENT_PAIRS),
            contacts_t=None,  # upper-limb exercises, no ground contact and no force plate
            rest_annotation_s=None,  # no rest period is annotated, and a check of the deposit measured that none exists
            trial_type=exercise,
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "figshare",
                "release": "processed HDF5 (All_subjects_data.h5); the two raw zips are not on disk",
                "exercise": exercise,
                "speed_level": speeds[speed_key],
                "deposit_keys": {"subject": _subject_group(subject), "exercise": exercise_key, "speed": speed_key},
                "dropped_prefix_s": float(t[0]) if len(t) else float("nan"),
                "dropped_prefix_note": (
                    "the processed release drops the first 2000 rows (20 s) of every trial with more than 4000 "
                    "rows (the authors' batch_processH5.py); the deposited time column keeps its "
                    "start, so this value is measured from t[0] and is 20.0 s for P03–P12 and 0.0 s for P01, "
                    "P02 and P13. About 40 % of the recorded data is not in this file."
                ),
                "antialias_filtered": False,
                "decimation": "2000 → 100 Hz by index picking, no anti-alias filter",
                "trailing_nan_samples": trailing_nan,
                "trailing_nan_note": "the deposit's last row is empty in every inertial column; it is kept as NaN",
                "unmapped_channel_groups": sorted(
                    {
                        name[-2:] if name[-2:].isdigit() else name[-1]
                        for name in columns
                        if name.startswith(("ACC", "GYRO"))
                        and (name[-2:] if name[-2:].isdigit() else name[-1]) not in "123456"
                    }
                ),
                "column_set_size": len(columns),
                "clock_offset_s_measured": CLOCK_OFFSET_S_MEASURED,
                "clock_offset_applied": False,
                "clock_offset_sign": "the inertial stream lags the optical reference",
                "reference_accuracy_limit_deg": REFERENCE_ACCURACY_LIMIT_DEG,
                "reference_is_model_fit": True,
                "sensor_to_segment_rotation_measurable": False,
                "shoulder_reference_only": (
                    "arm_flex_r, arm_add_r and arm_rot_r are in the reference but the thorax carries no sensor, "
                    "so the shoulder is not a pair"
                ),
                "deltoid_sensor_index": ("upper_arm", 2),
                "deltoid_note": "channel 6 does not move rigidly with the humerus (R² 0.778–0.811)",
                "starts_at_rest": False,
                "excitation_note": (
                    "the wrist is never excited (constant-map R² 0.94–0.97 in all five exercises); the elbow "
                    "only in the 39 elbow-flexion trials (R² 0.504 against 0.91–0.94 elsewhere)"
                ),
            },
        )

    def _poses(self, t: np.ndarray, angles: dict[str, np.ndarray]) -> dict[str, Pose]:
        """Segment rotations in the model thorax frame from the deposited joint angles (see the docstring)."""
        if not all(name in angles for name in (*SHOULDER_ANGLES, *ELBOW_ANGLES, *WRIST_ANGLES)):
            return {}
        upper_arm = _chain(_rot_z(angles["arm_flex_r"]), _rot_x(angles["arm_add_r"]), _rot_y(angles["arm_rot_r"]))
        forearm = _chain(upper_arm, _rot_z(angles["elbow_flex_r"]), _rot_y(angles["pro_sup_r"]))
        hand = _chain(forearm, _rot_z(angles["wrist_flex_r"]), _rot_x(angles["wrist_dev_r"]))
        position = np.full((len(t), 3), np.nan)  # the processed file ships no segment position of any kind

        def pose(rotation: np.ndarray, names: tuple[str, ...]) -> Pose:
            return Pose(
                t=t,
                R=rotation,
                p=position,
                rate_hz=RATE_HZ,
                kind="skeletal-fit",
                markers=None,  # the processed file ships no marker positions
                marker_names=None,
                meta={
                    "construction": "composition of the model's joint rotations from the deposited IK angles",
                    "frame": "model thorax, not the laboratory — gravity direction is not available here",
                    "axis_convention_assumed": True,
                    "axis_convention": (
                        "shoulder Z-X-Y (arm_flex, arm_add, arm_rot), elbow Z (elbow_flex) then Y (pro_sup), "
                        "wrist Z (wrist_flex) then X (wrist_dev), orthogonal axes; the .osim that defines the "
                        "joint frames ships only in the raw zips, which are not on disk"
                    ),
                    "angles_used": list(names),
                    "reference_accuracy_limit_deg": REFERENCE_ACCURACY_LIMIT_DEG,
                    "dof_expressed_per_joint": 2,
                    "on_housing": False,
                },
            )

        return {
            "upper_arm": pose(upper_arm, SHOULDER_ANGLES),
            "forearm": pose(forearm, SHOULDER_ANGLES + ELBOW_ANGLES),
            "hand": pose(hand, SHOULDER_ANGLES + ELBOW_ANGLES + WRIST_ANGLES),
        }


ADAPTER = UltraMocapAdapter()
