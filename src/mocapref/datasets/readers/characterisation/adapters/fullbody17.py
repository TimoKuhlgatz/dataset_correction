"""Hankyong full-body dataset (Lee & Lee, Hankyong National University): 17 Xsens IMUs, each with a rigid three-marker
cluster machined around its housing, so the optical system measures the sensor frame (checked through the
accelerometer: cluster and sensor frame agree to a per-subject median of 1.2-1.6 degrees, one forearm per subject
5-8 degrees off).

Files read (root from ``paths.dataset_dir("fullbody17")``): ``_extracted/S<nn>/[S<nn>/]S<nn>_<TrialName>.mat``, one
MATLAB 5 file per trial (S01 and S02 are nested one folder deeper, so the files are globbed recursively), with
``IMU.<label>.{Acc,Gyr,IMURot,IMUPos,Marker1,Marker2,Marker3}``, ``Info.Time``, ``GaitEvent.{RHS,LHS}``,
``Landmark.<joint>`` (twelve optical joint centres) and ``Segment.<label>`` (the Visual3D segment orientation).
Not read: the C3D files, scripts, the magnetometer and the vendor's magnetometer-aided fusion ``IMU.IMURotKF``.

Units: none converted (accelerometer m/s^2 with gravity, gyroscope rad/s, positions m); only the documented
full-scale ranges (+-2000 deg/s, +-160 m/s^2) are converted for ``SensorStream.range_gyr`` / ``range_acc``.

Deposit defects left in the data: the head IMU's Acc and Gyr are synthesised from the head marker cluster in nine
of twelve subjects (S01, S04, S06-S12) and in ``S02 Side_Squat_1`` (``meta["synthesised"] = True``); no trial
starts at rest (``rest_annotation_s`` is None); the static trials ``Npose`` and ``Tpose`` carry no inertial data
(``sensor=None``); the IMUs run at 60 Hz natively, cubic-spline resampled to 100 Hz; the optical stream is
low-passed near 15-20 Hz; one time shift per trial for all sensors; marker gap filling is not flagged; ``S11
Full_Squat_2`` has a single 30.7 rad/s spike on the head gyroscope at sample 0.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream
from ._frames import DEG  # only the full-scale ranges are converted; the signals need no conversion

DATASET = "fullbody-17imu"
FOLDER = "fullbody17"

SENSOR_SEGMENTS: dict[str, str] = {
    # deposit label -> the segment name used across the characterisation module (descriptor Table 4)
    "HE": "head",
    "TA": "thorax",
    "PV": "pelvis",
    "RSO": "right_shoulder",
    "LSO": "left_shoulder",
    "RUA": "right_upper_arm",
    "RFA": "right_forearm",
    "RHA": "right_hand",
    "LUA": "left_upper_arm",
    "LFA": "left_forearm",
    "LHA": "left_hand",
    "RTH": "right_thigh",
    "RSK": "right_shank",
    "RFT": "right_foot",
    "LTH": "left_thigh",
    "LSK": "left_shank",
    "LFT": "left_foot",
}

MODEL_SEGMENTS: dict[str, str] = {
    # the fifteen Visual3D model segments, which cover every sensor segment except the two shoulder blades
    label: name for label, name in SENSOR_SEGMENTS.items() if label not in ("RSO", "LSO")
}

# The sixteen connected pairs, from the descriptor's segment and joint tables. The joint word
# follows ``schema.Pair``; the anatomical name and the degrees of freedom sit beside it. The scapulothoracic
# and sternoclavicular span has no fixed joint centre, so its joint is "none" in the schema's sense.
PAIRS: list[tuple[str, str, str, int | None, str, str | None]] = [
    # parent, child, joint, dof, anatomical name, deposited joint centre in ``Landmark``
    ("head", "thorax", "ball", 3, "neck / cervical spine (compound)", None),
    ("thorax", "pelvis", "ball", 3, "lumbar spine (compound)", None),
    ("thorax", "right_shoulder", "none", 3, "sternoclavicular + scapulothoracic, gliding", None),
    ("thorax", "left_shoulder", "none", 3, "sternoclavicular + scapulothoracic, gliding", None),
    ("right_shoulder", "right_upper_arm", "ball", 3, "glenohumeral", "RSJC"),
    ("left_shoulder", "left_upper_arm", "ball", 3, "glenohumeral", "LSJC"),
    ("right_upper_arm", "right_forearm", "2-dof", 2, "elbow (humeroulnar + proximal radioulnar)", "REJC"),
    ("left_upper_arm", "left_forearm", "2-dof", 2, "elbow (humeroulnar + proximal radioulnar)", "LEJC"),
    ("right_forearm", "right_hand", "2-dof", 2, "wrist", "RWJC"),
    ("left_forearm", "left_hand", "2-dof", 2, "wrist", "LWJC"),
    ("pelvis", "right_thigh", "ball", 3, "hip", "RHJC"),
    ("pelvis", "left_thigh", "ball", 3, "hip", "LHJC"),
    ("right_thigh", "right_shank", "hinge", 1, "knee (hinge with laxity)", "RKJC"),
    ("left_thigh", "left_shank", "hinge", 1, "knee (hinge with laxity)", "LKJC"),
    ("right_shank", "right_foot", "2-dof", 2, "ankle (talocrural + subtalar)", "RAJC"),
    ("left_shank", "left_foot", "2-dof", 2, "ankle (talocrural + subtalar)", "LAJC"),
]
"""Units: degrees of freedom.
Provenance: read-from-dataset — the connected pairs of the deposit and each joint's degrees of freedom."""

STATIC_TRIALS = ("Npose", "Tpose")

# Which head streams are an optical reconstruction rather than a measurement: measured by three independent tests
# over all 264 movement trials of all twelve subjects (accelerometer-versus-cluster residual, the 60 Hz resampling
# fingerprint, and the deposit's own angular-velocity script).
SYNTHESISED_HEAD_SUBJECTS = ("S01", "S04", "S06", "S07", "S08", "S09", "S10", "S11", "S12")
SYNTHESISED_HEAD_TRIALS = (("S02", "Side_Squat_1"),)

# Vendor full-scale ranges of the Xsens MTw,  converted to SI here.
NATIVE_RATE_HZ = 60.0
"""The unit's native rate before the deposit's own resampling.

Units: Hz.
Provenance: read-from-dataset — read from the deposit."""

RANGE_GYR_RAD_S = 2000.0 * DEG
"""Units: rad/s.
Provenance: read-from-dataset — the hardware full scale the deposit states."""
RANGE_ACC_M_S2 = 160.0
"""Units: m/s².
Provenance: read-from-dataset — the hardware full scale the deposit states."""


def _trial_type(trial: str) -> str:
    if trial in STATIC_TRIALS:
        return "static"
    if trial.startswith("Walking"):
        return "walking"
    if trial.startswith("Running"):
        return "running"
    return "functional"


def _as_time_series(a: np.ndarray) -> np.ndarray:
    """MATLAB column-major (3, N) -> (N, 3); (3, 3, N) -> (N, 3, 3)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)
    if a.ndim == 3:
        return np.ascontiguousarray(np.transpose(a, (2, 0, 1)))
    raise ValueError(f"unexpected shape {a.shape}")


class FullBody17Adapter(Adapter):
    """The 12-subject, 17-sensor, 24-trial deposit; 288 trials, 264 of them with inertial data."""

    name = DATASET
    reference_kind = "housing"
    sensor_family = "Xsens MTw (Awinda)"
    licence = "CC BY 4.0 (Zenodo record 19632046; the article PDF is CC BY-NC-ND and is not the data licence)"
    source = "doi:10.5281/zenodo.19632046 — see <dataset root>/SOURCE.md"

    def __init__(self, root: os.PathLike | str | None = None) -> None:
        self.root = dataset_dir(FOLDER) if root is None else root
        self._index: dict[tuple[str, str], str] | None = None

    # -- discovery -------------------------------------------------------------------------------------

    def _build_index(self) -> dict[tuple[str, str], str]:
        if self._index is None:
            index: dict[tuple[str, str], str] = {}
            pattern = os.path.join(str(self.root), "_extracted", "S*", "**", "S*_*.mat")
            for path in sorted(glob.glob(pattern, recursive=True)):
                base = os.path.basename(path)[: -len(".mat")]
                m = re.match(r"^(S\d\d)_(.+)$", base)
                if m is None or m.group(2) == "SubjectInfo":
                    continue
                index[(m.group(1), m.group(2))] = path
            self._index = index
        return self._index

    def trials(self) -> list[tuple[str, str]]:
        return sorted(self._build_index())

    # -- loading ---------------------------------------------------------------------------------------

    def load(self, subject: str, trial: str) -> Recording:
        from scipy.io import loadmat  # imported here so ``trials()`` costs nothing

        path = self._build_index()[(subject, trial)]
        mat = loadmat(
            path,
            struct_as_record=False,
            squeeze_me=True,
            variable_names=["IMU", "Info", "GaitEvent", "Landmark", "Segment"],
        )
        imu = mat["IMU"]
        t = np.asarray(mat["Info"].Time, dtype=float)
        rate_hz = float(np.asarray(mat["Info"].SamplingRate))
        model = mat.get("Segment")
        landmark = mat.get("Landmark")

        head_synthesised = subject in SYNTHESISED_HEAD_SUBJECTS or (subject, trial) in SYNTHESISED_HEAD_TRIALS
        has_inertial = trial not in STATIC_TRIALS

        segments: dict[str, Segment] = {}
        for label, seg_name in SENSOR_SEGMENTS.items():
            if label not in imu._fieldnames:
                continue
            s = getattr(imu, label)
            fields = set(s._fieldnames)

            sensor = None
            if {"Acc", "Gyr"} <= fields:
                synthesised = seg_name == "head" and head_synthesised
                sensor = SensorStream(
                    t=t,
                    gyr=_as_time_series(s.Gyr),  # rad/s as deposited, no conversion
                    acc=_as_time_series(s.Acc),  # m/s² as deposited, gravity included, no conversion
                    rate_hz=rate_hz,
                    family=self.sensor_family,
                    range_gyr=RANGE_GYR_RAD_S,
                    range_acc=RANGE_ACC_M_S2,
                    native_rate_hz=NATIVE_RATE_HZ,
                    meta={
                        "deposit_label": label,
                        "deposited_units": {"acc": "m/s^2 (gravity included)", "gyr": "rad/s"},
                        "synthesised": bool(synthesised),
                        "synthesised_evidence": (
                            "the head Acc/Gyr are a reconstruction "
                            "from the head marker cluster in this subject; measures must skip this stream"
                        )
                        if synthesised
                        else None,
                        "resampled": "60 Hz native, cubic spline to 100 Hz",
                    },
                )

            reference = None
            if "IMURot" in fields:
                markers = None
                marker_names = None
                if {"Marker1", "Marker2", "Marker3"} <= fields:
                    markers = np.stack(
                        [_as_time_series(s.Marker1), _as_time_series(s.Marker2), _as_time_series(s.Marker3)],
                        axis=1,
                    )  # (T, 3, 3) metres
                    marker_names = [f"{label}_cluster_{i}" for i in (1, 2, 3)]
                reference = Pose(
                    t=t,
                    R=_as_time_series(s.IMURot),  # sensor(cluster) frame -> optical world
                    p=_as_time_series(s.IMUPos) if "IMUPos" in fields else np.full((len(t), 3), np.nan),
                    rate_hz=rate_hz,
                    kind="housing",
                    markers=markers,
                    marker_names=marker_names,
                    meta={
                        "construction": (
                            "the deposit's own three-marker cluster pose (IMURot/IMUPos): the cluster is "
                            "machined around the housing and its frame is aligned with the IMU frame by "
                            "construction (descriptor §3); the three markers ship alongside so the "
                            "cluster-rigidity route of docs/characterisation.md §2.9 can be recomputed"
                        ),
                        "cluster_to_sensor_offset_deg": (
                            "per-subject median 1.19–1.58°, 90th percentile 2.7–4.2°, and one forearm per "
                            "subject at 5.2–8.4°"
                        ),
                        "low_pass": "optical stream filtered near 15–20 Hz",
                    },
                )

            anatomical = None
            if model is not None and label in MODEL_SEGMENTS and label in model._fieldnames:
                seg_model = getattr(model, label)
                if "SegRot" in set(seg_model._fieldnames):
                    anatomical = Pose(
                        t=t,
                        R=_as_time_series(seg_model.SegRot),
                        p=_as_time_series(seg_model.CoMPos),
                        rate_hz=rate_hz,
                        kind="skeletal-fit",
                        meta={
                            "construction": (
                                "Visual3D model segment orientation from the 45 anatomical markers, as "
                                "deposited; the origin is the segment centre of mass. A second, model-based "
                                "reference — not the depositors' housing reference and not this adapter's "
                                "own construction"
                            ),
                        },
                    )

            segments[seg_name] = Segment(
                name=seg_name,
                sensor=sensor,
                reference=reference,
                anatomical=anatomical,
                mount="3D-printed marker-cluster holder, Velcro-strapped over an optical motion-capture suit",
            )

        pairs: list[Pair] = []
        for parent, child, joint, dof, anatomical_name, landmark_name in PAIRS:
            if parent not in segments or child not in segments:
                continue
            centre = None
            if landmark is not None and landmark_name is not None and landmark_name in landmark._fieldnames:
                centre = _as_time_series(getattr(landmark, landmark_name))  # metres, optical time base
            pairs.append(
                Pair(
                    parent=parent,
                    child=child,
                    joint=joint,
                    joint_dof=dof,
                    joint_centre_world=centre,
                    joint_centre_t=t if centre is not None else None,
                    meta={
                        "joint_name": anatomical_name,
                        "joint_centre_source": (
                            f"deposited optical landmark {landmark_name} (Visual3D model)"
                            if centre is not None
                            else "none deposited for this pair"
                        ),
                        "head_stream_synthesised": bool(head_synthesised) if "head" in (parent, child) else False,
                    },
                )
            )

        contacts = None
        if "GaitEvent" in mat:
            events = mat["GaitEvent"]
            times = []
            for field in ("RHS", "LHS"):
                if field in events._fieldnames:
                    arr = np.atleast_2d(np.asarray(getattr(events, field), dtype=float))
                    if arr.size:
                        times.append(arr[:, 0])  # column 0 is the event time in seconds, column 1 the frame
            if times:
                contacts = np.sort(np.concatenate(times))

        return Recording(
            dataset=DATASET,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],  # one sensor per segment; no same-segment pair exists in this deposit
            contacts_t=contacts,
            rest_annotation_s=None,  # the ≈ 20 s standing prefix is NOT stillness
            trial_type=_trial_type(trial),
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "Zenodo",
                "sensor_family": self.sensor_family,
                "reference": "OptiTrack Prime 13w, 12 cameras, 100 Hz, 3 markers machined around every housing",
                "path": path,
                "has_inertial": has_inertial,
                "standing_prefix_s": (
                    "≈ 20.5 s (first heel strike 19.97–21.61 s in all 144 locomotion trials) of standing and "
                    "stepping onto the treadmill — reproducible but NOT still; no rest annotation"
                )
                if _trial_type(trial) in ("walking", "running")
                else None,
                "head_stream_synthesised": bool(head_synthesised),
                "time_alignment": (
                    "one gyroscope-to-cluster shift per trial, not per sensor; measured residual below one "
                    "sample on 95–100 % of sensor-trials"
                ),
                "known_defects": [
                    "head Acc/Gyr synthesised from the optical cluster in nine of twelve subjects (§8.1)",
                    "no still stretch with inertial data anywhere in the deposit (§4, §8.4)",
                    "static trials ship no Acc/Gyr (§8.4)",
                    "60 Hz native resampled to 100 Hz; nothing above ≈ 30 Hz is genuine (§8.2)",
                    (
                        "optical stream low-pass filtered near 15–20 Hz; its derivative is not an "
                        "angular-rate reference above ≈ 4 Hz (§8.3)"
                    ),
                    "marker gap filling under 1 s is unflagged per sample (§8.5)",
                    "S11 Full_Squat_2 head gyroscope: single-sample 30.7 rad/s spike at sample 0 (§8.1)",
                ],
            },
        )


ADAPTER = FullBody17Adapter()
