"""Hip-ROM-Y (Yanshan): eight inertial units on the lower body plus optical motion capture, standing hip exercises,
16 participants. Deposit: Science Data Bank 10.57760/sciencedb.27443 V5, CC BY 4.0.

The accelerometer has gravity removed by the vendor firmware: every sample is the gravity-compensated linear
acceleration (the authors' decoder calls it "acceleration without gravity"). Every stream carries
``SensorStream.meta["gravity_removed"] = True``; gravity is not added back here (error 22 restores it from the
vendor quaternion, which is carried in ``SensorStream.meta["vendor_quaternion_wxyz"]`` and is a nine-axis,
magnetometer-aided firmware output, never an input or a reference). The gyroscope is unaffected.

Files read (per participant, under ``_extracted/hip_ROM_dataset/hip_ROM_dataset``):
- ``Final_dataset/P<k>_dataset.csv``: frame counter, then ``ax,ay,az,gx,gy,gz,w,x,y,z,rx,ry,rz`` per sensor, the
  six released hip Euler angles and the action label ``pose``;
- ``Angle_data/P<k>_angle.csv``: the XINGYING segment export (parent-relative and world pose per segment); row k
  matches row k of the merged product.

Reference (``kind="skeletal-fit"``): ``R`` from the world quaternion ``Q?ToGlobal`` (segment to world, z up) and
``p`` from ``X/Y/ZToGlobal``: a marker-driven segment-model fit (19 skin markers, NOKOV at 50 Hz), not a cluster
on a housing. ``Pose.markers`` is None.

Units: gyroscope deg/s -> rad/s; accelerometer already m/s^2; positions mm -> m; time = frame index / 50 Hz for
both systems (the raw optical Unix-millisecond stamps are in ``Pose.meta["timestamp_unix_ms"]``).

Deposit defects left in the data: the undocumented ``Other data/`` folder (permuted participant labels) is not
read (``Recording.meta["not_exposed"]``); optical glitches up to 165 degrees per frame (indices of jumps above
10 degrees in ``Pose.meta["jump_gt_10deg_idx"]``); unit 6 (right thigh) has impossible single-frame rates
(``SensorStream.meta["impossible_rate_idx"]``); missing optical frames are exact zeros in the deposit and become
NaN (``Pose.meta["invalid_idx"]``); the documented product keeps only the first 217-457 s of each session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .. import paths
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

DEG2RAD = np.pi / 180.0
"""Units: rad per degree.
Provenance: physical — an exact unit conversion (physical)."""
MM2M = 1e-3
"""Units: m per mm.
Provenance: physical — an exact unit conversion (physical)."""
RATE_HZ = 50.0
"""Units: Hz.
Provenance: read-from-dataset — the rate the deposit is written at (data-free: the deposit's documented and measured rate for both systems)."""
DEG_PER_S_FULL_SCALE = 2000.0
"""Units: °/s.
Provenance: read-from-dataset — the unit's full-scale range as the vendor states it (documented: IM948 gyroscope full scale (data-descriptor paper))."""
JUMP_DEG = 10.0
"""Units: degrees.
Provenance: data-free — data-free."""
IMPOSSIBLE_RATE_DEG_S = 500.0
"""Units: °/s.
Provenance: read-from-dataset — read from the deposit."""

#: The eight optical model segments in file order; the last is the pelvis in a second laboratory convention.
OPTICAL_SEGMENTS = ["Pelvis", "R.Thigh", "L.Thigh", "R.Shank", "L.Shank", "R.Foot", "L.Foot", "PelvisWRTLab"]

#: segment name -> (primary sensor index 1..8, further sensor indices, optical segment index or None)
LAYOUT: dict[str, tuple[int, tuple[int, ...], int | None]] = {
    "clavicle": (1, (2,), None),
    "pelvis": (3, (4,), 0),
    "right_thigh": (6, (), 1),
    "left_thigh": (5, (), 2),
    "right_shank": (8, (), 3),
    "left_shank": (7, (), 4),
}
"""Units: channel index.
Provenance: read-from-dataset — which deposited channel sits on which segment."""

#: (parent, child, joint type, degrees of freedom, note)
PAIRS: list[tuple[str, str, str, int | None, str]] = [
    ("pelvis", "right_thigh", "ball", 3, "right hip"),
    ("pelvis", "left_thigh", "ball", 3, "left hip"),
    (
        "right_thigh",
        "right_shank",
        "hinge",
        1,
        "right knee, hinge-dominant: one large and two small degrees of freedom",
    ),
    ("left_thigh", "left_shank", "hinge", 1, "left knee, hinge-dominant: one large and two small degrees of freedom"),
    (
        "clavicle",
        "pelvis",
        "unknown",
        None,
        (
            "the whole trunk and lumbar spine, not one joint; the optical model has no trunk segment, so this "
            "pair carries no reference"
        ),
    ),
]
"""Units: degrees of freedom.
Provenance: read-from-dataset — the connected pairs of the deposit and each joint's degrees of freedom."""

MOUNT = "elastic strap with an adjustable buckle"  # documented in the data-descriptor paper

#: Documented placement per sensor index. The paper names the pelvis pair as "left and right anterior superior
#: iliac spines" without saying which index is which, so both are given the same undifferentiated site.
_SITE = {
    1: "mid-clavicle (upper-body harness)",
    2: "mid-clavicle (upper-body harness)",
    3: "anterior superior iliac spine (side not distinguished in the deposit)",
    4: "anterior superior iliac spine (side not distinguished in the deposit)",
    5: "distal left thigh, just above the femur-patella junction",
    6: "distal right thigh, just above the femur-patella junction",
    7: "proximal left shank, just below the tibia-patella junction",
    8: "proximal right shank, just below the tibia-patella junction",
}
"""Units: 1.
Provenance: read-from-dataset — the placement the deposit states for each channel."""


def _root() -> Path:
    return paths.dataset_dir("hip_rom_y") / "_extracted" / "hip_ROM_dataset" / "hip_ROM_dataset"


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True in a boolean mask."""
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return [(int(a), int(b - a)) for a, b in zip(idx[0::2], idx[1::2], strict=True)]


class HipRomYAdapter(Adapter):
    name = "hip-rom-y"
    reference_kind = "skeletal-fit"
    sensor_family = "IM948 (ChenYi Technology)"
    licence = "CC BY 4.0 (Science Data Bank record; the data-descriptor paper is CC BY-NC-ND 4.0)"
    source = "10.57760/sciencedb.27443 (see <eval_datasets>/hip_rom_y/SOURCE.md)"

    def trials(self) -> list[tuple[str, str]]:
        """One continuous recording per participant; the twelve action types are labels inside it, not trials.

        Read from the file names of ``Final_dataset/`` alone — no file is opened.
        """
        root = _root() / "Final_dataset"
        subjects = sorted(
            (p.name[: -len("_dataset.csv")] for p in root.glob("P*_dataset.csv")),
            key=lambda s: int(s[1:]),
        )
        return [(s, "documented-recording") for s in subjects]

    def load(self, subject: str, trial: str) -> Recording:
        if trial != "documented-recording":
            raise KeyError(f"hip-rom-y has one trial per subject, 'documented-recording'; got {trial!r}")
        root = _root()
        merged = pd.read_csv(root / "Final_dataset" / f"{subject}_dataset.csv")
        n = len(merged)
        angle = pd.read_csv(
            root / "Angle_data" / f"{subject}_angle.csv", skiprows=19, header=None, low_memory=False, nrows=n
        )
        if len(angle) != n:
            raise ValueError(
                f"{subject}: the optical export has {len(angle)} rows but the merged product has {n}; the deposit's "
                "row-for-row pairing does not hold"
            )

        t = np.arange(n) / RATE_HZ
        stamp_ms = angle.iloc[:, 1].to_numpy(float)
        zero_stamp = np.flatnonzero(stamp_ms <= 0)

        poses = {k: self._pose(angle, k, t, stamp_ms, zero_stamp) for k in range(len(OPTICAL_SEGMENTS))}

        segments: dict[str, Segment] = {}
        for name, (primary, extra, optical) in LAYOUT.items():
            streams = [self._stream(merged, i, t) for i in (primary, *extra)]
            segments[name] = Segment(
                name=name,
                sensor=streams[0],
                reference=poses[optical] if optical is not None else None,
                anatomical=None,
                extra_sensors=streams[1:],
                mount=MOUNT,
            )

        pairs = [Pair(parent=a, child=b, joint=j, joint_dof=d, meta={"note": note}) for a, b, j, d, note in PAIRS]

        label = merged["pose"].to_numpy(float)
        lead = int(np.argmax(label != 0)) if (label != 0).any() else n

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            # both sensors of each rigid site index into [sensor] + extra_sensors
            same_segment_pairs=[("pelvis", 0, 1), ("clavicle", 0, 1)],
            contacts_t=None,  # standing exercises: no force plates and no contact annotation in the deposit
            rest_annotation_s=None,  # the deposit documents no rest period; see meta["initial_zero_label_s"]
            trial_type="standing-hip-range-of-motion",
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "Science Data Bank (CNIC, Chinese Academy of Sciences), record version V5",
                "sensor_family": self.sensor_family,
                "gravity_removed": True,
                "gravity_removed_note": (
                    "the deposited accelerometer is gravity-compensated linear acceleration; measures that need "
                    "specific force must skip this dataset and gravity is never added back"
                ),
                "reference_side": True,
                "reference_note": (
                    "nineteen skin markers (Helen Hayes), twelve-camera NOKOV at 50 Hz, XINGYING segment model; "
                    "no marker on any sensor housing, so every derived number is reference-side"
                ),
                "action_label": label,
                "action_label_note": "0 = rest or transition, 1..12 = the twelve action types, as the deposit labels them",
                "initial_zero_label_s": float(lead / RATE_HZ),
                "released_hip_euler_deg": merged[["rxt", "ryt", "rzt", "lxt", "lyt", "lzt"]].to_numpy(float),
                "released_hip_euler_note": (
                    "the extrinsic x-y-z decomposition of the thigh-to-pelvis quaternion, to floating-point "
                    "round-off; carried for provenance only"
                ),
                "not_exposed": (
                    "the undocumented 'Other data/' folder holds the full 631-1291 s session but its participant "
                    "labels are permuted (fifteen of sixteen files carry another participant's number); this "
                    "adapter reads only the documented Final_dataset/Angle_data pair, which is the first "
                    "217-457 s of the session"
                ),
                "defects": [
                    "accelerometer released with gravity removed (disqualifying for any gravity-using measure)",
                    "the per-sensor orientation quaternion is nine-axis, magnetometer-aided firmware fusion",
                    "optical reference glitches up to 165 deg between consecutive frames in six of sixteen subjects",
                    "inertial unit #6 (right thigh) emits impossible single-frame rates of 1129-3018 deg/s",
                    "missing optical frames are written as exact zero quaternions",
                    "the documented product discards roughly two thirds of the session",
                ],
            },
        )

    def _stream(self, merged: pd.DataFrame, index: int, t: np.ndarray) -> SensorStream:
        acc = merged[[f"a{c}{index}" for c in "xyz"]].to_numpy(float)  # m/s², gravity removed by the vendor
        gyr_deg = merged[[f"g{c}{index}" for c in "xyz"]].to_numpy(float)  # °/s as deposited
        gyr = gyr_deg * DEG2RAD
        impossible = np.flatnonzero(np.linalg.norm(gyr_deg, axis=1) > IMPOSSIBLE_RATE_DEG_S)
        return SensorStream(
            t=t,
            gyr=gyr,
            acc=acc,
            rate_hz=RATE_HZ,
            family=self.sensor_family,
            range_gyr=DEG_PER_S_FULL_SCALE * DEG2RAD,
            range_acc=None,  # the accelerometer full scale is not stated in the deposit or the paper
            native_rate_hz=None,
            meta={
                "sensor_index": index,
                "site": _SITE[index],
                "units_as_deposited": {"acc": "m/s^2 (gravity removed)", "gyr": "deg/s"},
                "gravity_removed": True,
                "impossible_rate_idx": impossible,
                "impossible_rate_note": (
                    f"frames whose rate magnitude exceeds {IMPOSSIBLE_RATE_DEG_S:.0f} deg/s; on sensor #6 these are "
                    "the deposit's known impossible samples. Exposed, not de-spiked."
                ),
                "vendor_quaternion_wxyz": merged[[f"{c}{index}" for c in "wxyz"]].to_numpy(float),
                "vendor_quaternion_note": (
                    "nine-axis firmware fusion with magnetometer calibration and live magnetic-axis correction; "
                    "never an input and never a reference for a magnetometer-free method"
                ),
            },
        )

    def _pose(self, angle: pd.DataFrame, k: int, t: np.ndarray, stamp_ms: np.ndarray, zero_stamp: np.ndarray) -> Pose:
        base = 2 + 15 * k
        block = angle.iloc[:, base : base + 15].to_numpy(float)
        q_parent = block[:, 3:7]  # (x, y, z, w) to the parent segment
        q_world = block[:, 10:14]  # (x, y, z, w) to the world ("ToGlobal") frame
        p = block[:, 7:10] * MM2M  # ToGlobal translation, mm -> m

        invalid = np.linalg.norm(q_world, axis=1) < 0.5  # the deposit writes missing frames as exact zeros
        q_fit = np.where(invalid[:, None], np.array([0.0, 0.0, 0.0, 1.0]), q_world)
        R = Rotation.from_quat(q_fit).as_matrix()
        R[invalid] = np.nan
        p = p.copy()
        p[invalid] = np.nan

        ok = ~invalid
        jump_deg = np.zeros(len(R))
        step_ok = ok[:-1] & ok[1:]
        if step_ok.any():
            a = Rotation.from_quat(q_fit[:-1][step_ok])
            b = Rotation.from_quat(q_fit[1:][step_ok])
            jump_deg[1:][step_ok] = (a.inv() * b).magnitude() / DEG2RAD

        return Pose(
            t=t,
            R=R,
            p=p,
            rate_hz=RATE_HZ,
            kind="skeletal-fit",
            markers=None,  # the deposit ships model segment origins, not marker trajectories
            marker_names=None,
            meta={
                "segment": OPTICAL_SEGMENTS[k],
                "parent": OPTICAL_SEGMENTS[k].split(".")[-1] if k else None,
                "frame": "XINGYING 'ToGlobal' laboratory frame, z up (pelvis/knee/ankle at 0.99/0.49/0.09 m standing)",
                "quaternion_to_parent_xyzw": q_parent,
                "quaternion_to_parent_note": (
                    "the deposit's parent-relative quaternion, carried verbatim; R above is the world "
                    "('ToGlobal') rotation, and q_world(child) = q_world(parent) * q_parent(child) holds to a "
                    "median 9.2e-5 deg"
                ),
                "segment_length_mm": block[:, 14],
                "invalid_idx": np.flatnonzero(invalid),
                "invalid_note": "reference missing; the deposit writes an exact zero quaternion, here NaN rows",
                "jump_gt_10deg_idx": np.flatnonzero(jump_deg > JUMP_DEG),
                "jump_max_deg": float(jump_deg.max()) if len(jump_deg) else float("nan"),
                "jump_note": "frame-to-frame world rotation; glitches are flagged here and left in the signal",
                "timestamp_unix_ms": stamp_ms,
                "zero_timestamp_idx": zero_stamp,
                "timestamp_note": (
                    "the deposit's own optical clock; t above is the uniform 50 Hz frame index, because the "
                    "deposit pairs inertial and optical row by row and some frames carry a zero timestamp"
                ),
                "invalid_runs": _runs_of_true(invalid),
            },
        )


ADAPTER = HipRomYAdapter()
