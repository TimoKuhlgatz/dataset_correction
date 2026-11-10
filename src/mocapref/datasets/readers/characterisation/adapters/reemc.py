"""REEM-C: a humanoid robot walking, seven IMUs on the lower body, Qualisys optical reference. Licence CC BY 4.0,
doi:10.5281/zenodo.10253284 (no peer-reviewed data paper; the one-page Description.pdf defines no channels, units or
frames, so everything below was established from the data).

Files read: ``paths.dataset_dir("reemc")/*.h5``, one per trial (``ROBOT_CIRCLE_RUN_0003``, ``_0004``,
``ROBOT_TURNING_0003``, 403-453 s each), with ``/data/{acc_i, gyr_i}`` (T, 21), ``/data/{quat_ni, quat_ns}``
(T, 28; IMU and segment to world), ``/data/{i_n, s_n}`` (IMU and segment positions), ``/data/j_n`` (six joint
centres), ``/data/time`` (one clock for all streams, 0.01 s spacing, a session clock kept as deposited) and the
constant ``/skeleton`` group.

Units: already SI (accelerometer m/s^2 with gravity, gyroscope rad/s, positions m); quaternions scalar-first, body
to world. The magnetometer (arbitrary units) is not carried. IMUs at 100 Hz; Qualisys ran at 150 Hz and was
resampled to 100 Hz before deposit.

Reference: a rigid skeletal-model fit, verified algebraically over all frames: ``quat_ni = quat_ns * quat_si``
with a constant ``quat_si``, so the only measured reference quantity is the segment pose ``(s_n, quat_ns)``. That
pose is the segment reference (``kind="mechanical"``: on a robot the skeleton is genuinely rigid); ``quat_si`` is
exposed on the sensor as the deposited constant alignment and ``quat_ni`` is not carried. No marker positions are
deposited.

Joints: the six joints of ``listJointNamesAndConnPtIdx`` (two ankles, knees, hips); the deposit declares no degrees
of freedom (``Pair.joint_dof`` is None); the deposited joint centre ``j_n`` is carried as ``joint_centre_world``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..schema import Pair, Pose, Recording, Segment, SensorStream
from ._rigid_common import quaternion_to_matrix

SUBJECT = "reemc"
RATE_HZ = 100.0
"""Units: Hz.
Provenance: read-from-dataset — the rate the deposit is written at."""
QUALISYS_NATIVE_HZ = 150.0
"""Units: Hz.
Provenance: read-from-dataset — the optical system's native rate as the deposit states it."""

JOINT_NAME_OF_SEGMENTS = {
    frozenset({"LeftFoot", "LeftLowerLeg"}): "left ankle",
    frozenset({"RightFoot", "RightLowerLeg"}): "right ankle",
    frozenset({"LeftLowerLeg", "LeftUpperLeg"}): "left knee",
    frozenset({"RightLowerLeg", "RightUpperLeg"}): "right knee",
    frozenset({"LeftUpperLeg", "Pelvis"}): "left hip",
    frozenset({"RightUpperLeg", "Pelvis"}): "right hip",
}
"""Anatomical names inferred from the segment pair; the deposit names joints only by their connection points."""


class Reemc:
    """The three Reem-C trials; one subject (the robot), one trial per file, six joints each."""

    name = "reemc"
    reference_kind = "mechanical"
    sensor_family = "not stated in the deposit"
    licence = "CC BY 4.0 (Zenodo metadata.license = {'id': 'cc-by-4.0'})"
    source = (
        "Lorenz, M.; Aller, F.; Bleser-Taetz, G.; Taetz, B., 'IMU and marker-based optical motion capture from "
        "a humanoid robot', Zenodo (2023), doi:10.5281/zenodo.10253284. No peer-reviewed data paper; the "
        "deposit's one-page Description.pdf is the primary description"
    )

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else paths.dataset_dir("reemc")

    def trials(self) -> list[tuple[str, str]]:
        if not self.root.is_dir():
            return []
        return [(SUBJECT, p.stem) for p in sorted(self.root.glob("*.h5"))]

    def recordings(self, limit: int | None = None):
        for i, (subject, trial) in enumerate(self.trials()):
            if limit is not None and i >= limit:
                return
            yield self.load(subject, trial)

    def load(self, subject: str, trial: str) -> Recording:
        import h5py

        path = self.root / f"{trial}.h5"
        with h5py.File(path, "r") as handle:
            group = handle["data"]
            names = [str(s) for s in group.attrs["listIMUIDs"]]
            t = np.asarray(group["time"][:], float)
            acc = np.asarray(group["acc_i"][:], float)
            gyr = np.asarray(group["gyr_i"][:], float)
            quat_ns = np.asarray(group["quat_ns"][:], float)
            s_n = np.asarray(group["s_n"][:], float)
            j_n = np.asarray(group["j_n"][:], float)
            mag_present = "mag_i" in group
            skeleton = handle["skeleton"]
            joint_labels = [str(s) for s in skeleton.attrs["listJointNamesAndConnPtIdx"]]
            constants = {
                name: {
                    "sensor_origin_in_segment_m": np.asarray(skeleton[f"segment_{name}"]["i_s"][:], float),
                    "segment_from_sensor_quaternion": np.asarray(
                        skeleton[f"segment_{name}"]["quat_si"][:], float
                    ),
                    "connection_points_in_segment_m": np.asarray(
                        skeleton[f"segment_{name}"]["pSeg_s"][:], float
                    ),
                }
                for name in names
            }

        segments: dict[str, Segment] = {}
        for k, name in enumerate(names):
            quat = quat_ns[:, 4 * k: 4 * k + 4]
            constant = constants[name]
            segments[name] = Segment(
                name=name,
                sensor=SensorStream(
                    t=t,
                    gyr=gyr[:, 3 * k: 3 * k + 3],
                    acc=acc[:, 3 * k: 3 * k + 3],
                    rate_hz=RATE_HZ,
                    family=self.sensor_family,
                    range_gyr=None,  # no sensor model is stated anywhere in the deposit
                    range_acc=None,
                    native_rate_hz=RATE_HZ,
                    meta={
                        "units_as_deposited": "acc m/s², gyr rad/s — already SI, no conversion applied",
                        "gravity_included": True,
                        "frame": "the IMU frame",
                        "sensor_to_segment_quaternion": constant["segment_from_sensor_quaternion"].tolist(),
                        "sensor_to_segment_provenance": "deposited and time-invariant (/skeleton/segment_*/"
                                                        "quat_si); the deposit's quat_ni is exactly quat_ns ⊗ "
                                                        "quat_si, verified to 5.6e-16; this alignment "
                                                        "is a declared mapping, not a certified physical "
                                                        "calibration",
                        "sensor_origin_in_segment_m": constant["sensor_origin_in_segment_m"].tolist(),
                        "magnetometer_deposited": mag_present,
                        "magnetometer_not_carried": "the schema is six-axis; the deposit's magnetometer is "
                                                    "normalised to arbitrary units (median norm 0.91–1.14)",
                    },
                ),
                reference=Pose(
                    t=t,
                    R=quaternion_to_matrix(quat),
                    p=s_n[:, 3 * k: 3 * k + 3],
                    rate_hz=RATE_HZ,
                    kind="mechanical",
                    markers=None,
                    meta={
                        "construction": "a rigid skeletal model fitted to Qualisys markers; verified that "
                                        "quat_ni = quat_ns ⊗ quat_si with a constant quat_si (5.6e-16) and "
                                        "i_n = s_n + R(quat_ns)·i_s (4.4e-16), so there is no independent "
                                        "per-frame housing orientation in this deposit",
                        "kind_choice": "'mechanical' rather than 'skeletal-fit': the robot's skeleton is "
                                       "genuinely rigid, and classifying it reference-side would pool it with "
                                       "skin-marker human datasets, which rule 5 forbids",
                        "markers_not_deposited": True,
                        "system": "Qualisys, native 150 Hz, resampled to 100 Hz before deposit",
                        "quaternion_convention": "scalar-first (w, x, y, z), segment → world; the world frame "
                                                 "is Z-up and the accelerometer reads +g at rest",
                        "connection_points_in_segment_m": constant["connection_points_in_segment_m"].tolist(),
                    },
                ),
                anatomical=None,
                extra_sensors=[],
                mount="mounted on the robot segment; the sensor-to-segment rotation is deposited as a constant",
            )

        pairs = []
        for j, label in enumerate(joint_labels):
            left, right = label.split("_")
            parent, child = left.split(":")[0], right.split(":")[0]
            joint_name = JOINT_NAME_OF_SEGMENTS.get(frozenset({parent, child}), "unknown")
            pairs.append(
                Pair(
                    parent=parent,
                    child=child,
                    joint="mechanical",
                    joint_dof=None,  # the deposit declares none
                    joint_centre_world=j_n[:, 3 * j: 3 * j + 3],
                    joint_centre_t=t,
                    meta={
                        "joint_name": joint_name,
                        "deposit_label": label,
                        "joint_dof_note": "not declared anywhere in the deposit; the anatomical name is the "
                                          "only hint and is inferred from the two segment names",
                        "joint_centre_provenance": "deposited (/data/j_n); verified to be the midpoint of the "
                                                   "two segments' connection points to 8.9e-16, the two points "
                                                   "sitting 2.5–4.1 mm apart on average",
                    },
                )
            )

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],
            contacts_t=None,  # no force plates and no contact annotation are deposited
            rest_annotation_s=None,
            trial_type="circle run" if "CIRCLE_RUN" in trial else "turning",
            meta={
                "licence": self.licence,
                "source": self.source,
                "file": str(path),
                "duration_s": float(t[-1] - t[0]),
                "time_base": "the deposit's session clock; it does not start at zero because each trial is a "
                             "cut-out from a longer session. One clock serves every stream",
                "qualisys_native_rate_hz": QUALISYS_NATIVE_HZ,
                "reference_construction": "rigid skeletal model fit (verified algebraically over all frames)",
                "defects": [
                    ("no marker positions are deposited: the reference is a fitted segment pose, not a housing "
                    "cluster, so it is not comparable to a housing-referenced dataset"),
                    ("repeatable sensor/reference-frame inconsistency (5-8 degree tilt on several segments); "
                    "mounting, optical-model and sensor-axis causes are not separated; the deposited alignment "
                    "is kept"),
                    "no degrees-of-freedom count is declared for any joint",
                    "no sensor family, no full-scale ranges and no channel documentation exist in the deposit",
                    "the deposit's one-page Description.pdf is the only description",
                ],
            },
        )


ADAPTER = Reemc()
