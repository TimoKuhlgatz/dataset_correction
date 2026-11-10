"""OpenSense (Al Borno et al. 2022): eleven people walking and performing tasks, Xsens MTw IMUs with marker plates
machined onto the sensor housings, so the optical system measures the sensor's own frame. Licence CC BY 4.0.

Reference: the shipped ``markerplate_orientations.sto`` is defective (the left and right foot columns are
permutations of one another), so the reference is rebuilt from the raw markers by
:mod:`mocapref.datasets.readers.datasets.opensense`: per-frame Kabsch registration of a clean reference cluster, the
laboratory's Y-up frame turned to Z-up, and one constant plate-to-sensor rotation per sensor fitted from the
gyroscope. The rebuilt reference is stored once in its own data home (``paths.data_path("opensense_reference")``)
and read from there; deleting that folder rebuilds it from the deposit.

Frames and units: quaternions scalar-first, plate (= sensor) frame to the Z-up laboratory frame; accelerometer in
m/s^2 with gravity, gyroscope in rad/s, both at 100 Hz.

Trials: the joints listed in SEQUENCES (every joint whose two plates passed the rigidity, gap and synchronisation
checks) plus ADDED_SEQUENCES (Subject07's task block, read with the sensor mapping of its walk), minus the two
trials recorded at 40 Hz and interpolated to 100 Hz (EXCLUDED_TRIALS); 16 trials.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...datasets.opensense import store as reference_store
from ..schema import Pair, Pose, Recording, Segment, SensorStream
from ._rigid_common import quaternion_to_matrix

RATE_HZ = 100.0
"""The rate the reconstruction serves every trial at (the deposit's own rate for fifteen of the seventeen
scored trials; the two 40 Hz trials are resampled onto it by the reconstruction, which is where its resampling
fix lives).

Units: Hz.
Provenance: read-from-dataset — the deposit's own rate on nine of the eleven subjects, and the rate the
reconstruction serves all of them at."""

RATE_MATCH_TOLERANCE_HZ = 1e-6
"""Two rates closer than this count as the same rate, so a trial recorded at the read rate is not reported as
resampled.

Units: Hz.
Provenance: data-free — a floating-point tolerance on an equality of two rates, far below the only rate
difference this deposit contains (100 Hz against 40 Hz)."""

JOINT_BODIES: dict[str, tuple[str, str]] = {
    "hip_r": ("pelvis", "femur_r"),
    "knee_r": ("femur_r", "tibia_r"),
    "ankle_r": ("tibia_r", "calcn_r"),
    "hip_l": ("pelvis", "femur_l"),
    "knee_l": ("femur_l", "tibia_l"),
    "ankle_l": ("tibia_l", "calcn_l"),
    "lumbar": ("pelvis", "torso"),
}
"""Joint -> (parent body, child body). Transcribed from the deposit's own model at the pinned commit."""

JOINT_DOF = {"knee_r": 1, "knee_l": 1, "ankle_r": 2, "ankle_l": 2, "hip_r": 3, "hip_l": 3, "lumbar": 3}
"""Degrees of freedom per joint, as the deposit's model defines them: knees hinge, ankles two-axis, hips and the
lumbar joint three-axis. Used only to group sequences into the degrees-of-freedom classes the dataset value
averages over.

Units: degrees of freedom.
Provenance: read-from-dataset — as the deposit's own model defines them."""

JOINT_KIND = {1: "hinge", 2: "2-dof", 3: "ball"}
"""Units: degrees of freedom.
Provenance: data-free — a vocabulary: the name of a joint with that many degrees of freedom."""

SEQUENCES: tuple[str, ...] = (
    "Subject01/complexTasks/hip_l", "Subject01/complexTasks/hip_r", "Subject01/complexTasks/knee_l",
    "Subject01/complexTasks/knee_r", "Subject01/complexTasks/lumbar", "Subject01/walking/hip_l",
    "Subject01/walking/hip_r", "Subject01/walking/knee_l", "Subject01/walking/knee_r",
    "Subject01/walking/lumbar", "Subject02/complexTasks/ankle_l", "Subject02/complexTasks/ankle_r",
    "Subject02/complexTasks/hip_l", "Subject02/complexTasks/hip_r", "Subject02/complexTasks/knee_l",
    "Subject02/complexTasks/knee_r", "Subject02/walking/ankle_l", "Subject02/walking/ankle_r",
    "Subject02/walking/hip_l", "Subject02/walking/hip_r", "Subject02/walking/knee_l",
    "Subject02/walking/knee_r", "Subject02/walking/lumbar", "Subject03/complexTasks/ankle_l",
    "Subject03/complexTasks/ankle_r", "Subject03/complexTasks/hip_l", "Subject03/complexTasks/hip_r",
    "Subject03/complexTasks/knee_l", "Subject03/complexTasks/knee_r", "Subject03/walking/ankle_l",
    "Subject03/walking/ankle_r", "Subject03/walking/hip_l", "Subject03/walking/hip_r",
    "Subject03/walking/knee_l", "Subject03/walking/knee_r", "Subject03/walking/lumbar",
    "Subject04/complexTasks/ankle_l", "Subject04/complexTasks/ankle_r", "Subject04/complexTasks/hip_l",
    "Subject04/complexTasks/hip_r", "Subject04/complexTasks/knee_l", "Subject04/complexTasks/knee_r",
    "Subject04/complexTasks/lumbar", "Subject04/walking/ankle_l", "Subject04/walking/ankle_r",
    "Subject04/walking/hip_l", "Subject04/walking/hip_r", "Subject04/walking/knee_l",
    "Subject04/walking/knee_r", "Subject05/walking/hip_l", "Subject05/walking/knee_l",
    "Subject06/complexTasks/ankle_l", "Subject06/complexTasks/hip_r", "Subject06/complexTasks/knee_r",
    "Subject06/complexTasks/lumbar", "Subject07/walking/ankle_l", "Subject07/walking/ankle_r",
    "Subject07/walking/hip_l", "Subject07/walking/hip_r", "Subject07/walking/knee_l",
    "Subject07/walking/knee_r", "Subject08/walking/ankle_r", "Subject08/walking/hip_l",
    "Subject08/walking/hip_r", "Subject08/walking/knee_l", "Subject08/walking/knee_r",
    "Subject09/complexTasks/hip_l", "Subject09/complexTasks/knee_l", "Subject09/walking/hip_l",
    "Subject09/walking/hip_r", "Subject09/walking/knee_l", "Subject09/walking/knee_r",
    "Subject10/walking/hip_l", "Subject10/walking/hip_r", "Subject10/walking/knee_r",
    "Subject11/complexTasks/hip_l", "Subject11/complexTasks/hip_r", "Subject11/complexTasks/knee_l",
    "Subject11/walking/hip_l", "Subject11/walking/hip_r", "Subject11/walking/knee_l",
    "Subject11/walking/knee_r",
)
"""The 82 sequences, ``subject/task/joint``, whose two plates passed the rigidity, gap and synchronisation
checks: 11 subjects, 17 trials, 29 one-, 16 two- and 37 three-degree-of-freedom sequences. ``Subject06/walking``
is not among them: its marker folder is spelled ``MoCap``, which the selection's exact name match missed."""

ADDED_SEQUENCES: tuple[str, ...] = (
    "Subject07/complexTasks/ankle_l", "Subject07/complexTasks/ankle_r", "Subject07/complexTasks/hip_l",
    "Subject07/complexTasks/hip_r", "Subject07/complexTasks/knee_l", "Subject07/complexTasks/knee_r",
)
"""Sequences added to :data:`SEQUENCES` here. Subject07's task block ships no sensor mapping file; it is read with
the walking trial's mapping (``BORROWED_MAPPINGS`` in ``mocapref.datasets.readers.datasets.opensense.sensors``, where the
check is described). Its joints are those of Subject07's walk (same seven sensors, no torso sensor)."""

EXCLUDED_TRIALS: dict[tuple[str, str], str] = {
    ("Subject06", "complexTasks"): "recorded at 40 Hz and interpolated to 100 Hz",
    ("Subject10", "walking"): "recorded at 40 Hz and interpolated to 100 Hz",
}
"""Trials left out of :meth:`OpenSense.trials`: the 40 Hz trials are not used. Subject06's walk, also 40 Hz, is not
in :data:`SEQUENCES` in the first place."""


class OpenSense:
    """One trial is one subject × one task; its scored pairs are the joints of that trial in :data:`SEQUENCES`."""

    name = "opensense-s2"
    reference_kind = "housing"
    sensor_family = "Xsens MTw"
    licence = "CC BY 4.0 (as the host states it for the OpenSense dataset)"
    source = (
        "OpenSense IMU + optical dataset, read through the marker-plate reconstruction in "
        "mocapref.datasets.readers.datasets.opensense — see the module docstring"
    )

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else None

    # -- listing ----------------------------------------------------------- #
    def trials(self) -> list[tuple[str, str]]:
        seen: list[tuple[str, str]] = []
        for sequence in SEQUENCES + ADDED_SEQUENCES:  # with the added and without the excluded trials
            subject, task, _joint = sequence.split("/")
            if (subject, task) not in seen and (subject, task) not in EXCLUDED_TRIALS:
                seen.append((subject, task))
        return sorted(seen)

    def joints_of(self, subject: str, task: str) -> list[str]:
        return [s.split("/")[2] for s in SEQUENCES + ADDED_SEQUENCES if s.startswith(f"{subject}/{task}/")]

    def reference_file(self, subject: str, task: str) -> Path:
        """Where the rebuilt reference of one trial is materialised."""
        return reference_store.rebuilt_path(subject, task, self.store_dir)

    # -- loading ----------------------------------------------------------- #
    def load(self, subject: str, task: str) -> Recording:
        rebuilt = reference_store.load_trial(subject, task, self.store_dir)
        names, n = rebuilt["imus"], rebuilt["n"]
        quat, acc, gyr = rebuilt["quat"], rebuilt["acc"], rebuilt["gyr"]
        native_rate_hz = float(rebuilt["native_rate"])
        resampled = abs(native_rate_hz - RATE_HZ) > RATE_MATCH_TOLERANCE_HZ

        t = np.arange(n) / RATE_HZ
        segments = {
            name: Segment(
                name=name,
                sensor=SensorStream(
                    t=t, gyr=gyr[name], acc=acc[name], rate_hz=RATE_HZ, native_rate_hz=native_rate_hz,
                    family=self.sensor_family,
                    meta={"units_as_deposited": "acc m/s², gyr rad/s — already SI, no conversion applied",
                          "gravity_included": True,
                          "resampled_to_read_rate": resampled,
                          "true_bandwidth_hz": native_rate_hz / 2,
                          "synchronisation": "integer-lag alignment against the marker-derived angular rate, "
                                             "performed by this repository's reconstruction"},
                ),
                reference=Pose(
                    t=t, R=quaternion_to_matrix(quat[name]), p=np.full((n, 3), np.nan), rate_hz=RATE_HZ,
                    kind="housing",
                    meta={"construction": "per-frame Kabsch registration of the four-marker plate machined "
                                          "around the sensor housing, laboratory Y-up rotated to Z-up",
                          "reference_quaternion": quat[name],
                          "reference_quaternion_convention": "scalar-first (w, x, y, z), plate (= sensor) "
                                                             "frame → Z-up laboratory frame",
                          "shipped_orientation_file_not_used": "markerplate_orientations.sto is defective "
                                                               "(left and right foot columns are permutations "
                                                               "of one another) and is never read"},
                ),
                mount="strap, with a machined marker plate around the housing",
            )
            for name in names
        }

        pairs = []
        for joint in self.joints_of(subject, task):
            parent_body, child_body = JOINT_BODIES[joint]
            parent, child = f"{parent_body}_imu", f"{child_body}_imu"
            if parent not in segments or child not in segments:
                continue
            dof = JOINT_DOF[joint]
            pairs.append(Pair(
                parent=parent, child=child, joint=JOINT_KIND[dof], joint_dof=dof,
                meta={"joint": joint, "sequence_id": f"{subject}/{task}/{joint}", "task": task},
            ))

        defects = ["the shipped marker-plate orientation file is defective and is not used"]
        if resampled:
            named = ", ".join(p.meta["sequence_id"] for p in pairs)
            defects.append(
                f"recorded at {native_rate_hz:g} Hz and resampled to {RATE_HZ:g} Hz: nothing above "
                f"{native_rate_hz / 2:g} Hz is genuine in this trial's inertial streams, which affects the "
                f"sequences {named}"
            )
            defects.append(
                "the deposit's own reader mis-resamples this trial (it cross-correlates a 100 Hz stream "
                "against a 40 Hz one and truncates to the shorter); the reconstruction read here repairs that"
            )

        return Recording(
            dataset=self.name, subject=subject, trial=task, segments=segments, pairs=pairs,
            trial_type="walking" if task == "walking" else "functional tasks",
            meta={"licence": self.licence, "source": self.source, "rate_hz": RATE_HZ,
                  "native_rate_hz": native_rate_hz,
                  "n_samples": n, "duration_s": float(n / RATE_HZ),
                  "reference_rebuilt_into": str(self.reference_file(subject, task)),
                  "known_defects": defects},
        )

    def recordings(self, limit: int | None = None):
        for i, (subject, task) in enumerate(self.trials()):
            if limit is not None and i >= limit:
                return
            yield self.load(subject, task)


ADAPTER = OpenSense()
