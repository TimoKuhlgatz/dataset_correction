"""IS2B: a bone-pinned cadaver knee, one specimen, two Xsens IMUs screwed to femur and tibia, Vicon marker clusters on
the same bone pins (no skin or strap in the measurement chain). Data CC BY 4.0 (doi:10.6084/m9.figshare.c.5328773).

Files read (under ``paths.dataset_dir("is2b")``):
- the plane-movement folders (Vertical, Horizontal, Mixed): 48 trials ``<plane>_<duration>_<excitation>_<range>.mat``
  (plane V/H/M, duration 15/30/120 s, excitation s/f/m, range of motion 60 or 110 degrees);
- ``14170718_Functional_movements/``: 5 trials ``<plane>_Fun_<FemFix|TibFix>[_rot].mat``, one bone clamped;
- the gyroscope-bias and magnetometer-calibration recordings, only with ``include_calibration=True``;
- ``14170733_Inertial_sensor_alignment/align.mat``: two constant sensor-to-cluster quaternions, exposed on the
  sensor as ``meta["sensor_to_cluster_R"]`` (not used downstream);
- ``14170727_CT-data/ct.mat``, read on demand by :meth:`Is2b.ct_landmarks`. The per-trial notes of
  ``Data_Summary.xlsx`` are transcribed into TRIAL_NOTES.

Units: markers and landmarks mm -> m; joint angles deg -> rad; accelerometer (m/s^2) and gyroscope (rad/s) are
already SI.

Reference frames: ``Segment.reference`` is the marker-cluster frame (rigid registration of a constant four-marker
template, ``_rigid_common.cluster_frame``; ``kind="mechanical"``), its constant rotation set to the shipped
anatomical frame at the first sample. ``Segment.anatomical`` is the shipped CT-derived bone frame ``[I | J | K]``,
placed at the knee-centre landmark (FKC/TKC). A cluster frame with an inter-marker distance more than 2 mm off its
median is registered on the three markers that hold when exactly one triple does, keeps all four when the fault is
ambiguous, and is NaN where no triple holds.

Deposit defects: the vendor orientation ``imu.<segment>.q`` is not carried (a filter output that drifts 4-19
degrees within a trial); ``kin.flexion_gs`` is folded about 90 degrees, only ``flexion_dh`` is carried; stream
lengths differ by 0-2 samples in 27 of 64 files and are truncated to the common length
(``meta["truncated_samples"]``); the third rotation is named ``abduction`` or ``adduction`` depending on the file;
the accelerometer is tilted about 6 degrees against the vertical at rest (a calibration defect); the marker
clusters are not rigid in 3.2 % of frames; not every trial starts at rest (``rest_annotation_s`` follows the
transcribed notes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..schema import Pair, Pose, Recording, Segment, SensorStream
from ._rigid_common import DEG, cluster_frame, quaternion_to_matrix, rigid_frames

SUBJECT = "cadaver-limb-01"
"""n = 1 cadaveric lower limb; there is no specimen field in any file name."""

SEGMENTS = ("femur", "tibia")
MILLIMETRE = 1e-3
"""Units: m per mm.
Provenance: physical — an exact unit conversion."""

MOTION_DIRS = (
    "14170700_Vertical_plane_movements",
    "14170712_Horizontal_plane_movement",
    "14170715_Mixed_plane_movement",
    "14170718_Functional_movements",
)
CALIBRATION_DIRS = ("14170721_Gyroscope_bias", "14170724_Magnetometer_calibration")
ALIGNMENT_FILE = "14170733_Inertial_sensor_alignment/align.mat"
CT_FILE = "14170727_CT-data/ct.mat"

KNEE_CENTRE_LANDMARK = {"femur": "FKC", "tibia": "TKC"}
"""The origin the deposit's own visualisation draws each anatomical frame at."""

CLUSTER_MARKERS = ("O1", "O2", "O3", "O4")
RIGID_TOLERANCE_M = 0.002
"""An inter-marker distance further than this from its median over the recording marks the frame as not rigid.
Units: m.
Provenance: data-free — several times the sub-millimetre reconstruction noise of an optical system; across this
75–130 mm cluster it is about one degree of orientation."""

TRIAL_NOTES: dict[str, str] = {
    "V_30_s_60": "High levels of noise are present on accelerometer data",
    "V_30_f_60": "High levels of noise are present on accelerometer data",
    "V_15_m_110": "Trial start pose in 90° flexion.",
    "V_30_m_110": "No quasi-static time period at start.",
    "V_Fun_FemFix": "No quasi-static time period at start.",
    "V_Fun_TibFix": "2 second quasi-static time period at start.",
}
"""Transcribed verbatim from the deposit's ``Data_Summary.xlsx`` note column, so a load needs no spreadsheet
reader. Provenance: read-from-dataset."""

REST_ANNOTATION_S: dict[str, float] = {"V_30_m_110": 0.0, "V_Fun_FemFix": 0.0, "V_Fun_TibFix": 2.0}
"""The only rest periods the deposit states per trial. Every other trial gets ``None``: the source paper says
each trial was "preceded by ~5 s pseudo-static", but that is prose about the protocol and ~18 files were
measured to have no usable still window, so it is not carried as an annotation."""


class Is2b:
    """The bone-pinned cadaver knee: one subject, one 3-DoF-referenced joint, 53 motion trials."""

    name = "is2b"
    reference_kind = "mechanical"
    sensor_family = "Xsens MTw Awinda"
    licence = (
        "CC BY 4.0 on all nine figshare articles (verified from the figshare v2 API). The companion code "
        "repository github.com/IveW/IS2B is GPL-3.0 and is not vendored"
    )
    source = (
        "Weygers, I.; Kok, M.; Seel, T.; Shah, D.; Taylan, O.; Scheys, L.; et al., 'Reference in-vitro dataset "
        "for inertial-sensor-to-bone alignment applied to the tibiofemoral joint', Scientific Data 8:208 "
        "(2021), doi:10.1038/s41597-021-00995-8; deposit doi:10.6084/m9.figshare.c.5328773. See the deposit's "
        "SOURCE.md"
    )

    def __init__(self, root: Path | None = None, include_calibration: bool = False) -> None:
        self.root = Path(root) if root is not None else paths.dataset_dir("is2b")
        self.include_calibration = include_calibration

    # -- listing ----------------------------------------------------------- #
    def _files(self) -> dict[str, Path]:
        dirs = list(MOTION_DIRS) + (list(CALIBRATION_DIRS) if self.include_calibration else [])
        out: dict[str, Path] = {}
        for name in dirs:
            folder = self.root / name
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.mat")):
                out[path.stem] = path
        return out

    def trials(self) -> list[tuple[str, str]]:
        return [(SUBJECT, stem) for stem in sorted(self._files())]

    def recordings(self, limit: int | None = None):
        for i, (subject, trial) in enumerate(self.trials()):
            if limit is not None and i >= limit:
                return
            yield self.load(subject, trial)

    def alignment(self) -> dict[str, np.ndarray]:
        """The deposited sensor-to-bone alignment quaternions, scalar-first, one per segment."""
        from scipy.io import loadmat

        align = loadmat(self.root / ALIGNMENT_FILE, squeeze_me=True, struct_as_record=False)["align"]
        return {name: np.asarray(getattr(align, name), float) for name in SEGMENTS}

    def ct_landmarks(self):
        """The CT-frame landmark file as deposited (millimetres, CT scanner frame). Read on demand."""
        from scipy.io import loadmat

        return loadmat(self.root / CT_FILE, squeeze_me=True, struct_as_record=False)["CT"]

    # -- loading ----------------------------------------------------------- #
    def load(self, subject: str, trial: str) -> Recording:
        from scipy.io import loadmat

        path = self._files()[trial]
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        imu, kin, traj = mat["imu"], mat["kin"], mat["traj"]
        alignment = self.alignment()

        rate_hz = float(getattr(imu, SEGMENTS[0]).fs)
        n_imu = int(np.asarray(getattr(imu, SEGMENTS[0]).Acc_X).shape[0])
        n_ref = int(np.asarray(getattr(kin, SEGMENTS[0]).I).shape[0])
        n = min(n_imu, n_ref)
        t = np.arange(n) / rate_hz  # one shared 100 Hz clock: sample i ↔ t = i/100 s in every stream

        segments: dict[str, Segment] = {}
        for name in SEGMENTS:
            sensor = getattr(imu, name)
            acc = np.stack([np.asarray(getattr(sensor, f"Acc_{a}"), float) for a in "XYZ"], axis=-1)[:n]
            gyr = np.stack([np.asarray(getattr(sensor, f"Gyr_{a}"), float) for a in "XYZ"], axis=-1)[:n]
            has_mag = hasattr(sensor, "Mag_X")

            axes = getattr(kin, name)
            triad = ("I", "J", "K") if name == "femur" else ("i", "j", "k")
            anatomical_r = np.stack([np.asarray(getattr(axes, a), float)[:n] for a in triad], axis=2)
            landmark = KNEE_CENTRE_LANDMARK[name]
            anatomical_p = np.asarray(getattr(getattr(traj, name).a, landmark), float)[:n] * MILLIMETRE

            cluster = getattr(traj, name).o
            markers = np.stack(
                [np.asarray(getattr(cluster, m), float)[:n] for m in CLUSTER_MARKERS], axis=1
            ) * MILLIMETRE
            rot, centroid, residual = cluster_frame(markers, r_at_first_frame=anatomical_r,
                                                    rigid_tolerance_m=RIGID_TOLERANCE_M)
            not_rigid = ~rigid_frames(markers, RIGID_TOLERANCE_M)

            segments[name] = Segment(
                name=name,
                sensor=SensorStream(
                    t=t,
                    gyr=gyr,
                    acc=acc,
                    rate_hz=rate_hz,
                    family=self.sensor_family,
                    range_gyr=None,  # the deposit states no full-scale range
                    range_acc=None,
                    native_rate_hz=rate_hz,
                    meta={
                        "units_as_deposited": "acc m/s², gyr rad/s — already SI, no conversion applied",
                        "gravity_included": True,
                        "sensor_to_cluster_R": quaternion_to_matrix(alignment[name]).tolist(),
                        "sensor_to_cluster_quaternion": alignment[name].tolist(),
                        "sensor_to_cluster_provenance": "deposited (align.mat), an averaged constant over "
                                                        "trials that the deposit recommends over a per-trial "
                                                        "fit; it does NOT reproduce against a rates-only fit in "
                                                        "this module's frames (134° femur, 98° tibia off)",
                        "magnetometer_deposited": has_mag,
                        "magnetometer_not_carried": "the schema is six-axis; the deposit's magnetometer is "
                                                    "normalised to arbitrary units",
                        "xsens_filter_quaternion_not_carried": "imu.<segment>.q is the proprietary Xsens "
                                                               "filter's output, forbidden by specification "
                                                               "rule 2, transposed in 47 of 64 files and "
                                                               "empty in V_30_m_60",
                        "accelerometer_tilt_defect_deg": "5.5–6.3° between the aligned accelerometer and world "
                                                         "vertical at rest; a calibration/axis-scale error, "
                                                         "not the mounting rotation",
                    },
                ),
                reference=Pose(
                    t=t,
                    R=rot,
                    p=centroid,
                    rate_hz=rate_hz,
                    kind="mechanical",
                    markers=markers,
                    marker_names=list(CLUSTER_MARKERS),
                    meta={
                        "construction": "rigid registration of a constant four-marker template from the "
                                        "bone-pin cluster; the template's constant rotation is fixed to the "
                                        "shipped CT-derived anatomical frame at the first sample",
                        "frame_measures": "the bone pin, which also carries the IMU — sensor and reference are "
                                          "one rigid body, so this is the soft-tissue-free null case",
                        "system": "six-camera Vicon MX+, hardware-synchronised with the IMUs, both 100 Hz",
                        "units_converted": "marker positions mm → m (× 1e-3)",
                        "registration_residual_m_median": float(np.nanmedian(residual)),
                        "registration_residual_m_max": float(np.nanmax(residual)) if n else float("nan"),
                        "frames_not_rigid": int(not_rigid.sum()),
                        "frames_without_a_rigid_triple": int((not_rigid & ~np.isfinite(rot[:, 0, 0])).sum()),
                    },
                ),
                anatomical=Pose(
                    t=t,
                    R=anatomical_r,
                    p=anatomical_p,
                    rate_hz=rate_hz,
                    kind="anatomical",
                    markers=None,
                    meta={
                        "construction": "the shipped CT-derived bone frame, axes as columns of [I|J|K]; "
                                        "R_cluster(t)ᵀ R_anatomical(t) was verified constant to 1.3e-15, i.e. "
                                        "the CT registration is already baked into kin",
                        "not_skin": "'anatomical' here does NOT mean skin landmarks — the frame is CT-derived "
                                    "and carried by the bone pin",
                        "origin": f"the knee-centre landmark {landmark}",
                        "units_converted": "landmark positions mm → m (× 1e-3)",
                    },
                ),
                extra_sensors=[],
                mount="zip-tied to a bone pin screwed into the bone",
            )

        angle_field = "abduction" if hasattr(kin, "abduction") else "adduction"
        pair = Pair(
            parent="femur",
            child="tibia",
            joint="hinge",
            joint_dof=1,
            joint_centre_world=None,  # no optical joint centre is deposited; the two knee-centre landmarks are
            joint_centre_t=None,      # on the anatomical Pose of each segment instead
            meta={
                "joint_name": "tibiofemoral (knee)",
                "joint_dof_note": "a hinge: one fixed axis explains the relative rotation to 1.7-4.7 deg RMS "
                                  "per recording, median 3.3, over 60-148 deg of flexion, and one common axis fits "
                                  "all 26 recordings to 5.4 deg RMS. The reference expresses all three rotational "
                                  "degrees of freedom (flexion, internal/external rotation, ab/adduction), but that "
                                  "is how the reference is parameterised, not how the joint moves. In 9 of the 26 "
                                  "recordings the flexion axis is near vertical, where a hinge axis carries no "
                                  "heading information at all; those are split out of any hinge-based comparison",
                "flexion_rad": np.asarray(kin.flexion_dh, float)[:n] * DEG,
                "rotation_rad": np.asarray(kin.rotation, float)[:n] * DEG,
                f"{angle_field}_rad": np.asarray(getattr(kin, angle_field), float)[:n] * DEG,
                "angles_units_converted": "degrees → radians (× π/180)",
                "flexion_source": "kin.flexion_dh (Denavit-Hartenberg); kin.flexion_gs is NOT carried because "
                                  "it is folded about 90°",
                "third_angle_field_name": angle_field,
                "knee_centre_landmarks": "femur FKC and tibia TKC, carried as the origin of each segment's "
                                         "anatomical Pose",
            },
        )

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=[pair],
            same_segment_pairs=[],
            contacts_t=None,  # an in-vitro specimen moved by hand: no ground contact
            rest_annotation_s=REST_ANNOTATION_S.get(trial),
            trial_type=_trial_type(trial),
            meta={
                "licence": self.licence,
                "source": self.source,
                "file": str(path),
                "alignment_file": str(self.root / ALIGNMENT_FILE),
                "ct_file": str(self.root / CT_FILE),
                "duration_s": float(n / rate_hz),
                "truncated_samples": {"imu": n_imu - n, "reference": n_ref - n},
                "deposit_note": TRIAL_NOTES.get(trial),
                "factors": _factors(trial),
                "clock": "one shared 100 Hz clock; IMU and Vicon were hardware-synchronised and no time column "
                         "is deposited anywhere — sample i is t = i/100 s in every stream",
                "defects": [
                    ("the deposited Xsens filter quaternion is not carried (estimator output; transposed in 47 "
                    "of 64 files; empty in V_30_m_60)"),
                    "kin.flexion_gs is folded about 90° and is not carried",
                    "stream lengths disagree by 0–2 samples in 27 of 64 files; truncated to the common length",
                    "the third rotation is named abduction in motion files and adduction in calibration files",
                    "a 5.5–6.3° accelerometer tilt defect at rest, systematic by movement plane",
                    "the initial still stretch is not universal; ~18 files have no usable window",
                    "only the 120 s arm can yield a full 120 s readout (18 of 53 trials, of which 16 exist)",
                ],
            },
        )


def _factors(trial: str) -> dict:
    """The trial name's four factors, or the functional/calibration description."""
    parts = trial.split("_")
    if len(parts) == 4 and parts[1].isdigit():
        return {
            "plane": {"V": "vertical", "H": "horizontal", "M": "mixed"}.get(parts[0], parts[0]),
            "duration_s": int(parts[1]),
            "excitation": {"s": "slow", "f": "fast", "m": "mixed"}.get(parts[2], parts[2]),
            "range_of_motion_deg": int(parts[3]),
        }
    return {"description": trial.replace("_", " ")}


def _trial_type(trial: str) -> str:
    if "Fun" in trial:
        return "functional (one bone clamped)"
    if trial.startswith("Gyr_bias"):
        return "gyroscope bias calibration"
    if trial.startswith("Mag_cal"):
        return "magnetometer calibration"
    return "controlled flexion"


ADAPTER = Is2b()
