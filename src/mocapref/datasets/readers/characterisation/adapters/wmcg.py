"""WMCG ("Wearable Motion Capture Gait", University of Central Florida): eight Delsys Avanti IMUs and skin markers,
14 recordings per participant (overground, stairs, treadmill, slope). Reader name ``wmcg-ucf``.

File read: ``_extracted/WMCG_dataset.h5``, groups ``/participants/<P>/<mode>/<condition>``:
  ``imu/sensor_<k>_<location>/acc`` (g), ``.../gyro`` (deg/s), ``imu/time`` (exactly i/100);
  ``markers/data`` (mm, XYZ interleaved per marker), ``markers/marker_names``, ``markers/time`` (exactly
  (i+1)/100: the documented one-row shift between the streams is applied through the time values).
Not used: ``/imu/raw`` (its layout attribute claims a block layout while the data are interleaved per sensor),
``/kinematics`` (an OpenSim gait2354 fit with one-degree-of-freedom knee and ankle) and ``/hof``.

Units: accelerometer g -> m/s^2 (x 9.80665); gyroscope deg/s -> rad/s (the declared unit agrees with the
marker-derived rate within 1 %); markers mm -> m. Rate 100 Hz as released (native about 148 Hz).

Reference (``kind="skin"``): no marker sits on a sensor housing. Each segment frame is the Procrustes fit of the
segment's four to five markers to the participant's segment shape, estimated by iterative Procrustes (three
iterations) on one recording (SHAPE_TRIAL, else the next with a complete frame) and used for all fourteen
recordings; a shape per recording would turn the frame with the way the participant faced. ``Pose.p`` is the
marker centroid; frames with a missing marker are NaN. The world is +y up (turned to +z up by the loader). In P08
``treadmill/vfast`` the deposited lateral and medial right-ankle labels are exchanged back before frame
construction.

Deposit defects left in the data: the right-shank accelerometer is railed at +-16 g in P05, P06, P07 and P10
(``meta["defects"]["railed_accelerometers"]``); a gyroscope zero-rate offset of 9-18 deg/s on the foot units; the
foot accelerometers read 4-7 % low; a residual reference lag of about 12.5 ms; synthetic time stamps; no recording
starts at rest; the mounting is undocumented (``Segment.mount`` is None); ``P08/overground/round`` has 101 more
inertial than marker samples.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

G_MS2 = 9.80665
"""Standard gravity; converts the deposit's accelerometer from g to m/s².

Units: m/s².
Provenance: physical — the defined value of ISO 80000-3."""

DEG_TO_RAD = np.pi / 180.0
"""Provenance: physical; the deposit's gyroscope attribute says ``deg/s``."""

MM_TO_M = 1e-3
"""Provenance: physical; the deposit's marker attribute says ``mm``."""

RATE_HZ = 100.0
"""Provenance: data-free — the deposit's own ``fs_hz`` attribute on ``/imu``, ``/markers`` and ``/kinematics``."""

NATIVE_RATE_HZ = 148.0
"""Provenance: data-free — the deposit's ``/imu`` attribute ``original_fs = "~148 Hz (Delsys Avanti), resampled
to 100 Hz"``. Approximate, and not checkable from the file because the released timestamps are synthetic."""

SENSOR_FAMILY = "Delsys Trigno Avanti"

SENSORS = {
    "sternum": "sensor_1_sternum",
    "sacrum": "sensor_2_sacrum",
    "right_thigh": "sensor_3_right_thigh",
    "left_thigh": "sensor_4_left_thigh",
    "right_shank": "sensor_5_right_shank",
    "left_shank": "sensor_6_left_shank",
    "right_foot": "sensor_7_right_foot",
    "left_foot": "sensor_8_left_foot",
}
"""Segment name → HDF5 group, from the deposit's own ``/metadata/imu_sensor_map``."""

MARKERS = {
    "sternum": ("C7", "CLAV", "STRN", "LSHO", "RSHO"),
    "sacrum": ("LASI", "RASI", "LPSI", "RPSI"),
    "right_thigh": ("RTHI", "RTHI_front", "RKNE", "R_Knee_med"),
    "left_thigh": ("LTHI", "LTHI_front", "LKNE", "L_Knee_med"),
    "right_shank": ("RTIB", "R_Shank", "RANK", "R_ANK_med"),
    "left_shank": ("LTIB", "L_Shank", "LANK", "L_ANK_med"),
    "right_foot": ("RHEE", "RTOE", "R_1MT", "R_5MT"),
    "left_foot": ("LHEE", "LTOE", "L_1MT", "L_5MT"),
}
"""The marker group of each segment, as grouped them."""

REFERENCE_LAG_MS = 12.5
"""Lag of the optical reference behind the inertial streams.

Units: ms.
Provenance: read-from-dataset — read from the deposit."""

REFERENCE_LAG_SPREAD_MS = 6.4
"""Trial-to-trial spread of that lag.

Units: ms.
Provenance: read-from-dataset — read from the deposit."""

PAIRS = (
    ("sternum", "sacrum", "ball", 3, "lumbar spine (a multi-vertebra column, modelled as a 3-DoF ball)"),
    ("sacrum", "right_thigh", "ball", 3, "right hip"),
    ("sacrum", "left_thigh", "ball", 3, "left hip"),
    ("right_thigh", "right_shank", "hinge", 1, "right knee"),
    ("left_thigh", "left_shank", "hinge", 1, "left knee"),
    ("right_shank", "right_foot", "2-dof", 2, "right ankle complex (talocrural plus subtalar)"),
    ("left_shank", "left_foot", "2-dof", 2, "left ankle complex (talocrural plus subtalar)"),
)
"""The seven connected pairs. The knee is nominally one hinge degree of freedom (two to three in
reality, with rollback and axial laxity), the ankle complex two, the hip and the lumbar column three.

Units: degrees of freedom.
Provenance: read-from-dataset — the connected pairs of the deposit and each joint's degrees of freedom."""

ACC_FULL_SCALE_G = 16.0
"""Provenance: read-from-dataset — the accelerometer end-stop the deposit itself shows. In P05, P06, P07 and
P10 the ``sensor_5_right_shank`` accelerometer sits at a mean magnitude of 16.01–16.07 g with 34 distinct
values across 17,957 samples on one axis, which is the ±16 g full scale of the Delsys Trigno Avanti. Carried on
every stream of the deposit so that a range-saturation check has the stated range it needs."""

GYR_FULL_SCALE_DPS = 2000.0
"""Provenance: physical — physical."""

RAILED_RIGHT_SHANK_PARTICIPANTS = ("P05", "P06", "P07", "P10")
"""Provenance: measured on this dataset — the
``sensor_5_right_shank`` accelerometer sits at its ±16 g end-stop in all fourteen recordings of these four
participants. Recorded as a flag only; nothing is excluded and nothing is repaired."""

PROCRUSTES_ITERATIONS = 3
"""Provenance: data-free — data-free."""

SHAPE_TRIAL = "overground/normal"
"""The recording whose marker cloud defines every segment's shape, for all fourteen recordings of the participant.

Provenance: data-free — one of the participant's own recordings; any single one would do, and the shape is
arbitrary up to a rotation either way. With each recording's own first
complete frame as the shape, the segment frame turned with the way the participant faced at that frame, and
participants face one way overground and on the stairs and the opposite way on the treadmill and the slope."""


ANKLE_LABEL_REPAIR_GROUP = "/participants/P08/treadmill/vfast"
ANKLE_LABEL_REPAIR = {"RANK": "R_ANK_med", "R_ANK_med": "RANK"}
"""canonical marker name -> deposited name, in this one recording only.

Provenance: measured raw-marker geometry and anatomical laterality on the already exposed WMCG test data;
all six single swaps were compared independently of estimator outputs. No acquisition cause or absolute
reference accuracy is certified. The dictionary changes correspondence, not any deposited coordinate value.
"""


def _marker_columns(marker_names: list[str], segment: str, group_path: str) -> list[int]:
    names = MARKERS[segment]
    if group_path == ANKLE_LABEL_REPAIR_GROUP:
        names = tuple(ANKLE_LABEL_REPAIR.get(name, name) for name in names)
    return [marker_names.index(name) for name in names]


def _fit_frames(block: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment → world rotations and marker-cloud centroids of a (T, M, 3) marker block against a fixed shape.

    One Procrustes fit per frame, in the block's units. Frames with any missing marker come out as NaN.
    """
    good = ~np.isnan(block).any(axis=(1, 2))
    centroid = np.nanmean(block, axis=1)
    centred = block - centroid[:, None, :]
    h = np.einsum("tmi,mj->tij", np.nan_to_num(centred), shape)
    u, _, vt = np.linalg.svd(h)
    reflection = np.tile(np.eye(3), (len(block), 1, 1))
    reflection[:, 2, 2] = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", u, vt)))
    rotations = np.einsum("tij,tjk,tkl->til", u, reflection, vt)
    rotations[~good] = np.nan
    centroid[~good] = np.nan
    return rotations, centroid


def _segment_shape(block: np.ndarray) -> np.ndarray | None:
    """The segment's marker shape from one recording's (T, M, 3) block, or None if no frame is complete.

    Iterative Procrustes against a re-estimated shape, as the dataset probe's does;
    the first complete frame only starts the iteration. The shape is centred and expressed in the segment frame
    it defines, so it is arbitrary up to one rotation — which is why it must come from one recording per
    participant and not from each recording's own first frame.
    """
    good = ~np.isnan(block).any(axis=(1, 2))
    if not good.any():
        return None
    shape = block[good][0]
    shape = shape - shape.mean(0)
    centred = block - np.nanmean(block, axis=1)[:, None, :]
    for _ in range(PROCRUSTES_ITERATIONS):
        rotations, _ = _fit_frames(block, shape)
        in_segment = np.einsum("tji,tmj->tmi", rotations[good], centred[good])
        shape = in_segment.mean(axis=0)
        shape = shape - shape.mean(0)
    return shape


class Wmcg(Adapter):
    """The 140 recordings of the WMCG deposit: ten participants × fourteen walking conditions."""

    name = "wmcg-ucf"
    reference_kind = "skin"
    sensor_family = SENSOR_FAMILY
    licence = "CC BY 4.0 (data; machine-readable in the figshare record). The article is CC BY-NC-ND 4.0."
    source = "doi:10.25452/figshare.plus.29328746.v2 — see SOURCE.md in the dataset folder"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else dataset_dir("wmcg") / "_extracted" / "WMCG_dataset.h5"
        self._index: list[tuple[str, str]] | None = None
        self._shapes: dict[str, dict[str, tuple[np.ndarray | None, str]]] = {}

    def trials(self) -> list[tuple[str, str]]:
        """All ``(participant, "<mode>/<condition>")`` pairs.

        The deposit is one HDF5 file, so the index comes from walking the group tree; that reads the group
        directory only and touches no signal dataset. It is cached in memory on this instance (nothing is
        written to the dataset folder).
        """
        if self._index is None:
            found: list[tuple[str, str]] = []
            with h5py.File(self.path, "r") as h5:
                participants = h5["participants"]
                for participant in sorted(participants):
                    for mode in sorted(participants[participant]):
                        for condition in sorted(participants[participant][mode]):
                            found.append((participant, f"{mode}/{condition}"))
            self._index = found
        return list(self._index)

    @staticmethod
    def _marker_block(group, name: str) -> np.ndarray:
        """One segment's (T, M, 3) marker block in metres, from the recording's own marker names."""
        marker_names = [n.decode() for n in group["markers/marker_names"][:]]
        data = np.asarray(group["markers/data"], dtype=np.float64) * MM_TO_M
        columns = _marker_columns(marker_names, name, group.name)
        return np.stack([data[:, 3 * i : 3 * i + 3] for i in columns], axis=1)

    def segment_shapes(self, subject: str) -> dict[str, tuple[np.ndarray | None, str]]:
        """Per segment, the participant's marker shape and the recording it was taken from.

        ``SHAPE_TRIAL`` for every segment that has a complete frame there, otherwise the next recording of the
        participant in ``trials()`` order that has one; None when no recording of the participant does. The
        marker names carry the correspondence between recordings, so the 36-marker recordings match the
        34-marker ones. Cached in memory on this instance; nothing is written to the dataset folder.
        """
        if subject not in self._shapes:
            order = [t for s, t in self.trials() if s == subject]
            order = [SHAPE_TRIAL] + [t for t in order if t != SHAPE_TRIAL]
            shapes: dict[str, tuple[np.ndarray | None, str]] = {}
            with h5py.File(self.path, "r") as h5:
                for candidate in order:
                    missing = [n for n in MARKERS if n not in shapes]
                    if not missing:
                        break
                    mode, _, condition = candidate.partition("/")
                    group = h5[f"participants/{subject}/{mode}/{condition}"]
                    for name in missing:
                        shape = _segment_shape(self._marker_block(group, name))
                        if shape is not None:
                            shapes[name] = (shape, candidate)
            for name in MARKERS:
                shapes.setdefault(name, (None, "none: no complete marker frame in any recording"))
            self._shapes[subject] = shapes
        return self._shapes[subject]

    def load(self, subject: str, trial: str) -> Recording:
        mode, _, condition = trial.partition("/")
        shapes = self.segment_shapes(subject)
        with h5py.File(self.path, "r") as h5:
            group = h5[f"participants/{subject}/{mode}/{condition}"]
            t_imu = np.asarray(group["imu/time"], dtype=np.float64)
            t_ref = np.asarray(group["markers/time"], dtype=np.float64)
            marker_names = [n.decode() for n in group["markers/marker_names"][:]]
            marker_data = np.asarray(group["markers/data"], dtype=np.float64) * MM_TO_M
            inertial = {
                name: (
                    np.asarray(group[f"imu/{key}/acc"], dtype=np.float64) * G_MS2,
                    np.asarray(group[f"imu/{key}/gyro"], dtype=np.float64) * DEG_TO_RAD,
                )
                for name, key in SENSORS.items()
            }

        railed = subject in RAILED_RIGHT_SHANK_PARTICIPANTS
        segments: dict[str, Segment] = {}
        for name, (acc, gyr) in inertial.items():
            stream_meta: dict[str, object] = {
                "deposited_units": {"acc": "g", "gyr": "deg/s", "t": "s"},
                "hdf5_group": f"imu/{SENSORS[name]}",
                "acc_norm_median_ms2": float(np.median(np.linalg.norm(acc, axis=1))),
                "gyro_zero_rate_offset_corrected": False,
                "acc_scale_corrected": False,
            }
            if railed and name == "right_shank":
                stream_meta["accelerometer_railed"] = (
                    "at the +-16 g full-scale end-stop in every recording of this participant "
                    "; exposed, not excluded"
                )
            if name in ("right_foot", "left_foot"):
                stream_meta["acc_scale_low"] = "gravity magnitude reads 4-7 % low; not corrected"

            names = MARKERS[name]
            columns = _marker_columns(marker_names, name, f"/participants/{subject}/{trial}")
            block = np.stack([marker_data[:, 3 * i : 3 * i + 3] for i in columns], axis=1)
            shape, shape_trial = shapes[name]
            if shape is None:
                rotations = np.full((len(block), 3, 3), np.nan)
                centroid = np.full((len(block), 3), np.nan)
            else:
                rotations, centroid = _fit_frames(block, shape)
            segments[name] = Segment(
                name=name,
                sensor=SensorStream(
                    t=t_imu,
                    gyr=gyr,
                    acc=acc,
                    rate_hz=RATE_HZ,
                    family=SENSOR_FAMILY,
                    range_gyr=GYR_FULL_SCALE_DPS * DEG_TO_RAD,
                    range_acc=ACC_FULL_SCALE_G * G_MS2,
                    native_rate_hz=NATIVE_RATE_HZ,
                    meta=stream_meta,
                ),
                reference=Pose(
                    t=t_ref,
                    R=rotations,
                    p=centroid,
                    rate_hz=RATE_HZ,
                    kind="skin",
                    markers=block,
                    marker_names=list(names),
                    meta={
                        "frame": (
                            "Procrustes fit of the segment marker cloud against the participant's segment "
                            f"shape, which is the iterative Procrustes ({PROCRUSTES_ITERATIONS} iterations) of "
                            "that same cloud in one recording of the participant, as the dataset probe "
                            " estimates it; the same shape in all fourteen recordings, so "
                            "the frame is the same segment-fixed frame throughout"
                        ),
                        "shape_from_trial": shape_trial,
                        "origin": "marker-cloud centroid",
                        "deposited_units": "mm",
                        "on_housing": False,
                        **({"marker_correspondence_repair": {
                            "canonical_to_deposited": dict(ANKLE_LABEL_REPAIR),
                            "source_preserved": True,
                        }} if f"/participants/{subject}/{trial}" == ANKLE_LABEL_REPAIR_GROUP
                           and name == "right_shank" else {}),
                    },
                ),
                anatomical=None,
                mount=None,
            )

        pairs = [
            Pair(parent=parent, child=child, joint=joint, joint_dof=dof, meta={"plain_name": plain})
            for parent, child, joint, dof, plain in PAIRS
        ]

        defects: dict[str, object] = {
            "gyro_zero_rate_offset_uncorrected": (
                "9-18 deg/s on the foot units, a few deg/s on trunk and pelvis, stable within a participant "
                "; left in the data by design"
            ),
            "foot_accelerometer_scale_low": "both foot units read 4-7 % low in gravity magnitude; not corrected",
            "reference_lag_ms": REFERENCE_LAG_MS,
            "reference_lag_spread_ms": REFERENCE_LAG_SPREAD_MS,
            "reference_lag_corrected": False,
            "no_marker_on_any_housing": "reference is a bone-landmark frame; sensor-to-segment alignment unobservable",
            "mounting_undocumented": True,
            "no_initial_rest_period": "not one of the 140 recordings has 0.2 s with all eight units below 10 deg/s",
            "synthetic_timestamps": "t = i/100 exactly; dropped samples cannot be audited",
            "imu_raw_layout_attribute_wrong": "/imu/raw is interleaved, not the block layout its attribute claims",
            "released_kinematics_unusable_as_reference": (
                "OpenSim gait2354 fit: knee and ankle one degree of freedom each, subtalar identically zero"
            ),
        }
        if railed:
            defects["railed_accelerometers"] = {
                "sensor": "right_shank",
                "note": (
                    "at the +-16 g end-stop in all fourteen recordings of P05, P06, P07 and P10 "
                    ". Not excluded: the characterisation measures detect it."
                ),
            }
        if (subject, trial) == ("P08", "overground/round"):
            defects["marker_stream_short"] = (
                "19,306 marker samples against 19,407 inertial samples, a 101-sample gap where every other "
                "recording differs by exactly one"
            )

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],
            contacts_t=None,
            rest_annotation_s=None,
            trial_type=mode,
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "figshare+ (one HDF5 file; the per-participant video archives are not downloaded)",
                "sensor_family": SENSOR_FAMILY,
                "condition": condition,
                "reference": "Vicon Vero, twelve cameras, 100 Hz, 34 skin and bony-landmark markers",
                "clock": (
                    "hardware trigger at the start, manual trim at the end; the released time vectors put the "
                    "markers one row later than the inertial stream and are used as deposited"
                ),
                "defects": defects,
            },
        )


ADAPTER = Wmcg()
