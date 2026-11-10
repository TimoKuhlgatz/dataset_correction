"""Geneva asymptomatic gait and movements (Grouvel et al. 2023): eight Physilog6S sensors, each in a 3D-printed
housing carrying a four-marker cluster, so the optical system measures the sensor frame; plus 37 anatomical skin
markers and six deposited optical joint centres. Licence CC BY 4.0.

Files read (root from ``paths.dataset_dir("grouvel")``):
- ``P<nn>_S01/SYNC_DATA/P<nn>_S01_<TrialType>_<NN>.csv``: the deposit's synchronised record at the optical
  100 Hz: 231 marker columns in mm, then ``P6_<SEG>_acc_{x,y,z}`` in g and ``P6_<SEG>_gyro_{x,y,z}`` in deg/s.
- ``P<nn>_S01/RAW_DATA/P<nn>_S01_<SEG>_Inertial_sensor.BIN`` (``load_raw`` only): the vendor's container of the
  whole 256 Hz session, 512-byte pages, accelerometer at 16 g / 32768 and gyroscope at 2000 deg/s / 32768 per count.
Not read: the ``.c3d`` files, the spreadsheets, the magnetometer and barometer channels.

Units: markers and joint centres mm -> m; accelerometer g -> m/s^2 (x 9.80665); gyroscope deg/s -> rad/s.

Time base: the row index at 100 Hz (``x_Time``, present only with insoles, is the insole clock and is not used).
The deposited alignment of IMU and optical system is served as published, no timing correction is applied.

Reference (``kind="housing"``): the four housing markers Procrustes-fitted to one cluster template per session
(the first calibration task); exact-zero markers (the deposit's occlusion code) become NaN. The per-frame worst
inter-marker distance deviation is in ``meta["rigidity_deviation_m"]``, since the deposit's gap filling silently
distorts the clusters in 0.6-1.9 % of frames. Anatomical frames use the fixed three-point construction of
``_frames.frame_from_three_points`` (not a clinical convention).

Deposit defects left in the data: the deposited IMU columns are mistimed beyond a shift (rebuilt from the
``.BIN`` files by ``scripts/rebuild_geneva_from_raw.py``; ``load`` still serves the deposit); the eight IMUs are
not exactly co-timed; the 0.5 s ``Static`` trials repeat one marker frame; P05 has no left-thigh IMU (segment
served with ``sensor=None``), P03 has no ``2minWalk``; the ``Synchronization`` trials have no housing reference;
``P01_S01_LT_Inertial_sensor.BIN`` is written out of page order and loses 38.8 s (page accounting in
``meta["page_accounting"]``).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream
from ._frames import (
    DEG,
    STANDARD_GRAVITY,
    cluster_pose,
    cluster_rigidity_deviation,
    cluster_template,
    frame_from_three_points,
)

DATASET = "grouvel-geneva"
FOLDER = "grouvel"

RATE_HZ = 100.0
"""The synchronised record's grid. Data-free: the optical rate stated by the depositors and verified in the
checked against ``POINT:RATE = 100.0`` in every ``.c3d`` and 328/332 exact row-count matches.

Units: Hz.
Provenance: read-from-dataset — the grid the synchronised record is written on."""

NATIVE_RATE_HZ = 256.0
"""The Physilog6S hardware rate; the synchronised CSVs are resampled from it.

Units: Hz.
Provenance: read-from-dataset — the unit's hardware rate before the deposit's resampling."""

SEGMENTS: dict[str, str] = {
    # deposit code -> segment name used across the characterisation module
    "TR": "trunk",
    "SA": "pelvis",
    "LT": "left_thigh",
    "LS": "left_shank",
    "LF": "left_foot",
    "RT": "right_thigh",
    "RS": "right_shank",
    "RF": "right_foot",
}

# The seven connected pairs. Cross-body pairs are not connected and are deliberately absent.
PAIRS: list[tuple[str, str, str, int, str, str | None]] = [
    # parent, child, joint, dof, anatomical name, deposited optical joint centre
    ("pelvis", "trunk", "ball", 3, "lumbar / thoracolumbar spine (compound)", None),
    ("pelvis", "left_thigh", "ball", 3, "hip", "LHJC"),
    ("pelvis", "right_thigh", "ball", 3, "hip", "RHJC"),
    ("left_thigh", "left_shank", "hinge", 1, "knee (hinge with laxity and a moving axis)", "LKJC"),
    ("right_thigh", "right_shank", "hinge", 1, "knee (hinge with laxity and a moving axis)", "RKJC"),
    ("left_shank", "left_foot", "2-dof", 2, "ankle complex (talocrural + subtalar)", "LAJC"),
    ("right_shank", "right_foot", "2-dof", 2, "ankle complex (talocrural + subtalar)", "RAJC"),
]
"""Units: degrees of freedom.
Provenance: read-from-dataset — the connected pairs of the deposit and each joint's degrees of freedom."""

# origin, first-axis point, plane point — the three landmarks per segment, all deposited marker names.
ANATOMICAL_LANDMARKS: dict[str, tuple[str, str, str]] = {
    "pelvis": ("midASIS", "RASI", "SACR"),
    "trunk": ("CLAV", "C7", "STRN"),
    "left_thigh": ("LKJC", "LHJC", "LTHI"),
    "right_thigh": ("RKJC", "RHJC", "RTHI"),
    "left_shank": ("LAJC", "LKJC", "LTIB"),
    "right_shank": ("RAJC", "RKJC", "RTIB"),
    "left_foot": ("LHEE", "LTOE", "LFMH"),
    "right_foot": ("RHEE", "RTOE", "RFMH"),
}

TRIAL_TYPES: dict[str, str] = {
    "Gait": "walking",
    "SlowGait": "walking",
    "FastGait": "walking",
    "2minWalk": "walking",
    "Running": "running",
    "TUG": "functional",
    "SitToStand": "functional",
    "CalibrationTask": "functional",
    "Sitting": "static",
    "Static": "static",
    "Synchronization": "synchronisation",
}

BIN_ACC_CHANNEL, BIN_GYR_CHANNEL = 0x13, 0x14
BIN_ACC_FULL_SCALE_G = 16.0
"""Units: g.
Provenance: read-from-dataset — the vendor full-scale range of the unit."""
BIN_GYR_FULL_SCALE_DPS = 2000.0
"""Vendor full-scale ranges of the Physilog6S, and the ``.BIN`` channel identifiers decoded from the deposit

Units: °/s.
Provenance: read-from-dataset — the vendor full-scale range of the unit."""


def _marker(frame, name: str) -> np.ndarray | None:
    cols = [f"{name}_{a}" for a in "xyz"]
    if not all(c in frame.columns for c in cols):
        return None
    v = frame[cols].to_numpy(dtype=float) / 1000.0  # millimetres -> metres
    zero = np.all(v == 0.0, axis=1)  # the deposit's exact-zero occlusion sentinel
    v[zero] = np.nan
    return v


class GrouvelAdapter(Adapter):
    """Ten participants, one session each, 337 synchronised trials, eight segments."""

    name = DATASET
    reference_kind = "housing"
    sensor_family = "GaitUp Physilog6S"
    licence = "CC BY 4.0 (Yareta, Université de Genève; SPDX CC-BY-4.0 in the host's machine-readable record)"
    source = "doi:10.26037/yareta:xkxgaw6ewjdhfntdhtj7upepxy — see <dataset root>/SOURCE.md"

    def __init__(self, root: os.PathLike | str | None = None) -> None:
        self.root = dataset_dir(FOLDER) if root is None else root
        self._index: dict[tuple[str, str], str] | None = None
        self._templates: dict[str, dict[str, np.ndarray | None]] = {}

    # -- discovery -------------------------------------------------------------------------------------

    def _build_index(self) -> dict[tuple[str, str], str]:
        if self._index is None:
            index: dict[tuple[str, str], str] = {}
            for path in sorted(glob.glob(os.path.join(str(self.root), "P??_S01", "SYNC_DATA", "*.csv"))):
                m = re.match(r"^(P\d\d)_S01_(.+)$", os.path.basename(path)[: -len(".csv")])
                if m is not None:
                    index[(m.group(1), m.group(2))] = path
            self._index = index
        return self._index

    def trials(self) -> list[tuple[str, str]]:
        return sorted(self._build_index())

    def raw_subjects(self) -> list[str]:
        """Participants for whom the continuous ``.BIN`` session stream is present."""
        found = set()
        for path in glob.glob(os.path.join(str(self.root), "P??_S01", "RAW_DATA", "*.BIN")):
            m = re.match(r"^(P\d\d)_S01_", os.path.basename(path))
            if m is not None:
                found.add(m.group(1))
        return sorted(found)

    # -- the referenced record -------------------------------------------------------------------------

    def _session_template(self, subject: str, code: str) -> np.ndarray | None:
        """The housing cluster's shape in the participant's first calibration task (first trial otherwise): one
        body-fixed reference frame per sensor for the whole session. With each trial's own median
        shape the frame turned with the participant's median pose in the room, 177–180° between the calibration
        tasks and the gait passes, so no sensor-to-reference rotation held across trials."""
        import pandas as pd

        if subject not in self._templates:
            trials = sorted(t for s, t in self.trials() if s == subject)
            first = next((t for t in trials if t.startswith("CalibrationTask")), trials[0])
            frame = pd.read_csv(self._build_index()[(subject, first)])
            shapes = {}
            for seg_code in SEGMENTS:
                cluster = [_marker(frame, f"{seg_code}{i}") for i in (1, 2, 3, 4)]
                if all(c is not None for c in cluster):
                    shapes[seg_code] = cluster_template(np.stack(cluster, axis=1))
            self._templates[subject] = shapes
        return self._templates[subject].get(code)

    def load(self, subject: str, trial: str) -> Recording:
        import pandas as pd  # imported here so ``trials()`` costs nothing

        path = self._build_index()[(subject, trial)]
        frame = pd.read_csv(path)
        n = len(frame)
        t = np.arange(n) / RATE_HZ  # the row index at 100 Hz; never ``x_Time`` (defect 5)
        trial_type = TRIAL_TYPES.get(trial.rsplit("_", 1)[0], "unknown")

        segments: dict[str, Segment] = {}
        for code, seg_name in SEGMENTS.items():
            acc_cols = [f"P6_{code}_acc_{a}" for a in "xyz"]
            gyr_cols = [f"P6_{code}_gyro_{a}" for a in "xyz"]
            sensor = None
            if all(c in frame.columns for c in acc_cols + gyr_cols):
                sensor = SensorStream(
                    t=t,
                    gyr=frame[gyr_cols].to_numpy(dtype=float) * DEG,  # °/s -> rad/s
                    acc=frame[acc_cols].to_numpy(dtype=float) * STANDARD_GRAVITY,  # g -> m/s²
                    rate_hz=RATE_HZ,
                    family=self.sensor_family,
                    range_gyr=BIN_GYR_FULL_SCALE_DPS * DEG,
                    range_acc=BIN_ACC_FULL_SCALE_G * STANDARD_GRAVITY,
                    native_rate_hz=NATIVE_RATE_HZ,
                    meta={
                        "deposit_code": code,
                        "deposited_units": {"acc": "g", "gyr": "deg/s"},
                        "resampled": "256 Hz native, resampled and cropped to the optical 100 Hz grid",
                        "alignment": "as deposited; see the recording's clock_offset_* meta",
                    },
                )

            reference = None
            cluster = [_marker(frame, f"{code}{i}") for i in (1, 2, 3, 4)]
            if all(c is not None for c in cluster):
                markers = np.stack(cluster, axis=1)  # (T, 4, 3) metres
                rot, pos = cluster_pose(markers, self._session_template(subject, code))
                reference = Pose(
                    t=t,
                    R=rot,
                    p=pos,
                    rate_hz=RATE_HZ,
                    kind="housing",
                    markers=markers,
                    marker_names=[f"{code}{i}" for i in (1, 2, 3, 4)],
                    meta={
                        "construction": (
                            "rigid (Procrustes) fit of the four housing-cluster markers to their median shape "
                            "in the participant's first calibration task, so the frame is the same housing-fixed "
                            "frame in every trial of the participant; it is defined up to that arbitrary shape"
                        ),
                        "rigidity_deviation_m": cluster_rigidity_deviation(markers),
                        "rigidity_note": (
                            "per frame, the largest deviation of any of the six inter-marker distances from "
                            "its trial median; the deposit's gap fill reaches 69–311 mm in 0.6–1.9 % of "
                            "frames and is unflagged. No frame is removed here."
                        ),
                        "reference_accuracy_limit_deg": (
                            "0.79° median per segment, 2.30° at the 95th percentile, ≈ 1.1° on a relative "
                            "orientation, before the timing defect"
                        ),
                    },
                )

            anatomical = None
            names = ANATOMICAL_LANDMARKS.get(seg_name)
            if names is not None:
                pts = [_marker(frame, nm) for nm in names]
                if all(p is not None for p in pts):
                    with np.errstate(invalid="ignore"):
                        rot = frame_from_three_points(*pts)
                    anatomical = Pose(
                        t=t,
                        R=rot,
                        p=pts[0],
                        rate_hz=RATE_HZ,
                        kind="anatomical",
                        markers=np.stack(pts, axis=1),
                        marker_names=list(names),
                        meta={
                            "construction": (
                                "this adapter's fixed three-point convention (_frames.frame_from_three_points) "
                                f"from {names}; not a clinical convention — only its variation against the "
                                "housing cluster is meaningful"
                            ),
                            "side": "reference-side: skin markers, so the frame carries skin motion",
                        },
                    )

            if sensor is None and reference is None and anatomical is None:
                continue
            segments[seg_name] = Segment(
                name=seg_name,
                sensor=sensor,
                reference=reference,
                anatomical=anatomical,
                mount=(
                    "3D-printed marker-cluster housing on a Qualisys SuperWrap elastic strap "
                    "(trunk: harness with an elasticated band)"
                ),
            )

        pairs: list[Pair] = []
        for parent, child, joint, dof, anatomical_name, centre_name in PAIRS:
            if parent not in segments or child not in segments:
                continue
            if segments[parent].sensor is None or segments[child].sensor is None:
                continue  # P05's left thigh: markers but no IMU, so the two pairs across it are not scorable
            centre = _marker(frame, centre_name) if centre_name is not None else None
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
                            f"deposited virtual marker {centre_name} (hip centres from Hara's regression, "
                            "knee and ankle centres from a chord function)"
                            if centre is not None
                            else "none deposited for this pair"
                        ),
                    },
                )
            )

        lag_table = os.path.join(str(self.root), "_probe", "trial_lag.csv")
        return Recording(
            dataset=DATASET,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],  # one sensor per segment
            contacts_t=None,  # force plates ship in the .c3d only, which this adapter does not read
            rest_annotation_s=None,  # no rest period is annotated anywhere in the deposit
            trial_type=trial_type,
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "Yareta, Université de Genève",
                "sensor_family": self.sensor_family,
                "reference": "Qualisys Oqus 7+, 12 cameras, 100 Hz, four markers on every housing",
                "path": path,
                "alignment": "as deposited — no timing repair is applied here (defect 1)",
                "clock_offset_review_table": lag_table if os.path.exists(lag_table) else None,
                "clock_offset_review_summary": (
                    "IMU-to-optical offset median −0.60 s, 5th–95th percentile −1.01 to −0.12 s, worst "
                    "−1.12 s, against a stated 0.1 s; "
                    "between the eight IMUs, 0.03–0.12 s within a trial"
                ),
                "known_defects": [
                    "published synchronisation off by ≈ 0.6 s, exposed as deposited (§8.1)",
                    "the eight IMUs are not exactly co-timed (§8.2)",
                    "silent marker gap-fill corruption, exposed per frame as rigidity_deviation_m (§8.3)",
                    "Static trials are 0.5 s with one marker frame repeated fifty times (§8.4)",
                    "x_Time is the insole clock and is not used (§8.6)",
                    "P05 has no left-thigh IMU; P03 has no 2minWalk trials (§8.8)",
                    "longest referenced trial 79.85 s; the 2minWalk recordings are 2.25–4.54 s passes (§8.9)",
                    (
                        "P01's left-thigh raw .BIN is out of page order and loses 38.8 s, "
                        "affecting load_raw only"
                    ),
                ],
                "raw_stream": "load_raw(subject) serves the continuous 256 Hz .BIN session without reference",
            },
        )

    # -- the continuous native-rate stream -------------------------------------------------------------

    def load_raw(self, subject: str) -> Recording:
        """The unbroken ≈ 69-minute 256 Hz session from the ``.BIN`` files, with **no reference**.

        Returns one ``Recording`` (the type every measure consumes) whose segments carry the native-rate
        ``SensorStream`` and ``reference=None``, ``anatomical=None``; ``pairs`` is empty because nothing here
        is referenced. This is the stream for the envelope measures — noise level, effective bandwidth and
        range saturation (``docs/characterisation.md`` §2.11) — which the resampled 100 Hz record cannot
        answer. The eight sensors were switched on 12–32 s apart, so their sessions start at different wall
        clock times and are **not** mutually aligned: each stream's ``t`` starts at 0 at its own first
        sample, and the ``.BIN`` header's absolute start time is carried in the stream's ``meta`` for
        whoever wants to align them. Pages lost by the logger (0–10 per file, ≤ 0.02 % of a session) leave
        NaN rows at their true sample indices rather than shifting the stream, and the per-file page
        accounting (missing pages, duplicated pages, whether the file is in page order) rides on each
        stream's ``meta``. One file is badly affected: ``P01``'s left thigh loses 38.8 s in one hole, see
        ``_decode_bin``.
        """
        segments: dict[str, Segment] = {}
        for code, seg_name in SEGMENTS.items():
            path = os.path.join(
                str(self.root), f"{subject}_S01", "RAW_DATA", f"{subject}_S01_{code}_Inertial_sensor.BIN"
            )
            if not os.path.exists(path):
                continue
            acc, gyr, accounting, start = _decode_bin(path)
            segments[seg_name] = Segment(
                name=seg_name,
                sensor=SensorStream(
                    t=np.arange(len(acc)) / NATIVE_RATE_HZ,
                    gyr=gyr,
                    acc=acc,
                    rate_hz=NATIVE_RATE_HZ,
                    family=self.sensor_family,
                    range_gyr=BIN_GYR_FULL_SCALE_DPS * DEG,
                    range_acc=BIN_ACC_FULL_SCALE_G * STANDARD_GRAVITY,
                    native_rate_hz=NATIVE_RATE_HZ,
                    meta={
                        "deposit_code": code,
                        "deposited_units": {"acc": "int16 counts, 16 g full scale", "gyr": "int16 counts, 2000 °/s"},
                        "path": path,
                        "logger_start_time": start,
                        "page_accounting": accounting,
                        "missing_sample_fraction": float(np.isnan(acc).any(axis=1).mean()),
                        "sample_index_origin": (
                            "from the page's cumulative sample counter; the origin is determined to ±1 sample "
                            "(±4 ms), which is why this stream is for envelope measures and not for timing"
                        ),
                    },
                ),
                reference=None,
                anatomical=None,
                mount=(
                    "3D-printed marker-cluster housing on a Qualisys SuperWrap elastic strap "
                    "(trunk: harness with an elasticated band)"
                ),
            )
        if not segments:
            raise FileNotFoundError(f"no .BIN files for {subject} under {self.root}")
        return Recording(
            dataset=DATASET,
            subject=subject,
            trial="raw_session",
            segments=segments,
            pairs=[],
            rest_annotation_s=None,
            trial_type="session",
            meta={
                "licence": self.licence,
                "source": self.source,
                "reference": "none — this stream carries no optical data",
                "note": (
                    "≈ 1.06 million samples = 4136–4147 s per sensor; the optical passes cover only "
                    "7.5–12.5 % of a session"
                ),
                "page_accounting_note": (
                    "each stream reports missing_pages / duplicate_pages / pages_in_order; P01's left thigh "
                    "is the one file that is out of order and loses 38.8 s"
                ),
            },
        )


def _decode_bin(path: str) -> tuple[np.ndarray, np.ndarray, dict, str | None]:
    """Decode one Physilog6S ``.BIN`` into (acc m/s², gyr rad/s, page accounting, logger start time).

    Layout: see the module description.
    Samples are placed at the index implied by each page's **cumulative sample counter**, not at their file
    position, so a missing page leaves NaN rows instead of shifting everything after it and an out-of-order
    page lands where it belongs. The page accounting is returned rather than acted on:

    * ``missing_pages`` — page indices absent from the span the file covers;
    * ``duplicate_pages`` — page indices that occur more than once;
    * ``pages_in_order`` — whether the page index increases monotonically through the file.

    Measured: 15 of the 16 ``.BIN`` files of participants P01 and
    P03 are in order and miss 0–10 pages (≤ 0.02 % of a session), but
    ``P01_S01_LT_Inertial_sensor.BIN`` is **not**: a block of 512 pages appears twice at the wrong file
    offset and 517 page indices are absent, so that stream carries one 9,925-sample (38.8 s) hole, which this
    decoder leaves as NaN. That is a defect of the deposit and is exposed, never patched.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    n_pages = len(raw) // 512
    pages = raw[: n_pages * 512].reshape(n_pages, 512)
    counter = (
        (pages[:, 4].astype(np.int64) << 24)
        | (pages[:, 5].astype(np.int64) << 16)
        | (pages[:, 6].astype(np.int64) << 8)
        | pages[:, 7].astype(np.int64)
    )
    page_index = ((pages[:, 2].astype(np.int64) << 8) | pages[:, 3].astype(np.int64))[1:]
    records = pages[:, 8 : 8 + 63 * 8].reshape(n_pages, 63, 8)[1:]  # page 0 is the text-tagged header
    counter = counter[1:]
    unique, occurrences = np.unique(page_index, return_counts=True)
    accounting = {
        "pages": int(page_index.size),
        "missing_pages": int(page_index.max() - page_index.min() + 1 - unique.size),
        "duplicate_pages": int((occurrences > 1).sum()),
        "pages_in_order": bool(np.all(np.diff(page_index) > 0)),
    }

    channel = records[:, :, 0]
    value = (records[:, :, 2::2].astype(np.int32) << 8) | records[:, :, 3::2].astype(np.int32)
    value = np.where(value >= 32768, value - 65536, value)
    # The counter steps by two per sample and every record carries its own step within the page, so
    # (page counter + record step) / 2 is the record's sample index. A sample's accelerometer and gyroscope
    # records can lie on two pages; an ordinal within the page would put that page's gyroscope one sample late
    # (a third of the gyroscope samples, and one NaN per such page).
    sample = (counter[:, None] + records[:, :, 1].astype(np.int64)) // 2
    sample -= sample.min()

    out = {}
    for chan, scale in ((BIN_ACC_CHANNEL, BIN_ACC_FULL_SCALE_G * STANDARD_GRAVITY / 32768.0),
                        (BIN_GYR_CHANNEL, BIN_GYR_FULL_SCALE_DPS * DEG / 32768.0)):
        hit = channel == chan
        idx = sample[hit]
        n = int(idx.max()) + 1
        arr = np.full((n, 3), np.nan)
        arr[idx] = value[hit] * scale
        out[chan] = arr

    n = min(len(out[BIN_ACC_CHANNEL]), len(out[BIN_GYR_CHANNEL]))
    start = _bin_start_time(pages[0])
    return out[BIN_ACC_CHANNEL][:n], out[BIN_GYR_CHANNEL][:n], accounting, start


def _bin_start_time(page0: np.ndarray) -> str | None:
    """The logger's absolute start time from the page-0 tag ``0x0300``."""
    s = bytes(page0)
    i = 4
    while i < len(s) - 5:
        if s[i] == 0x50 and s[i + 1] == 0x35:
            tag, length = (s[i + 2], s[i + 3]), s[i + 4]
            payload = s[i + 5 : i + 5 + length]
            if tag == (0x03, 0x00) and len(payload) >= 7:
                year = (payload[5] << 8) | payload[6]
                return f"{year:04d}-{payload[4]:02d}-{payload[3]:02d} {payload[2]:02d}:{payload[1]:02d}:{payload[0]:02d}"
            i += 5 + length
            if s[i : i + 2] != b"\xff\xfe":
                break
            i += 2
        else:
            i += 1
    return None


ADAPTER = GrouvelAdapter()
