"""Trento upper limb (Bonfiglio et al., Front. Bioeng. Biotechnol. 12:1385750, 2024): five Movella DOT on one arm,
an Optotrak marker cluster on every housing. Deposit: Zenodo doi:10.5281/zenodo.10935873, CC BY 4.0.

Files read (the deposit's derived products are never read: its Optotrak product replaces occluded samples by 0
and low-passes everything at 3 Hz, its DOT product aligns the five sensors by row index):
- ``Data/subject <n>/0. raw data/DOT/*.csv``: one DOT export per sensor and trial, 120 Hz. The leading ``sep=,``
  line is sniffed and the first data row, all-zero on accelerometer and gyroscope in every file, is dropped.
- ``.../Optotrack/Trials/*_<NNN>_3d.xls``: tab-separated text (not Excel), 15 cluster markers, 120 Hz, mm; blank
  fields where a marker is occluded.
- ``.../Optotrack/Bony Landmark/*.xls``: the 19 probe digitisations, reconstructed as the deposit's MATLAB does.

Units: gyroscope deg/s -> rad/s (unit undocumented, established from the sensor's own quaternion increments);
accelerometer already m/s^2, gravity included; SampleTimeFine microseconds -> s, unwrapped modulo 2^32 us; markers
mm -> m.

Time bases: every sensor keeps its own recorded sample times relative to the smallest SampleTimeFine of the trial,
so the start skew between sensors (2-16 samples) stays visible. The optical time base is ``arange(n) / 120`` from
0 s: there is no hardware synchronisation. The lag between the systems is measured (``_bonfiglio_session.py``) and
reported in ``meta["reference_lag_s"]``, but not applied (APPLY_REFERENCE_LAG), so the synchronisation-offset
check sees the deposit's timing.

Trial pairing: optical files and IMU trials do not match by index (some subjects have 21-23 optical files for 20
IMU trials, subject 4 has 18). ``pairing="repaired"`` (reader ``bonfiglio-arm``) pairs each IMU trial with the
optical file whose angular rate correlates best, in one monotone assignment per session;
``pairing="as_deposited"`` (reader ``bonfiglio-arm-deposited``) pairs the k-th IMU trial with the k-th optical
file, as the deposit's own script assumes.

Reference (``kind="housing"``): per segment the three cluster markers are Procrustes-fitted to one marker template
of the session; frames with an occluded marker are NaN. ``Pose.R`` is the IMU's sensor frame in the world: the
cluster frame turned by the constant cluster-to-sensor rotation measured per sensor and session from angular rate
(and gravity where a sensor barely turns). Where no such rotation could be fitted at all (subject 9, whose sensors
share no clock) the cluster frame is served as it is; where the fit found no constant rotation (the scapula of
every subject: the cluster moves 15-42 degrees against the IMU between trials) the segment has no reference. The
world is turned from the deposit's y-up to z-up. ``Pose.p`` is the cluster centroid.

Anatomical frames: ISB thorax and scapula frames from the landmarks; ``None`` for upper arm, forearm and hand,
whose own landmarks do not define an ISB frame.

Deposit defects left in the data: subject 9's sensors are not on a common SampleTimeFine timeline
(``meta["common_sensor_timeline"]``); occluded frames are NaN, never interpolated; subject 2 wears the set on the
left arm with right-side names (``meta["worn_side"]``); subject 12 lacks the T8 and AA digitisations (no thorax or
scapula frame); two trials of subject 4 have no optical file.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream
from . import _bonfiglio_session as _repair

APPLY_REFERENCE_LAG = False
"""False: the optical time base is served as deposited (starting at 0 s, like the sensors' common epoch) and the
references of trials whose lag could not be measured are kept, so the synchronisation-offset check sees the
deposit's timing. True shifts the optical time base by the measured lag (``_bonfiglio_session.py``) and drops the
reference of trials whose lag could not be measured."""

RATE_HZ = 120.0
"""Units: Hz.
Provenance: read-from-dataset — the rate the deposit is written at."""
MM_TO_M = 1e-3
"""Units: m per mm.
Provenance: physical — an exact unit conversion."""
DEG_TO_RAD = np.pi / 180.0
"""Units: rad per degree.
Provenance: physical — an exact unit conversion."""
US_TO_S = 1e-6
"""Units: s per microsecond.
Provenance: physical — an exact unit conversion."""
STF_WRAP_US = 2**32

#: Segment name → (DOT filename tag, its three housing-cluster marker labels).
SEGMENTS = {
    "trunk": ("TRK", ("ST1", "ST2", "ST3")),
    "scapula": ("RSC", ("SC1", "SC2", "SC3")),
    "upper_arm": ("RUA", ("UA1", "UA2", "UA3")),
    "forearm": ("RLA", ("LA1", "LA2", "LA3")),
    "hand": ("RHA", ("HA1", "HA2", "HA3")),
}

#: (parent, child, joint, dof). The scapulothoracic pair is not a bony joint and has no joint centre.
PAIRS = [
    ("trunk", "scapula", "none", 3),
    ("scapula", "upper_arm", "ball", 3),
    ("upper_arm", "forearm", "2-dof", 2),
    ("forearm", "hand", "2-dof", 2),
]
"""Units: degrees of freedom.
Provenance: read-from-dataset — the connected pairs of the deposit and the degrees of freedom of each joint."""

#: Protocol order of the twenty trials, from the deposit's ``Codes/B_sortDotData.m`` output folders.
TRIAL_NAMES = [
    "n-pose",
    "elbow-flexion",
    "elbow-pronation",
    "wrist-flexion",
    "wrist-deviation",
    "shoulder-internal-rotation",
    "shoulder-flexion",
    "shoulder-abduction",
    "scapula-elevation",
    "scapula-protraction",
    *[f"task-{i}" for i in range(1, 11)],
]

#: Landmark order of ``bony_landmarks_001..019``, from ``Codes/A_Optotrack_sortData.m``.
LANDMARK_ORDER = [
    "C7", "T8", "PX", "IJ", "AC", "SC", "AI", "AA", "PC", "TS",
    "EL", "EM", "US", "RS", "MC3", "MCP2", "MCP3", "MCP5", "LAC",
]  # fmt: skip

#: Which cluster each landmark was digitised against (``computeTipPosition.m``).
LANDMARK_SEGMENT = {
    "C7": "trunk", "T8": "trunk", "PX": "trunk", "IJ": "trunk", "LAC": "trunk",
    "AC": "scapula", "SC": "scapula", "AI": "scapula", "AA": "scapula", "PC": "scapula", "TS": "scapula",
    "EL": "upper_arm", "EM": "upper_arm",
    "US": "forearm", "RS": "forearm",
    "MC3": "hand", "MCP2": "hand", "MCP3": "hand", "MCP5": "hand",
}  # fmt: skip

#: NDI digitising-probe tip offset in the probe frame, mm. Provenance: physical, quoted from the vendor
#: manual by the deposit's own ``computeTipPosition.m`` ("this is the only thing we need (from manual)").
PROBE_TIP_OFFSET_MM = np.array([-6.0, 0.0, -120.0])
"""Units: mm.
Provenance: read-from-dataset — the pointer geometry the depositors state."""

#: Movella DOT gyroscope full scale, deg/s (a check of the deposit measured a maximum of 1,996 units).
REFERENCE_LAG_REVIEW_MEDIAN_S = -0.733
"""Median lag of the optical reference behind the inertial streams, measured on the deposit.

Units: s.
Provenance: read-from-dataset — read from the deposit."""

DOT_GYRO_RANGE_DPS = 2000.0
"""Units: °/s.
Provenance: read-from-dataset — the unit's full-scale range as the vendor states it."""

#: Cluster orientation noise RMS in degrees, static N-pose of subject 1.
REFERENCE_ACCURACY_LIMIT_DEG = {
    "trunk": 0.119,
    "scapula": 0.189,
    "upper_arm": 0.089,
    "forearm": 0.090,
    "hand": 0.097,
}
"""Units: degrees.
Provenance: read-from-dataset — read from the deposit."""


def _reference_accuracy_limit(segment: str, fit: dict) -> float:
    """This deposit's reference accuracy limit for one segment, *after* the reference-side repair.

    Two independent terms, added in quadrature: the cluster frame's own orientation noise
    (:data:`REFERENCE_ACCURACY_LIMIT_DEG`, measured on the deposit) and the uncertainty of the constant
    cluster-to-sensor rotation the cluster frame is now rotated by, which the repair measures per sensor and
    session as the scatter of its per-trial fits (``_bonfiglio_session._diagnose``). The second is the larger:
    a number quoted on this dataset is quoted against roughly a degree, not against a tenth of one.
    """
    noise = REFERENCE_ACCURACY_LIMIT_DEG[segment]
    scatter = fit.get("between_trial_scatter_deg_median")
    return float(np.hypot(noise, scatter)) if scatter is not None else float(noise)

_DOT_NAME = re.compile(r"^(\d)([A-Z]+)_([0-9A-F]+)_(\d{8})_(\d{6})\.csv$")
_OPTO_NUMBER = re.compile(r"_(\d+)_3d\.xls$")


def _read_3d_xls(path: Path) -> tuple[dict[str, str], list[str], np.ndarray]:
    """Optotrak ``_3d.xls`` → (header, marker names, (T, K, 3) mm with ``NaN`` where occluded).

    Reproduces::read_3d_xls``.
    """
    lines = path.read_text(errors="replace").replace("\r\n", "\n").split("\n")
    meta: dict[str, str] = {}
    i = 0
    while i < len(lines) and lines[i].strip() and ":" in lines[i]:
        key, value = lines[i].split(":", 1)
        meta[key.strip()] = value.strip()
        i += 1
    while i < len(lines) and not lines[i].startswith("Frame"):
        i += 1
    header = lines[i].rstrip("\t").split("\t")
    names: list[str] = []
    for column in header[1:]:
        name = column.rsplit(" ", 1)[0]
        if name not in names:
            names.append(name)
    rows = [line.rstrip("\t").split("\t")[1:] for line in lines[i + 1 :] if line.strip()]
    arr = np.full((len(rows), len(header) - 1), np.nan)
    for r, parts in enumerate(rows):
        for c, field in enumerate(parts):
            field = field.strip()
            if field:
                try:
                    arr[r, c] = float(field)
                except ValueError:
                    arr[r, c] = np.nan
    arr[np.abs(arr) > 1e6] = np.nan
    return meta, names, arr.reshape(len(rows), -1, 3)


def _read_dot_csv(path: Path):
    """One Movella DOT export → a pandas frame with the leading ``sep=,`` line and the all-zero row gone."""
    import pandas as pd

    with path.open(errors="replace") as handle:
        first = handle.readline()
    frame = pd.read_csv(path, skiprows=1 if first.startswith("sep=") else 0)
    frame.columns = [c.strip() for c in frame.columns]
    return frame.iloc[1:].reset_index(drop=True)


def _unwrap_stf(stf: np.ndarray) -> np.ndarray:
    """``SampleTimeFine`` in µs, monotonic across the 32-bit wrap."""
    v = stf.astype(np.int64)
    return v + STF_WRAP_US * np.concatenate(([0], np.cumsum(np.diff(v) < -STF_WRAP_US // 2)))


def _read_trial_streams(paths: dict[str, Path]) -> dict[str, dict]:
    """One trial's five inertial streams on the trial's common epoch, as the deposit writes them.

    Read once here and used by both :meth:`BonfiglioAdapter.load` and the session repair, so that the streams
    the repair is fitted on and the streams the adapter emits are the same arrays by construction.
    """
    frames = {segment: _read_dot_csv(path) for segment, path in paths.items()}
    counters = {segment: _unwrap_stf(frame["SampleTimeFine"].to_numpy(np.int64))
                for segment, frame in frames.items()}
    epoch = min(int(v[0]) for v in counters.values())
    return {
        segment: {
            "t": (counters[segment] - epoch) * US_TO_S,
            "gyr": frame[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(float) * DEG_TO_RAD,
            "acc": frame[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(float),
            "sample_time_fine": counters[segment],
            "packet_counter": frame["PacketCounter"].to_numpy(np.int64),
            "file": paths[segment].name,
        }
        for segment, frame in frames.items()
    }


def _read_optical_markers(path: Path) -> dict[str, np.ndarray]:
    """One optical trial file → marker name → (T, 3) metres, ``NaN`` where occluded."""
    _, names, arr = _read_3d_xls(path)
    arr = arr * MM_TO_M
    return {name: arr[:, i, :] for i, name in enumerate(names)}


class _Reader:
    """The deposit's raw arrays for one subject, cached for the length of one session build.

    The repair module (``_bonfiglio_session.py``) reads the deposit only through this object, so the file
    formats stay in one place — this adapter — and the repair stays a protocol over arrays.
    """

    def __init__(self, adapter: BonfiglioAdapter) -> None:
        self._adapter = adapter
        self._streams: dict[tuple[str, int], dict] = {}
        self._markers: dict[tuple[str, int], dict] = {}

    def trial_indices(self, subject: str) -> list[int]:
        return sorted(self._adapter._dot_files(subject))

    def optical_indices(self, subject: str) -> list[int]:
        return list(range(len(self._adapter._optical_files(subject))))

    def streams(self, subject: str, trial_index: int) -> dict[str, dict]:
        key = (subject, trial_index)
        if key not in self._streams:
            self._streams[key] = _read_trial_streams(self._adapter._dot_files(subject)[trial_index])
        return self._streams[key]

    def markers(self, subject: str, optical_index: int) -> dict[str, np.ndarray | None]:
        """Segment → (T, 3, 3) cluster markers of one optical file, or ``None`` where a marker is absent."""
        key = (subject, optical_index)
        if key not in self._markers:
            columns = _read_optical_markers(self._adapter._optical_files(subject)[optical_index])
            self._markers[key] = {
                segment: (np.stack([columns[label] for label in labels], axis=1)
                          if all(label in columns for label in labels) else None)
                for segment, (_, labels) in SEGMENTS.items()
            }
        return self._markers[key]


@lru_cache(maxsize=2)
def _session(root: str, subject: str) -> _repair.Session:
    """The whole reference-side repair of one subject, computed once (``_bonfiglio_session.py``).

    Cached by (deposit root, subject) because it costs one pass over the subject's twenty trials and twenty-odd
    optical files, and every trial of that subject needs it. Two subjects are held, which is what a scoring run
    walking the deposit subject by subject needs.
    """
    adapter = BonfiglioAdapter(Path(root))
    return _repair.build(_Reader(adapter), subject, RATE_HZ)


def _local_frame(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``getLocalReference.m`` verbatim: (R with columns x, y, z; origin s1). Broadcasts over leading axes."""

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    v1, v2 = s2 - s1, s3 - s1
    x = unit(np.cross(v1, v2))
    y = unit(np.cross(v1, x))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1), s1


def _probe_tip(tip1: np.ndarray, tip2: np.ndarray, tip3: np.ndarray) -> np.ndarray:
    """``computeTipPosition.m`` verbatim: the digitising probe's tip in the laboratory frame, mm."""
    origin = (tip2 + tip3) / 2.0
    y = (tip3 - tip2) / np.linalg.norm(tip3 - tip2)
    z = (tip1 - tip2) / np.linalg.norm(tip1 - tip2)
    x = np.cross(z, y)
    x = x / np.linalg.norm(x)
    return np.stack([x, y, z], axis=-1) @ PROBE_TIP_OFFSET_MM + origin


@lru_cache(maxsize=4)
def _landmarks_local(subject_dir: str) -> dict[str, tuple[str, np.ndarray]]:
    """Each digitised landmark expressed in its cluster's local frame, metres. Cached per subject (19×3)."""
    folder = Path(subject_dir) / "0. raw data" / "Optotrack" / "Bony Landmark"
    out: dict[str, tuple[str, np.ndarray]] = {}
    if not folder.is_dir():
        return out
    files = sorted(
        (f for f in folder.glob("*_3d.xls") if _OPTO_NUMBER.search(f.name)),
        key=lambda f: int(_OPTO_NUMBER.search(f.name).group(1)),
    )
    for landmark, path in zip(LANDMARK_ORDER, files, strict=False):
        segment = LANDMARK_SEGMENT[landmark]
        _, names, arr = _read_3d_xls(path)
        index = {name: i for i, name in enumerate(names)}
        needed = [*SEGMENTS[segment][1], "TIP1", "TIP2", "TIP3"]
        if any(name not in index for name in needed):
            continue
        columns = {name: arr[:, index[name], :] for name in needed}
        if any(not np.isfinite(v).any() for v in columns.values()):
            continue  # a landmark whose file has a fully occluded marker cannot be reconstructed
        mean = {name: np.nanmean(v, axis=0) for name, v in columns.items()}
        if any(not np.isfinite(v).all() for v in mean.values()):
            continue
        tip = _probe_tip(mean["TIP1"], mean["TIP2"], mean["TIP3"])
        s1, s2, s3 = (mean[name] for name in SEGMENTS[segment][1])
        rotation, origin = _local_frame(s1, s2, s3)
        out[landmark] = (segment, rotation.T @ (tip - origin) * MM_TO_M)
    return out


def _isb_thorax(lm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """ISB thorax frame from IJ, C7, PX, T8 (origin IJ). Returns (R frame→world, p) or None."""
    if not {"IJ", "C7", "PX", "T8"} <= lm.keys():
        return None
    mid_low = (lm["PX"] + lm["T8"]) / 2.0
    y = lm["IJ"] + lm["C7"] - 2.0 * mid_low
    z = np.cross(lm["IJ"] - mid_low, lm["C7"] - mid_low)
    return _orthonormalise(y, z, order="yz"), lm["IJ"]


def _isb_scapula(lm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """ISB scapula frame from AA, AI, TS (origin AA). Returns (R frame→world, p) or None."""
    if not {"AA", "AI", "TS"} <= lm.keys():
        return None
    z = lm["AA"] - lm["TS"]
    x = np.cross(lm["AA"] - lm["TS"], lm["AI"] - lm["TS"])
    return _orthonormalise(x, z, order="xz"), lm["AA"]


def _orthonormalise(first: np.ndarray, second: np.ndarray, order: str) -> np.ndarray:
    """Two (T, 3) direction fields → (T, 3, 3) rotations with columns (x, y, z), right-handed."""

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    a, b = unit(first), unit(second)
    if order == "yz":  # y up, z right, x = y x z
        y, z = a, b
        x = np.cross(y, z)
        z = np.cross(x, y)
    else:  # order == "xz": x forward, z right, y = z x x
        x, z = a, b
        y = np.cross(z, x)
        z = np.cross(x, y)
    return np.stack([unit(x), unit(y), unit(z)], axis=-1)


class BonfiglioAdapter(Adapter):
    """The Bonfiglio upper-limb dataset as the characterisation schema sees it."""

    name = "bonfiglio-arm"
    reference_kind = "housing"
    sensor_family = "Movella DOT"
    licence = "CC BY 4.0"
    source = "eval_datasets/bonfiglio/SOURCE.md — doi:10.5281/zenodo.10935873"

    def __init__(self, root: Path | None = None, pairing: str = "repaired") -> None:
        # `pairing="as_deposited"` pairs the k-th inertial trial with the k-th optical file, as the deposit's own
        # B_sortDotData.m assumes, instead of the correlation-based assignment; the reference is built the same
        # way (session templates and cluster-to-sensor rotations), so only the pairing differs (error 8).
        if pairing not in ("repaired", "as_deposited"):
            raise ValueError(f"pairing must be 'repaired' or 'as_deposited', not {pairing!r}")
        self.pairing = pairing
        if pairing == "as_deposited":
            self.name = "bonfiglio-arm-deposited"
        self.root = root or dataset_dir("bonfiglio") / "_extracted" / "Optotrak vs DOT" / "Data"

    def _subject_dir(self, subject: str) -> Path:
        return self.root / f"subject {subject}"

    def _dot_files(self, subject: str) -> dict[int, dict[str, Path]]:
        """Trial index (0-based, by file timestamp) → segment → CSV path. Filenames only, nothing opened."""
        by_tag: dict[str, list[tuple[str, Path]]] = {}
        for path in (self._subject_dir(subject) / "0. raw data" / "DOT").glob("*.csv"):
            match = _DOT_NAME.match(path.name)
            if match:
                by_tag.setdefault(match.group(2), []).append((match.group(4) + match.group(5), path))
        out: dict[int, dict[str, Path]] = {}
        for segment, (tag, _) in SEGMENTS.items():
            for k, (_, path) in enumerate(sorted(by_tag.get(tag, []))):
                out.setdefault(k, {})[segment] = path
        return out

    def _optical_files(self, subject: str) -> list[Path]:
        folder = self._subject_dir(subject) / "0. raw data" / "Optotrack" / "Trials"
        files = [f for f in folder.glob("*_3d.xls") if _OPTO_NUMBER.search(f.name)]
        return sorted(files, key=lambda f: int(_OPTO_NUMBER.search(f.name).group(1)))

    def trials(self) -> list[tuple[str, str]]:
        """All (subject, trial); built from filenames alone, no file is opened."""
        out: list[tuple[str, str]] = []
        if not self.root.is_dir():
            return out
        subjects = sorted(
            (d.name[len("subject ") :] for d in self.root.glob("subject *") if d.is_dir()),
            key=lambda s: int(s) if s.isdigit() else 0,
        )
        for subject in subjects:
            for k in sorted(self._dot_files(subject)):
                out.append((subject, _trial_id(k)))
        return out

    def load(self, subject: str, trial: str) -> Recording:
        k = int(trial.split("-", 1)[0][1:]) - 1
        subject_dir = self._subject_dir(subject)
        dot = self._dot_files(subject).get(k, {})
        if not dot:
            raise KeyError(f"{self.name}: no DOT files for subject {subject} trial {trial}")

        session = _session(str(self.root), subject)
        streams_raw = _read_trial_streams(dot)
        trunk_start = float(streams_raw["trunk"]["t"][0]) if "trunk" in streams_raw else 0.0
        skew = {segment: float(v["t"][0]) - trunk_start for segment, v in streams_raw.items()}

        streams: dict[str, SensorStream] = {}
        for segment, stream in streams_raw.items():
            streams[segment] = SensorStream(
                t=stream["t"],
                gyr=stream["gyr"],
                acc=stream["acc"],
                rate_hz=RATE_HZ,
                family=self.sensor_family,
                range_gyr=DOT_GYRO_RANGE_DPS * DEG_TO_RAD,
                range_acc=None,  # undocumented in the deposit and in the paper
                meta={
                    "file": stream["file"],
                    "sample_time_fine_start_us": int(stream["sample_time_fine"][0]),
                    "start_skew_s_vs_trunk": skew[segment],
                    "streams_aligned": False,
                    "deposited_gyroscope_unit": "deg/s",
                    "deposited_accelerometer_unit": "m/s^2",
                    "first_row_dropped": "all-zero accelerometer and gyroscope in 1600 of 1600 files",
                    "magnetometer_and_vendor_quaternion_read": False,
                    "packet_counter_gaps": int((np.diff(stream["packet_counter"]) != 1).sum()),
                },
            )

        optical = self._optical_files(subject)
        if self.pairing == "as_deposited":  # see __init__
            optical_index = k if k < len(optical) else None
        else:
            optical_index = session.pairing.get(k)
        optical_path = optical[optical_index] if optical_index is not None else None
        lag = session.lag_s.get(k)
        markers: dict[str, np.ndarray] = {}
        t_ref = np.array([], dtype=float)
        if optical_path is not None:
            # The deposit's optical world frame is y-up (measured, session by session); every marker is put on
            # the adapters' z-up convention here, once, by one constant rotation of the world — a change of
            # gauge that leaves every relative orientation exactly as it was (``_bonfiglio_session.py``).
            markers = {name: column @ session.world_from_deposit.T
                       for name, column in _read_optical_markers(optical_path).items()}
            n_frames = len(next(iter(markers.values()))) if markers else 0
            # The deposit's optical time base is kept unless APPLY_REFERENCE_LAG: the lag is the error the
            # synchronisation-offset check measures.
            t_ref = np.arange(n_frames) / RATE_HZ + (lag if APPLY_REFERENCE_LAG and lag is not None else 0.0)

        landmarks_local = _landmarks_local(str(subject_dir))
        landmarks_world = _reconstruct_landmarks(markers, landmarks_local)
        unusable: dict[str, str] = {}

        segments: dict[str, Segment] = {}
        for segment, (_, labels) in SEGMENTS.items():
            reference = None
            fit = session.alignment.get(segment, {})
            cluster_from_sensor = session.cluster_from_sensor.get(segment)
            if APPLY_REFERENCE_LAG and lag is None and optical_path is not None:
                unusable[segment] = (session.lag_diagnostics.get(k, {}).get("reason")
                                     or "the trial's clock offset against the reference was not recovered")
            elif optical_path is None:
                unusable[segment] = (session.lag_diagnostics.get(k, {}).get("reason")
                                     or "no optical recording of this session matches this trial")
            elif not fit and all(label in markers for label in labels):
                # No cluster-to-sensor rotation could be fitted at all (subject 9, whose sensors share no clock):
                # the cluster's own orientation is served as deposited, since data that cannot be corrected are
                # used as they are; the amount the segment turns is exact, its axes are the cluster's. Where a fit
                # was made and found no constant rotation (the scapula of every subject: the cluster moves 15-42°
                # against the IMU between trials), the segment has no reference.
                cluster = np.stack([markers[label] for label in labels], axis=1)
                rotation, position = _repair.cluster_pose(cluster, session.templates[segment])
                reference = Pose(
                    t=t_ref, R=rotation, p=position, rate_hz=RATE_HZ, kind="housing",
                    markers=cluster, marker_names=list(labels),
                    meta={
                        "construction": "Procrustes fit of the housing cluster to one marker template of the "
                                        "session; NOT rotated into the sensor frame (cluster-to-sensor rotation "
                                        "not identified)",
                        "R_is": "the housing cluster's template frame in the optical world frame",
                        "occluded_fraction": float(np.isnan(cluster).any(axis=(1, 2)).mean()),
                        "on_housing": True,
                        "clock_offset_applied_s": 0.0,
                        "cluster_to_sensor_aligned": False,
                        "cluster_to_sensor_reason": "no cluster-to-sensor rotation could be fitted for this session",
                    },
                )
            elif cluster_from_sensor is None or not fit.get("aligned"):
                unusable[segment] = fit.get("reason", "the cluster-to-sensor rotation was not identified")
            elif not all(label in markers for label in labels):
                unusable[segment] = "the housing cluster's markers are not in the optical file"
            else:
                cluster = np.stack([markers[label] for label in labels], axis=1)
                rotation, position = _repair.cluster_pose(cluster, session.templates[segment])
                reference = Pose(
                    t=t_ref,
                    R=rotation @ cluster_from_sensor,
                    p=position,
                    rate_hz=RATE_HZ,
                    kind="housing",
                    markers=cluster,
                    marker_names=list(labels),
                    meta={
                        "construction": "Procrustes fit of the housing cluster to one marker template of the "
                                        "session, then rotated into the sensor's own frame by the measured "
                                        "constant cluster-to-sensor rotation of that sensor and session",
                        "R_is": "the IMU's own sensor frame in the optical system's world frame",
                        "p_is": "the centroid of the three housing markers, not the sensor's origin — the "
                                "lever arm between them is not deposited and is not measured here",
                        "occluded_fraction": float(np.isnan(cluster).any(axis=(1, 2)).mean()),
                        "reference_accuracy_limit_deg": _reference_accuracy_limit(segment, fit),
                        "reference_accuracy_limit_is": (
                            "the cluster orientation noise of this segment and the uncertainty of the constant "
                            "cluster-to-sensor rotation this reference is rotated by, added in quadrature; the "
                            "second term dominates"),
                        "cluster_orientation_noise_deg": REFERENCE_ACCURACY_LIMIT_DEG[segment],
                        "cluster_to_sensor_uncertainty_deg": fit.get("between_trial_scatter_deg_median"),
                        "on_housing": True,
                        "clock_offset_applied_s": lag if APPLY_REFERENCE_LAG else 0.0,
                        "cluster_to_sensor_aligned": True,
                        "cluster_to_sensor_fit": fit,
                        "world_frame": "the deposit's optical world frame rotated onto the adapters' z-up "
                                       "convention by the session's measured vertical",
                    },
                )
            segments[segment] = Segment(
                name=segment,
                sensor=streams.get(segment),
                reference=reference,
                anatomical=_anatomical_pose(segment, landmarks_world, t_ref),
                mount="elastic Kinesio tape, sensor directly on the skin",
            )

        pairs = [
            Pair(
                parent=parent,
                child=child,
                joint=joint,
                joint_dof=dof,
                joint_centre_world=None,  # no optical joint centre ships with this deposit
                joint_centre_t=None,
                meta={"no_joint_centre_exists": joint == "none"},
            )
            for parent, child, joint, dof in PAIRS
        ]

        imu_duration = min(float(s.t[-1] - s.t[0]) for s in streams.values()) if streams else float("nan")
        optical_duration = float(len(t_ref) / RATE_HZ) if len(t_ref) else float("nan")
        name = TRIAL_NAMES[k] if k < len(TRIAL_NAMES) else f"unnamed-{k + 1}"
        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],  # one sensor per segment
            contacts_t=None,  # seated and standing upper-limb tasks; no gait and no force plates
            rest_annotation_s=None,  # no rest duration is stated anywhere in the deposit
            trial_type=("static" if k == 0 else "functional" if k < 10 else "task"),
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "Zenodo",
                "protocol_step": name,
                "protocol_index": k + 1,
                "worn_side": "left" if subject == "2" else "right",
                "labels_mismatch_worn_side": subject == "2",
                "sample_time_fine_start_us": {seg: int(v["sample_time_fine"][0])
                                              for seg, v in streams_raw.items()},
                "start_skew_s": skew,
                "streams_aligned": False,
                "common_sensor_timeline": max(skew.values()) - min(skew.values()) < 1.0,
                "optical_file": optical_path.name if optical_path else None,
                "optical_pairing": ("by index, as deposited (k-th inertial trial with k-th optical file)"
                                    if self.pairing == "as_deposited" else
                                    "by the measured peak correlation of the two systems' angular rate, in one "
                                    "monotone assignment over the whole session (_bonfiglio_session.py)"),
                "optical_file_index": optical_index,
                "optical_pairing_moved_from_index": optical_index != k if optical_index is not None else None,
                "optical_pairing_peak": session.pairing_peak.get(k),
                "optical_pairing_moved_in_session": dict(sorted(session.pairing_moved.items())),
                "duration_mismatch_s": optical_duration - imu_duration,
                "reference_unsynchronised": lag is None,
                "reference_lag_s": lag,
                "reference_lag_is": "the seconds added to the optical time base to put it on the sensors' "
                                    "common SampleTimeFine clock; applied to the Poses only if "
                                    "APPLY_REFERENCE_LAG (False by default)",
                "reference_lag_applied": APPLY_REFERENCE_LAG,
                "reference_lag_diagnostics": session.lag_diagnostics.get(k, {}),
                "reference_lag_review_median_s": REFERENCE_LAG_REVIEW_MEDIAN_S,
                "reference_unusable_segments": unusable,
                "cluster_to_sensor_fits": session.alignment,
                "laboratory_vertical": session.vertical,
                "world_from_deposit_R": session.world_from_deposit.tolist(),
                "reference_accuracy_limit_deg": {
                    name: _reference_accuracy_limit(name, session.alignment.get(name, {}))
                    for name in REFERENCE_ACCURACY_LIMIT_DEG},
                "cluster_orientation_noise_deg": REFERENCE_ACCURACY_LIMIT_DEG,
                "anatomical_landmarks": landmarks_world,
                "anatomical_landmarks_t": t_ref if len(t_ref) else None,
                "derived_products_read": False,
            },
        )


def _trial_id(k: int) -> str:
    name = TRIAL_NAMES[k] if k < len(TRIAL_NAMES) else f"unnamed-{k + 1}"
    return f"t{k + 1:02d}-{name}"


def _reconstruct_landmarks(
    markers: dict[str, np.ndarray], local: dict[str, tuple[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Digitised landmarks put back on every frame of a trial from their cluster's markers, metres."""
    out: dict[str, np.ndarray] = {}
    for landmark, (segment, offset) in local.items():
        labels = SEGMENTS[segment][1]
        if any(label not in markers for label in labels):
            continue
        s1, s2, s3 = (markers[label] for label in labels)
        rotation, origin = _local_frame(s1, s2, s3)
        out[landmark] = origin + np.einsum("tij,j->ti", rotation, offset)
    return out


def _anatomical_pose(segment: str, landmarks: dict[str, np.ndarray], t_ref: np.ndarray) -> Pose | None:
    """The ISB frame a segment's own digitised landmarks define, or ``None`` (see the module docstring)."""
    builder = {"trunk": _isb_thorax, "scapula": _isb_scapula}.get(segment)
    if builder is None or not len(t_ref):
        return None
    built = builder(landmarks)
    if built is None:
        return None
    rotation, position = built
    used = sorted(name for name, (seg) in LANDMARK_SEGMENT.items() if seg == segment and name in landmarks)
    return Pose(
        t=t_ref,
        R=rotation,
        p=position,
        rate_hz=RATE_HZ,
        kind="anatomical",
        markers=np.stack([landmarks[name] for name in used], axis=1) if used else None,
        marker_names=used,
        meta={
            "construction": f"ISB {'thorax' if segment == 'trunk' else 'scapula'} frame from digitised landmarks",
            "landmarks_reconstructed_from": "the segment's own housing cluster",
        },
    )


ADAPTER = BonfiglioAdapter()
EXTRA_ADAPTERS = [BonfiglioAdapter(pairing="as_deposited")]  # see __init__
