"""Darmstadt stair ambulation (Grimmer et al., TU Darmstadt): sixteen Delsys Trigno sensors plus Qualisys, skin
markers. Data DOI 10.48328/tudatalib-1182, CC BY 4.0. 718 trials over twelve subjects.

Files read, straight out of the deposited zip archives (MATLAB level-5, ``scipy.io.loadmat``):
- ``Preprocessed.zip``: ``Preprocessed/IMU/IMU{s}.mat`` (the magnetometer is dropped) and
  ``Preprocessed/Marker/Marker{s}.mat`` (28 marker trajectories);
- ``Processed.zip``: ``Processed/Touchdowns/Touchdowns{s}.mat`` (heel-strike frame indices, used for
  ``contacts_t`` and as the trial index).
Each variable holds six stair configurations (low/medium/high a, low/medium/high b) of ten (rarely nine) trials.

Units (``*_unit`` and ``*_fs`` read back and asserted): accelerometer g -> m/s^2 (x 9.80665); gyroscope deg/s ->
rad/s; markers already m; touchdowns frame index -> s as (index - 1) / 200 (1-based indices assumed). Time vectors
are ``k / fs`` for both streams from zero (the deposit ships no time stamps). The 148.148 Hz inertial stream is raw
(values on the converter lattice) and is not resampled or filtered.

Reference (``kind="skin"``): Qualisys at 180 Hz, released resampled to 200 Hz; markers on anatomical landmarks.
Right-handed frames ``R = [e1 e2 e3]`` with e2 = unit(long axis), e1 = unit(ml x e2), e3 = e1 x e2:
    thigh  long Trc - KJC, ml Kne_lat - Kne_med   (KJC = knee marker midpoint)
    shank  long KJC - AJC, ml Ank_lat - Ank_med   (AJC = ankle marker midpoint)
    foot   long AJC - MTJC, ml Mt5 - Mt1          (MTJC = (Mt1 + Mt5) / 2)
``p`` is the marker centroid. Left and right use the same formula, so their axes are mirrored. Missing frames are
NaN, never interpolated (``Pose.meta["complete_fraction"]``).

Sensors: the vastus lateralis (thigh) and the "Shank" site (shank) are ``Segment.sensor``; the others are
``Segment.extra_sensors``, all same-segment pairs in ``Recording.same_segment_pairs``.

Deposit defects: a trailing block of exactly-zero inertial samples (median 9.9 s) in every trial of subjects 4-12
is container padding and is trimmed (``Recording.meta["zero_tail_trimmed_s"]``); the inertial stream lags the
optical one by 22.5-35 ms (not aligned); optical dropout of 2.4-2.7 % on the knee markers; recordings start
mid-stride (``rest_annotation_s`` is None); mounting undocumented; the longest trial is 34.6 s; a stride
exclusion list exists only for another release and is not applied.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

DEG_TO_RAD = np.pi / 180.0
"""Units: rad per degree.
Provenance: physical — an exact unit conversion (physical)."""
G_TO_M_PER_S2 = 9.80665
"""Units: m/s² per g.
Provenance: physical — the defined value of standard gravity (physical: the standard gravity the deposit's "g" unit refers to)."""

EXPECTED_IMU_RATE_HZ = 1000.0 / 6.75
"""Units: Hz.
Provenance: read-from-dataset — the unit's stated update rate (148.148...; data-free, asserted against the file's own Accelerometer_fs)."""
EXPECTED_MARKER_RATE_HZ = 200.0
"""Units: Hz.
Provenance: read-from-dataset — the optical rate the deposit states (data-free, asserted against the file's own Marker_fs)."""
NATIVE_MARKER_RATE_HZ = 180.0
"""Units: Hz.
Provenance: read-from-dataset — the optical system's native rate before the deposit's resampling (documented Qualisys rate before the deposit resampled to 200 Hz)."""
TOUCHDOWN_RATE_HZ = 200.0
"""Units: Hz.
Provenance: read-from-dataset — the rate the deposited event channel is written at (the deposit's TD_fs)."""

STAIR_SLOPE_DEG = {"low": 19.0, "medium": 30.4, "high": 39.6}
"""Slope of each of the deposit's three stair configurations.

Units: degrees.
Provenance: read-from-dataset — stated by the deposit for its three stair heights."""

RANGE_GYR_DPS = 1998.0
"""The unit's gyroscope full-scale range.

Units: °/s.
Provenance: read-from-dataset — recovered from the deposit's own quantisation step (0.060976 °/s over sixteen bits)."""

RANGE_ACC_G = 16.0
"""The unit's accelerometer full-scale range.

Units: g.
Provenance: read-from-dataset — recovered from the deposit's own quantisation step (16/32768 g over sixteen bits)."""

N_SUBJECTS = 12
"""Units: subjects.
Provenance: read-from-dataset — how many subjects the deposit contains."""
CONFIGURATIONS = (
    ("low", "a", 0.10),
    ("medium", "a", 0.17),
    ("high", "a", 0.24),
    ("low", "b", 0.10),
    ("medium", "b", 0.17),
    ("high", "b", 0.24),
)
"""The six cells in the deposit's documented top-to-bottom order, with the stair height in metres.
Slopes are 19.0 deg (low), 30.4 deg (medium), 39.6 deg (high); stair depth is fixed at 0.29 m.

Units: m.
Provenance: read-from-dataset — the deposit's own stair configurations and their heights."""

_SEGMENTS: dict[str, dict] = {
    "thigh_r": {"sensor": "vas_r", "extra": ["rcf_r", "bcf_r"], "markers": ["Trc_r", "Kne_lat_r", "Kne_med_r"]},
    "shank_r": {
        "sensor": "sha_r",
        "extra": ["tib_r", "gas_r", "sol_r"],
        "markers": ["Kne_lat_r", "Kne_med_r", "Ank_lat_r", "Ank_med_r"],
    },
    "foot_r": {"sensor": "foo_r", "extra": [], "markers": ["Ank_lat_r", "Ank_med_r", "Mt1_r", "Mt5_r"]},
    "thigh_l": {"sensor": "vas_l", "extra": ["rcf_l", "bcf_l"], "markers": ["Trc_l", "Kne_lat_l", "Kne_med_l"]},
    "shank_l": {
        "sensor": "sha_l",
        "extra": ["tib_l", "gas_l", "sol_l"],
        "markers": ["Kne_lat_l", "Kne_med_l", "Ank_lat_l", "Ank_med_l"],
    },
    "foot_l": {"sensor": "foo_l", "extra": [], "markers": ["Ank_lat_l", "Ank_med_l", "Mt1_l", "Mt5_l"]},
}
"""Segment → its sensors (primary first) and the markers its reference frame is built from."""

_SENSOR_SITE = {
    "rcf": "rectus femoris",
    "bcf": "biceps femoris",
    "vas": "vastus lateralis",
    "tib": "tibialis anterior",
    "gas": "gastrocnemius",
    "sol": "soleus",
    "sha": "shank (not on a muscle)",
    "foo": "os cuneiforme II, midfoot (not on a muscle)",
}

_LICENCE = "CC BY 4.0 (license_CC-BY-4.0.rdf shipped with the TUdatalib item)"
_SOURCE = "SOURCE.md in the deposit folder; data DOI 10.48328/tudatalib-1182 (TUdatalib, TU Darmstadt)"


def _frame(long_axis: np.ndarray, ml_axis: np.ndarray) -> np.ndarray:
    """Right-handed segment frame from a proximal-distal and a medio-lateral direction; columns are the axes."""
    e2 = long_axis / np.linalg.norm(long_axis, axis=1, keepdims=True)
    e1 = np.cross(ml_axis, e2)
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=2)


def _unit_pair(value) -> tuple[str, str]:
    arr = np.atleast_1d(np.asarray(value).ravel())
    return str(arr[0]), str(arr[-1])


class GrimmerStairsAdapter(Adapter):
    """One ``Recording`` per deposited trial: 718 stair-ambulation trials of twelve subjects."""

    name = "grimmer-stairs"
    reference_kind = "skin"
    sensor_family = "Delsys Trigno"
    licence = _LICENCE
    source = _SOURCE

    def __init__(self, root=None) -> None:
        self.root = dataset_dir("grimmer_stairs") if root is None else root
        self._index: list[tuple[str, str]] | None = None

    # -- discovery ---------------------------------------------------------------------------------------
    def trials(self) -> list[tuple[str, str]]:
        """``(subject, trial)`` for all 718 trials, read from the 3 kB ``Touchdowns`` files, no signal data.

        The trial *count* per configuration varies (nine instead of ten in two cases), so it has to be read
        from a file; the touchdown files are the cheapest place it exists — 3 kB each against 82 MB for the
        inertial file and 220 MB for the marker file. The index is cached in memory for the lifetime of the
        adapter; nothing is written to disk.
        """
        if self._index is not None:
            return self._index
        import scipy.io as sio

        index: list[tuple[str, str]] = []
        with zipfile.ZipFile(self.root / "Processed.zip") as z:
            for subject_number in range(1, N_SUBJECTS + 1):
                with z.open(f"Processed/Touchdowns/Touchdowns{subject_number}.mat") as fh:
                    mat = sio.loadmat(io.BytesIO(fh.read()), squeeze_me=True, struct_as_record=False)
                cells = mat["TD_ascent"]
                for cfg, (height, setup, _) in enumerate(CONFIGURATIONS):
                    for rep in range(int(np.shape(cells[cfg])[0])):
                        index.append((f"S{subject_number:02d}", f"{height}_{setup}_rep{rep + 1:02d}"))
        self._index = index
        return index

    def _parse_trial_id(self, subject: str, trial: str) -> tuple[int, int, int]:
        subject_number = int(subject.lstrip("S"))
        try:
            height, setup, rep = trial.split("_")
            cfg = next(i for i, (h, s, _) in enumerate(CONFIGURATIONS) if h == height and s == setup)
            rep_index = int(rep.replace("rep", "")) - 1
        except (ValueError, StopIteration) as exc:
            raise ValueError(f"unreadable trial id {trial!r}; expected e.g. 'medium_a_rep03'") from exc
        return subject_number, cfg, rep_index

    # -- loading -----------------------------------------------------------------------------------------
    def load(self, subject: str, trial: str) -> Recording:
        import scipy.io as sio

        subject_number, cfg, rep = self._parse_trial_id(subject, trial)

        def read(archive: str, member: str, names: list[str]) -> dict:
            with zipfile.ZipFile(self.root / archive) as z, z.open(member) as fh:
                return sio.loadmat(io.BytesIO(fh.read()), squeeze_me=True, struct_as_record=False, variable_names=names)

        imu = read(
            "Preprocessed.zip",
            f"Preprocessed/IMU/IMU{subject_number}.mat",
            ["Accelerometer", "Gyroscope", "Accelerometer_fs", "Gyroscope_fs", "Accelerometer_unit", "Gyroscope_unit"],
        )
        acc_fs, gyr_fs = float(imu["Accelerometer_fs"]), float(imu["Gyroscope_fs"])
        acc_unit, gyr_unit = _unit_pair(imu["Accelerometer_unit"])[1], _unit_pair(imu["Gyroscope_unit"])[1]
        if acc_unit != "g" or not gyr_unit.endswith("/s"):
            raise ValueError(f"{subject}/{trial}: deposited inertial units changed: {acc_unit!r}, {gyr_unit!r}")
        if abs(acc_fs - EXPECTED_IMU_RATE_HZ) > 1e-6 or abs(gyr_fs - EXPECTED_IMU_RATE_HZ) > 1e-6:
            raise ValueError(f"{subject}/{trial}: inertial rate changed: {acc_fs}, {gyr_fs}")
        acc_struct, gyr_struct = imu["Accelerometer"][cfg][rep], imu["Gyroscope"][cfg][rep]
        n_released = int(np.shape(acc_struct.sha_r)[0])
        n_keep = self._trim_length(acc_struct, gyr_struct, n_released)
        streams = {
            field: self._sensor_stream(acc_struct, gyr_struct, field, acc_fs, n_keep, n_released)
            for field in acc_struct._fieldnames
        }
        del imu, acc_struct, gyr_struct

        marker = read(
            "Preprocessed.zip",
            f"Preprocessed/Marker/Marker{subject_number}.mat",
            ["Marker", "Marker_fs", "Marker_unit"],
        )
        marker_fs = float(marker["Marker_fs"])
        if abs(marker_fs - EXPECTED_MARKER_RATE_HZ) > 1e-6:
            raise ValueError(f"{subject}/{trial}: marker rate changed: {marker_fs}")
        if _unit_pair(marker["Marker_unit"])[1] != "m":
            raise ValueError(f"{subject}/{trial}: deposited marker unit changed: {_unit_pair(marker['Marker_unit'])}")
        marker_struct = marker["Marker"][cfg][rep]
        n_marker = int(np.shape(marker_struct.Trc_r)[0])
        t_marker = np.arange(n_marker) / marker_fs
        poses = {
            seg: self._pose(marker_struct, spec["markers"], seg, t_marker, marker_fs) for seg, spec in _SEGMENTS.items()
        }
        del marker, marker_struct

        contacts, contact_detail = self._contacts(subject_number, cfg, rep)

        segments: dict[str, Segment] = {}
        same_segment_pairs: list[tuple[str, int, int]] = []
        for seg_name, spec in _SEGMENTS.items():
            names = [spec["sensor"], *spec["extra"]]
            seg_streams = [streams[name] for name in names]
            segments[seg_name] = Segment(
                name=seg_name,
                sensor=seg_streams[0],
                reference=poses[seg_name],
                anatomical=None,  # the skin-landmark frame IS the reference here; no second frame exists
                extra_sensors=seg_streams[1:],
                mount=None,  # the documentation says only "EMG (IMU) sensors were fixed at the subjects"
            )
            for a in range(len(seg_streams)):
                for b in range(a + 1, len(seg_streams)):
                    same_segment_pairs.append((seg_name, a, b))

        pairs = [
            Pair(parent="thigh_r", child="shank_r", joint="hinge", joint_dof=1, meta={"side": "right", "name": "knee"}),
            Pair(
                parent="shank_r",
                child="foot_r",
                joint="2-dof",
                joint_dof=2,
                meta={"side": "right", "name": "ankle (talocrural + subtalar)"},
            ),
            Pair(parent="thigh_l", child="shank_l", joint="hinge", joint_dof=1, meta={"side": "left", "name": "knee"}),
            Pair(
                parent="shank_l",
                child="foot_l",
                joint="2-dof",
                joint_dof=2,
                meta={"side": "left", "name": "ankle (talocrural + subtalar)"},
            ),
        ]

        height, setup, height_m = CONFIGURATIONS[cfg]
        n_trimmed = n_released - n_keep
        meta = {
            "licence": _LICENCE,
            "source": _SOURCE,
            "host": "TUdatalib (TU Darmstadt)",
            "dataset_folder": str(self.root),
            "stair_height_label": height,
            "stair_height_m": height_m,
            "stair_slope_deg": STAIR_SLOPE_DEG[height],
            "force_plate_setup": setup,
            "configuration_cell": cfg,
            "repetition": rep + 1,
            "imu_samples_as_released": n_released,
            "zero_tail_trimmed_samples": int(n_trimmed),
            "zero_tail_trimmed_s": float(n_trimmed / acc_fs),
            "zero_tail_note": (
                "subjects 4-12 carry a trailing block of exactly-zero inertial samples (median 9.92 s) in all "
                "538 of their trials; it is container padding, not measurement, and is trimmed here and "
                "reported rather than hidden. Subjects 1-3 have none."
            ),
            "inertial_lag_measured": (
                "inertial stream lags the optical stream by 22.5-35.0 ms (+3 to +9 samples at 200 Hz), the "
                "same to within one sample on eight sensors across three segments"
                "; NOT corrected here"
            ),
            "inertial_raw": (
                "148.148 Hz as recorded: 100.000 % of accelerometer values lie on the 16/32768 g converter "
                "lattice and the gyroscope on a 0.060976 deg/s lattice; no resampling or filtering applied"
            ),
            "magnetometer_present_but_dropped": (
                "the deposit ships a 74.074 Hz magnetometer; this paper is magnetometer-free"
            ),
            "optical_resampled": "Qualisys native 180 Hz, released resampled to 200 Hz (band-limited above ~60 Hz)",
            "reference_accuracy_limit_deg": (
                "3.0 rms on knee flexion between two constructions of the same optical reference"
            ),
            "starts_in_motion": "0 of 718 trials start below 20 deg/s; median 142 deg/s over the first second",
            "still_stretch_at_end": (
                "the only quiet stretch is terminal, median 6.57 s, ending within 1 s of the last sample "
                "in 93.3 % of trials"
            ),
            "sensor_to_segment_alignment": (
                "unknown; no marker on any housing, no calibration pose, sensor axes differ left/right"
            ),
            "sensor_choice": (
                "vastus lateralis (thigh) and the non-muscle shank site (shank) as Segment.sensor; "
                "a design choice with options"
            ),
            "window_ceiling_s": "no 60 s or 120 s window exists in the deposit; the longest trial is 34.62 s",
            "stride_exclusion_list_unavailable": (
                "the 95 excluded strides are marked only in the FullyProcessed release and do not map back "
                "to these frames"
            ),
            "contacts_note": (
                "heel-strike frame indices of the instrumented strides only (TD_ascent and TD_descent, both "
                "legs), converted as (index-1)/200 s assuming the MATLAB 1-based convention [assumed]"
            ),
            "contacts_detail": contact_detail,
        }

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=same_segment_pairs,
            contacts_t=contacts,
            rest_annotation_s=None,  # no initial rest is documented and none exists; the still stretch is terminal
            trial_type="stairs",
            meta=meta,
        )

    # -- pieces ------------------------------------------------------------------------------------------
    @staticmethod
    def _trim_length(acc_struct, gyr_struct, n_released: int) -> int:
        """Length after dropping the trailing block in which every sensor reads exactly zero (defect D1)."""
        nonzero = np.zeros(n_released, dtype=bool)
        for field in acc_struct._fieldnames:
            a = np.asarray(getattr(acc_struct, field), dtype=float)
            g = np.asarray(getattr(gyr_struct, field), dtype=float)
            nonzero |= (np.abs(a).sum(axis=1) != 0) | (np.abs(g).sum(axis=1) != 0)
        idx = np.flatnonzero(nonzero)
        return int(idx[-1]) + 1 if idx.size else n_released

    @staticmethod
    def _sensor_stream(acc_struct, gyr_struct, field: str, fs: float, n_keep: int, n_released: int) -> SensorStream:
        acc = np.asarray(getattr(acc_struct, field), dtype=float)[:n_keep] * G_TO_M_PER_S2
        gyr = np.asarray(getattr(gyr_struct, field), dtype=float)[:n_keep] * DEG_TO_RAD
        side = "right" if field.endswith("_r") else "left"
        return SensorStream(
            t=np.arange(n_keep) / fs,
            gyr=gyr,
            acc=acc,
            rate_hz=fs,
            family="Delsys Trigno",
            range_gyr=RANGE_GYR_DPS * DEG_TO_RAD,
            range_acc=RANGE_ACC_G * G_TO_M_PER_S2,
            native_rate_hz=None,  # rate_hz IS the hardware rate here; nothing was resampled
            meta={
                "field": field,
                "site": _SENSOR_SITE[field[:3]],
                "side": side,
                "deposited_units": {"acc": "g", "gyr": "deg/s"},
                "time_base": (
                    "synthesised as k/fs from the file's own Accelerometer_fs; the deposit ships no timestamps"
                ),
                "raw": True,
                "samples_as_released": n_released,
                "zero_tail_trimmed_samples": int(n_released - n_keep),
                "lag_behind_optical_ms": "22.5-35.0, not corrected",
                "axes": "Trigno user-manual convention, not the deposit's global convention; differs left/right",
            },
        )

    @staticmethod
    def _pose(marker_struct, marker_names: list[str], segment: str, t: np.ndarray, fs: float) -> Pose:
        tracks = [np.asarray(getattr(marker_struct, name), dtype=float) for name in marker_names]
        markers = np.stack(tracks, axis=1)  # (T, K, 3), already in metres
        if segment.startswith("thigh"):
            knee_centre = (tracks[1] + tracks[2]) / 2.0
            R = _frame(tracks[0] - knee_centre, tracks[1] - tracks[2])
        elif segment.startswith("shank"):
            knee_centre = (tracks[0] + tracks[1]) / 2.0
            ankle_centre = (tracks[2] + tracks[3]) / 2.0
            R = _frame(knee_centre - ankle_centre, tracks[2] - tracks[3])
        else:
            ankle_centre = (tracks[0] + tracks[1]) / 2.0
            toe_centre = (tracks[2] + tracks[3]) / 2.0
            R = _frame(ankle_centre - toe_centre, tracks[3] - tracks[2])
        p = markers.mean(axis=1)
        complete = np.isfinite(markers).all(axis=(1, 2))
        return Pose(
            t=t,
            R=R,
            p=p,
            rate_hz=fs,
            kind="skin",
            markers=markers,
            marker_names=list(marker_names),
            meta={
                "frame": (
                    "e2 = unit(proximal-distal); e1 = unit(ml x e2); e3 = e1 x e2; "
                    "columns of R, origin = marker centroid"
                ),
                "frame_note": "corrects the improper e3 = e2 x e1 of so that det R = +1",
                "marker_sites": "anatomical skin landmarks; no marker on any sensor housing",
                "deposited_unit": "m",
                "native_rate_hz": NATIVE_MARKER_RATE_HZ,
                "resampled": "Qualisys 180 Hz resampled to 200 Hz by the deposit",
                "complete_fraction": float(complete.mean()),
                "longest_complete_run_s": float(_longest_run(complete) / fs),
                "gaps_are_nan": True,
                "sides_mirrored": "left and right use the same formula, so their axis semantics are mirrored",
            },
        )

    def _contacts(self, subject_number: int, cfg: int, rep: int) -> tuple[np.ndarray, dict]:
        import scipy.io as sio

        with (
            zipfile.ZipFile(self.root / "Processed.zip") as z,
            z.open(f"Processed/Touchdowns/Touchdowns{subject_number}.mat") as fh,
        ):
            mat = sio.loadmat(io.BytesIO(fh.read()), squeeze_me=True, struct_as_record=False)
        fs = float(mat.get("TD_fs", TOUCHDOWN_RATE_HZ))
        detail: dict[str, list[float]] = {}
        all_times: list[float] = []
        for phase in ("TD_ascent", "TD_descent"):
            struct = mat[phase][cfg][rep]
            for leg_field, leg in (("tdR", "right"), ("tdL", "left")):
                frames = np.atleast_1d(np.asarray(getattr(struct, leg_field), dtype=float)).ravel()
                frames = frames[np.isfinite(frames)]
                times = (frames - 1.0) / fs
                detail[f"{phase.replace('TD_', '')}_{leg}"] = times.tolist()
                all_times.extend(times.tolist())
        return np.sort(np.asarray(all_times, dtype=float)), detail


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in ``mask`` (samples)."""
    if not mask.any():
        return 0
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return int((ends - starts).max()) + 1


ADAPTER = GrimmerStairsAdapter()
