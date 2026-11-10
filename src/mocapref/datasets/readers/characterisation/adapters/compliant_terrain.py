"""Locomotion over compliant terrains (University of Delaware, Variable Stiffness Treadmill 2): Delsys IMUs on thigh
and shank muscles, Vicon skin markers. Data DOI 10.6084/m9.figshare.27180288, CC BY 4.0. Only the two downloaded
participants (SUB01, SUB02) of twenty are visible.

File read: ``SUB{nn}.mat`` (MATLAB v7.3 = HDF5, via h5py). One trial is one treadmill speed (0.6, 0.8, 1.0 m/s) and
holds four ground-stiffness conditions, each a cell array with one cell per gait cycle. Every gap between cells is
exactly one sample, so the cells are concatenated in protocol order (stiff1000, stiff25, stiff80, stiff40) into one
14-19 minute recording; the one-sample continuity is checked on every load and the cycle and condition boundaries
are kept in ``Recording.meta``.

Units (read back from each group's ``unit`` field and asserted): ``imuAcc`` mm/s^2 -> m/s^2 (specific force);
``imuGyro`` deg/s -> rad/s; markers mm -> m; times in s.

Reference (``kind="skin"``): Plug-in-Gait markers on skin and tight clothing, never on a housing, 100 Hz. One
right-handed frame per segment from three markers (thigh ``{R,L}THI, {R,L}THIA, {R,L}KNE``; shank ``{R,L}TIB,
{R,L}TIBA, {R,L}ANK``): e1 = unit(m2 - m1), e3 = unit(e1 x (m3 - m1)), e2 = e3 x e1, ``p`` the marker centroid.
The raw markers travel with the Pose.

Sensors: the vastus lateralis (thigh) and tibialis anterior (shank) sensors are ``Segment.sensor``; the others are
``Segment.extra_sensors``, and every same-segment pair is in ``Recording.same_segment_pairs``. Knees only: no foot,
pelvis or trunk sensor.

Deposit defects left in the data: the 2000 Hz inertial stream carries about 148 Hz of information (served at
2000 Hz with ``native_rate_hz=148.148`` and ``SensorStream.meta["interpolated"]``); the inertial stream lags the
optical one by 18-34 ms on the shank and 0.5-27 ms on the thigh (not aligned); no inertial time stamps ship, so the
inertial time is ``t_optical[0] + k / 2000`` and the per-cycle deviation from 20 x the optical length is reported
(``meta["inertial_cycle_offset_samples"]``); the right gait events duplicate the left ones (only the left are
used, ``contacts_t`` = left heel strikes); the optical stream was gap-filled without a mask; one ``RTHI`` marker
glitch (``Pose.meta["marker_speed_glitch_frames"]``); no still stretch anywhere.

Memory: one trial holds ten inertial streams of about 2e6 samples (about 1 GB in float64); ``load`` builds one
trial and caches nothing.
"""

from __future__ import annotations

import numpy as np

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

DEG_TO_RAD = np.pi / 180.0
"""Units: rad per degree.
Provenance: physical — an exact unit conversion (physical)."""
MM_TO_M = 1e-3
"""Units: m per mm.
Provenance: physical — an exact unit conversion (physical)."""
MM_PER_S2_TO_M_PER_S2 = 1e-3
"""Units: (m/s²) per (mm/s²).
Provenance: physical — an exact unit conversion (physical)."""

NOMINAL_RATE_HZ = 2000.0
"""Units: Hz.
Provenance: read-from-dataset — the rate the deposit states (data-free: the rate the deposit labels the inertial stream with)."""
NATIVE_RATE_HZ = 148.148
"""Units: Hz.
Provenance: read-from-dataset — read from the deposit."""
OPTICAL_RATE_HZ = 100.0
"""Units: Hz.
Provenance: read-from-dataset — the optical system's rate as the deposit states it (data-free: the deposit's own time.time step, re-asserted on every load)."""
SAMPLES_PER_OPTICAL = 20
"""Units: samples.
Provenance: data-free — the nominal ratio of the two stated rates; the per-cycle deviation is reported, never enforced (data-free: the nominal ratio 2000/100; the per-cycle deviation is reported, not enforced)."""

CONDITION_ORDER = ("stiff1000", "stiff25", "stiff80", "stiff40")
"""The deposit's protocol order; ``timeseriesConvert.m`` concatenates in the same order."""

STIFFNESS_KN_PER_M = {"stiff1000": "rigid", "stiff25": 27.2, "stiff80": 89.4, "stiff40": 41.3}
"""Nominal condition name → the stiffness the authors measured, kN/m (data paper); ``stiff1000`` is rigid.

Units: kN/m.
Provenance: read-from-dataset — the stiffness the depositors measured for each terrain condition."""

FORCE_MAT_WARMUP_CYCLES = 150
"""The deposit's own force-mat warm-up, documented for the force channels only; kept, never dropped.

Units: cycles.
Provenance: read-from-dataset — the deposit's own stated force-mat warm-up."""

MARKER_GLITCH_MM_PER_S = 3000.0
"""Detection threshold for the known ``RTHI`` marker jump (the 99.9th percentile of marker speed in the deposit is 824 mm/s).
Provenance: it only *flags* frames, it changes no returned value.

Units: mm/s."""

_SEGMENTS: dict[str, dict] = {
    "thigh_r": {"sensor": "RVL", "extra": ["RRF", "RBF"], "markers": ["RTHI", "RTHIA", "RKNE"]},
    "shank_r": {"sensor": "RTA", "extra": ["RGA"], "markers": ["RTIB", "RTIBA", "RANK"]},
    "thigh_l": {"sensor": "LVL", "extra": ["LRF", "LBF"], "markers": ["LTHI", "LTHIA", "LKNE"]},
    "shank_l": {"sensor": "LTA", "extra": ["LGA"], "markers": ["LTIB", "LTIBA", "LANK"]},
}
"""Segment → its sensors (primary first) and the three markers its reference frame is built from.

Sensor sites: ``VL`` vastus lateralis, ``RF`` rectus femoris, ``BF`` biceps femoris (thigh); ``TA`` tibialis
anterior, ``GA`` gastrocnemius (shank). The primary choice (vastus lateralis, tibialis anterior) is a
registered design choice, not a tuned constant."""

_SENSOR_SITE = {
    "RVL": "vastus lateralis",
    "RRF": "rectus femoris",
    "RBF": "biceps femoris",
    "RTA": "tibialis anterior",
    "RGA": "gastrocnemius",
    "LVL": "vastus lateralis",
    "LRF": "rectus femoris",
    "LBF": "biceps femoris",
    "LTA": "tibialis anterior",
    "LGA": "gastrocnemius",
}

_MOUNT = "double-sided adhesive tape on shaved skin, over-wrapped with medical tape (SENIAM placement)"

_LICENCE = "CC BY 4.0 (figshare licence id 52, https://creativecommons.org/licenses/by/4.0/)"
_SOURCE = "SOURCE.md in the deposit folder; data DOI 10.6084/m9.figshare.27180288 (figshare v1)"


def _unit_string(group) -> str:
    """Read a MATLAB char array stored as uint16 back into a string (the ``unit`` dataset of each group)."""
    raw = np.asarray(group["unit"]).ravel().astype(np.uint16)
    return "".join(chr(c) for c in raw)


def _concat_cells(f, group, name: str) -> np.ndarray:
    """Resolve a MATLAB cell array of object references into one ``(N, d)`` array."""
    cells = group[name]
    parts = [np.asarray(f[cells[0, i]]) for i in range(cells.shape[1])]
    return np.concatenate(parts, axis=1).T


def _cell_lengths(f, group, name: str) -> np.ndarray:
    cells = group[name]
    return np.array([f[cells[0, i]].shape[1] for i in range(cells.shape[1])], dtype=np.int64)


def _frame_from_three_markers(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    """Right-handed segment frame from three markers; columns of the result are the axes in world."""
    e1 = m2 - m1
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e3 = np.cross(e1, m3 - m1)
    e3 = e3 / np.linalg.norm(e3, axis=1, keepdims=True)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=2)


class CompliantTerrainAdapter(Adapter):
    """Whole-trial recordings of the compliant-terrain deposit, cycles concatenated, defects exposed."""

    name = "compliant-terrain"
    reference_kind = "skin"
    sensor_family = "Delsys Trigno"
    licence = _LICENCE
    source = _SOURCE

    def __init__(self, root=None) -> None:
        self.root = dataset_dir("compliant_terrain") if root is None else root
        self._index: list[tuple[str, str]] | None = None

    # -- discovery ---------------------------------------------------------------------------------------
    def trials(self) -> list[tuple[str, str]]:
        """``(participant, speed)`` for every downloaded participant file; opens the files but reads no data.

        The speed keys are listed from the HDF5 group table, which costs no signal read. The result is cached
        in memory for the lifetime of the adapter (never on disk, never a ``Recording``)."""
        if self._index is not None:
            return self._index
        import h5py

        index: list[tuple[str, str]] = []
        for path in sorted(self.root.glob("SUB*.mat")):
            subject = path.stem
            with h5py.File(path, "r") as f:
                if subject not in f:
                    continue
                speeds = sorted(k for k in f[subject] if k.startswith("speed"))
            index.extend((subject, speed) for speed in speeds)
        self._index = index
        return index

    # -- loading -----------------------------------------------------------------------------------------
    def load(self, subject: str, trial: str) -> Recording:
        import h5py

        path = self.root / f"{subject}.mat"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; only the downloaded participants are readable here")
        with h5py.File(path, "r") as f:
            return self._load_open(f, subject, trial)

    def _load_open(self, f, subject: str, trial: str) -> Recording:
        speed_grp = f[subject][trial]
        conditions = [c for c in CONDITION_ORDER if c in speed_grp]
        if not conditions:
            raise ValueError(f"{subject}/{trial}: none of {CONDITION_ORDER} present")

        t_opt, cycle_start, cycle_len, cycle_condition, condition_start, gap_dev = self._time_base(
            f, speed_grp, conditions
        )
        n_opt = t_opt.size
        imu_cell_len = self._inertial_cell_lengths(f, speed_grp, conditions)
        n_imu = int(imu_cell_len.sum())
        cycle_start_imu = np.concatenate(([0], np.cumsum(imu_cell_len)[:-1])).astype(np.int64)
        cycle_offset = imu_cell_len - SAMPLES_PER_OPTICAL * cycle_len
        t_imu = t_opt[0] + np.arange(n_imu) / NOMINAL_RATE_HZ

        first = speed_grp[conditions[0]]
        acc_unit, gyr_unit, mrk_unit = (
            _unit_string(first["imuAcc"]),
            _unit_string(first["imuGyro"]),
            _unit_string(first["markers"]),
        )
        if (acc_unit, gyr_unit, mrk_unit) != ("mm/sec^2", "deg/sec", "mm"):
            raise ValueError(f"{subject}/{trial}: deposited units changed: {acc_unit}, {gyr_unit}, {mrk_unit}")

        segments: dict[str, Segment] = {}
        same_segment_pairs: list[tuple[str, int, int]] = []
        for seg_name, spec in _SEGMENTS.items():
            sensor_names = [spec["sensor"], *spec["extra"]]
            streams = [self._sensor_stream(f, speed_grp, conditions, name, t_imu, n_imu) for name in sensor_names]
            pose = self._pose(f, speed_grp, conditions, spec["markers"], t_opt, n_opt)
            segments[seg_name] = Segment(
                name=seg_name,
                sensor=streams[0],
                reference=pose,
                anatomical=None,  # no housing cluster and no second landmark frame ships
                extra_sensors=streams[1:],
                mount=_MOUNT,
            )
            for a in range(len(streams)):
                for b in range(a + 1, len(streams)):
                    same_segment_pairs.append((seg_name, a, b))

        pairs = [
            Pair(
                parent="thigh_r",
                child="shank_r",
                joint="hinge",
                joint_dof=1,
                meta={"side": "right", "note": "released Plug-in-Gait knee is 3-column; secondary axes are crosstalk"},
            ),
            Pair(
                parent="thigh_l",
                child="shank_l",
                joint="hinge",
                joint_dof=1,
                meta={"side": "left", "note": "released Plug-in-Gait knee is 3-column; secondary axes are crosstalk"},
            ),
        ]

        events = speed_grp["gaitEvents"]
        left_heel = np.asarray(events["leftHeelStrikes"]).ravel().astype(float)
        left_toe = np.asarray(events["leftToeOffs"]).ravel().astype(float)
        right_heel = np.asarray(events["rightHeelStrikes"]).ravel().astype(float)
        right_duplicates = bool(right_heel.shape == left_heel.shape and np.array_equal(right_heel, left_heel))

        n_first_condition = int(np.sum(np.asarray(cycle_condition) == conditions[0]))
        warmup_cycles = FORCE_MAT_WARMUP_CYCLES if n_first_condition >= 2 * FORCE_MAT_WARMUP_CYCLES else 0

        meta = {
            "licence": _LICENCE,
            "source": _SOURCE,
            "host": "figshare (Springer Nature portal)",
            "dataset_folder": str(self.root),
            "treadmill_speed_m_per_s": float(trial.replace("speed", "").replace("p", ".")),
            "stiffness_conditions": {c: STIFFNESS_KN_PER_M[c] for c in conditions},
            "condition_order": conditions,
            "cycle_start_index_optical": cycle_start,
            "cycle_start_index_inertial": cycle_start_imu,
            "cycle_length_optical": cycle_len,
            "cycle_condition": cycle_condition,
            "condition_start_index_optical": condition_start,
            "n_gait_cycles": len(cycle_start),
            "optical_gap_max_deviation_s": gap_dev,
            "inertial_cycle_offset_samples": cycle_offset,
            "inertial_cycle_offset_note": (
                "the 2000 Hz cycle boundaries do NOT sit on multiples of 20 optical samples: per cycle they "
                "deviate by up to +-18 samples (+-9 ms), while the whole-trial total stays within a few "
                "milliseconds. The reading 'exactly 20x in every cycle' reads the median of the per-cycle "
                "ratio, not the per-cycle value. The adapter therefore uses ONE uniform 2000 Hz time base "
                "anchored at the optical start and reports the deviation rather than resampling anything."
            ),
            "inertial_total_offset_samples": int(n_imu - n_opt * SAMPLES_PER_OPTICAL),
            "inertial_optical_span_mismatch_s": float(n_imu / NOMINAL_RATE_HZ - n_opt / OPTICAL_RATE_HZ),
            "force_mat_warmup_cycles": warmup_cycles,
            "force_mat_warmup_note": (
                "the deposit's README excludes the first 150 cycles of the rigid condition from force analysis "
                "only; the marker and inertial data of that stretch are kept here"
            ),
            "inertial_lag_measured": (
                "inertial stream lags the optical stream by 18.0-34.0 ms (shank sensors) and 0.5-26.5 ms "
                "(thigh sensors), /; NOT corrected here"
            ),
            "inertial_stream_interpolated": (
                "released at 2000 Hz but linearly interpolated from the Trigno's native 148.148 Hz; "
                "delivered as deposited, no decimation applied"
            ),
            "right_gait_events_are_duplicates": right_duplicates,
            "gait_events_note": (
                "rightHeelStrikes/rightToeOffs are bit-for-bit copies of the left-side arrays in every trial "
                "of both downloaded participants, so the right-side events are not in the deposit"
            ),
            "left_toe_offs_s": left_toe,
            "optical_gap_filled_no_mask": (
                "Vicon gap filling (Woltring, rigid-body with proximal markers, Nexus kinematic) was applied "
                "before release and no gap mask ships; a rigid-body fill imposes the segment rigidity the "
                "reference is used to measure"
            ),
            "reference_accuracy_limit_deg": (
                "2.06-4.54 rms on knee flexion between two constructions of the same optical reference"
            ),
            "no_still_stretch": "quietest one-second stretch of a whole trial is 56.9 deg/s; no rest anywhere",
            "sensor_to_segment_alignment": (
                "unknown; no marker on any housing, no calibration pose, SENIAM muscle-aligned sensors"
            ),
            "sensor_choice": (
                "vastus lateralis (thigh) and tibialis anterior (shank) as Segment.sensor; a design choice with options"
            ),
            "chain_note": "knees only; no foot, pelvis or trunk sensor, so no ankle and no hip pair",
        }

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=same_segment_pairs,
            contacts_t=left_heel,
            rest_annotation_s=None,  # the deposit documents none and a check of the deposit measured none
            trial_type="walking",
            meta=meta,
        )

    # -- pieces ------------------------------------------------------------------------------------------
    def _time_base(self, f, speed_grp, conditions):
        """Concatenate ``time.time`` over the conditions, re-checking the one-sample continuity."""
        chunks, cycle_start, cycle_len, cycle_condition, condition_start = [], [], [], [], {}
        offset = 0
        for cond in conditions:
            condition_start[cond] = offset
            cells = speed_grp[cond]["time"]["time"]
            for i in range(cells.shape[1]):
                a = np.asarray(f[cells[0, i]]).ravel()
                cycle_start.append(offset)
                cycle_len.append(a.size)
                cycle_condition.append(cond)
                chunks.append(a)
                offset += a.size
        t = np.concatenate(chunks)
        gap_dev = float(np.max(np.abs(np.diff(t) - 1.0 / OPTICAL_RATE_HZ)))
        if gap_dev > 1e-6:
            raise ValueError(
                f"time.time is not continuous across cells (worst step deviation {gap_dev:.3e} s); the "
                "lossless-concatenation property the adapter relies on does not hold for this trial"
            )
        return (
            t,
            np.asarray(cycle_start, dtype=np.int64),
            np.asarray(cycle_len, dtype=np.int64),
            cycle_condition,
            condition_start,
            gap_dev,
        )

    @staticmethod
    def _inertial_cell_lengths(f, speed_grp, conditions) -> np.ndarray:
        """Sample count of every gait-cycle cell in the 2000 Hz stream; shapes only, no data is read."""
        reference_sensor = _SEGMENTS["thigh_r"]["sensor"]
        lengths = [_cell_lengths(f, speed_grp[cond]["imuAcc"], reference_sensor) for cond in conditions]
        return np.concatenate(lengths)

    def _sensor_stream(self, f, speed_grp, conditions, sensor: str, t_imu, n_imu) -> SensorStream:
        acc_parts, gyr_parts = [], []
        for cond in conditions:
            grp = speed_grp[cond]
            acc_parts.append(_concat_cells(f, grp["imuAcc"], sensor))
            gyr_parts.append(_concat_cells(f, grp["imuGyro"], sensor))
        acc = np.concatenate(acc_parts, axis=0) * MM_PER_S2_TO_M_PER_S2
        gyr = np.concatenate(gyr_parts, axis=0) * DEG_TO_RAD
        if acc.shape[0] != n_imu or gyr.shape[0] != n_imu:
            raise ValueError(
                f"sensor {sensor}: {acc.shape[0]} accelerometer and {gyr.shape[0]} gyroscope samples against "
                f"{n_imu} in the reference sensor; the 2000 Hz channels of one trial must agree in length"
            )
        return SensorStream(
            t=t_imu,
            gyr=gyr,
            acc=acc,
            rate_hz=NOMINAL_RATE_HZ,
            family="Delsys Trigno",
            range_gyr=None,  # not documented in the deposit
            range_acc=None,
            native_rate_hz=NATIVE_RATE_HZ,
            meta={
                "field": sensor,
                "site": _SENSOR_SITE[sensor],
                "interpolated": True,
                "interpolation": "linear, from 148.148 Hz onto the 2000 Hz electromyography grid",
                "deposited_units": {"acc": "mm/sec^2", "gyr": "deg/sec"},
                "time_base": "synthesised as t_optical[0] + k/2000; the deposit ships no inertial timestamps",
                "lag_behind_optical_ms": "18.0-34.0 (shank) / 0.5-26.5 (thigh), not corrected",
                "axes": "undocumented; SENIAM muscle-fibre alignment, different for every sensor on a leg",
            },
        )

    def _pose(self, f, speed_grp, conditions, marker_names, t_opt, n_opt) -> Pose:
        tracks = []
        for name in marker_names:
            parts = [_concat_cells(f, speed_grp[cond]["markers"], name) for cond in conditions]
            tracks.append(np.concatenate(parts, axis=0) * MM_TO_M)
        markers = np.stack(tracks, axis=1)  # (T, K, 3) in metres
        if markers.shape[0] != n_opt:
            raise ValueError(f"markers {marker_names}: {markers.shape[0]} frames against {n_opt} optical samples")
        R = _frame_from_three_markers(tracks[0], tracks[1], tracks[2])
        p = markers.mean(axis=1)
        speed_mm_per_s = np.linalg.norm(np.diff(markers, axis=0), axis=2) * OPTICAL_RATE_HZ / MM_TO_M
        glitch = np.unique(np.where(speed_mm_per_s > MARKER_GLITCH_MM_PER_S)[0])
        return Pose(
            t=t_opt,
            R=R,
            p=p,
            rate_hz=OPTICAL_RATE_HZ,
            kind="skin",
            markers=markers,
            marker_names=list(marker_names),
            meta={
                "frame": (
                    "e1 = unit(m2-m1); e3 = unit(e1 x (m3-m1)); e2 = e3 x e1; columns of R, origin = marker centroid"
                ),
                "marker_sites": "skin and tight-fitting clothing; no marker on any sensor housing",
                "deposited_unit": "mm",
                "marker_speed_glitch_frames": glitch.astype(int).tolist(),
                "marker_speed_glitch_threshold_mm_per_s": MARKER_GLITCH_MM_PER_S,
                "gap_filled_before_release": True,
                "gap_mask_shipped": False,
            },
        )


ADAPTER = CompliantTerrainAdapter()
