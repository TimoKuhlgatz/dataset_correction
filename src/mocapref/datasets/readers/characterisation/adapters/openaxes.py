"""OpenAXES: a UR5 industrial robot arm with four IMUs on its links, referenced by the joint encoders.

Files read (under ``paths.dataset_dir("openaxes")/_extracted/``, folder ``measure_raw-<date>``):
- ``imu-<shape>-<speed>/{humerus,radius,carpus,digitus}.csv``: one CSV per IMU and scenario;
- ``robot/robot-<shape>-<speed>.csv``: the UR5 controller stream at 125 Hz.
Eight scenarios ({circle, triangle} x {100, 150, 200, 250} mm/s end-effector speed), three adjacent-IMU pairs each
(q2: humerus-radius, q3: radius-carpus, q4: carpus-digitus).

Units: already SI (accelerometer m/s^2, gyroscope rad/s, encoder angles rad); no conversion. The quaternion columns
are all zero placeholders and are ignored. Of the two accelerometers per board the ADXL355 is used; the BMI160 one
is uncalibrated in the deposit and only kept in ``SensorStream.meta``.

Reference (``kind="encoder"``): link orientations from the six encoder angles with the deposit's own forward
kinematics (calculate-robot-quaternions.py), rotated into each IMU's sensor frame by a constant per-trial,
per-segment Wahba rotation fitted between encoder-derived angular rate and gyroscope over well-excited frames
(the link rotation is kept in ``Pose.meta["link_from_sensor_R"]``). Segment positions are not deposited
(``Pose.p`` is NaN).

Clocks: the IMU sample clock is the firmware counter ``INDEX / 100 Hz`` (no packet lost), placed on the host clock
by the mean of ``TIME - INDEX / 100 Hz``. The robot runs on the controller's clock, and the host-to-controller
offset is not deposited (it moves by 185 ms across the trials). It is recovered per trial: for each unit, the
encoder-derived and the gyroscope angular-rate magnitudes are put on a 100 Hz grid, low-passed at 5 Hz (zero
phase) and cross-correlated over the full lag range; the median of the four units is subtracted from every sensor
time base. The applied offset and the per-unit values are in ``Recording.meta``. Because the reader aligns the
clocks, a measured synchronisation offset on this dataset is a residual of that alignment.

Other corrections and defects: ``circle-200`` has a constant gyroscope offset on the humerus only, removed by the
deposit's own initial-plateau rule before the clock and mounting fits (bias in ``SensorStream.meta``). IMU and
robot records are not co-extensive (``triangle-100``: 121 s against 63 s); nothing is trimmed, the overlap is in
``Recording.meta["overlap_s"]``. Leading rows with an exactly zero gyroscope are kept but left out of the fits.

Licence: CC BY-SA 3.0 (share-alike).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..schema import Pair, Pose, Recording, Segment, SensorStream

SUBJECT = "ur5"
LINKS = ("humerus", "radius", "carpus", "digitus")
"""The four instrumented links, proximal to distal, named as the deposit names its files."""

IMU_ID = {"humerus": "imu1", "radius": "imu2", "carpus": "imu6", "digitus": "imu9"}
IMU_RATE_HZ = 100.0
"""Units: Hz.
Provenance: read-from-dataset — the rate the deposit states."""
RANGE_GYR_DPS = 2000.0
"""The unit's gyroscope full-scale range.

Units: °/s.
Provenance: read-from-dataset — the BMI160's stated range, confirmed by the deposit's own least-significant bit of 0.001065."""

ROBOT_RATE_HZ = 125.0
"""Units: Hz.
Provenance: read-from-dataset — the robot controller's rate as the deposit states it."""

PAIRS = (("humerus", "radius", "q2"), ("radius", "carpus", "q3"), ("carpus", "digitus", "q4"))
"""The three adjacent-IMU pairs and the encoder that drives each, from the deposit's README."""

ACC_COLUMNS = ["ACCEL_ADXL355_0", "ACCEL_ADXL355_1", "ACCEL_ADXL355_2"]
ACC_COLUMNS_BMI = ["ACCEL_BMI160_0", "ACCEL_BMI160_1", "ACCEL_BMI160_2"]
GYR_COLUMNS = ["GYRO_0", "GYRO_1", "GYRO_2"]

MICROSECONDS_PER_S = 1e6
"""The deposit's ``TIME`` column is host Unix-epoch microseconds.

Units: µs per s.
Provenance: physical — an exact unit conversion."""

MILLISECONDS_PER_S = 1000.0
"""Used only to report the spread of the four recovered clock offsets in milliseconds.

Units: ms per s.
Provenance: physical — an exact unit conversion."""

NORM_FLOOR = 1e-12
"""Floor on a vector norm before a division, in the cross-correlation and in the axis-residual angle.

Units: dimensionless (the quantities divided are already normalised correlations and unit vectors).
Provenance: data-free — a numerical guard fifteen orders of magnitude below every quantity these expressions
carry; it changes no value and exists so that an all-zero window returns zero instead of a NaN."""

GYRO_BIAS_TRIGGER_SAMPLES = 20
GYRO_BIAS_TRIGGER_RAD_S = 0.05
GYRO_BIAS_PLATEAU_WINDOW = 50
GYRO_BIAS_PLATEAU_VARIANCE_RATIO = 20.0
"""Provenance: deposited calculate-quaternions.py, first-20 trigger and find_initial_plateau.
The triggered interval is verified as rest on the exposed OpenAXES data before applying this source rule.
These constants are not an estimator setting or a general-purpose rest detector."""


def initial_gyro_bias(gyr: np.ndarray) -> tuple[np.ndarray, dict]:
    """The deposit's initial-plateau offset, without modifying samples or consulting the reference.

    The caller must use this only on the OpenAXES recording protocol whose initial rest was checked.
    A trigger without a defined plateau fails explicitly; it is not evidence that arbitrary motion is rest.
    """
    g = np.asarray(gyr, float)
    if g.ndim != 2 or g.shape[1] != 3 or len(g) < GYRO_BIAS_TRIGGER_SAMPLES or not np.isfinite(g).all():
        raise ValueError("invalid_openaxes_gyro_for_initial_bias")
    first_mean = g[:GYRO_BIAS_TRIGGER_SAMPLES].mean(0)
    note = {"rule": "deposited initial-plateau bias rule; verified OpenAXES initial rest",
            "first_mean_rad_s": first_mean.tolist(), "trigger_rad_s": GYRO_BIAS_TRIGGER_RAD_S,
            "applied": False, "plateau_end_exclusive": None, "bias_rad_s": [0.0, 0.0, 0.0]}
    bias = np.zeros(3)
    if np.linalg.norm(first_mean) <= GYRO_BIAS_TRIGGER_RAD_S:
        return bias, note
    window = GYRO_BIAS_PLATEAU_WINDOW
    if len(g) <= window:
        raise ValueError("openaxes_initial_plateau_too_short")
    variance = np.var(g[:window, 0])
    if variance == 0:
        raise ValueError("openaxes_initial_plateau_variance_undefined")
    for i in range(len(g) - window):
        if np.var(g[i:i + window, 0]) / variance >= GYRO_BIAS_PLATEAU_VARIANCE_RATIO:
            break
    else:
        raise ValueError("openaxes_initial_plateau_end_not_identified")
    end = i + window // 2
    bias = g[:end].mean(0)
    note.update(applied=True, plateau_end_exclusive=end, bias_rad_s=bias.tolist(),
                plateau_gyro_std_rad_s=g[:end].std(0).tolist())
    return bias, note


# --------------------------------------------------------------------------- #
# The synchronisation and alignment protocol: cross-correlation of angular-rate magnitudes for the clock      #
# offset, Wahba fit of encoder angular rate to gyroscope for the sensor-to-link rotation.                     #
# --------------------------------------------------------------------------- #
SYNC_GRID_HZ = 100.0
"""The common grid the encoder rate and the gyroscope are compared on. Equal to the IMU's own rate, so the
gyroscope is not resampled in rate, only shifted in time.

Units: Hz.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

SYNC_LOWPASS_HZ = 5.0
"""Cut-off of the zero-phase low-pass applied to both angular-rate magnitudes before the cross-correlation and
before the Wahba fit. Zero-phase, so it introduces no lag of its own; it is a measurement-side filter and never
touches the samples the adapter emits.

Units: Hz.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

SYNC_LOWPASS_ORDER = 4
"""Order of that Butterworth low-pass (applied forwards and backwards, so the effective order is twice this).

Units: dimensionless.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

SYNC_MIN_OVERLAP_SAMPLES = 1500
"""A lag is only considered if the two series still overlap by at least this many samples (15 s at 100 Hz), so
that a spurious peak cannot be formed by a handful of samples at the extreme of the lag range.

Units: samples.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

SYNC_MIN_OVERLAP_S = 20.0
"""Sanity floor: after the recovered offset is applied, sensor and reference must still overlap by at least
this long, or the recovered offset is not believable and the adapter raises rather than emitting it.

Units: s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_MIN_OMEGA_RAD_S = 1.5
"""Only frames whose low-passed gyroscope magnitude exceeds this enter the Wahba fit: the sensor-to-link
rotation is only observable where the segment actually turns.

Units: rad/s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_FALLBACK_OMEGA_RAD_S = 0.5
"""The excitation gate falls back to this lower value when the primary gate leaves too few frames.

Units: rad/s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_MIN_FRAMES = 500
"""Below this many frames past the primary excitation gate, the fallback gate is used instead.

Units: samples.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_MIN_FRAMES_FALLBACK = 50
"""Below this many frames past the fallback gate, no rotation is fitted and the link frame is emitted as it is,
with ``aligned: False`` on the pose.

Units: samples.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_RESIDUAL_PERCENTILE = 95.0
"""The upper percentile of the axis residual reported beside its median; both are diagnostics, neither gates
anything here.

Units: percent.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""


def _zero_phase_lowpass(x: np.ndarray) -> np.ndarray:
    """Transcribed from ``gt.zero_phase_lp``: a fourth-order Butterworth applied forwards and backwards."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(SYNC_LOWPASS_ORDER, SYNC_LOWPASS_HZ / (SYNC_GRID_HZ / 2), btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float), axis=0)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(T, 3, 3) frame→world rotations to scalar-first quaternions."""
    import qmt

    q = qmt.quatFromRotMat(np.asarray(R, float))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _omega_central(q: np.ndarray) -> np.ndarray:
    """Body-frame angular rate (rad/s) by a central difference, transcribed from ``gt.omega_central``.

    Central rather than forward, because a half-sample bias would show up as a spurious one-sample lag against
    the gyroscope — which is exactly the quantity being measured here.
    """
    import qmt

    d = qmt.qmult(qmt.qinv(q[:-2]), q[2:])
    d = d * np.where(d[:, :1] < 0, -1.0, 1.0)  # the double cover: take the short way round
    v = qmt.quatToRotVec(d) * (SYNC_GRID_HZ / 2)
    return np.concatenate([v[:1], v, v[-1:]], axis=0)


def _resample(t: np.ndarray, x: np.ndarray, t_out: np.ndarray, quat: bool = False) -> np.ndarray:
    """Linear for vectors, SLERP for quaternions — transcribed from ``adapter_common.resample_uniform``."""
    import qmt

    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if quat:
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return qmt.quatInterp(x, np.interp(t_out, t, np.arange(len(t))))
    return np.stack([np.interp(t_out, t, x[:, k]) for k in range(x.shape[1])], axis=1)


def _full_xcorr_lag(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Integer lag of ``y`` behind ``x`` over the **full** lag range, and the correlation there.

    Transcribed from ``adapter_openaxes._full_xcorr_lag``. The correlation is divided by the overlap length so
    that lags near the extremes, where few samples overlap, are not favoured; lags whose overlap is shorter
    than :data:`SYNC_MIN_OVERLAP_SAMPLES` are excluded outright. Nothing bounds the search otherwise, which is
    the point: the true offset on this deposit is seconds, not milliseconds.
    """
    from scipy.signal import correlate

    x = x - x.mean()
    y = y - y.mean()
    c = correlate(x, y, mode="full", method="fft")
    n = np.arange(-len(y) + 1, len(x))
    overlap = np.minimum(np.minimum(len(x), len(y)), np.minimum(len(x) - n, len(y) + n)).clip(1)
    usable = overlap >= SYNC_MIN_OVERLAP_SAMPLES
    lag = int(n[usable][np.argmax((c / overlap)[usable])])
    if lag >= 0:
        a, b = x[lag:], y[: len(x) - lag]
    else:
        a, b = x[: len(x) + lag], y[-lag:]
    m = min(len(a), len(b))
    a, b = a[:m] - a[:m].mean(), b[:m] - b[:m].mean()
    peak = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + NORM_FLOOR))
    return lag, peak


def _axis_angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between two stacks of vectors, in degrees. Transcribed from ``gt._axis_angle_deg``."""
    un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), NORM_FLOOR)
    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), NORM_FLOOR)
    return np.rad2deg(np.arccos(np.clip(np.sum(un * vn, axis=1), -1.0, 1.0)))


def _wahba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares rotation ``R`` with ``a_i ≈ R b_i``, by the Kabsch/Markley singular-value decomposition."""
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def _sensor_align(q_link: np.ndarray, gyr: np.ndarray) -> tuple[np.ndarray, dict]:
    """The constant link←sensor rotation, measured from the encoders against the bias-corrected gyroscope.

    Transcribed from ``adapter_common.sensor_align``. ``q_link`` is the link's orientation in the robot base
    frame and ``gyr`` that link's bias-corrected gyroscope, both already on the same grid and the same clock. Returns the
    (3, 3) rotation to post-multiply the link rotations by, and the fit's diagnostics. Only the bias-corrected gyroscope
    enters — no fusion, no model, and no method's error.
    """
    w_link = _zero_phase_lowpass(_omega_central(q_link))
    w_gyr = _zero_phase_lowpass(np.asarray(gyr, float))
    excited = np.linalg.norm(w_gyr, axis=1) > ALIGN_MIN_OMEGA_RAD_S
    gate = ALIGN_MIN_OMEGA_RAD_S
    if excited.sum() < ALIGN_MIN_FRAMES:
        excited = np.linalg.norm(w_gyr, axis=1) > ALIGN_FALLBACK_OMEGA_RAD_S
        gate = ALIGN_FALLBACK_OMEGA_RAD_S
    if excited.sum() < ALIGN_MIN_FRAMES_FALLBACK:
        return np.eye(3), {"aligned": False, "reason": "too few excited frames", "n_align": int(excited.sum())}
    rotation = _wahba(w_link[excited], w_gyr[excited])  # link ← sensor
    after = _axis_angle_deg(w_link[excited], w_gyr[excited] @ rotation.T)
    before = _axis_angle_deg(w_link[excited], w_gyr[excited])
    return rotation, {
        "aligned": True,
        "n_align": int(excited.sum()),
        "excitation_gate_rad_s": float(gate),
        "link_from_sensor_angle_deg": float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))),
        "omega_axis_residual_deg_median": float(np.median(after)),
        "omega_axis_residual_deg_p95": float(np.percentile(after, ALIGN_RESIDUAL_PERCENTILE)),
        "omega_axis_before_align_deg_median": float(np.median(before)),
    }


def _segment_rotations(q: np.ndarray) -> dict[str, np.ndarray]:
    """Link orientations from the six encoder angles, transcribed from ``calculate-robot-quaternions.py``.

    ``q`` is ``(T, 6)`` in radians. Returns one ``(T, 3, 3)`` rotation matrix array per instrumented link.
    Provenance of the model: read-from-dataset (the deposit's own script). Positions are not part of it.
    """
    from scipy.spatial.transform import Rotation

    def _rot(axis: str, angle: np.ndarray):
        # scipy wants the last dimension to match the number of sequence axes, so a one-axis sequence over T
        # samples is passed as (T, 1)
        return Rotation.from_euler(axis, np.asarray(angle, float).reshape(-1, 1))

    base = _rot("y", -q[:, 0])
    humerus_joint = _rot("x", q[:, 1] + np.pi / 2)
    radius_joint = _rot("x", q[:, 2])
    carpus_joint = _rot("x", q[:, 3] + np.pi / 2)
    digitus_joint = _rot("y", -q[:, 4])
    rx = Rotation.from_euler("x", -np.pi / 2)

    humerus = rx * base * humerus_joint
    radius = humerus * radius_joint
    carpus = radius * carpus_joint
    digitus = carpus * digitus_joint
    return {
        "humerus": humerus.as_matrix(),
        "radius": radius.as_matrix(),
        "carpus": carpus.as_matrix(),
        "digitus": digitus.as_matrix(),
    }


class OpenAxes:
    """The eight UR5 scenarios; one subject (the robot), one trial per scenario, three pairs each."""

    name = "openaxes"
    reference_kind = "encoder"
    sensor_family = "OpenAXES node (ADXL355 accelerometer + BMI160 gyroscope), no magnetometer"
    licence = "CC BY-SA 3.0 — share-alike: derivatives must be licensed alike"
    source = (
        "Webering, F.; Fahnemann, C.; Stanislawski, N., 'OpenAXES Example Robot Dataset', Forschungsdaten-"
        "Repositorium der Leibniz Universität Hannover, v1.0 (2023), doi:10.25835/psqqs9w0. No data paper. "
        "See the deposit's SOURCE.md"
    )

    def __init__(self, root: Path | None = None) -> None:
        base = Path(root) if root is not None else paths.dataset_dir("openaxes")
        self.root = base / "_extracted" / "measure_raw-2022-09-15"

    def trials(self) -> list[tuple[str, str]]:
        if not self.root.is_dir():
            return []
        return [(SUBJECT, p.name.replace("imu-", "")) for p in sorted(self.root.glob("imu-*")) if p.is_dir()]

    def recordings(self, limit: int | None = None):
        for i, (subject, trial) in enumerate(self.trials()):
            if limit is not None and i >= limit:
                return
            yield self.load(subject, trial)

    def _read_imu(self, path: Path) -> dict:
        """One IMU CSV: bias-corrected samples and the host-clock time base of every sample.

        The sample clock is ``INDEX / 100 Hz``; the host ``TIME`` column only fixes its origin, through the
        mean of ``TIME − INDEX/100 Hz`` over the rows that carry signal. Leading rows whose gyroscope is
        exactly zero are counted and reported but **not** removed from the emitted samples; the fits below
        start after them.
        """
        import pandas as pd

        frame = pd.read_csv(path)
        index = frame["INDEX"].to_numpy(np.int64)
        gyr = frame[GYR_COLUMNS].to_numpy(float)
        acc = frame[ACC_COLUMNS].to_numpy(float)
        first_live = int(np.argmax(np.abs(gyr).sum(axis=1) > 0))
        host_s = frame["TIME"].to_numpy(float) / MICROSECONDS_PER_S
        counter_s = index / IMU_RATE_HZ
        origin = float((host_s[first_live:] - counter_s[first_live:]).mean())
        bias, bias_note = initial_gyro_bias(gyr)
        if bias_note["applied"]:
            gyr[first_live:] -= bias  # do not turn leading missing-data zeros into a calibration signal
        return {
            "index": index,
            "gyr": gyr,
            "acc": acc,
            "t_host": counter_s + origin,
            "first_live": first_live,
            "initial_gyro_bias": bias_note,
            "mac": str(frame["MAC"].iloc[0]),
            "pc_clock_first_sample_us": int(frame["TIME"].iloc[0]),
        }

    def _clock_offset(self, t_controller: np.ndarray, rotations: dict[str, np.ndarray], raw: dict) -> dict:
        """The per-trial host-minus-controller clock offset, by full cross-correlation, unit by unit.

        The encoder-derived angular-rate magnitude
        of a link and that link's gyroscope magnitude are put on their own 100 Hz grids, low-passed, and
        cross-correlated over the full lag range; the offset of the trial is the **median** of the four units,
        and their spread is published as an independent check.
        """
        grid_ref = np.arange(t_controller[0], t_controller[-1], 1 / SYNC_GRID_HZ)
        per_unit: dict[str, dict] = {}
        for link in LINKS:
            q_ref = _resample(t_controller, _rotmat_to_quat(rotations[link]), grid_ref, quat=True)
            w_ref = np.linalg.norm(_zero_phase_lowpass(_omega_central(q_ref)), axis=1)
            live = raw[link]["first_live"]
            t_live = raw[link]["t_host"][live:]
            grid_imu = np.arange(t_live[0], t_live[-1], 1 / SYNC_GRID_HZ)
            w_gyr = np.linalg.norm(_zero_phase_lowpass(_resample(t_live, raw[link]["gyr"][live:], grid_imu)), axis=1)
            lag, peak = _full_xcorr_lag(w_ref, w_gyr)
            per_unit[link] = {
                "host_minus_controller_clock_s": float(grid_imu[0] - grid_ref[0] - lag / SYNC_GRID_HZ),
                "cross_correlation_peak": peak,
                "lag_samples": lag,
            }
        offsets = [v["host_minus_controller_clock_s"] for v in per_unit.values()]
        return {
            "offset_s": float(np.median(offsets)),
            "spread_ms": float((max(offsets) - min(offsets)) * MILLISECONDS_PER_S),
            "per_unit": per_unit,
        }

    def load(self, subject: str, trial: str) -> Recording:
        import pandas as pd

        imu_dir = self.root / f"imu-{trial}"
        robot_csv = self.root / "robot" / f"robot-{trial}.csv"
        robot = pd.read_csv(robot_csv)
        q = robot[[f"q{i}" for i in range(6)]].to_numpy(float)  # rad, six UR5 joint encoders
        t_controller = robot["timestamp"].to_numpy(float)  # the UR controller's own uptime clock
        t_robot = t_controller - t_controller[0]
        rotations = _segment_rotations(q)

        raw = {link: self._read_imu(imu_dir / f"{link}.csv") for link in LINKS}
        sync = self._clock_offset(t_controller, rotations, raw)
        # the measured offset, applied to every unit of the trial by the same rule
        t_sensor = {link: raw[link]["t_host"] - sync["offset_s"] - t_controller[0] for link in LINKS}

        low = max([float(t_robot[0])] + [float(t_sensor[k][raw[k]["first_live"]]) for k in LINKS])
        high = min([float(t_robot[-1])] + [float(t_sensor[k][-1]) for k in LINKS])
        if high - low < SYNC_MIN_OVERLAP_S:
            raise ValueError(
                f"openaxes/{trial}: only {high - low:.1f} s of sensor-reference overlap after the measured "
                f"clock offset of {sync['offset_s']:.4f} s — the offset is not believable"
            )
        grid = np.arange(low, high, 1 / SYNC_GRID_HZ)

        segments: dict[str, Segment] = {}
        index_origin: dict[str, int] = {}
        alignment: dict[str, dict] = {}
        for link in LINKS:
            index = raw[link]["index"]
            index_origin[link] = int(index[0])
            t = t_sensor[link]
            steps = np.unique(np.diff(index))
            q_link = _resample(t_robot, _rotmat_to_quat(rotations[link]), grid, quat=True)
            link_from_sensor, alignment[link] = _sensor_align(q_link, _resample(t, raw[link]["gyr"], grid))
            segments[link] = Segment(
                name=link,
                sensor=SensorStream(
                    t=t,
                    gyr=raw[link]["gyr"],
                    acc=raw[link]["acc"],
                    rate_hz=IMU_RATE_HZ,
                    family=self.sensor_family,
                    range_gyr=float(np.deg2rad(RANGE_GYR_DPS)),
                    range_acc=None,  # the ADXL355 range is not stated in the deposit
                    native_rate_hz=IMU_RATE_HZ,
                    meta={
                        "units_as_deposited": "acc m/s², gyr rad/s — already SI, no conversion applied",
                        "gravity_included": True,
                        "device": IMU_ID[link],
                        "mac": raw[link]["mac"],
                        "accelerometer_used": "ADXL355",
                        "second_accelerometer_on_board": "BMI160, uncalibrated in the deposit (mean norm up to "
                                                         "4 % off); not exposed as a second sensor because it "
                                                         "is the same physical device",
                        "bmi160_columns": ACC_COLUMNS_BMI,
                        "time_base": "INDEX / 100 Hz — the firmware sample counter, which steps by exactly one "
                                     "everywhere — placed on the host clock through the mean of TIME minus "
                                     "INDEX/100 Hz, then shifted onto the robot controller's clock by the "
                                     "trial's measured offset and zeroed at the robot's first sample",
                        "clock_offset_applied_s": sync["offset_s"],
                        "clock_offset_this_unit_s": sync["per_unit"][link]["host_minus_controller_clock_s"],
                        "clock_cross_correlation_peak": sync["per_unit"][link]["cross_correlation_peak"],
                        "index_steps_present": steps.tolist(),
                        "index_origin": int(index[0]),
                        "leading_zero_gyroscope_rows": raw[link]["first_live"],
                        "initial_gyro_bias": raw[link]["initial_gyro_bias"],
                        "pc_clock_first_sample_us": raw[link]["pc_clock_first_sample_us"],
                        "deposited_quaternion_columns_are_zero": True,
                    },
                ),
                reference=Pose(
                    t=t_robot,
                    R=rotations[link] @ link_from_sensor,
                    p=np.full((len(t_robot), 3), np.nan),
                    rate_hz=ROBOT_RATE_HZ,
                    kind="encoder",
                    markers=None,
                    meta={
                        "construction": "forward kinematics from the six joint encoders, transcribed from the "
                                        "deposit's calculate-robot-quaternions.py, then rotated into the "
                                        "sensor frame by the measured constant link←sensor rotation",
                        "frame": "the IMU's own sensor frame in the robot base frame — the IMU-to-link "
                                 "mounting rotation is not deposited, so it is measured per trial and per "
                                 "segment by a Wahba fit of the encoder angular rate against the bias-corrected "
                                 "gyroscope (fixed protocol, not tuned on this dataset)",
                        "link_from_sensor_R": link_from_sensor.tolist(),
                        "link_from_sensor_fit": alignment[link],
                        "positions_stored": False,
                        "p_is": "NaN — only the tool-centre position is deposited, not segment origins",
                        "clock": "the UR controller's uptime clock, zeroed at the first sample; the sensor "
                                 "streams were shifted onto this clock by the trial's measured offset",
                    },
                ),
                mount="screwed to the robot link; the mounting rotation is identified from a calibration move, "
                      "not measured by markers",
            )

        pairs = [
            Pair(
                parent=parent,
                child=child,
                joint="hinge",
                joint_dof=1,
                joint_centre_world=None,
                joint_centre_t=None,
                meta={
                    "encoder": encoder,
                    "encoder_angle_rad": q[:, int(encoder[1:])],
                    "encoder_t": t_robot,
                    "encoder_rate_hz": ROBOT_RATE_HZ,
                    "encoder_note": "the deposit's README assigns this encoder to this adjacent-IMU pair; q0 "
                                    "and q1 both move the humerus and have no IMU above them, and q5 rotates "
                                    "the tool without moving the end-effector position",
                },
            )
            for parent, child, encoder in PAIRS
        ]

        defects = [
            (
                "the IMU-to-link mounting rotation is identified from a calibration move by an unreleased "
                "program; corrected here by a measured per-trial per-segment link←sensor Wahba rotation, "
                "applied to every trial by the same rule"
            ),
            (
                "the constant offset between the IMU clock and the robot clock is not deposited; corrected "
                "here by a measured per-trial offset from the full-lag-range cross-correlation of the encoder "
                "angular-rate magnitude against the gyroscope magnitude, applied to every trial by one rule"
            ),
        ]
        if trial.startswith("triangle"):
            defects.append("nine triangle-* sequences carry a 10–27° constant misalignment (concept.md §2); "
                           "removed by the measured link←sensor rotation above")
        if trial == "circle-200":
            defects.append(
                "a constant gyroscope offset on the humerus stream only — mean of the first twenty samples "
                "(0.268, −0.272, −0.273) rad/s; removed using the deposited initial-plateau "
                "rule before synchronization and mounting fits; raw source files preserved"
            )
        imu_durations = {k: float(s.sensor.t[-1] - s.sensor.t[0]) for k, s in segments.items()}
        if max(imu_durations.values()) > 1.5 * float(t_robot[-1]):
            defects.append(
                f"the IMU record ({max(imu_durations.values()):.0f} s) is far longer than the robot record "
                f"({t_robot[-1]:.0f} s); the streams are not co-extensive and nothing here trims them"
            )

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],
            contacts_t=None,  # a robot arm: no ground contact
            rest_annotation_s=None,
            trial_type=trial.split("-")[0],
            meta={
                "licence": self.licence,
                "source": self.source,
                "imu_dir": str(imu_dir),
                "robot_csv": str(robot_csv),
                "end_effector_speed_mm_s": int(trial.split("-")[1]),
                "clock_offset_sensor_to_reference_s": sync["offset_s"],
                "clock_offset_is": "the host-minus-controller offset in seconds that this adapter HAS ALREADY "
                                   "SUBTRACTED from every sensor time base, not a residual: as emitted, the "
                                   "sensor streams and the reference are on one clock. It is of order 1.66e9 "
                                   "because the host clock is a Unix epoch and the controller's is its uptime",
                "clock_offset_per_unit_s": {
                    k: v["host_minus_controller_clock_s"] for k, v in sync["per_unit"].items()
                },
                "clock_offset_unit_spread_ms": sync["spread_ms"],
                "clock_cross_correlation_peaks": {
                    k: v["cross_correlation_peak"] for k, v in sync["per_unit"].items()
                },
                "clock_note": "not deposited; the deposit's example notebook uses a hand-tuned per-scenario "
                              "constant (+265 s for circle-100). This adapter measures it instead, per trial, "
                              "by full-lag-range cross-correlation of the encoder angular-rate magnitude "
                              "against each unit's gyroscope magnitude (fixed protocol, not tuned on this dataset)",
                "link_from_sensor_fits": alignment,
                "overlap_s": float(high - low),
                "imu_index_origins": index_origin,
                "imu_durations_s": imu_durations,
                "robot_duration_s": float(t_robot[-1]),
                "defects": defects,
            },
        )


ADAPTER = OpenAxes()
