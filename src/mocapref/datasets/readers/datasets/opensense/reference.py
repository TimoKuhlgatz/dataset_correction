"""The OpenSense reference reconstruction: from marker plates and raw sensors to one housing orientation per
sensor, with the quality numbers of each trial. Linear algebra only: no skeleton model, inverse kinematics,
nonlinear solver or sensor fusion.

1. Read the marker file and group its markers into the eight sensor plates.
2. Pick, per plate, one clean reference cluster: the frame in which every marker is visible and the inter-marker
   distances are closest to the trial's medians, expressed in the plate's own O/X/Y frame.
3. Fit that cluster onto the observed markers frame by frame (Kabsch), giving laboratory <- plate over time, and
   fill short marker dropouts spherically.
4. Turn the laboratory frame from Y-up to Z-up.
5. Read the raw sensors (:mod:`.sensors`) and fit, per sensor, the one constant rotation between plate and housing
   that carries the gyroscope's angular rate onto the cluster's (raw gyroscope only, never the onboard fusion).
6. Synchronisation: the integer shift that maximises the correlation of the angular-rate magnitudes is measured
   and reported per sensor; removing a constant lag is switched off (REMOVE_CONSTANT_LAG), so the deposit's
   pairing is kept.

The marker folder is matched without regard to case (one trial names it ``MoCap``). The native sensor rate is
passed out with the trial. The result is ``q(Z-up laboratory <- sensor)(t)`` per sensor at 100 Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import markers as marker_reader
from .deposit_files import READ_RATE_HZ, trial_folder
from .sensors import native_rate_of, read_sensors

# --------------------------------------------------------------------------- #
# where the trial's files sit                                                  #
# --------------------------------------------------------------------------- #
MOCAP_FOLDER = "Mocap"
"""The name of the marker folder inside a trial folder — matched **case-insensitively**, because one subject's
walking trial spells it ``MoCap`` and the reconstruction this was ported from, which compared the name
exactly, dropped that whole trial as if it carried no markers."""


def marker_file_path(subject_dir: Path, task: str) -> Path | None:
    """The trial's ``.trc``, or ``None`` when the trial ships no marker folder or no marker file.

    The marker folder is matched without regard to case; see :data:`MOCAP_FOLDER`.
    """
    trial = trial_folder(task)
    trial_dir = Path(subject_dir) / trial
    if not trial_dir.is_dir():
        return None
    for child in sorted(trial_dir.iterdir()):
        if child.is_dir() and child.name.lower() == MOCAP_FOLDER.lower():
            candidate = child / f"{trial}.trc"
            if candidate.exists():
                return candidate
    return None


# --------------------------------------------------------------------------- #
# the reconstruction's own settings                                            #
# --------------------------------------------------------------------------- #
MAX_FILLED_GAP_S = 0.5
"""Marker dropouts no longer than this are filled by spherical interpolation of the plate quaternion; longer
ones stay missing and the trial reports how long its longest one was.

Units: seconds.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

MIN_KABSCH_MARKERS = 3
"""Fewest visible markers of a plate before a frame gets a fitted rotation at all.

Units: markers.
Provenance: data-free — three non-collinear points is the smallest set that fixes a rigid frame."""

MIN_SENSORS = 2
"""Fewest sensors a trial must yield before it is reconstructed at all — a relative orientation needs two.

Units: sensors.
Provenance: data-free — the arity of the quantity this dataset is scored on."""


MIN_COMPLETE_FRAMES = 10
"""Fewest frames with every marker of a plate visible before the plate is reconstructed; below it there is no
clean reference cluster to pick and the plate is reported as unusable.

Units: frames.
Provenance: data-free — a minimum sample count; below it nothing is produced for that plate."""

REPORTED_PERCENTILE = 95
"""The percentile the trial's diagnostics report beside every median they report — the fit residual, the
angular-rate axis residual and the quiet-frame tilt. Nothing is selected on it.

Units: percent.
Provenance: data-free — a reported percentile of the whole distribution, not a threshold."""

ZUP_FROM_LAB_QUATERNION = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
"""The fixed rotation from the motion-capture world's frame to a Z-up world: a quarter turn about x.

The OpenSense laboratory and the OpenSim world are Y-up, so without this the "vertical" the reference implies
would be a horizontal axis. Only the parts of the pipeline that use the *absolute* orientation care, but they
need the true vertical, so the rotation is applied here and checked afterwards: every trial re-measures where
gravity actually points in its own laboratory frame (``gravity_lab_measured`` in the diagnostics).

Units: quaternion, scalar-first.
Provenance: physical — the stated axis convention of the motion-capture world, verified per trial against the
measured direction of gravity."""

ALIGNMENT_LOWPASS_HZ = 5.0
"""Cut-off of the zero-phase low-pass applied to both angular-rate signals before the plate-to-sensor rotation
is fitted and before the synchronisation is estimated. It never touches a published signal: the reference and
the sensor streams this module returns are unfiltered.

Units: Hz.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

LOWPASS_ORDER = 4
"""Order of that Butterworth low-pass (applied forwards and backwards, so the effective order is twice it).

Units: filter order.
Provenance: textbook — a fourth-order Butterworth, the standard choice for a zero-phase biomechanics filter."""

ALIGNMENT_MIN_RATE_RAD_S = 1.5
"""Only frames in which the sensor is turning at least this fast enter the plate-to-sensor rotation fit: a
frame at rest carries no information about a rotation and only adds noise to it.

Units: rad/s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGNMENT_FALLBACK_RATE_RAD_S = 0.5
"""The gate the fit falls back to when the trial does not offer :data:`ALIGNMENT_MIN_FRAMES` frames above
:data:`ALIGNMENT_MIN_RATE_RAD_S` — a slow trial is fitted on its own best-excited frames rather than not at
all.

Units: rad/s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGNMENT_MIN_FRAMES = 500
"""Fewest well-excited frames before the fit is run at the full gate rather than the fallback one.

Units: frames.
Provenance: data-free — a minimum sample count for a three-parameter least-squares fit."""

MAX_LAG_SAMPLES = 300
"""Widest integer shift the synchronisation searches, in either direction — three seconds at the read rate.

Units: samples.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

MIN_OVERLAP_SAMPLES = 100
"""Fewest overlapping samples a candidate shift must leave before its correlation is formed at all.

Units: samples.
Provenance: data-free — a minimum sample count; a shorter overlap produces no correlation."""

WORSE_THAN_ANY_CORRELATION = -2.0
"""The value the best-correlation search starts from, below the −1 a correlation can reach, so that the first
candidate always replaces it.

Units: correlation.
Provenance: data-free — an initial value outside the range of the quantity it is compared against."""

TRIAL_SHIFT_TOLERANCE_SAMPLES = 2
"""A trial whose sensors all sit within this many samples of the marker stream is left alone; beyond it the
trial is a candidate for a constant-shift correction.

Units: samples.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

TRIAL_SHIFT_MIN_XCORR = 0.90
"""A constant shift is removed only when **every** sensor of the trial is this confidently synchronised. The
correction is deliberately conservative: it fires only where the trial already fails the tolerance above, all
sensors are confident, and they agree on the shift to within :data:`TRIAL_SHIFT_MAX_SPREAD_SAMPLES`.
Otherwise nothing is shifted.

Units: correlation.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

TRIAL_SHIFT_MAX_SPREAD_SAMPLES = 1
"""How far the per-sensor shifts of one trial may disagree before the trial-level correction refuses to fire.

Units: samples.
Provenance: data-free — the sensors of one trial share a clock, so a spread beyond one sample means the
estimates are not measuring one common offset."""

QUIET_ACCELERATION_TOLERANCE = 0.3
"""How far the accelerometer magnitude may sit from gravity for a frame to count as quiet.

Units: m/s².
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

QUIET_RATE_RAD_S = 0.1
"""How fast a frame may be turning and still count as quiet.

Units: rad/s.
Provenance: inherited-protocol — the same quiet-frame gate."""

QUIET_MIN_FRAMES = 200
"""Fewest quiet frames before the tilt agreement between the reference and the accelerometer is reported.

Units: frames.
Provenance: data-free — a minimum sample count; below it the trial reports the check as not measurable."""

GRAVITY_M_S2 = 9.81
"""The gravity magnitude the quiet-frame check compares the accelerometer against.

Units: m/s².
Provenance: physical — standard gravity, to the two decimals the check needs."""

SIGN_GUARD = 1e-300
"""Added to a quaternion's scalar part before taking its sign, so that an exactly zero scalar part maps to
``+1`` rather than to ``0`` and the quaternion is not zeroed.

Units: dimensionless.
Provenance: data-free — a numerical guard far below every quantity the expression carries."""

DIRECTION_NORM_FLOOR = 1e-12
"""Floor on a vector norm before it is used to normalise that vector, so a zero-length vector produces a
finite direction rather than a division by zero.

Units: same as the vector normalised (rad/s).
Provenance: data-free — a numerical guard far below every angular rate the expression carries."""

JOINT_BODIES = [
    ("hip_r", "pelvis", "femur_r"),
    ("knee_r", "femur_r", "tibia_r"),
    ("ankle_r", "tibia_r", "calcn_r"),
    ("hip_l", "pelvis", "femur_l"),
    ("knee_l", "femur_l", "tibia_l"),
    ("ankle_l", "tibia_l", "calcn_l"),
    ("lumbar", "pelvis", "torso"),
]
"""Joint -> (parent body, child body), as the deposit's own model names them. Used here only by the external
consistency check below."""


# --------------------------------------------------------------------------- #
# small numerical helpers                                                      #
# --------------------------------------------------------------------------- #
def normalise_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def angular_rate_central(q: np.ndarray, rate_hz: float = READ_RATE_HZ) -> np.ndarray:
    """Body-frame angular velocity (rad/s) from a quaternion series, by **central** difference.

    A forward difference would put the rate half a sample ahead of the quaternion, which the synchronisation
    below would read as a genuine one-sample lag against the gyroscope. The central difference has no such
    bias; the first and last samples are repeated to keep the length.
    """
    import qmt

    step = qmt.qmult(qmt.qinv(q[:-2]), q[2:])
    step = step * np.sign(step[:, :1] + SIGN_GUARD)
    rate = qmt.quatToRotVec(step) * (rate_hz / 2.0)
    return np.concatenate([rate[:1], rate, rate[-1:]], axis=0)


def zero_phase_lowpass(x: np.ndarray, cutoff_hz: float = ALIGNMENT_LOWPASS_HZ,
                       rate_hz: float = READ_RATE_HZ) -> np.ndarray:
    """Forward-backward Butterworth low-pass. Used on the two angular-rate signals only, never on an output."""
    from scipy.signal import butter, sosfiltfilt

    sections = butter(LOWPASS_ORDER, cutoff_hz / (0.5 * rate_hz), btype="low", output="sos")
    x = np.asarray(x, float)
    valid = np.isfinite(x).all(axis=1) if x.ndim > 1 else np.isfinite(x)
    if valid.all():
        return sosfiltfilt(sections, x, axis=0)  # preserve the historical finite route exactly
    out = np.full_like(x, np.nan)
    # SciPy's default SOS padding requirement; algorithmic, not a tuned gap rule.
    padlen = 3 * (2 * len(sections) + 1
                  - min((sections[:, 2] == 0).sum(), (sections[:, 5] == 0).sum()))
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int)))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start > padlen:
            out[start:stop] = sosfiltfilt(sections, x[start:stop], axis=0)
    return out


def best_shift(x: np.ndarray, y: np.ndarray, max_lag: int = MAX_LAG_SAMPLES) -> tuple[int, float]:
    """The integer shift ``k`` maximising ``corr(x[t + k], y[t])``, and that correlation."""
    n = min(len(x), len(y))
    x, y = np.asarray(x)[:n], np.asarray(y)[:n]
    best, best_lag = WORSE_THAN_ANY_CORRELATION, 0
    for k in range(-max_lag, max_lag + 1):
        a, b = (x[k:], y[:n - k]) if k >= 0 else (x[:n + k], y[-k:])
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.all():
            a, b = a[finite], b[finite]
        if len(a) < MIN_OVERLAP_SAMPLES:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        correlation = float(a @ b / denominator) if denominator > 0 else 0.0
        if correlation > best:
            best, best_lag = correlation, int(k)
    return best_lag, best


def wahba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The least-squares rotation ``R`` with ``a_i ≈ R b_i`` (Kabsch/Markley, with the reflection guard)."""
    covariance = a.T @ b
    left, _singular, right = np.linalg.svd(covariance)
    determinant = np.sign(np.linalg.det(left @ right))
    return left @ np.diag([1.0, 1.0, determinant]) @ right


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-row angle between two sets of vectors, in degrees."""
    un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), DIRECTION_NORM_FLOOR)
    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), DIRECTION_NORM_FLOOR)
    return np.rad2deg(np.arccos(np.clip(np.sum(un * vn, axis=1), -1.0, 1.0)))


def fill_short_gaps(q: np.ndarray, max_gap: int) -> tuple[np.ndarray, int, int]:
    """Fill missing quaternion runs no longer than ``max_gap`` spherically.

    Returns the filled series, the longest missing run found (filled or not) and how many samples were missing
    before the fill. A run that touches either end of the trial is never filled — there is nothing to
    interpolate between.
    """
    import qmt

    missing = ~np.isfinite(q).all(axis=1)
    if not missing.any():
        return q, 0, 0
    runs, i, longest = [], 0, 0
    n = len(q)
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            runs.append((i, j - i))
            longest = max(longest, j - i)
            i = j
        else:
            i += 1
    out = q.copy()
    for start, length in runs:
        if length <= max_gap and start > 0 and start + length < n:
            # Bound the interpolation to these two observed endpoints. Never let
            # nanInterp fill an unrelated long or endpoint gap.
            stop = start + length
            out[start:stop] = qmt.nanInterp(q[start - 1:stop + 1])[1:-1]
    return out, longest, int(missing.sum())


# --------------------------------------------------------------------------- #
# step 1-3: the plates                                                         #
# --------------------------------------------------------------------------- #
def plate_orientations(trc_path: Path) -> dict:
    """Every plate of one marker file -> its laboratory-frame quaternion series plus quality diagnostics."""
    marker_file = marker_reader.read_marker_file(trc_path)
    plates = marker_reader.plate_markers(marker_file)
    quaternions, diagnostics = {}, {}
    for sensor, cluster in plates.items():
        seen = marker_reader.visibility(cluster)
        names, vis = seen["names"], seen["vis"]
        rigid = marker_reader.rigidity(cluster, vis, names)

        # The clean reference cluster: among the frames where every marker is visible, the one whose pairwise
        # distances sit closest to the trial's medians — an undeformed, fully observed plate.
        complete = vis.all(axis=1)
        if complete.sum() < MIN_COMPLETE_FRAMES:
            diagnostics[sensor] = {"status": "too few complete frames", "n_full_frames": int(complete.sum())}
            continue
        complete_index = np.where(complete)[0]
        reference_frame = int(complete_index[np.nanargmin(rigid["worst_dev_m"][complete_index])])

        observed = np.stack([cluster[n][reference_frame] for n in names], axis=0)  # (K, 3) laboratory
        nominal = marker_reader.nominal_frame(
            cluster["O"][reference_frame:reference_frame + 1],
            cluster["X"][reference_frame:reference_frame + 1],
            cluster["Y"][reference_frame:reference_frame + 1],
        )[0]
        local = (observed - observed.mean(axis=0)) @ nominal  # the cluster in the plate's own frame

        rotation, rmse = marker_reader.kabsch_series(cluster, vis, names, local,
                                                     min_markers=MIN_KABSCH_MARKERS)
        q = marker_reader.rotation_matrix_to_quaternion(rotation)
        q_filled, longest_gap, n_missing = fill_short_gaps(
            q, round(MAX_FILLED_GAP_S * marker_file["rate"]))
        quaternions[sensor] = q_filled
        diagnostics[sensor] = {
            "status": "ok",
            "markers": names,
            "n_markers": len(names),
            "ref_frame": reference_frame,
            "rigidity_pairs": rigid["pairs"],
            "rigidity_sd_mm_max": max(p["sd_mm"] for p in rigid["pairs"]),
            "rigidity_p95_mm_max": max(p["p95_abs_mm"] for p in rigid["pairs"]),
            "rigidity_frac_gt5mm_max": max(p["frac_gt5mm"] for p in rigid["pairs"]),
            "frac_frames_all_markers": float(complete.mean()),
            "frac_frames_lt3_markers": float((vis.sum(axis=1) < MIN_KABSCH_MARKERS).mean()),
            "n_quat_nan_before_fill": int(n_missing),
            "longest_gap_samples": int(longest_gap),
            "n_quat_nan_after_fill": int((~np.isfinite(q_filled).all(axis=1)).sum()),
            "kabsch_rmse_mm_median": float(np.nanmedian(rmse) * marker_reader.MM_PER_M),
            "kabsch_rmse_mm_p95": float(np.nanpercentile(rmse, REPORTED_PERCENTILE) * marker_reader.MM_PER_M),
        }
    return {"rate": marker_file["rate"], "n_frames": marker_file["n_frames"],
            "quat_lab": quaternions, "diag": diagnostics}


# --------------------------------------------------------------------------- #
# step 4-6: the trial                                                          #
# --------------------------------------------------------------------------- #
REMOVE_CONSTANT_LAG = False
"""False: a well-identified constant lag between the marker plates and the sensors is measured but not removed,
since it is an error of the deposit (synchronisation offset) that the checks measure. True removes it."""


def build_trial(subject: str, task: str, root: Path, verbose: bool = False) -> dict:
    """Everything one subject-and-task needs, plus every quality number it can be judged by.

    The reference is built as ``q_reference(t) = q(Z-up<-laboratory) ⊗ q(laboratory<-plate)(t) ⊗
    q(plate<-sensor)``, where the constant ``q(plate<-sensor)`` is the least-squares rotation carrying the
    sensor's own gyroscope onto the marker cluster's angular rate. No fusion result, no model of the skeleton
    and no shipped orientation file enters it.

    Returns a dictionary with ``status`` ``"ok"`` or a reason string; on success the per-sensor reference
    quaternion, accelerometer, gyroscope and onboard orientation, the synchronisation diagnostics, the plate
    diagnostics and the trial's native sensor rate.
    """
    import qmt

    subject_dir = Path(root) / subject
    trial = trial_folder(task)
    trc_path = marker_file_path(subject_dir, trial)
    if trc_path is None:
        return {"status": "no trc", "subject": subject, "task": trial}
    sensors = read_sensors(subject_dir, trial, verbose=verbose)
    if len(sensors) < MIN_SENSORS:
        return {"status": f"only {len(sensors)} IMUs", "subject": subject, "task": trial}

    plates = plate_orientations(trc_path)
    n_mocap = plates["n_frames"]
    n_imu = min(len(d["acc"]) for d in sensors.values())
    n = min(n_mocap, n_imu)
    common = sorted(set(plates["quat_lab"]) & set(sensors))

    # --- the trial-level constant-shift correction ------------------------- #
    # A well-identified constant offset is removed rather than used to reject the trial, and only where every
    # sensor is confidently synchronised and they agree; otherwise the shift is zero and nothing changes.
    per_sensor: dict[str, tuple[int, float]] = {}
    for name in common:
        rate_marker = zero_phase_lowpass(angular_rate_central(normalise_quaternion(plates["quat_lab"][name][:n])))
        rate_gyro = zero_phase_lowpass(np.asarray(sensors[name]["gyr"], float)[:n])
        per_sensor[name] = best_shift(np.linalg.norm(rate_marker, axis=1), np.linalg.norm(rate_gyro, axis=1))
    lags = [v[0] for v in per_sensor.values()]
    correlations = [v[1] for v in per_sensor.values()]
    shift, shift_reason = 0, "not needed"
    if not REMOVE_CONSTANT_LAG:
        shift_reason = "not removed: the copy serves the timing as deposited (REMOVE_CONSTANT_LAG)"
    elif lags and max(abs(lag) for lag in lags) > TRIAL_SHIFT_TOLERANCE_SAMPLES:
        if (min(correlations) >= TRIAL_SHIFT_MIN_XCORR
                and (max(lags) - min(lags)) <= TRIAL_SHIFT_MAX_SPREAD_SAMPLES):
            shift = int(np.median(lags))
            shift_reason = (f"constant lag {shift} removed (min xcorr {min(correlations):.4f}, spread "
                            f"{max(lags) - min(lags)})")
        else:
            shift_reason = (f"NOT corrected: lags {min(lags)}..{max(lags)}, "
                            f"min xcorr {min(correlations):.4f}")

    out_quat, out_acc, out_gyr, out_onboard, sync = {}, {}, {}, {}, {}
    gravity = []
    n = n - abs(shift)
    for name in common:
        q_lab_full = normalise_quaternion(plates["quat_lab"][name])
        acc_full = np.asarray(sensors[name]["acc"], float)
        gyr_full = np.asarray(sensors[name]["gyr"], float)
        onboard_full = np.asarray(sensors[name]["quat_onboard"], float)
        # best_shift maximises corr(marker[t + k], sensor[t]), so marker[k:] pairs with sensor[:]
        if shift >= 0:
            q_lab, sensor_slice = q_lab_full[shift:shift + n], slice(0, n)
        else:
            q_lab, sensor_slice = q_lab_full[:n], slice(-shift, -shift + n)
        acc, gyr, onboard = acc_full[sensor_slice], gyr_full[sensor_slice], onboard_full[sensor_slice]

        rate_marker = angular_rate_central(q_lab)  # in the plate frame
        marker_filtered, gyro_filtered = zero_phase_lowpass(rate_marker), zero_phase_lowpass(gyr)

        # --- synchronisation: |ω| of the marker cluster against |ω| of its own sensor --- #
        lag, correlation = best_shift(np.linalg.norm(marker_filtered, axis=1),
                                      np.linalg.norm(gyro_filtered, axis=1))

        # --- the constant plate<-sensor rotation, from angular rate alone --- #
        supported = np.isfinite(marker_filtered).all(axis=1) & np.isfinite(gyro_filtered).all(axis=1)
        excited = supported & (np.linalg.norm(gyro_filtered, axis=1) > ALIGNMENT_MIN_RATE_RAD_S)
        if excited.sum() < ALIGNMENT_MIN_FRAMES:
            excited = supported & (np.linalg.norm(gyro_filtered, axis=1) > ALIGNMENT_FALLBACK_RATE_RAD_S)
        if not excited.any():
            raise ValueError(f"{subject}/{trial}/{name}: no supported excited optical/gyro alignment rows")
        rotation = wahba(marker_filtered[excited], gyro_filtered[excited])  # plate <- sensor
        residual_after = angle_between_deg(marker_filtered[excited], gyro_filtered[excited] @ rotation.T)
        residual_before = angle_between_deg(marker_filtered[excited], gyro_filtered[excited])
        q_plate_from_sensor = normalise_quaternion(
            marker_reader.rotation_matrix_to_quaternion(rotation[None]))[0]

        q_reference = qmt.qmult(qmt.qmult(np.tile(ZUP_FROM_LAB_QUATERNION, (n, 1)), q_lab),
                                np.tile(q_plate_from_sensor, (n, 1)))  # Z-up <- sensor
        q_lab_sensor = qmt.qmult(q_lab, np.tile(q_plate_from_sensor, (n, 1)))
        world_acc = qmt.rotate(q_lab_sensor, acc)
        gravity_valid = np.isfinite(world_acc).all(axis=1)
        gravity.append(np.mean(world_acc if gravity_valid.all() else world_acc[gravity_valid], axis=0))

        # --- quiet-frame check: the reference's vertical against the accelerometer's --- #
        magnitude = np.linalg.norm(acc, axis=1)
        quiet = ((np.abs(magnitude - GRAVITY_M_S2) < QUIET_ACCELERATION_TOLERANCE)
                 & (np.linalg.norm(gyr, axis=1) < QUIET_RATE_RAD_S)
                 & np.isfinite(q_reference).all(axis=1))
        tilt_median = tilt_p95 = float("nan")
        if quiet.sum() > QUIET_MIN_FRAMES:
            k = int(quiet.sum())
            vertical = qmt.rotate(qmt.qinv(q_reference[quiet]),
                                  np.tile(np.array([0.0, 0.0, 1.0]), (k, 1)))
            deviation = angle_between_deg(vertical, acc[quiet] / magnitude[quiet, None])
            tilt_median = float(np.median(deviation))
            tilt_p95 = float(np.percentile(deviation, REPORTED_PERCENTILE))

        out_quat[name], out_acc[name] = q_reference, acc
        out_gyr[name], out_onboard[name] = gyr, onboard
        sync[name] = {
            "lag_samples": lag,
            "xcorr": correlation,
            "align_residual_sto_rad": float(sensors[name]["align_residual"]),
            "imu_sto_lag": int(sensors[name]["lag"]),
            "plate_from_sensor_quat": q_plate_from_sensor.tolist(),
            "plate_from_sensor_angle_deg": float(np.rad2deg(
                2 * np.arccos(np.clip(abs(q_plate_from_sensor[0]), 0, 1)))),
            "omega_axis_resid_deg_median": float(np.median(residual_after)),
            "omega_axis_resid_deg_p95": float(np.percentile(residual_after, REPORTED_PERCENTILE)),
            "omega_axis_before_align_deg_median": float(np.median(residual_before)),
            "n_align_frames": int(excited.sum()),
            "quiet_frames": int(quiet.sum()),
            "gt_vs_acc_tilt_deg_quiet_median": tilt_median,
            "gt_vs_acc_tilt_deg_quiet_p95": tilt_p95,
        }
    return {"status": "ok", "subject": subject, "task": trial, "n": int(n),
            "n_mocap": int(n_mocap), "n_imu": int(n_imu),
            "recovery": {"trial_shift_samples": int(shift),
                         "reason": shift_reason,
                         "pre_lags": {k: int(v[0]) for k, v in per_sensor.items()},
                         "pre_xcorr": {k: float(v[1]) for k, v in per_sensor.items()},
                         "native_rate": native_rate_of(sensors)},
            "quat": out_quat, "acc": out_acc, "gyr": out_gyr,
            "onboard": out_onboard, "sync": sync, "plate_diag": plates["diag"],
            "gravity_lab_measured": np.mean(gravity, axis=0).tolist(),
            "imus": common}


def onboard_consistency_check(trial: dict) -> dict:
    """An **external** check that is no part of the reconstruction: the plate-to-plate relative orientation
    against the sensors' own onboard fusion, which the scored path never sees.

    Reported twice per joint: raw, and after removing the one constant rotation between the two (the mean of
    the residual), because a constant offset between the plate frames and the fusion's frames is expected and
    says nothing about the reference's quality over time.
    """
    import qmt

    out = {}
    for label, parent_body, child_body in JOINT_BODIES:
        parent, child = f"{parent_body}_imu", f"{child_body}_imu"
        if parent not in trial["quat"] or child not in trial["quat"]:
            continue
        q_reference = qmt.qrel(trial["quat"][parent], trial["quat"][child])
        q_onboard = qmt.qrel(trial["onboard"][parent], trial["onboard"][child])
        valid = np.isfinite(q_reference).all(axis=1) & np.isfinite(q_onboard).all(axis=1)
        if not valid.any():
            out[label] = {"raw_median_deg": float("nan"), "offset_removed_median_deg": float("nan")}
            continue
        if not valid.all():
            q_reference, q_onboard = q_reference[valid], q_onboard[valid]
        raw = np.rad2deg(np.abs(qmt.quatAngle(qmt.qrel(q_reference, q_onboard))))
        residual = qmt.qrel(q_reference, q_onboard)
        mean_rotation = normalise_quaternion(
            np.mean(residual * np.sign(residual[:, :1] + SIGN_GUARD), axis=0))
        without_offset = np.rad2deg(np.abs(qmt.quatAngle(
            qmt.qmult(residual, qmt.qinv(np.tile(mean_rotation, (len(residual), 1)))))))
        out[label] = {"raw_median_deg": float(np.median(raw)),
                      "offset_removed_median_deg": float(np.median(without_offset))}
    return out


def diagnostics_of(trial: dict) -> dict:
    """The trial without its arrays: exactly what the reconstruction writes beside each cached trial."""
    diagnostics = {k: v for k, v in trial.items() if k not in ("quat", "acc", "gyr", "onboard")}
    diagnostics["external_xsens_check"] = onboard_consistency_check(trial)
    return diagnostics
