"""Error 18 (mounting rotation): unknown or wrong mounting of the IMU on the segment (misalignment between sensor and
reference frame).

What it is: the IMU's axes are not aligned with the axes of the optical segment frame, and the deposit does not say
(or says wrongly) how they are rotated against each other. Relative orientation then mixes the joint's motion with the
unknown mounting rotation.

How we catch it: the gyroscope measures the angular rate in the sensor frame, the reference gives the angular rate in
the segment frame. If the IMU is rigidly mounted, one constant rotation maps one onto the other; the best such
rotation over the moving samples (least squares, Kabsch) is the misalignment. The fit also tells how the reference
stores its orientation (segment-to-world or the reverse) and how well a rotation explains the data at all.

In the order of correction: one rotation per trial and segment, fitted on the trial's first 10 s of motion (the
estimators' unscored convergence time), for every dataset (check_for_mounting_in_convergence); the rotation applied is
the certainty-weighted mean of that fit and the participant's other trials (comments below).

How we correct it: rotate the gyroscope and accelerometer samples by the applied rotation, so the IMU data are
expressed in the reference segment frame. Needs synchronised streams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..recording import Recording, Segment
from ..rotations import best_rotation, interpolate, reference_angular_rate
from ._result import CheckResult, entry
from ._session import as_session, segments_by_name
from .e03_reference_jumps import implausible_reference_frames

ERROR = 'mounting misalignment'


@dataclass
class GyroscopeFit:
    """Best constant maps from the gyroscope to the reference angular rate of one segment."""

    sensor_to_segment: np.ndarray  # 3x3 rotation, reference rate ≈ M @ gyroscope
    residual_dps: float  # rms misfit of the best rotation
    mirrored_residual_dps: float  # rms misfit of the best rotation-plus-mirror
    rate_spread_dps: float  # rms of the reference rate itself, for scale
    convention: str  # how the reference orientation is stored, as the fit suggests
    excitation: float  # smallest / largest spread of the rate over the three axes; near 0 = mostly one axis
    samples: int


def rate_pairs(segment: Segment, convention: str, min_rate_dps: float = 17.0, window_s: float | None = None,
               window_start_s: float | None = None, with_times: bool = False):
    """Gyroscope and reference angular rate at the same instants, for the moving, valid and plausible frames.

    window_s: only window_s seconds from window_start_s (default: the start of the time both streams cover; None:
    all of it). with_times: also return the samples' times (s)."""
    imu, ref = segment.imu, segment.reference
    excluded = implausible_reference_frames(segment, margin_s=0.5)
    t, w = reference_angular_rate(ref.orientation, ref.rate_hz, convention)
    t = t + ref.start_s
    g = interpolate(t, imu.times, np.where(imu.valid[:, None], imu.gyr, np.nan))
    use = (np.isfinite(g).all(1) & ~excluded[:-1] & ~excluded[1:]
           & (np.linalg.norm(w, axis=1) > np.radians(min_rate_dps)))
    if window_s is not None:
        start = common_start(segment) if window_start_s is None else window_start_s
        use &= (t >= start) & (t < start + window_s)
    return (g[use], w[use], t[use]) if with_times else (g[use], w[use])


def common_start(segment: Segment) -> float:
    """First instant covered by both the IMU and the reference."""
    return max(float(segment.imu.times[0]), float(segment.reference.start_s))


def fit_gyroscope_to_reference(segments: Segment | list[Segment], min_rate_dps: float = 17.0,
                               window_s: float | None = None, min_samples: int = 20,
                               window_start_s: float | None = None) -> GyroscopeFit | None:
    """Fit one constant rotation from the gyroscope to the reference angular rate, pooled over all given segments
    (the same body segment in several trials of one session).

    If the reference's storage convention is unknown, both are tried and the one whose best map (rotation or
    mirror) fits better is kept. Returns None if fewer than 20 usable samples exist."""
    segments = [segments] if isinstance(segments, Segment) else segments
    known = segments[0].reference.convention
    conventions = ('segment_to_world', 'world_to_segment') if known == 'unknown' else (known,)
    best = None
    for convention in conventions:
        pairs = [rate_pairs(s, convention, min_rate_dps, window_s, window_start_s) for s in segments]
        g = np.vstack([p[0] for p in pairs])
        w = np.vstack([p[1] for p in pairs])
        if len(g) < max(min_samples, 3):
            return None
        m, res = best_rotation(g, w, proper=True)
        _, res_mirror = best_rotation(g, w, proper=False)
        # the convention is chosen by the better of the two maps, so a mirrored gyroscope cannot hide behind it
        if best is None or min(res, res_mirror) < min(best.residual_dps, best.mirrored_residual_dps):
            centred = w - w.mean(0)
            spread = np.linalg.svd(centred, compute_uv=False)
            best = GyroscopeFit(m, res, res_mirror, float(np.sqrt((centred ** 2).sum(1).mean())), convention,
                                float(spread[2] / spread[0]), len(g))
    best.residual_dps, best.mirrored_residual_dps, best.rate_spread_dps = (
        float(np.degrees(v)) for v in (best.residual_dps, best.mirrored_residual_dps, best.rate_spread_dps))
    return best


def check_for_mounting_misalignment(recordings: Recording | list[Recording], threshold_deg: float = 2.0) -> CheckResult:
    """Angle between each IMU's axes and its reference segment frame, fitted over all given recordings of one
    session. Detected if any segment exceeds `threshold_deg` (exploratory default)."""
    per_segment, angles = {}, []
    for name, segs in segments_by_name(as_session(recordings)).items():
        fit = fit_gyroscope_to_reference(segs)
        if fit is None:
            per_segment[name] = entry(None, float('nan'), 'deg', fit=None)
            continue
        angle = float(np.degrees(Rotation.from_matrix(fit.sensor_to_segment).magnitude()))
        per_segment[name] = entry(angle > threshold_deg, angle, 'deg', fit.residual_dps, fit=fit)
        angles.append(angle)
    thresholds = dict(threshold_deg=threshold_deg)
    if not angles:
        return CheckResult(ERROR, None, 'Too little motion to fit the mounting in any segment.', per_segment,
                           thresholds)
    detected = max(angles) > threshold_deg
    summary = (f'IMU axes differ from the reference segment frames by {min(angles):.1f}° to {max(angles):.1f}° '
               f'({len(angles)} segments).')
    return CheckResult(ERROR, detected, summary, per_segment, thresholds)


# Per-trial fit in the convergence window: every trial gets its own constant rotation from the gyroscope to the
# reference, fitted on its convergence window, the first CONVERGENCE_S seconds of motion from the motion onset
# (motion_onset.py). Those seconds are never scored (the estimators converge there), so the scored part does not set
# the rotation. No dataset is exempt, also not those whose markers sit on the IMU: a cluster on the housing moves with
# the sensor but its axes are not the sensor's axes. A static trial (no onset) gets no rotation.
#
# Uncertainty and weighted mean: every window fit carries its uncertainty as an information matrix (fit_information),
# and the rotation applied is the certainty-weighted mean of the trial's own fit and the pooled rotation of the
# participant's other trials of that segment (pooled_rotations, combine_with_pool). The weighting is per direction: a
# segment turning about one axis contributes its well-determined axes, and the pool fills in the rotation about that
# axis. The pool's uncertainty holds the noise of its fits plus the real scatter between trials, so where a
# participant's trials truly differ (a slipped sensor), a precise own fit keeps its weight.
CONVERGENCE_S = 10.0  # = tuning_split.CONVERGENCE_S
MIN_RATE_DPS = 17.0  # reference rate a sample needs to enter a fit (the default of every fit below)
MIN_FIT = 3  # fewest samples a rotation can be fitted from at all

# Uncertainty of a fit. Noise correlation: neighbouring samples drift together (skin wobble, timing, smoothing), so
# the noise is estimated with Newey-West weights over NOISE_LAG_S; synthetic windows with known rotation keep the true
# error inside the 95 % uncertainty in 94-99 % for assumed lags from 0.25 to 2 s, where independent samples would give
# 9-43 %. NOISE_LAG_S = 0.25 s is the measured memory of the misfit (the noise of block sums peaks at 0.05-0.25 s in
# almost every dataset). Rest floor: never less noise than white noise at the reference's jitter at rest (REST_DPS,
# = the gyroscope offset's MAX_REF_DPS). Short stretches: with few independent pieces (moving time / lag) the noise
# estimate itself is unreliable, so the covariance is widened by the small-sample factor
# 3 F(0.95; 3, m) / chi2(0.95; 3), m = moving time / NOISE_LAG_S (synthetic stretches of 0.4-3 s: 12-88 % coverage
# without it, 99-100 % with it).
NOISE_LAG_S = 0.25
REST_DPS = 5.6
UNCERTAINTY_LEVEL = 0.95


def _hat(m: np.ndarray, g: np.ndarray, w: np.ndarray):
    """Centred rotated gyroscope v and misfit e of the rotation m (w ~ m g), as in rotations.best_rotation."""
    v = (g - g.mean(0)) @ m.T
    return v, (w - w.mean(0)) - v


def fit_information(m: np.ndarray, g: np.ndarray, w: np.ndarray, t: np.ndarray, rate_hz: float,
                    lag_s: float = NOISE_LAG_S) -> np.ndarray:
    """Information matrix (rad^-2, 3x3) of the rotation m fitted from g to w (w ~ m g) at times t: the inverse
    covariance of a small turn exp(d) m of the fit (d in the segment frame) that the data cannot rule out. Comment
    above."""
    from scipy.stats import chi2, f
    v, e = _hat(m, g, w)
    h = (v ** 2).sum() * np.eye(3) - v.T @ v
    score = np.zeros((int(np.rint((t[-1] - t[0]) * rate_hz)) + 1, 3))
    score[np.rint((t - t[0]) * rate_hz).astype(int)] = np.cross(v, e)
    lag = max(int(round(lag_s * rate_hz)), 0)
    c = score.T @ score
    for k in range(1, min(lag, len(score) - 1) + 1):
        gk = score[:-k].T @ score[k:]
        c += (1 - k / (lag + 1)) * (gk + gk.T)
    floor = np.radians(REST_DPS) ** 2 / 3 * h
    if np.trace(floor) > np.trace(c):
        c = floor
    pieces = len(g) / rate_hz / lag_s
    widen = 3 * f.ppf(UNCERTAINTY_LEVEL, 3, max(pieces, 1e-3)) / chi2.ppf(UNCERTAINTY_LEVEL, 3)
    info = h @ np.linalg.pinv(c) @ h / widen
    return (info + info.T) / 2


def uncertainty_deg(information: np.ndarray) -> float:
    """Largest angle (degrees) by which the true rotation may differ at UNCERTAINTY_LEVEL: the longest half-axis of
    the confidence region, capped at 180."""
    from scipy.stats import chi2
    low = float(np.linalg.eigvalsh(information)[0])
    if low <= 0:
        return 180.0
    return float(min(180.0, np.degrees(np.sqrt(chi2.ppf(UNCERTAINTY_LEVEL, 3) / low))))


def window_fit(segment: Segment, start_s: float | None, window_s: float = CONVERGENCE_S, threshold_deg: float = 2.0,
               min_rate_dps: float = 17.0) -> dict:
    """One segment's rotation fitted on `window_s` seconds from `start_s` (an entry of
    check_for_mounting_in_convergence's per_segment); no fit (detected None) with fewer than MIN_FIT moving samples or
    without a start."""
    fit = None if start_s is None else fit_gyroscope_to_reference(segment, min_rate_dps, window_s=window_s,
                                                                     min_samples=MIN_FIT, window_start_s=start_s)
    if fit is None:
        return entry(None, float('nan'), 'deg', fit=None, moving_samples=0, information=np.zeros((3, 3)),
                     uncertainty_deg=180.0)
    g, w, t = rate_pairs(segment, fit.convention, min_rate_dps, window_s, start_s, with_times=True)
    info = fit_information(fit.sensor_to_segment, g, w, t, segment.reference.rate_hz)
    angle = float(np.degrees(Rotation.from_matrix(fit.sensor_to_segment).magnitude()))
    return entry(angle > threshold_deg, angle, 'deg', fit.residual_dps, fit=fit, moving_samples=fit.samples,
                 excitation=fit.excitation, information=info, uncertainty_deg=uncertainty_deg(info))


def check_for_mounting_in_convergence(recording: Recording, window_s: float = CONVERGENCE_S,
                                      threshold_deg: float = 2.0, min_rate_dps: float = 17.0,
                                      onset_s: float | None = None) -> CheckResult:
    """One rotation per segment from this trial's convergence window: `window_s` seconds from the motion onset
    (`onset_s`, computed from the reference when None; comment above). Per segment: the angle, the rms misfit, the
    moving samples used, the information matrix of the fit and its uncertainty in degrees (uncertainty_deg). A
    segment with fewer than MIN_FIT moving samples, or every segment of a static trial, gets no rotation (detected
    None)."""
    from ..motion_onset import motion_onset_s  # here, since motion_onset imports from this package
    onset = motion_onset_s(recording) if onset_s is None else onset_s
    per_segment, angles, unfitted = {}, [], []
    for name, seg in recording.segments.items():
        per_segment[name] = window_fit(seg, onset, window_s, threshold_deg, min_rate_dps)
        if per_segment[name]['fit'] is None:
            unfitted.append(name)
        else:
            angles.append(per_segment[name]['estimate'])
    thresholds = dict(window_s=window_s, threshold_deg=threshold_deg, min_rate_dps=min_rate_dps, min_fit=MIN_FIT,
                      noise_lag_s=NOISE_LAG_S, rest_dps=REST_DPS, onset_s=onset)
    if onset is None:
        return CheckResult(ERROR, None, 'Static trial: the reference never exceeds the standing limit, no rotation '
                           'fitted.', per_segment, thresholds)
    flag = f' Too little movement to fit in the {window_s:g} s from motion onset: {", ".join(unfitted)}.' if unfitted else ''
    if not angles:
        return CheckResult(ERROR, None, f'No segment moves enough in the {window_s:g} s from motion onset to fit a '
                           'rotation.' + flag, per_segment, thresholds)
    summary = (f'IMU axes differ from the reference segment frames by {min(angles):.1f}° to {max(angles):.1f}° '
               f'({len(angles)} segments, {window_s:g} s from motion onset at {onset:.1f} s).' + flag)
    return CheckResult(ERROR, max(angles) > threshold_deg, summary, per_segment, thresholds)


def _pool_weights(infos: np.ndarray, tau2: float) -> np.ndarray:
    """Weights (K, 3, 3) of fits with information `infos` once each trial also scatters by tau2 (rad^2) per axis:
    (J^-1 + tau2 I)^-1 = J (I + tau2 J)^-1, without inverting a singular J."""
    eye = np.eye(3)
    return np.stack([j @ np.linalg.inv(eye + tau2 * j) for j in infos])


def _weighted_mean(rots: Rotation, weights: np.ndarray, start: Rotation | None = None, iterations: int = 20):
    """Rotation mean with per-fit 3x3 weights, on the tangent space of the current mean (segment frame, left)."""
    mean = rots.mean() if start is None else start
    total = weights.sum(0) + 1e-12 * np.eye(3)
    for _ in range(iterations):
        x = (rots * mean.inv()).as_rotvec()
        step = np.linalg.solve(total, np.einsum('kij,kj->i', weights, x))
        mean = Rotation.from_rotvec(step) * mean
        if np.linalg.norm(step) < 1e-10:
            break
    return mean, (rots * mean.inv()).as_rotvec()


def trial_scatter(rots: Rotation, infos: np.ndarray) -> float:
    """Scatter between trials (rad^2 per axis) beyond the fits' own noise, by the Paule-Mandel rule: the tau2 at
    which the weighted spread of the fits about their mean equals its expectation, 3 (K - 1); 0 if the noise alone
    explains the spread or K < 2."""
    k = len(rots)
    if k < 2:
        return 0.0

    def spread(tau2):
        wts = _pool_weights(infos, tau2)
        _, x = _weighted_mean(rots, wts)
        return float(np.einsum('ki,kij,kj->', x, wts, x))

    target = 3 * (k - 1)
    if spread(0.0) <= target:
        return 0.0
    lo, hi = 0.0, np.pi ** 2
    if spread(hi) > target:
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spread(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def pooled_rotations(rows) -> dict:
    """Leave-one-out pools per (participant, segment) from the window fits of its trials (rows: dicts with
    participant, segment, trial, quaternion xyzw and information, 3x3). For each trial: the certainty-weighted mean of
    the participant's other fits of that segment and its information, which includes the scatter between trials
    (trial_scatter, estimated from those other fits only, so a useless own fit cannot inflate the scatter and keep
    its own weight).
    With a single other fit the scatter is 0. Returns {(participant, segment): dict(tau_deg, n,
    per_trial={trial: (Rotation, information)}, all=(Rotation, information))}: `all` pools every fit, for a trial
    without a fit of its own; a group with a single fit has no pool for that fit."""
    groups = {}
    for r in rows:
        groups.setdefault((str(r['participant']), r['segment']), []).append(r)
    eye = np.eye(3)
    out = {}
    for key, rs in groups.items():
        rots = Rotation.from_quat(np.asarray([r['quaternion'] for r in rs], float))
        infos = np.asarray([r['information'] for r in rs], float)
        tau2 = trial_scatter(rots, infos)
        wts = _pool_weights(infos, tau2)
        full, _ = _weighted_mean(rots, wts)
        per_trial = {}
        for i, r in enumerate(rs):
            keep = np.arange(len(rs)) != i
            if not keep.any():
                continue
            tau2_i = trial_scatter(rots[keep], infos[keep])
            w_i = _pool_weights(infos[keep], tau2_i)
            mean, _ = _weighted_mean(rots[keep], w_i, start=full)
            s = w_i.sum(0)
            per_trial[r['trial']] = (mean, s @ np.linalg.inv(eye + tau2_i * s))
        s_all = wts.sum(0)
        out[key] = dict(tau_deg=float(np.degrees(np.sqrt(tau2))), n=len(rs), per_trial=per_trial,
                        all=(full, s_all @ np.linalg.inv(eye + tau2 * s_all)))
    return out


def combine_with_pool(own: np.ndarray | None, own_information: np.ndarray, pool: Rotation | None,
                      pool_information: np.ndarray | None):
    """The applied rotation: the certainty-weighted mean of the trial's own fit (sensor_to_segment matrix, or None)
    and the pool's rotation, per direction (comment above). Returns (matrix, information of the result, own_share),
    own_share the own fit's share of the total information (trace), or (None, None, None) without either."""
    if pool is None:
        if own is None:
            return None, None, None
        return own, own_information, 1.0
    if own is None:
        return pool.as_matrix(), pool_information, 0.0
    total = own_information + pool_information
    x = (Rotation.from_matrix(own) * pool.inv()).as_rotvec()
    step = np.linalg.lstsq(total, own_information @ x, rcond=None)[0]
    share = float(np.trace(own_information) / max(np.trace(total), 1e-300))
    return (Rotation.from_rotvec(step) * pool).as_matrix(), total, share


def remove_mounting_misalignment(recording: Recording, result: CheckResult | None = None) -> Recording:
    """Return a copy with every IMU's gyroscope and accelerometer rotated into its reference segment frame.

    `result` may come from a session-level check; its fitted rotations are applied to this recording."""
    if result is None:
        result = check_for_mounting_misalignment(recording)
    new = recording.copy()
    rotated = {}
    for name, seg in new.segments.items():
        fit = result.per_segment.get(name, {}).get('fit')
        if fit is None:
            continue
        seg.imu.gyr = seg.imu.gyr @ fit.sensor_to_segment.T
        seg.imu.acc = seg.imu.acc @ fit.sensor_to_segment.T
        rotated[name] = result.per_segment[name]['estimate']
    new.log_correction(ERROR, f'Rotated the IMU data of {len(rotated)} segments into the reference segment frames.',
                       angle_deg=rotated)
    return new


# Rotation applied per trial and segment. The rotation is the certainty-weighted mean of the trial's own window fit and
# the leave-one-out pool of the participant's other trials of that segment (the pool's uncertainty includes the
# scatter between trials). A segment without an own fit takes the pool of every fitted trial; one with neither is not
# corrected. Own-window rule: a segment without a pool that moves in at most half of the trial's window is fitted on
# its own first CONVERGENCE_S of motion (moving meaning faster than the fit's MIN_RATE_DPS, so the fit has samples);
# its joints are scored from the end of that window (metadata[SEGMENT_WINDOWS_KEY]).

INFORMATION_COLUMNS = ('j_xx', 'j_xy', 'j_xz', 'j_yy', 'j_yz', 'j_zz')  # upper triangle of a fit's information


def _fit_columns(prefix: str, p: dict) -> dict:
    """Quaternion (xyzw), information (upper triangle) and uncertainty of a window_fit entry, as row columns."""
    fit = p.get('fit')
    q = Rotation.from_matrix(fit.sensor_to_segment).as_quat() if fit is not None else [np.nan] * 4
    info = p.get('information', np.zeros((3, 3)))
    upper = [info[0, 0], info[0, 1], info[0, 2], info[1, 1], info[1, 2], info[2, 2]]
    out = {f'{prefix}q{a}': float(v) for a, v in zip('xyzw', q)}
    out.update({f'{prefix}{c}': float(v) for c, v in zip(INFORMATION_COLUMNS, upper)})
    out[f'{prefix}uncertainty_deg'] = float(p.get('uncertainty_deg', 180.0))
    out[f'{prefix}moving_samples'] = int(p.get('moving_samples') or 0)
    return out


def _information(r: dict, prefix: str) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = (float(r.get(f'{prefix}{c}', 0.0) or 0.0) for c in INFORMATION_COLUMNS)
    return np.nan_to_num(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], float))


def _quaternion(r: dict, prefix: str) -> np.ndarray | None:
    q = np.array([float(r.get(f'{prefix}q{a}', np.nan)) for a in 'xyzw'])
    return q if np.isfinite(q).all() else None


def window_fits(rec: Recording) -> list[dict]:
    """Per segment: the fit on the trial's convergence window and, for a segment that moves in at most half of that
    window, the alternative fit on its own convergence window (used only where the segment has no pool)."""
    from ..motion_onset import MAJORITY, convergence_window, segment_convergence_window, segment_motion_share
    window = convergence_window(rec)
    onset = window['onset_s']
    res = check_for_mounting_in_convergence(rec, onset_s=onset)
    rows = []
    for seg, p in res.per_segment.items():
        row = dict(segment=seg, onset_s=onset, window_share=window['share'], share_in_trial_window=np.nan,
                   alt_start_s=np.nan, **_fit_columns('own_', p))
        if onset is not None:
            share = segment_motion_share(rec, seg, onset)
            row['share_in_trial_window'] = share
            if share <= MAJORITY:
                w = segment_convergence_window(rec, seg, max_ref_dps=MIN_RATE_DPS)
                if w['onset_s'] is not None and w['onset_s'] != onset:
                    row.update(alt_start_s=w['onset_s'], alt_share=w['share'], alt_continuous=w['continuous'],
                               **_fit_columns('alt_', window_fit(rec[seg], w['onset_s'])))
        rows.append(row)
    return rows


def pools_from_rows(rows: list[dict]) -> dict:
    """Leave-one-out pools (pooled_rotations) from the own window fits of rows (participant, segment, trial)."""
    fitted = [dict(participant=str(r['participant']), segment=r['segment'], trial=str(r['trial']),
                   quaternion=_quaternion(r, 'own_'), information=_information(r, 'own_'))
              for r in rows if _quaternion(r, 'own_') is not None]
    return pooled_rotations(fitted)


def plan_rotation(r: dict, pools: dict) -> dict:
    """The rotation applied to one trial-segment (comment above), from its window_fits row and the pools: applied
    quaternion (a_qx..a_qw, NaN where not corrected), own share, uncertainties, pool size and scatter, own window."""
    from ..motion_onset import MAJORITY
    group = pools.get((str(r['participant']), r['segment']))
    trial = str(r['trial'])
    own_q, own_info = _quaternion(r, 'own_'), _information(r, 'own_')
    own_unc = float(r.get('own_uncertainty_deg', 180.0))
    has_pool = bool(group) and (trial in group['per_trial'] or own_q is None)
    onset, share = float(r.get('onset_s') if r.get('onset_s') is not None else np.nan), float(
        r.get('share_in_trial_window', np.nan))
    own_window = (not has_pool and np.isfinite(onset) and np.isfinite(share) and share <= MAJORITY
                  and np.isfinite(float(r.get('alt_start_s', np.nan))))
    if own_window:
        own_q, own_info = _quaternion(r, 'alt_'), _information(r, 'alt_')
        own_unc = float(r.get('alt_uncertainty_deg', 180.0))
    own = Rotation.from_quat(own_q).as_matrix() if own_q is not None else None
    pool, pool_info = None, None
    if group and not own_window:  # own fit: the other trials; none: every fitted trial of the group
        pool, pool_info = group['per_trial'].get(trial, (None, None)) if own is not None else group['all']
    m, info, own_share = combine_with_pool(own, own_info, pool, pool_info)
    out = dict(own_window=bool(own_window), corrected=m is not None,
               own_window_start_s=float(r['alt_start_s']) if own_window else np.nan,
               own_window_end_s=float(r['alt_start_s']) + CONVERGENCE_S if own_window else np.nan,
               own_share=own_share if own_share is not None else np.nan,
               pool_trials=(group['n'] - (own is not None)) if pool is not None else 0,
               pool_scatter_deg=group['tau_deg'] if group else np.nan, plan_own_uncertainty_deg=own_unc)
    q = Rotation.from_matrix(m).as_quat() if m is not None else [np.nan] * 4
    out.update({f'a_q{a}': float(v) for a, v in zip('xyzw', q)})
    out['applied_uncertainty_deg'] = uncertainty_deg(info) if m is not None else np.nan
    out['angle_deg'] = float(np.degrees(Rotation.from_matrix(m).magnitude())) if m is not None else np.nan
    return out


def apply_rotations(rec: Recording, rows: list[dict]) -> Recording:
    """Copy with each segment's gyroscope and accelerometer rotated by its applied rotation (rows of plan_rotation);
    segments fitted on their own window are recorded in metadata[SEGMENT_WINDOWS_KEY] {segment: (start, end)}."""
    from ..motion_onset import SEGMENT_WINDOWS_KEY
    new = rec.copy()
    rotated, weights, uncorrected, own_window = {}, {}, [], {}
    onset = None
    for r in rows:
        seg = r['segment']
        if seg not in new.segments:
            continue
        onset = r.get('onset_s') if onset is None else onset
        q = _quaternion(r, 'a_')
        if q is None:
            uncorrected.append(seg)
            continue
        m = Rotation.from_quat(q).as_matrix()
        new.segments[seg].imu.gyr = new.segments[seg].imu.gyr @ m.T
        new.segments[seg].imu.acc = new.segments[seg].imu.acc @ m.T
        rotated[seg] = float(r.get('angle_deg', np.nan))
        weights[seg] = dict(own_uncertainty_deg=round(float(r.get('plan_own_uncertainty_deg', 180.0)), 2),
                            own_share=round(float(r.get('own_share', np.nan)), 3),
                            applied_uncertainty_deg=round(float(r.get('applied_uncertainty_deg', np.nan)), 2),
                            pool_trials=int(float(r.get('pool_trials', 0) or 0)),
                            pool_scatter_deg=float(r.get('pool_scatter_deg', np.nan)))
        if str(r.get('own_window')) == 'True':
            own_window[seg] = dict(start_s=float(r['own_window_start_s']), end_s=float(r['own_window_end_s']),
                                   share_in_trial_window=round(float(r.get('share_in_trial_window', np.nan)), 3))
    new.log_correction(ERROR, f'Rotated the IMU data of {len(rotated)} segments into the reference segment frames.',
                       angle_deg=rotated, weights=weights, not_corrected=sorted(uncorrected), motion_onset_s=onset,
                       own_window=own_window)
    if own_window:
        new.metadata[SEGMENT_WINDOWS_KEY] = {k: (v['start_s'], v['end_s']) for k, v in own_window.items()}
    return new


def mount_trial(rec: Recording, pools: dict | None = None, participant=None, trial: str | None = None) -> Recording:
    """One trial rotated into its reference segment frames, with the given leave-one-out pools (pooled_rotations;
    none: own fits only)."""
    participant = rec.participant if participant is None else participant
    trial = rec.trial if trial is None else trial
    rows = [dict(r, participant=participant, trial=trial) for r in window_fits(rec)]
    return apply_rotations(rec, [dict(r, **plan_rotation(r, pools or {})) for r in rows])


# Dataset level: one row per trial and segment with its own window fit, the plan built from the participant's other
# trials and the quaternion applied. Counted: trial-segments whose applied rotation exceeds 2 degrees (the check's
# threshold); segments without any rotation are undecided.

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected as _corrected_loader, frame, per_trial, prevalence_of, run_trials  # noqa: E402

THRESHOLD_DEG = 2.0


def _fits_of_trial(loader, trial):
    rec = loader.load(trial)
    participant = str(loader.participant_of(trial))
    return [dict(trial=trial, participant=participant, error='', **r) for r in window_fits(rec)]


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """Window fit of every trial and segment, pools per participant and segment, and the rotation to apply."""
    records = run_trials(loader, _fits_of_trial, processes)
    ok = [r for r in records if not r.get('error')]
    pools = pools_from_rows(ok)
    for r in ok:
        r.update(plan_rotation(r, pools))
        r.update(detected=bool(r['angle_deg'] > THRESHOLD_DEG) if r['corrected'] else None,
                 estimate=r['angle_deg'], unit='deg')
    return frame(records, ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'error'])


def prevalence(rows) -> dict:
    """Trials / trial-segments whose applied mounting rotation exceeds 2 degrees; uncorrected segments undecided."""
    return prevalence_of(rows, 'trial-segments')


def correct(loader, rows):
    """Every segment's IMU rotated into its reference segment frame by the planned rotation."""
    return _corrected_loader(loader, apply_rotations, per_trial(rows), 'mounting')


SPEC = ErrorSpec(18, 'mounting', 'fix', 'mounting rotation between sensor and reference segment frame')
PARTS = {18: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_mounting = make_handler(PARTS[18])
