"""Error 17: gyroscope zero-rate offset not removed (gyroscope bias).

What it is: a gyroscope at rest reads a small constant rate instead of zero. Integrated, it makes the orientation
drift. Calibrated units read about 0.2-0.6 °/s, uncalibrated ones in the deposits up to 27 °/s.

Place in the order: after the mirrored gyroscope (error 16) and before the mounting (error 18). At rest the true rate
is zero in every frame, so the mean of each raw axis is its offset, whatever the axis signs, the mounting or the
reference.

How we catch it:
1. Rest. A window of WINDOW_S is at rest when the IMU is quiet AND the optical reference is still (both, so that a
   constant rotation is not taken for rest, and reference jitter alone does not make a moving sensor look still):
   - IMU quiet: the accelerometer magnitude varies by less than MAX_ACC_STD and every gyroscope axis by less than
     MAX_GYR_STD_DPS (standard deviation over the window); the spread, not the level, so an offset does not hide
     the rest.
   - Reference still: the net rotation of the reference over the window (angle between its orientation at the first
     and the last IMU sample of the window, over the time between them) is below MAX_REF_DPS, and the reference is
     valid throughout. Net rotation, not the frame-to-frame speed: marker jitter differentiated frame by frame reads
     2-9 °/s on still references, while a real constant rotation shows fully in the net angle.
   MAX_GYR_STD_DPS = 4.3 °/s: the largest gyroscope spread in the static recordings the datasets ship (2.87 °/s, a
   person standing in an N-pose) times 1.5. MAX_REF_DPS = 5.6 °/s, derived the same way from the net reference
   rotation (3.73 °/s, body sway). MAX_ACC_STD is provisional.
2. Per trial. The offset of a trial is the mean gyroscope reading over its rest samples if the trial has at least
   MIN_REST_S of rest; otherwise the rest samples of all given trials of the session are pooled (fallback). Without
   enough rest in the session either, the offset can only come from a fit against the reference rate
   (reference_fallback=True, the better of a rotation and a mirror image); the dataset-level check uses rest only.
3. Decision: an offset larger than THRESHOLD_DPS = 0.01 °/s counts: the specified bias stability of the sensors in
   these datasets is 0.003-0.006 °/s, so an offset removed at the start stays below 0.01 °/s.

Session level: pass all recordings of one session (the loader's session_of); each trial gets its own offset.

How we correct it: subtract each trial's estimated offset from every gyroscope sample of that segment.
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording, Segment
from ..rotations import best_rotation
from ._result import CheckResult, entry
from ._session import as_session
from .e15_corrupted_samples import reference_speed_at
from .e18_mounting import fit_gyroscope_to_reference, rate_pairs

ERROR = 'gyroscope offset'

# Rest limits and threshold (module description).
WINDOW_S = 0.5  # length of a rest window; windows overlap by half
MAX_ACC_STD = 0.1  # m/s^2, spread of the accelerometer magnitude within a rest window (draft value)
MAX_GYR_STD_DPS = 4.3  # °/s, spread of every gyroscope axis within a rest window: 1.5 x worst static recording
MAX_REF_DPS = 5.6  # °/s, net reference rotation over a rest window: 1.5 x worst static recording
MIN_REST_S = 1.0  # seconds of rest a trial needs for its own estimate (draft value)
THRESHOLD_DPS = 0.01  # °/s, offset size counted as an error: above the worst specified bias stability


def _window_std(x: np.ndarray, n: int, starts: np.ndarray) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    s, s2 = c[starts + n] - c[starts], c2[starts + n] - c2[starts]
    return np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0.0))


def rest_mask(segment: Segment, window_s: float = WINDOW_S, max_acc_std: float = MAX_ACC_STD,
              max_gyr_std_dps: float = MAX_GYR_STD_DPS, max_ref_dps: float = MAX_REF_DPS) -> np.ndarray:
    """True for the IMU samples inside windows where the IMU is quiet and the reference is still."""
    imu = segment.imu
    n = max(int(round(window_s * imu.rate_hz)), 2)
    still = np.zeros(len(imu), bool)
    if len(imu) < n:
        return still
    ok = imu.valid & np.isfinite(imu.gyr).all(1) & np.isfinite(imu.acc).all(1)
    gyr = np.where(ok[:, None], imu.gyr, 0.0)
    acc_norm = np.linalg.norm(np.where(ok[:, None], imu.acc, 0.0), axis=1)
    ref = reference_speed_at(segment)
    ref_ok = np.isfinite(ref)
    starts = np.arange(0, len(imu) - n + 1, max(n // 2, 1))
    bad = np.cumsum(np.insert(~(ok & ref_ok), 0, 0))
    complete = (bad[starts + n] - bad[starts]) == 0
    quiet = (_window_std(acc_norm, n, starts) < max_acc_std) & np.all(
        [_window_std(gyr[:, k], n, starts) < np.radians(max_gyr_std_dps) for k in range(3)], axis=0)
    for start in starts[complete & quiet]:
        if reference_net_speed(segment, start, start + n - 1) < np.radians(max_ref_dps):
            still[start:start + n] = True
    return still & ok


def reference_net_speed(segment: Segment, first: int, last: int) -> float:
    """Net rotation speed (rad/s) of the reference between IMU samples `first` and `last`: the angle between its
    orientations at the two sample times (nearest frames) divided by the time between them. The relative rotation's
    angle does not depend on the convention of the stored orientation."""
    imu, ref = segment.imu, segment.reference
    t0, t1 = imu.times[first], imu.times[last]
    i0, i1 = (int(np.clip(np.rint((t - ref.start_s) * ref.rate_hz), 0, len(ref) - 1)) for t in (t0, t1))
    if t1 <= t0 or i0 == i1:
        return np.inf
    return float((ref.orientation[i0].inv() * ref.orientation[i1]).magnitude() / (t1 - t0))


def offset_from_reference_fit(segments: list[Segment]):
    """Offset (rad/s) as the constant part of the fit of the gyroscope against the reference rate, pooled over the
    given segments, with the better of a rotation and a mirror image as the map; None if too little motion."""
    fit = fit_gyroscope_to_reference(segments)
    if fit is None:
        return None
    pairs = [rate_pairs(s, fit.convention) for s in segments]
    g = np.vstack([p[0] for p in pairs])
    w = np.vstack([p[1] for p in pairs])
    m, _ = best_rotation(g, w, proper=fit.residual_dps <= fit.mirrored_residual_dps)
    return (g - w @ m).mean(0)  # reference rate ≈ M gyroscope, so gyroscope - M^T reference rate = offset


def estimate_gyroscope_offsets(recordings: Recording | list[Recording], name: str, min_rest_s: float = MIN_REST_S,
                               reference_fallback: bool = True, **limits) -> list[dict]:
    """Offset of segment `name` in each given recording of one session: one dict per recording that has the segment,
    with trial, offset (rad/s or None), method ('rest (trial)', 'rest (session)', 'reference fit', 'none') and the
    seconds of rest found in the trial."""
    recs = [r for r in as_session(recordings) if name in r.segments]
    rests = [rest_mask(r[name], **limits) for r in recs]
    pooled = [r[name].imu.gyr[m] for r, m in zip(recs, rests)]
    pooled_rest = np.vstack(pooled) if pooled else np.zeros((0, 3))
    rate = recs[0][name].imu.rate_hz if recs else 1.0
    fallback = None
    out = []
    for r, m in zip(recs, rests):
        seg = r[name]
        rest_s = float(m.sum() / seg.imu.rate_hz)
        if rest_s >= min_rest_s:
            offset, method = seg.imu.gyr[m].mean(0), 'rest (trial)'
        elif len(pooled_rest) >= min_rest_s * rate:
            offset, method = pooled_rest.mean(0), 'rest (session)'
        elif reference_fallback:
            if fallback is None:
                fallback = offset_from_reference_fit([x[name] for x in recs])
                fallback = fallback if fallback is not None else False
            offset, method = (fallback, 'reference fit') if fallback is not False else (None, 'none')
        else:
            offset, method = None, 'none'
        out.append(dict(trial=r.trial, offset=offset, method=method, rest_s=rest_s))
    return out


def check_for_gyroscope_offset(recordings: Recording | list[Recording], threshold_dps: float = THRESHOLD_DPS,
                               min_rest_s: float = MIN_REST_S, reference_fallback: bool = True,
                               **limits) -> CheckResult:
    """Does any gyroscope read more than `threshold_dps` when it should read zero? One offset per trial and segment
    (module description); `limits` are the rest limits of rest_mask (window_s, max_acc_std, max_gyr_std_dps,
    max_ref_dps). Per segment: detected if any trial's offset exceeds the threshold, estimate = the largest offset
    (°/s), `offset` = the offset of the first trial (the only one for a single recording), `per_trial` = all."""
    recs = as_session(recordings)
    names = list(dict.fromkeys(n for r in recs for n in r.segments))
    per_segment, values = {}, []
    for name in names:
        trials = estimate_gyroscope_offsets(recs, name, min_rest_s, reference_fallback, **limits)
        sizes = [float(np.degrees(np.linalg.norm(t['offset']))) for t in trials if t['offset'] is not None]
        for t in trials:
            t['size_dps'] = float(np.degrees(np.linalg.norm(t['offset']))) if t['offset'] is not None else np.nan
        if not sizes:
            per_segment[name] = entry(None, float('nan'), 'deg/s', method='none', per_trial=trials, offset=None)
            continue
        size = max(sizes)
        per_segment[name] = entry(size > threshold_dps, size, 'deg/s', offset=trials[0]['offset'],
                                  method=trials[0]['method'], per_trial=trials)
        values.append(size)
    thresholds = dict(threshold_dps=threshold_dps, min_rest_s=min_rest_s, **limits)
    if not values:
        return CheckResult(ERROR, None, 'No rest phase and no fit to estimate any offset.', per_segment, thresholds)
    detected = max(values) > threshold_dps
    summary = f'Gyroscope offsets {min(values):.2f} to {max(values):.2f} °/s over {len(values)} segments.'
    return CheckResult(ERROR, detected, summary, per_segment, thresholds)


def remove_gyroscope_offset(recording: Recording, result: CheckResult | None = None) -> Recording:
    """Return a copy with this trial's estimated offset subtracted from each segment's gyroscope.

    `result` may come from a session-level check; the offset of this recording's trial is applied."""
    if result is None:
        result = check_for_gyroscope_offset(recording)
    new = recording.copy()
    removed = {}
    for name, seg in new.segments.items():
        e = result.per_segment.get(name, {})
        trials = [t for t in e.get('per_trial', []) if t['trial'] == recording.trial]
        offset = trials[0]['offset'] if trials else e.get('offset')
        if offset is not None:
            seg.imu.gyr = seg.imu.gyr - offset
            removed[name] = float(np.degrees(np.linalg.norm(offset)))
    new.log_correction(ERROR, f'Subtracted the estimated gyroscope offset in {len(removed)} segments.',
                       offset_dps=removed)
    return new


# Dataset level: the check runs per session (the loader's session_of), from rest only (the trial's own rest, else the
# session's pooled rest); one row per trial and segment with the offset to subtract.

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, frame, number, per_trial, prevalence_of, run_groups  # noqa: E402
from .e16_mirrored_gyroscope import _load_all  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'session', 'method', 'rest_s',
           'offset_x_dps', 'offset_y_dps', 'offset_z_dps', 'error']


def _check_session(loader, trials):
    recs, rows = _load_all(loader, trials)
    if not recs:
        return rows
    c = check_for_gyroscope_offset(recs, reference_fallback=False)
    for seg, p in c.per_segment.items():
        for tr in p['per_trial']:
            o = np.degrees(tr['offset']) if tr['offset'] is not None else [np.nan] * 3
            size = tr['size_dps']
            rows.append(dict(trial=tr['trial'], participant=str(loader.participant_of(tr['trial'])), segment=seg,
                             detected=bool(size > THRESHOLD_DPS) if np.isfinite(size) else None, estimate=size,
                             unit='deg/s', session=str(loader.session_of(tr['trial'])), method=tr['method'],
                             rest_s=round(tr['rest_s'], 2), offset_x_dps=o[0], offset_y_dps=o[1], offset_z_dps=o[2],
                             error=''))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """The gyroscope offset of every trial and segment, from rest phases, per session."""
    return frame(run_groups(loader, _check_session, processes, key='session'), COLUMNS)


def prevalence(rows) -> dict:
    """Trials / trial-segments whose offset at rest exceeds 0.01 °/s; segments without rest are undecided."""
    return prevalence_of(rows, 'trial-segments')


def _subtract(rec, rows):
    new, removed = rec.copy(), {}
    for r in rows:
        o = np.array([number(r.get(f'offset_{a}_dps')) for a in 'xyz'])
        if r['segment'] in new.segments and np.isfinite(o).all():
            new.segments[r['segment']].imu.gyr = new.segments[r['segment']].imu.gyr - np.radians(o)
            removed[r['segment']] = float(np.linalg.norm(o))
    new.log_correction(ERROR, f'Subtracted the gyroscope offset in {len(removed)} segments (rest phases).',
                       offset_dps=removed)
    return new


def correct(loader, rows):
    """Each trial's offset subtracted from every gyroscope sample of its segment; segments without rest keep theirs."""
    return corrected(loader, _subtract, per_trial(rows), 'gyroscope offset')


SPEC = ErrorSpec(17, 'gyroscope_offset', 'fix', 'gyroscope offset not removed')
PARTS = {17: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_gyroscope_offset = make_handler(PARTS[17])
