"""Error 16 (mirrored gyroscope, all three axes or a single flipped axis): the gyroscope is stored mirrored, all three
axes or one axis negated, so its frame is left-handed.

What it is: a sign error in the sensor's firmware, export or loader. Against the optical reference, a correctly stored
gyroscope differs only by a rotation (the mounting); a mirrored one by a rotation combined with a mirror.

How we catch it: two readings, and a gyroscope counts as mirrored only if both say so; where they disagree, or the
motion is too planar to read, it is undecided and reported, not corrected.
1. Fit test: the best rotation and the best mirror image mapping the gyroscope onto the reference's angular rate
   (both after removing the means, so a gyroscope offset does not enter); the better fit wins.
2. Volume test: for random triples of moving samples, the sign of the volume spanned by the three gyroscope rates
   (det of the 3x3 matrix) against the sign of the volume of the three reference rates at the same moments. A
   rotation keeps the sign, a mirror flips it. Only well-spread triples count (|det| at least TRIPLE_CONDITION of the
   product of the three lengths); fewer than MIN_TRIPLES: no reading. The share of agreeing signs decides: below
   MIRRORED_BELOW mirrored, above RIGHT_HANDED_ABOVE right-handed, in between no reading (right-handed sensors read a
   median 0.95 in the datasets, while 0.40-0.50 are coin flips).
The reference rate is differenced over RATE_WINDOW_S, not frame to frame, so marker jitter does not swamp it. Only
samples moving faster than MOVING_RAD_S carry a direction.

Pooling: the handedness is read over all trials of one sensor on one participant, split into separate sessions where
the sensor was clearly remounted: a trial joins a session when its own best map (rotation, or rotation of the negated
gyroscope) is within SESSION_ANGLE_DEG of the session's first trial and of the same handedness; trials with fewer than
OWN_FIT_MIN_SAMPLES moving samples join the largest session. A session other than the sensor's largest is marked
`split_off`: it is judged and corrected like any other but reported apart, since a broken trial (mis-paired, wrong
segment) can split off and read mirrored as well as a remounted sensor.

How we correct it: against the reference alone, negating one axis or all three fit equally well (one axis = all three
plus a 180° turn, which the mounting rotation absorbs). The accelerometer, which shares the gyroscope's frame,
decides: for each candidate (x, y, z, all three) the rotation-free cross-check (accelerometer_cross_check.py) gives
the slope of the measured gyroscope · accelerometer product on the one the reference predicts, about ±1 for the right
candidate and nearer zero for the others. The candidate with the largest |slope| wins if its 99 % interval excludes
zero and lies wholly beyond the runner-up's; otherwise, or where the cross-check cannot read, all three axes are
flipped and the axis is recorded as open. The trials' up direction comes from the sensors' majority (error 20's
trial_vertical).

Place in the order: after the corrupted samples and before the gyroscope offset, so the offset's backup fit against
the reference sees a right-handed gyroscope; the readings need the time alignment.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..recording import Recording, Segment
from ..rotations import best_rotation, interpolate
from ._result import CheckResult, entry
from ._session import as_session

ERROR = 'mirrored gyroscope'
# Constants of the readings (chosen without looking at the data):
RATE_WINDOW_S = 0.04  # differencing window of the reference rate: averages marker jitter, short against gait
MOVING_RAD_S = 0.5  # samples slower than this carry no direction
TRIPLES = 200_000  # random triples per session for the volume test
TRIPLE_CONDITION = 0.40  # a triple counts if |det| >= this share of the product of its three lengths
MIN_TRIPLES = 100  # fewer counting triples: no volume reading
SESSION_ANGLE_DEG = 30.0  # own maps further apart than this: remounted, a separate session
OWN_FIT_MIN_SAMPLES = 300  # moving samples a trial needs for its own map to place it in a session
FLIPS = {'x': np.array([-1.0, 1, 1]), 'y': np.array([1.0, -1, 1]), 'z': np.array([1.0, 1, -1]),
         'all three': np.array([-1.0, -1, -1])}
GRAVITY_MPS2 = 9.81
MIRRORED_BELOW, RIGHT_HANDED_ABOVE = 0.4, 0.6  # volume agreement: decided only outside this band


def _segment_to_world(orientation: Rotation, convention: str) -> Rotation:
    return orientation.inv() if convention == 'world_to_segment' else orientation


def rate_pairs(segment: Segment, convention: str, window_s: float = RATE_WINDOW_S,
               min_rate: float = MOVING_RAD_S) -> tuple[np.ndarray, np.ndarray]:
    """Gyroscope (rad/s, sensor frame) and reference rate (rad/s, segment frame) at the same moments, for the moving
    samples where both are valid. The reference rate is differenced over `window_s`."""
    ref, imu = segment.reference, segment.imu
    r = _segment_to_world(ref.orientation, convention)
    m = max(1, int(round(window_s * ref.rate_hz)))
    if len(r) <= m:
        return np.zeros((0, 3)), np.zeros((0, 3))
    w = (r[:-m].inv() * r[m:]).as_rotvec() * ref.rate_hz / m
    t = ref.start_s + (np.arange(len(w)) + m / 2) / ref.rate_hz
    ok = ref.valid[:-m] & ref.valid[m:]
    g = interpolate(t, imu.times, np.where(imu.valid[:, None], imu.gyr, np.nan))
    use = ok & np.isfinite(g).all(1) & np.isfinite(w).all(1) & (np.linalg.norm(g, axis=1) > min_rate)
    return g[use], w[use]


def volume_agreement(g: np.ndarray, w: np.ndarray, seed: int = 0) -> tuple[float, int]:
    """Share of well-spread triples whose gyroscope volume has the sign of the reference volume (NaN if fewer than
    MIN_TRIPLES count), and the number of counting triples."""
    if len(g) < 3:
        return float('nan'), 0
    idx = np.random.default_rng(seed).integers(0, len(g), size=(TRIPLES, 3))
    gt, wt = g[idx], w[idx]
    vg, vw = np.linalg.det(gt), np.linalg.det(wt)
    ok = np.abs(vg) >= TRIPLE_CONDITION * np.prod(np.linalg.norm(gt, axis=2), axis=1)
    if ok.sum() < MIN_TRIPLES:
        return float('nan'), int(ok.sum())
    return float(np.mean(np.sign(vg[ok]) == np.sign(vw[ok]))), int(ok.sum())


def read_handedness(g: np.ndarray, w: np.ndarray) -> dict | None:
    """Both readings on paired rates: fit residuals (°/s) of the best rotation and mirror, the volume agreement and
    the verdict (True mirrored, False right-handed, None undecided)."""
    if len(g) < 20:
        return None
    m_proper, res_proper = best_rotation(g, w, proper=True)
    m_mirror, res_mirror = best_rotation(g, w, proper=False)
    better = m_proper if res_proper <= res_mirror else m_mirror
    offset = g.mean(0) - w.mean(0) @ better  # w ≈ M (g - b): the gyroscope offset, removed before the volumes
    agreement, triples = volume_agreement(g - offset, w)
    fit_says = res_mirror < res_proper
    volume_says = (None if not np.isfinite(agreement) or MIRRORED_BELOW <= agreement <= RIGHT_HANDED_ABOVE
                   else agreement < MIRRORED_BELOW)
    verdict = fit_says if volume_says is not None and volume_says == fit_says else None
    return dict(residual_rotation_dps=float(np.degrees(res_proper)), residual_mirror_dps=float(np.degrees(res_mirror)),
                volume_agreement=agreement, triples=triples, fit_says_mirrored=bool(fit_says),
                volume_says_mirrored=volume_says, mirrored=verdict, samples=len(g))


def _own_map(g: np.ndarray, w: np.ndarray) -> tuple[int, np.ndarray]:
    """Handedness sign and rotation of the better of the proper maps of g and -g."""
    m_pos, r_pos = best_rotation(g, w, proper=True)
    m_neg, r_neg = best_rotation(-g, w, proper=True)
    return (1, m_pos) if r_pos <= r_neg else (-1, m_neg)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude()))


def _convention(segments: list[Segment]) -> str:
    """The stated convention, else the one whose better map (rotation or mirror) fits the pooled rates best."""
    known = segments[0].reference.convention
    if known != 'unknown':
        return known
    best, choice = np.inf, 'segment_to_world'
    for c in ('segment_to_world', 'world_to_segment'):
        pairs = [rate_pairs(s, c) for s in segments]
        g, w = np.vstack([p[0] for p in pairs]), np.vstack([p[1] for p in pairs])
        if len(g) < 20:
            continue
        res = min(best_rotation(g, w, proper=True)[1], best_rotation(g, w, proper=False)[1])
        if res < best:
            best, choice = res, c
    return choice


def sessions_of(trials: list[tuple[str, np.ndarray, np.ndarray]]) -> list[list[int]]:
    """Group trials (name, g, w) of one sensor into mounting sessions (module description); returns index lists."""
    clusters, pending = [], []
    for i, (_, g, w) in enumerate(trials):
        if len(g) < OWN_FIT_MIN_SAMPLES:
            pending.append(i)
            continue
        sign, m = _own_map(g, w)
        for c in clusters:
            if c['sign'] == sign and _angle_deg(c['map'], m) <= SESSION_ANGLE_DEG:
                c['members'].append(i)
                break
        else:
            clusters.append(dict(sign=sign, map=m, members=[i]))
    if not clusters:
        return [list(range(len(trials)))]
    largest = max(clusters, key=lambda c: len(c['members']))
    largest['members'].extend(pending)
    return [sorted(c['members']) for c in clusters]


def choose_flip_by_slope(slopes: dict) -> tuple[str, bool]:
    """The flip to apply and whether the cross-check decided it: slopes {candidate: dict(slope, low, high)}; the
    largest |slope| wins if its interval excludes zero and lies beyond the runner-up's whole interval (absolute
    values); otherwise all three axes, open."""
    ok = {k: v for k, v in slopes.items() if np.isfinite(v['slope'])}
    if len(ok) < 2:
        return 'all three', False
    ranked = sorted(ok, key=lambda k: -abs(ok[k]['slope']))
    best, runner = ok[ranked[0]], ok[ranked[1]]
    excludes_zero = best['low'] > 0 or best['high'] < 0
    beyond = min(abs(best['low']), abs(best['high'])) > max(abs(runner['low']), abs(runner['high']))
    return (ranked[0], True) if excludes_zero and beyond else ('all three', False)


def cross_check_slopes(recordings: list[Recording], segment: str, convention: str, ups: dict) -> dict:
    """Cross-check slope of each candidate flip of this segment's gyroscope over the session's trials."""
    from .accelerometer_cross_check import session_slope  # here: accelerometer_sign imports this module
    return {name: session_slope(recordings, segment, ups, convention, gyr_map=np.diag(flip))
            for name, flip in FLIPS.items()}


def check_for_mirrored_gyroscope(recordings: Recording | list[Recording]) -> CheckResult:
    """Is any gyroscope stored mirrored? Pass all trials of one participant: each sensor is read per mounting
    session (module description). Per segment: detected True if a session is mirrored, None if no session could be
    decided; `sessions` holds each session's trials, readings, candidate flips and chosen flip, `per_trial` each
    trial's own reading and session."""
    recs = as_session(recordings)
    names = list(dict.fromkeys(n for r in recs for n in r.segments))
    per_segment, mirrored_names, decided = {}, [], 0
    ups = None  # each trial's up direction from its sensors' majority, computed when a session is mirrored
    for name in names:
        have = [r for r in recs if name in r.segments]
        segs = [r[name] for r in have]
        convention = _convention(segs)
        pairs = [rate_pairs(s, convention) for s in segs]
        groups = sessions_of([(r.trial, g, w) for r, (g, w) in zip(have, pairs)])
        sessions, per_trial = [], {}
        main = max(range(len(groups)), key=lambda j: len(groups[j]))  # the largest; the first on a tie
        for label, members in enumerate(groups):
            g = np.vstack([pairs[i][0] for i in members])
            w = np.vstack([pairs[i][1] for i in members])
            reading = read_handedness(g, w)
            session = dict(session=f's{label + 1}', trials=[have[i].trial for i in members], reading=reading,
                           mirrored=None if reading is None else reading['mirrored'], flip=None, cross_slopes=None,
                           axis_decided=None, split_off=label != main)
            if session['mirrored']:
                if ups is None:
                    from .e19_e20_accelerometer_sign import check_for_negated_accelerometer
                    tv = check_for_negated_accelerometer(recs, cross_check=False).thresholds['trial_verticals']
                    ups = {t: np.asarray(v['up']) for t, v in tv.items()}
                slopes = cross_check_slopes([have[i] for i in members], name, convention, ups)
                session['cross_slopes'] = slopes
                session['flip'], session['axis_decided'] = choose_flip_by_slope(slopes)
            sessions.append(session)
            for i in members:
                per_trial[have[i].trial] = dict(session=session['session'],
                                                reading=read_handedness(*pairs[i]), flip=session['flip'])
        verdicts = [s['mirrored'] for s in sessions]
        if all(v is None for v in verdicts):
            detected = None
        else:
            detected = any(v for v in verdicts)
            decided += 1
        if detected:
            mirrored_names.append(name)
        agreements = [s['reading']['volume_agreement'] for s in sessions if s['reading']]
        per_segment[name] = entry(detected, float(np.nanmin(agreements)) if agreements and
                                  np.isfinite(agreements).any() else float('nan'), 'share of agreeing volumes',
                                  convention=convention, sessions=sessions, per_trial=per_trial)
    thresholds = dict(rate_window_s=RATE_WINDOW_S, moving_rad_s=MOVING_RAD_S, triple_condition=TRIPLE_CONDITION,
                      min_triples=MIN_TRIPLES, session_angle_deg=SESSION_ANGLE_DEG, mirrored_below=MIRRORED_BELOW,
                      right_handed_above=RIGHT_HANDED_ABOVE)
    if not decided:
        return CheckResult(ERROR, None, 'Motion too little or too planar, or the two readings disagree, for every '
                                        'gyroscope.', per_segment, thresholds)
    summary = (f'Mirrored gyroscope in {", ".join(mirrored_names)}.' if mirrored_names
               else f'All {decided} decided gyroscopes are right-handed.')
    return CheckResult(ERROR, bool(mirrored_names), summary, per_segment, thresholds)


def remove_mirrored_gyroscope(recording: Recording, result: CheckResult) -> Recording:
    """Copy with the chosen axes negated in every gyroscope whose session the check found mirrored (the session's
    flip applied to this trial)."""
    new, flipped = recording.copy(), {}
    for name, p in result.per_segment.items():
        trial = (p.get('per_trial') or {}).get(recording.trial)
        if name not in new.segments or not trial or not trial['flip']:
            continue
        new.segments[name].imu.gyr = new.segments[name].imu.gyr * FLIPS[trial['flip']]
        flipped[name] = trial['flip']
    new.log_correction(ERROR, f'Negated the gyroscope axes of {len(flipped)} segments stored mirrored: '
                       + (', '.join(f'{n} ({f})' for n, f in flipped.items()) or 'none') + '.', flipped=flipped)
    return new


# Sensors whose sign readings (mirrored gyroscope, error 16; negated accelerometer, error 20) are reported only, never
# counted or corrected: (dataset, participant, segment) -> reason.
SIGN_REPORT_ONLY = {
    ('ultramocap', 'P08', 'upper_arm'): 'the gyroscope-to-reference (mounting) fit explains almost none of the motion '
                                        '(misfit about the size of the reference rate, 9 apparent mounting sessions), '
                                        'so the mirror verdict and the accelerometer sign read through it are not '
                                        'trusted',
}


def sign_report_only(dataset: str, participant, segment: str) -> bool:
    return (dataset, str(participant), segment) in SIGN_REPORT_ONLY


# Dataset level: the check runs per participant (each sensor read over its trials, split into mounting sessions); one
# row per trial and segment with its session's verdict and flip.

from functools import partial  # noqa: E402

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, flag, frame, per_trial, read_rows, run_groups, truthy, undecided  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'session', 'split_off', 'flip',
           'axis_decided', 'report_only', 'trial_mirrored', 'volume_agreement', 'error']


def _load_all(loader, trials):
    recs, rows = [], []
    for t in trials:
        try:
            rec = loader.load(t)
            if not rec.segments:
                raise ValueError('no segment has both an IMU and a reference')
            rec.trial = t  # the checks key their trials by recording.trial: use the loader's identifier
            recs.append(rec)
        except Exception as exc:  # noqa: BLE001 - a broken trial is reported, not fatal
            rows.append(dict(trial=t, participant=str(loader.participant_of(t)), error=f'{type(exc).__name__}: {exc}'))
    return recs, rows


def _check_participant(loader, trials, dataset=''):
    recs, rows = _load_all(loader, trials)
    if not recs:
        return rows
    res = check_for_mirrored_gyroscope(recs)
    for seg, p in res.per_segment.items():
        sessions = {s['session']: s for s in p['sessions']}
        for trial, tr in p['per_trial'].items():
            s = sessions[tr['session']]
            participant = str(loader.participant_of(trial))
            reading = tr['reading'] or {}
            rows.append(dict(trial=trial, participant=participant, segment=seg, detected=s['mirrored'],
                             estimate=(s['reading'] or {}).get('volume_agreement', float('nan')),
                             unit='share of agreeing volumes', session=tr['session'], split_off=s['split_off'],
                             flip=tr['flip'] or '', axis_decided=s['axis_decided'],
                             report_only=sign_report_only(dataset, participant, seg),
                             trial_mirrored=reading.get('mirrored'),
                             volume_agreement=reading.get('volume_agreement', float('nan')), error=''))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """Handedness of every gyroscope, read per participant, sensor and mounting session."""
    return frame(run_groups(loader, partial(_check_participant, dataset=dataset), processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials / trial-segments of mirrored sessions within a sensor's main session; sessions split off from their
    sensor's main session are counted apart (split_off_trials, split_off_occurrences). Sensors reported only are
    left out."""
    ok = read_rows(rows)
    ok = ok[~ok.report_only.map(truthy)]
    hit = ok.detected.map(lambda v: flag(v) is True)
    split = ok.split_off.map(truthy)
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[hit & ~split].nunique()),
                occurrences=int((hit & ~split).sum()), occurrence_unit='trial-segments',
                undecided=int(ok.trial[ok.detected.map(lambda v: flag(v) is None)].nunique()),
                split_off_trials=int(ok.trial[hit & split].nunique()), split_off_occurrences=int((hit & split).sum()))


def _flip(rec, rows):
    new, flipped = rec.copy(), {}
    for r in rows:
        f = r.get('flip')
        if r['segment'] not in new.segments or undecided(f) or not f or truthy(r.get('report_only')):
            continue
        new.segments[r['segment']].imu.gyr = new.segments[r['segment']].imu.gyr * FLIPS[f]
        flipped[r['segment']] = f
    new.log_correction(ERROR, f'Negated the gyroscope axes of {len(flipped)} segments stored mirrored: '
                       + (', '.join(f'{n} ({f})' for n, f in flipped.items()) or 'none') + '.', flipped=flipped)
    return new


def correct(loader, rows):
    """The chosen axes negated in every trial of a mirrored session (split-off sessions too); sensors reported only
    are left as they are."""
    return corrected(loader, _flip, per_trial(rows), 'mirrored gyroscope')


SPEC = ErrorSpec(16, 'mirrored_gyroscope', 'fix', 'mirrored gyroscope or single flipped axis')
PARTS = {16: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_mirrored_gyroscope = make_handler(PARTS[16])
