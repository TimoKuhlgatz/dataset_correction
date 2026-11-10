"""Error 19 (reference vertical, report only) and error 20 (accelerometer negated: it reports the negative of the
specific force on all three axes).

What error 20 is: a sign setting in the sensor, its export or a loader that hits the accelerometer only; the gyroscope
in the same housing is right-handed by now (error 16 runs before).

How we catch it: averaged over a trial, the specific force carried into the room frame is gravity's reaction plus the
velocity change over the trial length, so a vector of about 9.81 m/s² along the room's vertical; a negated
accelerometer gives exactly the negative. The accelerometer is carried into the segment frame by a rotation fitted
from the gyroscope alone against the reference's rotation rate (no accelerometer enters it, so the reading is not
circular), then into the room by the reference orientation, and averaged. The mounting rotation is only a measuring
tool here: the data are not rotated by it.
- The vertical, per trial: which axis the optical system calls up differs between deposits and may change between
  sessions, but all units of one trial share one coordinate system. Each unit's average is compared with the common
  axis of the trial's other readable units (leave one out, sign ignored); the units within AXIS_TOLERANCE_DEG of it
  fit. With at least two fitting units, their common axis is the trial's vertical and the majority of them sets which
  way it points; on a tie the sign is taken from +z (`sign_assumed`). With fewer than two fitting units +z is assumed
  and the trial is reported (`vertical_assumed`); there a unit is read only within the same tolerance of ±z.
- Error 19 (reference vertical), report only: the trial's vertical is reported with its angle to the nearest signed
  coordinate axis; below AXIS_NAMED_DEG that axis is named (e.g. '+y'), from there on no axis is named and the trial
  is reported as suspicious. The sign verdicts use the found vertical either way.
- A unit pointing against its trial's vertical is negated in that trial; outside the tolerance it is unreadable there.
- Pooling: per sensor, participant and mounting session (sessions as for the mirrored gyroscope), the trial verdicts
  weighted by valid samples; more than half negated: negated, exactly half: undecided.
- Second reading: the rotation-free cross-check (accelerometer_cross_check.py) on the same sessions with the same
  trial verticals. A session counts as negated only if both readings say so and it is not a single trial split off
  from its sensor; a session the main reading calls correct and the cross-check negated is undecided. Every other case
  is reported, not corrected (`reason`). Where the cross-check cannot read, nothing is negated.
- Unreadable: an average shorter than MIN_LENGTH_G of gravity has no direction; this is where removed gravity
  (error 22) shows.
- Not detectable: every unit of a trial negated alike; harmless, since a common turn of the whole room cancels in the
  relative orientation. Where most units of a trial are negated the correct minority is flagged instead.
Not read: single negated axes and swapped axes; in walking, wrong axis orders reach trial averages longer and more
upward than the right one, so no axis-level rule held.

How we correct it: negate the three accelerometer columns of every trial of a negated unit-session.
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording, Segment
from ..rotations import best_rotation
from ._result import CheckResult, entry
from ._session import as_session
from .accelerometer_cross_check import session_slope
from .e16_mirrored_gyroscope import _convention, _segment_to_world, rate_pairs, sessions_of

ERROR = 'negated accelerometer'
GRAVITY_MPS2 = 9.81
MIN_LENGTH_G = 0.5  # pooled averages shorter than this share of gravity carry no readable direction
MIN_SAMPLES = 300  # valid samples a trial needs for its average
MIN_RATE_PAIRS = 20  # moving samples a session needs for the gyroscope-only mounting fit
# A unit's average fits its trial's vertical within this angle (sign ignored): the 99th percentile
# of the angle of every unit's average to the common axis of the other units of its trial, over 14,311 unit-trials of
# the datasets in the order of correction. Beyond it a unit is unreadable in that trial, never negated.
AXIS_TOLERANCE_DEG = 46.6
# Error 19 (reference vertical), which coordinate axis is the reference's vertical: the trial's
# vertical within AXIS_NAMED_DEG of a signed coordinate axis names that axis; from AXIS_NAMED_DEG to AXIS_REJECTED_DEG
# no axis is named and the trial is reported as suspicious. The nearest signed axis is never further than 54.7°, so in
# practice every vertical from 30° on is suspicious.
AXIS_NAMED_DEG, AXIS_REJECTED_DEG = 30.0, 60.0
SIGNED_AXES = {f'{s}{a}': np.eye(3)[i] * (1 if s == '+' else -1) for i, a in enumerate('xyz') for s in '+-'}


def trial_mean(segment: Segment, convention: str, sensor_to_segment: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Room-frame average of the accelerometer (m/s²) and its number of valid samples; None below MIN_SAMPLES."""
    ref, imu = segment.reference, segment.imu
    k = np.rint((imu.times - ref.start_s) * ref.rate_hz).astype(int)
    ok = (k >= 0) & (k < len(ref)) & imu.valid & np.isfinite(imu.acc).all(1)
    ok[ok] &= ref.valid[k[ok]]
    if ok.sum() < MIN_SAMPLES:
        return None
    room = _segment_to_world(ref.orientation[k[ok]], convention).apply(imu.acc[ok] @ sensor_to_segment.T)
    return room.mean(0), int(ok.sum())


def _readable(v: np.ndarray) -> bool:
    return bool(np.linalg.norm(v) >= MIN_LENGTH_G * GRAVITY_MPS2)


def _axis(directions: list[np.ndarray]) -> np.ndarray:
    """Sign-blind common axis of unit vectors: the largest eigenvector of the sum of their outer products."""
    d = np.asarray(directions)
    return np.linalg.eigh(d.T @ d)[1][:, -1]


def _angle_to_axis_deg(d: np.ndarray, u: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(d @ u), 0.0, 1.0))))


def name_vertical(up: np.ndarray) -> tuple[str | None, str, float]:
    """Error 19 (reference vertical): the signed coordinate axis nearest to a trial's vertical, its angle, and the
    status: 'named' below AXIS_NAMED_DEG, 'suspicious' up to AXIS_REJECTED_DEG (no axis named), 'none' beyond."""
    axis = max(SIGNED_AXES, key=lambda k: SIGNED_AXES[k] @ up)
    angle = float(np.degrees(np.arccos(np.clip(SIGNED_AXES[axis] @ up, -1.0, 1.0))))
    status = 'named' if angle < AXIS_NAMED_DEG else 'suspicious' if angle <= AXIS_REJECTED_DEG else 'none'
    return (axis if status == 'named' else None), status, angle


def trial_vertical(vectors: dict[str, np.ndarray], tolerance_deg: float) -> dict:
    """The vertical of one trial from its units' room-frame averages (module description). Returns up (unit vector),
    source ('sensors' or 'assumed +z'), sign_assumed, votes (with, against) and per unit: loo_angle_deg (to the axis
    of the other readable units, NaN with no other), angle_up_deg (to the line of the vertical, sign ignored) and
    negated (True / False / None where unreadable or outside the tolerance)."""
    dirs = {n: v / np.linalg.norm(v) for n, v in vectors.items() if _readable(v)}
    loo = {n: (_angle_to_axis_deg(d, _axis([e for m, e in dirs.items() if m != n])) if len(dirs) > 1 else np.nan)
           for n, d in dirs.items()}
    fitting = [n for n in dirs if loo[n] <= tolerance_deg]
    sign_assumed = False
    if len(fitting) >= 2:
        u = _axis([dirs[n] for n in fitting])
        with_u = sum(dirs[n] @ u > 0 for n in fitting)
        against = len(fitting) - with_u
        if with_u == against:
            sign_assumed = True
            u = u if u[2] >= 0 else -u
        elif against > with_u:
            u = -u
        source = 'sensors'
    else:
        u, source = np.array([0.0, 0.0, 1.0]), 'assumed +z'
    units = {}
    for n, v in vectors.items():
        if n not in dirs:
            units[n] = dict(loo_angle_deg=np.nan, angle_up_deg=np.nan, negated=None)
            continue
        angle = _angle_to_axis_deg(dirs[n], u)
        ok = (n in fitting) if source == 'sensors' else angle <= tolerance_deg
        units[n] = dict(loo_angle_deg=loo[n], angle_up_deg=angle, negated=bool(dirs[n] @ u < 0) if ok else None)
    votes = (sum(1 for n in units if units[n]['negated'] is False), sum(1 for n in units if units[n]['negated']))
    axis, status, angle = name_vertical(u) if source == 'sensors' else ('+z', 'assumed', 0.0)
    return dict(up=u, source=source, sign_assumed=sign_assumed, votes=votes, units=units, axis=axis,
                axis_status=status, axis_angle_deg=angle)


def check_for_negated_accelerometer(recordings: Recording | list[Recording], tolerance_deg: float | None = None,
                                    cross_check: bool = True) -> CheckResult:
    """Is any accelerometer negated? Pass all trials of one participant. Per segment: detected True if a session is
    negated, False if every readable session points with its trials' verticals, None if none is readable.
    `sessions` holds each session's trials, verdict, share of negated trial verdicts and fallback count; `per_trial`
    each trial's own reading. The thresholds dict lists the trials whose vertical was assumed (`vertical_assumed`)
    or whose sign was (`sign_assumed`), and `trial_verticals` the vertical and votes of every trial."""
    tol = AXIS_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
    recs = as_session(recordings)
    names = list(dict.fromkeys(n for r in recs for n in r.segments))
    # 1. per segment: sessions, gyroscope-only mounting per session, each trial's room-frame average
    units, means = {}, {}  # means[(trial, segment)] = (vector, samples, session label)
    for name in names:
        have = [r for r in recs if name in r.segments]
        segs = [r[name] for r in have]
        convention = _convention(segs)
        pairs = [rate_pairs(s, convention) for s in segs]
        groups = sessions_of([(r.trial, g, w) for r, (g, w) in zip(have, pairs)])
        units[name] = dict(convention=convention, sessions=[])
        for label, members in enumerate(groups):
            g = np.vstack([pairs[i][0] for i in members])
            w = np.vstack([pairs[i][1] for i in members])
            session = dict(session=f's{label + 1}', trials=[have[i].trial for i in members],
                           mounting_fitted=len(g) >= MIN_RATE_PAIRS,
                           split_off=label != max(range(len(groups)), key=lambda j: len(groups[j])))
            units[name]['sessions'].append(session)
            if not session['mounting_fitted']:
                continue
            m, misfit = best_rotation(g, w, proper=True)  # w ≈ m g: sensor frame -> segment frame
            centred = w - w.mean(0)
            spread = float(np.sqrt((centred ** 2).sum(1).mean()))
            axes = np.linalg.svd(centred, compute_uv=False)
            session.update(mounting_misfit_dps=float(np.degrees(misfit)),
                           misfit_share=float(misfit / spread) if spread > 0 else float('inf'),
                           excitation=float(axes[2] / axes[0]) if axes[0] > 0 else 0.0)
            for i in members:
                got = trial_mean(segs[i], convention, m)
                if got is not None:
                    means[(have[i].trial, name)] = (*got, session['session'])
    # 2. per trial: its vertical from its own units, the units' verdicts against it
    verticals = {}
    for r in recs:
        vectors = {n: means[(r.trial, n)][0] for n in r.segments if (r.trial, n) in means}
        verticals[r.trial] = trial_vertical(vectors, tol)
    # 3. per unit-session: the trial verdicts pooled, weighted by samples
    per_segment, negated, decided = {}, [], 0
    for name in names:
        per_trial, verdicts = {}, []
        for session in units[name]['sessions']:
            read = [(t, verticals[t]['units'][name]['negated'], means[(t, name)][1]) for t in session['trials']
                    if (t, name) in means and verticals[t]['units'][name]['negated'] is not None]
            weight = sum(n for _, _, n in read)
            share = sum(n for _, neg, n in read if neg) / weight if weight else float('nan')
            main = None if not weight or share == 0.5 else bool(share > 0.5)
            cross = (session_slope([r for r in recs if r.trial in session['trials']], name,
                                   {t: verticals[t]['up'] for t in session['trials']}, units[name]['convention'])
                     if cross_check else dict(negated=None, slope=float('nan'), low=float('nan'),
                                              high=float('nan'), trials=0))
            single_split = session['split_off'] and len(session['trials']) == 1
            if main is None:
                verdict, reason = None, 'main reading undecided'
            elif main is False:
                verdict, reason = (None, 'readings disagree') if cross['negated'] else (False, 'correct')
            elif cross['negated'] is not True:
                verdict, reason = None, ('cross-check unreadable' if cross['negated'] is None else 'readings disagree')
            elif single_split:
                verdict, reason = None, 'single trial split off'
            else:
                verdict, reason = True, 'both readings negated'
            session.update(negated=verdict, reason=reason, negated_main=main, negated_cross=cross['negated'],
                           cross_slope=cross['slope'], cross_low=cross['low'], cross_high=cross['high'],
                           cross_trials=cross['trials'],
                           negated_share=share, trials_read=len(read), readable=verdict is not None,
                           trials_vertical_assumed=sum(verticals[t]['source'] != 'sensors' for t, _, _ in read),
                           trials_disagreeing=[t for t, neg, _ in read if verdict is not None and neg != verdict])
            verdicts.append(verdict)
            for t in session['trials']:
                own = means.get((t, name))
                tv = verticals[t]['units'].get(name, {})
                per_trial[t] = dict(session=session['session'], negated=verdict, trial_negated=tv.get('negated'),
                                    loo_angle_deg=tv.get('loo_angle_deg', np.nan),
                                    angle_up_deg=tv.get('angle_up_deg', np.nan),
                                    angle_z_deg=_angle_to_axis_deg(own[0] / np.linalg.norm(own[0]),
                                                                   np.array([0.0, 0, 1])) if own else np.nan,
                                    length_mps2=float(np.linalg.norm(own[0])) if own else float('nan'),
                                    vertical_source=verticals[t]['source'])
        if all(v is None for v in verdicts):
            detected = None
        else:
            detected = any(v for v in verdicts if v is not None)
            decided += 1
        if detected:
            negated.append(name)
        shares = [s['negated_share'] for s in units[name]['sessions'] if s.get('readable')]
        per_segment[name] = entry(detected, max(shares) if shares else float('nan'),
                                  'share of trials read negated', convention=units[name]['convention'],
                                  sessions=units[name]['sessions'], per_trial=per_trial)
    assumed = [t for t, v in verticals.items() if v['source'] != 'sensors']
    sign_assumed = [t for t, v in verticals.items() if v['sign_assumed']]
    thresholds = dict(axis_tolerance_deg=tol, min_length_g=MIN_LENGTH_G, min_samples=MIN_SAMPLES,
                      min_rate_pairs=MIN_RATE_PAIRS, gravity_mps2=GRAVITY_MPS2, vertical_assumed=assumed,
                      sign_assumed=sign_assumed,
                      axis_named_deg=AXIS_NAMED_DEG, axis_rejected_deg=AXIS_REJECTED_DEG,
                      vertical_suspicious=[t for t, v in verticals.items() if v['axis_status'] == 'suspicious'],
                      trial_verticals={t: dict(up=v['up'].round(4).tolist(), source=v['source'], votes=v['votes'],
                                               axis=v['axis'], axis_status=v['axis_status'],
                                               axis_angle_deg=round(v['axis_angle_deg'], 2))
                                       for t, v in verticals.items()})
    note = (f' Vertical assumed +z in {len(assumed)} trials (no common axis of the units within {tol:g}°).'
            if assumed else '')
    if not decided:
        return CheckResult(ERROR, None, 'No accelerometer could be read.' + note, per_segment, thresholds)
    summary = (f'Negated accelerometer in {", ".join(negated)}.' if negated
               else f'All {decided} readable accelerometers point with their trials\' vertical.') + note
    return CheckResult(ERROR, bool(negated), summary, per_segment, thresholds)


def remove_negated_accelerometer(recording: Recording, result: CheckResult) -> Recording:
    """Copy with the three accelerometer columns negated in every segment whose session the check found negated."""
    new, changed = recording.copy(), []
    for name, p in result.per_segment.items():
        trial = (p.get('per_trial') or {}).get(recording.trial)
        if name not in new.segments or not trial or not trial['negated']:
            continue
        new.segments[name].imu.acc = -new.segments[name].imu.acc
        changed.append(name)
    new.log_correction(ERROR, f'Negated the accelerometer of {len(changed)} segments: '
                       + (', '.join(changed) or 'none') + '.', negated=changed)
    return new


# Dataset level: the check runs per participant (each sensor pooled over its trials per mounting session). Error 19
# keeps one row per trial (its vertical); error 20 one row per trial and segment (its session's verdict).

from functools import partial  # noqa: E402

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, flag, frame, per_trial, read_rows, run_groups, truthy  # noqa: E402
from .e16_mirrored_gyroscope import _load_all, sign_report_only  # noqa: E402

VERTICAL_COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'vertical_axis',
                    'vertical_axis_status', 'vertical_source', 'trial_up', 'votes_with', 'votes_against', 'error']
SIGN_COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'session', 'reason',
                'negated_main', 'negated_cross', 'split_off', 'report_only', 'trial_negated', 'length_mps2',
                'vertical_source', 'error']


def _vertical_rows(loader, trials):
    recs, rows = _load_all(loader, trials)
    if not recs:
        return rows
    tv = check_for_negated_accelerometer(recs, cross_check=False).thresholds['trial_verticals']
    for trial, v in tv.items():
        status = v['axis_status']
        detected = (v['axis'] != '+z') if status == 'named' else None
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), segment='', detected=detected,
                         estimate=v['axis_angle_deg'], unit='deg', vertical_axis=v['axis'] or '',
                         vertical_axis_status=status, vertical_source=v['source'],
                         trial_up='|'.join(f'{x:.3f}' for x in v['up']), votes_with=v['votes'][0],
                         votes_against=v['votes'][1], error=''))
    return rows


def check_vertical(loader, dataset: str, processes: int = 1, **options):
    """The reference's vertical of every trial, from its sensors' trial-averaged accelerometers."""
    return frame(run_groups(loader, _vertical_rows, processes), VERTICAL_COLUMNS)


def prevalence_vertical(rows) -> dict:
    """Trials whose vertical names an axis other than +z (one occurrence per trial). Trials whose vertical lies
    30-60 degrees from every axis are undecided and counted apart (suspicious_trials); trials without a common axis
    fall back to +z and are undecided too."""
    ok = read_rows(rows)
    hit = ok.detected.map(lambda v: flag(v) is True)
    suspicious = ok.vertical_axis_status == 'suspicious'
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[hit].nunique()),
                occurrences=int(ok.trial[hit].nunique()), occurrence_unit='trials',
                undecided=int(ok.trial[ok.detected.map(lambda v: flag(v) is None)].nunique()),
                suspicious_trials=int(ok.trial[suspicious].nunique()))


def correct_vertical(loader, rows):
    """Report only: the vertical is a property of the reference's coordinates, nothing is changed."""
    return loader


def _sign_rows(loader, trials, dataset=''):
    recs, rows = _load_all(loader, trials)
    if not recs:
        return rows
    res = check_for_negated_accelerometer(recs)
    tv = res.thresholds['trial_verticals']
    for seg, p in res.per_segment.items():
        sessions = {s['session']: s for s in p['sessions']}
        for trial, tr in p['per_trial'].items():
            s = sessions[tr['session']]
            participant = str(loader.participant_of(trial))
            rows.append(dict(trial=trial, participant=participant, segment=seg, detected=tr['negated'],
                             estimate=s.get('negated_share', float('nan')), unit='share of trials read negated',
                             session=tr['session'], reason=s.get('reason', ''), negated_main=s.get('negated_main'),
                             negated_cross=s.get('negated_cross'), split_off=s.get('split_off'),
                             report_only=sign_report_only(dataset, participant, seg),
                             trial_negated=tr['trial_negated'], length_mps2=tr['length_mps2'],
                             vertical_source=tv[trial]['source'], error=''))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """The accelerometer sign of every sensor, per participant and mounting session (both readings)."""
    return frame(run_groups(loader, partial(_sign_rows, dataset=dataset), processes), SIGN_COLUMNS)


def prevalence(rows) -> dict:
    """Trials of a negated sensor-session / negated sensor-sessions; sensors reported only are left out."""
    ok = read_rows(rows)
    ok = ok[~ok.report_only.map(truthy)]
    hit = ok.detected.map(lambda v: flag(v) is True)
    sessions = ok[hit][['participant', 'segment', 'session']].astype(str).drop_duplicates()
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[hit].nunique()),
                occurrences=len(sessions), occurrence_unit='sensor-sessions',
                undecided=int(ok.trial[ok.detected.map(lambda v: flag(v) is None)].nunique()))


def _negate(rec, rows):
    new, changed = rec.copy(), []
    for r in rows:
        if r['segment'] in new.segments and truthy(r.get('detected')) and not truthy(r.get('report_only')):
            new.segments[r['segment']].imu.acc = -new.segments[r['segment']].imu.acc
            changed.append(r['segment'])
    new.log_correction(ERROR, f'Negated the accelerometer of {len(changed)} segments: '
                       + (', '.join(changed) or 'none') + '.', negated=changed)
    return new


def correct(loader, rows):
    """The three accelerometer columns negated in every trial of a negated sensor-session."""
    return corrected(loader, _negate, per_trial(rows), 'negated accelerometer')


SPEC_VERTICAL = ErrorSpec(19, 'reference_vertical', 'report', 'reference vertical not +z')
SPEC = ErrorSpec(20, 'negated_accelerometer', 'fix', 'accelerometer negated')
PARTS = {19: dict(spec=SPEC_VERTICAL, check_dataset=check_vertical, prevalence=prevalence_vertical,
                  correct=correct_vertical),
         20: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_reference_vertical = make_handler(PARTS[19])
handle_negated_accelerometer = make_handler(PARTS[20])
