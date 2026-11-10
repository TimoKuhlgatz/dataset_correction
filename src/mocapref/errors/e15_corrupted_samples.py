"""Error 15: corrupted single samples. One or two IMU samples hold a value the sensor cannot have measured, while the
samples on both sides agree with each other (e.g. two consecutive gyroscope samples at +5,169 and -5,117 deg/s between
neighbours at 25 deg/s).

Place in the order: after the time alignment, so the aligned optical reference judges each suspicious sample; before
every fit (bias, signs, mounting), which one such sample would pull. Mounting, gyroscope offset and sign errors are
not corrected yet, so only rotation speeds (vector norms) are compared with the reference. All four criteria must
hold:

1. Needle. A run of 1 or 2 (MAX_RUN) samples leaves its two outer neighbours and comes back: the out-and-back jump
   (for each sample of the run the smaller vector distance to the two outer neighbours, then the smallest over the
   run) is at least NEEDLE = 0.9 times the largest rotation speed among the run and its outer neighbours. Runs of 2
   are looked for first, so a corrupted pair is judged against the samples outside it.
2. Score. The whole-trial score of each sample of the run, (|gyroscope| - |reference rate| - median) / spread, with
   median and spread (1.4826 median absolute deviation) of that difference over all valid samples of the segment,
   exceeds SCORE = 50 in absolute value, all in the same direction.
3. Reference. The reference does not show the jump: at every sample of the run the reference speed lies closer to
   the mean speed of the two outer neighbours than the gyroscope speed does.
4. One sensor. Corruption is a fault of one sensor; an impact (a foot stomp) shakes several sensors at once. No other
   sensor of the same recording scores above SCORE within OTHER_WINDOW = 1 sample of the run.

Only upward needles are counted and marked invalid. Downward needles that meet criteria 1-3 are reported but not
marked: in the data they are almost all real reversals of the direction of rotation, which this rule cannot tell from
a dropout. Where the reference is missing at an upward needle that would score high even against a resting reference,
the needle is counted as not examined and not marked.

The accelerometer cannot be judged by the reference at this stage. Its candidates (needles by the same shape rule) are
confirmed only where the gyroscope of the same IMU has a corrupted run at the same samples.

Known limitation: an impact the gyroscope catches and the smoother reference does not, and which the other sensors
feel only weakly, still passes.

Correction: the samples of a corrupted run are marked invalid (all channels of that IMU sample). NEEDLE and SCORE were
chosen after scanning the datasets (exploratory).
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording
from ..rotations import reference_angular_rate
from ._result import CheckResult, entry

NEEDLE = 0.9
SCORE = 50.0
MAX_RUN = 2
OTHER_WINDOW = 1


def _robust_spread(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def needle_runs(v: np.ndarray, valid: np.ndarray, needle: float = NEEDLE, max_run: int = MAX_RUN) -> list[dict]:
    """Runs of 1..max_run valid samples of a (N, 3) signal whose out-and-back jump (smallest vector distance of any
    run sample to either outer neighbour) is at least `needle` times the largest norm among the run and its outer
    neighbours. Longer runs are taken first, so two consecutive corrupted samples are judged against the samples
    outside the pair, not against each other (each sample of the TUK pair is also a needle on its own); a pair of one
    needle and one normal sample never qualifies, since the normal sample does not leave its outer neighbours. A
    shorter run overlapping one already found is skipped. One record per run: first sample, length, out-and-back jump,
    needle share."""
    ok = valid & np.isfinite(v).all(1)
    norm = np.linalg.norm(np.where(ok[:, None], v, 0.0), axis=1)
    n = len(v)
    taken = np.zeros(n, bool)
    runs = []
    for length in range(max_run, 0, -1):
        firsts = np.arange(1, n - length)
        if len(firsts) == 0:
            continue
        before, after = firsts - 1, firsts + length
        good = ok[before] & ok[after]
        jump = np.full(len(firsts), np.inf)
        top = np.maximum(norm[before], norm[after])
        for j in range(length):
            i = firsts + j
            good &= ok[i]
            jump = np.minimum(jump, np.minimum(np.linalg.norm(v[i] - v[before], axis=1),
                                               np.linalg.norm(v[i] - v[after], axis=1)))
            top = np.maximum(top, norm[i])
        with np.errstate(invalid='ignore', divide='ignore'):
            share = jump / top
        hit = good & (top > 0) & (share >= needle)
        for a, m, s in zip(firsts[hit], jump[hit], share[hit]):
            if taken[a:a + length].any():
                continue
            taken[a:a + length] = True
            runs.append(dict(first=int(a), length=length, jump=float(m), share=float(s)))
    return sorted(runs, key=lambda r: r['first'])


def reference_speed_at(segment) -> np.ndarray:
    """Norm of the reference angular rate (rad/s) at each IMU sample time; NaN where the reference is missing."""
    ref, imu = segment.reference, segment.imu
    if len(ref) < 2:
        return np.full(len(imu), np.nan)
    times, rates = reference_angular_rate(ref.orientation, ref.rate_hz, ref.convention)
    speed = np.linalg.norm(rates, axis=1)
    speed[~(ref.valid[:-1] & ref.valid[1:])] = np.nan  # np.interp spreads NaN to the intervals touching it
    return np.interp(imu.times, times + ref.start_s, speed, left=np.nan, right=np.nan)


def _scores(segment) -> dict:
    """Speed, reference speed, residual score and its spread for one segment (rad/s; score dimensionless)."""
    imu = segment.imu
    speed = np.linalg.norm(imu.gyr, axis=1)
    ref = reference_speed_at(segment)
    residual = speed - ref
    residual[~imu.valid] = np.nan
    centre = float(np.nanmedian(residual)) if np.isfinite(residual).any() else np.nan
    spread = _robust_spread(residual)
    judged = bool(np.isfinite(spread) and spread > 0)
    score = (residual - centre) / spread if judged else np.full(len(speed), np.nan)
    return dict(speed=speed, ref=ref, residual=residual, centre=centre, spread=spread, judged=judged, score=score,
                times=imu.times)


def _other_sensors(own_times: np.ndarray, others: dict, window: int = OTHER_WINDOW) -> tuple[float, str]:
    """Highest score of any other sensor within `window` samples of the given IMU times, and that sensor's name."""
    best, who = -np.inf, ''
    for name, o in others.items():
        if not o['judged'] or len(o['times']) == 0:
            continue
        idx = np.searchsorted(o['times'], own_times)
        lo, hi = max(0, int(idx.min()) - window), min(len(o['times']), int(idx.max()) + window + 1)
        part = o['score'][lo:hi]
        if np.isfinite(part).any() and np.nanmax(part) > best:
            best, who = float(np.nanmax(part)), name
    return best, who


def judge_segment(segment, needle: float = NEEDLE, score: float = SCORE, max_run: int = MAX_RUN,
                  others: dict | None = None, window: int = OTHER_WINDOW, own: dict | None = None) -> dict:
    """Needles, their scores and the reference's verdict, and the mask of samples to mark invalid, for one segment.
    `others` holds _scores() of the other sensors of the recording (criterion 4); None skips that criterion. Only
    the runs that matter are kept: upward candidates (examined or not) and downward needles meeting criteria 1-3."""
    imu = segment.imu
    own = own if own is not None else _scores(segment)
    speed, ref, residual, centre, spread = own['speed'], own['ref'], own['residual'], own['centre'], own['spread']
    mask = np.zeros(len(imu), bool)
    gyro_runs = []
    for run in needle_runs(imu.gyr, imu.valid, needle, max_run):
        a, length = run['first'], run['length']
        idx = np.arange(a, a + length)
        if not own['judged']:
            continue  # no reference anywhere in the segment: nothing can be judged (examined_fraction shows it)
        s = own['score'][idx]
        if not np.isfinite(s).all():
            if np.min((speed[idx] - centre) / spread) > score:  # would score high even against a resting reference
                gyro_runs.append(dict(run, direction='up', examined=False, score=np.nan, reference_agrees=None,
                                      impact=None, corrupted=False))
            continue
        direction = 'up' if s.min() > score else ('down' if s.max() < -score else None)
        if direction is None:
            continue
        neighbours = 0.5 * (speed[a - 1] + speed[a + length])
        agrees = bool(np.all(np.abs(ref[idx] - neighbours) < np.abs(speed[idx] - neighbours)))
        if direction == 'down' and not agrees:
            continue  # a downward needle the reference also shows: real motion, not kept
        entry_ = dict(run, direction=direction, examined=True, score=float(s.min() if direction == 'up' else s.max()),
                      reference_agrees=agrees, impact=None)
        if direction == 'up' and agrees and others is not None:
            other_max, other_name = _other_sensors(imu.times[idx], others, window)
            entry_.update(other_max_score=other_max if np.isfinite(other_max) else np.nan,
                          other_segment=other_name, impact=bool(other_max > score))
        entry_['corrupted'] = direction == 'up' and agrees and not entry_['impact']
        gyro_runs.append(entry_)
        if entry_['corrupted']:
            mask[idx] = True
    for run in gyro_runs:
        run['speed_dps'] = [float(np.degrees(speed[i])) for i in range(run['first'], run['first'] + run['length'])]
        run['neighbours_dps'] = [float(np.degrees(speed[run['first'] - 1])),
                                 float(np.degrees(speed[run['first'] + run['length']]))]
        run['reference_dps'] = [float(np.degrees(ref[i])) for i in range(run['first'], run['first'] + run['length'])]
    gyro_mask = mask.copy()
    acc_runs = needle_runs(imu.acc, imu.valid, needle, max_run)
    for run in acc_runs:
        run['corrupted'] = bool(gyro_mask[run['first']:run['first'] + run['length']].any())
        if run['corrupted']:
            mask[run['first']:run['first'] + run['length']] = True
    return dict(gyro_runs=gyro_runs, acc_runs=acc_runs, mask=mask & imu.valid,
                examined_fraction=float(np.isfinite(residual).mean()) if len(residual) else 0.0,
                residual_spread_dps=float(np.degrees(spread)))


def judge_recording(recording: Recording, needle: float = NEEDLE, score: float = SCORE, max_run: int = MAX_RUN,
                    window: int = OTHER_WINDOW) -> dict:
    """judge_segment for every segment, each with the other sensors of the recording for criterion 4."""
    scores = {name: _scores(seg) for name, seg in recording.segments.items()}
    return {name: judge_segment(seg, needle, score, max_run, {n: v for n, v in scores.items() if n != name}, window,
                                scores[name]) for name, seg in recording.segments.items()}


def check_for_corrupted_samples(recording: Recording, needle: float = NEEDLE, score: float = SCORE,
                                max_run: int = MAX_RUN, window: int = OTHER_WINDOW,
                                judged: dict | None = None) -> CheckResult:
    """Corrupted single samples per segment: detected True if an upward needle is confirmed by the reference and no
    other sensor shows an impact, False if none is, None if an upward candidate could not be examined (or the segment
    has no reference) and none was confirmed. Downward needles and impacts are reported in the evidence, never
    counted. `judged`: judge_recording's result, where already computed."""
    per_segment = {}
    judged = judge_recording(recording, needle, score, max_run, window) if judged is None else judged
    for name, j in judged.items():
        seg = recording[name]
        g, a = j['gyro_runs'], j['acc_runs']
        up = [r for r in g if r['direction'] == 'up']
        corrupted = [r for r in up if r['corrupted']]
        not_examined = [r for r in up if not r['examined']]
        impacts = [r for r in up if r['impact']]
        down = [r for r in g if r['direction'] == 'down']
        no_reference = j['examined_fraction'] == 0 and len(seg.imu) > 0
        detected = True if corrupted else (None if not_examined or no_reference else False)
        per_segment[name] = entry(
            detected, int(j['mask'].sum()), 'samples',
            gyro_candidates=len(up) - len(not_examined), gyro_corrupted=len(corrupted),
            gyro_not_examined=len(not_examined), gyro_impacts=len(impacts),
            gyro_rejected=len(up) - len(corrupted) - len(not_examined) - len(impacts),
            down_reported=len(down),
            acc_candidates=len(a), acc_confirmed=sum(r['corrupted'] for r in a),
            acc_unconfirmed=sum(not r['corrupted'] for r in a),
            examined_fraction=j['examined_fraction'], residual_spread_dps=j['residual_spread_dps'],
            gyro_runs=g, acc_runs=[r for r in a if r['corrupted']])
    decisions = [p['detected'] for p in per_segment.values()]
    hit = [n for n, p in per_segment.items() if p['detected']]
    detected = True if hit else (None if None in decisions else False)
    if hit:
        summary = f'Corrupted samples confirmed by the reference in {", ".join(hit)}.'
    elif detected is None:
        summary = 'No corrupted sample confirmed; some candidates fall where the reference is missing.'
    else:
        summary = 'No corrupted samples found.'
    down = sum(p['down_reported'] for p in per_segment.values())
    impacts = sum(p['gyro_impacts'] for p in per_segment.values())
    if impacts:
        summary += f' {impacts} needle(s) rejected as impacts (another sensor also scores high).'
    if down:
        summary += f' {down} downward needle(s) reported, not counted.'
    return CheckResult('corrupted samples', detected, summary, per_segment,
                       dict(needle=needle, score=score, max_run=max_run, other_window=window))


def remove_corrupted_samples(recording: Recording, needle: float = NEEDLE, score: float = SCORE,
                             max_run: int = MAX_RUN, window: int = OTHER_WINDOW) -> Recording:
    """Return a copy with the samples of every confirmed (upward, one-sensor) corrupted run marked invalid (all
    channels)."""
    judged = judge_recording(recording, needle, score, max_run, window)
    new = recording.copy()
    marked = {}
    for name, seg in new.segments.items():
        bad = judged[name]['mask']
        seg.imu.valid = seg.imu.valid & ~bad
        marked[name] = int(bad.sum())
    new.log_correction('corrupted samples', f'Marked {sum(marked.values())} corrupted IMU samples invalid '
                                            '(upward needles of one sensor, confirmed by the reference after the time '
                                            'alignment).', samples_marked=marked)
    return new


# Dataset level: one row per trial and segment; the marked samples are kept as runs [first, length] (JSON).

import json  # noqa: E402

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, frame, loads, number, per_trial, read_rows, run_trials  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'samples_marked', 'gyro_corrupted',
           'gyro_impacts', 'gyro_not_examined', 'down_reported', 'acc_confirmed', 'marked_runs', 'error']


def _runs(mask: np.ndarray) -> list[list[int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    return [[int(r[0]), len(r)] for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)]


def _check_trial(loader, trial):
    rec = loader.load(trial)
    if not rec.segments:
        raise ValueError('no segment has both an IMU and a reference')
    judged = judge_recording(rec)
    c = check_for_corrupted_samples(rec, judged=judged)
    return [dict(trial=trial, participant=str(loader.participant_of(trial)), segment=name, detected=p['detected'],
                 estimate=p['estimate'], unit='samples', samples_marked=p['estimate'],
                 gyro_corrupted=p['gyro_corrupted'], gyro_impacts=p['gyro_impacts'],
                 gyro_not_examined=p['gyro_not_examined'], down_reported=p['down_reported'],
                 acc_confirmed=p['acc_confirmed'], marked_runs=json.dumps(_runs(judged[name]['mask'])), error='')
            for name, p in c.per_segment.items()]


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """Corrupted single samples in every trial (meant to run after the time alignment)."""
    return frame(run_trials(loader, _check_trial, processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials with a marked sample / corrupted runs of one or two samples."""
    ok = read_rows(rows)
    marked = ok.samples_marked.map(number).fillna(0) > 0
    undecided = ok.detected.map(lambda v: v is None or str(v) in ('', 'None', 'nan'))
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[marked].nunique()),
                occurrences=int(ok.gyro_corrupted.map(number).fillna(0).sum()), occurrence_unit='runs',
                undecided=int(ok.trial[undecided].nunique()))


def _mask(rec, rows):
    new, marked = rec.copy(), {}
    for r in rows:
        runs = loads(r.get('marked_runs'), [])
        if r['segment'] not in new.segments or not runs:
            continue
        imu = new.segments[r['segment']].imu
        bad = np.zeros(len(imu), bool)
        for first, n in runs:
            bad[first:first + n] = True
        imu.valid = imu.valid & ~bad
        marked[r['segment']] = int(bad.sum())
    new.log_correction('corrupted samples', f'Marked {sum(marked.values())} corrupted IMU samples invalid.',
                       samples_marked=marked)
    return new


def correct(loader, rows):
    """The samples of every confirmed corrupted run marked invalid (all channels of that IMU sample)."""
    return corrected(loader, _mask, per_trial(rows), 'corrupted samples')


SPEC = ErrorSpec(15, 'corrupted_samples', 'mask', 'corrupted single samples')
PARTS = {15: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_corrupted_samples = make_handler(PARTS[15])
