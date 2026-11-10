"""Errors 5 (clipping) and 7 (railed channel) without knowing the measurement range: a flat stretch at a channel's
extreme that the signal around it does not lead into.

Per channel and side (largest positive, largest negative values):
    1. Noise: the channel's own noise standard deviation, read from its quietest stretches (the NOISE_QUANTILE of the
       per-window robust second-difference noise over windows of NOISE_WINDOW_S), at least the quantisation noise
       step / sqrt(12) of coarsely stored data. Every tolerance below is a multiple of it, so every dataset gets its
       own scale.
    2. Plateaus: the top level is the highest value two consecutive samples both reach (one spike cannot set it).
       Runs of at least MIN_PLATEAU (2) consecutive samples within 2 * PLATEAU_SIGMA (3) noise of that level are the
       candidate plateaus (a noisy limit wanders by 3 noise either side of its level).
    3. Fit to the neighbours: a cubic (DEGREE 3; a parabola cannot follow a fast rise and a slow fall) is fitted to
       the FLANK (3) samples before and after the plateau and extended across it. A natural peak meets the plateau.
       The plateau does not fit, and is cut off, when the extended curve rises past the plateau's level by more than
       EXCEED_SIGMA (5) times the noise of that comparison (the noise and the fit's own uncertainty, combined):
       clipped at a limit; or when the plateau sits that far above the whole curve for at least RAILED_S: jumped to
       a rail (a shorter one is a sharp natural peak the cubic undershoots, as seen in Central Florida walking).
       The fit's own misfit to the flanks (residual) counts into that noise, so a curve that follows the flanks
       poorly must miss by more to decide.
    4. Undecided: a plateau at the edge of the trial or next to missing values (no flanks), or whose flanks the
       cubic does not fit (residual above EXCEED_SIGMA noise: too few samples per peak, e.g. a heel strike at a low
       rate) while the miss lies within that misfit. These are counted, not flagged, with the number of plateaus
       sharing the side's top level (a limit recurs at the same level, a natural maximum rarely does).
    4b. Rest: a plateau counts only while another channel of the same IMU varies during it, its standard deviation
       above PLATEAU_SIGMA times its noise (as in the dead-channel check). At rest every channel is flat, and a
       one-sided signal resting at its minimum looks like a limit; a range limit is reached in motion.
    4c. Smallest possible range: a limit lies at least MIN_LIMIT from zero, 1 g for an
       accelerometer (a smaller range could not measure gravity) and 125 deg/s for a gyroscope (the smallest range
       of common sensors). A cut-off plateau closer to zero is a natural top the cubic overshoots.
    4d. Recurrence: a range limit belongs to the sensor and is reached again. A cut-off plateau
       counts only when another cut-off plateau of the same channel and side, in the same trial or in another trial
       of the same participant (other_levels), lies within RECUR_SIGMA (6) noise of its level; otherwise 'single',
       reported, not flagged.
    4e. Not exceeded: a range limit cannot be exceeded, so a cut-off plateau counts only when no
       peak of the same sensor (any of its axes, either sign; the highest value two consecutive samples reach) in
       the same trial lies more than EXCEED_FRACTION (5 %, calibration differences between axes) beyond its level;
       otherwise 'exceeded', reported, not flagged. Only the same trial: the range may have been changed between
       trials. Checked before the recurrence, so an exceeded plateau does not vote.
    5. Clipping (error 5) or railed (error 7): a cut-off plateau shorter than RAILED_S (1 s) is clipping, a longer one a
       railed channel. Both are masked: clipping by this error's correction, railed channels by error 7's.
    6. Railed for most of the trial (a gravity window of 0.5-2 g instead flagged running, hands and data stored in
       g): a railed channel has no flanks for step 3. A channel whose samples lie
       within PLATEAU_SIGMA noise of its median for more than WHOLE_TRIAL (half) of the trial, while another channel
       of the IMU varies during those samples (as 4b), is railed when the median lies at least RAIL_LIMIT from zero:
       2 g for an accelerometer (an axis that holds still while the IMU moves can naturally only read gravity, at
       most 1 g; with 1 g, upright axes in Trento, Geneva and REEM-C were flagged) and 125 deg/s
       for a gyroscope. Its band samples are masked. The band is centred on the median, not the top value: a few
       samples above the rail would otherwise move the band off its bulk (2 of 56 Central Florida trials missed).
       Recurrence is not required: the dwell is itself the evidence.
Two further readings, reported as evidence, not used to decide:
    - pile-up: more samples within the top band than in the band of equal width below it; a clipped channel piles up
      at its limit, a natural signal thins out towards its extremes.
    - the median length of the accelerometer vector over the trial (near 9.81 m/s^2 for any orientation).

Exactly constant channels in the middle of their range are not this check's business: they are found by the dead
channel rule (e07_dead_channels.py), which needs no limit.
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording
from ._result import CheckResult, entry

ERROR = 'cut-off peaks'
MIN_PLATEAU = 2
PLATEAU_SIGMA = 3.0
EXCEED_SIGMA = 5.0
FLANK = 3
DEGREE = 3
RAILED_S = 1.0
NOISE_WINDOW_S = 0.5
NOISE_QUANTILE = 0.1
GRAVITY = 9.81
MIN_LIMIT = {'gyr': np.deg2rad(125.0), 'acc': GRAVITY}
RECUR_SIGMA = 2 * PLATEAU_SIGMA
WHOLE_TRIAL = 0.5
EXCEED_FRACTION = 0.05
RAIL_LIMIT = {'gyr': np.deg2rad(125.0), 'acc': 2 * GRAVITY}
CANDIDATE = ('cut off', 'single')  # passed the shape, rest and range tests; recurrence decides between them


def channel_noise(x: np.ndarray, rate_hz: float) -> float:
    """Noise standard deviation of one channel from its quietest windows, at least its quantisation noise."""
    x = np.asarray(x, float)
    finite = x[np.isfinite(x)]
    if len(finite) < 3:
        return float('nan')
    steps = np.abs(np.diff(np.unique(finite)))
    span = finite.max() - finite.min()
    steps = steps[steps > 1e-6 * span] if span > 0 else steps[:0]
    floor = steps.min() / np.sqrt(12) if len(steps) else 0.0
    w = max(int(round(NOISE_WINDOW_S * rate_hz)), 8)
    d2 = np.diff(x, 2)
    n = len(d2) // w
    if n == 0:
        per_window = np.array([np.nanmedian(np.abs(d2 - np.nanmedian(d2)))])
    else:
        blocks = d2[:n * w].reshape(n, w)
        per_window = np.nanmedian(np.abs(blocks - np.nanmedian(blocks, axis=1, keepdims=True)), axis=1)
    per_window = 1.4826 * per_window[np.isfinite(per_window) & (per_window > 0)] / np.sqrt(6)
    quiet = float(np.quantile(per_window, NOISE_QUANTILE)) if len(per_window) else 0.0
    return max(quiet, floor)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _fit_across(y: np.ndarray, a: int, b: int, sigma: float):
    """Cubic through the flanks of plateau y[a:b], extended across it: (largest predicted value, its standard
    error, residual standard deviation of the fit), or None without two full flanks."""
    if a - FLANK < 0 or b + FLANK > len(y):
        return None
    t = np.concatenate([np.arange(a - FLANK, a), np.arange(b, b + FLANK)]).astype(float)
    v = y[t.astype(int)]
    if not np.isfinite(v).all():
        return None
    centre, scale = (a + b - 1) / 2, max((b - a + 2 * FLANK) / 2, 1.0)
    X = np.vander((t - centre) / scale, DEGREE + 1)
    coef, *_ = np.linalg.lstsq(X, v, rcond=None)
    residual = float(np.sqrt(np.sum((X @ coef - v) ** 2) / (len(v) - DEGREE - 1)))
    Xp = np.vander((np.arange(a, b) - centre) / scale, DEGREE + 1)
    pred = Xp @ coef
    k = int(np.argmax(pred))
    cov = np.linalg.inv(X.T @ X)
    se = float(sigma * np.sqrt(Xp[k] @ cov @ Xp[k]))
    return float(pred[k]), se, residual


def side_plateaus(x: np.ndarray, sigma: float, sign: float, rate_hz: float) -> list[dict]:
    """Every candidate plateau on one side of one channel, with the fit's verdict."""
    y = sign * np.asarray(x, float)
    pair = np.fmin(y[:-1], y[1:])
    if not np.isfinite(pair).any() or not np.isfinite(sigma):
        return []
    top = float(np.nanmax(pair))
    band = np.isfinite(y) & (y >= top - 2 * PLATEAU_SIGMA * sigma)
    gap = np.zeros_like(band)  # one noisy sample dipping out of a long plateau does not split it
    gap[1:-1] = ~band[1:-1] & band[:-2] & band[2:] & np.isfinite(y[1:-1])
    band |= gap
    out = []
    for a, b in _runs(band):
        if b - a < MIN_PLATEAU:
            continue
        level = float(np.median(y[a:b]))
        fit = _fit_across(y, a, b, sigma)
        plateau = dict(first=a, samples=b - a, level=sign * level)
        if fit is None:
            out.append(dict(plateau, verdict='undecided', reason='no flanks'))
            continue
        peak, se, residual = fit
        noise = float(np.sqrt(se ** 2 + sigma ** 2 + residual ** 2))  # how far the fit's misfit lets it be off
        exceed = peak - level
        if exceed > EXCEED_SIGMA * noise:
            verdict = 'cut off'  # the signal continues past the plateau: clipped at a limit
        elif exceed < -EXCEED_SIGMA * noise and (b - a) >= RAILED_S * rate_hz:
            verdict = 'cut off'  # held far above everything its neighbours lead to, for a second: jumped to a rail
        elif residual > EXCEED_SIGMA * sigma:
            verdict = 'undecided'  # the cubic does not follow the flanks, and the gap is within its misfit
        else:
            verdict = 'fits'
        out.append(dict(plateau, verdict=verdict, reason='flanks not smooth' if verdict == 'undecided' else '',
                        exceed_noise=exceed / noise if noise > 0 else float('inf'), residual_noise=residual / sigma))
    for p in out:  # how many plateaus share this side's top level: a limit recurs, a natural maximum rarely does
        p['plateaus_at_level'] = len(out)
    return out


def pile_up(x: np.ndarray, sigma: float, sign: float) -> bool:
    """More samples in the top band than in the band of equal width just below it."""
    y = sign * np.asarray(x, float)
    pair = np.fmin(y[:-1], y[1:])
    if not np.isfinite(pair).any() or not np.isfinite(sigma) or sigma <= 0:
        return False
    top, width = float(np.nanmax(pair)), 2 * PLATEAU_SIGMA * sigma
    upper = np.sum(y >= top - width)
    lower = np.sum((y >= top - 2 * width) & (y < top - width))
    return bool(upper > lower)


def _others_move(others: np.ndarray, noise: np.ndarray, a: int, b: int) -> bool:
    """Does any other channel of the IMU vary during the plateau, its standard deviation above PLATEAU_SIGMA times
    its noise (as in the dead-channel check)? At rest every channel is quiet; a limit is reached in motion."""
    if others.shape[1] == 0:
        return True
    return bool(np.any(np.nanstd(others[a:b], axis=0) > PLATEAU_SIGMA * noise))


def recurring(records: list[dict], key) -> None:
    """Step 4d, in place: a candidate plateau stays 'cut off' when another candidate with the same key(record)
    (channel and side) lies within RECUR_SIGMA noise of its level, else becomes 'single'. Records marked external
    (from other trials) only vote; whole-trial rails are exempt."""
    groups = {}
    for p in records:
        if p['verdict'] in CANDIDATE and not p.get('whole_trial'):
            groups.setdefault(key(p), []).append(p)
    for group in groups.values():
        for p in group:
            if p.get('external'):
                continue
            agree = sum(abs(p['level'] - q['level']) <= RECUR_SIGMA * max(p['noise'], q['noise'])
                        for q in group if q is not p)
            p.update(verdict='cut off', reason='') if agree else p.update(verdict='single',
                                                                           reason='its level does not recur')


def cut_off_masks(records: list[dict], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Clipped and railed samples (any channel) of the records judged cut off."""
    clipped, railed = np.zeros(n, bool), np.zeros(n, bool)
    for p in records:
        if p['verdict'] == 'cut off' and not p.get('external'):
            long = p.get('whole_trial') or p['samples'] >= p['rail_samples']
            p['kind'] = 'railed' if long else 'clipping'
            (railed if long else clipped)[p['first']:p['first'] + p['samples']] = True
    return clipped, railed


def whole_trial_rail(x: np.ndarray, sigma: float, others: np.ndarray, noise: np.ndarray, floor: float) -> list[dict]:
    """Step 6: one record per run of band samples when the channel sits within PLATEAU_SIGMA noise of its median for
    more than WHOLE_TRIAL of the trial while another channel moves during those samples, with the median at least
    floor from zero; else none."""
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    if not finite.any() or not np.isfinite(sigma):
        return []
    level = float(np.median(x[finite]))
    band = finite & (np.abs(x - level) <= PLATEAU_SIGMA * sigma)
    fraction = band.sum() / finite.sum()
    if fraction <= WHOLE_TRIAL or abs(level) < floor or not _others_move(others[band], noise, 0, int(band.sum())):
        return []
    return [dict(first=a, samples=b - a, level=level, side='positive' if level > 0 else 'negative',
                 verdict='cut off', whole_trial=True, fraction=fraction, reason='at one level for most of the trial')
            for a, b in _runs(band)]


def peak_reach(x: np.ndarray) -> float:
    """The largest absolute value two consecutive samples both reach (one spike cannot set it)."""
    a = np.abs(np.asarray(x, float))
    pair = np.fmin(a[:-1], a[1:])
    return float(np.nanmax(pair)) if np.isfinite(pair).any() else float('nan')


def cut_off_samples(signal: np.ndarray, rate_hz: float, minimum_limit=None, other_levels: dict | None = None,
                    rail_limit=None, sensors=None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Samples inside a cut-off plateau of any column: (clipped, railed, one record per plateau found).
    signal: the channels of one IMU (gyroscope and accelerometer together, six columns), so that a plateau counts
    only while another channel moves; with a single column that condition is not applied.
    minimum_limit: per column, the smallest level a range limit can have (step 4c); None applies none.
    other_levels: {(column, side): [(level, noise), ...]} of the candidate plateaus of the same channels in the
    participant's other trials (step 4d); None counts only this trial.
    rail_limit: per column, the smallest level of a channel railed for most of the trial (step 6); None applies
    none.
    sensors: per column, which sensor it belongs to (step 4e compares a plateau with every column of its sensor);
    None treats each column as its own sensor."""
    signal = np.asarray(signal, float)
    n, m = signal.shape
    floor = np.zeros(m) if minimum_limit is None else np.asarray(minimum_limit, float)
    rail_floor = np.zeros(m) if rail_limit is None else np.asarray(rail_limit, float)
    noise = np.array([channel_noise(signal[:, c], rate_hz) for c in range(m)])
    sensors = np.arange(m) if sensors is None else np.asarray(sensors)
    reach = np.array([peak_reach(signal[:, c]) for c in range(m)])
    sensor_reach = np.array([np.nanmax(reach[sensors == sensors[c]]) for c in range(m)])
    records = []
    for axis in range(m):
        x = signal[:, axis]
        rest = [c for c in range(m) if c != axis]
        for p in whole_trial_rail(x, noise[axis], signal[:, rest], noise[rest], rail_floor[axis]):
            p.update(axis=axis, noise=noise[axis], pile_up=False, rail_samples=RAILED_S * rate_hz)
            records.append(p)
        for side, sign in (('positive', 1.0), ('negative', -1.0)):
            found = side_plateaus(x, noise[axis], sign, rate_hz)
            for p in found:
                p.update(axis=axis, side=side, noise=noise[axis], pile_up=pile_up(x, noise[axis], sign),
                         rail_samples=RAILED_S * rate_hz)
                if p['verdict'] == 'cut off' and not _others_move(signal[:, rest], noise[rest], p['first'],
                                                                  p['first'] + p['samples']):
                    p.update(verdict='fits', reason='the whole IMU is quiet (rest)')
                if p['verdict'] == 'cut off' and abs(p['level']) < floor[axis]:
                    p.update(verdict='fits', reason='closer to zero than any range')
                if p['verdict'] == 'cut off' and sensor_reach[axis] > (1 + EXCEED_FRACTION) * abs(p['level']):
                    p.update(verdict='exceeded', reason='a peak of the same sensor in this trial goes beyond it',
                             sensor_reach=sensor_reach[axis])
            records += found
    external = [dict(axis=axis, side=side, level=level, noise=s, verdict='cut off', external=True)
                for (axis, side), levels in (other_levels or {}).items() for level, s in levels]
    recurring(records + external, key=lambda p: (p['axis'], p['side']))
    clipped, railed = cut_off_masks(records, n)
    return clipped, railed, records


SENSORS = np.array([0, 0, 0, 1, 1, 1])  # gyroscope x, y, z, accelerometer x, y, z


def minimum_limits(limits: dict = MIN_LIMIT) -> np.ndarray:
    """Step 4c (or with RAIL_LIMIT step 6) for the six columns gyroscope x, y, z, accelerometer x, y, z."""
    return np.array([limits['gyr']] * 3 + [limits['acc']] * 3)


def candidate_levels(records: list[dict]) -> dict:
    """{(column, side): [(level, noise), ...]} of the candidate plateaus in records, for other_levels."""
    out = {}
    for p in records:
        if p['verdict'] in CANDIDATE and not p.get('whole_trial') and not p.get('external'):
            out.setdefault((p['axis'], p['side']), []).append((p['level'], p['noise']))
    return out


def median_specific_force(acc: np.ndarray) -> float:
    """Median length of the accelerometer vector over the trial (m/s^2)."""
    acc = np.asarray(acc, float)
    ok = np.isfinite(acc).all(1)
    return float(np.median(np.linalg.norm(acc[ok], axis=1))) if ok.any() else float('nan')


def check_for_cut_off_peaks(recording: Recording, other_levels: dict | None = None) -> CheckResult:
    """Per segment: clipped and railed samples (gyroscope and accelerometer), undecided and single plateaus.
    other_levels: {segment: candidate_levels(...)} from the participant's other trials (step 4d); each segment's
    candidates are returned as evidence ('candidates') so that the caller can collect them."""
    per_segment = {}
    for name, seg in recording.segments.items():
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        c, r, records = cut_off_samples(both, seg.imu.rate_hz, minimum_limits(), (other_levels or {}).get(name),
                                        minimum_limits(RAIL_LIMIT), SENSORS)
        per_segment[name] = summarise(records, c, r, len(both))
        per_segment[name].update(candidates=candidate_levels(records),
                                 median_specific_force=median_specific_force(seg.imu.acc))
    hit = [n for n, p in per_segment.items() if p['detected']]
    summary = f'Cut-off peaks or railed channels in {", ".join(hit)}.' if hit else 'No cut-off peaks found.'
    return CheckResult(ERROR, bool(hit), summary, per_segment,
                       dict(plateau_sigma=PLATEAU_SIGMA, exceed_sigma=EXCEED_SIGMA, flank=FLANK, railed_s=RAILED_S,
                            recur_sigma=RECUR_SIGMA, minimum_limit=MIN_LIMIT, whole_trial=WHOLE_TRIAL,
                            rail_limit=RAIL_LIMIT, exceed_fraction=EXCEED_FRACTION))


def summarise(records: list[dict], clipped: np.ndarray, railed: np.ndarray, n: int) -> dict:
    """The per-segment entry from one segment's plateau records and masks."""
    records = [dict(p, sensor='gyr' if p['axis'] < 3 else 'acc', axis=p['axis'] % 3) for p in records]
    cut = [p for p in records if p['verdict'] == 'cut off']
    on = {'gyr': np.zeros(n, bool), 'acc': np.zeros(n, bool)}
    for p in cut:
        on[p['sensor']][p['first']:p['first'] + p['samples']] = True
    bad = clipped | railed
    return entry(bool(bad.any()), int(bad.sum()), 'samples', clipped=int(clipped.sum()), railed=int(railed.sum()),
                 gyroscope_samples=int(on['gyr'].sum()), accelerometer_samples=int(on['acc'].sum()),
                 plateaus=len(records), cut_off=len(cut),
                 undecided=sum(p['verdict'] == 'undecided' for p in records),
                 single=sum(p['verdict'] == 'single' for p in records),
                 exceeded=sum(p['verdict'] == 'exceeded' for p in records),
                 whole_trial=sorted({f"{p['sensor']}{p['axis']}{p['side'][0]}" for p in cut if p.get('whole_trial')}),
                 cut_off_with_pile_up=sum(p['pile_up'] for p in cut),
                 channels=sorted({f"{p['sensor']}{p['axis']}{p['side'][0]}" for p in cut}),
                 levels={f"{p['sensor']}{p['axis']}{p['side'][0]}": p['level'] for p in cut},
                 longest=max((p['samples'] for p in cut), default=0))


def remove_cut_off_peaks(recording: Recording, other_levels: dict | None = None) -> Recording:
    """Return a copy with every clipped or railed sample marked invalid (all channels of that IMU sample)."""
    new = recording.copy()
    marked = {}
    for name, seg in new.segments.items():
        c, r, _ = cut_off_samples(np.hstack([seg.imu.gyr, seg.imu.acc]), seg.imu.rate_hz, minimum_limits(),
                                  (other_levels or {}).get(name), minimum_limits(RAIL_LIMIT), SENSORS)
        bad = (c | r) & seg.imu.valid
        seg.imu.valid = seg.imu.valid & ~bad
        marked[name] = int(bad.sum())
    new.log_correction(ERROR, f'Marked {sum(marked.values())} clipped or railed IMU samples invalid.',
                       samples_marked=marked)
    return new


# --- handler: one dataset at a time -------------------------------------------------------------------------------

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import by_participant, dumps, from_runs, loads, parallel_map, reader_of, runs, truthy  # noqa: E402


def plateaus_of(rec: Recording) -> dict:
    """{segment: (IMU samples, plateau records other than 'fits')}, recurrence counted within the trial only."""
    out = {}
    for name, seg in rec.segments.items():
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        _, _, records = cut_off_samples(both, seg.imu.rate_hz, minimum_limits(), None, minimum_limits(RAIL_LIMIT),
                                        SENSORS)
        out[name] = (len(both), [p for p in records if p['verdict'] != 'fits'])
    return out


def with_recurrence(found: dict) -> dict:
    """Step 4d across one participant's trials, in place: found {trial: {segment: (samples, records)}} as from
    plateaus_of; every candidate is judged against the candidates of the same channel and side in the other trials.
    Returns {trial: {segment: (clipped mask, railed mask, records)}}, each cut-off record with its 'kind'."""
    levels = {(t, s): candidate_levels(v[1]) for t, segs in found.items() for s, v in segs.items()}
    out = {}
    for trial, segs in found.items():
        out[trial] = {}
        for seg, (n, records) in segs.items():
            external = [dict(axis=a, side=side, level=lv, noise=sd, verdict='cut off', external=True)
                        for (t, s), lev in levels.items() if s == seg and t != trial
                        for (a, side), pairs in lev.items() for lv, sd in pairs]
            recurring(records + external, key=lambda p: (p['axis'], p['side']))
            out[trial][seg] = (*cut_off_masks(records, n), records)
    return out


def _check_participant(loader, group) -> list[dict]:
    _, trials = group
    found = {}
    for t in trials:
        rec = loader.load(t)
        if rec.segments:
            found[t] = plateaus_of(rec)
    rows = []
    for t, segs in with_recurrence(found).items():
        for seg, (clipped, railed, records) in segs.items():
            cut = [p for p in records if p['verdict'] == 'cut off']
            clip = [p for p in cut if p['kind'] == 'clipping']
            s = summarise(records, clipped, railed, len(clipped))
            rows.append(dict(trial=t, participant=str(loader.participant_of(t)), segment=seg,
                             detected=bool(clip), estimate=len(clip), unit='plateaus',
                             clipped_samples=int(clipped.sum()), railed_samples=int(railed.sum()),
                             railed_channels=len({(p['axis'], p['side']) for p in cut if p['kind'] == 'railed'}),
                             undecided=s['undecided'], single=s['single'], exceeded=s['exceeded'],
                             channels=';'.join(s['channels']), clipping_runs=dumps(runs(clipped)),
                             railed_runs=dumps(runs(railed))))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: the clipped plateaus (estimate; cut-off plateaus shorter than RAILED_S, every
    gyroscope or accelerometer channel and side counted apart), with recurrence counted across the participant's
    trials. Railed channels (longer plateaus, or a channel at one level for most of the trial) are listed in
    railed_channels and railed_runs but belong to error 7."""
    rows = []
    for out in parallel_map(loader, _check_participant, by_participant(loader), processes):
        if isinstance(out, dict):  # a participant's trials could not be read
            rows += [dict(trial=t, participant=str(loader.participant_of(t)), segment='', detected=None,
                          error=out['error']) for t in out['trial'][1]]
        else:
            rows += out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with a clipped plateau / clipped plateaus."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'plateaus', int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def _mask(rec: Recording, trial, *, stored) -> Recording:
    """Clipped samples marked invalid: from the stored runs ({trial: {segment: runs}}), or, where `stored` is None,
    found again in this trial alone (another reader than the check's: no recurrence across trials)."""
    if stored is not None and str(trial) not in stored:
        return rec
    new, marked = rec.copy(), {}
    for name, seg in new.segments.items():
        if stored is None:
            bad = cut_off_samples(np.hstack([seg.imu.gyr, seg.imu.acc]), seg.imu.rate_hz, minimum_limits(), None,
                                  minimum_limits(RAIL_LIMIT), SENSORS)[0]
        else:
            bad = from_runs(stored[str(trial)].get(name), len(seg.imu))
        bad &= seg.imu.valid
        seg.imu.valid = seg.imu.valid & ~bad
        marked[name] = int(bad.sum())
    new.log_correction('clipping', f'Marked {sum(marked.values())} clipped IMU samples invalid.',
                       samples_marked=marked)
    return new


def correct(loader, rows: pd.DataFrame):
    """The clipped samples marked invalid (all channels of that IMU sample)."""
    if len(rows) and 'reader' in rows and rows.reader.iloc[0] != reader_of(loader):
        return CorrectedLoader(loader, partial(_mask, stored=None), 'clipping')
    stored = {}
    for r in rows[rows.detected.map(truthy)].itertuples():
        stored.setdefault(str(r.trial), {})[r.segment] = loads(r.clipping_runs, [])
    if not stored:
        return loader
    return CorrectedLoader(loader, partial(_mask, stored=stored), 'clipping')


SPEC = ErrorSpec(5, 'clipping', 'mask', 'clipping at the measurement range')
PARTS = {5: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_clipping = make_handler(PARTS[5])
