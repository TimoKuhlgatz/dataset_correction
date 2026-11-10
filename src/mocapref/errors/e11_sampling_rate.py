"""Error 11: the sampling rate of an IMU is not the one declared.

Two different things go wrong, and they need different checks:

Kind 1, the time axis is stretched (`check_for_time_scale`): the samples are read at the declared rate but were taken
at another one, so every motion appears too fast or too slow by one factor f = true rate / declared rate. The clock
drift check (error 13) is the fine version of this (f within a few hundred ppm of 1); a gross factor makes the two
speed curves disagree everywhere, so the offset and drift checks return "undecided", not "detected".
    How we catch it: the rotation speed of the gyroscope and of the reference (independent of mounting and sign
    errors) are cross-correlated for every factor of a log grid (1/16 to 16), the IMU time axis divided by the
    factor, the best shift searched within +-10 s; coarse to fine on windows that double until the whole recording
    is used.
    Decision (method='rank-blocks', default): ranks in every correlation; a segment's factor is trusted if it is not
    at the edge of the grid and, with the IMU time axis divided by it, the offsets of 5 blocks lie on a straight line
    within half a reference frame (a remaining fine drift is error 13's business); it deviates if the factor is more
    than 2 % from 1 and at the declared rate the 5 blocks do not agree on one offset (error 12's rule). The recording
    is detected if most trusted segments deviate; its factor is the median over trusted segments.
    Former decision (method='peak'): a factor trusted at a peak correlation of 0.9, or 0.7 with three agreeing
    segments; it deviates if more than 2 % from 1 with a correlation at least 0.1 higher than at f = 1.
    Limit: IMU against reference measures only the ratio of the two time scales, not which stream is wrong.
    Correction (`remove_time_scale`): the IMU rate becomes declared rate x f; no sample changed.

Kind 2, the stream was resampled (`check_for_effective_rate`): the time axis is right, but the samples were computed
from samples at another rate, by repeating samples (zero-order hold) or by interpolating. Needs no reference.
Measured per IMU, on the six channels together:
    hold_fraction: share of samples equal to the previous one on all six channels (zero-order hold leaves 1 - r0/r).
    linear_fraction: share of samples whose second difference is zero (to rounding): linear interpolation from
        r0 < r/2 leaves about 1 - 2 r0/r of them; a real sensor with noise almost none.
    line_ratio, line_hz: the strongest narrow line of the spectrum of |second difference| above 10 Hz, as its height
        over the median of the spectrum within +-1 Hz, and its frequency. Decided only with at least 20 frequency
        points in that neighbourhood (recordings of about 12.5 s or more). Interpolation from any rate puts the knots
        of the original grid at a fixed pattern, so the second differences repeat with its period (60 -> 100 Hz:
        every 5 samples, a line at 20 Hz). Without a pattern the highest bin lies 10-20 above the median; a pattern
        gives 10^3-10^5.
    top_band_ratio: gyroscope power in the top fifth below the Nyquist frequency over the power at 30-50 % of it.
        Band-limited upsampling leaves that band numerically empty (below 1e-6).
    Two outcomes, reported separately: resampled (line_ratio >= 200; clean datasets reach at most about 120), and
    below declared (hold_fraction >= 0.05, or linear_fraction >= 0.2, or top_band_ratio < 1e-6); the effective rate
    is estimated from the zero-order hold only (declared x (1 - hold_fraction)). No correction: the timing is right;
    the stream only carries less (or other) information than its rate suggests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..recording import Recording, Segment
from ._result import CheckResult, entry
from .e12_time_offset import (block_agreement, block_offsets, blocks_on_a_line, estimate_time_lag,
                                     speed_curves)

ERROR_SCALE = 'time scale (rate not as declared)'
ERROR_EFFECTIVE = 'resampled, or effective rate below the declared rate'

FACTOR_RANGE = (1 / 16, 16.0)
FIRST_WINDOW_S = 30.0
MIN_WINDOW_S = 10.0
STEP_TIMES_WINDOW = 0.05  # grid step in ln f times the window length (s); measured on synthetic motion of 0.2-2 Hz:
# f off by 0.0024 on 60 s already lowers the peak from 1.0 to 0.89, so the peak's half width is about 0.1 / window
COARSE_GRID_HZ = 50.0
FINE_GRID_HZ = 200.0
MIN_NEIGHBOUR_BINS = 20  # frequency points within +-1 Hz of a line; bins are 1 / duration apart, so about 12.5 s


def _lag_at(curves, factor, window_s, max_offset_s, grid_hz, ranks=False):
    """Cross-correlation of the speed curves with the IMU time axis divided by `factor`, on true times up to
    `window_s` after the start (None: the whole recording). Returns the LagEstimate or None if the span is too short."""
    imu_t, imu_v, ref_t, ref_v = curves
    t_imu = imu_t[0] + (imu_t - imu_t[0]) / factor
    if window_s is not None:
        end = min(t_imu[0], ref_t[0]) + window_s
        keep_imu, keep_ref = t_imu <= end, ref_t <= end + max_offset_s
        t_imu, imu_v, ref_t, ref_v = t_imu[keep_imu], imu_v[keep_imu], ref_t[keep_ref], ref_v[keep_ref]
    if len(t_imu) < 10 or len(ref_t) < 10:
        return None
    span = min(t_imu[-1], ref_t[-1]) - max(t_imu[0], ref_t[0])
    if span < MIN_WINDOW_S:
        return None
    return estimate_time_lag(t_imu, imu_v, ref_t, ref_v, max_offset_s, grid_hz, ranks=ranks)


def _blocks_agree(curves, factor, offset_s, tolerance_s, allow_drift=False) -> bool:
    """Block agreement (offset check's rule) with the IMU time axis divided by `factor` (method='rank-blocks').

    allow_drift=True: the block offsets may lie on a straight line (a residual clock drift, left to the clock-drift
    check); trusted if twice the standard error of the residuals around that line is within the tolerance.
    Without it the offset check's own rule: the blocks agree on one constant offset."""
    imu_t, imu_v, ref_t, ref_v = curves
    t_imu = imu_t[0] + (imu_t - imu_t[0]) / factor
    est = estimate_time_lag(t_imu, imu_v, ref_t, ref_v, 10.0, ranks=True)
    est.offset_s = offset_s if np.isfinite(offset_s) else est.offset_s
    b = block_offsets(t_imu, imu_v, ref_t, ref_v, est.offset_s)
    est.block_offsets_s = tuple(b)
    if not allow_drift:
        return block_agreement(est, tolerance_s)
    return bool(not est.at_search_limit and blocks_on_a_line(b, tolerance_s))


def estimate_segment_time_scale(segment: Segment, factor_range=FACTOR_RANGE, max_offset_s: float = 10.0,
                                candidates: int = 3, ranks: bool = False) -> dict:
    """Best time-scale factor f (true rate / declared rate) of one segment's IMU against its reference.

    Returns dict(factor, peak_correlation, correlation_at_1 (f = 1 on the whole recording), offset_s, at_limit,
    second_factor, second_correlation (best coarse candidate at least 10 % away from the best), evaluations)."""
    imu_t, imu_v, ref_t, ref_v = speed_curves(segment)
    block = int(segment.imu.rate_hz // (2 * FINE_GRID_HZ))
    if block > 1:  # average high-rate speed in blocks first (a low-pass); the search only needs a few Hz of motion
        n = len(imu_t) // block * block
        imu_t = imu_t[:n].reshape(-1, block).mean(1)
        imu_v = np.nanmean(imu_v[:n].reshape(-1, block), axis=1)
    curves = imu_t, imu_v, ref_t, ref_v
    duration = max(imu_t[-1] - imu_t[0], ref_t[-1] - ref_t[0])
    window = min(FIRST_WINDOW_S, duration)
    lo, hi = np.log(factor_range[0]), np.log(factor_range[1])
    step = STEP_TIMES_WINDOW / window
    grid = np.arange(lo, hi + step / 2, step)
    coarse = []
    for g in grid:
        est = _lag_at(curves, np.exp(g), window, max_offset_s, COARSE_GRID_HZ, ranks)
        coarse.append(est.peak_correlation if est is not None else -np.inf)
    coarse = np.array(coarse)
    evaluations = len(grid)
    order = np.argsort(coarse)[::-1]
    picked = []
    for i in order:  # the best local candidates, at least 5 grid steps apart
        if not np.isfinite(coarse[i]) or len(picked) == candidates:
            break
        if all(abs(i - j) > 5 for j in picked):
            picked.append(i)
    if not picked:
        return dict(factor=float('nan'), peak_correlation=float('nan'), correlation_at_1=float('nan'),
                    offset_s=float('nan'), at_limit=False, second_factor=float('nan'),
                    second_correlation=float('nan'), evaluations=evaluations)
    best_coarse = grid[picked[0]]
    away = np.abs(grid - best_coarse) > np.log(1.1)
    second = int(np.argmax(np.where(away, coarse, -np.inf))) if away.any() else picked[0]
    refined = []
    for i in picked:  # coarse to fine: double the window, refine within +-2 of the previous steps
        centre, w, s = grid[i], window, step
        est = None
        while True:
            w = min(2 * w, duration)
            full = w >= duration
            new_step = STEP_TIMES_WINDOW / w
            local = np.arange(centre - 2 * s, centre + 2 * s + new_step / 2, new_step)
            results = [(g, _lag_at(curves, np.exp(g), None if full else w, max_offset_s, FINE_GRID_HZ, ranks))
                       for g in local]
            evaluations += len(local)
            results = [(g, e) for g, e in results if e is not None]
            if not results:
                break
            centre, est = max(results, key=lambda r: r[1].peak_correlation)
            s = new_step
            if full:
                break
        if est is not None:
            refined.append((centre, est))
    at_one = _lag_at(curves, 1.0, None, max_offset_s, FINE_GRID_HZ, ranks)
    if at_one is not None:
        refined.append((0.0, at_one))  # the declared rate itself is always a candidate
    if not refined:
        g, est = best_coarse, None
    else:
        g, est = max(refined, key=lambda r: r[1].peak_correlation)
    return dict(factor=float(np.exp(g)),
                peak_correlation=est.peak_correlation if est is not None else float(coarse[picked[0]]),
                correlation_at_1=at_one.peak_correlation if at_one is not None else float('nan'),
                offset_s=est.offset_s if est is not None else float('nan'),
                at_limit=bool(g <= lo + step or g >= hi - step),
                second_factor=float(np.exp(grid[second])), second_correlation=float(coarse[second]),
                evaluations=evaluations, curves=curves,
                offset_at_1_s=at_one.offset_s if at_one is not None else float('nan'))


@dataclass
class TimeScaleResult(CheckResult):
    factor: float = float('nan')  # true rate / declared rate, median over trusted segments
    spread: float = float('nan')  # largest / smallest trusted factor


def check_for_time_scale(recording: Recording, factor_range=FACTOR_RANGE, max_offset_s: float = 10.0,
                         min_peak_correlation: float = 0.9, min_agreeing_correlation: float = 0.7,
                         min_agreeing: int = 3, tolerance: float = 0.02,
                         min_gain: float = 0.1, method: str = 'rank-blocks') -> TimeScaleResult:
    """Is the IMU time axis stretched by a gross factor against the reference (kind 1 of the module description)?

    method='rank-blocks' (default): ranks in every correlation; a segment's factor is
    trusted if, with the IMU time axis divided by it, the offsets of 5 blocks lie on a straight line within half a reference frame (a residual fine drift is the
    clock-drift check's business); it deviates if the factor is more than `tolerance` from 1 and at the declared rate
    (factor 1) the blocks do not agree on one offset (the offset check's rule). method='peak': the former rule
    (peak 0.9, or 0.7 with three agreeing segments; min_peak_correlation, min_agreeing_correlation, min_agreeing,
    min_gain apply only to it)."""
    if method == 'rank-blocks':
        return _time_scale_rank_blocks(recording, factor_range, max_offset_s, tolerance)
    per_segment, trusted = {}, []
    estimates = {name: estimate_segment_time_scale(seg, factor_range, max_offset_s)
                 for name, seg in recording.segments.items()}
    for e in estimates.values():
        e.pop('curves', None)
    usable = {n for n, e in estimates.items() if np.isfinite(e['peak_correlation']) and not e['at_limit']}
    fair = [estimates[n]['factor'] for n in usable if estimates[n]['peak_correlation'] >= min_agreeing_correlation]
    for name, est in estimates.items():
        fair_here = name in usable and est['peak_correlation'] >= min_agreeing_correlation
        est['agreeing_segments'] = int(sum(abs(np.log(f / est['factor'])) <= np.log(1 + tolerance) for f in fair)
                                       if fair_here else 0)
        ok = name in usable and (est['peak_correlation'] >= min_peak_correlation
                                 or est['agreeing_segments'] >= min_agreeing)
        deviates = None
        if ok:
            gain = est['peak_correlation'] - (est['correlation_at_1'] if np.isfinite(est['correlation_at_1']) else -1)
            deviates = bool(abs(np.log(est['factor'])) > np.log(1 + tolerance) and gain >= min_gain)
            trusted.append((est['factor'], deviates))
        per_segment[name] = entry(deviates, est['factor'], 'factor', est['peak_correlation'], **est)
    thresholds = dict(factor_range=factor_range, max_offset_s=max_offset_s,
                      min_peak_correlation=min_peak_correlation, min_agreeing_correlation=min_agreeing_correlation,
                      min_agreeing=min_agreeing, tolerance=tolerance, min_gain=min_gain)
    if not trusted:
        return TimeScaleResult(ERROR_SCALE, None, 'No segment gave a trustworthy correlation peak at any factor, so '
                               'the time scale cannot be decided.', per_segment, thresholds, unit='factor')
    factors = np.array([f for f, _ in trusted])
    factor = float(np.median(factors))
    detected = sum(d for _, d in trusted) > len(trusted) / 2
    summary = (f'The IMU time axis fits the reference best when divided by {factor:.4g} (median over {len(trusted)} '
               f'of {len(per_segment)} segments; true rate = declared x {factor:.4g}). '
               + ('This deviates' if detected else 'This does not deviate') + f' from 1 by more than {tolerance:.0%} '
               'with a clearly better fit than at the declared rate.')
    return TimeScaleResult(ERROR_SCALE, detected, summary, per_segment, thresholds, estimate=factor, unit='factor',
                           factor=factor, spread=float(factors.max() / factors.min()))


def _time_scale_rank_blocks(recording, factor_range, max_offset_s, tolerance) -> TimeScaleResult:
    tol_s = 0.5 / min(s.reference.rate_hz for s in recording.segments.values())
    per_segment, trusted = {}, []
    for name, seg in recording.segments.items():
        est = estimate_segment_time_scale(seg, factor_range, max_offset_s, ranks=True)
        curves = est.pop('curves', None)
        ok = bool(curves is not None and np.isfinite(est['factor']) and not est['at_limit']
                  and _blocks_agree(curves, est['factor'], est['offset_s'], tol_s, allow_drift=True))
        est['blocks_agree'] = ok
        deviates = None
        if ok:
            far = abs(np.log(est['factor'])) > np.log(1 + tolerance)
            est['declared_rate_agrees'] = (bool(_blocks_agree(curves, 1.0, est.get('offset_at_1_s', np.nan), tol_s))
                                           if far else True)
            deviates = bool(far and not est['declared_rate_agrees'])
            trusted.append((est['factor'], deviates))
        per_segment[name] = entry(deviates, est['factor'], 'factor', est['peak_correlation'], **est)
    thresholds = dict(factor_range=factor_range, max_offset_s=max_offset_s, tolerance=tolerance,
                      method='rank-blocks', tolerance_s=tol_s)
    if not trusted:
        return TimeScaleResult(ERROR_SCALE, None, 'No segment gave a factor whose blocks agree, so the time scale '
                               'cannot be decided.', per_segment, thresholds, unit='factor')
    factors = np.array([f for f, _ in trusted])
    factor = float(np.median(factors))
    detected = sum(d for _, d in trusted) > len(trusted) / 2
    summary = (f'Best factor {factor:.4g} (median over {len(trusted)} of {len(per_segment)} segments whose blocks '
               'agree); ' + ('deviates' if detected else 'does not deviate') + ' from the declared rate.')
    return TimeScaleResult(ERROR_SCALE, detected, summary, per_segment, thresholds, estimate=factor, unit='factor',
                           factor=factor, spread=float(factors.max() / factors.min()))


def remove_time_scale(recording: Recording, result: TimeScaleResult | None = None, **check_options) -> Recording:
    """Copy with every IMU rate set to declared rate x factor (where detected); no sample changed."""
    if result is None:
        result = check_for_time_scale(recording, **check_options)
    new = recording.copy()
    if not result.detected:
        new.log_correction(ERROR_SCALE, f'Not applied: {result.summary}')
        return new
    for seg in new.segments.values():
        seg.imu.rate_hz *= result.factor
    new.log_correction(ERROR_SCALE, f'Set every IMU rate to its declared rate x {result.factor:.5g}.',
                       factor=result.factor)
    return new


def effective_rate_measures(segment: Segment) -> dict:
    """The four measures of kind 2 (module description) for one IMU.

    Samples marked invalid (`imu.valid` False, e.g. by remove_dead_channels, which runs first in the agreed order of
    correction) and missing values are left out in place: a difference is used only if every sample it spans is
    valid, so a stuck stretch already counted as error 7 (dead or stuck channel) is not counted again as repeated
    samples, and no difference bridges a gap."""
    imu = segment.imu
    x = np.hstack([imu.gyr, imu.acc]).astype(float)
    x[~imu.valid] = np.nan
    good = np.all(np.isfinite(x), axis=1)
    n = int(good.sum())
    out = dict(samples=n, hold_fraction=float('nan'), linear_fraction=float('nan'), line_ratio=float('nan'),
               line_hz=float('nan'), neighbour_bins=0, top_band_ratio=float('nan'))
    if n < 100:
        return out
    d1 = np.diff(x, axis=0)
    ok1 = good[:-1] & good[1:]
    held = np.all(d1 == 0, axis=1) & ok1
    out['hold_fraction'] = float(held.sum() / ok1.sum())
    scale = np.nanmax(np.abs(x), axis=0) + 1e-12
    d2 = np.diff(x, n=2, axis=0)
    ok2 = good[:-2] & good[1:-1] & good[2:]
    flat = np.all(np.abs(np.nan_to_num(d2)) <= 1e-7 * scale, axis=1) & ok2
    moving = ok2 & ~(held[:-1] & held[1:])  # a run of held samples is not a linear stretch
    out['linear_fraction'] = float((flat & moving).sum() / max(1, moving.sum()))
    a = np.abs(d2 / scale).sum(axis=1)
    a = np.where(ok2, a - a[ok2].mean(), 0.0)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    freqs = np.fft.rfftfreq(len(a), 1 / imu.rate_hz)
    band = freqs > 10.0
    if band.sum() > 10:
        k = int(np.argmax(np.where(band, spec, -np.inf)))
        near = np.abs(freqs - freqs[k]) <= 1.0
        near[max(0, k - 2):k + 3] = False  # the line itself and its window leakage
        out['line_hz'] = float(freqs[k])
        out['neighbour_bins'] = int(near.sum())
        if near.sum() >= MIN_NEIGHBOUR_BINS:  # shorter recordings: no reliable neighbourhood, line undecided
            out['line_ratio'] = float(spec[k] / (np.median(spec[near]) + 1e-30))
    g = x[:, :3]
    g = np.where(good[:, None], g - np.nanmean(g[good], axis=0), 0.0)
    ps = (np.abs(np.fft.rfft(g, axis=0)) ** 2).sum(axis=1)
    f = np.fft.rfftfreq(len(g), 1 / imu.rate_hz)
    nyq = imu.rate_hz / 2
    top, mid = (f >= 0.8 * nyq) & (f < 0.98 * nyq), (f >= 0.3 * nyq) & (f < 0.5 * nyq)
    if top.any() and mid.any():
        out['top_band_ratio'] = float(np.mean(ps[top]) / (np.mean(ps[mid]) + 1e-30))
    return out


@dataclass
class EffectiveRateResult(CheckResult):
    resampled: bool | None = None  # a resampling line in any IMU
    below_declared: bool | None = None  # repeated samples, linear stretches or an empty top band in any IMU


def check_for_effective_rate(recording: Recording, min_hold_fraction: float = 0.05,
                             min_linear_fraction: float = 0.2, min_line_ratio: float = 200.0,
                             max_top_band_ratio: float = 1e-6) -> EffectiveRateResult:
    """Was any IMU stream resampled from another rate, or does it carry a lower effective rate than declared (kind 2
    of the module description)? Needs no reference."""
    per_segment, decided = {}, 0
    for name, seg in recording.segments.items():
        m = effective_rate_measures(seg)
        if not np.isfinite(m['hold_fraction']):
            per_segment[name] = entry(None, float('nan'), 'Hz', resampled=None, below_declared=None, **m)
            continue
        decided += 1
        resampled = bool(m['line_ratio'] >= min_line_ratio) if np.isfinite(m['line_ratio']) else None
        below = bool(m['hold_fraction'] >= min_hold_fraction or m['linear_fraction'] >= min_linear_fraction
                     or (np.isfinite(m['top_band_ratio']) and m['top_band_ratio'] < max_top_band_ratio))
        per_segment[name] = entry(bool(resampled) or below, seg.imu.rate_hz * (1 - m['hold_fraction']), 'Hz',
                                  resampled=resampled, below_declared=below, **m)
    thresholds = dict(min_hold_fraction=min_hold_fraction, min_linear_fraction=min_linear_fraction,
                      min_line_ratio=min_line_ratio, max_top_band_ratio=max_top_band_ratio)
    if not decided:
        return EffectiveRateResult(ERROR_EFFECTIVE, None, 'Too few valid samples to decide.', per_segment, thresholds)
    res = [n for n, p in per_segment.items() if p['resampled']]
    low = [n for n, p in per_segment.items() if p['below_declared']]
    summary = (f'{len(res)} of {decided} IMUs show a resampling pattern'
               + (f' ({", ".join(res)})' if res else '') + f'; {len(low)} carry a lower effective rate than declared'
               + (f' ({", ".join(low)}).' if low else '.'))
    line_decided = [p['resampled'] for p in per_segment.values() if p['resampled'] is not None]
    resampled = True if res else (False if line_decided else None)
    if not line_decided:
        summary += ' Too short for the resampling line (fewer than 20 frequency points within +-1 Hz).'
    return EffectiveRateResult(ERROR_EFFECTIVE, bool(res or low), summary, per_segment, thresholds,
                               unit='Hz', resampled=resampled, below_declared=bool(low))


# Dataset level: one row per trial and IMU. An IMU counts if its time axis is stretched, its stream resampled or its
# effective rate below the declared one; a trial counts if the trial-level decision of either check holds.

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, flag, frame, per_trial, read_rows, run_trials, truthy, number  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'factor', 'scale_detected',
           'scale_deviates', 'resampled', 'below_declared', 'trial_detected', 'imu_hz', 'peak_correlation',
           'hold_fraction', 'linear_fraction', 'line_ratio', 'line_hz', 'top_band_ratio', 'error']


def _check_trial(loader, trial, **options):
    rec = loader.load(trial)
    ts, er = check_for_time_scale(rec), check_for_effective_rate(rec)
    trial_hit = bool(truthy(ts.detected) or truthy(er.resampled) or truthy(er.below_declared))
    rows = []
    for name, seg in rec.segments.items():
        a, b = ts.per_segment[name], er.per_segment[name]
        flags = [a['detected'], b['resampled'], b['below_declared']]
        hit = any(f is True for f in flags)
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), segment=name,
                         detected=True if hit else (None if a['detected'] is None and b['below_declared'] is None
                                                    else False),
                         estimate=a['factor'], unit='factor', factor=ts.factor, scale_detected=ts.detected,
                         scale_deviates=a['detected'], resampled=b['resampled'], below_declared=b['below_declared'],
                         trial_detected=trial_hit, imu_hz=seg.imu.rate_hz, peak_correlation=a['peak_correlation'],
                         hold_fraction=b['hold_fraction'], linear_fraction=b['linear_fraction'],
                         line_ratio=b['line_ratio'], line_hz=b['line_hz'], top_band_ratio=b['top_band_ratio'],
                         error=''))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """Both checks (stretched time axis, resampled or lower effective rate) on every trial."""
    return frame(run_trials(loader, _check_trial, processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials with a stretched time axis, a resampled stream or a lower effective rate / IMUs affected."""
    ok = read_rows(rows)
    hit = ok.detected.map(lambda v: flag(v) is True)
    trial_hit = ok.trial_detected.map(truthy)
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[trial_hit].nunique()),
                occurrences=int(hit.sum()), occurrence_unit='IMUs',
                undecided=int(ok.trial[ok.detected.map(lambda v: flag(v) is None)].nunique()))


def _rescale(rec, rows):
    r = rows[0]
    if not truthy(r.get('scale_detected')):
        return rec
    factor = number(r.get('factor'))
    new = rec.copy()
    for seg in new.segments.values():
        seg.imu.rate_hz *= factor
    new.log_correction(ERROR_SCALE, f'Set every IMU rate to its declared rate x {factor:.5g}.', factor=factor)
    return new


def correct(loader, rows):
    """A stretched time axis: every IMU rate of the trial set to declared rate x factor. A resampled stream or a lower
    effective rate is reported only."""
    return corrected(loader, _rescale, per_trial(rows), 'sampling rate')


SPEC = ErrorSpec(11, 'sampling_rate', 'fix', 'sampling rate not as declared')
PARTS = {11: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_sampling_rate = make_handler(PARTS[11])
