"""Error 13: clock drift between the IMU and the optical reference.

What it is: the two clocks run at slightly different speeds, so the time offset is not constant but changes during
the recording. 100 ppm (parts per million) means 0.1 ms more offset per second, 6 ms per minute. Drift is reported
per minute (and in ppm), not as the offset it adds up to, so trials of different length can be compared.

How we catch it:
    1. Cut the time both streams cover into windows of WINDOW_S (30 s) that overlap by at least OVERLAP (50 %),
       spread evenly so the first starts at the beginning and the last ends at the end of that time.
    2. In every window, estimate the time offset exactly as the synchronisation-offset check does (same speed
       curves, same left-out frames: only reference frames without an orientation, same search window capped at a
       third of the window), with both speed curves replaced by their ranks. A window counts if the offsets of 5
       equal blocks of the window lie on a straight line: twice the standard error of their residuals around the
       line within half a reference frame, and the offset not at the search edge (the offset check's rule, with a
       line instead of one constant offset, since the offset changes inside a window by the drift itself). The former rule, a correlation
       peak of at least 0.9, is kept as method='peak'.
    3. Fit a straight line offset(t) = a + d * t through the counted windows of one segment (least squares, every
       counted window weighted equally). The slope d is the segment's drift, with its standard error.
    4. All segments of one recording share the IMU clock, so the recording's drift is the median over segments
       with at least three counted windows; the spread between them is reported as a plausibility check.
    Detection (exploratory): the drift is detected if it differs from zero by more than two standard errors and
    adds up to more than half a reference frame over the recording. Recordings shorter than MIN_DURATION_S (60 s)
    are undecided.

The constant part of the offset (the line's level) is NOT reported here: it is the synchronisation offset, error 12,
reported by its own check. Likewise the correction removes only the drift.

How we correct it: the IMU time stamps are rescaled about the middle of the recording, t -> t - d * (t - t_mid),
which is a new start time and a new sampling rate (rate / (1 - d)). The offset at t_mid is left unchanged, so the
constant offset the synchronisation-offset check measures stays for its own correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress

from ..recording import Recording, Segment
from ._result import CheckResult, entry
from .e12_time_offset import block_offsets, blocks_on_a_line, estimate_time_lag, speed_curves

ERROR = 'clock drift'
WINDOW_S = 30.0
OVERLAP = 0.5
MIN_DURATION_S = 60.0
MIN_WINDOWS = 3  # a line with an uncertainty needs at least three points


@dataclass
class ClockDriftResult(CheckResult):
    drift_ms_per_min: float = float('nan')  # median over segments; > 0: the offset grows, the IMU clock runs fast
    stderr_ms_per_min: float = float('nan')  # median standard error of the segments' slopes
    drift_ppm: float = float('nan')
    spread_ms_per_min: float = float('nan')  # largest minus smallest segment drift
    pivot_s: float = float('nan')  # middle of the time both streams cover; the correction keeps the offset here


def common_span(segment: Segment) -> tuple[float, float]:
    """First and last moment covered by both the IMU and the reference of a segment."""
    return (max(segment.imu.times[0], segment.reference.times[0]),
            min(segment.imu.times[-1], segment.reference.times[-1]))


def window_starts(start_s: float, end_s: float, window_s: float = WINDOW_S, overlap: float = OVERLAP) -> np.ndarray:
    """Evenly spread window starts from `start_s` to `end_s - window_s`, overlapping by at least `overlap`."""
    if end_s - start_s < window_s:
        return np.array([])
    n = int(np.ceil((end_s - start_s - window_s) / (window_s * (1 - overlap)) - 1e-9)) + 1
    return np.linspace(start_s, end_s - window_s, n) if n > 1 else np.array([start_s])


def windowed_offsets(segment: Segment, window_s: float = WINDOW_S, overlap: float = OVERLAP,
                     max_offset_s: float = 10.0, exclude: str = 'invalid',
                     min_peak_correlation: float = 0.9, method: str = 'rank-blocks',
                     tolerance_s: float | None = None) -> list[dict]:
    """The time offset in each window of one segment (steps 1-2 of the module description).

    method='rank-blocks' (default): ranks, and the window counts if the offsets of 5
    blocks of the window lie on a straight line within `tolerance_s` (half a reference frame; blocks_on_a_line): the
    offset check's rule, except that the offset may change inside the window, which is the drift being measured (a
    constant offset failed the known-answer tests from about 35 ms/min on at 100 Hz).
    method='peak' (former rule): a window counts if its Pearson peak reaches `min_peak_correlation`."""
    if tolerance_s is None:
        tolerance_s = 0.5 / segment.reference.rate_hz
    imu_t, imu_speed, ref_t, ref_speed = speed_curves(segment, exclude)
    out = []
    for a in window_starts(*common_span(segment), window_s, overlap):
        i, r = (imu_t >= a) & (imu_t < a + window_s), (ref_t >= a) & (ref_t < a + window_s)
        if i.sum() < 10 or r.sum() < 10:
            continue
        est = estimate_time_lag(imu_t[i], imu_speed[i], ref_t[r], ref_speed[r], max_offset_s,
                                ranks=method == 'rank-blocks')
        if method == 'rank-blocks':
            est.block_offsets_s = tuple(block_offsets(imu_t[i], imu_speed[i], ref_t[r], ref_speed[r], est.offset_s))
            counted = not est.at_search_limit and blocks_on_a_line(est.block_offsets_s, tolerance_s)
        else:
            counted = est.peak_correlation >= min_peak_correlation and not est.at_search_limit
        out.append(dict(centre_s=float(a + window_s / 2), offset_s=est.offset_s, peak=est.peak_correlation,
                        at_limit=est.at_search_limit, counted=bool(counted)))
    return out


def estimate_segment_drift(segment: Segment, **window_options) -> dict:
    """Windowed offsets of one segment and the straight line through the counted ones (step 3).

    Returns the windows, the number counted, and (if at least MIN_WINDOWS count) the slope and its standard error
    in ms per minute; otherwise those are NaN."""
    windows = windowed_offsets(segment, **window_options)
    counted = [w for w in windows if w['counted']]
    slope = stderr = level = float('nan')
    if len(counted) >= MIN_WINDOWS:
        t = np.array([w['centre_s'] for w in counted])
        if np.ptp(t) > 0:
            fit = linregress(t, [w['offset_s'] for w in counted])
            slope, stderr, level = fit.slope * 60e3, fit.stderr * 60e3, fit.intercept
    return dict(windows=windows, counted=len(counted), slope_ms_per_min=slope, stderr_ms_per_min=stderr,
                offset_at_zero_s=level)


def check_for_clock_drift(recording: Recording, min_duration_s: float = MIN_DURATION_S,
                          tolerance_s: float | None = None, **window_options) -> ClockDriftResult:
    """Does the time offset between IMU and reference change during the recording?

    tolerance_s: the drift must add up to more than this over the recording. Default: half a reference frame.
    window_options: window_s, overlap, max_offset_s, exclude, method, min_peak_correlation (see windowed_offsets).
    Defaults are exploratory."""
    if not recording.segments:
        return ClockDriftResult(ERROR, None, 'No segment with both an IMU and a reference.', {},
                                dict(min_duration_s=min_duration_s, tolerance_s=tolerance_s), unit='ms/min')
    if tolerance_s is None:
        tolerance_s = 0.5 / min(s.reference.rate_hz for s in recording.segments.values())
    thresholds = dict(min_duration_s=min_duration_s, tolerance_s=tolerance_s, window_s=WINDOW_S, overlap=OVERLAP,
                      min_windows=MIN_WINDOWS, **window_options)
    spans = [common_span(s) for s in recording.segments.values()]
    duration = min(b - a for a, b in spans)
    pivot = float(np.median([(a + b) / 2 for a, b in spans]))
    if duration < min_duration_s:
        return ClockDriftResult(ERROR, None, f'Recording is {duration:.0f} s long; drift is only tested from '
                                             f'{min_duration_s:.0f} s on.', {}, thresholds, unit='ms/min',
                                pivot_s=pivot)
    per_segment = {}
    for name, seg in recording.segments.items():
        d = estimate_segment_drift(seg, **window_options)
        ok = np.isfinite(d['slope_ms_per_min'])
        decision = _detected(d['slope_ms_per_min'], d['stderr_ms_per_min'], duration, tolerance_s) if ok else None
        per_segment[name] = entry(decision, d['slope_ms_per_min'], 'ms/min', d['counted'], **d)
    trusted = [p for p in per_segment.values() if np.isfinite(p['slope_ms_per_min'])]
    if not trusted:
        return ClockDriftResult(ERROR, None, f'No segment has {MIN_WINDOWS} windows with a trustworthy offset.',
                                per_segment, thresholds, unit='ms/min', pivot_s=pivot)
    slopes = [p['slope_ms_per_min'] for p in trusted]
    drift = float(np.median(slopes))
    stderr = float(np.median([p['stderr_ms_per_min'] for p in trusted]))
    detected = _detected(drift, stderr, duration, tolerance_s)
    summary = (f'The offset changes by {drift:+.2f} ± {stderr:.2f} ms per minute ({drift / 0.06:+.0f} ppm; median '
               f'over {len(trusted)} of {len(per_segment)} segments, segments differ by up to '
               f'{np.ptp(slopes):.2f} ms/min). Detected if beyond two standard errors and more than '
               f'{tolerance_s * 1e3:.1f} ms over the {duration:.0f} s recording.')
    return ClockDriftResult(ERROR, detected, summary, per_segment, thresholds, estimate=drift, unit='ms/min',
                            drift_ms_per_min=drift, stderr_ms_per_min=stderr, drift_ppm=drift / 0.06,
                            spread_ms_per_min=float(np.ptp(slopes)), pivot_s=pivot)


def _detected(slope_ms_per_min, stderr_ms_per_min, duration_s, tolerance_s) -> bool:
    beyond_noise = abs(slope_ms_per_min) > 2 * stderr_ms_per_min
    return bool(beyond_noise and abs(slope_ms_per_min) / 60e3 * duration_s > tolerance_s)


def remove_clock_drift(recording: Recording, result: ClockDriftResult | None = None, **check_options) -> Recording:
    """Return a copy whose IMU time stamps are rescaled about the middle of the recording, removing the drift but
    not the constant offset (error 12, time offset, corrected by remove_synchronisation_offset)."""
    if result is None:
        result = check_for_clock_drift(recording, **check_options)
    new = recording.copy()
    if not result.detected:
        new.log_correction(ERROR, f'Not applied: {result.summary}')
        return new
    d, pivot = result.drift_ms_per_min / 60e3, result.pivot_s
    for seg in new.segments.values():
        # stamp t = start + k / rate becomes t - d (t - pivot) = start (1 - d) + d pivot + k (1 - d) / rate
        seg.imu.start_s = seg.imu.start_s * (1 - d) + d * pivot
        seg.imu.rate_hz = seg.imu.rate_hz / (1 - d)
    new.log_correction(ERROR, f'Rescaled the IMU time base by {-result.drift_ppm:+.0f} ppm about '
                              f'{pivot:.1f} s ({-result.drift_ms_per_min:+.2f} ms per minute).',
                       drift_ms_per_min=result.drift_ms_per_min, pivot_s=pivot)
    return new


# Dataset level: one row per trial (its IMUs share one clock).

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, frame, number, per_trial, prevalence_of, run_trials, truthy  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'decided', 'drift_ms_per_min',
           'stderr_ms_per_min', 'drift_ppm', 'pivot_s', 'error']


def _check_trial(loader, trial):
    rec = loader.load(trial)
    c = check_for_clock_drift(rec)
    return [dict(trial=trial, participant=str(loader.participant_of(trial)), segment='', detected=c.detected,
                 estimate=c.drift_ms_per_min, unit='ms/min', decided=c.detected is not None,
                 drift_ms_per_min=c.drift_ms_per_min, stderr_ms_per_min=c.stderr_ms_per_min, drift_ppm=c.drift_ppm,
                 pivot_s=c.pivot_s, error='')]


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """The clock drift of every trial of at least 60 s."""
    return frame(run_trials(loader, _check_trial, processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials with a drift / the same trials (one drift per trial)."""
    return prevalence_of(rows, 'trials')


def _rescale(rec, rows):
    r = rows[0]
    if not truthy(r.get('detected')):
        return rec
    result = ClockDriftResult(ERROR, True, '', drift_ms_per_min=number(r['drift_ms_per_min']),
                              drift_ppm=number(r['drift_ppm']), pivot_s=number(r['pivot_s']))
    return remove_clock_drift(rec, result)


def correct(loader, rows):
    """IMU time stamps rescaled about the middle of the recording (the constant offset stays)."""
    return corrected(loader, _rescale, per_trial(rows), 'clock drift')


SPEC = ErrorSpec(13, 'clock_drift', 'fix', 'clock drift between IMU and reference')
PARTS = {13: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_clock_drift = make_handler(PARTS[13])
