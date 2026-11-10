"""Error 14, between the IMUs: the IMUs of one recording do not share one time scale.

What it is: each IMU runs at a slightly different true sampling rate, so the IMUs drift apart during the recording.
Unlike the clock drift against the reference (error 13), which all IMUs share, this shifts the IMUs of a joint
against each other.

How we catch it:
    1. For every sensor, the drift against the reference from the windowed offsets of the clock-drift check (error 13) (30 s windows,
       at least 50 % overlap, least-squares line). Its standard error comes from every second window only, so the
       windows used do not overlap and their offsets are closer to independent (with all windows the standard error
       was too small: 74 % of TUK trials differed "significantly" at 0.24 ms/min). At least three such windows.
    2. The drift all sensors share is the median over sensors (error 13, clock drift against the reference). Each
       sensor's own deviation from that median is this error.
    3. Threshold: a sensor deviates if its deviation exceeds two standard
       errors (of the sensor and of the median) and adds up over the recording to more than half a reference frame.
       Recordings shorter than 60 s, or with fewer than two sensors with a drift, are undecided.

How we correct it: the time of each deviating IMU is stretched by its own deviation about the middle of the
recording (as remove_clock_drift, but per sensor), t -> t - e * (t - t_mid): a new start time and sampling rate, no
sample changed. The drift shared by all sensors and the constant offset stay for their own corrections. With two
sensors both deviate by half their difference, since which one is off cannot be told.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress

from ..recording import Recording
from ._result import CheckResult, entry
from .e13_clock_drift import MIN_DURATION_S, MIN_WINDOWS, common_span, estimate_segment_drift

ERROR = 'sensor rate mismatch'


@dataclass
class SensorRateMismatchResult(CheckResult):
    largest_deviation_ms_per_min: float = float('nan')  # largest |deviation| among the sensors with a drift
    shared_drift_ms_per_min: float = float('nan')  # median over sensors
    pivot_s: float = float('nan')


def sensor_drift(segment, **window_options) -> dict:
    """One sensor's drift (ms/min, line through all counted windows) and its standard error from every second window."""
    d = estimate_segment_drift(segment, **window_options)
    alternate = [w for w in d['windows'][::2] if w['counted']]
    stderr = float('nan')
    if len(alternate) >= MIN_WINDOWS and np.isfinite(d['slope_ms_per_min']):
        stderr = linregress([w['centre_s'] for w in alternate], [w['offset_s'] for w in alternate]).stderr * 60e3
    return dict(slope_ms_per_min=d['slope_ms_per_min'], stderr_ms_per_min=stderr, counted=d['counted'],
                windows=d['windows'])


def check_for_sensor_rate_mismatch(recording: Recording, min_duration_s: float = MIN_DURATION_S,
                                   tolerance_s: float | None = None, **window_options) -> SensorRateMismatchResult:
    """Do the IMUs of this recording drift against each other?

    The drift windows use the drift check's rule (default method='rank-blocks': ranks, 5 blocks on a line);
    method='peak' gives the former rule (peak correlation 0.9)."""
    if tolerance_s is None:
        tolerance_s = 0.5 / min(s.reference.rate_hz for s in recording.segments.values())
    thresholds = dict(min_duration_s=min_duration_s, tolerance_s=tolerance_s, **window_options)
    spans = [common_span(s) for s in recording.segments.values()]
    duration = min(b - a for a, b in spans)
    pivot = float(np.median([(a + b) / 2 for a, b in spans]))
    undecided = dict(unit='ms/min', pivot_s=pivot)
    if duration < min_duration_s or len(recording.segments) < 2:
        return SensorRateMismatchResult(ERROR, None, f'Recording is {duration:.0f} s long with '
                                        f'{len(recording.segments)} sensors; tested from {min_duration_s:.0f} s and '
                                        'two sensors on.', {}, thresholds, **undecided)
    drifts = {n: sensor_drift(s, **window_options) for n, s in recording.segments.items()}
    usable = {n: d for n, d in drifts.items() if np.isfinite(d['slope_ms_per_min'] + d['stderr_ms_per_min'])}
    if len(usable) < 2:
        return SensorRateMismatchResult(ERROR, None, 'Fewer than two sensors have a drift with an uncertainty.',
                                        {n: entry(None, d['slope_ms_per_min'], 'ms/min', d['counted'], **d)
                                         for n, d in drifts.items()}, thresholds, **undecided)
    slopes = np.array([d['slope_ms_per_min'] for d in usable.values()])
    errors = np.array([d['stderr_ms_per_min'] for d in usable.values()])
    shared = float(np.median(slopes))
    shared_se = float(1.2533 * np.sqrt(np.mean(errors ** 2) / len(errors)))  # standard error of a median
    per_segment = {}
    for name, d in drifts.items():
        if name not in usable:
            per_segment[name] = entry(None, float('nan'), 'ms/min', d['counted'], **d)
            continue
        deviation = d['slope_ms_per_min'] - shared
        se = float(np.hypot(d['stderr_ms_per_min'], shared_se))
        deviates = bool(abs(deviation) > 2 * se and abs(deviation) / 60e3 * duration > tolerance_s)
        per_segment[name] = entry(deviates, deviation, 'ms/min', d['counted'], deviation_ms_per_min=deviation,
                                  deviation_stderr_ms_per_min=se, **d)
    deviations = [p['deviation_ms_per_min'] for p in per_segment.values() if p['detected'] is not None]
    largest = float(max(deviations, key=abs))
    off = [n for n, p in per_segment.items() if p['detected']]
    detected = bool(off)
    summary = (f'Sensors drift against the reference by {slopes.min():+.2f} to {slopes.max():+.2f} ms/min (shared '
               f'{shared:+.2f}); largest own deviation {largest:+.2f} ms/min. '
               + (f'Deviating: {", ".join(off)}.' if off else 'No sensor deviates beyond the threshold.'))
    return SensorRateMismatchResult(ERROR, detected, summary, per_segment, thresholds, estimate=largest,
                                    unit='ms/min', largest_deviation_ms_per_min=largest,
                                    shared_drift_ms_per_min=shared, pivot_s=pivot)


def remove_sensor_rate_mismatch(recording: Recording, result: SensorRateMismatchResult | None = None,
                                **check_options) -> Recording:
    """Return a copy in which every deviating IMU's time is stretched by its own deviation about mid-recording."""
    if result is None:
        result = check_for_sensor_rate_mismatch(recording, **check_options)
    new = recording.copy()
    if not result.detected:
        new.log_correction(ERROR, f'Not applied: {result.summary}')
        return new
    done = {}
    for name, p in result.per_segment.items():
        if not p['detected']:
            continue
        e = p['deviation_ms_per_min'] / 60e3
        imu = new.segments[name].imu
        imu.start_s = imu.start_s * (1 - e) + e * result.pivot_s
        imu.rate_hz = imu.rate_hz / (1 - e)
        done[name] = p['deviation_ms_per_min']
    new.log_correction(ERROR, 'Rescaled the time of ' + ', '.join(f'{n} by {-v:+.2f} ms/min' for n, v in done.items())
                       + f' about {result.pivot_s:.1f} s.', deviations_ms_per_min=done, pivot_s=result.pivot_s)
    return new


# Dataset level: one row per trial and IMU (one row with an empty segment where the trial is undecided as a whole).

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, flag, frame, number, per_trial, read_rows, run_trials, truthy  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'trial_detected',
           'deviation_ms_per_min', 'slope_ms_per_min', 'shared_drift_ms_per_min', 'pivot_s', 'error']


def _check_trial(loader, trial):
    rec = loader.load(trial)
    c = check_for_sensor_rate_mismatch(rec)
    base = dict(trial=trial, participant=str(loader.participant_of(trial)), unit='ms/min', trial_detected=c.detected,
                shared_drift_ms_per_min=c.shared_drift_ms_per_min, pivot_s=c.pivot_s, error='')
    if c.detected is None and not any(p['detected'] is not None for p in c.per_segment.values()):
        return [dict(base, segment='', detected=None, estimate=float('nan'))]
    return [dict(base, segment=n, detected=p['detected'], estimate=p['estimate'],
                 deviation_ms_per_min=p.get('deviation_ms_per_min', float('nan')),
                 slope_ms_per_min=p.get('slope_ms_per_min', float('nan')))
            for n, p in c.per_segment.items()]


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """Every trial of at least 60 s with two IMUs: which IMUs drift against the others."""
    return frame(run_trials(loader, _check_trial, processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials with a deviating IMU / deviating IMUs; undecided trials have no IMU with a drift to compare."""
    ok = read_rows(rows)
    hit = ok.detected.map(lambda v: flag(v) is True)
    trial_flags = ok.groupby('trial').trial_detected.first().map(flag)
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[hit].nunique()),
                occurrences=int(hit.sum()), occurrence_unit='IMUs', undecided=int(trial_flags.isna().sum()))


def _rescale(rec, rows):
    new, done, pivot = rec.copy(), {}, float('nan')
    for r in rows:
        if r['segment'] not in new.segments or not truthy(r.get('detected')):
            continue
        dev, pivot = number(r['deviation_ms_per_min']), number(r['pivot_s'])
        e = dev / 60e3
        imu = new.segments[r['segment']].imu
        imu.start_s = imu.start_s * (1 - e) + e * pivot
        imu.rate_hz = imu.rate_hz / (1 - e)
        done[r['segment']] = dev
    if done:
        new.log_correction(ERROR, 'Rescaled the time of ' + ', '.join(f'{n} by {-v:+.2f} ms/min'
                                                                       for n, v in done.items())
                           + f' about {pivot:.1f} s.', deviations_ms_per_min=done, pivot_s=pivot)
    return new


def correct(loader, rows):
    """The time of every deviating IMU stretched by its own deviation about the middle of the recording."""
    return corrected(loader, _rescale, per_trial(rows), 'IMU time scales')


SPEC = ErrorSpec(14, 'time_scales', 'fix', 'IMUs of one trial on different time scales')
PARTS = {14: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_time_scales = make_handler(PARTS[14])
