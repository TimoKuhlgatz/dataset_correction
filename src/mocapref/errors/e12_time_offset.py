"""Error 12: time offset between the IMU and the optical reference (synchronisation offset).

What it is: the IMU and the optical system run on different clocks. If the deposit pairs IMU sample k with the
reference frame taken at the same time stamp, but the clocks were started at different moments, every IMU sample
is paired with the reference of a different moment.

How we catch it (step 1):
    1. Turn the reference orientation into an angular rate: the rotation between two consecutive frames divided by
       the frame period. This is what a perfect gyroscope on the segment would measure ("simulated gyroscope").
    2. Take the magnitude (speed) of both the simulated and the measured gyroscope. The magnitude does not depend
       on how the IMU is mounted or on sign errors of its axes, so this step works before those are fixed.
    3. Replace each speed curve by its ranks (its position when all values are sorted), so a single extreme value,
       such as a corrupted gyroscope sample or an impossible step of the reference, counts no more than the
       largest true value. Slide one curve against the other and compute their correlation at every shift
       (cross-correlation). The shift with the highest correlation is the time offset; a parabola through the
       peak refines it below the grid step.
    4. Decide whether the offset is known precisely enough (block agreement): the recording is cut into 5 equal
       time blocks and the offset is estimated again in each block, within +-1.5 s of the whole-recording offset
       (wide enough to hold a peak one stride away). A segment is trusted if at least 3 blocks give an estimate,
       twice the standard error of the block offsets is at most the tolerance (half a reference frame), their
       median lies within the tolerance of the whole-recording offset, and the offset is not at the edge of the
       search window. No limit on the correlation height. The offset counts as
       detected if it exceeds the tolerance.
    All segments of one recording share the IMU clock, so the recording's offset is the median over the
    trusted segments; the spread between segments is reported as a plausibility check.
    Why ranks and blocks: with the former rule (Pearson correlation, trusted at a peak of at least 0.9) one
    corrupted sample per segment took most segments below 0.9, the 0.9 limit rejected good offsets in slow motion,
    and it trusted wrong peaks. The former rule is kept as method='peak'.

How we correct it (step 2): move the IMU time stamps by the offset (`ImuStream.start_s`). No sample is changed,
cut or resampled. (Resampling onto one common grid happens only when corrected data are saved, see resample.py.)

How we check the correction (validation): the cross-correlation cannot be reused, because after the shift it
returns zero by construction. Instead `gyroscope_audit` compares, over 1 s windows, how far the integrated
gyroscope says the segment turned with how far the reference says it turned. A correct shift makes the two agree
better; `validate_offset_correction` reports this before and after.

Sign convention: offset_s > 0 means the IMU time stamps are late: the IMU sample stamped t was taken at the
moment the reference stamps t - offset_s.

Known limits: on periodic motion (walking) the correlation also has peaks one stride apart; blocks that disagree
on the peak leave the segment untrusted, but a wrong peak preferred in every block is not caught (the median over
segments may still be right). A recording whose clock drifts (error 13) has no single offset: its
block offsets rise with time and it stays undecided unless the drift over the recording is small against the
tolerance. Recordings too short for 5 informative blocks (a few seconds) stay undecided.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.stats import rankdata

from ..recording import Recording, Segment
from scipy.spatial.transform import Rotation

from ..rotations import reference_angular_rate
from ._result import CheckResult, entry
from .e03_reference_jumps import implausible_reference_frames

ERROR = 'synchronisation offset'


@dataclass
class LagEstimate:
    """Cross-correlation result for one pair of speed curves."""

    offset_s: float
    peak_correlation: float  # correlation at the best shift, 1 = identical shape
    correlation_at_zero: float  # correlation without any shift
    at_search_limit: bool  # best shift lies at the edge of the window: the true offset may be larger
    second_peak_offset_s: float  # best shift at least `peak_separation_s` away from the best one
    second_peak_correlation: float
    block_offsets_s: tuple = ()  # offsets of the time blocks (block agreement), empty for method='peak'

    @property
    def block_standard_error_s(self) -> float:
        b = np.asarray(self.block_offsets_s)
        return float(np.std(b, ddof=1) / np.sqrt(len(b))) if len(b) >= 3 else float('inf')


def estimate_time_lag(imu_times, imu_speed, ref_times, ref_speed, max_offset_s=2.0, grid_hz=1000.0,
                      peak_separation_s=0.2, ranks=False) -> LagEstimate:
    """Find the shift that best aligns the IMU speed curve with the reference speed curve.

    Both curves are interpolated onto a common time grid (`grid_hz`) over the time both cover, normalised to zero
    mean and unit standard deviation, and correlated at every shift up to `max_offset_s` (at most a third of the
    common duration). NaN samples (invalid or excluded) contribute nothing. ranks=True replaces each curve by its
    ranks before normalising (robust to single extreme values). Returns the shift in seconds with the sign
    convention of this module.
    """
    t0, t1 = max(imu_times[0], ref_times[0]), min(imu_times[-1], ref_times[-1])
    grid = np.arange(t0, t1, 1.0 / grid_hz)
    standardise = _rank_standardise if ranks else _standardise
    x = standardise(np.interp(grid, imu_times, imu_speed))
    y = standardise(np.interp(grid, ref_times, ref_speed))
    n = len(grid)
    max_shift = int(min(max_offset_s, (t1 - t0) / 3) * grid_hz)
    # correlation[s] = mean over the overlap of x(t + s) * y(t): IMU shows at t + s what the reference shows at t
    sums = correlate(x, y, mode='full', method='fft')
    shifts = correlation_lags(n, n, mode='full')
    keep = np.abs(shifts) <= max_shift
    shifts, corr = shifts[keep], sums[keep] / (n - np.abs(shifts[keep]))
    best = int(np.argmax(corr))
    refinement = 0.0
    if 0 < best < len(corr) - 1:
        curvature = corr[best - 1] - 2 * corr[best] + corr[best + 1]
        refinement = 0.5 * (corr[best - 1] - corr[best + 1]) / curvature if curvature != 0 else 0.0
    away = np.abs(shifts - shifts[best]) > peak_separation_s * grid_hz
    second = int(np.argmax(np.where(away, corr, -np.inf))) if away.any() else best
    return LagEstimate(
        offset_s=float((shifts[best] + refinement) / grid_hz),
        peak_correlation=float(corr[best]),
        correlation_at_zero=float(corr[shifts == 0][0]),
        at_search_limit=bool(abs(shifts[best]) >= max_shift),
        second_peak_offset_s=float(shifts[second] / grid_hz),
        second_peak_correlation=float(corr[second]))


@dataclass
class SynchronisationOffsetResult(CheckResult):
    """CheckResult plus the recording's offset and the spread between its segments."""

    offset_s: float = float('nan')
    spread_s: float = float('nan')


def _standardise(v):
    v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    return np.nan_to_num(v)  # missing samples contribute nothing to the correlation


def _rank_standardise(v):
    """Ranks of the finite samples (ties: average rank), normalised like _standardise; missing samples give 0."""
    r = np.asarray(rankdata(v, nan_policy='omit'), dtype=float)
    return np.nan_to_num((r - np.nanmean(r)) / (np.nanstd(r) + 1e-12))


def block_agreement(est: LagEstimate, tolerance_s: float) -> bool:
    """Trust rule of method='rank-blocks': at least 3 blocks, twice the standard error of the block offsets at most
    the tolerance, their median within the tolerance of the whole-recording offset (blocks that agree on another
    shift do not confirm it), and the offset not at the edge of the search window."""
    b = np.asarray(est.block_offsets_s)
    return bool(len(b) >= 3 and 2 * est.block_standard_error_s <= tolerance_s
                and abs(np.median(b) - est.offset_s) <= tolerance_s and not est.at_search_limit)


def blocks_on_a_line(block_offsets_s, tolerance_s: float) -> bool:
    """Variant of block_agreement for a stretch in which the offset may change steadily (a clock drift inside a drift
    window, a remaining fine drift after a time-scale factor): at least 4 blocks, and twice the standard error of
    their residuals around a straight line through them (least squares, blocks in order) at most the tolerance."""
    b = np.asarray(block_offsets_s, dtype=float)
    if len(b) < 4:
        return False
    x = np.arange(len(b), dtype=float)
    residual = b - np.polyval(np.polyfit(x, b, 1), x)
    se = np.sqrt(np.sum(residual ** 2) / (len(b) - 2)) / np.sqrt(len(b))
    return bool(2 * se <= tolerance_s)


EXCLUSIONS = ('implausible', 'invalid')
METHODS = ('rank-blocks', 'peak')
N_BLOCKS = 5
BLOCK_SEARCH_S = 1.5


def block_offsets(imu_times, imu_speed, ref_times, ref_speed, offset_s, n_blocks=N_BLOCKS,
                  search_s=BLOCK_SEARCH_S) -> np.ndarray:
    """Offsets (s) of `n_blocks` equal time blocks of the span both curves cover, each searched within
    +-`search_s` around `offset_s` with ranks; blocks without data are skipped."""
    its = np.asarray(imu_times) - offset_s  # IMU moved onto the reference clock by the whole-recording offset
    t0, t1 = max(ref_times[0], its[0]), min(ref_times[-1], its[-1])
    edges = np.linspace(t0, t1, n_blocks + 1)
    out = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        r = (ref_times >= b0) & (ref_times <= b1)
        i = (its >= b0 - search_s) & (its <= b1 + search_s)
        if r.sum() < 4 or i.sum() < 4 or not np.isfinite(ref_speed[r]).any() or not np.isfinite(imu_speed[i]).any():
            continue
        e = estimate_time_lag(its[i], imu_speed[i], ref_times[r], ref_speed[r], search_s, ranks=True)
        out.append(offset_s + e.offset_s)
    return np.array(out)


def estimate_segment_offset(segment: Segment, max_offset_s: float = 10.0, exclude: str = 'invalid',
                            method: str = 'rank-blocks') -> LagEstimate:
    """Time offset of one segment's IMU against its own reference (steps 1-3 of the module description).

    exclude='invalid' (default): only the frames without an orientation in the
    deposit (e.g. occluded markers) are left out; every other frame is used, however noisy. exclude='implausible'
    (former default): also leaves out reference frames that rotate more
    than 300 °/s faster or slower than the gyroscope, widened by 0.5 s; on skin-marker references this removes
    25-99 % of the frames and mixes the quality of the marker placement, a separate error, into this one.
    method='rank-blocks' (default): ranks, plus the offsets of the time blocks for the trust
    decision; method='peak': the former Pearson correlation without blocks."""
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    curves = speed_curves(segment, exclude)
    imu_times, _, ref_times, _ = curves
    if len(imu_times) < 2 or len(ref_times) < 2 or min(imu_times[-1], ref_times[-1]) <= max(imu_times[0], ref_times[0]):
        # IMU and reference cover no common time (e.g. Trento participant 9, whose sensors each run on their own
        # clock): the offset cannot be estimated, so the segment is undecided instead of failing the whole trial
        nan = float('nan')
        return LagEstimate(nan, nan, nan, False, nan, nan)
    est = estimate_time_lag(*curves, max_offset_s, ranks=method == 'rank-blocks')
    if method == 'rank-blocks':
        est.block_offsets_s = tuple(block_offsets(*curves, est.offset_s))
    return est


def speed_curves(segment: Segment, exclude: str = 'invalid'):
    """Time stamps and rotation speed of the IMU (gyroscope norm) and of the reference (simulated gyroscope norm),
    as (imu_times, imu_speed, ref_times, ref_speed); left-out samples are NaN (see estimate_segment_offset)."""
    if exclude not in EXCLUSIONS:
        raise ValueError(f'exclude must be one of {EXCLUSIONS}')
    imu, ref = segment.imu, segment.reference
    step_times, rates = reference_angular_rate(ref.orientation, ref.rate_hz)
    ref_speed = np.linalg.norm(rates, axis=1)
    excluded = implausible_reference_frames(segment, margin_s=0.5) if exclude == 'implausible' else ~ref.valid
    ref_speed[excluded[:-1] | excluded[1:]] = np.nan
    imu_speed = np.where(imu.valid, np.linalg.norm(imu.gyr, axis=1), np.nan)
    return imu.times, imu_speed, step_times + ref.start_s, ref_speed


def check_for_synchronisation_offset(recording: Recording, max_offset_s: float = 10.0,
                                     tolerance_s: float | None = None,
                                     min_peak_correlation: float = 0.9,
                                     exclude: str = 'invalid', method: str = 'rank-blocks') -> SynchronisationOffsetResult:
    """Is there a time offset between the IMU and the optical reference in this recording?

    max_offset_s: largest offset searched for (some deposits need about 10 s; capped at a third of the trial).
    tolerance_s: offsets up to this size count as no offset. Default: half a reference frame period.
    method: 'rank-blocks' (default): a segment is trusted if its block offsets agree (twice their standard error at
        most the tolerance, at least 3 blocks) and the offset is not at the search edge; 'peak' (former rule): if
        the Pearson correlation peak reaches `min_peak_correlation` and is not at the search edge.
    min_peak_correlation: used by method='peak' only.
    exclude: which reference frames are left out of the correlation ('implausible' or 'invalid', see
    estimate_segment_offset).
    All defaults are exploratory.

    Returns a result with `offset_s` (the recording's offset, median over trusted segments),
    `spread_s` (largest minus smallest trusted segment offset) and `per_segment` (offset in ms per segment, its
    peak correlation as quality, and the full LagEstimate as `lag`).
    """
    if tolerance_s is None:
        tolerance_s = 0.5 / min(s.reference.rate_hz for s in recording.segments.values())
    per_segment, trusted = {}, []
    for name, seg in recording.segments.items():
        est = estimate_segment_offset(seg, max_offset_s, exclude, method)
        if method == 'peak':
            trustworthy = est.peak_correlation >= min_peak_correlation and not est.at_search_limit
        else:
            trustworthy = block_agreement(est, tolerance_s)
        decision = (abs(est.offset_s) > tolerance_s) if trustworthy else None
        per_segment[name] = entry(decision, est.offset_s * 1e3, 'ms', est.peak_correlation, lag=est)
        if trustworthy:
            trusted.append(est.offset_s)
    thresholds = dict(max_offset_s=max_offset_s, tolerance_s=tolerance_s, exclude=exclude, method=method)
    thresholds.update(min_peak_correlation=min_peak_correlation) if method == 'peak' else thresholds.update(
        n_blocks=N_BLOCKS, block_search_s=BLOCK_SEARCH_S)
    if not trusted:
        summary = ('No segment gave a trustworthy offset, so the offset cannot be decided (too little motion, too '
                   'short, clock drift, or larger than the search window).')
        return SynchronisationOffsetResult(ERROR, None, summary, per_segment, thresholds, unit='ms')
    offset = float(np.median(trusted))
    spread = float(np.ptp(trusted))
    detected = abs(offset) > tolerance_s
    summary = (f'The IMU time stamps are {offset * 1e3:+.1f} ms off the reference (median over {len(trusted)} of '
               f'{len(per_segment)} segments; segments differ by up to {spread * 1e3:.1f} ms). '
               + ('This exceeds' if detected else 'This is within')
               + f' the tolerance of {tolerance_s * 1e3:.1f} ms.')
    return SynchronisationOffsetResult(ERROR, detected, summary, per_segment, thresholds, estimate=offset * 1e3,
                                       unit='ms', offset_s=offset, spread_s=spread)


def remove_synchronisation_offset(recording: Recording, result: SynchronisationOffsetResult | None = None,
                                  **check_options) -> Recording:
    """Return a copy of the recording with the IMU time stamps moved by the detected offset.

    If `result` is not given, check_for_synchronisation_offset is run first (with `check_options`). If no offset
    was detected, the copy is unchanged and the log says why. The same offset is applied to every IMU, because
    they share one clock.

    Note: the correction cannot be validated with the same cross-correlation (it returns zero afterwards by
    construction). Validate with injected offsets or with the gyroscope audit instead.
    """
    if result is None:
        result = check_for_synchronisation_offset(recording, **check_options)
    new = recording.copy()
    if not result.detected:
        new.log_correction(ERROR, f'Not applied: {result.summary}')
        return new
    for seg in new.segments.values():
        seg.imu.start_s -= result.offset_s
    new.log_correction(ERROR, f'Moved all IMU time stamps by {-result.offset_s * 1e3:+.1f} ms to match the reference.',
                       offset_s=result.offset_s, spread_s=result.spread_s)
    return new


# Participants whose IMUs each run on their own clock, {dataset: {participant}}. Trento
# participant 9: its five sensors are about 2.94 s apart from each other. Everywhere else the IMUs share one clock,
# so the median over the trusted segments is the better estimate for every IMU (the segments' own estimates scatter
# by 0-3 ms around it, and the rare large disagreements are estimation failures or other errors, not offsets).
SEPARATE_CLOCKS = {'bonfiglio': {'9'}}


def on_separate_clocks(dataset: str, participant) -> bool:
    return str(participant) in SEPARATE_CLOCKS.get(dataset, set())


def remove_sensor_offsets(recording: Recording, sensor_offsets_s: dict, applied_s: float) -> Recording:
    """Second step for IMUs on separate clocks: after remove_synchronisation_offset moved every IMU by the trial's
    offset `applied_s` (0 if none was applied), move each IMU with a trusted offset of its own (`sensor_offsets_s`,
    {segment: offset in s against the reference's clock}) by the rest, so every such IMU ends on the reference's
    clock. IMUs without a trusted offset keep the trial's shift and are logged as not time-aligned."""
    new = recording.copy()
    moved, not_aligned = {}, []
    for name, seg in new.segments.items():
        own = sensor_offsets_s.get(name)
        if own is None or not np.isfinite(own):
            not_aligned.append(name)
            continue
        seg.imu.start_s -= own - applied_s
        moved[name] = round((own - applied_s) * 1e3, 2)
    new.log_correction(ERROR, f'IMUs on separate clocks: moved {len(moved)} IMUs by their own remaining offset; '
                              f'not time-aligned: {", ".join(not_aligned) or "none"}.',
                       moved_ms=moved, not_time_aligned=not_aligned)
    return new


def gyroscope_audit(segment: Segment, window_s: float = 1.0) -> float:
    """Median disagreement in degrees between gyroscope and reference over non-overlapping windows.

    For every window of `window_s` on the reference time base: integrate the gyroscope (rates interpolated at
    the IMU time stamps, so a shifted IMU clock is respected) into one rotation, take the reference's rotation
    between the window's first and last frame, and compare the two rotation angles. Only the angles are
    compared, not the axes, so the audit works before the mounting is known and is not affected by the
    reference convention. Windows touching samples without data (invalid IMU samples, reference frames without
    an orientation) are skipped; nothing else is.
    Returns NaN if no window is usable. A gyroscope offset not yet removed raises the value for every window
    alike, so the audit compares before and after a correction, not across datasets.
    """
    imu, ref = segment.imu, segment.reference
    n = int(round(window_s * ref.rate_hz))
    bad = ~ref.valid
    starts = np.arange(0, len(ref) - n, n)
    t0 = ref.start_s + starts / ref.rate_hz
    t1 = t0 + n / ref.rate_hz
    inside = (t0 >= imu.times[0]) & (t1 <= imu.times[-1])
    imu_bad_at = np.interp(np.arange(len(ref)) / ref.rate_hz + ref.start_s, imu.times, (~imu.valid).astype(float))
    usable = [inside[i] and not bad[s:s + n + 1].any() and not imu_bad_at[s:s + n + 1].any()
              for i, s in enumerate(starts)]
    starts, t0 = starts[usable], t0[usable]
    if not len(starts):
        return float('nan')
    m = max(1, int(round(window_s * imu.rate_hz)))  # integration steps per window, at the IMU rate
    dt = window_s / m
    rotation = Rotation.identity(len(starts))
    for j in range(m):
        mid = t0 + (j + 0.5) * dt
        rates = np.column_stack([np.interp(mid, imu.times, imu.gyr[:, i]) for i in range(3)])
        rotation = rotation * Rotation.from_rotvec(rates * dt)
    reference_change = ref.orientation[starts].inv() * ref.orientation[starts + n]
    return float(np.degrees(np.median(np.abs(rotation.magnitude() - reference_change.magnitude()))))


def validate_offset_correction(before: Recording, after: Recording, window_s: float = 1.0) -> dict:
    """Gyroscope audit per segment before and after the offset correction, in degrees per window.

    The correction is confirmed for a segment if the audit got smaller. Returns
    {segment: dict(before_deg, after_deg, improved)} plus '(all)': the medians over segments."""
    out = {}
    for name in before.segments:
        b, a = gyroscope_audit(before[name], window_s), gyroscope_audit(after[name], window_s)
        out[name] = dict(before_deg=b, after_deg=a, improved=bool(a < b) if np.isfinite(a + b) else None)
    b = float(np.nanmedian([v['before_deg'] for v in out.values()]))
    a = float(np.nanmedian([v['after_deg'] for v in out.values()]))
    out['(all)'] = dict(before_deg=b, after_deg=a, improved=bool(a < b) if np.isfinite(a + b) else None)
    return out


# Dataset level: one row per trial (its IMUs share one clock); the offsets of the single segments are kept as JSON for
# the IMUs on separate clocks.

import json  # noqa: E402

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, frame, loads, number, per_trial, prevalence_of, run_trials, truthy  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'decided', 'offset_ms', 'spread_ms',
           'tolerance_ms', 'segments_trusted', 'separate_clocks', 'segments_json', 'error']


def _check_trial(loader, trial, dataset=''):
    rec = loader.load(trial)
    if not rec.segments:
        raise ValueError('no segment has both an IMU and a reference')
    c = check_for_synchronisation_offset(rec)
    segs = {n: dict(offset_ms=round(e['lag'].offset_s * 1e3, 2), trusted=e['detected'] is not None)
            for n, e in c.per_segment.items()}
    participant = str(loader.participant_of(trial))
    return [dict(trial=trial, participant=participant, segment='', detected=c.detected, estimate=c.offset_s * 1e3,
                 unit='ms', decided=c.detected is not None, offset_ms=c.offset_s * 1e3, spread_ms=c.spread_s * 1e3,
                 tolerance_ms=c.thresholds['tolerance_s'] * 1e3,
                 segments_trusted=sum(v['trusted'] for v in segs.values()),
                 separate_clocks=on_separate_clocks(dataset, participant), segments_json=json.dumps(segs), error='')]


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """The time offset of every trial (ranks and block agreement)."""
    from functools import partial
    return frame(run_trials(loader, partial(_check_trial, dataset=dataset), processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials with an offset beyond half a reference frame / the same trials (one offset per trial)."""
    return prevalence_of(rows, 'trials')


def _shift(rec, rows):
    r = rows[0]
    hit = truthy(r.get('detected'))
    offset_s = number(r.get('offset_ms')) / 1e3 if hit else 0.0
    new = rec.copy()
    if hit:
        for seg in new.segments.values():
            seg.imu.start_s -= offset_s
        new.log_correction(ERROR, f'Moved all IMU time stamps by {-offset_s * 1e3:+.1f} ms to match the reference.',
                           offset_s=offset_s, spread_s=number(r.get('spread_ms')) / 1e3)
    if truthy(r.get('separate_clocks')):
        segs = loads(r.get('segments_json'), {})
        own = {n: e['offset_ms'] / 1e3 for n, e in segs.items() if e.get('trusted')}
        new = remove_sensor_offsets(new, own, offset_s)
    return new


def correct(loader, rows):
    """IMU time stamps moved by the trial's offset; on separate clocks each trusted IMU then by its own rest."""
    return corrected(loader, _shift, per_trial(rows), 'time offset')


SPEC = ErrorSpec(12, 'time_offset', 'fix', 'time offset between IMU and reference')
PARTS = {12: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_time_offset = make_handler(PARTS[12])
