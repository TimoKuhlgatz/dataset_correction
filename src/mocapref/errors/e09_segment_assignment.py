"""Error 9: IMU on the wrong segment: the IMU stream of one segment belongs to another segment.

What it is: the deposit files an IMU under a segment it was not mounted on (left and right exchanged, or two
neighbouring segments). Each stream is real motion of the body, so nothing looks broken; only the comparison with
the reference of each segment shows it.

How we catch it. Assignment errors are checked before the time offset, the gyroscope
offset and the sensor-to-reference rotation, so the check must not depend on their correction; it is built to be
insensitive to them instead:
    1. Common time offset without using the assignment. All IMUs are assumed to share one clock. The mean rotation
       speed of all IMUs is correlated with the mean rotation speed of all reference segments over the full lag
       range (e12_time_offset.estimate_time_lag). Exchanging IMUs only reorders the terms of these means,
       so this offset is the same whether or not an IMU sits on the wrong segment. A drifting common clock is
       followed by refining the common lag in each window within +-DRIFT_SEARCH_S.
    2. Assignment at zero lag after shifting by the common lag. The trial is cut into consecutive windows of
       WINDOW_S; in each window every IMU is scored against every segment's simulated gyroscope (5 Hz low-pass)
       with two scores:
         speed:  correlation of the rotation speeds |w| (blind to any sensor-to-reference rotation; a gyroscope
                 offset cancels only approximately);
         fitted: both 3-axis rates minus their means (removes a constant gyroscope offset exactly), the best
                 constant rotation between them (least squares), score = share of the gyroscope's variance it
                 explains.
       The best one-to-one assignment of IMUs to segments (largest total score) is the window's vote. An IMU is on
       the wrong segment if more than half of its windows assign it to one other segment, correctly placed if more
       than half assign it to its declared segment, undecided otherwise (no score threshold).
    3. Individual offsets. With the assignment of step 2, each IMU's own offset against its segment is estimated
       (rotation speeds, within +-MAX_INDIVIDUAL_S of the common lag), and step 2 is run again with these offsets
       applied. An IMU whose decision changes is undecided: step 2 holds only if the individual offsets are small.
    The two scores are reported separately; the IMU's decision is theirs where they agree and undecided where they
    differ. An undecided IMU keeps its declared segment (without enough evidence the deposit's assignment is not
    doubted); `undecided` in its entry records that the evidence was not conclusive.

How we correct it: give each segment the IMU stream the check assigned to it, if the detected IMUs exchange their
segments among themselves (a permutation); otherwise nothing is changed.

Known limits: when two segments move alike (both legs in phase in a squat, pelvis and trunk in walking), their
IMUs cannot be told apart and the windows disagree (undecided, kept as declared). Windows at rest vote at random. All parameters are
exploratory defaults.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt

from ..recording import Recording, Segment
from ..rotations import reference_angular_rate
from ._result import CheckResult, entry
from .e12_time_offset import estimate_time_lag

ERROR = 'segment assignment'
WINDOW_S = 10.0
LOWPASS_HZ = 5.0
MAX_GRID_HZ = 100.0
MAX_COMMON_OFFSET_S = 10.0
DRIFT_SEARCH_S = 0.25
MAX_INDIVIDUAL_S = 1.0
SCORES = ('speed', 'fitted')


def _lowpass(x: np.ndarray, rate_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth at LOWPASS_HZ along time; missing samples are filled linearly for the
    filter only and are missing again afterwards."""
    x = np.array(x, float)
    bad = ~np.isfinite(x).all(axis=1)
    if bad.all() or len(x) < 30 or LOWPASS_HZ >= rate_hz / 2:
        return x
    k = np.arange(len(x))
    for j in range(x.shape[1]):
        x[bad, j] = np.interp(k[bad], k[~bad], x[~bad, j])
    sos = butter(4, LOWPASS_HZ / (rate_hz / 2), output='sos')
    out = sosfiltfilt(sos, x, axis=0)
    out[bad] = np.nan
    return out


def _sample(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """values (N, d) given at `times`, read at `at` by linear interpolation; NaN outside the span and next to
    missing samples."""
    out = np.full((len(at), values.shape[1]), np.nan)
    inside = (at >= times[0]) & (at <= times[-1])
    if not inside.any():
        return out
    for j in range(values.shape[1]):
        out[inside, j] = np.interp(at[inside], times, values[:, j])  # NaN neighbours give NaN
    return out


class _Streams:
    """The low-passed gyroscope of every IMU and the simulated gyroscope of every segment, ready to be read at any
    time. A reference whose convention is unknown is kept in both conventions; the fitted score takes the better."""

    def __init__(self, recording: Recording):
        self.names = recording.segment_names
        self.imu, self.ref = {}, {}
        for name, seg in recording.segments.items():
            imu = seg.imu
            gyr = np.where(imu.valid[:, None], imu.gyr, np.nan)
            self.imu[name] = (imu.times, _lowpass(gyr, imu.rate_hz))
            ref = seg.reference
            versions = []
            for conv in (['world_to_segment', 'segment_to_world'] if ref.convention == 'unknown'
                         else [ref.convention]):
                t, rates = reference_angular_rate(ref.orientation, ref.rate_hz, conv)
                rates = np.where((ref.valid[:-1] & ref.valid[1:])[:, None], rates, np.nan)
                versions.append(_lowpass(rates, ref.rate_hz))
            self.ref[name] = (t + ref.start_s, versions)
        self.grid_hz = min(MAX_GRID_HZ, min(s.reference.rate_hz for s in recording.segments.values()))
        starts = [t[0] for t, _ in self.ref.values()]
        ends = [t[-1] for t, _ in self.ref.values()]
        self.grid = np.arange(min(starts), max(ends), 1 / self.grid_hz)
        self.ref_on_grid = {n: [_sample(t, v, self.grid) for v in vs] for n, (t, vs) in self.ref.items()}

    def imu_at(self, name: str, times: np.ndarray) -> np.ndarray:
        t, g = self.imu[name]
        return _sample(t, g, times)


def common_offset(streams: _Streams, max_offset_s: float = MAX_COMMON_OFFSET_S):
    """Step 1: the IMUs' common lag against the reference from the mean rotation speeds (independent of which IMU
    is filed under which segment). Returns the LagEstimate (offset_s > 0: IMU time stamps late)."""
    imu_speed = np.nanmean([np.linalg.norm(streams.imu_at(n, streams.grid), axis=1) for n in streams.names], axis=0)
    ref_speed = np.nanmean([np.linalg.norm(streams.ref_on_grid[n][0], axis=1) for n in streams.names], axis=0)
    # the IMU curve is read on the reference grid: the lag search then runs on one time base
    return estimate_time_lag(streams.grid, imu_speed, streams.grid, ref_speed, max_offset_s,
                             grid_hz=streams.grid_hz)


def _windows(streams: _Streams, window_s: float) -> list[np.ndarray]:
    n = int(round(window_s * streams.grid_hz))
    return [np.arange(k, k + n) for k in range(0, len(streams.grid) - n + 1, n)]


def _window_lag(streams: _Streams, idx: np.ndarray, base: float) -> float:
    """Common lag in one window, refined within +-DRIFT_SEARCH_S of the trial's common lag (mean speeds)."""
    tau = streams.grid[idx]
    ref = np.nanmean([np.linalg.norm(streams.ref_on_grid[n][0][idx], axis=1) for n in streams.names], axis=0)
    best, best_c = base, -np.inf
    for d in np.arange(-DRIFT_SEARCH_S, DRIFT_SEARCH_S + 1e-9, 1 / streams.grid_hz):
        imu = np.nanmean([np.linalg.norm(streams.imu_at(n, tau + base + d), axis=1) for n in streams.names], axis=0)
        c = _corr(imu, ref)
        if np.isfinite(c) and c > best_c:
            best, best_c = base + d, c
    return best


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def _explained(p: np.ndarray, q: np.ndarray) -> float:
    """Share of q's variance explained by the best constant rotation of p (both minus their means)."""
    ok = np.isfinite(p).all(1) & np.isfinite(q).all(1)
    if ok.sum() < 10:
        return np.nan
    pc, qc = p[ok] - p[ok].mean(0), q[ok] - q[ok].mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    m = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    total = (qc * qc).sum()
    return float(1 - ((qc - pc @ m.T) ** 2).sum() / total) if total > 0 else np.nan


def _window_scores(streams: _Streams, idx: np.ndarray, lag: float, own: dict) -> dict:
    """Score matrices (IMU x segment) of one window for both scores; `own` holds each IMU's individual lag."""
    tau = streams.grid[idx]
    n = len(streams.names)
    out = {s: np.full((n, n), np.nan) for s in SCORES}
    imu = {name: streams.imu_at(name, tau + lag + own.get(name, 0.0)) for name in streams.names}
    for i, a in enumerate(streams.names):
        for j, b in enumerate(streams.names):
            refs = [v[idx] for v in streams.ref_on_grid[b]]
            out['speed'][i, j] = _corr(np.linalg.norm(imu[a], axis=1), np.linalg.norm(refs[0], axis=1))
            fitted = [_explained(r, imu[a]) for r in refs]
            out['fitted'][i, j] = np.nan if np.isnan(fitted).all() else np.nanmax(fitted)
    return out


def _votes(streams: _Streams, windows: list, lags: list, own: dict) -> dict:
    """Per score and IMU: the segment each usable window assigns it to."""
    votes = {s: {name: [] for name in streams.names} for s in SCORES}
    for idx, lag in zip(windows, lags):
        scores = _window_scores(streams, idx, lag, own)
        for s, m in scores.items():
            rows, cols = linear_sum_assignment(-np.nan_to_num(m, nan=-1.0))
            for i, j in zip(rows, cols):
                if np.isfinite(m[i, i]) and np.isfinite(m[i, j]):  # declared and assigned segment both scored
                    votes[s][streams.names[i]].append(streams.names[j])
    return votes


def _decide(name: str, votes: list) -> tuple[bool | None, str | None]:
    """More than half of the windows on one other segment: wrong (and which); on the declared one: right."""
    if not votes:
        return None, None
    values, counts = np.unique(votes, return_counts=True)
    top = int(np.argmax(counts))
    if counts[top] * 2 <= len(votes):
        return None, None
    return (values[top] != name), str(values[top])


def check_for_segment_assignment(recording: Recording, window_s: float = WINDOW_S) -> CheckResult:
    """Is an IMU of this recording filed under the wrong segment? Steps 1-3 of the module description."""
    thresholds = dict(window_s=window_s, lowpass_hz=LOWPASS_HZ, max_common_offset_s=MAX_COMMON_OFFSET_S,
                      drift_search_s=DRIFT_SEARCH_S, max_individual_s=MAX_INDIVIDUAL_S, rule='majority of windows')
    if len(recording.segments) < 2:
        return CheckResult(ERROR, None, 'Only one segment: nothing to exchange.', {}, thresholds)
    streams = _Streams(recording)
    lag = common_offset(streams)
    windows = _windows(streams, window_s)
    if not windows:
        return CheckResult(ERROR, None, f'Shorter than one window of {window_s:g} s.', {}, thresholds)
    lags = [_window_lag(streams, idx, lag.offset_s) for idx in windows]
    first = _votes(streams, windows, lags, {})
    decided = {s: {n: _decide(n, first[s][n]) for n in streams.names} for s in SCORES}
    # step 3: individual offsets against the segment the IMU is assigned to (its declared one if not moved)
    own = {}
    for name in streams.names:
        flagged = [decided[s][name] for s in SCORES if decided[s][name][0]]
        target = flagged[0][1] if flagged else name
        t_ref, versions = streams.ref[target]
        ref_speed = np.linalg.norm(versions[0], axis=1)
        t_imu, g = streams.imu[name]
        est = estimate_time_lag(t_imu - lag.offset_s, np.linalg.norm(g, axis=1), t_ref, ref_speed,
                                MAX_INDIVIDUAL_S, grid_hz=streams.grid_hz)
        own[name] = est.offset_s
    second = _votes(streams, windows, lags, own)
    per_segment, any_wrong, any_undecided = {}, False, False
    for name in streams.names:
        final = {}
        for s in SCORES:
            d1, seg1 = decided[s][name]
            d2, seg2 = _decide(name, second[s][name])
            final[s] = (d1, seg1) if (d1, seg1) == (d2, seg2) else (None, None)
        (ds, segs), (df, segf) = final['speed'], final['fitted']
        detected, belongs_to = ((ds, segs) if (ds, segs) == (df, segf) else (None, None))
        undecided = detected is None
        if undecided:  # not enough evidence to doubt the deposit: keep the declared segment
            detected, belongs_to = False, name
        n_votes = len(first['fitted'][name])
        wrong_share = (sum(v != name for v in first['fitted'][name]) / n_votes) if n_votes else np.nan
        per_segment[name] = entry(
            detected, wrong_share, 'share of windows on another segment (fitted)', None,
            belongs_to=belongs_to, undecided=undecided, speed=dict(zip(('detected', 'segment'), final['speed'])),
            fitted=dict(zip(('detected', 'segment'), final['fitted'])),
            votes_speed=dict(zip(*np.unique(first['speed'][name], return_counts=True))) if first['speed'][name] else {},
            votes_fitted=dict(zip(*np.unique(first['fitted'][name], return_counts=True))) if n_votes else {},
            individual_offset_ms=own[name] * 1e3)
        any_wrong |= detected is True
    detected = any_wrong
    wrong = [f'{n} -> {e["belongs_to"]}' for n, e in per_segment.items() if e['detected']]
    undecided = [n for n, e in per_segment.items() if e['undecided']]
    summary = (f'Common IMU offset {lag.offset_s * 1e3:+.0f} ms (peak correlation {lag.peak_correlation:.2f}), '
               f'{len(windows)} windows of {window_s:g} s. '
               + (f'On the wrong segment: {", ".join(wrong)}. ' if wrong else 'No IMU on another segment. ')
               + (f'Undecided, kept as declared: {", ".join(undecided)}.' if undecided else ''))
    return CheckResult(ERROR, detected, summary.strip(), per_segment, thresholds, estimate=float(len(wrong)),
                       unit='IMUs on the wrong segment')


def remove_segment_assignment(recording: Recording, result: CheckResult | None = None) -> Recording:
    """A copy in which each segment gets the IMU stream the check assigned to it, if the detected IMUs exchange
    their segments among themselves; otherwise the copy is unchanged and the log says why."""
    if result is None:
        result = check_for_segment_assignment(recording)
    new = recording.copy()
    moves = {n: e['belongs_to'] for n, e in result.per_segment.items() if e['detected']}
    if not moves:
        new.log_correction(ERROR, f'Not applied: {result.summary}')
        return new
    if sorted(moves) != sorted(moves.values()):
        new.log_correction(ERROR, f'Not applied: the detected IMUs do not exchange their segments among '
                                  f'themselves ({moves}).')
        return new
    for source, target in moves.items():
        new.segments[target] = Segment(target, recording.segments[source].imu, recording.segments[target].reference)
    new.log_correction(ERROR, 'Gave the IMU streams to the segments they belong to: '
                       + ', '.join(f'{s} -> {t}' for s, t in moves.items()) + '.', moves=moves)
    return new


# --- handler: one dataset at a time -------------------------------------------------------------------------------

from functools import partial  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import parallel_map, reader_of, truthy  # noqa: E402


def _check_trial(loader, trial) -> list[dict]:
    rec = loader.load(trial)
    if not rec.segments:
        return []
    result = check_for_segment_assignment(rec)
    base = dict(trial=trial, participant=str(loader.participant_of(trial)))
    if not result.per_segment:  # one segment only, or shorter than one window
        return [dict(base, segment='', detected=None, unit='IMUs', reason=result.summary)]
    return [dict(base, segment=seg, detected=e['detected'], estimate=e['estimate'],
                 unit='share of windows on another segment (fitted)', belongs_to=e['belongs_to'],
                 undecided=e['undecided'], speed=e['speed']['detected'], speed_segment=e['speed']['segment'],
                 fitted=e['fitted']['detected'], fitted_segment=e['fitted']['segment'],
                 individual_offset_ms=e['individual_offset_ms'])
            for seg, e in result.per_segment.items()]


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and IMU: detected where the majority of its windows put it on one other segment
    (belongs_to), by both scores; an undecided IMU is kept as declared (undecided True, detected False). A trial with
    one segment or shorter than one window gives one undecided row."""
    rows = []
    for out in parallel_map(loader, _check_trial, loader.trials(), processes):
        rows += [dict(out, segment='', detected=None)] if isinstance(out, dict) else out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with an IMU on another segment / such IMUs."""
    return count_trials(rows, 'IMUs')


def _reassign(rec: Recording, trial, *, moves) -> Recording:
    per = moves.get(str(trial))
    if not per:
        return rec
    result = SimpleNamespace(per_segment={s: dict(detected=True, belongs_to=b) for s, b in per.items()},
                             summary='no IMU on another segment')
    return remove_segment_assignment(rec, result)


def correct(loader, rows: pd.DataFrame):
    """Each segment given the IMU stream the check assigned to it, where the detected IMUs exchange their segments
    among themselves (otherwise unchanged, with the reason in the correction log)."""
    moves = {}
    for r in rows[rows.detected.map(truthy)].itertuples():
        moves.setdefault(str(r.trial), {})[r.segment] = r.belongs_to
    if not moves:
        return loader
    return CorrectedLoader(loader, partial(_reassign, moves=moves), ERROR)


SPEC = ErrorSpec(9, 'segment_assignment', 'fix', 'IMU on the wrong segment')
PARTS = {9: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_segment_assignment = make_handler(PARTS[9])
