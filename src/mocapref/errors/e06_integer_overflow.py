"""Error 6: integer overflow (wrap-around), found without knowing the measurement range (exploratory rule).

A sensor stores each sample as a signed integer. When the true value passes the top of the range R, the integer
wraps to the bottom: the stored value is the true value minus the full span S = 2R, until the true value returns
inside the range, where it jumps back. No information is lost: true = stored + S over the wrapped stretch.

Per channel:
    1. Noise sigma: the channel's own noise (e05_clipping.channel_noise), so every tolerance has the data's scale.
    2. Entry and exit jumps: one sample step x[k] -> x[k+1] between opposite signs, both beyond NEAR (half) of the
       channel's largest reach M (the largest absolute value two consecutive samples both reach). The size of the
       wrap is the step minus the true change over one sample, estimated from the two neighbouring steps:
       S_hat = |step - mean(neighbouring steps)|.
    3. The channel's span S0 = 2 * the largest stored absolute value (a wrapped channel stores nothing beyond its
       range and comes close to it before a wrap; estimating the span from each jump alone lost most wraps in fast
       motion). A jump counts if the change between its two neighbouring steps (the local
       curvature, the error of the neighbours' estimate) is at most CURVATURE_FRACTION (10 %) of S0, and S_hat agrees
       with S0 within SPAN_SIGMA noise (of a step) and SPAN_FRACTION (5 %) of S0. Letting the curvature into the
       tolerance instead flagged 53 fast natural reversals at 59-200 Hz with ranges that changed within a channel.
       An entry jump is paired with the first later jump of the opposite direction, which must agree too; if it does not, the entry
       is rejected (a later jump may not stand in, or the stretch would run over normal samples). The wrapped stretch is the
       samples between the two jumps.
    4. The stored stretch stays on the opposite side: every stored sample of the stretch has the sign opposite to
       the overflow (within PLATEAU noise of zero), as it must while the true value lies between R and 2R. A
       natural fast reversal followed by ordinary motion crosses back through zero and fails this.
    5. Smooth junctions: at both ends the neighbouring steps are small against the span (at most SMOOTH_FRACTION,
       a quarter, of S; a swing near the sampling rate fails), and the corrected step is no larger than the
       neighbouring steps by more than JUMP_SIGMA noise: |corrected step - mean(neighbouring steps)| <=
       JUMP_SIGMA * sigma * sqrt(6) + the difference between the two neighbouring steps (the local curvature).
    5b. Shorter than its surroundings: the wrapped stretch must be shorter than the stretch from the previous jump
       candidate to its entry and from its exit to the next jump candidate. A range is chosen so that the signal
       lies inside it most of the time; locally the exit of a wrap at the bottom looks exactly like the entry of a
       wrap at the top, and without this rule the in-range samples between two wraps of a one-sided signal
       (gravity on an accelerometer axis) were taken as a wrapped stretch (wraps injected into Delaware compliant
       terrain: 3.2 million false samples).
    6. Range: R = S0 / 2 at least MIN_LIMIT from zero (1 g, 125 deg/s, as for clipping).
    7. Undecided: an entry jump that passes steps 2, 4 (up to the trial end) and 5 but has no exit, because the trial
       ends or values go missing while wrapped; reported, not flagged.
The span used for the correction is the median of the spans of all accepted jumps of the channel (the hardware fixes
it); unwrapping is then exact up to the error of that median, not bit-exact.
Not separable: a single sample flipped to exactly the negative of its neighbour at the channel's peak looks like a
one-sample wrap of a true value at the range limit.
Several wraps in a row (true value beyond 3R) are not handled: they would appear as a second entry inside the
stretch, and the stretch then fails step 4.
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording
from ._result import CheckResult, entry
from .e05_clipping import MIN_LIMIT, SENSORS, channel_noise, minimum_limits, peak_reach

ERROR = 'integer overflow'
NEAR = 0.5
SPAN_SIGMA = 5.0
SPAN_FRACTION = 0.05
CURVATURE_FRACTION = 0.10
JUMP_SIGMA = 5.0
SMOOTH_FRACTION = 0.25
SIGN_SIGMA = 3.0
MODES = ('unwrap', 'mask')


def _true_step(x: np.ndarray, k: int) -> float:
    """The change over one sample at k -> k+1 expected from the neighbouring steps (NaN where they are missing)."""
    before = x[k] - x[k - 1] if k >= 1 else np.nan
    after = x[k + 2] - x[k + 1] if k + 2 < len(x) else np.nan
    steps = [s for s in (before, after) if np.isfinite(s)]
    return float(np.mean(steps)) if steps else float('nan')


def _curvature(x: np.ndarray, k: int) -> float:
    before = x[k] - x[k - 1] if k >= 1 else np.nan
    after = x[k + 2] - x[k + 1] if k + 2 < len(x) else np.nan
    return float(abs(after - before)) if np.isfinite(before) and np.isfinite(after) else 0.0


def jump_candidates(x: np.ndarray, reach: float) -> list[dict]:
    """Step 2: every one-sample step between opposite signs with both samples beyond NEAR * reach."""
    x = np.asarray(x, float)
    a, b = x[:-1], x[1:]
    ok = np.isfinite(a) & np.isfinite(b) & (np.sign(a) * np.sign(b) < 0)
    ok &= (np.abs(a) >= NEAR * reach) & (np.abs(b) >= NEAR * reach)
    out = []
    for k in np.flatnonzero(ok):
        true = _true_step(x, int(k))
        if not np.isfinite(true):
            continue
        step = x[k + 1] - x[k]
        out.append(dict(k=int(k), direction=int(np.sign(step)), span=float(abs(step - true)),
                        curvature=_curvature(x, int(k))))
    return out


def _agrees(jump: dict, span0: float, sigma: float) -> bool:
    """Step 3: the jump's span (step minus the neighbours' change) equals the channel's span within the noise, the
    local curvature (the error of the neighbours' estimate) and SPAN_FRACTION of the span."""
    if jump['curvature'] > CURVATURE_FRACTION * span0:
        return False  # the neighbouring steps change too much to tell a wrap from a fast natural reversal
    return abs(jump['span'] - span0) <= SPAN_SIGMA * sigma * np.sqrt(6) + SPAN_FRACTION * span0


def _smooth(y: np.ndarray, k: int, sigma: float, span: float) -> bool:
    """Step 5 on the corrected series y at the step k -> k+1."""
    true = _true_step(y, k)
    if not np.isfinite(true):
        return False
    neighbours = [abs(y[k] - y[k - 1]) if k >= 1 else 0.0, abs(y[k + 2] - y[k + 1]) if k + 2 < len(y) else 0.0]
    if max(neighbours) > SMOOTH_FRACTION * span:
        return False
    return abs((y[k + 1] - y[k]) - true) <= (JUMP_SIGMA * sigma * np.sqrt(6) + _curvature(y, k)
                                             + SPAN_FRACTION * span)


def overflow_stretches(x: np.ndarray, rate_hz: float, minimum_limit: float = 0.0, log: list | None = None) -> list[dict]:
    """Every wrapped stretch of one channel: dict(first, samples, span, range, side, verdict, reason), verdict
    'overflow' (steps 2-6 passed) or 'undecided' (no exit, step 7). first/samples index the stored (wrapped)
    samples; the correction adds sign * span to them (sign +1 for an overflow at the top, -1 at the bottom).
    log: if a list, (sample index of the entry jump, the rule that rejected it) is appended for every rejected jump
    candidate, for diagnosis."""
    reject = (lambda k, why: log.append((k, why))) if log is not None else (lambda k, why: None)
    x = np.asarray(x, float)
    sigma = channel_noise(x, rate_hz)
    reach = peak_reach(x)
    if not np.isfinite(reach) or not np.isfinite(sigma) or reach <= 0:
        return []
    jumps = jump_candidates(x, reach)
    stored_max = float(np.nanmax(np.abs(x)))
    span0 = 2 * stored_max  # a wrapped channel stores nothing beyond its range and comes close to it before a wrap
    if span0 / 2 < minimum_limit:
        return []  # step 6: no sensor has a range this small
    out, used = [], set()
    for i, j in enumerate(jumps):
        if j['k'] in used:
            continue
        if not _agrees(j, span0, sigma):
            reject(j['k'], 'range')
            continue  # step 3: the jump is not the channel's full span
        span = span0
        side = -j['direction']  # a downward entry jump (+ to -) is an overflow at the top (+1)
        exit_ = next((e for e in jumps[i + 1:] if e['direction'] == -j['direction']), None)
        if exit_ is not None and not _agrees(exit_, span0, sigma):
            reject(j['k'], 'exit span')
            continue  # the first jump back is not a full span: not this wrap's exit, and no later one may stand in
        start = j['k'] + 1
        stop = exit_['k'] + 1 if exit_ else len(x)
        stretch = x[start:stop]
        if exit_ is None:
            missing = np.flatnonzero(~np.isfinite(stretch))
            if len(missing):
                stop = start + int(missing[0])
                stretch = x[start:stop]
        if len(stretch) == 0:
            continue
        corrected = x.copy()
        corrected[start:stop] = stretch + side * span
        opposite = np.all(side * stretch <= SIGN_SIGMA * sigma)
        if not opposite:
            reject(j['k'], 'sign')
            continue
        if not _smooth(corrected, j['k'], sigma, span) or (exit_ is not None and
                                                           not _smooth(corrected, exit_['k'], sigma, span)):
            reject(j['k'], 'smooth')
            continue
        k_prev = max((e['k'] for e in jumps if e['k'] < j['k']), default=-1)
        k_next = min((e['k'] for e in jumps if e['k'] >= stop), default=len(x) - 1)
        if exit_ is not None and (stop - start >= j['k'] - k_prev or stop - start >= k_next - exit_['k']):
            reject(j['k'], 'longer')
            continue  # longer than the in-range stretch beside it: more likely the complement of a wrap
        spans = [span] + ([exit_['span']] if exit_ else [])
        record = dict(first=start, samples=stop - start, event_spans=spans, side=side, noise=sigma)
        if exit_ is None:
            out.append(dict(record, verdict='undecided', reason='no exit jump before the end or a gap'))
        else:
            used.add(exit_['k'])
            out.append(dict(record, verdict='overflow', reason=''))
    accepted = [v for r in out if r['verdict'] == 'overflow' for v in r['event_spans']]
    for r in out:  # one span per channel: the median over its accepted jumps (undecided ones use their own)
        r['span'] = float(np.median(accepted)) if accepted else float(np.median(r['event_spans']))
        r['span0'] = span0
        r['range'] = r['span'] / 2
    return out


def overflow_in_imu(signal: np.ndarray, rate_hz: float, minimum_limit=None) -> list[dict]:
    """Stretches of every column of one IMU (gyroscope x, y, z, accelerometer x, y, z), with the column."""
    signal = np.asarray(signal, float)
    floor = np.zeros(signal.shape[1]) if minimum_limit is None else np.asarray(minimum_limit, float)
    out = []
    for c in range(signal.shape[1]):
        out += [dict(s, axis=c) for s in overflow_stretches(signal[:, c], rate_hz, floor[c])]
    return out


def _summary(records: list[dict], n: int) -> dict:
    wrapped = np.zeros(n, bool)
    for r in records:
        if r['verdict'] == 'overflow':
            wrapped[r['first']:r['first'] + r['samples']] = True
    hits = [r for r in records if r['verdict'] == 'overflow']
    name = lambda r: f"{'gyr' if r['axis'] < 3 else 'acc'}{r['axis'] % 3}{'p' if r['side'] > 0 else 'n'}"
    return entry(bool(hits), int(wrapped.sum()), 'samples', stretches=len(hits),
                 undecided=sum(r['verdict'] == 'undecided' for r in records),
                 channels=sorted({name(r) for r in hits}),
                 ranges={name(r): r['range'] for r in hits})


def check_for_integer_overflow(recording: Recording) -> CheckResult:
    """Per segment: samples inside a wrapped stretch (gyroscope and accelerometer), and undecided stretches."""
    per_segment = {}
    for name, seg in recording.segments.items():
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        per_segment[name] = _summary(overflow_in_imu(both, seg.imu.rate_hz, minimum_limits()), len(both))
    hit = [n for n, p in per_segment.items() if p['detected']]
    summary = f'Integer overflow in {", ".join(hit)}.' if hit else 'No integer overflow found.'
    return CheckResult(ERROR, bool(hit), summary, per_segment,
                       dict(near=NEAR, span_sigma=SPAN_SIGMA, span_fraction=SPAN_FRACTION,
                            jump_sigma=JUMP_SIGMA, minimum_limit=MIN_LIMIT))


def remove_integer_overflow(recording: Recording, mode: str = 'unwrap') -> Recording:
    """Return a copy with every wrapped stretch unwrapped (mode='unwrap', the correction of the order: stored value
    + or - the channel's span, exact up to the error of the span estimate if the true value stayed within one wrap) or, on request, marked invalid (mode='mask', all channels of that IMU sample).
    Undecided stretches (no exit jump) are left unchanged."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    new = recording.copy()
    changed = {}
    for name, seg in new.segments.items():
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        records = [r for r in overflow_in_imu(both, seg.imu.rate_hz, minimum_limits()) if r['verdict'] == 'overflow']
        for r in records:
            rows = slice(r['first'], r['first'] + r['samples'])
            if mode == 'unwrap':
                target = seg.imu.gyr if r['axis'] < 3 else seg.imu.acc
                target[rows, r['axis'] % 3] += r['side'] * r['span']
            else:
                seg.imu.valid[rows] = False
        changed[name] = int(sum(r['samples'] for r in records))
    verb = 'Unwrapped' if mode == 'unwrap' else 'Marked invalid'
    new.log_correction(ERROR, f'{verb} {sum(changed.values())} overflowed IMU samples.', samples=changed, mode=mode)
    return new


__all__ = ['check_for_integer_overflow', 'remove_integer_overflow', 'overflow_stretches', 'SENSORS', 'check_dataset',
           'prevalence', 'correct', 'SPEC', 'PARTS', 'handle_integer_overflow']


# --- handler: one dataset at a time -------------------------------------------------------------------------------

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import dumps, loads, parallel_map, reader_of, truthy  # noqa: E402


def _check_trial(loader, trial) -> list[dict]:
    rec = loader.load(trial)
    rows = []
    for seg_name, seg in rec.segments.items():
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        records = overflow_in_imu(both, seg.imu.rate_hz, minimum_limits())
        hits = [r for r in records if r['verdict'] == 'overflow']
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), segment=seg_name,
                         detected=bool(hits), estimate=len(hits), unit='stretches',
                         wrapped_samples=int(sum(r['samples'] for r in hits)),
                         undecided=sum(r['verdict'] == 'undecided' for r in records),
                         stretches=dumps([dict(axis=r['axis'], first=r['first'], samples=r['samples'],
                                               side=r['side'], span=r['span']) for r in hits])))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: the wrapped stretches (estimate; every channel counted apart), each with its
    channel, samples, side and span in `stretches`; entry jumps without an exit are counted in undecided."""
    rows = []
    for out in parallel_map(loader, _check_trial, loader.trials(), processes):
        rows += [dict(out, segment='', detected=None)] if isinstance(out, dict) else out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with a wrapped stretch / wrapped stretches."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'stretches', int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def _unwrap(rec: Recording, trial, *, stored) -> Recording:
    """The stored stretches ({trial: {segment: [stretch]}}) unwrapped, or, where `stored` is None, the trial's own
    stretches found again (another reader than the check's)."""
    if stored is None:
        return remove_integer_overflow(rec)
    if str(trial) not in stored:
        return rec
    new, changed = rec.copy(), {}
    for name, records in stored[str(trial)].items():
        if name not in new.segments:
            continue
        imu = new.segments[name].imu
        for r in records:
            target = imu.gyr if r['axis'] < 3 else imu.acc
            target[r['first']:r['first'] + r['samples'], r['axis'] % 3] += r['side'] * r['span']
        changed[name] = int(sum(r['samples'] for r in records))
    new.log_correction(ERROR, f'Unwrapped {sum(changed.values())} overflowed IMU samples.', samples=changed,
                       mode='unwrap')
    return new


def correct(loader, rows: pd.DataFrame):
    """Every wrapped stretch unwrapped: stored value + or - the channel's span."""
    if len(rows) and 'reader' in rows and rows.reader.iloc[0] != reader_of(loader):
        return CorrectedLoader(loader, partial(_unwrap, stored=None), ERROR)
    stored = {}
    for r in rows[rows.detected.map(truthy)].itertuples():
        stored.setdefault(str(r.trial), {})[r.segment] = loads(r.stretches, [])
    if not stored:
        return loader
    return CorrectedLoader(loader, partial(_unwrap, stored=stored), ERROR)


SPEC = ErrorSpec(6, 'integer_overflow', 'fix', 'integer overflow (wrap-around)')
PARTS = {6: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_integer_overflow = make_handler(PARTS[6])
