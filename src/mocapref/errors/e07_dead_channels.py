"""Error 7: dead, stuck or railed channel: IMU samples that cannot be repaired, only marked.

Dead or stuck channel: a channel repeats exactly the same value for at least MIN_RUN (5) samples while the sensor is
    moving; moving means the optical reference or another channel of the same IMU varies during
    that stretch by more than N_SIGMA (3) times its own noise (noise from second differences over the whole trial).
    Without the reference or motion: missing values, and an accelerometer reading exactly zero on all three axes
    (impossible under gravity). Time as deposited: the synchronisation is checked later, so a large time offset makes
    the reference test look at the wrong moment.
Railed channel: a channel jumped to a rail and held there, found with the range-free rule of error 5
    (e05_clipping.py): a cut-off plateau of at least RAILED_S, or a channel at one level for most of the trial while
    another channel moves.
Held samples (a tool, not counted): consecutive samples identical on all six channels, and channels that never
    change.

How we correct them: the true value is unknown, so the affected samples are marked invalid (`ImuStream.valid`).
Later steps skip invalid samples instead of using an invented value.

Not for RepoIMU (NO_DEAD_STAGE): its repeated whole samples count as a sampling-rate error (error 11).
"""
from __future__ import annotations

import numpy as np

from ..recording import ImuStream, Recording
from ._result import CheckResult, entry


# --- held samples and dead channels ---------------------------------------------------------------------------

def held_samples(imu: ImuStream) -> np.ndarray:
    """True for samples identical to the previous sample on all six channels (the first of a run stays valid)."""
    both = np.hstack([imu.gyr, imu.acc])
    repeated = np.zeros(len(imu), bool)
    repeated[1:] = np.all(np.diff(both, axis=0) == 0, axis=1)
    return repeated


def check_for_held_samples(recording: Recording) -> CheckResult:
    """Count repeated samples and channels that never change, per segment."""
    per_segment = {}
    for name, seg in recording.segments.items():
        held = held_samples(seg.imu)
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        dead = [i for i in range(6) if np.all(both[:, i] == both[0, i])]
        per_segment[name] = entry(bool(held.any() or dead), int(held.sum()), 'samples', fraction=float(held.mean()),
                                  longest_run=_longest_run(held), dead_channels=dead)
    hit = [n for n, p in per_segment.items() if p['detected']]
    summary = f'Held samples or dead channels in {", ".join(hit)}.' if hit else 'No held samples found.'
    return CheckResult('held samples', bool(hit), summary, per_segment)


def remove_held_samples(recording: Recording) -> Recording:
    """Return a copy with every repeated sample marked invalid."""
    return _mark_invalid(recording, 'held samples', held_samples, 'held')


# --- dead or stuck channels -----------------------------------------------------------------------------------

MIN_RUN = 5
N_SIGMA = 3.0


def noise_std(x: np.ndarray) -> np.ndarray:
    """Noise standard deviation per column from second differences (median absolute deviation scaled to a
    standard deviation; white noise of variance s^2 gives second differences of variance 6 s^2)."""
    d2 = np.diff(x, 2, axis=0)
    d2 = np.where(np.isfinite(d2), d2, np.nan)
    return 1.4826 * np.nanmedian(np.abs(d2 - np.nanmedian(d2, axis=0)), axis=0) / np.sqrt(6)


def constant_runs(x: np.ndarray, min_run: int = MIN_RUN) -> list[tuple[int, int]]:
    """(first, last + 1) of every run of at least min_run exactly equal finite consecutive samples of one channel."""
    same = np.zeros(len(x), bool)
    same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
    edges = np.diff(np.concatenate([[0], same.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1) - 1, np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_run]


def reference_noise(segment) -> np.ndarray:
    """Noise standard deviation of the reference orientation per axis (rad), from the second differences of the
    orientation: the change of the frame-to-frame increments."""
    ref = segment.reference
    increments = (ref.orientation[:-1].inv() * ref.orientation[1:]).as_rotvec()
    increments[~(ref.valid[:-1] & ref.valid[1:])] = np.nan
    d = np.diff(increments, axis=0)
    return 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d, axis=0)), axis=0) / np.sqrt(6)


def _reference_moves(segment, t0: float, t1: float, noise: np.ndarray, n_sigma: float) -> bool:
    """Does the reference orientation vary between t0 and t1 by more than n_sigma times its noise on any axis?"""
    ref = segment.reference
    t = ref.times
    i0 = max(int(np.searchsorted(t, t0, 'right')) - 1, 0)  # the frames spanning the stretch
    i1 = min(int(np.searchsorted(t, t1, 'left')) + 1, len(t))
    keep = np.flatnonzero(ref.valid[i0:i1]) + i0
    if len(keep) < 2:
        return False
    window = (ref.orientation[keep[0]].inv() * ref.orientation[keep]).as_rotvec()  # relative to its first frame
    return bool(np.any(np.std(window, axis=0) > n_sigma * noise))


def stuck_samples(segment, min_run: int = MIN_RUN, n_sigma: float = N_SIGMA) -> tuple[np.ndarray, list[dict]]:
    """Samples of one segment inside a stuck stretch, and one record per stretch (channel, samples, which rule)."""
    imu = segment.imu
    both = np.hstack([imu.gyr, imu.acc])
    noise = noise_std(both)
    ref_noise = reference_noise(segment)
    bad = np.zeros(len(imu), bool)
    stretches = []
    for ch in range(6):
        for a, b in constant_runs(both[:, ch], min_run):
            flat = [c for c in range(6) if np.all(both[a:b, c] == both[a, c])]
            others = [c for c in range(6) if c not in flat]
            other_moves = any(np.nanstd(both[a:b, c]) > n_sigma * noise[c] for c in others)
            ref_moves = _reference_moves(segment, imu.times[a], imu.times[b - 1], ref_noise, n_sigma)
            if other_moves or ref_moves:
                bad[a:b] = True
                stretches.append(dict(channel=ch, first=a, samples=b - a, flat_channels=flat,
                                      reference_moves=ref_moves, other_channels_move=other_moves))
    missing = ~np.isfinite(both).all(1)
    zero_acc = np.all(imu.acc == 0, axis=1)
    return bad | missing | zero_acc, stretches


def check_for_dead_channels(recording: Recording, min_run: int = MIN_RUN, n_sigma: float = N_SIGMA) -> CheckResult:
    """Count samples inside stuck stretches, missing samples and all-zero accelerometer samples, per segment."""
    per_segment = {}
    for name, seg in recording.segments.items():
        bad, stretches = stuck_samples(seg, min_run, n_sigma)
        both = np.hstack([seg.imu.gyr, seg.imu.acc])
        per_segment[name] = entry(bool(bad.any()), int(bad.sum()), 'samples', fraction=float(bad.mean()),
                                  missing=int((~np.isfinite(both).all(1)).sum()),
                                  zero_accelerometer=int(np.all(seg.imu.acc == 0, axis=1).sum()),
                                  stretches=len(stretches),
                                  longest_stretch=max((s['samples'] for s in stretches), default=0),
                                  by_reference=sum(s['reference_moves'] for s in stretches),
                                  by_other_channels=sum(s['other_channels_move'] for s in stretches))
    hit = [n for n, p in per_segment.items() if p['detected']]
    summary = f'Dead or stuck channels in {", ".join(hit)}.' if hit else 'No dead or stuck channels found.'
    return CheckResult('dead channels', bool(hit), summary, per_segment, dict(min_run=min_run, n_sigma=n_sigma))


def remove_dead_channels(recording: Recording, min_run: int = MIN_RUN, n_sigma: float = N_SIGMA) -> Recording:
    """Return a copy with every sample of a stuck stretch, missing or all-zero accelerometer marked invalid (all
    channels of that IMU sample)."""
    new = recording.copy()
    marked = {}
    for name, seg in new.segments.items():
        bad = stuck_samples(seg, min_run, n_sigma)[0] & seg.imu.valid
        seg.imu.valid = seg.imu.valid & ~bad
        marked[name] = int(bad.sum())
    new.log_correction('dead channels', f'Marked {sum(marked.values())} IMU samples of dead or stuck channels invalid.',
                       samples_marked=marked)
    return new


# --- shared ---------------------------------------------------------------------------------------------------

def _mark_invalid(recording: Recording, error: str, find, word: str) -> Recording:
    new = recording.copy()
    marked = {}
    for name, seg in new.segments.items():
        bad = find(seg.imu) & seg.imu.valid
        seg.imu.valid = seg.imu.valid & ~bad
        marked[name] = int(bad.sum())
    new.log_correction(error, f'Marked {sum(marked.values())} {word} IMU samples invalid.', samples_marked=marked)
    return new


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


# --- handler: one dataset at a time -------------------------------------------------------------------------------

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import by_participant, dumps, from_runs, loads, parallel_map, reader_of, runs, truthy  # noqa: E402
from .e05_clipping import (RAIL_LIMIT, SENSORS, cut_off_samples, minimum_limits, plateaus_of,  # noqa: E402
                           with_recurrence)

NO_DEAD_STAGE = {'repoimu'}
"""Readers whose repeated samples count as a sampling-rate error (error 11), not as stuck channels."""


def _check_participant(loader, group) -> list[dict]:
    _, trials = group
    found, stuck = {}, {}
    for t in trials:
        rec = loader.load(t)
        if not rec.segments:
            continue
        found[t] = plateaus_of(rec)
        stuck[t] = {}
        for name, seg in rec.segments.items():
            bad, _ = stuck_samples(seg)
            both = np.hstack([seg.imu.gyr, seg.imu.acc])
            earlier = np.isfinite(both).all(1) & ~seg.imu.valid  # marked by an earlier correction (clipping)
            stuck[t][name] = (bad, len(runs(bad & ~earlier)))
    rows = []
    for t, segs in with_recurrence(found).items():
        for name, (_, railed, records) in segs.items():
            bad, n_stuck = stuck[t][name]
            channels = len({(p['axis'], p['side']) for p in records
                            if p['verdict'] == 'cut off' and p.get('kind') == 'railed'})
            rows.append(dict(trial=t, participant=str(loader.participant_of(t)), segment=name,
                             detected=n_stuck + channels > 0, estimate=n_stuck + channels,
                             unit='stuck runs and railed channels', stuck_runs=n_stuck, railed_channels=channels,
                             stuck_samples=int(bad.sum()), railed_samples=int(railed.sum()),
                             mask_runs=dumps(runs(bad | railed))))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: runs of dead or stuck samples of the IMU (any channel; samples an earlier
    correction already marked invalid are left out) plus railed channels (channel and side, with recurrence across
    the participant's trials as for clipping); mask_runs holds every sample the correction marks. RepoIMU is not
    checked (NO_DEAD_STAGE) and counts 0."""
    reader = reader_of(loader)
    if reader in NO_DEAD_STAGE:
        rows = pd.DataFrame([dict(trial=t, participant=str(loader.participant_of(t)), segment='', detected=False,
                                  estimate=0, unit='stuck runs and railed channels',
                                  reason='repeated samples count as a sampling-rate error (error 11)')
                             for t in loader.trials()])
    else:
        rows = []
        for out in parallel_map(loader, _check_participant, by_participant(loader), processes):
            if isinstance(out, dict):  # a participant's trials could not be read
                rows += [dict(trial=t, participant=str(loader.participant_of(t)), segment='', detected=None,
                              error=out['error']) for t in out['trial'][1]]
            else:
                rows += out
        rows = pd.DataFrame(rows)
    rows['reader'] = reader
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with a stuck run or a railed channel / stuck runs plus railed channels."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'stuck runs and railed channels',
                        int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def _mask(rec: Recording, trial, *, stored) -> Recording:
    """Dead, stuck and railed samples marked invalid: from the stored runs ({trial: {segment: runs}}), or, where
    `stored` is None, found again in this trial alone (another reader than the check's)."""
    if stored is not None and str(trial) not in stored:
        return rec
    new, marked = rec.copy(), {}
    for name, seg in new.segments.items():
        if stored is None:
            railed = cut_off_samples(np.hstack([seg.imu.gyr, seg.imu.acc]), seg.imu.rate_hz, minimum_limits(), None,
                                     minimum_limits(RAIL_LIMIT), SENSORS)[1]
            bad = stuck_samples(seg)[0] | railed
        else:
            bad = from_runs(stored[str(trial)].get(name), len(seg.imu))
        bad &= seg.imu.valid
        seg.imu.valid = seg.imu.valid & ~bad
        marked[name] = int(bad.sum())
    new.log_correction('dead channels', f'Marked {sum(marked.values())} IMU samples of dead, stuck or railed '
                                        'channels invalid.', samples_marked=marked)
    return new


def correct(loader, rows: pd.DataFrame):
    """The dead, stuck and railed samples marked invalid (all channels of that IMU sample); RepoIMU unchanged."""
    reader = reader_of(loader)
    if reader in NO_DEAD_STAGE:
        return loader
    if len(rows) and 'reader' in rows and rows.reader.iloc[0] != reader:
        return CorrectedLoader(loader, partial(_mask, stored=None), 'dead channels')
    stored = {}
    for r in rows[rows.detected.map(truthy) | (pd.to_numeric(rows.get('stuck_samples', 0)) > 0)].itertuples():
        stored.setdefault(str(r.trial), {})[r.segment] = loads(r.mask_runs, [])
    if not stored:
        return loader
    return CorrectedLoader(loader, partial(_mask, stored=stored), 'dead channels')


SPEC = ErrorSpec(7, 'dead_channels', 'mask', 'dead, stuck or railed channel')
PARTS = {7: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_dead_channels = make_handler(PARTS[7])
