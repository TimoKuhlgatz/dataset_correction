"""Error 8: wrong trial pairing, an IMU recording paired with the optical recording of another trial.

What it is: each stream is real motion, so joint angles stay plausible, but the two streams do not show the same
movement (Trento: the deposit pairs the k-th IMU trial with the k-th optical file although aborted and repeated
optical takes shift the numbering).

How we catch it (the Trento reader's own pairing repair, `_bonfiglio_session.py`, applied to every dataset): every
candidate (IMU trial, optical recording) is scored by the peak normalised cross-correlation of the
gyroscope's rotation speed with the reference's rotation speed over the full lag range (at least half of the shorter
series overlapping), both low-passed at 5 Hz (zero phase) on a grid at the reference's rate; a candidate's score is
the best segment's peak. The IMU trials are then assigned to optical recordings so that the total gain (score - 0.5)
is largest, each optical recording used at most once and no pairing below 0.5 (the trial stays unpaired). A trial
is wrongly paired if the assignment gives it another optical recording than the declared one (or none).
    Default (the reader's procedure): per participant, the candidates lie within NEIGHBOURS (5; the reader uses 4)
        positions of the trial's own position, and the assignment keeps the chronological order of both lists
        (the reader's monotone assignment, imported unchanged).
    all_trials=True: every optical recording of the dataset is a candidate for every IMU trial, with no assumption
        about names, order or participants; the assignment is then the best one-to-one matching (Hungarian
        method) with the same gain and floor, since there is no order to keep.

How we correct it: pair each trial with its assigned optical recording; unpaired trials have no reference. Only
Trento is corrected: it is read with its repaired pairing (reader 'bonfiglio' instead of 'bonfiglio_deposited').
The other wrongly paired trials (Darmstadt, ULTRA-MoCap) are reported only; Geneva's are a consequence of its
mis-assembled streams (error 10) and vanish on the streams rebuilt from its raw logs, which its pairing is checked on
in the order of correction.
A static recording without a match is not counted: it has no motion to match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..datasets.readers.characterisation.adapters._bonfiglio_session import (MIN_OVERLAP_FRACTION, MIN_PEAK,
                                                                     _monotone_pairing, full_xcorr_lag,
                                                                     interpolate_gaps, zero_phase_lowpass)
from ..recording import Recording
from ..rotations import reference_angular_rate

ERROR = 'trial pairing'
NEIGHBOURS = 5


@dataclass
class Speeds:
    """Rotation speed per segment of one stream type (IMU or reference) of one recording, low-passed at 5 Hz, on
    a uniform grid at rate_hz."""
    rate_hz: float
    by_segment: dict


def _usable(x: np.ndarray | None) -> bool:
    return x is not None and len(x) >= 2 / MIN_OVERLAP_FRACTION


def imu_speeds(recording: Recording, rate_hz: float | None = None) -> Speeds:
    """The gyroscope's rotation speed per segment on a grid at rate_hz (default: the reference's rate), as the
    Trento reader builds it: each axis interpolated onto the grid, then the norm, then the low-pass."""
    rate_hz = rate_hz or next(iter(recording.segments.values())).reference.rate_hz
    out = {}
    for name, seg in recording.segments.items():
        t, ok = seg.imu.times, seg.imu.valid & np.isfinite(seg.imu.gyr).all(1)
        if ok.sum() < 2:
            continue
        grid = np.arange(t[ok][0], t[ok][-1], 1 / rate_hz)
        g = np.stack([np.interp(grid, t[ok], seg.imu.gyr[ok, c]) for c in range(3)], axis=1)
        if _usable(grid):
            out[name] = zero_phase_lowpass(np.linalg.norm(g, axis=1), rate_hz)
    return Speeds(rate_hz, out)


def reference_speeds(recording: Recording) -> Speeds:
    """The reference's rotation speed per segment at its own rate, as the Trento reader builds it: one-step angular rate,
    missing frames filled linearly for the correlation only, then the low-pass."""
    rate_hz = next(iter(recording.segments.values())).reference.rate_hz
    out = {}
    for name, seg in recording.segments.items():
        ref = seg.reference
        _, rates = reference_angular_rate(ref.orientation, ref.rate_hz)
        speed = np.linalg.norm(rates, axis=1)
        speed[~(ref.valid[:-1] & ref.valid[1:])] = np.nan
        speed = interpolate_gaps(speed)
        if _usable(speed):
            out[name] = zero_phase_lowpass(speed, ref.rate_hz)
    return Speeds(rate_hz, out)


def score(imu: Speeds, reference: Speeds) -> tuple[float, dict]:
    """The best segment's peak correlation of this IMU recording against this optical recording (0 if no segment
    can be compared, as in the Trento reader), and the peak of every segment."""
    peaks = {}
    for name in imu.by_segment.keys() & reference.by_segment.keys():
        x, y = reference.by_segment[name], imu.by_segment[name]
        if reference.rate_hz != imu.rate_hz:  # put the reference on the IMU speed's grid
            x = np.interp(np.arange(0, len(x) / reference.rate_hz, 1 / imu.rate_hz),
                          np.arange(len(x)) / reference.rate_hz, x)
        got = full_xcorr_lag(x, y, int(MIN_OVERLAP_FRACTION * min(len(x), len(y))))
        if got is not None:
            peaks[name] = got[1]
    return (max(peaks.values()) if peaks else 0.0), peaks


def window_candidates(n_imu: int, n_optical: int, neighbours: int = NEIGHBOURS) -> list[tuple[int, int]]:
    """(IMU position, optical position) pairs within `neighbours` positions of each other."""
    return [(k, j) for k in range(n_imu) for j in range(n_optical) if abs(j - k) <= neighbours]


def assign_in_order(n_imu: int, n_optical: int, scores: dict) -> tuple[dict, dict]:
    """The Trento reader's monotone assignment over positions (each list in its chronological order)."""
    return _monotone_pairing(list(range(n_imu)), list(range(n_optical)), scores)


def assign_any_order(imu_ids: list, optical_ids: list, scores: dict) -> tuple[dict, dict]:
    """The one-to-one assignment with the largest total gain (score - MIN_PEAK) and no pairing below MIN_PEAK,
    without any order (all_trials). scores: {(imu_id, optical_id): score}; missing pairs score 0."""
    gain = np.zeros((len(imu_ids), len(optical_ids)))
    index_imu = {t: i for i, t in enumerate(imu_ids)}
    index_opt = {t: j for j, t in enumerate(optical_ids)}
    for (t, o), s in scores.items():
        gain[index_imu[t], index_opt[o]] = max(s - MIN_PEAK, 0.0)
    rows, cols = linear_sum_assignment(gain, maximize=True)
    pairing, peak = {t: None for t in imu_ids}, {}
    for i, j in zip(rows, cols):
        if gain[i, j] > 0:
            pairing[imu_ids[i]] = optical_ids[j]
            peak[imu_ids[i]] = scores[(imu_ids[i], optical_ids[j])]
    return pairing, peak


# --- handler: one dataset at a time -------------------------------------------------------------------------------

import re  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import by_participant, parallel_map, reader_of, rebase, truthy  # noqa: E402

STATIC = re.compile('static', re.I)
"""Trial names of static recordings: no motion to match, so not counted when left without a match."""


def _status(assigned, declared) -> str:
    return 'unpaired' if assigned is None else ('kept' if assigned == declared else 're-paired')


def _check_participant(loader, group) -> list[dict]:
    """Speeds of the participant's IMU trials and optical recordings, scores within NEIGHBOURS positions, the
    chronological assignment and each trial's status. Trento: every optical file, the aborted takes included."""
    participant, trials = group
    imu, optical, declared = {}, {}, {}
    trento = hasattr(loader, 'optical_speeds')
    if trento:
        optical = loader.optical_speeds(participant)
    failed = []
    for trial in trials:
        try:
            if trento:
                imu[trial] = loader.trial_imu_speeds(trial)
                declared[trial] = loader.declared_optical(trial)
                continue
            rec = loader.load(trial)
            if not rec.segments:
                continue
            imu[trial], optical[trial], declared[trial] = imu_speeds(rec), reference_speeds(rec), trial
        except Exception as exc:  # noqa: BLE001 - one unreadable trial is reported, the others still paired
            imu.pop(trial, None)
            failed.append(dict(trial=trial, participant=str(participant), segment='', detected=None,
                               error=f'{type(exc).__name__}: {exc}'))
    imu_ids, opt_ids = list(imu), list(optical)
    scores = {(k, j): score(imu[imu_ids[k]], optical[opt_ids[j]])[0]
              for k, j in window_candidates(len(imu_ids), len(opt_ids))}
    pairing, peak = assign_in_order(len(imu_ids), len(opt_ids), scores)
    rows = failed
    for k, t in enumerate(imu_ids):
        got = None if pairing[k] is None else opt_ids[pairing[k]]
        status = _status(got, declared[t])
        j = opt_ids.index(declared[t]) if declared[t] in opt_ids else None
        detected = status == 're-paired' or (status == 'unpaired' and not STATIC.search(str(t)))
        rows.append(dict(trial=t, participant=str(participant), segment='',
                         detected=None if status == 'unpaired' and STATIC.search(str(t)) else detected,
                         estimate=int(detected), unit='trials', status=status, declared=declared[t], assigned=got,
                         peak=peak.get(k, float('nan')),
                         declared_score=scores.get((k, j), float('nan')) if j is not None else float('nan')))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per IMU trial: its status ('kept', 're-paired': another optical recording of the participant matches
    clearly better, 'unpaired': none matches, the declared one included), the declared and the assigned optical
    recording. Detected: re-paired or unpaired; an unpaired static recording is undecided."""
    rows = []
    for out in parallel_map(loader, _check_participant, by_participant(loader), processes):
        if isinstance(out, dict):  # a participant could not be read
            rows += [dict(trial=t, participant=str(out['trial'][0]), segment='', detected=None, error=out['error'])
                     for t in out['trial'][1]]
        else:
            rows += out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials re-paired or unpaired (one occurrence per trial)."""
    return count_trials(rows, 'trials')


def correct(loader, rows: pd.DataFrame):
    """Trento read with its repaired pairing: the same corrections on top of the reader 'bonfiglio'. Every other
    reader is returned unchanged (in the order of correction, correction.base_reader already switches the reader
    from this error's check on)."""
    if reader_of(loader) != 'bonfiglio_deposited':
        return loader
    from ..datasets import load_dataset
    return rebase(loader, load_dataset('bonfiglio'))


def loader_for(dataset: str, data: str):
    """Trento as deposited is read with the deposit's own pairing (the k-th IMU trial with the k-th optical file);
    None for everything else (the usual source loader)."""
    if dataset == 'bonfiglio' and data == 'deposited':
        from ..datasets import load_dataset
        return load_dataset('bonfiglio_deposited')
    return None


SPEC = ErrorSpec(8, 'trial_pairing', 'fix', 'wrong trial pairing')
PARTS = {8: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct,
                 loader_for=loader_for)}
handle_trial_pairing = make_handler(PARTS[8])
