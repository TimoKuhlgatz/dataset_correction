"""The effect of an error and its correction on relative orientation (joint angle) estimation.

Estimator: MEKF-acc (Weygers & Kok 2020) from the `dfjimu` package, `mekf_acc`: both sensors start at the identity
quaternion, the lever arms are estimated from the data passed in, no gyroscope bias is removed, and two parameters
are tuned on a logarithmic grid:
    R: the measurement noise variance (`R_diag`), R_GRID;
    Q: a factor on the process noise; Q_cov = Q * (variance of each gyroscope axis over the first 50 samples of
       the trial), Q_GRID.
The error of one joint is the angle (degrees) between the estimated and the reference relative orientation
R_a^-1 R_b, averaged over the scored samples where both references are valid; the error of a trial is the mean
over its joints.

Procedure: the dataset is split into a tuning half and a test half (tuning_split.py); `tune_on_trials` runs the grid
search once over all tuning trials (one (R, Q) per dataset, trials weighted by their scored length), `test_on_trial`
scores each test trial with it. IMU and reference are first resampled onto one grid at the IMU rate (resample.py).
Every trial is filtered from its start and scored from CONVERGENCE_S (10 s) after its motion onset onwards
(motion_onset.py), in tuning and in testing; the base Q comes from each trial's own first 50 samples. Pass `onset_s`
to score two versions of one trial (deposited and corrected) from the same instant; an explicit `skip_s` (seconds
after the part's start) overrides the onset.
A joint with a segment whose mounting was fitted on that segment's own convergence window (no other trial to pool
with and too little motion in the trial's window; the mounting correction, error 18) is scored only from the end of
that window (segment_window_skip_s); to score another version of the trial (the deposited data) over the same span,
copy metadata[motion_onset.SEGMENT_WINDOWS_KEY] onto it.
Static trials are neither tuned on nor tested: the analysis is about relative motion, not pose at rest. `trial_grid`
and `test_on_trial` raise NotScored for a trial without motion onset, and for one too short from its onset for its
role (`motion_onset.long_enough`: tuning 20 s, test 15 s); a trial split in time is checked for motion only. Static
trials stay available to the error corrections that need them (rest phases, session pooling).

Estimator choice (`method`, every function below): 'MEKF' (default, as above) or 'zoe', the calibration-free EKF
of zoe.py, which estimates the lever arms online. For 'zoe' the grid's R is its measurement noise cov_lnk and its Q
the gyroscope process noise cov_w, both absolute (zoe.COV_LNK_GRID, zoe.COV_W_GRID, around the published values);
the base Q is not used. Tuning, scoring and the error measure are the same for both methods.
"""
from __future__ import annotations

import warnings

import numpy as np
from dfjimu import estimate_lever_arms, mekf_acc
from scipy.spatial.transform import Rotation

from . import zoe
from .recording import ImuStream, Recording, ReferenceStream
from .resample import resample_to_common_grid
from .tuning_split import CONVERGENCE_S
from .motion_onset import SEGMENT_WINDOWS_KEY, long_enough, motion_onset_s

R_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
Q_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
BASE_Q_SAMPLES = 50
Q_INIT = np.array([1.0, 0.0, 0.0, 0.0])
METHODS = ('MEKF', 'zoe')
TIME_TOLERANCE_S = 1e-9  # time stamps closer than this count as equal (rounding of start + k / rate)


def default_grids(method: str = 'MEKF') -> tuple[tuple, tuple]:
    """(R grid, Q grid) of the method; for 'zoe' (cov_lnk grid, cov_w grid)."""
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}, not {method!r}')
    return (R_GRID, Q_GRID) if method == 'MEKF' else (zoe.COV_LNK_GRID, zoe.COV_W_GRID)


def relative_orientation_error_deg(estimate: Rotation, reference: Rotation) -> np.ndarray:
    """Angle in degrees of the rotation between estimated and reference relative orientation, per sample."""
    return np.degrees((reference.inv() * estimate).magnitude())


def trial_start_s(recording: Recording) -> float:
    """The first moment at which every stream of the trial has data."""
    return max(s.start_s for seg in recording.segments.values() for s in (seg.imu, seg.reference))


def trial_end_s(recording: Recording) -> float:
    """The last moment at which every stream of the trial still has data (time of the last sample)."""
    return min(s.times[-1] for seg in recording.segments.values() for s in (seg.imu, seg.reference))


def _cut_imu(imu: ImuStream, keep: np.ndarray) -> ImuStream:
    first = int(np.argmax(keep)) if keep.any() else 0
    return ImuStream(imu.gyr[keep], imu.acc[keep], imu.rate_hz, imu.start_s + first / imu.rate_hz, imu.valid[keep])


def _cut_reference(ref: ReferenceStream, keep: np.ndarray) -> ReferenceStream:
    first = int(np.argmax(keep)) if keep.any() else 0
    markers = {name: xyz[keep] for name, xyz in ref.markers.items()} if ref.markers else ref.markers
    return ReferenceStream(ref.orientation[np.flatnonzero(keep)], ref.rate_hz, ref.start_s + first / ref.rate_hz,
                           ref.valid[keep], ref.convention, markers)


def half_of_trial(recording: Recording, half: str) -> Recording:
    """The 'first half' or 'second half' of a trial between its start and end (trial_start_s, trial_end_s), for a
    dataset split in time. A sample stamped exactly at the middle goes to the second half; time stamps are kept."""
    if half not in ('first half', 'second half'):
        raise ValueError("half must be 'first half' or 'second half'")
    start, end = trial_start_s(recording), trial_end_s(recording)
    middle = start + (end - start) / 2

    def keep(times: np.ndarray) -> np.ndarray:
        if half == 'first half':
            return (times >= start - TIME_TOLERANCE_S) & (times < middle - TIME_TOLERANCE_S)
        return (times >= middle - TIME_TOLERANCE_S) & (times <= end + TIME_TOLERANCE_S)

    new = recording.copy()
    for seg in new.segments.values():
        seg.imu = _cut_imu(seg.imu, keep(seg.imu.times))
        seg.reference = _cut_reference(seg.reference, keep(seg.reference.times))
    return new


def base_q(recording: Recording, joint: tuple[str, str]) -> np.ndarray:
    """Gyroscope variance per axis of both sensors over the first BASE_Q_SAMPLES samples."""
    a, b = (recording[name].imu.gyr[:BASE_Q_SAMPLES] for name in joint)
    return np.concatenate([np.var(a, axis=0), np.var(b, axis=0)])


class JointRun:
    """The inputs of one joint in one recording part, prepared once (lever arms or filtered signals, reference) for
    many runs of the estimator `method` ('MEKF' or 'zoe')."""

    def __init__(self, recording: Recording, joint: tuple[str, str], q_base: np.ndarray, skip_s: float = 0.0,
                 score_valid_imu_only: bool = False, method: str = 'MEKF'):
        default_grids(method)  # checks the name
        a, b = recording[joint[0]], recording[joint[1]]
        self.method = method
        self.rate = a.imu.rate_hz
        self.gyr = (a.imu.gyr, b.imu.gyr)
        self.acc = (a.imu.acc, b.imu.acc)
        self.q_base = q_base
        if method == 'MEKF':
            self.lever_arms = estimate_lever_arms(*self.gyr, *self.acc, self.rate)
        else:
            self.zoe = zoe.ZoeInputs(self.gyr[0], self.acc[0], self.gyr[1], self.acc[1], self.rate)
        self.reference = a.reference.orientation.inv() * b.reference.orientation
        scored = a.reference.valid & b.reference.valid
        if score_valid_imu_only:  # leave samples marked invalid in either IMU (e.g. clipped) out of the error
            scored &= a.imu.valid & b.imu.valid
        scored[:int(round(skip_s * self.rate))] = False
        self.scored = scored

    def errors(self, r: float, q: float) -> np.ndarray:
        """Error per sample in degrees (NaN where not scored) for one parameter pair (for 'zoe': r = cov_lnk,
        q = cov_w)."""
        if self.method == 'MEKF':
            q1, q2 = mekf_acc(*self.gyr, *self.acc, *self.lever_arms, Fs=self.rate, q_init=Q_INIT,
                              Q_cov=self.q_base * q, R_diag=r)
        else:
            try:
                q1, q2 = self.zoe.run(cov_lnk=r, cov_w=q)[:2]
            except np.linalg.LinAlgError:  # the filter diverged (non-finite covariance): no estimate, counts as worst
                return np.full(len(self.scored), np.nan)
        estimate = Rotation.from_quat(q1[:, [1, 2, 3, 0]]).inv() * Rotation.from_quat(q2[:, [1, 2, 3, 0]])
        error = relative_orientation_error_deg(estimate, self.reference)
        return np.where(self.scored, error, np.nan)

    def mean_error(self, r: float, q: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # a diverged run gives NaN, which counts as worst
            value = float(np.nanmean(self.errors(r, q)))
        return value if np.isfinite(value) else np.inf


def segment_window_skip_s(part: Recording, segment: str) -> float:
    """Seconds from the part's start to the end of the segment's own mounting window, where fix_mounting fitted the
    segment on it (motion_onset.SEGMENT_WINDOWS_KEY, in the time base before resampling); 0 for every other
    segment."""
    window = part.metadata.get(SEGMENT_WINDOWS_KEY, {}).get(segment)
    if window is None:
        return 0.0
    grid = [c.parameters['grid_start_s'] for c in part.corrections if c.error == 'resampling']
    end = window[1] - (grid[-1] if grid else 0.0)
    return max(end - trial_start_s(part), 0.0)


def _joint_runs(part: Recording, q_bases: dict, skip_s: float, score_valid_imu_only: bool = False,
                method: str = 'MEKF') -> dict:
    """Each joint scored from `skip_s`, or later where one of its segments was fitted on its own mounting window
    (scored only after that window)."""
    return {joint: JointRun(part, joint, q_bases[joint],
                            max([skip_s] + [segment_window_skip_s(part, s) for s in joint]), score_valid_imu_only,
                            method)
            for joint in part.joints}


def _grids(method: str, r_grid, q_grid) -> tuple[tuple, tuple]:
    default_r, default_q = default_grids(method)
    return (default_r if r_grid is None else r_grid), (default_q if q_grid is None else q_grid)


def _joint_errors(part: Recording, q_bases: dict, r: float, q: float, skip_s: float,
                  score_valid_imu_only: bool = False, method: str = 'MEKF') -> dict:
    """Mean error per joint on one part, scored from `skip_s` after its start."""
    return {joint: run.mean_error(r, q)
            for joint, run in _joint_runs(part, q_bases, skip_s, score_valid_imu_only, method).items()}


def scoring_skip_s(part: Recording, onset_s: float | None = None) -> float:
    """Seconds from the part's start until scoring starts: CONVERGENCE_S after the motion onset (computed from the
    part's reference when `onset_s` is None; an onset before the part's start, e.g. in the second half of a trial
    split in time, counts from the part's start). A static part is not scored at all."""
    start = trial_start_s(part)
    onset = motion_onset_s(part) if onset_s is None else onset_s
    if onset is None:
        return trial_end_s(part) - start + 1.0
    return max(onset, start) - start + CONVERGENCE_S


def _prepared(recording: Recording, half: str | None = None) -> tuple[Recording, dict]:
    """One trial on the common grid (or the first or second half of it, for a dataset split in time) with the base
    Q of each joint from its first samples."""
    part = resample_to_common_grid(recording)
    if half is not None:
        part = half_of_trial(part, half)
    return part, {joint: base_q(part, joint) for joint in part.joints}


def tune_on_trials(recordings: list, r_grid=None, q_grid=None, skip_s: float | None = None,
                   score_valid_imu_only: bool = False, half: str | None = None,
                   method: str = 'MEKF') -> tuple[float, float, dict]:
    """Grid search over the tuning half of a dataset (tuning_split.py): the (R, Q) with the smallest error, where a
    trial's error is the mean over its joints and the tuning error the mean over the trials weighted by each trial's
    assessed length (scored seconds, mean over its joints), so long trials count more; each trial
    scored from CONVERGENCE_S after its motion onset (or from `skip_s` after its start). Returns R, Q and the full
    grid {(R, Q): tuning error}."""
    grids = []
    for rec in recordings:  # each trial from its own motion onset; static or too short trials take no part
        try:
            grids.append(trial_grid(rec, r_grid, q_grid, skip_s, score_valid_imu_only, half, method))
        except NotScored:
            continue
    return combine_trial_grids(grids)


class NotScored(ValueError):
    """A trial that takes no part in tuning or testing (module description: static or too short from its onset)."""


def _check_scored(recording: Recording, role: str, half: str | None, onset_s: float | None) -> None:
    onset = motion_onset_s(recording) if onset_s is None else onset_s
    if onset is None:
        raise NotScored('static: the reference never exceeds the standing limit; not tuned on or tested')
    if half is None:
        ok, why = long_enough(recording, role, onset)
        if not ok:
            raise NotScored(why)


def trial_grid(recording: Recording, r_grid=None, q_grid=None, skip_s: float | None = None,
               score_valid_imu_only: bool = False, half: str | None = None,
               method: str = 'MEKF', onset_s: float | None = None) -> tuple[float, dict]:
    """One tuning trial's share of `tune_on_trials`, so trials can be run in parallel: its weight (scored seconds,
    mean over its joints) and its error (mean over its joints) for every (R, Q) of the grid."""
    if skip_s is None:
        _check_scored(recording, 'tuning', half, onset_s)
    r_grid, q_grid = _grids(method, r_grid, q_grid)
    part, q_bases = _prepared(recording, half)
    skip_s = scoring_skip_s(part, onset_s) if skip_s is None else skip_s
    runs = _joint_runs(part, q_bases, skip_s, score_valid_imu_only, method)
    weight = float(np.mean([run.scored.sum() / run.rate for run in runs.values()]))
    if weight <= 0:  # a trial with nothing to score does not take part
        return 0.0, {}
    return weight, {(r, q): float(np.mean([run.mean_error(r, q) for run in runs.values()]))
                    for r in r_grid for q in q_grid}


def combine_trial_grids(trial_grids: list) -> tuple[float, float, dict]:
    """The tuning error per (R, Q) over trials, weighted by their scored length, and the best pair."""
    kept = [(w, g) for w, g in trial_grids if w > 0]
    grid = {rq: float(np.average([g[rq] for _, g in kept], weights=[w for w, _ in kept])) for rq in kept[0][1]}
    best = min(grid, key=grid.get)
    return best[0], best[1], grid


def test_on_trial(recording: Recording, r: float, q: float, skip_s: float | None = None,
                  score_valid_imu_only: bool = False, half: str | None = None, method: str = 'MEKF',
                  onset_s: float | None = None) -> dict:
    """Error of one test trial with the tuned (R, Q), the filter started afresh and scored from CONVERGENCE_S after
    its motion onset (or from `skip_s` after its start). Raises NotScored for a static or too short trial."""
    if skip_s is None:
        _check_scored(recording, 'test', half, onset_s)
    part, q_bases = _prepared(recording, half)
    skip_s = scoring_skip_s(part, onset_s) if skip_s is None else skip_s
    per_joint = _joint_errors(part, q_bases, r, q, skip_s, score_valid_imu_only, method)
    return dict(test_error_deg=float(np.mean(list(per_joint.values()))),
                per_joint={f'{a}-{b}': e for (a, b), e in per_joint.items()})
