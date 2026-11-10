"""Cross-check of error 20 (negated accelerometer) that fits no sensor rotation (a fit-free reading).

Idea: the gyroscope and the accelerometer sit in one housing, so the product of their readings, h = ω · a, is the
same in every frame the housing could be turned to: it does not depend on the mounting. A negated accelerometer flips
its sign. The optical reference predicts the same product: with R the segment-to-room rotation, ω_w the segment's
rotation rate in the room, p a point fixed to the segment, r the sensor's unknown position relative to p (segment
frame) and u the room's up direction,

    a (room) = p̈ + R̈ r + g u,   so   ω · a = ω_w · (p̈ + g u) + (ω_w R̈) r.

The unknown r enters linearly, so the three columns of ω_w R̈ (and a constant) are projected out of both sides per
trial (Frisch-Waugh). The slope of the measured on the predicted product is then about +1 for a correct
accelerometer and about -1 for a negated one. Pooled per sensor-session over its trials; a bootstrap over 2 s blocks
gives a 99 % interval: negated if it lies below zero, correct if above, else undecided.

Where p comes from:
- markers: the centroid of the segment's markers (any point fixed to the segment will do, the lever arm is
  projected out);
- a chain hanging from a fixed point (ULTRA-MoCap's arm from the model's shoulder, RepoIMU's pendulum from its
  pivot): p̈ of a segment's joint is the sum of R̈_k l_k over the segments k above it, with unknown lengths l_k, so
  the columns ω_w R̈_k are projected out as well. Assumption: the fixed point does not accelerate (ULTRA-MoCap: the
  trunk is still; its reference is the model thorax, so a moving trunk breaks this).
The up direction u is the trial's vertical from check_for_negated_accelerometer (its sensors' majority). Data in the
agreed order are already synchronised, so no lag is searched. The
orientation and positions are low-passed at 10 Hz before they are differentiated, and a trial whose gyroscope and
reference rotation speeds correlate below 0.5 is left out.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from ..recording import Recording
from ..rotations import interpolate
from .e16_mirrored_gyroscope import _segment_to_world

GRAVITY_MPS2 = 9.81
CUTOFF_HZ = 10.0  # low-pass before differentiating
EDGE_S = 0.6  # trimmed at both ends of a run (filter transients)
MIN_RUN_S = 5.0  # shortest stretch of valid frames used
MIN_SPEED_CORR = 0.5  # correlation of gyroscope and reference rotation speed below this: trial left out
BLOCK_S = 2.0  # bootstrap block
BOOTSTRAP = 400
INTERVAL = (0.5, 99.5)  # percentiles of the bootstrap interval


def _lowpass(x: np.ndarray, rate: float) -> np.ndarray:
    b, a = butter(4, min(CUTOFF_HZ / (rate / 2), 0.99), 'low')
    return filtfilt(b, a, x, axis=0)


def _longest_run(ok: np.ndarray) -> slice | None:
    idx = np.flatnonzero(ok)
    if not len(idx):
        return None
    run = max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1), key=len)
    return slice(run[0], run[-1] + 1)


def trial_terms(recording: Recording, segment: str, up: np.ndarray, convention: str,
                chain: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Predicted and measured ω·a of one trial and segment, with the lever arm (and the chain lengths) projected out,
    and the speed correlation. `chain`: the segments above this one in a chain hanging from a fixed point (empty for
    the root); None: the segment's markers give p. None if the trial cannot be read."""
    seg = recording[segment]
    ref, imu = seg.reference, seg.imu
    rate = ref.rate_hz
    t = ref.start_s + np.arange(len(ref)) / rate
    ok = ref.valid.copy()
    if chain is None:
        if not ref.markers:
            return None
        pts = np.stack([np.asarray(v, float) for v in ref.markers.values()], axis=1) / 1000.0  # (M, k, 3) in m
        ok &= np.isfinite(pts).all(axis=(1, 2))
    else:
        others = [recording[c].reference for c in chain]
        if any(len(o) != len(ref) or o.rate_hz != rate for o in others):
            return None
        for o in others:
            ok &= o.valid
    g = interpolate(t, imu.times, np.where(imu.valid[:, None], imu.gyr, np.nan))
    a = interpolate(t, imu.times, np.where(imu.valid[:, None], imu.acc, np.nan))
    ok &= np.isfinite(g).all(1) & np.isfinite(a).all(1)
    run = _longest_run(ok)
    if run is None or (run.stop - run.start) < MIN_RUN_S * rate:
        return None

    def rot(r):  # segment-to-room matrices of the run, low-passed elementwise
        return _lowpass(_segment_to_world(r.orientation[run], convention).as_matrix().reshape(-1, 9), rate).reshape(-1, 3, 3)

    dt = 1.0 / rate
    R = rot(ref)
    dR = np.gradient(R, dt, axis=0)
    ddR = np.gradient(dR, dt, axis=0)
    W = np.einsum('nij,nkj->nik', dR, R)  # Ṙ Rᵀ: the room-frame rate as a skew matrix
    w = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
    columns = [np.einsum('ni,nij->nj', w, ddR)]  # ω_w R̈: the lever arm
    if chain is None:
        p = _lowpass(pts[run].mean(axis=1), rate)
        pdd = np.gradient(np.gradient(p, dt, axis=0), dt, axis=0)
    else:
        pdd = np.zeros_like(w)  # the fixed point does not accelerate; the joints above enter through their lengths
        for c in chain:
            Rc = rot(recording[c].reference)
            columns.append(np.einsum('ni,nij->nj', w, np.gradient(np.gradient(Rc, dt, axis=0), dt, axis=0)))
    gl, al = _lowpass(g[run], rate), _lowpass(a[run], rate)
    k = int(EDGE_S * rate)
    if len(w) <= 2 * k + 10:
        return None
    sl = slice(k, len(w) - k)
    speed_corr = float(np.corrcoef(np.linalg.norm(w[sl], axis=1), np.linalg.norm(gl[sl], axis=1))[0, 1])
    predicted = np.einsum('ni,ni->n', w, pdd + GRAVITY_MPS2 * up)[sl]
    measured = np.einsum('ni,ni->n', gl, al)[sl]
    P = np.column_stack([c[sl] for c in columns] + [np.ones(sl.stop - sl.start)])

    def resid(v):
        c, *_ = np.linalg.lstsq(P, v, rcond=None)
        return v - P @ c

    return resid(predicted), resid(measured), speed_corr


def pooled_slope(parts: list[tuple[np.ndarray, np.ndarray]], rate_hz: float, seed: int = 7) -> dict:
    """Slope of measured on predicted over the pooled trials, with a block-bootstrap interval and the verdict
    (True negated, False correct, None undecided)."""
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    n = max(1, int(BLOCK_S * rate_hz))
    nb = len(a) // n
    if nb < 5:
        return dict(slope=float('nan'), low=float('nan'), high=float('nan'), negated=None, samples=len(a))
    ab, bb = a[:nb * n].reshape(nb, n), b[:nb * n].reshape(nb, n)
    num, den = (ab * bb).sum(1), (ab * ab).sum(1)
    rng = np.random.default_rng(seed)
    boot = [num[k].sum() / den[k].sum() for k in rng.integers(0, nb, (BOOTSTRAP, nb))]
    low, high = (float(x) for x in np.percentile(boot, INTERVAL))
    verdict = True if high < 0 else False if low > 0 else None
    return dict(slope=float(num.sum() / den.sum()), low=low, high=high, negated=verdict, samples=len(a))


# Where the point on the segment comes from, per dataset (recording.dataset): marker positions, or a chain hanging from
# a fixed point (each segment: the segments above it). Other datasets cannot be cross-checked.
MARKER_DATASETS = {'camargo', 'fullbody17', 'bonfiglio', 'bonfiglio_deposited', 'wmcg', 'compliant_terrain', 'is2b',
                   'grouvel', 'grouvel_raw', 'grimmer_stairs', 'diodem', 'synthetic'}
CHAINS = {'ultramocap': {'upper_arm': [], 'forearm': ['upper_arm'], 'hand': ['upper_arm', 'forearm']},
          'repoimu': {'segment_1': [], 'segment_2': ['segment_1'], 'segment_3': ['segment_1', 'segment_2']}}


def point_source(recording: Recording, segment: str):
    """None for markers, a list of segments above for a chain, False where the cross-check cannot read."""
    if recording.dataset in CHAINS:
        chain = CHAINS[recording.dataset].get(segment)
        return False if chain is None else chain
    return None if recording.dataset in MARKER_DATASETS else False


def session_slope(recordings: list[Recording], segment: str, ups: dict, convention: str,
                  gyr_map: np.ndarray | None = None) -> dict:
    """Pooled slope over the trials of one sensor-session (pooled_slope), `ups` {trial: the trial's up vector}.
    `gyr_map` (3x3) is applied to the gyroscope first (a candidate axis flip). negated None and slope NaN where
    no trial can be read."""
    parts, rate = [], None
    for rec in recordings:
        if segment not in rec.segments or rec.trial not in ups:
            continue
        chain = point_source(rec, segment)
        if chain is False:
            continue
        if gyr_map is not None:
            rec = rec.copy()
            rec.segments[segment].imu.gyr = rec.segments[segment].imu.gyr @ gyr_map.T
        got = trial_terms(rec, segment, np.asarray(ups[rec.trial], float), convention, chain)
        if got is None or not got[2] >= MIN_SPEED_CORR:
            continue
        parts.append(got[:2])
        rate = rec[segment].reference.rate_hz
    if not parts:
        return dict(slope=float('nan'), low=float('nan'), high=float('nan'), negated=None, samples=0, trials=0)
    return dict(pooled_slope(parts, rate), trials=len(parts))
