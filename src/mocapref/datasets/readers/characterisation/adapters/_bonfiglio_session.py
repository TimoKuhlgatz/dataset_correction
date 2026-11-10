"""Per-session corrections of the Trento upper-limb deposit (bonfiglio.py), computed once per subject and session.

The deposit ships a three-marker cluster taped on every sensor housing, but no sensor-to-cluster calibration, no
hardware synchronisation between the optical and the inertial system, and a trial numbering that does not line up
between the two. Everything here is a cross-correlation of two measured signals, a rigid registration of a marker
cluster to a template, or one constant rotation fitted by least squares (Wahba/Kabsch); no orientation filter.

1. Trial pairing. Six of the sixteen subjects have 21-23 optical files for 20 inertial trials (aborted and
   repeated takes), one has 18. Pairing by index puts 71 of the 320 trials with the wrong optical recording. Every
   candidate pair within PAIRING_SEARCH_WINDOW files is scored by the peak correlation of the angular-rate
   magnitudes, and the monotone assignment with the largest total is taken (a session is chronological); a trial
   whose best candidate stays below MIN_PEAK is left unpaired.
2. Clock offset. The angular-rate magnitudes of reference and gyroscope of each segment are put on uniform grids,
   low-passed (zero phase) and cross-correlated over the full lag range. The trial's offset is the median of the
   sensors whose peak reaches MIN_PEAK; it counts only when at least MIN_SENSORS_AGREEING of them agree within
   MAX_DISAGREEMENT_SAMPLES. Sensors further out are listed in ``sensors_outlying_ms``.
3. Cluster frame and cluster-to-sensor rotation. Every trial of a session is Procrustes-fitted to one template
   (the marker triangle of one real frame of the session), not to its own mean shape: a trial's mean shape is a
   shrunken triangle oriented along the trial's average orientation, which put a trial-dependent rotation into the
   frame. The cluster-to-sensor rotation is one constant per sensor and session, fitted over all trials at once,
   from angular rate (omega_cluster = R omega_gyroscope above ALIGN_MIN_OMEGA_RAD_S) or, for a sensor that never
   turns fast enough (trunk and scapula in this seated protocol), from angular-rate directions and the gravity
   direction over still frames, with the laboratory vertical measured per session. The fit is accepted only if the
   per-trial fits agree within MAX_BETWEEN_TRIAL_SCATTER_DEG; otherwise the sensor is marked not aligned. The
   identifiability measure (MIN_DIRECTION_SPREAD_DEG) comes from measures/alignment.py. All fit diagnostics are
   returned for ``Pose.meta`` and ``Recording.meta``.

Not corrected here: marker occlusions (NaN), the 2-16 sample start skew between sensors, subject 9's sensors not
sharing one SampleTimeFine timeline, subject 2's left-arm wear under right-side labels, and the lever arm between
cluster centroid and sensor (``Pose.p`` stays the cluster centroid).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..measures.alignment import MIN_DIRECTION_SPREAD_DEG, direction_spread

# --------------------------------------------------------------------------- #
# constants (every one declares its provenance)                                #
# --------------------------------------------------------------------------- #
LOWPASS_HZ = 5.0
"""Cut-off of the zero-phase low-pass applied to both angular-rate streams before the cross-correlation and
before the Wahba fit. Zero-phase, so it introduces no lag of its own; it is a measurement-side filter and never
touches the samples the adapter emits.

Units: Hz.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

LOWPASS_ORDER = 4
"""Order of that Butterworth low-pass, applied forwards and backwards.

Units: dimensionless.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

MIN_OVERLAP_FRACTION = 0.5
"""A lag is only considered while the two series still overlap by at least this fraction of the shorter one.

The robot-arm adapter fixes the same guard at 1,500 samples (15 s), which no trial of this deposit could
survive: its trials are 4.6–33 s and its lags reach several seconds, so the guard has to scale with the
recording rather than be an absolute count.

Units: fraction of the shorter series.
Provenance: data-free — data-free."""

MIN_PEAK = 0.5
"""The normalised peak correlation a sensor's angular-rate magnitude must reach against its cluster's for that
sensor to carry a lag, and for an (inertial trial, optical file) pair to be a pairing at all.

Units: dimensionless (a normalised correlation).
Provenance: tuned-on-bonfiglio-arm — chosen by looking at this deposit. The value sits in a gap rather than on a slope: the true pairing of a trial reads 0.88–1.00 and
a wrong one 0.26–0.83, and the recovered per-trial lag is unchanged at 0.4, 0.5, 0.6 and 0.7 for 174 of the
trials that carry four or more usable sensors. Two independent facts say what it buys: two sensors that pass it
and agree reproduce the median of four or more to 0.6 ms (95th percentile 5.1 ms), and two sensors that fail it
agree by chance in 5.6 % of cases."""

MIN_SENSORS_AGREEING = 2
"""How many of a trial's five sensors must carry a lag before the trial has one.

Units: sensors.
Provenance: data-free — one sensor is one measurement with no check on it, and this deposit's exercises leave
two or three sensors of any trial nearly still, so requiring more than two would discard the exercises that
move one segment. What makes two enough is agreement, not number: see :data:`MAX_DISAGREEMENT_SAMPLES`."""

MAX_DISAGREEMENT_SAMPLES = 1.0
"""The median absolute deviation the agreeing sensors' lags may have from their median, in samples of the
deposit's own rate, before the trial is declared unsynchronisable.

Units: samples.
Provenance: data-free — the five sensors watch one arm through one optical system on one clock, so their lags
are one quantity and must agree to the resolution at which it is measurable, which is the sample. Measured on
this deposit: 87 % of the trials with three or more usable sensors agree to within one sample and 93 % to
within two, and the rest are not near the boundary but hundreds of milliseconds to seconds away."""

PAIRING_SEARCH_WINDOW = 4
"""How far from the deposit's own index an optical file is looked for.

Units: files.
Provenance: read-from-dataset — the largest surplus of optical files over inertial trials in the deposit is
three (23 against 20, subjects 2 and 9) and the largest deficit is two (18 against 20, subject 4), so a window
of four covers every offset the deposit can produce."""

ALIGN_MIN_OMEGA_RAD_S = 1.5
"""Only frames whose band-limited gyroscope magnitude exceeds this enter the angular-rate route: the
cluster-to-sensor rotation is only observable where the segment actually turns.

Units: rad/s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_FALLBACK_OMEGA_RAD_S = 0.5
"""The lower gate the second route uses for its angular-rate pairs, where gravity carries the other two degrees
of freedom of the rotation.

Units: rad/s.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_MIN_FRAMES = 500
"""Below this many frames past the excitation gate, the angular-rate route is not run at all and the sensor
goes to the second route.

Units: frames.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

ALIGN_MIN_FRAMES_FALLBACK = 50
"""Below this many direction pairs the second route is not run either and the sensor is published as not
aligned.

Units: frames.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

RESIDUAL_PERCENTILE = 95.0
"""The upper percentile of the axis residual reported beside its median. Both are diagnostics; neither gates.

Units: percent.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

MAX_BETWEEN_TRIAL_SCATTER_DEG = MIN_DIRECTION_SPREAD_DEG
"""How far the fits of the individual trials of one session may scatter around the pooled fit before the
sensor is published as **not aligned**.

This is the *direct* test of the constant this repair claims to have measured — the tape does not move, so
fitting each trial separately must give the same rotation — where the conditioning of the direction cloud
(:data:`MIN_DIRECTION_SPREAD_DEG`) is only a necessary condition on the geometry. Measured on this deposit, the
two disagree in both directions: the scapula's directions spread by 9.8–21.2° (conditioned) while its per-trial
fits scatter by 13.4–42.4° (not one constant), and the trunk of subjects 8 and 15 spreads by 5.28°/4.45° (below
the conditioning gate) while its per-trial fits scatter by 2.56°/5.38° and its residual is 0.53°/0.60° (one
constant, well measured). The scatter decides; the conditioning number is published beside it.

Units: degrees.
Provenance: inherited-protocol — the same ``MIN_DIRECTION_SPREAD_DEG`` the characterisation's alignment measure
uses, in its second natural role: an angle below which the measure declines to call a direction cloud a
direction is also an angle above which repeated fits of one rotation are not one rotation. No new number is
introduced and none is read off this deposit's scores."""

STILL_OMEGA_RAD_S = 0.1
"""A frame counts as still, and its specific force as the direction of gravity, when the gyroscope magnitude is
below this.

Units: rad/s.
Provenance: data-free — a body turning this slowly rotates by 0.048° over one sample of the deposit's 120 Hz,
which is below the reference's own orientation noise (0.089–0.189° rms), so a still frame's gravity direction
cannot be spoiled by the rotation the frame still contains."""

STILL_ACC_TOLERANCE_M_S2 = 0.2
"""How far a still frame's specific-force magnitude may sit from standard gravity.

Units: m/s².
Provenance: data-free — a residual specific force of this size can tilt the measured direction by at most
``arcsin(0.2 / 9.80665) = 1.17°``, a fifth of the smallest direction spread
(``MIN_DIRECTION_SPREAD_DEG = 5.74°``) at which a rotation is declared identifiable at all."""

STANDARD_GRAVITY = 9.80665
"""Units: m/s².
Provenance: physical — the defined value of ISO 80000-3 / CODATA."""

NORM_FLOOR = 1e-12
"""Floor on a vector norm before a division, in the cross-correlation and in the axis-residual angle.

Units: dimensionless (the divided quantities are normalised correlations and unit vectors).
Provenance: data-free — a numerical guard twelve orders of magnitude below every quantity these expressions
carry; it changes no value and exists so that an all-zero window returns zero instead of a NaN."""

MILLISECONDS_PER_S = 1000.0
"""Units: ms per s.
Provenance: physical — an exact unit conversion."""


# --------------------------------------------------------------------------- #
# primitives                                                                   #
# --------------------------------------------------------------------------- #
def zero_phase_lowpass(x: np.ndarray, rate_hz: float) -> np.ndarray:
    """A fourth-order Butterworth applied forwards and backwards (no phase, therefore no lag of its own)."""
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, float)
    sos = butter(LOWPASS_ORDER, LOWPASS_HZ / (rate_hz / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def log_map(R: np.ndarray) -> np.ndarray:
    """(T, 3, 3) rotations → (T, 3) rotation vectors."""
    R = np.asarray(R, float)
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    out = axis / 2
    sin = np.sin(angle)
    big = sin > NORM_FLOOR
    out[big] = axis[big] * (angle[big] / (2 * sin[big]))[:, None]
    return out


def angular_rate(t: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The reference's angular rate in the *moving* frame, from the deposited rotations alone.

    ``ω(t + ½Δt) = log(R(t)ᵀ R(t+Δt)) / Δt`` — a one-step difference of the reference labelled at the
    mid-sample, so it is unbiased in time; not a filter and not an estimator; the same expression as
    ``measures/_common.pose_angular_rate``, with the frames whose rotation is missing (an occluded marker) left
    ``NaN``. Returns the mid-sample times and ``ω`` in rad/s.
    """
    t = np.asarray(t, float)
    R = np.asarray(R, float)
    ok = np.isfinite(R[:-1]).all((1, 2)) & np.isfinite(R[1:]).all((1, 2))
    w = np.full((len(t) - 1, 3), np.nan)
    if ok.any():
        dR = np.einsum("tji,tjk->tik", R[:-1][ok], R[1:][ok])
        w[ok] = log_map(dR) / np.diff(t)[ok][:, None]
    return 0.5 * (t[:-1] + t[1:]), w


def interpolate_gaps(x: np.ndarray) -> np.ndarray | None:
    """A one-dimensional series with its missing samples filled by linear interpolation, for the filter and the
    correlation only — the filled samples are never emitted and never enter a fit."""
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    index = np.arange(len(x))
    x[bad] = np.interp(index[bad], index[~bad], x[~bad])
    return x


def full_xcorr_lag(x: np.ndarray, y: np.ndarray, min_overlap: int) -> tuple[float, float] | None:
    """Lag of ``y`` against ``x`` over the **full** lag range, and the normalised correlation there.

    Transcribed from ``openaxes._full_xcorr_lag`` and extended by the parabolic sub-sample refinement the
    earlier check of the deposit used, because this deposit's offset has to be recovered to better than the
    sample: the correlation is divided by the overlap length so that lags near the extremes are not favoured,
    and lags whose overlap is shorter than ``min_overlap`` are excluded outright. The returned lag ``L`` is the
    one for which ``x[i] ≈ y[i − L]``.
    """
    from scipy.signal import correlate

    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    c = correlate(x, y, mode="full", method="fft")
    n = np.arange(-len(y) + 1, len(x))
    overlap = np.minimum(np.minimum(len(x), len(y)), np.minimum(len(x) - n, len(y) + n)).clip(1)
    usable = overlap >= min_overlap
    if not usable.any():
        return None
    score = (c / overlap)[usable]
    lags = n[usable]
    i = int(np.argmax(score))
    lag = float(lags[i])
    if 0 < i < len(score) - 1:  # parabola through the peak and its two neighbours
        left, top, right = score[i - 1], score[i], score[i + 1]
        curvature = left - 2 * top + right
        if curvature != 0:
            lag += 0.5 * (left - right) / curvature
    whole = int(lags[i])
    a, b = (x[whole:], y[: len(x) - whole]) if whole >= 0 else (x[: len(x) + whole], y[-whole:])
    m = min(len(a), len(b))
    if m < 2:
        return None
    a, b = a[:m] - a[:m].mean(), b[:m] - b[:m].mean()
    peak = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + NORM_FLOOR))
    return lag, peak


def wahba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The constant rotation ``R`` minimising ``Σ‖a − R b‖²`` (Kabsch/Markley singular value decomposition).

    A constant fit over two measured vector streams, not an orientation estimator.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    u, _, vt = np.linalg.svd(a[ok].T @ b[ok])
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def rotation_angle_deg(R: np.ndarray) -> float:
    """The angle of a rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def axis_angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between two stacks of vectors, in degrees."""
    un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), NORM_FLOOR)
    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), NORM_FLOOR)
    return np.rad2deg(np.arccos(np.clip(np.sum(un * vn, axis=1), -1.0, 1.0)))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), NORM_FLOOR)


def cluster_pose(markers: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes fit of a three-marker cluster to **one session template** → (R frame→world, p centroid).

    ``markers`` is (T, K, 3) in metres with ``NaN`` where a marker is occluded; a frame missing any marker of
    the cluster is ``NaN`` in both outputs. ``template`` is the (K, 3) centred marker configuration of one real
    frame of the session, so every trial of the session is expressed in **one** gauge — see the module
    docstring for what the trial-mean gauge this replaces did to the fit.
    """
    n = markers.shape[0]
    r = np.full((n, 3, 3), np.nan)
    p = np.full((n, 3), np.nan)
    ok = ~np.isnan(markers).any(axis=(1, 2))
    if not ok.any():
        return r, p
    x = markers[ok]
    centroids = x.mean(axis=1)
    y = x - centroids[:, None, :]
    h = np.einsum("ki,tkj->tij", template, y)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1))))
    flip = np.tile(np.eye(3), (len(d), 1, 1))
    flip[:, 2, 2] = d
    r[ok] = np.einsum("tij,tjk,tkl->til", vt.transpose(0, 2, 1), flip, u.transpose(0, 2, 1))
    p[ok] = centroids
    return r, p


# --------------------------------------------------------------------------- #
# the session                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    """Everything the adapter needs that is a property of the whole session rather than of one trial."""

    subject: str
    rate_hz: float
    templates: dict[str, np.ndarray] = field(default_factory=dict)
    #: inertial trial index → optical file index, or ``None`` where no optical file matched
    pairing: dict[int, int | None] = field(default_factory=dict)
    pairing_peak: dict[int, float] = field(default_factory=dict)
    pairing_moved: dict[int, int] = field(default_factory=dict)
    #: inertial trial index → the seconds to add to the optical time base to put it on the sensors' clock
    lag_s: dict[int, float | None] = field(default_factory=dict)
    lag_diagnostics: dict[int, dict] = field(default_factory=dict)
    #: segment → the constant rotation mapping a vector in the sensor frame to the cluster frame
    cluster_from_sensor: dict[str, np.ndarray] = field(default_factory=dict)
    alignment: dict[str, dict] = field(default_factory=dict)
    vertical: dict = field(default_factory=dict)
    #: the constant rotation that takes the deposit's own world frame to the adapters' z-up convention
    world_from_deposit: np.ndarray = field(default_factory=lambda: np.eye(3))


def build(reader, subject: str, rate_hz: float) -> Session:
    """Compute the whole repair for one subject.

    ``reader`` supplies the deposit's raw arrays and hides the file formats from this module; see
    ``bonfiglio.BonfiglioAdapter._reader`` for the three methods it must have.
    """
    session = Session(subject=subject, rate_hz=rate_hz)
    trials = reader.trial_indices(subject)
    optical = reader.optical_indices(subject)

    session.templates = _templates(reader, subject, optical)
    scores, lags, peaks = _candidate_scores(reader, subject, trials, optical, session.templates, rate_hz)
    session.pairing, session.pairing_peak = _monotone_pairing(trials, optical, scores)
    session.pairing_moved = {k: j for k, j in session.pairing.items() if j != k}
    session.lag_s, session.lag_diagnostics = _trial_lags(session.pairing, lags, peaks, scores, rate_hz)
    _fit_rotations(reader, subject, session, rate_hz)
    return session


def _templates(reader, subject: str, optical: list[int]) -> dict[str, np.ndarray]:
    """One marker template per cluster: the centred triangle of the first frame of the session in which the
    cluster is fully visible. Its orientation is arbitrary and constant, which is the whole point."""
    out: dict[str, np.ndarray] = {}
    for j in optical:
        markers = reader.markers(subject, j)
        for segment, cluster in markers.items():
            if segment in out or cluster is None:
                continue
            ok = ~np.isnan(cluster).any(axis=(1, 2))
            if ok.any():
                first = cluster[int(np.argmax(ok))]
                out[segment] = first - first.mean(axis=0)
        if len(out) == len(markers):
            break
    return out


def _candidate_scores(reader, subject, trials, optical, templates, rate_hz):
    """Peak correlation and lag for every candidate (inertial trial, optical file, segment)."""
    scores: dict[tuple[int, int], float] = {}
    lags: dict[tuple[int, int], dict[str, float]] = {}
    peaks: dict[tuple[int, int], dict[str, float]] = {}
    reference: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for k in trials:
        streams = reader.streams(subject, k)
        gyro_magnitude = {}
        for segment, stream in streams.items():
            grid = np.arange(stream["t"][0], stream["t"][-1], 1 / rate_hz)
            if len(grid) < 2 / MIN_OVERLAP_FRACTION:
                continue
            g = np.stack([np.interp(grid, stream["t"], stream["gyr"][:, c]) for c in range(3)], axis=1)
            gyro_magnitude[segment] = (grid, zero_phase_lowpass(np.linalg.norm(g, axis=1), rate_hz))
        for j in optical:
            if abs(j - k) > PAIRING_SEARCH_WINDOW:
                continue
            per_segment_lag: dict[str, float] = {}
            per_segment_peak: dict[str, float] = {}
            markers = reader.markers(subject, j)
            for segment, (grid, gm) in gyro_magnitude.items():
                if (j, segment) not in reference:
                    cluster = markers.get(segment)
                    if cluster is None or segment not in templates:
                        reference[(j, segment)] = (np.zeros(0), np.zeros(0))
                    else:
                        R, _ = cluster_pose(cluster, templates[segment])
                        t = np.arange(len(R)) / rate_hz
                        tm, w = angular_rate(t, R)
                        magnitude = interpolate_gaps(np.linalg.norm(w, axis=1))
                        reference[(j, segment)] = (
                            (tm, zero_phase_lowpass(magnitude, rate_hz))
                            if magnitude is not None and len(magnitude) >= 2 / MIN_OVERLAP_FRACTION
                            else (np.zeros(0), np.zeros(0))
                        )
                tm, wm = reference[(j, segment)]
                if len(wm) == 0:
                    continue
                got = full_xcorr_lag(wm, gm, int(MIN_OVERLAP_FRACTION * min(len(wm), len(gm))))
                if got is None:
                    continue
                lag, peak = got
                per_segment_lag[segment] = float(grid[0] - tm[0] - lag / rate_hz)
                per_segment_peak[segment] = peak
            scores[(k, j)] = max(per_segment_peak.values()) if per_segment_peak else 0.0
            lags[(k, j)] = per_segment_lag
            peaks[(k, j)] = per_segment_peak
        for key in list(reference):
            if key[0] < k - PAIRING_SEARCH_WINDOW:
                del reference[key]
    return scores, lags, peaks


def _monotone_pairing(trials, optical, scores):
    """The monotone assignment of inertial trials to optical files maximising the total peak correlation.

    Both recordings run through the session in order, so an assignment must be strictly increasing; a trial may
    be left unassigned and an optical file left unused (the aborted and repeated takes). An assignment is only
    worth making when its peak exceeds :data:`MIN_PEAK`, which is what the gain ``score − MIN_PEAK`` says.
    """
    n, m = len(trials), len(optical)
    best = np.full((n + 1, m + 1), -np.inf)
    back: dict[tuple[int, int], tuple[int, int, int | None]] = {}
    best[0, 0] = 0.0
    for a in range(n + 1):
        for b in range(m + 1):
            if not np.isfinite(best[a, b]):
                continue
            if a < n and best[a, b] > best[a + 1, b]:  # this inertial trial gets no optical file
                best[a + 1, b] = best[a, b]
                back[(a + 1, b)] = (a, b, None)
            if b < m and best[a, b] > best[a, b + 1]:  # this optical file is an aborted or repeated take
                best[a, b + 1] = best[a, b]
                back[(a, b + 1)] = (a, b, None)
            if a < n and b < m:
                gain = scores.get((trials[a], optical[b]), 0.0) - MIN_PEAK
                if best[a, b] + gain > best[a + 1, b + 1]:
                    best[a + 1, b + 1] = best[a, b] + gain
                    back[(a + 1, b + 1)] = (a, b, b)
    pairing: dict[int, int | None] = {k: None for k in trials}
    peak: dict[int, float] = {}
    a, b = n, m
    while (a, b) != (0, 0):
        pa, pb, taken = back[(a, b)]
        if taken is not None:
            pairing[trials[pa]] = optical[taken]
            peak[trials[pa]] = scores.get((trials[pa], optical[taken]), 0.0)
        a, b = pa, pb
    return pairing, peak


def _trial_lags(pairing, lags, peaks, scores, rate_hz):
    """One clock offset per trial: the median of the sensors that carry a lag, when they agree."""
    out: dict[int, float | None] = {}
    diagnostics: dict[int, dict] = {}
    for k, j in pairing.items():
        if j is None:
            out[k] = None
            # Why a trial ended up unpaired is not always "no file left": a session can have as many files as
            # trials and still leave one out, because every candidate correlates below the floor.
            candidates = {jj: v for (kk, jj), v in scores.items() if kk == k}
            best = max(candidates, key=lambda jj: candidates[jj]) if candidates else None
            if best is None:
                reason = "no optical file of this session lies within the pairing search window of this trial"
            elif candidates[best] < MIN_PEAK:
                reason = (f"the best candidate optical file of this session (index {best}) correlates at "
                          f"{candidates[best]:.2f} against this trial's gyroscopes, below the {MIN_PEAK} floor "
                          f"— the trial has a candidate, not a partner")
            else:
                reason = (f"the session's monotone assignment left this trial unpaired; its best candidate "
                          f"(index {best}) correlates at {candidates[best]:.2f} but is taken by another trial")
            diagnostics[k] = {"reason": reason, "best_candidate_optical_index": best,
                              "best_candidate_peak": candidates.get(best)}
            continue
        per_segment = lags.get((k, j), {})
        per_peak = peaks.get((k, j), {})
        accepted = {s: v for s, v in per_segment.items() if per_peak.get(s, 0.0) >= MIN_PEAK}
        entry = {
            "per_sensor_lag_s": per_segment,
            "per_sensor_peak": per_peak,
            "sensors_accepted": sorted(accepted),
            "minimum_peak": MIN_PEAK,
        }
        if len(accepted) < MIN_SENSORS_AGREEING:
            out[k] = None
            entry["reason"] = (f"only {len(accepted)} of {len(per_segment)} sensors reach a peak correlation of "
                               f"{MIN_PEAK} — the trial's motion does not locate a lag")
            diagnostics[k] = entry
            continue
        values = np.array(sorted(accepted.values()))
        median = float(np.median(values))
        deviation = float(np.median(np.abs(values - median)))
        gate = MAX_DISAGREEMENT_SAMPLES / rate_hz
        agreeing = sorted(s for s, v in accepted.items() if abs(v - median) <= gate)
        outliers = {s: (accepted[s] - median) * MILLISECONDS_PER_S for s in sorted(accepted) if s not in agreeing}
        entry.update({
            "lag_s": median,
            "median_absolute_deviation_ms": deviation * MILLISECONDS_PER_S,
            "spread_ms": float((values.max() - values.min()) * MILLISECONDS_PER_S),
            "maximum_disagreement_ms": gate * MILLISECONDS_PER_S,
            "sensors_agreeing": agreeing,
            # Sensors that passed the correlation floor and still sit further than one sample from the trial's
            # published offset. The offset is the median and is not moved by them, but they are a failure of
            # that floor and are named rather than left inside a robust statistic: measured over this deposit,
            # 106 of the accepted trials have one, nine of them further than 3 s out.
            "sensors_outlying_ms": outliers,
        })
        if len(agreeing) < MIN_SENSORS_AGREEING or deviation > gate:
            out[k] = None
            entry["reason"] = (f"only {len(agreeing)} of the {len(accepted)} sensors that carry a lag are "
                               f"within one sample of their median (median absolute deviation "
                               f"{deviation * MILLISECONDS_PER_S:.1f} ms against the "
                               f"{MAX_DISAGREEMENT_SAMPLES:g} sample = {gate * MILLISECONDS_PER_S:.2f} ms at "
                               f"which one quantity measured five times is one quantity)")
        else:
            out[k] = median
        diagnostics[k] = entry
    return out, diagnostics


def _pairs_of_one_trial(reader, subject, k, j, lag, templates, rate_hz):
    """The two vector streams the rotation fit reads, per segment, for one repaired trial.

    Returns segment → dict with the reference's angular rate in the cluster frame, the gyroscope at the same
    instants, the accelerometer there, and the cluster rotations — everything already on one clock.
    """
    out: dict[str, dict] = {}
    markers = reader.markers(subject, j)
    streams = reader.streams(subject, k)
    for segment, stream in streams.items():
        cluster = markers.get(segment)
        if cluster is None or segment not in templates:
            continue
        R, _ = cluster_pose(cluster, templates[segment])
        t = np.arange(len(R)) / rate_hz + lag
        tm, w = angular_rate(t, R)
        if len(tm) < ALIGN_MIN_FRAMES_FALLBACK:
            continue
        ok_w = np.isfinite(w).all(1)
        if ok_w.sum() < ALIGN_MIN_FRAMES_FALLBACK:
            continue
        filled = np.stack([interpolate_gaps(np.where(ok_w, w[:, c], np.nan)) for c in range(3)], axis=1)
        w_low = zero_phase_lowpass(filled, rate_hz)
        gyr = np.stack([np.interp(tm, stream["t"], stream["gyr"][:, c], left=np.nan, right=np.nan)
                        for c in range(3)], axis=1)
        acc = np.stack([np.interp(tm, stream["t"], stream["acc"][:, c], left=np.nan, right=np.nan)
                        for c in range(3)], axis=1)
        ok_g = np.isfinite(gyr).all(1)
        if ok_g.sum() < ALIGN_MIN_FRAMES_FALLBACK:
            continue
        gyr_filled = np.stack([interpolate_gaps(np.where(ok_g, gyr[:, c], np.nan)) for c in range(3)], axis=1)
        g_low = zero_phase_lowpass(gyr_filled, rate_hz)
        ok = ok_w & ok_g
        out[segment] = {
            "trial": k,
            "omega_cluster": w_low[ok],
            "omega_sensor": g_low[ok],
            "gyroscope_raw": gyr[ok],
            "accelerometer": acc[ok],
            "cluster_rotation": R[:-1][ok],  # the left frame of each one-step difference, i.e. R(t)
        }
    return out


def _fit_rotations(reader, subject, session: Session, rate_hz: float) -> None:
    """One constant cluster-to-sensor rotation per sensor, from the whole session."""
    pooled: dict[str, list[dict]] = {}
    for k, j in session.pairing.items():
        lag = session.lag_s.get(k)
        if j is None or lag is None:
            continue
        for segment, data in _pairs_of_one_trial(reader, subject, k, j, lag, session.templates, rate_hz).items():
            pooled.setdefault(segment, []).append(data)

    # --- route one: angular rate, the transcribed protocol ----------------- #
    for segment, blocks in pooled.items():
        omega_cluster = np.concatenate([b["omega_cluster"] for b in blocks])
        omega_sensor = np.concatenate([b["omega_sensor"] for b in blocks])
        excited = np.linalg.norm(omega_sensor, axis=1) > ALIGN_MIN_OMEGA_RAD_S
        if excited.sum() < ALIGN_MIN_FRAMES:
            session.alignment[segment] = {
                "aligned": False,
                "route": "angular rate",
                "reason": (f"{int(excited.sum())} frames of the session exceed {ALIGN_MIN_OMEGA_RAD_S} rad/s, "
                           f"below the {ALIGN_MIN_FRAMES} the angular-rate route needs"),
                "frames_above_gate": int(excited.sum()),
                "frames_total": len(omega_sensor),
            }
            continue
        rotation = wahba(omega_cluster[excited], omega_sensor[excited])
        diagnostics = _diagnose(
            "angular rate", rotation, omega_cluster[excited], omega_sensor[excited],
            [(b["trial"], b["omega_cluster"], b["omega_sensor"],
              np.linalg.norm(b["omega_sensor"], axis=1) > ALIGN_MIN_OMEGA_RAD_S) for b in blocks],
            {"excitation_gate_rad_s": ALIGN_MIN_OMEGA_RAD_S},
        )
        # Only a fit that passed its own verdict is kept: a rotation the session does not determine must not
        # survive as the sensor's rotation just because the route ran.
        if diagnostics["aligned"]:
            session.cluster_from_sensor[segment] = rotation
        session.alignment[segment] = diagnostics

    # --- the laboratory vertical, measured from the sensors that aligned --- #
    session.vertical = _laboratory_vertical(pooled, session)
    session.world_from_deposit = _world_from_deposit(session.vertical.get("world_direction"))

    # --- route two: angular rate and gravity ------------------------------- #
    vertical = session.vertical.get("world_direction")
    for segment, blocks in pooled.items():
        if session.alignment.get(segment, {}).get("aligned"):
            continue
        if vertical is None:
            session.alignment.setdefault(segment, {}).update(
                {"reason_second_route": "no sensor of this session aligned by angular rate, so the laboratory "
                                        "vertical the gravity route needs is not measured"})
            continue
        pairs = [_direction_pairs(b, np.asarray(vertical, float)) for b in blocks]
        pairs = [p for p in pairs if p is not None]
        if not pairs:
            continue
        cluster = np.concatenate([p[0] for p in pairs])
        sensor = np.concatenate([p[1] for p in pairs])
        source = np.concatenate([p[2] for p in pairs])
        if len(cluster) < ALIGN_MIN_FRAMES_FALLBACK:
            session.alignment[segment].update(
                {"reason_second_route": f"{len(cluster)} direction pairs, below {ALIGN_MIN_FRAMES_FALLBACK}"})
            continue
        rotation = wahba(cluster, sensor)
        previous = session.alignment.get(segment, {})
        diagnostics = _diagnose(
            "angular rate and gravity", rotation, cluster, sensor,
            [(p[3], p[0], p[1], np.ones(len(p[0]), bool)) for p in pairs],
            {
                "excitation_gate_rad_s": ALIGN_FALLBACK_OMEGA_RAD_S,
                "still_gate_rad_s": STILL_OMEGA_RAD_S,
                "gravity_pairs": int((source == 0).sum()),
                "angular_rate_pairs": int((source == 1).sum()),
                "laboratory_vertical": session.vertical,
                "why_second_route": previous.get("reason", ""),
            },
        )
        if diagnostics["aligned"]:
            session.cluster_from_sensor[segment] = rotation
        session.alignment[segment] = diagnostics


def _direction_pairs(block: dict, vertical: np.ndarray):
    """Unit direction pairs of one trial for the second route: gravity where the sensor is still, angular rate
    where it turns. Returns (cluster-side, sensor-side, source flag, trial index)."""
    gyr = block["gyroscope_raw"]
    acc = block["accelerometer"]
    R = block["cluster_rotation"]
    speed = np.linalg.norm(gyr, axis=1)
    norm = np.linalg.norm(acc, axis=1)
    still = (speed < STILL_OMEGA_RAD_S) & (np.abs(norm - STANDARD_GRAVITY) < STILL_ACC_TOLERANCE_M_S2)
    still &= np.isfinite(R).all((1, 2))
    excited = np.linalg.norm(block["omega_sensor"], axis=1) > ALIGN_FALLBACK_OMEGA_RAD_S
    cluster, sensor, source = [], [], []
    if still.any():
        cluster.append(np.einsum("tji,j->ti", R[still], vertical))
        sensor.append(unit(acc[still]))
        source.append(np.zeros(int(still.sum()), int))
    if excited.any():
        cluster.append(unit(block["omega_cluster"][excited]))
        sensor.append(unit(block["omega_sensor"][excited]))
        source.append(np.ones(int(excited.sum()), int))
    if not cluster:
        return None
    return (np.concatenate(cluster), np.concatenate(sensor), np.concatenate(source), block["trial"])


def _laboratory_vertical(pooled, session: Session) -> dict:
    """The world-frame direction of gravity in the optical system's own frame, measured from the sensors that
    aligned by angular rate alone. Never assumed: the angle to the adapters' z-up convention is published."""
    per_segment: dict[str, list[float]] = {}
    for segment, blocks in pooled.items():
        rotation = session.cluster_from_sensor.get(segment)
        if rotation is None or not session.alignment.get(segment, {}).get("aligned"):
            continue
        directions = []
        for b in blocks:
            acc, R = b["accelerometer"], b["cluster_rotation"]
            speed = np.linalg.norm(b["gyroscope_raw"], axis=1)
            norm = np.linalg.norm(acc, axis=1)
            still = ((speed < STILL_OMEGA_RAD_S) & (np.abs(norm - STANDARD_GRAVITY) < STILL_ACC_TOLERANCE_M_S2)
                     & np.isfinite(R).all((1, 2)))
            if still.any():
                directions.append(np.einsum("tij,jk,tk->ti", R[still], rotation, unit(acc[still])))
        if directions:
            per_segment[segment] = unit(np.concatenate(directions).mean(axis=0)).tolist()
    if not per_segment:
        return {"world_direction": None, "reason": "no sensor aligned by angular rate alone"}
    stack = np.array(list(per_segment.values()))
    mean = unit(stack.mean(axis=0))
    z_up = np.array([0.0, 0.0, 1.0])
    return {
        "world_direction": mean.tolist(),
        "per_sensor": per_segment,
        "sensor_disagreement_deg": float(np.max(axis_angle_deg(stack, np.tile(mean, (len(stack), 1))))),
        "angle_to_z_up_deg": float(axis_angle_deg(mean[None], z_up[None])[0]),
        "note": "the direction the still-frame specific force of the aligned sensors points in, in the "
                "optical system's own world frame; measured, never assumed",
    }


def _world_from_deposit(vertical) -> np.ndarray:
    """The constant world rotation that puts the deposit's measured vertical on ``+z``.

    A change of gauge, not of data: it is one rotation applied to every segment of the recording, so no
    relative orientation moves by it. It exists because ``schema.py`` declares the adapters' world frame to be
    z-up and every measure that reads gravity (``measures/alignment.py``) takes that at its word, while this
    deposit's optical world frame is **y-up** — measured, 89.5–90.4° from the z axis on all sixteen subjects.
    The rotation is the smallest one carrying the measured vertical onto ``+z``, so the residual tilt of the
    camera frame (0.25–0.81° between the sensors of one session) is corrected with it; the heading of the
    resulting frame is arbitrary, as the heading of a laboratory frame always is.
    """
    if vertical is None:
        return np.eye(3)
    v = unit(np.asarray(vertical, float))
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = float(np.linalg.norm(axis))
    c = float(v @ z)
    if s < NORM_FLOOR:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s**2)


def _diagnose(route, rotation, cluster, sensor, blocks, extra) -> dict:
    """Everything a reader needs to judge one fit: conditioning, residual, per-trial scatter, split halves.

    The verdict — ``aligned`` — is the **between-trial scatter** against
    :data:`MAX_BETWEEN_TRIAL_SCATTER_DEG`, because that is the direct measurement of the constant being claimed
    (see that constant's own docstring for the two sensors on which it and the conditioning gate disagree).
    The conditioning of the direction cloud decides only where the session has fewer than two usable trials and
    the scatter therefore does not exist.
    """
    spread = direction_spread(cluster)
    after = axis_angle_deg(cluster, sensor @ rotation.T)
    before = axis_angle_deg(cluster, sensor)
    per_trial = []
    within = []
    for trial, a, b, mask in blocks:
        if mask.sum() < ALIGN_MIN_FRAMES_FALLBACK:
            continue
        per_trial.append((trial, wahba(a[mask], b[mask])))
        # The split half *inside* one trial: unlike the split over trials below, pooling cannot average the
        # per-trial disagreement away here, so this one does discriminate a constant from a drifting fit.
        am, bm = a[mask], b[mask]
        half = len(am) // 2
        if half >= ALIGN_MIN_FRAMES_FALLBACK:
            within.append(rotation_angle_deg(wahba(am[:half], bm[:half]).T @ wahba(am[half:], bm[half:])))
    scatter = [rotation_angle_deg(rotation.T @ r) for _, r in per_trial]
    halves: list[list] = [[], []]
    for i, (trial, _) in enumerate(per_trial):
        halves[i % 2].append(trial)
    split = None
    if len(per_trial) >= 2:
        fits = []
        for half in halves:
            picked = [(a, b, m) for t, a, b, m in blocks if t in half]
            if not picked or sum(int(m.sum()) for _, _, m in picked) < ALIGN_MIN_FRAMES_FALLBACK:
                fits = []
                break
            fits.append(wahba(np.concatenate([a[m] for a, _, m in picked]),
                              np.concatenate([b[m] for _, b, m in picked])))
        if len(fits) == 2:
            split = rotation_angle_deg(fits[0].T @ fits[1])
    identifiable = bool(np.isfinite(spread["spread_deg"]) and spread["spread_deg"] >= MIN_DIRECTION_SPREAD_DEG)
    scatter_median = float(np.median(scatter)) if scatter else None
    determined = None if len(per_trial) < 2 else bool(scatter_median < MAX_BETWEEN_TRIAL_SCATTER_DEG)
    out = {
        "aligned": identifiable if determined is None else determined,
        "constant_over_trials": determined,
        "identifiable": identifiable,
        "within_trial_split_half_deg_median": float(np.median(within)) if within else None,
        "maximum_between_trial_scatter_deg": MAX_BETWEEN_TRIAL_SCATTER_DEG,
        "route": route,
        "cluster_from_sensor_R": rotation.tolist(),
        "rotation_deg": rotation_angle_deg(rotation),
        "n_pairs": len(cluster),
        "n_trials": len(per_trial),
        "direction_spread_deg": spread["spread_deg"],
        "smallest_direction_spread_deg": spread["smallest_spread_deg"],
        "singular_value_ratio": spread["condition_number"],
        "minimum_direction_spread_deg": MIN_DIRECTION_SPREAD_DEG,
        "residual_deg_median": float(np.median(after)),
        "residual_deg_p95": float(np.percentile(after, RESIDUAL_PERCENTILE)),
        "residual_before_deg_median": float(np.median(before)),
        "between_trial_scatter_deg_median": scatter_median,
        "between_trial_scatter_deg_p95": float(np.percentile(scatter, RESIDUAL_PERCENTILE)) if scatter else None,
        "split_half_disagreement_deg": split,
        "split_half_disagreement_is": "the two halves of the session fitted pooled; it scales as twice the "
                                      "between-trial scatter over the root of the trial count and carries no "
                                      "information beyond it — the discriminating number is "
                                      "within_trial_split_half_deg_median",
        **extra,
    }
    if determined is False:
        out["reason"] = (
            f"not one constant over this session: the {len(per_trial)} trials fitted separately scatter by "
            f"{scatter_median:.3g}° (median) around the pooled rotation, above the "
            f"{MAX_BETWEEN_TRIAL_SCATTER_DEG:.3g}° at which repeated fits of one rotation are one rotation; "
            f"the direction cloud is {'well' if identifiable else 'not'} conditioned "
            f"({spread['spread_deg']:.3g}° spread), so this is a fit that is not constant, not a fit the "
            f"geometry could not condition"
        )
    elif determined is None and not identifiable:
        out["reason"] = (
            f"unidentifiable over this session, and fewer than two trials to test the constant against: the "
            f"reference-side directions spread by {spread['spread_deg']:.3g}°, below the "
            f"{MIN_DIRECTION_SPREAD_DEG:.3g}° at which the rotation about the dominant direction is determined "
            f"within an order of magnitude as well as the other two (the characterisation's own gate, "
            f"measures/alignment.py)"
        )
    return out
