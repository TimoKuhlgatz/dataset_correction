"""Measurement primitives shared by the effect modules — no estimator among them.

Everything here is either a **constant fit over a whole trial** (a rotation by singular value decomposition,
a constant vector by linear least squares), a **zero-phase** filter or differentiator, or plain descriptive
statistics. None of it estimates an attitude, a heading or a joint centre from an inertial method
(``docs/characterisation.md`` rule 2). The two fits used everywhere are named where they are used:
*Wahba fit* (the constant rotation between two vector streams) and *fixed-point fit* (the constant position
that is common to two moving frames).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

RAD2DEG = 180.0 / np.pi
"""Radians to degrees.

Units: degrees per radian.
Provenance: physical — the exact definition of the degree, no data enters it."""

GRAVITY = 9.80665
"""Standard gravity, used only to print an acceleration in units of g.

Units: m/s².
Provenance: physical — the defined value of ISO 80000-3 / CODATA."""

M_TO_CM = 100.0
"""Metres to centimetres.

Units: cm per m.
Provenance: physical — an exact unit conversion."""


# --- small rotation utilities -------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    z = np.zeros(v.shape[:-1])
    return np.stack(
        [
            np.stack([z, -v[..., 2], v[..., 1]], -1),
            np.stack([v[..., 2], z, -v[..., 0]], -1),
            np.stack([-v[..., 1], v[..., 0], z], -1),
        ],
        -2,
    )


def vee(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    return np.stack([S[..., 2, 1] - S[..., 1, 2], S[..., 0, 2] - S[..., 2, 0], S[..., 1, 0] - S[..., 0, 1]], -1) / 2.0


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """The rotation angle of one or many rotation matrices, in degrees."""
    R = np.asarray(R, float)
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def log_map(R: np.ndarray) -> np.ndarray:
    """The rotation vector of one or many rotation matrices (axis times angle, radians).

    Computed by SciPy's ``Rotation.from_matrix(...).as_rotvec()`` (quaternion route), which keeps its
    accuracy near half a turn (an ``arccos``/``sin θ`` form loses up to 6° there). Non-finite matrices give a non-finite vector; a slightly non-orthogonal
    matrix (a spline-resampled pose) is projected onto the nearest rotation first, as SciPy does.
    """
    from scipy.spatial.transform import Rotation

    R = np.asarray(R, float)
    lead = R.shape[:-2]
    flat = R.reshape(-1, 3, 3)
    ok = np.isfinite(flat).all(axis=(1, 2))
    out = np.full((len(flat), 3), np.nan)
    if ok.any():
        out[ok] = Rotation.from_matrix(flat[ok]).as_rotvec()
    return out.reshape(*lead, 3)


def project_to_rotation(M: np.ndarray) -> np.ndarray:
    """The nearest rotation matrix to ``M`` in the Frobenius sense (a projection, not an estimator)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def median_rotation(R: np.ndarray) -> np.ndarray:
    """The trial median of a sequence of rotations: the element-wise median matrix projected onto rotations.

    A constant fit over the whole trial (a measurement, not a filter). Element-wise so that a few missing or
    wild frames do not move it; the projection makes the result a rotation again.
    """
    R = np.asarray(R, float)
    ok = np.isfinite(R).all(axis=(1, 2))
    if not ok.any():
        return np.full((3, 3), np.nan)
    return project_to_rotation(np.median(R[ok], axis=0))


def wahba_fit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The single constant rotation ``R`` minimising ``Σ_t ‖b(t) − R a(t)‖²`` (Wahba's problem, solved once
    over the whole trial by singular value decomposition). A constant fit, not an orientation estimator."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if ok.sum() < 3:
        return np.full((3, 3), np.nan)
    H = b[ok].T @ a[ok]
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


# --- resampling and filtering (zero-phase; a measurement may, a causal method may not) ------------------

def strictly_increasing_mask(t: np.ndarray) -> np.ndarray:
    """True where a sample's time exceeds every earlier time: the first row of a tied or backward timestamp
    stays, the later rows are masked. A deposit that quotes its time column too coarsely (RepoIMU, defect 7)
    keeps its rows in the adapter; a spline needs a strictly increasing knot vector, so the tie is resolved
    here and only here."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return np.zeros(0, dtype=bool)
    running_max = np.maximum.accumulate(np.where(np.isfinite(t), t, -np.inf))
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = t[1:] > running_max[:-1]
    return keep & np.isfinite(t)


def resample_cubic(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Cubic-spline resampling onto another time base; NaN outside the source range or where too few samples.
    Tied or backward source timestamps keep their first row only (``strictly_increasing_mask``)."""
    t_src = np.asarray(t_src, float)
    x = np.asarray(x, float)
    single = x.ndim == 1
    X = x[:, None] if single else x
    out = np.full((len(t_dst), X.shape[1]), np.nan)
    increasing = strictly_increasing_mask(t_src)
    for j in range(X.shape[1]):
        ok = increasing & np.isfinite(X[:, j])
        if ok.sum() < 4:
            continue
        cs = CubicSpline(t_src[ok], X[ok, j], extrapolate=False)
        out[:, j] = cs(t_dst)
    return out[:, 0] if single else out


def resample_cubic_no_bridge(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray,
                             max_gap_steps: float = 2.0) -> np.ndarray:
    """:func:`resample_cubic`, then NaN wherever a destination time falls inside a **gap** of the source —
    two consecutive finite source rows more than ``max_gap_steps`` median steps apart — so that nothing is
    bridged (the contract of :func:`finite_runs`). Rows finite in every column count as present. Data-free:
    two steps is the smallest gap a sample-and-hold or a single dropped frame cannot produce."""
    out = resample_cubic(t_src, x, t_dst)
    t_src = np.asarray(t_src, float)
    X = np.asarray(x, float)
    X = X[:, None] if X.ndim == 1 else X
    ok = strictly_increasing_mask(t_src) & np.isfinite(X).all(1)
    ts = t_src[ok]
    if len(ts) < 2:
        return out
    step = float(np.median(np.diff(ts)))
    destination = np.asarray(t_dst, float)
    idx = np.searchsorted(ts, destination, side="right")
    inside = (idx > 0) & (idx < len(ts))
    preceding = np.clip(idx - 1, 0, len(ts) - 1)
    exact_source_knot = (idx > 0) & (ts[preceding] == destination)
    gap = np.zeros(len(idx))  # outside the source range resample_cubic has already left NaN
    gap[inside] = ts[idx[inside]] - ts[idx[inside] - 1]
    bridged = inside & ~exact_source_knot & (gap > max_gap_steps * step)
    out[bridged] = np.nan
    return out


def finite_runs(x: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` of the contiguous stretches of rows that are finite in every column and at
    least ``min_len`` long. A reference with marker gaps is handled stretch by stretch; nothing is bridged."""
    x = np.asarray(x, float)
    ok = np.isfinite(x).all(axis=1) if x.ndim > 1 else np.isfinite(x)
    if len(ok) == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], ok, [False])).astype(int)))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=True) if b - a >= min_len]


def zero_phase_lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low pass (``filtfilt``). Zero-phase is allowed for a measurement and forbidden
    for the causal method (specification rule 2). Returns the input unchanged if the cut-off is above Nyquist."""
    x = np.asarray(x, float)
    if not np.isfinite(fs) or fc >= 0.5 * fs:
        return x
    b, a = butter(order, fc / (0.5 * fs), btype="low")
    pad = 3 * max(len(a), len(b))
    if x.shape[0] <= pad:
        return x
    return filtfilt(b, a, x, axis=0)


def uniform_rate(t: np.ndarray) -> float:
    t = np.asarray(t, float)
    if len(t) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(t)))


# --- pose derived quantities --------------------------------------------------------------------------

def pose_angular_rate(t: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The reference's angular rate in the moving frame, from the deposited rotations alone.

    ``ω(t + ½Δt) = log(R(t)ᵀ R(t+Δt)) / Δt`` — a symmetric (zero-phase) difference of the reference, not a
    filter and not an estimator. Returns the mid-sample times and ``ω`` in rad/s.
    """
    t = np.asarray(t, float)
    R = np.asarray(R, float)
    dt = np.diff(t)
    dR = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = log_map(dR) / dt[:, None]
    w[dt <= 0] = np.nan  # a tied or backward timestamp spans no time and has no rate (RepoIMU defect 7)
    return 0.5 * (t[:-1] + t[1:]), w


def pose_linear_acceleration(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Second time derivative of a position track by a symmetric (zero-phase) second difference, m/s²."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    dt = float(np.median(np.diff(t)))
    a = np.full_like(p, np.nan)
    a[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt**2
    return a


# --- descriptive statistics used by several definitions ------------------------------------------------

def _window_bounds(n: int, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``[lo, hi)`` sample bounds of a centred moving window of ``win`` samples at every position, and
    whether the window is complete. A window that would run past either end is **not** truncated to what is
    there: it is flagged incomplete and its statistic is NaN, so an argument-minimum over the moving mean
    cannot be attracted to a half-window at the record ends."""
    i = np.arange(n)
    half = win // 2
    lo = i - half
    hi = lo + win
    complete = (lo >= 0) & (hi <= n)
    return np.clip(lo, 0, n), np.clip(hi, 0, n), complete


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Root mean square of ``x`` in a centred moving window of ``win`` samples (same length, NaN at the ends).

    Computed from the prefix sums at every position at once — the same arithmetic as a window-by-window
    evaluation, without a Python loop over samples (a loop over a 2 kHz seventeen-minute trial is minutes).
    """
    x = np.asarray(x, float)
    win = max(int(win), 1)
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x**2))])
    n = np.concatenate([[0], np.cumsum(np.isfinite(x).astype(int))])
    lo, hi, complete = _window_bounds(len(x), win)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(complete & (cnt > 0), np.sqrt((c[hi] - c[lo]) / np.where(cnt > 0, cnt, 1)), np.nan)


def moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Mean of ``x`` in a centred moving window of ``win`` samples; prefix sums, no loop over samples."""
    x = np.asarray(x, float)
    win = max(int(win), 1)
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    n = np.concatenate([[0], np.cumsum(np.isfinite(x).astype(int))])
    lo, hi, complete = _window_bounds(len(x), win)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(complete & (cnt > 0), (c[hi] - c[lo]) / np.where(cnt > 0, cnt, 1), np.nan)


def most_active_window(x: np.ndarray, win: int) -> tuple[int, int]:
    """The ``win``-sample window with the largest mean of ``x`` — an argument-maximum, so no threshold."""
    x = np.asarray(x, float)
    n = len(x)
    win = int(min(max(win, 1), n))
    if n <= win:
        return 0, n
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    sums = c[win:] - c[:-win]
    lo = int(np.nanargmax(sums))
    return lo, lo + win


def log_grid(lo: float, hi: float, n: int = 40) -> np.ndarray:
    """A logarithmic grid of levels; the module reports curves over such grids instead of choosing a level."""
    lo = max(float(lo), 1e-6)
    hi = max(float(hi), lo * 10)
    return np.geomspace(lo, hi, int(n))


def fraction_below(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(len(levels), np.nan)
    xs = x[ok]
    return np.array([float(np.mean(xs < lv)) for lv in levels])


def _longest_run(m: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array, from the run boundaries (no loop over samples)."""
    if not m.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], m.view(np.int8) if m.dtype == bool else m, [0]))))
    return int((edges[1::2] - edges[0::2]).max())


def longest_stretch_below(x: np.ndarray, levels: np.ndarray, dt: float) -> np.ndarray:
    """Longest contiguous stretch (seconds) during which ``x`` stays below each level."""
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    out = np.zeros(len(levels))
    for i, lv in enumerate(levels):
        out[i] = _longest_run(finite & (x < lv)) * dt
    return out


def longest_prefix_below(x: np.ndarray, levels: np.ndarray, dt: float) -> np.ndarray:
    """Length (seconds) of the longest **prefix** of the recording during which ``x`` stays below each level."""
    x = np.asarray(x, float)
    out = np.zeros(len(levels))
    for i, lv in enumerate(levels):
        m = np.isfinite(x) & (x < lv)
        k = int(np.argmin(m)) if not m.all() else len(m)
        out[i] = k * dt
    return out


# --- the quietest stretch, and the noise level read off it ---------------------------------------------

QUIET_WINDOW_S = 1.0
"""The window the quietest stretch is searched in (data-free: one second, long enough for a
sample-to-sample statistic at every deposit rate in the census, short enough to fit inside a pause)."""


def quietest_window(stream, window_s: float = QUIET_WINDOW_S) -> tuple[int, int, float]:
    """The quietest stretch of a recording: the ``window_s`` window with the smallest mean angular rate.

    An argument-minimum, so it needs **no threshold** (specification §2.11, which locates the still stretch
    "by the same curve as §2.1 and never by a chosen threshold"). Returns ``(start, stop, mean rate in °/s)``.
    """
    t = np.asarray(stream.t, float)
    g = np.asarray(stream.gyr, float)
    fs = uniform_rate(t)
    win = max(int(np.rint(window_s * fs)), 4)
    rate = np.linalg.norm(g, axis=1) * RAD2DEG
    if len(rate) < win:
        return 0, len(rate), float(np.nanmean(rate)) if len(rate) else float("nan")
    m = moving_mean(rate, win)  # NaN wherever the window would be incomplete: only full windows compete
    if not np.isfinite(m).any():
        return 0, len(rate), float(np.nanmean(rate))
    i = int(np.nanargmin(m))
    lo = i - win // 2
    return lo, lo + win, float(m[i])


def noise_level(stream, window_s: float = QUIET_WINDOW_S) -> dict:
    """Sensor noise from the quietest stretch: ``std(sample-to-sample difference) / √2`` per axis.

    Specification §2.11. Returns levels in °/s and m/s², the densities in °/s/√Hz and (m/s²)/√Hz at the rate
    given (the caller passes the **native** rate where it recovered one), the quietest window's own mean
    angular rate, and the sample count. No threshold is used to find the window (see :func:`quietest_window`).
    """
    lo, hi, quiet_rate = quietest_window(stream, window_s)
    g = np.asarray(stream.gyr, float)[lo:hi] * RAD2DEG
    a = np.asarray(stream.acc, float)[lo:hi]
    fs = uniform_rate(np.asarray(stream.t, float))
    lvl_g = np.std(np.diff(g, axis=0), axis=0) / np.sqrt(2.0) if len(g) > 2 else np.full(3, np.nan)
    lvl_a = np.std(np.diff(a, axis=0), axis=0) / np.sqrt(2.0) if len(a) > 2 else np.full(3, np.nan)
    return {
        "gyr_level_deg_s": lvl_g,
        "acc_level_m_s2": lvl_a,
        "gyr_level_norm_deg_s": float(np.linalg.norm(lvl_g) / np.sqrt(3.0)),
        "acc_level_norm_m_s2": float(np.linalg.norm(lvl_a) / np.sqrt(3.0)),
        "quiet_window_rate_deg_s": quiet_rate,
        "window": (lo, hi),
        "rate_hz": fs,
    }


NOISE_FLOOR_DEG_S = 1e-3
"""Numerical floor for a measured noise level of exactly zero (a noiseless synthetic stream), so that a
logarithmic grid anchored on the noise level stays defined.

Units: °/s.
Provenance: data-free; it is never a detection threshold."""
