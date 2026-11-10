"""Small rotation and signal helpers shared by the error checks."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_from_wxyz(q) -> Rotation:
    """scipy Rotation from quaternions stored scalar first (w, x, y, z). Rows with NaN become identity;
    the caller is responsible for marking them invalid."""
    q = np.asarray(q, float)
    q = np.where(np.isfinite(q).all(axis=-1, keepdims=True), q, [1.0, 0.0, 0.0, 0.0])
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def reference_angular_rate(orientation: Rotation, rate_hz: float, convention: str = 'segment_to_world'):
    """Angular rate of the segment in its own frame, computed from consecutive reference orientations.

    This is the "simulated gyroscope": what a perfect gyroscope fixed to the segment would measure.
    Returns (times, rates): the rate between frame k and k+1 is assigned to the middle of the two frames,
    times relative to the stream start. `convention` tells how the orientation is stored; the magnitude of the rate
    (`np.linalg.norm(rates, axis=1)`) is the same for both conventions.
    """
    if convention == 'world_to_segment':
        rates = -(orientation[1:] * orientation[:-1].inv()).as_rotvec() * rate_hz
    else:
        rates = (orientation[:-1].inv() * orientation[1:]).as_rotvec() * rate_hz
    times = (np.arange(len(orientation) - 1) + 0.5) / rate_hz
    return times, rates


def best_rotation(p: np.ndarray, q: np.ndarray, proper: bool = True):
    """Rotation matrix M that best maps the rows of p onto the rows of q (q ≈ M p), after removing the means.

    proper=False returns the best improper map (a rotation combined with a mirror, determinant -1).
    Comparing the two tells whether the data are mirrored. Returns (M, rms residual)."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T)) * (1.0 if proper else -1.0)
    m = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    residual = qc - pc @ m.T
    return m, float(np.sqrt((residual ** 2).sum(1).mean()))


def interpolate(times_out: np.ndarray, times_in: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column; NaN outside the input time range."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        return np.interp(times_out, times_in, values, left=np.nan, right=np.nan)
    return np.column_stack([np.interp(times_out, times_in, values[:, i], left=np.nan, right=np.nan)
                            for i in range(values.shape[1])])


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Widen every True region of a boolean mask by n samples on both sides."""
    if n <= 0 or not mask.any():
        return mask.copy()
    full = np.convolve(mask.astype(float), np.ones(2 * n + 1), mode='full')  # length len(mask) + 2n
    return full[n:n + len(mask)] > 0
