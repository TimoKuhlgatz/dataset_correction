"""Calibration-free relative orientation estimator ("zoe"): the 12-state EKF of Alg. 1 in the public repository
Calibration-free-IMU-orientation-location-reproducibility, ported to numba.

What the filter does (unchanged from the published code, `run_kf_with_lpf_strategy(data, 'dwacc')`): the state is
the orientation error of both sensors and both sensor-to-joint-centre vectors r1, r2 (started at [1, 1, 1] m,
estimated online, so no lever-arm calibration). Time update: strapdown integration of the raw gyroscope.
Measurement: the accelerations of both sensors, moved to the joint centre with the lever arms, must agree in the
world frame. The measurement uses 6 Hz low-pass filtered accelerations and the derivative of the 6 Hz low-pass
filtered gyroscope (4th-order Butterworth, forward-backward), and the raw gyroscope in the centripetal term. Samples
with an acceleration component above 300 m/s^2 skip the measurement update. Both orientations start gravity-aligned
(Kok 2017, eq. 3.64a) from the first sample whose acceleration norm is within 0.6-1.4 g, as published.

Tuned parameters (an absolute grid around the published OpenSense values, same value on every axis as published):
cov_w, the gyroscope process noise (published 1e-2), and cov_lnk, the measurement noise (published
0.35^2 * 10 = 1.225); the lever-arm process noise stays 1e-9 * cov_lnk and the initial covariance the identity, as
published.

Changes against the published code, implementation only: numba loop instead of Python (orientations and lever arms
agree with the published code to 1e-9 on the same input at 100 Hz); the low-pass filter uses the recording's rate
(the published `lowpass_filter` assumes 100 Hz unless told otherwise, which would move the cutoff away from 6 Hz on
datasets at other rates); arrays are (N, 3).
"""
from __future__ import annotations

import numba
import numpy as np
from scipy.signal import butter, filtfilt

CUTOFF_HZ = 6.0
FILTER_ORDER = 4
ACC_REJECT_MPS2 = 300.0
LEVER_ARM_INIT_M = 1.0
STABLE_ACC_MPS2 = (5.89, 13.73)  # 0.6 g ... 1.4 g, for the gravity-aligned start
PUBLISHED_COV_W = 1e-2
PUBLISHED_COV_LNK = 0.35 ** 2 * 10
# cov_lnk takes the place of the MEKF's R, cov_w that of its Q; as many values as the MEKF grids (9 and 14)
COV_LNK_GRID = tuple(float(f'{PUBLISHED_COV_LNK * 10.0 ** k:.4g}') for k in range(-4, 5))
COV_W_GRID = tuple(float(f'{PUBLISHED_COV_W * 10.0 ** k:.4g}') for k in range(-7, 7))


def lowpass(x: np.ndarray, rate_hz: float) -> np.ndarray:
    b, a = butter(FILTER_ORDER, CUTOFF_HZ / (0.5 * rate_hz), btype='low')
    return filtfilt(b, a, x, axis=0)


def five_point_derivative(y: np.ndarray, rate_hz: float) -> np.ndarray:
    """As `approx_derivative` of the published code: central five-point difference, zero at the first and last two
    samples."""
    dy = np.zeros_like(y)
    dy[2:-2] = (y[:-4] - 8 * y[1:-3] + 8 * y[3:-1] - y[4:]) * (rate_hz / 12)
    return dy


def gravity_aligned_start(acc: np.ndarray) -> np.ndarray:
    """`kok_init_quat` of the published code at the first sample with a plausible gravity norm; quaternion [w, x, y, z]
    rotating the measured gravity onto [0, 0, -1]."""
    norms = np.linalg.norm(acc, axis=1)
    stable = np.flatnonzero((norms >= STABLE_ACC_MPS2[0]) & (norms <= STABLE_ACC_MPS2[1]))
    g_b = acc[stable[0] if len(stable) else 0]
    g_b = g_b / (np.linalg.norm(g_b) + 1e-12)
    g_n = np.array([0.0, 0.0, -1.0])
    dot = float(np.clip(g_b @ g_n, -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(g_b, g_n)
    axis /= np.linalg.norm(axis) + 1e-12
    half = np.arccos(dot) / 2.0
    q = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    return q / (np.linalg.norm(q) + 1e-12)


class ZoeInputs:
    """The signals of one joint prepared once (filtering, derivative, start orientation) for many filter runs."""

    def __init__(self, gyr_1, acc_1, gyr_2, acc_2, rate_hz: float):
        self.rate = float(rate_hz)
        self.gyr = (np.ascontiguousarray(gyr_1, float), np.ascontiguousarray(gyr_2, float))
        self.acc_filtered = tuple(np.ascontiguousarray(lowpass(a, rate_hz)) for a in (acc_1, acc_2))
        self.dgyr_filtered = tuple(np.ascontiguousarray(five_point_derivative(lowpass(g, rate_hz), rate_hz))
                                   for g in (gyr_1, gyr_2))
        self.q_start = (gravity_aligned_start(np.asarray(acc_1)), gravity_aligned_start(np.asarray(acc_2)))

    def run(self, cov_lnk: float, cov_w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Orientations of both sensors (N, 4) [w, x, y, z], sensor to world, and the final lever arms (m)."""
        return _ekf(self.gyr[0], self.gyr[1], self.acc_filtered[0], self.acc_filtered[1], self.dgyr_filtered[0],
                    self.dgyr_filtered[1], _normalized(self.q_start[0]), _normalized(self.q_start[1]),
                    1.0 / self.rate, float(cov_w), float(cov_lnk), ACC_REJECT_MPS2, LEVER_ARM_INIT_M)


def _normalized(q: np.ndarray) -> np.ndarray:
    """`quatnormalize` of the published code: positive scalar part, unit norm."""
    q = -q if q[0] < 0 else q
    return q / np.linalg.norm(q)


@numba.njit(cache=True)
def _cross(v):
    m = np.zeros((3, 3))
    m[0, 1], m[0, 2] = -v[2], v[1]
    m[1, 0], m[1, 2] = v[2], -v[0]
    m[2, 0], m[2, 1] = -v[1], v[0]
    return m


@numba.njit(cache=True)
def _matrix(q):
    n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    w, x, y, z = q[0] / n, q[1] / n, q[2] / n, q[3] / n
    m = np.empty((3, 3))
    m[0, 0], m[0, 1], m[0, 2] = w * w + x * x - y * y - z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)
    m[1, 0], m[1, 1], m[1, 2] = 2 * (x * y + z * w), w * w - x * x + y * y - z * z, 2 * (y * z - x * w)
    m[2, 0], m[2, 1], m[2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), w * w - x * x - y * y + z * z
    return m


@numba.njit(cache=True)
def _multiply(q, r):
    out = np.empty(4)
    out[0] = q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3]
    out[1] = q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2]
    out[2] = q[0] * r[2] + q[2] * r[0] + q[3] * r[1] - q[1] * r[3]
    out[3] = q[0] * r[3] + q[3] * r[0] + q[1] * r[2] - q[2] * r[1]
    return out


@numba.njit(cache=True)
def _exp(v):
    """`EXPq` of the published code: [cos|v|, v/|v| sin|v|]."""
    n = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    out = np.empty(4)
    out[0] = np.cos(n)
    s = np.sin(n) / (n + 2.220446049250313e-16)
    out[1], out[2], out[3] = v[0] * s, v[1] * s, v[2] * s
    return out


@numba.njit(cache=True)
def _ekf(gyr_1, gyr_2, acc_1, acc_2, dgyr_1, dgyr_2, q1, q2, T, cov_w, cov_lnk, acc_reject, r_init):
    N = gyr_1.shape[0]
    out_1, out_2 = np.zeros((N, 4)), np.zeros((N, 4))
    out_1[0], out_2[0] = q1, q2
    q1, q2 = q1.copy(), q2.copy()
    P = np.zeros((12, 12))
    for i in range(12):
        P[i, i] = 1.0
    x = np.zeros(12)
    x[6:12] = r_init
    r1, r2 = x[6:9].copy(), x[9:12].copy()
    Q_eta = T * T * cov_w  # G Q G^T with G = T * rotation and Q = cov_w * I
    Q_r = cov_lnk * 1e-9
    R_meas = np.eye(3) * cov_lnk
    for t in range(1, N):
        x[0:6] = 0.0
        q1 = _multiply(q1, _exp(T / 2 * gyr_1[t - 1]))
        q2 = _multiply(q2, _exp(T / 2 * gyr_2[t - 1]))
        for i in range(6):
            P[i, i] += Q_eta
            P[6 + i, 6 + i] += Q_r
        if np.any(np.abs(acc_1[t]) > acc_reject) or np.any(np.abs(acc_2[t]) > acc_reject):
            out_1[t], out_2[t] = q1, q2
            continue
        C1 = _cross(gyr_1[t]) @ _cross(gyr_1[t]) + _cross(dgyr_1[t])
        C2 = _cross(gyr_2[t]) @ _cross(gyr_2[t]) + _cross(dgyr_2[t])
        a1 = acc_1[t] - C1 @ r1
        a2 = acc_2[t] - C2 @ r2
        R1, R2 = _matrix(q1), _matrix(q2)
        g1, g2 = R1 @ a1, R2 @ a2
        H = np.zeros((3, 12))
        H[:, 0:3] = _cross(g1)
        H[:, 3:6] = -_cross(g2)
        H[:, 6:9] = R1 @ C1
        H[:, 9:12] = -(R2 @ C2)
        e = g1 - g2
        PHt = P @ H.T
        S = H @ PHt + R_meas
        K = PHt @ np.linalg.inv(S)
        x = x + K @ e
        r1, r2 = x[6:9].copy(), x[9:12].copy()
        q1 = _multiply(_exp(x[0:3] / 2), q1)
        q2 = _multiply(_exp(x[3:6] / 2), q2)
        P = P - K @ S @ K.T
        out_1[t], out_2[t] = q1, q2
    return out_1, out_2, r1, r2
