"""Synthetic recordings with known motion and known injected errors, for the known-answer tests.

The body rotates with a smooth random angular rate (a sum of sines). Its orientation is integrated finely (midpoint
rule at 2 kHz) and sampled as the optical reference; the gyroscope measures the exact body rate and the
accelerometer measures gravity only (no linear acceleration), unless `linear_acc_mps2` is set: then all segments
share one smooth linear acceleration in the world frame, as sensors at a common joint centre would. Gravity alone
gives each sensor its inclination but not their relative heading, so MEKF-acc needs this shared acceleration to
estimate the relative orientation (measured: 72° error without it, 0.8° with it, effect.py). Every error check
must find nothing on the clean recording and recover the injected error on the defective one.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .recording import ImuStream, Recording, ReferenceStream, Segment

GRAVITY = np.array([0.0, 0.0, 9.81])


def smooth_rate(t: np.ndarray, rng: np.random.Generator, peak_dps: float = 200.0) -> np.ndarray:
    """A smooth, non-periodic three-axis angular rate (rad/s) at times t."""
    freqs = rng.uniform(0.2, 2.0, (3, 4))
    phases = rng.uniform(0, 2 * np.pi, (3, 4))
    w = np.stack([np.sin(2 * np.pi * freqs[i][:, None] * t + phases[i][:, None]).sum(0) for i in range(3)], 1)
    return w / np.abs(w).max() * np.radians(peak_dps)


def _integrate(start: Rotation, rate: np.ndarray, dt: float) -> np.ndarray:
    """Orientation quaternions (x, y, z, w) from a body-frame rate, midpoint rule (segment_to_world convention)."""
    steps = Rotation.from_rotvec(0.5 * (rate[:-1] + rate[1:]) * dt).as_quat()
    out = np.empty((len(rate), 4))
    x, y, z, w = start.as_quat()
    out[0] = x, y, z, w
    for k, (bx, by, bz, bw) in enumerate(steps.tolist(), 1):  # q_{k} = q_{k-1} * step, plain floats for speed
        x, y, z, w = (w * bx + x * bw + y * bz - z * by, w * by - x * bz + y * bw + z * bx,
                      w * bz + x * by - y * bx + z * bw, w * bw - x * bx - y * by - z * bz)
        out[k] = x, y, z, w
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_recording(duration_s: float = 30.0, imu_rate_hz: float = 100.0, reference_rate_hz: float = 100.0,
                   segment_names=('upper', 'lower'), offset_s: float = 0.0, drift_ppm: float = 0.0,
                   mounting: Rotation | None = None, seed: int = 0, linear_acc_mps2: float = 0.0,
                   segment_drift_ppm: dict | None = None) -> Recording:
    """A clean recording, optionally with injected timing or mounting errors.

    offset_s: the IMU sample stamped t was taken at true time t - offset_s (sign convention of
        synchronisation_offset.py).
    drift_ppm: the IMU clock runs fast by this many parts per million: sample k stamped k / rate was taken at
        true time k / rate * (1 - drift_ppm * 1e-6) - offset_s.
    mounting: rotation from segment frame to sensor frame; the IMU measures in the rotated frame.
    linear_acc_mps2: peak of a linear acceleration shared by all segments (world frame); 0 = gravity only.
    segment_drift_ppm: extra drift of single IMUs, {segment: ppm}, about the middle of the recording (the IMUs run
        at slightly different rates; at mid-recording the extra offset is zero).
    """
    rng = np.random.default_rng(seed)
    fine_dt = 1 / 2000.0
    t_fine = np.arange(-1.0, duration_s + 1.0, fine_dt)  # one second of margin on both sides for shifted samples
    linear = np.zeros((len(t_fine), 3))
    if linear_acc_mps2:
        linear = smooth_rate(t_fine, np.random.default_rng(seed + 1000), peak_dps=1.0)
        linear *= linear_acc_mps2 / np.abs(linear).max()
    segments = {}
    for name in segment_names:
        w_fine = smooth_rate(t_fine, rng)
        quats = _integrate(Rotation.random(random_state=rng.integers(1 << 31)), w_fine, fine_dt)
        k_ref = np.round((np.arange(int(duration_s * reference_rate_hz)) / reference_rate_hz + 1.0) / fine_dt).astype(int)
        reference = ReferenceStream(Rotation.from_quat(quats[k_ref]), reference_rate_hz, convention='segment_to_world')
        stamps = np.arange(int(duration_s * imu_rate_hz)) / imu_rate_hz
        extra = (segment_drift_ppm or {}).get(name, 0.0) * 1e-6
        true_t = stamps * (1 - drift_ppm * 1e-6) - offset_s - extra * (stamps - duration_s / 2)
        gyr = np.column_stack([np.interp(true_t, t_fine, w_fine[:, i]) for i in range(3)])
        k_imu = np.round((true_t + 1.0) / fine_dt).astype(int)
        acc = Rotation.from_quat(quats[k_imu]).inv().apply(GRAVITY + linear[k_imu])
        if mounting is not None:
            gyr, acc = mounting.apply(gyr), mounting.apply(acc)
        segments[name] = Segment(name, ImuStream(gyr, acc, imu_rate_hz), reference)
    joints = list(zip(segment_names[:-1], segment_names[1:]))
    return Recording('synthetic', f'seed{seed}', segments, joints)
