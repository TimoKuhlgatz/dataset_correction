"""Put the IMU and the reference of every segment on one common time grid.

After timing corrections (time offset, clock drift) the IMU time stamps no longer coincide with the reference
frames, and the two streams may run at different rates. Estimators such as the MEKF expect IMU sample k and
reference frame k to belong to the same instant. This step creates such a recording:

- grid: from the latest start to the earliest end of all streams, at `rate_hz` (default: the IMU rate);
- IMU: linear interpolation of gyroscope and accelerometer between the two neighbouring samples;
- reference: spherical interpolation (slerp) between the two neighbouring frames;
- validity: a grid sample is valid only if both neighbours it was interpolated from are valid.

Resampling smooths the signal slightly; it is logged as a correction like every other change.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .recording import ImuStream, Recording, ReferenceStream


def _neighbours(times: np.ndarray, start_s: float, rate_hz: float, n: int):
    """Index of the sample before each time, and the fraction of the way to the next one."""
    position = (times - start_s) * rate_hz
    k = np.clip(np.floor(position).astype(int), 0, n - 2)
    return k, position - k


def resample_imu(imu: ImuStream, times: np.ndarray, rate_hz: float) -> ImuStream:
    k, frac = _neighbours(times, imu.start_s, imu.rate_hz, len(imu))
    f = frac[:, None]
    gyr = (1 - f) * imu.gyr[k] + f * imu.gyr[k + 1]
    acc = (1 - f) * imu.acc[k] + f * imu.acc[k + 1]
    return ImuStream(gyr, acc, rate_hz, start_s=0.0, valid=imu.valid[k] & imu.valid[k + 1])


def resample_reference(ref: ReferenceStream, times: np.ndarray, rate_hz: float) -> ReferenceStream:
    k, frac = _neighbours(times, ref.start_s, ref.rate_hz, len(ref))
    before, after = ref.orientation[k], ref.orientation[k + 1]
    step = (before.inv() * after).as_rotvec()
    orientation = before * Rotation.from_rotvec(frac[:, None] * step)
    return ReferenceStream(orientation, rate_hz, start_s=0.0, valid=ref.valid[k] & ref.valid[k + 1],
                           convention=ref.convention)  # raw markers are not carried over to the grid


def resample_to_common_grid(recording: Recording, rate_hz: float | None = None) -> Recording:
    """Return a copy with every IMU and reference on one grid starting at 0 s, at `rate_hz` (default: the IMU rate
    of the first segment). Time 0 of the new grid is the latest start of all streams."""
    streams = [s for seg in recording.segments.values() for s in (seg.imu, seg.reference)]
    if rate_hz is None:
        rate_hz = next(iter(recording.segments.values())).imu.rate_hz
    start = max(s.start_s for s in streams)
    end = min(s.start_s + (len(s) - 1) / s.rate_hz for s in streams)
    times = start + np.arange(int(np.floor((end - start) * rate_hz)) + 1) / rate_hz
    new = recording.copy()
    for seg in new.segments.values():
        seg.imu = resample_imu(seg.imu, times, rate_hz)
        seg.reference = resample_reference(seg.reference, times, rate_hz)
    new.log_correction('resampling', f'Resampled IMU and reference onto one grid at {rate_hz:g} Hz, '
                                     f'{len(times)} samples from {start:.4f} s to {times[-1]:.4f} s of the old time base.',
                       rate_hz=rate_hz, grid_start_s=float(start), samples=len(times))
    return new
