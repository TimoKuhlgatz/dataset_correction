"""The four deposited file formats this reconstruction reads, and the three signal repairs they need.

One trial of the OpenSense deposit is spread over four kinds of file:

* ``myIMUMappings_<trial>.xml`` — which raw sensor file belongs to which body. Sensors the depositors
  excluded are *commented out* in this file, so stripping the comments is what makes the exclusion effective.
* ``MT_..._<sensor id>.txt`` — the raw Xsens export of one sensor: accelerometer, gyroscope, and the onboard
  fusion's orientation written out as the nine elements of a rotation matrix. Its header states the sensor's
  own update rate, which is 100 Hz for nine of the eleven subjects and 40 Hz for two of them.
* ``<trial>_orientations.sto`` — the same onboard orientations after the depositors' own trimming, in
  OpenSim's quaternion table format. It is used for one thing only: to find where the trimmed trial starts
  inside the untrimmed raw recording.
* ``<trial>.trc`` — the marker file, read by :mod:`mocapref.datasets.readers.datasets.opensense.markers`.

The three repairs are: filling dropped Xsens packets, resampling a 40 Hz sensor onto the 100 Hz read rate, and
estimating the integer start offset between the raw stream and the trimmed one.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

READ_RATE_HZ = 100.0
"""The rate everything downstream of this module is served at. Nine of the eleven subjects were recorded at
it; the two 40 Hz subjects are resampled onto it here, and their true bandwidth travels separately as the
trial's native rate so that a reader can exclude them.

Units: Hz.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""

RATE_TOLERANCE_HZ = 1e-6
"""Two rates closer than this count as the same rate, so that a header that states ``100.0`` and a time column
that implies ``99.999999`` do not trigger a resampling.

Units: Hz.
Provenance: data-free — a floating-point tolerance on an equality of two rates, far below any rate difference
the deposit contains (the only difference in it is 100 Hz against 40 Hz)."""

ROTATION_MATRIX_INDICES = (1, 2, 3)
"""The one-based row and column indices the Xsens export writes into its orientation column names
(``Mat[1][1]`` ... ``Mat[3][3]``).

Units: matrix index, one-based.
Provenance: read-from-dataset — the vendor's own column naming."""

MIN_TIME_SAMPLES_FOR_RATE = 2
"""Fewest samples a time column must hold before a rate is read off it rather than assumed.

Units: samples.
Provenance: data-free — a minimum sample count: one difference needs two samples, and a median of differences
is only meaningful beyond that."""

TASK_FOLDERS = {"walk": "walking", "complex": "complexTasks"}
"""The two task names of the deposit, keyed by the prefix a caller may use."""


def trial_folder(task: str) -> str:
    """``'walking'`` / ``'complex'`` / ``'complexTasks'`` -> the folder name the deposit uses."""
    key = str(task).strip().lower()
    for prefix, folder in TASK_FOLDERS.items():
        if key.startswith(prefix):
            return folder
    raise ValueError(f"unknown task {task!r}, expected 'walking' or 'complex'")


def read_sensor_mapping(path: str | Path) -> dict[str, str]:
    """``myIMUMappings_<trial>.xml`` -> ``{raw file name: sensor name}``.

    Sensors the depositors excluded are commented out in this file, so the comments are stripped first and an
    excluded sensor simply never appears.
    """
    text = re.sub(r"<!--.*?-->", "", Path(path).read_text(errors="replace"), flags=re.DOTALL)
    prefix = re.search(r"<trial_prefix>([^<]+)</trial_prefix>", text).group(1)
    return {
        f"{prefix}{sensor_id}.txt": name.strip()
        for sensor_id, name in re.findall(
            r'<ExperimentalSensor name="([^"]+)">\s*<name_in_model>([^<]+)</name_in_model>', text
        )
    }


def read_xsens_export(path: str | Path) -> dict:
    """Raw Xsens export -> ``{'acc' [m/s²], 'gyr' [rad/s], 'quat', 'rate' [Hz]}``.

    All three signals are in the sensor frame. ``quat`` is the onboard fusion's own result, rebuilt from the
    rotation-matrix columns the file writes and left ``NaN`` where the radio packet was lost.
    """
    import pandas as pd
    from scipy.spatial.transform import Rotation

    skip, columns, rate = 0, None, READ_RATE_HZ
    with open(path) as handle:
        for i, line in enumerate(handle):
            stated = re.match(r"//\s*Update Rate:\s*([\d.]+)\s*Hz", line)
            if stated:
                rate = float(stated.group(1))
            if line.startswith("PacketCounter"):
                skip, columns = i + 1, line.strip().split("\t")
                break
    table = pd.read_csv(path, sep="\t", skiprows=skip, names=columns)

    out = {
        "acc": table[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(float),
        "gyr": table[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(float),
        "rate": rate,
    }
    matrix_columns = [f"Mat[{i}][{j}]" for j in ROTATION_MATRIX_INDICES for i in ROTATION_MATRIX_INDICES]
    quat = np.full((len(table), 4), np.nan)
    if all(c in table.columns for c in matrix_columns):
        flat = table[matrix_columns].to_numpy(float)
        matrices = np.stack([flat[:, 0:3], flat[:, 3:6], flat[:, 6:9]], axis=2)  # the file writes columns
        finite = np.isfinite(matrices).all(axis=(1, 2))
        quat[finite] = Rotation.from_matrix(matrices[finite]).as_quat()[:, [3, 0, 1, 2]]
    out["quat"] = quat
    return out


def read_orientation_table(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """OpenSim quaternion ``.sto`` -> ``(time [s], {column name: (T, 4) scalar-first quaternion})``."""
    import pandas as pd

    skip = 0
    with open(path) as handle:
        for i, line in enumerate(handle):
            if line.strip().lower() == "endheader":
                skip = i + 1
                break
    table = pd.read_csv(path, sep="\t", skiprows=skip)
    quaternions = {
        name: np.array([[float(x) for x in cell.split(",")] for cell in table[name]])
        for name in table.columns[1:]
    }
    return table[table.columns[0]].to_numpy(float), quaternions


def fill_dropped_samples(x: np.ndarray) -> np.ndarray:
    """Fill lost Xsens packets: spherical interpolation for quaternions, linear otherwise."""
    import qmt

    x = np.asarray(x, dtype=float)
    return qmt.nanInterp(x) if np.isnan(x).any() else x


def resample_to_read_rate(x: np.ndarray, rate: float, quat: bool = False) -> np.ndarray:
    """Resample a stream from ``rate`` onto :data:`READ_RATE_HZ`; a stream already at it is returned unchanged.

    The target grid starts at the source's own ``t = 0``, so there is no half-sample offset, and quaternions
    are interpolated spherically rather than linearly and renormalised.
    """
    import qmt

    if abs(rate - READ_RATE_HZ) < RATE_TOLERANCE_HZ:
        return x
    t_in = np.arange(len(x)) / rate
    t_out = np.arange(0.0, t_in[-1], 1.0 / READ_RATE_HZ)
    if quat:
        return qmt.quatInterp(x, np.interp(t_out, t_in, np.arange(len(x))))
    return np.stack([np.interp(t_out, t_in, x[:, k]) for k in range(x.shape[1])], axis=1)


def estimate_start_offset(q_raw: np.ndarray, q_trimmed: np.ndarray) -> tuple[int, float]:
    """Where sample 0 of the trimmed orientation table sits inside the untrimmed raw stream.

    Both series describe the same sensor up to one constant global rotation, so their frame-to-frame rotation
    angle is identical and can be cross-correlated. Returns the offset in samples and the residual root mean
    square of that angle at the best offset.
    """
    import qmt
    from scipy.signal import correlate

    def step_angle(q: np.ndarray) -> np.ndarray:
        return np.abs(qmt.quatAngle(qmt.qrel(q[:-1], q[1:])))

    raw, trimmed = step_angle(q_raw), step_angle(q_trimmed)
    if len(raw) < len(trimmed):
        raise ValueError("raw recording is shorter than the trimmed reference")
    correlation = correlate(raw - raw.mean(), trimmed - trimmed.mean(), mode="valid", method="fft")
    offset = int(np.argmax(correlation))
    return offset, float(np.sqrt(np.mean((raw[offset:offset + len(trimmed)] - trimmed) ** 2)))


def read_rate_of(time: np.ndarray) -> float:
    """The rate a time column implies, or :data:`READ_RATE_HZ` when it is too short to imply one."""
    if len(time) > MIN_TIME_SAMPLES_FOR_RATE:
        return float(1.0 / np.median(np.diff(time)))
    return READ_RATE_HZ
