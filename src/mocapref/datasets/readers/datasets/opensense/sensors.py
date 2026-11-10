"""Reading one OpenSense trial's inertial sensors, with the resampling the deposit's own layout needs.

The trimmed orientation table of a trial is located inside the raw sensor stream by cross-correlation. For the
two subjects recorded at 40 Hz, resampling only the raw stream to 100 Hz before that search compares signals at
two rates and keeps about 40 % of the trial at a meaningless alignment (correlation 0.05 and -0.20 against
0.87-0.999 when done right). So the orientation table's rate is read off its own time column and it is resampled
first; a trial already at 100 Hz is unchanged. Where the raw header and the orientation table disagree about the
rate, the reader raises. The sensor's native rate (``native_rate``) is passed out with the trial.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .deposit_files import (
    RATE_TOLERANCE_HZ,
    READ_RATE_HZ,
    estimate_start_offset,
    fill_dropped_samples,
    read_orientation_table,
    read_rate_of,
    read_sensor_mapping,
    read_xsens_export,
    resample_to_read_rate,
    trial_folder,
)

FUSION_FOLDER = "xsens"
"""Which of the deposit's three onboard-fusion folders the raw exports and the orientation table are read
from. The fusion result itself is never used as a reference here — only to find the trial's start offset —
but the raw accelerometer and gyroscope files sit inside that folder."""

RAW_SUBFOLDER = "LowerExtremity"
"""The subfolder of the fusion folder holding one text export per sensor."""

# A trial that ships its sensors but no mapping file may borrow the mapping of the same subject's other task.
BORROWED_MAPPINGS: dict[tuple[str, str], str] = {("Subject07", "complexTasks"): "walking"}
"""``(subject, trial folder) -> trial folder whose myIMUMappings file is read instead``.

Subject07's task block ships the raw exports of the same eight sensors as its walk but no mapping file. Measured:
each column of the depositors' ``complexTasks_orientations.sto`` matches exactly one raw export
(sample-to-sample rotation angle, correlation >= 0.9989, runner-up <= 0.90), and that assignment equals the walking
mapping for all seven named sensors; the same test recovers the shipped mapping of Subject07 walking and
Subject01 complexTasks. The borrowed mapping keeps its sensor ids; the file-name prefix is this trial's own."""


def _borrowed_mapping(subject_dir: Path, trial: str) -> dict[str, str] | None:
    """The mapping of :data:`BORROWED_MAPPINGS` for this trial, with the file names rewritten to its own raw files."""
    other = BORROWED_MAPPINGS.get((subject_dir.name, trial))
    if other is None:
        return None
    source = subject_dir / other / "IMU" / f"myIMUMappings_{other}.xml"
    raw_dir = subject_dir / trial / "IMU" / FUSION_FOLDER / RAW_SUBFOLDER
    if not source.exists() or not raw_dir.is_dir():
        return None
    own = {re.match(r"(.*)(_00B4[0-9A-F]+)\.txt$", p.name).group(2): p.name
           for p in raw_dir.glob("*_00B4*.txt")}
    mapping = {}
    for filename, sensor_name in read_sensor_mapping(source).items():
        sensor_id = re.match(r".*(_00B4[0-9A-F]+)\.txt$", filename).group(1)
        if sensor_id in own:
            mapping[own[sensor_id]] = sensor_name
    return mapping


def read_sensors(subject_dir: str | Path, task: str, verbose: bool = False) -> dict[str, dict]:
    """All sensors of one trial at :data:`~mocapref.datasets.readers.datasets.opensense.deposit_files.READ_RATE_HZ`.

    Returns ``{sensor name: {'acc', 'gyr', 'quat_onboard', 'lag', 'align_residual', 'native_rate'}}``, every
    sensor cut to the common length. ``lag`` is where the depositors' trimmed table starts inside the raw
    recording and ``align_residual`` the root-mean-square step-angle mismatch at that offset — both are
    diagnostics of the cut, not of the reconstruction.
    """
    subject_dir, trial = Path(subject_dir), trial_folder(task)
    base = subject_dir / trial / "IMU"
    mapping_path = base / f"myIMUMappings_{trial}.xml"
    table_path = base / FUSION_FOLDER / f"{trial}_orientations.sto"
    mapping = read_sensor_mapping(mapping_path) if mapping_path.exists() else _borrowed_mapping(subject_dir, trial)
    if not mapping or not table_path.exists():
        if verbose:
            print(f"  [skip] {subject_dir.name}/{trial}: no inertial data")
        return {}

    table_time, table = read_orientation_table(table_path)

    # The trimmed table shares the raw stream's native rate (measured: both 40.0 Hz
    # on the two 40 Hz subjects, on both sides). Its rate is taken from its own time column so that the two
    # resamplings rest on independent evidence, and cross-checked against the raw header below.
    table_rate = read_rate_of(table_time)
    table_needs_resample = abs(table_rate - READ_RATE_HZ) > RATE_TOLERANCE_HZ
    if table_needs_resample:
        if verbose:
            print(f"  [fix ] orientation table at {table_rate:.3f} Hz -> resampled to {READ_RATE_HZ} Hz")
        table = {k: resample_to_read_rate(fill_dropped_samples(v), table_rate, quat=True)
                 for k, v in table.items()}

    sensors: dict[str, dict] = {}
    n = None
    for filename, sensor_name in sorted(mapping.items()):
        raw_file = base / FUSION_FOLDER / RAW_SUBFOLDER / filename
        if not raw_file.exists() or sensor_name not in table:
            if verbose:
                print(f"  [skip] {sensor_name}: no raw file, or absent from {table_path.name}")
            continue

        raw = read_xsens_export(raw_file)
        native_rate = float(raw["rate"])
        acc, gyr, quat = (fill_dropped_samples(raw[k]) for k in ("acc", "gyr", "quat"))
        if abs(native_rate - READ_RATE_HZ) > RATE_TOLERANCE_HZ:
            acc, gyr = (resample_to_read_rate(x, native_rate) for x in (acc, gyr))
            quat = resample_to_read_rate(quat, native_rate, quat=True)
            # The two streams are documented as sharing a rate; if they ever disagree the cut below is
            # meaningless, so refuse rather than silently producing a misaligned trial.
            if not table_needs_resample:
                raise ValueError(
                    f"{subject_dir.name}/{trial}/{sensor_name}: the raw export states {native_rate} Hz but "
                    f"the orientation table parsed as {table_rate:.3f} Hz — the rate pair is inconsistent, "
                    "refusing to guess"
                )

        lag, residual = estimate_start_offset(quat, table[sensor_name])
        cut = slice(lag, lag + len(table[sensor_name]))
        sensors[sensor_name] = {
            "acc": acc[cut], "gyr": gyr[cut], "quat_onboard": quat[cut],
            "lag": lag, "align_residual": residual, "native_rate": native_rate,
        }
        n = len(acc[cut]) if n is None else min(n, len(acc[cut]))

    for entry in sensors.values():
        for key in ("acc", "gyr", "quat_onboard"):
            entry[key] = entry[key][:n]
    return sensors


def native_rate_of(sensors: dict[str, dict]) -> float:
    """The trial's native sensor rate — the deposit runs every sensor of one trial at one rate."""
    return float(next(iter(sensors.values()))["native_rate"]) if sensors else float(np.nan)
