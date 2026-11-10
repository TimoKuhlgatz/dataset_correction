"""Every loader reads its first trial into a consistent Recording with plausible units. Skipped where the data
are not on this machine."""
import numpy as np
import pytest

from mocapref.datasets import LOADERS, load_dataset

# Properties of the deposits (measured on the first trial), not loader faults: the check they would fail is skipped.
KNOWN = {
    'hip_rom_y': {'gravity'},  # the vendor removed gravity from the accelerometer (error 22)
    'openaxes': {'duration'},  # IMU streams run up to 3.6 s longer than the robot's record
    'repoimu': {'reference_coverage'},  # first trial is static and has no reference at all
}


@pytest.mark.parametrize('name', list(LOADERS))
def test_loader_reads_first_trial(name):
    try:
        loader = load_dataset(name)
    except FileNotFoundError:
        pytest.skip(f'{name} data not available')
    trials = loader.trials()
    assert trials
    rec = loader.load(trials[0])
    assert rec.dataset == name and rec.segments
    for joint in rec.joints:
        assert joint[0] in rec.segments and joint[1] in rec.segments
    for seg in rec.segments.values():
        imu, ref = seg.imu, seg.reference
        assert imu.gyr.shape == imu.acc.shape == (len(imu), 3)
        assert len(imu.valid) == len(imu) and len(ref.valid) == len(ref)
        known = KNOWN.get(name, set())
        if 'duration' not in known:
            assert abs(imu.duration_s - ref.duration_s) < 1.0  # both streams cover the same trial
        if 'gravity' not in known:
            assert 7.0 < np.median(np.linalg.norm(imu.acc[imu.valid], axis=1)) < 13.0  # m/s^2: gravity dominates
        assert np.percentile(np.linalg.norm(imu.gyr[imu.valid], axis=1), 99) < 50.0  # rad/s, not deg/s
        if 'reference_coverage' not in known:
            assert ref.valid.mean() > 0.5
