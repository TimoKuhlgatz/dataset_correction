"""Known-answer tests for the rotation-free cross-check of error 20 (the ω·a reading): on a
synthetic segment with a known path, a sensor off the marker centroid and an unknown mounting, the slope is +1 for a
correct accelerometer and -1 for a negated one, from markers and from a fixed pivot."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors.accelerometer_cross_check import pooled_slope, trial_terms
from mocapref.synthetic import make_recording

MOUNTING = Rotation.from_euler('xyz', [30, -50, 120], degrees=True)  # segment frame -> sensor frame
UP = np.array([0.0, 0.0, 1.0])


def segment_with_path(seed, moving_point=True, negate=False):
    rec = make_recording(duration_s=30.0, imu_rate_hz=200.0, reference_rate_hz=200.0, seed=seed,
                         mounting=MOUNTING, segment_names=('seg',))
    seg = rec.segments['seg']
    R = seg.reference.orientation
    t = np.arange(len(R)) / 200.0
    p = (np.column_stack([0.3 * np.sin(1.3 * t), 0.2 * np.sin(0.7 * t + 1), 0.1 * np.sin(2 * t)])
         if moving_point else np.zeros((len(t), 3)))
    sensor = p + R.apply([0.05, 0.02, 0.10])  # the sensor sits 12 cm from the point p
    sdd = np.gradient(np.gradient(sensor, t, axis=0), t, axis=0)
    seg.imu.acc = MOUNTING.apply(R.inv().apply(sdd + 9.81 * UP)) * (-1 if negate else 1)
    seg.reference.markers = {f'm{i}': 1000 * (p + R.apply(c)) for i, c in
                             enumerate([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])}
    return rec


@pytest.mark.parametrize('negate, expected', [(False, 1.0), (True, -1.0)])
@pytest.mark.parametrize('mode', ['markers', 'fixed point'])
def test_slope_is_plus_one_or_minus_one(mode, negate, expected):
    parts = []
    for seed in range(3):
        rec = segment_with_path(seed, moving_point=mode == 'markers', negate=negate)
        got = trial_terms(rec, 'seg', UP, 'segment_to_world', chain=None if mode == 'markers' else [])
        assert got is not None and got[2] > 0.9
        parts.append(got[:2])
    res = pooled_slope(parts, 200.0)
    assert res['slope'] == pytest.approx(expected, abs=0.1)
    assert res['negated'] is negate
