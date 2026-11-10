"""Known-answer tests for the accelerometer bias correction (error 21):
a known bias is recovered from rest poses spread over the sphere, poses in one plane or too few are not judged, a
gravity-removed sensor is skipped, and the bias is subtracted in the frame the accelerometer has been turned into."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors import e21_accelerometer_bias as AB
from mocapref.synthetic import make_recording

BIAS = np.array([0.15, -0.30, 0.90])  # m/s², about IS2B's tibia


def _poses(n, rng, plane=False):
    d = rng.standard_normal((n, 3))
    if plane:
        d[:, 2] = 0.0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return AB.G * d


def test_known_bias_is_recovered_from_spread_poses():
    rng = np.random.default_rng(1)
    readings = _poses(30, rng) + BIAS + 0.01 * rng.standard_normal((30, 3))
    out = AB.judge_unit(readings)
    assert out['verdict'] == AB.READ
    assert np.allclose([out['bias_x'], out['bias_y'], out['bias_z']], BIAS, atol=0.02)


def test_poses_in_one_plane_or_too_few_are_not_judged():
    rng = np.random.default_rng(2)
    assert AB.judge_unit(_poses(30, rng, plane=True) + BIAS)['verdict'] == 'poses in one plane'
    assert AB.judge_unit(_poses(5, rng) + BIAS)['verdict'] == 'too few poses'


def test_small_bias_is_left():
    rng = np.random.default_rng(3)
    assert AB.judge_unit(_poses(30, rng) + 0.05)['verdict'] == 'bias below 0.2 m/s²'


def test_gravity_removed_unit_is_skipped():
    """Rest readings near zero (error 22): a sphere of radius g through the origin would read a bias of g."""
    rng = np.random.default_rng(4)
    assert AB.judge_unit(0.2 * rng.standard_normal((40, 3)))['verdict'] == 'gravity removed (error 22)'


@pytest.mark.parametrize('negate', [False, True])
def test_bias_is_subtracted_in_the_turned_frame(negate):
    """The bias lives in the sensor frame; the stored accelerometer has since been turned (mounting) and maybe
    negated (error 20). Subtracting the turned bias gives exactly the turned clean accelerometer."""
    rec = make_recording(duration_s=10.0)
    seg = rec.segments['upper']
    clean = seg.imu.acc.copy()
    raw = clean + BIAS
    turn = Rotation.from_euler('xyz', [30, -20, 75], degrees=True).as_matrix() * (-1 if negate else 1)
    seg.imu.acc = raw @ turn.T
    fixed = AB.remove_accelerometer_bias(rec, {'upper': BIAS}, {'upper': raw})
    assert np.allclose(fixed.segments['upper'].imu.acc, clean @ turn.T, atol=1e-9)
    assert fixed.corrections[-1].parameters['bias_mps2'] == {'upper': pytest.approx(np.linalg.norm(BIAS), abs=1e-3)}


def test_changed_accelerometer_is_not_touched():
    rec = make_recording(duration_s=10.0)
    seg = rec.segments['upper']
    before = seg.imu.acc.copy()
    other = before + np.random.default_rng(5).standard_normal(before.shape)  # not a turned copy
    fixed = AB.remove_accelerometer_bias(rec, {'upper': BIAS}, {'upper': other})
    assert np.allclose(fixed.segments['upper'].imu.acc, before)
    assert 'upper' in fixed.corrections[-1].parameters['skipped']


def test_handler_reads_the_bias_below_the_mounting_and_subtracts_it_turned():
    """Dataset level: a sensor with a known bias in its own frame rests in many poses over two trials; the check reads
    through a chain whose mounting correction turned the accelerometer, finds the bias in the sensor frame, and the
    corrected loader subtracts it in the turned frame."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e21_accelerometer_bias as M
    from mocapref.errors._report import CorrectedLoader
    rng = np.random.default_rng(3)
    bias = np.array([0.6, -0.4, 0.3])
    recs = {}
    for k in range(2):
        rec = make_recording(duration_s=10.0, segment_names=('upper',), seed=k)
        imu = rec['upper'].imu
        poses = Rotation.random(20, random_state=rng.integers(1 << 31)).apply([0, 0, 9.80665])
        n = len(imu) // 20
        acc = np.repeat(poses, n, axis=0)[:len(imu)]
        acc = np.vstack([acc, np.repeat(poses[-1:], len(imu) - len(acc), axis=0)])
        imu.acc, imu.gyr = acc + bias, np.zeros_like(imu.gyr)
        ref = rec['upper'].reference
        ref.orientation = Rotation.from_quat(np.repeat(ref.orientation[:1].as_quat(), len(ref), axis=0))
        recs[f't{k}'] = rec
    turn = Rotation.from_euler('xyz', [30, -20, 100], degrees=True).as_matrix()

    def rotate(rec, trial):
        new = rec.copy()
        new['upper'].imu.acc = -(new['upper'].imu.acc @ turn.T)  # mounting turned it, the sign stage negated it
        return new
    base = MemoryLoader(recs)
    chain = CorrectedLoader(CorrectedLoader(base, rotate, 'mounting'), lambda r, t: r, 'negated accelerometer')
    rows = M.check_dataset(chain, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 2, 1)
    assert np.allclose(rows[['bias_x', 'bias_y', 'bias_z']].iloc[0], bias, atol=0.02)
    fixed = M.correct(chain, rows).load('t0')['upper'].imu.acc
    assert np.allclose(fixed, -((recs['t0']['upper'].imu.acc - bias) @ turn.T), atol=0.02)
