"""Known-answer tests of the corrupted-sample check (error 15): injected corrupted samples are found and
marked, nothing is found on clean noisy recordings, a fast motion the reference also shows is kept, a downward
needle is reported but not marked, a candidate where the reference is missing is not examined, and an accelerometer
candidate counts only where the gyroscope of the same IMU is corrupted. The check runs after the time alignment, so
the recordings here are aligned."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref import errors as E
from mocapref.errors.e15_corrupted_samples import needle_runs, reference_speed_at
from mocapref.synthetic import make_recording


@pytest.fixture(scope='module')
def noisy():
    rec = make_recording(duration_s=60.0, segment_names=('pelvis', 'thigh', 'shank'), seed=5)
    rng = np.random.default_rng(1)
    for seg in rec.segments.values():
        seg.imu.gyr = seg.imu.gyr + rng.normal(0, np.radians(1.0), seg.imu.gyr.shape) + np.radians([2.0, -1.0, 0.5])
        seg.imu.acc = seg.imu.acc + rng.normal(0, 0.05, seg.imu.acc.shape)
    return rec


def test_clean_noisy_recording_has_no_corrupted_samples(noisy):
    result = E.check_for_corrupted_samples(noisy)
    assert result.detected is False
    assert all(p['estimate'] == 0 for p in result.per_segment.values())


def test_single_spike_and_tuk_like_pair_are_found_and_marked(noisy):
    rec = noisy.copy()
    rec['thigh'].imu.gyr[1000] = np.radians([400.0, 5000.0, -10.0])  # one sample
    rec['shank'].imu.gyr[2000] = np.radians([445.0, 5169.0, -12.0])  # two consecutive samples as in TUK
    rec['shank'].imu.gyr[2001] = np.radians([-461.0, -5117.0, -12.0])
    result = E.check_for_corrupted_samples(rec)
    assert result.detected is True
    assert result.per_segment['pelvis']['detected'] is False
    assert result.per_segment['thigh']['estimate'] == 1
    assert result.per_segment['shank']['estimate'] == 2
    assert [r['length'] for r in result.per_segment['shank']['gyro_runs'] if r['corrupted']] == [2]
    fixed = E.remove_corrupted_samples(rec)
    assert not fixed['thigh'].imu.valid[1000] and fixed['thigh'].imu.valid.sum() == len(rec['thigh'].imu) - 1
    assert not fixed['shank'].imu.valid[2000:2002].any()
    assert fixed.corrections[-1].error == 'corrupted samples'
    assert E.check_for_corrupted_samples(fixed).detected is False


def test_a_spike_the_reference_also_shows_is_kept():
    """The reference turns by 0.1 rad in one frame and straight back (10 rad/s for one step) and the gyroscope
    shows that speed at that moment, on top of 1 deg/s sensor noise: a spike shape in the gyroscope, but real
    motion, so not corrupted."""
    rec = make_recording(duration_s=30.0, segment_names=('upper',), seed=2)
    seg = rec['upper']
    seg.imu.gyr = seg.imu.gyr + np.random.default_rng(0).normal(0, np.radians(1.0), seg.imu.gyr.shape)
    i = 1500
    kick = Rotation.from_rotvec([0.0, 0.1, 0.0])
    q = seg.reference.orientation
    seg.reference.orientation = Rotation.concatenate([q[:i + 1], q[i + 1:i + 2] * kick, q[i + 2:]])
    seg.imu.gyr[i + 1] = [0.0, reference_speed_at(seg)[i + 1], 0.0]
    assert any(r['first'] == i + 1 for r in needle_runs(seg.imu.gyr, seg.imu.valid))  # a needle by shape
    result = E.check_for_corrupted_samples(rec)
    p = result.per_segment['upper']
    assert p['gyro_corrupted'] == 0 and p['estimate'] == 0
    assert result.detected is False


def test_downward_needle_is_reported_not_marked(noisy):
    """One gyroscope sample drops to zero at the fastest moment of the motion, the reference keeps moving: a
    downward needle meeting all three criteria. It is reported, not counted and not marked."""
    rec = noisy.copy()
    seg = rec['shank']
    i = int(np.argmax(np.linalg.norm(seg.imu.gyr[100:-100], axis=1))) + 100
    seg.imu.gyr[i] = 0.0
    p = E.check_for_corrupted_samples(rec).per_segment['shank']
    assert p['down_reported'] == 1 and p['estimate'] == 0 and p['detected'] is False
    assert [r['first'] for r in p['gyro_runs'] if r['direction'] == 'down'] == [i]
    assert E.remove_corrupted_samples(rec)['shank'].imu.valid[i]


def test_candidate_without_reference_is_not_examined(noisy):
    rec = noisy.copy()
    rec['thigh'].imu.gyr[1000] = np.radians([0.0, 5000.0, 0.0])
    rec['thigh'].reference.valid[990:1010] = False  # marker gap around the spike
    result = E.check_for_corrupted_samples(rec)
    p = result.per_segment['thigh']
    assert p['detected'] is None and p['gyro_not_examined'] == 1 and p['estimate'] == 0
    assert result.detected is None
    assert E.remove_corrupted_samples(rec)['thigh'].imu.valid[1000]


def test_accelerometer_candidate_counts_only_with_the_gyroscope(noisy):
    rec = noisy.copy()
    imu = rec['pelvis'].imu
    imu.acc[700] = imu.acc[700] + [0.0, 300.0, 0.0]  # accelerometer alone: reported, not marked
    imu.acc[1200] = imu.acc[1200] + [0.0, -300.0, 0.0]  # same sample as a gyroscope spike: confirmed
    imu.gyr[1200] = np.radians([0.0, 0.0, 5000.0])
    p = E.check_for_corrupted_samples(rec).per_segment['pelvis']
    assert p['acc_candidates'] == 2 and p['acc_confirmed'] == 1 and p['acc_unconfirmed'] == 1
    fixed = E.remove_corrupted_samples(rec)['pelvis'].imu
    assert fixed.valid[700] and not fixed.valid[1200]


def test_check_follows_the_time_alignment():
    """A spike judged against a reference 0.5 s off: the reference is examined at the wrong moment, which is why the
    check comes after the time alignment. With the offset removed the spike is confirmed."""
    rec = make_recording(duration_s=60.0, segment_names=('upper', 'lower'), seed=7, offset_s=0.5)
    rec['lower'].imu.gyr[3000] = np.radians([0.0, 5000.0, 0.0])
    aligned = E.remove_synchronisation_offset(rec, E.check_for_synchronisation_offset(rec))
    p = E.check_for_corrupted_samples(aligned).per_segment['lower']
    assert p['gyro_corrupted'] == 1 and p['estimate'] == 1


def test_impact_felt_by_another_sensor_is_not_marked(noisy):
    """An impact: the shank and the thigh both jump to several hundred deg/s at the same sample while the reference
    stays smooth. Each alone passes criteria 1-3; because another sensor also scores high, neither is marked. The
    same jump on one sensor alone is marked."""
    rec = noisy.copy()
    rec['shank'].imu.gyr[1500] = np.radians([0.0, 900.0, 0.0])
    rec['thigh'].imu.gyr[1501] = np.radians([600.0, 0.0, 0.0])  # one sample later: within the window
    result = E.check_for_corrupted_samples(rec)
    for name in ('shank', 'thigh'):
        p = result.per_segment[name]
        assert p['gyro_impacts'] == 1 and p['estimate'] == 0
    assert result.detected is False
    assert E.remove_corrupted_samples(rec)['shank'].imu.valid[1500]
    alone = noisy.copy()
    alone['shank'].imu.gyr[1500] = np.radians([0.0, 900.0, 0.0])
    assert E.check_for_corrupted_samples(alone).per_segment['shank']['estimate'] == 1


def test_handler_counts_runs_and_masks_them(noisy):
    """Dataset level: one trial with a single spike and a TUK-like pair, one clean; two runs counted, three samples
    marked invalid by the corrected loader."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e15_corrupted_samples as M
    rec = noisy.copy()
    rec['thigh'].imu.gyr[1000] = np.radians([400.0, 5000.0, -10.0])
    rec['shank'].imu.gyr[2000] = np.radians([445.0, 5169.0, -12.0])
    rec['shank'].imu.gyr[2001] = np.radians([-461.0, -5117.0, -12.0])
    loader = MemoryLoader({'clean': noisy, 'spiky': rec})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 1, 2)
    fixed = M.correct(loader, rows).load('spiky')
    assert not fixed['thigh'].imu.valid[1000] and not fixed['shank'].imu.valid[2000:2002].any()
    assert sum((~s.imu.valid).sum() for s in fixed.segments.values()) == 3
    assert M.check_for_corrupted_samples(fixed).detected is False
