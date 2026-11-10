"""Known-answer tests for catching and restoring removed gravity (error 22) from the vendor quaternion: gravity
taken off a synthetic accelerometer with the sensor's true orientation comes back exactly, also after the mounting
stage has turned the accelerometer into the segment frame; a segment without a quaternion is left alone."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors.e22_gravity_removed import check_for_gravity_removed, restore_removed_gravity, vendor_up_in_sensor
from mocapref.synthetic import GRAVITY, make_recording

MOUNTING = Rotation.from_euler('xyz', [30, -50, 120], degrees=True)  # segment frame -> sensor frame
G = float(np.linalg.norm(GRAVITY))


def with_gravity_removed(turn=None):
    """A recording as Hip-ROM-Y's reader gives it: linear acceleration plus the vendor quaternion (here the true
    sensor-to-room orientation), and optionally already turned by a mounting stage (`turn`, 3x3)."""
    rec = make_recording(duration_s=20.0, seed=3, mounting=MOUNTING, linear_acc_mps2=2.0)
    truth = {}
    rec.metadata = {'segments': {}}
    for name, seg in rec.segments.items():
        n = len(seg.imu)
        sensor_to_room = seg.reference.orientation[:n] * MOUNTING.inv()  # same rate, same instants (no offset)
        q = sensor_to_room.as_quat()[:, [3, 0, 1, 2]]
        linear = seg.imu.acc - G * vendor_up_in_sensor(q)
        rec.metadata['segments'][name] = {'vendor_quaternion_wxyz': q, 'acc_as_read': linear.copy()}
        truth[name] = seg.imu.acc.copy()
        seg.imu.acc = linear if turn is None else linear @ turn.T
        if turn is not None:
            truth[name] = truth[name] @ turn.T
    return rec, truth


@pytest.mark.parametrize('turned', [False, True], ids=['as read', 'after the mounting stage'])
def test_gravity_comes_back(turned):
    turn = MOUNTING.inv().as_matrix() if turned else None
    rec, truth = with_gravity_removed(turn)
    assert np.median(np.linalg.norm(rec.segments['upper'].imu.acc, axis=1)) < 3  # gravity really gone
    out = restore_removed_gravity(rec, gravity_mps2=G)
    for name, seg in out.segments.items():
        np.testing.assert_allclose(seg.imu.acc, truth[name], atol=1e-9)
    assert sorted(out.corrections[-1].parameters['restored']) == ['lower', 'upper']


def test_segment_without_quaternion_or_with_changed_samples_is_left_alone():
    rec, _ = with_gravity_removed()
    del rec.metadata['segments']['lower']
    rec.metadata['segments']['upper']['acc_as_read'] = rec.metadata['segments']['upper']['acc_as_read'][:-5]
    before = {n: s.imu.acc.copy() for n, s in rec.segments.items()}
    out = restore_removed_gravity(rec, gravity_mps2=G)
    for n, s in out.segments.items():
        assert np.array_equal(s.imu.acc, before[n])
    assert out.corrections[-1].parameters['restored'] == []


def test_a_sensor_moved_to_another_segment_is_found_by_its_data_and_missing_quaternions_are_invalid():
    rec, truth = with_gravity_removed()
    up, lo = rec.segments['upper'].imu, rec.segments['lower'].imu
    up.acc, lo.acc = lo.acc, up.acc  # a later stage exchanged the two sensors (error 9)
    rec.metadata['segments']['upper']['vendor_quaternion_wxyz'][7] = 0.0
    out = restore_removed_gravity(rec, gravity_mps2=G)
    p = out.corrections[-1].parameters
    assert p['read_as'] == {'upper': 'lower', 'lower': 'upper'} and p['samples_without_quaternion'] == {'lower': 1}
    np.testing.assert_allclose(out.segments['upper'].imu.acc, truth['lower'], atol=1e-9)
    assert np.isfinite(out.segments['lower'].imu.acc).all() and not out.segments['lower'].imu.valid[7]


def resting(n_s=40.0):
    """Segments that rest for the first half (still reference, quiet IMU) and turn slowly afterwards."""
    rec = make_recording(duration_s=n_s, seed=5, mounting=MOUNTING)
    for seg in rec.segments.values():
        half = len(seg.imu) // 2
        q0 = seg.reference.orientation[0]
        seg.reference.orientation = Rotation.concatenate([q0] * half + list(seg.reference.orientation[half:]))
        seg.imu.gyr[:half] = 0.0
        seg.imu.acc[:half] = MOUNTING.apply(q0.inv().apply(GRAVITY))
    return rec


def test_removed_gravity_is_caught_at_rest_and_gravity_present_is_not():
    rec = resting()
    assert check_for_gravity_removed(rec).detected is False
    for seg in rec.segments.values():
        q = (seg.reference.orientation[:len(seg.imu)] * MOUNTING.inv()).as_quat()[:, [3, 0, 1, 2]]
        seg.imu.acc = seg.imu.acc - G * vendor_up_in_sensor(q)
    res = check_for_gravity_removed(rec)
    assert res.detected is True and all(p['detected'] for p in res.per_segment.values())
    assert all(p['estimate'] < 0.1 for p in res.per_segment.values())


def test_no_rest_is_not_judged():
    rec = make_recording(duration_s=20.0, seed=5, mounting=MOUNTING)
    assert check_for_gravity_removed(rec).detected is None


def test_handler_counts_and_restores_removed_gravity():
    """Dataset level: a resting trial whose gravity was taken off with the vendor quaternion is counted per sensor
    and gets gravity back from the corrected loader; a clean resting trial is not counted."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e22_gravity_removed as M
    clean = resting()
    bad = resting()
    q = bad.metadata = {'segments': {}}
    for name, seg in bad.segments.items():
        k = np.rint((seg.imu.times - seg.reference.start_s) * seg.reference.rate_hz).astype(int)
        sensor_to_room = seg.reference.orientation[np.clip(k, 0, len(seg.reference) - 1)] * MOUNTING.inv()
        quat = sensor_to_room.as_quat()[:, [3, 0, 1, 2]]
        seg.imu.acc = seg.imu.acc - 9.8 * sensor_to_room.inv().apply([0, 0, 1.0])
        q['segments'][name] = {'vendor_quaternion_wxyz': quat, 'acc_as_read': seg.imu.acc.copy()}
    loader = MemoryLoader({'clean': clean, 'removed': bad})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 1, 2)
    fixed = M.correct(loader, rows).load('removed')
    assert np.allclose(fixed['upper'].imu.acc, clean['upper'].imu.acc, atol=0.05)
