"""Known-answer tests for error 16, the mirrored gyroscope: a gyroscope stored with one axis or all
three negated is found by both readings, the accelerometer picks the axis that was flipped (also with the
accelerometer itself negated and with a gyroscope offset left in), the correction restores the original data, a
clean or remounted sensor is not flagged, and motion about one axis only stays undecided."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors.e16_mirrored_gyroscope import check_for_mirrored_gyroscope, choose_flip_by_slope, remove_mirrored_gyroscope
from mocapref.recording import ImuStream, Recording, ReferenceStream, Segment
from mocapref.synthetic import GRAVITY, _integrate, make_recording

MOUNTING = Rotation.from_euler('xyz', [30, -50, 120], degrees=True)


def with_markers(rec):
    """Three markers per segment about its fixed origin, so the cross-check that picks the axis can read it."""
    for seg in rec.segments.values():
        seg.reference.markers = {f'm{i}': 1000 * seg.reference.orientation.apply(c)
                                 for i, c in enumerate([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])}
    return rec


def trials(n=2, **kw):
    return [with_markers(make_recording(duration_s=40.0, seed=s, mounting=MOUNTING, **kw)) for s in range(n)]


def flipped(recs, axes, negate_acc=False, offset_dps=0.0):
    out = []
    for r in recs:
        r = r.copy()
        imu = r.segments['lower'].imu
        imu.gyr[:, axes] *= -1
        imu.gyr += np.radians(offset_dps)
        if negate_acc:
            imu.acc = -imu.acc
        out.append(r)
    return out


def test_clean_sensors_are_right_handed():
    res = check_for_mirrored_gyroscope(trials())
    assert res.detected is False
    for p in res.per_segment.values():
        assert p['detected'] is False and len(p['sessions']) == 1
        assert p['sessions'][0]['reading']['volume_agreement'] > 0.95


@pytest.mark.parametrize('axes, name', [([0], 'x'), ([1], 'y'), ([2], 'z'), ([0, 1, 2], 'all three')])
@pytest.mark.parametrize('negate_acc', [False, True])
def test_flipped_axis_is_found_and_restored(axes, name, negate_acc):
    clean = trials()
    recs = flipped(clean, axes, negate_acc=negate_acc, offset_dps=15.0)
    res = check_for_mirrored_gyroscope(recs)
    lower, upper = res.per_segment['lower'], res.per_segment['upper']
    assert lower['detected'] is True and upper['detected'] is False
    session = lower['sessions'][0]
    assert session['flip'] == name and session['axis_decided'] is True
    slopes = session['cross_slopes']
    assert slopes[name]['slope'] == pytest.approx(-1.0 if negate_acc else 1.0, abs=0.15)
    assert all(abs(v['slope']) < 0.9 for k, v in slopes.items() if k != name)  # 'all three' reaches 0.84
    for rec, orig in zip(recs, clean):
        fixed = remove_mirrored_gyroscope(rec, res)
        expected = orig.segments['lower'].imu.gyr + np.radians(15.0) * (np.ones(3) * np.where(
            np.isin(np.arange(3), axes), -1, 1))  # the offset was added after the flip, so it is flipped back too
        assert np.allclose(fixed.segments['lower'].imu.gyr, expected)
        assert np.array_equal(fixed.segments['upper'].imu.gyr, rec.segments['upper'].imu.gyr)
    assert check_for_mirrored_gyroscope([remove_mirrored_gyroscope(r, res) for r in recs]).detected is False


def test_remounted_sensor_forms_its_own_session():
    a = with_markers(make_recording(duration_s=40.0, seed=0, mounting=MOUNTING))
    b = with_markers(make_recording(duration_s=40.0, seed=1,
                                    mounting=Rotation.from_euler('z', 90, degrees=True) * MOUNTING))
    res = check_for_mirrored_gyroscope(flipped([a, b], [1]))
    sessions = res.per_segment['lower']['sessions']
    assert [s['trials'] for s in sessions] == [[a.trial], [b.trial]]
    assert all(s['mirrored'] for s in sessions)
    # one trial each: a decided axis must be y; a single trial may leave it open (all three flipped), as seed 0 does,
    # where 'all three' reads -0.95 against +1.00 for y
    assert all(s['flip'] == 'y' if s['axis_decided'] else s['flip'] == 'all three' for s in sessions)
    assert any(s['axis_decided'] for s in sessions)
    assert [s['split_off'] for s in sessions] == [False, True]  # equal size: the first is the main group


def test_motion_about_one_axis_is_undecided():
    rate_hz, t = 100.0, np.arange(0, 40.0, 0.01)
    w = np.zeros((len(t), 3))
    w[:, 2] = np.radians(150) * np.sin(2 * np.pi * 0.7 * t)  # a hinge
    q = Rotation.from_quat(_integrate(Rotation.identity(), w, 1 / rate_hz))
    gyr = MOUNTING.apply(w)
    gyr[:, 1] *= -1
    acc = MOUNTING.apply(q.inv().apply(GRAVITY))
    seg = Segment('hinge', ImuStream(gyr, acc, rate_hz), ReferenceStream(q, rate_hz, convention='segment_to_world'))
    res = check_for_mirrored_gyroscope(Recording('synthetic', 'hinge', {'hinge': seg}, []))
    assert res.detected is None
    assert res.per_segment['hinge']['detected'] is None


def test_axis_is_open_unless_one_slope_stands_out():
    s = lambda v, lo, hi: dict(slope=v, low=lo, high=hi)
    # a real case: y only +1.03 against x -0.65, z +0.38, all three -0.77
    assert choose_flip_by_slope({'x': s(-0.65, -0.7, -0.6), 'y': s(1.03, 0.95, 1.1), 'z': s(0.38, 0.3, 0.45),
                                 'all three': s(-0.77, -0.86, -0.68)}) == ('y', True)
    # two candidates equally far from zero (P09, one trial): open
    assert choose_flip_by_slope({'x': s(0.55, 0.2, 0.9), 'y': s(-1.65, -2.2, -1.1), 'z': s(1.63, 1.1, 2.2),
                                 'all three': s(-0.52, -0.8, -0.1)}) == ('all three', False)
    nan = float('nan')
    assert choose_flip_by_slope({k: s(nan, nan, nan) for k in ('x', 'y', 'z', 'all three')}) == ('all three', False)


def test_handler_counts_the_mirrored_session_and_flips_it_back():
    """Dataset level: participant P1's lower gyroscope is mirrored in both trials, P2 is clean; P1's two
    trial-segments count, and the corrected loader returns P1's original gyroscope."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e16_mirrored_gyroscope as M
    clean = trials(2)
    bad = flipped(trials(2), [1])
    recs = {'p1a': bad[0], 'p1b': bad[1], 'p2a': clean[0], 'p2b': clean[1]}
    loader = MemoryLoader(recs, participant={'p1a': 'P1', 'p1b': 'P1', 'p2a': 'P2', 'p2b': 'P2'})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (4, 2, 2)
    fixed = M.correct(loader, rows)
    assert np.allclose(fixed.load('p1a')['lower'].imu.gyr, clean[0]['lower'].imu.gyr)
    assert np.allclose(fixed.load('p2a')['lower'].imu.gyr, clean[0]['lower'].imu.gyr)
