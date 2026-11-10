"""Known-answer tests for error 20, the negated accelerometer: the trial's vertical comes from its
own units (majority of the units on a common axis), so a negated unit is found whatever axis the optical system calls up
and also in an upside-down reference; it is restored by the correction; clean units pass; removed gravity is
unreadable; a single unit falls back to +z and is reported."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors.e19_e20_accelerometer_sign import (check_for_negated_accelerometer, name_vertical,
                                               remove_negated_accelerometer, trial_vertical)
from mocapref.synthetic import make_recording

MOUNTING = Rotation.from_euler('xyz', [30, -50, 120], degrees=True)
THREE = ('a', 'b', 'c')


def trials(n=3, segments=THREE, world=None):
    """`world`: a constant rotation of the optical coordinate system (the room is the same, its axes are not). Each
    segment rotates about a fixed point and carries three markers, so the cross-check can read it too."""
    out = []
    for s in range(n):
        r = make_recording(duration_s=40.0, seed=s, mounting=MOUNTING, segment_names=segments)
        for seg in r.segments.values():
            if world is not None:
                seg.reference.orientation = world * seg.reference.orientation
            seg.reference.markers = {f'm{i}': 1000 * seg.reference.orientation.apply(c)
                                     for i, c in enumerate([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])}
        out.append(r)
    return out


def negate(recs, segment='b'):
    out = []
    for r in recs:
        r = r.copy()
        r.segments[segment].imu.acc = -r.segments[segment].imu.acc
        out.append(r)
    return out


def test_trial_vertical_by_majority():
    g = np.array([0.0, 9.81, 0.0])
    tv = trial_vertical({'a': g, 'b': -g, 'c': g * 1.01 + [0.3, 0, 0]}, 30.0)
    assert tv['source'] == 'sensors' and np.allclose(tv['up'], [0, 1, 0], atol=0.05) and tv['votes'] == (2, 1)
    assert tv['units']['b']['negated'] is True and tv['units']['a']['negated'] is False
    tilted = trial_vertical({'a': g, 'b': g, 'c': np.array([9.81, 0, 0])}, 30.0)
    assert tilted['units']['c']['negated'] is None and tilted['units']['c']['loo_angle_deg'] == pytest.approx(90)


def test_clean_units_pass():
    res = check_for_negated_accelerometer(trials())
    assert res.detected is False and res.thresholds['vertical_assumed'] == []


@pytest.mark.parametrize('world', [None, Rotation.from_euler('x', -90, degrees=True),  # +y up
                                   Rotation.from_euler('x', 180, degrees=True)],  # upside down
                         ids=['z up', 'y up', 'z down'])
def test_negated_unit_is_found_whatever_axis_is_up(world):
    clean = trials(world=world)
    recs = negate(clean)
    res = check_for_negated_accelerometer(recs)
    assert res.per_segment['b']['detected'] is True
    assert res.per_segment['a']['detected'] is False and res.per_segment['c']['detected'] is False
    assert res.thresholds['vertical_assumed'] == [] and res.thresholds['sign_assumed'] == []
    fixed = [remove_negated_accelerometer(r, res) for r in recs]
    for f, c in zip(fixed, clean):
        for n in THREE:
            assert np.array_equal(f.segments[n].imu.acc, c.segments[n].imu.acc)
    assert check_for_negated_accelerometer(fixed).detected is False


def test_two_units_disagreeing_take_the_sign_from_plus_z_and_say_so():
    recs = negate(trials(segments=('a', 'b')))
    res = check_for_negated_accelerometer(recs)
    assert res.per_segment['b']['detected'] is True and res.per_segment['a']['detected'] is False
    assert sorted(res.thresholds['sign_assumed']) == sorted(r.trial for r in recs)


def test_gravity_removed_is_unreadable():
    recs = trials()
    for r in recs:
        for seg in r.segments.values():
            g_seg = seg.reference.orientation.inv().apply(np.array([0, 0, 9.81]))
            seg.imu.acc = seg.imu.acc - MOUNTING.apply(g_seg)
    res = check_for_negated_accelerometer(recs)
    assert res.detected is None
    assert sorted(res.thresholds['vertical_assumed']) == sorted(r.trial for r in recs)


def test_single_unit_falls_back_to_plus_z_and_is_reported():
    recs = negate(trials(segments=('b',)))
    res = check_for_negated_accelerometer(recs)
    assert res.per_segment['b']['detected'] is True
    assert sorted(res.thresholds['vertical_assumed']) == sorted(r.trial for r in recs)
    turned = negate(trials(segments=('b',), world=Rotation.from_euler('x', -90, degrees=True)))
    assert check_for_negated_accelerometer(turned).per_segment['b']['detected'] is None  # y up: nothing near ±z


def test_all_units_negated_alike_leaves_the_relative_orientation_unchanged():
    """The case the vote cannot see is harmless: the MEKF's relative-orientation error is the same when every
    accelerometer of the trial is negated (equal to 1e-11 degrees), about 180° when one is."""
    from mocapref.effect import test_on_trial as score
    rec = make_recording(duration_s=40.0, seed=0, linear_acc_mps2=2.0)
    both, one = rec.copy(), rec.copy()
    for name in ('upper', 'lower'):
        both.segments[name].imu.acc = -both.segments[name].imu.acc
    one.segments['lower'].imu.acc = -one.segments['lower'].imu.acc
    clean = score(rec, 1e-2, 1e-2, skip_s=10.0)['test_error_deg']
    assert score(both, 1e-2, 1e-2, skip_s=10.0)['test_error_deg'] == pytest.approx(clean, abs=1e-6)
    assert score(one, 1e-2, 1e-2, skip_s=10.0)['test_error_deg'] > 90


def test_vertical_axis_is_named_below_30_degrees_and_suspicious_beyond():
    assert name_vertical(np.array([0.0, 1.0, 0.0])) == ('+y', 'named', 0.0)
    tilt = lambda deg: Rotation.from_euler('x', deg, degrees=True).apply([0.0, 0.0, 1.0])
    axis, status, angle = name_vertical(tilt(-20))
    assert (axis, status) == ('+z', 'named') and angle == pytest.approx(20)
    axis, status, angle = name_vertical(tilt(-40))
    assert (axis, status) == (None, 'suspicious') and angle == pytest.approx(40)
    res = check_for_negated_accelerometer(trials(world=Rotation.from_euler('x', -90, degrees=True)))
    assert {v['axis'] for v in res.thresholds['trial_verticals'].values()} == {'+y'}
    tilted = check_for_negated_accelerometer(negate(trials(world=Rotation.from_euler('x', 45, degrees=True))))
    assert len(tilted.thresholds['vertical_suspicious']) == 3
    assert tilted.per_segment['b']['detected'] is True  # the sign is read against the found vertical anyway


def test_a_negated_unit_needs_both_readings():
    recs = negate(trials())
    res = check_for_negated_accelerometer(recs)
    b = res.per_segment['b']['sessions'][0]
    assert b['negated_main'] is True and b['negated_cross'] is True and b['reason'] == 'both readings negated'
    assert b['cross_slope'] == pytest.approx(-1.0, abs=0.15)
    for r in recs:
        for seg in r.segments.values():
            seg.reference.markers = None  # no point on the segment: the cross-check cannot read
    res = check_for_negated_accelerometer(recs)
    b = res.per_segment['b']['sessions'][0]
    assert b['negated_main'] is True and b['negated'] is None and b['reason'] == 'cross-check unreadable'
    assert res.per_segment['b']['detected'] is None


def test_handlers_count_the_negated_session_and_the_vertical():
    """Dataset level: participant P1's unit b is negated in all three trials, P2 is clean but its optical system
    calls +y up. Error 20 counts P1's three trials and one sensor-session and negates b back; error 19 counts P2's
    three trials (vertical +y) and changes nothing."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e19_e20_accelerometer_sign as M
    p1, p2 = negate(trials()), trials(world=Rotation.from_euler('x', -90, degrees=True))
    recs = {**{f'p1_{i}': r for i, r in enumerate(p1)}, **{f'p2_{i}': r for i, r in enumerate(p2)}}
    loader = MemoryLoader(recs, participant={t: t[:2].upper() for t in recs})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (6, 3, 1)
    fixed = M.correct(loader, rows).load('p1_0')
    assert np.allclose(fixed['b'].imu.acc, trials()[0]['b'].imu.acc)
    vertical = M.check_vertical(loader, 'synthetic', 1)
    v = M.prevalence_vertical(vertical)
    assert (v['trials_detected'], v['occurrences']) == (3, 3)
    assert set(vertical[vertical.detected == True].vertical_axis) == {'+y'}  # noqa: E712
    assert M.correct_vertical(loader, vertical) is loader
