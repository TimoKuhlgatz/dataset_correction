"""Known-answer tests of the gyroscope offset check (error 17): the offset is recovered from rest phases
found by the IMU and the reference together, per trial, with the session's pooled rest as fallback; a constant
rotation (IMU quiet, reference turning) is not taken for rest; a mirrored axis does not change the estimate from
rest, nor the fallback fit against the reference."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref import errors as E
from mocapref.errors.e17_gyroscope_offset import rest_mask
from mocapref.recording import ImuStream, Recording, ReferenceStream, Segment
from mocapref.synthetic import GRAVITY, _integrate, smooth_rate

RATE = 100.0
FINE = 2000.0


def recording(trial, rests=(), spin=None, offset_dps=(0, 0, 0), duration_s=40.0, seed=0, noise_dps=0.2):
    """One segment moving smoothly, still during each (start, stop) in `rests` (seconds), turning at a constant
    rate `spin` (rad/s vector, start, stop) if given; gyroscope with a constant offset and white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / FINE)
    w = smooth_rate(t, rng)
    envelope = np.ones_like(t)
    for a, b in rests:  # smooth stop and start over 0.3 s
        envelope = np.minimum(envelope, np.clip((np.abs(t - (a + b) / 2) - (b - a) / 2) / 0.3, 0, 1))
    w = w * envelope[:, None]
    if spin is not None:
        rate, a, b = spin
        inside = (t >= a) & (t < b)
        w[inside] = rate
    quats = _integrate(Rotation.identity(), w, 1 / FINE)
    k = np.round(np.arange(int(duration_s * RATE)) / RATE * FINE).astype(int)
    gyr = w[k] + np.radians(offset_dps) + rng.normal(0, np.radians(noise_dps), (len(k), 3))
    acc = Rotation.from_quat(quats[k]).inv().apply(GRAVITY) + rng.normal(0, 0.01, (len(k), 3))
    seg = Segment('shank', ImuStream(gyr, acc, RATE), ReferenceStream(Rotation.from_quat(quats[k]), RATE,
                                                                     convention='segment_to_world'))
    return Recording('synthetic', trial, {'shank': seg}, session='s1')


def test_offset_is_recovered_from_rest_in_each_trial():
    a = recording('a', rests=[(5, 8), (20, 23)], offset_dps=(2.0, -1.0, 0.5), seed=1)
    b = recording('b', rests=[(10, 13)], offset_dps=(-3.0, 0.0, 1.0), seed=2)  # another offset (power cycle)
    result = E.check_for_gyroscope_offset([a, b])
    trials = {t['trial']: t for t in result.per_segment['shank']['per_trial']}
    assert trials['a']['method'] == trials['b']['method'] == 'rest (trial)'
    assert np.allclose(np.degrees(trials['a']['offset']), [2.0, -1.0, 0.5], atol=0.05)
    assert np.allclose(np.degrees(trials['b']['offset']), [-3.0, 0.0, 1.0], atol=0.05)
    assert result.detected
    fixed = E.remove_gyroscope_offset(b, result)
    assert np.allclose(np.degrees(fixed['shank'].imu.gyr[rest_mask(fixed['shank'])].mean(0)), 0, atol=0.05)


def test_rest_is_found_only_where_the_segment_is_still():
    rec = recording('a', rests=[(5, 8)], seed=3)
    mask = rest_mask(rec['shank'])
    times = rec['shank'].imu.times
    assert mask.any() and np.all((times[mask] > 4.9) & (times[mask] < 8.1))


def test_trial_without_rest_takes_the_pooled_session_rest():
    a = recording('a', rests=[(5, 8)], offset_dps=(2.0, -1.0, 0.5), seed=4)
    b = recording('b', rests=[], offset_dps=(2.0, -1.0, 0.5), seed=5)
    trials = {t['trial']: t for t in E.check_for_gyroscope_offset([a, b]).per_segment['shank']['per_trial']}
    assert trials['b']['method'] == 'rest (session)' and trials['b']['rest_s'] == 0
    assert np.allclose(np.degrees(trials['b']['offset']), [2.0, -1.0, 0.5], atol=0.05)


def test_constant_rotation_is_not_taken_for_rest():
    """30 °/s about the vertical for 10 s: gyroscope and accelerometer magnitude stay constant (the IMU is quiet),
    but the reference turns, so it is not rest and the offset is not 30 °/s."""
    rec = recording('a', rests=[(25, 28)], spin=(np.radians([0, 0, 30.0]), 5, 15), offset_dps=(1.0, 0, 0), seed=6,
                    noise_dps=0.05)
    seg = rec['shank']
    spin = (seg.imu.times > 6) & (seg.imu.times < 14)
    assert not rest_mask(seg)[spin].any()
    imu_only = rest_mask(seg, max_ref_dps=np.inf)  # without the reference the spin would pass as rest
    assert imu_only[spin].mean() > 0.9
    trial = E.check_for_gyroscope_offset(rec).per_segment['shank']['per_trial'][0]
    assert np.allclose(np.degrees(trial['offset']), [1.0, 0, 0], atol=0.05)


@pytest.mark.parametrize('axes', [[1], [0, 1, 2]])
def test_mirrored_axes_do_not_change_the_estimate(axes):
    """The offset is read in the raw sensor frame, so a flipped axis just carries its offset with it; the fallback
    fit against the reference allows a mirror image and recovers the offset as well."""
    offset = np.array([2.0, -1.0, 0.5])
    flip = np.ones(3)
    flip[axes] = -1
    rest = recording('a', rests=[(5, 8)], offset_dps=offset, seed=7)
    moving = recording('b', rests=[], offset_dps=offset, seed=8)
    for rec in (rest, moving):
        rec['shank'].imu.gyr *= flip  # mirrored sensor: raw reading = flip * (rate + offset)
    from_rest = E.check_for_gyroscope_offset(rest).per_segment['shank']
    assert from_rest['method'] == 'rest (trial)'
    assert np.allclose(np.degrees(from_rest['offset']), flip * offset, atol=0.05)
    from_fit = E.check_for_gyroscope_offset(moving).per_segment['shank']
    assert from_fit['method'] == 'reference fit'
    assert np.allclose(np.degrees(from_fit['offset']), flip * offset, atol=0.1)
    assert E.check_for_gyroscope_offset(moving, reference_fallback=False).per_segment['shank']['method'] == 'none'


def test_reference_jitter_does_not_hide_rest():
    """0.3° of marker jitter per frame reads about 40 °/s frame to frame on a still reference; the net rotation over
    the window does not add it up, so the rest is still found and the offset recovered."""
    rec = recording('a', rests=[(5, 8)], offset_dps=(2.0, -1.0, 0.5), seed=7)
    ref = rec['shank'].reference
    jitter = Rotation.from_rotvec(np.random.default_rng(8).normal(0, np.radians(0.3), (len(ref), 3)))
    ref.orientation = ref.orientation * jitter
    seg = rec['shank']
    times = seg.imu.times
    mask = rest_mask(seg)
    assert mask[(times > 5.5) & (times < 7.5)].mean() > 0.8
    trial = E.check_for_gyroscope_offset(rec).per_segment['shank']['per_trial'][0]
    assert trial['method'] == 'rest (trial)'
    assert np.allclose(np.degrees(trial['offset']), [2.0, -1.0, 0.5], atol=0.05)


def test_handler_counts_offsets_and_subtracts_them():
    """Dataset level: two trials of one session, each with rest and an offset; both counted, both removed."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e17_gyroscope_offset as M
    recs = {'a': recording('a', rests=[(5, 8)], offset_dps=(1.0, -0.5, 0.3), seed=1),
            'b': recording('b', rests=[(20, 23)], offset_dps=(1.0, -0.5, 0.3), seed=2)}
    loader = MemoryLoader(recs, session={'a': 's1', 'b': 's1'})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 2, 2)
    fixed = M.correct(loader, rows).load('a')
    left = M.check_for_gyroscope_offset(fixed, reference_fallback=False).per_segment['shank']['estimate']
    assert left < 0.1
