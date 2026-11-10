"""Known-answer tests: every check finds nothing on a clean synthetic recording, recovers an
injected error, and the corrected copy passes the check again."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref import errors as E
from mocapref.synthetic import make_recording


@pytest.fixture(scope='module')
def clean():
    return make_recording(duration_s=60.0, imu_rate_hz=100.0, reference_rate_hz=120.0)


def test_clean_recording_shows_no_error(clean):
    for check in (E.check_for_synchronisation_offset, E.check_for_held_samples,
                  E.check_for_corrupted_samples, E.check_for_reference_jumps, E.check_for_gyroscope_offset,
                  E.check_for_mirrored_gyroscope, E.check_for_mounting_misalignment):
        assert check(clean).detected is False, check.__name__


@pytest.mark.parametrize('offset_s', [0.137, -0.02, 0.0063])
def test_synchronisation_offset_is_recovered_and_removed(offset_s):
    rec = make_recording(duration_s=60.0, offset_s=offset_s, reference_rate_hz=120.0)
    result = E.check_for_synchronisation_offset(rec)
    assert result.detected
    assert result.offset_s == pytest.approx(offset_s, abs=0.5e-3)
    fixed = E.remove_synchronisation_offset(rec, result)
    assert abs(E.check_for_synchronisation_offset(fixed).offset_s) < 0.5e-3
    assert rec.segments['upper'].imu.start_s == 0.0  # the input is untouched
    assert len(fixed.corrections) == 1


@pytest.mark.parametrize('offset_s', [0.137, 3.0])
def test_offset_is_recovered_by_default_with_only_missing_frames_left_out(offset_s):
    rec = make_recording(duration_s=60.0, offset_s=offset_s, reference_rate_hz=120.0)
    for seg in rec.segments.values():
        seg.reference.valid[1000:1100] = False  # an occlusion
    result = E.check_for_synchronisation_offset(rec)  # default: only frames without an orientation left out, ±10 s
    assert result.detected and result.offset_s == pytest.approx(offset_s, abs=0.5e-3)
    assert result.thresholds['exclude'] == 'invalid'


@pytest.mark.parametrize('spikes', [1, 10])
def test_offset_survives_corrupted_gyroscope_samples(spikes):
    """Single samples of 5000 deg/s took a correlation rule's peak below 0.9; the rank rule ignores them."""
    rec = make_recording(duration_s=60.0, offset_s=0.137, reference_rate_hz=120.0)
    rng = np.random.default_rng(1)
    for seg in rec.segments.values():
        idx = rng.choice(np.arange(1, len(seg.imu.gyr) - 1), spikes, replace=False)
        seg.imu.gyr[idx, 0] = np.radians(5000.0)
    result = E.check_for_synchronisation_offset(rec)
    assert result.detected and result.offset_s == pytest.approx(0.137, abs=0.5e-3)
    assert all(e['detected'] is not None for e in result.per_segment.values())


def test_unrelated_recordings_are_not_trusted():
    """The IMU of one recording against the reference of another (different motion): no segment is trusted."""
    a = make_recording(duration_s=60.0, seed=1)
    b = make_recording(duration_s=60.0, seed=2)
    for name in a.segments:
        a.segments[name].imu = b.segments[name].imu
    assert E.check_for_synchronisation_offset(a).detected is None


def test_segment_without_common_time_is_undecided_and_the_others_still_count():
    """One IMU stamped on its own clock, far from the reference's time (Trento participant 9): that segment is
    undecided, the trial's offset comes from the other segment."""
    rec = make_recording(duration_s=30.0, offset_s=0.05)
    rec.segments['upper'].imu.start_s = 3000.0
    result = E.check_for_synchronisation_offset(rec)
    assert result.per_segment['upper']['detected'] is None
    assert result.detected is True and abs(result.offset_s - 0.05) < 1e-3


def test_imus_on_separate_clocks_end_on_the_reference_clock():
    """Three IMUs, each on its own clock (0.05, 0.05 and 2.0 s against the reference): the trial's median is removed
    first, then each IMU by its own rest; afterwards no IMU shows an offset."""
    from mocapref.errors.e12_time_offset import remove_sensor_offsets
    rec = make_recording(duration_s=30.0, offset_s=0.05, segment_names=('a', 'b', 'c'))
    rec.segments['c'].imu.start_s += 2.0
    result = E.check_for_synchronisation_offset(rec)
    own = {n: e['lag'].offset_s for n, e in result.per_segment.items() if e['detected'] is not None}
    assert abs(own['c'] - 2.05) < 1e-3
    fixed = remove_sensor_offsets(E.remove_synchronisation_offset(rec, result), own, result.offset_s)
    after = E.check_for_synchronisation_offset(fixed)
    assert all(abs(e['lag'].offset_s) < 1e-3 for e in after.per_segment.values())
    partial = remove_sensor_offsets(rec, {'a': 0.05}, 0.0)
    assert partial.corrections[-1].parameters['not_time_aligned'] == ['b', 'c']


def test_offset_beyond_search_window_is_undecided():
    rec = make_recording(duration_s=30.0, offset_s=0.8)
    assert E.check_for_synchronisation_offset(rec, max_offset_s=0.3).detected is None


@pytest.mark.parametrize('drift_ppm', [150.0, -60.0])
def test_clock_drift_is_recovered_and_removed(drift_ppm):
    rec = make_recording(duration_s=300.0, drift_ppm=drift_ppm, offset_s=0.01)
    result = E.check_for_clock_drift(rec)
    assert result.detected
    assert result.drift_ppm == pytest.approx(drift_ppm, abs=5.0)
    assert result.drift_ms_per_min == pytest.approx(drift_ppm * 0.06, abs=0.3)
    windows = result.per_segment['upper']['windows']
    assert len(windows) == 19  # 300 s in 30 s windows overlapping by 50 %: starts every 15 s
    fixed = E.remove_clock_drift(rec, result)
    assert not E.check_for_clock_drift(fixed).detected
    # a drifting recording has no single offset: its block offsets rise with time, so the offset check (block
    # agreement) leaves it undecided; only the drift is removed, and the constant offset (the offset at
    # mid-recording) stays for its own correction
    assert E.check_for_synchronisation_offset(rec).detected is None
    after = E.check_for_synchronisation_offset(fixed)
    assert after.detected is not None  # decided (at -60 ppm the 1 ms left is within the tolerance)
    assert after.offset_s == pytest.approx(0.01 + drift_ppm * 1e-6 * 150.0, abs=1.5e-3)


@pytest.mark.parametrize('drift_ppm', [150.0, -60.0])
def test_clock_drift_with_the_rank_block_rule(drift_ppm):
    """method='rank-blocks' for the windows: the drift is recovered, also
    with one corrupted sample of 5000 deg/s per segment."""
    rec = make_recording(duration_s=300.0, drift_ppm=drift_ppm, offset_s=0.01)
    for seg in rec.segments.values():
        seg.imu.gyr[5000, 0] = np.radians(5000.0)
    result = E.check_for_clock_drift(rec, method='rank-blocks')
    assert result.detected
    assert result.drift_ppm == pytest.approx(drift_ppm, abs=5.0)


def test_constant_offset_is_not_reported_as_drift():
    rec = make_recording(duration_s=120.0, offset_s=0.137)
    result = E.check_for_clock_drift(rec)
    assert result.detected is False and abs(result.drift_ms_per_min) < 0.1


def test_drift_is_tested_from_60_s_on():
    assert E.check_for_clock_drift(make_recording(duration_s=50.0)).detected is None
    result = E.check_for_clock_drift(make_recording(duration_s=62.0, drift_ppm=500.0))
    assert result.detected and len(result.per_segment['upper']['windows']) == 4  # 62 s: starts 0, 10.7, 21.3, 32 s


def test_windows_cover_the_whole_recording_with_at_least_half_overlap():
    from mocapref.errors.e13_clock_drift import window_starts
    starts = window_starts(0.0, 70.0)
    assert starts[0] == 0.0 and starts[-1] + 30.0 == pytest.approx(70.0)
    assert np.all(np.diff(starts) <= 15.0 + 1e-9)


def test_mounting_is_recovered_and_removed():
    mounting = Rotation.from_euler('xyz', [20, -10, 90], degrees=True)
    rec = make_recording(duration_s=60.0, mounting=mounting)
    result = E.check_for_mounting_misalignment(rec)
    assert result.detected
    assert result.per_segment['upper']['estimate'] == pytest.approx(np.degrees(mounting.magnitude()), abs=0.2)
    assert E.check_for_mounting_misalignment(E.remove_mounting_misalignment(rec, result)).per_segment['upper'][
        'estimate'] < 0.2


@pytest.mark.parametrize('convention', ['segment_to_world', 'unknown'])
@pytest.mark.parametrize('axes', [[0, 1, 2], [1]])
def test_mirrored_gyroscope_is_detected(clean, axes, convention):
    rec = clean.copy()
    for seg in rec.segments.values():
        seg.reference.convention = convention
    rec.segments['lower'].imu.gyr[:, axes] *= -1
    result = E.check_for_mirrored_gyroscope(rec)
    assert result.detected
    assert result.per_segment['lower']['detected'] and not result.per_segment['upper']['detected']


def test_gyroscope_offset_is_recovered_and_removed(clean):
    rec = clean.copy()
    offset = np.radians([2.0, -1.0, 0.5])
    rec.segments['upper'].imu.gyr += offset
    result = E.check_for_gyroscope_offset(rec)
    assert result.detected and result.per_segment['upper']['method'] == 'reference fit'  # no rest phase
    assert np.allclose(result.per_segment['upper']['offset'], offset, atol=np.radians(0.1))
    assert not E.check_for_gyroscope_offset(E.remove_gyroscope_offset(rec, result)).detected


def test_sample_defects_are_found_and_marked_invalid(clean):
    rec = clean.copy()
    lower = rec.segments['lower'].imu
    lower.gyr[500] += 20.0  # isolated spike
    lower.gyr[800:803] = lower.gyr[799]  # three held samples
    lower.acc[800:803] = lower.acc[799]
    spikes = E.check_for_corrupted_samples(rec)
    assert spikes.detected and spikes.per_segment['lower']['estimate'] == 1
    held = E.check_for_held_samples(rec)
    assert held.detected and held.per_segment['lower']['estimate'] == 3
    fixed = E.remove_held_samples(E.remove_corrupted_samples(rec))
    assert fixed.segments['lower'].imu.valid.sum() == len(lower) - 4
    assert [c.error for c in fixed.corrections] == ['corrupted samples', 'held samples']


def test_dead_channels_are_found_only_while_moving_and_from_five_samples(clean):
    rng = np.random.default_rng(7)
    rec = clean.copy()
    for seg in rec.segments.values():  # sensor noise, so a live channel never repeats exactly
        seg.imu.gyr += rng.normal(0, 0.005, seg.imu.gyr.shape)
        seg.imu.acc += rng.normal(0, 0.02, seg.imu.acc.shape)
    assert E.check_for_dead_channels(rec).detected is False
    upper, lower = rec.segments['upper'].imu, rec.segments['lower'].imu
    upper.gyr[1000:1020, 1] = upper.gyr[1000, 1]  # one gyroscope channel stuck for 20 samples
    upper.acc[3000:3004, 0] = upper.acc[3000, 0]  # 4 equal samples: shorter than the minimum
    lower.gyr[2000:2050] = 0.0  # the whole sensor drops out for 50 samples
    lower.acc[2000:2050] = 0.0
    lower.gyr[4000, 2] = np.nan  # one missing sample
    result = E.check_for_dead_channels(rec)
    assert result.per_segment['upper']['estimate'] == 20
    assert result.per_segment['upper']['by_other_channels'] == 1 and result.per_segment['upper']['by_reference'] == 1
    assert result.per_segment['lower']['estimate'] == 51 and result.per_segment['lower']['missing'] == 1
    fixed = E.remove_dead_channels(rec)
    assert fixed.segments['upper'].imu.valid.sum() == len(upper) - 20


def test_a_constant_channel_at_rest_is_not_called_stuck():
    rec = make_recording(duration_s=30.0, imu_rate_hz=100.0, reference_rate_hz=100.0)
    rng = np.random.default_rng(8)
    for seg in rec.segments.values():
        seg.reference.orientation = Rotation.from_quat(np.tile(seg.reference.orientation[0].as_quat(), (len(seg.reference), 1)))
        seg.reference.orientation = seg.reference.orientation * Rotation.from_rotvec(rng.normal(0, 1e-3, (len(seg.reference), 3)))
        seg.imu.gyr = rng.normal(0, 0.005, seg.imu.gyr.shape)  # at rest: noise only
        seg.imu.acc = np.tile([0.0, 0.0, 9.81], (len(seg.imu), 1)) + rng.normal(0, 0.02, seg.imu.acc.shape)
    rec.segments['upper'].imu.gyr[500:520] = 0.0  # all gyroscope channels exactly 0 for 20 samples, at rest
    rec.segments['upper'].imu.acc[500:520] = rec.segments['upper'].imu.acc[500]
    assert E.check_for_dead_channels(rec).per_segment['upper']['estimate'] == 0


def test_reference_jumps_are_found_and_marked_invalid(clean):
    rec = clean.copy()
    ref = rec.segments['lower'].reference
    q = ref.orientation.as_quat()
    q[3000] = (ref.orientation[3000] * Rotation.from_euler('x', 170, degrees=True)).as_quat()
    ref.orientation = Rotation.from_quat(q)
    result = E.check_for_reference_jumps(rec)
    assert result.detected and result.per_segment['lower']['estimate'] == 3  # the frame and both neighbours' steps
    fixed = E.remove_reference_jumps(rec, margin_s=0.0)
    assert not fixed.segments['lower'].reference.valid[3000]


def test_imus_on_different_time_scales_are_found_and_aligned():
    names = ('a', 'b', 'c', 'd')
    rec = make_recording(duration_s=180.0, segment_names=names, segment_drift_ppm={'c': 800.0})
    result = E.check_for_sensor_rate_mismatch(rec)
    assert result.detected
    assert [n for n, p in result.per_segment.items() if p['detected']] == ['c']
    assert result.per_segment['c']['deviation_ms_per_min'] == pytest.approx(800.0 * 0.06, abs=1.5)
    fixed = E.remove_sensor_rate_mismatch(rec, result)
    assert not E.check_for_sensor_rate_mismatch(fixed).detected
    assert not E.check_for_clock_drift(fixed).detected  # the IMUs now share the reference's time scale again


def test_shared_drift_is_not_a_rate_mismatch():
    rec = make_recording(duration_s=180.0, segment_names=('a', 'b', 'c'), drift_ppm=500.0)
    assert E.check_for_clock_drift(rec).detected
    assert E.check_for_sensor_rate_mismatch(rec).detected is False


def test_trial_pairing_recovers_a_shift_by_an_extra_take_and_a_swap():
    from mocapref.errors.e08_trial_pairing import (assign_any_order, assign_in_order, imu_speeds, reference_speeds, score,
                                              window_candidates)
    recs = [make_recording(duration_s=40.0, seed=10 + k, segment_names=('a', 'b', 'c')) for k in range(9)]
    imu = [imu_speeds(r) for r in recs[:8]]  # eight IMU trials
    optical = [reference_speeds(r) for r in recs[:3]]  # optical files: an aborted take is inserted at position 3,
    optical += [reference_speeds(recs[8])] + [reference_speeds(r) for r in recs[3:8]]  # so trials 3-7 shift by one
    scores = {(k, j): score(imu[k], optical[j])[0] for k, j in window_candidates(8, 9)}
    pairing, _ = assign_in_order(8, 9, scores)
    assert pairing == {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}  # declared k -> k is wrong from trial 3 on
    everything = {(k, j): score(imu[k], optical[j])[0] for k in range(8) for j in range(9)}
    assert assign_any_order(list(range(8)), list(range(9)), everything)[0] == pairing
    swapped = optical[:]  # trials 2 and 6 swapped: no chronological assignment can express it, the any-order one can
    swapped[2], swapped[7] = optical[7], optical[2]
    everything = {(k, j): score(imu[k], swapped[j])[0] for k in range(8) for j in range(9)}
    expected = {**pairing, 2: 7, 6: 2}
    assert assign_any_order(list(range(8)), list(range(9)), everything)[0] == expected
