"""Known-answer tests of the segment assignment check (error 9): exchanged IMUs are found and put back, nothing is
found on clean recordings, and the check does not depend on a time offset, a gyroscope offset, a sensor rotation
or clock drift (all corrected only later in the order)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors.e09_segment_assignment import check_for_segment_assignment, remove_segment_assignment
from mocapref.recording import ImuStream, Recording, ReferenceStream, Segment
from mocapref.synthetic import _integrate, make_recording

NAMES = ('pelvis', 'thigh', 'shank', 'foot')


def exchanged(rec: Recording, moves: dict) -> Recording:
    """A copy whose segment `target` carries the IMU of `source` for every source -> target in moves."""
    new = rec.copy()
    for source, target in moves.items():
        new.segments[target] = Segment(target, rec[source].imu, rec[target].reference)
    return new


def decisions(result):
    return {n: (e['detected'], e['belongs_to']) for n, e in result.per_segment.items()}


@pytest.fixture(scope='module')
def clean():
    return make_recording(duration_s=60.0, segment_names=NAMES, seed=3)


def test_clean_recording_has_every_imu_in_place(clean):
    result = check_for_segment_assignment(clean)
    assert result.detected is False
    assert all(d == (False, n) for n, d in decisions(result).items())


def test_two_exchanged_imus_are_found_and_put_back(clean):
    rec = exchanged(clean, {'thigh': 'foot', 'foot': 'thigh'})
    result = check_for_segment_assignment(rec)
    assert result.detected is True
    d = decisions(result)
    assert d['thigh'] == (True, 'foot') and d['foot'] == (True, 'thigh')
    assert d['pelvis'] == (False, 'pelvis') and d['shank'] == (False, 'shank')
    fixed = remove_segment_assignment(rec, result)
    assert check_for_segment_assignment(fixed).detected is False
    assert np.array_equal(fixed['thigh'].imu.gyr, clean['thigh'].imu.gyr)


def test_three_way_exchange_is_found_and_put_back(clean):
    rec = exchanged(clean, {'pelvis': 'thigh', 'thigh': 'shank', 'shank': 'pelvis'})
    result = check_for_segment_assignment(rec)
    d = decisions(result)
    assert d['pelvis'] == (True, 'shank') and d['thigh'] == (True, 'pelvis') and d['shank'] == (True, 'thigh')
    assert check_for_segment_assignment(remove_segment_assignment(rec, result)).detected is False


def test_exchange_found_despite_offset_bias_rotation_and_drift():
    """Common time offset 0.7 s, clock drift 500 ppm, 3 deg/s gyroscope offset and a 40 deg sensor rotation at
    once: none of them is corrected before this check."""
    rec = make_recording(duration_s=90.0, segment_names=NAMES, seed=4, offset_s=0.7, drift_ppm=500.0,
                         mounting=Rotation.from_euler('xyz', [40, -25, 10], degrees=True))
    for seg in rec.segments.values():
        seg.imu.gyr = seg.imu.gyr + np.radians([3.0, -2.0, 1.5])
    result = check_for_segment_assignment(exchanged(rec, {'pelvis': 'shank', 'shank': 'pelvis'}))
    assert abs(float(result.summary.split('offset ')[1].split(' ms')[0]) - 700) < 30
    d = decisions(result)
    assert d['pelvis'] == (True, 'shank') and d['shank'] == (True, 'pelvis')
    assert d['thigh'] == (False, 'thigh') and d['foot'] == (False, 'foot')


def gait(duration_s=60.0, rate=100.0, stride_s=1.1, seed=0, periodic=False):
    """Left and right thigh and shank doing the same motion, the right side half a stride later. Unless
    `periodic`, the stride time varies by up to 8 % and the walk starts and stops, as real walking does; a perfectly
    periodic walk repeats itself every half stride across the sides."""
    rng = np.random.default_rng(seed)
    t = np.arange(-1.0, duration_s + 2.0, 1 / 2000.0)
    if periodic:
        cycles, envelope = t / stride_s, np.ones_like(t)
    else:
        f = (1 + 0.05 * np.sin(2 * np.pi * t / 13.0) + 0.03 * np.sin(2 * np.pi * t / 5.3)) / stride_s
        cycles = np.cumsum(f) / 2000.0
        envelope = (np.clip(t / 3, 0, 1) * np.clip((duration_s - t) / 3, 0, 1)
                    * (1 + 0.2 * np.sin(2 * np.pi * t / 17)))
    harmonics = {part: rng.uniform(0.2, 1.0, (3, 3)) for part in ('thigh', 'shank')}
    segments = {}
    for side, phase in (('left', 0.0), ('right', 0.5)):
        for part, amp in (('thigh', 1.0), ('shank', 1.6)):
            w = np.stack([sum(harmonics[part][a, h] * np.sin(2 * np.pi * (h + 1) * (cycles - phase) + a + h)
                              for h in range(3)) for a in range(3)], 1) * amp * envelope[:, None]
            quats = _integrate(Rotation.identity(), w, 1 / 2000.0)
            k = np.round((np.arange(int(duration_s * rate)) / rate + 1.0) * 2000).astype(int)
            name = f'{side}_{part}'
            segments[name] = Segment(name, ImuStream(w[k], np.tile([0, 0, 9.81], (len(k), 1)), rate),
                                     ReferenceStream(Rotation.from_quat(quats[k]), rate,
                                                     convention='segment_to_world'))
    return Recording('synthetic', 'gait', segments)


def test_left_right_exchange_in_periodic_gait_is_found():
    """The two sides do the same motion half a stride apart: a per-sensor lag search would match the exchanged
    IMUs at half a stride; at the common lag they are told apart."""
    rec = gait()
    assert check_for_segment_assignment(rec).detected is False
    result = check_for_segment_assignment(exchanged(rec, {'left_shank': 'right_shank', 'right_shank': 'left_shank'}))
    d = decisions(result)
    assert d['left_shank'] == (True, 'right_shank') and d['right_shank'] == (True, 'left_shank')


@pytest.mark.parametrize('own_s', [0.2, 0.55])
def test_individual_offset_in_gait_is_not_called_a_wrong_segment(own_s):
    """The right shank IMU on its own clock, 0.2 s or half a stride (0.55 s) late: every IMU stays in place."""
    rec = gait()
    rec['right_shank'].imu.start_s += own_s
    assert all(d == (False, n) for n, d in decisions(check_for_segment_assignment(rec)).items())


def test_half_stride_individual_offset_in_periodic_gait_is_undecided(monkeypatch):
    """Perfectly periodic gait with the common lag known: half a stride on one IMU's own clock cannot be told from
    a left/right exchange, so the two shanks must come out undecided (step 3's re-run) and keep their declared
    segments, not be exchanged."""
    from mocapref.errors import e09_segment_assignment as S
    monkeypatch.setattr(S, 'common_offset', lambda streams, *a: S.estimate_time_lag(
        streams.grid, np.ones(len(streams.grid)), streams.grid, np.ones(len(streams.grid)), 0.0))
    rec = gait(periodic=True)
    rec['right_shank'].imu.start_s += 0.55
    result = check_for_segment_assignment(rec)
    d = decisions(result)
    assert d['left_shank'] == (False, 'left_shank') and d['right_shank'] == (False, 'right_shank')
    assert result.per_segment['left_shank']['undecided'] and result.per_segment['right_shank']['undecided']
    assert result.detected is False
    assert d['left_thigh'] == (False, 'left_thigh') and d['right_thigh'] == (False, 'right_thigh')


def test_large_individual_offset_is_not_called_a_wrong_segment(clean):
    """One IMU 0.4 s off its own clock (not shared by the others): the check must not report it as being on
    another segment (it may be decided or undecided, and then keeps its segment)."""
    rec = clean.copy()
    rec['shank'].imu.start_s += 0.4
    d = decisions(check_for_segment_assignment(rec))
    assert d['shank'][0] is not True
    assert all(d[n] == (False, n) for n in ('pelvis', 'thigh', 'foot'))


def test_handler_counts_imus_and_gives_them_back(clean):
    from memory_loader import MemoryLoader
    from mocapref.errors import e09_segment_assignment as H

    short = make_recording(duration_s=5.0, segment_names=NAMES, seed=4)  # shorter than one window: not checked
    loader = MemoryLoader({'clean': clean, 'swapped': exchanged(clean, {'thigh': 'foot', 'foot': 'thigh'}),
                           'short': short})
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences'], p['undecided']) == (3, 1, 2, 1)
    fixed = H.correct(loader, rows).load('swapped')
    assert np.array_equal(fixed['thigh'].imu.gyr, clean['thigh'].imu.gyr)
    assert np.array_equal(fixed['foot'].imu.gyr, clean['foot'].imu.gyr)
