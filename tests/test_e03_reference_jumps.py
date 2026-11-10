"""Known-answer tests for the first pass of error 3 (reference jumps or flips) from the reference alone."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.errors.e03_reference_jumps import (MAX_SEGMENT_RATE_DPS, check_for_impossible_reference_steps,
                                            impossible_steps, remove_impossible_reference_steps)
from mocapref.synthetic import make_recording


@pytest.fixture(scope='module')
def clean():
    return make_recording(duration_s=60.0, imu_rate_hz=100.0, reference_rate_hz=120.0)


def _with(rec, segment, frames, rotation):
    ref = rec.segments[segment].reference
    q = ref.orientation.as_quat()
    q[frames] = (ref.orientation[frames] * rotation).as_quat()
    ref.orientation = Rotation.from_quat(q)


def test_limit_is_mean_plus_one_sd_of_the_fastest_condition():
    assert MAX_SEGMENT_RATE_DPS == pytest.approx(7649.0)


def test_clean_reference_has_no_impossible_step(clean):
    assert check_for_impossible_reference_steps(clean).detected is False


def test_negated_quaternion_is_not_a_step(clean):
    rec = clean.copy()
    ref = rec.segments['lower'].reference
    q = ref.orientation.as_quat()
    q[100:200] *= -1  # same orientations, other sign
    ref.orientation = Rotation.from_quat(q)
    assert check_for_impossible_reference_steps(rec).detected is False


def test_one_frame_glitch_masks_only_that_frame(clean):
    rec = clean.copy()
    _with(rec, 'lower', [3000], Rotation.from_euler('x', 170, degrees=True))
    r = check_for_impossible_reference_steps(rec).per_segment['lower']
    assert r['detected'] and r['impossible_steps'] == 2 and r['pairs'] == 1 and r['unpaired'] == 0
    fixed = remove_impossible_reference_steps(rec)
    assert np.flatnonzero(~fixed.segments['lower'].reference.valid).tolist() == [3000]
    assert fixed.segments['upper'].reference.valid.all()


def test_flip_masks_the_whole_flipped_stretch(clean):
    rec = clean.copy()
    _with(rec, 'upper', slice(1200, 1440), Rotation.from_euler('z', 180, degrees=True))  # 2 s at 120 Hz
    fixed = remove_impossible_reference_steps(rec)
    assert np.flatnonzero(~fixed.segments['upper'].reference.valid).tolist() == list(range(1200, 1440))


def test_jump_below_the_limit_is_not_found(clean):
    rec = clean.copy()
    _with(rec, 'lower', [3000], Rotation.from_euler('x', 40, degrees=True))  # 40° x 120 Hz = 4,800 °/s
    assert check_for_impossible_reference_steps(rec).detected is False


def test_unpaired_step_is_reported_not_marked(clean):
    rec = clean.copy()
    n = len(rec.segments['lower'].reference)
    _with(rec, 'lower', slice(n - 600, n), Rotation.from_euler('y', 150, degrees=True))  # flip to the end
    r = check_for_impossible_reference_steps(rec).per_segment['lower']
    assert r['detected'] and r['unpaired'] == 1 and r['estimate'] == 0
    assert remove_impossible_reference_steps(rec).segments['lower'].reference.valid.all()


def test_step_across_invalid_frames_is_measured_over_the_gap(clean):
    rec = clean.copy()
    _with(rec, 'lower', slice(3000, 3001), Rotation.from_euler('x', 170, degrees=True))
    rec.segments['lower'].reference.valid[3000] = False  # the glitch frame is already invalid
    assert impossible_steps(rec.segments['lower'].reference)['steps'] == []


def test_too_low_frame_rate_is_not_checkable():
    rec = make_recording(duration_s=10.0, imu_rate_hz=100.0, reference_rate_hz=40.0)
    result = check_for_impossible_reference_steps(rec)
    assert result.detected is None and result.per_segment['upper']['detected'] is None


def test_two_different_wrong_frames_do_not_shift_the_later_pairs(clean):
    rec = clean.copy()
    _with(rec, 'lower', [2000], Rotation.from_euler('x', 170, degrees=True))
    _with(rec, 'lower', [2001], Rotation.from_euler('y', 90, degrees=True))  # second wrong frame, other offset
    _with(rec, 'lower', [5000], Rotation.from_euler('z', 160, degrees=True))  # a later one-frame glitch
    r = impossible_steps(rec.segments['lower'].reference)
    assert len(r['steps']) == 5
    assert (5000, 5000) in r['pairs']
    assert not r['mask'][2002:5000].any()  # nothing between the two glitches is masked


def test_unrelated_jumps_are_not_paired(clean):
    rec = clean.copy()
    n = len(rec.segments['upper'].reference)
    _with(rec, 'upper', slice(1000, n), Rotation.from_euler('y', 150, degrees=True))  # flips at 1000 ...
    _with(rec, 'upper', slice(4000, n), Rotation.from_euler('x', 150, degrees=True))  # ... and again, differently
    r = impossible_steps(rec.segments['upper'].reference)
    assert len(r['steps']) == 2 and r['pairs'] == [] and len(r['unpaired']) == 2 and not r['mask'].any()


def test_glitches_with_agreeing_offsets_are_not_taken_for_a_flip(clean):
    """Two unrelated glitches whose offsets agree pass the undo rule; the other segment shows nothing flipped between
    them, so the stretch is not marked (check against the other segments)."""
    rec = clean.copy()
    flip = Rotation.from_euler('x', 170, degrees=True)
    _with(rec, 'lower', [2000], flip)
    _with(rec, 'lower', [2001], Rotation.from_euler('y', 90, degrees=True))  # comes back through another wrong frame
    _with(rec, 'lower', [5000], flip)  # a later one-frame glitch with the same offset
    alone = impossible_steps(rec.segments['lower'].reference)
    assert alone['mask'][2002:5000].all()  # the undo rule alone takes it for a flip
    r = impossible_steps(rec.segments['lower'].reference, others=[rec.segments['upper'].reference])
    assert r['rejected'] >= 1 and (5000, 5000) in r['pairs']
    assert not r['mask'][2002:5000].any()


def test_flip_is_confirmed_by_the_other_segment(clean):
    rec = clean.copy()
    _with(rec, 'upper', slice(1200, 1440), Rotation.from_euler('z', 180, degrees=True))
    r = impossible_steps(rec.segments['upper'].reference, others=[rec.segments['lower'].reference])
    assert r['pairs'] == [(1200, 1439)] and r['rejected'] == 0 and r['unconfirmed'] == 0


def test_handler_counts_steps_and_masks_the_flipped_stretch():
    from memory_loader import MemoryLoader
    from mocapref.errors import e03_reference_jumps as H

    rec = make_recording(duration_s=20.0, imu_rate_hz=100.0, reference_rate_hz=120.0, seed=1)
    flipped = rec.copy()
    _with(flipped, 'lower', slice(1000, 1100), Rotation.from_euler('x', 170, degrees=True))
    loader = MemoryLoader({'clean': rec, 'flip': flipped})
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_detected'], p['occurrences'], p['occurrence_unit']) == (1, 2, 'steps')  # out and back
    fixed = H.correct(loader, rows)
    valid = fixed.load('flip').segments['lower'].reference.valid
    assert not valid[1000:1100].any() and valid[:1000].all() and valid[1100:].all()
    assert fixed.load('clean').corrections == []
