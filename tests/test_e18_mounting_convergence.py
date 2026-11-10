"""Known-answer tests for the per-trial mounting fit on the first 10 s of motion from the motion onset (error 18,
mounting rotation)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref import errors as E
from mocapref.synthetic import make_recording

MOUNTING = Rotation.from_euler('xyz', [20, -10, 90], degrees=True)


def test_rotation_is_recovered_from_the_first_10_s_only():
    """The sensor is remounted at 10 s: the fit must return the rotation of the first 10 s, not a blend."""
    rec = make_recording(duration_s=40.0, imu_rate_hz=100.0, reference_rate_hz=120.0, mounting=MOUNTING)
    later = Rotation.from_euler('z', 60, degrees=True)
    for seg in rec.segments.values():
        after = seg.imu.times >= 10.5
        seg.imu.gyr[after] = later.apply(seg.imu.gyr[after])
    result = E.check_for_mounting_in_convergence(rec)
    for name, p in result.per_segment.items():
        assert p['uncertainty_deg'] < 2
        assert p['estimate'] == pytest.approx(np.degrees(MOUNTING.magnitude()), abs=0.5), name
        fitted = Rotation.from_matrix(p['fit'].sensor_to_segment)
        assert np.degrees((fitted * MOUNTING).magnitude()) < 0.5  # sensor_to_segment undoes segment-to-sensor


def test_corrected_copy_has_no_mounting_left():
    rec = make_recording(duration_s=30.0, mounting=MOUNTING)
    fixed = E.remove_mounting_misalignment(rec, E.check_for_mounting_in_convergence(rec))
    again = E.check_for_mounting_in_convergence(fixed)
    assert again.detected is False
    assert all(p['estimate'] < 0.5 for p in again.per_segment.values())
    assert np.allclose(rec.segments['upper'].imu.gyr, make_recording(duration_s=30.0, mounting=MOUNTING)
                       .segments['upper'].imu.gyr)  # the original is untouched


def _still_until(rec, t_s):
    """Reference and gyroscope at rest until t_s, moving afterwards."""
    for seg in rec.segments.values():
        ref = seg.reference
        k = min(int(t_s * ref.rate_hz), len(ref) - 1)
        q = ref.orientation.as_quat()
        q[:k] = q[k]
        ref.orientation = Rotation.from_quat(q)
        seg.imu.gyr[seg.imu.times < t_s] = 0.0


def test_rest_at_the_start_moves_the_window_to_the_motion_onset():
    """Still for the first 10.5 s: the window starts at the motion onset, not at the trial start,
    so the rotation is recovered from the 10 s after it instead of being left out."""
    rec = make_recording(duration_s=40.0, mounting=MOUNTING)
    _still_until(rec, 10.5)
    result = E.check_for_mounting_in_convergence(rec)
    assert result.thresholds['onset_s'] == pytest.approx(10.5, abs=0.5)
    for p in result.per_segment.values():
        assert p['uncertainty_deg'] < 2
        assert np.degrees((Rotation.from_matrix(p['fit'].sensor_to_segment) * MOUNTING).magnitude()) < 0.5


def test_static_trial_gets_no_rotation():
    rec = make_recording(duration_s=30.0, mounting=MOUNTING)
    _still_until(rec, 30.0)
    result = E.check_for_mounting_in_convergence(rec)
    assert result.detected is None and result.thresholds['onset_s'] is None
    assert all(p['fit'] is None for p in result.per_segment.values())
    fixed = E.remove_mounting_misalignment(rec, result)
    assert np.array_equal(fixed.segments['upper'].imu.gyr, rec.segments['upper'].imu.gyr)


def test_little_movement_is_fitted_with_a_large_uncertainty():
    """Motion that stops shortly after onset: a window with only a few moving samples is fitted, but its uncertainty
    is large, so the weighted mean (fix_mounting) leans on the participant's other trials."""
    rec = make_recording(duration_s=30.0, mounting=MOUNTING)
    for seg in rec.segments.values():
        ref = seg.reference
        k0 = int(5.0 * ref.rate_hz)
        q = ref.orientation.as_quat()
        q[:k0] = q[k0]
        q[k0 + 10:] = q[k0 + 10]  # ten frames of motion at 5 s, then still to the end
        ref.orientation = Rotation.from_quat(q)
    result = E.check_for_mounting_in_convergence(rec)
    for p in result.per_segment.values():
        assert p['fit'] is not None and 3 <= p['moving_samples'] < 20 and p['uncertainty_deg'] > 20
    fixed = E.remove_mounting_misalignment(rec, result)
    assert not np.array_equal(fixed.segments['upper'].imu.gyr, rec.segments['upper'].imu.gyr)
