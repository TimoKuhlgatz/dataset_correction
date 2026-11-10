"""Known-answer tests for the motion onset: the convergence time and the
mounting window start when the reference first exceeds the standing limit, scoring starts 10 s later, and trials
too short after onset (or static) are not used."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref.effect import scoring_skip_s
from mocapref.motion_onset import long_enough, motion_onset_s, scoring_start_s
from mocapref.resample import resample_to_common_grid
from mocapref.effect import trial_start_s
from mocapref.synthetic import make_recording


def still_until(rec, t_s, jitter_deg=0.0, seed=0):
    """Reference at rest until t_s (with optional marker-like jitter of the stored orientation), moving afterwards."""
    rng = np.random.default_rng(seed)
    for seg in rec.segments.values():
        ref = seg.reference
        k = min(int(t_s * ref.rate_hz), len(ref) - 1)
        rest = Rotation.from_quat(np.repeat(ref.orientation[k].as_quat()[None], k, axis=0))
        if jitter_deg:
            rest = rest * Rotation.from_rotvec(np.radians(jitter_deg) * rng.standard_normal((k, 3)))
        q = ref.orientation.as_quat()
        q[:k] = rest.as_quat()
        ref.orientation = Rotation.from_quat(q)
        seg.imu.gyr[seg.imu.times < t_s] = 0.0


def test_onset_is_where_motion_starts():
    rec = make_recording(duration_s=40.0)
    still_until(rec, 6.0)
    assert motion_onset_s(rec) == pytest.approx(6.0, abs=0.5)
    assert scoring_start_s(rec) == pytest.approx(16.0, abs=0.5)


def test_moving_from_the_start_keeps_the_old_rule():
    rec = make_recording(duration_s=40.0)
    assert motion_onset_s(rec) == pytest.approx(0.0, abs=0.3)
    part = resample_to_common_grid(rec)
    assert scoring_skip_s(part) == pytest.approx(10.0, abs=0.3)  # 10 s after the start


def test_marker_jitter_at_rest_does_not_start_it():
    """Jitter of 0.3° per frame (rms per axis) makes frame-to-frame rates of tens of °/s at 100 Hz, above the
    fit's 17 °/s, but no net rotation over half a second above the standing limit."""
    rec = make_recording(duration_s=40.0)
    still_until(rec, 12.0, jitter_deg=0.3)
    assert motion_onset_s(rec) == pytest.approx(12.0, abs=0.5)


def test_scoring_skip_counts_from_the_onset():
    rec = make_recording(duration_s=40.0)
    still_until(rec, 6.0)
    part = resample_to_common_grid(rec)
    onset = motion_onset_s(rec)
    assert scoring_skip_s(part) == pytest.approx(onset - trial_start_s(part) + 10.0, abs=0.05)
    assert scoring_skip_s(part, onset_s=2.0) == pytest.approx(2.0 - trial_start_s(part) + 10.0, abs=0.05)


def test_trials_too_short_after_onset_or_static_are_not_used():
    rec = make_recording(duration_s=30.0)
    still_until(rec, 12.0)  # 18 s left after onset
    assert long_enough(rec, 'test')[0] is True
    ok, why = long_enough(rec, 'tuning')
    assert ok is False and 'tuning needs 20' in why
    static = make_recording(duration_s=30.0)
    still_until(static, 30.0)
    assert motion_onset_s(static) is None and scoring_start_s(static) is None
    ok, why = long_enough(static, 'test')
    assert ok is False and why.startswith('static')
    part = resample_to_common_grid(static)
    assert scoring_skip_s(part) > 30.0  # nothing scored
