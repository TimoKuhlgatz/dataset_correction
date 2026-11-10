"""Static trials are neither tuned on nor tested: the scoring functions refuse them before any
filter runs, and a trial too short from its motion onset is refused for its role."""
import pytest

from mocapref.effect import NotScored, test_on_trial as score_test_trial, trial_grid
from mocapref.synthetic import make_recording


def _still(rec):
    for seg in rec.segments.values():
        ref = seg.reference
        ref.orientation = ref.orientation[[0] * len(ref)]
        seg.imu.gyr[:] = 0.0


def test_static_trial_is_refused_for_tuning_and_testing():
    rec = make_recording(duration_s=40.0)
    _still(rec)
    with pytest.raises(NotScored, match='static'):
        score_test_trial(rec, 1.0, 1.0)
    with pytest.raises(NotScored, match='static'):
        trial_grid(rec)


def test_trial_too_short_from_onset_is_refused():
    rec = make_recording(duration_s=12.0)
    with pytest.raises(NotScored, match='test needs'):
        score_test_trial(rec, 1.0, 1.0)
