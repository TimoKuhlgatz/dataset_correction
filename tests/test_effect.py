"""Known-answer tests of the effect harness: on a synthetic joint with an injected timing error, the corrected trial
reaches the error of the clean trial and the uncorrected one is clearly worse."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocapref import errors as E
from mocapref import effect
from mocapref.effect import relative_orientation_error_deg
from mocapref.synthetic import make_recording

SKIP_S = 10.0  # scored from 10 s after the trial start (the synthetic trials move from the start)


def tuned_error(rec, tuning=None, method='MEKF'):
    """Test error of `rec` with (R, Q) tuned on `tuning` (default: on `rec` itself)."""
    r, q, _ = effect.tune_on_trials([rec if tuning is None else tuning], skip_s=SKIP_S, method=method)
    return effect.test_on_trial(rec, r, q, skip_s=SKIP_S, method=method)['test_error_deg']


def test_error_angle_of_a_known_rotation():
    a = Rotation.from_euler('z', [[0], [10], [30]], degrees=True)
    assert np.allclose(relative_orientation_error_deg(a, Rotation.identity(3)), [0, 10, 30])


@pytest.mark.parametrize('tuned_on_clean', [False, True])
def test_offset_correction_restores_the_clean_error(tuned_on_clean):
    kw = dict(duration_s=60.0, reference_rate_hz=100.0, seed=1, linear_acc_mps2=3.0)
    clean_rec = make_recording(**kw)
    tuning = clean_rec if tuned_on_clean else None
    clean = tuned_error(clean_rec)
    rec = make_recording(offset_s=0.05, **kw)
    check = E.check_for_synchronisation_offset(rec)
    assert check.detected
    corrected = E.remove_synchronisation_offset(rec, check)
    assert tuned_error(corrected, tuning) == pytest.approx(clean, abs=0.05)
    assert tuned_error(rec, tuning) > clean + 3.0
    assert clean < 2.0


def test_drift_correction_restores_the_clean_error():
    """1000 ppm drift with zero offset at mid-trial: removing the drift alone must give the clean error."""
    kw = dict(duration_s=120.0, reference_rate_hz=100.0, seed=1, linear_acc_mps2=3.0)
    clean = tuned_error(make_recording(**kw))
    rec = make_recording(drift_ppm=1000.0, offset_s=-1000e-6 * 60.0, **kw)  # offset -60 ms at 0 s, +60 ms at 120 s
    check = E.check_for_clock_drift(rec)
    assert check.detected and check.drift_ms_per_min == pytest.approx(60.0, abs=1.0)
    assert tuned_error(E.remove_clock_drift(rec, check)) == pytest.approx(clean, abs=0.1)
    assert tuned_error(rec) > clean + 2.0


def test_rate_mismatch_correction_restores_the_clean_error():
    """One IMU of a joint 1000 ppm off the others (zero at mid-trial): aligning it alone gives the clean error."""
    kw = dict(duration_s=120.0, reference_rate_hz=100.0, seed=1, linear_acc_mps2=3.0, segment_names=('a', 'b', 'c'))
    clean = tuned_error(make_recording(**kw))
    rec = make_recording(segment_drift_ppm={'b': 1000.0}, **kw)
    check = E.check_for_sensor_rate_mismatch(rec)
    assert [n for n, p in check.per_segment.items() if p['detected']] == ['b']
    assert tuned_error(E.remove_sensor_rate_mismatch(rec, check)) == pytest.approx(clean, abs=0.1)
    assert tuned_error(rec) > clean + 2.0
