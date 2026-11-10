"""Known-answer tests of the calibration-free estimator (zoe.py) and of its use in the effect harness."""
import numpy as np
import pytest

from mocapref import effect
from mocapref.synthetic import make_recording
from mocapref.zoe import COV_LNK_GRID, COV_W_GRID, PUBLISHED_COV_LNK, PUBLISHED_COV_W, lowpass

def test_clean_joint_gives_a_small_error_after_tuning():
    """Clean synthetic trials: tuned on two, the relative orientation of two unseen trials is recovered."""
    trials = [make_recording(duration_s=40.0, reference_rate_hz=100.0, seed=s, linear_acc_mps2=3.0) for s in range(4)]
    r, q, grid = effect.tune_on_trials(trials[:2], method='zoe')
    assert set(grid) == {(a, b) for a in COV_LNK_GRID for b in COV_W_GRID}
    assert grid[(r, q)] < 2.0
    for rec in trials[2:]:
        assert effect.test_on_trial(rec, r, q, method='zoe')['test_error_deg'] < 2.0


@pytest.mark.parametrize('rate', [60.0, 100.0, 200.0])
def test_low_pass_cutoff_is_6_hz_at_every_rate(rate):
    """A 2 Hz sine passes, a 6 Hz sine is halved (forward-backward Butterworth: -6 dB at the cutoff), a 15 Hz sine
    is removed, whatever the sampling rate."""
    t = np.arange(0, 20, 1 / rate)
    middle = slice(len(t) // 4, -len(t) // 4)  # away from the edges; gain as an RMS ratio (few samples per period)
    gain = {}
    for f in (2.0, 6.0, 15.0):
        x = np.sin(2 * np.pi * f * t)[:, None]
        gain[f] = np.sqrt(np.mean(lowpass(x, rate)[middle] ** 2) / np.mean(x[middle] ** 2))
    assert gain[2.0] == pytest.approx(1.0, abs=0.02)
    assert gain[6.0] == pytest.approx(0.5, abs=0.02)
    assert gain[15.0] < 0.01


def test_offset_correction_restores_the_clean_error_with_zoe():
    """The time-offset known-answer case of test_effect.py, run with the calibration-free estimator."""
    from mocapref import errors as E
    kw = dict(duration_s=60.0, reference_rate_hz=100.0, seed=1, linear_acc_mps2=3.0)

    def tuned_error(rec):
        r, q, _ = effect.tune_on_trials([rec], skip_s=10.0, method='zoe')
        return effect.test_on_trial(rec, r, q, skip_s=10.0, method='zoe')['test_error_deg']

    clean = tuned_error(make_recording(**kw))
    rec = make_recording(offset_s=0.05, **kw)
    check = E.check_for_synchronisation_offset(rec)
    assert check.detected
    assert tuned_error(E.remove_synchronisation_offset(rec, check)) == pytest.approx(clean, abs=0.05)
    assert tuned_error(rec) > clean + 3.0
    assert clean < 2.0


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        effect.test_on_trial(make_recording(duration_s=20.0), 1.0, 1.0, method='mekf2')


def test_a_diverging_grid_point_counts_as_worst_and_keeps_the_trial(monkeypatch):
    """A filter run that diverges (non-finite covariance, LinAlgError) gives an infinite error at that grid point;
    the trial still takes part with its other grid points (on Camargo, two trials had otherwise been dropped)."""
    from mocapref import zoe
    real_run = zoe.ZoeInputs.run

    def run(self, cov_lnk, cov_w):
        if cov_lnk == 1e-4:
            raise np.linalg.LinAlgError('Array must not contain infs or NaNs.')
        return real_run(self, cov_lnk, cov_w)

    monkeypatch.setattr(zoe.ZoeInputs, 'run', run)
    rec = make_recording(duration_s=30.0, reference_rate_hz=100.0, seed=0, linear_acc_mps2=3.0)
    weight, grid = effect.trial_grid(rec, r_grid=(1e-4, PUBLISHED_COV_LNK), q_grid=(PUBLISHED_COV_W,), method='zoe')
    assert weight > 0
    assert grid[(1e-4, PUBLISHED_COV_W)] == np.inf
    assert np.isfinite(grid[(PUBLISHED_COV_LNK, PUBLISHED_COV_W)])
