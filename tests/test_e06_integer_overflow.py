"""Known-answer tests for the range-free integer overflow check (e06_integer_overflow.py): wraps at a
range the check is not told are found and unwrapped exactly; clean data, clipping, a single spike, a natural fast
reversal and a step are not flagged; a wrap that runs to the end of the trial is undecided."""
import numpy as np
import pytest

from mocapref import errors as E
from mocapref.errors.e06_integer_overflow import overflow_stretches
from mocapref.synthetic import make_recording

R_GYR = np.deg2rad(500.0)  # a range the check is not told
R_ACC = 4 * 9.81


def wrap(x, r):
    """What a signed integer of range +-r stores for the true value x."""
    return (x + r) % (2 * r) - r


def bursts(rate_hz, peak, n=6, width_s=0.3, period_s=4.0, sign=1.0, seed=0, noise=0.01):
    """Smooth bursts reaching `peak` (beyond the range) on a slow background, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, n * period_s, 1 / rate_hz)
    x = 0.2 * peak * np.sin(2 * np.pi * 0.3 * t)
    for k in range(n):
        x += sign * peak * rng.uniform(0.9, 1.0) * np.exp(-((t - (k + 0.5) * period_s) / width_s) ** 2)
    return x + rng.normal(0, noise, len(t))


@pytest.mark.parametrize('rate_hz', [60.0, 100.0, 200.0, 1000.0])
@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_wraps_at_an_unknown_range_are_found_on_both_sides(rate_hz, sign):
    true = bursts(rate_hz, 1.4 * R_GYR, sign=sign)
    stored = wrap(true, R_GYR)
    found = [s for s in overflow_stretches(stored, rate_hz, np.deg2rad(125)) if s['verdict'] == 'overflow']
    wrapped = np.abs(true) > R_GYR
    assert len(found) >= 5  # 6 bursts, all beyond the range (a burst at 0.9 * 1.4 R may be cut in two by noise)
    flagged = np.zeros(len(true), bool)
    for s in found:
        assert s['side'] == sign
        assert abs(s['range'] - R_GYR) < 0.02 * R_GYR
        flagged[s['first']:s['first'] + s['samples']] = True
    assert flagged.sum() >= 0.95 * wrapped.sum() and not (flagged & ~wrapped).any()


@pytest.mark.parametrize('rate_hz', [60.0, 200.0])
def test_unwrapping_restores_the_true_signal(rate_hz):
    rec = make_recording(duration_s=24.0, imu_rate_hz=rate_hz, seed=3)
    seg = next(iter(rec.segments.values()))
    n = len(seg.imu.gyr)
    true_gyr = bursts(rate_hz, 1.3 * R_GYR, seed=1)[:n]
    true_acc = bursts(rate_hz, 1.3 * R_ACC, sign=-1.0, seed=2, noise=0.05)[:n]
    seg.imu.gyr[:, 0], seg.imu.acc[:, 2] = wrap(true_gyr, R_GYR), wrap(true_acc, R_ACC)
    result = E.check_for_integer_overflow(rec)
    assert result.detected is True
    fixed = E.remove_integer_overflow(rec)  # unwrap is the default correction
    g = next(iter(fixed.segments.values())).imu
    assert np.abs(g.gyr[:, 0] - true_gyr).max() < 0.01 * R_GYR and np.abs(g.acc[:, 2] - true_acc).max() < 0.01 * R_ACC
    masked = next(iter(E.remove_integer_overflow(rec, mode='mask').segments.values())).imu
    flagged = np.abs(true_gyr) > R_GYR
    flagged |= np.abs(true_acc) > R_ACC
    assert (~masked.valid & ~flagged).sum() == 0 and (~masked.valid).sum() >= 0.95 * flagged.sum()
    with pytest.raises(ValueError):
        E.remove_integer_overflow(rec, mode='guess')


@pytest.mark.parametrize('rate_hz', [60.0, 100.0, 200.0])
def test_clean_recording_has_no_overflow(rate_hz):
    rec = make_recording(duration_s=60.0, imu_rate_hz=rate_hz, linear_acc_mps2=5.0, seed=4)
    assert E.check_for_integer_overflow(rec).detected is False


@pytest.mark.parametrize('rate_hz', [60.0, 200.0])
def test_clipping_is_not_overflow(rate_hz):
    clipped = np.clip(bursts(rate_hz, 1.4 * R_GYR), -R_GYR, R_GYR)
    assert not overflow_stretches(clipped, rate_hz, np.deg2rad(125))


def test_single_spike_of_opposite_sign_is_not_overflow():
    x = bursts(100.0, 0.95 * R_GYR)
    k = int(np.argmax(x))
    x[k + 1] = -0.7 * x[k]  # one corrupted sample on the opposite side, not a full span away
    # (a sample flipped to exactly -x[k] at the channel's peak cannot be told from a one-sample wrap whose true value
    # lies at the range limit; see the module docstring)
    assert not [s for s in overflow_stretches(x, 100.0, np.deg2rad(125)) if s['verdict'] == 'overflow']


@pytest.mark.parametrize('rate_hz', [60.0, 100.0])
def test_natural_fast_reversal_is_not_overflow(rate_hz):
    """An oscillation that swings from near +M to near -M between two samples, then moves on normally."""
    t = np.arange(0, 20, 1 / rate_hz)
    x = 8.0 * np.sin(2 * np.pi * (rate_hz / 2.2) * t) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    x += np.random.default_rng(0).normal(0, 0.01, len(t))
    assert not [s for s in overflow_stretches(x, rate_hz, np.deg2rad(125)) if s['verdict'] == 'overflow']


def test_step_without_return_is_at_most_undecided():
    t = np.arange(0, 20, 0.01)
    x = np.where(t < 10, 8.0, -8.0) + np.random.default_rng(0).normal(0, 0.01, len(t))
    assert all(s['verdict'] == 'undecided' for s in overflow_stretches(x, 100.0, np.deg2rad(125)))


def test_wrap_running_to_the_end_of_the_trial_is_undecided():
    true = bursts(100.0, 1.4 * R_GYR, n=3)
    t = np.arange(150) / 100.0
    true = np.concatenate([true, true[-1] + (1.3 * R_GYR - true[-1]) * (1 - np.exp(-t / 0.2))])  # still beyond the range at the end
    stored = wrap(true, R_GYR)
    verdicts = [s['verdict'] for s in overflow_stretches(stored, 100.0, np.deg2rad(125))]
    assert verdicts.count('undecided') == 1 and verdicts.count('overflow') >= 2


def test_range_smaller_than_any_sensor_is_not_overflow():
    true = bursts(100.0, 1.4 * np.deg2rad(50.0))  # wraps at 50 deg/s, below the smallest gyroscope range
    assert not overflow_stretches(wrap(true, np.deg2rad(50.0)), 100.0, np.deg2rad(125))


def test_handler_counts_stretches_and_unwraps_from_the_rows():
    import pandas as pd
    from memory_loader import MemoryLoader
    from mocapref.errors import e06_integer_overflow as H

    rec = make_recording(duration_s=24.0, imu_rate_hz=100.0, seed=3)
    wrapped = rec.copy()
    seg = next(iter(wrapped.segments.values()))
    true_gyr = bursts(100.0, 1.3 * R_GYR, seed=1)[:len(seg.imu)]
    seg.imu.gyr[:, 0] = wrap(true_gyr, R_GYR)
    loader = MemoryLoader({'clean': rec, 'wrapped': wrapped})
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    expected = sum(s['verdict'] == 'overflow' for s in overflow_stretches(seg.imu.gyr[:, 0], 100.0, np.deg2rad(125)))
    assert (p['trials_detected'], p['occurrences'], p['occurrence_unit']) == (1, expected, 'stretches')
    rows = pd.read_csv(pd.io.common.StringIO(rows.to_csv(index=False)), keep_default_na=False, na_values=[''])
    fixed = next(iter(H.correct(loader, rows).load('wrapped').segments.values())).imu
    assert np.abs(fixed.gyr[:, 0] - true_gyr).max() < 0.01 * R_GYR
    assert H.correct(loader, rows).load('clean').corrections == []
