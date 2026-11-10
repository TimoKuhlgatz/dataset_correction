"""Known-answer tests for the range-free clipping and railing check (e05_clipping.py): defects at
a level the check is not told are found, natural peaks, coarse storage and a mid-range plateau are not."""
import numpy as np
import pytest

from mocapref import errors as E
from mocapref.errors.e05_clipping import (RAIL_LIMIT, SENSORS, channel_noise, cut_off_samples, minimum_limits,
                                          side_plateaus)
from mocapref.synthetic import make_recording


def noisy(rate_hz=100.0, seed=0):
    rec = make_recording(duration_s=60.0, imu_rate_hz=rate_hz, reference_rate_hz=120.0, linear_acc_mps2=5.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for seg in rec.segments.values():
        seg.imu.gyr += rng.normal(0, 0.005, seg.imu.gyr.shape)
        seg.imu.acc += rng.normal(0, 0.02, seg.imu.acc.shape)
    return rec


def gait_like(rate_hz, rise_s=0.02, n_steps=60, seed=0):
    """Asymmetric peaks like heel strikes: a rise of width rise_s, a fall four times slower, of varying height,
    plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_steps, 1 / rate_hz)
    x = np.zeros_like(t)
    for k in range(n_steps):
        dt = t - k - 0.3
        x += rng.uniform(20, 40) * np.exp(-(np.minimum(dt, 0) / rise_s) ** 2 - np.maximum(dt, 0) / (4 * rise_s))
    return x + rng.normal(0, 0.05, len(t))


@pytest.mark.parametrize('rate_hz', [60.0, 100.0, 200.0])
def test_clean_recording_has_no_cut_off_peak(rate_hz):
    result = E.check_for_cut_off_peaks(noisy(rate_hz))
    assert result.detected is False
    assert all(p['estimate'] == 0 for p in result.per_segment.values())


@pytest.mark.parametrize('rate_hz', [60.0, 100.0, 200.0])
def test_clipping_at_an_unknown_calibrated_level_is_found(rate_hz):
    rec = noisy(rate_hz)
    imu = rec.segments['upper'].imu
    rng = np.random.default_rng(2)
    true = imu.gyr.copy()
    limit = 0.8 * true[:, 1].max()  # the range holds for every axis of the sensor
    over = (np.abs(true) > limit).any(axis=1)
    imu.gyr = np.clip(true, -limit, limit) * 1.02 + 0.01 + rng.normal(0, 0.005, true.shape)  # calibrated after
    result = E.check_for_cut_off_peaks(rec)
    up = result.per_segment['upper']
    assert up['detected'] and up['railed'] == 0
    clipped, railed, _ = cut_off_samples(np.hstack([imu.gyr, imu.acc]), rate_hz, sensors=SENSORS)
    assert (clipped & over).sum() >= 0.9 * over.sum()  # nearly every clipped sample found
    extra = clipped & ~over  # the others are samples whose true value lies within the noise band of the limit
    band = 2 * 3.0 * max(channel_noise(imu.gyr[:, k], rate_hz) for k in range(3)) / 1.02
    assert (np.abs(true[extra]).max(axis=1) > limit - band).all()
    assert not result.per_segment['lower']['detected']
    fixed = E.remove_cut_off_peaks(rec)
    assert fixed.segments['upper'].imu.valid.sum() == len(imu) - up['estimate']


@pytest.mark.parametrize('rate_hz, rise_s', [(1000.0, 0.02), (100.0, 0.08), (60.0, 0.1)])
def test_resolved_asymmetric_peaks_are_not_flagged_and_clipped_ones_are(rate_hz, rise_s):
    x = gait_like(rate_hz, rise_s)  # the rise spans at least 6 samples
    other = 0.5 * gait_like(rate_hz, rise_s, seed=1)  # a second channel of the same IMU, moving with it
    clipped, railed, _ = cut_off_samples(np.column_stack([x, other]), rate_hz)
    assert not (clipped | railed).any()
    over = x > 25.0
    clipped, railed, _ = cut_off_samples(np.column_stack([np.minimum(x, 25.0), other]), rate_hz)
    assert clipped[over].mean() >= 0.8 and not railed.any()


@pytest.mark.parametrize('rate_hz', [60.0, 100.0])
def test_unresolved_peaks_are_undecided_not_flagged(rate_hz):
    x = np.minimum(gait_like(rate_hz, 0.02), 25.0)  # a rise of 1-2 samples: the shape cannot be read
    other = 0.5 * gait_like(rate_hz, 0.02, seed=1)
    clipped, railed, records = cut_off_samples(np.column_stack([x, other]), rate_hz)
    assert not (clipped | railed).any()
    top = [r for r in records if r['side'] == 'positive' and r['axis'] == 0]
    assert sum(r['verdict'] == 'undecided' for r in top) >= 0.8 * len(top)
    assert top[0]['plateaus_at_level'] >= 10  # the limit recurs: reported as evidence


def railed_once(seed=0):
    rec = noisy(seed=seed)
    imu = rec.segments['lower'].imu
    rail = 1.5 * np.abs(noisy().segments['lower'].imu.acc).max()  # the same limit for every trial, beyond every peak
    imu.acc[3000:3300, 0] = rail + np.random.default_rng(seed + 4).normal(0, 0.02, 300)  # jumps to it for 3 s
    return rec


def test_a_railed_stretch_is_found_and_called_railed():
    rec = railed_once()
    imu = rec.segments['lower'].imu
    imu.acc[4000:4300, 0] = imu.acc[3000:3300, 0]  # reached a second time: the level recurs
    result = E.check_for_cut_off_peaks(rec).per_segment['lower']
    assert result['detected'] and result['railed'] >= 590 and 'acc0p' in result['channels']
    imu = noisy().segments['lower'].imu  # driven into the limit twice: the true value rises past it and stays
    rng = np.random.default_rng(4)
    true = imu.acc[:, 1].copy()
    limit = np.quantile(true, 0.9)
    imu.acc[:, [0, 2]] = np.clip(imu.acc[:, [0, 2]], -limit, limit)  # the range holds for every axis
    beyond = np.zeros(len(true), bool)
    for a in (3000, 4000):
        true[a:a + 300] += np.linspace(0, 3 * np.abs(true).max(), 300)
        beyond[a:a + 300] = true[a:a + 300] > limit
    imu.acc[:, 1] = np.minimum(true, limit) + rng.normal(0, 0.02, len(true))
    clipped, railed, _ = cut_off_samples(np.hstack([imu.gyr, imu.acc]), 100.0)
    assert (railed | clipped)[beyond].mean() >= 0.95 and railed.sum() >= 400


def test_a_level_reached_once_counts_only_when_another_trial_of_the_participant_reaches_it():
    first, second = railed_once(0), railed_once(10)
    alone = E.check_for_cut_off_peaks(first).per_segment['lower']
    assert not alone['detected'] and alone['single'] >= 1  # reported, not flagged
    others = {n: p['candidates'] for n, p in E.check_for_cut_off_peaks(second).per_segment.items()}
    together = E.check_for_cut_off_peaks(first, other_levels=others).per_segment['lower']
    assert together['detected'] and together['railed'] >= 295 and together['channels'] == ['acc0p']
    assert E.remove_cut_off_peaks(first, others).segments['lower'].imu.valid.sum() == len(first.segments['lower'].imu) - together['estimate']


def test_a_level_that_another_peak_of_the_same_sensor_exceeds_in_the_trial_is_not_a_limit():
    rec = noisy()
    imu = rec.segments['upper'].imu
    limit = 0.8 * imu.gyr[:, 1].max()
    imu.gyr[:, :2] = np.clip(imu.gyr[:, :2], -limit, limit) + np.random.default_rng(9).normal(0, 0.005, (len(imu), 2))
    imu.gyr[:, 2] = imu.gyr[:, 2] * (1.5 * limit / np.abs(imu.gyr[:, 2]).max())  # another axis goes 1.5 x beyond it
    result = E.check_for_cut_off_peaks(rec).per_segment['upper']
    assert not result['detected'] and result['exceeded'] >= 1
    imu.gyr[:, 2] *= 1.04 / 1.5  # 4 % beyond it: within the calibration difference between axes, still a limit
    assert E.check_for_cut_off_peaks(rec).per_segment['upper']['detected']


def test_a_limit_closer_to_zero_than_any_range_is_not_flagged():
    rec = noisy()
    imu = rec.segments['upper'].imu
    small = 0.5 * np.deg2rad(125.0)  # a gyroscope range of 62.5 deg/s does not exist
    imu.gyr[:, 1] = np.clip(imu.gyr[:, 1], -small, small) + np.random.default_rng(7).normal(0, 0.005, len(imu))
    clipped, _, _ = cut_off_samples(np.hstack([imu.gyr, imu.acc]), 100.0)
    assert clipped.sum() > 100  # the shape alone calls it clipped
    assert not E.check_for_cut_off_peaks(rec).per_segment['upper']['detected']


def test_rest_at_the_minimum_of_a_one_sided_channel_is_not_clipping():
    x, other = gait_like(60.0, 0.02), 0.5 * gait_like(60.0, 0.02, seed=1)  # both rest at 0 between the peaks
    _, _, alone = cut_off_samples(x[:, None], 60.0)  # without the other channel, the rest passes the shape test
    assert sum(r['verdict'] in ('cut off', 'single', 'exceeded') and r['side'] == 'negative' for r in alone) >= 50
    clipped, railed, records = cut_off_samples(np.column_stack([x, other]), 60.0)
    assert not (clipped | railed).any()
    assert sum(r['reason'] == 'the whole IMU is quiet (rest)' for r in records) >= 50


def test_coarse_storage_is_not_clipping():
    rec = noisy()
    for seg in rec.segments.values():
        seg.imu.gyr = np.round(seg.imu.gyr / 0.02) * 0.02  # stored in steps 4 times the noise
        seg.imu.acc = np.round(seg.imu.acc / 0.1) * 0.1
    assert channel_noise(rec.segments['upper'].imu.gyr[:, 0], 100.0) >= 0.02 / np.sqrt(12)
    assert E.check_for_cut_off_peaks(rec).detected is False


def test_a_plateau_without_flanks_is_undecided_not_flagged():
    x = np.concatenate([np.full(50, 10.0), np.linspace(10, 0, 50)]) + np.random.default_rng(5).normal(0, 0.01, 100)
    records = side_plateaus(x, 0.01, 1.0, 100.0)
    assert records and all(r['verdict'] == 'undecided' for r in records)


def test_a_channel_railed_for_the_whole_trial_is_found():
    rec = noisy()
    n = len(rec.segments['lower'].imu)
    rec.segments['lower'].imu.acc[:, 2] = 16 * 9.81 + np.random.default_rng(6).normal(0, 0.05, n)
    result = E.check_for_cut_off_peaks(rec).per_segment['lower']
    assert result['detected'] and result['whole_trial'] == ['acc2p'] and result['railed'] >= 0.9 * n
    assert not E.check_for_cut_off_peaks(noisy()).per_segment['lower']['whole_trial']


def test_a_channel_still_at_gravity_while_the_imu_turns_about_it_is_not_railed():
    """A sensor whose z axis points up and that turns only about that axis reads 1 g on z for the whole trial while
    its gyroscope z moves: no rail (found on real upright axes with a 1 g minimum)."""
    rng = np.random.default_rng(8)
    t = np.arange(0, 60, 0.01)
    gyr = np.column_stack([rng.normal(0, 0.005, len(t)), rng.normal(0, 0.005, len(t)), 3 * np.sin(2 * np.pi * 0.5 * t)])
    acc = np.column_stack([rng.normal(0, 0.02, len(t)), rng.normal(0, 0.02, len(t)), 9.81 + rng.normal(0, 0.02, len(t))])
    limits = minimum_limits()
    clipped, railed, _ = cut_off_samples(np.hstack([gyr, acc]), 100.0, limits, None, minimum_limits(
        RAIL_LIMIT))
    assert not (clipped | railed).any()
    _, railed, _ = cut_off_samples(np.hstack([gyr, acc]), 100.0, limits, None, limits)  # at 1 g it would be railed
    assert railed.sum() > 0.9 * len(t)


def test_handler_counts_clipped_plateaus_and_masks_only_clipping():
    """check_dataset / prevalence / correct on a small in-memory dataset: one clipped trial, one clean trial and two
    trials railed at the same level (railed channels belong to error 7: counted there, not here)."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e05_clipping as H

    clipped = noisy(seed=3)
    imu = clipped.segments['upper'].imu
    limit = 0.8 * np.abs(imu.gyr).max()
    imu.gyr = np.clip(imu.gyr, -limit, limit) + np.random.default_rng(2).normal(0, 0.005, imu.gyr.shape)
    loader = MemoryLoader({'clipped': clipped, 'clean': noisy(seed=5), 'rail1': railed_once(0), 'rail2': railed_once(10)})
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    hit = rows[rows.detected == True]  # noqa: E712
    assert p['trials_detected'] == 1 and set(hit.trial) == {'clipped'} and p['occurrences'] == hit.estimate.sum() > 0
    assert rows[rows.trial.isin(['rail1', 'rail2'])].railed_channels.sum() == 2
    fixed = H.correct(loader, rows)
    up = hit[hit.segment == 'upper'].iloc[0]
    assert fixed.load('clipped').segments['upper'].imu.valid.sum() == len(imu) - up.clipped_samples
    assert fixed.load('rail1').segments['lower'].imu.valid.all()  # railed samples are masked by error 7
