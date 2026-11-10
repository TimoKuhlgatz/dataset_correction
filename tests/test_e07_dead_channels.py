"""Known-answer tests for the handler of error 7, dead, stuck or railed channel: on a small in-memory dataset a
stuck channel and a railed channel are counted and masked, samples an earlier correction already marked invalid are
not counted again, and RepoIMU is not checked. The rules themselves are tested in test_errors_known_answer.py (dead
and stuck channels) and test_e05_clipping.py (railed channels)."""
import numpy as np
from memory_loader import MemoryLoader

from mocapref.errors import e07_dead_channels as H
from mocapref.synthetic import make_recording


def noisy(seed=0):
    rec = make_recording(duration_s=60.0, imu_rate_hz=100.0, reference_rate_hz=120.0, linear_acc_mps2=5.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for seg in rec.segments.values():  # sensor noise, so a live channel never repeats exactly
        seg.imu.gyr += rng.normal(0, 0.005, seg.imu.gyr.shape)
        seg.imu.acc += rng.normal(0, 0.02, seg.imu.acc.shape)
    return rec


def railed(seed, rail):
    rec = noisy(seed)
    rec.segments['lower'].imu.acc[3000:3300, 0] = rail + np.random.default_rng(seed + 4).normal(0, 0.02, 300)
    return rec


def dataset():
    stuck = noisy(1)
    stuck.segments['upper'].imu.gyr[1000:1020, 1] = stuck.segments['upper'].imu.gyr[1000, 1]  # one run of 20
    rail = 1.5 * np.abs(noisy().segments['lower'].imu.acc).max()  # the same rail in both trials: it recurs
    return MemoryLoader({'stuck': stuck, 'clean': noisy(2), 'rail1': railed(0, rail), 'rail2': railed(10, rail)})


def test_stuck_runs_and_railed_channels_are_counted_and_masked():
    loader = dataset()
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_detected'], p['occurrences']) == (3, 3)  # one stuck run, one railed channel in each rail trial
    assert rows.set_index(['trial', 'segment']).loc[('stuck', 'upper'), 'stuck_runs'] == 1
    fixed = H.correct(loader, rows)
    up = fixed.load('stuck').segments['upper'].imu.valid
    assert not up[1000:1020].any() and up.sum() == len(up) - 20
    low = fixed.load('rail1').segments['lower'].imu.valid
    assert not low[3000:3300].any() and low.sum() >= len(low) - 310
    assert fixed.load('clean').corrections == []


def test_samples_masked_before_are_not_counted_again():
    loader = dataset()
    loader.recordings['stuck'].segments['upper'].imu.valid[1000:1020] = False  # e.g. masked as clipped by error 5
    rows = H.check_dataset(loader, 'synthetic', 1)
    assert rows.set_index(['trial', 'segment']).loc[('stuck', 'upper'), 'stuck_runs'] == 0


def test_repoimu_is_not_checked():
    loader = dataset()
    loader.name = 'repoimu'
    rows = H.check_dataset(loader, 'repoimu', 1)
    p = H.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (4, 0, 0)
    assert H.correct(loader, rows) is loader
