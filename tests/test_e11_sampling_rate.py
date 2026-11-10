"""Known-answer tests for the two checks of a sampling rate not as declared (error 11):
a stretched time axis is recovered from the reference, and an upsampled stream is recognised from the IMU alone."""
import numpy as np
import pytest
from scipy.signal import resample

from mocapref.errors.e11_sampling_rate import check_for_effective_rate, check_for_time_scale, remove_time_scale
from mocapref.synthetic import make_recording


def declared(true_hz, declared_hz, duration_s=60.0):
    rec = make_recording(duration_s=duration_s, imu_rate_hz=true_hz, reference_rate_hz=120.0, offset_s=0.3)
    for seg in rec.segments.values():
        seg.imu.rate_hz = declared_hz
    return rec


def test_declared_rate_is_not_flagged():
    result = check_for_time_scale(declared(100.0, 100.0))
    assert result.detected is False and result.factor == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize('true_hz, declared_hz', [(240.0, 100.0), (100.0, 150.0), (148.148, 2000.0), (100.0, 103.0)])
def test_stretched_time_axis_is_recovered_and_removed(true_hz, declared_hz):
    rec = declared(true_hz, declared_hz, 40.0 if declared_hz > 1000 else 60.0)
    result = check_for_time_scale(rec)
    assert result.detected
    assert result.factor == pytest.approx(true_hz / declared_hz, rel=5e-3)
    fixed = remove_time_scale(rec, result)
    assert check_for_time_scale(fixed).detected is False
    assert rec.segments['upper'].imu.rate_hz == declared_hz  # the original is untouched


@pytest.mark.parametrize('segments, expected', [(('a', 'b', 'c', 'd'), True), (('a', 'b'), None)])
def test_agreeing_segments_are_trusted_below_the_correlation_threshold(segments, expected):
    rec = make_recording(duration_s=60.0, imu_rate_hz=240.0, reference_rate_hz=120.0, segment_names=segments)
    rng = np.random.default_rng(0)
    for seg in rec.segments.values():
        seg.imu.rate_hz = 100.0
        seg.imu.gyr = seg.imu.gyr + rng.normal(0, 0.5, seg.imu.gyr.shape)  # peak correlation about 0.88
    result = check_for_time_scale(rec, method='peak')  # the former rule's agreeing-segments clause
    assert all(0.7 <= p['quality'] < 0.9 for p in result.per_segment.values())
    assert result.detected is expected  # four agreeing segments are trusted, two are not enough
    if expected:
        assert result.factor == pytest.approx(2.4, rel=5e-3)


def upsampled(source_hz, target_hz, how, seed=2):
    rec = make_recording(duration_s=60.0, imu_rate_hz=source_hz, seed=seed)
    rng = np.random.default_rng(seed)
    for seg in rec.segments.values():
        imu = seg.imu
        gyr = imu.gyr + rng.normal(0, 0.005, imu.gyr.shape)  # sensor noise, so a real stream is not smooth
        acc = imu.acc + rng.normal(0, 0.05, imu.acc.shape)
        n = len(gyr)
        m = int(n * target_hz / source_hz)
        t0, t1 = np.arange(n) / source_hz, np.arange(m) / target_hz

        def convert(x):
            if how == 'hold':
                return x[np.minimum((t1 * source_hz + 1e-9).astype(int), n - 1)]
            if how == 'linear':
                return np.column_stack([np.interp(t1, t0, x[:, i]) for i in range(3)])
            if how == 'fft':
                return resample(x, m, axis=0)
            return x

        imu.gyr, imu.acc, imu.rate_hz, imu.valid = convert(gyr), convert(acc), target_hz, np.ones(m, bool)
    return rec


def test_a_noisy_stream_at_its_own_rate_is_not_flagged():
    assert check_for_effective_rate(upsampled(100.0, 100.0, 'none')).detected is False


@pytest.mark.parametrize('source_hz, target_hz, how, measure, outcome', [
    (25.0, 100.0, 'hold', 'hold_fraction', 'below_declared'),
    (148.148, 2000.0, 'linear', 'linear_fraction', 'below_declared'),
    (60.0, 100.0, 'linear', 'line_ratio', 'resampled'),
    (60.0, 100.0, 'fft', 'top_band_ratio', 'below_declared')])
def test_resampled_streams_are_recognised(source_hz, target_hz, how, measure, outcome):
    result = check_for_effective_rate(upsampled(source_hz, target_hz, how))
    assert result.detected and getattr(result, outcome)
    for seg in result.per_segment.values():
        m = seg[measure]
        hit = {'hold_fraction': m >= 0.05, 'linear_fraction': m >= 0.2, 'line_ratio': m >= 200,
               'top_band_ratio': m < 1e-6}[measure]
        assert hit and seg[outcome], (measure, m)
    if how == 'hold':
        assert result.per_segment['upper']['estimate'] == pytest.approx(source_hz, rel=0.01)


def test_resampling_down_from_148_hz_leaves_a_weak_line_at_its_alias():
    """Known limit (as ULTRA): interpolation from 148.148 Hz onto 100 Hz leaves its line at the alias 48.15 Hz, but
    too weak for the threshold of 200."""
    result = check_for_effective_rate(upsampled(148.148, 100.0, 'linear'))
    for seg in result.per_segment.values():
        assert seg['line_hz'] == pytest.approx(148.148 - 100.0, abs=0.05)
        assert seg['line_ratio'] < 200 and not seg['resampled']


def test_samples_marked_invalid_are_not_counted_as_repeated():
    """Dead and stuck samples are removed first (agreed order of correction); a stuck stretch marked invalid must not
    count again as repeated samples, and no difference may bridge the gap."""
    rec = upsampled(100.0, 100.0, 'none')
    seg = rec.segments['upper']
    seg.imu.gyr[1000:2500] = seg.imu.gyr[1000]
    seg.imu.acc[1000:2500] = seg.imu.acc[1000]  # the whole IMU stuck for 15 s (25 % of the samples)
    assert check_for_effective_rate(rec).per_segment['upper']['below_declared']
    seg.imu.valid[1000:2500] = False
    measures = check_for_effective_rate(rec).per_segment['upper']
    assert measures['hold_fraction'] == 0.0 and not measures['below_declared']


def test_the_resampling_line_is_undecided_on_short_recordings():
    """Below about 12.5 s the +-1 Hz neighbourhood of a line holds fewer than 20 frequency points: undecided."""
    rec = make_recording(duration_s=5.0, imu_rate_hz=60.0, seed=3)
    for seg in rec.segments.values():  # 60 -> 100 Hz linear: a clear line on a long recording
        t0, t1 = np.arange(len(seg.imu)) / 60.0, np.arange(int(len(seg.imu) * 100 / 60)) / 100.0
        seg.imu.gyr = np.column_stack([np.interp(t1, t0, seg.imu.gyr[:, i]) for i in range(3)])
        seg.imu.acc = np.column_stack([np.interp(t1, t0, seg.imu.acc[:, i]) for i in range(3)])
        seg.imu.rate_hz, seg.imu.valid = 100.0, np.ones(len(t1), bool)
    result = check_for_effective_rate(rec)
    assert result.resampled is None
    assert all(p['resampled'] is None and p['neighbour_bins'] < 20 for p in result.per_segment.values())


@pytest.mark.parametrize('true_hz, declared_hz', [(100.0, 100.0), (240.0, 100.0), (100.0, 103.0)])
def test_rank_block_rule_recovers_the_time_scale(true_hz, declared_hz):
    """method='rank-blocks': the factor is found and decided as with the former rule."""
    rec = declared(true_hz, declared_hz)
    result = check_for_time_scale(rec, method='rank-blocks')
    assert result.factor == pytest.approx(true_hz / declared_hz, rel=5e-3)
    assert result.detected is (true_hz != declared_hz)


def test_rank_block_rule_ignores_a_corrupted_sample():
    rec = declared(240.0, 100.0)
    for seg in rec.segments.values():
        seg.imu.gyr[1000, 0] = np.radians(5000.0)
    result = check_for_time_scale(rec, method='rank-blocks')
    assert result.detected and result.factor == pytest.approx(2.4, rel=5e-3)


def test_handler_counts_and_corrects_a_stretched_trial():
    """Dataset level: one stretched and one clean trial; the stretched one counts with both IMUs and is rescaled."""
    from memory_loader import MemoryLoader
    from mocapref.errors import e11_sampling_rate as M
    loader = MemoryLoader({'clean': declared(100.0, 100.0, 40.0), 'stretched': declared(240.0, 100.0, 40.0)})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 1, 2)
    fixed = M.correct(loader, rows)
    assert fixed.load('stretched')['upper'].imu.rate_hz == pytest.approx(240.0, rel=5e-3)
    assert fixed.load('clean')['upper'].imu.rate_hz == 100.0
    assert check_for_time_scale(fixed.load('stretched')).detected is False
