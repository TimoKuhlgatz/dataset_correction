"""Known-answer tests of the translation from a reader's Recording to this package's (_bridge.py): a synthetic
recording with an injected time offset is written in the readers' format, translated back, and the
offset check must recover the injected offset, also when the clock is irregular and has to be put on a grid."""
import numpy as np
import pytest

from mocapref import errors as E
from mocapref.datasets.readers.characterisation.schema import Pair, Pose
from mocapref.datasets.readers.characterisation.schema import Recording as ReaderRecording
from mocapref.datasets.readers.characterisation.schema import Segment as ReaderSegment
from mocapref.datasets.readers.characterisation.schema import SensorStream
from mocapref.datasets._bridge import to_recording
from mocapref.synthetic import make_recording


def as_reader_format(rec, epoch_s=50.0, imu_times=None):
    """The synthetic recording in the readers' format, on a time base starting at `epoch_s`."""
    segments = {}
    for name, seg in rec.segments.items():
        t_imu = epoch_s + (seg.imu.times if imu_times is None else imu_times)
        sensor = SensorStream(t_imu, seg.imu.gyr, seg.imu.acc, seg.imu.rate_hz, 'synthetic')
        matrices = seg.reference.orientation.as_matrix()
        matrices[~seg.reference.valid] = np.nan
        pose = Pose(epoch_s + seg.reference.times, matrices, np.zeros((len(matrices), 3)), seg.reference.rate_hz,
                    'housing')
        segments[name] = ReaderSegment(name, sensor, pose)
    segments['markers_only'] = ReaderSegment('markers_only', None, segments['upper'].reference)
    pairs = [Pair('upper', 'lower', 'hinge'), Pair('lower', 'markers_only', 'hinge')]
    return ReaderRecording('synthetic', 'S1', 'walk', segments, pairs)


@pytest.mark.parametrize('offset_s', [0.0, 0.04, -0.137])
def test_injected_offset_survives_the_translation(offset_s):
    rec = make_recording(duration_s=40.0, offset_s=offset_s, reference_rate_hz=120.0)
    out = to_recording(as_reader_format(rec), 'synthetic')
    assert out.trial == 'S1/walk' and out.participant == 'S1'
    assert list(out.segments) == ['upper', 'lower'] and out.joints == [('upper', 'lower')]
    assert out.metadata['segments_left_out'] == {'markers_only': 'no IMU'}
    for name, seg in rec.segments.items():
        new = out[name]
        assert new.imu.start_s == pytest.approx(0.0) and new.reference.start_s == pytest.approx(0.0)
        assert np.array_equal(new.imu.gyr, seg.imu.gyr)
        assert new.reference.convention == 'segment_to_world'
        assert np.allclose((new.reference.orientation.inv() * seg.reference.orientation).magnitude(), 0, atol=1e-9)
    check = E.check_for_synchronisation_offset(out)
    assert check.offset_s == pytest.approx(offset_s, abs=1e-3)


def test_streams_keep_their_own_start():
    rec = make_recording(duration_s=40.0)
    served = as_reader_format(rec)
    for seg in served.segments.values():
        if seg.sensor is not None:
            seg.sensor.t = seg.sensor.t + 0.25  # IMU stamps start 250 ms after the reference
    out = to_recording(served, 'synthetic')
    assert out['upper'].imu.start_s == pytest.approx(0.25) and out['upper'].reference.start_s == pytest.approx(0.0)
    assert E.check_for_synchronisation_offset(out).offset_s == pytest.approx(0.25, abs=1e-3)


def test_irregular_clock_is_put_on_a_grid_without_moving_the_offset():
    rec = make_recording(duration_s=40.0, offset_s=0.05)
    rng = np.random.default_rng(1)
    n = len(rec['upper'].imu)
    jitter = np.cumsum(rng.normal(0, 0.002, n))  # wanders off the uniform grid by many samples
    irregular = rec['upper'].imu.times + jitter - jitter[0]
    served = as_reader_format(rec, imu_times=irregular)
    for seg in served.segments.values():  # the reference shares the irregular stamps, as in RepoIMU
        if seg.sensor is not None:
            seg.reference.t = seg.sensor.t.copy()
    out = to_recording(served, 'synthetic')
    assert set(out.metadata['resampled_from_irregular_clock']) == {'upper.imu', 'upper.reference', 'lower.imu',
                                                                     'lower.reference'}
    # both streams were stamped on the same irregular clock, so they stay paired: the offset is the injected one
    assert E.check_for_synchronisation_offset(out).offset_s == pytest.approx(0.05, abs=2e-3)
