"""Known-answer checks of the Geneva raw rebuild reader (rule 3), on the data where they are on disk."""
import numpy as np
import pytest

from mocapref.datasets.grouvel_raw import REBUILD_DIR

pytestmark = pytest.mark.skipif(not (REBUILD_DIR / 'match.csv').exists(), reason='Geneva raw rebuild not on disk')


@pytest.fixture(scope='module')
def loaders():
    from mocapref.datasets import load_dataset
    return load_dataset('grouvel'), load_dataset('grouvel_raw')


def test_same_trials_segments_and_reference(loaders):
    deposited, rebuilt = loaders
    assert deposited.trials() == rebuilt.trials()
    a, b = deposited.load('P01/Gait_01'), rebuilt.load('P01/Gait_01')
    assert a.segment_names == b.segment_names
    for name in a.segment_names:
        assert np.allclose(a[name].reference.orientation.as_quat(), b[name].reference.orientation.as_quat())
        assert len(a[name].imu) == len(b[name].imu) and a[name].imu.start_s == b[name].imu.start_s


def test_clean_copy_trial_equals_the_deposit_shifted(loaders):
    """Where the deposit is a clean copy of the raw stream (deposit_corr >= 0.999), it is that stream shifted by the
    rebuild's deposit_shift_s (-0.3 to -1.1 s in Geneva): the rebuilt gyroscope moved by that shift reproduces the
    deposited one, so reader and rebuild read the same sensor."""
    import pandas as pd
    deposited, rebuilt = loaders
    match = pd.read_csv(REBUILD_DIR / 'match.csv')
    row = match[match.deposit_corr >= 0.999].iloc[0]
    trial = f'{row.subject}/{row.trial}'
    a, b = deposited.load(trial)[row.segment].imu, rebuilt.load(trial)[row.segment].imu
    k = int(round(row.deposit_shift_s * a.rate_hz))  # deposited row j = rebuilt row j + k
    j = np.arange(max(0, -k), min(len(a), len(b) - k))
    ok = a.valid[j] & b.valid[j + k]
    assert ok.sum() > 100
    assert np.corrcoef(a.gyr[j][ok].ravel(), b.gyr[j + k][ok].ravel())[0, 1] > 0.99


def test_rebuilt_gait_trial_matches_the_reference_where_the_deposit_does_not(loaders):
    from mocapref.errors import check_for_synchronisation_offset
    deposited, rebuilt = loaders
    peak = {name: np.median([e['quality'] for e in check_for_synchronisation_offset(loader.load('P01/Gait_01'))
                             .per_segment.values()]) for name, loader in zip(('deposited', 'rebuilt'), loaders)}
    assert peak['rebuilt'] > 0.9 and peak['deposited'] < 0.5
