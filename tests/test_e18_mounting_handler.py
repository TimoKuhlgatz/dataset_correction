"""Known-answer test for the dataset-level handler of error 18 (mounting rotation): three trials of one participant
with a known mounting; every trial-segment is counted, and the corrected loader rotates the IMU into the segment
frame (the gyroscope of the corrected trial matches the clean one)."""
import numpy as np
from scipy.spatial.transform import Rotation

from memory_loader import MemoryLoader
from mocapref.errors import e18_mounting as M
from mocapref.synthetic import make_recording

MOUNTING = Rotation.from_euler('xyz', [20, -10, 90], degrees=True)


def test_handler_counts_and_removes_the_mounting():
    recs = {f't{s}': make_recording(duration_s=30.0, seed=s, mounting=MOUNTING) for s in range(3)}
    loader = MemoryLoader(recs)
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (3, 3, 6)
    assert (rows.pool_trials == 2).all()  # each trial pooled with the participant's two others
    fixed = M.correct(loader, rows).load('t1')
    clean = make_recording(duration_s=30.0, seed=1)
    assert np.allclose(fixed['upper'].imu.gyr, clean['upper'].imu.gyr, atol=np.radians(1.0))
    assert np.allclose(fixed['lower'].imu.acc, clean['lower'].imu.acc, atol=0.2)
