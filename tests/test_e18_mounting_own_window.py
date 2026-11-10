"""Known-answer test for error 18, mounting rotation, on a segment that stands still through the
trial's fit window and has no other trial to pool with: it is fitted on
its own first 10 s of motion, the known mounting is recovered, and its joints are scored only from the end of that
window."""
import numpy as np
from scipy.spatial.transform import Rotation

from mocapref.effect import _joint_runs, segment_window_skip_s
from mocapref.errors.e18_mounting import mount_trial
from mocapref.motion_onset import SEGMENT_WINDOWS_KEY, motion_onset_s
from mocapref.resample import resample_to_common_grid
from mocapref.rotations import best_rotation
from mocapref.synthetic import make_recording

MOUNTING = Rotation.from_euler('xyz', [30, -50, 120], degrees=True)
STILL_S = 30.0


def late_segment(still_s=STILL_S):
    """'upper' moves from the start; 'lower' stands still for `still_s`, then moves (its first pose held before)."""
    rec = make_recording(duration_s=80.0, mounting=MOUNTING, seed=3)
    seg = rec.segments['lower']
    n = int(still_s * seg.imu.rate_hz)
    k = np.maximum(np.arange(len(seg.imu)) - n, 0)
    gyr = seg.imu.gyr[k].copy()
    gyr[:n] = 0.0
    seg.imu.gyr, seg.imu.acc = gyr, seg.imu.acc[k].copy()
    kr = np.maximum(np.arange(len(seg.reference)) - int(still_s * seg.reference.rate_hz), 0)
    seg.reference.orientation = seg.reference.orientation[kr]
    return rec


def test_still_segment_is_fitted_on_its_own_window_and_scored_after_it():
    rec = late_segment()
    onset = motion_onset_s(rec)
    assert onset < 1.0
    fixed = mount_trial(rec)
    windows = fixed.metadata[SEGMENT_WINDOWS_KEY]
    assert set(windows) == {'lower'}  # 'upper' moves in the trial's window and keeps it
    start, end = windows['lower']
    assert STILL_S - 1.0 <= start <= STILL_S + 1.0 and end == start + 10.0
    applied, _ = best_rotation(rec['lower'].imu.gyr[3000:], fixed['lower'].imu.gyr[3000:])
    error = Rotation.from_matrix(applied) * MOUNTING
    assert np.degrees(error.magnitude()) < 2.0
    assert fixed.corrections[-1].parameters['own_window']['lower']['share_in_trial_window'] == 0.0

    part = resample_to_common_grid(fixed)
    assert abs(segment_window_skip_s(part, 'lower') - end) < 0.05
    assert segment_window_skip_s(part, 'upper') == 0.0
    runs = _joint_runs(part, {j: np.ones(3) for j in part.joints}, 10.0)
    first = int(np.argmax(runs[('upper', 'lower')].scored)) / part['upper'].imu.rate_hz
    assert abs(first - end) < 0.05


def test_without_the_rule_the_still_segment_gets_no_rotation():
    """The trial's window holds no motion of 'lower': the window fit is empty, so without the own-window rule its
    sensor axes would be left as they were."""
    from mocapref.errors.e18_mounting import check_for_mounting_in_convergence
    res = check_for_mounting_in_convergence(late_segment())
    assert res.per_segment['lower']['fit'] is None and res.per_segment['upper']['fit'] is not None
