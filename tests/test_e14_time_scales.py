"""Known-answer test for the dataset-level handler of error 14 (IMUs of one trial on different time scales): the one
IMU running at its own rate is counted and aligned by the corrected loader. The check itself is tested in
test_errors_known_answer.py."""
from memory_loader import MemoryLoader
from mocapref.errors import e14_time_scales as M
from mocapref.synthetic import make_recording


def test_handler_counts_and_aligns_the_deviating_imu():
    names = ('a', 'b', 'c', 'd')
    loader = MemoryLoader({'short': make_recording(duration_s=30.0, segment_names=names),
                           'apart': make_recording(duration_s=180.0, segment_names=names,
                                                   segment_drift_ppm={'c': 800.0})})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences'], p['undecided']) == (2, 1, 1, 1)
    fixed = M.correct(loader, rows)
    assert not M.check_for_sensor_rate_mismatch(fixed.load('apart')).detected
