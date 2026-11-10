"""Known-answer test for the dataset-level handler of error 13 (clock drift between IMU and reference): an injected
drift is counted and removed by the corrected loader; a clean trial is not counted. The check itself is tested in
test_errors_known_answer.py."""
from memory_loader import MemoryLoader
from mocapref.errors import e13_clock_drift as M
from mocapref.synthetic import make_recording


def test_handler_counts_and_removes_a_drift():
    loader = MemoryLoader({'clean': make_recording(duration_s=120.0, seed=1),
                           'drifting': make_recording(duration_s=120.0, seed=2, drift_ppm=400.0)})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 1, 1)
    fixed = M.correct(loader, rows)
    assert not M.check_for_clock_drift(fixed.load('drifting')).detected
