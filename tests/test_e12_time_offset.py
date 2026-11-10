"""Known-answer test for the dataset-level handler of error 12 (time offset between IMU and reference): an injected
offset is counted once per trial and removed by the corrected loader; a clean trial is not counted. The check itself
is tested in test_errors_known_answer.py."""
import pytest

from memory_loader import MemoryLoader
from mocapref.errors import e12_time_offset as M
from mocapref.synthetic import make_recording


def test_handler_counts_and_removes_an_offset():
    loader = MemoryLoader({'clean': make_recording(duration_s=30.0, seed=1),
                           'late': make_recording(duration_s=30.0, seed=2, offset_s=0.08)})
    rows = M.check_dataset(loader, 'synthetic', 1)
    p = M.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences']) == (2, 1, 1)
    assert rows.set_index('trial').offset_ms['late'] == pytest.approx(80.0, abs=2.0)
    fixed = M.correct(loader, rows)
    assert M.check_for_synchronisation_offset(fixed.load('late')).detected is False
    assert fixed.load('clean')['upper'].imu.start_s == 0.0
