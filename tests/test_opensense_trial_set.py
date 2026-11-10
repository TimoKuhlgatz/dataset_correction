"""OpenSense trial set of this package: the 40 Hz trials are left out, Subject07's task block
is read with the sensor mapping of its walk."""
from pathlib import Path

import pytest

from mocapref.datasets.readers.characterisation.adapters.opensense import ADAPTER, EXCLUDED_TRIALS
from mocapref.datasets.readers.datasets.opensense import sensors

MAPPING = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000"><XsensDataReaderSettings>
<trial_prefix>MT_X_004-000</trial_prefix><ExperimentalSensors>
<ExperimentalSensor name="_00B4226B"><name_in_model>pelvis_imu</name_in_model></ExperimentalSensor>
<ExperimentalSensor name="_00B421EF"><name_in_model>calcn_r_imu</name_in_model></ExperimentalSensor>
<!--<ExperimentalSensor name="_00B421F3"><name_in_model>torso_imu</name_in_model></ExperimentalSensor>-->
</ExperimentalSensors></XsensDataReaderSettings></OpenSimDocument>"""


def test_borrowed_mapping_keeps_sensor_ids_and_takes_the_trials_own_file_names(tmp_path, monkeypatch):
    subject = tmp_path / 'SubjectX'
    (subject / 'walking' / 'IMU').mkdir(parents=True)
    (subject / 'walking' / 'IMU' / 'myIMUMappings_walking.xml').write_text(MAPPING)
    raw = subject / 'complexTasks' / 'IMU' / 'xsens' / 'LowerExtremity'
    raw.mkdir(parents=True)
    for sensor_id in ('_00B4226B', '_00B421EF', '_00B421F3', '_00B42268'):  # one commented out, one never mapped
        (raw / f'MT_X_005-000{sensor_id}.txt').write_text('')
    monkeypatch.setitem(sensors.BORROWED_MAPPINGS, ('SubjectX', 'complexTasks'), 'walking')
    assert sensors._borrowed_mapping(subject, 'complexTasks') == {
        'MT_X_005-000_00B4226B.txt': 'pelvis_imu', 'MT_X_005-000_00B421EF.txt': 'calcn_r_imu'}
    assert sensors._borrowed_mapping(subject, 'walking') is None  # not listed: nothing is borrowed


def test_trial_set():
    trials = ADAPTER.trials()
    assert len(trials) == 16
    assert ('Subject07', 'complexTasks') in trials
    assert not set(EXCLUDED_TRIALS) & set(trials)
    assert ADAPTER.joints_of('Subject07', 'complexTasks') == ADAPTER.joints_of('Subject07', 'walking')


DEPOSIT = Path('/home/data/imu-orientation/raw/opensense/_extracted/Subject07')


@pytest.mark.skipif(not DEPOSIT.is_dir(), reason='OpenSense deposit not on this machine')
def test_subject07_task_block_reads_all_seven_sensors_against_the_depositors_table():
    """With the borrowed mapping every sensor's onboard orientation lines up with its column of the depositors'
    orientation table to float precision (measured 2e-7 to 4e-7 rad per step)."""
    read = sensors.read_sensors(DEPOSIT, 'complexTasks')
    assert sorted(read) == ['calcn_l_imu', 'calcn_r_imu', 'femur_l_imu', 'femur_r_imu', 'pelvis_imu',
                            'tibia_l_imu', 'tibia_r_imu']
    assert max(s['align_residual'] for s in read.values()) < 1e-3
