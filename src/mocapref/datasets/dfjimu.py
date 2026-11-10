"""dfjimu: ABB IRB120 robot arm with two Xsens IMUs on links connected by one joint, Vicon reference.

Deposit: 15 MATLAB files in the dfjimu repository, each a struct `data` with
    rate        sampling rate (50 Hz, IMU and reference share it)
    sensorData  (N, 12): sensor 1 acc (0:3), gyr (3:6), sensor 2 acc (6:9), gyr (9:12)
    ref         (N, 8): Vicon quaternion (w, x, y, z) of sensor 1 (0:4) and sensor 2 (4:8)
Units as in the repository's own use: m/s^2 and rad/s.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io

from ..recording import ImuStream, Recording, ReferenceStream, Segment
from ..rotations import rotation_from_wxyz
from .base import DATA_ROOT, DatasetLoader


class DfjimuLoader(DatasetLoader):
    name = 'dfjimu'
    default_root = Path(os.environ.get('MOCAPREF_DFJIMU', DATA_ROOT / 'dfjimu'))  # folder of the deposit's data files

    def trials(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob('*.mat'))

    def participant_of(self, trial: str) -> str:
        return 'robot'

    def load(self, trial: str) -> Recording:
        data = scipy.io.loadmat(self.root / f'{trial}.mat', squeeze_me=True, struct_as_record=False)['data']
        rate = float(data.rate)
        s, q = np.asarray(data.sensorData, float), np.asarray(data.ref, float)
        segments = {}
        for name, acc, gyr, quat in (('link1', s[:, 0:3], s[:, 3:6], q[:, 0:4]),
                                     ('link2', s[:, 6:9], s[:, 9:12], q[:, 4:8])):
            reference = ReferenceStream(rotation_from_wxyz(quat), rate, valid=np.isfinite(quat).all(1))
            segments[name] = Segment(name, ImuStream(gyr, acc, rate), reference)
        return Recording(self.name, trial, segments, joints=[('link1', 'link2')], metadata=dict(source=str(self.root)),
                         participant='robot', session=self.session_of(trial))
