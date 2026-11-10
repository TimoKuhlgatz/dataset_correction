"""TUK six-minute walk: 25 participants, two sessions (test, retest), seven Xsens MTw Awinda IMUs on pelvis,
thighs, shanks and feet. Every IMU sits in a 3D-printed box carrying a marker cluster that was hand-eye calibrated
to the IMU, so the reference is the orientation of the IMU housing itself.

Deposit: one HDF5 file per recording in `tuk6min/_extracted/`, group `data`:
    gyr_i, acc_i   (N, 21): 7 IMUs × 3 axes, in the order of the attribute listIMUIDs
                   (Pelvis, RightUpperLeg, RightLowerLeg, RightFoot, LeftUpperLeg, LeftLowerLeg, LeftFoot)
    quat_ni        (N + 1, 28): optical orientation of each IMU housing, scalar first, in the order pelvis,
                   left thigh, right thigh, left shank, right shank, left foot, right foot (this order gives
                   consistent fits of IMU to housing)
    quat_ns        (N + 1, 28): anatomical segment frames from 32 landmarks, same order
    gyr_biases_i   (7, 3): gyroscope biases shipped with the file (not subtracted here)
    time           (N,) Unix time stamps
The optical stream has one frame more than the IMU stream. Which end carries the surplus frame is not documented,
so both streams start at 0 s here and the synchronisation-offset check measures it.
The sampling rate is taken from the time stamps (about 58.8 Hz), not from the documentation (60 Hz).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..recording import ImuStream, Recording, ReferenceStream, Segment
from ..rotations import rotation_from_wxyz
from .base import DATA_ROOT, DatasetLoader

IMU_ORDER = ['pelvis', 'r_thigh', 'r_shank', 'r_foot', 'l_thigh', 'l_shank', 'l_foot']
OPTICAL_ORDER = ['pelvis', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank', 'l_foot', 'r_foot']
JOINTS = [('pelvis', 'l_thigh'), ('l_thigh', 'l_shank'), ('l_shank', 'l_foot'),
          ('pelvis', 'r_thigh'), ('r_thigh', 'r_shank'), ('r_shank', 'r_foot')]
SUFFIX = '_6_min_walking_test.h5'


class TukLoader(DatasetLoader):
    name = 'tuk6min'
    default_root = DATA_ROOT / 'tuk6min' / '_extracted'

    def trials(self) -> list[str]:
        return sorted(p.name[:-len(SUFFIX)] for p in self.root.glob(f'P*{SUFFIX}'))  # e.g. 'P11_Test_1'

    def participant_of(self, trial: str) -> str:
        return trial.split('_')[0]  # 'P11_Retest' -> 'P11'; test and retest are separate sessions

    def load(self, trial: str) -> Recording:
        with h5py.File(self.root / f'{trial}{SUFFIX}', 'r') as f:
            d = f['data']
            gyr = d['gyr_i'][()].reshape(-1, 7, 3)
            acc = d['acc_i'][()].reshape(-1, 7, 3)
            housing = d['quat_ni'][()].reshape(-1, 7, 4)
            anatomical = d['quat_ns'][()].reshape(-1, 7, 4)
            biases = d['gyr_biases_i'][()]
            time = d['time'][()]
        rate = float(1.0 / np.median(np.diff(time)))
        segments, anatomical_frames, shipped_biases = {}, {}, {}
        for name in IMU_ORDER:
            i, o = IMU_ORDER.index(name), OPTICAL_ORDER.index(name)
            quat = housing[:, o]
            reference = ReferenceStream(rotation_from_wxyz(quat), rate, valid=np.isfinite(quat).all(1))
            segments[name] = Segment(name, ImuStream(gyr[:, i], acc[:, i], rate), reference)
            anatomical_frames[name] = rotation_from_wxyz(anatomical[:, o])
            shipped_biases[name] = biases[i]
        return Recording(self.name, trial, segments, JOINTS,
                         metadata=dict(source=str(self.root), unix_start_s=float(time[0]),
                                       anatomical_frames=anatomical_frames, shipped_gyroscope_biases=shipped_biases),
                         participant=self.participant_of(trial), session=self.session_of(trial))
