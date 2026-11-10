"""mfimt_d01: 3D-printed chain of four segments joined by a 1D, a 2D and a 3D joint, OptiTrack reference.

Deposit: one file `chain_experiment_d01.npz` in the mfimt package, 100 Hz, IMU and reference on one time base,
    gyr, acc   (4, N, 3) rad/s and m/s^2
    quat_opt   (4, N, 4) optical quaternions, scalar first
    segment_names, fs
Chain (from the package documentation): seg4 -(1D)- seg1 -(2D)- seg5 -(3D)- seg2. The recording starts with about
20 s of rest. The package documents a handful of optical glitches; they stay in the data here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..recording import ImuStream, Recording, ReferenceStream, Segment
from ..rotations import rotation_from_wxyz
from .base import DATA_ROOT, DatasetLoader


class MfimtLoader(DatasetLoader):
    name = 'mfimt_d01'
    default_root = Path(os.environ.get('MOCAPREF_MFIMT', DATA_ROOT / 'mfimt'))  # folder of the deposit's data files

    def trials(self) -> list[str]:
        return ['d01']

    def participant_of(self, trial: str) -> str:
        return 'chain'

    def load(self, trial: str = 'd01') -> Recording:
        z = np.load(self.root / 'chain_experiment_d01.npz')
        rate = float(z['fs'])
        names = [str(n) for n in z['segment_names']]
        segments = {}
        for i, name in enumerate(names):
            quat = z['quat_opt'][i].astype(float)
            reference = ReferenceStream(rotation_from_wxyz(quat), rate, valid=np.isfinite(quat).all(1))
            segments[name] = Segment(name, ImuStream(z['gyr'][i], z['acc'][i], rate), reference)
        joints = [('seg4', 'seg1'), ('seg1', 'seg5'), ('seg5', 'seg2')]
        joints = [j for j in joints if j[0] in segments and j[1] in segments]
        return Recording(self.name, trial, segments, joints, metadata=dict(source=str(self.root)),
                         participant='chain', session=trial)
