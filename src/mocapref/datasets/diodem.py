"""DIODEM: five-segment mechanical chains (arm and gait set), two IMUs per segment (one rigid, one on foam), optical
markers and an optical orientation per segment (Bachhuber et al., Scientific Data 2025).

Deposit: `diodem/dataset/<arm|gait>/expNN/motionMM_<name>/` with three CSV files, each starting with a line
"# sampling frequency: <Hz>" and a units line (seconds, meters, radians):
    expNN_motionMM_imu_rigid.csv, expNN_motionMM_imu_nonrigid.csv   seg<k>_acc_*, seg<k>_gyr_*, seg<k>_mag_*
    expNN_motionMM_omc.csv   seg<k>_marker<j>_* (m) and seg<k>_quat_w/x/y/z
The IMUs run at 40 Hz natively; the optical rate differs per experiment (30 or 120 Hz).

Trials: one trial is one whole experiment (`arm/exp01`, 80-368 s). The deposit cuts each experiment into motions
(2-142 s), but they are consecutive pieces of one continuous recording (measured: at every motion boundary the step of the optical orientation, of the markers and of the gyroscope is as large as an ordinary step
between neighbouring samples). The loader joins them back together; the motion boundaries are kept in
metadata['motions'] as (name, start_s, end_s) on the IMU time base. IMU and optical stream of one motion can
differ by less than one optical frame in length, because both are cut at the same moment; each stream is joined on
its own, so this does not accumulate. `split='motion'` gives the motions as separate trials instead.

Choose which IMU represents each segment with `imu='rigid'` (default) or `imu='nonrigid'` (foam).

Joints: the data description (docs/diodem.pdf of the diodem package) names the joints, the arm chain three hinges
and a spherical joint, the gait chain y-hinge, saddle, saddle, y-hinge. The order of the segments along each chain:
arm seg1-seg2-seg3-seg4-seg5 (spherical joint first), gait seg5-seg1-seg2-seg3-seg4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..recording import ImuStream, Recording, ReferenceStream, Segment
from ..rotations import rotation_from_wxyz
from .base import DATA_ROOT, DatasetLoader

CHAINS = {'arm': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5'], 'gait': ['seg5', 'seg1', 'seg2', 'seg3', 'seg4']}


def read_diodem_csv(path: Path) -> tuple[float, dict[str, np.ndarray]]:
    """Sampling rate and columns of one DIODEM CSV file."""
    with open(path) as f:
        rate = float(f.readline().split(':')[1])
        f.readline()
        header = f.readline().strip().split(',')
    values = np.genfromtxt(path, delimiter=',', skip_header=3)
    return rate, {h: values[:, i] for i, h in enumerate(header)}


class DiodemLoader(DatasetLoader):
    name = 'diodem'
    default_root = DATA_ROOT / 'diodem' / 'dataset'

    def __init__(self, root=None, imu: str = 'rigid', split: str = 'experiment'):
        super().__init__(root)
        if imu not in ('rigid', 'nonrigid'):
            raise ValueError("imu must be 'rigid' or 'nonrigid'")
        if split not in ('experiment', 'motion'):
            raise ValueError("split must be 'experiment' or 'motion'")
        self.imu, self.split = imu, split

    def trials(self) -> list[str]:
        pattern = '*/exp*' if self.split == 'experiment' else '*/exp*/motion*'
        return sorted(str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_dir())

    def participant_of(self, trial: str) -> str:
        return trial.split('/')[0]  # the mechanical chain: 'arm' or 'gait'

    def session_of(self, trial: str) -> str:
        return '/'.join(trial.split('/')[:2])  # 'arm/exp01'; assumed: IMUs stay mounted between the motions of one experiment

    def load(self, trial: str) -> Recording:
        """An experiment ('arm/exp01', the motions joined) or a single motion ('arm/exp01/motion01_canonical')."""
        folder = self.root / trial
        if trial.count('/') == 2:
            motions = [folder]
        else:
            motions = sorted(p for p in folder.glob('motion*') if p.is_dir())
        pieces = [self._read_motion(m) for m in motions]
        imu_rate, ref_rate = pieces[0][0], pieces[0][1]
        if any(p[0] != imu_rate or p[1] != ref_rate for p in pieces):
            raise ValueError(f'{trial}: sampling rates differ between motions')
        boundaries, start = [], 0.0
        for m, p in zip(motions, pieces):
            end = start + len(next(iter(p[2].values()))['gyr']) / imu_rate
            boundaries.append((m.name, start, end))
            start = end
        segments = {}
        for name in pieces[0][2]:
            parts = [p[2][name] for p in pieces]
            join = lambda key: np.concatenate([q[key] for q in parts])
            valid = join('valid')
            markers = {k: np.concatenate([q['markers'][k] for q in parts]) for k in parts[0]['markers']}
            reference = ReferenceStream(rotation_from_wxyz(np.where(valid[:, None], join('quat'), np.nan)), ref_rate,
                                        valid=valid, markers=markers)
            segments[name] = Segment(name, ImuStream(join('gyr'), join('acc'), imu_rate), reference)
        chain = [s for s in CHAINS[trial.split('/')[0]] if s in segments]
        return Recording(self.name, trial, segments, joints=list(zip(chain, chain[1:])),
                         metadata=dict(source=str(folder), imu=self.imu, motions=boundaries),
                         participant=self.participant_of(trial), session=self.session_of(trial))

    def _read_motion(self, folder: Path):
        """IMU rate, optical rate and per segment the raw arrays of one motion folder."""
        imu_rate, imu = read_diodem_csv(next(folder.glob(f'*_imu_{self.imu}.csv')))
        ref_rate, omc = read_diodem_csv(next(folder.glob('*_omc.csv')))
        segments = {}
        for k in range(1, 6):
            name = f'seg{k}'
            if f'{name}_gyr_x' not in imu or f'{name}_quat_w' not in omc:
                continue
            quat = np.column_stack([omc[f'{name}_quat_{a}'] for a in 'wxyz'])
            segments[name] = dict(
                gyr=np.column_stack([imu[f'{name}_gyr_{a}'] for a in 'xyz']),
                acc=np.column_stack([imu[f'{name}_acc_{a}'] for a in 'xyz']),
                quat=quat,
                valid=np.isfinite(quat).all(1) & (np.linalg.norm(np.nan_to_num(quat), axis=1) > 0.5),
                markers={f'marker{j}': 1000.0 * np.column_stack([omc[f'{name}_marker{j}_{a}'] for a in 'xyz'])
                         for j in range(1, 5) if f'{name}_marker{j}_x' in omc})
        return imu_rate, ref_rate, segments
