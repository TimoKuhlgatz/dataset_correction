"""Geneva gait with the IMU streams rebuilt from the raw session logs (mis-assembled streams, error 10).

Why: the deposit's per-trial files carry IMU columns that are not the sensors' recordings of that trial (measured:
the best correlation of the deposited gyroscope with the same sensor's raw recording, over shifts of ±1.5 s, has a
median of 0.10-0.12 in the gait-type trials; 100 of 2,218 sensor-trials are clean shifted copies). The raw `.BIN`
files hold every sensor's whole session at 256 Hz.

How the IMU streams are rebuilt (readers/scripts/rebuild_geneva_from_raw.py): per participant and sensor, each
trial is found in the sensor's raw session by correlating the optical reference's rotation speed (from the
housing's marker cluster) with the raw gyroscope's; the trial's position is the median over the sensors (each
logger keeps one constant offset to the others), refined per sensor to 1 ms; the raw stream is low-passed (zero
phase, 40 Hz) and read at the trial's 100 Hz rows, on the vendor's factory calibration from the file header. Only
time is fitted.

The rebuild is run once (73 s on 10 processes) and stored outside the repository:
    uv run python -m mocapref.datasets.readers.scripts.rebuild_geneva_from_raw --out <REBUILD_DIR>

This loader serves the trials, segments and references of `grouvel` unchanged and replaces each IMU's gyroscope and
accelerometer by the rebuilt one; a segment the rebuild could not place keeps no valid IMU sample. The match of each
segment (position in the raw session, correlation, how it was placed) is in metadata['raw_rebuild'].
"""
from __future__ import annotations

import pathlib
from functools import cached_property

import numpy as np
import pandas as pd

from ..recording import ImuStream, Recording
from ._bridge import READER_LOADERS

REBUILD_DIR = pathlib.Path('/home/data/imu-orientation/prepared/grouvel/raw-rebuild-v1')
_GROUVEL = next(loader for loader in READER_LOADERS if loader.name == 'grouvel')


class GrouvelRawLoader(_GROUVEL):
    """Geneva gait, IMU streams rebuilt from the raw 256 Hz session logs on the optical clock (error 10)."""

    name = 'grouvel_raw'

    def __init__(self, root=None, rebuild_dir: pathlib.Path | str = REBUILD_DIR):
        super().__init__(root)
        self.rebuild_dir = pathlib.Path(rebuild_dir)
        if not (self.rebuild_dir / 'match.csv').exists():
            raise FileNotFoundError(f'{self.rebuild_dir}/match.csv missing: run uv run python -m '
                                    f'mocapref.datasets.readers.scripts.rebuild_geneva_from_raw --out {self.rebuild_dir}')

    @cached_property
    def match(self) -> pd.DataFrame:
        return pd.read_csv(self.rebuild_dir / 'match.csv')

    def load(self, trial: str) -> Recording:
        rec = super().load(trial)
        subject, reader_trial = self._trial_keys[trial]
        path = self.rebuild_dir / subject / f'{reader_trial}.npz'
        arrays = dict(np.load(path)) if path.exists() else {}
        rows = self.match[(self.match.subject == subject) & (self.match.trial == reader_trial)]
        info = {}
        for name, seg in rec.segments.items():
            gyr, acc = arrays.get(f'gyr/{name}'), arrays.get(f'acc/{name}')
            if gyr is None or len(gyr) != len(seg.imu):
                gyr = acc = np.full((len(seg.imu), 3), np.nan)
            valid = np.isfinite(gyr).all(1) & np.isfinite(acc).all(1)
            seg.imu = ImuStream(np.where(valid[:, None], gyr, 0.0), np.where(valid[:, None], acc, 0.0),
                                seg.imu.rate_hz, seg.imu.start_s, valid)
            row = rows[rows.segment == name]
            info[name] = (dict(start_in_session_s=float(row.start_s.iloc[0]), corr=float(row['corr'].iloc[0]),
                               source=str(row.source.iloc[0]), deposit_corr=float(row.deposit_corr.iloc[0]))
                          if len(row) else dict(source='not in the rebuild'))
        rec.dataset = self.name
        rec.metadata['raw_rebuild'] = info
        rec.metadata['timing'] = ('IMU rebuilt from the raw .BIN session on the optical 100 Hz rows '
                                  '(rebuild_geneva_from_raw.py); reference as deposited')
        return rec
