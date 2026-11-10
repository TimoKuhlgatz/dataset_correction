"""Error 10: mis-assembled streams: the IMU columns of a published trial file are not the sensors' recordings of that
trial.

What it is: the depositors cut and joined the sensor streams wrongly, so a trial file carries IMU data from another
moment of the session (or a mixture). Each column is real sensor data, so nothing looks broken in the file itself.

How we catch it: there is no general rule; it needs what the files should contain. Found only on Geneva gait, whose
raw 256 Hz session logs (.BIN) hold every sensor's whole session. For every sensor-trial the rebuild from the raw
logs (datasets/grouvel_raw.py, table match.csv) stores `deposit_corr`, the best correlation of the published
gyroscope (3 axes) with the same sensor's raw gyroscope over shifts of ±1.5 s around the trial's place in the
session. A clean shifted copy gives about 1; a sensor-trial below CLEAN_COPY (0.99) is mis-assembled. Trials the
rebuild could not place are undecided. Every other dataset is not checked (no general rule).

How we correct it: Geneva is read from the streams rebuilt from its raw logs (reader 'grouvel_raw'). In the order of
correction, correction.base_reader already switches to that reader from error 8's check on (the trial pairing cannot
be judged on the mis-assembled files), so the correction here only matters for a report on the deposited files.
"""
from __future__ import annotations

import pandas as pd

from ._report import ErrorSpec, count_trials, make_handler
from ._rows import reader_of, rebase

CLEAN_COPY = 0.99
"""A published sensor-trial whose best correlation with the sensor's raw recording reaches this is a clean copy."""
CHECKED = {'grouvel', 'grouvel_raw'}
NOT_CHECKED = 'not checked: no general rule (needs what the files should contain, such as raw session logs)'


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """Geneva: one row per sensor-trial placed by the rebuild (estimate = deposit_corr, detected below CLEAN_COPY)
    and one undecided row per trial the rebuild could not place. Every other dataset: one undecided row per trial
    with the reason."""
    reader = reader_of(loader)
    if dataset not in CHECKED and reader not in CHECKED:
        rows = pd.DataFrame([dict(trial=t, participant=str(loader.participant_of(t)), segment='', detected=None,
                                  unit='sensor-trials', reason=NOT_CHECKED) for t in loader.trials()])
        rows['reader'] = reader
        return rows
    from ..datasets import load_dataset
    match = load_dataset('grouvel_raw').match
    rows = [dict(trial=f'{m.subject}/{m.trial}', participant=str(m.subject), segment=m.segment,
                 detected=bool(m.deposit_corr < CLEAN_COPY), estimate=float(m.deposit_corr), unit='correlation',
                 deposit_shift_s=m.deposit_shift_s, placed_by=m.source) for m in match.itertuples()]
    placed = {r['trial'] for r in rows}
    rows += [dict(trial=t, participant=str(loader.participant_of(t)), segment='', detected=None,
                  unit='correlation', reason='not placed in the raw session logs')
             for t in loader.trials() if t not in placed and loader.load(t).segments]
    rows = pd.DataFrame(rows)
    rows['reader'] = reader
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with a mis-assembled sensor-trial / mis-assembled sensor-trials."""
    return count_trials(rows, 'sensor-trials')


def correct(loader, rows: pd.DataFrame):
    """Geneva read from the streams rebuilt from its raw logs: the same corrections on top of the reader
    'grouvel_raw'. Every other reader (and Geneva already rebuilt) is returned unchanged."""
    if reader_of(loader) != 'grouvel':
        return loader
    from ..datasets import load_dataset
    return rebase(loader, load_dataset('grouvel_raw'))


SPEC = ErrorSpec(10, 'mis_assembled_streams', 'fix', 'mis-assembled streams')
PARTS = {10: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
_handle = make_handler(PARTS[10])


def handle_mis_assembled_streams(datasets=None, **kwargs):
    report = _handle(datasets, **kwargs)
    for name in report.prevalence.dataset:
        if name not in CHECKED:
            report.add_note(name, NOT_CHECKED)
    return report


handle_mis_assembled_streams.__doc__ = _handle.__doc__
