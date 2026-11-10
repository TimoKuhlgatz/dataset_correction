"""Small helpers shared by the dataset-level checks of errors 11-22 (rows in, rows out, loaders that correct)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ._report import CorrectedLoader, _safe, count_trials


def truthy(v) -> bool:
    """True for True and 'True' (rows read back from CSV hold strings)."""
    return v is True or str(v).strip().lower() == 'true'


def undecided(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() in ('', 'None', 'nan')


def flag(v):
    """A row's detected value as True / False / None, whether it comes from memory or from a CSV."""
    return None if undecided(v) else truthy(v)


def number(v, default=float('nan')) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x


def loads(v, default=None):
    """JSON column back to Python ('' or NaN gives `default`)."""
    if isinstance(v, (dict, list)):
        return v
    if undecided(v):
        return default
    return json.loads(v)


def failed(rows: pd.DataFrame) -> pd.Series:
    """Rows of trials that could not be read."""
    if 'error' not in rows:
        return pd.Series(False, index=rows.index)
    return rows['error'].map(lambda e: not undecided(e))


def read_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows of trials that were read (failed trials dropped)."""
    return rows[~failed(rows)]


def prevalence_of(rows: pd.DataFrame, unit: str, occurrences: int | None = None) -> dict:
    """count_trials on the rows that were read; trials_checked counts the trials read."""
    return count_trials(read_rows(rows), unit, occurrences)


def frame(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Rows as a DataFrame with at least `columns` (an empty dataset still has them)."""
    out = pd.DataFrame(records)
    for c in columns:
        if c not in out:
            out[c] = None
    return out


def by_group(loader, key: str = 'participant') -> list[tuple]:
    """The loader's trials grouped by participant or session (each group a tuple of trials)."""
    of = loader.participant_of if key == 'participant' else loader.session_of
    groups: dict = {}
    for t in loader.trials():
        groups.setdefault(of(t), []).append(t)
    return [tuple(g) for g in groups.values()]


_SHARED: dict = {}


def _call(job):
    return _safe(_SHARED['fn'], _SHARED['loader'], job)


def map_jobs(loader, fn, processes: int, jobs: list) -> list:
    """fn(loader, job) for every job, in forked worker processes when processes > 1. The loader and fn reach the
    workers by fork, not by pickling (some loaders cannot be pickled); a failing job gives dict(trial, error)."""
    if processes <= 1 or len(jobs) <= 1:
        return [_safe(fn, loader, j) for j in jobs]
    import multiprocessing
    _SHARED.update(fn=fn, loader=loader)
    try:
        with multiprocessing.get_context('fork').Pool(processes) as pool:
            return pool.map(_call, jobs, chunksize=1)
    finally:
        _SHARED.clear()


def run_groups(loader, fn, processes: int, key: str = 'participant') -> list[dict]:
    """fn(loader, trials) for every participant (or session) group; returns the concatenated row lists. A group whose
    check failed gives one error row per trial."""
    groups = by_group(loader, key)
    out = []
    for group, got in zip(groups, map_jobs(loader, fn, processes, groups)):
        if isinstance(got, dict) and 'error' in got:
            out += [dict(trial=t, participant=str(loader.participant_of(t)), error=got['error']) for t in group]
        else:
            out += got
    return out


def run_trials(loader, fn, processes: int) -> list[dict]:
    """fn(loader, trial) -> list of row dicts for every trial; a failed trial gives one error row."""
    out = []
    trials = list(loader.trials())
    for t, got in zip(trials, map_jobs(loader, fn, processes, trials)):
        if isinstance(got, dict) and 'error' in got:
            out.append(dict(trial=t, participant=str(loader.participant_of(t)), error=got['error']))
        else:
            out += got
    return out


def per_trial(rows: pd.DataFrame) -> dict:
    """{trial: list of row dicts} of the rows that were read."""
    out: dict = {}
    for r in read_rows(rows).to_dict('records'):
        out.setdefault(str(r['trial']), []).append(r)
    return out


def corrected(loader, fix, table: dict, label: str) -> CorrectedLoader:
    """A loader applying fix(recording, rows_of_trial) to every trial (picklable: fix must be module-level)."""
    return CorrectedLoader(loader, _Fix(fix, table), label)


class _Fix:
    def __init__(self, fix, table):
        self.fix, self.table = fix, table

    def __call__(self, rec, trial):
        rows = self.table.get(str(trial))
        return self.fix(rec, rows) if rows else rec


def vector(r: dict, prefix: str) -> np.ndarray:
    return np.array([number(r.get(f'{prefix}{a}')) for a in 'xyz'])
