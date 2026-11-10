"""Small helpers shared by the error handlers of errors 1-10: reading back saved rows, grouping trials, and walking
a chain of corrected loaders."""
from __future__ import annotations

import json

import numpy as np


def truthy(value) -> bool:
    """True for True and for 'True' read back from a saved table."""
    return value is True or str(value).strip().lower() == 'true'


def dumps(value) -> str:
    """A JSON string of `value` (numpy numbers and tuples allowed), so that rows survive a round trip through CSV."""
    return json.dumps(value, default=lambda v: v.item() if hasattr(v, 'item') else str(v))


def loads(value, default=None):
    """The value of a JSON column, whether still a string, already parsed, or empty."""
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and value != value) or value == '':
        return default
    return json.loads(value)


def runs(mask: np.ndarray) -> list[list[int]]:
    """[first, length] of every continuous run of True."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, bool).astype(np.int8), [0]]))
    return [[int(a), int(b - a)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def from_runs(stored, n: int) -> np.ndarray:
    """The boolean mask of length n from [first, length] runs."""
    mask = np.zeros(n, bool)
    for first, length in stored or []:
        mask[int(first):int(first) + int(length)] = True
    return mask


def by_participant(loader, trials=None, session: bool = False) -> list[tuple[str, list[str]]]:
    """[(key, trials)] in the loader's order: one group per participant, or per participant and session."""
    groups: dict[str, list[str]] = {}
    for trial in (loader.trials() if trials is None else trials):
        key = str(loader.participant_of(trial))
        if session:
            s = loader.session_of(trial)
            key = f"{key}|{'' if s == trial else s}"  # a loader without sessions returns the trial itself
        groups.setdefault(key, []).append(trial)
    return list(groups.items())


def reader_of(loader) -> str:
    """The name of the dataset reader at the bottom of a chain of corrected loaders."""
    from ._report import CorrectedLoader
    while isinstance(loader, CorrectedLoader):
        loader = loader.base
    return getattr(loader, 'name', '')


def rebase(loader, new_base):
    """The same chain of corrections as `loader`, on top of another reader (`new_base`)."""
    from ._report import CorrectedLoader
    if isinstance(loader, CorrectedLoader):
        return CorrectedLoader(rebase(loader.base, new_base), loader.fix, loader.label)
    return new_base


def rows_for(rows, trial) -> list:
    """The rows of one trial (as named tuples)."""
    return list(rows[rows.trial.astype(str) == str(trial)].itertuples()) if len(rows) else []


_JOB = None  # (fn, loader) of the running parallel_map, inherited by the forked workers


def _run(item):
    from ._report import _safe
    fn, loader = _JOB
    return _safe(fn, loader, item)


def parallel_map(loader, fn, items, processes: int = 1) -> list:
    """fn(loader, item) for every item, in forked worker processes when processes > 1. Unlike a plain process pool,
    the loader is inherited by the workers instead of pickled (the dataset readers built with type() cannot be
    pickled). A failing item gives dict(trial=item, error=...)."""
    global _JOB
    from ._report import _safe
    items = list(items)
    if processes <= 1 or len(items) <= 1:
        return [_safe(fn, loader, item) for item in items]
    import multiprocessing
    _JOB = (fn, loader)
    try:
        with multiprocessing.get_context('fork').Pool(min(processes, len(items))) as pool:
            return pool.map(_run, items, chunksize=1)
    finally:
        _JOB = None
