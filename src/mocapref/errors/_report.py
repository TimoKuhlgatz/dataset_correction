"""The common interface of the error handlers: one `handle_<error>()` per error, one `ErrorReport` back.

Every handler is built from three small functions that live in the error's own module:

    check_dataset(loader, dataset, processes, **options) -> pandas.DataFrame
        one row per checked unit (a trial, a trial and segment, or a session), with at least the columns
        trial, participant, segment ('' where the unit is the whole trial), detected (True / False / None),
        estimate and unit. Anything the correction needs later is stored in further columns, so a saved
        report can be corrected again without re-running the check.
    prevalence(rows) -> dict
        trials_checked, trials_detected, occurrences, occurrence_unit, undecided for one dataset.
    correct(loader, rows) -> loader
        a loader whose load(trial) returns the trial corrected (repaired, or flagged samples marked invalid)
        for this error. Errors without a repair return the input loader unchanged.

`run_error()` turns them into a handler: it reads every requested dataset either as deposited
(data='deposited', the independent count) or corrected for every earlier error in the agreed order
(data='in_order', see mocapref.correction), runs the check, tabulates the prevalence and attaches the notes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

PREVALENCE_COLUMNS = ['dataset', 'trials_checked', 'trials_detected', 'occurrences', 'occurrence_unit', 'undecided',
                      'note']
ROW_COLUMNS = ['dataset', 'trial', 'participant', 'segment', 'detected', 'estimate', 'unit']


@dataclass(frozen=True)
class ErrorSpec:
    """What an error is: its number in the agreed order of correction, its name, and what happens to the data
    (kind 'fix' = repaired, 'mask' = samples or frames marked invalid, 'report' = flagged only)."""
    number: int
    name: str
    kind: str
    description: str = ''


@dataclass
class ErrorReport:
    """Outcome of one error handler over one or more datasets.

    prevalence: one row per dataset (PREVALENCE_COLUMNS). trials_detected counts trials with at least one
        detection; occurrences counts them in the error's own unit (occurrence_unit, e.g. 'segments', 'runs',
        'samples').
    rows: every checked unit of every dataset (ROW_COLUMNS plus the check's own columns).
    notes: free-text notes per dataset, for suspicious findings that the numbers do not show.
    data: 'deposited' or 'in_order', the data the check ran on.
    """
    spec: ErrorSpec
    data: str
    prevalence: pd.DataFrame
    rows: pd.DataFrame
    notes: dict[str, list[str]] = field(default_factory=dict)
    _loaders: dict = field(default_factory=dict, repr=False)
    _correct: Callable | None = field(default=None, repr=False)

    def add_note(self, dataset: str, text: str) -> None:
        """Attach a note to a dataset (shown in the prevalence table and saved with the report)."""
        self.notes.setdefault(dataset, []).append(text)
        self.prevalence.loc[self.prevalence.dataset == dataset, 'note'] = ' | '.join(self.notes[dataset])

    def corrected(self, dataset: str):
        """A loader of `dataset` with this error corrected on top of the data the check ran on."""
        if dataset not in self._loaders:
            raise KeyError(f'{dataset} was not part of this report')
        rows = self.rows[self.rows.dataset == dataset]
        return self._correct(self._loaders[dataset], rows) if self._correct else self._loaders[dataset]

    def save(self, folder: str | Path) -> Path:
        """Write prevalence.csv, rows.csv and notes.json into folder/<NN>_<name>/<data>/."""
        out = Path(folder) / f'{self.spec.number:02d}_{self.spec.name}' / self.data
        out.mkdir(parents=True, exist_ok=True)
        self.prevalence.to_csv(out / 'prevalence.csv', index=False)
        self.rows.to_csv(out / 'rows.csv', index=False)
        (out / 'notes.json').write_text(json.dumps(self.notes, indent=1))
        return out

    def __str__(self):
        head = f'Error {self.spec.number} - {self.spec.name} ({self.data}, {self.spec.kind})'
        return head + '\n' + self.prevalence.to_string(index=False)


def _as_list(datasets) -> list[str]:
    from ..tuning_split import effect_datasets
    if datasets is None:
        return list(effect_datasets())
    return [datasets] if isinstance(datasets, str) else list(datasets)


def source_loader(dataset: str, data: str, number: int, cache_dir: str | Path | None = None):
    """The loader a check of error `number` reads: the deposit, or the data corrected for errors 1..number-1."""
    if data == 'deposited':
        from ..datasets import load_dataset
        # Trento as published pairs the k-th IMU trial with the k-th optical file; Geneva's published files.
        return load_dataset('bonfiglio_deposited' if dataset == 'bonfiglio' else dataset)
    if data == 'in_order':
        from ..correction import corrected_before
        return corrected_before(dataset, number, cache_dir=cache_dir)
    raise ValueError(f"data must be 'deposited' or 'in_order', not {data!r}")


def run_error(spec: ErrorSpec, datasets, data: str, notes: dict | None, processes: int,
              check_dataset: Callable, prevalence: Callable, correct: Callable | None,
              loader_for: Callable | None = None, cache_dir=None, **options) -> ErrorReport:
    """Run one error's check over `datasets` and assemble the report (module description).

    loader_for(dataset, data) may replace source_loader for errors that read a dataset through another reader
    (trial pairing reads Trento with the deposit's own pairing)."""
    tables, rows_all, loaders = [], [], {}
    for name in _as_list(datasets):
        loader = (loader_for(name, data) if loader_for else None) or source_loader(name, data, spec.number, cache_dir)
        loaders[name] = loader
        rows = check_dataset(loader, name, processes, **options)
        rows = rows.copy()
        rows.insert(0, 'dataset', name)
        for col in ROW_COLUMNS:
            if col not in rows:
                rows[col] = '' if col in ('segment', 'unit') else None
        rows_all.append(rows)
        tables.append(dict(dataset=name, note='', **prevalence(rows)))
    prevalence_table = pd.DataFrame(tables)
    extra = [c for c in prevalence_table.columns if c not in PREVALENCE_COLUMNS]  # error-specific counts
    prevalence_table = prevalence_table.reindex(columns=PREVALENCE_COLUMNS[:-1] + extra + ['note'])
    rows = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame(columns=ROW_COLUMNS)
    report = ErrorReport(spec, data, prevalence_table, rows, {}, loaders, correct)
    for name, texts in (notes or {}).items():
        for text in ([texts] if isinstance(texts, str) else texts):
            report.add_note(name, text)
    return report


def count_trials(rows: pd.DataFrame, occurrence_unit: str, occurrences: int | None = None) -> dict:
    """The usual prevalence: trials with any detection, occurrences = detected rows unless given."""
    detected = rows.detected.map(lambda v: v is True or str(v) == 'True')
    undecided = rows.detected.map(lambda v: v is None or (isinstance(v, float) and v != v) or str(v) == 'None')
    return dict(trials_checked=int(rows.trial.nunique()), trials_detected=int(rows.trial[detected].nunique()),
                occurrences=int(detected.sum()) if occurrences is None else int(occurrences),
                occurrence_unit=occurrence_unit, undecided=int(rows.trial[undecided].nunique()))


class CorrectedLoader:
    """A loader that applies `fix(recording, trial)` to every trial of `base`; everything else is passed on."""

    def __init__(self, base, fix: Callable, name: str = ''):
        self.base, self.fix, self.label = base, fix, name

    def trials(self):
        return self.base.trials()

    def load(self, trial):
        return self.fix(self.base.load(trial), trial)

    def __iter__(self):
        for trial in self.trials():
            yield self.load(trial)

    def __getattr__(self, attr):  # participant_of, session_of, name, reader-specific helpers
        return getattr(self.base, attr)


def map_trials(loader, fn: Callable, processes: int = 1, trials=None) -> list:
    """fn(loader, trial) for every trial, in a process pool when processes > 1 (fn must be a module-level
    function). Trials whose reading fails give None, so one broken file does not stop a dataset."""
    trials = list(loader.trials() if trials is None else trials)
    if processes <= 1:
        return [_safe(fn, loader, t) for t in trials]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    _FORKED.update(fn=fn, loader=loader)  # inherited by the forked workers, so nothing is pickled but the trial
    with ProcessPoolExecutor(processes, mp_context=get_context('fork')) as pool:
        return list(pool.map(_forked, trials))


_FORKED: dict = {}


def _forked(trial):
    return _safe(_FORKED['fn'], _FORKED['loader'], trial)


def _safe(fn, loader, trial):
    try:
        return fn(loader, trial)
    except Exception as exc:  # noqa: BLE001 - a broken trial is reported, not fatal
        return dict(trial=trial, error=f'{type(exc).__name__}: {exc}')


def make_handler(parts: dict) -> Callable:
    """Build the public handle_<error>() from an error module's PARTS entry (spec, check_dataset, prevalence,
    correct, optional loader_for)."""
    spec = parts['spec']

    def handle(datasets=None, *, data: str = 'deposited', notes: dict | None = None, processes: int = 1,
               cache_dir=None, **options) -> ErrorReport:
        return run_error(spec, datasets, data, notes, processes, parts['check_dataset'], parts['prevalence'],
                         parts.get('correct'), parts.get('loader_for'), cache_dir=cache_dir, **options)

    handle.__name__ = f'handle_{spec.name}'
    handle.__doc__ = (f"Error {spec.number} - {spec.description or spec.name}: check `datasets` (default: all 18) and "
                      "return an ErrorReport.\n\n"
                      "data='deposited' checks the files as published (independent count); data='in_order' checks the "
                      "data corrected for every earlier error. notes: {dataset: text or [texts]}. processes: parallel "
                      "workers per dataset. options: thresholds passed on to the check.")
    return handle
