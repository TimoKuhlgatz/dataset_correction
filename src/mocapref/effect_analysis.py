"""Influence of the error correction on relative orientation (joint angle) estimation, with the MEKF, zoe or both.

    from mocapref.effect_analysis import EffectAnalysis

    ea = EffectAnalysis(['camargo', 'is2b'], methods=('MEKF', 'zoe'), processes=16)
    summary = ea.run()          # one row per dataset and method: tuned parameters and test errors per condition
    ea.tests                    # per test trial and joint: error in degrees per condition

    # the effect of one error alone: data corrected for errors 1..11 against 1..12
    EffectAnalysis.for_error(12, ['dfjimu']).run()

Procedure (the same for every condition and method; estimators in effect.py and zoe.py):
    trials      each dataset's saved split (splits/<dataset>.json): half of the participants tune, the other half
                test (by trial for single-subject datasets, by time for one recording). Static trials and trials too
                short from their motion onset (tuning 20 s, test 15 s) are skipped.
    joints      the joints present in every condition of the trial.
    tuning      one parameter pair (measurement noise R, process noise Q) per dataset, condition and method, chosen
                on a logarithmic grid over all tuning trials together (trials weighted by their scored length). When
                the best pair lies on the grid's edge, the grid is widened by two decades on that side, until a step
                lowers the tuning error by less than 0.01 deg or the limits are reached.
    scoring     every trial filtered from its start and scored from 10 s after its motion onset, the filters'
                convergence time. The onset is read once from the first condition, so all conditions are scored
                from the same instant. Masked IMU samples (marked invalid by a correction) run through the filters
                but are not scored; masked reference frames are never scored. Where the mounting correction fitted
                a segment on its own motion window, its joints are scored from that window's end in all conditions.
    error       per joint, the mean angle between estimated and reference relative orientation over the scored
                samples; per trial, the mean over its joints.

Conditions: by default 'deposited' (the files as published; Trento upper limb read with its repaired trial
pairing, as in the paper's effect numbers) and 'corrected' (every handled error corrected in the
agreed order, mocapref.correction.correct_in_order). Any {name: loader factory(dataset)} can be passed instead.
"""
from __future__ import annotations

import os
import traceback
import warnings
from typing import Callable

import pandas as pd

from .effect import combine_trial_grids, default_grids, test_on_trial, trial_grid
from .motion_onset import SEGMENT_WINDOWS_KEY, motion_onset_s
from .correction import previous as _previous
from .tuning_split import effect_datasets, load_split

GRID_LIMITS = dict(MEKF=((1e-14, 1e10), (1e-13, 1e14)), zoe=((1e-14, 1e10), (1e-14, 1e10)))  # (R range, Q range)
STEP_DECADES = 2
MIN_GAIN_DEG = 0.01


def _deposited(dataset):
    from .datasets import load_dataset
    return load_dataset(dataset)


def _widened(values, best, limits):
    values = sorted(values)
    out = []
    if best == values[0]:
        out = [values[0] / 10 ** k for k in range(1, STEP_DECADES + 1) if values[0] / 10 ** k >= limits[0] * (1 - 1e-9)]
    elif best == values[-1]:
        out = [values[-1] * 10 ** k for k in range(1, STEP_DECADES + 1) if values[-1] * 10 ** k <= limits[1] * (1 + 1e-9)]
    return [float(f'{v:g}') for v in out]


def _name(j):
    return f'{j[0]}-{j[1]}'


# Per worker process: the loaders of the analysis being run (set by _init).
_STATE: dict = {}


def _init(conditions, cache_dir):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    _STATE.clear()
    _STATE.update(conditions=conditions, cache_dir=cache_dir, loaders={})


def _recordings(dataset, trial):
    """The trial in every condition, restricted to the joints all conditions have, with one motion onset."""
    conditions = _STATE['conditions']
    if dataset not in _STATE['loaders']:
        _STATE['loaders'][dataset] = {c: f(dataset) for c, f in conditions.items()}
    recs = {c: loader.load(trial) for c, loader in _STATE['loaders'][dataset].items()}
    windows = {}
    for rec in recs.values():
        windows.update(rec.metadata.get(SEGMENT_WINDOWS_KEY) or {})
    if windows:
        for rec in recs.values():
            rec.metadata[SEGMENT_WINDOWS_KEY] = dict(windows)
    common = set.intersection(*({_name(j) for j in rec.joints} for rec in recs.values()))
    for rec in recs.values():
        rec.joints = [j for j in rec.joints if _name(j) in common]
    onset = motion_onset_s(next(iter(recs.values())))
    return recs, sorted(common), onset


def _half(dataset, part):
    halves = load_split(dataset).get('halves')
    return halves[part] if halves else None


def _valid_only(condition):
    return condition != 'deposited'


def _tuning_job(job):
    dataset, trial, method, blocks = job
    warnings.simplefilter('ignore')
    try:
        recs, common, onset = _recordings(dataset, trial)
        if not common:
            return dataset, trial, None, 'no joint in common'
        out = {}
        for c, grids in blocks.items():
            weight, merged = 0.0, {}
            for r_values, q_values in grids:
                weight, grid = trial_grid(recs[c], r_grid=tuple(r_values), q_grid=tuple(q_values), method=method,
                                          half=_half(dataset, 'tuning'), onset_s=onset,
                                          score_valid_imu_only=_valid_only(c))
                merged.update(grid)
            out[c] = (weight, merged)
        return dataset, trial, out, '|'.join(common)
    except Exception:  # noqa: BLE001 - static or too short trials are skipped, the reason kept
        return dataset, trial, None, traceback.format_exc(limit=1).strip().splitlines()[-1]


def _test_job(job):
    dataset, trial, parameters, method = job
    warnings.simplefilter('ignore')
    try:
        recs, common, onset = _recordings(dataset, trial)
        if not common:
            return dataset, trial, None, 'no joint in common'
        return dataset, trial, {c: test_on_trial(recs[c], *parameters[c], method=method, half=_half(dataset, 'test'),
                                                 onset_s=onset, score_valid_imu_only=_valid_only(c))
                                for c in recs}, '|'.join(common)
    except Exception:  # noqa: BLE001
        return dataset, trial, None, traceback.format_exc(limit=1).strip().splitlines()[-1]


class EffectAnalysis:
    """Tune and test MEKF and/or zoe on each dataset under several conditions of the data (module description).

    datasets: names (default: the 18 datasets with a tuning split). methods: 'MEKF', 'zoe' or both.
    conditions: {name: callable(dataset) -> loader}; default deposited against corrected in the agreed order.
    processes: parallel workers. cache_dir: where the in-order corrections keep their check results.
    """

    def __init__(self, datasets=None, methods=('MEKF', 'zoe'), conditions: dict[str, Callable] | None = None,
                 processes: int = 1, cache_dir=None):
        self.datasets = list(effect_datasets()) if datasets is None else (
            [datasets] if isinstance(datasets, str) else list(datasets))
        self.methods = (methods,) if isinstance(methods, str) else tuple(methods)
        self.conditions = conditions or {'deposited': _deposited,
                                         'corrected': _CorrectedFactory(cache_dir)}
        self.processes, self.cache_dir = processes, cache_dir
        self.tuning: pd.DataFrame | None = None
        self.tests: pd.DataFrame | None = None
        self.summary: pd.DataFrame | None = None

    @classmethod
    def for_error(cls, number: int, datasets=None, methods=('MEKF', 'zoe'), processes: int = 1, cache_dir=None):
        """Data corrected for every error before `number` against data corrected also for `number`."""
        return cls(datasets, methods, {f'before_{number}': _CorrectedFactory(cache_dir, _previous(number)),
                                       f'after_{number}': _CorrectedFactory(cache_dir, number)},
                   processes, cache_dir)

    def _map(self, fn, jobs):
        if self.processes <= 1:
            _init(self.conditions, self.cache_dir)
            return [fn(j) for j in jobs]
        from multiprocessing import get_context
        with get_context('fork').Pool(self.processes, _init, (self.conditions, self.cache_dir),
                                      maxtasksperchild=20) as pool:
            return list(pool.imap_unordered(fn, jobs))

    def run(self) -> pd.DataFrame:
        """Tune and test every dataset with every method; returns the summary (also kept in .summary)."""
        tunings, tests, summary = [], [], []
        for method in self.methods:
            t, s, sm = self._run_method(method)
            tunings.append(t), tests.append(s), summary.append(sm)
        self.tuning = pd.concat(tunings, ignore_index=True)
        self.tests = pd.concat(tests, ignore_index=True)
        self.summary = pd.concat(summary, ignore_index=True)
        return self.summary

    def _run_method(self, method):
        conds = list(self.conditions)
        plan = {n: (list(load_split(n)['tuning']), list(load_split(n)['test'])) for n in self.datasets}
        r0, q0 = default_grids(method)
        r_lim, q_lim = GRID_LIMITS[method]
        axes = {n: {c: (list(r0), list(q0)) for c in conds} for n in self.datasets}
        per_trial = {n: {c: {} for c in conds} for n in self.datasets}
        joints, failed = {}, {}
        pending = {n: {c: [(list(r0), list(q0))] for c in conds} for n in self.datasets}
        chosen = {n: {} for n in self.datasets}
        while pending:
            jobs = [(n, t, method, blocks) for n, blocks in pending.items() for t in plan[n][0]
                    if t not in failed.get(n, {})]
            for dataset, trial, result, note in self._map(_tuning_job, jobs):
                if result is None:
                    failed.setdefault(dataset, {})[trial] = note
                    for c in conds:
                        per_trial[dataset][c].pop(trial, None)
                    continue
                joints[(dataset, trial)] = note
                for c, (weight, grid) in result.items():
                    old = per_trial[dataset][c].get(trial, (weight, {}))[1]
                    per_trial[dataset][c][trial] = (weight, {**old, **grid})
            next_pending = {}
            for n in pending:
                for c in pending[n]:
                    if not per_trial[n][c]:
                        continue
                    r_values, q_values = axes[n][c]
                    inside = {(r, q) for r in r_values for q in q_values}
                    r, q, grid = combine_trial_grids([(w, {k: v for k, v in g.items() if k in inside})
                                                      for w, g in per_trial[n][c].values()])
                    previous = chosen[n].get(c)
                    if previous is not None and previous[2] - grid[(r, q)] < MIN_GAIN_DEG:
                        continue
                    chosen[n][c] = (r, q, grid[(r, q)])
                    new_r, new_q = _widened(r_values, r, r_lim), _widened(q_values, q, q_lim)
                    blocks = ([(new_r, q_values)] if new_r else []) + ([(r_values + new_r, new_q)] if new_q else [])
                    if blocks:
                        axes[n][c] = (r_values + new_r, q_values + new_q)
                        next_pending.setdefault(n, {})[c] = blocks
            pending = next_pending
        tuning_rows = [dict(method=method, dataset=n, trial=t, condition='', note=note)
                       for n in self.datasets for t, note in failed.get(n, {}).items()]
        tuning_rows += [dict(method=method, dataset=n, trial=t, condition=c, weight_s=w, joints=joints[(n, t)],
                             **{f'R={r:g},Q={q:g}': e for (r, q), e in sorted(g.items())})
                        for n in self.datasets for c in conds for t, (w, g) in per_trial[n][c].items()]
        jobs = [(n, t, {c: chosen[n][c][:2] for c in conds}, method) for n in self.datasets
                if all(c in chosen[n] for c in conds) for t in plan[n][1]]
        test_rows = []
        for dataset, trial, result, note in self._map(_test_job, jobs):
            row = dict(method=method, dataset=dataset, trial=trial)
            if result is None:
                row['note'] = note
            else:
                row['joints'] = note
                for c, res in result.items():
                    row[f'{c}_deg'] = res['test_error_deg']
                    row.update({f'{c}:{j}': e for j, e in res['per_joint'].items()})
            test_rows.append(row)
        tests = pd.DataFrame(test_rows)
        summary = []
        for n in self.datasets:
            t = tests[tests.dataset == n] if len(tests) else tests
            cols = [f'{c}_deg' for c in conds]
            ok = t.dropna(subset=cols) if all(c in t for c in cols) else t.iloc[:0]
            row = dict(method=method, dataset=n, tuning_trials=len(plan[n][0]), test_trials=len(plan[n][1]),
                       scored=len(ok))
            for c in conds:
                r, q, e = chosen[n].get(c, (None, None, None))
                row.update({f'{c}_R': r, f'{c}_Q': q, f'{c}_tuning_deg': e,
                            f'{c}_test_median_deg': ok[f'{c}_deg'].median() if len(ok) else None,
                            f'{c}_test_mean_deg': ok[f'{c}_deg'].mean() if len(ok) else None})
            summary.append(row)
        return pd.DataFrame(tuning_rows), tests, pd.DataFrame(summary)


class _CorrectedFactory:
    """Picklable loader factory: the dataset corrected in the agreed order up to error `up_to` (None = all)."""

    def __init__(self, cache_dir=None, up_to: int | None = None):
        self.cache_dir, self.up_to = cache_dir, up_to

    def __call__(self, dataset):
        from .correction import correct_in_order
        return correct_in_order(dataset, up_to=self.up_to, cache_dir=self.cache_dir)
