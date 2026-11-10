"""Correct the errors in the agreed order.

The errors are numbered in the order in which they are detected and corrected: the check of error N runs on data
already corrected for every error with a smaller number. "Corrected" means: flagged samples (clipped, overflowed,
dead, stuck, corrupted) and unreliable reference frames are marked invalid, real repairs are applied (swapped labels
exchanged back, trials re-paired, IMUs given back to their segments, time axes rescaled and shifted, mirrored axes
negated, offsets subtracted, mounting rotated out, gravity restored), and errors without a repair stay.

    from mocapref.correction import correct_in_order, run_in_order

    loader = correct_in_order('camargo')            # every handled error corrected
    rec = loader.load(loader.trials()[0])
    loader = correct_in_order('camargo', up_to=12)  # corrected for errors 1-12 only
    reports = run_in_order(['camargo', 'wmcg'])     # {number: ErrorReport}, each check on the data corrected before it

Each check runs over a whole dataset (several corrections pool a participant's trials), so the first call per
dataset and error takes as long as the check itself. With cache_dir set, every report's rows are saved there and
read back on the next call instead of running the check again.

Readers: Trento upper limb ('bonfiglio') is read with the deposit's own trial pairing until its pairing is
corrected (error 8), with the repaired pairing afterwards. Geneva ('grouvel') is read from the streams rebuilt from
its raw session logs (error 10, mis-assembled streams) already from the pairing check on, since its pairing cannot be
judged on the mis-assembled files.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Number of each handled error -> module in mocapref.errors. Not in the list: error 23 (unused) and 24 (set aside).
ORDER: dict[int, str] = {
    1: 'e01_label_swaps', 2: 'e02_misplaced_markers', 3: 'e03_reference_jumps', 4: 'e04_gap_fill',
    5: 'e05_clipping', 6: 'e06_integer_overflow', 7: 'e07_dead_channels',
    8: 'e08_trial_pairing', 9: 'e09_segment_assignment', 10: 'e10_mis_assembled_streams',
    11: 'e11_sampling_rate', 12: 'e12_time_offset', 13: 'e13_clock_drift', 14: 'e14_time_scales',
    15: 'e15_corrupted_samples', 16: 'e16_mirrored_gyroscope', 17: 'e17_gyroscope_offset', 18: 'e18_mounting',
    19: 'e19_e20_accelerometer_sign', 20: 'e19_e20_accelerometer_sign', 21: 'e21_accelerometer_bias',
    22: 'e22_gravity_removed',
}
ALL_CORRECTED = max(ORDER) + 1


def parts(number: int) -> dict:
    """The PARTS entry (spec, check_dataset, prevalence, correct) of error `number`."""
    module = importlib.import_module(f'mocapref.errors.{ORDER[number]}')
    return module.PARTS[number]


def base_reader(dataset: str, number: int) -> str:
    """The reader the check of error `number` starts from (module description, "Readers")."""
    if dataset == 'bonfiglio' and number <= 8:
        return 'bonfiglio_deposited'
    if dataset == 'grouvel' and number >= 8:
        return 'grouvel_raw'
    return dataset


def _saved_rows(cache_dir, number: int, dataset: str) -> pd.DataFrame | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f'{number:02d}_{parts(number)["spec"].name}' / 'in_order' / f'{dataset}.csv'
    if not path.exists():
        return None
    text = {c: str for c in ('dataset', 'trial', 'participant', 'session', 'segment')}  # '01' stays '01'
    return pd.read_csv(path, keep_default_na=False, na_values=[''], dtype=text)


@lru_cache(maxsize=256)
def _rows_in_order(dataset: str, number: int, cache_dir: str | None) -> pd.DataFrame:
    """The rows of error `number`'s check on `dataset` in the agreed order (from the cache, or by running it)."""
    rows = _saved_rows(cache_dir, number, dataset)
    if rows is not None:
        return rows
    from .errors._report import make_handler
    report = make_handler(parts(number))(dataset, data='in_order', cache_dir=cache_dir)
    rows = report.rows
    if cache_dir is not None:
        path = Path(cache_dir) / f'{number:02d}_{parts(number)["spec"].name}' / 'in_order'
        path.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path / f'{dataset}.csv', index=False)
    return rows


def corrected_before(dataset: str, number: int, cache_dir=None):
    """A loader of `dataset` corrected for every handled error with a number smaller than `number`."""
    from .datasets import load_dataset
    loader = load_dataset(base_reader(dataset, number))
    cache = None if cache_dir is None else str(cache_dir)
    for n in sorted(ORDER):
        if n >= number:
            break
        correct = parts(n).get('correct')
        if correct is not None:
            rows = _rows_in_order(dataset, n, cache)
            loader = correct(loader, rows[rows.dataset == dataset] if 'dataset' in rows else rows)
    return loader


def correct_in_order(dataset: str, up_to: int | None = None, cache_dir=None):
    """A loader of `dataset` with errors 1..up_to corrected in the agreed order (default: every handled error)."""
    return corrected_before(dataset, ALL_CORRECTED if up_to is None else up_to + 1, cache_dir=cache_dir)


def run_in_order(datasets=None, errors=None, cache_dir=None, notes: dict | None = None) -> dict:
    """{number: ErrorReport} of every handled error (or of `errors`), each check on the data corrected for all
    earlier errors. notes: {number: {dataset: text}}."""
    from .errors._report import make_handler
    out = {}
    for n in sorted(ORDER if errors is None else errors):
        out[n] = make_handler(parts(n))(datasets, data='in_order', cache_dir=cache_dir,
                                        notes=(notes or {}).get(n))
    return out


def prevalence_table(reports: dict) -> pd.DataFrame:
    """One table from several reports: a row per dataset, a 'trials / occurrences' column per error."""
    cols = {}
    for n, report in sorted(reports.items()):
        p = report.prevalence.set_index('dataset')
        cols[f'{n} {report.spec.name}'] = p.trials_detected.astype(str) + ' / ' + p.occurrences.astype(str)
    return pd.DataFrame(cols)
