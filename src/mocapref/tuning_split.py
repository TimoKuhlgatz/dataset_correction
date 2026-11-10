"""The split of each dataset into a tuning half and a test half for the effect analyses.

Rules:
    - 50/50 by participants: half of the participants (rounded down) tune the filter, the others test it, so no
      participant is in both halves. The recorded time in the two halves need not be equal.
    - A dataset with a single participant (a specimen, a robot, an apparatus), and DIODEM (its trials vary a lot in
      motion), is split by trial instead: half of the trials (rounded down) tune, the others test.
    - A dataset with a single trial (mfimt_d01) is split in time: the first half of the trial tunes, the second half
      tests.
    - When a dataset cannot be halved exactly, the test half gets the extra participant or trial.
    - Tuning trials must be at least MIN_TUNING_S (20 s) long, test trials at least MIN_TEST_S (15 s); shorter trials
      are not used. The tuning participants (or trials) are drawn among those with at least one trial of 20 s; a
      tuning participant's trials of 15-20 s are not used, so they cannot put that participant into the test half.
    - The first CONVERGENCE_S (10 s) after a trial's motion onset (motion_onset.py) are not scored, while the filter
      converges, in tuning and in testing. The minimum lengths are applied from the onset too when the trials are
      used: a test trial needs 15 s from onset, a tuning trial 20 s, and a static trial (no onset) is not used.
      Trials failing this are left out at use time (`motion_onset.long_enough`).
    - The tuning error weights each tuning trial by its scored length, so long trials count more (effect.py,
      `tune_on_trials`).

The draw is random with a seed per dataset. Each split is saved as splits/<dataset>.json and read back, never drawn
again, so the deposited and the corrected data of a dataset are tuned and tested on exactly the same trials. The
trial lengths used for the draw are the shortest overlap of IMU and reference over a trial's segments, on the
deposited data.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import zlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / 'splits'
MIN_TUNING_S = 20.0
MIN_TEST_S = 15.0
CONVERGENCE_S = 10.0
BY_TRIAL = {'diodem'}  # several participants (chains) in the loader, but split by trial
# The 18 datasets in reporting order: humans, then mechanical chains and robots.
OVERVIEW_ORDER = ('tuk6min', 'opensense', 'fullbody17', 'camargo', 'wmcg', 'compliant_terrain', 'hip_rom_y', 'is2b',
                  'grouvel', 'bonfiglio', 'grimmer_stairs', 'ultramocap', 'diodem', 'dfjimu', 'mfimt_d01', 'reemc',
                  'openaxes', 'repoimu')


def default_seed(dataset: str) -> int:
    """A fixed seed per dataset, so that drawing a split again gives the same split."""
    return zlib.crc32(dataset.encode())


def _draw(candidates: list, n: int, rng: np.random.Generator) -> list:
    return sorted(rng.choice(sorted(candidates), size=min(n, len(candidates)), replace=False).tolist())


def make_split(dataset: str, trials: pd.DataFrame, seed: int | None = None) -> dict:
    """Draw the split of one dataset from its trials (columns trial, participant, duration_s; NaN duration = not
    readable)."""
    seed = default_seed(dataset) if seed is None else seed
    rng = np.random.default_rng(seed)
    excluded = {}
    for t, d in zip(trials.trial, trials.duration_s):
        if not np.isfinite(d):
            excluded[t] = 'not readable'
        elif d < MIN_TEST_S:
            excluded[t] = f'shorter than {MIN_TEST_S:g} s'
    usable = trials[~trials.trial.isin(list(excluded))]
    long_enough = usable[usable.duration_s >= MIN_TUNING_S]
    split = dict(dataset=dataset, seed=seed, created=datetime.date.today().isoformat(),
                 min_tuning_s=MIN_TUNING_S, min_test_s=MIN_TEST_S, convergence_s=CONVERGENCE_S)
    participants = sorted(usable.participant.unique())
    if len(usable) == 1:
        trial = usable.trial.iloc[0]
        if usable.duration_s.iloc[0] / 2 < MIN_TEST_S:
            raise ValueError(f'{dataset}: the single trial is too short to split in time')
        split.update(unit='time', tuning=[trial], test=[trial], halves=dict(tuning='first half', test='second half'))
    elif len(participants) >= 2 and dataset not in BY_TRIAL:
        tuning = _draw(long_enough.participant.unique().tolist(), len(participants) // 2, rng)
        test = [p for p in participants if p not in tuning]
        for t in usable.trial[usable.participant.isin(tuning) & (usable.duration_s < MIN_TUNING_S)]:
            excluded[t] = f'shorter than {MIN_TUNING_S:g} s, participant in the tuning half'
        split.update(unit='participant', tuning_participants=tuning, test_participants=test,
                     tuning=long_enough.trial[long_enough.participant.isin(tuning)].tolist(),
                     test=usable.trial[usable.participant.isin(test)].tolist())
    else:
        tuning = _draw(long_enough.trial.tolist(), len(usable) // 2, rng)
        split.update(unit='trial', tuning=[t for t in usable.trial if t in tuning],
                     test=[t for t in usable.trial if t not in tuning])
    split['excluded'] = excluded
    return split


def split_path(dataset: str) -> pathlib.Path:
    return SPLITS_DIR / f'{dataset}.json'


def save_split(split: dict, overwrite: bool = False) -> pathlib.Path:
    path = split_path(split['dataset'])
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists; a saved split is reused, not drawn again')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + '\n')
    return path


def load_split(dataset: str) -> dict:
    return json.loads(split_path(dataset).read_text())


def get_split(dataset: str) -> dict:
    """The saved split of the dataset (splits/<dataset>.json)."""
    return load_split(dataset)


def has_usable_split(dataset: str) -> bool:
    """True if the dataset has a saved split with at least one tuning and one test trial."""
    if not split_path(dataset).exists():
        return False
    split = load_split(dataset)
    return bool(split.get('tuning')) and bool(split.get('test'))


def effect_datasets(candidates=OVERVIEW_ORDER) -> list[str]:
    """The datasets that go into the effect analyses: those of `candidates` with a usable saved split, in the
    order given (by default OVERVIEW_ORDER)."""
    return [d for d in candidates if has_usable_split(d)]
