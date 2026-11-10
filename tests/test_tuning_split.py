"""Known-answer tests of the tuning/test split (tuning_split.py) and of tuning across trials (effect.py)."""
import numpy as np
import pandas as pd
import pytest

from mocapref import tuning_split as S


def lengths(rows):
    return pd.DataFrame(rows, columns=['trial', 'participant', 'duration_s'])


def five_participants():
    rows = [(f'P{p}/t{k}', f'P{p}', 60.0) for p in range(5) for k in range(3)]
    rows += [('P0/short', 'P0', 17.0), ('P1/short', 'P1', 17.0), ('P2/tiny', 'P2', 10.0), ('P3/broken', 'P3', np.nan)]
    return lengths(rows)


def test_participants_are_halved_with_the_extra_one_in_the_test_half():
    split = S.make_split('x', five_participants())
    assert split['unit'] == 'participant'
    assert len(split['tuning_participants']) == 2 and len(split['test_participants']) == 3
    assert not set(split['tuning_participants']) & set(split['test_participants'])
    tuning_of = {t.split('/')[0] for t in split['tuning']}
    test_of = {t.split('/')[0] for t in split['test']}
    assert tuning_of == set(split['tuning_participants']) and test_of == set(split['test_participants'])


def test_short_trials_only_in_the_test_half_and_never_below_15_s():
    split = S.make_split('x', five_participants())
    assert split['excluded']['P2/tiny'] == 'shorter than 15 s' and split['excluded']['P3/broken'] == 'not readable'
    for t in ('P0/short', 'P1/short'):  # 17 s: test if its participant tests, otherwise not used
        participant = t.split('/')[0]
        assert (t in split['test']) == (participant in split['test_participants'])
        assert t not in split['tuning']
        assert (t in split['excluded']) == (participant in split['tuning_participants'])


def test_the_same_seed_gives_the_same_split_and_another_seed_can_differ():
    a, b = S.make_split('x', five_participants()), S.make_split('x', five_participants())
    assert a['tuning'] == b['tuning']
    draws = {tuple(S.make_split('x', five_participants(), seed=s)['tuning_participants']) for s in range(20)}
    assert len(draws) > 1


def test_single_participant_and_diodem_split_by_trial():
    rows = lengths([(f'robot/t{k}', 'robot', 30.0) for k in range(7)])
    split = S.make_split('x', rows)
    assert split['unit'] == 'trial' and len(split['tuning']) == 3 and len(split['test']) == 4
    chains = lengths([(f'{c}/t{k}', c, 100.0) for c in ('arm', 'gait') for k in range(5)])
    assert S.make_split('diodem', chains)['unit'] == 'trial'
    assert S.make_split('other', chains)['unit'] == 'participant'


def test_a_single_trial_is_split_in_time():
    split = S.make_split('x', lengths([('chain/rec', 'chain', 475.0)]))
    assert split['unit'] == 'time' and split['tuning'] == split['test'] == ['chain/rec']


def test_a_saved_split_is_read_back_and_not_drawn_again(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'SPLITS_DIR', tmp_path)
    split = S.make_split('x', five_participants())
    S.save_split(split)
    assert S.load_split('x') == split
    with pytest.raises(FileExistsError):
        S.save_split(S.make_split('x', five_participants(), seed=1))


def test_tuning_across_trials_gives_a_low_error_on_unseen_trials():
    """Clean synthetic trials: parameters tuned on two trials give a small error on two others, and a trial split in
    time is scored on its second half only."""
    from mocapref.effect import test_on_trial, tune_on_trials
    from mocapref.synthetic import make_recording
    trials = [make_recording(duration_s=40.0, reference_rate_hz=100.0, seed=s, linear_acc_mps2=3.0) for s in range(4)]
    r, q, grid = tune_on_trials(trials[:2])
    assert grid[(r, q)] < 2.0
    for rec in trials[2:]:
        assert test_on_trial(rec, r, q)['test_error_deg'] < 2.0
    assert test_on_trial(trials[2], r, q, half='second half')['test_error_deg'] < 2.0


def test_tuning_ignores_the_first_10_s():
    """A reference rotated by 90 degrees during the first 10 s of a tuning trial must not change the tuning error."""
    from scipy.spatial.transform import Rotation

    from mocapref.effect import tune_on_trials
    from mocapref.synthetic import make_recording
    clean = make_recording(duration_s=30.0, reference_rate_hz=100.0, seed=0, linear_acc_mps2=3.0)
    broken = clean.copy()
    ref = next(iter(broken.segments.values())).reference
    early = ref.times < ref.start_s + 9.0
    q = ref.orientation.as_quat()
    q[early] = (ref.orientation[np.flatnonzero(early)] * Rotation.from_euler('z', 90, degrees=True)).as_quat()
    ref.orientation = Rotation.from_quat(q)
    grid = dict(r_grid=(1e-2,), q_grid=(1.0,))
    assert tune_on_trials([broken], **grid)[2] == pytest.approx(tune_on_trials([clean], **grid)[2])
    assert tune_on_trials([broken], skip_s=0.0, **grid)[2][(1e-2, 1.0)] > \
        tune_on_trials([clean], skip_s=0.0, **grid)[2][(1e-2, 1.0)] + 10.0


def test_tuning_weights_trials_by_their_assessed_length():
    """Tuning error of a 70 s and a 21 s trial = their errors weighted by 60 s and 11 s scored (a trial needs 20 s
    from its motion onset to tune)."""
    from mocapref.effect import test_on_trial, tune_on_trials
    from mocapref.synthetic import make_recording
    long, short = (make_recording(duration_s=d, reference_rate_hz=100.0, seed=s, linear_acc_mps2=3.0)
                   for d, s in ((70.0, 0), (21.0, 1)))
    r, q = 1e-2, 1.0
    tuned = tune_on_trials([long, short], r_grid=(r,), q_grid=(q,))[2][(r, q)]
    e_long, e_short = (test_on_trial(rec, r, q)['test_error_deg'] for rec in (long, short))
    assert tuned == pytest.approx((60 * e_long + 11 * e_short) / 71, rel=1e-3)
    assert abs(e_long - e_short) > 1e-3  # otherwise the test could not tell weighted from unweighted


def test_effect_datasets_are_the_18_with_a_saved_tuning_and_test_half():
    chosen = S.effect_datasets()
    assert len(chosen) == 18
    assert {'bonfiglio', 'grimmer_stairs', 'ultramocap'} <= set(chosen)  # too short under the old 40 s rule
    assert chosen == [d for d in S.OVERVIEW_ORDER if d in chosen]


def test_overview_order_lists_every_paper_dataset_once():
    from mocapref.datasets import PAPER_DATASETS
    assert sorted(S.OVERVIEW_ORDER) == sorted(PAPER_DATASETS) and len(set(S.OVERVIEW_ORDER)) == 18
