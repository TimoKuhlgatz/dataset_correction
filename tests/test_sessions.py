"""Sessions as the documentation states them (plus Trento's running clocks and IS2B's blocks): Geneva per
participant, Trento per participant except subject 9's last trial, IS2B per plane and range of motion with the
functional trials alone, every other dataset each trial alone. No data needed: the trial keys are set directly."""
from mocapref.datasets._bridge import READER_LOADERS


def loader(name, trials):
    cls = next(c for c in READER_LOADERS if c.name == name)
    obj = object.__new__(cls)
    obj.__dict__['_trial_keys'] = {t: tuple(t.split('/', 1)) for t in trials}
    return obj


def test_geneva_one_session_per_participant():
    g = loader('grouvel', ['P01/2minWalk_01', 'P01/Static_01', 'P02/2minWalk_01'])
    assert [g.session_of(t) for t in g.trials()] == ['P01', 'P01', 'P02']


def test_trento_per_participant_except_subject_9_last_trial():
    for name in ('bonfiglio', 'bonfiglio_deposited'):
        b = loader(name, ['9/t19-task-9', '9/t20-task-10', '1/t20-task-10'])
        assert [b.session_of(t) for t in b.trials()] == ['9', '9/t20-task-10', '1']


def test_is2b_blocks_of_plane_and_range_of_motion():
    i = loader('is2b', ['cadaver-limb-01/H_120_f_110', 'cadaver-limb-01/H_15_s_110', 'cadaver-limb-01/H_15_s_60',
                        'cadaver-limb-01/V_Fun_FemFix'])
    assert [i.session_of(t) for t in i.trials()] == ['cadaver-limb-01/H_110', 'cadaver-limb-01/H_110',
                                                     'cadaver-limb-01/H_60', 'cadaver-limb-01/V_Fun_FemFix']


def test_undocumented_datasets_keep_each_trial_alone():
    for name in ('fullbody17', 'wmcg', 'grimmer_stairs', 'camargo', 'ultramocap'):
        d = loader(name, ['S01/a', 'S01/b'])
        assert [d.session_of(t) for t in d.trials()] == ['S01/a', 'S01/b']
