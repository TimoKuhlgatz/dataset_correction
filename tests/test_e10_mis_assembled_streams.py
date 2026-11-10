"""Known-answer tests for the handler of error 10, mis-assembled streams: a dataset without a general rule is
reported as not checked (undecided, with a note), and the correction switches Geneva's reader to the streams rebuilt
from its raw logs while keeping every earlier correction. The rebuild itself is tested in test_grouvel_raw.py."""
from memory_loader import MemoryLoader

from mocapref.errors import e10_mis_assembled_streams as H
from mocapref.errors._report import CorrectedLoader
from mocapref.errors._rows import reader_of
from mocapref.synthetic import make_recording


def test_other_datasets_are_not_checked():
    loader = MemoryLoader({'t0': make_recording(duration_s=5.0), 't1': make_recording(duration_s=5.0, seed=1)})
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_checked'], p['trials_detected'], p['occurrences'], p['undecided']) == (2, 0, 0, 2)
    assert H.correct(loader, rows) is loader


def test_handler_notes_unchecked_datasets(monkeypatch):
    loader = MemoryLoader({'t0': make_recording(duration_s=5.0)})
    import mocapref.errors._report as R
    monkeypatch.setattr(R, 'source_loader', lambda name, data, number, cache_dir=None: loader)
    report = H.handle_mis_assembled_streams(['synthetic'])
    assert report.notes['synthetic'] == [H.NOT_CHECKED]
    assert report.prevalence.trials_detected.iloc[0] == 0


def test_geneva_is_switched_to_the_rebuilt_streams_with_the_earlier_corrections_kept(monkeypatch):
    class Published:
        name = 'grouvel'

    chain = CorrectedLoader(Published(), lambda rec, t: rec, 'clipping')
    import mocapref.datasets as D
    monkeypatch.setattr(D, 'load_dataset', lambda name: type('Rebuilt', (), {'name': name})())
    fixed = H.correct(chain, None)
    assert reader_of(fixed) == 'grouvel_raw' and fixed.label == 'clipping'
    assert H.correct(fixed, None) is fixed  # already rebuilt: unchanged
