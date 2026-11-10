"""Known-answer tests for the handler of error 8, wrong trial pairing: on a small in-memory dataset whose optical
recordings are shifted by one from the third trial on (an aborted take), the shifted trials are found re-paired or
unpaired, the others kept, and an unpaired static recording is not counted. The assignment itself is tested in
test_errors_known_answer.py."""
from memory_loader import MemoryLoader

from mocapref.errors import e08_trial_pairing as H
from mocapref.synthetic import make_recording


def test_shifted_references_are_found_and_static_recordings_not_counted():
    recs = [make_recording(duration_s=40.0, seed=20 + k, segment_names=('a', 'b', 'c')) for k in range(6)]
    data = {f't{k}': recs[k] for k in range(5)}
    for k in (2, 3, 4):  # trial k carries the reference of trial k + 1: no reference of t2's motion is left
        data[f't{k}'] = recs[k].copy()
        for s in ('a', 'b', 'c'):
            data[f't{k}'].segments[s].reference = recs[k + 1].segments[s].reference
    static = make_recording(duration_s=40.0, seed=30, segment_names=('a', 'b', 'c'))
    other = make_recording(duration_s=40.0, seed=31, segment_names=('a', 'b', 'c'))
    for s in ('a', 'b', 'c'):  # matches nothing: its IMU and its reference come from different recordings
        static.segments[s].imu = other.segments[s].imu
    data['static_01'] = static
    loader = MemoryLoader(data)
    rows = H.check_dataset(loader, 'synthetic', 1).set_index('trial')
    assert list(rows.status) == ['kept', 'kept', 'unpaired', 're-paired', 're-paired', 'unpaired']
    assert rows.loc['t3', 'assigned'] == 't2' and rows.loc['static_01', 'detected'] is None
    p = H.prevalence(rows.reset_index())
    assert (p['trials_checked'], p['trials_detected'], p['occurrences'], p['undecided']) == (6, 3, 3, 1)
    assert H.correct(loader, rows.reset_index()) is loader  # only Trento's reader is switched


def test_trento_is_corrected_to_its_repaired_pairing_with_the_earlier_corrections_kept(monkeypatch):
    from mocapref.errors._report import CorrectedLoader
    from mocapref.errors._rows import reader_of

    class Deposited:
        name = 'bonfiglio_deposited'

    chain = CorrectedLoader(CorrectedLoader(Deposited(), lambda rec, t: rec, 'first'), lambda rec, t: rec, 'second')
    import mocapref.datasets as D
    monkeypatch.setattr(D, 'load_dataset', lambda name: type('Repaired', (), {'name': name})())
    fixed = H.correct(chain, None)
    assert reader_of(fixed) == 'bonfiglio' and fixed.label == 'second' and fixed.base.label == 'first'
    assert H.loader_for('bonfiglio', 'deposited').name == 'bonfiglio_deposited'
    assert H.loader_for('camargo', 'deposited') is None
