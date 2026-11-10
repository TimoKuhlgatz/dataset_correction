"""Known-answer tests for error 2, a cluster marker reconstructed in the wrong place: an injected
displacement of one marker for a few frames is found, the right marker is blamed and the orientation is rebuilt; a
clean trial gives nothing beyond its noise tail, three-marker clusters are not checkable, and a shift to the end of
the trial is reported, not counted."""
import numpy as np
from scipy.spatial.transform import Rotation

from mocapref.errors.e01_label_swaps import _kabsch
from mocapref.errors.e02_misplaced_markers import check_for_misplaced_markers, find_misplaced, remove_misplaced_markers
from mocapref.synthetic import make_recording

LOCAL = np.array([[60.0, 0, 0], [0, 70.0, 0], [0, 0, 80.0], [-50.0, -50.0, 0]])


def cluster(rec, noise_mm=0.1, seed=0):
    ref = rec.segments['upper'].reference
    pos = np.stack([ref.orientation.apply(p) + [0, 0, 1200.0] for p in LOCAL], axis=1)
    return pos + np.random.default_rng(seed).normal(0, noise_mm, pos.shape)


def attach(rec, m):
    """Markers on the segment and the stored orientation built from them, as a deposit does."""
    ref = rec.segments['upper'].reference
    ref.markers = {f'm{k}': m[:, k] for k in range(m.shape[1])}
    ref.orientation = Rotation.from_matrix(np.stack([_kabsch(LOCAL[:m.shape[1]], f) for f in m]))
    return rec


def test_injected_misplacement_is_found_and_repaired():
    rec = make_recording(duration_s=10.0, seed=3)
    truth = rec.segments['upper'].reference.orientation
    m = cluster(rec)
    m[400:408, 2] += [15.0, -10.0, 5.0]  # marker 2 off by about 19 mm for 8 frames
    m[700:703, 0] += [0, 6.0, 0]  # marker 0 off by 6 mm for 3 frames
    rec = attach(rec, m)
    res = check_for_misplaced_markers(rec)
    runs = res.per_segment['upper']['runs']
    found = [(r['first'], r['frames'], r['marker']) for r in runs]
    assert (400, 8, 2) in found and (700, 3, 0) in found
    noise = [r for r in runs if r['first'] not in (400, 700)]  # the far-out fence's tail of pure noise
    assert noise == []  # pure noise stays below the 1 mm floor
    before = np.degrees((rec.segments['upper'].reference.orientation.inv() * truth).magnitude())[400:408]
    fixed = remove_misplaced_markers(rec, res).segments['upper'].reference
    after = np.degrees((fixed.orientation.inv() * truth).magnitude())[400:408]
    assert before.min() > 3 and after.max() < 0.5
    assert fixed.valid[400:408].all()


def test_clean_trial_lasting_shift_and_three_markers():
    rec = make_recording(duration_s=10.0, seed=4)
    m = cluster(rec, seed=1)
    res = find_misplaced(m)
    assert res['runs'] == []  # the noise tail stays below the 1 mm floor
    shifted = m.copy()
    shifted[800:, 1] += [0, 0, 12.0]  # marker 1 moved for good: reaches the trial's end
    res = find_misplaced(shifted)
    assert not any(r['marker'] == 1 and r['frames'] > 10 for r in res['runs'])
    assert any(r['marker'] == 1 and r['first'] == 800 for r in res['lasting'])
    three = attach(make_recording(duration_s=5.0, seed=5), cluster(make_recording(duration_s=5.0, seed=5))[:, :3])
    assert check_for_misplaced_markers(three).per_segment['upper']['detected'] is None


def test_one_millimetre_floor():
    """Below the accuracy of current optical systems (1 mm, Topley & Richards 2020) a displacement does not count,
    even where the trial's own fence is far lower (a near noise-free cluster, as DIODEM's computed markers)."""
    rec = make_recording(duration_s=10.0, seed=6)
    m = cluster(rec, noise_mm=0.001)
    m[300:305, 3] += [0.8, 0, 0]
    m[600:605, 3] += [1.5, 0, 0]
    res = find_misplaced(m)
    assert res['fence_mm'] < 0.1 and res['limit_mm'] == 1.0
    assert [(r['first'], r['frames'], r['marker']) for r in res['runs']] == [(600, 5, 3)]


def test_handler_counts_runs_and_leaves_the_frames_as_deposited():
    from memory_loader import MemoryLoader
    from mocapref.errors import e02_misplaced_markers as H

    data = {}
    for i in range(3):
        rec = make_recording(duration_s=10.0, seed=10 + i)
        m = cluster(rec, seed=i)
        if i == 1:
            m[400:408, 2] += [15.0, -10.0, 5.0]
        data[f't{i}'] = attach(rec, m)
    loader = MemoryLoader(data)
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_detected'], p['occurrences'], p['occurrence_unit']) == (1, 1, 'runs')
    assert rows.fence_mm_vs_session.notna().sum() == 3
    assert H.correct(loader, rows) is loader  # report only
