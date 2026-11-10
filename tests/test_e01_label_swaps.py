"""Known-answer tests for error 1, marker labels swapped within a cluster: a swap for a whole
trial, from a frame on and for a few frames is found and repaired; fast real motion and clean trials are not swaps."""
import numpy as np

from scipy.spatial.transform import Rotation

from mocapref.errors.e01_label_swaps import (SessionReference, _kabsch, check_for_label_swaps, local_positions,
                                        remove_label_swaps)
from mocapref.synthetic import make_recording

LOCAL = np.array([[60.0, 0, 0], [0, 70.0, 0], [0, 0, 80.0], [-50.0, -50.0, 0]])


def cluster(rec, noise_mm=0.05, seed=0):
    ref = rec.segments['upper'].reference
    pos = np.stack([ref.orientation.apply(p) + [0, 0, 1200.0] for p in LOCAL], axis=1)
    return pos + np.random.default_rng(seed).normal(0, noise_mm, pos.shape)


def session(seeds, noise_mm=0.05):
    recs = [make_recording(duration_s=10.0, seed=s) for s in seeds]
    return recs, [cluster(r, noise_mm, s) for r, s in zip(recs, seeds)]


def references(clusters, static=()):
    return {'upper': SessionReference(clusters, static)}


def attach(rec, m, built_from_markers=False):
    """Give the segment its markers; optionally rebuild the stored orientation from them, as a deposit does (so
    swapped markers give a wrong orientation)."""
    ref = rec.segments['upper'].reference
    ref.markers = {f'm{k}': m[:, k] for k in range(4)}
    if built_from_markers:
        ref.orientation = Rotation.from_matrix(np.stack([_kabsch(LOCAL, f) for f in m]))
    return rec


def test_three_kinds_of_swap_are_found_and_repaired():
    recs, clusters = session(range(8))
    refs = references(clusters)
    for rec, m in zip(recs[:5], clusters[:5]):  # clean trials: nothing found
        assert check_for_label_swaps(attach(rec, m), refs).detected is False
    whole, onward, brief = clusters[5].copy(), clusters[6].copy(), clusters[7].copy()
    whole[:, [0, 1]] = whole[:, [1, 0]]
    onward[500:, [0, 2]] = onward[500:, [2, 0]]
    brief[300:306, [1, 3]] = brief[300:306, [3, 1]]
    expected = {5: (0, 1000, (0, 1)), 6: (500, 500, (0, 2)), 7: (300, 6, (1, 3))}
    truths = {i: recs[i].segments['upper'].reference.orientation for i in expected}
    session_local = {'upper': np.median([local_positions(m, r.segments['upper'].reference.orientation,
                                                         np.ones(len(m), bool)) for r, m in zip(recs[:5], clusters[:5])],
                                        axis=0)}
    for i, m in ((5, whole), (6, onward), (7, brief)):
        rec = attach(recs[i], m, built_from_markers=True)
        res = check_for_label_swaps(rec, refs)
        swaps = res.per_segment['upper']['swaps']
        assert [(s['first'], s['frames'], s['pair']) for s in swaps] == [expected[i]]
        fixed = remove_label_swaps(rec, res, session_local).segments['upper'].reference
        truth = truths[i]
        err = np.degrees((fixed.orientation.inv() * truth).magnitude())
        assert err.max() < 0.5  # orientation of the repaired frames within half a degree
        assert np.allclose(fixed.markers['m0'], clusters[i][:, 0])


def test_static_trial_sets_the_threshold_and_fast_motion_is_no_swap():
    recs, clusters = session(range(6), noise_mm=0.5)  # skin-like noise
    static = [clusters[0][:200]]
    refs = references(clusters, static)
    assert refs['upper'].source == 'static trial' and refs['upper'].threshold > 0
    fast = make_recording(duration_s=10.0, seed=11)
    fast.segments['upper'].reference.orientation = fast.segments['upper'].reference.orientation ** 6.0
    m = cluster(fast, 0.5, 11)
    assert check_for_label_swaps(attach(fast, m), refs).detected is False


def test_mirror_pair_is_not_checkable():
    """Markers 2 and 3 are mirror images within the cluster: exchanging them leaves every distance unchanged, so
    the pair is named as not checkable and never reported as swapped."""
    local = np.array([[0.0, 0, 0], [100.0, 0, 0], [50.0, 40.0, 0], [50.0, -40.0, 0]])
    recs = [make_recording(duration_s=10.0, seed=s) for s in range(5)]
    clusters = [np.stack([r.segments['upper'].reference.orientation.apply(p) + [0, 0, 1200.0] for p in local], axis=1)
                + np.random.default_rng(s).normal(0, 0.05, (1000, 4, 3)) for s, r in enumerate(recs)]
    refs = references(clusters)
    m = clusters[4].copy()
    m[:, [2, 3]] = m[:, [3, 2]]
    res = check_for_label_swaps(attach(recs[4], m), refs).per_segment['upper']
    assert res['detected'] is False and ('m2', 'm3') in res['not_checkable']


def test_handler_counts_and_repairs_a_swap_in_a_dataset():
    """check_dataset / prevalence / correct on a small in-memory dataset: one participant, six trials, one brief swap."""
    import pandas as pd
    from memory_loader import MemoryLoader
    from mocapref.errors import e01_label_swaps as H

    recs, clusters = session(range(6))
    truth = recs[5].segments['upper'].reference.orientation
    swapped = clusters[5].copy()
    swapped[300:306, [1, 3]] = swapped[300:306, [3, 1]]
    data = {f't{i}': attach(r, swapped if i == 5 else m, built_from_markers=True)
            for i, (r, m) in enumerate(zip(recs, clusters))}
    loader = MemoryLoader(data)
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_detected'], p['occurrences'], p['occurrence_unit']) == (1, 1, 'pieces')
    assert p['undecided'] == 6  # 'lower' has no markers in any trial
    rows = pd.read_csv(pd.io.common.StringIO(rows.to_csv(index=False)), keep_default_na=False, na_values=[''])
    fixed = H.correct(loader, rows).load('t5').segments['upper'].reference  # also from rows read back from CSV
    assert np.degrees((fixed.orientation.inv() * truth).magnitude()).max() < 0.5
    assert np.allclose(fixed.markers['m1'], clusters[5][:, 1])
    assert H.correct(loader, rows).load('t0').corrections == []
