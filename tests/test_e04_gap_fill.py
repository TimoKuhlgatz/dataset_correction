"""Known-answer tests for error 4, gaps filled without a flag: injected gaps and straight-line
fills in a rigid marker cluster are found, correctly flagged gaps and fills that leave three measured markers are
not, and the correction marks exactly the filled frames invalid."""
import numpy as np

from mocapref.errors.e04_gap_fill import (check_for_gap_fill, filled_frames, remove_gap_fill, stretches,
                                     unmeasured_markers)
from mocapref.synthetic import make_recording


def cluster_from(rec, segment, seed=1, noise_mm=0.05):
    """Four markers rigidly on the segment, 50-80 mm from its origin, at 1.2 m height, with measurement noise."""
    ref = rec.segments[segment].reference
    local = np.array([[60.0, 0, 0], [0, 70.0, 0], [0, 0, 80.0], [-50.0, -50.0, 0]])
    pos = np.stack([ref.orientation.apply(p) + [0, 0, 1200.0] for p in local], axis=1)
    return pos + np.random.default_rng(seed).normal(0, noise_mm, pos.shape)


def fill_linear(m, k, first, last):
    """Replace marker k's frames first..last by a straight line between its neighbours (what gap filling does)."""
    a, b = m[first - 1, k], m[last + 1, k]
    w = (np.arange(first, last + 1) - (first - 1)) / (last + 1 - (first - 1))
    m[first:last + 1, k] = a + w[:, None] * (b - a)


def test_injected_fills_and_gaps_are_found():
    rec = make_recording(duration_s=10.0, reference_rate_hz=100.0)
    m = cluster_from(rec, 'upper')
    fill_linear(m, 0, 100, 119)
    fill_linear(m, 1, 100, 119)  # two of four markers filled: 2 measured -> no orientation
    fill_linear(m, 2, 300, 319)  # one marker filled: 3 measured -> orientation still defined
    m[500:510, :2] = np.nan  # visible gap, flagged in the deposit below
    m[600:605, 1:] = np.nan  # visible gap, not flagged
    valid = np.ones(len(m), bool)
    valid[500:510] = False
    bad = filled_frames(m, valid)
    assert stretches(bad) == [(100, 20), (600, 5)]

    seg = rec.segments['upper']
    seg.reference.markers = {f'm{k}': m[:, k] for k in range(4)}
    seg.reference.valid = valid
    result = check_for_gap_fill(rec)
    assert result.detected is True
    assert result.per_segment['upper']['estimate'] == 2 and result.per_segment['upper']['frames'] == 25
    assert result.per_segment['lower']['detected'] is None  # no markers: undecided, not "none"
    fixed = remove_gap_fill(rec)
    assert (~fixed.segments['upper'].reference.valid).sum() == 10 + 25
    assert check_for_gap_fill(fixed).detected is False


def test_real_motion_rest_and_rounding_are_not_fills():
    rec = make_recording(duration_s=10.0)
    m = cluster_from(rec, 'upper', noise_mm=0.05)
    assert not unmeasured_markers(m).any()  # noisy real motion is never exactly on a line
    m[200:260] = m[199]  # held position: first difference exactly zero, a different fingerprint
    assert not unmeasured_markers(m)[201:259].any()
    r = cluster_from(rec, 'upper', noise_mm=0.05)
    fill_linear(r, 0, 400, 409)
    fill_linear(r, 3, 400, 409)
    fill_linear(r, 2, 400, 409)
    stored = np.round(r, 2)  # written with 0.01 mm resolution
    assert not stretches(filled_frames(stored, np.ones(len(r), bool)))  # rounding hides the exact line ...
    assert stretches(filled_frames(stored, np.ones(len(r), bool), resolution=0.01)) == [(400, 10)]  # ... unless known


def test_fixed_construction_needs_every_marker():
    """A triad frame is lost when one of its three markers is filled; a fit to four markers is not."""
    rec = make_recording(duration_s=10.0)
    m = cluster_from(rec, 'upper')
    fill_linear(m, 0, 100, 109)
    seg = rec.segments['upper']
    seg.reference.markers = {f'm{k}': m[:, k] for k in range(4)}
    assert check_for_gap_fill(rec).per_segment['upper']['estimate'] == 0
    triad = check_for_gap_fill(rec, needs={'upper': (('m0', 'm1', 'm2'), None)}).per_segment['upper']
    assert (triad['estimate'], triad['frames'], triad['needed']) == (1, 10, 3)


def test_float32_storage():
    """Markers stored in single precision: the fill is found with resolution='float32', real motion is not."""
    rec = make_recording(duration_s=10.0)
    m = cluster_from(rec, 'upper', noise_mm=0.05)
    for k in (0, 1):
        fill_linear(m, k, 250, 269)
    stored = m.astype(np.float32).astype(float)
    assert stretches(filled_frames(stored, np.ones(len(m), bool), resolution='float32')) == [(250, 20)]


def test_handler_counts_stretches_and_masks_the_filled_frames():
    from memory_loader import MemoryLoader
    from mocapref.errors import e04_gap_fill as H

    data = {}
    for i in range(2):
        rec = make_recording(duration_s=10.0, reference_rate_hz=100.0, seed=i)
        m = cluster_from(rec, 'upper', seed=i)
        if i == 1:
            m[600:605, 1:] = np.nan  # three of four markers missing, reference still valid
            m[800:802, :2] = np.nan
        rec.segments['upper'].reference.markers = {f'm{k}': m[:, k] for k in range(4)}
        data[f't{i}'] = rec
    loader = MemoryLoader(data)  # reader 'synthetic': rounded-file rule, missing markers only
    rows = H.check_dataset(loader, 'synthetic', 1)
    p = H.prevalence(rows)
    assert (p['trials_detected'], p['occurrences'], p['occurrence_unit']) == (1, 2, 'stretches')
    valid = H.correct(loader, rows).load('t1').segments['upper'].reference.valid
    assert (~valid).sum() == 7 and not valid[600:605].any() and not valid[800:802].any()
