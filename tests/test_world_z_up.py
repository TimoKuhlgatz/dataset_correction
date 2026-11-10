"""Known-answer test for turning a +y-up reference world to +z up (reference vertical, not handled as an error, since
a constant rotation of the world cancels in the relative orientation): the vertical moves to +z, marker positions turn with it, and the relative
orientation of two segments, the quantity scored, is unchanged."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from mocapref.datasets._bridge import WORLD_TO_Z_UP, _reference_stream


def _pose(rotations, markers):
    return SimpleNamespace(t=np.arange(len(rotations)) / 100.0, R=rotations.as_matrix(), rate_hz=100.0,
                           markers=markers, marker_names=['a', 'b'])


def test_turn_carries_y_up_to_z_up_and_keeps_relative_orientation():
    rng = np.random.default_rng(0)
    upper, lower = Rotation.random(50, random_state=1), Rotation.random(50, random_state=2)
    markers = rng.normal(size=(50, 2, 3))
    markers[:, :, 1] += 1.0  # markers lie above the floor along +y
    markers[7, 0] = np.nan
    turn = WORLD_TO_Z_UP['camargo']
    assert np.allclose(turn.apply([0, 1, 0]), [0, 0, 1])
    a, _ = _reference_stream(_pose(upper, markers), 0.0, turn)
    b, _ = _reference_stream(_pose(lower, markers), 0.0, turn)
    a0, _ = _reference_stream(_pose(upper, markers), 0.0)
    b0, _ = _reference_stream(_pose(lower, markers), 0.0)
    assert np.allclose(((a.orientation.inv() * b.orientation) * (a0.orientation.inv() * b0.orientation).inv())
                       .magnitude(), 0.0, atol=1e-9)
    assert np.allclose(a.markers['b'][:, 2], a0.markers['b'][:, 1])  # height moved from y to z
    assert np.isnan(a.markers['a'][7]).all() and np.isfinite(a.markers['a'][8]).all()


def test_only_the_two_y_up_datasets_are_turned():
    assert set(WORLD_TO_Z_UP) == {'camargo', 'wmcg'}


def test_empty_samples_at_the_end_are_counted():
    from mocapref.datasets._bridge import _empty_tail
    gyr, acc = np.ones((10, 3)), np.ones((10, 3))
    assert _empty_tail(gyr, acc) == 0
    gyr[-1], acc[-1] = np.nan, np.nan
    assert _empty_tail(gyr, acc) == 1
    gyr[5, 0] = np.nan  # a gap inside the trial is not the end
    assert _empty_tail(gyr, acc) == 1
    acc[-2] = np.nan  # gyroscope still read: not empty
    assert _empty_tail(gyr, acc) == 1
