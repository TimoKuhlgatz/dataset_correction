"""Known-answer tests of the soft-tissue measures: constant frame offsets alone give 0 deg; added wobble gives a
positive segment motion and joint floor."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocapref.soft_tissue import joint_error, segment_error

N = 2000


def _frames(seed_h, seed_c, wobble_deg=0.0, rng=None):
    h = R.random(N, random_state=seed_h)
    c = R.random(random_state=seed_c)
    s = h * c.inv()
    if wobble_deg:
        s = s * R.from_rotvec(rng.normal(size=(N, 3)) * np.radians(wobble_deg / np.sqrt(3)))
    return h.as_matrix(), s.as_matrix()


def test_constant_offset_gives_no_soft_tissue_motion():
    h, s = _frames(1, 3)
    r = segment_error(h, s)
    assert r['dev_max_deg'] < 1e-6
    ha, sa = _frames(1, 3)
    hb, sb = _frames(2, 4)
    assert joint_error(ha, sa, hb, sb).max() < 1e-6


def test_wobble_is_measured():
    rng = np.random.default_rng(0)
    h, s = _frames(1, 3, 2.0, rng)
    assert 1.0 < segment_error(h, s)['dev_median_deg'] < 4.0
    ha, sa = _frames(1, 3, 2.0, rng)
    hb, sb = _frames(2, 4, 2.0, rng)
    assert joint_error(ha, sa, hb, sb).mean() > 1.0
