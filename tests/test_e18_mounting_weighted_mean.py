"""Known-answer tests for the convergence window of continuous motion and for
the uncertainty-weighted mounting rotation (error 18): the uncertainty holds the true
error, a precise own fit keeps its weight, a vague one leans on the participant's other trials, the weighting works
per direction, and real scatter between trials is recognised."""
import numpy as np
import pytest
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from mocapref import errors as E
from mocapref.errors.e18_mounting import (combine_with_pool, fit_information, mount_trial, pooled_rotations,
                                          trial_scatter, uncertainty_deg)
from mocapref.motion_onset import convergence_window, motion_onset_s
from mocapref.synthetic import make_recording, smooth_rate

MOUNTING = Rotation.from_euler('xyz', [20, -10, 90], degrees=True)


def _hold(rec, start_s, end_s):
    """Reference and gyroscope at rest from start_s to end_s (the pose of start_s held)."""
    for seg in rec.segments.values():
        ref = seg.reference
        a, b = int(start_s * ref.rate_hz), min(int(end_s * ref.rate_hz), len(ref) - 1)
        q = ref.orientation.as_quat()
        q[a:b] = q[a]
        ref.orientation = Rotation.from_quat(q)
        seg.imu.gyr[(seg.imu.times >= start_s) & (seg.imu.times < end_s)] = 0.0


def _twitch_then_rest(rec):
    """Still until 2 s, moving for 0.5 s, still again until 12 s, moving afterwards (DIODEM gait/exp11's pattern)."""
    _hold(rec, 0.0, 2.0)
    _hold(rec, 2.5, 12.0)


def test_a_twitch_followed_by_rest_does_not_start_the_motion():
    rec = make_recording(duration_s=40.0, mounting=MOUNTING)
    _twitch_then_rest(rec)
    assert motion_onset_s(rec) == pytest.approx(12.0, abs=0.5)


def test_twitch_only_trial_takes_the_window_with_most_motion():
    """Moving windows but no 10 s with a majority of them: the 10 s with the most motion, the trial is not static."""
    rec = make_recording(duration_s=40.0)
    _hold(rec, 0.0, 2.0)
    _hold(rec, 2.5, 40.0)
    assert motion_onset_s(rec) == pytest.approx(2.0, abs=0.5)


def test_window_is_continuous_where_the_majority_moves():
    rec = make_recording(duration_s=40.0)
    _hold(rec, 0.0, 6.0)
    w = convergence_window(rec)
    assert w['continuous'] and w['onset_s'] == pytest.approx(6.0, abs=0.5) and w['share'] > 0.9
    twitch = make_recording(duration_s=40.0)
    _hold(twitch, 0.0, 2.0)
    _hold(twitch, 2.5, 40.0)
    assert not convergence_window(twitch)['continuous']


def test_window_of_continuous_motion_still_recovers_the_mounting():
    rec = make_recording(duration_s=40.0, mounting=MOUNTING)
    _twitch_then_rest(rec)
    for p in E.check_for_mounting_in_convergence(rec).per_segment.values():
        assert p['uncertainty_deg'] < 2
        assert np.degrees((Rotation.from_matrix(p['fit'].sensor_to_segment) * MOUNTING).magnitude()) < 0.5


def _window(seed, peak=200.0, minor=1.0, noise_dps=10.0, grow=0.15, seconds=10.0, rate=100.0, cutoff_hz=1.0):
    """Gyroscope and reference rates of one window with skin-like reference noise (low-passed at cutoff_hz, partly
    growing with the motion); returns g, w, t, the true segment-to-sensor rotation."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / rate)
    w = smooth_rate(t, rng, peak_dps=peak)
    w[:, 1:] *= minor
    w = Rotation.random(random_state=rng.integers(1 << 31)).apply(w)
    true = Rotation.random(random_state=rng.integers(1 << 31))
    g = true.apply(w) + np.radians(1.0) * rng.standard_normal(w.shape)
    b, a = butter(2, cutoff_hz / (rate / 2))

    def coloured():
        x = filtfilt(b, a, rng.standard_normal((len(t), 3)), axis=0)
        return x / np.sqrt((x ** 2).sum(1).mean())
    w_ref = w + np.radians(noise_dps) * coloured() + grow * np.linalg.norm(w, axis=1, keepdims=True) * coloured()
    use = np.linalg.norm(w_ref, axis=1) > np.radians(17)
    return g[use], w_ref[use], t[use], true


def _fit(g, w):
    from mocapref.rotations import best_rotation
    return best_rotation(g, w)[0]


@pytest.mark.parametrize('minor', [1.0, 0.1])
def test_uncertainty_holds_the_true_error(minor):
    """Synthetic windows with known rotation and slowly drifting noise: the 95 % uncertainty holds the true error in
    at least 90 % of 120 windows."""
    inside = []
    for seed in range(120):
        g, w, t, true = _window(seed, minor=minor)
        m = _fit(g, w)
        err = np.degrees((Rotation.from_matrix(m) * true).magnitude())
        inside.append(err <= uncertainty_deg(fit_information(m, g, w, t, 100.0)))
    assert np.mean(inside) >= 0.9


def test_short_stretch_is_not_overconfident():
    """One second of motion with noise whose memory is the measured one (low-passed at 4 Hz, correlation fading
    within about 0.1-0.2 s): the small-sample widening keeps the true error inside."""
    inside = []
    for seed in range(100):
        g, w, t, true = _window(seed, cutoff_hz=4.0)
        k = slice(200, 300)
        m = _fit(g[k], w[k])
        err = np.degrees((Rotation.from_matrix(m) * true).magnitude())
        inside.append(err <= uncertainty_deg(fit_information(m, g[k], w[k], t[k], 100.0)))
    assert np.mean(inside) >= 0.9


def test_precise_own_fit_keeps_its_weight_and_vague_one_leans_on_the_pool():
    pool = MOUNTING.inv()
    own = (Rotation.from_euler('z', 10, degrees=True) * pool).as_matrix()
    precise, vague, pool_info = np.eye(3) / np.radians(0.5) ** 2, np.eye(3) / np.radians(30) ** 2, np.eye(3) / np.radians(3) ** 2
    m, _, share = combine_with_pool(own, precise, pool, pool_info)
    assert np.degrees((Rotation.from_matrix(m) * Rotation.from_matrix(own).inv()).magnitude()) < 0.5 and share > 0.95
    m, _, share = combine_with_pool(own, vague, pool, pool_info)
    assert np.degrees((Rotation.from_matrix(m) * pool.inv()).magnitude()) < 0.2 and share < 0.05


def test_weighting_is_per_direction():
    """An own fit that knows x and y well but not z (motion about one axis) keeps its x and y and takes z from the
    pool."""
    pool = Rotation.identity()
    own = Rotation.from_rotvec(np.radians([5, -4, 20])).as_matrix()
    own_info = np.diag([1 / np.radians(0.3) ** 2, 1 / np.radians(0.3) ** 2, 1 / np.radians(90) ** 2])
    m, _, _ = combine_with_pool(own, own_info, pool, np.eye(3) / np.radians(2) ** 2)
    x = np.degrees(Rotation.from_matrix(m).as_rotvec())
    assert x[0] == pytest.approx(5, abs=0.3) and x[1] == pytest.approx(-4, abs=0.3) and abs(x[2]) < 0.5


def test_no_pool_keeps_the_own_fit_and_no_fit_takes_the_pool():
    own = MOUNTING.inv().as_matrix()
    assert np.allclose(combine_with_pool(own, np.eye(3), None, None)[0], own)
    assert np.allclose(combine_with_pool(None, np.zeros((3, 3)), MOUNTING.inv(), np.eye(3))[0], own)
    assert combine_with_pool(None, np.zeros((3, 3)), None, None)[0] is None


def test_scatter_between_trials_is_recognised():
    """Fits 1 deg precise but truly 8 deg apart: the scatter comes out near 8 deg; fits that agree within their
    noise give zero scatter."""
    rng = np.random.default_rng(3)
    info = np.repeat((np.eye(3) / np.radians(1) ** 2)[None], 30, 0)
    scattered = Rotation.from_rotvec(np.radians(8) * rng.standard_normal((30, 3))) * MOUNTING
    assert np.degrees(np.sqrt(trial_scatter(scattered, info))) == pytest.approx(8, rel=0.3)
    tight = Rotation.from_rotvec(np.radians(1) * rng.standard_normal((30, 3))) * MOUNTING
    assert np.degrees(np.sqrt(trial_scatter(tight, info))) < 1.0


def test_pool_leaves_the_trial_itself_out():
    rows = [dict(participant='P1', segment='upper', trial=f't{i}', information=np.eye(3) / np.radians(1) ** 2,
                 quaternion=(Rotation.from_euler('z', z, degrees=True) * MOUNTING).as_quat())
            for i, z in enumerate([0, 0, 0, 40])]
    pools = pooled_rotations(rows)[('P1', 'upper')]
    mean, _ = pools['per_trial']['t3']
    assert np.degrees((mean * MOUNTING.inv()).magnitude()) < 1.0  # t3's own 40 deg is not in its pool
    assert pooled_rotations(rows[:1])[('P1', 'upper')]['per_trial'] == {}


def test_mounting_uses_the_pool_for_a_window_without_motion():
    """A trial whose window holds a single twitch gets a vague own fit; the weighted mean lands on the pool."""
    rec = make_recording(duration_s=30.0, mounting=MOUNTING)
    for seg in rec.segments.values():
        ref = seg.reference
        k0 = int(5.0 * ref.rate_hz)
        q = ref.orientation.as_quat()
        q[:k0] = q[k0]
        q[k0 + 10:] = q[k0 + 10]
        ref.orientation = Rotation.from_quat(q)
    pool = {('P1', 'upper'): dict(tau_deg=1.0, n=5, per_trial={rec.trial: (MOUNTING.inv(), np.eye(3) / np.radians(2) ** 2)})}
    fixed = mount_trial(rec, pool, 'P1')
    log = fixed.corrections[-1].parameters
    assert log['weights']['upper']['own_share'] < 0.2 and log['weights']['upper']['pool_trials'] == 4
    clean = make_recording(duration_s=30.0).segments['upper'].imu
    assert np.allclose(fixed.segments['upper'].imu.gyr, clean.gyr, atol=np.radians(0.5) * 10)
    assert 'lower' not in log['not_corrected']  # own fit, no pool: kept


def test_trial_without_own_fit_takes_the_pool_of_every_fit():
    rows = [dict(participant='P1', segment='upper', trial=f't{i}', information=np.eye(3) / np.radians(1) ** 2,
                 quaternion=MOUNTING.inv().as_quat()) for i in range(3)]
    mean, info = pooled_rotations(rows)[('P1', 'upper')]['all']
    assert np.degrees((mean * MOUNTING).magnitude()) < 1e-6 and np.linalg.eigvalsh(info)[0] > 0


def test_a_wrong_own_fit_does_not_inflate_its_own_scatter():
    """Four fits agree within 1 deg, the fifth is 130 deg off: the fifth trial's pool
    carries the scatter of the other four only, so its own fit keeps almost no weight."""
    rng = np.random.default_rng(5)
    info = np.eye(3) / np.radians(1) ** 2
    rots = [Rotation.from_rotvec(np.radians(0.5) * rng.standard_normal(3)) * MOUNTING for _ in range(4)]
    rots.append(Rotation.from_euler('x', 130, degrees=True) * MOUNTING)
    rows = [dict(participant='P1', segment='upper', trial=f't{i}', information=info / (100 if i == 4 else 1),
                 quaternion=r.as_quat()) for i, r in enumerate(rots)]
    mean, pool_info = pooled_rotations(rows)[('P1', 'upper')]['per_trial']['t4']
    assert np.degrees(np.sqrt(1 / np.linalg.eigvalsh(pool_info)[0])) < 2.0  # pool not widened by the 130 deg fit
    applied, _, share = combine_with_pool(rots[4].as_matrix(), info / 100, mean, pool_info)
    assert share < 0.05 and np.degrees((Rotation.from_matrix(applied) * MOUNTING.inv()).magnitude()) < 8.0
