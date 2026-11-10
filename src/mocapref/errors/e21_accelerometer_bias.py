"""Error 21: accelerometer bias. A constant offset in the accelerometer: at rest the sensor reads gravity plus that
offset, so the length is not 9.81 m/s² in every pose. Corrected where it can be measured; a scale error is not fitted.

How it is measured (the rest-pose rule). Rest is error 17's rest_mask (IMU quiet, the accelerometer's length varying
by less than 0.1 m/s², and optical reference still). Each rest stretch is cut into 0.5 s pieces; the mean
accelerometer of a piece is one rest reading. Readings closer than POSE_APART_DEG in direction are one pose (2.8°: the
most the segment can turn within one reading under the reference's rest limit of 5.6 °/s). With at least MIN_POSES
poses whose directions do not lie in one plane (smallest eigenvalue of their mean outer product at least MIN_SPREAD),
the bias b is the centre of the sphere of radius g through the readings (|a - b| = g, least squares). It is read if
the fit's rms residual is at most MAX_RESIDUAL and |b| at least MIN_BIAS; otherwise it is not corrected and the
verdict says why. Rest is found by how much the accelerometer's length varies, not by its size, so a constant bias
does not change which stretches count as rest.

Pooling: one bias per sensor and participant, over all its trials. This assumes that the sensor was not switched off
between trials (an accelerometer's offset may change at each switch-on); pooled per documented session instead, the
poses of a session usually lie in one plane and nothing can be read.

Units with gravity removed (error 22) are skipped: their rest readings lie near zero and a sphere of radius g through
the origin would read a "bias" of g. Skipped is a unit whose median rest-reading length is below error 22's
MAX_REST_LENGTH_G of gravity.

Frame. The bias is fitted in the sensor's own frame (the data before the mounting rotation, error 18). At its place in
the order (after the mounting rotation and the accelerometer sign, error 20) the accelerometer has been turned and
maybe negated; that map is recovered from the accelerometer before the mounting correction and as it now stands (the
same samples, turned), and the bias is turned by it before it is subtracted (see check_dataset below).
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording, Segment
from .e22_gravity_removed import GRAVITY_MPS2, MAX_REST_LENGTH_G, current_turn
from .e17_gyroscope_offset import WINDOW_S, rest_mask

ERROR = 'accelerometer bias'
G = 9.80665
POSE_APART_DEG = 2.8
MIN_POSES, MIN_SPREAD, MAX_RESIDUAL, MIN_BIAS = 8, 0.05, 0.05, 0.2  # gates of the rest-pose rule (m/s² for the last two)
READ = 'bias read'


def rest_readings(segment: Segment, window_s: float = WINDOW_S) -> np.ndarray:
    """Mean accelerometer (sensor frame as it stands, N x 3) of every whole `window_s` piece of rest."""
    still = rest_mask(segment)
    n = max(int(round(window_s * segment.imu.rate_hz)), 2)
    idx = np.flatnonzero(still)
    out = []
    if len(idx):
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            out += [segment.imu.acc[run[k * n:(k + 1) * n]].mean(0) for k in range(len(run) // n)]
    return np.asarray(out, float).reshape(-1, 3)


def distinct_poses(readings: np.ndarray, apart_deg: float = POSE_APART_DEG) -> np.ndarray:
    """The readings kept greedily so that no two directions lie within `apart_deg`."""
    d = readings / np.linalg.norm(readings, axis=1, keepdims=True)
    keep = []
    cos = np.cos(np.radians(apart_deg))
    for i in range(len(readings)):
        if all(d[i] @ d[j] < cos for j in keep):
            keep.append(i)
    return readings[keep]


def fit_bias(poses: np.ndarray, gravity: float = G, iterations: int = 50):
    """Centre b of the sphere of radius `gravity` through the poses (Gauss-Newton from 0); returns b and the
    residuals |p - b| - gravity."""
    b = np.zeros(3)
    for _ in range(iterations):
        v = poses - b
        n = np.linalg.norm(v, axis=1)
        step = np.linalg.lstsq(-v / n[:, None], gravity - n, rcond=None)[0]
        b = b + step
        if np.linalg.norm(step) < 1e-10:
            break
    return b, np.linalg.norm(poses - b, axis=1) - gravity


def judge_unit(readings: np.ndarray) -> dict:
    """The rest-pose rule on one sensor's pooled rest readings: verdict, number of poses, spread, bias (vector and size)
    and fit residual."""
    out = dict(readings=len(readings), poses=0, spread=np.nan, residual=np.nan, bias_norm=np.nan,
               bias_x=np.nan, bias_y=np.nan, bias_z=np.nan, rest_length=np.nan)
    if not len(readings):
        out['verdict'] = 'no rest'
        return out
    out['rest_length'] = float(np.median(np.linalg.norm(readings, axis=1)))
    if out['rest_length'] < MAX_REST_LENGTH_G * GRAVITY_MPS2:
        out['verdict'] = 'gravity removed (error 22)'
        return out
    p = distinct_poses(readings)
    out['poses'] = len(p)
    if len(p) < MIN_POSES:
        out['verdict'] = 'too few poses'
        return out
    d = p / np.linalg.norm(p, axis=1, keepdims=True)
    out['spread'] = float(np.linalg.eigvalsh(d.T @ d / len(d))[0])
    if out['spread'] < MIN_SPREAD:
        out['verdict'] = 'poses in one plane'
        return out
    b, res = fit_bias(p)
    out.update(residual=float(np.sqrt(np.mean(res ** 2))), bias_norm=float(np.linalg.norm(b)),
               bias_x=float(b[0]), bias_y=float(b[1]), bias_z=float(b[2]))
    out['verdict'] = ('fit residual too large' if out['residual'] > MAX_RESIDUAL else
                      'bias below 0.2 m/s²' if out['bias_norm'] < MIN_BIAS else READ)
    return out


def remove_accelerometer_bias(recording: Recording, biases: dict, acc_in_sensor_frame: dict) -> Recording:
    """Copy with the bias subtracted in every segment of `biases` ({segment: bias (3,) in the sensor's own frame}).
    `acc_in_sensor_frame` {segment: accelerometer before the mounting stage} gives the turn applied since; a
    segment whose accelerometer is not a turned copy of it is left as it is and logged."""
    new = recording.copy()
    removed, skipped = {}, {}
    for name, b in biases.items():
        if name not in new.segments:
            continue
        seg = new.segments[name]
        before = acc_in_sensor_frame.get(name)
        turn = None if before is None or len(before) != len(seg.imu) else current_turn(np.asarray(before), seg.imu.acc)
        if turn is None:
            skipped[name] = 'accelerometer is not a turned copy of the one before the mounting stage'
            continue
        seg.imu.acc = seg.imu.acc - np.asarray(b, float) @ turn.T
        removed[name] = round(float(np.linalg.norm(b)), 4)
    new.log_correction(ERROR, f'Subtracted the accelerometer bias in {len(removed)} segments.',
                       bias_mps2=removed, skipped=skipped)
    return new


# Dataset level. The bias is fitted in the sensor's own frame, from the rest readings of all trials of a sensor and
# participant; but at its place in the order the accelerometer has been rotated by the mounting correction (error 18)
# and maybe negated (error 20), by a map that differs per trial. check_dataset therefore reads every trial twice: from
# the loader below the mounting correction (found by walking back the chain of corrected loaders to the one labelled
# 'mounting'; on deposited data the loader itself), where the rest readings are taken, and from the loader it was
# given, as it stands. The map between the two (current_turn) turns the sensor-frame bias into the frame the
# correction will see, and that turned bias is stored per trial and segment (now_bias_x/y/z), so correct() needs
# nothing but the rows.


from ._report import CorrectedLoader, ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, frame, number, per_trial, read_rows, run_trials, truthy  # noqa: E402

MOUNTING_LABEL = 'mounting'  # label of error 18's corrected loader
COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'verdict', 'poses', 'spread',
           'residual', 'bias_x', 'bias_y', 'bias_z', 'now_bias_x', 'now_bias_y', 'now_bias_z', 'rest_length',
           'error']


def sensor_frame_loader(loader):
    """The loader below the mounting correction in a chain of corrected loaders (else the loader itself)."""
    current = loader
    while isinstance(current, CorrectedLoader):
        if current.label == MOUNTING_LABEL:
            return current.base
        current = current.base
    return loader


def _readings_of_trial(loader, trial):
    below = sensor_frame_loader(loader)
    sensor = below.load(trial)
    now = loader.load(trial) if below is not loader else sensor
    rows = []
    for name, seg in sensor.segments.items():
        turn = np.eye(3)
        if now is not sensor:
            other = now.segments.get(name)
            turn = (current_turn(seg.imu.acc, other.imu.acc)
                    if other is not None and len(other.imu) == len(seg.imu) else None)
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), segment=name,
                         readings=rest_readings(seg), turn=turn, error=''))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """One bias per sensor and participant from the rest poses of all its trials (sensor frame)."""
    records = run_trials(loader, _readings_of_trial, processes)
    ok = [r for r in records if not r.get('error')]
    units = {}
    for r in ok:
        units.setdefault((r['participant'], r['segment']), []).append(r['readings'])
    judged = {k: judge_unit(np.vstack(v)) for k, v in units.items()}
    for r in ok:
        j = judged[(r['participant'], r['segment'])]
        b = np.array([j['bias_x'], j['bias_y'], j['bias_z']])
        now = b @ r['turn'].T if r['turn'] is not None else np.full(3, np.nan)
        r.update(detected=True if j['verdict'] == READ else (False if j['verdict'] == 'bias below 0.2 m/s²'
                                                             else None),
                 estimate=j['bias_norm'], unit='m/s^2', verdict=j['verdict'], poses=j['poses'], spread=j['spread'],
                 residual=j['residual'], bias_x=b[0], bias_y=b[1], bias_z=b[2], now_bias_x=now[0],
                 now_bias_y=now[1], now_bias_z=now[2], rest_length=j['rest_length'])
    for r in records:
        r.pop('readings', None)
        r.pop('turn', None)
    return frame(records, COLUMNS)


def prevalence(rows) -> dict:
    """Trials of a sensor with a bias read / sensor-participants with a bias read. Undecided: no rest, too few
    poses, poses in one plane, a fit residual too large, or gravity removed."""
    ok = read_rows(rows)
    hit = ok.detected.map(truthy)
    units = ok[hit][['participant', 'segment']].astype(str).drop_duplicates()
    undecided = ok.detected.map(lambda v: v is None or str(v) in ('', 'None', 'nan'))
    return dict(trials_checked=int(ok.trial.nunique()), trials_detected=int(ok.trial[hit].nunique()),
                occurrences=len(units), occurrence_unit='sensor-participants',
                undecided=int(ok.trial[undecided].nunique()))


def _subtract(rec, rows):
    new, removed, skipped = rec.copy(), {}, {}
    for r in rows:
        if r['segment'] not in new.segments or not truthy(r.get('detected')):
            continue
        b = np.array([number(r.get(f'now_bias_{a}')) for a in 'xyz'])
        if not np.isfinite(b).all():
            skipped[r['segment']] = 'accelerometer is not a turned copy of the one before the mounting correction'
            continue
        new.segments[r['segment']].imu.acc = new.segments[r['segment']].imu.acc - b
        removed[r['segment']] = round(float(np.linalg.norm(b)), 4)
    new.log_correction(ERROR, f'Subtracted the accelerometer bias in {len(removed)} segments.',
                       bias_mps2=removed, skipped=skipped)
    return new


def correct(loader, rows):
    """The bias subtracted where the rest-pose rule reads it, turned like the accelerometer; a scale error is not
    corrected."""
    return corrected(loader, _subtract, per_trial(rows), 'accelerometer bias')


SPEC = ErrorSpec(21, 'accelerometer_bias', 'fix', 'accelerometer bias')
PARTS = {21: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_accelerometer_bias = make_handler(PARTS[21])
