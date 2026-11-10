"""Error 22: gravity removed from the accelerometer by the depositors (e.g. vendor firmware that writes the
gravity-compensated linear acceleration). Reported as the depositors' processing choice, and repaired for the
estimators, which need gravity to know which way is down.

How it is caught: at rest an accelerometer reads gravity, 9.81 m/s² in any pose. A sensor whose median accelerometer
length over its rest samples (error 17's rest_mask: IMU quiet and optical reference still) is below MAX_REST_LENGTH_G
of gravity has its gravity removed. No rotation and no mounting enters, so a poor mounting fit cannot shorten it. In
the datasets the affected sensor-trials read 0.08-0.36 m/s², every other one at least 8.9 m/s². A sensor-trial with
less than MIN_REST_S of rest is not judged. One occurrence per sensor and trial.

How it is restored. The firmware subtracted gravity as it saw it: its own orientation estimate turned the room's "up"
into the sensor frame, and that vector times 1 g was taken off every sample. The deposit keeps that orientation
estimate, the vendor quaternion (one per sample, rows aligned with the accelerometer). Adding the same vector back
gives the specific force the sensor measured:

    a_restored(t) = a_deposited(t) + g * R_vendor(t)^T e_z,

with R_vendor the sensor-to-room rotation of the quaternion and e_z the room's up. Checked on the deposit: the
quaternion's frame-to-frame rotation matches the gyroscope (correlation 0.98-1.00), its room is +z up, and the
restored length at rest is 9.78-9.84 m/s². g = 9.8 m/s², the 1 g of the deposit's own scale constant. Only the
quaternion's tilt enters (a heading error turns e_z about itself); its tilt errors enter the restored accelerometer.
Assumed, not tested: the firmware subtracted exactly this vector (not, e.g., a low-passed one).

Place in the order: after the mounting rotation (error 18), so the vector is turned by the rotation applied to the
accelerometer since it was read. That rotation is recovered from the accelerometer as read and as it now stands (the
same samples, rotated), so no stage needs to hand it over.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..recording import Recording
from ..rotations import best_rotation
from ._result import CheckResult, entry
from .e17_gyroscope_offset import rest_mask

ERROR = 'gravity removed'
VENDOR_GRAVITY_MPS2 = 9.8  # the deposit's 1 g (scale constant 9.8/2048 m/s² per count)
QUATERNION_KEY = 'vendor_quaternion_wxyz'
AS_READ_KEY = 'acc_as_read'
MAX_REST_LENGTH_G = 0.5  # rest length below half gravity: gravity removed (gap 0.36 to 8.9 m/s² in the data)
MIN_REST_S = 1.0  # rest a sensor-trial needs to be judged (as for the gyroscope offset)
GRAVITY_MPS2 = 9.81
MAX_MAP_RESIDUAL = 1e-6  # m/s²: the accelerometer now is the one as read, turned; anything larger is not a pure turn


def check_for_gravity_removed(recording: Recording, max_rest_length_g: float = MAX_REST_LENGTH_G,
                              min_rest_s: float = MIN_REST_S) -> CheckResult:
    """Per segment: the median accelerometer length over its rest samples; detected below max_rest_length_g of
    gravity, None with less than min_rest_s of rest."""
    per = {}
    for name, seg in recording.segments.items():
        still = rest_mask(seg)
        rest_s = float(still.sum() / seg.imu.rate_hz)
        if rest_s < min_rest_s:
            per[name] = entry(None, float('nan'), 'm/s^2', rest_s=rest_s)
            continue
        length = float(np.median(np.linalg.norm(seg.imu.acc[still], axis=1)))
        per[name] = entry(length < max_rest_length_g * GRAVITY_MPS2, length, 'm/s^2', rest_s=rest_s)
    found = [p['detected'] for p in per.values() if p['detected'] is not None]
    hits = [n for n, p in per.items() if p['detected']]
    return CheckResult(ERROR, (bool(hits) if found else None),
                       f'{len(hits)} of {len(found)} judged sensors read less than {max_rest_length_g:g} g at rest.',
                       per, dict(max_rest_length_g=max_rest_length_g, min_rest_s=min_rest_s))


def vendor_up_in_sensor(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """The room's up (unit vectors, N x 3) in the sensor frame, from the vendor's sensor-to-room quaternions."""
    q = np.asarray(quaternion_wxyz, float)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).inv().apply([0.0, 0.0, 1.0])


def current_turn(as_read: np.ndarray, now: np.ndarray) -> np.ndarray | None:
    """The 3x3 map (rotation, or a mirror after a negation) that turned the accelerometer as read into the one now,
    from the samples finite in both; None if it is not a pure turn."""
    ok = np.isfinite(as_read).all(1) & np.isfinite(now).all(1)
    if ok.sum() < 10:
        return None
    best = None
    for proper in (True, False):
        m, _ = best_rotation(as_read[ok], now[ok], proper=proper)
        res = float(np.sqrt(np.mean(np.sum((as_read[ok] @ m.T - now[ok]) ** 2, axis=1))))
        if best is None or res < best[1]:
            best = (m, res)
    return best[0] if best[1] <= MAX_MAP_RESIDUAL * max(1.0, float(np.abs(now[ok]).max())) else None


def restore_removed_gravity(recording: Recording, gravity_mps2: float = VENDOR_GRAVITY_MPS2,
                            segments=None) -> Recording:
    """Copy with gravity added back to every segment whose accelerometer is a turned copy of one the reader carried
    with its vendor quaternion (recording.metadata['segments'][name]); other segments are left as they are. The
    stream is matched by its data, not its name, since a later stage may have moved a sensor to another segment
    (error 9). Samples whose quaternion is missing (zero) take the next one's and become invalid. `segments`: restore
    only these (default: every segment)."""
    new = recording.copy()
    carried = {k: m for k, m in new.metadata.get('segments', {}).items() if QUATERNION_KEY in m}
    restored, skipped, moved, no_quaternion = [], {}, {}, {}
    for name, seg in new.segments.items():
        if not carried:
            break
        if segments is not None and name not in segments:
            continue
        match = None
        for source in sorted(carried, key=lambda k: k != name):  # its own name first
            m = carried[source]
            if len(m[AS_READ_KEY]) == len(seg.imu) == len(m[QUATERNION_KEY]):
                turn = current_turn(np.asarray(m[AS_READ_KEY], float), seg.imu.acc)
                if turn is not None:
                    match = (source, turn)
                    break
        if match is None:
            if name in carried:
                skipped[name] = 'no stream read with a quaternion is a turned copy of this accelerometer'
            continue
        source, turn = match
        q = np.array(carried[source][QUATERNION_KEY], float)
        bad = ~(np.isfinite(q).all(1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0.5))
        if bad.all():
            skipped[name] = 'no sample has a quaternion'
            continue
        good = np.flatnonzero(~bad)  # a missing quaternion takes its nearest neighbour's, the sample stays finite
        q[bad] = q[good[np.clip(np.searchsorted(good, np.flatnonzero(bad)), 0, len(good) - 1)]]  # (zoe stops on NaN)
        seg.imu.acc = seg.imu.acc + gravity_mps2 * vendor_up_in_sensor(q) @ turn.T
        seg.imu.valid = seg.imu.valid & ~bad
        restored.append(name)
        if source != name:
            moved[name] = source
        if bad.any():
            no_quaternion[name] = int(bad.sum())
    new.log_correction(ERROR, f'Added gravity back from the vendor quaternion to {len(restored)} segments.',
                       restored=restored, skipped=skipped, read_as=moved, samples_without_quaternion=no_quaternion,
                       gravity_mps2=gravity_mps2)
    return new


# Dataset level: one row per trial and segment (one occurrence per sensor and trial).

from ._report import ErrorSpec, make_handler  # noqa: E402
from ._rowtools import corrected, frame, per_trial, prevalence_of, run_trials, truthy  # noqa: E402

COLUMNS = ['trial', 'participant', 'segment', 'detected', 'estimate', 'unit', 'rest_s', 'error']


def _check_trial(loader, trial):
    rec = loader.load(trial)
    res = check_for_gravity_removed(rec)
    return [dict(trial=trial, participant=str(loader.participant_of(trial)), segment=s, detected=p['detected'],
                 estimate=p['estimate'], unit='m/s^2', rest_s=p['rest_s'], error='')
            for s, p in res.per_segment.items()]


def check_dataset(loader, dataset: str, processes: int = 1, **options):
    """Median accelerometer length at rest of every sensor and trial."""
    return frame(run_trials(loader, _check_trial, processes), COLUMNS)


def prevalence(rows) -> dict:
    """Trials with a sensor whose rest length is below half gravity / such sensor-trials; sensor-trials with less
    than 1 s of rest are undecided."""
    return prevalence_of(rows, 'sensor-trials')


def _restore(rec, rows):
    hit = {r['segment'] for r in rows if truthy(r.get('detected'))}
    return restore_removed_gravity(rec, segments=hit) if hit else rec


def correct(loader, rows):
    """Gravity added back from the vendor quaternion in the sensor-trials found without it, where the reader carries
    that quaternion; the error is still reported."""
    return corrected(loader, _restore, per_trial(rows), 'gravity removed')


SPEC = ErrorSpec(22, 'gravity_removed', 'fix', 'gravity removed from the accelerometer')
PARTS = {22: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_gravity_removed = make_handler(PARTS[22])
