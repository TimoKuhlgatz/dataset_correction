"""Error 3: jumps or flips in the optical reference orientation.

What it is: single reference frames (or short stretches) where the optical orientation jumps, often by close to
180°, because markers were mislabelled or occluded. The body cannot rotate that fast; the IMU on it did not.

How we correct it: we cannot know the true orientation in such frames, so they are marked invalid.

How we catch it (first pass, from the reference alone): a step between two consecutive valid reference frames is
impossible if the rotation between them, divided by the time between them, exceeds
MAX_SEGMENT_RATE_DPS, the fastest rotation of a human body part we have read a measurement of. The rate is the
geodesic angle between the two orientations, so the quaternion sign does not matter, and frames already invalid are
skipped (a step across a gap is measured over the gap's whole time).
Impossible steps are paired out and back: an out step is paired with the first later step that undoes it, that is,
the stretch is left with the same offset it was entered with. The offset entering is the rotation from the frame
before the out step to the first frame of the stretch, the offset leaving the rotation from the frame after the back
step to the last frame of the stretch; they must agree within one frame of the limit (MAX_SEGMENT_RATE_DPS / frame
rate: 38° at 200 Hz, 76° at 100 Hz), since the true orientation moves by at most that much over one frame. The
frames between the two steps are marked invalid, a single frame for a one-frame glitch, the whole flipped stretch for
a flip; impossible steps inside the stretch belong to it.
Check against the other segments: a pair is accepted only if the stretch is also offset against every other segment
of the trial. The rotation of the segment relative to the other segment (the joint rotation) is averaged inside the
stretch and over as many valid frames directly before and after it; the two means must differ by at least
OTHER_SEGMENT_SHARE (half) of the angle of the out step. A real flip of the segment by F shifts the joint rotation by
about F; two unrelated glitches whose offsets happen to agree leave it in place. On 75 real stretches longer than 1 s
(whole-trial means) the 3 real flips shifted by 156-177 degrees against steps of 163-179, the 72 others by 0.4-12
against steps of 38-171. It reads only the optical references, not the IMU. A candidate that fails is skipped and the
next later step that undoes the out step is tried. A segment without another segment of the same frame count and
rate is paired on the undo rule alone (reported as unconfirmed). Pairing consecutive steps instead failed on real
data: a glitch of two different wrong frames gives three steps and shifted every later pair, so the stretches between
unrelated glitches were masked. A step without a partner is counted and reported as unpaired, not marked: the
reference alone cannot tell which side of it is wrong. Below MIN_CHECKABLE_RATE_HZ even a 180° step stays below the
limit, so such references are reported as not checkable, not as clean. Jumps smaller than the limit allows per frame
(64° at 120 Hz, 76° at 100 Hz) are not found by this pass.

A second rule, kept as a tool but not part of the order of correction since it needs synchronised streams
(check_for_reference_jumps, remove_reference_jumps): where the reference's angular speed differs from the gyroscope's
by more than a threshold, the frame is implausible and marked invalid with a margin (rule taken from the mfimt
package's `optical_valid`).
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording, Segment
from ..rotations import dilate, reference_angular_rate
from ._result import CheckResult, entry

ERROR = 'reference jumps'
OTHER_SEGMENT_SHARE = 0.5
"""A stretch counts as offset against another segment if its joint rotation moves by at least this share of the out
step's angle."""

MAX_SEGMENT_RATE_DPS = 6756.0 + 893.0
"""°/s, the fastest rotation of a human body part we have read a measurement of, mean + 1 SD:
shoulder internal rotation of 25 high-school and college pitchers throwing a 4-oz ball from flat ground with a crow
hop, 6,756 ± 893 °/s, the fastest of the ten throwing conditions (Fleisig et al. 2017, Sports Health 9(3):210-215,
Table 1; skin markers at 240 Hz). It is a joint rate (upper arm against trunk); the upper arm's rate in space adds
the trunk's (about 1,100 °/s in the same table), so a real segment can slightly exceed it in a throw. None of the 18
datasets holds throwing."""
MIN_CHECKABLE_RATE_HZ = MAX_SEGMENT_RATE_DPS / 180.0
"""Hz, below this frame rate even a 180° step between two frames stays below MAX_SEGMENT_RATE_DPS."""


def implausible_reference_frames(segment: Segment, threshold_dps: float = 300.0, margin_s: float = 0.0) -> np.ndarray:
    """Boolean mask over the reference frames: True where the reference rotates implausibly fast.

    A step between frame k and k+1 counts as implausible if its angular speed differs from the gyroscope's
    angular speed at the same time by more than `threshold_dps`. Both frames of such a step are marked, then
    widened by `margin_s`. Frames already invalid in the deposit are marked too.
    """
    ref, imu = segment.reference, segment.imu
    step_times, rates = reference_angular_rate(ref.orientation, ref.rate_hz)
    step_speed = np.linalg.norm(rates, axis=1)
    gyro_speed = np.interp(step_times + ref.start_s, imu.times, np.linalg.norm(imu.gyr, axis=1))
    bad_step = ~np.isfinite(step_speed) | (np.abs(step_speed - gyro_speed) > np.radians(threshold_dps))
    bad_frame = np.zeros(len(ref), bool)
    bad_frame[:-1] |= bad_step
    bad_frame[1:] |= bad_step
    bad_frame |= ~ref.valid
    return dilate(bad_frame, int(round(margin_s * ref.rate_hz)))


def joint_shift_deg(reference, other, first: int, last: int) -> float:
    """Angle (degrees) between the mean joint rotation (reference relative to other) inside frames first..last and
    the mean over as many frames directly before and after; NaN where either side has no valid frame."""
    n, length = len(reference), last - first + 1
    ok = reference.valid & other.valid
    inside = np.zeros(n, bool)
    inside[first:last + 1] = True
    near = np.zeros(n, bool)
    near[max(0, first - length):first] = True
    near[last + 1:last + 1 + length] = True
    if not (ok & inside).any() or not (ok & near).any():
        return float('nan')
    joint = reference.orientation.inv() * other.orientation
    a, b = joint[ok & inside].mean(), joint[ok & near].mean()
    return float(np.degrees((a.inv() * b).magnitude()))


def impossible_steps(reference, limit_dps: float = MAX_SEGMENT_RATE_DPS, others=()) -> dict:
    """First pass, reference alone: the steps between consecutive valid frames faster than `limit_dps`, paired out
    and back where the back step undoes the out step. Returns mask (frames between the two steps of each pair),
    steps (list of dict(before, after, angle_deg, rate_dps), frame indices), pairs (list of (first, last) masked
    frame) and unpaired (the out steps without a partner). `others`: the other segments' references of the same
    recording; a pair is kept only if the stretch is offset against each of them (see OTHER_SEGMENT_SHARE).
    Also returns rejected (candidate pairs that undid but were not offset against the others) and unconfirmed
    (pairs kept without another segment to check against)."""
    n = len(reference)
    mask = np.zeros(n, bool)
    finite = np.isfinite(reference.orientation.as_quat()).all(axis=1) if n else np.zeros(0, bool)
    idx = np.flatnonzero(reference.valid & finite)
    if len(idx) < 2:
        return dict(mask=mask, steps=[], pairs=[], unpaired=[], rejected=0, unconfirmed=0)
    rot = reference.orientation[idx]
    angle = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())
    rate = angle / (np.diff(idx) / reference.rate_hz)
    steps = [dict(before=int(idx[k]), after=int(idx[k + 1]), angle_deg=float(angle[k]), rate_dps=float(rate[k]))
             for k in np.flatnonzero(rate > limit_dps)]
    tolerance = np.radians(limit_dps / reference.rate_hz)  # one frame at the limit
    orientation = reference.orientation
    others = [o for o in others if len(o) == n and o.rate_hz == reference.rate_hz]
    pairs, unpaired, k, rejected = [], [], 0, 0
    while k < len(steps):
        out = steps[k]
        offset_in = orientation[out['before']].inv() * orientation[out['after']]
        for j in range(k + 1, len(steps)):
            back = steps[j]
            offset_out = orientation[back['after']].inv() * orientation[back['before']]
            if (offset_in.inv() * offset_out).magnitude() <= tolerance:
                shifts = [joint_shift_deg(reference, o, out['after'], back['before']) for o in others]
                shifts = [x for x in shifts if np.isfinite(x)]
                if shifts and min(shifts) < OTHER_SEGMENT_SHARE * out['angle_deg']:
                    rejected += 1
                    continue
                mask[out['after']:back['before'] + 1] = True
                pairs.append((out['after'], back['before']))
                k = j + 1
                break
        else:
            unpaired.append(out)
            k += 1
    return dict(mask=mask, steps=steps, pairs=pairs, unpaired=unpaired, rejected=rejected,
                unconfirmed=len(pairs) if not others else 0)


def _others(recording: Recording, name: str) -> list:
    """The references of the recording's other segments, as deposited in `recording`."""
    return [seg.reference for other, seg in recording.segments.items() if other != name]


def check_for_impossible_reference_steps(recording: Recording,
                                         limit_dps: float = MAX_SEGMENT_RATE_DPS) -> CheckResult:
    """First pass of error 3, from the reference alone: steps faster than any measured human rotation."""
    per_segment, found, decided = {}, False, False
    for name, seg in recording.segments.items():
        ref = seg.reference
        if ref.rate_hz < limit_dps / 180.0:
            per_segment[name] = entry(None, float('nan'), 'frames', reason=f'frame rate {ref.rate_hz:g} Hz: even '
                                      f'a 180° step stays below {limit_dps:g} °/s')
            continue
        r = impossible_steps(ref, limit_dps, _others(recording, name))
        hit = bool(r['steps'])
        per_segment[name] = entry(hit, int(r['mask'].sum()), 'frames', impossible_steps=len(r['steps']),
                                  pairs=len(r['pairs']), unpaired=len(r['unpaired']), rejected_pairs=r['rejected'],
                                  unconfirmed_pairs=r['unconfirmed'],
                                  fastest_step_dps=max((x['rate_dps'] for x in r['steps']), default=0.0))
        found |= hit
        decided = True
    n_steps = sum(p.get('impossible_steps', 0) for p in per_segment.values())
    summary = (f'{n_steps} reference steps are faster than {limit_dps:g} °/s.' if found else
               f'No reference step is faster than {limit_dps:g} °/s.' if decided else
               'Not checkable: every reference frame rate is too low for the limit.')
    return CheckResult(ERROR, found if decided else None, summary, per_segment, dict(limit_dps=limit_dps))


def remove_impossible_reference_steps(recording: Recording, limit_dps: float = MAX_SEGMENT_RATE_DPS) -> Recording:
    """Copy with the frames between the two impossible steps of each out-and-back pair (the back step undoing the
    out step) marked invalid."""
    new = recording.copy()
    marked, unpaired = {}, {}
    for name, seg in new.segments.items():
        if seg.reference.rate_hz < limit_dps / 180.0:
            continue
        r = impossible_steps(seg.reference, limit_dps, _others(recording, name))
        marked[name] = int((r['mask'] & seg.reference.valid).sum())
        unpaired[name] = len(r['unpaired'])
        seg.reference.valid = seg.reference.valid & ~r['mask']
    if sum(marked.values()) or sum(unpaired.values()):
        new.log_correction(
            ERROR, f'Marked {sum(marked.values())} reference frames invalid between out-and-back steps faster than '
                   f'{limit_dps:g} °/s; {sum(unpaired.values())} unpaired steps reported, not marked.',
            frames_marked=marked, unpaired_steps=unpaired, limit_dps=limit_dps)
    return new


def check_for_reference_jumps(recording: Recording, threshold_dps: float = 300.0) -> CheckResult:
    """Count reference frames whose angular speed contradicts the gyroscope.

    Needs the streams to be synchronised: with a large time offset the gyroscope and the reference disagree
    everywhere. Run check_for_synchronisation_offset first."""
    per_segment, any_found = {}, False
    for name, seg in recording.segments.items():
        before = ~seg.reference.valid
        bad = implausible_reference_frames(seg, threshold_dps) & ~before
        steps = (seg.reference.orientation[:-1].inv() * seg.reference.orientation[1:]).magnitude()
        per_segment[name] = entry(bool(bad.any()), int(bad.sum()), 'frames', fraction=float(bad.mean()),
                                  largest_step_deg=float(np.degrees(np.nanmax(steps))))
        any_found |= bool(bad.any())
    total = sum(p['estimate'] for p in per_segment.values())
    summary = (f'{total} reference frames rotate implausibly fast compared with the gyroscope.' if any_found
               else 'No reference frame contradicts the gyroscope.')
    return CheckResult(ERROR, any_found, summary, per_segment, dict(threshold_dps=threshold_dps))


def remove_reference_jumps(recording: Recording, threshold_dps: float = 300.0, margin_s: float = 0.5) -> Recording:
    """Return a copy in which every implausible reference frame, plus `margin_s` on both sides, is marked invalid."""
    marked = {}
    new = recording.copy()
    for name, seg in new.segments.items():
        bad = implausible_reference_frames(seg, threshold_dps, margin_s)
        marked[name] = int((bad & seg.reference.valid).sum())
        seg.reference.valid = seg.reference.valid & ~bad
    new.log_correction(
        ERROR, f'Marked {sum(marked.values())} reference frames invalid where the reference contradicts the '
               f'gyroscope by more than {threshold_dps:g} °/s (widened by {margin_s:g} s).',
        frames_marked=marked, threshold_dps=threshold_dps, margin_s=margin_s)
    return new


# --- handler: one dataset at a time -------------------------------------------------------------------------------

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import parallel_map, reader_of, truthy  # noqa: E402


def _check_trial(loader, trial) -> list[dict]:
    rec = loader.load(trial)
    rows = []
    for seg, p in check_for_impossible_reference_steps(rec).per_segment.items():
        ref = rec.segments[seg].reference
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), segment=seg,
                         detected=p['detected'], estimate=p.get('impossible_steps', float('nan')), unit='steps',
                         rate_hz=ref.rate_hz, frames=len(ref), valid_frames=int(ref.valid.sum()),
                         frames_marked=p['estimate'], pairs=p.get('pairs'), unpaired=p.get('unpaired'),
                         rejected_pairs=p.get('rejected_pairs'), unconfirmed_pairs=p.get('unconfirmed_pairs'),
                         fastest_dps=p.get('fastest_step_dps'), reason=p.get('reason', '')))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: the reference steps faster than MAX_SEGMENT_RATE_DPS (estimate), the frames
    marked between paired out-and-back steps, and the unpaired steps (reported, not marked). Undecided where the
    reference's frame rate is too low for the limit."""
    rows = []
    for out in parallel_map(loader, _check_trial, loader.trials(), processes):
        rows += [dict(out, segment='', detected=None)] if isinstance(out, dict) else out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with an impossible step / impossible steps."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'steps', int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def _mask(rec: Recording, trial, *, trials) -> Recording:
    return remove_impossible_reference_steps(rec) if trials is None or str(trial) in trials else rec


def correct(loader, rows: pd.DataFrame):
    """The frames between paired out-and-back steps marked invalid. The mask depends only on the trial's own
    references, so it is recomputed on loading; on the reader the check ran on, only for the trials with a step,
    on another reader (Trento re-paired) for every trial."""
    same = len(rows) == 0 or 'reader' not in rows or rows.reader.iloc[0] == reader_of(loader)
    trials = {str(t) for t in rows.trial[rows.detected.map(truthy)]} if same else None
    if trials == set():
        return loader
    return CorrectedLoader(loader, partial(_mask, trials=trials), 'reference jumps')


SPEC = ErrorSpec(3, 'reference_jumps', 'mask', 'reference jumps or flips (steps faster than any human rotation)')
PARTS = {3: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_reference_jumps = make_handler(PARTS[3])
