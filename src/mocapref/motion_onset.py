"""Motion onset of a trial: where the convergence time starts.

The estimators converge during the first CONVERGENCE_S (10 s) of motion, which are not scored, and the mounting
rotation is fitted on the same seconds (errors/e18_mounting.py), so the scored part never sets it. A trial that
starts at rest would otherwise spend its convergence time standing still, leaving the mounting fit without motion.

Onset: the start of the first window in which the trial exceeds the standing limit, i.e. the rest test of the
gyroscope offset turned around: a window of WINDOW_S (0.5 s, overlapping by half) in which the reference of any
segment turns by a net rotation faster than MAX_REF_DPS (5.6 °/s = 1.5 x the worst static recording, 3.73 °/s,
Trento N-pose). The net rotation over half a second is used, not the frame-to-frame rate, so marker jitter does not
start it. Only the reference is read: the IMU may still carry errors that are corrected later. One onset per trial
(all joints are scored from the same instant). A trial without any such window is static.

Continuous motion: the filters need motion to converge, so the convergence window is the first CONVERGENCE_S of
continuous motion: it starts at a moving window and the majority (more than MAJORITY) of the windows in the
CONVERGENCE_S from there move, a window moving where any segment moves. A brief twitch followed by rest therefore
does not start it (DIODEM gait/exp11: twitches at 1.5 s and 6.5 s, motion from about 12 s; its 10 s window from the
twitch held almost no motion and gave mounting rotations of 64-129 degrees on housing-referenced segments). A trial
with moving windows but no such stretch takes the CONVERGENCE_S with the most motion (it is not static);
convergence_window() reports which.

A segment that barely moves in the trial's window: where a participant has no other trial of that segment to pool
its mounting with (errors/e18_mounting.py), a segment moving in at most MAJORITY of the window's WINDOW_S steps
(segment_motion_share) gets its mounting fitted on its own convergence window instead (segment_convergence_window:
the same rule, read on that segment alone, with moving meaning faster than the fit's 17 °/s instead of the standing
limit, since the fit can use only such samples), and every joint with that segment is scored only from the end of
that window (effect.py; Hip-ROM-Y: the right leg moves later than the left, so the first 10 s of the trial held no
right-shank motion).

Scoring starts at onset + CONVERGENCE_S. Trials too short after onset are not used: a test trial needs MIN_TEST_S
(15 s) from onset, a tuning trial MIN_TUNING_S (20 s), the rule of tuning_split.py counted from the onset instead of
the trial start; static trials have no onset and are not used either. Static trials make no sense for relative
orientation estimation, which has nothing to estimate at rest.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .errors.e17_gyroscope_offset import MAX_REF_DPS, WINDOW_S
from .recording import Recording, ReferenceStream
from .tuning_split import CONVERGENCE_S, MIN_TEST_S, MIN_TUNING_S

SEGMENT_WINDOWS_KEY = 'segment_fit_windows_s'  # metadata: {segment: (start, end)} fitted on its own window
MAJORITY = 0.5  # share of moving windows a convergence window must exceed


def _moving_windows(ref: ReferenceStream, window_s: float, max_ref_dps: float):
    """Start times (in the recording's time base) of the valid windows and whether each one's net reference rotation
    exceeds the limit."""
    n = max(int(round(window_s * ref.rate_hz)), 1)
    if len(ref) <= n:
        return np.array([]), np.array([], bool)
    first = np.arange(0, len(ref) - n, max(n // 2, 1))
    last = first + n
    ok = ref.valid[first] & ref.valid[last]
    first, last = first[ok], last[ok]
    angle = (ref.orientation[first].inv() * ref.orientation[last]).magnitude()
    return ref.start_s + first / ref.rate_hz, angle / (n / ref.rate_hz) > np.radians(max_ref_dps)


def convergence_window(recording: Recording, window_s: float = WINDOW_S, max_ref_dps: float = MAX_REF_DPS) -> dict:
    """The trial's convergence window (module description): onset_s (None for a static trial), the share of moving
    windows in it, and continuous (False where no stretch reached the majority and the one with most motion is taken)."""
    step = window_s / 2
    per = [_moving_windows(seg.reference, window_s, max_ref_dps) for seg in recording.segments.values()
           if seg.reference is not None and len(seg.reference)]
    per = [(t, m) for t, m in per if len(t)]
    if not per or not any(m.any() for _, m in per):
        return dict(onset_s=None, share=0.0, continuous=False)
    t0 = min(t[0] for t, _ in per)
    size = int(round((max(t[-1] for t, _ in per) - t0) / step)) + 1
    moving = np.zeros(size, bool)
    for t, m in per:  # a grid window moves where any segment moves
        moving[np.round((t[m] - t0) / step).astype(int)] = True
    k = max(int(round(CONVERGENCE_S / step)), 1)
    share = np.convolve(moving.astype(float), np.ones(k), 'full')[k - 1:] / k  # windows beyond the end count as rest
    candidates = moving & (share > MAJORITY)
    continuous = bool(candidates.any())
    i = int(np.argmax(candidates)) if continuous else int(np.argmax(np.where(moving, share, -1.0)))
    return dict(onset_s=float(t0 + i * step), share=float(share[i]), continuous=continuous)


def segment_motion_share(recording: Recording, segment: str, start_s: float, window_s: float = CONVERGENCE_S,
                         step_s: float = WINDOW_S, max_ref_dps: float = MAX_REF_DPS) -> float:
    """Share of the segment's own WINDOW_S steps in [start_s, start_s + window_s) whose net reference rotation exceeds
    the standing limit; steps without a valid reference count as rest (as in convergence_window)."""
    ref = recording[segment].reference
    t, m = _moving_windows(ref, step_s, max_ref_dps)
    k = max(int(round(window_s / (step_s / 2))), 1)  # steps of the window, overlapping by half
    inside = (t >= start_s - 1e-9) & (t < start_s + window_s - 1e-9)
    return float(m[inside].sum() / k)


def segment_convergence_window(recording: Recording, segment: str, window_s: float = WINDOW_S,
                               max_ref_dps: float = MAX_REF_DPS) -> dict:
    """convergence_window read on one segment alone: its own first CONVERGENCE_S of continuous motion."""
    return convergence_window(replace(recording, segments={segment: recording[segment]}), window_s, max_ref_dps)


def motion_onset_s(recording: Recording, window_s: float = WINDOW_S, max_ref_dps: float = MAX_REF_DPS) -> float | None:
    """Start of the trial's convergence window (module description), or None for a static trial."""
    return convergence_window(recording, window_s, max_ref_dps)['onset_s']


def scoring_start_s(recording: Recording, onset_s: float | None = None) -> float | None:
    """Instant from which the trial is scored: onset + CONVERGENCE_S; None for a static trial."""
    onset = motion_onset_s(recording) if onset_s is None else onset_s
    return None if onset is None else onset + CONVERGENCE_S


def long_enough(recording: Recording, role: str, onset_s: float | None = None) -> tuple[bool, str]:
    """Whether a trial may be used for `role` ('tuning' or 'test') under the onset rule, and why not.

    The end of the trial is the end of the time all segments' IMU and reference cover."""
    onset = motion_onset_s(recording) if onset_s is None else onset_s
    if onset is None:
        return False, 'static: the reference never exceeds the standing limit'
    end = min(min(seg.imu.times[-1], seg.reference.start_s + (len(seg.reference) - 1) / seg.reference.rate_hz)
              for seg in recording.segments.values())
    need = MIN_TUNING_S if role == 'tuning' else MIN_TEST_S
    left = end - onset
    if left < need:
        return False, f'{left:.1f} s from motion onset, {role} needs {need:g} s'
    return True, ''
