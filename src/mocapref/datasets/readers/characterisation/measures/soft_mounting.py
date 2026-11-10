"""Soft mounting — the housing moves against the segment it is strapped to (row E5, specification §2.6)

**Definition, two constructions that are not two routes to one number.** *Construction A — housing against
anatomy:* where a segment carries both a housing marker cluster and an anatomical frame, the rotation
``Q(t) = R_cluster(t)ᵀ R_anatomical(t)``; reported is the angle of ``Q(t) Q̄ᵀ`` about the trial median ``Q̄``,
as a root mean square and a 95th percentile in degrees, with its power spectrum beside it (where the mount's
resonance sits is an open question of the catalogue). The constant part ``Q̄`` is not the mount — it is the
sensor-to-segment alignment, measured in its own module. *Construction B — same-segment pair:* for two sensors
on one rigid segment, rigidity means ``ω_B(t) = R_BA ω_A(t)`` with ``R_BA`` constant; ``R_BA`` is fitted once
per sequence by least squares over the gyroscope vectors (a Wahba fit over the whole trial, a measurement, not
an estimator) and the reported number is the residual ``‖ω_B − R_BA ω_A‖`` as a root mean square, divided by
the root mean square of ``‖ω_A‖`` — a dimensionless fraction of the signal — plus the same in °/s. The
gyroscope is used and not the accelerometer because rotation carries no lever-arm term, so this number
contains no joint-centre contribution.

**Null value.** Construction B on two sensors bolted to one rigid link reads the gyroscope noise level (zero
on a noiseless recording); Construction A does not exist on a rigid system, since there is no anatomy.

**Side.** Construction A is reference-side — it is the mount as a skin-defined segment frame sees it, and it
contains skin motion. Construction B is sensor-side and contains no reference at all.

**Known gap.** No simulation arm injects a slow or multi-mode mount (specification §4(a)), so every
soft-mounting number is printed with that gap named; it is carried in ``extras["uncalibrated_gap"]``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

from mocapref.datasets.readers.characterisation.schema import Recording

from . import Constant, Result, not_measurable, side_of
from ._common import RAD2DEG, median_rotation, resample_cubic, rotation_angle_deg, uniform_rate, wahba_fit

EFFECT = "soft mounting"

CONSTANTS = (
    Constant("percentile reported", 95, "percentile", "data-free", "the spread reported beside the root mean square"),
    Constant("radians to degrees", RAD2DEG, "°/rad", "physical", "unit conversion"),
)

UNCALIBRATED_GAP = (
    "no simulation arm injects a slow or multi-mode mount (docs/characterisation.md §4(a)); the number is "
    "calibrated for a fast mount only"
)


def housing_against_anatomy(recording: Recording, segment: str):
    """``Q(t) = R_clusterᵀ R_anatomical`` on the cluster's time base, with the trial median ``Q̄``."""
    seg = recording.segments.get(segment)
    if seg is None or seg.reference is None or seg.anatomical is None:
        return None, "the segment does not carry both a housing cluster and an anatomical frame"
    if seg.reference.kind != "housing":
        return None, f"the segment's reference is '{seg.reference.kind}', not a marker cluster on the housing"
    t = seg.reference.t
    Ra = resample_cubic(seg.anatomical.t, seg.anatomical.R.reshape(len(seg.anatomical.t), 9), t).reshape(-1, 3, 3)
    Q = np.einsum("tji,tjk->tik", seg.reference.R, Ra)
    return (t, Q, median_rotation(Q)), None


def measure(recording: Recording, **options) -> list[Result]:
    """Construction A per segment where both frames exist; Construction B per same-segment pair."""
    results: list[Result] = []

    for seg_name in recording.segments:
        got, reason = housing_against_anatomy(recording, seg_name)
        if got is None:
            results.append(
                not_measurable(EFFECT, recording, seg_name, "housing against anatomy",
                               side_of(recording, seg_name), reason, "°")
            )
            continue
        t, Q, Qbar = got
        ang = rotation_angle_deg(np.einsum("tij,kj->tik", Q, Qbar))
        fs = uniform_rate(t)
        f, pxx = periodogram(ang - np.nanmean(ang), fs=fs)
        results.append(
            Result(
                effect=EFFECT,
                dataset=recording.dataset,
                subject=recording.subject,
                trial=recording.trial,
                unit=seg_name,
                value={
                    "rms_deg": float(np.sqrt(np.nanmean(ang**2))),
                    "p95_deg": float(np.nanpercentile(ang, 95)),
                    "max_deg": float(np.nanmax(ang)),
                },
                units="°",
                route="housing against anatomy",
                side="reference-side",
                n_samples=int(np.isfinite(ang).sum()),
                extras={
                    "spectrum": {"frequency_hz": f, "power_deg2_per_hz": pxx},
                    "constant_part_deg": float(rotation_angle_deg(Qbar)),
                    "note": "the constant part is the sensor-to-segment alignment and is reported there",
                    "uncalibrated_gap": UNCALIBRATED_GAP,
                },
                constants=CONSTANTS,
            )
        )

    for segment, ia, ib in recording.same_segment_pairs:
        streams = recording.sensors_of(segment)
        unit = f"{segment}#{ia}-#{ib}"
        if max(ia, ib) >= len(streams):
            results.append(not_measurable(EFFECT, recording, unit, "same-segment pair", "sensor-side",
                                          "the recording does not carry both sensors of the declared pair", "1"))
            continue
        A, B = streams[ia], streams[ib]
        t = np.asarray(A.t, float)
        wa = np.asarray(A.gyr, float)
        wb = resample_cubic(np.asarray(B.t, float), np.asarray(B.gyr, float), t)
        R_BA = wahba_fit(wa, wb)
        resid = wb - wa @ R_BA.T
        rms_resid = float(np.sqrt(np.nanmean(np.sum(resid**2, axis=1))))
        rms_signal = float(np.sqrt(np.nanmean(np.sum(wa**2, axis=1))))
        results.append(
            Result(
                effect=EFFECT,
                dataset=recording.dataset,
                subject=recording.subject,
                trial=recording.trial,
                unit=unit,
                value={
                    "residual_fraction_of_signal": rms_resid / rms_signal if rms_signal > 0 else float("nan"),
                    "residual_rms_deg_s": rms_resid * RAD2DEG,
                    "signal_rms_deg_s": rms_signal * RAD2DEG,
                },
                units="fraction of the signal; °/s",
                route="same-segment pair",
                side="sensor-side",
                n_samples=int(np.isfinite(resid).all(1).sum()),
                extras={
                    "constant_rotation_deg": float(rotation_angle_deg(R_BA)),
                    "note": "constant rotation fitted once over the whole trial by least squares (Wahba); "
                            "the gyroscope carries no lever-arm term, so no joint-centre contribution enters",
                    "uncalibrated_gap": UNCALIBRATED_GAP,
                },
                constants=CONSTANTS,
            )
        )
    return results
