"""Sensor-to-segment alignment — the constant rotation between a sensor and its segment 

**Definition.** The **constant** part of the housing-against-anatomy rotation: the trial median ``Q̄`` of
``R_cluster(t)ᵀ R_anatomical(t)``, reported as an angle in degrees per sensor, with the spread over the trial
printed beside it. The spread over sequences is the estimability limit of the quantity; the spread across
subjects is the effect, and it is formed when the per-sequence results are pooled, not here. *Second route
(§3):* the cluster-to-sensor rotation from a Wahba fit of the reference-predicted specific force against the
accelerometer — the reference's own second derivative of position minus gravity, expressed in the reference
frame, matched to the accelerometer by one constant rotation over the whole trial. It is a constant fit, not
an orientation estimator, and on a still or slowly moving stretch it reduces to matching the gravity direction.

**When the second route cannot be run.** The accelerometer route reports itself **not measurable** where the
specific-force direction does not rotate enough over the sequence for a constant rotation to be identifiable — the
root-mean-square transverse spread of the direction is below :data:`MIN_DIRECTION_SPREAD_DEG`. On a sequence that
holds one attitude the direction of gravity in the sensor frame is fixed and the rotation about it is free. The
direction spread and the conditioning of the fit are reported with every value.

**Null value.** Zero by construction on a system whose sensor frame *is* the segment frame — a machined rigid
chain or a robot link. On a housing-referenced human dataset the quantity the second route measures is not the
alignment but the **cluster-to-sensor** offset, of the order of a degree.

**Side.** Reference-side by definition — the segment frame is the reference's construction. **Not measurable**
(never zero) on a dataset with no marker on any housing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from mocapref.datasets.readers.characterisation.schema import Recording

from . import Constant, Result, not_measurable, side_of
from ._common import (
    GRAVITY,
    pose_linear_acceleration,
    resample_cubic,
    rotation_angle_deg,
    wahba_fit,
)
from .soft_mounting import housing_against_anatomy

EFFECT = "sensor-to-segment alignment"

GRAVITY_WORLD = np.array([0.0, 0.0, -GRAVITY])

IDENTIFIABILITY_RATIO = 10.0
"""Data-free. A constant rotation between two vector streams is determined about an axis only by the spread of
the directions **transverse** to that axis. Requiring that no degree of freedom of the rotation be determined
more than an order of magnitude worse than the other two fixes the smallest admissible transverse spread at
``arcsin(1 / 10)``. The factor of ten is the choice; the angle follows from it."""

MIN_DIRECTION_SPREAD_DEG = float(np.degrees(np.arcsin(1.0 / IDENTIFIABILITY_RATIO)))  # 5.74°

CONSTANTS = (
    Constant("standard gravity", GRAVITY, "m/s²", "physical",
             "the reference-predicted specific force is the reference's own acceleration minus gravity"),
    Constant("gravity direction", (0.0, 0.0, -1.0), "unit vector", "read-from-dataset",
             "the laboratory frame is z-up in the adapters' convention (schema.py); a dataset with another "
             "convention has it fixed in its adapter, not here"),
    Constant("identifiability ratio", IDENTIFIABILITY_RATIO, "1", "data-free",
             "the accelerometer route is reported as not measurable when the specific-force direction does "
             "not rotate enough in the sequence for the rotation about it to be determined within an order "
             "of magnitude as well as the other two degrees of freedom"),
    Constant("minimum direction spread", MIN_DIRECTION_SPREAD_DEG, "°", "data-free",
             "arcsin(1 / identifiability ratio); the root-mean-square transverse spread of the specific-force "
             "direction below which the fit is unidentifiable"),
)


def direction_spread(v: np.ndarray) -> dict:
    """How much a stream of vectors changes **direction** over a sequence, and how well conditioned a constant
    rotation fitted to it therefore is.

    The unit directions' scatter matrix ``S = mean(u uᵀ)`` has eigenvalues ``λ₁ ≥ λ₂ ≥ λ₃`` summing to one.
    ``√λ₂`` is the root-mean-square sine of the excursion away from the dominant direction in the better of
    the two transverse directions, so ``arcsin √λ₂`` is the angular spread that constrains the rotation about
    the dominant direction — the one degree of freedom that a sequence holding one attitude cannot determine.
    ``λ₁ / λ₂`` is reported as the conditioning of the fit.
    """
    v = np.asarray(v, float)
    ok = np.isfinite(v).all(1) & (np.linalg.norm(v, axis=1) > 0)
    if ok.sum() < 3:
        return {"spread_deg": float("nan"), "condition_number": float("nan"),
                "eigenvalues": [float("nan")] * 3, "n_samples": int(ok.sum())}
    u = v[ok] / np.linalg.norm(v[ok], axis=1, keepdims=True)
    lam = np.sort(np.linalg.eigvalsh(u.T @ u / len(u)))[::-1]
    lam = np.clip(lam, 0.0, 1.0)
    return {
        "spread_deg": float(np.degrees(np.arcsin(np.sqrt(lam[1])))),
        "smallest_spread_deg": float(np.degrees(np.arcsin(np.sqrt(lam[2])))),
        "condition_number": float(lam[0] / lam[1]) if lam[1] > 0 else float("inf"),
        "eigenvalues": [float(x) for x in lam],
        "n_samples": int(ok.sum()),
    }


def measure(recording: Recording, **options) -> list[Result]:
    """One result per segment per route: the constant housing-against-anatomy rotation, and the
    cluster-to-sensor rotation from the accelerometer."""
    results: list[Result] = []
    for seg_name, seg in recording.segments.items():
        side = side_of(recording, seg_name)

        got, reason = housing_against_anatomy(recording, seg_name)
        if got is None:
            results.append(
                not_measurable(EFFECT, recording, seg_name, "constant part of housing against anatomy",
                               "reference-side", reason, "°")
            )
        else:
            _t, Q, Qbar = got
            ang = rotation_angle_deg(np.einsum("tij,kj->tik", Q, Qbar))
            results.append(
                Result(
                    effect=EFFECT,
                    dataset=recording.dataset,
                    subject=recording.subject,
                    trial=recording.trial,
                    unit=seg_name,
                    value={
                        "alignment_deg": float(rotation_angle_deg(Qbar)),
                        "within_trial_spread_deg": float(np.nanstd(ang)),
                    },
                    units="°",
                    route="constant part of housing against anatomy",
                    side="reference-side",
                    n_samples=int(np.isfinite(ang).sum()),
                    extras={
                        "rotation_matrix": Qbar,
                        "estimability_limit_deg": float(np.nanstd(ang)),
                        "note": "trial median of the rotation (element-wise median projected onto rotations)",
                    },
                    constants=CONSTANTS,
                )
            )

        if seg.reference is None or seg.sensor is None:
            results.append(
                not_measurable(EFFECT, recording, seg_name, "cluster-to-sensor rotation from the accelerometer",
                               side, "the segment carries no reference pose or no sensor", "°")
            )
            continue
        t_ref = np.asarray(seg.reference.t, float)
        acc_world = pose_linear_acceleration(t_ref, np.asarray(seg.reference.p, float))
        f_ref = np.einsum("tji,tj->ti", seg.reference.R, acc_world - GRAVITY_WORLD)
        a_sensor = resample_cubic(np.asarray(seg.sensor.t, float), np.asarray(seg.sensor.acc, float), t_ref)
        spread_ref = direction_spread(f_ref)
        spread_sen = direction_spread(a_sensor)
        conditioning = {
            "reference_direction_spread": spread_ref,
            "sensor_direction_spread": spread_sen,
            "spread_deg": float(np.nanmin([spread_ref["spread_deg"], spread_sen["spread_deg"]])),
            "minimum_spread_deg": MIN_DIRECTION_SPREAD_DEG,
            "note": "the rotation about the dominant specific-force direction is constrained only by the "
                    "spread transverse to it; on a sequence that holds one attitude that direction does not "
                    "rotate in the sensor frame and one degree of freedom of the fit is free, which is how a "
                    "measure whose answer is 0° returns 179°",
        }
        if not np.isfinite(conditioning["spread_deg"]) or conditioning["spread_deg"] < MIN_DIRECTION_SPREAD_DEG:
            results.append(
                replace(
                    not_measurable(
                        EFFECT, recording, seg_name, "cluster-to-sensor rotation from the accelerometer", side,
                        f"unidentifiable on this sequence: the specific-force direction spreads by only "
                        f"{conditioning['spread_deg']:.3g}° over the trial, below the "
                        f"{MIN_DIRECTION_SPREAD_DEG:.3g}° at which the rotation about it is determined within "
                        f"a factor of {IDENTIFIABILITY_RATIO:g} as well as the other two axes", "°",
                    ),
                    extras=conditioning, constants=CONSTANTS,
                )
            )
            continue
        M = wahba_fit(f_ref, a_sensor)
        resid = a_sensor - f_ref @ M.T
        results.append(
            Result(
                effect=EFFECT,
                dataset=recording.dataset,
                subject=recording.subject,
                trial=recording.trial,
                unit=seg_name,
                value={
                    "rotation_deg": float(rotation_angle_deg(M)),
                    "residual_rms_m_s2": float(np.sqrt(np.nanmean(np.sum(resid**2, axis=1)))),
                },
                units="°",
                route="cluster-to-sensor rotation from the accelerometer",
                side=side,
                n_samples=int(np.isfinite(resid).all(1).sum()),
                extras=dict(
                    conditioning,
                    rotation_matrix=M,
                    reference_kind=seg.reference.kind,
                    note="one constant rotation fitted over the whole trial (Wahba) between the "
                         "reference-predicted specific force and the accelerometer; on a housing-referenced "
                         "dataset this is the cluster-to-sensor offset, not the alignment",
                ),
                constants=CONSTANTS,
            )
        )
    return results
